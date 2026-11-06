"""Shell-command hooks, configured in the ``hooks`` section of config.json::

    "hooks": {
      "tool.before": [{"matcher": "Bash", "command": "~/bin/guard.sh", "fail_closed": true}],
      "tool.after":  [{"matcher": "Edit|Write", "command": "ruff check -q . >&2 || exit 2"}],
      "agent.start": [{"command": "git status --short"}],
      "agent.end":   [{"command": "notify-send 'agent done'"}]
    }

Each command runs via ``/bin/sh -c`` in the project dir, gets the event as
JSON on stdin and ``PAH_HOOK_EVENT`` / ``PAH_PROJECT_DIR`` in its env.
Secret-looking env vars are removed (Bash tool's filter) unless listed in
the hook's ``env_passthrough``.

===========  =====================================================
tool.before  exit 2 blocks the call; stderr is the reason
tool.after   exit 2: stderr is appended to the result as feedback
agent.start  exit 0: stdout is context for the turn
agent.end    notification only
===========  =====================================================

Any other exit, a timeout, or a start failure is logged (user only, not
the model) and ignored -- a ``fail_closed`` tool.before then blocks.

Entry keys: ``command`` (required); ``matcher`` (regex, full match on the
tool name; tool events only); ``timeout`` (s, default 30); ``fail_closed``
(tool.before only); ``env_passthrough`` (secret vars this hook may see).
Keys starting with ``_`` (``_comment``, ``_examples``) are ignored.

Security: the payload (``tool_args``, ``result``, ``text``, ``answer``) is
model-written and UNTRUSTED -- never eval it or splice it unquoted into a
shell command.  Block reasons, feedback and context go to the model and
the session file -- never print secrets.

Behavior:

- Safety: no method raises (``HookRunner._guard``); a hook cannot break,
  freeze or exhaust the agent.
- Tool hooks run in the tool runner, sub-agent calls included
  (``is_subagent``).  tool.before runs before the plan-mode guard (which
  still applies) and may run concurrently in parallel readonly rounds.
  Calls rejected before dispatch (bad JSON, missing args, parent-only
  tools) skip tool.before; tool.after sees every delivered result,
  errors included.  Results discarded by a cancel (or a superseded run)
  are never delivered, so their tool.after hooks do not run even when
  the tool already executed.
- Stopping (timeout, Ctrl-C): SIGTERM to the process group, SIGKILL
  after 0.5s.  Ctrl-C acts within ~0.1s; an interrupted tool.before
  blocks its call.  A hook is done when its shell exits.
- Output: 64 KiB kept per stream; at most 8000 chars reach the model.
- agent.end runs on a background thread, is never interrupted, and gets
  ``status`` (``done``/``error``/``cancelled``/``superseded``; error =
  the run raised or returned ``Error: ...``) and ``answer`` (the answer
  shown to the user; the error text on error).
- agent.start context stays in the history -- keep it short.
- Hooks get pipes, not the terminal (/dev/tty output garbles the TUI).
- A running hook shows in the TUI status bar (``hook_running`` notify);
  ``python-agent-harness config`` lists the configured hooks.
- ``Session.hooks`` may be replaced by an object with the same methods.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

EVENTS = ("tool.before", "tool.after", "agent.start", "agent.end")
TOOL_EVENTS = ("tool.before", "tool.after")
RunStatus = Literal["done", "error", "cancelled", "superseded"]
DEFAULT_TIMEOUT = 30.0
BLOCK_EXIT_CODE = 2
MAX_TEXT_CHARS = 8000  # per text sent to the model (reason/feedback/context)
MAX_CAPTURE_BYTES = 64 * 1024  # per output stream; the rest is drained and dropped
_READ_CHUNK = 64 * 1024
_POLL_INTERVAL = 0.1  # exit/cancel check period
_TERM_GRACE = 0.5  # SIGTERM -> SIGKILL
_DRAIN_AFTER_EXIT = 0.25  # wait for pipes to close once the shell has exited
_REAP_TIMEOUT = 1.0  # bound on reaping after a kill
_ENTRY_KEYS = {"command", "matcher", "timeout", "fail_closed", "env_passthrough"}


@dataclass(frozen=True)
class Hook:
    event: str
    command: str
    matcher: re.Pattern[str] | None = None
    timeout: float = DEFAULT_TIMEOUT
    fail_closed: bool = False
    env_passthrough: tuple[str, ...] = ()

    def matches(self, tool_name: str) -> bool:
        return self.matcher is None or self.matcher.fullmatch(tool_name) is not None


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def parse_hooks(section: Any) -> list[Hook]:
    """Validate the ``hooks`` section; raises ValueError (strict: a
    silently skipped entry could be a dropped guard).  Keys starting with
    ``_`` (``_comment``, ``_examples``) are documentation and ignored."""
    if section is None:
        return []
    if not isinstance(section, dict):
        raise ValueError("hooks: must be an object")
    hooks: list[Hook] = []
    for event, entries in section.items():
        if event.startswith("_"):
            continue
        if event not in EVENTS:
            raise ValueError(
                f"hooks: unknown event {event!r} (expected one of: {', '.join(EVENTS)})"
            )
        if not isinstance(entries, list):
            raise ValueError(f"hooks.{event}: must be a list")
        hooks.extend(_parse_entry(event, f"hooks.{event}[{i}]", e) for i, e in enumerate(entries))
    return hooks


def _parse_entry(event: str, where: str, entry: Any) -> Hook:
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: must be an object")
    unknown = {k for k in entry if not k.startswith("_")} - _ENTRY_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"{where}: 'command' must be a non-empty string")
    matcher = None
    if "matcher" in entry:
        if event not in TOOL_EVENTS:
            raise ValueError(f"{where}: 'matcher' is only valid for tool events")
        if not isinstance(entry["matcher"], str) or not entry["matcher"]:
            raise ValueError(f"{where}: 'matcher' must be a non-empty string")
        try:
            matcher = re.compile(entry["matcher"])
        except re.error as e:
            raise ValueError(f"{where}: invalid 'matcher' regex: {e}") from e
    timeout = entry.get("timeout", DEFAULT_TIMEOUT)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)  # json accepts NaN / Infinity
        or timeout <= 0
    ):
        raise ValueError(f"{where}: 'timeout' must be a positive number")
    fail_closed = entry.get("fail_closed", False)
    if not isinstance(fail_closed, bool):
        raise ValueError(f"{where}: 'fail_closed' must be true or false")
    if fail_closed and event != "tool.before":
        raise ValueError(f"{where}: 'fail_closed' is only valid for tool.before")
    passthrough = entry.get("env_passthrough", [])
    if not isinstance(passthrough, list) or not all(
        isinstance(n, str) and n.strip() for n in passthrough
    ):
        raise ValueError(f"{where}: 'env_passthrough' must be a list of variable names")
    return Hook(
        event,
        command,
        matcher,
        float(timeout),
        fail_closed,
        tuple(n.strip() for n in passthrough),
    )


# ---------------------------------------------------------------------------
# running a command
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Outcome:
    code: int | None  # None: did not complete (see error)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    cancelled: bool = False


class _Capture:
    """One output stream: at most MAX_CAPTURE_BYTES kept."""

    __slots__ = ("data", "dropped")

    def __init__(self) -> None:
        self.data = bytearray()
        self.dropped = False

    def text(self) -> str:
        text = self.data.decode("utf-8", "replace")
        if self.dropped:
            text += "\n… [output beyond the capture limit discarded]"
        return text


def _run(
    hook: Hook,
    payload: dict[str, Any],
    cwd: str,
    env: dict[str, str],
    cancelled: Callable[[], bool] | None,
) -> _Outcome:
    """Run HOOK with PAYLOAD on stdin.  Never raises; bounded in memory
    (capped readers keep draining, so no full-pipe stall) and time
    (timeout / CANCELLED -> ``_stop``).  A background process holding a
    pipe after the shell exits keeps only its (daemon) reader alive."""
    try:  # before spawning: a failure here must not leave a process behind
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8", "replace")
    except (TypeError, ValueError, RecursionError) as e:
        return _Outcome(None, error=f"payload not serializable: {e}")
    try:
        proc = subprocess.Popen(
            ["/bin/sh", "-c", hook.command],
            cwd=cwd if os.path.isdir(cwd) else None,
            env={**env, "PAH_HOOK_EVENT": hook.event, "PAH_PROJECT_DIR": cwd},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # own group: signals reach all it spawned
        )
    except (OSError, ValueError) as e:  # ValueError: NUL in the command
        return _Outcome(None, error=f"cannot start: {e}")
    out, err = _Capture(), _Capture()
    readers = [_start_reader(proc.stdout, out), _start_reader(proc.stderr, err)]
    _start_writer(proc.stdin, data)

    deadline = time.monotonic() + hook.timeout
    error: str | None = None
    was_cancelled = False
    while True:
        try:
            proc.wait(timeout=max(0.0, min(_POLL_INTERVAL, deadline - time.monotonic())))
            break
        except subprocess.TimeoutExpired:
            pass
        if cancelled is not None and cancelled():
            error, was_cancelled = "cancelled", True
        elif time.monotonic() >= deadline:
            error = f"timed out after {hook.timeout:g}s"
        if error is not None:
            _stop(proc, readers)
            break
    # pipes normally close with the shell; a background job holding them
    # costs at most this much
    drain_end = time.monotonic() + _DRAIN_AFTER_EXIT
    for reader in readers:
        reader.join(max(0.0, drain_end - time.monotonic()))
    if error is not None:
        return _Outcome(None, error=error, cancelled=was_cancelled)
    return _Outcome(proc.returncode, out.text(), err.text())


def _start_reader(stream: Any, sink: _Capture) -> threading.Thread:
    """Drain STREAM to EOF into SINK (capped)."""

    def read() -> None:
        try:
            while chunk := stream.read1(_READ_CHUNK):
                room = MAX_CAPTURE_BYTES - len(sink.data)
                if len(chunk) > room:
                    sink.dropped = True
                if room > 0:
                    sink.data.extend(chunk[:room])
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError, ValueError):
                stream.close()

    thread = threading.Thread(target=read, name="hook-reader", daemon=True)
    thread.start()
    return thread


def _start_writer(stream: Any, data: bytes) -> None:
    """Write DATA to stdin and close it (EPIPE from a non-reader is fine)."""

    def write() -> None:
        try:
            stream.write(data)
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError, ValueError):
                stream.close()

    threading.Thread(target=write, name="hook-writer", daemon=True).start()


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


def _stop(proc: subprocess.Popen[bytes], readers: list[threading.Thread]) -> None:
    """SIGTERM the group, wait up to _TERM_GRACE for its pipes to close,
    SIGKILL, then reap.  Reaping last keeps the pid (= group id) from
    being reused, so the signals cannot hit an unrelated process."""
    _signal_group(proc, signal.SIGTERM)
    grace_end = time.monotonic() + _TERM_GRACE
    for reader in readers:
        reader.join(max(0.0, grace_end - time.monotonic()))
    _signal_group(proc, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        proc.wait(timeout=_REAP_TIMEOUT)


def _env_builder() -> Callable[[Hook], dict[str, str]]:
    """Per-event environment: os.environ snapshotted and scrubbed ONCE
    (the scrub can stat paths), then each hook's passthrough added back."""
    from ..tools.bash import scrubbed_env  # lazy: avoids an import cycle

    full = dict(os.environ)
    base = scrubbed_env(full)

    def env_for(hook: Hook) -> dict[str, str]:
        if not hook.env_passthrough:
            return base
        wanted = {n.upper() for n in hook.env_passthrough}
        return {**base, **{k: v for k, v in full.items() if k.upper() in wanted}}

    return env_for


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"\n… [truncated {len(text) - MAX_TEXT_CHARS} chars]"
    return text


def _parsed_args(arguments: Any) -> Any:
    """Arguments as JSON data; the raw string if it does not parse."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (ValueError, RecursionError):  # invalid / absurdly nested
            return arguments
    return arguments


_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# A shell word: single-quoted, double-quoted (both may contain spaces), or
# a bare run of non-space characters.
_WORD = re.compile(r"""'([^']*)'|"([^"]*)"|(\S+)""")
_LABEL_CHARS = 40


def _label(hook: Hook) -> str:
    """Status-bar text for a running hook: its program, never the rest.

    Only the executable's own name is shown: arguments (or a lone
    ``VAR=value`` assignment) may inline a credential and the status bar
    is on screen.  No program at all -> the event alone.
    """
    words: list[str] = []
    for match in _WORD.finditer(hook.command):
        # group(1)/(2) are the quoted bodies and may be empty: '' is a
        # word, but an empty one, so it must not become a program.
        word = match.group(1) or match.group(2) or match.group(3)
        if word and word.strip():
            words.append(word)
    while words and _ASSIGNMENT.match(words[0]):  # VAR=value cmd ...
        words.pop(0)
    # a quoted program: '"/opt/my dir/guard.sh" --flag' -> guard.sh
    program = os.path.basename(words[0]) if words else ""
    if not program:
        return f"hook {hook.event}"
    if len(program) > _LABEL_CHARS:
        program = program[: _LABEL_CHARS - 1] + "…"
    return f"hook {hook.event}: {program}"


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
class HookRunner:
    """Runs the configured hooks, in config order.

    Safety: every public method is one ``_guard`` call -- the single
    boundary that turns any internal error into that method's safe
    default.  Callers pass raw inputs (unparsed arguments, a callable for
    the end-of-run outcome) so all hook work happens behind it.  A single
    hook failing (exit code, timeout, start error) is an ordinary outcome:
    logged, or a block for a ``fail_closed`` tool.before.

    PROJECT_DIR is re-read per hook (project switches); LOG gets failures;
    CANCELLED (the session's cancel flag) stops all but agent.end; STATUS
    gets a label while a hook runs and None when it ends (agent.end
    excepted: it runs after the run is over).
    """

    def __init__(
        self,
        hooks: list[Hook],
        project_dir: Callable[[], str],
        log: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        status: Callable[[str | None], None] | None = None,
    ) -> None:
        self._hooks = list(hooks)
        self._project_dir = project_dir
        self._log = log
        self._cancelled = cancelled
        self._status = status

    def __len__(self) -> int:
        return len(self._hooks)

    def _guard(self, event: str, fallback: Callable[[], Any], work: Callable[[], Any]) -> Any:
        """Run WORK; on any exception, log it and return ``fallback()``."""
        try:
            return work()
        except Exception as e:  # noqa: BLE001 - the containment boundary
            self._emit(f"{event} hooks: internal error ({type(e).__name__}: {e}); skipped")
            return fallback()

    def _emit(self, message: str) -> None:
        if self._log is not None:
            with contextlib.suppress(Exception):
                self._log(message)

    # -- public API: one guarded call each ------------------------------------
    def before_tool(
        self, name: str, arguments: Any, call_id: str | None, is_subagent: bool
    ) -> str | None:
        """Return the block message to deliver instead of running the tool,
        or None to run it.  A cancel blocks.  Fallback: allow, or block if
        a fail_closed tool.before hook covers this tool."""

        def fallback() -> str | None:
            try:
                closed = any(h.fail_closed and h.matches(name) for h in self._for("tool.before"))
            except Exception:  # noqa: BLE001 - cannot tell: assume it is covered
                closed = any(h.fail_closed for h in self._hooks if h.event == "tool.before")
            if closed:
                return f"Error: tool {name} blocked by hook — internal hook error (fail_closed)"
            return None

        return self._guard(
            "tool.before",
            fallback,
            lambda: self._before_tool(name, arguments, call_id, is_subagent),
        )

    def after_tool(
        self, name: str, arguments: Any, call_id: str | None, is_subagent: bool, result: str
    ) -> str:
        """Return RESULT with hook feedback appended.  Fallback: RESULT."""
        return self._guard(
            "tool.after",
            lambda: result,
            lambda: self._after_tool(name, arguments, call_id, is_subagent, result),
        )

    def agent_start(self, text: str) -> str | None:
        """Return context for the turn, or None.  Fallback: None."""
        return self._guard("agent.start", lambda: None, lambda: self._agent_start(text))

    def agent_end(
        self, text: str, outcome: Callable[[], tuple[str | None, RunStatus]]
    ) -> threading.Thread | None:
        """Start agent.end hooks on a background thread and return it (None
        if there are none, or on error).  OUTCOME -> ``(answer, status)``
        is evaluated now, while the history is still this run's.

        Not awaited, so a slow hook never keeps the run busy (TUI and
        headless/serve wait for the worker); non-daemon, so a headless
        exit still lets it finish.  Never interrupted by a cancel."""
        return self._guard("agent.end", lambda: None, lambda: self._agent_end(text, outcome))

    # -- event bodies: only called through _guard -----------------------------
    def _for(self, event: str, tool_name: str | None = None) -> list[Hook]:
        return [
            h
            for h in self._hooks
            if h.event == event and (tool_name is None or h.matches(tool_name))
        ]

    def _is_cancelled(self) -> bool:
        try:
            return self._cancelled is not None and bool(self._cancelled())
        except Exception:  # noqa: BLE001 - a broken flag must not break the run
            return False

    def _show(self, label: str | None) -> None:
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status(label)

    def _fire(
        self,
        hook: Hook,
        payload: dict[str, Any],
        env_for: Callable[[Hook], dict[str, str]],
        interruptible: bool = True,
    ) -> _Outcome:
        """Run one hook; its failures are an outcome, never an exception."""
        cancelled = self._is_cancelled if interruptible else None
        if cancelled is not None and cancelled():
            return _Outcome(None, error="cancelled", cancelled=True)
        if interruptible:
            self._show(_label(hook))
        try:
            return _run(hook, payload, self._project_dir(), env_for(hook), cancelled)
        except Exception as e:  # noqa: BLE001 - one hook failing is an outcome
            return _Outcome(None, error=str(e) or type(e).__name__)
        finally:
            if interruptible:
                self._show(None)

    def _failure(self, hook: Hook, outcome: _Outcome) -> str:
        """Log a failed hook; return the message."""
        if outcome.error is not None:
            detail = outcome.error
        else:
            stderr = outcome.stderr.strip()
            detail = f"exited with {outcome.code}" + (f": {_clip(stderr)}" if stderr else "")
        message = f"{hook.event} hook `{hook.command}` failed: {detail}"
        self._emit(message)
        return message

    def _before_tool(
        self, name: str, arguments: Any, call_id: str | None, is_subagent: bool
    ) -> str | None:
        hooks = self._for("tool.before", name)
        if not hooks:
            return None
        env_for = _env_builder()
        payload = {
            "event": "tool.before",
            "tool_name": name,
            "tool_args": _parsed_args(arguments),
            "call_id": call_id,
            "is_subagent": is_subagent,
        }
        for hook in hooks:
            outcome = self._fire(hook, payload, env_for)
            reason: str | None = None
            if outcome.cancelled:
                reason = "cancelled by the user"
            elif outcome.code == BLOCK_EXIT_CODE:
                reason = _clip(outcome.stderr) or f"blocked by `{hook.command}`"
            elif outcome.code != 0:
                message = self._failure(hook, outcome)
                if hook.fail_closed:
                    reason = f"{message} (fail_closed)"
            if reason is not None:
                return f"Error: tool {name} blocked by hook — {reason}"
        return None

    def _after_tool(
        self, name: str, arguments: Any, call_id: str | None, is_subagent: bool, result: str
    ) -> str:
        hooks = self._for("tool.after", name)
        if not hooks:
            return result
        env_for = _env_builder()
        payload = {
            "event": "tool.after",
            "tool_name": name,
            "tool_args": _parsed_args(arguments),
            "call_id": call_id,
            "is_subagent": is_subagent,
            "result": result,
        }
        feedback: list[str] = []
        for hook in hooks:
            outcome = self._fire(hook, payload, env_for)
            if outcome.cancelled:
                return result  # run aborting; results are discarded
            if outcome.code == BLOCK_EXIT_CODE:
                text = _clip(outcome.stderr)
                if text:
                    feedback.append(text)
            elif outcome.code != 0:
                self._failure(hook, outcome)
        if not feedback:
            return result
        # suffix, not prefix: the TUI marks failures by an "Error" prefix
        return f"{result}\n\n[tool.after hook feedback]\n" + "\n\n".join(feedback)

    def _agent_start(self, text: str) -> str | None:
        hooks = self._for("agent.start")
        if not hooks:
            return None
        env_for = _env_builder()
        context: list[str] = []
        for hook in hooks:
            outcome = self._fire(hook, {"event": "agent.start", "text": text}, env_for)
            if outcome.cancelled:
                return None
            if outcome.code == 0:
                out = _clip(outcome.stdout)
                if out:
                    context.append(out)
            else:
                self._failure(hook, outcome)
        return "\n\n".join(context) or None

    def _agent_end(
        self, text: str, outcome: Callable[[], tuple[str | None, RunStatus]]
    ) -> threading.Thread | None:
        hooks = self._for("agent.end")
        if not hooks:
            return None
        answer, status = outcome()
        payload = {"event": "agent.end", "text": text, "answer": answer, "status": status}

        def run() -> None:
            env_for = _env_builder()
            for hook in hooks:
                result = self._fire(hook, payload, env_for, interruptible=False)
                if result.code != 0:
                    self._failure(hook, result)

        # guarded body: an escaping exception would print over the TUI
        thread = threading.Thread(
            target=lambda: self._guard("agent.end", lambda: None, run),
            name="hook-agent-end",
            daemon=False,
        )
        thread.start()
        return thread
