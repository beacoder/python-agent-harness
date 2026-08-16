"""Bash tool.

Asynchronous (mirrors ``:async t`` in gptel-agent-tools): ``run``
spawns the process and returns a ``PendingToolResult`` immediately; a
background thread collects the output and delivers it when the process
exits.  A long-running command therefore never blocks the parent's
sequential tool loop — it runs concurrently with sibling async tools
(Agent) while sync tools execute one at a time.

A session cancel (Ctrl-C) kills the process group and delivers a
cancelled error.

Output is read incrementally with a bounded buffer (head + tail), so a
huge stream (e.g. ``cat`` of a multi-GB log) can never exhaust memory,
and delivery never waits on a detached child that keeps the stdout
pipe open after the shell has exited.

Normal completion appends ``Exit code: N`` (N negative = killed by a
signal).  A command that produces no output for ``BASH_TIMEOUT_SILENCE``
seconds is killed (SIGTERM then SIGKILL) and reported as timed out;
``BASH_TIMEOUT_MAX`` optionally caps the total runtime.  Ctrl-C kills
the process group immediately.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import select
import signal
import subprocess
import threading
import time
from collections import deque

from ..config import BASH_TIMEOUT_MAX, BASH_TIMEOUT_SILENCE
from ..config import MAX_OUTPUT_CHARS as _MAX_OUTPUT
from .base import PendingToolResult, Tool, ToolContext

# Tail lines kept after truncation (the head budget is derived from
# MAX_OUTPUT_CHARS, shared with the filesystem spool threshold).
_TAIL_LINES = 50  # lines kept from the tail after truncation
_READ_CHUNK = 64 * 1024  # bytes read from the pipe per iteration
_POLL_INTERVAL = 0.02  # seconds between cancel/exit checks
_DRAIN_GRACE = 0.25  # seconds to keep reading after the process exits


def _kill_pgid(pgid: int) -> None:
    """Kill the process group PGID, ignoring "already gone" errors.

    The group id is captured at spawn (with ``start_new_session=True``
    the child is the group leader, so its pid IS the pgid): it must
    never be resolved at kill time via ``os.getpgid`` — by then the
    shell may already be dead while a detached child that keeps the
    stdout pipe open is still running in the group.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGKILL)


def _kill_graceful(pgid: int, proc: subprocess.Popen) -> None:
    """SIGTERM the process group; SIGKILL if it is still alive 2s later.

    Used for timeouts so shells/compilers get a chance to clean up
    children before the hard kill.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _kill_pgid(pgid)


def _timeout_message(out: str, silence: bool) -> str:
    """Report a timed-out command, preserving any output so far."""
    if silence:
        timeout = BASH_TIMEOUT_SILENCE or 0.0
        reason = f"no output for {timeout:.0f}s"
    else:
        timeout = BASH_TIMEOUT_MAX or 0.0
        reason = f"exceeded the {timeout:.0f}s maximum"
    out = out.rstrip("\n")
    suffix = f"Error: Bash command timed out ({reason})."
    return f"{out}\n\n{suffix}" if out else suffix


def _append_exit_code(out: str, proc: subprocess.Popen) -> str:
    """Append ``Exit code: N`` (N negative = killed by a signal).

    The exit-code line is added AFTER truncation, so it is always the
    last line and survives the head+tail retention.
    """
    try:
        rc = proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        return out  # still alive; nothing useful to report
    if out and not out.endswith("\n"):
        out += "\n"
    return f"{out}Exit code: {rc}"


def _assemble_truncated(head: str, tail: deque[str]) -> str:
    """Assemble a truncated output within the cap.

    ``head`` holds the first ``_MAX_OUTPUT`` chars, ``tail`` the last
    ``_TAIL_LINES`` lines (each already line-capped).  The tail is
    preferred: as many trailing lines as fit are kept and the head gets
    the remaining budget, so the delivered string never exceeds
    ``_MAX_OUTPUT`` (plus the truncation notice).
    """
    notice = f"... [truncated: output exceeded {_MAX_OUTPUT} chars] ..."
    budget = _MAX_OUTPUT - len(notice) - 4  # room for the "\n\n" separators
    tail_parts: list[str] = []
    used = 0
    for line in reversed(tail):
        cost = len(line) + (1 if tail_parts else 0)
        if used + cost > budget:
            break
        tail_parts.append(line)
        used += cost
    head = head[: max(0, budget - used)]
    out = f"{head}\n\n{notice}"
    if tail_parts:
        out += "\n\n" + "\n".join(reversed(tail_parts))
    return out


def _collect_output(proc: subprocess.Popen, cancel: threading.Event | None) -> tuple[str, str]:
    """Read PROC's merged output incrementally; return (text, status).

    Status is one of ``"ok"``, ``"cancelled"``, ``"timeout_silence"``,
    ``"timeout_max"``.  Keeps the head (first ``_MAX_OUTPUT`` chars) and
    the tail (last ``_TAIL_LINES`` lines) and discards the middle, so
    memory stays bounded no matter how much the process writes.  The
    read loop is poll-based: a cancel is noticed promptly, a process
    silent for ``BASH_TIMEOUT_SILENCE`` seconds (or running past
    ``BASH_TIMEOUT_MAX``) is reported as timed out, and a process that
    has exited is only drained for ``_DRAIN_GRACE`` seconds — a
    detached child holding the pipe open can never wedge delivery.
    """
    stdout = proc.stdout
    if stdout is None:  # unreachable (stdout=PIPE), kept for the type checker
        return "", "ok"
    fd = stdout.fileno()
    os.set_blocking(fd, False)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    head: list[str] = []
    head_len = 0
    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    pending_line = ""
    total = 0
    exited = False
    drain_until: float | None = None
    start = time.monotonic()
    last_output = start

    def finish() -> str:
        nonlocal pending_line
        if pending_line:
            if len(pending_line) > _MAX_OUTPUT:
                pending_line = pending_line[:_MAX_OUTPUT]
            tail.append(pending_line)
        head_text = "".join(head)
        if total > _MAX_OUTPUT:
            return _assemble_truncated(head_text, tail)
        return head_text

    while True:
        if cancel is not None and cancel.is_set():
            return "", "cancelled"
        now = time.monotonic()
        if not exited:
            if BASH_TIMEOUT_SILENCE is not None and now - last_output >= BASH_TIMEOUT_SILENCE:
                return finish(), "timeout_silence"
            if BASH_TIMEOUT_MAX is not None and now - start >= BASH_TIMEOUT_MAX:
                return finish(), "timeout_max"
        if exited and drain_until is not None and now >= drain_until:
            break
        readable, _, _ = select.select([fd], [], [], _POLL_INTERVAL)
        if not readable:
            if not exited and proc.poll() is not None:
                exited = True
                drain_until = time.monotonic() + _DRAIN_GRACE
            continue
        try:
            raw = os.read(fd, _READ_CHUNK)
        except BlockingIOError:
            continue
        except OSError:
            break
        if not raw:
            break  # EOF: every writer closed the pipe
        chunk = decoder.decode(raw)
        total += len(chunk)
        last_output = time.monotonic()
        if head_len < _MAX_OUTPUT:
            take = chunk[: _MAX_OUTPUT - head_len]
            head.append(take)
            head_len += len(take)
        parts = chunk.split("\n")
        parts[0] = pending_line + parts[0]
        pending_line = parts.pop()
        for line in parts:
            if len(line) > _MAX_OUTPUT:
                line = line[:_MAX_OUTPUT]
            tail.append(line)
    return finish(), "ok"


class Bash(Tool):
    name = "Bash"
    _timeout_silence = BASH_TIMEOUT_SILENCE
    _timeout_note = (
        f"A command silent for {_timeout_silence:.0f}s is killed and reported as timed out. "
        if _timeout_silence is not None
        else ""
    )
    description = (
        "Execute a shell command. Returns stdout followed by 'Exit code: N' "
        "(N is the command's exit status; negative means killed by a signal). "
        + _timeout_note
        + "A session cancel (Ctrl-C) kills the process."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
        },
        "required": ["command"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str | PendingToolResult:
        command = args["command"]
        return self._execute(command, ctx)

    def _execute(self, command: str, ctx: ToolContext) -> str | PendingToolResult:
        cancel = ctx.cancel_event
        if cancel is not None and cancel.is_set():
            # Ctrl-C already pending: do not spawn a process that would
            # be killed moments later.
            return "Error: Bash command cancelled."
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                cwd=ctx.cwd,
                start_new_session=True,
            )
        except OSError as e:
            return f"Error: {e}"

        # start_new_session=True makes the child the session/group
        # leader, so its pid IS the group id — captured once here,
        # never resolved again at kill time.
        pgid = proc.pid
        pending = PendingToolResult()

        def deliverer() -> None:
            """Collect output; deliver it once the process exits.

            The kill happens HERE (not in a separate watcher thread):
            the collector is the thread that observed the condition, so
            there is no race window in which a watcher exits without
            killing and the process group survives.  Cancel is an
            immediate SIGKILL; a timeout kills gracefully (SIGTERM,
            then SIGKILL after 2s) so children get a chance to clean up.
            """
            try:
                out, status = _collect_output(proc, cancel)
            except Exception as e:  # noqa: BLE001 - delivered as an error string
                out = f"Error: Bash failed — {e}"
            else:
                if status == "cancelled":
                    _kill_pgid(pgid)
                    out = "Error: Bash command cancelled."
                elif status == "timeout_silence":
                    _kill_graceful(pgid, proc)
                    out = _timeout_message(out, silence=True)
                elif status == "timeout_max":
                    _kill_graceful(pgid, proc)
                    out = _timeout_message(out, silence=False)
                elif status == "ok":
                    out = _append_exit_code(out, proc)
            pending.deliver(out)
            with contextlib.suppress(Exception):
                if proc.stdout is not None:
                    proc.stdout.close()
            with contextlib.suppress(Exception):
                proc.wait(timeout=2)  # reap (bounded; never wedges)

        threading.Thread(target=deliverer, daemon=True).start()
        return pending
