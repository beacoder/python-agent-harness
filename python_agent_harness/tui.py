"""Rich TUI for the agent harness.

Layout: conversation panel + status bar + input line.  The agent loop
runs in a worker thread; the main thread renders with rich Live and
services interactive questions (Question tool, PlanExit confirmation).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shlex
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich import box
from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config
from .agent import run_agent_loop
from .agent_session import AgentSession
from .commands import find_command
from .diffrender import render_diff
from .models import Message
from .prompts import _is_mode_reminder_text
from .session_store import (
    SessionStore,
    escape_role_headers,
    split_role_header,
    title_from_filename,
    unescape_role_header,
)

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# message-body colors - distinct from tool colors (tool calls are magenta,
# tool results dim) so roles never blend into tool activity
USER_STYLE = "cyan"
ASSISTANT_STYLE = "green"


def _tail_lines(text: str, n: int) -> str:
    """Keep the last N lines of TEXT, marking the cut with an ellipsis."""
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "…\n" + "\n".join(lines[-n:])


def _tail_chars(text: str, n: int) -> str:
    """Keep the last N chars of TEXT, marking the cut with an ellipsis."""
    if len(text) <= n:
        return text
    return "…" + text[-n:]


def _head_lines(text: str, n: int) -> str:
    """Keep the first N lines of TEXT, marking the cut with a count."""
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[:n]) + f"\n… [{len(lines) - n} more lines]"


def _head_chars(text: str, n: int) -> str:
    """Keep the first N chars of TEXT, marking the cut with an ellipsis."""
    if len(text) <= n:
        return text
    return text[:n] + "…"


def _tool_result_preview(content: str) -> str:
    """Preview of a tool result: first N lines, capped at N chars.

    Tool results (file reads, command output) can be huge — showing only
    a few lines keeps the TUI fast and readable.  The beginning is kept
    (it carries the result/error); the cut is marked explicitly.
    """
    from . import config

    preview = _head_lines(content, config.TOOL_RESULT_PREVIEW_LINES)
    return _head_chars(preview, config.TOOL_RESULT_PREVIEW_CHARS)


# the completion-check filter: a [FINAL CHECK] header followed by the
# Goal:/Status:/Evidence: labels (anywhere in the block, any lines)
_FINAL_CHECK_RE = re.compile(r"\[FINAL CHECK\].*Goal:.*Status:.*Evidence:", re.DOTALL)

def _is_injected_user_text(text: str) -> bool:
    """True for harness-injected user messages (not user input).

    Live sessions flag injected messages directly; these content checks
    cover restored sessions, where the flag is lost in the markdown
    round-trip: the completion nudge, the <system-reminder>-wrapped
    plan/build-switch prompts (plan.md, plan-mode.md, build-switch.md
    all start with the reminder tag), and the plan-exit approval notice.
    """
    if text == config.NUDGE_MESSAGE:
        return True
    return _is_mode_reminder_text(text)


def _strip_final_check(text: str) -> str:
    """Drop a completion-check block from an assistant reply.

    The task-completion rules make the model end with a [FINAL CHECK]
    block (Goal / Status / Evidence) — verification bookkeeping, not
    content the user wants to read.  The filter is the
    "[FINAL CHECK].*Goal:.*Status:.*Evidence:" pattern: everything
    from the header onward is dropped.

    The block is hidden even when it is the reply's ONLY content —
    check-only replies never render.  Replies without the header are
    untouched.  The agent loop still produces and stores the message
    unchanged; this only trims it from the TUI display.
    """
    m = _FINAL_CHECK_RE.search(text)
    if m is not None:
        return text[: m.start()].rstrip()
    return text


def _strip_reasoning(text: str, reasoning: str) -> str:
    """Remove the leading REASONING block from TEXT, or TEXT unchanged.

    Reasoning content is streamed before the answer, so it forms the
    leading part of the stored message content.  The TUI collapses it
    to a marker once the stream is done, so it stops eating the
    visible-row budget; the stored message is never modified.
    """
    if not reasoning:
        return text
    if text.startswith(reasoning):
        return text[len(reasoning) :]
    stripped = text.lstrip()
    if stripped.startswith(reasoning):
        return stripped[len(reasoning) :]
    return text


def _history_path() -> str:
    d = config.SESSION_DIR / "python-agent-harness"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "input_history")


def _make_key_bindings() -> KeyBindings:
    """Esc+Enter (or Alt+Enter) submits; plain Enter inserts a newline.

    Tab triggers completion explicitly (first Tab inserts the common
    part / opens the menu, further Tabs cycle), Shift+Tab cycles
    backwards — prompt_toolkit's defaults don't reliably bind Tab in
    every mode/version.
    """
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("c-i")
    def _complete(event: Any) -> None:
        b = event.current_buffer
        if b.complete_state:
            b.complete_next()
        else:
            b.start_completion(insert_common_part=True)

    @kb.add("s-tab")
    def _complete_backward(event: Any) -> None:
        b = event.current_buffer
        if b.complete_state:
            b.complete_previous()
        else:
            b.start_completion(select_first=True)

    return kb


def _make_prompt_session(
    history: FileHistory, completer: Completer, **kwargs: Any
) -> PromptSession:
    """Create the TUI's input session.

    ``complete_while_typing`` is off on purpose: it races with Tab's
    ``start_completion`` (a keystroke-triggered completion can create
    the completion state just before the Tab-triggered task runs, which
    then bails out without inserting the common part).  Tab must be the
    single, deterministic trigger.
    """
    return PromptSession(
        history=history,
        key_bindings=_make_key_bindings(),
        completer=completer,
        complete_while_typing=False,
        multiline=True,
        enable_suspend=True,
        **kwargs,
    )


SLASH_COMMANDS = [
    "/plan",
    "/build",
    "/init",
    "/review",
    "/explain",
    "/compact",
    "/save",
    "/summary",
    "/sessions",
    "/restore",
    "/clear",
    "/model",
    "/exit",
]


def _custom_slash_commands() -> list[str]:
    from .commands import load_custom_commands

    return sorted(f"/{c.name}" for c in load_custom_commands())


class SlashCompleter(Completer):
    """Tab-completion for the input line.

    - A first token starting with ``/`` completes against the known
      slash commands (builtins + custom commands from
      prompts/commands/*.md); if no command matches, it is treated as
      an absolute path.
    - After a slash command's space, Tab completes paths relative to
      the session's project dir (absolute and ``~`` paths work too).
    - Any other ``~``-prefixed or ``/``-containing token (e.g.
      ``~/wor``, ``docs/``) completes as a path: ``~`` against $HOME,
      otherwise relative to the project dir. Plain words without ``/``
      are left alone.
    - Directories get a trailing ``/`` so repeated Tab drills deeper;
      ``~`` alone completes to ``~/``.
    """

    def __init__(self, get_project_dir: Callable[[], str]) -> None:
        self.get_project_dir = get_project_dir

    def _slash_commands(self) -> list[str]:
        return sorted(set(SLASH_COMMANDS + _custom_slash_commands()))

    def _complete_paths(self, arg: str) -> Iterable[Completion]:
        expanded = os.path.expanduser(arg)
        if not arg:
            directory, prefix = self.get_project_dir() or os.getcwd(), ""
        elif expanded.endswith(os.sep):
            base = (
                expanded
                if os.path.isabs(expanded)
                else os.path.join(self.get_project_dir() or os.getcwd(), expanded)
            )
            directory, prefix = base, ""
        elif os.path.isdir(expanded):
            # "~" or an existing dir without a trailing slash: complete
            # the trailing slash itself (bash-style), not its siblings.
            yield Completion(text="/", start_position=0, display=arg + "/")
            return
        else:
            base = (
                expanded
                if os.path.isabs(expanded)
                else os.path.join(self.get_project_dir() or os.getcwd(), expanded)
            )
            directory, prefix = os.path.dirname(base), os.path.basename(base)
        try:
            entries = sorted(os.listdir(directory or "."))
        except OSError:
            return
        for name in entries:
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if os.path.isdir(os.path.join(directory, name)):
                suffix += "/"
                display = name + "/"
            else:
                display = name
            # start_position=0 appends at the cursor; the typed prefix is
            # already in the buffer, so only the remaining suffix is inserted.
            yield Completion(text=suffix, start_position=0, display=display)

    def get_completions(self, document: Any, complete_event: Any):
        text = document.text_before_cursor
        if text.startswith("/"):
            if " " not in text:
                cmds = [c for c in self._slash_commands() if c.startswith(text)]
                for cmd in cmds:
                    yield Completion(cmd, start_position=-len(text))
                if cmds:
                    return
                yield from self._complete_paths(text)  # absolute path
                return
            arg = text.split(" ", 1)[1]
            yield from self._complete_paths(arg)
            return
        token = text.rsplit(" ", 1)[-1] if " " in text else text
        if token.startswith("~") or "/" in token:
            yield from self._complete_paths(token)


class UiQuestion:
    def __init__(
        self,
        prompt: str,
        multiple: bool = False,
        options: list[str] | None = None,
        custom: bool = True,
        keys: list[str] | None = None,
    ) -> None:
        self.prompt = prompt
        self.multiple = multiple
        self.options = options or []
        self.custom = custom
        # keyed choices (e.g. ["y", "n"] for a confirm): render the
        # options as a keyed list and resolve typed keys to labels,
        # instead of the numbered-list style of the Question tool
        self.keys = keys or []
        self.answer: str | None = None
        self.event = threading.Event()


def _resolve_keyed_choice(answer: str, options: list[str], keys: list[str]) -> str:
    """Map bare keys in ANSWER to the matching option label.

    Comma-separated keys pick several options (multiple select);
    non-key tokens pass through unchanged as free-text answers.
    """
    if not options or not keys or not answer.strip():
        return answer
    resolved: list[str] = []
    for part in answer.split(","):
        part = part.strip()
        if part.lower() in keys:
            resolved.append(options[keys.index(part.lower())])
            continue
        resolved.append(part)
    return ", ".join(resolved)


def _resolve_numbered_choice(answer: str, options: list[str]) -> str:
    """Map bare numbers in ANSWER (1-based) to the matching option label.

    Comma-separated numbers pick several options (multiple select);
    non-numeric tokens pass through unchanged as free-text answers;
    out-of-range numbers are kept as typed.  Empty answers stay empty.
    """
    if not options or not answer.strip():
        return answer
    resolved: list[str] = []
    for part in answer.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(options):
                resolved.append(options[idx - 1])
                continue
        resolved.append(part)
    return ", ".join(resolved)


class Tui:
    def __init__(self, session: AgentSession, console: Console | None = None) -> None:
        self.session = session
        self.console = console or Console()
        self.stream_text = ""
        self.lock = threading.Lock()
        self.question: UiQuestion | None = None
        self.agent_running = False
        self.status = " idle"
        self._current_tool = ""
        self.run_seq = 0
        self._restore: Callable[[], None] | None = None
        self.conversation_history: list[Message] = []
        # Index into ``session.last_messages`` where the current round
        # begins: the live panel renders only from here on (the latest
        # round of interactions), while the end-of-run scrollback dump
        # prints the full conversation.  0 means "show everything" —
        # the default until the first run sets a boundary.
        self.round_start = 0
        # Text the user submitted for the current round, shown as a live
        # user row until it is mirrored into ``session.last_messages``
        # (which only happens once the first assistant response lands).
        self.round_user_text = ""
        self._data_event = threading.Event()
        self._history_cache: list[Any] | None = None
        self._history_dirty = True
        # wall-clock start of each agent run (one per user round), used
        # by the end-of-run dump to timestamp the round separators
        self._round_times: list[float] = []
        # wall-clock start of the current run, used to report the total
        # time spent once the run finishes
        self._run_start: float | None = None
        self.prompt_session: PromptSession = _make_prompt_session(
            FileHistory(_history_path()),
            SlashCompleter(lambda: str(self.session.project_dir)),
        )

        session.on_delta = self._on_delta
        session.notify_fn = self._on_notify
        session.log_fn = self._on_log
        session.confirm_fn = self._ui_confirm
        session.ask_fn = self._ui_ask

    # ------------------------------------------------------------------
    # session callbacks (called from the worker thread)
    # ------------------------------------------------------------------
    def _on_delta(self, text: str) -> None:
        with self.lock:
            self.stream_text += text
            # bound the buffer so every frame's tail-slicing is
            # constant-time, no matter how long the generation runs
            if len(self.stream_text) > 100_000:
                self.stream_text = self.stream_text[-100_000:]
        # wake the render loop immediately: streaming text pushes the
        # display without waiting for the next fixed tick
        self._data_event.set()

    def _on_notify(self, kind: str, data: Any = None) -> None:
        if kind == "tool_start":
            # Tool execution is starting: clear the stale stream text
            # (the assistant message is committed to messages and will
            # appear in history once delivered) and show which tools are
            # running, so the display stays alive during long operations.
            with self.lock:
                self.stream_text = ""
            names = data if isinstance(data, list) else []
            label = ", ".join(names) if names else "tools"
            self._current_tool = label
            self.status = f" ⏳ {label}"
            self._history_dirty = True
        elif kind == "tool_running":
            # Per-tool notification: update the current tool name shown
            # beside the spinner as each sync tool starts executing.
            name = data if isinstance(data, str) else ""
            self._current_tool = name
            self.status = f" ⏳ {name}" if name else " ⏳ tools"
        elif kind == "tools":
            self._current_tool = ""
            self.status = " running tools"
            # tool round finished: session.last_messages now contains the
            # tool-call + result rows — rebuild the cached history so
            # they show up live instead of after the run ends
            self._history_dirty = True
        elif kind == "compact":
            self.status = " compacted"
            self._history_dirty = True
        elif kind == "retry":
            # A connection error mid-stream: the client discarded the
            # partial response and is retrying on a fresh connection —
            # drop the partial stream text so the restarted stream
            # doesn't duplicate it on screen.
            with self.lock:
                self.stream_text = ""
            self.status = " connection lost — retrying"
        elif kind == "todos":
            # TodoWrite updated the task list: the cached history rows
            # (which include the Todos panel) must be rebuilt
            self._history_dirty = True
        elif kind == "error":
            self.status = " error"
        elif kind == "save-error":
            self.status = " auto-save failed"
        else:
            self.status = " running"
        self._data_event.set()

    def _on_log(self, msg: str) -> None:
        self.status = f" {msg}"

    def _ui_confirm(self, prompt: str) -> bool:
        """PlanExit confirmation: same look as the Question tool, but a
        y/n keyed choice list instead of numbers (two choices only)."""
        q = UiQuestion(
            prompt,
            options=list(config.PLAN_EXIT_OPTIONS),
            keys=["y", "n"],
            custom=False,
        )
        answer = self._ask_sync(q).strip().lower()
        # resolved answers arrive as the option label; legacy free-text
        # (y/yes/a/1/true) keeps working for muscle memory
        return answer == config.PLAN_EXIT_OPTIONS[0].lower() or answer in (
            "y",
            "yes",
            "a",
            "true",
            "1",
        )

    def _ui_ask(self, questions: list[dict]) -> str:
        lines = []
        for q in questions:
            prompt = q.get("question", "")
            options = q.get("options") or []
            multiple = bool(q.get("multiple"))
            custom = q.get("custom", True)
            ui_q = UiQuestion(
                prompt,
                multiple=multiple,
                options=list(options),
                custom=custom,
            )
            answer = self._ask_sync(ui_q)
            if multiple:
                answer = ", ".join(a.strip() for a in answer.split(",") if a.strip())
            lines.append(f'"{prompt}" = "{answer}"')
        return "\n".join(lines) if lines else "Unanswered"

    def _ask_sync(self, q: UiQuestion) -> str:
        """Block the worker thread until the main thread answers.

        Cancel-aware: the wait polls the session cancel event, so a
        Ctrl-C outside the answer prompt (e.g. during the render loop)
        unblocks the worker immediately instead of wedging it until a
        question is answered.  A cancelled run returns an empty answer.
        """
        self.question = q
        cancel = getattr(self.session, "cancel_event", None)
        while not q.event.wait(0.1):
            if cancel is not None and cancel.is_set():
                return ""
        return q.answer or ""

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _build_history_rows(self, full: bool = False) -> list[Any]:
        """Rows for the stored conversation (messages + todos). No stream.

        FULL=True renders the WHOLE conversation with uncapped
        user/assistant bodies (the post-run scrollback dump, where every
        round must be readable).  FULL=False renders only the LATEST
        round — the messages from ``self.round_start`` onward — with the
        tail caps, so the live panel focuses on the current interaction
        instead of replaying every previous round.
        """
        rows: list[Any] = []
        calls_by_id: dict[str, Any] = {}
        all_messages = self.session.last_messages or []
        # tool-call lookup spans the whole conversation so a diff still
        # resolves even if its call landed in an earlier round
        for m in all_messages:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    calls_by_id[tc.id] = tc

        if full:
            messages = all_messages
        else:
            # clamp: compaction / clear / restore can shrink
            # last_messages below the recorded boundary
            start = min(self.round_start, len(all_messages))
            messages = all_messages[start:]
            # before the round's user message is mirrored into
            # last_messages (during the first assistant stream), show
            # the text the user just submitted so the round isn't blank
            if len(all_messages) <= self.round_start and self.round_user_text:
                body = _tail_lines(self.round_user_text, 12)
                if body.strip():
                    rows.append(Markdown(f"**user:** {body}", style=USER_STYLE))

        for m in messages:
            # compacted summaries live in the user turn (system prompt is
            # separate); match on content so both live sessions and
            # restored files (role lost in the markdown round-trip) render
            if m.text().startswith("**[Compacted Summary]**"):
                rows.append(Text("📦 " + _tail_chars(m.text(), 200), style="dim italic"))
                continue
            if m.role == "user":
                # harness-injected messages (completion nudge,
                # plan/build-mode prompts, build-switch notices,
                # plan-exit approval): they drive the agent loop, but the
                # user never typed them — keep them out of the
                # conversation panel.  The flag covers live sessions; the
                # content checks catch restored sessions, where the flag
                # is lost in the markdown round-trip.
                if m.injected or _is_injected_user_text(m.text()):
                    continue
                body = m.text() if full else _tail_lines(m.text(), 12)
                if body.strip():
                    rows.append(Markdown(f"**user:** {body}", style=USER_STYLE))
            elif m.role == "assistant":
                body = m.text()
                collapsed_reasoning = False
                if isinstance(m.reasoning, str) and m.reasoning:
                    stripped = _strip_reasoning(body, m.reasoning)
                    if stripped != body:
                        body = stripped
                        collapsed_reasoning = True
                body = _strip_final_check(body)
                if not full:
                    body = _tail_lines(body, 12)
                if collapsed_reasoning:
                    # the reasoning streamed live while it was being
                    # produced; once it is done it collapses to a marker
                    # so it doesn't eat the visible-row budget
                    rows.append(Text("reasoning ...", style="dim"))
                if m.tool_calls:
                    for tc in m.tool_calls:
                        args = tc.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                args = {}
                        if isinstance(args, dict):
                            params = " ".join(
                                f"{k}={v!r}"
                                for k, v in args.items()
                                if k != "content" and len(repr(v)) < 80
                            )
                        else:
                            params = ""
                        label = f"tool: {tc.name}({params})" if params else f"tool: {tc.name}"
                        rows.append(Text(f"▶ {label}", style="magenta"))
                if body.strip():
                    rows.append(Markdown(f"**assistant:** {body}", style=ASSISTANT_STYLE))
            elif m.role == "tool":
                preview = _tool_result_preview(m.text())
                name = (m.name or "tool").lower()
                call = calls_by_id.get(m.tool_call_id)
                # tool failures surface as "Error: ..." results (agent
                # containment, missing args, MCP-reported errors)
                failed = (m.text() or "").startswith("Error")
                marker = "✗" if failed else "✓"
                marker_style = "bold red" if failed else "green"
                elapsed = ""
                if call is not None and call.elapsed is not None:
                    elapsed = f" ({call.elapsed:.1f}s)"
                row = Text(style="dim")
                row.append(f"{marker} {name} result{elapsed}:", style=marker_style)
                row.append(f"\n{preview}")
                rows.append(row)
                if call is not None and call.diff:
                    rows.append(render_diff(call.diff))
        return rows

    def _todos_panel(self) -> Group | None:
        """Todos section — rebuilt every frame (not cached), so a
        TodoWrite call shows up immediately even mid-run."""
        if not self.session.todos:
            return None
        t = Table.grid(padding=(0, 1))
        for todo in self.session.todos[-8:]:
            status = todo.get("status", "")
            mark = {"completed": "✅", "in_progress": "⏳", "pending": "⬜"}.get(status, "•")
            t.add_row(mark, todo.get("content", ""))
        return Group(Text("Todos", style="bold"), t)

    def _history_rows(self) -> list[Any]:
        """Cached history rows; rebuilt only when the conversation changes.

        During streaming only the stream row changes, so we must NOT
        rebuild (and re-parse Markdown for) the whole conversation every
        frame — that cost is what made the scroll lag behind the text.
        """
        if self._history_dirty or self._history_cache is None:
            self._history_cache = self._build_history_rows()
            self._history_dirty = False
        return list(self._history_cache)

    def _stream_row(self) -> Text | None:
        """Live stream row (cheap Text, tail-capped)."""
        with self.lock:
            stream = self.stream_text
        stream = _strip_final_check(stream)
        if not stream:
            return None
        cap = self._visible_row_cap()
        width = getattr(self.console, "width", None) or 80
        lines = max(3, cap - 3)
        preview = _tail_lines(stream, lines)
        preview = _tail_chars(preview, lines * max(1, width))
        row = Text(f"assistant: {preview}", style=ASSISTANT_STYLE)
        # blinking block cursor, 2 Hz phase (same clock as the spinner)
        if int(time.time() * 2) % 2 == 0:
            row.append("▍")
        return row

    def _render_conversation(self) -> Group | Text:
        rows = self._history_rows()
        stream_row = self._stream_row()
        if stream_row is not None:
            rows.append(stream_row)
        rows = self._apply_budget(rows)
        return Group(*rows) if rows else Text("(empty)")

    def _apply_budget(self, rows: list[Any]) -> list[Any]:
        """Keep the NEWEST rows that fit the visible terminal area.

        rich's Live crops a too-tall frame from the bottom, which would
        hide exactly the rows that matter (the latest progress), so we
        drop old rows first and keep the newest content on screen.
        """
        width = getattr(self.console, "width", None) or 80
        budget = self._visible_row_cap()
        kept: list[Any] = []
        for row in reversed(rows):
            est = self._est_lines(row, width)
            if budget - est < 0:
                continue  # older rows are dropped once the budget is spent
            kept.append(row)
            budget -= est
        return kept[::-1]

    def _round_time(self, round_no: int) -> str | None:
        """Formatted HH:MM:SS start time of round N, if recorded.

        Live runs record their start times here; for restored sessions
        the times come back from the persisted session metadata
        (``store.round_times``).
        """
        idx = round_no - 1
        times = self._round_times or self.session.store.round_times
        if 0 <= idx < len(times):
            return time.strftime("%H:%M:%S", time.localtime(times[idx]))
        return None

    def _dump_conversation(self) -> None:
        """Print the full conversation into the terminal scrollback.

        The Live display overwrites its frames in place and the frame
        budget drops anything that doesn't fit the visible area, so the
        conversation never reaches the terminal's scrollback during a
        run.  When the run finishes, print it again as plain lines so
        the user can scroll back through everything that happened.

        Each user message starts a new round; rounds after the first
        are separated by a rule line (with the round's start time when
        it was recorded live).

        Unlike the live panel, message bodies are printed UNCAPPED
        (``full=True``): the live panel tail-caps long replies to the
        newest lines, but the dump is where the whole answer becomes
        readable — a capped dump would hide the head of long
        summaries exactly like the live view.
        """
        rows = self._build_history_rows(full=True)
        if not rows:
            return
        self.console.print()
        self.console.print("[dim]— full conversation —[/dim]")
        round_no = 0
        for row in rows:
            # a displayed user row starts a new round (injected prompts
            # are already filtered out of the rows)
            if isinstance(row, Markdown) and getattr(row, "markup", "").startswith("**user:**"):
                round_no += 1
                if round_no > 1:
                    title = f"round {round_no}"
                    ts = self._round_time(round_no)
                    if ts is not None:
                        title += f" · {ts}"
                    self.console.rule(title, style="dim")
            self.console.print(row)
        # report the total time spent on this run, mirroring the per-tool
        # elapsed shown on tool results
        if self._run_start is not None:
            elapsed = time.time() - self._run_start
            self.console.print(f"[green]✓ time spent ({elapsed:.1f}s):[/green]")

    def _visible_row_cap(self) -> int:
        """Max conversation rows that fit the visible terminal area.

        Uses the live terminal height when known (rich reports None for
        non-terminals, e.g. tests), reserving lines for the status bar,
        the input prompt and the pinned Todos section when visible.
        """
        height = getattr(self.console, "height", None)
        if not height or height <= 0:
            return 60
        # reserve: status bar (1) + input prompt (1)
        # + the pinned Todos section when visible (its title line + rows)
        reserved = 2
        if self.session.todos:
            reserved += min(len(self.session.todos), 8) + 1
        return max(5, height - reserved)

    @staticmethod
    def _est_lines(row: Any, width: int) -> int:
        """Rough wrapped-line estimate for a row (used for the budget)."""
        if isinstance(row, Text):
            return max(1, math.ceil(len(row.plain) / max(1, width)))
        if isinstance(row, Markdown):
            return max(1, math.ceil(len(row.markup) / max(1, width)))
        if isinstance(row, Panel):
            inner = getattr(row.renderable, "renderables", None)
            return 3 + (len(inner) if isinstance(inner, (list, tuple)) else 1)
        return 1

    def _render_frame(self) -> Group:
        """Full frame: status bar + Todos pinned on top, conversation below.

        The status bar and the Todos panel are placed FIRST so they stay
        visible no matter how tall the conversation gets — the Todos list
        is pinned like a second mode line instead of competing with the
        conversation rows for the visible budget.
        """
        parts: list[Any] = [self._status_bar()]
        todos = self._todos_panel()
        if todos is not None:
            parts.append(todos)
        parts.append(self._render_conversation())
        return Group(*parts)

    def _status_bar(self) -> Text:
        mode = self.session.plan_mode.mode.value
        mode_style = "bold yellow" if mode == "plan" else "bold green"
        ratio = self.session.context_ratio
        ctx = ""
        if ratio is not None:
            pct = round(ratio * 100)
            trigger = round(config.CONTEXT_TRIGGER * 100)
            filled = round(ratio * 10)
            bar = "▓" * filled + "░" * (10 - filled)
            ctx = f" [Ctx:{bar} {pct}%/{trigger}%]"
        t = Text()
        t.append(f" [{mode.upper()}]", style=mode_style)
        if ctx:
            over = ratio is not None and ratio >= config.CONTEXT_TRIGGER
            t.append(ctx, style="bold" if over else "")
        if getattr(self.session, "_save_error", None):
            t.append(" [!save]", style="red bold")
        if self.agent_running:
            frame = SPINNER_FRAMES[int(time.time() * 10) % len(SPINNER_FRAMES)]
            t.append(f" {frame}", style="bold cyan")
            if self._current_tool:
                t.append(f" {self._current_tool}", style="bold cyan")
        elif self.question is not None:
            t.append(" ❓", style="yellow")
        t.append(self._fit_status(cell_len(str(t)), self.status), style=self._status_style())
        return t

    def _fit_status(self, used: int, msg: str) -> str:
        """Fit the status message on the status-bar line: newlines are
        flattened and the message is truncated with an ellipsis only
        when it would overflow the terminal width."""
        msg = " ".join(msg.splitlines())
        width = self.console.width or 80
        avail = max(1, width - used - 1)
        if cell_len(msg) <= avail:
            return msg
        out = ""
        used_cells = 0
        for ch in msg:
            w = cell_len(ch)
            if used_cells + w > avail - 1:
                break
            out += ch
            used_cells += w
        return out + "…"

    def _status_style(self) -> str:
        """Status-bar color by state: errors red, activity cyan, idle dim."""
        s = self.status
        if "error" in s or "failed" in s:
            return "bold red"
        if " ⏳" in s or " running" in s or "retrying" in s:
            return "cyan"
        return "dim"

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.console.print(
            Panel(
                Text.from_markup(
                    "[bold]Commands:[/bold] /plan /build /init /review /explain "
                    "/compact /save /summary /sessions /restore /help /exit\n\n"
                    "Ctrl-C cancels the current execution (the app stays open); "
                    "Ctrl-D or /exit quits.\n"
                    "Type a message — Enter for a new line, Esc then Enter "
                    "(or Alt+Enter) to submit. Up/Down recall history.\n\n"
                    "[dim]Type [bold]/help[/bold] for the full command reference.[/dim]"
                ),
                title="[bold cyan]python-agent-harness — interactive AI coding agent[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
        if config.LLM_LOG_ENABLED:
            self.console.print(f"[dim]LLM logs: {self.session.client.log_path}[/dim]")
        while True:
            try:
                if self.question is not None:
                    self._ask_question_blocking()
                    continue
                self.console.print(self._status_bar())
                self._flush()
                text = self._read_multiline()
                if text is None:
                    break
                if not text.strip():
                    continue
                if text.startswith("/"):
                    if self._handle_slash(text):
                        break
                    continue
                self._start_agent(text)
            except KeyboardInterrupt:
                # stray Ctrl-C outside input/execution: stay in the app
                self.console.print("[dim]cancelled — Ctrl-D or /exit to quit[/dim]")

    def _ask_question_blocking(self) -> None:
        q = self.question
        if q is None:
            return
        self.console.print(self._render_frame())
        self.console.print()
        self._flush()
        options = q.options or []
        keys = q.keys or []
        if keys and options and len(keys) == len(options):
            # keyed choices (e.g. y/n confirm): type the key to pick —
            # same list look as the Question tool, keys instead of numbers
            self.console.print(Text(q.prompt))
            for key, opt in zip(keys, options, strict=True):
                line = Text(f"  {key}) ", style="cyan")
                line.append(opt)
                self.console.print(line)
            hint = "Enter keys, comma-separated" if q.multiple else "Enter a key"
            if q.custom:
                hint += ", or type your own answer"
            self.console.print(f"[dim]{hint}[/dim]")
            prompt = "> "
        elif options:
            # option labels get a numbered list: type the number to pick
            self.console.print(Text(q.prompt))
            for i, opt in enumerate(options, 1):
                line = Text(f"  {i}) ", style="cyan")
                line.append(opt)
                self.console.print(line)
            hint = "Enter numbers, comma-separated" if q.multiple else "Enter a number"
            if q.custom:
                hint += ", or type your own answer"
            self.console.print(f"[dim]{hint}[/dim]")
            prompt = "> "
        else:
            prompt = q.prompt + " > "
        try:
            with patch_stdout():
                answer = self.prompt_session.prompt(prompt, multiline=False)
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if keys:
            q.answer = _resolve_keyed_choice(answer, options, keys)
        else:
            q.answer = _resolve_numbered_choice(answer, options)
        q.event.set()
        self.question = None
        self._data_event.set()  # re-render promptly after the answer

    def _input_prompt(self) -> FormattedText:
        """Styled input prompt: short model name + caret.

        Shows the model actually in use (the part after the last '/',
        e.g. ``deepseek-ai/deepseek-flash-v4`` → ``deepseek-flash-v4``)
        so the active model stays visible while typing.
        """
        model = self.session.model or ""
        short = model.rsplit("/", 1)[-1] if "/" in model else model
        return FormattedText(
            [
                ("bold cyan", f"{short} " if short else ""),
                ("ansibrightblack", "> "),
            ]
        )

    def _read_multiline(self) -> str | None:
        try:
            with patch_stdout():
                text = self.prompt_session.prompt(self._input_prompt())
        except EOFError:
            # Ctrl-D: quit
            return None
        except KeyboardInterrupt:
            # Ctrl-C: cancel this input, stay in the app
            self.console.print("[dim]input cancelled[/dim]")
            return ""
        return text

    def _start_agent(
        self,
        text: str,
        system: str | None = None,
        restore: Callable[[], None] | None = None,
    ) -> None:
        """Run the agent loop on TEXT in a worker thread.

        SYSTEM overrides the session's system prompt for this run only.
        RESTORE (if given) runs when the run finishes — used by the
        slash commands to put back state they borrowed (e.g. project_dir).
        """
        self.stream_text = ""
        self.status = " running"
        self._current_tool = ""
        # Mark where this round begins in the shared history: the live
        # panel renders only from here on (the latest round), while the
        # end-of-run dump prints the full conversation.  Captured before
        # the run so it points just past the previous round's messages —
        # the new user message will be the first mirrored row.
        self.round_start = len(self.session.last_messages or [])
        self.round_user_text = text
        self._round_times.append(time.time())
        self._run_start = time.time()
        # keep the persisted metadata in sync so auto-save /save
        # capture the round timestamps (restore reads them back)
        self.session.store.round_times = list(self._round_times)
        # A new top-level run starts here: drop any todo list left over
        # from a previous run so a finished task's todos don't stay
        # pinned into the next task.
        self.session.clear_todos()
        # A new top-level run starts here: invalidate any worker still
        # unwinding from a previous run — from this point on it is stale
        # and must never touch shared state.  Bump before clearing the
        # event so there is no instant where an old worker sees "not
        # cancelled".
        self.session.run_generation += 1
        self.session.cancel_event.clear()
        self._data_event.clear()
        self._history_dirty = True
        self.run_seq += 1
        seq = self.run_seq
        self._restore = restore
        self.agent_running = True
        worker = threading.Thread(
            target=self._run_agent, args=(text, seq, system, restore), daemon=True
        )
        worker.start()
        cancelled = False
        try:
            if self.console.is_dumb_terminal:
                cancelled = self._run_dumb(worker)
            else:
                cancelled = self._run_live(worker)
        except KeyboardInterrupt:
            # Ctrl-C during execution: cancel the run, keep the app open.
            # The worker is a daemon and cancel-aware; don't join it — a
            # hung HTTP read may take a while, and the UI must return to
            # the input prompt immediately.
            cancelled = True
            self.session.cancel()
            if self.question is not None:
                # a pending question wedges the worker in its wait: the
                # run is cancelled, so release it now — it must not be
                # re-prompted after the run is over
                self.question.answer = ""
                self.question.event.set()
                self.question = None
            if self._restore is not None:
                # release state the cancelled run borrowed (e.g. a slash
                # command's project dir) now: the worker may finish late
                # (stale) and its own finally must not touch it then
                self._restore()
                self._restore = None
            self.console.print("\n[dim]execution cancelled — add more messages or /exit[/dim]")
            self._flush()
        finally:
            self.agent_running = False
            if not cancelled:
                self.console.print()
                self._flush()

    def _run_live(self, worker: threading.Thread) -> bool:
        """Live-based display (real terminal). Returns True if cancelled.

        Event-driven: the render loop blocks on `_data_event` and wakes
        the moment new stream text arrives, so the text pushes the
        scroll immediately instead of waiting for a fixed tick.  The
        short timeout keeps the spinner animating between data bursts.
        """
        with Live(
            self._render_frame(),
            console=self.console,
            refresh_per_second=30,
            screen=False,
        ) as live:
            while worker.is_alive():
                if self.question is not None:
                    live.stop()
                    self._ask_question_blocking()
                    live.start()
                    continue
                self._data_event.wait(timeout=0.1)
                self._data_event.clear()
                live.update(self._render_frame())
                self._flush()
            live.update(self._render_frame())
            self._flush()
        # run finished: the last Live frame stays on screen, but the
        # in-place redraws never reached the scrollback — print the
        # full conversation so the user can scroll back through it
        self._dump_conversation()
        return False

    def _run_dumb(self, worker: threading.Thread) -> bool:
        """Dumb-terminal fallback: print each frame as a normal line.

        rich's Live intentionally renders nothing on dumb terminals
        (TERM unset/"dumb"), so without this the status bar and spinner
        would never appear there.  Same event-driven wakeup as `_run_live`.
        """
        self.console.print(self._render_frame())
        self._flush()
        while worker.is_alive():
            if self.question is not None:
                self._ask_question_blocking()
                continue
            self._data_event.wait(timeout=0.1)
            self._data_event.clear()
            self.console.print(self._render_frame())
            self._flush()
        self.console.print(self._render_frame())
        self._flush()
        self._dump_conversation()
        return False

    def _flush(self) -> None:
        """Force stdout through so Live frames render in real time.

        rich's Live does not flush after each refresh, and tty stdout is
        line-buffered — without this the status bar/spinner would sit in
        the stdio buffer and only appear once the run ends (or the 8KB
        buffer fills).
        """
        with contextlib.suppress(AttributeError, OSError):
            self.console.file.flush()

    def _run_agent(
        self,
        text: str,
        seq: int,
        system: str | None = None,
        restore: Callable[[], None] | None = None,
    ) -> None:
        try:
            self.conversation_history.append(Message(role="user", content=text))
            run_agent_loop(
                self.session,
                messages=list(self.conversation_history),
                top_level=True,
                system=system or self.session.system_prompt,
            )
            # Only the current run may update shared state: a stale
            # worker (a newer run started — `run_seq` advanced) must not
            # clobber the next run.  A cancelled run with no successor
            # is still current, so it adopts its salvaged partial
            # history and the interrupted turn is not lost (the seq
            # check is the staleness guard; the cancel event no longer
            # blocks the adoption).
            if seq == self.run_seq and self.session.last_messages:
                self.conversation_history = list(self.session.last_messages)
        except Exception as e:  # noqa: BLE001
            if seq == self.run_seq:
                self._on_log(f"agent error: {e}")
        finally:
            # Only the current run may touch shared UI state: a stale
            # worker from a cancelled run that finishes late must not
            # wipe the next run's live stream or fire its restore
            # callback (which could reset e.g. a borrowed project dir
            # while the new run is mid-execution).  The restore for a
            # cancelled run is released by the Ctrl-C handler instead.
            if seq == self.run_seq:
                # the run is done: the final assistant message is now
                # part of the conversation history, so drop the live
                # stream buffer — otherwise the same text renders twice
                # (stream row + history row) and eats the visible-row
                # budget
                with self.lock:
                    self.stream_text = ""
                self._history_dirty = True
                self._data_event.set()
                if restore is not None:
                    restore()

    # ------------------------------------------------------------------
    # slash commands
    # ------------------------------------------------------------------
    def _handle_slash(self, line: str) -> bool:
        cmd, _, arg = line.partition(" ")
        cmd = cmd.strip().lower()
        arg = arg.strip()
        if cmd == "/exit":
            return True
        if cmd == "/plan":
            self.session.switch_to_plan()
            self.console.print(
                "[yellow]Plan mode — read-only; only the plan file is writable.[/yellow]"
            )
        elif cmd == "/build":
            self.session.switch_to_build()
            self.console.print("[green]Build mode.[/green]")
        elif cmd == "/compact":
            self._run_compact()
        elif cmd == "/save":
            path = self.session.store.save(self._conversation_text())
            self.console.print(f"saved: {path}")
        elif cmd == "/summary":
            self._run_summary()
        elif cmd in ("/init", "/review", "/explain"):
            self._run_slash_command(cmd[1:], arg)
        elif cmd == "/sessions":
            self._run_sessions()
        elif cmd == "/restore":
            self._run_restore(arg)
        elif cmd == "/clear":
            # Replacing the conversation is a new generation: invalidate any
            # worker still winding down from a cancelled run, or its
            # salvaged-history commit would resurrect what we just wiped.
            self.session.run_generation += 1
            self.conversation_history = []
            self.session.last_messages = []
            self.session.clear_todos()
            self.console.print("[yellow]Conversation history cleared.[/yellow]")
        elif cmd == "/model":
            self._run_model_command(arg)
        elif cmd == "/help":
            self.console.print(
                "/plan /build /init /review /explain /compact "
                "/save /summary /sessions /restore /clear /model /exit\n"
                "/init [project] [--extra TEXT]       create/update AGENTS.md\n"
                "/review [project] [commit|branch|PR] review code changes\n"
                "/explain [project] [target]          explain code\n"
                "/sessions                            list saved sessions\n"
                "/restore [path | title | --latest | latest]   restore a saved session\n"
                "/model [name]                        switch LLM model profile\n"
                "Ctrl-C cancels the current execution (app stays open); "
                "Ctrl-D or /exit quits.",
                markup=False,
            )
        else:
            self.console.print(f"unknown command: {cmd}")
        return False

    # ------------------------------------------------------------------
    # command slash commands (/init /review /explain)
    # ------------------------------------------------------------------
    @staticmethod
    def _split_args(arg: str) -> list[str]:
        """Split a slash-command argument string (shell-like quoting)."""
        try:
            return shlex.split(arg)
        except ValueError:
            return arg.split()

    def _command_args(self, name: str, arg: str) -> tuple[str | None, str | None]:
        """Parse slash-command args into (project, extra).

        Positional order matches the CLI: [project] then the command's
        argument (commit/branch/PR for review, target for explain,
        --extra TEXT for init).  A lone first token that isn't an
        existing directory is treated as the command's argument instead
        of a project, so `/review main` reviews the branch `main` of the
        current project and `/explain client.py` explains `client.py`.
        """
        parts = self._split_args(arg)
        if not parts:
            return None, None
        if name == "init":
            project = None
            extra = None
            rest = parts
            if rest and rest[0] != "--extra":
                project = rest[0]
                rest = rest[1:]
            if rest:
                if rest[0] != "--extra":
                    return None, None
                extra = " ".join(rest[1:]) if len(rest) > 1 else None
            return project, extra
        first_is_dir = os.path.isdir(os.path.abspath(os.path.expanduser(parts[0])))
        if first_is_dir:
            return parts[0], " ".join(parts[1:]) or None
        return None, " ".join(parts)

    def _run_slash_command(self, name: str, arg: str) -> None:
        """Run a SessionCommand (/init /review /explain) in this session.

        The command's prompt replaces the system prompt for this run
        only; the output streams into the conversation panel and stays
        in history, so the user can follow up on the result.  When a
        different project is given, the session's project dir is
        borrowed for the run (tool cwd) and restored afterwards.
        """
        cmd = find_command(name)
        if cmd is None:
            self.console.print(f"[yellow]unknown command: /{name}[/yellow]")
            return
        project, extra = self._command_args(name, arg)
        if name == "explain" and project is None and not extra:
            self.console.print(
                "[yellow]/explain needs a target — e.g. /explain client.py "
                "or /explain the retry logic in client.py[/yellow]"
            )
            return
        if project:
            project = os.path.abspath(os.path.expanduser(project))
        cwd, prompt, kickoff = cmd.prepare(
            project_dir=project or self.session.project_dir, extra=extra
        )
        if self.conversation_history or self.session.last_messages:
            # The commands' kickoffs ("Proceed with the task described
            # in your instructions.") assume a fresh conversation: the
            # task lives only in the system prompt.  Mid-conversation
            # that reads as a continuation of the previous — already
            # finished — task, so the model keeps working on the old
            # one instead of the command's.  Anchor the new task by
            # naming the command (and target) and marking the earlier
            # conversation as background context only.
            target = f": {extra}" if extra else ""
            kickoff = (
                f"{kickoff.strip()}\n\n"
                f"This is a NEW /{name} request{target} — the messages "
                "above are background context from an earlier task; "
                "follow the NEW instructions in your system prompt."
            )
        # keep the project context + task-completion rules in front of
        # the command's prompt (the "actual agent prompt" for this run)
        from .prompts import assemble_agent_prompt

        system = assemble_agent_prompt(
            cwd, prompt, context_path=self.session._configured_context_path
        )
        prev_project = self.session.project_dir
        if cwd != prev_project:
            self.session.project_dir = cwd

            def _restore() -> None:
                # idempotent: only undo OUR borrow, never clobber a
                # newer run's borrow (or a restore already performed)
                if self.session.project_dir == cwd:
                    self.session.project_dir = prev_project

            restore = _restore
        else:
            restore = None
        if not cmd.allow_planexit:
            # init/review: all tools except PlanExit — hide it for the
            # run (sub-agents share the session registry, so they are
            # covered too) and put it back when the run finishes.
            from .commands import hide_planexit

            planexit_restore = hide_planexit(self.session)
            if planexit_restore is not None:
                prev_restore = restore
                state = {"done": False}

                def _restore() -> None:
                    if state["done"]:
                        return
                    state["done"] = True
                    if prev_restore is not None:
                        prev_restore()
                    planexit_restore()

                restore = _restore
        self.console.print(f"[cyan]/{name}: {kickoff.strip()}[/cyan]")
        self._start_agent(kickoff, system=system, restore=restore)

    def _conversation_text(self) -> str:
        msgs = self.session.last_messages or []
        parts = []
        for m in msgs:
            # escaped: see session_store.escape_role_headers
            body = escape_role_headers(m.text())
            if body:
                parts.append(f"**{m.role}**: {body}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # direct commands (no LLM agent loop)
    # ------------------------------------------------------------------
    def _run_compact(self) -> None:
        """Compact the current conversation directly."""
        ok, msg = self.session.compact_conversation()
        if ok:
            # The shared conversation was replaced: sync the TUI's own
            # history too, or the next run would restart from the old
            # full conversation and immediately re-compact it.
            self.conversation_history = list(self.session.last_messages)
            self._history_dirty = True
        self.console.print(msg)

    def _refresh_model_profiles(self) -> None:
        """Re-read the config file's ``models`` section so profiles
        added/removed while the TUI is running show up on the next
        /model call.  A malformed config keeps the last loaded set."""
        try:
            self.session.model_profiles = config.load_models_config(self.session.config_path)
        except ValueError as e:
            self.console.print(f"[red]{e}[/red]")

    def _model_list_names(self) -> list[str]:
        """Names shown by /model: the original default model (as
        ``default``) followed by every configured profile.

        ``default`` always restores the main ``llm`` settings the
        session started with, so the original model stays reachable
        after switching to profiles — and the listing count stays
        stable across switches.  Used by both the listing and the
        numbered-selection paths so the numbers always match what was
        displayed.
        """
        return ["default", *sorted(self.session.model_profiles.keys())]

    def _model_switch_by_name(self, name: str) -> None:
        """Switch to a named profile (or ``default``) and report the outcome."""
        success, msg = self.session.switch_model(name)
        if success:
            self.console.print(f"[green]{msg}[/green]")
            self._data_event.set()
        else:
            self.console.print(f"[red]{msg}[/red]")

    def _run_model_command(self, arg: str) -> None:
        """Handle /model command for switching LLM profiles."""
        self._refresh_model_profiles()
        if not arg:
            # List all models: the original default plus every profile
            # currently in the config file
            all_names = self._model_list_names()
            current_model = self.session.model or "(unknown)"
            default_model = (self.session.llm_settings or {}).get("model") or current_model
            default_base_url = (self.session.llm_settings or {}).get(
                "base_url"
            ) or self.session.client.base_url

            self.console.print("\n[bold cyan]Available model profiles:[/bold cyan]")
            if not self.session.model_profiles:
                self.console.print(
                    "[yellow]  (none configured — add a 'models' section to use /model)[/yellow]"
                )
            for idx, name in enumerate(all_names, 1):
                if name == "default":
                    marker = " *" if current_model == default_model else ""
                    self.console.print(
                        f"  [cyan]{idx})[/cyan] default{marker} — "
                        f"{default_model} @ {default_base_url}"
                    )
                else:
                    profile = self.session.model_profiles[name]
                    model_name = profile.get("model", "(inherited)")
                    base_url = profile.get("base_url", "(inherited)")
                    marker = " *" if model_name == current_model else ""
                    self.console.print(
                        f"  [cyan]{idx})[/cyan] {name}{marker} — {model_name} @ {base_url}"
                    )

            total_count = len(all_names)
            self.console.print(f"\n[dim]Current: {current_model}[/dim]")
            self.console.print(
                f"[dim]Type a number (1-{total_count}) to switch, or enter a model name directly[/dim]\n"
            )

            # Prompt user for selection
            try:
                selection = input("Select model: ").strip()
                if not selection:
                    return

                # Check if selection is a number
                if selection.isdigit():
                    idx = int(selection) - 1
                    if 0 <= idx < len(all_names):
                        selected = all_names[idx]
                        if selected == "default" and current_model == default_model:
                            self.console.print("[yellow]Already using this model.[/yellow]")
                        else:
                            self._model_switch_by_name(selected)
                    else:
                        self.console.print(
                            f"[red]Invalid selection: {selection}. Choose 1-{len(all_names)}[/red]"
                        )
                else:
                    # Treat as model name
                    self._model_switch_by_name(selection)
            except EOFError:
                pass
            except KeyboardInterrupt:
                self.console.print("\n[dim]cancelled[/dim]")
            return

        # Check if arg is a number
        if arg.strip().isdigit():
            all_names = self._model_list_names()
            idx = int(arg.strip()) - 1
            if 0 <= idx < len(all_names):
                selected = all_names[idx]
                default_model = (self.session.llm_settings or {}).get("model") or (
                    self.session.model or ""
                )
                if selected == "default" and (self.session.model or "") == default_model:
                    self.console.print("[yellow]Already using this model.[/yellow]")
                else:
                    self._model_switch_by_name(selected)
            else:
                self.console.print(
                    f"[red]Invalid selection: {arg}. Choose 1-{len(all_names)}[/red]"
                )
            return
        # Switch to named model
        self._model_switch_by_name(arg)

    def _run_summary(self) -> None:
        """Append a summary of the conversation (tools disabled).

        The summary request is synchronous (non-streaming), so it runs
        in a worker thread like the agent loop: the main thread keeps
        rendering the status bar / spinner while the request is in
        flight, and the result is printed when it lands.
        """
        self.status = " ⏳ summarizing"
        self._current_tool = ""
        self.agent_running = True
        self._data_event.clear()
        result: dict[str, str] = {}

        def worker() -> None:
            try:
                result["msg"] = self.session.summarize_conversation()
            except Exception as e:  # noqa: BLE001 - surfaced to the user
                result["msg"] = f"Summary failed: {e}"

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            if self.console.is_dumb_terminal:
                while thread.is_alive():
                    self._data_event.wait(timeout=0.1)
                    self._data_event.clear()
                    self.console.print(self._status_bar())
                    self._flush()
            else:
                with Live(
                    self._status_bar(),
                    console=self.console,
                    refresh_per_second=30,
                    screen=False,
                ) as live:
                    while thread.is_alive():
                        self._data_event.wait(timeout=0.1)
                        self._data_event.clear()
                        live.update(self._status_bar())
                        self._flush()
        except KeyboardInterrupt:
            self.console.print("\n[dim]summary cancelled — the result may still be appended[/dim]")
            self._flush()
        finally:
            self.agent_running = False
            self.status = ""
            self._data_event.set()
        msg = result.get("msg", "Summary failed: unknown error.")
        self.conversation_history = list(self.session.last_messages)
        self._history_dirty = True
        if msg == "Summary appended.":
            last_msg = self.session.last_messages[-1]
            if last_msg.role == "assistant" and last_msg.content:
                self.console.print(last_msg.content)
            else:
                self.console.print(msg)
        else:
            self.console.print(msg)

    def _run_sessions(self) -> None:
        """List saved sessions with metadata."""
        files = SessionStore.list_sessions()
        if not files:
            self.console.print("[dim]no saved sessions[/dim]")
            return
        for f in files:
            try:
                with open(f, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                continue
            meta = SessionStore.parse_metadata(text)
            basename = os.path.basename(f)
            model = meta.get("gptel-model", "?")
            project = meta.get("python-agent-harness--project-dir", "?")
            self.console.print(f"  {basename:50s}  model={model:20s}  project={project}")

    def _run_restore(self, arg: str) -> None:
        """Restore a saved session into the current TUI.

        Usage: /restore <path>  or  /restore --latest  or  /restore latest
               or  /restore <title>
        Loads the conversation history so the user can continue from
        where they left off.  When the argument is not a file path,
        it is matched as a substring against session filenames/titles
        (case-insensitive).
        """
        path: str | None = None
        if not arg or arg in ("--latest", "latest"):
            path = SessionStore.latest_session()
        elif os.path.isfile(arg):
            path = arg
        else:
            # Try title-based matching: find sessions whose filename
            # contains the argument as a case-insensitive substring
            path = self._find_session_by_title(arg)
        if not path:
            self.console.print(
                "[yellow]no session found "
                "(use /restore <path>, /restore <title>, "
                "/restore --latest, or /restore latest)[/yellow]"
            )
            return
        if not os.path.isfile(path):
            self.console.print(f"[red]file not found: {path}[/red]")
            return
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            self.console.print(f"[red]cannot read {path}: {e}[/red]")
            return
        meta = SessionStore.parse_metadata(text)
        body = SessionStore.strip_metadata(text)
        # Rebuild conversation history from the saved markdown format
        messages = self._parse_saved_body(body)
        # Round timestamps are persisted in the metadata block; restore
        # them so the dump separators keep their HH:MM:SS times.
        round_times = meta.get("python-agent-harness--round-times")
        if round_times:
            try:
                self._round_times = [float(x) for x in round_times.split()]
            except ValueError:
                self._round_times = []
        else:
            self._round_times = []
        self.session.store.round_times = list(self._round_times)
        # Update the session store to point at the restored file
        self.session.store.file_path = path
        title = title_from_filename(path)
        if title:
            self.session.store.title = title
        # Replace conversation history: a new generation.  Invalidate any
        # worker still winding down from a cancelled run so its salvaged
        # history can't clobber the restored session.
        self.session.run_generation += 1
        self.conversation_history = messages
        self.session.last_messages = list(messages)
        self.session.clear_todos()
        self._history_dirty = True
        model = meta.get("gptel-model", "?")
        project = meta.get("python-agent-harness--project-dir", "?")
        self.console.print(
            f"[green]restored:[/green] {os.path.basename(path)} "
            f"(model={model}, project={project}, {len(messages)} messages)"
        )

    @staticmethod
    def _parse_saved_body(body: str) -> list[Message]:
        """Parse a saved session body back into Message objects.

        The save format is markdown with **role**: content blocks
        separated by blank lines.

        ``tool`` blocks are dropped: the saved markdown does not keep
        ``tool_call_id``/``name`` (assistant tool calls are flattened to
        plain text), so a restored ``role="tool"`` message would form an
        API-invalid payload (a tool message with no preceding assistant
        ``tool_calls``).  The following assistant reply already
        summarizes the results, so dropping them loses no essential
        context.

        Body lines that merely look like a block header are escaped by
        the renderer (see `escape_role_headers`) and unescaped here, so
        a message quoting this format no longer splits into extra
        messages.  Sessions saved before escaping existed can still
        split — that ambiguity is in the file, not in this parser.
        """
        messages: list[Message] = []
        current_role: str | None = None
        current_lines: list[str] = []

        for line in body.splitlines():
            # Check for a role header: **user**: ... or **assistant**: ...
            header = split_role_header(line)
            if header is not None:
                role, rest = header
                # Save the previous block (tool blocks are dropped:
                # their tool_call_id/name were not persisted)
                if current_role is not None and current_role != "tool":
                    content = "\n".join(current_lines).strip()
                    if content:
                        messages.append(Message(role=current_role, content=content))
                current_role = role
                current_lines = [unescape_role_header(rest)]
                continue
            current_lines.append(unescape_role_header(line))

        # Don't forget the last block (tool blocks are dropped)
        if current_role is not None and current_role != "tool":
            content = "\n".join(current_lines).strip()
            if content:
                messages.append(Message(role=current_role, content=content))

        return messages

    @staticmethod
    def _find_session_by_title(query: str) -> str | None:
        """Find a session file by title substring (case-insensitive).

        Matches against the full filename, the filename without .md,
        and the derived title.  Returns the most recent match, or None.
        """
        query_lower = query.lower()
        # Strip .md from query if present, for cleaner substring matching
        query_stem = query_lower[:-3] if query_lower.endswith(".md") else query_lower
        files = SessionStore.list_sessions()  # already sorted by mtime desc
        for f in files:
            basename = os.path.basename(f)
            basename_lower = basename.lower()
            # Exact basename match (with or without .md)
            if basename_lower == query_lower or basename_lower == query_lower + ".md":
                return f
            # Substring match against filename (minus .md)
            name_part = basename[:-3] if basename.endswith(".md") else basename
            if query_stem in name_part.lower():
                return f
            # Match against derived title (dashes → spaces)
            title = title_from_filename(f)
            if title and query_stem in title.lower():
                return f
        return None
