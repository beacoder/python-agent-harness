"""Rich TUI for the agent harness.

Layout: conversation panel + status bar + input line.  The agent loop
runs in a worker thread; the main thread renders with rich Live and
services interactive questions (Bash approval, Question tool, PlanExit
confirmation).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config
from .agent import run_agent_loop
from .diffrender import render_diff
from .harness import AgentSession
from .models import Message

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


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


def _history_path() -> str:
    d = config.SESSION_DIR / "python-agent-harness"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "input_history")


def _make_key_bindings() -> KeyBindings:
    """Esc+Enter (or Alt+Enter) submits; plain Enter inserts a newline."""
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    return kb


class UiQuestion:
    def __init__(self, prompt: str, multiple: bool = False,
                 options: list[str] | None = None) -> None:
        self.prompt = prompt
        self.multiple = multiple
        self.options = options or []
        self.answer: str | None = None
        self.event = threading.Event()


class Tui:
    def __init__(self, session: AgentSession, console: Console | None = None) -> None:
        self.session = session
        self.console = console or Console()
        self.stream_text = ""
        self.lock = threading.Lock()
        self.question: UiQuestion | None = None
        self.agent_running = False
        self.status = " idle"
        self.run_seq = 0
        self.conversation_history: list[Message] = []
        self._data_event = threading.Event()
        self._history_cache: list[Any] | None = None
        self._history_dirty = True
        self.prompt_session: PromptSession = PromptSession(
            history=FileHistory(_history_path()),
            key_bindings=_make_key_bindings(),
            multiline=True,
        )

        session.on_delta = self._on_delta
        session.notify_fn = self._on_notify
        session.log_fn = self._on_log
        session.confirm_fn = self._ui_confirm
        session.ask_fn = self._ui_ask
        session.bash_approval_fn = self._ui_bash_approval

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

    def _on_notify(self, kind: str) -> None:
        if kind == "tools":
            self.status = " running tools"
            # tool round finished: session.last_messages now contains the
            # tool-call + result rows — rebuild the cached history so
            # they show up live instead of after the run ends
            self._history_dirty = True
        elif kind == "compact":
            self.status = " compacted"
            self._history_dirty = True
        elif kind == "todos":
            # TodoWrite updated the task list: the cached history rows
            # (which include the Todos panel) must be rebuilt
            self._history_dirty = True
        elif kind == "error":
            self.status = " error"
        else:
            self.status = " running"
        self._data_event.set()

    def _on_log(self, msg: str) -> None:
        self.status = f" {msg[:60]}"

    def _ui_confirm(self, prompt: str) -> bool:
        q = UiQuestion(prompt)
        return self._ask_sync(q) in ("y", "yes", "true", "1", "a")

    def _ui_ask(self, questions: list[dict]) -> str:
        lines = []
        for q in questions:
            prompt = q.get("question", "")
            options = q.get("options") or []
            multiple = bool(q.get("multiple"))
            custom = q.get("custom", True)
            ui_q = UiQuestion(prompt, multiple=multiple, options=list(options))
            answer = self._ask_sync(ui_q)
            if multiple:
                answer = ", ".join(a.strip() for a in answer.split(",") if a.strip())
            lines.append(f'"{prompt}" = "{answer}"')
        return "\n".join(lines) if lines else "Unanswered"

    def _ui_bash_approval(self, command: str) -> tuple[bool, str]:
        prompt = (
            f"[bold yellow]Dangerous Bash command[/bold yellow]\n\n"
            f"{command}\n\n"
            "Run it?  [y]es / [n]o / [a]lways allow (session) / [d]eny (session)"
        )
        q = UiQuestion(prompt, options=["y", "n", "a", "d"])
        answer = self._ask_sync(q).strip().lower()
        if answer in ("y", "yes", "a", "always"):
            return True, ("allow" if answer.startswith("a") else "run")
        if answer in ("d", "deny"):
            return False, "deny"
        return False, "run"

    def _ask_sync(self, q: UiQuestion) -> str:
        """Block the worker thread until the main thread answers."""
        self.question = q
        q.event.wait()
        return q.answer or ""

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _build_history_rows(self) -> list[Any]:
        """Rows for the stored conversation (messages + todos). No stream."""
        rows: list[Any] = []
        calls_by_id: dict[str, Any] = {}
        for m in self.session.last_messages or []:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    calls_by_id[tc.id] = tc

        for m in self.session.last_messages or []:
            if m.role == "system" and m.text().startswith("**[Compacted Summary]**"):
                rows.append(Text("📦 " + _tail_chars(m.text(), 200), style="dim italic"))
                continue
            if m.role == "user":
                body = _tail_lines(m.text(), 12)
                if body.strip():
                    rows.append(Markdown(f"**user:** {body}"))
            elif m.role == "assistant":
                body = _tail_lines(m.text(), 12)
                if m.tool_calls:
                    for tc in m.tool_calls:
                        args = tc.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                args = {}
                        if isinstance(args, dict):
                            params = " ".join(f"{k}={v!r}" for k, v in args.items()
                                             if k != "content" and len(repr(v)) < 80)
                        else:
                            params = ""
                        label = f"🤖 {tc.name}({params})" if params else f"🤖 {tc.name}"
                        rows.append(Text(label, style="cyan"))
                if body.strip():
                    rows.append(Markdown(f"**assistant:** {body}"))
            elif m.role == "tool":
                preview = _tool_result_preview(m.text())
                rows.append(Text(f"tool: {m.name or 'tool'}: {preview}", style="dim"))
                call = calls_by_id.get(m.tool_call_id)
                if call is not None and call.diff:
                    rows.append(render_diff(call.diff))
        return rows

    def _todos_panel(self) -> Panel | None:
        """Todos panel — rebuilt every frame (not cached), so a
        TodoWrite call shows up immediately even mid-run.  When a
        sub-agent is running, its scoped list is shown with a `sub:`
        label so the parent's list isn't mistaken for the sub's."""
        if not self.session.todos:
            return None
        title = "Todos"
        label = self.session.todo_scope_label
        if label:
            # parentheses, not markup brackets: rich parses panel titles
            # as markup and `[sub: ...]` would be eaten as a style tag
            title = f"Todos (sub: {_tail_chars(label, 40)})"
        t = Table.grid(padding=(0, 1))
        for todo in self.session.todos[-8:]:
            status = todo.get("status", "")
            mark = {"completed": "✅", "in_progress": "⏳", "pending": "⬜"}.get(status, "•")
            t.add_row(mark, todo.get("content", ""))
        return Panel(t, title=title, border_style="blue", expand=False)

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
        if not stream:
            return None
        cap = self._visible_row_cap()
        width = getattr(self.console, "width", None) or 80
        lines = max(3, cap - 3)
        preview = _tail_lines(stream, lines)
        preview = _tail_chars(preview, lines * max(1, width))
        return Text(f"assistant: {preview}")

    def _render_conversation(self) -> Panel:
        from rich.console import Group

        rows = self._history_rows()
        stream_row = self._stream_row()
        if stream_row is not None:
            rows.append(stream_row)
        rows = self._apply_budget(rows)
        group = Group(*rows) if rows else Text("(empty)")
        return Panel(group, title="python-agent-harness", border_style="green")

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

    def _visible_row_cap(self) -> int:
        """Max conversation rows that fit the visible terminal area.

        Uses the live terminal height when known (rich reports None for
        non-terminals, e.g. tests), reserving lines for the status bar,
        the panel borders and the input prompt.
        """
        height = getattr(self.console, "height", None)
        if not height or height <= 0:
            return 60
        # reserve: status bar (1) + panel borders (2) + input prompt (1)
        # + the pinned Todos panel when visible (its rows + 2 borders)
        reserved = 4
        if self.session.todos:
            reserved += min(len(self.session.todos), 8) + 2
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
        from rich.console import Group

        parts: list[Any] = [self._status_bar()]
        todos = self._todos_panel()
        if todos is not None:
            parts.append(todos)
        parts.append(self._render_conversation())
        return Group(*parts)

    def _status_bar(self) -> Text:
        mode = self.session.plan_mode.mode.value
        mode_style = "yellow" if mode == "plan" else "green"
        ratio = self.session.context_ratio
        ctx = ""
        if ratio is not None:
            pct = round(ratio * 100)
            style = "red" if pct >= 80 else ("yellow" if pct >= 50 else "green")
            ctx = f" [Ctx:{pct}%/{round(config.CONTEXT_TRIGGER * 100)}%]"
        t = Text()
        t.append(f" [{mode.upper()}]", style=mode_style)
        t.append(ctx)
        if self.agent_running:
            frame = SPINNER_FRAMES[int(time.time() * 10) % len(SPINNER_FRAMES)]
            t.append(f" {frame}", style="bold cyan")
        elif self.question is not None:
            t.append(" ❓", style="yellow")
        t.append(f"{self.status}", style="dim")
        return t

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.console.print(
            Panel(
                "python-agent-harness — agent execution harness\n"
                "Commands: /plan /build /compact /undo /history /save "
                "/summary /help /exit\n"
                "Ctrl-C cancels the current execution (the app stays open); "
                "Ctrl-D or /exit quits.\n"
                "Type a message — Enter for a new line, Esc then Enter "
                "(or Alt+Enter) to submit. Up/Down recall history.",
                border_style="blue",
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
                self.console.print(
                    "[dim]cancelled — Ctrl-D or /exit to quit[/dim]"
                )

    def _ask_question_blocking(self) -> None:
        q = self.question
        self.console.print(self._render_frame())
        self.console.print()
        self._flush()
        prompt = q.prompt
        if q.options:
            prompt += " [choices: " + ", ".join(q.options) + "]"
        try:
            with patch_stdout():
                answer = self.prompt_session.prompt(prompt + " > ", multiline=False)
        except (EOFError, KeyboardInterrupt):
            answer = ""
        q.answer = answer
        q.event.set()
        self.question = None
        self._data_event.set()  # re-render promptly after the answer

    def _read_multiline(self) -> str | None:
        try:
            with patch_stdout():
                text = self.prompt_session.prompt("> ")
        except EOFError:
            # Ctrl-D: quit
            return None
        except KeyboardInterrupt:
            # Ctrl-C: cancel this input, stay in the app
            self.console.print("[dim]input cancelled[/dim]")
            return ""
        return text

    def _start_agent(self, text: str) -> None:
        self.stream_text = ""
        self.status = " running"
        self.session.cancel_event.clear()
        self._data_event.clear()
        self._history_dirty = True
        self.run_seq += 1
        seq = self.run_seq
        self.agent_running = True
        worker = threading.Thread(
            target=self._run_agent, args=(text, seq), daemon=True
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
            self.console.print(
                "\n[dim]execution cancelled — add more messages or /exit[/dim]"
            )
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
        return False

    def _flush(self) -> None:
        """Force stdout through so Live frames render in real time.

        rich's Live does not flush after each refresh, and tty stdout is
        line-buffered — without this the status bar/spinner would sit in
        the stdio buffer and only appear once the run ends (or the 8KB
        buffer fills).
        """
        try:
            self.console.file.flush()
        except (AttributeError, OSError):
            pass

    def _run_agent(self, text: str, seq: int) -> None:
        try:
            self.conversation_history.append(Message(role="user", content=text))
            run_agent_loop(
                self.session,
                messages=list(self.conversation_history),
                top_level=True,
                system=self.session.system_prompt,
            )
            # Only the current run may update shared state; a cancelled
            # worker that finishes late must not clobber the next run.
            if seq == self.run_seq and self.session.last_messages:
                self.conversation_history = list(self.session.last_messages)
        except Exception as e:  # noqa: BLE001
            if seq == self.run_seq:
                self._on_log(f"agent error: {e}")
        finally:
            # the run is done: the final assistant message is now part of
            # the conversation history, so drop the live stream buffer —
            # otherwise the same text renders twice (stream row + history
            # row) and eats the visible-row budget
            with self.lock:
                self.stream_text = ""
            self._history_dirty = True
            self._data_event.set()

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
            self.console.print("[yellow]Plan mode — read-only; only the plan file is writable.[/yellow]")
        elif cmd == "/build":
            self.session.switch_to_build()
            self.console.print("[green]Build mode.[/green]")
        elif cmd == "/compact":
            self._run_compact()
        elif cmd == "/undo":
            ok, msg = self.session.undo.undo_last()
            self.console.print(msg)
        elif cmd == "/history":
            for h in self.session.undo.history():
                self.console.print(h)
        elif cmd == "/save":
            path = self.session.store.save(self._conversation_text())
            self.console.print(f"saved: {path}")
        elif cmd == "/summary":
            self._run_summary()
        elif cmd == "/clear":
            self.conversation_history = []
            self.session.last_messages = []
            self.console.print("[yellow]Conversation history cleared.[/yellow]")
        elif cmd == "/help":
            self.console.print(
                "/plan /build /compact /undo /history /save /summary /clear /exit\n"
                "Ctrl-C cancels the current execution (app stays open); "
                "Ctrl-D or /exit quits."
            )
        else:
            self.console.print(f"unknown command: {cmd}")
        return False

    def _conversation_text(self) -> str:
        msgs = self.session.last_messages or []
        parts = []
        for m in msgs:
            body = m.text()
            if body:
                parts.append(f"**{m.role}**: {body}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # direct commands (no LLM agent loop)
    # ------------------------------------------------------------------
    def _run_compact(self) -> None:
        """Compact the current conversation directly."""
        ok, msg = self.session.compact_conversation()
        self.console.print(msg)

    def _run_summary(self) -> None:
        """Append a summary of the conversation (tools disabled)."""
        msg = self.session.summarize_conversation()
        self.console.print(msg)
