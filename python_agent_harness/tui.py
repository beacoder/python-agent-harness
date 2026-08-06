"""Rich TUI for the agent harness.

Layout: conversation panel + status bar + input line.  The agent loop
runs in a worker thread; the main thread renders with rich Live and
services interactive questions (Bash approval, Question tool, PlanExit
confirmation).
"""

from __future__ import annotations

import json
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
        self.conversation_history: list[Message] = []
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

    def _on_notify(self, kind: str) -> None:
        if kind == "tools":
            self.status = " running tools"
        elif kind == "compact":
            self.status = " compacted"
        elif kind == "error":
            self.status = " error"
        else:
            self.status = " running"

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
    def _render_conversation(self) -> Panel:
        from rich.console import Group

        with self.lock:
            stream = self.stream_text
        rows: list[Any] = []

        calls_by_id: dict[str, Any] = {}
        for m in self.session.last_messages or []:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    calls_by_id[tc.id] = tc

        # compact summary marker
        for m in self.session.last_messages or []:
            if m.role == "system" and m.text().startswith("**[Compacted Summary]**"):
                rows.append(Text("📦 " + m.text()[:200], style="dim italic"))
                continue
            if m.role == "user":
                if m.text().strip():
                    rows.append(Markdown(f"**You:** {m.text()}"))
            elif m.role == "assistant":
                body = m.text()
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
                if body:
                    rows.append(Markdown(f"**Agent:** {body}"))
            elif m.role == "tool":
                content = m.text()
                preview = content[:400] + ("…" if len(content) > 400 else "")
                rows.append(Text(f"🔧 {m.name or 'tool'}: {preview}", style="dim"))
                call = calls_by_id.get(m.tool_call_id)
                if call is not None and call.diff:
                    rows.append(render_diff(call.diff))

        if stream:
            rows.append(Markdown(f"**Agent:** {stream}"))

        if self.session.todos:
            t = Table.grid(padding=(0, 1))
            for todo in self.session.todos[-8:]:
                status = todo.get("status", "")
                mark = {"completed": "✅", "in_progress": "⏳", "pending": "⬜"}.get(status, "•")
                t.add_row(mark, todo.get("content", ""))
            rows.append(Panel(t, title="Todos", border_style="blue", expand=False))

        group = Group(*rows) if rows else Text("(empty)")
        return Panel(group, title="python-agent-harness", border_style="green")

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
        self.console.print(self._render_conversation())
        self.console.print()
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
        self.agent_running = True
        worker = threading.Thread(target=self._run_agent, args=(text,), daemon=True)
        worker.start()
        cancelled = False
        try:
            with Live(
                self._render_conversation(),
                console=self.console,
                refresh_per_second=10,
                screen=False,
            ) as live:
                while worker.is_alive():
                    if self.question is not None:
                        live.stop()
                        self._ask_question_blocking()
                        live.start()
                    live.update(self._render_conversation())
                    time.sleep(0.05)
                live.update(self._render_conversation())
        except KeyboardInterrupt:
            # Ctrl-C during execution: stop the run, keep the app open
            cancelled = True
            self.session.cancel()
            worker.join(timeout=5)
            self.console.print(
                "\n[dim]execution cancelled — add more messages or /exit[/dim]"
            )
        finally:
            self.agent_running = False
            if not cancelled:
                self.console.print()

    def _run_agent(self, text: str) -> None:
        try:
            self.conversation_history.append(Message(role="user", content=text))
            run_agent_loop(
                self.session,
                messages=list(self.conversation_history),
                top_level=True,
                system=self.session.system_prompt,
            )
            if self.session.last_messages:
                self.conversation_history = list(self.session.last_messages)
        except Exception as e:  # noqa: BLE001
            self._on_log(f"agent error: {e}")
        finally:
            self.agent_running = False

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
