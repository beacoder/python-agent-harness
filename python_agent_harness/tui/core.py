"""Core TUI class — the coordinator that owns shared state, the main
event loop, agent run lifecycle, and session callbacks.

Rendering, input, and slash commands are mixed in from their respective
modules; this module provides the glue: ``__init__``, ``run``,
``_start_agent``, ``_run_live``, ``_run_dumb``, ``_run_agent``, and the
session callbacks (``_on_delta``, ``_on_notify``, ``_on_log``).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from typing import Any

from prompt_toolkit.history import FileHistory
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .. import config
from ..agent import run_agent_loop
from ..agent_session import AgentSession
from ..models import Message
from .commands import CommandMixin
from .input import InputMixin, SlashCompleter, UiQuestion, _history_path, _make_prompt_session
from .render import RenderMixin


class Tui(RenderMixin, InputMixin, CommandMixin):
    """Interactive Rich TUI for the agent harness.

    Layout: conversation panel + status bar + input line.  The agent loop
    runs in a worker thread; the main thread renders with rich Live and
    services interactive questions (Question tool, PlanExit confirmation).
    """

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
        self.prompt_session = _make_prompt_session(
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
