"""Rendering helpers and mixin for the TUI.

Contains all text-trimming helpers, final-check / reasoning strippers,
and the RenderMixin that provides conversation panel, status bar,
Todos panel, and scrollback dump rendering.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from rich.cells import cell_len
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.models import ImagePart, display_args
from ..io.diffrender import render_diff
from ..io.text_filter import strip_final_check, strip_reasoning
from ..prompts import _is_mode_reminder_text
from ..session import config

if TYPE_CHECKING:
    import threading

    from ..entry.controller import Controller
    from .input import UiQuestion

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
    preview = _head_lines(content, config.TOOL_RESULT_PREVIEW_LINES)
    return _head_chars(preview, config.TOOL_RESULT_PREVIEW_CHARS)


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


def _image_indicator(msg: Any) -> str:
    """A short ``[📎 N image(s)]`` suffix for user messages with images.

    ``Message.text()`` skips ``ImagePart`` objects, so without this the
    conversation panel gives no visual cue that an image was attached.
    """
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return ""
    count = sum(1 for p in content if isinstance(p, ImagePart))
    if not count:
        return ""
    noun = "image" if count == 1 else "images"
    return f" [📎 {count} {noun}]"


# shared with the headless view (python_agent_harness.io.text_filter); the
# underscore-prefixed names are kept for existing importers
_strip_final_check = strip_final_check
_strip_reasoning = strip_reasoning


class RenderMixin:
    """Rendering methods for the TUI.

    Expects the host class to provide: ``_controller``, ``console``,
    ``stream_text``, ``lock``, ``question``, ``agent_running``,
    ``status``, ``_current_tool``, ``round_start``, ``round_user_text``,
    ``_data_event``, ``_history_cache``, ``_history_dirty``,
    ``_round_times``, ``_run_start``.
    """

    if TYPE_CHECKING:
        _controller: Controller
        console: Console
        stream_text: str
        lock: threading.Lock
        question: UiQuestion | None
        agent_running: bool
        status: str
        _current_tool: str
        round_start: int
        round_user_text: str
        _data_event: threading.Event
        _history_cache: list[Any] | None
        _history_dirty: bool
        _round_times: list[float]
        _run_start: float | None

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
        all_messages = self._controller.last_messages or []
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
                indicator = _image_indicator(m)
                if body.strip() or indicator:
                    rows.append(Markdown(f"**user:** {body.strip()}{indicator}", style=USER_STYLE))
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
                        params = " ".join(f"{k}={v}" for k, v in display_args(tc.arguments).items())
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
        TodoWrite call shows up immediately even mid-run.

        Defensive against non-dict items / non-str fields: todos come
        from the model, and this runs on the main render thread where a
        raise would kill the TUI (session.update_todos already
        normalizes at the boundary; this covers direct assignment).
        """
        if not self._controller.todos:
            return None
        t = Table.grid(padding=(0, 1))
        for raw in self._controller.todos[-8:]:
            if isinstance(raw, dict):
                status = raw.get("status", "")
                if not isinstance(status, str):
                    status = ""  # dict.get needs a hashable key
                mark = {"completed": "✅", "in_progress": "⏳", "pending": "⬜"}.get(status, "•")
                content = str(raw.get("content", ""))
            else:
                mark, content = "•", repr(raw)
            t.add_row(mark, content)
        return Group(Text("Todos", style="bold"), t)

    def _history_rows(self) -> list[Any]:
        """Cached history rows; rebuilt only when the conversation changes.

        During streaming only the stream row changes, so we must NOT
        rebuild (and re-parse Markdown for) the whole conversation every
        frame — that cost is what made the scroll lag behind the text.
        """
        with self.lock:
            dirty = self._history_dirty
            cache = self._history_cache
        if dirty or cache is None:
            built = self._build_history_rows()
            with self.lock:
                self._history_cache = built
                self._history_dirty = False
            return list(built)
        return list(cache)

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
        times = self._round_times or self._controller.store.round_times
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
        # reserve: status bar (up to 2 lines) + input prompt (1)
        # + the pinned Todos section when visible (its title line + rows)
        reserved = 3
        if self._controller.todos:
            reserved += min(len(self._controller.todos), 8) + 1
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
        mode = self._controller.plan_mode.mode.value
        mode_style = "bold yellow" if mode == "plan" else "bold green"
        ratio = self._controller.context_ratio
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
        if self._controller.save_error:
            t.append(" [!save]", style="red bold")
        if self.agent_running:
            frame = SPINNER_FRAMES[int(time.time() * 10) % len(SPINNER_FRAMES)]
            t.append(f" {frame}", style="bold cyan")
            with self.lock:
                tool = self._current_tool
                status = self.status
            if tool:
                t.append(f" {tool}", style="bold cyan")
        elif self.question is not None:
            t.append(" ❓", style="yellow")
            with self.lock:
                status = self.status
        else:
            with self.lock:
                status = self.status
        t.append(self._fit_status(cell_len(str(t)), status), style=self._status_style())
        return t

    def _fit_status(self, used: int, msg: str) -> str:
        """Fit the status message on the status bar: up to two lines.

        Line 1 shares its row with the mode/context prefix (``used``
        cells); line 2 gets the full terminal width.  Newlines in the
        message are flattened, words break at spaces where possible,
        and anything beyond two lines is truncated with an ellipsis.
        """
        msg = " ".join(msg.splitlines())
        width = self.console.width or 80
        avail1 = max(1, width - used - 1)
        avail2 = max(1, width - 2)
        line1, rest = self._wrap_line(msg, avail1)
        if not rest:
            return line1
        line2, rest2 = self._wrap_line(rest, avail2)
        if rest2:
            # re-wrap reserving a cell so the ellipsis fits the line
            line2, _ = self._wrap_line(rest, max(1, avail2 - 1))
            return f"{line1}\n{line2}…"
        return f"{line1}\n{line2}"

    @staticmethod
    def _wrap_line(text: str, avail: int) -> tuple[str, str]:
        """Split ``text`` into ``(line, remainder)`` where ``line`` is the
        longest prefix fitting ``avail`` cells.

        Breaks at a space so words stay whole; a single word wider than
        the line (long JSON tokens, URLs) is hard-broken at the cell
        boundary instead of dropped.
        """
        if cell_len(text) <= avail:
            return text, ""
        words = text.split(" ")
        line = ""
        for idx, word in enumerate(words):
            candidate = word if not line else f"{line} {word}"
            if cell_len(candidate) <= avail:
                line = candidate
                continue
            if not line:
                # one word wider than the whole line: cut mid-word
                take = ""
                for ch in word:
                    if cell_len(take) + cell_len(ch) > avail:
                        break
                    take += ch
                rest_word = word[len(take) :]
                tail = words[idx + 1 :]
                remainder = " ".join([rest_word, *tail]) if rest_word else " ".join(tail)
                return take, remainder
            return line, " ".join(words[idx:])
        return line, ""

    def _status_style(self) -> str:
        """Status-bar color by state: errors red, activity cyan, idle dim."""
        with self.lock:
            s = self.status
        if "error" in s or "failed" in s:
            return "bold red"
        if " ⏳" in s or " running" in s or "retrying" in s:
            return "cyan"
        return "dim"
