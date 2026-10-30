"""Rendering helpers and mixin for the TUI.

Contains all text-trimming helpers, final-check / reasoning strippers,
and the RenderMixin that provides conversation panel, status bar,
Todos panel, and scrollback dump rendering.
"""

from __future__ import annotations

import math
import re
import time
from typing import TYPE_CHECKING, Any

from rich._wrap import divide_line
from rich.cells import cell_len
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.models import ImagePart, api_safe, display_args
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

# Terminal control sequences, most specific first: string sequences
# (OSC title/clipboard/hyperlink, DCS, SOS, PM, APC) up to their BEL/ST
# terminator -- or the end of the line when unterminated, so a stray
# introducer cannot hide every following line from the user --, CSI
# sequences (cursor movement, screen clearing, colors), both in their
# 7-bit (ESC-prefixed) and 8-bit (C1) forms, other two-byte ESC
# sequences, then any remaining C0/C1 control character except tab and
# newline (\r included: it rewinds the line, letting later text
# overwrite what came before).
_TERMINAL_CONTROL = re.compile(
    r"(?:\x1b[\]PX^_]|[\x90\x98\x9d\x9e\x9f])[^\x07\x1b\x9c\n]*(?:\x07|\x1b\\|\x9c)?"
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"
    r"|\x1b[ -~]?"
    r"|[\x00-\x08\x0b-\x1f\x7f-\x9f]"
)


def sanitize_terminal(text: str) -> str:
    """Strip terminal control sequences from untrusted display text.

    Model output, tool results (file contents, command output, MCP
    results) and diffs are rendered verbatim, and Rich passes escape
    sequences straight through to the terminal: a file the agent read
    could retitle the window, clear or overwrite the screen (hiding
    what actually ran), spoof hyperlinks, or -- on terminals that honor
    OSC 52 -- write the user's clipboard.  Tabs and newlines survive;
    colors in command output are dropped along with everything else.

    Lone surrogates (a non-UTF-8 file read with ``surrogateescape``)
    become U+FFFD: Rich would hand them to the terminal verbatim and the
    UTF-8 encoder raised ``UnicodeEncodeError`` — killing the TUI while
    rendering a restored session.
    """
    return _TERMINAL_CONTROL.sub("", api_safe(text))


# Characters sanitize_terminal would change: any C0/C1 control except
# tab/newline, or a lone surrogate.  A cheap pre-check lets callers skip
# the full regex pass on the common (clean) case.
_NEEDS_SANITIZE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\ud800-\udfff]")


def needs_sanitize(text: str) -> bool:
    """Whether sanitize_terminal would change TEXT (cheap pre-check)."""
    return _NEEDS_SANITIZE.search(text) is not None


def safe_markup(value: object) -> str:
    """VALUE made safe to embed in a Rich markup string.

    Paths, error messages and user input interpolated into
    ``f"[red]...{value}...[/red]"`` are parsed as markup: a value like
    ``notes[/draft].md`` raised MarkupError (killing the TUI), and
    ``arr[i]`` silently lost its ``[i]``.  Escapes markup and strips
    terminal control sequences.
    """
    return escape(sanitize_terminal(str(value)))


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
    round-trip: the completion nudge, the post-compaction resume message,
    the <system-reminder>-wrapped plan/build-switch prompts (plan.md,
    plan-mode.md, build-switch.md all start with the reminder tag), and
    the plan-exit approval notice.
    """
    injected = {config.NUDGE_MESSAGE, config.RESUME_MESSAGE}
    if text in injected:
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


class ToolRow(Text):
    """A panel row owned by a tool call, and therefore a click target.

    ``tool_id`` is the ``ToolCall.id`` the row displays; asking rich for
    the rendered lines of a row and walking the frame with it is how a
    terminal click is traced back to the call it landed on.
    """

    def __init__(self, tool_id: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tool_id = tool_id


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
        _expanded_tools: set[str]
        _last_frame: Any
        _mouse: Any

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
                body = _tail_lines(sanitize_terminal(self.round_user_text), 12)
                if body.strip():
                    rows.append(Markdown(f"**user:** {body}", style=USER_STYLE, hyperlinks=False))

        # Markdown rows use hyperlinks=False: Rich would otherwise turn
        # [text](url) into a terminal hyperlink showing only TEXT, so
        # model output or a file could display one URL while the click
        # opens another.  Unlinked, the real target is printed beside it.
        for m in messages:
            text = sanitize_terminal(m.text())
            # compacted summaries live in the user turn (system prompt is
            # separate); match on content so both live sessions and
            # restored files (role lost in the markdown round-trip) render
            if text.startswith("**[Compacted Summary]**"):
                rows.append(Text("📦 " + _tail_chars(text, 200), style="dim italic"))
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
                body = text if full else _tail_lines(text, 12)
                indicator = _image_indicator(m)
                if body.strip() or indicator:
                    rows.append(
                        Markdown(
                            f"**user:** {body.strip()}{indicator}",
                            style=USER_STYLE,
                            hyperlinks=False,
                        )
                    )
            elif m.role == "assistant":
                body = text
                collapsed_reasoning = False
                if isinstance(m.reasoning, str) and m.reasoning:
                    stripped = _strip_reasoning(body, sanitize_terminal(m.reasoning))
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
                        rows.extend(self._tool_call_rows(tc))
                if body.strip():
                    rows.append(
                        Markdown(f"**assistant:** {body}", style=ASSISTANT_STYLE, hyperlinks=False)
                    )
            elif m.role == "tool":
                preview = _tool_result_preview(text)
                name = sanitize_terminal((m.name or "tool").lower())
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
                    rows.append(render_diff(sanitize_terminal(call.diff)))
        return rows

    def _tool_call_rows(self, tc: Any) -> list[Any]:
        """Rows for one tool call: collapsed by default, expanded on click.

        Collapsed: a single magenta ``▶ tool: Name(k=v, ...)`` line with
        truncated argument values — the compact panel row.

        Expanded: the ``▶`` header line plus one indented row per
        argument carrying the value in full, so a long Bash command can
        be read in place.  Expansion is per tool-call id and happens
        only when the user clicks the ``▶`` row (there is no other
        trigger).

        The expanded rows are sized to the visible area as a *block*:
        the whole call (header + values) has to fit the row budget,
        because ``_apply_budget`` drops a call's rows oldest-first —
        otherwise a tall value would push the header off the panel and
        the expansion would look like a bare ``▾``.  Values are trimmed
        with an ellipsis to make that fit, never silently hidden.

        Every returned row is a ``ToolRow`` carrying the call id, which
        is how ``_on_mouse_click`` maps a clicked terminal row back to
        the call it belongs to (the continuation rows included, so a
        click anywhere in an expanded call collapses it again).
        """
        expanded = tc.id in getattr(self, "_expanded_tools", ())
        if not expanded:
            args = display_args(tc.arguments, limit=100)
            if not args:
                return [ToolRow(tc.id, f"▶ tool: {sanitize_terminal(tc.name)}", style="magenta")]
            params = " ".join(f"{k}={v}" for k, v in args.items())
            label = f"tool: {tc.name}({params})"
            return [ToolRow(tc.id, f"▶ {sanitize_terminal(label)}", style="magenta")]
        args = display_args(tc.arguments, limit=config.TOOL_EXPANDED_ARG_CHARS)
        if not args:
            return [ToolRow(tc.id, f"▶ tool: {sanitize_terminal(tc.name)}", style="magenta")]
        head = ToolRow(tc.id, "▶ ", style="magenta")
        head.append(sanitize_terminal(f"tool: {tc.name} "), style="magenta")
        count = len(args)
        head.append(f"({count} argument{'s' if count != 1 else ''}) ", style="dim")
        head.append("▾", style="dim")
        rows: list[Any] = [head]
        # lines the block may use: the header plus a share of the
        # remaining budget for each argument value
        width = max(1, getattr(self.console, "width", None) or 80)
        budget = max(1, self._expanded_block_lines() - 1)
        for index, (key, value) in enumerate(args.items()):
            share = max(1, budget // max(1, len(args) - index))
            chars = max(20, share * width - len(key) - 8)
            if len(value) > chars:
                value = value[: max(1, chars - 1)] + "…"
            row = ToolRow(tc.id, style="dim")
            row.append(sanitize_terminal(f"    {key} = {value}"))
            budget -= self._est_lines(row, width)
            rows.append(row)
        return rows

    def _expanded_block_lines(self) -> int:
        """Lines an expanded tool call may occupy on screen.

        The block has to fit next to the status bar and the pinned
        Todos panel, so it gets the visible row budget minus a couple
        of rows for the neighbouring conversation rows.  Staying inside
        the budget is what keeps the header and its values together and
        the frame no taller than the terminal (a cropped frame has no
        usable click mapping).
        """
        return max(1, self._visible_row_cap() - 3)

    def _expanded_arg_limit(self, argument_count: int = 1) -> int:
        """Char cap for one expanded argument value.

        ``TOOL_EXPANDED_ARG_CHARS`` is an absolute ceiling, and it is
        taller than a short terminal can show.  The cap is therefore
        scaled to the visible area and divided between the arguments,
        so the whole expanded block fits the row budget whatever the
        terminal size (see ``_tool_call_rows``).
        """
        width = max(1, getattr(self.console, "width", None) or 80)
        share = max(1, self._expanded_block_lines() - 1) // max(1, argument_count)
        return max(40, min(config.TOOL_EXPANDED_ARG_CHARS, share * width - 8))

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
            # Text, not str: add_row parses plain strings as markup, so a
            # model-written todo like "fix [/tmp] paths" raised
            # MarkupError on the render thread and killed the TUI
            t.add_row(mark, Text(sanitize_terminal(content)))
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
        cap = self._visible_row_cap()
        width = getattr(self.console, "width", None) or 80
        lines = max(3, cap - 3)
        # Sanitize BEFORE stripping: a control sequence inside the
        # [FINAL CHECK] header ("[FINAL\x1b[0m CHECK]") would otherwise
        # defeat the header match and leak the block.  The full-buffer
        # sanitize is skipped when the text is already clean (the common
        # case), so the expensive regex pass only runs when needed.
        if needs_sanitize(stream):
            stream = sanitize_terminal(stream)
        # Strip on the FULL buffer: the header can sit arbitrarily far
        # before the tail, so windowing first would leak the block.
        stream = _strip_final_check(stream)
        window = max(lines * max(1, width) * 2, 8192)
        if len(stream) > window:
            stream = stream[-window:]
        if not stream:
            return None
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

        Rows belonging to one tool call form a unit: an expanded call is
        kept or dropped whole, never split, so a click target can never
        end up on screen without the header (or the values) it
        describes.
        """
        width = getattr(self.console, "width", None) or 80
        budget = self._visible_row_cap()
        units: list[list[Any]] = []
        for row in rows:
            tool_id = getattr(row, "tool_id", None)
            if tool_id is not None and units:
                previous = units[-1]
                if getattr(previous[0], "tool_id", None) == tool_id:
                    previous.append(row)
                    continue
            units.append([row])
        kept: list[list[Any]] = []
        for unit in reversed(units):
            est = sum(self._est_lines(row, width) for row in unit)
            if budget - est < 0:
                continue  # older rows are dropped once the budget is spent
            kept.append(unit)
            budget -= est
        return [row for unit in reversed(kept) for row in unit]

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

        The status bar is measured instead of assumed: it wraps on a
        narrow terminal, and reserving too little made the whole frame
        taller than the screen (rich then crops it, which leaves no
        usable click mapping).  The result is never below 1, so a tiny
        terminal degrades to "hardly any conversation" rather than to a
        frame that no longer fits.
        """
        height = getattr(self.console, "height", None)
        if not height or height <= 0:
            return 60
        status = self._frame_height(self._status_bar()) or 2
        reserved = status + 1  # + the input prompt
        if self._controller.todos:
            reserved += min(len(self._controller.todos), 8) + 1
        return max(1, height - reserved)

    @staticmethod
    def _est_lines(row: Any, width: int) -> int:
        """Estimated wrapped-line count for ROW at WIDTH (budget math).

        The estimate must never be a LOWER bound: ``_apply_budget`` keeps
        the newest rows by subtracting estimates from a fixed budget, so
        under-estimating a tall row lets it through, the frame overflows,
        and rich's ``Live`` crops it from the BOTTOM -- hiding exactly the
        newest rows the budget exists to protect.

        Two shapes made the old per-type estimate collapse to 1 line:

        - multi-line ``Text`` (the old formula divided the total length by
          the width, so 20 short lines counted as ~2 and were billed as 1);
        - ``Group`` (a rendered diff is one ``Text`` per diff line, so a
          300-line Edit diff was counted as a single line and disabled the
          budget entirely).

        Both now count real newlines, and ``Group`` recurses into its
        renderables.

        The per-line count comes from rich's own wrapping primitive
        (``divide_line``), so word wrapping and double-width characters
        are counted exactly: a naive ``len(line) / width`` billing
        under-counted by a line per row on a value like ``k0 = 'vvv…'``
        (wrapping happens at the space), which was enough for a frame to
        outgrow the terminal and, with it, to disable click mapping.
        """
        width = max(1, width)

        def _wrapped(text: str) -> int:
            total = 0
            for line in text.split("\n"):
                if not line:
                    total += 1
                    continue
                try:
                    breaks = divide_line(line, width, fold=True)
                except Exception:  # pragma: no cover - defensive
                    total += max(1, math.ceil(cell_len(line) / width))
                else:
                    total += len(breaks) + 1
            return total

        if isinstance(row, Text):
            return _wrapped(row.plain)
        if isinstance(row, Markdown):
            return _wrapped(row.markup)
        if isinstance(row, Panel):
            inner = getattr(row.renderable, "renderables", None)
            if isinstance(inner, (list, tuple)):
                return 3 + sum(RenderMixin._est_lines(child, width) for child in inner)
            return 3 + RenderMixin._est_lines(row.renderable, width)
        if isinstance(row, Group):
            return sum(RenderMixin._est_lines(child, width) for child in row.renderables)
        return 1

    def _render_frame(self) -> Group:
        """Full frame: status bar + Todos pinned on top, conversation below.

        The status bar and the Todos panel are placed FIRST so they stay
        visible no matter how tall the conversation gets — the Todos list
        is pinned like a second mode line instead of competing with the
        conversation rows for the visible budget.

        The frame is remembered (``_last_frame``) so a mouse click can be
        traced back to the row it landed on: the click handler needs the
        exact renderable that produced the visible lines.
        """
        parts: list[Any] = [self._status_bar()]
        todos = self._todos_panel()
        if todos is not None:
            parts.append(todos)
        parts.append(self._render_conversation())
        frame = Group(*parts)
        self._last_frame = frame
        return frame

    # ------------------------------------------------------------------
    # mouse clicks (expand / collapse a tool call)
    # ------------------------------------------------------------------
    def _frame_rows(self, frame: Any) -> list[Any]:
        """Flatten FRAME into its rows, in render order."""
        rows: list[Any] = []
        for part in getattr(frame, "renderables", (frame,)):
            inner = getattr(part, "renderables", None)
            if isinstance(inner, (list, tuple)) and isinstance(part, Group):
                rows.extend(inner)
            else:
                rows.append(part)
        return rows

    def _frame_top(self, total: int) -> int | None:
        """Screen row (1-based) where the frame's first line is drawn.

        The terminal is asked for its cursor row, which sits on the
        frame's last line (rich paints the live renderable through a
        render hook, so its trailing newline lands *before* the frame):
        the top is therefore ``cursor - total + 1``.  Asking is the only
        exact option — a short frame is *not* bottom-anchored, since rich
        repaints from wherever the cursor happened to be, so assuming the
        panel ends at the last row would shift every click by the empty
        space left below it.

        Falls back to the bottom-anchored assumption when no terminal
        answers (``TERM`` without DSR support, unit tests), and returns
        None when the frame is taller than the screen: rich then crops
        it and rows no longer map onto it.
        """
        height = getattr(self.console, "height", None)
        mouse = getattr(self, "_mouse", None)
        if mouse is not None and height and total <= height:
            cursor = mouse.cursor_row()
            if cursor is not None:
                top = cursor - total + 1
                if 1 <= top <= height:
                    return top
        if not height or total > height:
            return None
        return height - total + 1

    def _frame_height(self, frame: Any) -> int | None:
        """Rendered line count of FRAME, or None if it cannot be rendered."""
        try:
            return len(self.console.render_lines(frame, self.console.options, pad=False))
        except Exception:
            return None

    def _panel_top(self) -> tuple[Any, int, int] | None:
        """(frame, line count, screen row of its first line) for a click.

        The frame is captured, measured, and only then is the terminal
        asked for the cursor row — and if the panel repainted in
        between, the answer describes the *newer* frame while the
        measurement describes the old one, so the mapping would be off
        by however many rows the panel shifted.  The identity check
        catches that and measures the fresh frame instead (bounded, so
        a panel that repaints non-stop still yields the best guess
        rather than ignoring the click).
        """
        result: tuple[Any, int, int] | None = None
        for _ in range(3):
            frame = getattr(self, "_last_frame", None)
            if frame is None:
                return None
            total = self._frame_height(frame)
            if total is None:
                return None
            top = self._frame_top(total)
            if top is None:
                return None
            result = (frame, total, top)
            if getattr(self, "_last_frame", None) is frame:
                break
        return result

    def _toggle_tool_at(self, row: int) -> Any:
        """Tool-call id displayed at terminal ROW (1-based), or None.

        Lines are counted with rich's own renderer, so wrapped rows and
        diffs are accounted for exactly: an estimate would drift by a
        line or two and toggle the wrong call.  Only runs on a click,
        so the extra render pass costs nothing in the redraw loop.
        """
        located = self._panel_top()
        if located is None:
            return None
        frame, total, top = located
        line = row - top
        if line < 0 or line >= total:
            return None
        offset = 0
        for part in self._frame_rows(frame):
            row_height = self._frame_height(part)
            if row_height is None:
                return None
            tool_id = getattr(part, "tool_id", None)
            if tool_id is not None and offset <= line < offset + row_height:
                return tool_id
            offset += row_height
        return None

    def _on_mouse_click(self, column: int, row: int) -> None:
        """Expand/collapse the tool call under a left click.

        ``row`` is 1-based, counted from the top of the terminal.  The
        click is ignored unless it lands on a `▶` tool row of the live
        panel; the run's stream keeps rendering either way.
        """
        with self.lock:
            if not self.agent_running:
                return
        tool_id = self._toggle_tool_at(row)
        if tool_id is None:
            return
        expanded = getattr(self, "_expanded_tools", None)
        if expanded is None:
            return
        if tool_id in expanded:
            expanded.discard(tool_id)
        else:
            expanded.add(tool_id)
        with self.lock:
            self._history_dirty = True
        self._data_event.set()

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
                t.append(f" {sanitize_terminal(tool)}", style="bold cyan")
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

        The status is the one display path that reaches the terminal
        unsanitized: it is set from endpoint/MCP error text
        (``error_display_text``) and ``_on_log`` messages (``agent
        error: {e}``, ``compaction failed: {e}``, ``MCP: [{server}]
        {err}``), any of which can carry a hostile server's escape
        sequences.  Strip them here -- the single sink for every status
        source -- mirroring the sibling paths (tool name, history rows)
        that already call ``sanitize_terminal``.  Sanitizing BEFORE the
        newline flatten below keeps a stray OSC introducer from hiding
        the rest of the line.

        The pre-check mirrors ``_stream_row``: the sanitizer is a regex
        pass over the whole message, and the status bar is now measured
        on every frame (``_visible_row_cap``), so running it
        unconditionally would bill that pass to every clean frame too.
        """
        if needs_sanitize(msg):
            msg = sanitize_terminal(msg)
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
