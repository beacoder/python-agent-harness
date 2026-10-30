"""Tests for click-to-expand tool calls (tui/mouse.py + the click map).

Covers the mouse report parser, the terminal-mode lifecycle guards, and
the mapping from a clicked terminal row to the tool call it landed on.
"""

import io
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from rich.console import Console
from rich.text import Text

from python_agent_harness.core.models import Message, ToolCall
from python_agent_harness.tui import mouse as mouse_mod
from tests.support.tui_test_utils import make_tui


def _tool_call_message(tool_id="1", name="Bash", command="ls -la"):
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=tool_id, name=name, arguments={"command": command})],
    )


class TestMouseParsing(unittest.TestCase):
    def test_parse_sgr_press_and_release(self):
        data = b"\x1b[<0;12;7M\x1b[<0;12;7m"
        events = mouse_mod.parse_events(data)
        self.assertEqual(events, [(0, 12, 7, "press"), (0, 12, 7, "release")])

    def test_parse_ignores_non_mouse_input(self):
        """Typed text and other escapes never look like a click."""
        self.assertEqual(mouse_mod.parse_events(b"hello"), [])
        self.assertEqual(mouse_mod.parse_events(b"\x1b[A"), [])

    def test_parse_large_coordinates(self):
        """SGR (1006) reports columns/rows beyond the X10 limit."""
        events = mouse_mod.parse_events(b"\x1b[<0;300;120M")
        self.assertEqual(events, [(0, 300, 120, "press")])

    def test_is_left_click_excludes_wheel_and_drag(self):
        self.assertTrue(mouse_mod.is_left_click(0))
        self.assertFalse(mouse_mod.is_left_click(32))  # drag
        self.assertFalse(mouse_mod.is_left_click(64))  # wheel up
        self.assertFalse(mouse_mod.is_left_click(65))  # wheel down
        self.assertFalse(mouse_mod.is_left_click(2))  # right button

    def test_split_separates_events_text_and_carry(self):
        events, carry, text = mouse_mod.split(b"a\x1b[<0;5;9Mb")
        self.assertEqual(events, [(0, 5, 9, "press")])
        self.assertEqual(carry, b"")
        self.assertEqual(text, "ab")

    def test_split_keeps_incomplete_sequence_for_next_chunk(self):
        """A report split across reads must not be decoded as text."""
        events, carry, text = mouse_mod.split(b"x\x1b[<0;5")
        self.assertEqual(events, [])
        self.assertEqual(text, "x")
        self.assertTrue(carry.startswith(b"\x1b[<0;5"))
        # the rest of the report arrives in the next chunk
        events, carry, text = mouse_mod.split(carry + b";9M")
        self.assertEqual(events, [(0, 5, 9, "press")])
        self.assertEqual(carry, b"")
        self.assertEqual(text, "")

    def test_split_drops_control_characters(self):
        _events, _carry, text = mouse_mod.split(b"a\x01\x03b")
        self.assertEqual(text, "ab")

    def test_take_cursor_reports_extracts_and_strips(self):
        """A DSR answer becomes a row, and never typed text."""
        rest, rows = mouse_mod.take_cursor_reports(b"a\x1b[24;7Rb")
        self.assertEqual(rows, [(24, 7)])
        self.assertEqual(rest, b"ab")
        self.assertEqual(mouse_mod.take_cursor_reports(b"\x1b[A"), (b"\x1b[A", []))

    def test_split_never_returns_a_dsr_reply_as_text(self):
        data, _rows = mouse_mod.take_cursor_reports(b"\x1b[12;1Rhi")
        _events, _carry, text = mouse_mod.split(data)
        self.assertEqual(text, "hi")


class TestMouseSession(unittest.TestCase):
    def test_supported_is_false_without_a_tty(self):
        console = Console(file=io.StringIO(), width=80)
        session = mouse_mod.MouseSession(console, lambda c, r: None)
        self.assertFalse(session.supported())
        self.assertFalse(session.enable())

    def test_enable_swallows_terminal_errors(self):
        """A terminal that rejects cbreak mode must not break the run."""
        console = Console(file=io.StringIO(), width=80)
        session = mouse_mod.MouseSession(console, lambda c, r: None)
        with (
            mock.patch.object(session, "supported", return_value=True),
            mock.patch.object(mouse_mod.termios, "tcgetattr", side_effect=OSError),
        ):
            self.assertFalse(session.enable())
        self.assertFalse(session._active)

    def test_enable_writes_mouse_mode_and_disable_restores(self):
        buf = io.StringIO()
        console = Console(file=buf, width=80)
        session = mouse_mod.MouseSession(console, lambda c, r: None)
        attrs = ["saved"]
        with (
            mock.patch.object(session, "supported", return_value=True),
            mock.patch.object(mouse_mod.termios, "tcgetattr", return_value=attrs),
            mock.patch.object(mouse_mod.tty, "setcbreak"),
            mock.patch.object(mouse_mod.termios, "tcsetattr") as setattr_mock,
            mock.patch.object(mouse_mod.threading, "Thread") as thread_cls,
        ):
            thread_cls.return_value.is_alive.return_value = False
            self.assertTrue(session.enable())
            self.assertIn(mouse_mod.ENABLE, buf.getvalue())
            session.disable()
            self.assertIn(mouse_mod.DISABLE, buf.getvalue())
            setattr_mock.assert_called_once()
        self.assertEqual(session._saved_attrs, None)

    def test_click_callback_errors_do_not_escape(self):
        """A raising click handler must never take the reader down."""
        raised = threading.Event()

        def boom(_col, _row):
            raised.set()
            raise RuntimeError("boom")

        buf = io.StringIO()
        console = Console(file=buf, width=80)
        session = mouse_mod.MouseSession(console, boom)
        fd_reads = [b"\x1b[<0;4;5M", b""]
        with (
            mock.patch.object(mouse_mod.select, "select", return_value=([0], [], [])),
            mock.patch.object(mouse_mod.os, "read", side_effect=fd_reads),
        ):
            session._fd = 0
            session._read_loop()  # returns on EOF instead of raising
        self.assertTrue(raised.wait(1.0))
        self.assertEqual(session.pending_text, "")

    def test_clicks_are_dispatched_off_the_reader_thread(self):
        """The reader must stay free to read the cursor-row answer."""
        seen = []
        done = threading.Event()

        def on_click(col, row):
            seen.append((col, row, threading.current_thread().name))
            done.set()

        buf = io.StringIO()
        console = Console(file=buf, width=80)
        session = mouse_mod.MouseSession(console, on_click)
        fd_reads = [b"\x1b[<0;4;5M", b""]
        with (
            mock.patch.object(mouse_mod.select, "select", return_value=([0], [], [])),
            mock.patch.object(mouse_mod.os, "read", side_effect=fd_reads),
        ):
            session._fd = 0
            session._read_loop()
        self.assertTrue(done.wait(1.0))
        self.assertEqual(seen[0][:2], (4, 5))
        self.assertEqual(seen[0][2], "tui-mouse-click")

    def test_cursor_row_reads_the_terminal_answer(self):
        buf = io.StringIO()
        console = Console(file=buf, width=80)
        session = mouse_mod.MouseSession(console, lambda c, r: None)
        session._active = True

        def fake_read(_fd, _n):
            # a real terminal only answers ~ESC[6n once it is written
            if mouse_mod.CURSOR_QUERY in buf.getvalue():
                return b"\x1b[17;3R"
            time.sleep(0.01)
            return b"\x00"

        with (
            mock.patch.object(mouse_mod.select, "select", return_value=([0], [], [])),
            mock.patch.object(mouse_mod.os, "read", side_effect=fake_read),
        ):
            session._fd = 0
            reader = threading.Thread(target=session._read_loop, daemon=True)
            reader.start()
            try:
                self.assertEqual(session.cursor_row(timeout=2.0), 17)
            finally:
                session._stop.set()
                reader.join(timeout=2.0)
        self.assertIn(mouse_mod.CURSOR_QUERY, buf.getvalue())

    def test_cursor_row_is_none_when_reporting_is_off(self):
        console = Console(file=io.StringIO(), width=80)
        session = mouse_mod.MouseSession(console, lambda c, r: None)
        self.assertIsNone(session.cursor_row(timeout=0.01))

    def test_cursor_row_gives_up_when_nothing_answers(self):
        """A terminal without DSR support must not hang the click."""
        buf = io.StringIO()
        console = Console(file=buf, width=80)
        session = mouse_mod.MouseSession(console, lambda c, r: None)
        session._active = True
        self.assertIsNone(session.cursor_row(timeout=0.01))
        self.assertIn(mouse_mod.CURSOR_QUERY, buf.getvalue())


class TestClickExpandsToolCall(unittest.TestCase):
    def _frame_line_of(self, tui, needle):
        """Screen-independent index of the frame line containing NEEDLE."""
        lines = tui.console.render_lines(tui._render_frame(), tui.console.options, pad=False)
        for idx, line in enumerate(lines):
            if needle in "".join(segment.text for segment in line):
                return idx, len(lines)
        self.fail(f"{needle!r} not found in the frame")

    def _screen_row(self, tui, frame_index, total):
        height = tui.console.height
        return (height - total) + frame_index + 1

    def _tui_with_tool_call(self):
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            _tool_call_message(command="ls -la --time-style=full-iso ~/projects"),
        ]
        return tui, buf

    def test_click_expands_the_clicked_tool_row(self):
        tui, _buf = self._tui_with_tool_call()
        tui.agent_running = True
        frame_index, total = self._frame_line_of(tui, "▶ tool: Bash")
        row = self._screen_row(tui, frame_index, total)
        tui._on_mouse_click(5, row)
        self.assertIn("1", tui._expanded_tools)

    def test_second_click_collapses(self):
        tui, _buf = self._tui_with_tool_call()
        tui.agent_running = True
        frame_index, total = self._frame_line_of(tui, "▶ tool: Bash")
        row = self._screen_row(tui, frame_index, total)
        tui._on_mouse_click(5, row)
        self.assertIn("1", tui._expanded_tools)
        # re-render: the expanded rows shift the layout, so recompute
        frame_index, total = self._frame_line_of(tui, "▶ tool: Bash")
        tui._on_mouse_click(5, self._screen_row(tui, frame_index, total))
        self.assertNotIn("1", tui._expanded_tools)

    def test_click_outside_a_tool_row_does_nothing(self):
        tui, _buf = self._tui_with_tool_call()
        tui.agent_running = True
        self._frame_line_of(tui, "▶ tool: Bash")  # renders + stores the frame
        tui._on_mouse_click(5, 1)  # status bar
        self.assertEqual(tui._expanded_tools, set())

    def test_click_ignored_when_not_running(self):
        """The panel only exists during a run; a stray click is inert."""
        tui, _buf = self._tui_with_tool_call()
        tui.agent_running = False
        frame_index, total = self._frame_line_of(tui, "▶ tool: Bash")
        tui._on_mouse_click(5, self._screen_row(tui, frame_index, total))
        self.assertEqual(tui._expanded_tools, set())

    def test_click_without_a_frame_is_inert(self):
        tui, _buf = self._tui_with_tool_call()
        tui.agent_running = True
        tui._last_frame = None
        tui._on_mouse_click(1, 1)  # must not raise
        self.assertEqual(tui._expanded_tools, set())

    def test_expanded_rows_show_the_full_argument(self):
        """The whole command is visible once expanded (no truncation)."""
        tui, buf = make_tui()
        command = "grep -rn " + "x" * 300 + " ."
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            _tool_call_message(command=command),
        ]
        tui._expanded_tools.add("1")
        tui._history_dirty = True
        buf.seek(0)
        buf.truncate()
        tui.console.print(tui._render_conversation())
        # wrap the captured frame back into logical lines: the panel
        # wraps to the console width, the value itself is intact
        out = "".join(buf.getvalue().split("\n"))
        self.assertIn(command, out)  # full value, not truncated
        self.assertNotIn("…", out)

    def test_collapsed_row_is_a_single_truncated_line(self):
        """Unclicked tool calls keep the compact one-line form."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            _tool_call_message(command="y" * 300),
        ]
        tui._history_dirty = True
        buf.seek(0)
        buf.truncate()
        tui.console.print(tui._render_conversation())
        out = "".join(buf.getvalue().split("\n"))
        self.assertIn("▶ tool: Bash(command=", out)
        self.assertIn("…", out)
        self.assertEqual(buf.getvalue().count("▶"), 1)

    def test_tool_rows_carry_their_call_id(self):
        tui, _buf = self._tui_with_tool_call()
        rows = tui._build_history_rows()
        tool_rows = [r for r in rows if getattr(r, "tool_id", None) == "1"]
        self.assertEqual(len(tool_rows), 1)
        tui._expanded_tools.add("1")
        tui._history_dirty = True
        expanded = tui._build_history_rows()
        # header + one argument row, all clickable
        self.assertEqual(len([r for r in expanded if getattr(r, "tool_id", None) == "1"]), 2)

    def test_expanded_value_stays_inside_a_short_terminal(self):
        """An expansion always shows its value, even when it is huge.

        The cap is scaled to the visible area: a 3000-char command used
        to be dropped by the row budget (its estimate no longer fit), so
        clicking showed a ``▾`` header with nothing under it.
        """
        tui, _buf = make_tui()
        tui.console = Console(file=io.StringIO(), width=100, height=25, force_terminal=False)
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            _tool_call_message(command="grep -rn " + "x" * 3000),
        ]
        tui._expanded_tools.add("1")
        tui._history_dirty = True
        lines = tui.console.render_lines(tui._render_frame(), tui.console.options, pad=False)
        texts = ["".join(segment.text for segment in line) for line in lines]
        self.assertTrue(any("▶ tool: Bash (1 argument) ▾" in t for t in texts))
        self.assertTrue(any("command =" in t for t in texts))
        # the frame still fits, so a click maps onto the rows
        self.assertLessEqual(len(texts), tui.console.height)
        self.assertIsNotNone(tui._frame_top(len(texts)))

    def test_clicked_row_never_vanishes_when_the_block_no_longer_fits(self):
        """A block the budget cannot pay for degrades, never disappears.

        The expanded block sits *before* its tool result, so a long
        result spends the budget first; the block used to be dropped
        whole, leaving the user without the row they had just clicked
        and with no way to collapse it again.
        """
        tui, _buf = make_tui()
        tui.console = Console(file=io.StringIO(), width=100, height=12, force_terminal=False)
        tui.session.todos = [{"content": "task one", "status": "in_progress"}]
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            _tool_call_message(command="grep -rn " + "x" * 3000),
            Message(
                role="tool",
                content="\n".join(f"line {i}" for i in range(60)),
                tool_call_id="1",
                name="Bash",
            ),
        ]
        tui._expanded_tools.add("1")
        tui._history_dirty = True
        kept = [
            r
            for r in tui._apply_budget(tui._build_history_rows())
            if getattr(r, "tool_id", None) == "1"
        ]
        self.assertEqual(len(kept), 1, "the clicked call must still be on screen")
        self.assertIn("▶ tool: Bash", kept[0].plain)
        self.assertNotIn("▾", kept[0].plain)  # the collapsed form, not a bare header
        # the click target is still live: clicking it toggles the call off
        self.assertIn("1", tui._expanded_tools)

    def test_wide_characters_are_budgeted_in_cells_not_chars(self):
        """A CJK value is twice as wide as its character count.

        Cutting the value at "N chars" overflowed the row budget by a
        factor of two, so the frame outgrew the terminal and the whole
        expansion was dropped.
        """
        tui, _buf = make_tui()
        tui.console = Console(file=io.StringIO(), width=100, height=25, force_terminal=False)
        tui.session.todos = []
        tui.session.last_messages = [
            Message(role="user", content="go"),
            _tool_call_message(command="x" + "命" * 3000),
        ]
        tui.agent_running = True
        tui._expanded_tools.add("1")
        tui._history_dirty = True
        rows = [r for r in tui._build_history_rows() if getattr(r, "tool_id", None) == "1"]
        self.assertEqual(len(rows), 2, "the expansion must be shown, not dropped")
        used = sum(tui._est_lines(r, 100) for r in rows)
        self.assertLessEqual(used, tui._expanded_block_lines())
        lines = tui.console.render_lines(tui._render_frame(), tui.console.options, pad=False)
        self.assertLessEqual(len(lines), tui.console.height)
        self.assertIsNotNone(tui._frame_top(len(lines)))

    def test_long_todo_stays_on_one_line_in_the_pinned_panel(self):
        """The panel height is reserved per todo, so a todo must not wrap.

        A wrapping item made the pinned panel taller than
        ``_visible_row_cap`` reserved for it: the frame outgrew the
        terminal, rich cropped the newest rows, and every click lost its
        row mapping.
        """
        tui, _buf = make_tui()
        tui.console = Console(file=io.StringIO(), width=100, height=20, force_terminal=False)
        tui.session.todos = [{"content": "a very long todo item " * 30, "status": "pending"}]
        tui.session.last_messages = [Message(role="user", content=f"m{i}") for i in range(40)]
        tui._history_dirty = True
        lines = tui.console.render_lines(tui._render_frame(), tui.console.options, pad=False)
        texts = ["".join(segment.text for segment in line) for line in lines]
        # title + exactly one line for the single todo
        self.assertEqual(texts[1], "Todos")
        self.assertTrue(texts[2].startswith("⬜ a very long todo item"))
        self.assertIn("…", texts[2])
        self.assertLessEqual(len(texts), tui.console.height)
        self.assertIsNotNone(tui._frame_top(len(texts)))

    def test_expanded_value_limit_scales_with_the_visible_area(self):
        """The value cap never exceeds the configured ceiling."""
        from python_agent_harness.session import config

        value = "x" * 10000

        def shown_value(height: int, width: int) -> int:
            tui, _buf = make_tui()
            tui.console = Console(
                file=io.StringIO(), width=width, height=height, force_terminal=False
            )
            tui.session.todos = []
            tui.session.last_messages = [
                Message(role="user", content="run it"),
                _tool_call_message(command=value),
            ]
            tui._expanded_tools.add("1")
            tui._history_dirty = True
            rows = [r for r in tui._build_history_rows() if getattr(r, "tool_id", None) == "1"]
            self.assertEqual(len(rows), 2, "the expansion must be shown, not dropped")
            return rows[1].plain.count("x")

        short = shown_value(25, 100)
        self.assertLess(short, config.TOOL_EXPANDED_ARG_CHARS)
        # "fills the lines it was given" and not "shows something": the
        # cap used to be derived as ``lines * width - len(prefix)``, which
        # ignores that a value that cannot fit after the prefix is folded
        # onto a FRESH line -- every candidate then measured one line too
        # tall and the retry loop bottomed out at a single line (84 cells
        # on a 25x100 terminal instead of the ~1800 that fit).
        self.assertGreaterEqual(short, 15 * 100)
        tall = shown_value(60, 200)
        # the ceiling counts cells, and the ellipsis takes one of them
        self.assertGreaterEqual(tall, config.TOOL_EXPANDED_ARG_CHARS - 2)

    def test_many_argument_expansion_keeps_the_frame_on_screen(self):
        """Eight arguments still leave a frame that fits and maps clicks.

        Word wrapping cost one extra line per row over the old naive
        estimate, so the frame outgrew a 25-row terminal; rich then
        crops it, ``_frame_top`` returns None, and *every* click in the
        panel stopped working.
        """
        tui, _buf = make_tui()
        tui.console = Console(file=io.StringIO(), width=100, height=25, force_terminal=False)
        tui.session.todos = []
        tui.session.last_messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="1", name="Custom", arguments={f"k{i}": "v" * 300 for i in range(8)}
                    )
                ],
            ),
        ]
        tui.agent_running = True
        tui._expanded_tools.add("1")
        tui._history_dirty = True
        lines = tui.console.render_lines(tui._render_frame(), tui.console.options, pad=False)
        texts = ["".join(segment.text for segment in line) for line in lines]
        self.assertLessEqual(len(texts), tui.console.height)
        header = next(i for i, t in enumerate(texts) if "▶ tool: Custom (8 arguments) ▾" in t)
        # header and every value row survive together
        self.assertEqual(len([t for t in texts if t.startswith("    k")]), 8)
        # the shared budget is actually spent on the values, not wasted by
        # a per-row provider that measures the wrap wrong
        used = sum(
            tui._est_lines(r, 100) for r in tui._build_history_rows() if getattr(r, "tool_id", None)
        )
        self.assertEqual(used, tui._expanded_block_lines())
        top = tui._frame_top(len(texts))
        self.assertEqual(tui._toggle_tool_at(header + top), "1")

    def test_frame_budget_never_splits_a_tool_call(self):
        """A call's rows are dropped or kept as one unit."""
        tui, _buf = make_tui()
        tui.console = Console(file=io.StringIO(), width=100, height=12, force_terminal=False)
        block = tui._tool_call_rows(
            ToolCall(id="1", name="Bash", arguments={"command": "x" * 400, "timeout": 5})
        )
        tui._expanded_tools.add("1")
        rows = [Text("older"), *block]
        kept = tui._apply_budget(rows)
        kept_ids = [getattr(r, "tool_id", None) for r in kept]
        self.assertIn("1", kept_ids)
        # header + every value row, never a subset
        self.assertEqual(kept_ids.count("1"), len(block))
        self.assertEqual(len([r for r in kept if getattr(r, "tool_id", None) == "1"]), len(block))
        # exhausted budget: the whole block goes, never half of it
        kept = tui._apply_budget(block)
        self.assertEqual(len(kept), len(block))

    def test_est_lines_matches_the_rendered_height(self):
        """The budget estimate must never under-count a wrapped row.

        A row like ``k0 = 'vvv…'`` wraps at the space, so counting
        ``len(text) / width`` billed one line less than rich draws.
        """
        from python_agent_harness.io.diffrender import render_diff

        tui, _buf = make_tui()
        tui.console = Console(file=io.StringIO(), width=40, height=40, force_terminal=False)
        rows = [
            Text("    k0 = " + "'" + "v" * 130 + "'"),
            Text("word " * 40),
            Text("短い日本語のテキスト" * 6),
            Text("a\n\nb"),
            render_diff("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"),
        ]
        for row in rows:
            real = len(tui.console.render_lines(row, tui.console.options, pad=False))
            self.assertEqual(tui._est_lines(row, 40), real)


class TestFramePositionOnScreen(unittest.TestCase):
    """Clicks use the panel's real position, not a bottom-anchored guess.

    rich's Live repaints from wherever the cursor is, so a panel shorter
    than the screen leaves blank rows below it: a click row is only
    meaningful together with the row the panel starts on, which the
    terminal reports (DSR).
    """

    class _FakeMouse:
        def __init__(self, row):
            self.row = row
            self.queries = 0

        def cursor_row(self, timeout=0.2):
            self.queries += 1
            return self.row

    def _tui(self):
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            _tool_call_message(command="ls -la --time-style=full-iso ~/projects"),
        ]
        tui.agent_running = True
        return tui, buf

    def _frame(self, tui):
        lines = tui.console.render_lines(tui._render_frame(), tui.console.options, pad=False)
        texts = ["".join(seg.text for seg in line) for line in lines]
        return texts

    def test_click_uses_the_reported_panel_top(self):
        """A panel floating mid-screen still maps clicks onto its rows."""
        tui, _buf = self._tui()
        texts = self._frame(tui)
        total = len(texts)
        frame_index = next(i for i, t in enumerate(texts) if "▶ tool: Bash" in t)
        # pretend the frame starts at screen row 4 (blank space below it);
        # the reported cursor sits on the frame's last li
        top = 4
        tui._mouse = self._FakeMouse(top + total - 1)
        tui._on_mouse_click(5, top + frame_index)
        self.assertIn("1", tui._expanded_tools)

    def test_bottom_anchor_guess_is_not_used_when_a_row_is_reported(self):
        """The same click lands elsewhere under the old bottom-anchor math."""
        tui, _buf = self._tui()
        texts = self._frame(tui)
        total = len(texts)
        frame_index = next(i for i, t in enumerate(texts) if "▶ tool: Bash" in t)
        top = 4
        tui._mouse = self._FakeMouse(top + total - 1)
        # the bottom-anchored row for this frame line is a different line
        height = tui.console.height
        wrong_row = (height - total) + frame_index + 1
        self.assertNotEqual(wrong_row, top + frame_index)
        tui._on_mouse_click(5, wrong_row)
        self.assertEqual(tui._expanded_tools, set())

    def test_click_below_the_panel_does_nothing(self):
        tui, _buf = self._tui()
        texts = self._frame(tui)
        total = len(texts)
        top = 4
        tui._mouse = self._FakeMouse(top + total - 1)
        tui._on_mouse_click(5, top + total + 2)
        self.assertEqual(tui._expanded_tools, set())

    def test_falls_back_to_bottom_anchor_without_an_answer(self):
        """A terminal that ignores DSR keeps the previous behaviour."""
        tui, _buf = self._tui()
        texts = self._frame(tui)
        total = len(texts)
        frame_index = next(i for i, t in enumerate(texts) if "▶ tool: Bash" in t)
        tui._mouse = self._FakeMouse(None)
        height = tui.console.height
        tui._on_mouse_click(5, (height - total) + frame_index + 1)
        self.assertIn("1", tui._expanded_tools)

    def test_no_click_mapping_when_the_frame_is_cropped(self):
        """A frame taller than the screen is cropped: rows do not map."""
        tui, _buf = self._tui()
        self._frame(tui)
        height = tui.console.height
        self.assertIsNone(tui._frame_top(height + 1))

    def test_repaint_during_measurement_remeasures_the_fresh_frame(self):
        """A repaint mid-click must not shift the mapping onto a neighbour.

        The cursor answer describes whatever the terminal last painted:
        if that paint happened after the frame was measured, the two
        belong to different frames and the click would land elsewhere.
        The handler notices and measures the fresh frame instead.
        """

        class _RepaintingMouse:
            def __init__(self, tui, row):
                self.tui = tui
                self.row = row
                self.queries = 0

            def cursor_row(self, timeout=0.2):
                self.queries += 1
                if self.queries == 1:
                    # the render loop repaints while we are answering
                    self.tui._render_frame()
                return self.row

        tui, _buf = self._tui()
        texts = self._frame(tui)
        total = len(texts)
        frame_index = next(i for i, t in enumerate(texts) if "▶ tool: Bash" in t)
        top = 4
        fake = _RepaintingMouse(tui, top + total - 1)
        tui._mouse = fake
        tui._on_mouse_click(5, top + frame_index)
        self.assertEqual(fake.queries, 2)  # asked again for the fresh frame
        self.assertIn("1", tui._expanded_tools)

    def test_click_still_works_when_the_panel_never_settles(self):
        """A panel that repaints on every query still yields a best guess."""

        class _RestlessMouse:
            def __init__(self, tui, row):
                self.tui = tui
                self.row = row
                self.queries = 0

            def cursor_row(self, timeout=0.2):
                self.queries += 1
                self.tui._render_frame()
                return self.row

        tui, _buf = self._tui()
        texts = self._frame(tui)
        total = len(texts)
        frame_index = next(i for i, t in enumerate(texts) if "▶ tool: Bash" in t)
        top = 4
        fake = _RestlessMouse(tui, top + total - 1)
        tui._mouse = fake
        tui._on_mouse_click(5, top + frame_index)
        self.assertEqual(fake.queries, 3)  # bounded, no endless retry
        self.assertIn("1", tui._expanded_tools)


class TestTypedDuringRun(unittest.TestCase):
    def test_typed_text_is_replayed_into_the_next_prompt(self):
        """Keystrokes typed while mouse reporting owned stdin are kept."""
        tui, _buf = make_tui()
        tui._collect_typed("hel")
        tui._collect_typed("lo")
        with mock.patch.object(tui.prompt_session, "prompt", return_value="hello") as m:
            tui._read_multiline()
        self.assertEqual(m.call_args.kwargs.get("default"), "hello")
        # consumed once
        with mock.patch.object(tui.prompt_session, "prompt", return_value="") as m:
            tui._read_multiline()
        self.assertEqual(m.call_args.kwargs.get("default"), "")


if __name__ == "__main__":
    unittest.main()
