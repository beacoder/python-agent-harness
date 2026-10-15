"""Untrusted text in the TUI: markup injection and terminal escapes.

Regressions:
- dynamic values (paths, errors, user input) interpolated into Rich
  markup strings raised MarkupError on text like ``notes[/draft].md``,
  and the main loop only caught KeyboardInterrupt, so the TUI exited;
- model-written todos went through ``Table.add_row`` as markup strings,
  so a todo mentioning ``[/tmp]`` crashed the render thread;
- model output, tool results and diffs reached the terminal with their
  escape sequences intact (window title, screen clear, OSC 52
  clipboard writes, ...).
"""

import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from rich.console import Console

from python_agent_harness.core.models import Message, ToolCall
from python_agent_harness.io.attachments import parse_at_references
from python_agent_harness.tui.render import safe_markup, sanitize_terminal
from tests.support import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from tests.support.tui_test_utils import make_tui

ESC = "\x1b"
HOSTILE = (
    "before"
    "\x1b]0;PWNED-TITLE\x07"  # OSC set title (BEL-terminated)
    "\x1b]52;c;ZWNobyBwd25lZA==\x1b\\"  # OSC 52 clipboard write (ST-terminated)
    "\x1b[2J\x1b[H"  # CSI clear screen + cursor home
    "\x9b31m"  # 8-bit C1 CSI
    "\rafter"  # carriage return (overwrite the line)
)


def _plain_console() -> tuple[Console, io.StringIO]:
    """No color system: any ESC in the output came from the content."""
    buf = io.StringIO()
    return Console(file=buf, width=120, force_terminal=False, color_system=None), buf


class TestSanitizeTerminal(unittest.TestCase):
    def test_strips_sequences_and_control_chars(self):
        self.assertEqual(sanitize_terminal(HOSTILE), "beforeafter")

    def test_keeps_tabs_newlines_and_unicode(self):
        text = "a\tb\nc — ✓ 日本語 [bold]x[/bold]"
        self.assertEqual(sanitize_terminal(text), text)

    def test_unterminated_osc_is_dropped_to_end_of_line_only(self):
        self.assertEqual(sanitize_terminal("ok\x1b]0;no terminator"), "ok")
        # a stray introducer must not hide the following lines
        self.assertEqual(
            sanitize_terminal("ok\x1b]0;junk\nrm -rf / is visible"), "ok\nrm -rf / is visible"
        )

    def test_safe_markup_renders_literally(self):
        console, buf = _plain_console()
        for value in ("notes[/draft].md", "arr[i] out of range", "use [/] here"):
            with self.subTest(value=value):
                console.print(f"[red]cannot read {safe_markup(value)}: boom[/red]")
                self.assertIn(f"cannot read {value}: boom", buf.getvalue())


class TestTuiMarkupInjection(unittest.TestCase):
    def test_bracketed_attachment_error_prints_and_does_not_raise(self):
        """The exact core.py path: a missing @file whose name contains a
        closing-tag lookalike."""
        _text, _atts, errors = parse_at_references("summarize @notes[/draft].md please", "/tmp")
        self.assertTrue(errors)
        console, buf = _plain_console()
        for err in errors:
            console.print(f"[red]@{safe_markup(err.path)}: {safe_markup(err.message)}[/red]")
        self.assertIn("@notes[/draft].md", buf.getvalue())

    def test_model_written_todo_with_brackets_renders(self):
        tui, buf = make_tui()
        tui.session.todos = [{"content": "fix parser for [/tmp] paths", "status": "pending"}]
        tui.console.print(tui._render_frame())  # raised MarkupError before
        self.assertIn("fix parser for [/tmp] paths", buf.getvalue())

    def test_main_loop_survives_an_unexpected_exception(self):
        """An exception escaping a command/run must be reported, not
        close the app: the loop keeps reading input."""
        tui, buf = make_tui()
        inputs = iter(["do something", None])  # one prompt, then Ctrl-D
        with (
            mock.patch.object(tui, "_read_multiline", side_effect=lambda: next(inputs)),
            mock.patch.object(tui, "_start_agent", side_effect=RuntimeError("render bug")),
            mock.patch.object(tui, "_status_bar", return_value=""),
        ):
            tui.run()  # must return normally at the Ctrl-D
        self.assertIn("internal error: RuntimeError: render bug", buf.getvalue())
        self.assertFalse(tui.agent_running)

    def test_display_loop_failure_cancels_and_releases_the_run(self):
        """A rendering bug while a run is live: the run is cancelled and
        what it borrowed (a slash command's project dir) is restored --
        like Ctrl-C -- before the error reaches the main loop."""
        tui, _ = make_tui()
        restored = mock.Mock()
        handle = mock.Mock(errors=[], warnings=[], display_text="hi", seq=1)
        with (
            mock.patch.object(tui._controller, "submit", return_value=handle),
            mock.patch.object(tui._controller, "cancel") as cancel,
            mock.patch.object(tui, "_run_live", side_effect=RuntimeError("render bug")),
            mock.patch.object(tui, "_run_dumb", side_effect=RuntimeError("render bug")),
            self.assertRaises(RuntimeError),
        ):
            tui._start_agent("hi", restore=restored)
        cancel.assert_called_once()
        restored.assert_called_once()
        self.assertIsNone(tui._restore)
        self.assertFalse(tui.agent_running)

    def test_error_outside_a_run_does_not_cancel_anything(self):
        tui, _ = make_tui()
        inputs = iter(["/broken", None])
        with (
            mock.patch.object(tui, "_read_multiline", side_effect=lambda: next(inputs)),
            mock.patch.object(tui, "_handle_slash", side_effect=RuntimeError("cmd bug")),
            mock.patch.object(tui, "_status_bar", return_value=""),
            mock.patch.object(tui._controller, "cancel") as cancel,
        ):
            tui.run()
        cancel.assert_not_called()

    def test_persistent_error_is_reraised_instead_of_spinning(self):
        """A failure that repeats with no successful read in between
        (dead terminal: every read raises) must end the app, not busy-
        loop forever at full CPU."""
        tui, _ = make_tui()
        reads = {"n": 0}

        def dead_terminal():
            reads["n"] += 1
            raise OSError(5, "Input/output error")

        with (
            mock.patch.object(tui, "_read_multiline", side_effect=dead_terminal),
            mock.patch.object(tui, "_status_bar", return_value=""),
            self.assertRaises(OSError),
        ):
            tui.run()
        self.assertEqual(reads["n"], 3)

    def test_successful_reads_reset_the_error_budget(self):
        """Occasional errors separated by working input never add up."""
        tui, buf = make_tui()
        inputs = iter(["a", "b", "c", "d", None])
        with (
            mock.patch.object(tui, "_read_multiline", side_effect=lambda: next(inputs)),
            mock.patch.object(tui, "_start_agent", side_effect=RuntimeError("flaky")),
            mock.patch.object(tui, "_status_bar", return_value=""),
        ):
            tui.run()  # four errors, each after a successful read: no raise
        self.assertEqual(buf.getvalue().count("internal error: RuntimeError: flaky"), 4)


class TestTuiTerminalEscapes(unittest.TestCase):
    def _render(self, messages, stream: str = "") -> str:
        tui, _ = make_tui()
        console, buf = _plain_console()
        tui.console = console
        tui.session.todos = [{"content": "todo" + HOSTILE, "status": "pending"}]
        tui.session.last_messages = messages
        tui._history_dirty = True
        tui.stream_text = stream
        console.print(tui._render_frame())
        tui._dump_conversation()
        return buf.getvalue()

    def test_no_escape_reaches_the_terminal(self):
        diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new" + HOSTILE + "\n"
        call = ToolCall(id="1", name="Read", arguments='{"file_path": "x' + HOSTILE + '"}')
        call.diff = diff
        out = self._render(
            [
                Message(role="user", content="user says" + HOSTILE),
                Message(role="assistant", content="assistant says" + HOSTILE, tool_calls=[call]),
                Message(
                    role="tool", content="tool output" + HOSTILE, tool_call_id="1", name="Read"
                ),
            ],
            stream="streaming" + HOSTILE,
        )
        self.assertNotIn(ESC, out)
        self.assertNotIn("\x9b", out)
        self.assertNotIn("\x07", out)
        self.assertNotIn("PWNED-TITLE", out)
        # the visible text survives
        for visible in ("user says", "assistant says", "tool output", "streaming", "+new"):
            self.assertIn(visible, out)

    def test_markdown_links_show_their_real_target(self):
        """[trusted text](evil url) must not render as a clickable link
        whose target is hidden behind the trusted-looking text."""
        tui, _ = make_tui()
        buf = io.StringIO()
        tui.console = Console(file=buf, width=120, force_terminal=True)
        tui.session.last_messages = [
            Message(role="user", content="where is it?"),
            Message(
                role="assistant",
                content="see [https://github.com/org/repo](https://evil.example/x)",
            ),
        ]
        tui._history_dirty = True
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertNotIn("\x1b]8;", out)  # no terminal hyperlink
        self.assertIn("https://evil.example/x", out)  # the real target is visible


if __name__ == "__main__":
    unittest.main()
