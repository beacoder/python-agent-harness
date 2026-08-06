"""TUI rendering regression tests."""

import io
import unittest

from rich.console import Console
from rich.live import Live

from python_agent_harness.client import Client
from python_agent_harness.harness import AgentSession
from python_agent_harness.models import Message, ToolCall
from python_agent_harness.tools import default_registry
from python_agent_harness.tui import Tui


def make_tui() -> tuple[Tui, io.StringIO]:
    c = Client(base_url="http://127.0.0.1:1/v1", api_key="x", model="fake")
    s = AgentSession(
        project_dir="/tmp/fakeproj", client=c, model="fake",
        registry=default_registry(),
    )
    s.last_messages = [
        Message(role="user", content="hello agent"),
        Message(
            role="assistant", content="hi",
            tool_calls=[ToolCall(id="1", name="Read", arguments="{}")],
        ),
        Message(role="tool", content="file contents", tool_call_id="1", name="Read"),
    ]
    s.todos = [{"content": "task one", "status": "in_progress"}]
    s.context_ratio = 0.55
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False)
    return Tui(s, console), buf


class TestTui(unittest.TestCase):
    def test_render_conversation(self):
        tui, buf = make_tui()
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("hello agent", out)
        self.assertIn("file contents", out)
        self.assertIn("Todos", out)

    def test_status_bar(self):
        tui, buf = make_tui()
        tui.console.print(tui._status_bar())
        out = buf.getvalue()
        self.assertIn("[BUILD]", out)
        self.assertIn("Ctx:55%", out)

    def test_frame_includes_status_bar(self):
        """The full frame (used by Live and idle loop) must show the status bar."""
        tui, buf = make_tui()
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertIn("hello agent", out)
        self.assertIn("[BUILD]", out)
        self.assertIn("Ctx:55%", out)

    def test_status_bar_pinned_on_top(self):
        """The status bar must come BEFORE the conversation panel so a
        tall panel can't push it off the bottom of the terminal."""
        tui, buf = make_tui()
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertLess(out.index("[BUILD]"), out.index("hello agent"))

    def test_role_labels(self):
        """Roles render as user / assistant / tool, not You / Agent."""
        tui, buf = make_tui()
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("user: hello agent", out)
        self.assertIn("assistant: hi", out)
        self.assertIn("tool: Read: file contents", out)
        self.assertNotIn("You:", out)
        self.assertNotIn("Agent:", out)

    def test_long_conversation_bounded(self):
        """A long conversation is capped so rendering stays fast."""
        tui, buf = make_tui()
        msgs = []
        for i in range(120):
            msgs.append(Message(role="user", content=f"message number {i}"))
        tui.session.last_messages = msgs
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        # oldest rows are dropped (only the last ~60 rows are rendered)
        self.assertNotIn("message number 0", out)
        self.assertIn("message number 119", out)

    def test_stream_preview_capped(self):
        tui, buf = make_tui()
        tui.stream_text = "x" * 10_000
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        # only the tail is rendered, so output stays bounded
        self.assertLess(len(out), 20_000)
        self.assertIn("assistant:", out)

    def test_height_budget_keeps_newest(self):
        """On a 24-line terminal only ~18 rows are kept — the NEWEST ones."""
        from types import SimpleNamespace

        tui, _ = make_tui()
        tui.session.last_messages = [
            Message(role="user", content=f"m{i}") for i in range(100)
        ]
        tui.console = SimpleNamespace(height=24, width=100)  # cap_rows = 18
        panel = tui._render_conversation()
        out_console, buf = make_tui()
        out_console.console.print(panel)
        out = buf.getvalue()
        self.assertIn("m99", out)    # newest row visible
        self.assertNotIn("m0", out)  # oldest dropped
        self.assertNotIn("m50", out) # middle rows dropped too

    def test_stream_tail_visible_when_huge(self):
        """A huge stream must keep its TAIL (the progress) on screen."""
        tui, buf = make_tui()
        tui.agent_running = True
        tui.stream_text = "".join(f"line {i}\n" for i in range(2000))
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("line 1999", out)    # newest progress visible
        self.assertNotIn("line 0\n", out)  # head dropped
        self.assertLess(len(out), 20_000)

    def test_delta_wakes_render_event(self):
        """Each stream delta must set the data event so the render loop
        wakes immediately (text pushes the scroll, no tick delay)."""
        tui, _ = make_tui()
        self.assertFalse(tui._data_event.is_set())
        tui._on_delta("chunk one ")
        self.assertTrue(tui._data_event.is_set())
        tui._data_event.clear()
        tui._on_delta("chunk two")
        self.assertTrue(tui._data_event.is_set())
        self.assertEqual(tui.stream_text, "chunk one chunk two")

    def test_delta_buffer_bounded(self):
        """The stream buffer is bounded so per-frame slicing stays fast
        even for very long generations."""
        tui, _ = make_tui()
        for _ in range(5000):
            tui._on_delta("x" * 100)  # 500KB total
        self.assertLessEqual(len(tui.stream_text), 100_000)
        self.assertEqual(tui.stream_text, "x" * 100_000)  # tail kept

    def test_history_rows_cached_between_stream_updates(self):
        """History rows must be cached while streaming: only the stream
        row changes per frame, so rendering keeps up with fast text."""
        tui, _ = make_tui()
        rows_a = tui._history_rows()
        tui.stream_text = "streaming data"  # no history change
        rows_b = tui._history_rows()
        self.assertEqual(len(rows_a), len(rows_b))
        self.assertTrue(all(x is y for x, y in zip(rows_a, rows_b)))
        tui._history_dirty = True
        rows_c = tui._history_rows()
        self.assertFalse(all(x is y for x, y in zip(rows_c, rows_b)))

    def test_render_fast_with_cached_history(self):
        """Rendering with cached history + growing stream must stay fast."""
        import time

        tui, _ = make_tui()
        tui._history_rows()  # warm the cache
        tui.stream_text = "x" * 2000
        t0 = time.perf_counter()
        for _ in range(50):
            tui._render_frame()
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 1.0, f"50 frames took {elapsed:.2f}s")

    def test_frame_spinner_while_running(self):
        tui, buf = make_tui()
        tui.agent_running = True
        tui.status = " running tools"
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertIn("[BUILD]", out)
        self.assertIn("running tools", out)
        self.assertTrue(any(ch in out for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"))

    def test_live_render_no_crash(self):
        """Regression: Live(console) used to crash with NotRenderableError.

        The renderable must be the first positional argument; the console
        must be passed by keyword.
        """
        tui, _ = make_tui()
        with Live(
            tui._render_conversation(),
            console=tui.console,
            refresh_per_second=10,
            screen=False,
        ):
            pass

    def test_diff_rendered_for_edit_tool_call(self):
        """A ToolCall.diff on an Edit/Write call is rendered in the panel."""
        tui, buf = make_tui()
        diff_text = (
            "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n"
            " line1\n-old line\n+new line\n"
        )
        tui.session.last_messages = [
            Message(role="user", content="edit the file"),
            Message(
                role="assistant", content="",
                tool_calls=[ToolCall(id="e1", name="Edit", arguments="{}", diff=diff_text)],
            ),
            Message(
                role="tool", content="Successfully replaced text in f.py",
                tool_call_id="e1", name="Edit",
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("old line", out)
        self.assertIn("new line", out)
        self.assertIn("@@", out)

    def test_no_diff_panel_when_diff_absent(self):
        """Tool calls without a diff (e.g. Read) render no diff block."""
        tui, buf = make_tui()
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertNotIn("@@", out)
        self.assertNotIn("no changes", out)


if __name__ == "__main__":
    unittest.main()
