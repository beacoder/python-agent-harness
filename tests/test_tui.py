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
