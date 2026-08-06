"""TUI rendering regression tests."""

import io
import os
import tempfile
import unittest
import unittest.mock as mock

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
        tui.console.print(tui._render_frame())
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

    def test_tool_result_line_limited(self):
        """A tool result with many lines shows only the first few lines,
        with an explicit 'N more lines' marker — not the whole output."""
        tui, buf = make_tui()
        big = "".join(f"output line {i}\n" for i in range(200))
        tui.session.last_messages = [
            Message(role="user", content="read it"),
            Message(
                role="assistant", content="",
                tool_calls=[ToolCall(id="1", name="Read", arguments="{}")],
            ),
            Message(role="tool", content=big, tool_call_id="1", name="Read"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("output line 0", out)   # head shown
        self.assertIn("output line 4", out)   # within the 5-line cap
        self.assertNotIn("output line 5\n", out)
        self.assertNotIn("output line 199", out)
        self.assertIn("more lines", out)      # truncation marker

    def test_tool_result_short_untouched(self):
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            Message(
                role="assistant", content="",
                tool_calls=[ToolCall(id="1", name="Bash", arguments="{}")],
            ),
            Message(role="tool", content="ok\n", tool_call_id="1", name="Bash"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("ok", out)
        self.assertNotIn("more lines", out)

    def test_tool_result_long_single_line_capped(self):
        """Even without newlines, a huge tool result is char-capped."""
        tui, buf = make_tui()
        huge = "x" * 50_000
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            Message(
                role="assistant", content="",
                tool_calls=[ToolCall(id="1", name="Bash", arguments="{}")],
            ),
            Message(role="tool", content=huge, tool_call_id="1", name="Bash"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertLess(len(out), 5000)
        self.assertIn("…", out)

    def test_todos_notify_invalidates_history_cache(self):
        """A TodoWrite call notifies 'todos', which must mark the cached
        history rows dirty so the Todos panel appears."""
        tui, _ = make_tui()
        tui._history_rows()  # warm the cache
        self.assertFalse(tui._history_dirty)
        tui._on_notify("todos")
        self.assertTrue(tui._history_dirty)
        tui._data_event.clear()
        tui._on_notify("todos")
        self.assertTrue(tui._data_event.is_set())  # render wakes promptly

    def test_tools_notify_invalidates_history_cache(self):
        """A tool round ('tools' notify) marks history dirty so the
        tool-call and result rows appear live, not after the run."""
        tui, _ = make_tui()
        tui._history_rows()  # warm the cache
        self.assertFalse(tui._history_dirty)
        tui._on_notify("tools")
        self.assertTrue(tui._history_dirty)

    def test_stream_cleared_when_run_finishes(self):
        """When the run completes the live stream buffer is dropped so the
        final text isn't rendered twice (stream row + history row)."""
        from types import SimpleNamespace

        tui, _ = make_tui()
        tui.stream_text = "final words"
        s = tui.session
        fake = SimpleNamespace(close=lambda: None)
        s.client = fake
        tui._run_agent("hi", tui.run_seq + 1)  # run completes (no API key -> error path)
        self.assertEqual(tui.stream_text, "")

    def test_todos_panel_visible_after_midrun_update(self):
        """TodoWrite mid-run: session.todos is set while the history cache
        is already built; the pinned Todos panel must still appear
        without waiting for the run to finish."""
        tui, buf = make_tui()
        tui.session.todos = []          # no todos at run start
        tui._history_rows()             # build cache without todos
        self.assertIsNone(tui._todos_panel())
        # TodoWrite runs mid-run:
        tui.session.update_todos(
            [{"content": "task one", "status": "in_progress"},
             {"content": "task two", "status": "pending"}]
        )
        self.assertEqual(tui.session.todos[0]["content"], "task one")
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertIn("Todos", out)
        self.assertIn("task one", out)
        self.assertIn("task two", out)

    def test_todos_pinned_above_conversation(self):
        """The Todos panel is pinned below the status bar and ABOVE the
        conversation, so a long conversation can't push it away."""
        tui, buf = make_tui()
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertLess(out.index("[BUILD]"), out.index("Todos"))
        self.assertLess(out.index("Todos"), out.index("hello agent"))
        # even with a huge conversation, the Todos panel stays pinned
        tui.session.last_messages = [
            Message(role="user", content=f"m{i}") for i in range(200)
        ]
        tui._history_dirty = True
        buf2 = io.StringIO()
        out_c2 = Console(file=buf2, width=100, force_terminal=False)
        out_c2.print(tui._render_frame())
        out2 = buf2.getvalue()
        self.assertLess(out2.index("Todos"), out2.index("m199"))
        self.assertNotIn("m0", out2)

    def test_todos_panel_hidden_when_empty(self):
        tui, buf = make_tui()
        tui.session.todos = []
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertNotIn("Todos", out)

    def test_todos_panel_subagent_label(self):
        """While a sub-agent runs, its scoped todos show with a `sub:`
        label so they're not mistaken for the parent's list."""
        tui, buf = make_tui()
        tui.session.update_todos(
            [{"content": "parent task", "status": "pending"}]
        )
        tui.session.push_todo_scope("sub:1", "find the bug")
        tui.session.update_todos(
            [{"content": "search", "status": "in_progress"}]
        )
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertIn("Todos (sub: find the bug)", out)
        self.assertIn("search", out)
        # parent's list is restored on pop and the label disappears
        tui.session.pop_todo_scope()
        buf2 = io.StringIO()
        out_c2 = Console(file=buf2, width=100, force_terminal=False)
        out_c2.print(tui._render_frame())
        out2 = buf2.getvalue()
        self.assertIn("parent task", out2)
        self.assertNotIn("sub: find the bug", out2)

    # ------------------------------------------------------------------
    # slash commands (/init /review /explain)
    # ------------------------------------------------------------------
    def test_slash_command_args_parsing(self):
        """Arg parsing matches the CLI signatures: [project] first, then
        the command's argument; a lone non-directory token is the
        argument (so `/review main` reviews the branch, not a project)."""
        tui, _ = make_tui()
        self.assertEqual(tui._command_args("init", ""), (None, None))
        self.assertEqual(tui._command_args("init", "myproj"), ("myproj", None))
        self.assertEqual(
            tui._command_args("init", 'myproj --extra "focus CI"'),
            ("myproj", "focus CI"),
        )
        self.assertEqual(
            tui._command_args("init", "--extra x"), (None, "x")
        )
        self.assertEqual(tui._command_args("review", ""), (None, None))
        self.assertEqual(tui._command_args("review", "main"), (None, "main"))
        self.assertEqual(tui._command_args("review", "abc123"), (None, "abc123"))
        self.assertEqual(
            tui._command_args("explain", "client.py"), (None, "client.py")
        )
        self.assertEqual(
            tui._command_args("explain", "the retry logic"),
            (None, "the retry logic"),
        )
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(tui._command_args("review", d), (d, None))
            self.assertEqual(tui._command_args("explain", d), (d, None))
            self.assertEqual(
                tui._command_args("review", f"{d} main"), (d, "main")
            )

    def test_slash_dispatch_runs_command_in_session(self):
        """/init, /review and /explain run their SessionCommand in the
        current session: the command prompt becomes the run's system
        prompt and the kickoff message is the user text."""
        tui, _ = make_tui()
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["text"] = text
            captured["system"] = system
            captured["restore"] = restore

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            self.assertFalse(tui._handle_slash("/init"))
            self.assertIn("AGENTS.md", captured["text"])
            self.assertIn("Create or update", captured["system"])

            tui._handle_slash("/review main")
            self.assertIn("Review the requested code changes", captured["text"])
            self.assertIn("code reviewer", captured["system"])
            self.assertIn("main", captured["system"])  # $ARGUMENTS substituted

            tui._handle_slash("/explain client.py")
            self.assertIn("instructions", captured["text"])  # custom kickoff
            self.assertIn("client.py", captured["system"])
            self.assertIn("explain", captured["system"])

    def test_slash_command_project_borrowed_and_restored(self):
        """A project given to a slash command borrows the session's
        project dir for the run (tool cwd) and restores it afterwards."""
        tui, _ = make_tui()
        with tempfile.TemporaryDirectory() as d:
            def fake_start(text, system=None, restore=None):
                self.assertEqual(
                    tui.session.project_dir, os.path.abspath(d)
                )
                restore()  # simulate the run finishing

            with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
                tui._handle_slash(f"/init {d}")
        self.assertEqual(tui.session.project_dir, "/tmp/fakeproj")

    def test_slash_command_defaults_to_session_project(self):
        """Without a project the command runs in the session's project."""
        tui, _ = make_tui()
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["text"] = text
            captured["restore"] = restore

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/init")
        self.assertIn("/tmp/fakeproj", captured["text"])
        self.assertIsNone(captured["restore"])

    def test_explain_requires_target(self):
        tui, buf = make_tui()
        with mock.patch.object(tui, "_start_agent") as start:
            tui._handle_slash("/explain")
        start.assert_not_called()
        self.assertIn("needs a target", buf.getvalue())

    def test_unknown_slash_command(self):
        tui, buf = make_tui()
        self.assertFalse(tui._handle_slash("/bogus"))
        self.assertIn("unknown command", buf.getvalue())

    def test_help_lists_command_slashes(self):
        tui, buf = make_tui()
        tui._handle_slash("/help")
        out = buf.getvalue()
        for s in ("/init", "/review", "/explain"):
            self.assertIn(s, out)


if __name__ == "__main__":
    unittest.main()
