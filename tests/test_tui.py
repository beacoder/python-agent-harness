"""TUI rendering regression tests."""

import io
import os
import tempfile
import unittest
import unittest.mock as mock

from rich.console import Console
from rich.live import Live

from python_agent_harness.client import Client
from python_agent_harness.agent_session import AgentSession
from python_agent_harness.models import Message, ToolCall
from python_agent_harness.tools import default_registry
from python_agent_harness.tui import (
    Tui, UiQuestion, _resolve_keyed_choice, _resolve_numbered_choice,
)


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

    def test_message_colors_distinct(self):
        """User and assistant bodies get distinct colors that do NOT
        collide with the tool colors (tool calls are cyan, tool results
        dim): bright green for user, bright blue for assistant."""
        from python_agent_harness.tui import ASSISTANT_STYLE, USER_STYLE

        tui, _ = make_tui()
        rows = tui._build_history_rows()
        styles = [getattr(r, "style", None) for r in rows]
        self.assertIn(USER_STYLE, styles)       # user body
        self.assertIn(ASSISTANT_STYLE, styles)  # assistant body
        self.assertIn("cyan", styles)           # tool call label
        self.assertIn("dim", styles)            # tool result
        self.assertNotEqual(USER_STYLE, ASSISTANT_STYLE)
        # user must not be confused with tool activity (cyan/dim) nor
        # with the panel border (plain green)
        self.assertNotIn(USER_STYLE, ("cyan", "dim", "green"))
        self.assertNotIn(ASSISTANT_STYLE, ("cyan", "dim", "green"))

    def test_stream_row_assistant_color(self):
        """The live stream row uses the assistant color so the in-flight
        response matches the stored assistant rows."""
        from python_agent_harness.tui import ASSISTANT_STYLE

        tui, _ = make_tui()
        tui.stream_text = "streaming answer text"
        row = tui._stream_row()
        self.assertEqual(row.style, ASSISTANT_STYLE)

    def test_dump_rows_keep_message_colors(self):
        """The full scrollback dump shares the role colors (same row
        builder as the live panel)."""
        from python_agent_harness.tui import ASSISTANT_STYLE, USER_STYLE

        tui, _ = make_tui()
        rows = tui._build_history_rows(full=True)
        styles = [getattr(r, "style", None) for r in rows]
        self.assertIn(USER_STYLE, styles)
        self.assertIn(ASSISTANT_STYLE, styles)

    def test_injected_user_prompts_hidden(self):
        """Auto-injected user prompts (plan / plan-mode / build-switch
        reminders, plan-exit notices) are harness bookkeeping — the TUI
        shows only what the user actually typed.  Messages flagged
        ``injected`` never render as user rows; real user input stays."""
        from python_agent_harness import config

        tui, buf = make_tui()
        plan_file = "/tmp/python-agent-plans-proj-ab12cd/PLAN.md"
        tui.session.last_messages = [
            Message(role="user", content="real typed request"),
            Message(role="user", content="plan.md contents", injected=True),
            Message(role="user", content="plan-mode.md contents", injected=True),
            Message(
                role="user",
                content=config.PLAN_EXIT_APPROVED_MESSAGE % plan_file,
                injected=True,
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("real typed request", out)
        self.assertNotIn("plan.md contents", out)
        self.assertNotIn("plan-mode.md contents", out)
        self.assertNotIn("plan at ", out)
        self.assertNotIn(plan_file, out)
        # the stored messages are untouched
        self.assertEqual(len(tui.session.last_messages), 4)

    def test_restored_injected_prompts_hidden(self):
        """Sessions restored from disk lose the ``injected`` flag (plain
        markdown round-trip) — the content checks must still keep
        harness prompts out of the panel: nudge, the
        <system-reminder>-wrapped plan/build-switch prompts, and the
        plan-exit approval notice."""
        from python_agent_harness import config

        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="typed during the restored session"),
            Message(role="user", content=config.NUDGE_MESSAGE),
            Message(
                role="user",
                content=(
                    "<system-reminder>\n# Plan Mode - System Reminder\n\n"
                    "plan.md body"
                ),
            ),
            Message(
                role="user",
                content=(
                    "<system-reminder>\nYour operational mode has changed "
                    "from plan to build.\nYou are no longer in read-only mode."
                ),
            ),
            Message(
                role="user",
                content=config.PLAN_EXIT_APPROVED_MESSAGE % "/old/plan/PLAN.md",
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("typed during the restored session", out)
        self.assertNotIn("Plan Mode - System Reminder", out)
        self.assertNotIn("no longer in read-only mode", out)
        self.assertNotIn(config.NUDGE_MESSAGE, out)
        self.assertNotIn("The plan at ", out)

    def test_nudge_and_final_check_hidden(self):
        """Harness bookkeeping is hidden from the panel: the injected
        completion-nudge user prompt and the assistant's [FINAL CHECK]
        block never show up — even when the check block is the reply's
        ONLY content (a reply carrying real content keeps its content).
        The stored messages are untouched — the agent loop keeps
        working exactly as before."""
        from python_agent_harness import config

        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="build the thing"),
            Message(
                role="assistant",
                content="Done. All tests pass.\n\n[FINAL CHECK]\n"
                "- Goal: build the thing\n- Status: SUCCESS\n"
                "- Evidence: tests pass",
            ),
            Message(role="user", content=config.NUDGE_MESSAGE),
            Message(
                role="assistant",
                content="[FINAL CHECK]\n- Goal: build the thing\n"
                "- Status: SUCCESS\n- Evidence: tests pass",
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("build the thing", out)
        self.assertIn("Done. All tests pass.", out)
        # the [FINAL CHECK] block is hidden even when it is the whole
        # reply — check-only messages never render
        self.assertNotIn("FINAL CHECK", out)
        self.assertNotIn("Status: SUCCESS", out)
        self.assertNotIn(config.NUDGE_MESSAGE, out)
        # the agent loop's history is untouched
        self.assertEqual(
            tui.session.last_messages[1].text(),
            "Done. All tests pass.\n\n[FINAL CHECK]\n"
            "- Goal: build the thing\n- Status: SUCCESS\n"
            "- Evidence: tests pass",
        )
        self.assertEqual(
            tui.session.last_messages[-1].text(),
            "[FINAL CHECK]\n- Goal: build the thing\n"
            "- Status: SUCCESS\n- Evidence: tests pass",
        )

    def test_final_check_without_header_kept(self):
        """A reply WITHOUT the [FINAL CHECK] header (just checklist
        bullets) is NOT hidden — only the header pattern filters."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="build the thing"),
            Message(
                role="assistant",
                content="Done.\n\n- Goal: build the thing\n"
                "- Status: SUCCESS\n- Evidence: tests pass",
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("Done.", out)
        self.assertIn("Goal:", out)
        self.assertIn("Status: SUCCESS", out)
        self.assertIn("Evidence:", out)

    def test_stream_pure_final_check_hidden(self):
        """A stream that is only the [FINAL CHECK] block never renders
        — the row goes blank on the final reply instead of flashing
        bookkeeping."""
        tui, _ = make_tui()
        tui.stream_text = "[FINAL CHECK]\n- Goal: x\n- Status: SUCCESS\n- Evidence: y"
        row = tui._stream_row()
        self.assertIsNone(row)

    def test_reasoning_streams_normally(self):
        """While streaming, reasoning content shows up live like any
        other text — the collapse only happens in the final history."""
        tui, _ = make_tui()
        tui.stream_text = "thinking hard...\n\nanswer here"
        row = tui._stream_row()
        self.assertIsNotNone(row)
        self.assertIn("thinking hard...", row.plain)
        self.assertIn("answer here", row.plain)

    def test_reasoning_collapsed_in_history(self):
        """Once the stream is done, the stored reasoning collapses to a
        '...' marker in the history so it stops eating TUI space; the
        answer stays fully visible.  The stored message is untouched."""
        tui, buf = make_tui()
        reasoning = "Let me think: Rayleigh scattering dominates..."
        tui.session.last_messages = [
            Message(role="user", content="why is the sky blue?"),
            Message(
                role="assistant",
                content=reasoning + "\n\nThe sky is blue due to Rayleigh scattering.",
                reasoning=reasoning,
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("💭 ...", out)
        self.assertIn("Rayleigh scattering.", out)
        self.assertNotIn("Let me think", out)
        # the stored message keeps its reasoning — only the display hides it
        self.assertEqual(
            tui.session.last_messages[1].reasoning, reasoning
        )
        self.assertIn("Let me think", tui.session.last_messages[1].text())

    def test_reasoning_collapsed_marker_shows_even_without_answer(self):
        """A reasoning-only assistant message (no answer content) still
        shows the collapse marker instead of vanishing entirely."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant", content="pensive thoughts here",
                reasoning="pensive thoughts here",
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("💭 ...", out)
        self.assertNotIn("pensive thoughts", out)

    def test_reasoning_marker_before_tool_call_label(self):
        """For a reasoned tool call the marker comes first: reasoning
        happened before the tool invocation, so it renders above the
        tool label."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="read the file"),
            Message(
                role="assistant", content="let me check the path first",
                reasoning="let me check the path first",
                tool_calls=[ToolCall(id="1", name="Read", arguments="{}")],
            ),
            Message(role="tool", content="file contents", tool_call_id="1", name="Read"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("💭 ...", out)
        self.assertIn("🤖 Read", out)
        self.assertLess(out.index("💭 ..."), out.index("🤖 Read"))
        self.assertNotIn("let me check", out)

    def test_non_string_reasoning_does_not_crash(self):
        """A malformed (non-string) reasoning value must not raise —
        display falls back to the raw content."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant", content="full text here",
                reasoning=["not", "a", "string"],
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("full text here", out)
        self.assertNotIn("💭 ...", out)

    def test_strip_reasoning(self):
        """_strip_reasoning removes the leading reasoning prefix and
        leaves non-matching text untouched."""
        from python_agent_harness.tui import _strip_reasoning

        self.assertEqual(
            _strip_reasoning("ABCanswer", "ABC"), "answer"
        )
        self.assertEqual(
            _strip_reasoning("  ABCanswer", "ABC"), "answer"
        )
        self.assertEqual(_strip_reasoning("answer", "ABC"), "answer")
        self.assertEqual(_strip_reasoning("x", ""), "x")

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

    def test_dump_conversation_full_history(self):
        """After a run, the full conversation is printed as plain lines
        (not Live frames) so it lands in the terminal scrollback —
        including rows the visible frame budget would have dropped."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content=f"message number {i}")
            for i in range(120)
        ]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("message number 0", out)    # oldest rows included
        self.assertIn("message number 119", out)  # newest rows included
        self.assertIn("full conversation", out)

    def test_dump_conversation_unlimited(self):
        """No line cap: a huge conversation is dumped in full, oldest
        and newest rows alike."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content=f"m{i}") for i in range(3000)
        ]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("m0", out)      # oldest row included
        self.assertIn("m2999", out)   # newest row included
        self.assertNotIn("omitted", out)

    def test_dump_conversation_empty_noop(self):
        """No messages → no dump, no separator line."""
        tui, buf = make_tui()
        tui.session.last_messages = []
        tui._dump_conversation()
        self.assertEqual(buf.getvalue(), "")

    def test_dump_conversation_full_long_reply(self):
        """The scrollback dump must show long assistant replies in FULL.

        The live panel tail-caps long messages to the newest lines
        (regression: the dump reused those same capped rows, so the
        head of a long summary was never readable anywhere in the TUI
        — only its tail, prefixed with a "…" marker).
        """
        tui, buf = make_tui()
        body = "\n".join(f"summary line {i}" for i in range(40))
        tui.session.last_messages = [Message(role="assistant", content=body)]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("summary line 0", out)    # head visible
        self.assertIn("summary line 39", out)   # tail visible
        self.assertNotIn("…\n", out)            # no tail-cut marker
        self.assertNotIn("more lines", out)     # no head-cut marker

    def test_dump_conversation_full_long_user_message(self):
        """Long user messages are dumped uncapped too (the live panel
        tail-caps them to 12 lines)."""
        tui, buf = make_tui()
        body = "\n".join(f"user line {i}" for i in range(30))
        tui.session.last_messages = [Message(role="user", content=body)]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("user line 0", out)
        self.assertIn("user line 29", out)
        self.assertNotIn("…\n", out)

    def test_dump_conversation_still_filters_and_strips(self):
        """The full dump keeps the same display hygiene as the panel:
        injected prompts hidden, final-check blocks stripped."""
        from python_agent_harness import config

        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="real question"),
            Message(role="user", content=config.NUDGE_MESSAGE),
            Message(
                role="assistant",
                content="Answer body.\n\n[FINAL CHECK]\n- Goal: x\n"
                        "- Status: SUCCESS\n- Evidence: y",
            ),
        ]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("real question", out)
        self.assertIn("Answer body.", out)
        self.assertNotIn(config.NUDGE_MESSAGE, out)
        self.assertNotIn("[FINAL CHECK]", out)

    def test_run_live_dumps_conversation_at_end(self):
        """When a run finishes normally, _run_live prints the full
        conversation into the scrollback after the Live frame."""
        from types import SimpleNamespace

        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="scrollback content"),
            Message(role="assistant", content="final answer"),
        ]
        tui._run_live(SimpleNamespace(is_alive=lambda: False))
        out = buf.getvalue()
        self.assertIn("scrollback content", out)
        self.assertIn("final answer", out)
        self.assertIn("full conversation", out)

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
        """When the current run completes the live stream buffer is
        dropped so the final text isn't rendered twice (stream row +
        history row)."""
        from types import SimpleNamespace

        tui, _ = make_tui()
        tui.stream_text = "final words"
        s = tui.session
        fake = SimpleNamespace(close=lambda: None)
        s.client = fake
        tui._run_agent("hi", tui.run_seq)  # current run completes (no API key -> error path)
        self.assertEqual(tui.stream_text, "")

    def test_stale_worker_does_not_clobber_next_run(self):
        """A worker from a cancelled run that finishes late must not
        clear the next run's stream or fire its restore callback."""
        from types import SimpleNamespace

        tui, _ = make_tui()
        s = tui.session
        s.client = SimpleNamespace(close=lambda: None)
        s.cancel_event.set()
        s.cancel_generation += 1
        s.run_generation += 1  # a newer run has already started
        tui.stream_text = "next run's stream"
        restored: list[int] = []
        tui.run_seq += 1  # a newer run has already started
        tui._run_agent("old", tui.run_seq - 1, restore=lambda: restored.append(1))
        self.assertEqual(tui.stream_text, "next run's stream")
        self.assertEqual(restored, [])

    def test_cancelled_current_run_adopts_history(self):
        """A cancelled run with no successor is still current: the TUI
        adopts its salvaged partial history so the interrupted turn is
        not lost (staleness is judged by seq, not the cancel event)."""
        from types import SimpleNamespace

        tui, _ = make_tui()
        s = tui.session
        s.client = SimpleNamespace(close=lambda: None)
        s.cancel_event.set()
        s.cancel_generation += 1
        # what the agent loop's finally salvaged before the worker died
        s.last_messages = [
            Message(role="user", content="q2"),
            Message(role="assistant", content="partial answer"),
        ]
        with mock.patch(
            "python_agent_harness.tui.run_agent_loop", return_value=None
        ):
            tui._run_agent("q2", tui.run_seq)
        self.assertEqual(
            [m.text() for m in tui.conversation_history],
            ["q2", "partial answer"],
        )

    def test_clear_bumps_run_generation(self):
        """/clear replaces the conversation generation: an in-flight worker
        from a cancelled run must be marked stale, or its salvaged
        history would resurrect what /clear just wiped."""
        tui, _ = make_tui()
        gen = tui.session.run_generation
        tui._handle_slash("/clear")
        self.assertEqual(tui.session.run_generation, gen + 1)
        self.assertEqual(tui.conversation_history, [])
        self.assertEqual(tui.session.last_messages, [])

    def test_restore_bumps_run_generation(self):
        """/restore replaces the conversation generation: a dying worker from
        a cancelled run must be marked stale so it can't clobber the
        restored session."""
        from python_agent_harness import config

        tui, _ = make_tui()
        with tempfile.TemporaryDirectory() as d:
            old = config.SESSION_DIR
            config.SESSION_DIR = __import__("pathlib").Path(d)
            try:
                path = tui.session.store.save(
                    "**user**: hello\n\n**assistant**: hi"
                )
                gen = tui.session.run_generation
                tui._run_restore(path)
            finally:
                config.SESSION_DIR = old
        self.assertEqual(tui.session.run_generation, gen + 1)
        self.assertEqual(
            [m.text() for m in tui.conversation_history], ["hello", "hi"]
        )
        self.assertEqual(
            [m.text() for m in tui.session.last_messages], ["hello", "hi"]
        )

    def test_restore_drops_tool_messages(self):
        """/restore of a session that used tools must not resurrect
        orphan ``tool`` messages: the saved markdown has no
        ``tool_call_id``/``name``, so they would make the next API
        request invalid."""
        from python_agent_harness import config

        tui, _ = make_tui()
        with tempfile.TemporaryDirectory() as d:
            old = config.SESSION_DIR
            config.SESSION_DIR = __import__("pathlib").Path(d)
            try:
                path = tui.session.store.save(
                    "**user**: find the files\n\n"
                    "**assistant**: [tool calls: Glob, Read]\n\n"
                    "**tool**: tests/test_agent.py\n"
                    "tests/test_tui.py\n\n"
                    "**assistant**: I found them."
                )
                tui._run_restore(path)
            finally:
                config.SESSION_DIR = old
        roles = [m.role for m in tui.session.last_messages]
        self.assertEqual(roles, ["user", "assistant", "assistant"])
        self.assertNotIn("tool", roles)

    def test_parse_saved_body_drops_tool_blocks(self):
        """``**tool**:`` blocks are dropped from restored history,
        including a trailing tool block; all remaining messages must
        serialize to an API-valid payload."""
        body = (
            "**user**: find the files\n\n"
            "**assistant**: [tool calls: Glob, Read]\n\n"
            "**tool**: tests/test_agent.py\n"
            "tests/test_tui.py\n\n"
            "**assistant**: I found them.\n\n"
            "**tool**: trailing result"
        )
        msgs = Tui._parse_saved_body(body)
        self.assertEqual(
            [(m.role, m.text()) for m in msgs],
            [
                ("user", "find the files"),
                ("assistant", "[tool calls: Glob, Read]"),
                ("assistant", "I found them."),
            ],
        )
        for m in msgs:
            self.assertNotEqual(m.to_api()["role"], "tool")

    def test_restore_idempotent(self):
        """The slash-command restore may run more than once (cancel
        path + worker finally) and must only undo its own borrow."""
        tui, _ = make_tui()
        with tempfile.TemporaryDirectory() as d:
            def fake_start(text, system=None, restore=None):
                restore()
                restore()  # double invocation must be a no-op

            with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
                tui._handle_slash(f"/init {d}")
        self.assertEqual(tui.session.project_dir, "/tmp/fakeproj")

    def test_cancel_releases_borrowed_project_dir(self):
        """Cancelling a slash-command run releases the borrowed project
        dir immediately (the stale worker's finally must not run it)."""
        tui, _ = make_tui()
        with tempfile.TemporaryDirectory() as d:
            captured: dict = {}

            def fake_start(text, system=None, restore=None):
                captured["restore"] = restore
                # simulate Ctrl-C: the main thread releases the borrow
                tui._restore = restore
                tui.session.cancel()
                if tui._restore is not None:
                    tui._restore()
                    tui._restore = None

            with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
                tui._handle_slash(f"/init {d}")
        self.assertEqual(tui.session.project_dir, "/tmp/fakeproj")
        # a second release (stale worker's finally, seq-guarded) no-ops
        captured["restore"]()
        self.assertEqual(tui.session.project_dir, "/tmp/fakeproj")

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

    def test_slash_init_hides_planexit_in_plan_mode(self):
        """/init runs with every tool except PlanExit: the tool is
        hidden for the run (sub-agents share the session registry, so
        they are covered too) and restored when the run finishes."""
        tui, _ = make_tui()
        tui.session.switch_to_plan()
        self.assertIsNotNone(tui.session.registry.get("PlanExit"))

        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["restore"] = restore
            self.assertIsNone(tui.session.registry.get("PlanExit"))
            restore()  # simulate the run finishing

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/init")
        self.assertIsNotNone(captured["restore"])
        self.assertIsNotNone(tui.session.registry.get("PlanExit"))

    def test_slash_command_keeps_planexit_for_custom(self):
        """Custom commands (/explain) may use all tools, incl. PlanExit."""
        tui, _ = make_tui()
        tui.session.switch_to_plan()
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["restore"] = restore
            self.assertIsNotNone(tui.session.registry.get("PlanExit"))

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/explain client.py")
        self.assertIsNone(captured["restore"])
        self.assertIsNotNone(tui.session.registry.get("PlanExit"))

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
        # bracket usage text must not be swallowed by rich markup
        for s in (
            "/init [project] [--extra TEXT]       create/update AGENTS.md",
            "/review [project] [commit|branch|PR] review code changes",
            "/explain [project] [target]",
            "/restore [path | title | --latest]   restore a saved session",
        ):
            self.assertIn(s, out)

    def test_completer_slash_commands(self):
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
        completions = list(
            c.get_completions(Document(text="/ini", cursor_position=4), None)
        )
        names = [x.text for x in completions]
        self.assertIn("/init", names)
        self.assertNotIn("/plan", names)
        completions = list(
            c.get_completions(Document(text="/", cursor_position=1), None)
        )
        names = [x.text for x in completions]
        for cmd in ("/plan", "/build", "/init", "/review", "/exit"):
            self.assertIn(cmd, names)

    def test_completer_tilde_paths(self):
        """~/wor + Tab must complete to ~/workspace (the user's case),
        bare ~ completes to ~/, and mid-sentence ~-tokens complete too."""
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "workspace"))
            os.mkdir(os.path.join(d, "workbench"))
            with mock.patch.dict(os.environ, {"HOME": d}):
                c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
                completions = list(
                    c.get_completions(Document(text="~/wor", cursor_position=5), None)
                )
                names = [x.text for x in completions]
                self.assertIn("kspace/", names)   # workspace
                self.assertIn("kbench/", names)   # workbench
                # bare ~ -> the trailing slash only (home dir itself)
                completions = list(
                    c.get_completions(Document(text="~", cursor_position=1), None)
                )
                self.assertEqual([x.text for x in completions], ["/"])
                # mid-sentence token completes
                completions = list(
                    c.get_completions(
                        Document(text="see ~/wor", cursor_position=9), None
                    )
                )
                self.assertIn("kspace/", [x.text for x in completions])

    def test_completer_plain_text_no_completion(self):
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
        for text in ("hello", "fix the /init bug", ""):
            completions = list(
                c.get_completions(Document(text=text, cursor_position=len(text)), None)
            )
            self.assertEqual(completions, [], f"unexpected completions for {text!r}")

    def test_completer_directories(self):
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "alpha"))
            os.mkdir(os.path.join(d, "beta"))
            open(os.path.join(d, "file.txt"), "w").close()
            open(os.path.join(d, "alpha", "inner.py"), "w").close()
            c = SlashCompleter(get_project_dir=lambda: d)
            completions = list(
                c.get_completions(Document(text="/init ", cursor_position=6), None)
            )
            names = [x.text for x in completions]
            self.assertIn("alpha/", names)   # directories get a trailing slash
            self.assertIn("beta/", names)
            self.assertIn("file.txt", names)  # files complete too (e.g. /explain)
            # partial dir prefix: only the suffix is inserted at the cursor
            completions = list(
                c.get_completions(Document(text="/init al", cursor_position=8), None)
            )
            self.assertEqual([x.text for x in completions], ["pha/"])
            # empty arg lists the project dir's own contents, not its siblings
            completions = list(
                c.get_completions(Document(text="/init ", cursor_position=6), None)
            )
            self.assertIn("alpha/", [x.text for x in completions])
            # trailing slash drills into the subdirectory
            completions = list(
                c.get_completions(Document(text="/init alpha/", cursor_position=12), None)
            )
            self.assertIn("inner.py", [x.text for x in completions])

    def test_tab_key_binding_completes(self):
        """Tab (c-i) must trigger completion end-to-end, and Shift+Tab
        must cycle backwards through the completion menu."""
        import asyncio

        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from python_agent_harness.tui import SlashCompleter, _make_prompt_session

        async def run(text: str, keys: str) -> str:
            with tempfile.TemporaryDirectory() as d, create_pipe_input() as inp:
                c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
                s = _make_prompt_session(
                    FileHistory(os.path.join(d, "hist")), c,
                    input=inp, output=DummyOutput(),
                )
                task = asyncio.ensure_future(s.prompt_async("> "))
                await asyncio.sleep(0.1)
                inp.send_text(text)
                await asyncio.sleep(0.2)
                inp.send_text(keys)
                await asyncio.sleep(0.3)
                inp.send_text("\x1b\r")
                return await asyncio.wait_for(task, 5)

        self.assertEqual(asyncio.run(run("/ini", "\t")), "/init")

    def test_tab_burst_input_completes(self):
        """Text and Tab arriving in a single input burst must still
        complete (regression: complete_while_typing's background task
        used to create the completion state first, so the Tab-triggered
        task bailed out without inserting)."""
        import asyncio

        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from python_agent_harness.tui import SlashCompleter, _make_prompt_session

        async def run(burst: str) -> str:
            with tempfile.TemporaryDirectory() as d, create_pipe_input() as inp:
                c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
                s = _make_prompt_session(
                    FileHistory(os.path.join(d, "hist")), c,
                    input=inp, output=DummyOutput(),
                )
                task = asyncio.ensure_future(s.prompt_async("> "))
                await asyncio.sleep(0.1)
                inp.send_text(burst)  # text + Tab in one chunk
                await asyncio.sleep(0.5)
                inp.send_text("\x1b\r")
                return await asyncio.wait_for(task, 5)

        self.assertEqual(asyncio.run(run("/ini\t")), "/init")

    def test_shift_tab_key_binding(self):
        """Shift+Tab (s-tab) must exist as a key binding handler."""
        from prompt_toolkit.key_binding.key_bindings import KeyBindings

        from python_agent_harness.tui import _make_key_bindings

        kb = _make_key_bindings()
        self.assertIsInstance(kb, KeyBindings)
        handlers = {b.keys: b.handler for b in kb.bindings}
        self.assertIn(("c-i",), handlers)    # Tab
        self.assertIn(("s-tab",), handlers)  # Shift+Tab


    # ------------------------------------------------------------------
    # question selection (number keys)
    # ------------------------------------------------------------------
    def test_numbered_choice_resolution(self):
        """Bare numbers map to option labels; everything else passes
        through untouched (free-text answers, out-of-range, empty)."""
        options = ["foo bar", "baz", "qux"]
        self.assertEqual(_resolve_numbered_choice("1", options), "foo bar")
        self.assertEqual(_resolve_numbered_choice("2", options), "baz")
        self.assertEqual(_resolve_numbered_choice("3", options), "qux")
        self.assertEqual(_resolve_numbered_choice("1,3", options), "foo bar, qux")
        self.assertEqual(_resolve_numbered_choice("2, custom", options), "baz, custom")
        self.assertEqual(_resolve_numbered_choice("custom", options), "custom")
        self.assertEqual(_resolve_numbered_choice("0", options), "0")
        self.assertEqual(_resolve_numbered_choice("9", options), "9")
        self.assertEqual(_resolve_numbered_choice("", options), "")
        self.assertEqual(_resolve_numbered_choice("1", []), "1")
        self.assertEqual(_resolve_numbered_choice("1", ["only"]), "only")

    def test_ask_question_short_options_numbered_too(self):
        """Single-letter options (y/n/a/d) also render as a numbered
        list — numbers apply to ALL option lists now — and a number
        resolves to the matching option."""
        tui, buf = make_tui()
        q = UiQuestion("Proceed?", options=["y", "n"])
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="2") as m:
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "n")
        m.assert_called_once_with("> ", multiline=False)
        out = buf.getvalue()
        self.assertIn("1) y", out)
        self.assertIn("2) n", out)

    def test_ask_question_custom_answer_passthrough(self):
        """Free-text answers (not numbers) are returned verbatim."""
        tui, _ = make_tui()
        q = UiQuestion("Pick one", options=["long option a", "long option b"])
        tui.question = q
        with mock.patch.object(
            tui.prompt_session, "prompt", return_value="something else"
        ):
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "something else")

    def test_ask_question_multiple_numbers(self):
        """Comma-separated numbers select several options (multiple)."""
        tui, _ = make_tui()
        q = UiQuestion(
            "Pick several", multiple=True,
            options=["first choice", "second choice", "third choice"],
        )
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="1,3"):
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "first choice, third choice")

    # ------------------------------------------------------------------
    # PlanExit confirmation (y/n keyed list, like Question but keys)
    # ------------------------------------------------------------------
    def test_keyed_choice_resolution(self):
        """Keys map to the matching option label; non-keys pass through."""
        options = ["Yes, switch to build agent", "No, keep refining the plan"]
        keys = ["y", "n"]
        self.assertEqual(_resolve_keyed_choice("y", options, keys), options[0])
        self.assertEqual(_resolve_keyed_choice("n", options, keys), options[1])
        self.assertEqual(_resolve_keyed_choice("Y", options, keys), options[0])
        self.assertEqual(_resolve_keyed_choice("y, custom", options, keys),
                         f"{options[0]}, custom")
        self.assertEqual(_resolve_keyed_choice("custom", options, keys), "custom")
        self.assertEqual(_resolve_keyed_choice("", options, keys), "")
        self.assertEqual(_resolve_keyed_choice("y", [], []), "y")

    def test_ask_question_keyed_list_renders_and_resolves(self):
        """A keyed choice renders as a list (y) label / n) label) with a
        hint line, and a typed key resolves to the option label."""
        tui, buf = make_tui()
        q = UiQuestion(
            "Approve plan?",
            options=["Yes, switch to build agent", "No, keep refining the plan"],
            keys=["y", "n"],
            custom=False,
        )
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="n"):
            tui._ask_question_blocking()
        out = buf.getvalue()
        self.assertIn("Approve plan?", out)
        self.assertIn("y) Yes, switch to build agent", out)
        self.assertIn("n) No, keep refining the plan", out)
        self.assertIn("Enter a key", out)
        self.assertEqual(q.answer, "No, keep refining the plan")

    def test_ui_confirm_accepts_y_n_and_legacy_yes(self):
        """_ui_confirm approves on y/yes, rejects on n; it renders a
        y/n keyed choice list (not a bare prompt)."""
        from python_agent_harness import config

        tui, _ = make_tui()
        for raw, expected in (("y", True), ("n", False), ("yes", True),
                              ("a", True), ("1", True), ("", False)):
            with mock.patch.object(tui, "_ask_sync", return_value=raw) as ask:
                self.assertEqual(tui._ui_confirm("Switch to build?"), expected)
            q = ask.call_args[0][0]
            self.assertEqual(q.options, list(config.PLAN_EXIT_OPTIONS))
            self.assertEqual(q.keys, ["y", "n"])
            self.assertFalse(q.custom)

    # ------------------------------------------------------------------
    # cancel-aware question wait + auto-save error surfacing
    # ------------------------------------------------------------------
    def test_ask_sync_unblocks_on_cancel(self):
        """A Ctrl-C (session cancel) while a question is pending must
        unblock the worker's wait promptly with an empty answer."""
        import threading
        import time

        tui, _ = make_tui()
        q = UiQuestion("Pick one", options=["a", "b"])
        results = {}
        worker = threading.Thread(
            target=lambda: results.update(r=tui._ask_sync(q))
        )
        worker.start()
        time.sleep(0.2)
        tui.session.cancel()  # sets cancel_event
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive(), "question wait wedged by cancel")
        self.assertEqual(results["r"], "")

    def test_ask_sync_returns_answer_when_not_cancelled(self):
        tui, _ = make_tui()
        q = UiQuestion("Pick one", options=["a", "b"])
        q.answer = "b"
        q.event.set()
        self.assertEqual(tui._ask_sync(q), "b")

    def test_status_bar_shows_save_error_marker(self):
        tui, buf = make_tui()
        tui.session._save_error = "disk full"
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertIn("[!save]", out)
        buf.seek(0)
        buf.truncate()
        tui.session._save_error = None
        tui.console.print(tui._render_frame())
        self.assertNotIn("[!save]", buf.getvalue())

    def test_compacted_summary_renders_as_user_message(self):
        """The compacted summary lives in the user turn: the frame must
        render as the dim 📦 row even with role=user (live sessions),
        not as a normal user body."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(
                role="user",
                content="**[Compacted Summary]**\n\nfixed the bug\n\n---\n\n"
                        "**[Context compacted]**\n\n---\n\n",
            ),
        ]
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertIn("📦", out)
        self.assertIn("fixed the bug", out)
        self.assertNotIn("**user:**", out)

    def test_compact_syncs_tui_conversation_history(self):
        """/compact replaces the shared conversation with the summary
        frame: the TUI's own history must follow, or the next run
        restarts from the old full conversation and immediately
        re-compacts it."""
        tui, _ = make_tui()
        tui.conversation_history = [
            Message(role="user", content="old full history"),
        ]
        compacted = [
            Message(role="user", content="**[Compacted Summary]**\n\nx"),
        ]
        with mock.patch.object(
            tui.session, "compact_conversation", return_value=(True, "ok"),
        ), mock.patch.object(
            tui.session, "last_messages", compacted, create=True,
        ):
            tui._run_compact()
        self.assertEqual(
            [m.role for m in tui.conversation_history], ["user"]
        )
        self.assertEqual(
            [m.text() for m in tui.conversation_history],
            [m.text() for m in compacted],
        )

    def test_summary_syncs_tui_conversation_history(self):
        tui, _ = make_tui()
        tui.conversation_history = [Message(role="user", content="hello")]
        summarized = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="summary appended"),
        ]
        with mock.patch.object(
            tui.session, "summarize_conversation", return_value="summary appended",
        ), mock.patch.object(
            tui.session, "last_messages", summarized, create=True,
        ):
            tui._run_summary()
        self.assertEqual(
            [m.text() for m in tui.conversation_history],
            [m.text() for m in summarized],
        )

    # ------------------------------------------------------------------
    # key-binding handlers (Tab / Shift+Tab with an open completion menu)
    # ------------------------------------------------------------------
    def _kb_handlers(self):
        from python_agent_harness.tui import _make_key_bindings

        return {b.keys: b.handler for b in _make_key_bindings().bindings}

    def test_complete_handler_with_menu_cycles_forward(self):
        """Tab while a completion menu is open cycles to the next entry."""
        buffer = mock.Mock()
        buffer.complete_state = object()  # menu open
        self._kb_handlers()[("c-i",)](mock.Mock(current_buffer=buffer))
        buffer.complete_next.assert_called_once_with()
        buffer.start_completion.assert_not_called()

    def test_complete_handler_starts_menu(self):
        """Tab with no menu open starts completion with the common part."""
        buffer = mock.Mock()
        buffer.complete_state = None
        self._kb_handlers()[("c-i",)](mock.Mock(current_buffer=buffer))
        buffer.start_completion.assert_called_once_with(insert_common_part=True)

    def test_complete_backward_handler_cycles(self):
        """Shift+Tab while a menu is open cycles to the previous entry."""
        buffer = mock.Mock()
        buffer.complete_state = object()
        self._kb_handlers()[("s-tab",)](mock.Mock(current_buffer=buffer))
        buffer.complete_previous.assert_called_once_with()
        buffer.start_completion.assert_not_called()

    def test_complete_backward_handler_starts_menu(self):
        """Shift+Tab with no menu open starts completion selecting the first."""
        buffer = mock.Mock()
        buffer.complete_state = None
        self._kb_handlers()[("s-tab",)](mock.Mock(current_buffer=buffer))
        buffer.start_completion.assert_called_once_with(select_first=True)

    # ------------------------------------------------------------------
    # completer edge cases
    # ------------------------------------------------------------------
    def test_completer_absolute_path_fallback(self):
        """A /-token matching no slash command completes as an absolute
        path (and yields nothing when nothing matches)."""
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
        completions = list(
            c.get_completions(Document(text="/zzzz-no-such", cursor_position=13), None)
        )
        self.assertEqual(completions, [])

    def test_completer_unlistable_directory_no_crash(self):
        """A path whose directory cannot be listed yields no completions
        instead of raising."""
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
        completions = list(
            c.get_completions(
                Document(text="/init /no/such/dir/", cursor_position=16), None
            )
        )
        self.assertEqual(completions, [])

    # ------------------------------------------------------------------
    # notify / log status updates
    # ------------------------------------------------------------------
    def test_on_notify_compact(self):
        tui, _ = make_tui()
        tui._on_notify("compact")
        self.assertEqual(tui.status, " compacted")
        self.assertTrue(tui._history_dirty)

    def test_on_notify_save_error(self):
        tui, _ = make_tui()
        tui._on_notify("save-error")
        self.assertEqual(tui.status, " auto-save failed")

    def test_on_notify_default_status(self):
        tui, _ = make_tui()
        tui._on_notify("some-other-kind")
        self.assertEqual(tui.status, " running")
        self.assertTrue(tui._data_event.is_set())

    def test_on_log_sets_status(self):
        tui, _ = make_tui()
        tui._on_log("checking files")
        self.assertEqual(tui.status, " checking files")

    def test_on_log_truncates_long_messages(self):
        tui, _ = make_tui()
        tui._on_log("x" * 100)
        self.assertEqual(tui.status, " " + "x" * 60)

    # ------------------------------------------------------------------
    # _ui_ask (Question tool)
    # ------------------------------------------------------------------
    def test_ui_ask_single_question(self):
        """A single Question returns one 'prompt' = 'answer' line."""
        tui, _ = make_tui()
        with mock.patch.object(tui, "_ask_sync", return_value="42"):
            result = tui._ui_ask(
                [{"question": "How many?", "options": ["one", "two"]}]
            )
        self.assertEqual(result, '"How many?" = "42"')

    def test_ui_ask_multiple_questions(self):
        tui, _ = make_tui()
        with mock.patch.object(tui, "_ask_sync", side_effect=["x", "y"]):
            result = tui._ui_ask([{"question": "Q1"}, {"question": "Q2"}])
        self.assertEqual(result, '"Q1" = "x"\n"Q2" = "y"')

    def test_ui_ask_multiple_joins_and_cleans_answers(self):
        """Multiple-select answers are joined, dropping empty parts."""
        tui, _ = make_tui()
        with mock.patch.object(tui, "_ask_sync", return_value="a, , b"):
            result = tui._ui_ask(
                [{"question": "Pick", "multiple": True, "options": ["a", "b"]}]
            )
        self.assertEqual(result, '"Pick" = "a, b"')

    def test_ui_ask_no_questions_returns_unanswered(self):
        tui, _ = make_tui()
        self.assertEqual(tui._ui_ask([]), "Unanswered")

    # ------------------------------------------------------------------
    # tool-call argument rendering edge cases
    # ------------------------------------------------------------------
    def test_tool_call_bad_json_arguments(self):
        """Unparseable tool-call JSON renders as a bare tool label."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant", content="",
                tool_calls=[ToolCall(id="1", name="Read", arguments="{oops")],
            ),
            Message(role="tool", content="x", tool_call_id="1", name="Read"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("🤖 Read", out)

    def test_tool_call_non_dict_arguments(self):
        """A JSON array of arguments renders as a bare tool label."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant", content="",
                tool_calls=[
                    ToolCall(id="1", name="Bash", arguments='["ls", "-la"]')
                ],
            ),
            Message(role="tool", content="x", tool_call_id="1", name="Bash"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("🤖 Bash", out)
        self.assertNotIn("ls", out)

    # ------------------------------------------------------------------
    # row budget: visible-row cap and line estimates
    # ------------------------------------------------------------------
    def test_visible_row_cap_defaults_without_height(self):
        """Without a terminal height the cap falls back to 60 rows."""
        from types import SimpleNamespace

        tui, _ = make_tui()
        tui.console = SimpleNamespace(height=0, width=80)
        self.assertEqual(tui._visible_row_cap(), 60)

    def test_est_lines_for_panels(self):
        """Panel rows estimate 3 lines plus their inner renderables."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        from python_agent_harness.tui import Tui

        self.assertEqual(Tui._est_lines(Panel("short"), 80), 4)
        self.assertEqual(
            Tui._est_lines(Panel(Group(Text("a"), Text("b"))), 80), 5
        )

    # ------------------------------------------------------------------
    # the main run loop
    # ------------------------------------------------------------------
    def test_run_quits_on_eof(self):
        """Ctrl-D at the prompt exits the app after showing the banner."""
        tui, buf = make_tui()
        with mock.patch.object(tui, "_read_multiline", return_value=None):
            tui.run()
        out = buf.getvalue()
        self.assertIn("python-agent-harness — agent execution harness", out)
        self.assertIn("Commands:", out)

    def test_run_handles_empty_and_slash_input(self):
        """Blank lines are skipped, non-exit slashes continue the loop,
        /exit breaks it and plain text starts a run."""
        tui, buf = make_tui()
        with mock.patch.object(
            tui, "_read_multiline", side_effect=["", "hello", "/help", "/exit"]
        ), mock.patch.object(tui, "_start_agent") as start:
            tui.run()
        start.assert_called_once_with("hello")
        self.assertIn("/sessions", buf.getvalue())  # /help rendered

    def test_run_keyboard_interrupt_stays_open(self):
        """A stray Ctrl-C outside input prints a hint and keeps looping."""
        tui, buf = make_tui()
        with mock.patch.object(
            tui, "_read_multiline", side_effect=[KeyboardInterrupt, None]
        ):
            tui.run()
        self.assertIn("cancelled", buf.getvalue())

    def test_run_services_pending_question_first(self):
        """A pending question is answered before reading new input."""
        tui, _ = make_tui()
        tui.question = UiQuestion("Approve?")
        with mock.patch.object(
            tui, "_ask_question_blocking",
            side_effect=lambda: setattr(tui, "question", None),
        ), mock.patch.object(tui, "_read_multiline", return_value=None):
            tui.run()
        self.assertIsNone(tui.question)

    def test_run_shows_llm_log_path(self):
        """With LLM logging enabled the log path is printed at startup."""
        import python_agent_harness.tui as tui_mod

        tui, buf = make_tui()
        tui.session.client.log_path = "/tmp/llm.log"
        with mock.patch.object(tui_mod.config, "LLM_LOG_ENABLED", True), \
             mock.patch.object(tui, "_read_multiline", return_value=None):
            tui.run()
        self.assertIn("/tmp/llm.log", buf.getvalue())

    # ------------------------------------------------------------------
    # question prompt variants
    # ------------------------------------------------------------------
    def test_ask_question_keyed_multiple_custom_hints(self):
        """A keyed list with multiple+custom shows both hint extensions
        and resolves comma-separated keys."""
        tui, buf = make_tui()
        q = UiQuestion(
            "Approve?", multiple=True,
            options=["Yes, switch", "No, refine"], keys=["y", "n"], custom=True,
        )
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="y,n"):
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "Yes, switch, No, refine")
        out = buf.getvalue()
        self.assertIn("Enter keys, comma-separated", out)
        self.assertIn("or type your own answer", out)

    def test_ask_question_plain_prompt(self):
        """A question without options uses a bare 'prompt > ' line."""
        tui, _ = make_tui()
        q = UiQuestion("What is your name?")
        tui.question = q
        with mock.patch.object(
            tui.prompt_session, "prompt", return_value="Ada"
        ) as m:
            tui._ask_question_blocking()
        m.assert_called_once_with("What is your name? > ", multiline=False)
        self.assertEqual(q.answer, "Ada")

    def test_ask_question_eof_returns_empty(self):
        """Ctrl-D/Ctrl-C at a question prompt answers with an empty string."""
        tui, _ = make_tui()
        q = UiQuestion("Pick", options=["a", "b"])
        tui.question = q
        with mock.patch.object(
            tui.prompt_session, "prompt", side_effect=EOFError
        ):
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "")
        self.assertIsNone(tui.question)

    # ------------------------------------------------------------------
    # _read_multiline
    # ------------------------------------------------------------------
    def test_read_multiline_returns_text(self):
        tui, _ = make_tui()
        with mock.patch.object(tui.prompt_session, "prompt", return_value="hello"):
            self.assertEqual(tui._read_multiline(), "hello")

    def test_read_multiline_eof_quits(self):
        tui, _ = make_tui()
        with mock.patch.object(
            tui.prompt_session, "prompt", side_effect=EOFError
        ):
            self.assertIsNone(tui._read_multiline())

    def test_read_multiline_interrupt_cancels_input(self):
        tui, buf = make_tui()
        with mock.patch.object(
            tui.prompt_session, "prompt", side_effect=KeyboardInterrupt
        ):
            self.assertEqual(tui._read_multiline(), "")
        self.assertIn("input cancelled", buf.getvalue())

    # ------------------------------------------------------------------
    # _start_agent / run loops
    # ------------------------------------------------------------------
    def test_start_agent_normal_completion(self):
        """A normal run starts a worker, renders live and clears the
        running flag when done."""
        import threading

        tui, _ = make_tui()
        gen = tui.session.run_generation
        done = threading.Event()

        def boom(*a, **k):
            done.set()
            raise RuntimeError("stop")

        with mock.patch("python_agent_harness.tui.run_agent_loop",
                        side_effect=boom), \
             mock.patch.object(tui, "_run_live", return_value=False) as live:
            tui._start_agent("hello")
        self.assertTrue(done.wait(2.0), "worker thread never ran")
        self.assertEqual(tui.session.run_generation, gen + 1)
        self.assertEqual(tui.run_seq, 1)
        self.assertFalse(tui.agent_running)
        live.assert_called_once()

    def test_start_agent_dumb_terminal(self):
        """Dumb terminals use the line-printing fallback display."""
        from types import SimpleNamespace

        tui, _ = make_tui()
        tui.console = SimpleNamespace(
            is_dumb_terminal=True, print=lambda *a, **k: None,
            file=io.StringIO(),
        )
        with mock.patch("python_agent_harness.tui.run_agent_loop",
                        side_effect=RuntimeError("stop")), \
             mock.patch.object(tui, "_run_dumb", return_value=False) as dumb:
            tui._start_agent("hello")
        dumb.assert_called_once()
        self.assertFalse(tui.agent_running)

    def test_start_agent_keyboard_interrupt(self):
        """Ctrl-C during execution cancels the run and releases pending
        questions and borrowed state (the worker's own finally may also
        fire the idempotent restore later)."""
        tui, buf = make_tui()
        released = []
        q = UiQuestion("Approve?")
        tui.question = q
        with mock.patch("python_agent_harness.tui.run_agent_loop",
                        side_effect=RuntimeError("stop")), \
             mock.patch.object(tui, "_run_live", side_effect=KeyboardInterrupt):
            tui._start_agent("hello", restore=lambda: released.append(1))
        self.assertIn("execution cancelled", buf.getvalue())
        self.assertIsNone(tui.question)
        self.assertTrue(q.event.is_set())
        self.assertIsNone(tui._restore)
        self.assertGreaterEqual(len(released), 1)  # released synchronously
        self.assertTrue(tui.session.cancel_event.is_set())
        self.assertFalse(tui.agent_running)

    def test_run_live_services_question_while_running(self):
        """A pending question pauses the Live display, is answered, and
        rendering resumes."""
        from types import SimpleNamespace

        tui, _ = make_tui()
        tui.question = UiQuestion("Approve?")
        tui._data_event.set()  # render loop wakes without sleeping
        worker = SimpleNamespace(is_alive=iter([True, True, False]).__next__)
        answered = []

        def ask():
            answered.append(1)
            tui.question = None  # the question is now answered

        with mock.patch.object(tui, "_ask_question_blocking", side_effect=ask), \
             mock.patch.object(tui, "_dump_conversation"):
            tui._run_live(worker)
        self.assertEqual(answered, [1])

    def test_run_dumb_services_question_and_prints_frames(self):
        """The dumb-terminal loop prints frames as lines and answers
        pending questions."""
        from types import SimpleNamespace

        tui, buf = make_tui()
        tui.question = UiQuestion("Approve?")
        tui._data_event.set()
        worker = SimpleNamespace(is_alive=iter([True, True, False]).__next__)
        answered = []

        def ask():
            answered.append(1)
            tui.question = None

        with mock.patch.object(tui, "_ask_question_blocking", side_effect=ask), \
             mock.patch.object(tui, "_dump_conversation"):
            result = tui._run_dumb(worker)
        self.assertFalse(result)
        self.assertEqual(answered, [1])
        self.assertIn("[BUILD]", buf.getvalue())

    def test_flush_tolerates_flush_errors(self):
        """A failing stdout flush must not crash the render loop."""
        tui, _ = make_tui()
        with mock.patch.object(
            tui.console.file, "flush", side_effect=OSError("boom")
        ):
            tui._flush()  # must not raise

    def test_run_agent_error_logged(self):
        """An agent-loop exception on the current run is surfaced in the
        status bar."""
        tui, _ = make_tui()
        with mock.patch("python_agent_harness.tui.run_agent_loop",
                        side_effect=RuntimeError("boom")):
            tui._run_agent("hi", tui.run_seq)
        self.assertIn("agent error: boom", tui.status)

    def test_run_agent_calls_restore(self):
        """The current run's finally fires the restore callback."""
        tui, _ = make_tui()
        restored = []
        with mock.patch("python_agent_harness.tui.run_agent_loop",
                        side_effect=RuntimeError("boom")):
            tui._run_agent("hi", tui.run_seq, restore=lambda: restored.append(1))
        self.assertEqual(restored, [1])

    # ------------------------------------------------------------------
    # remaining slash commands
    # ------------------------------------------------------------------
    def test_exit_slash(self):
        tui, _ = make_tui()
        self.assertTrue(tui._handle_slash("/exit"))

    def test_plan_and_build_slashes(self):
        tui, buf = make_tui()
        self.assertFalse(tui._handle_slash("/plan"))
        self.assertEqual(tui.session.plan_mode.mode.value, "plan")
        self.assertIn("Plan mode", buf.getvalue())
        self.assertFalse(tui._handle_slash("/build"))
        self.assertEqual(tui.session.plan_mode.mode.value, "build")
        self.assertIn("Build mode", buf.getvalue())

    def test_save_slash(self):
        tui, buf = make_tui()
        with mock.patch.object(
            tui.session.store, "save", return_value="/tmp/x.md"
        ):
            self.assertFalse(tui._handle_slash("/save"))
        self.assertIn("saved: /tmp/x.md", buf.getvalue())

    def test_compact_and_summary_slashes_dispatch(self):
        tui, _ = make_tui()
        with mock.patch.object(tui, "_run_compact") as c, \
             mock.patch.object(tui, "_run_summary") as s:
            tui._handle_slash("/compact")
            tui._handle_slash("/summary")
        c.assert_called_once_with()
        s.assert_called_once_with()

    def test_sessions_and_restore_slashes_dispatch(self):
        tui, _ = make_tui()
        with mock.patch.object(tui, "_run_sessions") as s, \
             mock.patch.object(tui, "_run_restore") as r:
            tui._handle_slash("/sessions")
            tui._handle_slash("/restore foo.md")
        s.assert_called_once_with()
        r.assert_called_once_with("foo.md")

    def test_split_args_unbalanced_quote_falls_back(self):
        """An unterminated quote falls back to whitespace splitting."""
        tui, _ = make_tui()
        self.assertEqual(
            tui._split_args('unterminated "quote'), ['unterminated', '"quote']
        )

    def test_command_args_init_invalid_returns_none(self):
        """/init with a non-project token after the project is invalid."""
        tui, _ = make_tui()
        self.assertEqual(tui._command_args("init", "proj --extra"), ("proj", None))
        self.assertEqual(tui._command_args("init", "a b"), (None, None))

    def test_run_slash_command_unknown(self):
        """A slash command with no registered SessionCommand is reported."""
        tui, buf = make_tui()
        with mock.patch("python_agent_harness.tui.find_command", return_value=None):
            tui._run_slash_command("bogus", "")
        self.assertIn("unknown command: /bogus", buf.getvalue())

    def test_planexit_restore_idempotent_with_prev_restore(self):
        """The planexit-restore wrapper undoes the project borrow first
        and ignores repeat invocations."""
        tui, _ = make_tui()
        with tempfile.TemporaryDirectory() as d:
            tui.session.switch_to_plan()
            seen = []

            def fake_start(text, system=None, restore=None):
                restore()
                restore()  # second call must be a no-op
                seen.append(tui.session.project_dir)

            with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
                tui._handle_slash(f"/init {d}")
        self.assertEqual(tui.session.project_dir, "/tmp/fakeproj")
        self.assertEqual(seen, ["/tmp/fakeproj"])
        self.assertIsNotNone(tui.session.registry.get("PlanExit"))

    def test_conversation_text(self):
        """_conversation_text renders non-empty messages, skipping blanks."""
        tui, _ = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content=""),
            Message(role="tool", content="result"),
        ]
        self.assertEqual(
            tui._conversation_text(),
            "**user**: hello\n\n**tool**: result",
        )

    def test_conversation_text_empty(self):
        tui, _ = make_tui()
        tui.session.last_messages = []
        self.assertEqual(tui._conversation_text(), "")

    def test_run_sessions_empty(self):
        tui, buf = make_tui()
        with mock.patch(
            "python_agent_harness.tui.SessionStore.list_sessions", return_value=[]
        ):
            tui._run_sessions()
        self.assertIn("no saved sessions", buf.getvalue())

    def test_run_sessions_lists_metadata(self):
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "my session_250101120000.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hello\n\n;; Local Variables:\n"
                    ";; gptel-model: gpt-4\n"
                    ";; python-agent-harness--project-dir: /tmp/p\n"
                    ";; End:\n"
                )
            with mock.patch(
                "python_agent_harness.tui.SessionStore.list_sessions",
                return_value=[path],
            ):
                tui._run_sessions()
        out = buf.getvalue()
        self.assertIn("my session_250101120000.md", out)
        self.assertIn("gpt-4", out)
        self.assertIn("/tmp/p", out)

    def test_run_sessions_skips_unreadable_files(self):
        tui, buf = make_tui()
        with mock.patch(
            "python_agent_harness.tui.SessionStore.list_sessions",
            return_value=["/nonexistent/session.md"],
        ):
            tui._run_sessions()  # must not raise
        self.assertEqual(buf.getvalue(), "")

    # ------------------------------------------------------------------
    # /restore paths
    # ------------------------------------------------------------------
    def test_restore_no_session_found(self):
        """/restore with nothing to restore prints the yellow hint."""
        tui, buf = make_tui()
        with mock.patch(
            "python_agent_harness.tui.SessionStore.latest_session",
            return_value=None,
        ):
            tui._run_restore("")
        self.assertIn("no session found", buf.getvalue())

    def test_restore_latest_session(self):
        """/restore --latest loads the most recent session file."""
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("**user**: hello\n\n**assistant**: hi")
            with mock.patch(
                "python_agent_harness.tui.SessionStore.latest_session",
                return_value=path,
            ):
                tui._run_restore("--latest")
        out = buf.getvalue()
        self.assertIn("restored:", out)
        self.assertIn("session.md", out)
        self.assertEqual(
            [m.text() for m in tui.session.last_messages], ["hello", "hi"]
        )

    def test_restore_resolved_path_not_a_file(self):
        """A resolved path that is not a file reports an error."""
        tui, buf = make_tui()
        with mock.patch(
            "python_agent_harness.tui.SessionStore.latest_session",
            return_value="/nonexistent/session.md",
        ):
            tui._run_restore("--latest")
        self.assertIn("file not found", buf.getvalue())

    def test_restore_unreadable_file(self):
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("**user**: hello")
            with mock.patch("builtins.open", side_effect=OSError("denied")):
                tui._run_restore(path)
        self.assertIn("cannot read", buf.getvalue())

    def test_restore_by_title_match(self):
        """A non-path /restore arg matches session filenames/titles, and a
        title-bearing filename sets the store title."""
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "my session_250101120000.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("**user**: hello\n\n**assistant**: hi")
            with mock.patch(
                "python_agent_harness.tui.SessionStore.list_sessions",
                return_value=[path],
            ):
                tui._run_restore("MY SESSION")
        self.assertIn("restored:", buf.getvalue())
        self.assertEqual(tui.session.store.title, "my session")

    def test_find_session_by_title(self):
        """Title lookup: exact basename, .md-less, substring and
        derived-title matches; unmatched queries return None."""
        with tempfile.TemporaryDirectory() as d:
            dash = os.path.join(d, "fix-bugs_250101000000.md")
            spaced = os.path.join(d, "Add feature_250101000001.md")
            for f in (dash, spaced):
                open(f, "w", encoding="utf-8").close()
            files = [dash, spaced]
            with mock.patch(
                "python_agent_harness.tui.SessionStore.list_sessions",
                return_value=files,
            ):
                # exact basename match (with and without .md)
                self.assertEqual(
                    Tui._find_session_by_title("Add feature_250101000001.md"),
                    spaced,
                )
                self.assertEqual(
                    Tui._find_session_by_title("add feature_250101000001"), spaced
                )
                # filename substring match
                self.assertEqual(Tui._find_session_by_title("fix-bugs"), dash)
                self.assertEqual(Tui._find_session_by_title("feature"), spaced)
                # derived-title match (dashes -> spaces)
                self.assertEqual(Tui._find_session_by_title("fix bugs"), dash)
                self.assertIsNone(Tui._find_session_by_title("nothing here"))


if __name__ == "__main__":
    unittest.main()
