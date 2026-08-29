"""TUI rendering regression tests (panel, status bar, rows)."""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from rich.console import Console
from rich.live import Live
from tui_test_utils import make_tui

from python_agent_harness.models import Message, ToolCall


class TestTuiRender(unittest.TestCase):
    def test_render_conversation(self):
        tui, buf = make_tui()
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertIn("hello agent", out)
        self.assertIn("file contents", out)
        self.assertIn("Todos", out)

    def test_tool_result_marker_with_elapsed(self):
        """Tool results render as a plain dim text row whose first line
        carries the ✓/✗ marker (and the elapsed time when recorded)."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="1", name="Bash", arguments="{}", elapsed=1.26)],
            ),
            Message(role="tool", content="ok\n", tool_call_id="1", name="Bash"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("✓ bash result (1.3s):", out)  # title with marker + elapsed
        self.assertIn("ok", out)
        # failure results get a red ✗ marker
        buf2 = io.StringIO()
        tui2, _ = make_tui()
        tui2.console = Console(file=buf2, width=100, force_terminal=False)
        tui2.session.last_messages = [
            Message(role="user", content="run it"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="1", name="Bash", arguments="{}")],
            ),
            Message(role="tool", content="Error: boom", tool_call_id="1", name="Bash"),
        ]
        tui2.console.print(tui2._render_conversation())
        out2 = buf2.getvalue()
        self.assertIn("✗ bash result:", out2)

    def test_frame_includes_status_bar(self):
        """The full frame (used by Live and idle loop) must show the status bar."""
        tui, buf = make_tui()
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertIn("hello agent", out)
        self.assertIn("[BUILD]", out)
        self.assertIn("Ctx:", out)
        self.assertIn("55%", out)  # context percentage
        self.assertIn("▓", out)  # context mini progress bar
        self.assertIn("░", out)

    def test_status_bar_pinned_on_top(self):
        """The status bar must come BEFORE the conversation panel so a
        tall panel can't push it off the bottom of the terminal."""
        tui, buf = make_tui()
        tui.console.print(tui._render_frame())
        out = buf.getvalue()
        self.assertLess(out.index("[BUILD]"), out.index("hello agent"))

    def test_status_bar_fits_terminal_width(self):
        """The status bar stays on one line: the status text is
        truncated with an ellipsis only when the terminal is too
        narrow to show it, and never wraps."""
        tui, buf = make_tui()
        long_msg = "agent error: " + "y" * 120  # 131 cells: fits a wide terminal
        tui._on_log(long_msg)
        # wide terminal: message shown in full (no fixed 60-char cap)
        tui.console.width = 200
        tui.console.file = io.StringIO()
        tui.console.print(tui._status_bar())
        self.assertIn(long_msg, tui.console.file.getvalue())
        # narrow terminal: one line, ellipsized
        tui.console.width = 40
        tui.console.file = io.StringIO()
        tui.console.print(tui._status_bar())
        lines = tui.console.file.getvalue().rstrip("\n").split("\n")
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].endswith("…"))

    def test_role_labels(self):
        """Roles render as user / assistant / tool, not You / Agent."""
        tui, buf = make_tui()
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("user: hello agent", out)
        self.assertIn("assistant: hi", out)
        self.assertIn("read result:", out)
        self.assertIn("file contents", out)
        self.assertNotIn("read result: file contents", out)  # body on its own line
        self.assertNotIn("You:", out)
        self.assertNotIn("Agent:", out)

    def test_message_colors_distinct(self):
        """User and assistant bodies get distinct colors that do NOT
        collide with the tool colors (tool calls are magenta, tool
        results dim): cyan for user, green for assistant."""
        from python_agent_harness.tui import ASSISTANT_STYLE, USER_STYLE

        tui, _ = make_tui()
        rows = tui._build_history_rows()
        styles = [getattr(r, "style", None) for r in rows]
        self.assertIn(USER_STYLE, styles)  # user body
        self.assertIn(ASSISTANT_STYLE, styles)  # assistant body
        self.assertIn("magenta", styles)  # tool call label
        self.assertIn("dim", styles)  # tool result
        self.assertNotEqual(USER_STYLE, ASSISTANT_STYLE)
        # user must not be confused with tool activity (magenta/dim) nor
        # with the panel border (plain blue)
        self.assertNotIn(USER_STYLE, ("magenta", "dim", "blue"))
        self.assertNotIn(ASSISTANT_STYLE, ("magenta", "dim", "blue"))

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
                content=("<system-reminder>\n# Plan Mode - System Reminder\n\nplan.md body"),
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
            "[FINAL CHECK]\n- Goal: build the thing\n- Status: SUCCESS\n- Evidence: tests pass",
        )

    def test_final_check_markdown_variants_hidden(self):
        """Markdown-decorated check blocks are hidden too.

        Models reformat the block: bold header, heading header, labels
        with the colon outside the emphasis ("**Goal**:"), and a reply
        that starts with blank lines (what's left after the leading
        reasoning is stripped).  None of these carry a literal
        "[FINAL CHECK] ... Goal:" sequence, and all of them leaked into
        the panel before the pattern was made decoration-tolerant.
        """
        from python_agent_harness.tui import _strip_final_check

        variants = [
            "\n\n**[FINAL CHECK]**\n\n- **Goal**: g\n- **Status**: SUCCESS\n- **Evidence**: e",
            "[FINAL CHECK]\n- **Goal:** g\n- **Status:** SUCCESS\n- **Evidence:** e",
            "## Final Check\n\n- Goal: g\n- Status: SUCCESS\n- Evidence: e\n",
            "answer.\n\n[Final check]\n* `Goal`: g\n* `Status`: SUCCESS\n* `Evidence`: e",
        ]
        for text in variants:
            with self.subTest(text=text):
                self.assertNotIn("Goal", _strip_final_check(text))
                # no dangling markdown decoration left behind
                self.assertNotIn("*", _strip_final_check(text))
        # a reply's real content survives the strip
        self.assertEqual(_strip_final_check(variants[-1]), "answer.")

    def test_final_check_prose_mention_kept(self):
        """ "...the final check..." in prose is not a header: only a
        bracketed or line-starting header truncates a reply, so talking
        about the check block doesn't delete the answer."""
        from python_agent_harness.tui import _strip_final_check

        text = (
            "I ran the final check pass. The Goal: field, the Status: "
            "field and the Evidence: field are all filled in."
        )
        self.assertEqual(_strip_final_check(text), text)

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
        self.assertIn("reasoning ...", out)
        self.assertIn("Rayleigh scattering.", out)
        self.assertNotIn("Let me think", out)
        # the stored message keeps its reasoning — only the display hides it
        self.assertEqual(tui.session.last_messages[1].reasoning, reasoning)
        self.assertIn("Let me think", tui.session.last_messages[1].text())

    def test_reasoning_collapsed_marker_shows_even_without_answer(self):
        """A reasoning-only assistant message (no answer content) still
        shows the collapse marker instead of vanishing entirely."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="pensive thoughts here",
                reasoning="pensive thoughts here",
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("reasoning ...", out)
        self.assertNotIn("pensive thoughts", out)

    def test_reasoning_marker_before_tool_call_label(self):
        """For a reasoned tool call the marker comes first: reasoning
        happened before the tool invocation, so it renders above the
        tool label."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="read the file"),
            Message(
                role="assistant",
                content="let me check the path first",
                reasoning="let me check the path first",
                tool_calls=[ToolCall(id="1", name="Read", arguments="{}")],
            ),
            Message(role="tool", content="file contents", tool_call_id="1", name="Read"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("reasoning ...", out)
        self.assertIn("tool: Read", out)
        self.assertLess(out.index("reasoning ..."), out.index("tool: Read"))
        self.assertNotIn("let me check", out)

    def test_non_string_reasoning_does_not_crash(self):
        """A malformed (non-string) reasoning value must not raise —
        display falls back to the raw content."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="full text here",
                reasoning=["not", "a", "string"],
            ),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("full text here", out)
        self.assertNotIn("reasoning ...", out)

    def test_strip_reasoning(self):
        """_strip_reasoning removes the leading reasoning prefix and
        leaves non-matching text untouched."""
        from python_agent_harness.tui import _strip_reasoning

        self.assertEqual(_strip_reasoning("ABCanswer", "ABC"), "answer")
        self.assertEqual(_strip_reasoning("  ABCanswer", "ABC"), "answer")
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
        tui.session.last_messages = [Message(role="user", content=f"m{i}") for i in range(100)]
        tui.console = SimpleNamespace(height=24, width=100)  # cap_rows = 18
        panel = tui._render_conversation()
        out_console, buf = make_tui()
        out_console.console.print(panel)
        out = buf.getvalue()
        self.assertIn("m99", out)  # newest row visible
        self.assertNotIn("m0", out)  # oldest dropped
        self.assertNotIn("m50", out)  # middle rows dropped too

    def test_stream_tail_visible_when_huge(self):
        """A huge stream must keep its TAIL (the progress) on screen."""
        tui, buf = make_tui()
        tui.agent_running = True
        tui.stream_text = "".join(f"line {i}\n" for i in range(2000))
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("line 1999", out)  # newest progress visible
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
        self.assertTrue(all(x is y for x, y in zip(rows_a, rows_b, strict=True)))
        tui._history_dirty = True
        rows_c = tui._history_rows()
        self.assertFalse(all(x is y for x, y in zip(rows_c, rows_b, strict=True)))

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
        diff_text = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n line1\n-old line\n+new line\n"
        tui.session.last_messages = [
            Message(role="user", content="edit the file"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="e1", name="Edit", arguments="{}", diff=diff_text)],
            ),
            Message(
                role="tool",
                content="Successfully replaced text in f.py",
                tool_call_id="e1",
                name="Edit",
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
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="1", name="Read", arguments="{}")],
            ),
            Message(role="tool", content=big, tool_call_id="1", name="Read"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("output line 0", out)  # head shown
        self.assertIn("output line 4", out)  # within the 5-line cap
        self.assertNotIn("output line 5\n", out)
        self.assertNotIn("output line 199", out)
        self.assertIn("more lines", out)  # truncation marker

    def test_tool_result_short_untouched(self):
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="run it"),
            Message(
                role="assistant",
                content="",
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
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="1", name="Bash", arguments="{}")],
            ),
            Message(role="tool", content=huge, tool_call_id="1", name="Bash"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertLess(len(out), 5000)
        self.assertIn("…", out)

    def test_tool_call_bad_json_arguments(self):
        """Unparseable tool-call JSON renders as a bare tool label."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="1", name="Read", arguments="{oops")],
            ),
            Message(role="tool", content="x", tool_call_id="1", name="Read"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("tool: Read", out)

    def test_tool_call_non_dict_arguments(self):
        """A JSON array of arguments renders as a bare tool label."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="1", name="Bash", arguments='["ls", "-la"]')],
            ),
            Message(role="tool", content="x", tool_call_id="1", name="Bash"),
        ]
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("tool: Bash", out)
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
        self.assertEqual(Tui._est_lines(Panel(Group(Text("a"), Text("b"))), 80), 5)

    # ------------------------------------------------------------------
    # Todos panel
    # ------------------------------------------------------------------
    def test_todos_panel_visible_after_midrun_update(self):
        """TodoWrite mid-run: session.todos is set while the history cache
        is already built; the pinned Todos panel must still appear
        without waiting for the run to finish."""
        tui, buf = make_tui()
        tui.session.todos = []  # no todos at run start
        tui._history_rows()  # build cache without todos
        self.assertIsNone(tui._todos_panel())
        # TodoWrite runs mid-run:
        tui.session.update_todos(
            [
                {"content": "task one", "status": "in_progress"},
                {"content": "task two", "status": "pending"},
            ]
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
        tui.session.last_messages = [Message(role="user", content=f"m{i}") for i in range(200)]
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
    # status bar markers / compacted summary rendering
    # ------------------------------------------------------------------
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


if __name__ == "__main__":
    unittest.main()
