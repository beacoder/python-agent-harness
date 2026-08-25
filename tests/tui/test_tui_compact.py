"""TUI /compact and /summary command tests."""

import os
import sys
import threading
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from tui_test_utils import make_tui

from python_agent_harness.models import Message


class TestTuiCompact(unittest.TestCase):
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
        with (
            mock.patch.object(
                tui.session,
                "compact_conversation",
                return_value=(True, "ok"),
            ),
            mock.patch.object(
                tui.session,
                "last_messages",
                compacted,
                create=True,
            ),
        ):
            tui._run_compact()
        self.assertEqual([m.role for m in tui.conversation_history], ["user"])
        self.assertEqual(
            [m.text() for m in tui.conversation_history],
            [m.text() for m in compacted],
        )

    def test_compact_resets_running_state(self):
        tui, _ = make_tui()
        compacted = [
            Message(role="user", content="**[Compacted Summary]**\n\nx"),
        ]
        with (
            mock.patch.object(
                tui.session,
                "compact_conversation",
                return_value=(True, "ok"),
            ),
            mock.patch.object(
                tui.session,
                "last_messages",
                compacted,
                create=True,
            ),
        ):
            tui._run_compact()
        self.assertFalse(tui.agent_running)
        self.assertEqual(tui.status, "")

    def test_compact_failure_prints_error_and_resets_state(self):
        tui, buf = make_tui()
        with mock.patch.object(
            tui.session,
            "compact_conversation",
            side_effect=RuntimeError("boom"),
        ):
            tui._run_compact()
        self.assertIn("Compaction failed: boom", buf.getvalue())
        self.assertFalse(tui.agent_running)
        self.assertEqual(tui.status, "")

    def test_compact_renders_status_bar_while_running(self):
        tui, buf = make_tui()
        started = threading.Event()
        release = threading.Event()

        def slow_compact():
            started.set()
            release.wait(2)
            return (True, "ok")

        compacted = [
            Message(role="user", content="**[Compacted Summary]**\n\nx"),
        ]
        with (
            mock.patch.object(
                tui.session,
                "compact_conversation",
                side_effect=slow_compact,
            ),
            mock.patch.object(
                tui.session,
                "last_messages",
                compacted,
                create=True,
            ),
        ):
            run_thread = threading.Thread(target=tui._run_compact, daemon=True)
            run_thread.start()
            self.assertTrue(started.wait(2))
            release.set()
            run_thread.join(2)
        self.assertIn("compacting", buf.getvalue())

    def test_summary_syncs_tui_conversation_history(self):
        tui, _ = make_tui()
        tui.conversation_history = [Message(role="user", content="hello")]
        summarized = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="summary appended"),
        ]
        with (
            mock.patch.object(
                tui.session,
                "summarize_conversation",
                return_value="summary appended",
            ),
            mock.patch.object(
                tui.session,
                "last_messages",
                summarized,
                create=True,
            ),
        ):
            tui._run_summary()
        self.assertEqual(
            [m.text() for m in tui.conversation_history],
            [m.text() for m in summarized],
        )

    def test_summary_prints_actual_summary_content(self):
        tui, buf = make_tui()
        summarized = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="DONE SUMMARY TEXT"),
        ]
        with (
            mock.patch.object(
                tui.session,
                "summarize_conversation",
                return_value="Summary appended.",
            ),
            mock.patch.object(
                tui.session,
                "last_messages",
                summarized,
                create=True,
            ),
        ):
            tui._run_summary()
        self.assertIn("DONE SUMMARY TEXT", buf.getvalue())

    def test_summary_resets_running_state(self):
        tui, buf = make_tui()
        with (
            mock.patch.object(
                tui.session,
                "summarize_conversation",
                return_value="Summary appended.",
            ),
            mock.patch.object(
                tui.session,
                "last_messages",
                [
                    Message(role="user", content="hello"),
                    Message(role="assistant", content="S"),
                ],
                create=True,
            ),
        ):
            tui._run_summary()
        self.assertFalse(tui.agent_running)
        self.assertEqual(tui.status, "")

    def test_summary_failure_prints_error_and_resets_state(self):
        tui, buf = make_tui()
        with mock.patch.object(
            tui.session,
            "summarize_conversation",
            side_effect=RuntimeError("boom"),
        ):
            tui._run_summary()
        self.assertIn("Summary failed: boom", buf.getvalue())
        self.assertFalse(tui.agent_running)
        self.assertEqual(tui.status, "")

    def test_summary_renders_status_bar_while_running(self):
        tui, buf = make_tui()
        started = threading.Event()
        release = threading.Event()

        def slow_summarize():
            started.set()
            release.wait(2)
            return "Summary appended."

        with (
            mock.patch.object(
                tui.session,
                "summarize_conversation",
                side_effect=slow_summarize,
            ),
            mock.patch.object(
                tui.session,
                "last_messages",
                [
                    Message(role="user", content="hello"),
                    Message(role="assistant", content="S"),
                ],
                create=True,
            ),
        ):
            run_thread = threading.Thread(target=tui._run_summary, daemon=True)
            run_thread.start()
            self.assertTrue(started.wait(2))
            release.set()
            run_thread.join(2)
        self.assertIn("summarizing", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
