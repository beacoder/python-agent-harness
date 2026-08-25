"""TUI run-loop and lifecycle tests (_start_agent, _run_live/_run_dumb,
worker staleness, run generations, notify/log status updates)."""

import io
import os
import sys
import tempfile
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from tui_test_utils import make_tui

from python_agent_harness.models import Message
from python_agent_harness.tui import UiQuestion


class TestTuiRun(unittest.TestCase):
    # ------------------------------------------------------------------
    # round boundaries in the live panel
    # ------------------------------------------------------------------
    def test_live_panel_shows_only_current_round(self):
        """The live panel renders only the LATEST round (messages from
        round_start on), so a second request does not replay the first
        round's interactions."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="first round question"),
            Message(role="assistant", content="first round answer"),
            Message(role="user", content="second round question"),
            Message(role="assistant", content="second round answer"),
        ]
        # the second round begins at index 2
        tui.round_start = 2
        tui._history_dirty = True
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("second round question", out)
        self.assertIn("second round answer", out)
        self.assertNotIn("first round question", out)
        self.assertNotIn("first round answer", out)

    def test_live_panel_shows_pending_user_text_before_mirror(self):
        """Before the round's user message is mirrored into
        last_messages (the initial assistant stream), the panel shows
        the text the user just submitted so the round isn't blank."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="first round question"),
            Message(role="assistant", content="first round answer"),
        ]
        # a new round just started: boundary is past the mirrored
        # history, and the user text is not mirrored yet
        tui.round_start = 2
        tui.round_user_text = "brand new question"
        tui._history_dirty = True
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertIn("brand new question", out)
        self.assertNotIn("first round question", out)

    def test_round_start_clamped_when_history_shrinks(self):
        """A stale round_start past the end of last_messages (e.g. after
        compaction / clear) must not raise — it clamps to the list."""
        tui, buf = make_tui()
        tui.session.last_messages = [Message(role="user", content="only message")]
        tui.round_start = 99
        tui.round_user_text = ""
        tui._history_dirty = True
        # must render without error; empty slice → just the panel frame
        tui.console.print(tui._render_conversation())
        out = buf.getvalue()
        self.assertNotIn("only message", out)

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

    # ------------------------------------------------------------------
    # notify / log status updates
    # ------------------------------------------------------------------
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

    def test_on_log_keeps_long_messages(self):
        """Log messages are kept in full in the status slot; width
        truncation (with ellipsis) happens only at render time."""
        tui, _ = make_tui()
        tui._on_log("x" * 100)
        self.assertEqual(tui.status, " " + "x" * 100)

    # ------------------------------------------------------------------
    # stream / worker lifecycle
    # ------------------------------------------------------------------
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
        with mock.patch("python_agent_harness.tui.core.run_agent_loop", return_value=None):
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
                path = tui.session.store.save("**user**: hello\n\n**assistant**: hi")
                gen = tui.session.run_generation
                tui._run_restore(path)
            finally:
                config.SESSION_DIR = old
        self.assertEqual(tui.session.run_generation, gen + 1)
        self.assertEqual([m.text() for m in tui.conversation_history], ["hello", "hi"])
        self.assertEqual([m.text() for m in tui.session.last_messages], ["hello", "hi"])

    def test_restore_persists_round_timestamps(self):
        """/restore reads the persisted round start times back into
        ``_round_times``, so dump separators keep their timestamps."""
        from python_agent_harness import config

        tui, _ = make_tui()
        tui.session.store.round_times = [1700000000.0, 1700000100.5]
        with tempfile.TemporaryDirectory() as d:
            old = config.SESSION_DIR
            config.SESSION_DIR = __import__("pathlib").Path(d)
            try:
                path = tui.session.store.save(
                    "**user**: q1\n\n**assistant**: a1\n\n**user**: q2\n\n**assistant**: a2"
                )
                tui._round_times = []  # simulate a fresh TUI
                tui._run_restore(path)
            finally:
                config.SESSION_DIR = old
        self.assertEqual(tui._round_times, [1700000000.0, 1700000100.5])
        import time as _time

        expected = _time.strftime("%H:%M:%S", _time.localtime(1700000100.5))
        self.assertEqual(tui._round_time(2), expected)
        # round times keep flowing into the restored store for resaves
        self.assertEqual(tui.session.store.round_times, [1700000000.0, 1700000100.5])

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

    # ------------------------------------------------------------------
    # the main run loop
    # ------------------------------------------------------------------
    def test_run_quits_on_eof(self):
        """Ctrl-D at the prompt exits the app after showing the banner."""
        tui, buf = make_tui()
        with mock.patch.object(tui, "_read_multiline", return_value=None):
            tui.run()
        out = buf.getvalue()
        self.assertIn("python-agent-harness — interactive AI coding agent", out)
        self.assertIn("Commands:", out)

    def test_run_handles_empty_and_slash_input(self):
        """Blank lines are skipped, non-exit slashes continue the loop,
        /exit breaks it and plain text starts a run."""
        tui, buf = make_tui()
        with (
            mock.patch.object(tui, "_read_multiline", side_effect=["", "hello", "/help", "/exit"]),
            mock.patch.object(tui, "_start_agent") as start,
        ):
            tui.run()
        start.assert_called_once_with("hello")
        self.assertIn("/sessions", buf.getvalue())  # /help rendered

    def test_run_keyboard_interrupt_stays_open(self):
        """A stray Ctrl-C outside input prints a hint and keeps looping."""
        tui, buf = make_tui()
        with mock.patch.object(tui, "_read_multiline", side_effect=[KeyboardInterrupt, None]):
            tui.run()
        self.assertIn("cancelled", buf.getvalue())

    def test_run_services_pending_question_first(self):
        """A pending question is answered before reading new input."""
        tui, _ = make_tui()
        tui.question = UiQuestion("Approve?")
        with (
            mock.patch.object(
                tui,
                "_ask_question_blocking",
                side_effect=lambda: setattr(tui, "question", None),
            ),
            mock.patch.object(tui, "_read_multiline", return_value=None),
        ):
            tui.run()
        self.assertIsNone(tui.question)

    def test_run_shows_llm_log_path(self):
        """With LLM logging enabled the log path is printed at startup."""
        import python_agent_harness.tui.core as tui_core

        tui, buf = make_tui()
        tui.session.client.log_path = "/tmp/llm.log"
        with (
            mock.patch.object(tui_core.config, "LLM_LOG_ENABLED", True),
            mock.patch.object(tui, "_read_multiline", return_value=None),
        ):
            tui.run()
        self.assertIn("/tmp/llm.log", buf.getvalue())

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

        with (
            mock.patch("python_agent_harness.tui.core.run_agent_loop", side_effect=boom),
            mock.patch.object(tui, "_run_live", return_value=False) as live,
        ):
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
            is_dumb_terminal=True,
            print=lambda *a, **k: None,
            file=io.StringIO(),
        )
        with (
            mock.patch(
                "python_agent_harness.tui.core.run_agent_loop", side_effect=RuntimeError("stop")
            ),
            mock.patch.object(tui, "_run_dumb", return_value=False) as dumb,
        ):
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
        with (
            mock.patch(
                "python_agent_harness.tui.core.run_agent_loop", side_effect=RuntimeError("stop")
            ),
            mock.patch.object(tui, "_run_live", side_effect=KeyboardInterrupt),
        ):
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

        with (
            mock.patch.object(tui, "_ask_question_blocking", side_effect=ask),
            mock.patch.object(tui, "_dump_conversation"),
        ):
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

        with (
            mock.patch.object(tui, "_ask_question_blocking", side_effect=ask),
            mock.patch.object(tui, "_dump_conversation"),
        ):
            result = tui._run_dumb(worker)
        self.assertFalse(result)
        self.assertEqual(answered, [1])
        self.assertIn("[BUILD]", buf.getvalue())

    def test_flush_tolerates_flush_errors(self):
        """A failing stdout flush must not crash the render loop."""
        tui, _ = make_tui()
        with mock.patch.object(tui.console.file, "flush", side_effect=OSError("boom")):
            tui._flush()  # must not raise

    def test_run_agent_error_logged(self):
        """An agent-loop exception on the current run is surfaced in the
        status bar."""
        tui, _ = make_tui()
        with mock.patch(
            "python_agent_harness.tui.core.run_agent_loop", side_effect=RuntimeError("boom")
        ):
            tui._run_agent("hi", tui.run_seq)
        self.assertIn("agent error: boom", tui.status)

    def test_run_agent_calls_restore(self):
        """The current run's finally fires the restore callback."""
        tui, _ = make_tui()
        restored = []
        with mock.patch(
            "python_agent_harness.tui.core.run_agent_loop", side_effect=RuntimeError("boom")
        ):
            tui._run_agent("hi", tui.run_seq, restore=lambda: restored.append(1))
        self.assertEqual(restored, [1])

    def test_start_agent_clears_todos(self):
        """A new top-level run drops any todo list left over from a
        previous run so a finished task's todos don't stay pinned into
        the next task."""
        tui, _ = make_tui()
        self.assertTrue(tui.session.todos)  # make_tui seeds a todo list
        with (
            mock.patch.object(tui, "_run_agent"),
            mock.patch.object(tui, "_run_live", return_value=False),
        ):
            tui._start_agent("next task")
        self.assertEqual(tui.session.todos, [])


if __name__ == "__main__":
    unittest.main()
