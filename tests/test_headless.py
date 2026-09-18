"""Tests for headless mode: HeadlessView and the CLI --headless path."""

from __future__ import annotations

import io
import os
import sys
import threading
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(__file__))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
import session_sandbox  # noqa: F401,E402  (side-effect: redirect SESSION_DIR)

from python_agent_harness import cli
from python_agent_harness.controller import RunHandle
from python_agent_harness.headless import (
    HeadlessView,
    final_answer_text,
    restore_session,
    run_headless,
)
from python_agent_harness.models import Message


class TestHeadlessView(unittest.TestCase):
    def test_deltas_ignored(self):
        out = io.StringIO()
        err = io.StringIO()
        view = HeadlessView(out=out, err=err)
        view.on_delta("hello ")
        view.on_delta("world")
        self.assertEqual(out.getvalue(), "")

    def test_notify_tool_start_to_err(self):
        out = io.StringIO()
        err = io.StringIO()
        view = HeadlessView(out=out, err=err)
        view.on_notify("tool_start", ["Read", "Grep"])
        self.assertIn("Read, Grep", err.getvalue())

    def test_notify_error_to_err(self):
        out = io.StringIO()
        err = io.StringIO()
        view = HeadlessView(out=out, err=err)
        view.on_notify("error", "no quota")
        self.assertIn("no quota", err.getvalue())

    def test_confirm_auto_approves(self):
        view = HeadlessView()
        self.assertTrue(view.confirm("switch to build?"))

    def test_ask_returns_unanswered(self):
        view = HeadlessView()
        self.assertEqual(view.ask([{"question": "pick one"}]), "Unanswered")

    def test_run_is_noop(self):
        view = HeadlessView()
        self.assertIsNone(view.run())


class TestFinalAnswerText(unittest.TestCase):
    def test_strips_final_check_block(self):
        session = mock.Mock()
        session.last_messages = [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content="The answer is 4.\n\n[FINAL CHECK]\n- Goal: g\n- Status: SUCCESS\n- Evidence: e",
            ),
        ]
        self.assertEqual(final_answer_text(session), "The answer is 4.")

    def test_strips_reasoning_preamble(self):
        session = mock.Mock()
        session.last_messages = [
            Message(
                role="assistant",
                content="Let me think... 1+1=2. So the answer is 2.",
                reasoning="Let me think... 1+1=2. ",
            ),
        ]
        self.assertEqual(final_answer_text(session), "So the answer is 2.")

    def test_empty_when_no_assistant_message(self):
        session = mock.Mock()
        session.last_messages = [Message(role="user", content="hi")]
        self.assertEqual(final_answer_text(session), "")

    def test_empty_when_history_empty(self):
        session = mock.Mock()
        session.last_messages = []
        self.assertEqual(final_answer_text(session), "")


class TestRunHeadless(unittest.TestCase):
    def _fake_handle(self, worker: threading.Thread) -> RunHandle:
        worker.start()
        return RunHandle(
            worker=worker,
            seq=1,
            display_text="hi",
            errors=[],
            warnings=[],
        )

    def test_submits_and_joins_worker(self):
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        with mock.patch("python_agent_harness.headless.Controller") as ctrl_cls:
            ctrl = ctrl_cls.return_value
            ctrl.submit.return_value = handle
            rc = run_headless(session, "hello")
        self.assertEqual(rc, 0)
        ctrl_cls.assert_called_once_with(session)
        ctrl.attach_view.assert_called_once()
        ctrl.submit.assert_called_once_with("hello")
        self.assertFalse(worker.is_alive())  # join() completed

    def test_returns_1_when_nothing_to_send(self):
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        with mock.patch("python_agent_harness.headless.Controller") as ctrl_cls:
            ctrl = ctrl_cls.return_value
            ctrl.submit.return_value = None
            rc = run_headless(session, "@missing.txt")
        self.assertEqual(rc, 1)

    def test_warnings_printed_to_err(self):
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        handle.warnings = ["model x does not support image input"]
        err = io.StringIO()
        with mock.patch("python_agent_harness.headless.Controller") as ctrl_cls:
            ctrl = ctrl_cls.return_value
            ctrl.submit.return_value = handle
            rc = run_headless(session, "hi", err=err)
        self.assertEqual(rc, 0)
        self.assertIn("model x does not support image input", err.getvalue())

    def test_returns_1_when_agent_error_notified(self):
        """An "error" notification during the run must yield exit code 1
        so CI can detect a failed agent run."""
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        err = io.StringIO()
        fake_view = mock.Mock()
        fake_view.errors = ["no quota"]
        fake_view.err = err
        with (
            mock.patch("python_agent_harness.headless.Controller") as ctrl_cls,
            mock.patch("python_agent_harness.headless.HeadlessView", return_value=fake_view),
        ):
            ctrl = ctrl_cls.return_value
            ctrl.submit.return_value = handle
            rc = run_headless(session, "hi", err=err)
        self.assertEqual(rc, 1)

    def test_final_answer_written_to_out(self):
        session = mock.Mock()
        session.last_messages = [
            Message(
                role="assistant",
                content="Done.\n\n[FINAL CHECK]\n- Goal: g\n- Status: SUCCESS\n- Evidence: e",
            ),
        ]
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        out = io.StringIO()
        with mock.patch("python_agent_harness.headless.Controller") as ctrl_cls:
            ctrl = ctrl_cls.return_value
            ctrl.submit.return_value = handle
            rc = run_headless(session, "hi", out=out)
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue(), "Done.\n")

    def test_model_profile_switch(self):
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        with (
            mock.patch("python_agent_harness.headless.Controller") as ctrl_cls,
            mock.patch("python_agent_harness.headless.HeadlessView") as view_cls,
        ):
            ctrl = ctrl_cls.return_value
            ctrl.submit.return_value = handle
            ctrl.switch_model.return_value = (True, "switched to fast (m-fast)")
            view = view_cls.return_value
            view.errors = []
            rc = run_headless(session, "hi", model="fast")
        self.assertEqual(rc, 0)
        ctrl.switch_model.assert_called_once_with("fast")

    def test_model_raw_name_fallback(self):
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        err = io.StringIO()
        with (
            mock.patch("python_agent_harness.headless.Controller") as ctrl_cls,
            mock.patch("python_agent_harness.headless.HeadlessView") as view_cls,
        ):
            ctrl = ctrl_cls.return_value
            ctrl.submit.return_value = handle
            ctrl.switch_model.return_value = (False, "unknown model: x")
            view = view_cls.return_value
            view.errors = []
            view.err = err
            rc = run_headless(session, "hi", model="raw-model-name", err=err)
        self.assertEqual(rc, 0)
        self.assertEqual(ctrl.session.client.model, "raw-model-name")
        self.assertEqual(ctrl.session.model, "raw-model-name")
        self.assertIn("using model name directly", err.getvalue())


class TestRestoreSession(unittest.TestCase):
    def _write_session(self, d: str, name: str = "my-session") -> str:

        from python_agent_harness.persistence import SessionPersistence

        store = SessionPersistence(
            project_dir=d,
            model="gpt-x",
            agent="default",
        )
        store.file_path = os.path.join(d, f"{name}.md")
        body = "**user**: hello\n\n**assistant**: hi there\n"
        path = store.save(body)
        assert path is not None
        return path

    def _fake_controller(self):
        import unittest.mock as mock

        ctrl = mock.Mock()
        ctrl.model = "gpt-x"
        ctrl.llm_settings = {"model": "gpt-x"}
        ctrl.model_profiles = {}
        ctrl.run_generation = 0
        ctrl.switch_agent.return_value = (True, "agent ok")
        ctrl.switch_model.return_value = (True, "model ok")
        return ctrl

    def test_restore_by_path(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = self._write_session(d)
            ctrl = self._fake_controller()
            err = io.StringIO()
            ok = restore_session(ctrl, path, err)
        self.assertTrue(ok)
        self.assertEqual(ctrl.store.file_path, path)
        self.assertEqual(len(ctrl.last_messages), 2)
        self.assertIn("restored:", err.getvalue())
        self.assertEqual(ctrl.run_generation, 1)
        ctrl.clear_todos.assert_called_once()

    def test_restore_latest(self):
        from python_agent_harness.persistence import session_dir

        d = str(session_dir())
        session_dir().mkdir(parents=True, exist_ok=True)
        path = self._write_session(d, name="latest-copy")
        try:
            ctrl = self._fake_controller()
            err = io.StringIO()
            ok = restore_session(ctrl, "latest", err)
            self.assertTrue(ok)
            self.assertIn("latest-copy.md", ctrl.store.file_path)
        finally:
            os.unlink(path)

    def test_restore_missing_returns_false(self):
        from python_agent_harness.persistence import session_dir

        if session_dir().is_dir():
            for f in session_dir().glob("*.md"):
                f.unlink()
        ctrl = self._fake_controller()
        err = io.StringIO()
        ok = restore_session(ctrl, "latest", err)
        self.assertFalse(ok)
        self.assertIn("no session found", err.getvalue())

    def test_restore_switches_agent_and_model(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = self._write_session(d)
            ctrl = self._fake_controller()
            ctrl.model = "gpt-other"
            ctrl.llm_settings = {"model": "gpt-other"}
            ctrl.switch_agent.return_value = (True, "agent ok")
            ctrl.switch_model.return_value = (True, "model ok")
            err = io.StringIO()
            ok = restore_session(ctrl, path, err)
        self.assertTrue(ok)
        ctrl.switch_agent.assert_called_once_with("default")
        ctrl.switch_model.assert_called_once_with("gpt-x")


class TestCliHeadless(unittest.TestCase):
    def test_headless_flag_routes_to_run_headless(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch("python_agent_harness.cli.make_session_with_mcp", return_value=session),
            mock.patch("python_agent_harness.headless.run_headless") as rh,
        ):
            rc = cli.main(["headless", "fix it", "--project", "/tmp/proj"])
        self.assertEqual(rc, rh.return_value)
        rh.assert_called_once()
        self.assertEqual(rh.call_args.args[0], session)
        self.assertEqual(rh.call_args.args[1], "fix it")
        session.close.assert_called_once()

    def test_headless_reads_prompt_from_stdin(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch("python_agent_harness.cli.make_session_with_mcp", return_value=session),
            mock.patch("python_agent_harness.headless.run_headless") as rh,
            mock.patch("sys.stdin", io.StringIO("from stdin")),
        ):
            rc = cli.main(["headless", "--project", "/tmp/proj"])
        self.assertEqual(rc, rh.return_value)
        self.assertEqual(rh.call_args.args[1], "from stdin")
        session.close.assert_called_once()

    def test_headless_closes_session_on_error(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch("python_agent_harness.cli.make_session_with_mcp", return_value=session),
            mock.patch(
                "python_agent_harness.headless.run_headless",
                side_effect=RuntimeError("boom"),
            ),
            self.assertRaises(RuntimeError),
        ):
            cli.main(["headless", "x", "--project", "/tmp/proj"])
        session.close.assert_called_once()

    def test_headless_model_flag_passed_through(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch("python_agent_harness.cli.make_session_with_mcp", return_value=session),
            mock.patch("python_agent_harness.headless.run_headless") as rh,
        ):
            cli.main(["headless", "hi", "--model", "fast"])
        self.assertEqual(rh.call_args.kwargs["model"], "fast")

    def test_headless_restore_flag_passed_through(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch("python_agent_harness.cli.make_session_with_mcp", return_value=session),
            mock.patch("python_agent_harness.headless.run_headless") as rh,
        ):
            cli.main(["headless", "fix it", "--restore", "my-session"])
        self.assertEqual(rh.call_args.kwargs["restore"], "my-session")

    def test_headless_bare_restore_defaults_to_latest(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch("python_agent_harness.cli.make_session_with_mcp", return_value=session),
            mock.patch("python_agent_harness.headless.run_headless") as rh,
        ):
            cli.main(["headless", "fix it", "--restore"])
        self.assertEqual(rh.call_args.kwargs["restore"], "latest")

    def test_parser_has_headless_and_prompt(self):
        parser = cli.build_parser()
        args = parser.parse_args(["headless", "p", "--project", "/tmp/proj"])
        self.assertEqual(args.command, "headless")
        self.assertEqual(args.prompt, "p")
        self.assertEqual(args.project, "/tmp/proj")


if __name__ == "__main__":
    unittest.main()
