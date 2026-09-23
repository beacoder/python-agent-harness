"""Tests for headless mode: HeadlessView and the CLI --headless path."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import unittest
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from python_agent_harness.core.models import Message
from python_agent_harness.entry import cli
from python_agent_harness.entry.controller import RunHandle
from python_agent_harness.entry.headless import (
    HeadlessView,
    JsonlView,
    final_answer_text,
    restore_session,
    run_headless,
    run_headless_jsonl,
    signal_canceller,
)
from tests.support import (
    plan_cleanup,  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
    session_sandbox,  # noqa: F401,E402  (side-effect: redirect SESSION_DIR)
)


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

    def test_notify_tool_start_without_names(self):
        err = io.StringIO()
        view = HeadlessView(err=err)
        view.on_notify("tool_start", None)
        self.assertIn("[tools: tools]", err.getvalue())

    def test_notify_run_done_to_err(self):
        err = io.StringIO()
        view = HeadlessView(err=err)
        view.on_notify("run_done")
        self.assertIn("[done]", err.getvalue())

    def test_on_log_to_err(self):
        err = io.StringIO()
        view = HeadlessView(err=err)
        view.on_log("compacting")
        self.assertIn("[log: compacting]", err.getvalue())

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

    def test_skips_check_only_trailing_message(self):
        session = mock.Mock()
        session.last_messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="Hello! How can I help you today?"),
            Message(
                role="assistant",
                content="[FINAL CHECK]\n- Goal: g\n- Status: SUCCESS\n- Evidence: e",
            ),
        ]
        self.assertEqual(final_answer_text(session), "Hello! How can I help you today?")

    def test_empty_when_all_assistant_messages_check_only(self):
        session = mock.Mock()
        session.last_messages = [
            Message(
                role="assistant",
                content="[FINAL CHECK]\n- Goal: g\n- Status: SUCCESS\n- Evidence: e",
            ),
        ]
        self.assertEqual(final_answer_text(session), "")

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
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl = ctrl_cls.return_value
            ctrl.submit.return_value = handle
            rc = run_headless(session, "hello")
        self.assertEqual(rc, 0)
        ctrl_cls.assert_called_once_with(session)
        ctrl.attach_view.assert_called_once()
        ctrl.submit.assert_called_once_with("hello", max_rounds=None, timeout=None)
        self.assertFalse(worker.is_alive())  # join() completed

    def test_returns_1_when_nothing_to_send(self):
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
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
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
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
            mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls,
            mock.patch("python_agent_harness.entry.headless.HeadlessView", return_value=fake_view),
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
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
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
            mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls,
            mock.patch("python_agent_harness.entry.headless.HeadlessView") as view_cls,
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
            mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls,
            mock.patch("python_agent_harness.entry.headless.HeadlessView") as view_cls,
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

        from python_agent_harness.io.persistence import SessionPersistence

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
        from python_agent_harness.io.persistence import session_dir

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
        from python_agent_harness.io.persistence import session_dir

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

    def test_restore_by_title_substring(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = self._write_session(d, name="my session_250101120000")
            ctrl = self._fake_controller()
            err = io.StringIO()
            with mock.patch(
                "python_agent_harness.io.persistence.find_session_by_title", return_value=path
            ) as find:
                ok = restore_session(ctrl, "MY SESSION", err)
        self.assertTrue(ok)
        find.assert_called_once_with("MY SESSION")
        self.assertEqual(ctrl.store.title, "my session")

    def test_restore_agent_failure_resets_to_default(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from python_agent_harness.io.persistence import SessionPersistence

            store = SessionPersistence(project_dir=d, model="gpt-x", agent="ghost")
            store.file_path = os.path.join(d, "ghost-session.md")
            path = store.save("**user**: hi\n\n**assistant**: yo\n")
            ctrl = self._fake_controller()
            ctrl.switch_agent.side_effect = [(False, "no such agent: ghost"), (True, "agent ok")]
            err = io.StringIO()
            ok = restore_session(ctrl, path, err)
        self.assertTrue(ok)
        self.assertEqual(
            ctrl.switch_agent.call_args_list,
            [mock.call("ghost"), mock.call("default")],
        )
        self.assertIn("warning: no such agent: ghost", err.getvalue())

    def test_restore_default_model_via_pseudo_profile(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = self._write_session(d)  # saved model gpt-x
            ctrl = self._fake_controller()
            ctrl.model = "drifted-model"
            ctrl.llm_settings = {"model": "gpt-x"}
            err = io.StringIO()
            ok = restore_session(ctrl, path, err)
        self.assertTrue(ok)
        ctrl.switch_model.assert_called_once_with("default")

    def test_restore_model_via_matching_profile(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = self._write_session(d)  # saved model gpt-x
            ctrl = self._fake_controller()
            ctrl.model = "m1"
            ctrl.llm_settings = {"model": "other"}
            ctrl.model_profiles = {"prof": {"model": "gpt-x"}}
            ctrl.switch_model.side_effect = [(False, "unknown model"), (True, "switched")]
            err = io.StringIO()
            ok = restore_session(ctrl, path, err)
        self.assertTrue(ok)
        self.assertEqual(
            ctrl.switch_model.call_args_list,
            [mock.call("gpt-x"), mock.call("prof")],
        )

    def test_restore_model_without_profile_warns(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = self._write_session(d)  # saved model gpt-x
            ctrl = self._fake_controller()
            ctrl.model = "m1"
            ctrl.llm_settings = {"model": "other"}
            ctrl.model_profiles = {}
            ctrl.switch_model.side_effect = [(False, "unknown model")]
            err = io.StringIO()
            ok = restore_session(ctrl, path, err)
        self.assertTrue(ok)
        self.assertIn("has no matching profile", err.getvalue())

    def test_restore_failure_exit_code(self):
        session = mock.Mock()
        session.last_messages = []
        with mock.patch("python_agent_harness.entry.headless.restore_session", return_value=False):
            rc = run_headless(session, "hi", restore="nope")
        self.assertEqual(rc, 1)


class TestJsonlView(unittest.TestCase):
    def _lines(self, out: io.StringIO) -> list[dict]:
        import json

        return [json.loads(line) for line in out.getvalue().splitlines() if line]

    def test_events_emitted_as_json_lines(self):
        import json

        out = io.StringIO()
        err = io.StringIO()
        view = JsonlView(out=out, err=err)
        view.emit_start("do it", ["w1"])
        view.on_delta("hello ")
        view.on_notify("tool_start", ["Read"])
        view.on_notify("error", "no quota")
        view.on_log("warming up")
        view.emit_result("Done.")
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        self.assertEqual(
            lines[0], {"seq": 1, "type": "start", "prompt": "do it", "warnings": ["w1"]}
        )
        self.assertEqual(lines[1], {"seq": 2, "type": "delta", "text": "hello "})
        self.assertEqual(
            lines[2], {"seq": 3, "type": "notify", "kind": "tool_start", "data": ["Read"]}
        )
        self.assertEqual(
            lines[3], {"seq": 4, "type": "notify", "kind": "error", "data": "no quota"}
        )
        self.assertEqual(lines[4], {"seq": 5, "type": "log", "message": "warming up"})
        # the error seen during the run is reported on the result line
        self.assertEqual(
            lines[5],
            {
                "seq": 6,
                "type": "result",
                "answer": "Done.",
                "errors": ["no quota"],
                "cancelled": False,
            },
        )
        # each line is compact single-line JSON: no embedded newlines
        self.assertTrue(all("\n" not in line for line in out.getvalue().splitlines()))

    def test_events_before_start_are_buffered_behind_start_line(self):
        """The worker starts before emit_start runs, so early events must
        be buffered and flushed after start — start is always first."""
        import json

        out = io.StringIO()
        view = JsonlView(out=out)
        # simulate the race: events arrive before emit_start
        view.on_delta("early ")
        view.on_notify("tool_start", ["Bash"])
        view.emit_start("go", [])
        view.on_delta("live")
        view.emit_result("ok")
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        self.assertEqual(lines[0]["type"], "start")
        self.assertEqual(lines[1], {"seq": 2, "type": "delta", "text": "early "})
        self.assertEqual(
            lines[2], {"seq": 3, "type": "notify", "kind": "tool_start", "data": ["Bash"]}
        )
        self.assertEqual(lines[3], {"seq": 4, "type": "delta", "text": "live"})
        self.assertEqual(lines[-1]["type"], "result")

    def test_concurrent_emit_and_start_never_corrupts_stream(self):
        """A worker hammering _emit while emit_start swaps the buffer to
        live mode must not raise (None.append race) and every line must
        stay valid JSON with start first."""
        import json

        for _ in range(20):  # the race is timing-dependent: hammer it
            out = io.StringIO()
            view = JsonlView(out=out)
            errors: list[BaseException] = []

            def worker(view=view, errors=errors) -> None:
                try:
                    for i in range(200):
                        view.on_delta(f"d{i}")
                except BaseException as exc:  # noqa: BLE001 - recorded below
                    errors.append(exc)

            t = threading.Thread(target=worker)
            t.start()
            view.emit_start("go", [])
            t.join()
            view.emit_result("ok")
            self.assertEqual(errors, [])
            lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
            self.assertEqual(lines[0]["type"], "start")
            self.assertEqual(lines[-1]["type"], "result")
            self.assertEqual(len(lines), 202)  # start + 200 deltas + result

    def test_error_notified_recorded_and_echoed_to_err(self):
        out = io.StringIO()
        err = io.StringIO()
        view = JsonlView(out=out, err=err)
        view.on_notify("error", "no quota")
        self.assertEqual(view.errors, ["no quota"])  # parent HeadlessView behavior
        self.assertIn("[error: no quota]", err.getvalue())

    def test_non_error_notify_not_echoed_to_err(self):
        out = io.StringIO()
        err = io.StringIO()
        view = JsonlView(out=out, err=err)
        view.on_notify("tool_start", ["Read"])
        self.assertEqual(err.getvalue(), "")

    def test_deltas_recorded_but_errors_only_from_notify(self):
        out = io.StringIO()
        view = JsonlView(out=out)
        view.on_delta("text")
        self.assertEqual(view.errors, [])

    def test_confirm_auto_approves_and_ask_unanswered(self):
        view = JsonlView()
        self.assertTrue(view.confirm("switch to build?"))
        self.assertEqual(view.ask([{"question": "pick one"}]), "Unanswered")

    def test_run_is_noop(self):
        self.assertIsNone(JsonlView().run())


class TestRunHeadlessJsonl(unittest.TestCase):
    def _fake_handle(self, worker: threading.Thread) -> RunHandle:
        worker.start()
        return RunHandle(
            worker=worker,
            seq=1,
            display_text="hi",
            errors=[],
            warnings=[],
        )

    def test_emits_start_and_result_lines(self):
        import json
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = [
            Message(
                role="assistant",
                content="Done.\n\n[FINAL CHECK]\n- Goal: g\n- Status: SUCCESS\n- Evidence: e",
            ),
        ]
        session.cancel_event = threading.Event()
        session.model = "m-test"
        session.usage_totals = {"input": 0, "output": 0, "rounds": 0}
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl = ctrl_cls.return_value
            ctrl.submit.return_value = handle
            rc = run_headless_jsonl(session, "hello", out=out)
        self.assertEqual(rc, 0)
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        self.assertEqual(lines[0], {"seq": 1, "type": "start", "prompt": "hello", "warnings": []})
        self.assertEqual(
            lines[-1],
            {
                "seq": 2,
                "type": "result",
                "answer": "Done.",
                "errors": [],
                "cancelled": False,
                "usage": {"input": 0, "output": 0, "rounds": 0},
                "model": "m-test",
            },
        )
        ctrl.submit.assert_called_once_with("hello", max_rounds=None, timeout=None)

    def test_submit_warnings_on_start_line(self):
        import json
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        handle.warnings = ["model x does not support image input"]
        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl_cls.return_value.submit.return_value = handle
            rc = run_headless_jsonl(session, "hi", out=out)
        self.assertEqual(rc, 0)
        start = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual(start["warnings"], ["model x does not support image input"])

    def test_returns_1_when_nothing_to_send(self):
        import json
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        session.cancel_event = threading.Event()
        session.model = "m-test"
        session.usage_totals = {"input": 0, "output": 0, "rounds": 0}
        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl_cls.return_value.submit.return_value = None
            rc = run_headless_jsonl(session, "@missing.txt", out=out)
        self.assertEqual(rc, 1)
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        # result is the only line (no start), with the failure reason
        self.assertEqual(len(lines), 1)
        self.assertEqual(
            lines[0],
            {
                "seq": 1,
                "type": "result",
                "answer": "",
                "errors": ["nothing to send"],
                "cancelled": False,
                "usage": {"input": 0, "output": 0, "rounds": 0},
                "model": "m-test",
            },
        )

    def test_returns_1_when_error_notified(self):
        """An "error" notification during the run must yield exit code 1."""
        import json
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        out = io.StringIO()
        err = io.StringIO()

        real_view = JsonlView(out=out, err=err)

        with (
            mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls,
            mock.patch("python_agent_harness.entry.headless.JsonlView", return_value=real_view),
        ):
            ctrl_cls.return_value.submit.return_value = handle
            # inject the error the way the agent loop would
            real_view.on_notify("error", "no quota")
            rc = run_headless_jsonl(session, "hi", out=out, err=err)
        self.assertEqual(rc, 1)
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        # the pre-start error event is buffered behind the start line
        notify_idx = next(i for i, line in enumerate(lines) if line["type"] == "notify")
        self.assertEqual(lines[notify_idx]["kind"], "error")
        result = lines[-1]
        self.assertEqual(result["type"], "result")
        self.assertEqual(result["errors"], ["no quota"])

    def test_restore_failure_emits_result_line(self):
        import json
        import unittest.mock as mock

        session = mock.Mock()
        session.last_messages = []
        session.cancel_event = threading.Event()
        session.model = "m-test"
        session.usage_totals = {"input": 0, "output": 0, "rounds": 0}
        out = io.StringIO()
        err = io.StringIO()
        with (
            mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls,
            mock.patch(
                "python_agent_harness.entry.headless.restore_session",
                return_value=False,
            ) as restore,
        ):
            ctrl_cls.return_value.attach_view.return_value = None
            rc = run_headless_jsonl(session, "hi", out=out, err=err, restore="nope")
        self.assertEqual(rc, 1)
        restore.assert_called_once()
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        # result is the only line (no start), with the failure reason
        self.assertEqual(len(lines), 1)
        self.assertEqual(
            lines[0],
            {
                "seq": 1,
                "type": "result",
                "answer": "",
                "errors": ["restore failed"],
                "cancelled": False,
                "usage": {"input": 0, "output": 0, "rounds": 0},
                "model": "m-test",
            },
        )

    def test_jsonl_model_selection_applied(self):
        session = mock.Mock()
        session.last_messages = []
        worker = threading.Thread(target=lambda: None)
        handle = self._fake_handle(worker)
        out = io.StringIO()
        with (
            mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls,
            mock.patch("python_agent_harness.entry.headless._select_model") as select,
        ):
            ctrl_cls.return_value.submit.return_value = handle
            rc = run_headless_jsonl(session, "hi", model="fast", out=out)
        self.assertEqual(rc, 0)
        select.assert_called_once()
        self.assertEqual(select.call_args.args[1], "fast")


class TestCliHeadless(unittest.TestCase):
    def test_headless_flag_routes_to_run_headless(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
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
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
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
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch(
                "python_agent_harness.entry.headless.run_headless",
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
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
        ):
            cli.main(["headless", "hi", "--model", "fast"])
        self.assertEqual(rh.call_args.kwargs["model"], "fast")

    def test_headless_restore_flag_passed_through(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
        ):
            cli.main(["headless", "fix it", "--restore", "my-session"])
        self.assertEqual(rh.call_args.kwargs["restore"], "my-session")

    def test_headless_bare_restore_defaults_to_latest(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
        ):
            cli.main(["headless", "fix it", "--restore"])
        self.assertEqual(rh.call_args.kwargs["restore"], "latest")

    def test_headless_json_flag_routes_to_run_headless_jsonl(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless_jsonl") as rhj,
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
        ):
            rc = cli.main(["headless", "fix it", "--json"])
        self.assertEqual(rc, rhj.return_value)
        rhj.assert_called_once()
        self.assertEqual(rhj.call_args.args[1], "fix it")
        rh.assert_not_called()  # plain runner must not run too
        session.close.assert_called_once()

    def test_headless_without_json_flag_keeps_plain_runner(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless_jsonl") as rhj,
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
        ):
            cli.main(["headless", "fix it"])
        rh.assert_called_once()
        rhj.assert_not_called()

    def test_parser_has_json_flag(self):
        parser = cli.build_parser()
        args = parser.parse_args(["headless", "p", "--json"])
        self.assertTrue(args.json)
        args = parser.parse_args(["headless", "p"])
        self.assertFalse(args.json)

    def test_parser_has_headless_and_prompt(self):
        parser = cli.build_parser()
        args = parser.parse_args(["headless", "p", "--project", "/tmp/proj"])
        self.assertEqual(args.command, "headless")
        self.assertEqual(args.prompt, "p")
        self.assertEqual(args.project, "/tmp/proj")


class TestProtocolHardening(unittest.TestCase):
    """--json exec-protocol hardening: seq/run_id correlation, usage and
    model on every result line, graceful signal cancel, run budgets."""

    def _session(self) -> mock.Mock:
        s = mock.Mock()
        s.last_messages = []
        s.cancel_event = threading.Event()
        s.model = "m-test"
        s.usage_totals = {"input": 120, "output": 45, "rounds": 2}
        return s

    def test_seq_monotonic_across_all_lines(self):
        out = io.StringIO()
        view = JsonlView(out=out)
        view.on_delta("a")
        view.on_log("l")
        view.emit_start("p", [])
        view.on_delta("b")
        view.emit_result("done")
        seqs = [json.loads(line)["seq"] for line in out.getvalue().splitlines() if line]
        self.assertEqual(seqs, [1, 2, 3, 4, 5])

    def test_run_id_echoed_on_every_line(self):
        out = io.StringIO()
        view = JsonlView(out=out, run_id="exec-42")
        view.emit_start("p", [])
        view.on_notify("tool_start", ["Read"])
        view.emit_result("done")
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        self.assertTrue(lines)
        self.assertTrue(all(line["run_id"] == "exec-42" for line in lines))

    def test_no_run_id_field_when_unset(self):
        out = io.StringIO()
        view = JsonlView(out=out)
        view.emit_start("p", [])
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        self.assertNotIn("run_id", lines[0])

    def test_result_line_carries_usage_model_cancelled(self):
        session = self._session()
        session.last_messages = [Message(role="assistant", content="done")]
        worker = threading.Thread(target=lambda: None)
        worker.start()
        handle = RunHandle(worker=worker, seq=1, display_text="hi", errors=[], warnings=[])
        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl_cls.return_value.submit.return_value = handle
            rc = run_headless_jsonl(session, "hi", out=out)
        self.assertEqual(rc, 0)
        result = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual(result["usage"], {"input": 120, "output": 45, "rounds": 2})
        self.assertEqual(result["model"], "m-test")
        self.assertIs(result["cancelled"], False)

    def test_usage_read_after_submit_swaps_totals(self):
        """submit() replaces session.usage_totals with a fresh dict; the
        result line must snapshot the CURRENT totals (the ones the run
        actually bumped), not the pre-submit reference (always zeros)."""
        session = self._session()
        session.last_messages = [Message(role="assistant", content="done")]
        worker = threading.Thread(target=lambda: None)
        worker.start()
        handle = RunHandle(worker=worker, seq=1, display_text="hi", errors=[], warnings=[])

        def swap_totals(prompt, **kwargs):
            # emulate Controller.submit: the run bumps a NEW dict
            session.usage_totals = {"input": 500, "output": 42, "rounds": 3}
            return handle

        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl_cls.return_value.submit.side_effect = swap_totals
            rc = run_headless_jsonl(session, "hi", out=out)
        self.assertEqual(rc, 0)
        result = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual(result["usage"], {"input": 500, "output": 42, "rounds": 3})

    def test_cancel_produces_result_line_and_exit_1(self):
        session = self._session()
        session.cancel_event.set()
        worker = threading.Thread(target=lambda: None)
        worker.start()
        handle = RunHandle(worker=worker, seq=1, display_text="hi", errors=[], warnings=[])
        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl_cls.return_value.submit.return_value = handle
            rc = run_headless_jsonl(session, "hi", out=out)
        self.assertEqual(rc, 1)
        result = json.loads(out.getvalue().splitlines()[-1])
        self.assertIs(result["cancelled"], True)
        self.assertEqual(result["errors"], [])

    def test_budget_params_passed_to_submit(self):
        session = self._session()
        worker = threading.Thread(target=lambda: None)
        worker.start()
        handle = RunHandle(worker=worker, seq=1, display_text="hi", errors=[], warnings=[])
        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl_cls.return_value.submit.return_value = handle
            rc = run_headless_jsonl(session, "hi", out=out, max_rounds=7, timeout=120.0)
        self.assertEqual(rc, 0)
        ctrl_cls.return_value.submit.assert_called_once_with("hi", max_rounds=7, timeout=120.0)

    def test_signal_handler_cancels_and_restores(self):
        import signal as signal_mod

        class _Cancellable:
            def __init__(self) -> None:
                self.cancel_event = threading.Event()

            def cancel(self) -> None:
                self.cancel_event.set()

        session = _Cancellable()
        delivered: list[int] = []
        orig_int = signal_mod.getsignal(signal_mod.SIGINT)
        orig_term = signal_mod.getsignal(signal_mod.SIGTERM)
        self.addCleanup(signal_mod.signal, signal_mod.SIGINT, orig_int)
        self.addCleanup(signal_mod.signal, signal_mod.SIGTERM, orig_term)
        canceller = signal_canceller(session)
        with canceller:
            for sig in (signal_mod.SIGINT, signal_mod.SIGTERM):
                handler = signal_mod.getsignal(sig)
                # the installed handler IS the canceller's: invoke it the
                # way the OS would deliver the signal
                self.assertTrue(callable(handler))
                handler(sig, None)
                delivered.append(sig)
            self.assertTrue(session.cancel_event.is_set())
        # previous handlers restored on exit
        self.assertIs(signal_mod.getsignal(signal_mod.SIGINT), orig_int)
        self.assertIs(signal_mod.getsignal(signal_mod.SIGTERM), orig_term)
        self.assertEqual(delivered, [signal_mod.SIGINT, signal_mod.SIGTERM])

    def test_signal_canceller_noop_outside_main_thread(self):
        session = self._session()
        errors: list[BaseException] = []
        done = threading.Event()

        def run() -> None:
            try:
                with signal_canceller(session):
                    pass  # ValueError from signal.signal is swallowed
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=5)
        self.assertTrue(done.is_set())
        self.assertEqual(errors, [])

    def test_plain_runner_reports_cancelled(self):
        session = self._session()
        session.cancel_event.set()
        worker = threading.Thread(target=lambda: None)
        worker.start()
        handle = RunHandle(worker=worker, seq=1, display_text="hi", errors=[], warnings=[])
        err = io.StringIO()
        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl_cls.return_value.submit.return_value = handle
            rc = run_headless(session, "hi", out=out, err=err)
        self.assertEqual(rc, 1)
        self.assertIn("[cancelled]", err.getvalue())

    def test_plain_runner_budget_params_passed(self):
        session = self._session()
        worker = threading.Thread(target=lambda: None)
        worker.start()
        handle = RunHandle(worker=worker, seq=1, display_text="hi", errors=[], warnings=[])
        with mock.patch("python_agent_harness.entry.headless.Controller") as ctrl_cls:
            ctrl_cls.return_value.submit.return_value = handle
            rc = run_headless(session, "hi", max_rounds=3, timeout=10.0)
        self.assertEqual(rc, 0)
        ctrl_cls.return_value.submit.assert_called_once_with("hi", max_rounds=3, timeout=10.0)


class TestCliBudgetFlags(unittest.TestCase):
    """CLI plumbing for the unattended-run budget flags."""

    def test_parser_accepts_budget_and_run_id_flags(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            ["headless", "p", "--max-rounds", "25", "--timeout", "300.5", "--run-id", "exec-9"]
        )
        self.assertEqual(args.max_rounds, 25)
        self.assertEqual(args.timeout, 300.5)
        self.assertEqual(args.run_id, "exec-9")

    def test_budget_flags_default_to_none(self):
        parser = cli.build_parser()
        args = parser.parse_args(["headless", "p"])
        self.assertIsNone(args.max_rounds)
        self.assertIsNone(args.timeout)
        self.assertIsNone(args.run_id)

    def test_flags_passed_to_jsonl_runner(self):
        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless_jsonl") as rhj,
        ):
            cli.main(
                [
                    "headless",
                    "hi",
                    "--json",
                    "--max-rounds",
                    "7",
                    "--timeout",
                    "120",
                    "--run-id",
                    "exec-1",
                ]
            )
        self.assertEqual(rhj.call_args.kwargs["max_rounds"], 7)
        self.assertEqual(rhj.call_args.kwargs["timeout"], 120.0)
        self.assertEqual(rhj.call_args.kwargs["run_id"], "exec-1")

    def test_flags_passed_to_plain_runner(self):
        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
        ):
            cli.main(["headless", "hi", "--max-rounds", "7"])
        self.assertEqual(rh.call_args.kwargs["max_rounds"], 7)

    def test_config_defaults_used_when_flags_unset(self):
        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
            mock.patch(
                "python_agent_harness.session.config.load_headless_limits",
                return_value=(15, 90.0),
            ),
        ):
            cli.main(["headless", "hi"])
        self.assertEqual(rh.call_args.kwargs["max_rounds"], 15)
        self.assertEqual(rh.call_args.kwargs["timeout"], 90.0)

    def test_flag_zero_disables_config_budget(self):
        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.entry.cli.make_session_with_mcp", return_value=session
            ),
            mock.patch("python_agent_harness.entry.headless.run_headless") as rh,
            mock.patch(
                "python_agent_harness.session.config.load_headless_limits",
                return_value=(15, 90.0),
            ),
        ):
            cli.main(["headless", "hi", "--max-rounds", "0", "--timeout", "0"])
        self.assertIsNone(rh.call_args.kwargs["max_rounds"])
        self.assertIsNone(rh.call_args.kwargs["timeout"])


if __name__ == "__main__":
    unittest.main()
