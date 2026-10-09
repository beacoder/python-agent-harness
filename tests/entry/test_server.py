"""Tests for the resident agent server (entry/server.py)."""

from __future__ import annotations

import ast
import inspect
import io
import json
import os
import signal
import threading
import time
import unittest
from unittest import mock

from python_agent_harness.entry import server as server_module
from python_agent_harness.entry.headless import PROTOCOL_VERSION
from python_agent_harness.entry.server import (
    CAPABILITIES,
    AgentServer,
    ServerView,
    _AskState,
    _restorable_handler,
    _ShutdownSignal,
    graceful_signal_shutdown,
    run_serve,
)


def _lines(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines() if line]


def _of_type(lines: list[dict], type_: str) -> list[dict]:
    return [line for line in lines if line.get("type") == type_]


class FakeSession:
    """Minimal Session double: callbacks + cancel + usage, no agent loop."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self.model = "m-test"
        self.usage_totals: dict = {"input": 1, "output": 2, "rounds": 3}
        self.last_messages: list = []
        self.on_delta = None
        self.log_fn = None
        self.notify_fn = None
        self.confirm_fn = None
        self.ask_fn = None
        self.cancel_calls = 0

    def notify(self, kind: str, data=None) -> None:
        if self.notify_fn:
            self.notify_fn(kind, data)

    def log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)

    def confirm(self, prompt: str) -> bool:
        if self.confirm_fn:
            return self.confirm_fn(prompt)
        return True

    def ask_questions(self, questions: list) -> str:
        if self.ask_fn:
            return self.ask_fn(questions)
        return "Unanswered"

    def cancel(self) -> None:
        self.cancel_calls += 1
        self.cancel_event.set()


class RunScript:
    """What the fake agent's worker thread does for one run.

    Modes:
    - "plain": notify run_finished, done (the happy path).
    - "error": notify error first, then run_finished.
    - "ask": call session.ask_fn (blocking on the view) with a question;
      the returned answer is stored on ``ask_answer``.
    - "confirm": call session.confirm_fn (blocking on the view); the
      boolean is stored on ``confirm_answer``.
    - "raise": raise before finishing (the run thread must error-line).
    """

    def __init__(self, mode: str = "plain", error_message: str = "llm unreachable") -> None:
        self.mode = mode
        self.error_message = error_message
        self.ask_answer: str | None = None
        self.ask_answer_2: str | None = None
        self.confirm_answer: bool | None = None

    def drive(self, session: FakeSession, prompt: str) -> None:
        if self.mode == "raise":
            raise RuntimeError("boom")
        if self.mode == "error":
            session.notify("error", self.error_message)
        if self.mode == "ask":
            self.ask_answer = session.ask_questions([{"question": f"{prompt}?"}])
        if self.mode == "ask_twice":
            # The dangerous interleaving: the first question is
            # abandoned (timeout) and a second takes its place, so a
            # late reply for the first could land on the second.
            self.ask_answer = session.ask_questions([{"question": "Deploy to PRODUCTION?"}])
            self.ask_answer_2 = session.ask_questions([{"question": "Pick a colour"}])
        if self.mode == "confirm":
            self.confirm_answer = session.confirm(f"{prompt}?")
        session.last_messages = [mock.Mock(role="assistant")]
        session.last_messages[0].text_without_reasoning.return_value = f"answer to: {prompt}"
        session.notify("run_finished")


class FakeController:
    """Controller double: attach_view wires callbacks like the real one."""

    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.view = None
        self.submits: list[str] = []
        self.submit_kwargs: list[dict] = []
        self.script = RunScript()
        self.gate: threading.Event | None = None
        # real Controller.submit() swaps in a fresh zeroed usage_totals
        # dict per run (the run mutates it in place); these knobs let
        # tests exercise that lifecycle and the nothing-to-send path.
        self.reset_usage_on_submit = False
        self.submit_returns_none = False

    def attach_view(self, view) -> None:
        self.view = view
        session = self.session
        session.on_delta = view.on_delta
        session.notify_fn = view.on_notify
        session.log_fn = view.on_log
        session.confirm_fn = view.confirm
        session.ask_fn = view.ask

    def detach_view(self) -> None:
        self.view = None
        session = self.session
        session.on_delta = None
        session.notify_fn = None
        session.log_fn = None
        session.confirm_fn = None
        session.ask_fn = None

    def submit(self, prompt: str, **kwargs):
        self.submits.append(prompt)
        # Record the budget kwargs: a double that swallows **kwargs
        # would hide an unplumbed argument entirely.
        self.submit_kwargs.append(dict(kwargs))
        if self.submit_returns_none:
            return None
        if self.reset_usage_on_submit:
            # real Controller.submit(): a fresh dict per run, zeroed
            self.session.usage_totals = {
                "input": 0,
                "output": 0,
                "rounds": 0,
                "_lock": threading.Lock(),
                "_accumulate": True,
            }
        if self.gate is not None:
            self.gate.wait(5)
        script = self.script

        def _drive() -> None:
            try:
                totals = self.session.usage_totals
                if self.reset_usage_on_submit and isinstance(totals, dict):
                    # the agent loop mutates the totals in place during
                    # the run (mirrors the real run's token accounting)
                    totals["input"] += 5
                    totals["output"] += 7
                    totals["rounds"] += 2
                script.drive(self.session, prompt)
            except RuntimeError as e:
                self.session.log(f"agent error: {e}")

        worker = threading.Thread(target=_drive, daemon=True)
        worker.start()
        return mock.Mock(warnings=[], worker=worker)


class ServerTestBase(unittest.TestCase):
    """Base: builds an AgentServer wired to a FakeController."""

    def _server(self, session: FakeSession | None = None, **kwargs) -> AgentServer:
        session = session or FakeSession()
        self.session = session
        self.controller = FakeController(session)
        inp = io.StringIO()
        out = io.StringIO()
        server = AgentServer(session, inp, out, err=io.StringIO(), **kwargs)
        server.controller = self.controller  # type: ignore[assignment]
        return server

    def _wait_idle(self, server: AgentServer, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with server._active_guard:
                if server._active_run_id is None:
                    return
            time.sleep(0.01)
        self.fail("run never became idle")

    def _wait_pending(self, server: AgentServer, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            view = server.view
            if view is not None:
                with view._pending_lock:
                    if view._pending is not None:
                        return
            time.sleep(0.01)
        self.fail("ask never became pending")

    def _wait_view(self, server: AgentServer, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if server.view is not None:
                return server.view
            time.sleep(0.01)
        self.fail("view never appeared")

    def _submit_and_wait(self, server: AgentServer, prompt: str = "hello", run_id: str = "r1"):
        """Submit via the op path and wait for the run to finish."""
        server.op_submit({"op": "submit", "prompt": prompt, "run_id": run_id})
        self._wait_idle(server)
        return _lines(server.out)  # type: ignore[arg-type]


class TestReadyAndControl(ServerTestBase):
    def test_ready_line_first(self):
        server = self._server()
        server.serve_forever()
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[0]["type"], "ready")
        self.assertIn("pid", lines[0])
        self.assertNotIn("run_id", lines[0])

    def test_ping_pong(self):
        server = self._server()
        server.inp.write(json.dumps({"op": "ping"}) + "\n")
        server.inp.seek(0)
        server.serve_forever()
        self.assertEqual(
            [line["type"] for line in _lines(server.out)],  # type: ignore[arg-type]
            ["ready", "pong"],
        )

    def test_shutdown_stops_loop(self):
        server = self._server()
        server.inp.write(json.dumps({"op": "shutdown"}) + "\n")
        server.inp.seek(0)
        server.serve_forever()
        self.assertTrue(server._stopped.is_set())

    def test_eof_stops_loop(self):
        server = self._server()
        server.serve_forever()
        self.assertEqual(len(_lines(server.out)), 1)  # type: ignore[arg-type]

    def test_malformed_line_yields_error(self):
        server = self._server()
        server.inp.write("not json\n")
        server.inp.seek(0)
        server.serve_forever()
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[1]["type"], "error")
        self.assertIn("malformed", lines[1]["error"]["message"])

    def test_non_object_op_yields_error(self):
        server = self._server()
        server.inp.write(json.dumps(["op"]) + "\n")
        server.inp.seek(0)
        server.serve_forever()
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[1]["type"], "error")

    def test_unknown_op_yields_error(self):
        server = self._server()
        server.inp.write(json.dumps({"op": "teleport"}) + "\n")
        server.inp.seek(0)
        server.serve_forever()
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[1]["type"], "error")
        self.assertIn("teleport", lines[1]["error"]["message"])


class TestSubmit(ServerTestBase):
    def test_submit_runs_and_streams(self):
        server = self._server()
        lines = self._submit_and_wait(server)
        self.assertEqual(lines[0]["type"], "start")
        self.assertEqual(lines[-1]["type"], "result")
        self.assertEqual(self.controller.submits, ["hello"])
        self.assertEqual(lines[0]["run_id"], "r1")
        self.assertEqual(lines[-1]["run_id"], "r1")
        self.assertEqual(lines[-1]["answer"], "answer to: hello")
        self.assertEqual(lines[-1]["usage"], {"input": 1, "output": 2, "rounds": 3})
        self.assertEqual(lines[-1]["model"], "m-test")
        self.assertFalse(lines[-1]["cancelled"])

    def test_usage_reports_run_totals_not_zeros(self):
        """usage on the result line must be the run's accumulated totals.

        Regression: the snapshot was taken right after submit(), which
        (like the real Controller) swaps in a fresh zeroed totals dict
        — result lines always reported zeros and billing lost every run.
        """
        server = self._server()
        self.controller.reset_usage_on_submit = True
        lines = self._submit_and_wait(server)
        self.assertEqual(lines[-1]["usage"], {"input": 5, "output": 7, "rounds": 2})

    def test_nothing_to_send_reports_zero_usage(self):
        """A run that never started must not leak previous totals."""
        server = self._server()
        self.controller.submit_returns_none = True
        lines = self._submit_and_wait(server, prompt="@missing.txt")
        self.assertEqual(lines[-1]["errors"], [{"code": "nothing", "message": "nothing to send"}])
        self.assertEqual(lines[-1]["error_messages"], ["nothing to send"])
        self.assertEqual(lines[-1]["usage"], {"input": 0, "output": 0, "rounds": 0})

    def test_submit_requires_run_id(self):
        server = self._server()
        server.op_submit({"op": "submit", "prompt": "hi"})
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[0]["type"], "error")
        self.assertIn("run_id", lines[0]["error"]["message"])
        self.assertEqual(lines[0]["error"]["code"], "protocol")
        self.assertEqual(self.controller.submits, [])

    def test_submit_requires_prompt(self):
        server = self._server()
        server.op_submit({"op": "submit", "run_id": "r1"})
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[0]["type"], "error")
        self.assertIn("prompt", lines[0]["error"]["message"])

    def test_submit_rejects_blank_prompt(self):
        server = self._server()
        server.op_submit({"op": "submit", "run_id": "r1", "prompt": "  "})
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[0]["type"], "error")

    def test_submit_rejects_while_active(self):
        server = self._server()
        self.controller.gate = threading.Event()
        server.op_submit({"op": "submit", "prompt": "one", "run_id": "r1"})
        self._wait_view(server)
        server.op_submit({"op": "submit", "prompt": "two", "run_id": "r2"})
        errors = [
            line
            for line in _lines(server.out)
            if line["type"] == "error"  # type: ignore[arg-type]
        ]
        self.assertEqual(len(errors), 1)
        self.assertIn("already active", errors[0]["error"]["message"])
        self.controller.gate.set()
        self._wait_idle(server)
        self.assertEqual(self.controller.submits, ["one"])

    def test_worker_exception_yields_result_with_error(self):
        """An agent-loop crash is logged by the real Controller (the
        worker never re-raises); the run ends with an empty answer, and
        the result line must carry a reason rather than a silent blank."""
        server = self._server()
        self.controller.script = RunScript(mode="raise")
        server.op_submit({"op": "submit", "prompt": "hi", "run_id": "r1"})
        self._wait_idle(server)
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[-1]["type"], "result")
        self.assertIn("run produced no answer", [e["message"] for e in lines[-1]["errors"]])
        self.assertIn("no_answer", [e["code"] for e in lines[-1]["errors"]])
        self.assertIn(
            "agent error: boom", [line["message"] for line in lines if line["type"] == "log"]
        )

    def test_submit_returning_none_writes_single_result(self):
        server = self._server()

        def none_submit(prompt, **kwargs):
            self.controller.submits.append(prompt)
            return None

        self.controller.submit = none_submit  # type: ignore[method-assign]
        server.op_submit({"op": "submit", "prompt": "@missing.txt", "run_id": "r1"})
        self._wait_idle(server)
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual([line["type"] for line in lines], ["result"])
        self.assertEqual(lines[0]["errors"], [{"code": "nothing", "message": "nothing to send"}])
        self.assertEqual(lines[0]["run_id"], "r1")
        self.assertEqual(lines[0]["seq"], 1)

    def test_error_notify_lands_in_result_errors(self):
        server = self._server()
        self.controller.script = RunScript(mode="error")
        lines = self._submit_and_wait(server, prompt="hi")
        self.assertEqual(lines[-1]["type"], "result")
        self.assertIn("llm unreachable", [e["message"] for e in lines[-1]["errors"]])
        self.assertEqual([e["code"] for e in lines[-1]["errors"]], ["unknown"])

    def test_two_runs_carry_history(self):
        server = self._server()
        server.op_submit({"op": "submit", "prompt": "a", "run_id": "r1"})
        self._wait_idle(server)
        server.op_submit({"op": "submit", "prompt": "b", "run_id": "r2"})
        self._wait_idle(server)
        results = _of_type(_lines(server.out), "result")  # type: ignore[arg-type]
        self.assertEqual(results[0]["answer"], "answer to: a")
        self.assertEqual(results[1]["answer"], "answer to: b")
        self.assertEqual(self.controller.submits, ["a", "b"])

    def test_seq_restarts_per_run(self):
        server = self._server()
        server.op_submit({"op": "submit", "prompt": "a", "run_id": "r1"})
        self._wait_idle(server)
        server.op_submit({"op": "submit", "prompt": "b", "run_id": "r2"})
        self._wait_idle(server)
        lines = _lines(server.out)  # type: ignore[arg-type]
        starts = [line for line in lines if line["type"] == "start"]
        self.assertEqual([line["seq"] for line in starts], [1, 1])
        self.assertEqual([line["run_id"] for line in starts], ["r1", "r2"])


class TestAnswer(ServerTestBase):
    def _submit_with_ask(self, server: AgentServer) -> RunScript:
        """Start a run whose agent blocks on a Question via the view."""
        script = RunScript(mode="ask")
        self.controller.script = script
        server.op_submit({"op": "submit", "prompt": "q", "run_id": "r1"})
        self._wait_pending(server)
        return script

    def test_answer_resolves_pending_ask(self):
        server = self._server()
        script = self._submit_with_ask(server)
        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["blue"]})
        self._wait_idle(server)
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[-1]["type"], "result")
        ask_events = [
            line for line in lines if line.get("type") == "notify" and line.get("kind") == "ask"
        ]
        self.assertEqual(len(ask_events), 1)
        self.assertEqual(ask_events[0]["data"]["kind"], "ask")
        self.assertEqual(script.ask_answer, "blue")

    def test_blank_lines_ignored(self):
        server = self._server()
        server.inp.write("\n   \n" + json.dumps({"op": "ping"}) + "\n")
        server.inp.seek(0)
        server.serve_forever()
        self.assertEqual(
            [line["type"] for line in _lines(server.out)],  # type: ignore[arg-type]
            ["ready", "pong"],
        )

    def test_submit_with_non_string_prompt_rejected(self):
        server = self._server()
        server.op_submit({"op": "submit", "run_id": "r1", "prompt": 123})
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[0]["type"], "error")
        self.assertIn("prompt", lines[0]["error"]["message"])

    def test_op_handlers_routed_through_serve_forever(self):
        """answer/cancel ops dispatched by the main loop reach the run."""
        server = self._server()
        script = RunScript(mode="ask")
        self.controller.script = script

        def _host() -> None:
            # the reader thread processes ops sequentially; drive via the
            # in-memory stream once the run thread is up
            server.inp.write(json.dumps({"op": "submit", "prompt": "q", "run_id": "r1"}) + "\n")
            server.inp.seek(0)
            # serve_forever blocks on readline; use a fake stdin that
            # returns the pending-submit then answers once asked
            raise AssertionError("placeholder")  # pragma: no cover

        # Simpler: submit via op (spawns thread), then feed answer/cancel
        # lines through serve_forever on a thread with a pipe-like input.
        r, w = os.pipe()
        reader_fd = os.fdopen(r, "r")
        out = io.StringIO()
        server2 = AgentServer(self.session, reader_fd, out, err=io.StringIO())
        server2.controller = self.controller  # type: ignore[assignment]
        thread = threading.Thread(target=server2.serve_forever, daemon=True)
        thread.start()
        os.write(w, (json.dumps({"op": "submit", "prompt": "q", "run_id": "r1"}) + "\n").encode())
        deadline = time.time() + 5
        while time.time() < deadline:
            view = server2.view
            if view is not None:
                with view._pending_lock:
                    if view._pending is not None:
                        break
            time.sleep(0.01)
        else:
            self.fail("ask never became pending")
        os.write(
            w, (json.dumps({"op": "answer", "run_id": "r1", "answers": ["blue"]}) + "\n").encode()
        )
        deadline = time.time() + 5
        while time.time() < deadline:
            with server2._active_guard:
                if server2._active_run_id is None:
                    break
            time.sleep(0.01)
        else:
            self.fail("run never became idle")
        os.write(w, (json.dumps({"op": "shutdown"}) + "\n").encode())
        thread.join(timeout=5)
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        self.assertEqual(lines[-1]["type"], "result")
        self.assertEqual(script.ask_answer, "blue")
        os.close(w)
        reader_fd.close()
        del _host

    def test_answer_to_active_run_without_pending_question_rejected(self):
        """op_answer while the named run is active but nothing pending."""
        server = self._server()
        self.controller.gate = threading.Event()
        server.op_submit({"op": "submit", "prompt": "hi", "run_id": "r1"})
        self._wait_view(server)
        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["x"]})
        errors = [
            line
            for line in _lines(server.out)
            if line["type"] == "error"  # type: ignore[arg-type]
        ]
        self.assertEqual(len(errors), 1)
        self.assertIn("no pending question", errors[0]["error"]["message"])
        self.controller.gate.set()
        self._wait_idle(server)

    def test_answer_to_finished_run_rejected(self):
        """op_answer naming a run that already finished (stale host)."""
        server = self._server()
        self._submit_and_wait(server)
        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["x"]})
        errors = [
            line
            for line in _lines(server.out)
            if line["type"] == "error"  # type: ignore[arg-type]
        ]
        self.assertEqual(len(errors), 1)
        self.assertIn("no pending question", errors[0]["error"]["message"])

    def test_answer_multi_values_join(self):
        server = self._server()
        script = self._submit_with_ask(server)
        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["a", "b"]})
        self._wait_idle(server)
        self.assertEqual(script.ask_answer, "a, b")

    def test_answer_unknown_run_rejected(self):
        server = self._server()
        server.op_answer({"op": "answer", "run_id": "nope", "answers": ["x"]})
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[-1]["type"], "error")

    def test_answer_requires_answers_list(self):
        server = self._server()
        server.op_answer({"op": "answer", "run_id": "r1"})
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[-1]["type"], "error")
        self.assertIn("answers", lines[-1]["error"]["message"])

    def test_answer_with_no_pending_question_rejected(self):
        server = self._server()
        self._submit_and_wait(server)
        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["late"]})
        errors = [
            line
            for line in _lines(server.out)
            if line["type"] == "error"  # type: ignore[arg-type]
        ]
        self.assertEqual(len(errors), 1)
        self.assertIn("no pending question", errors[0]["error"]["message"])

    def test_answer_after_wrong_run_rejected_then_correct_resolves(self):
        server = self._server()
        self._submit_with_ask(server)
        server.op_answer({"op": "answer", "run_id": "r2", "answers": ["x"]})
        errors = [
            line
            for line in _lines(server.out)
            if line["type"] == "error"  # type: ignore[arg-type]
        ]
        self.assertEqual(len(errors), 1)
        # the correct run_id still resolves the still-pending question
        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["blue"]})
        self._wait_idle(server)

    def test_confirm_answer_routes_to_confirm_fn(self):
        server = self._server()
        script = RunScript(mode="confirm")
        self.controller.script = script
        server.op_submit({"op": "submit", "prompt": "plan", "run_id": "r1"})
        self._wait_pending(server)
        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["y"]})
        self._wait_idle(server)
        self.assertTrue(script.confirm_answer)
        result = _of_type(_lines(server.out), "result")[-1]  # type: ignore[arg-type]
        self.assertFalse(result["cancelled"])


class TestCancel(ServerTestBase):
    def setUp(self) -> None:
        self._server()  # populates self.session / self.controller

    def test_cancel_unknown_run_rejected(self):
        server = self._server()
        server.op_cancel({"op": "cancel", "run_id": "nope"})
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[-1]["type"], "error")
        self.assertIn("not active", lines[-1]["error"]["message"])

    def test_cancel_flags_session_and_unblocks_worker(self):
        server = self._server()
        script = RunScript(mode="ask")
        self.controller.script = script
        server.op_submit({"op": "submit", "prompt": "q", "run_id": "r1"})
        self._wait_pending(server)
        server.op_cancel({"op": "cancel", "run_id": "r1"})
        self.assertEqual(self.session.cancel_calls, 1)
        self._wait_idle(server)
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[-1]["type"], "result")
        self.assertTrue(lines[-1]["cancelled"])
        # the worker's pending ask was unblocked with "" (cancel parity)
        self.assertEqual(script.ask_answer, "")

    def test_execute_run_crash_yields_error_line_and_releases(self):
        """A crash inside _execute_run itself (not the worker) must not
        wedge the server: an error line is written and the slot frees."""
        server = self._server()

        def boom(prompt: str, run_id: str) -> None:
            raise RuntimeError("execute exploded")

        server._execute_run = boom  # type: ignore[method-assign]
        server.op_submit({"op": "submit", "prompt": "hi", "run_id": "r1"})
        self._wait_idle(server)
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[-1]["type"], "error")
        self.assertIn("execute exploded", lines[-1]["error"]["message"])
        self.assertEqual(lines[-1]["error"]["code"], "protocol")
        # the server accepts a new run afterwards
        server.op_submit({"op": "submit", "prompt": "next", "run_id": "r2"})
        self._wait_idle(server)

    def test_cancel_op_routed_through_main_loop(self):
        """The reader loop dispatches a cancel op while the run is live."""
        r, w = os.pipe()
        reader_fd = os.fdopen(r, "r")
        out = io.StringIO()
        server = AgentServer(self.session, reader_fd, out, err=io.StringIO())
        server.controller = self.controller  # type: ignore[assignment]
        script = RunScript(mode="ask")
        self.controller.script = script
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        os.write(w, (json.dumps({"op": "submit", "prompt": "q", "run_id": "r1"}) + "\n").encode())
        deadline = time.time() + 5
        while time.time() < deadline:
            view = server.view
            if view is not None:
                with view._pending_lock:
                    if view._pending is not None:
                        break
            time.sleep(0.01)
        else:
            self.fail("ask never became pending")
        os.write(w, (json.dumps({"op": "cancel", "run_id": "r1"}) + "\n").encode())
        deadline = time.time() + 5
        while time.time() < deadline:
            with server._active_guard:
                if server._active_run_id is None:
                    break
            time.sleep(0.01)
        else:
            self.fail("run never became idle")
        os.write(w, (json.dumps({"op": "shutdown"}) + "\n").encode())
        thread.join(timeout=5)
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        result = lines[-1]
        self.assertEqual(result["type"], "result")
        self.assertTrue(result["cancelled"])
        self.assertEqual(self.session.cancel_calls, 1)
        os.close(w)
        reader_fd.close()

    def test_cancel_racing_run_start_is_not_swallowed(self):
        """A cancel op landing between op_submit and Controller.submit
        must survive submit()'s cancel_event.clear() — the run thread
        re-applies it after submit (verified race without the fix)."""
        server = self._server()
        script = RunScript(mode="ask")
        self.controller.script = script
        # make submit() simulate the race: the cancel op fires while the
        # run thread is between op_submit's guard-set and submit()
        original = self.controller.submit

        def racing_submit(prompt, **kwargs):
            # reader thread processes op_cancel right here
            server.op_cancel({"op": "cancel", "run_id": "r1"})
            return original(prompt, **kwargs)

        self.controller.submit = racing_submit  # type: ignore[method-assign]
        server.op_submit({"op": "submit", "prompt": "q", "run_id": "r1"})
        self._wait_idle(server, timeout=10)
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[-1]["type"], "result")
        self.assertTrue(lines[-1]["cancelled"])
        # once from op_cancel, once re-applied by the run thread after
        # submit cleared the event
        self.assertEqual(self.session.cancel_calls, 2)

    def test_shutdown_during_active_run_cancels_and_waits(self):
        """EOF/shutdown with a run still active: cancel it and wait for
        the run thread to unwind before the loop returns."""
        r, w = os.pipe()
        reader_fd = os.fdopen(r, "r")
        out = io.StringIO()
        server = AgentServer(self.session, reader_fd, out, err=io.StringIO())
        server.controller = self.controller  # type: ignore[assignment]
        script = RunScript(mode="ask")
        self.controller.script = script
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        os.write(w, (json.dumps({"op": "submit", "prompt": "q", "run_id": "r1"}) + "\n").encode())
        deadline = time.time() + 5
        while time.time() < deadline:
            view = server.view
            if view is not None:
                with view._pending_lock:
                    if view._pending is not None:
                        break
            time.sleep(0.01)
        else:
            self.fail("ask never became pending")
        os.write(w, (json.dumps({"op": "shutdown"}) + "\n").encode())
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.session.cancel_calls, 1)
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        self.assertEqual(lines[-1]["type"], "result")
        self.assertTrue(lines[-1]["cancelled"])
        os.close(w)
        reader_fd.close()

    def test_close_cancels_pending_ask(self):
        """AgentServer.close unblocks a wedged interactive prompt."""
        out = io.StringIO()
        server = AgentServer(self.session, io.StringIO(), out, err=io.StringIO())
        view = ServerView(out=io.StringIO(), err=io.StringIO(), run_id="r1")
        server.view = view
        result: dict = {}

        def _ask() -> None:
            result["answer"] = view.ask([{"question": "q?"}])

        thread = threading.Thread(target=_ask, daemon=True)
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            with view._pending_lock:
                if view._pending is not None:
                    break
            time.sleep(0.01)
        server.close()
        thread.join(timeout=5)
        self.assertEqual(result["answer"], "")

    def test_cancel_without_pending_ask_still_flags(self):
        server = self._server()
        self.controller.gate = threading.Event()
        server.op_submit({"op": "submit", "prompt": "hi", "run_id": "r1"})
        self._wait_view(server)
        server.op_cancel({"op": "cancel", "run_id": "r1"})
        self.assertEqual(self.session.cancel_calls, 1)
        self.controller.gate.set()
        self._wait_idle(server)
        result = _of_type(_lines(server.out), "result")[-1]  # type: ignore[arg-type]
        self.assertTrue(result["cancelled"])


class TestServerView(unittest.TestCase):
    def _wait_pending(self, view: ServerView, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with view._pending_lock:
                if view._pending is not None:
                    return view._pending
            time.sleep(0.01)
        self.fail("pending never appeared")

    def test_wait_answer_cancel_before_event(self):
        """Cancel while waiting, event never set: the loop's own
        cancelled check (not the event branch) returns the empty answer."""
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1")
        ask_state = _AskState("r1", "ask")
        result: dict = {}

        def _wait() -> None:
            result["answer"] = view._wait_answer(ask_state)

        thread = threading.Thread(target=_wait, daemon=True)
        thread.start()
        time.sleep(0.05)
        view._cancelled.set()
        thread.join(timeout=5)
        self.assertEqual(result["answer"], "")

    def test_wait_answer_cancel_racing_event(self):
        """Cancellation lands while the waiter sits inside event.wait().

        Direct race injection: the event fires first but the cancelled
        flag is set before the waiter re-checks it — the empty answer
        must win (TUI parity).
        """
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1")
        ask_state = _AskState("r1", "ask")
        result: dict = {}

        def _wait() -> None:
            result["answer"] = view._wait_answer(ask_state)

        thread = threading.Thread(target=_wait, daemon=True)
        thread.start()
        time.sleep(0.05)  # let the waiter enter event.wait()
        ask_state.event.set()
        view._cancelled.set()
        thread.join(timeout=5)
        self.assertEqual(result["answer"], "")

    def test_final_answer_skips_non_assistant_and_strips_final_check(self):
        """The result answer takes the LAST assistant message, TUI-filtered."""
        server = ServerTestBase._server(self)
        m1 = mock.Mock(role="user")
        m2 = mock.Mock(role="assistant")
        m2.text_without_reasoning.return_value = (
            "real answer\n\n[FINAL CHECK]\n- Goal: g\n- Status: SUCCESS\n- Evidence: e"
        )
        m3 = mock.Mock(role="tool")
        self.session.last_messages = [m1, m2, m3]
        self.assertEqual(server._final_answer(), "real answer")
        m4 = mock.Mock(role="assistant")
        m4.text_without_reasoning.return_value = "  "
        self.session.last_messages = [m4]
        self.assertEqual(server._final_answer(), "")
        m5 = mock.Mock(role="assistant")
        m5.text_without_reasoning.return_value = "real answer"
        m6 = mock.Mock(role="assistant")
        m6.text_without_reasoning.return_value = (
            "[FINAL CHECK]\n- Goal: g\n- Status: SUCCESS\n- Evidence: e"
        )
        self.session.last_messages = [m5, m6]
        self.assertEqual(server._final_answer(), "real answer")

    def test_usage_snapshot_shape(self):
        """Non-numeric/bool usage entries are ignored; missing dict -> None."""
        server = ServerTestBase._server(self)
        self.session.usage_totals = {"input": "x", "output": True, "rounds": 2.9}
        self.assertEqual(server._usage_snapshot(), {"input": 0, "output": 0, "rounds": 2})
        self.session.usage_totals = None
        self.assertIsNone(server._usage_snapshot())

    def test_confirm_true_and_false(self):
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1")
        self.assertIsNone(view.take_pending())
        result: dict = {}

        def _confirm() -> None:
            result["yes"] = view.confirm("proceed?")

        thread = threading.Thread(target=_confirm, daemon=True)
        thread.start()
        pending = self._wait_pending(view)
        pending.answer = "y"
        pending.event.set()
        thread.join(timeout=5)
        self.assertTrue(result["yes"])

        thread2 = threading.Thread(
            target=lambda: result.setdefault("no", view.confirm("again?")), daemon=True
        )
        thread2.start()
        pending2 = self._wait_pending(view)
        pending2.answer = "n"
        pending2.event.set()
        thread2.join(timeout=5)
        self.assertFalse(result["no"])

    def test_confirm_defaults_false_when_unanswered(self):
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1")
        result: dict = {}

        def _confirm() -> None:
            result["ok"] = view.confirm("proceed?")

        thread = threading.Thread(target=_confirm, daemon=True)
        thread.start()
        pending = self._wait_pending(view)
        pending.answer = "Unanswered"
        pending.event.set()
        thread.join(timeout=5)
        self.assertFalse(result["ok"])

    def test_confirm_timeout_falls_back_false(self):
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1", answer_timeout=0.2)
        self.assertFalse(view.confirm("proceed?"))

    def test_ask_multi_answer_joins(self):
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1")
        result: dict = {}

        def _ask() -> None:
            result["answer"] = view.ask([{"question": "q?"}])

        thread = threading.Thread(target=_ask, daemon=True)
        thread.start()
        self._wait_pending(view)
        view.answer(["a", "b"])
        thread.join(timeout=5)
        self.assertEqual(result["answer"], "a, b")

    def test_ask_single_answer(self):
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1")
        result: dict = {}

        def _ask() -> None:
            result["answer"] = view.ask([{"question": "q?"}])

        thread = threading.Thread(target=_ask, daemon=True)
        thread.start()
        self._wait_pending(view)
        view.answer(["solo"])
        thread.join(timeout=5)
        self.assertEqual(result["answer"], "solo")

    def test_ask_cancel_unblocks_empty(self):
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1")
        result: dict = {}

        def _ask() -> None:
            result["answer"] = view.ask([{"question": "q?"}])

        thread = threading.Thread(target=_ask, daemon=True)
        thread.start()
        self._wait_pending(view)
        view.cancel_pending()
        thread.join(timeout=5)
        self.assertEqual(result["answer"], "")

    def test_ask_timeout_falls_back_unanswered(self):
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1", answer_timeout=0.2)
        self.assertEqual(view.ask([{"question": "q?"}]), "Unanswered")

    def test_reset_for_run_renumbers(self):
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1")
        view.emit_start("p1", [])
        view.on_delta("x")
        view.reset_for_run("r2")
        # events before the next emit_start are buffered (pre-start
        # contract), so arm run 2 before emitting again
        view.emit_start("p2", [])
        view.on_delta("y")
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
        self.assertEqual([line["seq"] for line in lines], [1, 2, 1, 2])
        self.assertEqual([line.get("run_id") for line in lines], ["r1", "r1", "r2", "r2"])
        self.assertEqual(lines[0]["prompt"], "p1")
        self.assertEqual(lines[2]["prompt"], "p2")

    def test_answer_without_pending_is_noop(self):
        out = io.StringIO()
        view = ServerView(out=out, err=io.StringIO(), run_id="r1")
        self.assertFalse(view.answer(["x"]))


class TestRunServe(unittest.TestCase):
    def test_run_serve_sequential_runs_via_ops(self):
        """Two sequential runs through the op path: history carries, and
        each run's lines are numbered from 1 under its own run_id (the
        multi-turn memory the resident server exists to provide)."""
        server = ServerTestBase._server(self)
        for prompt, run_id in (("one", "r1"), ("two", "r2")):
            server.op_submit({"op": "submit", "prompt": prompt, "run_id": run_id})
            ServerTestBase._wait_idle(self, server)
        lines = _lines(server.out)  # type: ignore[arg-type]
        results = _of_type(lines, "result")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["answer"], "answer to: one")
        self.assertEqual(results[1]["answer"], "answer to: two")
        self.assertEqual(results[0]["run_id"], "r1")
        self.assertEqual(results[1]["run_id"], "r2")
        starts = [line for line in lines if line["type"] == "start"]
        self.assertEqual([line["seq"] for line in starts], [1, 1])

    def test_run_serve_shutdown_only(self):
        session = FakeSession()
        controller = FakeController(session)
        inp = io.StringIO(json.dumps({"op": "shutdown"}) + "\n")
        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.server.Controller", lambda s: controller):
            rc = run_serve(session, inp=inp, out=out, err=io.StringIO())
        self.assertEqual(rc, 0)

    def test_run_serve_eof_only(self):
        session = FakeSession()
        controller = FakeController(session)
        out = io.StringIO()
        with mock.patch("python_agent_harness.entry.server.Controller", lambda s: controller):
            rc = run_serve(session, inp=io.StringIO(""), out=out, err=io.StringIO())
        self.assertEqual(rc, 0)
        self.assertEqual(_lines(out)[0]["type"], "ready")


class StragglerFencingTests(ServerTestBase):
    """A run's worker can outlive its ``result`` line.

    Such a straggler (a detached sub-agent thread, a tool unwinding
    after a cancel) must not append to its own finished stream, and
    must never be attributed to a LATER run.
    """

    def test_run_thread_is_daemon(self):
        """The run thread must not keep the interpreter alive at exit.

        ``_join_run_thread`` bounds how long shutdown waits for a run
        that ignores cancel; a non-daemon thread would additionally
        block interpreter exit past that budget, so ``shutdown`` could
        never terminate the process.
        """
        server = self._server()
        gate = threading.Event()
        self.controller.gate = gate
        server.op_submit({"op": "submit", "prompt": "p", "run_id": "r1"})
        run_threads = [t for t in threading.enumerate() if t.name.startswith("serve-run-")]
        self.assertEqual(len(run_threads), 1)
        self.assertTrue(
            run_threads[0].daemon,
            "serve run thread must be a daemon or `shutdown` cannot exit the process",
        )
        gate.set()
        self._wait_idle(server)

    def test_no_line_after_result(self):
        """A straggler must not append behind the terminal result line."""
        server = self._server()
        server.op_submit({"op": "submit", "prompt": "p", "run_id": "r1"})
        self._wait_idle(server)
        view = server.controller.view or self._last_view
        # The run is over; the straggler still holds its bound callback.
        view.on_notify("tool", "straggler")
        view.on_log("late log")
        lines = _lines(server.out)
        self.assertEqual(lines[-1]["type"], "result", "result must stay the terminal line")
        self.assertNotIn("straggler", server.out.getvalue())
        self.assertNotIn("late log", server.out.getvalue())

    def test_straggler_not_attributed_to_next_run(self):
        """Run 1's straggler must not land in run 2's stream.

        ``attach_view`` repoints ``session.notify_fn`` per run and
        ``Session.notify`` resolves it at call time, so without the
        detach a late emission from run 1 would be stamped with run
        2's run_id/seq — the user would see the previous turn's tool
        output inside the current turn.
        """
        server = self._server()
        server.op_submit({"op": "submit", "prompt": "one", "run_id": "RUN-1"})
        self._wait_idle(server)
        run1_view = self._capture_view(server)

        server.op_submit({"op": "submit", "prompt": "two", "run_id": "RUN-2"})
        self._wait_idle(server)

        # Run 1's straggler emits through the view it captured, and
        # through the session (whose callbacks run 2 had repointed).
        run1_view.on_notify("tool", "STRAGGLER")
        self.session.notify("tool", "STRAGGLER-VIA-SESSION")

        lines = _lines(server.out)
        self.assertNotIn("STRAGGLER", server.out.getvalue())
        for line in lines:
            if line.get("run_id") == "RUN-2":
                self.assertNotIn("STRAGGLER", json.dumps(line))
        # Both runs still produced exactly one clean terminal line.
        results = _of_type(lines, "result")
        self.assertEqual([r["run_id"] for r in results], ["RUN-1", "RUN-2"])

    def test_detach_unwires_session_callbacks(self):
        """After a run, nothing is attached to emit into."""
        server = self._server()
        server.op_submit({"op": "submit", "prompt": "p", "run_id": "r1"})
        self._wait_idle(server)
        self.assertIsNone(self.session.notify_fn)
        self.assertIsNone(self.session.log_fn)
        self.assertIsNone(self.session.on_delta)
        self.assertIsNone(self.session.ask_fn)
        self.assertIsNone(self.session.confirm_fn)

    def test_detach_failure_does_not_wedge_the_server(self):
        """A raise in teardown must not strand the active-run id.

        ``_active_run_id`` gates every submit; if a detach failure
        escaped the finally the server would reject all later runs
        with "a run is already active" forever.
        """
        server = self._server()

        def _boom() -> None:
            raise RuntimeError("detach exploded")

        self.controller.detach_view = _boom  # type: ignore[assignment]
        server.op_submit({"op": "submit", "prompt": "one", "run_id": "r1"})
        self._wait_idle(server)
        # The next run must still be accepted.
        server.op_submit({"op": "submit", "prompt": "two", "run_id": "r2"})
        self._wait_idle(server)
        results = _of_type(_lines(server.out), "result")
        self.assertEqual([r["run_id"] for r in results], ["r1", "r2"])

    # -- helpers ---------------------------------------------------------

    def _capture_view(self, server: AgentServer):
        """The view of the run that just finished (detach clears it)."""
        return self._last_view

    def setUp(self) -> None:
        self._last_view = None

    def _server(self, *args, **kwargs):  # type: ignore[override]
        server = super()._server(*args, **kwargs)
        attach = self.controller.attach_view

        def _tracking_attach(view):
            self._last_view = view
            attach(view)

        self.controller.attach_view = _tracking_attach  # type: ignore[assignment]
        return server


class SignalShutdownTests(ServerTestBase):
    """SIGINT/SIGTERM must drain the server, not kill it mid-run.

    ``serve`` is resident, so the default handler used to kill the
    process before the active run could emit its terminal ``result``
    line — losing the partial answer, the salvaged history, and the
    token usage a driving host bills on.
    """

    def test_handler_flags_stop_and_breaks_the_blocking_read(self):
        """A flag alone is not enough: readline is retried (PEP 475)."""
        server = self._server()
        guard = graceful_signal_shutdown(server)
        with self.assertRaises(_ShutdownSignal):
            guard._handle(signal.SIGTERM, None)
        self.assertTrue(server._stopped.is_set())

    def test_handlers_installed_and_restored(self):
        server = self._server()
        before = signal.getsignal(signal.SIGTERM)
        with graceful_signal_shutdown(server):
            self.assertIsNot(signal.getsignal(signal.SIGTERM), before)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_handler_is_one_shot(self):
        """A second signal takes its default action, not another drain."""
        server = self._server()
        original = signal.getsignal(signal.SIGTERM)
        try:
            with graceful_signal_shutdown(server) as guard:
                installed = signal.getsignal(signal.SIGTERM)
                with self.assertRaises(_ShutdownSignal):
                    guard._handle(signal.SIGTERM, None)
                self.assertIsNot(
                    signal.getsignal(signal.SIGTERM),
                    installed,
                    "handler must uninstall itself before raising",
                )
        finally:
            signal.signal(signal.SIGTERM, original)

    def test_off_main_thread_degrades_to_noop(self):
        """Only the main thread may install handlers (ValueError)."""
        server = self._server()
        failures: list = []

        def body() -> None:
            try:
                with graceful_signal_shutdown(server):
                    pass
            except Exception as e:  # noqa: BLE001 - recorded for the assert
                failures.append(e)

        thread = threading.Thread(target=body)
        thread.start()
        thread.join(5)
        self.assertEqual(failures, [])

    def test_signal_drains_active_run_to_a_result_line(self):
        """The payload of the fix: a signal still yields ``result``.

        Without the drain the process died here and the run's result
        line — with ``cancelled`` and its usage — was never written.
        """
        server = self._server()
        self.controller.script = RunScript("ask")
        server.op_submit({"op": "submit", "prompt": "q", "run_id": "r1"})
        self._wait_pending(server)

        def _interrupted() -> None:
            raise _ShutdownSignal  # as the signal handler would

        server._read_ops = _interrupted  # type: ignore[method-assign]
        server.serve_forever()

        results = _of_type(_lines(server.out), "result")
        self.assertEqual(len(results), 1, "a drained run must still emit result")
        self.assertEqual(results[0]["run_id"], "r1")
        self.assertTrue(results[0]["cancelled"])
        self.assertIn("usage", results[0], "usage must survive for billing")
        self.assertEqual(_lines(server.out)[-1]["type"], "result")

    def test_tolerates_none_previous_handler(self):
        """``signal.signal`` returns None when the prior disposition
        was not installed from Python (C code, PID-1/supervisor
        setups — the containerised case).  Restoring that None raises
        TypeError, which would escape __exit__.
        """
        server = self._server()
        real = signal.signal

        def reports_none(sig, handler):
            real(sig, handler)
            return None

        try:
            with (
                mock.patch("signal.signal", reports_none),
                graceful_signal_shutdown(server),
            ):
                pass
        finally:
            real(signal.SIGINT, signal.default_int_handler)
            real(signal.SIGTERM, signal.SIG_DFL)

    def test_handler_raises_shutdown_even_if_restore_fails(self):
        """The raise is what the drain depends on.

        A failure while uninstalling must not replace _ShutdownSignal
        with an exception the loop does not expect — that would crash
        serve_forever and skip the drain entirely.
        """
        server = self._server()
        guard = graceful_signal_shutdown(server)
        guard._previous[signal.SIGTERM] = None  # restoring this raises TypeError
        with self.assertRaises(_ShutdownSignal):
            guard._handle(signal.SIGTERM, None)
        self.assertTrue(server._stopped.is_set())

    def test_restorable_handler_normalizes_none(self):
        self.assertIs(_restorable_handler(None), signal.SIG_DFL)
        self.assertIs(_restorable_handler(signal.SIG_IGN), signal.SIG_IGN)
        self.assertIs(_restorable_handler(signal.SIG_DFL), signal.SIG_DFL)
        handler = signal.getsignal(signal.SIGINT)
        self.assertIs(_restorable_handler(handler), handler)


class ProtocolDocTests(unittest.TestCase):
    """The module docstring is the protocol spec other hosts read."""

    @staticmethod
    def _code_string_literals() -> set[str]:
        """String literals in executable code (docstrings excluded)."""
        tree = ast.parse(inspect.getsource(server_module))
        docstrings = {
            ast.get_docstring(n, clean=False)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and ast.get_docstring(n, clean=False)
        }
        return {
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value not in docstrings
        }

    def test_no_phantom_ack_line(self):
        """``ack`` was documented but never emitted.

        A host written against the docstring would wait forever for a
        receipt that never arrives.
        """
        self.assertNotIn('"ack"', server_module.__doc__ or "")
        self.assertNotIn("ack", self._code_string_literals())

    def test_documented_control_lines_are_really_emitted(self):
        emitted = self._code_string_literals()
        for line_type in ("ready", "pong", "error"):
            self.assertIn(line_type, emitted)
            self.assertIn(line_type, server_module.__doc__ or "")


class ServeBudgetTests(ServerTestBase):
    """Per-run round/wall-clock budgets for the resident server.

    ``serve`` runs were unbounded: ``_execute_run`` called
    ``controller.submit(prompt)`` with no budget, and
    ``AgentLoop.__init__`` leaves ``max_rounds`` at None for a
    top-level run unless ``budget_top_level`` is set.  That is the mode
    that sandboxes untrusted code, so it needs a ceiling.
    """

    def test_budget_is_forwarded_to_every_submit(self):
        server = self._server(max_rounds=12, timeout=45.5)
        server.op_submit({"op": "submit", "prompt": "one", "run_id": "r1"})
        self._wait_idle(server)
        server.op_submit({"op": "submit", "prompt": "two", "run_id": "r2"})
        self._wait_idle(server)
        # Per-submit, not per-process: EVERY run carries the budget.
        self.assertEqual(
            self.controller.submit_kwargs,
            [
                {"max_rounds": 12, "timeout": 45.5},
                {"max_rounds": 12, "timeout": 45.5},
            ],
        )

    def test_unlimited_by_default(self):
        """Disabled by default — same as headless and the TUI."""
        server = self._server()
        server.op_submit({"op": "submit", "prompt": "p", "run_id": "r1"})
        self._wait_idle(server)
        self.assertEqual(self.controller.submit_kwargs, [{"max_rounds": None, "timeout": None}])

    def test_run_serve_passes_budget_through(self):
        session = FakeSession()
        controller = FakeController(session)
        inp = io.StringIO(json.dumps({"op": "submit", "prompt": "p", "run_id": "r1"}) + "\n")
        with mock.patch("python_agent_harness.entry.server.Controller", lambda s: controller):
            rc = run_serve(
                session,
                inp=inp,
                out=io.StringIO(),
                err=io.StringIO(),
                max_rounds=5,
                timeout=9.0,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(controller.submit_kwargs, [{"max_rounds": 5, "timeout": 9.0}])

    def test_tripped_budget_surfaces_as_a_typed_error(self):
        """A tripped budget must end the run normally, not kill it.

        The agent loop notifies an error and unwinds, so the result
        line still carries the structured code the host branches on
        plus the usage it bills.
        """
        server = self._server(max_rounds=1)
        self.controller.script = RunScript(
            "error", error_message="round budget exhausted (1 round)"
        )
        server.op_submit({"op": "submit", "prompt": "p", "run_id": "r1"})
        self._wait_idle(server)
        result = _of_type(_lines(server.out), "result")[-1]
        self.assertEqual([e["code"] for e in result["errors"]], ["budget"])
        self.assertIn("usage", result)
        self.assertFalse(result["cancelled"])


class ServeBudgetEndToEndTests(unittest.TestCase):
    """The budget must actually TRIP, not just be forwarded.

    Drives a REAL Session/Controller/AgentLoop through ``run_serve``
    with a model scripted to loop on tool calls forever.  The unit
    tests above assert plumbing against a controller double; this one
    asserts enforcement.
    """

    def _serve_once(self, max_rounds=None, timeout=None) -> list[dict]:
        from python_agent_harness.core.models import ToolCall
        from tests.support.agent_test_utils import RecordingSession

        session = RecordingSession()
        # A model that never stops calling tools: only a budget ends it.
        session.client.script = [
            ("", [ToolCall(id=str(i), name="Read", arguments='{"file_path": "/tmp/x.py"}')])
            for i in range(50)
        ]
        out = io.StringIO()
        # Drive op_submit directly rather than through run_serve: stdin
        # EOF makes serve_forever drain, and the drain cancels the
        # active run — which would end the run before any budget could
        # trip (and mask enforcement behind `cancelled`).
        server = AgentServer(session, io.StringIO(), out, err=io.StringIO())
        server.max_rounds = max_rounds
        server.timeout = timeout
        server.op_submit({"op": "submit", "prompt": "loop", "run_id": "r1"})
        deadline = time.time() + 30
        while time.time() < deadline:
            with server._active_guard:
                if server._active_run_id is None:
                    break
            time.sleep(0.01)
        else:
            self.fail("run never finished")
        self.assertFalse(session.cancel_event.is_set(), "run was cancelled, not budget-limited")
        return _lines(out)

    def test_round_budget_trips_and_reports_code_budget(self):
        lines = self._serve_once(max_rounds=2)
        result = _of_type(lines, "result")[-1]
        self.assertIn(
            "budget",
            [e["code"] for e in result["errors"]],
            f"expected a budget error, got {result['errors']}",
        )
        # The run ended normally: a terminal result line, not a crash,
        # and the usage the host bills on is present.
        self.assertEqual(lines[-1]["type"], "result")
        self.assertFalse(result["cancelled"])
        self.assertIn("usage", result)

    def test_wall_clock_budget_trips_and_reports_code_timeout(self):
        # 1 microsecond: the deadline is set at loop start, so it has
        # certainly passed by the first budget check (one LLM call plus
        # a tool execution later).  Deliberately NOT a "small" timeout
        # raced against the run finishing — the whole 50-round script
        # completes in ~40ms here, so a 10ms budget would be only a 4x
        # margin and could invert on a faster CI runner.
        lines = self._serve_once(timeout=1e-6)
        result = _of_type(lines, "result")[-1]
        self.assertIn(
            "timeout",
            [e["code"] for e in result["errors"]],
            f"expected a timeout error, got {result['errors']}",
        )
        self.assertEqual(lines[-1]["type"], "result")

    def test_no_budget_means_no_budget_error(self):
        """Default stays unlimited: the run ends on the model, not a cap."""
        lines = self._serve_once()
        result = _of_type(lines, "result")[-1]
        codes = [e["code"] for e in result["errors"]]
        self.assertNotIn("budget", codes)
        self.assertNotIn("timeout", codes)


class AskCorrelationTests(ServerTestBase):
    """``answer`` must name the question it answers.

    Without an id, ``answer`` could only target "whatever is pending".
    An ask that timed out stayed in the pending slot, so a reply the
    host sent for it was accepted — and once the agent published its
    NEXT question, that stale reply resolved THAT one instead.  A user
    could approve a production deploy and have the approval applied to
    an unrelated question.
    """

    def _view(self, answer_timeout: float = 0.0) -> ServerView:
        return ServerView(
            out=io.StringIO(), err=io.StringIO(), run_id="r1", answer_timeout=answer_timeout
        )

    def test_ask_line_carries_an_ask_id(self):
        view = self._view()
        out = view.out
        threading.Thread(target=lambda: view.ask([{"question": "q?"}]), daemon=True).start()
        deadline = time.time() + 5
        while time.time() < deadline and view.pending_ask_id() is None:
            time.sleep(0.01)
        ask_id = view.pending_ask_id()
        self.assertIsNotNone(ask_id)
        view.emit_start("p", [])
        payload = [
            line
            for line in _lines(out)
            if line.get("kind") == "ask"  # type: ignore[arg-type]
        ]
        self.assertEqual(payload[0]["data"]["ask_id"], ask_id)
        view.cancel_pending()

    def test_confirm_line_carries_an_ask_id(self):
        view = self._view()
        threading.Thread(target=lambda: view.confirm("ok?"), daemon=True).start()
        deadline = time.time() + 5
        while time.time() < deadline and view.pending_ask_id() is None:
            time.sleep(0.01)
        self.assertIsNotNone(view.pending_ask_id())
        view.cancel_pending()

    def test_matching_ask_id_resolves(self):
        view = self._view()
        got: dict = {}
        threading.Thread(
            target=lambda: got.setdefault("a", view.ask([{"question": "q?"}])), daemon=True
        ).start()
        deadline = time.time() + 5
        while time.time() < deadline and view.pending_ask_id() is None:
            time.sleep(0.01)
        self.assertTrue(view.answer(["yes"], ask_id=view.pending_ask_id()))
        deadline = time.time() + 5
        while time.time() < deadline and "a" not in got:
            time.sleep(0.01)
        self.assertEqual(got["a"], "yes")

    def test_mismatched_ask_id_is_refused(self):
        view = self._view()
        threading.Thread(target=lambda: view.ask([{"question": "q?"}]), daemon=True).start()
        deadline = time.time() + 5
        while time.time() < deadline and view.pending_ask_id() is None:
            time.sleep(0.01)
        self.assertFalse(view.answer(["yes"], ask_id="not-the-pending-one"))
        # Still waiting: the wrong answer was not applied.
        self.assertIsNotNone(view.pending_ask_id())
        view.cancel_pending()

    def test_answered_ask_leaves_the_slot_empty(self):
        """A resolved question must stop being 'pending'.

        Otherwise a second answer lands on a dead question and
        ``pending_ask_id`` reports an ask nobody waits on.
        """
        view = self._view()
        got: dict = {}
        threading.Thread(
            target=lambda: got.setdefault("a", view.ask([{"question": "q?"}])), daemon=True
        ).start()
        deadline = time.time() + 5
        while time.time() < deadline and view.pending_ask_id() is None:
            time.sleep(0.01)
        view.answer(["yes"], ask_id=view.pending_ask_id())
        deadline = time.time() + 5
        while time.time() < deadline and "a" not in got:
            time.sleep(0.01)
        self.assertIsNone(view.pending_ask_id())
        self.assertFalse(view.answer(["again"]))

    def test_timed_out_ask_cannot_be_resolved_later(self):
        """The wrong-question bug, end to end.

        Q1 times out; the agent asks Q2; the host's late reply for Q1
        arrives.  It must not become Q2's answer.
        """
        view = self._view(answer_timeout=0.3)
        got: dict = {}

        def agent() -> None:
            got["q1"] = view.ask([{"question": "Deploy to PRODUCTION?"}])
            got["q2"] = view.ask([{"question": "Pick a colour"}])

        thread = threading.Thread(target=agent, daemon=True)
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline and view.pending_ask_id() is None:
            time.sleep(0.01)
        q1_id = view.pending_ask_id()
        # Let Q1 time out and Q2 take its place.
        deadline = time.time() + 5
        while time.time() < deadline and "q1" not in got:
            time.sleep(0.01)
        self.assertEqual(got["q1"], "Unanswered")
        # The late reply for Q1 must be refused, not applied to Q2.
        self.assertFalse(view.answer(["yes"], ask_id=q1_id))
        view.cancel_pending()
        thread.join(5)
        self.assertNotEqual(got.get("q2"), "yes")

    def test_id_less_answer_refused_once_an_ask_was_abandoned(self):
        """An uncorrelated answer cannot be disambiguated, so refuse it.

        Q1 times out and Q2 takes its place while the run is still
        live.  An id-less reply could be meant for either, so it must
        be rejected rather than silently applied to Q2.

        Only reachable with a non-zero answer_timeout; with the default
        (wait forever) nothing is ever abandoned and an id-less answer
        stays acceptable, so an older host keeps working.
        """
        server = self._server(answer_timeout=0.3)
        self.controller.script = RunScript("ask_twice")
        server.op_submit({"op": "submit", "prompt": "q", "run_id": "r1"})
        self._wait_pending(server)
        view = server.view
        assert view is not None
        self.assertFalse(view.requires_ask_id())
        # Q1 times out; Q2 becomes the pending question.
        deadline = time.time() + 5
        while time.time() < deadline and not view.requires_ask_id():
            time.sleep(0.01)
        self.assertTrue(view.requires_ask_id())

        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["yes"]})
        errors = _of_type(_lines(server.out), "error")  # type: ignore[arg-type]
        self.assertIn("ambiguous", errors[-1]["error"]["message"])
        self.assertIn("ask_id", errors[-1]["error"]["message"])
        # The refused answer reached neither question.
        server.op_cancel({"op": "cancel", "run_id": "r1"})
        self._wait_idle(server)
        self.assertNotEqual(self.controller.script.ask_answer, "yes")
        self.assertNotEqual(self.controller.script.ask_answer_2, "yes")

    def test_op_answer_reports_a_stale_ask_id_distinctly(self):
        """A stale answer and "nothing pending" are different faults.

        The host needs to tell "my reply was for a question that has
        moved on" from "there is no question at all" — the first means
        re-read the current ask, the second means stop replying.
        """
        server = self._server()
        self.controller.script = RunScript("ask")
        server.op_submit({"op": "submit", "prompt": "q", "run_id": "r1"})
        self._wait_pending(server)
        view = server.view
        assert view is not None
        pending = view.pending_ask_id()

        server.op_answer(
            {"op": "answer", "run_id": "r1", "answers": ["yes"], "ask_id": "some-other-id"}
        )
        message = _of_type(_lines(server.out), "error")[-1]["error"]["message"]  # type: ignore[arg-type]
        self.assertIn("stale answer", message)
        self.assertIn("some-other-id", message)
        self.assertIn(str(pending), message)
        # The real question is untouched and still answerable.
        self.assertEqual(view.pending_ask_id(), pending)
        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["yes"], "ask_id": pending})
        self._wait_idle(server)
        self.assertEqual(self.controller.script.ask_answer, "yes")

    def test_id_less_answer_still_works_by_default(self):
        """Back-compat: a host that predates ask_id keeps working."""
        server = self._server()
        self.controller.script = RunScript("ask")
        server.op_submit({"op": "submit", "prompt": "q", "run_id": "r1"})
        self._wait_pending(server)
        server.op_answer({"op": "answer", "run_id": "r1", "answers": ["yes"]})
        self._wait_idle(server)
        self.assertEqual(self.controller.script.ask_answer, "yes")
        self.assertEqual(_of_type(_lines(server.out), "error"), [])  # type: ignore[arg-type]


class CapabilityTests(ServerTestBase):
    """``ready`` advertises what this build supports.

    ``protocol_version`` is a single integer a host can only accept or
    reject wholesale; it cannot say "does this build support mid-run
    answers?".  Without a capability list a host has to infer a
    feature's absence from events that never arrive.
    """

    def test_ready_lists_capabilities(self):
        server = self._server()
        server.serve_forever()
        ready = _lines(server.out)[0]  # type: ignore[arg-type]
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["capabilities"], list(CAPABILITIES))
        self.assertIn("protocol_version", ready)

    def test_capabilities_cover_the_implemented_ops(self):
        """Every op the loop dispatches must be advertised."""
        for op in ("submit", "answer", "cancel", "ping", "shutdown"):
            self.assertIn(op, CAPABILITIES)

    def test_capabilities_are_unique(self):
        self.assertEqual(len(set(CAPABILITIES)), len(CAPABILITIES))


class IncrementalUsageTests(unittest.TestCase):
    """Usage must be observable DURING a run, not only at the end.

    ``result.usage`` arrives when the run is already over, so a host
    could bill but never stop a run that outlives its user's
    allowance.  A per-round ``notify`` of kind ``usage`` lets it meter
    live and cancel.
    """

    def _run(self, rounds: int = 3):
        from python_agent_harness.core.agent import run_agent_loop
        from python_agent_harness.core.models import Message, ToolCall
        from tests.support.agent_test_utils import RecordingSession

        session = RecordingSession()
        session.client.script = [
            ("", [ToolCall(id=str(i), name="Read", arguments='{"file_path": "/tmp/x.py"}')])
            for i in range(rounds)
        ] + [("done", None)]
        seen: list = []
        session.notify_fn = lambda kind, data=None: seen.append((kind, data))
        run_agent_loop(session, messages=[Message(role="user", content="hi")])
        return seen

    def test_usage_is_emitted_per_round(self):
        seen = self._run(rounds=3)
        usage = [data for kind, data in seen if kind == "usage"]
        self.assertGreaterEqual(len(usage), 3, f"expected a usage line per round, got {usage}")

    def test_usage_totals_are_cumulative_and_monotonic(self):
        seen = self._run(rounds=3)
        usage = [data for kind, data in seen if kind == "usage"]
        rounds = [u["rounds"] for u in usage]
        self.assertEqual(rounds, sorted(rounds))
        self.assertEqual(rounds, list(range(1, len(rounds) + 1)))
        inputs = [u["input"] for u in usage]
        self.assertEqual(inputs, sorted(inputs), "input tokens must never decrease")
        for u in usage:
            self.assertEqual(set(u), {"input", "output", "rounds"})

    def test_usage_snapshot_is_not_the_live_dict(self):
        """A snapshot, not a reference: a host must not see it mutate."""
        seen = self._run(rounds=2)
        usage = [data for kind, data in seen if kind == "usage"]
        self.assertNotEqual(usage[0]["rounds"], usage[-1]["rounds"])

    def test_subagent_rounds_do_not_emit_their_own_lines(self):
        """Sub-agent tokens are already inside the snapshot.

        Per-sub-agent lines would add noise, not information.
        """
        from python_agent_harness.core.agent import run_agent_loop
        from python_agent_harness.core.models import Message
        from tests.support.agent_test_utils import RecordingSession

        session = RecordingSession()
        session.client.script = [("sub done", None)]
        seen: list = []
        session.notify_fn = lambda kind, data=None: seen.append((kind, data))
        run_agent_loop(session, messages=[Message(role="user", content="hi")], top_level=False)
        self.assertEqual([d for k, d in seen if k == "usage"], [])

    def test_usage_line_does_not_corrupt_the_delta_stream(self):
        """A usage line between deltas must not break reconstruction.

        A host concatenates ``delta`` lines by type; the per-round
        ``usage`` line sits among them, so this checks the answer still
        reassembles and ``seq`` stays monotonic.
        """
        from python_agent_harness.entry.headless import JsonlView

        out = io.StringIO()
        view = JsonlView(out=out, err=io.StringIO(), run_id="r1")
        view.emit_start("p", [])
        view.on_delta("Hello ")
        view.on_notify("usage", {"input": 1, "output": 1, "rounds": 1})
        view.on_delta("world")
        view.on_notify("tool_start", ["Bash"])  # flushes the held delta tail
        view.emit_result("Hello world")
        lines = _lines(out)  # type: ignore[arg-type]
        answer = "".join(line["text"] for line in lines if line["type"] == "delta")
        self.assertEqual(answer, "Hello world")
        seqs = [line["seq"] for line in lines]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))


class RaiseSiteErrorCodeTests(unittest.TestCase):
    """Error codes belong to the code that knows the outcome.

    Deriving them downstream by matching words in the message means
    rewording a sentence silently reclassifies the failure to
    ``unknown`` — the message text is not an API, but was being used
    as one.
    """

    def test_raise_site_code_survives_rewording(self):
        from python_agent_harness.entry.headless import _as_error

        structured = _as_error({"code": "budget", "message": "Error: ran out of steps"})
        self.assertEqual(structured["code"], "budget")

    def test_heuristic_remains_for_plain_strings(self):
        """Back-compat: an un-migrated call site still classifies."""
        from python_agent_harness.entry.headless import _as_error

        self.assertEqual(_as_error("Error: round budget exhausted")["code"], "budget")
        self.assertEqual(_as_error("Error: wall-clock limit")["code"], "timeout")

    def test_reworded_string_without_a_code_degrades(self):
        """Exactly the fragility the raise-site code removes."""
        from python_agent_harness.entry.headless import _as_error

        self.assertEqual(_as_error("Error: ran out of steps")["code"], "unknown")

    def test_budget_trip_carries_its_code_to_the_result_line(self):
        from python_agent_harness.core.agent import AgentLoop
        from python_agent_harness.core.models import Message, ToolCall
        from tests.support.agent_test_utils import RecordingSession

        session = RecordingSession()
        session.client.script = [
            ("", [ToolCall(id=str(i), name="Read", arguments='{"file_path": "/tmp/x.py"}')])
            for i in range(5)
        ]
        seen: list = []
        session.notify_fn = lambda kind, data=None: seen.append((kind, data))
        AgentLoop(
            session,
            messages=[Message(role="user", content="hi")],
            max_rounds=2,
            budget_top_level=True,
        ).run()
        errors = [data for kind, data in seen if kind == "error"]
        self.assertEqual(errors[-1]["code"], "budget")
        self.assertIn("round budget", errors[-1]["message"])

    def test_structured_error_stays_human_readable(self):
        """Views must render the message, not a dict repr.

        The LSP client already notified ``{"message": ...}`` dicts, so
        this also fixes a pre-existing display regression.
        """
        from python_agent_harness.entry.headless import error_display_text

        self.assertEqual(
            error_display_text({"code": "budget", "message": "Error: out of rounds"}),
            "Error: out of rounds",
        )
        self.assertEqual(error_display_text({"message": "LSP server exited"}), "LSP server exited")
        self.assertEqual(error_display_text("plain string"), "plain string")
        # No message key: fall back to the code rather than a dict repr.
        self.assertEqual(error_display_text({"code": "budget"}), "budget")

    def test_headless_view_renders_structured_errors(self):
        from python_agent_harness.entry.headless import HeadlessView

        err = io.StringIO()
        view = HeadlessView(out=io.StringIO(), err=err)
        view.on_notify("error", {"code": "timeout", "message": "Error: too slow"})
        self.assertIn("[error: Error: too slow]", err.getvalue())
        self.assertNotIn("'code'", err.getvalue())
        self.assertEqual(view.errors, ["Error: too slow"])

    def test_jsonl_notify_error_wire_stays_a_string(self):
        """Backward compat: an already-deployed host renders notify
        data directly, so a structured error must NOT widen that field
        to a dict on the wire — the code travels on the result line."""
        from python_agent_harness.entry.headless import JsonlView

        out = io.StringIO()
        view = JsonlView(out=out, err=io.StringIO(), run_id="r1")
        view.emit_start("p", [])
        view.on_notify("error", {"code": "budget", "message": "Error: out of rounds"})
        notify = _of_type(_lines(out), "notify")[-1]  # type: ignore[arg-type]
        self.assertEqual(notify["data"], "Error: out of rounds")
        self.assertIsInstance(notify["data"], str)
        # ...but the canonical result line still carries the raise-site code.
        view.emit_result("", errors=None)
        result = _of_type(_lines(out), "result")[-1]  # type: ignore[arg-type]
        self.assertEqual([e["code"] for e in result["errors"]], ["budget"])
        self.assertEqual(result["error_messages"], ["Error: out of rounds"])

    def test_display_text_and_result_message_cannot_diverge(self):
        """The live message and the recorded message are one projection.

        ``error_display_text`` (wire/TUI) and ``_as_error`` (result
        line) must return the SAME message for any error value, or a
        user could see one thing live and another in the final record.
        Single-sourced, so this holds across every shape.
        """
        from python_agent_harness.entry.headless import _as_error, error_display_text

        for value in (
            {"code": "budget", "message": "Error: out of rounds"},
            {"code": "budget"},  # code-only
            {"error": {"code": "timeout", "message": "slow"}},  # nested
            {"message": "LSP server exited"},  # LSP shape
            "Error: a plain string",
            {"error": "flat error string"},
        ):
            self.assertEqual(
                error_display_text(value),
                _as_error(value)["message"],
                f"projections diverged for {value!r}",
            )


if __name__ == "__main__":
    unittest.main()


class TestHelloNegotiation(ServerTestBase):
    """``hello`` is the host's half of version negotiation.

    ``ready`` announces what this build speaks, which a host can only
    accept or reject wholesale.  ``hello`` lets the host state what IT
    can parse, so a mismatch is settled once, before any run, instead
    of surfacing as misparsed events.
    """

    def _hello(self, server: AgentServer, op: dict) -> list[dict]:
        server.inp = io.StringIO(json.dumps(op) + "\n")
        server._read_ops()
        return _lines(server.out)

    def test_hello_is_advertised_as_a_capability(self) -> None:
        self.assertIn("hello", CAPABILITIES)
        self.assertIn("op_id", CAPABILITIES)

    def test_a_shared_version_is_confirmed(self) -> None:
        server = self._server()
        lines = self._hello(server, {"op": "hello", "protocol_versions": [1, 2]})
        hello = _of_type(lines, "hello")
        self.assertEqual(len(hello), 1)
        self.assertEqual(hello[0]["protocol_version"], PROTOCOL_VERSION)
        self.assertIn("submit", hello[0]["capabilities"])
        self.assertEqual(server.host_protocol_versions, [1, 2])
        self.assertEqual(_of_type(lines, "error"), [])

    def test_no_shared_version_is_refused_before_any_run(self) -> None:
        server = self._server()
        lines = self._hello(server, {"op": "hello", "protocol_versions": [7, 9]})
        self.assertEqual(_of_type(lines, "hello"), [])
        errors = _of_type(lines, "error")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error"]["code"], "protocol")
        self.assertIn("no shared protocol version", errors[0]["message"])
        self.assertIn("[7, 9]", errors[0]["message"])
        self.assertEqual(server.host_protocol_versions, [7, 9])

    def test_a_scalar_version_is_accepted(self) -> None:
        server = self._server()
        lines = self._hello(server, {"op": "hello", "protocol_version": 1})
        self.assertEqual(len(_of_type(lines, "hello")), 1)

    def test_a_versionless_greeting_just_confirms(self) -> None:
        """A hello with nothing parseable is a greeting, not a
        negotiation: it must not be treated as "no overlap"."""
        server = self._server()
        lines = self._hello(server, {"op": "hello"})
        self.assertEqual(len(_of_type(lines, "hello")), 1)
        self.assertEqual(_of_type(lines, "error"), [])

    def test_garbled_versions_do_not_refuse(self) -> None:
        server = self._server()
        lines = self._hello(server, {"op": "hello", "protocol_versions": ["x", None]})
        self.assertEqual(len(_of_type(lines, "hello")), 1)

    def test_hello_echoes_the_op_id(self) -> None:
        server = self._server()
        lines = self._hello(server, {"op": "hello", "protocol_versions": [1], "op_id": "h1"})
        self.assertEqual(_of_type(lines, "hello")[0]["op_id"], "h1")


class TestOpIdAttribution(ServerTestBase):
    """A refusal must name the op it refused.

    There is no generic ack on this protocol, so an op's effect on the
    stream IS its acknowledgement -- but an effect that never happened
    cannot be correlated.  Echoing the caller's own id makes failures
    attributable without adding receipts to the success path.
    """

    def _drive(self, server: AgentServer, ops: list[dict]) -> list[dict]:
        server.inp = io.StringIO("".join(json.dumps(o) + "\n" for o in ops))
        server._read_ops()
        return _lines(server.out)

    def test_unknown_op_error_carries_the_op_id(self) -> None:
        server = self._server()
        lines = self._drive(server, [{"op": "teleport", "op_id": "a7"}])
        errors = _of_type(lines, "error")
        self.assertEqual(errors[0]["op_id"], "a7")
        self.assertIn("unknown op", errors[0]["message"])

    def test_submit_refusals_carry_the_op_id(self) -> None:
        server = self._server()
        lines = self._drive(server, [{"op": "submit", "op_id": "s1"}])
        self.assertEqual(_of_type(lines, "error")[0]["op_id"], "s1")

    def test_answer_refusal_carries_the_op_id(self) -> None:
        server = self._server()
        lines = self._drive(
            server, [{"op": "answer", "run_id": "r1", "answers": ["x"], "op_id": "ans-9"}]
        )
        self.assertEqual(_of_type(lines, "error")[0]["op_id"], "ans-9")

    def test_cancel_refusal_carries_the_op_id(self) -> None:
        server = self._server()
        lines = self._drive(server, [{"op": "cancel", "run_id": "nope", "op_id": "c3"}])
        self.assertEqual(_of_type(lines, "error")[0]["op_id"], "c3")

    def test_two_pipelined_refusals_are_distinguishable(self) -> None:
        """The point of the whole thing: without the echo both come back
        as bare errors and the host cannot tell them apart."""
        server = self._server()
        lines = self._drive(
            server,
            [
                {"op": "answer", "run_id": "r1", "answers": ["x"], "op_id": "the-answer"},
                {"op": "cancel", "run_id": "r1", "op_id": "the-cancel"},
            ],
        )
        ids = [e.get("op_id") for e in _of_type(lines, "error")]
        self.assertEqual(ids, ["the-answer", "the-cancel"])

    def test_ping_echoes_the_op_id(self) -> None:
        server = self._server()
        lines = self._drive(server, [{"op": "ping", "op_id": "p1"}])
        self.assertEqual(_of_type(lines, "pong")[0]["op_id"], "p1")

    def test_an_op_without_an_id_omits_the_field(self) -> None:
        """Wire-compatible with a host that does not correlate."""
        server = self._server()
        lines = self._drive(server, [{"op": "cancel", "run_id": "nope"}])
        errors = _of_type(lines, "error")
        self.assertNotIn("op_id", errors[0])
        lines2 = self._drive(server, [{"op": "ping"}])
        self.assertNotIn("op_id", _of_type(lines2, "pong")[0])
