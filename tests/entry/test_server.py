"""Tests for the resident agent server (entry/server.py)."""

from __future__ import annotations

import io
import json
import os
import threading
import time
import unittest
from unittest import mock

from python_agent_harness.entry.server import AgentServer, ServerView, _AskState, run_serve


def _lines(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines() if line]


def _of_type(lines: list[dict], type_: str) -> list[dict]:
    return [line for line in lines if line.get("type") == type_]


class FakeSession:
    """Minimal Session double: callbacks + cancel + usage, no agent loop."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self.model = "m-test"
        self.usage_totals = {"input": 1, "output": 2, "rounds": 3}
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

    def __init__(self, mode: str = "plain") -> None:
        self.mode = mode
        self.ask_answer: str | None = None
        self.confirm_answer: bool | None = None

    def drive(self, session: FakeSession, prompt: str) -> None:
        if self.mode == "raise":
            raise RuntimeError("boom")
        if self.mode == "error":
            session.notify("error", "llm unreachable")
        if self.mode == "ask":
            self.ask_answer = session.ask_questions([{"question": f"{prompt}?"}])
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
        self.script = RunScript()
        self.gate: threading.Event | None = None

    def attach_view(self, view) -> None:
        self.view = view
        session = self.session
        session.on_delta = view.on_delta
        session.notify_fn = view.on_notify
        session.log_fn = view.on_log
        session.confirm_fn = view.confirm
        session.ask_fn = view.ask

    def submit(self, prompt: str, **kwargs):
        self.submits.append(prompt)
        if self.gate is not None:
            self.gate.wait(5)
        script = self.script

        def _drive() -> None:
            try:
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
        self.assertIn("malformed", lines[1]["error"])

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
        self.assertIn("teleport", lines[1]["error"])


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

    def test_submit_requires_run_id(self):
        server = self._server()
        server.op_submit({"op": "submit", "prompt": "hi"})
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[0]["type"], "error")
        self.assertIn("run_id", lines[0]["error"])
        self.assertEqual(self.controller.submits, [])

    def test_submit_requires_prompt(self):
        server = self._server()
        server.op_submit({"op": "submit", "run_id": "r1"})
        lines = _lines(server.out)  # type: ignore[arg-type]
        self.assertEqual(lines[0]["type"], "error")
        self.assertIn("prompt", lines[0]["error"])

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
        self.assertIn("already active", errors[0]["error"])
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
        self.assertIn("run produced no answer", lines[-1]["errors"])
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
        self.assertEqual(lines[0]["errors"], ["nothing to send"])
        self.assertEqual(lines[0]["run_id"], "r1")
        self.assertEqual(lines[0]["seq"], 1)

    def test_error_notify_lands_in_result_errors(self):
        server = self._server()
        self.controller.script = RunScript(mode="error")
        lines = self._submit_and_wait(server, prompt="hi")
        self.assertEqual(lines[-1]["type"], "result")
        self.assertIn("llm unreachable", lines[-1]["errors"])

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
        self.assertIn("prompt", lines[0]["error"])

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
        self.assertIn("no pending question", errors[0]["error"])
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
        self.assertIn("no pending question", errors[0]["error"])

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
        self.assertIn("answers", lines[-1]["error"])

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
        self.assertIn("no pending question", errors[0]["error"])

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
        self.assertIn("not active", lines[-1]["error"])

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
        self.assertIn("execute exploded", lines[-1]["error"])
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


if __name__ == "__main__":
    unittest.main()
