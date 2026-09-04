"""P1 agent scenario tests: deterministic end-to-end agent-loop scenarios.

Each scenario drives a full AgentLoop with a scripted client/session (or a
real HTTP client against an in-process fake server) and asserts the
harness-level outcome: completion, tool-failure recovery, nudging, Ctrl-C
cancellation, stale-worker containment, compaction resume, sub-agent
containment, plan->build handoff, result sanitization, and HTTP retry
semantics.
"""

import contextlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
import session_sandbox  # noqa: F401,E402  (side-effect: redirect SESSION_DIR)

from python_agent_harness import config
from python_agent_harness.agent import AgentLoop
from python_agent_harness.client import Client
from python_agent_harness.models import Message, ToolCall, Usage
from python_agent_harness.persistence import SessionPersistence
from python_agent_harness.session import Session
from python_agent_harness.tools import default_registry

# `discover -s tests` puts the tests dir on sys.path, but a direct
# `-m unittest tests.test_scenarios` invocation does not — make the
# sibling helper importable either way.
sys.path.insert(0, os.path.dirname(__file__))

import fake_openai_server  # noqa: E402  (state overrides for sync tests)
from fake_openai_server import serve  # noqa: E402


class ScriptedClient:
    """Scripted chat responses: (assistant_text, tool_calls) per call."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = []
        self.chat_sync_calls = []

    def chat(
        self,
        messages,
        tools=None,
        system=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
        on_delta=None,
        stream=True,
        cancel_check=None,
        on_retry=None,
    ):
        self.calls.append([m.to_api() for m in messages])
        if not self.script:
            return Message(role="assistant", content="done"), Usage(input_tokens=100)
        item = self.script.pop(0)
        if isinstance(item, tuple):
            text, tool_calls = item
        else:
            text, tool_calls = item, None
        return Message(role="assistant", content=text, tool_calls=tool_calls), Usage(
            input_tokens=100
        )

    def chat_sync(
        self,
        messages,
        system=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
        cancel_check=None,
    ):
        self.chat_sync_calls.append([m.to_api() for m in messages])
        return Message(role="assistant", content="SYNC-OK"), Usage()


class BlockingClient(ScriptedClient):
    """Chat blocks until aborted (mimics a stalled HTTP stream)."""

    def __init__(self):
        super().__init__([])
        self.started = threading.Event()
        self.unblock = threading.Event()
        self.aborted = False

    def chat(self, *a, **k):
        self.started.set()
        self.unblock.wait(timeout=15)
        raise RuntimeError("aborted read")

    def abort(self):
        self.aborted = True
        self.unblock.set()


class ScenarioSession(Session):
    _test_session_dir = None

    def __init__(self, project_dir="/tmp/fakeproj"):
        if ScenarioSession._test_session_dir is None:
            import tempfile as _tf

            ScenarioSession._test_session_dir = _tf.mkdtemp(prefix="pah-scenarios-")
            import python_agent_harness.config as cfg

            cfg.SESSION_DIR = __import__("pathlib").Path(ScenarioSession._test_session_dir)
        super().__init__(
            project_dir=project_dir,
            client=ScriptedClient(),
            model="gpt-5-mini",
            registry=default_registry(),
        )
        self.executed = []
        self.store = SessionPersistence(
            project_dir=project_dir,
            model=self.model,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tool_names=self.store.tool_names,
        )

    def execute_tool(self, name, args, call_id=None):
        self.executed.append((name, args))
        if name in ("Agent", "PlanExit"):
            return Session.execute_tool(self, name, args, call_id=call_id)
        if name == "Read":
            return "file content"
        if name == "Bash":
            return "bash output"
        return f"result of {name}"


def agent_call(call_id, description, prompt="do it"):
    return ToolCall(
        id=call_id,
        name="Agent",
        arguments=json.dumps(
            {
                "subagent_type": "subagent",
                "description": description,
                "prompt": prompt,
            }
        ),
    )


def run_in_thread(loop):
    out = {}

    def target():
        try:
            out["r"] = loop.run()
        except BaseException as e:  # noqa: BLE001 - record for the assertion
            out["e"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=target)
    t.start()
    return t, out


def make_http_session(tmpdir):
    srv = serve()
    host, port = srv.server_address
    client = Client(
        base_url=f"http://{host}:{port}/v1",
        api_key="test",
        model="fake",
        retry_max=3,
        retry_base_delay=0.01,
        retry_max_delay=0.05,
    )
    session = ScenarioSession(project_dir=tmpdir)
    session.client = client
    session.stream = False
    session.tools_enabled = False
    session.store.title = "T"  # skip LLM title generation (extra request)
    return session, srv


class TestScenarioSimple(unittest.TestCase):
    def test_simple_task_done(self):
        """simple task -> DONE: one terminal response ends the run."""
        session = ScenarioSession()
        session.tools_enabled = False
        session.client.script = ["task complete"]
        loop = AgentLoop(session, messages=[Message(role="user", content="do the thing")])
        result = loop.run()
        self.assertEqual(result, "task complete")
        self.assertEqual(loop.state, AgentLoop.DONE)
        self.assertEqual(loop.history, [AgentLoop.WAIT, AgentLoop.SUPERVISE])


class TestScenarioToolFailure(unittest.TestCase):
    def test_tool_failure_then_recovery(self):
        """tool failure -> recovery: a crashing tool becomes an error
        result, the model gets it, and the next round completes."""
        session = ScenarioSession()
        session.tools_enabled = False

        def flaky(name, args, call_id=None):
            if name == "Bash":
                raise RuntimeError("boom")
            return "ok"

        session.execute_tool = flaky
        session.client.script = [
            ("", [ToolCall(id="1", name="Bash", arguments='{"command": "false"}')]),
            "recovered with a fallback plan",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="run risky command")])
        result = loop.run()
        self.assertEqual(result, "recovered with a fallback plan")
        rows = [m for m in loop.messages if m.role == "tool"]
        self.assertEqual(len(rows), 1)
        self.assertIn("crashed during execution", rows[0].text())
        self.assertIn("boom", rows[0].text())
        self.assertEqual(loop.state, AgentLoop.DONE)


class TestScenarioNudging(unittest.TestCase):
    def test_premature_completion_nudged_to_finish(self):
        """LLM premature completion -> nudge -> completion: a terminal
        answer on an agentic loop is intercepted once and nudged."""
        session = ScenarioSession()
        session.tools_enabled = True
        session.client.script = ["I'll get back to this", "here is the finished work"]
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 1):
            loop = AgentLoop(session, messages=[Message(role="user", content="implement feature")])
            result = loop.run()
        self.assertEqual(result, "here is the finished work")
        self.assertEqual(loop.supervisor.nudge_count, 1)
        nudges = [m for m in loop.messages if m.role == "user" and m.injected]
        self.assertEqual([m.text() for m in nudges], [config.NUDGE_MESSAGE])

    def test_two_nudges_then_fail_closed(self):
        """two nudges -> fail closed: the nudge budget (2) is spent and
        the third terminal answer ends the run."""
        session = ScenarioSession()
        session.tools_enabled = True
        session.client.script = ["a", "b", "c"]
        loop = AgentLoop(session, messages=[Message(role="user", content="task")])
        result = loop.run()
        self.assertEqual(result, "c")
        self.assertEqual(loop.supervisor.nudge_count, config.MAX_NUDGES)
        self.assertFalse(loop.supervisor.can_nudge())
        self.assertEqual(
            [m.text() for m in loop.messages if m.role == "user" and m.injected],
            [config.NUDGE_MESSAGE, config.NUDGE_MESSAGE],
        )

    def test_dead_session_fails_closed_without_nudging(self):
        """A dead session has no nudge budget: the first terminal answer
        ends the run immediately (never loops)."""
        session = ScenarioSession()
        session.tools_enabled = True
        session.alive = False
        session.client.script = ["first answer"]
        loop = AgentLoop(session, messages=[Message(role="user", content="task")])
        result = loop.run()
        self.assertEqual(result, "first answer")
        self.assertEqual(loop.supervisor.nudge_count, 0)


class TestScenarioCancellation(unittest.TestCase):
    def test_ctrl_c_during_llm_request(self):
        """Ctrl-C during the LLM request: the run stops cleanly (None,
        not an error), the client is aborted, and the partial history
        is salvaged."""
        session = ScenarioSession()
        session.tools_enabled = False
        blocker = BlockingClient()
        session.client = blocker
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        t, out = run_in_thread(loop)
        self.assertTrue(blocker.started.wait(timeout=5))
        time.sleep(0.2)
        session.cancel()
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertNotIn("e", out)
        self.assertIsNone(out.get("r"))
        self.assertTrue(blocker.aborted)
        self.assertEqual([m.role for m in session.last_messages], ["user"])
        self.assertEqual(session.last_messages[0].text(), "hi")

    def test_ctrl_c_during_bash(self):
        """Ctrl-C while a real Bash command runs: the process group is
        killed, the run stops promptly, and no tool result leaks into
        the conversation."""
        with tempfile.TemporaryDirectory(prefix="pah-scen-bash-") as tmpdir:
            session = ScenarioSession(project_dir=tmpdir)
            session.tools_enabled = False

            def real_execute(name, args, call_id=None):
                return Session.execute_tool(session, name, args, call_id=call_id)

            started = threading.Event()

            def tracking(name, args, call_id=None):
                started.set()
                return real_execute(name, args, call_id=call_id)

            session.execute_tool = tracking
            session.client.script = [
                (
                    "",
                    [ToolCall(id="b1", name="Bash", arguments=json.dumps({"command": "sleep 30"}))],
                ),
            ]
            loop = AgentLoop(session, messages=[Message(role="user", content="run")])
            t, out = run_in_thread(loop)
            self.assertTrue(started.wait(timeout=5))
            time.sleep(0.3)
            t0 = time.monotonic()
            session.cancel()
            t.join(timeout=10)
            elapsed = time.monotonic() - t0
            self.assertFalse(t.is_alive())
            self.assertNotIn("e", out)
            self.assertIsNone(out.get("r"))
            self.assertLess(elapsed, 10)  # killed promptly, not after sleep 30
            self.assertFalse(any(m.role == "tool" for m in loop.messages))

    def test_old_worker_finishes_after_new_run_starts(self):
        """old worker finishes after new run starts: a worker superseded
        mid-tool-round never delivers its results or touches the shared
        history of the newer run."""
        session = ScenarioSession()
        session.tools_enabled = False
        started = threading.Event()

        def slow_bash(name, args, call_id=None):
            if name == "Bash":
                started.set()
                time.sleep(0.4)
                return "bash output"
            return "ok"

        session.execute_tool = slow_bash
        session.client.script = [
            ("", [ToolCall(id="1", name="Bash", arguments=json.dumps({"command": "sleep 1"}))]),
            "new run answer",
        ]
        loop1 = AgentLoop(session, messages=[Message(role="user", content="q1")])
        t, out = run_in_thread(loop1)
        self.assertTrue(started.wait(timeout=5))
        session.run_generation += 1  # a newer top-level run starts
        loop2 = AgentLoop(session, messages=[Message(role="user", content="q2")])
        r2 = loop2.run()
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertNotIn("e", out)
        self.assertIsNone(out.get("r"))
        self.assertEqual(r2, "new run answer")
        self.assertEqual([m.role for m in session.last_messages], ["user", "assistant"])
        self.assertEqual(session.last_messages[0].text(), "q2")
        self.assertEqual(session.last_messages[1].text(), "new run answer")
        self.assertFalse(any(m.role == "tool" for m in loop1.messages))


class TestScenarioCompaction(unittest.TestCase):
    def test_context_compaction_resume(self):
        """context compaction -> resume: past the trigger the history is
        replaced by the summary + last user request, and the run resumes
        to a final answer."""
        session = ScenarioSession()
        session.tools_enabled = True
        session.client.script = ["answer after compaction"]
        calls = {"n": 0}

        def fake_estimate(*a, **k):
            calls["n"] += 1
            return 1_000_000 if calls["n"] <= 2 else 100

        with (
            mock.patch(
                "python_agent_harness.agent.estimate_payload_tokens",
                side_effect=fake_estimate,
            ),
            mock.patch("python_agent_harness.config.MAX_NUDGES", 0),
        ):
            loop = AgentLoop(session, messages=[Message(role="user", content="long task")])
            result = loop.run()
        self.assertEqual(result, "answer after compaction")
        self.assertEqual(loop.messages[0].role, "user")
        self.assertIn("Compacted Summary", loop.messages[0].text())
        self.assertEqual(loop.messages[1].text(), "long task")
        resumed = session.client.calls[-1]
        contents = [m.get("content") for m in resumed if m.get("role") == "user"]
        self.assertTrue(any("Compacted Summary" in c for c in contents))
        self.assertIn("long task", contents)


class TestScenarioSubagent(unittest.TestCase):
    def test_subagent_failure_parent_survives(self):
        """sub-agent failure -> parent survives: an LLM error inside the
        sub-agent loop becomes an error string for the parent, which
        continues to its own completion."""
        session = ScenarioSession()
        session.tools_enabled = False

        class BoomSubClient(ScriptedClient):
            def chat(self, *a, **k):
                raise RuntimeError("subagent API down")

        session.subagent_client = BoomSubClient()
        session.client.script = [
            ("", [agent_call("1", "risky task", "do it")]),
            "parent final answer",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="delegate")])
        result = loop.run()
        self.assertEqual(result, "parent final answer")
        rows = [m.text() for m in loop.messages if m.role == "tool"]
        self.assertEqual(len(rows), 1)
        self.assertIn("Error: subagent API down", rows[0])
        self.assertEqual([n for n, _ in session.executed], ["Agent"])

    def test_subagent_cancellation_parent_survives(self):
        """sub-agent cancellation -> parent survives: Ctrl-C while a
        sub-agent is blocked aborts it; the parent stops cleanly with
        no exception and no zombie worker."""
        session = ScenarioSession()
        session.tools_enabled = False
        blocker = BlockingClient()
        session.subagent_client = blocker
        session.client.script = [("", [agent_call("1", "deep dive", "research")])]
        loop = AgentLoop(session, messages=[Message(role="user", content="delegate")])
        t, out = run_in_thread(loop)
        self.assertTrue(blocker.started.wait(timeout=5))
        time.sleep(0.2)
        session.cancel()
        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        self.assertNotIn("e", out)
        self.assertIsNone(out.get("r"))
        self.assertTrue(blocker.aborted)
        self.assertEqual([n for n, _ in session.executed], ["Agent"])


class TestScenarioPlanBuild(unittest.TestCase):
    def test_plan_planexit_build(self):
        """plan -> PlanExit -> build: the approved handoff switches the
        session to build mode, unregisters PlanExit, and injects the
        approved message into the next request."""
        session = ScenarioSession()
        session.tools_enabled = False
        session.switch_to_plan()
        session.confirm_fn = lambda prompt: True
        session.client.script = [
            ("", [ToolCall(id="p1", name="PlanExit", arguments="{}")]),
            "implementing now",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="plan the change")])
        result = loop.run()
        self.assertEqual(result, "implementing now")
        self.assertFalse(session.plan_mode.is_plan)
        self.assertIsNone(session.registry.get("PlanExit"))
        rows = [m for m in loop.messages if m.role == "tool"]
        self.assertEqual(len(rows), 1)
        self.assertIn("approved switching to build", rows[0].text())
        contents = [m.get("content") for m in session.client.calls[-1] if m.get("role") == "user"]
        self.assertTrue(any("has been approved" in c for c in contents))


class TestScenarioToolResults(unittest.TestCase):
    def test_tool_returns_none(self):
        """tool returns None -> the NIL placeholder is delivered to the
        model (never a crash or a missing row) and the run completes."""
        session = ScenarioSession()
        session.tools_enabled = False

        def none_tool(name, args, call_id=None):
            if name == "Bash":
                return None
            return "ok"

        session.execute_tool = none_tool
        session.client.script = [
            ("", [ToolCall(id="1", name="Bash", arguments='{"command": "echo hi"}')]),
            "final answer",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        result = loop.run()
        self.assertEqual(result, "final answer")
        rows = [m for m in loop.messages if m.role == "tool"]
        self.assertEqual(len(rows), 1)
        self.assertIn("produced no result", rows[0].text())

    def test_tool_returns_malformed_result(self):
        """tool returns a malformed (non-str) result -> sanitized to a
        string, delivered, and the run completes."""
        session = ScenarioSession()
        session.tools_enabled = False

        def weird_tool(name, args, call_id=None):
            if name == "Bash":
                return {"data": 42, "items": [1, 2]}
            return "ok"

        session.execute_tool = weird_tool
        session.client.script = [
            ("", [ToolCall(id="1", name="Bash", arguments='{"command": "echo hi"}')]),
            "final answer",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        result = loop.run()
        self.assertEqual(result, "final answer")
        rows = [m for m in loop.messages if m.role == "tool"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].text(), str({"data": 42, "items": [1, 2]}))


class TestScenarioHttpRetry(unittest.TestCase):
    def setUp(self):
        fake_openai_server.reset_state()

    def tearDown(self):
        fake_openai_server.reset_state()

    def test_http_429_retry_then_success(self):
        """HTTP 429 -> retry: the first attempt is rate-limited and the
        retry succeeds; the loop completes with exactly two requests."""
        fake_openai_server.STATUS_QUEUE = [429]
        fake_openai_server.NON_STREAM_SEQUENCE = [
            {
                "choices": [{"message": {"role": "assistant", "content": "rate limited but done"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        ]
        with tempfile.TemporaryDirectory(prefix="pah-http-") as tmpdir:
            session, srv = make_http_session(tmpdir)
            try:
                loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
                result = loop.run()
            finally:
                session.client.close()
                srv.shutdown()
        self.assertEqual(result, "rate limited but done")
        self.assertEqual(loop.state, AgentLoop.DONE)
        self.assertEqual(len(fake_openai_server.REQUEST_BODIES), 2)

    def test_http_500_retry_then_success(self):
        """HTTP 500 -> retry: a server error is transient and the retry
        succeeds."""
        fake_openai_server.STATUS_QUEUE = [500]
        fake_openai_server.NON_STREAM_SEQUENCE = [
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "server hiccup but done"}}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        ]
        with tempfile.TemporaryDirectory(prefix="pah-http-") as tmpdir:
            session, srv = make_http_session(tmpdir)
            try:
                loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
                result = loop.run()
            finally:
                session.client.close()
                srv.shutdown()
        self.assertEqual(result, "server hiccup but done")
        self.assertEqual(loop.state, AgentLoop.DONE)
        self.assertEqual(len(fake_openai_server.REQUEST_BODIES), 2)

    def test_http_permanent_4xx_fails(self):
        """permanent 4xx -> fail: a 400 is not retried; the loop ends in
        ERRS with the API error surfaced."""
        fake_openai_server.STATUS_QUEUE = [400]
        with tempfile.TemporaryDirectory(prefix="pah-http-") as tmpdir:
            session, srv = make_http_session(tmpdir)
            try:
                loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
                result = loop.run()
            finally:
                session.client.close()
                srv.shutdown()
        self.assertTrue(result.startswith("Error: API error 400"), result)
        self.assertEqual(loop.state, AgentLoop.ERRS)
        self.assertEqual(len(fake_openai_server.REQUEST_BODIES), 1)


class DropOnceHandler(BaseHTTPRequestHandler):
    """Streams one delta on the first request, then drops the connection
    mid-body (chunked body ends without the terminating chunk)."""

    protocol_version = "HTTP/1.1"
    attempts = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length or b"{}")
        type(self).attempts += 1
        if type(self).attempts == 1:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunk = b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)
            return
        chunks = [
            b'data: {"choices": [{"delta": {"content": "final "}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "answer"}}]}\n\n',
            b'data: {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}\n\n',
            b"data: [DONE]\n\n",
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, *a):
        pass


class TestScenarioStreamDisconnect(unittest.TestCase):
    def test_disconnect_mid_stream_retried_without_duplicate_output(self):
        """stream disconnect -> retry without duplicate output: a stream
        dropping mid-body is retried on a fresh connection; the stored
        conversation carries the retried response exactly once."""
        DropOnceHandler.attempts = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), DropOnceHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        host, port = server.server_address
        client = Client(
            base_url=f"http://{host}:{port}/v1",
            api_key="test",
            model="fake",
            retry_max=3,
            retry_base_delay=0.01,
            retry_max_delay=0.05,
        )
        with tempfile.TemporaryDirectory(prefix="pah-drop-") as tmpdir:
            session = ScenarioSession(project_dir=tmpdir)
            session.client = client
            session.tools_enabled = False
            session.store.title = "T"
            deltas = []
            notified = []
            session.on_delta = deltas.append
            session.notify_fn = lambda kind, data=None: notified.append(kind)
            try:
                loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
                result = loop.run()
            finally:
                client.close()
                server.shutdown()
        self.assertEqual(result, "final answer")
        self.assertEqual(DropOnceHandler.attempts, 2)
        self.assertIn("retry", notified)
        self.assertEqual("".join(deltas), "partialfinal answer")
        self.assertEqual(loop.messages[-1].text(), "final answer")
        self.assertFalse(any("partial" in m.text() for m in loop.messages))


if __name__ == "__main__":
    unittest.main()
