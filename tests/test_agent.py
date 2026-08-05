"""End-to-end agent loop tests with a fake client and fake session."""

import unittest
from unittest import mock

from python_agent_harness.agent import AgentLoop
from python_agent_harness.harness import AgentSession
from python_agent_harness.models import Message, ToolCall, Usage
from python_agent_harness.planmode import PlanMode
from python_agent_harness.session import SessionStore
from python_agent_harness.tools import default_registry


class FakeClient:
    """Scripted chat responses: (assistant_text, tool_calls) per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.kwargs = []

    def chat(self, messages, tools=None, system=None, temperature=None,
             max_tokens=None, reasoning_effort=None, on_delta=None):
        self.calls.append([m.to_api() for m in messages])
        self.kwargs.append({
            "tools": tools, "system": system, "temperature": temperature,
            "max_tokens": max_tokens, "reasoning_effort": reasoning_effort,
        })
        if not self.script:
            return Message(role="assistant", content="done"), Usage()
        item = self.script.pop(0)
        if isinstance(item, tuple):
            text, tool_calls = item
        else:
            text, tool_calls = item, None
        return Message(role="assistant", content=text, tool_calls=tool_calls), Usage(input_tokens=100)

    def chat_sync(self, messages, system=None, temperature=None, max_tokens=None,
                  reasoning_effort=None):
        return Message(role="assistant", content="SYNC-OK"), Usage()


class RecordingSession(AgentSession):
    _test_session_dir: str | None = None

    def __init__(self, project_dir="/tmp/fakeproj"):
        if RecordingSession._test_session_dir is None:
            import tempfile as _tf

            RecordingSession._test_session_dir = _tf.mkdtemp(prefix="pah-test-sessions-")
            import python_agent_harness.config as cfg

            cfg.SESSION_DIR = __import__("pathlib").Path(RecordingSession._test_session_dir)
        super().__init__(
            project_dir=project_dir,
            client=FakeClient([]),
            model="gpt-5-mini",
            registry=default_registry(),
        )
        self.executed = []
        self.store = SessionStore(
            project_dir=project_dir,
            model=self.model,
            backend=self.backend,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tool_names=self.store.tool_names,
        )

    def execute_tool(self, name, args):
        self.executed.append((name, args))
        if name == "Read":
            return "file content"
        if name == "Bash":
            return "bash output"
        return f"result of {name}"


class TestAgentLoop(unittest.TestCase):
    def test_simple_turn(self):
        session = RecordingSession()
        session.tools_enabled = False  # non-agentic: no completion nudges
        session.client.script = ["hello there"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        result = loop.run()
        self.assertEqual(result, "hello there")

    def test_tool_round(self):
        session = RecordingSession()
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 1):
            session.client.script = [
                ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
                "final answer",
                "final answer",
            ]
            loop = AgentLoop(session, messages=[Message(role="user", content="read it")])
            result = loop.run()
        self.assertEqual(result, "final answer")
        self.assertEqual(session.executed[0][0], "Read")
        self.assertTrue(any(
            m.role == "tool" and m.text() == "file content" for m in loop.messages
        ))

    def test_nudge_redirect(self):
        session = RecordingSession()
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 1):
            session.client.script = ["almost done", "done now"]
            loop = AgentLoop(session, messages=[Message(role="user", content="do it")])
            result = loop.run()
        self.assertEqual(result, "done now")
        # the nudge message was injected
        self.assertTrue(any(
            m.role == "user" and "Task Completion Rules" in m.text()
            for m in loop.messages
        ))

    def test_plan_mode_queues_prompts(self):
        session = RecordingSession()
        session.client.script = ["ok"]
        session.plan_mode = PlanMode("/tmp/fakeproj")
        session.plan_mode.set_mode(session.plan_mode.mode.PLAN, {
            "plan": "P1", "plan-mode": "P2 ${planInfo}", "build-switch": "B",
        })
        loop = AgentLoop(session, messages=[Message(role="user", content="plan it")])
        loop.run()
        sent = session.client.calls[0]
        contents = [m.get("content") for m in sent if m.get("role") == "user"]
        self.assertIn("P1", contents)
        self.assertTrue(any("P2 " in c for c in contents))

    def test_compact_on_high_context(self):
        session = RecordingSession()
        session.client.script = ["answer after compaction"]
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 0):
            calls = {"n": 0}

            def fake_estimate(*a, **k):
                calls["n"] += 1
                return 1_000_000 if calls["n"] <= 2 else 100

            with mock.patch(
                "python_agent_harness.agent.estimate_payload_tokens",
                side_effect=fake_estimate,
            ):
                loop = AgentLoop(session, messages=[Message(role="user", content="long task")])
                result = loop.run()
        self.assertEqual(result, "answer after compaction")
        # compaction replaced the conversation with summary frame + request
        self.assertTrue(any(
            m.role == "system" and "Compacted Summary" in m.text()
            for m in loop.messages
        ))

    def test_auto_save_and_last_messages(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["bye"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.run()
        self.assertTrue(session.last_messages)
        self.assertTrue(session.store.file_path)

    def test_reasoning_effort_reaches_client(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.reasoning_effort = "high"
        session.client.script = ["ok"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.run()
        # FakeClient.chat records the tool kwargs; verify it was passed
        last_call = session.client.kwargs[-1]
        self.assertEqual(last_call.get("reasoning_effort"), "high")

    def test_reasoning_effort_none_omitted(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.reasoning_effort = None
        session.client.script = ["ok"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.run()
        last_call = session.client.kwargs[-1]
        self.assertIsNone(last_call.get("reasoning_effort"))

    def test_cancel_aborts_blocking_chat(self):
        """Ctrl-C during a blocking stream must stop the run, not error."""
        import threading
        import time

        session = RecordingSession()
        session.tools_enabled = True

        class BlockingClient(FakeClient):
            def __init__(self):
                super().__init__([])
                self.unblock = threading.Event()
                self.aborted = False

            def chat(self, *a, **k):
                self.unblock.wait(timeout=10)
                raise RuntimeError("aborted")

            def abort(self):
                self.aborted = True
                self.unblock.set()

        session.client = BlockingClient()
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(
                r=AgentLoop(
                    session, messages=[Message(role="user", content="hi")]
                ).run()
            )
        )
        worker.start()
        time.sleep(0.3)
        session.cancel()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(result.get("r"))
        self.assertTrue(session.client.aborted)

    def test_cancel_cleared_per_run(self):
        """A new run must not inherit a stale cancel from the previous one."""
        session = RecordingSession()
        session.tools_enabled = False
        session.cancel_event.set()
        session.cancel_event.clear()  # TUI clears before each run
        session.client.script = ["works"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        self.assertEqual(loop.run(), "works")

    def test_cancel_mid_tool_round(self):
        session = RecordingSession()
        session.tools_enabled = True
        session.client.script = [
            ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
        ]
        session.cancel_event.set()
        loop = AgentLoop(session, messages=[Message(role="user", content="read it")])
        self.assertIsNone(loop.run())


if __name__ == "__main__":
    unittest.main()
