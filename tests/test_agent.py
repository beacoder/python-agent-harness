"""End-to-end agent loop tests with a fake client and fake session."""

import json
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))  # sibling fake server import

from python_agent_harness.agent import AgentLoop, Supervisor, sanitize_tool_result
from python_agent_harness.agent_session import AgentSession
from python_agent_harness.models import Message, ToolCall, Usage
from python_agent_harness.planmode import PlanMode
from python_agent_harness.session_store import SessionStore
from python_agent_harness.tools import default_registry


class FakeClient:
    """Scripted chat responses: (assistant_text, tool_calls) per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.kwargs = []

    def chat(self, messages, tools=None, system=None, temperature=None,
             max_tokens=None, reasoning_effort=None, on_delta=None, stream=True):
        self.calls.append([m.to_api() for m in messages])
        self.kwargs.append({
            "tools": tools, "system": system, "temperature": temperature,
            "max_tokens": max_tokens, "reasoning_effort": reasoning_effort,
            "stream": stream,
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

    def execute_tool(self, name, args, call_id=None):
        self.executed.append((name, args))
        if name == "Read":
            return "file content"
        if name == "Bash":
            return "bash output"
        return f"result of {name}"


class ParallelSubagentSession(RecordingSession):
    """Session whose run_subagent blocks (DURATION seconds, or until
    cancel when DURATION is None) while tracking peak concurrency."""

    def __init__(self, duration=0.4):
        super().__init__()
        self.duration = duration
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        self.started = threading.Event()

    def execute_tool(self, name, args, call_id=None):
        if name == "Agent":
            return self.run_subagent(
                args.get("subagent_type", "subagent"),
                args.get("description", "task"),
                args.get("prompt", ""),
            )
        return super().execute_tool(name, args, call_id=call_id)

    def run_subagent(self, subagent_type, description, prompt):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.duration is None:
                deadline = time.monotonic() + 30
                while not self.cancel_event.is_set() and time.monotonic() < deadline:
                    time.sleep(0.02)
            else:
                time.sleep(self.duration)
            return f"done:{description}"
        finally:
            with self._lock:
                self.active -= 1


def agent_call(call_id, description, prompt="do it"):
    return ToolCall(
        id=call_id,
        name="Agent",
        arguments=json.dumps({
            "subagent_type": "subagent",
            "description": description,
            "prompt": prompt,
        }),
    )


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

    def test_multiple_agent_calls_run_concurrently(self):
        """Several Agent calls in one round execute in parallel: peak
        concurrency exceeds 1, and results are delivered in the original
        tool-call order."""
        session = ParallelSubagentSession(duration=0.4)
        session.tools_enabled = False
        session.client.script = [
            ("", [
                agent_call("1", "task one"),
                agent_call("2", "task two"),
                agent_call("3", "task three"),
            ]),
            "all done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="delegate")])
        start = time.monotonic()
        result = loop.run()
        elapsed = time.monotonic() - start
        self.assertEqual(result, "all done")
        # parallel: all three ran at the same time
        self.assertGreaterEqual(session.max_active, 2)
        # ...and finished in roughly one task duration, not three
        self.assertLess(elapsed, 1.0)
        # results delivered in original call order
        tool_rows = [m.text() for m in loop.messages if m.role == "tool"]
        self.assertEqual(
            tool_rows,
            ["done:task one", "done:task two", "done:task three"],
        )

    def test_mixed_round_sequential_tools_then_parallel_agents(self):
        """Non-Agent tools stay sequential; Agent calls in the same round
        still run concurrently and all results are delivered."""
        session = ParallelSubagentSession(duration=0.3)
        session.tools_enabled = False
        session.client.script = [
            ("", [
                agent_call("1", "alpha"),
                ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/x.py"}'),
                agent_call("3", "beta"),
            ]),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        result = loop.run()
        self.assertEqual(result, "done")
        self.assertGreaterEqual(session.max_active, 2)
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "done:alpha")
        self.assertEqual(by_id["2"], "file content")
        self.assertEqual(by_id["3"], "done:beta")
        # delivered in original order
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3"],
        )

    def test_cancel_during_parallel_subagents(self):
        """Ctrl-C while sub-agents run in parallel: the round stops, the
        run returns None, and no exception escapes the thread pool."""
        session = ParallelSubagentSession(duration=None)
        session.tools_enabled = False
        session.client.script = [
            ("", [agent_call("1", "alpha"), agent_call("2", "beta")]),
        ]
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(
                r=AgentLoop(
                    session, messages=[Message(role="user", content="delegate")]
                ).run()
            )
        )
        worker.start()
        self.assertTrue(session.started.wait(timeout=5))
        time.sleep(0.2)  # let both sub-agents spin up
        session.cancel()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(result.get("r"))

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

    def test_stream_defaults_true_on_session_and_client(self):
        """Streaming is the default: the session opts in unless the
        config file (or --no-stream) says otherwise, and the loop must
        forward that to the client."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["ok"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.run()
        self.assertIs(session.stream, True)
        self.assertIs(session.client.kwargs[-1]["stream"], True)

    def test_non_streaming_reaches_client(self):
        """A session configured non-streaming must send stream=False to
        the client and still complete the loop normally."""
        session = RecordingSession()
        session.stream = False
        session.tools_enabled = False
        session.client.script = ["ok"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        result = loop.run()
        self.assertEqual(result, "ok")
        self.assertIs(session.client.kwargs[-1]["stream"], False)

    def test_non_streaming_tool_round(self):
        """Non-streaming mode must support full tool rounds (the client
        parses tool_calls from the single response)."""
        session = RecordingSession()
        session.stream = False
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
        # every request in the run went out non-streaming
        self.assertTrue(all(k["stream"] is False for k in session.client.kwargs))

    def test_non_streaming_http_tool_round_end_to_end(self):
        """Non-streaming through the REAL HTTP client: tool_calls parsed
        from a single response drive a full tool round, the loop finishes
        with the final answer, and every request went out stream=False."""
        import tempfile
        from pathlib import Path

        import fake_openai_server
        from fake_openai_server import serve

        import python_agent_harness.config as cfg
        from python_agent_harness.client import Client

        with tempfile.TemporaryDirectory() as d:
            cfg.SESSION_DIR = Path(d)  # session store writes land in tmp
            data_file = Path(d) / "data.txt"
            data_file.write_text("hello data", encoding="utf-8")
            fake_openai_server.reset_state()
            try:
                fake_openai_server.NON_STREAM_SEQUENCE = [
                    {  # 1st request: a tool call, no text
                        "choices": [{"message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": "call_1", "type": "function",
                                "function": {"name": "Read", "arguments": json.dumps(
                                    {"file_path": str(data_file)}
                                )},
                            }],
                        }}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                    },
                    {  # 2nd request: the final answer
                        "choices": [{"message": {
                            "role": "assistant", "content": "http non-streaming done",
                        }}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                    },
                ]
                srv = serve()
                host, port = srv.server_address
                client = Client(
                    base_url=f"http://{host}:{port}/v1", api_key="test", model="fake"
                )
                session = AgentSession(
                    project_dir=d, client=client, model="fake",
                    registry=default_registry(), stream=False,
                )
                # non-agentic: no completion nudges — the loop must terminate
                # on the scripted final answer; the fake server still returns
                # tool_calls, so the tool round runs regardless
                session.tools_enabled = False
                try:
                    loop = AgentLoop(
                        session, messages=[Message(role="user", content="read it")]
                    )
                    result = loop.run()
                finally:
                    client.close()
                # snapshot the bodies before resetting shared server state
                bodies = list(fake_openai_server.REQUEST_BODIES)
            finally:
                fake_openai_server.reset_state()  # don't leak server state
            self.assertEqual(result, "http non-streaming done")
            # the tool round really executed against the HTTP response
            tool_rows = [m for m in loop.messages if m.role == "tool"]
            self.assertEqual(len(tool_rows), 1)
            self.assertIn("hello data", tool_rows[0].text())
            # every request (loop chats + title generation) went out
            # non-streaming with the stream flag set
            self.assertTrue(bodies)
            self.assertTrue(
                all(b.get("stream") is False for b in bodies)
            )
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

    def test_title_generated_after_loop_finishes(self):
        """The session must get an LLM title once the agent loop completes."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["bye"]
        session.client.chat_sync_calls = []

        orig_chat_sync = session.client.chat_sync

        def tracking_chat_sync(messages, system=None, temperature=None,
                               max_tokens=None, reasoning_effort=None):
            session.client.chat_sync_calls.append((messages, system))
            return orig_chat_sync(messages, system=system)

        session.client.chat_sync = tracking_chat_sync
        with mock.patch(
            "python_agent_harness.prompts.read_prompt_file",
            return_value="TITLE-PROMPT",
        ):
            loop = AgentLoop(session, messages=[Message(role="user", content="hi there")])
            loop.run()
        self.assertEqual(len(session.client.chat_sync_calls), 1)
        self.assertEqual(session.client.chat_sync_calls[0][1], "TITLE-PROMPT")
        self.assertEqual(session.store.title, "SYNC-OK")
        self.assertTrue(os.path.basename(session.store.file_path).startswith("SYNC-OK_"))
        # one-shot: a second run must not re-generate the title
        session.client.script = ["again"]
        AgentLoop(session, messages=[Message(role="user", content="hi there")]).run()
        self.assertEqual(len(session.client.chat_sync_calls), 1)

    def test_no_title_for_empty_first_message(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["bye"]
        session.client.chat_sync_calls = []
        loop = AgentLoop(session, messages=[Message(role="user", content="")])
        loop.run()
        self.assertEqual(session.client.chat_sync_calls, [])
        self.assertIsNone(session.store.title)

    def test_stale_cancelled_run_does_not_clobber_next_run(self):
        """A cancelled worker finishing late must not overwrite the next
        run's shared history even after the new run cleared the event."""
        session = RecordingSession()
        session.tools_enabled = False

        class AbortClient(FakeClient):
            """Ctrl-C aborts the request; the blocked read raises late,
            after the next run already cleared the shared event."""

            def __init__(self):
                super().__init__([])

            def chat(self, *a, **k):
                session.cancel()  # Ctrl-C: cancel() aborts the HTTP client
                # the next run started meanwhile: `_start_agent` bumps
                # the run generation and clears the shared event
                session.run_generation += 1
                session.cancel_event.clear()
                raise RuntimeError("aborted read")

        session.client = AbortClient()
        # run 1: user asks q1, presses Ctrl-C mid-flight; the stale worker
        # finishes late (after the next run cleared the event)
        loop1 = AgentLoop(session, messages=[Message(role="user", content="q1")])
        # must be treated as a cancel (None), not a spurious error, and
        # must not clobber the shared history with its partial messages
        self.assertIsNone(loop1.run())
        self.assertEqual(session.last_messages, [])
        self.assertIsNone(session.store.title)

        # run 2 completes normally: full history must be present
        session.client = FakeClient(["second answer"])
        loop2 = AgentLoop(session, messages=[Message(role="user", content="q2")])
        self.assertEqual(loop2.run(), "second answer")
        roles = [m.role for m in session.last_messages]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(session.last_messages[0].text(), "q2")

    def test_cancel_sticks_after_event_cleared(self):
        """A cancelled run stays cancelled once the next run clears the
        shared event (cancel generation + run generation protect state)."""
        session = RecordingSession()
        session.tools_enabled = False
        loop = AgentLoop(session, messages=[Message(role="user", content="q1")])
        loop._cancel_gen = session.cancel_generation  # run() start
        loop._run_gen = session.run_generation
        session.cancel()
        session.run_generation += 1  # next run started
        session.cancel_event.clear()  # ...and cleared the shared event
        self.assertTrue(loop._is_cancelled())
        self.assertTrue(loop._is_stale())

    def test_cancelled_run_keeps_completed_rounds(self):
        """A fully completed tool round survives a cancel that lands in
        a later round: the partial history is cut back to the last
        complete round, so the next turn sends a valid request."""
        session = RecordingSession()
        session.tools_enabled = True
        calls = {"n": 0}
        orig_execute = RecordingSession.execute_tool

        def cancelling_execute(name, args, call_id=None):
            calls["n"] += 1
            if calls["n"] == 2:
                session.cancel()  # Ctrl-C during the second round
            return orig_execute(session, name, args, call_id=call_id)

        session.execute_tool = cancelling_execute
        session.client.script = [
            ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}')]),
            ("", [
                ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/b.py"}'),
                ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
            ]),
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="read all")])
        self.assertIsNone(loop.run())
        # the second round's tool 3 never ran (cancel stops the round)
        self.assertEqual(len(session.executed), 2)
        # the completed round survives; the dangling second round
        # (tool call 3 unanswered) is cut from the shared history
        roles = [m.role for m in session.last_messages]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertEqual(session.last_messages[-1].text(), "file content")

    def test_cancelled_run_salvages_partial_history(self):
        """Ctrl-C mid-tool-round with no successor must not lose the
        turn: completed content survives, and the dangling round is cut
        so the next turn sends a valid request."""
        session = RecordingSession()
        session.tools_enabled = True
        calls = {"n": 0}
        orig_execute = RecordingSession.execute_tool

        def cancelling_execute(name, args, call_id=None):
            calls["n"] += 1
            if calls["n"] == 2:
                session.cancel()  # Ctrl-C while the 2nd tool runs
            return orig_execute(session, name, args, call_id=call_id)

        session.execute_tool = cancelling_execute
        session.client.script = [
            ("", [
                ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/b.py"}'),
                ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
            ]),
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="read all")])
        self.assertIsNone(loop.run())
        # tool 3 never ran: the cancelled run stops mid-round
        self.assertEqual(len(session.executed), 2)
        # the salvaged history is a valid prefix (user message only —
        # the round's lone tool result can't stand without its peers)
        self.assertEqual([m.role for m in session.last_messages], ["user"])
        self.assertEqual(session.last_messages[0].text(), "read all")

    def test_cancel_between_chat_and_tools_skips_tools(self):
        """Ctrl-C after the model emitted tool calls but before the tools
        run: the tools must not execute; with no successor the run still
        salvages its (user-only) partial history."""
        session = RecordingSession()
        session.tools_enabled = True

        class CancelAfterChat(FakeClient):
            def __init__(self):
                super().__init__([])

            def chat(self, *a, **k):
                result = super().chat(*a, **k)
                session.cancel()  # Ctrl-C right after the response
                return result

        session.client = CancelAfterChat()
        session.client.script = [
            ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="read it")])
        self.assertIsNone(loop.run())
        self.assertEqual(session.executed, [])
        # the assistant tool-call message was dropped with the cancel:
        # only the user message is salvaged
        self.assertEqual([m.text() for m in session.last_messages], ["read it"])

    def test_cancelled_worker_does_not_resurrect_after_clear(self):
        """A cancelled worker winding down after /clear (which bumps the
        run generation WITHOUT starting a new run) must not resurrect
        its salvaged history over the cleared state."""
        session = RecordingSession()
        session.tools_enabled = False
        session.cancel_event.set()
        session.cancel_generation += 1   # Ctrl-C
        loop = AgentLoop(session, messages=[Message(role="user", content="q2")])
        loop._run_gen = session.run_generation  # captured before /clear
        session.last_messages = []       # /clear wiped the shared state
        session.run_generation += 1      # /clear invalidated in-flight workers
        self.assertIsNone(loop.run())
        self.assertEqual(session.last_messages, [])

    def test_compact_and_summary_bump_run_generation(self):
        """/compact and /summary replace the shared conversation, so they
        must invalidate in-flight workers just like /clear and /restore
        (otherwise a dying cancelled worker's salvaged-history commit
        would clobber the compacted/summarized buffer)."""
        session = RecordingSession()
        session.tools_enabled = False
        session.last_messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        gen = session.run_generation
        session.compact_conversation()
        self.assertEqual(session.run_generation, gen + 1)
        self.assertEqual(
            [m.role for m in session.last_messages], ["system", "user"]
        )
        session.summarize_conversation()
        self.assertEqual(session.run_generation, gen + 2)

    def test_stale_worker_does_not_stream_deltas(self):
        """A stale cancelled worker must not stream into the live row."""
        session = RecordingSession()
        session.tools_enabled = False
        deltas = []
        session.on_delta = deltas.append

        class StreamingClient(FakeClient):
            def __init__(self):
                super().__init__([])

            def chat(self, messages, **k):
                session.cancel()  # Ctrl-C while the request is in flight
                on_delta = k.get("on_delta")
                if on_delta:
                    on_delta("partial text")
                return super().chat(messages, **k)

        session.client = StreamingClient()
        session.client.script = ["full answer"]
        loop1 = AgentLoop(session, messages=[Message(role="user", content="q1")])
        self.assertIsNone(loop1.run())
        self.assertEqual(deltas, [])


class FakeSupervisorSession:
    def __init__(self, alive=True, tools=True, compacting=False):
        self.alive = alive
        self.tools_enabled = tools
        self.compacting = compacting


class TestSupervisor(unittest.TestCase):
    def test_terminal_agentic_top_level_nudges(self):
        sup = Supervisor(FakeSupervisorSession())
        self.assertTrue(sup.supervise(
            terminal=True, agentic=True, top_level=True, pending=False,
        ))
        self.assertEqual(sup.nudge_count, 1)

    def test_nudge_budget_exhausted(self):
        sup = Supervisor(FakeSupervisorSession())
        for _ in range(2):
            sup.supervise(terminal=True, agentic=True, top_level=True, pending=False)
        self.assertEqual(sup.nudge_count, 2)
        self.assertFalse(sup.supervise(
            terminal=True, agentic=True, top_level=True, pending=False,
        ))

    def test_dead_session_fails_closed(self):
        sup = Supervisor(FakeSupervisorSession(alive=False))
        self.assertFalse(sup.supervise(
            terminal=True, agentic=True, top_level=True, pending=False,
        ))

    def test_reset_nudges_on_tool_calls(self):
        sup = Supervisor(FakeSupervisorSession())
        sup.supervise(terminal=True, agentic=True, top_level=True, pending=False)
        sup.reset_nudges()
        self.assertEqual(sup.nudge_count, 0)

    def test_compacting_blocks_supervision(self):
        sup = Supervisor(FakeSupervisorSession(compacting=True))
        self.assertFalse(sup.supervise(
            terminal=True, agentic=True, top_level=True, pending=False,
        ))

    def test_non_agentic_does_not_nudge(self):
        sup = Supervisor(FakeSupervisorSession(tools=False))
        self.assertFalse(sup.supervise(
            terminal=True, agentic=False, top_level=True, pending=False,
        ))

    def test_non_top_level_does_not_nudge(self):
        sup = Supervisor(FakeSupervisorSession())
        self.assertFalse(sup.supervise(
            terminal=True, agentic=True, top_level=False, pending=False,
        ))

    def test_pending_tools_does_not_nudge(self):
        sup = Supervisor(FakeSupervisorSession())
        self.assertFalse(sup.supervise(
            terminal=True, agentic=True, top_level=True, pending=True,
        ))

    def test_non_terminal_does_not_nudge(self):
        sup = Supervisor(FakeSupervisorSession())
        self.assertFalse(sup.supervise(
            terminal=False, agentic=True, top_level=True, pending=False,
        ))


class TestSanitizeToolResult(unittest.TestCase):
    def test_none_becomes_error_placeholder(self):
        self.assertEqual(
            sanitize_tool_result(None),
            "Error: tool produced no result (it may have been interrupted or failed to return).",
        )

    def test_empty_string_kept(self):
        self.assertEqual(sanitize_tool_result(""), "")

    def test_string_kept(self):
        self.assertEqual(sanitize_tool_result("x"), "x")

    def test_non_string_str_converted(self):
        self.assertEqual(sanitize_tool_result(42), "42")


if __name__ == "__main__":
    unittest.main()
