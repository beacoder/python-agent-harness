"""End-to-end agent loop tests with a fake client and fake session."""

import json
import os
import sys
import tempfile
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
             max_tokens=None, reasoning_effort=None, on_delta=None, stream=True,
             cancel_check=None):
        self.calls.append([m.to_api() for m in messages])
        self.kwargs.append({
            "tools": tools, "system": system, "temperature": temperature,
            "max_tokens": max_tokens, "reasoning_effort": reasoning_effort,
            "stream": stream, "cancel_check": cancel_check,
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


class ParallelToolSession(RecordingSession):
    """Session whose tool calls block (DURATION seconds, or until cancel
    when DURATION is None) while tracking peak concurrency across ALL
    tools."""

    def __init__(self, duration=0.4):
        super().__init__()
        self.duration = duration
        self.active = 0
        self.max_active = 0
        self.executed_count = 0
        self._lock = threading.Lock()
        self.started = threading.Event()

    def execute_tool(self, name, args, call_id=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.executed_count += 1
        self.started.set()
        try:
            if self.duration is None:
                deadline = time.monotonic() + 30
                while not self.cancel_event.is_set() and time.monotonic() < deadline:
                    time.sleep(0.02)
            else:
                time.sleep(self.duration)
            if name == "Agent":
                return f"done:{args.get('description', 'task')}"
            return f"result of {name}"
        finally:
            with self._lock:
                self.active -= 1


class SerialPromptSession(RecordingSession):
    """Session whose interactive prompt handlers (Question / PlanExit
    confirm) each block DURATION seconds while tracking peak
    concurrency — to verify the session serializes them."""

    def __init__(self, duration=0.3):
        super().__init__()
        self.duration = duration
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        self.ask_fn = self._prompt
        self.confirm_fn = self._prompt

    def _prompt(self, prompt):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.duration)
            return "yes"
        finally:
            with self._lock:
                self.active -= 1


class StaggeredSession(RecordingSession):
    """Session with per-tool durations that records the real execution
    completion order — so a test can prove the round delivers results
    in ORIGINAL call order even when execution finishes in a different
    order."""

    def __init__(self, durations):
        super().__init__()
        self.durations = durations
        self.active = 0
        self.max_active = 0
        self.completed = []
        self._lock = threading.Lock()

    def execute_tool(self, name, args, call_id=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.durations[name])
            return f"result of {name}"
        finally:
            with self._lock:
                self.active -= 1
                self.completed.append(name)


class RealParallelSession(RecordingSession):
    """Session that runs REAL sub-agents (the Agent tool delegates to
    the real AgentSession.execute_tool) while every other tool blocks
    DURATION seconds, tracking concurrency across the parent round and
    the sub-agent's own round alike (a sub-agent shares this session)."""

    def __init__(self, duration=0.4):
        super().__init__()
        self.duration = duration
        self.active = 0
        self.max_active = 0
        self.executed_names = []
        self._lock = threading.Lock()

    def execute_tool(self, name, args, call_id=None):
        if name == "Agent":
            return AgentSession.execute_tool(self, name, args, call_id=call_id)
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.executed_names.append(name)
        try:
            time.sleep(self.duration)
            return f"result of {name}"
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

    def test_subagent_budget_exhausted_returns_last_real_text(self):
        """Round-budget exhaustion must surface the last real assistant
        text, never a trailing tool result or an empty string."""
        session = RecordingSession()
        session.tools_enabled = False
        # round 1: text + tool call; round 2: tool call only; budget 2
        session.client.script = [
            ("partial answer", [ToolCall(
                id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
            ("", [ToolCall(
                id="2", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
        ]
        loop = AgentLoop(
            session,
            messages=[Message(role="user", content="do it")],
            top_level=False, max_rounds=2,
        )
        result = loop.run()
        # the trailing messages are [tool result, assistant+tool_call,
        # tool result]; the only real text is "partial answer"
        self.assertEqual(result, "partial answer")

    def test_subagent_budget_exhausted_no_text_returns_error(self):
        """Exhaustion with NO assistant text at all (only tool rounds)
        must return an informative error, not raw tool output or ''."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = [
            ("", [ToolCall(
                id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
            ("", [ToolCall(
                id="2", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
        ]
        loop = AgentLoop(
            session,
            messages=[Message(role="user", content="do it")],
            top_level=False, max_rounds=2,
        )
        result = loop.run()
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Error: sub-agent round budget"), result)
        self.assertNotEqual(result, "file content")  # not raw tool output
        self.assertNotEqual(result, "")

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

    def test_multiple_tool_calls_run_concurrently(self):
        """Several tool calls in one round — Agent and plain tools alike
        — execute in parallel: peak concurrency exceeds 1, and results
        are delivered in the original tool-call order."""
        session = ParallelToolSession(duration=0.4)
        session.tools_enabled = False
        session.client.script = [
            ("", [
                agent_call("1", "task one"),
                ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/x.py"}'),
                ToolCall(id="3", name="Bash", arguments='{"command": "echo hi"}'),
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
            ["done:task one", "result of Read", "result of Bash"],
        )

    def test_mixed_round_all_tools_run_concurrently(self):
        """Every tool in a round — Agent and plain tools alike — runs
        concurrently, and all results are delivered in original order."""
        session = ParallelToolSession(duration=0.3)
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
        self.assertEqual(by_id["2"], "result of Read")
        self.assertEqual(by_id["3"], "done:beta")
        # delivered in original order
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3"],
        )

    def test_cancel_during_parallel_subagents(self):
        """Ctrl-C while sub-agents run in parallel: the round stops, the
        run returns None, and no exception escapes the thread pool."""
        session = ParallelToolSession(duration=None)
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

    def test_cancelled_queued_tool_does_not_run(self):
        """A tool still QUEUED when Ctrl-C lands must never execute: with
        a single pool worker, call 1 blocks, the cancel arrives while it
        runs, and the queued call 2 is skipped by the per-task pre-check
        (deterministic counterpart of the salvage tests' 2-3 range)."""
        session = ParallelToolSession(duration=None)
        session.tools_enabled = False
        session.client.script = [
            ("", [
                ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/b.py"}'),
            ]),
        ]
        result = {}
        with mock.patch("python_agent_harness.config.PARALLEL_TOOL_MAX", 1):
            worker = threading.Thread(
                target=lambda: result.update(
                    r=AgentLoop(
                        session, messages=[Message(role="user", content="read")]
                    ).run()
                )
            )
            worker.start()
            self.assertTrue(session.started.wait(timeout=5))
            time.sleep(0.2)  # worker 1 is now blocked; call 2 is queued
            session.cancel()
            worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(result.get("r"))
        # only the already-running call executed; the queued call was
        # skipped after Ctrl-C and never reached execute_tool
        self.assertEqual(session.executed_count, 1)
        self.assertEqual(session.max_active, 1)

    def test_pool_caps_concurrency_but_runs_all_calls(self):
        """More tool calls than PARALLEL_TOOL_MAX still ALL execute (the
        excess queue), and peak concurrency never exceeds the cap."""
        session = ParallelToolSession(duration=0.15)
        session.tools_enabled = False
        calls = [
            ToolCall(id=str(i), name="Read", arguments='{"file_path": "/tmp/x.py"}')
            for i in range(1, 9)
        ]
        session.client.script = [("", calls), "done"]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        with mock.patch("python_agent_harness.config.PARALLEL_TOOL_MAX", 4):
            start = time.monotonic()
            result = loop.run()
            elapsed = time.monotonic() - start
        self.assertEqual(result, "done")
        # all 8 calls ran, capped at 4 concurrent workers
        self.assertEqual(session.executed_count, 8)
        self.assertLessEqual(session.max_active, 4)
        # ...yet finished in ~2 batches, not 8 sequential durations
        self.assertLess(elapsed, 0.8)
        # results delivered in original order
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            [str(i) for i in range(1, 9)],
        )

    def test_malformed_tool_arguments_do_not_break_round(self):
        """A tool call whose arguments are not valid JSON (or parse to a
        non-object) must not break the round: the arguments degrade to
        {} (an error result for the tool), sibling calls still run, and
        all results are delivered."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = [
            ("", [
                ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                ToolCall(id="2", name="Bash", arguments="not-json{{{"),
                ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
                ToolCall(id="4", name="Read", arguments="[1, 2, 3]"),
            ]),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        result = loop.run()
        self.assertEqual(result, "done")
        # the malformed calls degraded to empty args (never the raw
        # string / raw list); the healthy siblings ran with theirs intact
        read_args = [args for name, args in session.executed if name == "Read"]
        bash_args = [args for name, args in session.executed if name == "Bash"]
        self.assertEqual(
            sorted(str(a["file_path"]) for a in read_args if a),
            ["/tmp/a.py", "/tmp/c.py"],
        )
        self.assertEqual([a for a in read_args if not a], [{}])
        self.assertEqual(bash_args, [{}])
        # all four results delivered in original order
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3", "4"],
        )
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "file content")
        self.assertEqual(by_id["3"], "file content")
        self.assertEqual(by_id["4"], "file content")

    def test_parallel_round_contains_tool_crash(self):
        """A tool that raises inside a parallel round must not kill the
        round: sibling tools still run, the crash becomes an error
        result, and all results are delivered in original order."""
        session = RecordingSession()
        session.tools_enabled = False
        orig_execute = RecordingSession.execute_tool

        def exploding_execute(name, args, call_id=None):
            if name == "Grep":
                raise RuntimeError("boom")
            return orig_execute(session, name, args, call_id=call_id)

        session.execute_tool = exploding_execute
        session.client.script = [
            ("", [
                ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                ToolCall(id="2", name="Bash", arguments='{"command": "echo hi"}'),
                ToolCall(id="3", name="Grep", arguments='{"regex": "x", "path": "/tmp"}'),
            ]),
            "all done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        self.assertEqual(loop.run(), "all done")
        # the two healthy tools ran; the crash was contained
        self.assertEqual(sorted(n for n, _ in session.executed), ["Bash", "Read"])
        tool_rows = [(m.tool_call_id, m.text()) for m in loop.messages if m.role == "tool"]
        self.assertEqual(tool_rows[0], ("1", "file content"))
        self.assertEqual(tool_rows[1], ("2", "bash output"))
        self.assertEqual(tool_rows[2][0], "3")
        self.assertIn("crashed in a worker thread", tool_rows[2][1])
        self.assertIn("boom", tool_rows[2][1])

    def test_interactive_prompts_serialized_under_parallel_rounds(self):
        """Interactive prompts (Question tool, PlanExit confirmation)
        must be strictly serialized: concurrent calls never present more
        than one prompt at a time."""
        session = SerialPromptSession(duration=0.3)
        threads = [
            threading.Thread(target=lambda: session.ask_questions([{"question": "q1"}])),
            threading.Thread(target=lambda: session.confirm("switch?")),
            threading.Thread(target=lambda: session.ask_questions([{"question": "q2"}])),
        ]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        # 3 prompts x 0.3s, one at a time
        self.assertEqual(session.max_active, 1)
        self.assertGreaterEqual(elapsed, 0.6)

    def test_parallel_edits_attach_diffs_to_their_own_call(self):
        """Two real Edit calls running in parallel must attach each
        unified diff to ITS OWN tool call: the thread-local diff slot
        (not a shared one) is what keeps concurrent file mutations from
        cross-attributing diffs in the TUI."""
        import tempfile as _tf

        with _tf.TemporaryDirectory(prefix="pah-parallel-diff-") as tmpdir:
            fa = os.path.join(tmpdir, "a.py")
            fb = os.path.join(tmpdir, "b.py")
            with open(fa, "w") as f:
                f.write("x = 1\n")
            with open(fb, "w") as f:
                f.write("y = 2\n")
            session = RecordingSession(project_dir=tmpdir)
            session.tools_enabled = False
            # run the REAL Edit implementation (RecordingSession normally
            # returns canned results)
            def real_execute(name, args, call_id=None):
                return AgentSession.execute_tool(session, name, args, call_id=call_id)

            session.execute_tool = real_execute
            loop = AgentLoop(session, messages=[Message(role="user", content="edit")])
            calls = [
                ToolCall(id="e1", name="Edit", arguments=json.dumps(
                    {"path": fa, "old_str": "x = 1", "new_str": "x = 42"})),
                ToolCall(id="e2", name="Edit", arguments=json.dumps(
                    {"path": fb, "old_str": "y = 2", "new_str": "y = 43"})),
            ]
            loop.pending = list(calls)
            loop._run_tool_round()
            # both files really changed
            with open(fa) as f:
                self.assertEqual(f.read(), "x = 42\n")
            with open(fb) as f:
                self.assertEqual(f.read(), "y = 43\n")
            # each call carries its own file's diff — not the other's
            self.assertIn("a.py", calls[0].diff)
            self.assertIn("x = 42", calls[0].diff)
            self.assertNotIn("b.py", calls[0].diff)
            self.assertIn("b.py", calls[1].diff)
            self.assertIn("y = 43", calls[1].diff)
            self.assertNotIn("a.py", calls[1].diff)
            # results delivered in original order
            self.assertEqual(
                [m.tool_call_id for m in loop.messages if m.role == "tool"],
                ["e1", "e2"],
            )

    def test_none_result_becomes_placeholder_in_parallel_round(self):
        """A tool returning None inside a parallel round must yield the
        NIL placeholder (never a crash or a missing tool row), while
        sibling results are delivered normally."""
        session = RecordingSession()
        session.tools_enabled = False
        orig_execute = RecordingSession.execute_tool

        def none_execute(name, args, call_id=None):
            if name == "Bash":
                return None
            return orig_execute(session, name, args, call_id=call_id)

        session.execute_tool = none_execute
        session.client.script = [
            ("", [
                ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                ToolCall(id="2", name="Bash", arguments='{"command": "echo hi"}'),
                ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
            ]),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        self.assertEqual(loop.run(), "done")
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "file content")
        self.assertIn("produced no result", by_id["2"])
        self.assertEqual(by_id["3"], "file content")
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3"],
        )

    def test_plan_mode_blocks_writes_in_parallel_round(self):
        """A parallel round under plan mode: mutating tools are blocked
        by the read-only guard while read-only tools run — every result
        is still delivered in original order, and no forbidden write
        lands."""
        session = RecordingSession()
        session.tools_enabled = False
        session.plan_mode = PlanMode("/tmp/fakeproj")
        session.plan_mode.set_mode(session.plan_mode.mode.PLAN, {
            "plan": "P1", "plan-mode": "P2", "build-switch": "B",
        })

        def real_execute(name, args, call_id=None):
            return AgentSession.execute_tool(session, name, args, call_id=call_id)

        session.execute_tool = real_execute
        with tempfile.TemporaryDirectory(prefix="pah-plan-") as tmpdir:
            blocked_path = os.path.join(tmpdir, "blocked.txt")
            readable = os.path.join(tmpdir, "readable.txt")
            with open(readable, "w") as f:
                f.write("readable content\n")
            session.client.script = [
                ("", [
                    ToolCall(id="1", name="Write", arguments=json.dumps(
                        {"path": tmpdir,
                         "filename": "blocked.txt",
                         "content": "should not land"})),
                    ToolCall(id="2", name="Read", arguments=json.dumps(
                        {"file_path": readable})),
                ]),
                "done",
            ]
            loop = AgentLoop(session, messages=[Message(role="user", content="go")])
            self.assertEqual(loop.run(), "done")
            # the Write was blocked by plan mode; the Read still ran
            self.assertFalse(os.path.exists(blocked_path))
            by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
            self.assertIn("blocked by plan mode", by_id["1"])
            self.assertEqual(by_id["2"], "readable content\n")
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2"],
        )

    def test_real_bash_commands_run_in_parallel(self):
        """Two REAL Bash invocations in one round complete in ~one sleep
        duration, not two: execution is not serialized (async delivery,
        no pool slot held while waiting)."""
        with tempfile.TemporaryDirectory(prefix="pah-bash-") as tmpdir:
            session = RecordingSession(project_dir=tmpdir)
            session.tools_enabled = False

            def real_execute(name, args, call_id=None):
                return AgentSession.execute_tool(session, name, args, call_id=call_id)

            session.execute_tool = real_execute
            loop = AgentLoop(session, messages=[Message(role="user", content="run")])
            calls = [
                ToolCall(id="b1", name="Bash", arguments=json.dumps(
                    {"command": "sleep 0.5 && echo one"})),
                ToolCall(id="b2", name="Bash", arguments=json.dumps(
                    {"command": "sleep 0.5 && echo two"})),
            ]
            loop.pending = list(calls)
            start = time.monotonic()
            loop._run_tool_round()
            elapsed = time.monotonic() - start
            by_id = {m.tool_call_id: m.text().strip() for m in loop.messages if m.role == "tool"}
            self.assertEqual(by_id["b1"], "one")
            self.assertEqual(by_id["b2"], "two")
            # ~0.5s if the subprocesses ran concurrently, ~1.0s if serialized
            self.assertLess(elapsed, 0.9)

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
        # as USER messages (the system prompt stays untouched)
        self.assertEqual(loop.messages[0].role, "user")
        self.assertEqual(loop.messages[1].role, "user")
        self.assertIn("Compacted Summary", loop.messages[0].text())
        self.assertTrue(any(
            "Compacted Summary" in m.text() for m in loop.messages
        ))

    def test_compact_resets_nudge_budget(self):
        """Compaction must reset the nudge budget: a terminal answer right
        after compaction must not end the run just because the budget was
        spent before compaction (elisp parity)."""
        session = RecordingSession()
        session.client.script = [
            "premature final answer 1",
            "premature final answer 2",
            "premature final answer 3",
            "answer after compaction",
        ]
        session.tools_enabled = True
        calls = {"n": 0}

        def fake_estimate(*a, **k):
            calls["n"] += 1
            return 1_000_000 if calls["n"] == 3 else 100

        with mock.patch(
            "python_agent_harness.agent.estimate_payload_tokens",
            side_effect=fake_estimate,
        ):
            loop = AgentLoop(session, messages=[Message(role="user", content="long task")])
            result = loop.run()
        self.assertEqual(result, "done")
        self.assertEqual(loop.messages[0].role, "user")
        self.assertIn("Compacted Summary", loop.messages[0].text())
        self.assertGreater(loop.supervisor.nudge_count, 0)

    def test_manual_compact_replaces_history_with_summary_only(self):
        """Manual /compact replaces the history with the summary frame
        only — re-appending the last user request is the automatic
        path's resume step, not the manual command's (elisp parity with
        gptel-agent-harness-commands-compact-buffer, which erases the
        buffer and inserts just the summary frame)."""
        session = RecordingSession()
        session.tools_enabled = False
        session.last_messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
            Message(role="user", content="please continue"),
        ]
        ok, msg = session.compact_conversation()
        self.assertTrue(ok)
        self.assertEqual(len(session.last_messages), 1)
        self.assertEqual(session.last_messages[0].role, "user")
        self.assertIn("Compacted Summary", session.last_messages[0].text())

    def test_manual_compact_does_not_require_user_request(self):
        """Compacting a summary-only history (e.g. a second /compact)
        must still work: the manual command no longer needs a user
        request to resume with."""
        session = RecordingSession()
        session.tools_enabled = False
        session.last_messages = [
            Message(role="user", content="**[Compacted Summary]**\n\nold summary"),
        ]
        ok, _ = session.compact_conversation()
        self.assertTrue(ok)
        self.assertEqual(len(session.last_messages), 1)
        self.assertEqual(session.last_messages[0].role, "user")
        self.assertIn("Compacted Summary", session.last_messages[0].text())

    def test_auto_compact_mirrors_shared_history(self):
        """The in-loop compaction must mirror the compacted conversation
        onto session.last_messages, so the TUI and a later manual
        /compact see the summary — not the old full history."""
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
                loop.run()
        # the compacted frame reached the shared history as a user message
        self.assertEqual(session.last_messages[0].role, "user")
        self.assertIn("Compacted Summary", session.last_messages[0].text())

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

    def test_abort_unblocks_stalled_http_stream(self):
        """Ctrl-C during a REAL stalled chunked SSE stream must unblock
        the worker promptly: closing the pool is not enough on Linux
        (close() cannot wake a blocked recv), so abort() must shutdown
        the in-flight connection socket.  Regression test for Ctrl-C
        leaving a zombie worker that would stall history adoption."""
        import tempfile
        import threading as _threading
        import time as _time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path

        import python_agent_harness.config as cfg
        from python_agent_harness.client import Client

        stall_started = _threading.Event()

        class StallHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length or b"{}")
                stall_started.set()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                first = b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
                self.wfile.write(("%x" % len(first)).encode() + b"\r\n" + first + b"\r\n")
                self.wfile.flush()
                _time.sleep(60)
                try:
                    self.wfile.write(b"0\r\n\r\n")
                except Exception:
                    pass

            def log_message(self, *a):
                pass

        with tempfile.TemporaryDirectory() as d:
            cfg.SESSION_DIR = Path(d)
            server = ThreadingHTTPServer(("127.0.0.1", 0), StallHandler)
            _threading.Thread(target=server.serve_forever, daemon=True).start()
            host, port = server.server_address
            client = Client(
                base_url=f"http://{host}:{port}/v1", api_key="test", model="fake"
            )
            out = {}

            def worker():
                try:
                    client.chat([Message(role="user", content="hi")], stream=True)
                    out["r"] = "completed"
                except BaseException as e:  # noqa: BLE001
                    out["r"] = f"{type(e).__name__}: {e}"

            t = _threading.Thread(target=worker, daemon=True)
            t.start()
            try:
                self.assertTrue(stall_started.wait(timeout=5))
                _time.sleep(0.3)  # worker is now blocked in iter_lines
                client.abort()  # what session.cancel() calls on Ctrl-C
                t.join(timeout=5)
                self.assertFalse(t.is_alive(), "worker stuck after abort()")
            finally:
                server.shutdown()
                client.close()
            # the unblocked read surfaces as a network error (the agent
            # loop maps a cancelled state to a clean None, not an error)
            self.assertIn("Error", out["r"])

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

    def test_title_strips_reasoning_and_uses_session_temperature(self):
        """Reasoning content merged by the client must not leak into the
        title (it would become the first 50 chars of the session name),
        and the title request must use the session temperature."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["bye"]
        session.client.chat_sync_calls = []

        def chat_sync(messages, system=None, temperature=None,
                      max_tokens=None, reasoning_effort=None):
            session.client.chat_sync_calls.append(temperature)
            return (
                Message(
                    role="assistant",
                    content=("We need to generate a title for the conversation. "
                             "Adding MCP support to agent harness"),
                    reasoning="We need to generate a title for the conversation.",
                ),
                Usage(),
            )

        session.client.chat_sync = chat_sync
        with mock.patch(
            "python_agent_harness.prompts.read_prompt_file",
            return_value="TITLE-PROMPT",
        ):
            loop = AgentLoop(session, messages=[Message(role="user", content="add mcp")])
            loop.run()
        self.assertEqual(session.store.title, "Adding-MCP-support-to-agent-harness")
        self.assertEqual(session.client.chat_sync_calls, [session.temperature])

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
        # tool 1 (round 1) and tool 2 (which triggered the cancel) ran;
        # tool 3 may or may not have started before the cancel landed
        # (queued tools are skipped after Ctrl-C) — but its result was
        # never delivered to the conversation
        self.assertGreaterEqual(len(session.executed), 2)
        self.assertLessEqual(len(session.executed), 3)
        self.assertFalse(any(m.role == "tool" and m.tool_call_id in ("2", "3")
                             for m in loop.messages))
        # the completed round survives; the dangling second round
        # (tool calls 2+3 unanswered) is cut from the shared history
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
        # tool 1 and tool 2 (which triggered the cancel) ran; tool 3 may
        # or may not have started before the cancel landed — but no
        # results were delivered to the conversation
        self.assertGreaterEqual(len(session.executed), 2)
        self.assertLessEqual(len(session.executed), 3)
        self.assertFalse(any(m.role == "tool" for m in loop.messages))
        # the salvaged history is a valid prefix (user message only —
        # the round's tool results can't stand without the full set)
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
            [m.role for m in session.last_messages], ["user"]
        )
        self.assertIn(
            "Compacted Summary", session.last_messages[0].text()
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

    def test_compact_no_user_request_returns_false(self):
        """Compaction needs a real (non-nudge) user request to resume
        with; without one it must fail cleanly."""
        session = RecordingSession()
        loop = AgentLoop(session, messages=[Message(role="assistant", content="hi")])
        self.assertFalse(loop.compact())

    def test_compact_empty_summary_returns_false(self):
        """A compaction response with no text must not replace the
        conversation (fail cleanly, reset the compacting flag)."""
        session = RecordingSession()

        def empty_chat_sync(messages, system=None, temperature=None,
                            max_tokens=None, reasoning_effort=None):
            return Message(role="assistant", content=""), Usage()

        session.client.chat_sync = empty_chat_sync
        loop = AgentLoop(session, messages=[Message(role="user", content="do it")])
        self.assertFalse(loop.compact())
        self.assertFalse(session.compacting)

    def test_compact_client_error_returns_false(self):
        """A failing compaction request is non-fatal: it is logged and
        the loop continues without replacing the history."""
        session = RecordingSession()

        def boom_chat_sync(messages, system=None, temperature=None,
                           max_tokens=None, reasoning_effort=None):
            raise RuntimeError("compaction API down")

        session.client.chat_sync = boom_chat_sync
        loop = AgentLoop(session, messages=[Message(role="user", content="do it")])
        self.assertFalse(loop.compact())
        self.assertFalse(session.compacting)

    def test_error_state_beats_terminal_response(self):
        """If the loop carries an error state, the error text wins over
        a terminal response (defensive path for the post-supervision
        error check)."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["hello"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.error = "boom"
        self.assertEqual(loop.run(), "Error: boom")

    def test_zero_budget_subagent_returns_none(self):
        """A sub-agent loop with max_rounds=0 and no assistant text has
        nothing to surface: run() returns None."""
        session = RecordingSession()
        loop = AgentLoop(
            session,
            messages=[Message(role="user", content="hi")],
            top_level=False, max_rounds=0,
        )
        self.assertIsNone(loop.run())


class TestParallelToolRounds(unittest.TestCase):
    def test_cancel_before_round_skips_all_tools(self):
        """Ctrl-C landing BEFORE a tool round starts must skip every
        queued tool (tools have side effects): the round-level guard
        refuses to submit anything, mirroring the sequential loop's
        per-call pre-check — no tool executes, no result is delivered."""
        session = RecordingSession()
        session.tools_enabled = False
        session.cancel()
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        loop.pending = [
            ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
            ToolCall(id="2", name="Bash", arguments='{"command": "echo hi"}'),
        ]
        loop._run_tool_round()
        self.assertEqual(session.executed, [])
        self.assertFalse(any(m.role == "tool" for m in loop.messages))

    def test_empty_pending_round_is_noop(self):
        """A round with nothing pending must return without touching
        the conversation or executing anything."""
        session = RecordingSession()
        session.tools_enabled = False
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        loop.pending = []
        loop._run_tool_round()
        self.assertEqual(session.executed, [])
        self.assertEqual(len(loop.messages), 1)
        self.assertEqual(loop.pending, [])

    def test_results_delivered_in_order_regardless_of_completion(self):
        """Results must be delivered in ORIGINAL tool-call order even
        when execution completes in a different order: here the slowest
        call (Read) is issued first and the fastest (Bash) last, so the
        recorded completion order is the reverse of the delivery
        order — pinning the ordering guarantee deterministically."""
        session = StaggeredSession({"Read": 0.5, "Bash": 0.05, "Grep": 0.1})
        session.tools_enabled = False
        session.client.script = [
            ("", [
                ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                ToolCall(id="2", name="Bash", arguments='{"command": "echo hi"}'),
                ToolCall(id="3", name="Grep", arguments='{"regex": "x", "path": "/tmp"}'),
            ]),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        start = time.monotonic()
        result = loop.run()
        elapsed = time.monotonic() - start
        self.assertEqual(result, "done")
        # the calls really overlapped (this is the deterministic proof;
        # the elapsed check below is only a smoke test for the batch
        # completing in ~one Read duration, 0.5s, not the 0.65s serial
        # sum — with a generous margin for slow CI machines)
        self.assertGreaterEqual(session.max_active, 2)
        # ...and finished fastest-first — NOT in call order (Read is
        # the slowest call, yet it is delivered first)
        self.assertEqual(session.completed, ["Bash", "Grep", "Read"])
        self.assertLess(elapsed, 0.75)
        # delivery follows the original call order regardless
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3"],
        )
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "result of Read")
        self.assertEqual(by_id["2"], "result of Bash")
        self.assertEqual(by_id["3"], "result of Grep")

    def test_pool_cap_one_still_runs_every_call(self):
        """With PARALLEL_TOOL_MAX=1 the round degrades to a serialized
        pool: EVERY call still executes (nothing is dropped), peak
        concurrency stays 1, the round takes ~the sum of the durations,
        and results still arrive in original order."""
        session = ParallelToolSession(duration=0.15)
        session.tools_enabled = False
        calls = [
            ToolCall(id=str(i), name="Read", arguments='{"file_path": "/tmp/x.py"}')
            for i in range(1, 4)
        ]
        session.client.script = [("", calls), "done"]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        with mock.patch("python_agent_harness.config.PARALLEL_TOOL_MAX", 1):
            start = time.monotonic()
            result = loop.run()
            elapsed = time.monotonic() - start
        self.assertEqual(result, "done")
        self.assertEqual(session.executed_count, 3)
        self.assertEqual(session.max_active, 1)
        # ~3 x 0.15s serialized, not a single 0.15s batch
        self.assertGreaterEqual(elapsed, 0.4)
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3"],
        )

    def test_cancel_during_delivery_discards_partial_round(self):
        """Ctrl-C landing while results are being DELIVERED (after the
        pool already finished) must stop the delivery loop: the tools'
        side effects are done, but already-delivered results stay local
        to the dead run, and the salvaged shared history cuts the
        dangling round — no tool call is left without its response."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = [
            ("", [
                ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/b.py"}'),
                ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
            ]),
        ]
        calls = {"n": 0}
        orig_deliver = AgentLoop._deliver_tool_result

        def cancelling_deliver(self, p, result):
            calls["n"] += 1
            if calls["n"] == 2:
                session.cancel()  # Ctrl-C during the 2nd delivery
            return orig_deliver(self, p, result)

        loop = AgentLoop(session, messages=[Message(role="user", content="read all")])
        with mock.patch.object(AgentLoop, "_deliver_tool_result", cancelling_deliver):
            self.assertIsNone(loop.run())
        # all three tools ran (side effects are done), but only two
        # results were delivered before the cancel landed
        self.assertEqual(len(session.executed), 3)
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2"],
        )
        # the salvaged history cuts the dangling round entirely: only
        # the user message survives (the round's results cannot stand
        # without the full set)
        self.assertEqual([m.role for m in session.last_messages], ["user"])
        self.assertEqual(session.last_messages[0].text(), "read all")

    def test_real_subagent_parallel_round_runs_inside_parent_round(self):
        """A REAL Agent call in a parallel round spawns a sub-agent
        whose own tool round executes through the same shared session
        CONCURRENTLY with the parent's sibling tools: parent Glob +
        sub-agent Bash + sub-agent Read all overlap (peak concurrency
        3), results arrive in original order, and the sub-agent's
        internals never leak into the parent's shared history."""
        session = RealParallelSession(duration=0.4)
        session.tools_enabled = False
        session.client.script = [
            ("", [
                agent_call("1", "sub task", "sub prompt"),
                ToolCall(id="2", name="Glob", arguments='{"pattern": "*.py"}'),
            ]),
            ("", [
                ToolCall(id="s1", name="Bash", arguments='{"command": "echo hi"}'),
                ToolCall(id="s2", name="Read", arguments='{"file_path": "/tmp/x.py"}'),
            ]),
            "sub done",
            "parent done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="delegate")])
        result = loop.run()
        self.assertEqual(result, "parent done")
        # the sub-agent's two tools overlapped with the parent's Glob:
        # all three were in flight at once
        self.assertGreaterEqual(session.max_active, 3)
        self.assertEqual(sorted(session.executed_names), ["Bash", "Glob", "Read"])
        # the parent round delivered in original order; the Agent result
        # is the sub-agent's return string
        tool_rows = [(m.tool_call_id, m.text()) for m in loop.messages if m.role == "tool"]
        self.assertEqual([t[0] for t in tool_rows], ["1", "2"])
        self.assertEqual(tool_rows[0][1], "sub done")
        self.assertEqual(tool_rows[1][1], "result of Glob")
        # the sub-agent's internal conversation never reached the
        # parent's shared history
        texts = [m.text() for m in session.last_messages]
        self.assertFalse(any("sub prompt" in t for t in texts))

    def test_salvage_cuts_open_round_variants(self):
        """_salvage_messages must cut any round that is not fully
        closed — a second tool-call message while one is open, an
        assistant text reply mid-round, or a stray tool result — back
        to the last complete round (the defensive branches protecting
        the parallel-round cancellation recovery)."""
        a1 = Message(role="assistant", content="", tool_calls=[
            ToolCall(id="1", name="Read", arguments="{}")])
        a2 = Message(role="assistant", content="", tool_calls=[
            ToolCall(id="2", name="Read", arguments="{}")])
        t1 = Message(role="tool", content="r1", tool_call_id="1", name="Read")
        session = RecordingSession()
        # a second tool-call message while a round is open → cut back to
        # the last COMPLETE round (before the open one)
        loop = AgentLoop(session, messages=[
            Message(role="user", content="u"), a1, t1, a2,
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="3", name="Read", arguments="{}")]),
        ])
        self.assertEqual(
            [m.role for m in loop._salvage_messages()],
            ["user", "assistant", "tool"],
        )
        # an assistant TEXT reply while a round is open → cut to the
        # last complete prefix (the user message only)
        loop = AgentLoop(session, messages=[
            Message(role="user", content="u"), a1,
            Message(role="assistant", content="text"),
        ])
        self.assertEqual([m.role for m in loop._salvage_messages()], ["user"])
        # a USER message while a round is open → cut to the last
        # complete prefix (the user message only)
        loop = AgentLoop(session, messages=[
            Message(role="user", content="u"), a1,
            Message(role="user", content="interrupt"),
        ])
        self.assertEqual([m.role for m in loop._salvage_messages()], ["user"])


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


class TestBashAsync(unittest.TestCase):
    """Async Bash contract: run() returns a PendingToolResult and the
    result is delivered when the process exits (mirrors :async t in
    gptel-agent-tools.el — no thread-pool slot held while waiting)."""

    def make_round(self, tmpdir):
        from python_agent_harness.agent import AgentLoop

        session = RecordingSession(project_dir=tmpdir)
        session.tools_enabled = False

        def real_execute(name, args, call_id=None):
            return AgentSession.execute_tool(session, name, args, call_id=call_id)

        session.execute_tool = real_execute
        loop = AgentLoop(session, messages=[Message(role="user", content="run")])
        return session, loop

    def test_run_returns_pending_result(self):
        from python_agent_harness.tools import PendingToolResult, ToolContext
        from python_agent_harness.tools.bash import Bash

        with tempfile.TemporaryDirectory(prefix="pah-bash-") as tmpdir:
            session = RecordingSession(project_dir=tmpdir)
            result = Bash().run({"command": "echo hello"}, ToolContext(session))
            self.assertIsInstance(result, PendingToolResult)
            self.assertEqual(result.wait().strip(), "hello")

    def test_round_delivers_async_result(self):
        with tempfile.TemporaryDirectory(prefix="pah-bash-") as tmpdir:
            with open(os.path.join(tmpdir, "x.txt"), "w") as f:
                f.write("file content\n")
            session, loop = self.make_round(tmpdir)
            loop.pending = [
                ToolCall(id="b1", name="Bash", arguments=json.dumps(
                    {"command": "echo one"})),
                ToolCall(id="b2", name="Read", arguments=json.dumps(
                    {"file_path": os.path.join(tmpdir, "x.txt")})),
            ]
            loop._run_tool_round()
            by_id = {m.tool_call_id: m.text().strip() for m in loop.messages if m.role == "tool"}
            self.assertEqual(by_id["b1"], "one")
            # sync sibling delivered alongside the async one
            self.assertEqual(by_id["b2"], "file content")
            self.assertEqual(
                [m.tool_call_id for m in loop.messages if m.role == "tool"],
                ["b1", "b2"],  # original order preserved
            )

    def test_cancel_kills_process_and_delivers_error(self):
        from python_agent_harness.tools import PendingToolResult, ToolContext
        from python_agent_harness.tools.bash import Bash

        with tempfile.TemporaryDirectory(prefix="pah-bash-") as tmpdir:
            session = RecordingSession(project_dir=tmpdir)
            result = Bash().run({"command": "sleep 30"}, ToolContext(session))
            self.assertIsInstance(result, PendingToolResult)
            threading.Timer(0.5, session.cancel).start()
            start = time.monotonic()
            delivered = result.wait()
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 5)  # killpg unblocked the wait promptly
            self.assertIn("cancelled", delivered)

    def test_deliver_is_idempotent(self):
        from python_agent_harness.tools import PendingToolResult

        p = PendingToolResult()
        p.deliver("first")
        p.deliver("second")  # late duplicate must be a no-op
        self.assertEqual(p.wait(), "first")

    def test_bad_command_returns_error_string_not_pending(self):
        from python_agent_harness.tools import ToolContext
        from python_agent_harness.tools.bash import Bash

        # Popen with shell=True never fails on syntax; simulate the
        # OSError path via an impossible cwd instead
        class FakeSess:
            project_dir = "/nonexistent-pah-dir"

        result = Bash().run({"command": "echo hi"}, ToolContext(FakeSess()))
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Error:"))


class TestAutoSave(unittest.TestCase):
    """Auto-save failures must not be silent: retry once, then leave a
    persistent, visible error state (cleared by the next success)."""

    def make_session(self):
        session = RecordingSession()
        session.logs = []
        session.log_fn = session.logs.append
        session.notified = []
        session.notify_fn = session.notified.append
        return session

    def test_retry_then_persistent_error(self):
        session = self.make_session()
        fails = {"n": 0}

        def flaky_save(text):
            fails["n"] += 1
            if fails["n"] <= 2:
                raise OSError("disk full")
            return None

        with mock.patch.object(session.store, "save", side_effect=flaky_save):
            session.auto_save([Message(role="user", content="hi")], None)
        self.assertEqual(fails["n"], 2)  # retried once, failed again
        self.assertEqual(session._save_error, "disk full")
        self.assertIn("auto-save failed", session.logs[-1])
        self.assertIn("save-error", session.notified)
        # the next successful save clears the persistent error
        with mock.patch.object(session.store, "save", return_value=None):
            session.auto_save([Message(role="user", content="hi")], None)
        self.assertIsNone(session._save_error)

    def test_transient_failure_recovers(self):
        """A one-off failure (retry succeeds) must not leave an error
        state behind."""
        session = self.make_session()
        fails = {"n": 0}

        def transient_save(text):
            fails["n"] += 1
            if fails["n"] == 1:
                raise OSError("nfs hiccup")
            return None

        with mock.patch.object(session.store, "save", side_effect=transient_save):
            session.auto_save([Message(role="user", content="hi")], None)
        self.assertEqual(fails["n"], 2)  # retried
        self.assertIsNone(session._save_error)  # success cleared it
        self.assertNotIn("save-error", session.notified)


if __name__ == "__main__":
    unittest.main()
