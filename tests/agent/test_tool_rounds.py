"""End-to-end agent loop tests with a fake client and fake session."""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/ for plan_cleanup

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
import session_sandbox  # noqa: F401,E402  (side-effect: redirect SESSION_DIR)
from agent_test_utils import (  # noqa: E402,F401
    FakeClient,
    ParallelToolSession,
    RealParallelSession,
    RecordingSession,
    SerialPromptSession,
    StaggeredSession,
    agent_call,
)

from python_agent_harness.agent import AgentLoop
from python_agent_harness.models import Message, ToolCall


class TestToolRounds(unittest.TestCase):
    def test_cancel_before_round_skips_all_tools(self):
        """Ctrl-C landing BEFORE a tool round starts must skip every
        queued tool (tools have side effects): the round-level guard
        refuses to run anything, mirroring the sequential loop's
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

    def test_sync_tools_execute_in_call_order(self):
        """Sync tools execute in model-emitted call order: the recorded
        completion order equals the call order (Read 0.5s first, then
        Bash 0.05s, then Grep 0.1s — the fake session treats all three
        as sync), peak concurrency stays 1, and results are delivered
        in the same order."""
        session = StaggeredSession({"Read": 0.5, "Bash": 0.05, "Grep": 0.1})
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                    ToolCall(id="2", name="Bash", arguments='{"command": "echo hi"}'),
                    ToolCall(id="3", name="Grep", arguments='{"regex": "x", "path": "/tmp"}'),
                ],
            ),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        start = time.monotonic()
        result = loop.run()
        elapsed = time.monotonic() - start
        self.assertEqual(result, "done")
        # one at a time...
        self.assertEqual(session.max_active, 1)
        # ...and finished in call order, ~the 0.65s serial sum
        self.assertEqual(session.completed, ["Read", "Bash", "Grep"])
        self.assertGreaterEqual(elapsed, 0.6)
        # delivery follows the original call order
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3"],
        )
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "result of Read")
        self.assertEqual(by_id["2"], "result of Bash")
        self.assertEqual(by_id["3"], "result of Grep")

    def test_cancel_during_delivery_discards_partial_round(self):
        """Ctrl-C landing while results are being DELIVERED (after all
        tools already ran) must stop the delivery loop: the tools'
        side effects are done, but already-delivered results stay local
        to the dead run, and the salvaged shared history cuts the
        dangling round — no tool call is left without its response."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                    ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/b.py"}'),
                    ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
                ],
            ),
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

    def test_real_subagent_round_runs_inside_parent_round(self):
        """A REAL Agent call is dispatched (async) and spawns a
        sub-agent whose own tool round executes through the same
        shared session while the parent's sync sibling tool runs
        inline: the sub-agent's tools overlap with the parent's Glob
        (peak concurrency 2 — the parent never waits for the
        sub-agent before running its next sync tool), results arrive
        in original order, and the sub-agent's internals never leak
        into the parent's shared history."""
        session = RealParallelSession(duration=0.4)
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    agent_call("1", "sub task", "sub prompt"),
                    ToolCall(id="2", name="Glob", arguments='{"pattern": "*.py"}'),
                ],
            ),
            (
                "",
                [
                    ToolCall(id="s1", name="Bash", arguments='{"command": "echo hi"}'),
                    ToolCall(id="s2", name="Read", arguments='{"file_path": "/tmp/x.py"}'),
                ],
            ),
            "sub done",
            "parent done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="delegate")])
        result = loop.run()
        self.assertEqual(result, "parent done")
        # the sub-agent's tools overlapped with the parent's Glob:
        # both were in flight at once
        self.assertGreaterEqual(session.max_active, 2)
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
        the tool-round cancellation recovery)."""
        a1 = Message(
            role="assistant", content="", tool_calls=[ToolCall(id="1", name="Read", arguments="{}")]
        )
        a2 = Message(
            role="assistant", content="", tool_calls=[ToolCall(id="2", name="Read", arguments="{}")]
        )
        t1 = Message(role="tool", content="r1", tool_call_id="1", name="Read")
        session = RecordingSession()
        # a second tool-call message while a round is open → cut back to
        # the last COMPLETE round (before the open one)
        loop = AgentLoop(
            session,
            messages=[
                Message(role="user", content="u"),
                a1,
                t1,
                a2,
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="3", name="Read", arguments="{}")],
                ),
            ],
        )
        self.assertEqual(
            [m.role for m in loop._salvage_messages()],
            ["user", "assistant", "tool"],
        )
        # an assistant TEXT reply while a round is open → cut to the
        # last complete prefix (the user message only)
        loop = AgentLoop(
            session,
            messages=[
                Message(role="user", content="u"),
                a1,
                Message(role="assistant", content="text"),
            ],
        )
        self.assertEqual([m.role for m in loop._salvage_messages()], ["user"])
        # a USER message while a round is open → cut to the last
        # complete prefix (the user message only)
        loop = AgentLoop(
            session,
            messages=[
                Message(role="user", content="u"),
                a1,
                Message(role="user", content="interrupt"),
            ],
        )
        self.assertEqual([m.role for m in loop._salvage_messages()], ["user"])


class TestParallelReadonly(unittest.TestCase):
    """When every call in a round is readonly (Read, Glob, Grep, Skill),
    the runner dispatches them concurrently via a thread pool: peak
    concurrency equals the call count, wall time is roughly the slowest
    tool (not the serial sum), and results are still delivered in
    original call order."""

    def test_all_readonly_round_runs_in_parallel(self):
        """Three readonly tools (Read, Grep, Glob) with staggered
        durations run concurrently: peak concurrency is 3, wall time
        is ~max(0.5, 0.1, 0.2)=0.5s (not the 0.8s serial sum), and
        results are delivered in original call order."""
        session = StaggeredSession({"Read": 0.5, "Grep": 0.1, "Glob": 0.2})
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                    ToolCall(id="2", name="Grep", arguments='{"regex": "x", "path": "/tmp"}'),
                    ToolCall(id="3", name="Glob", arguments='{"pattern": "*.py"}'),
                ],
            ),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        start = time.monotonic()
        result = loop.run()
        elapsed = time.monotonic() - start
        self.assertEqual(result, "done")
        self.assertEqual(session.max_active, 3)
        self.assertLess(elapsed, 0.8)
        self.assertGreaterEqual(elapsed, 0.45)
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3"],
        )
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "result of Read")
        self.assertEqual(by_id["2"], "result of Grep")
        self.assertEqual(by_id["3"], "result of Glob")

    def test_mixed_round_stays_sequential(self):
        """A round with any non-readonly tool (Bash) falls back to
        sequential dispatch: peak concurrency is 1, wall time is the
        serial sum."""
        session = StaggeredSession({"Read": 0.3, "Bash": 0.2, "Grep": 0.1})
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                    ToolCall(id="2", name="Bash", arguments='{"command": "echo hi"}'),
                    ToolCall(id="3", name="Grep", arguments='{"regex": "x", "path": "/tmp"}'),
                ],
            ),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        start = time.monotonic()
        result = loop.run()
        elapsed = time.monotonic() - start
        self.assertEqual(result, "done")
        self.assertEqual(session.max_active, 1)
        self.assertGreaterEqual(elapsed, 0.55)

    def test_single_readonly_tool_runs(self):
        """A single readonly tool still works (no parallelism needed,
        but the code path must handle len(calls)==1)."""
        session = StaggeredSession({"Read": 0.1})
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}')],
            ),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        result = loop.run()
        self.assertEqual(result, "done")
        self.assertEqual(session.max_active, 1)
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "result of Read")

    def test_readonly_round_cancel_before_start(self):
        """Ctrl-C before a readonly round starts must skip all tools."""
        session = StaggeredSession({"Read": 0.1, "Grep": 0.1})
        session.tools_enabled = False
        session.cancel()
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        loop.pending = [
            ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
            ToolCall(id="2", name="Grep", arguments='{"regex": "x", "path": "/tmp"}'),
        ]
        loop._run_tool_round()
        self.assertEqual(session.max_active, 0)
        self.assertFalse(any(m.role == "tool" for m in loop.messages))

    def test_large_readonly_round_caps_concurrency(self):
        """A readonly round larger than MAX_PARALLEL_READONLY must not
        spawn one thread per call: peak concurrency is capped, yet
        every call still runs and results are delivered in order."""
        from python_agent_harness.tool_runner import MAX_PARALLEL_READONLY

        n = MAX_PARALLEL_READONLY + 4
        session = ParallelToolSession(duration=0.1)
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    ToolCall(id=str(i), name="Read", arguments='{"file_path": "/tmp/a.py"}')
                    for i in range(n)
                ],
            ),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        result = loop.run()
        self.assertEqual(result, "done")
        # every call ran, but never more than the cap at once
        self.assertEqual(session.executed_count, n)
        self.assertEqual(session.max_active, MAX_PARALLEL_READONLY)
        # results delivered in original call order
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            [str(i) for i in range(n)],
        )
