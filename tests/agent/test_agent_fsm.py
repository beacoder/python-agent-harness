"""End-to-end agent loop tests with a fake client and fake session."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/ for plan_cleanup

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
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


class TestAgentFSM(unittest.TestCase):
    """FSM structure and state traces.

    Pins the state machine (loop -> FSM refactor): terminal states,
    visited-state history, per-round info reset, and the driver's
    fail-loudly routing.
    """

    def test_terminal_states_have_no_transitions(self):
        """DONE/ERRS/ABRT are terminals: no table entry, driver stops."""
        self.assertEqual(AgentLoop.TERMINAL, {AgentLoop.DONE, AgentLoop.ERRS, AgentLoop.ABRT})
        for state in AgentLoop.TERMINAL:
            self.assertNotIn(state, AgentLoop.TRANSITIONS)

    def test_simple_turn_trace(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["hello"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        self.assertEqual(loop.run(), "hello")
        self.assertEqual(loop.state, AgentLoop.DONE)
        self.assertEqual(loop.history, [AgentLoop.WAIT, AgentLoop.SUPERVISE])
        self.assertEqual(loop.result, "hello")

    def test_tool_round_trace(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = [
            ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
            "final",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="read it")])
        self.assertEqual(loop.run(), "final")
        self.assertEqual(loop.state, AgentLoop.DONE)
        self.assertEqual(
            loop.history,
            [AgentLoop.WAIT, AgentLoop.TOOL, AgentLoop.TRET, AgentLoop.WAIT, AgentLoop.SUPERVISE],
        )

    def test_cancel_trace_ends_in_abrt(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.cancel()
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        self.assertIsNone(loop.run())
        self.assertEqual(loop.state, AgentLoop.ABRT)
        self.assertEqual(loop.history, [AgentLoop.WAIT])
        self.assertIsNone(loop.result)

    def test_error_trace_ends_in_errs(self):
        session = RecordingSession()
        session.tools_enabled = False

        class Boom:
            def chat(self, *a, **k):
                raise RuntimeError("api down")

            def chat_sync(self, *a, **k):
                raise RuntimeError("x")

        session.client = Boom()
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        self.assertEqual(loop.run(), "Error: api down")
        self.assertEqual(loop.state, AgentLoop.ERRS)
        self.assertEqual(loop.history, [AgentLoop.WAIT])

    def test_budget_exhaustion_skips_supervise(self):
        """Sub-agent budget exhaustion routes WAIT -> DONE directly
        (the terminal-text backward scan runs; SUPERVISE is never
        entered)."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = [
            ("partial", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
        ]
        loop = AgentLoop(
            session,
            messages=[Message(role="user", content="do it")],
            top_level=False,
            max_rounds=1,
        )
        self.assertEqual(loop.run(), "partial")
        self.assertEqual(loop.state, AgentLoop.DONE)
        self.assertEqual(
            loop.history,
            [AgentLoop.WAIT, AgentLoop.TOOL, AgentLoop.TRET, AgentLoop.WAIT],
        )
        self.assertNotIn(AgentLoop.SUPERVISE, loop.history)

    def test_compaction_reenters_wait(self):
        """A successful in-loop compaction re-enters WAIT (the
        compacted round sends the request instead of classifying an
        empty response)."""
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
        self.assertEqual(loop.state, AgentLoop.DONE)
        self.assertIn(
            (AgentLoop.WAIT, AgentLoop.WAIT),
            zip(loop.history, loop.history[1:], strict=False),
        )

    def test_nudge_cycles_supervise_back_to_wait(self):
        """The nudge flag routes SUPERVISE -> WAIT: a terminal answer
        on an agentic top-level loop with nudge budget left never ends
        the run until the budget is spent."""
        session = RecordingSession()
        session.client.script = ["a", "b", "c"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 2):
            self.assertEqual(loop.run(), "c")
        self.assertEqual(loop.state, AgentLoop.DONE)
        self.assertEqual(loop.supervisor.nudge_count, 2)
        self.assertEqual(
            loop.history,
            [
                AgentLoop.WAIT,
                AgentLoop.SUPERVISE,
                AgentLoop.WAIT,
                AgentLoop.SUPERVISE,
                AgentLoop.WAIT,
                AgentLoop.SUPERVISE,
            ],
        )

    def test_wait_resets_info_between_rounds(self):
        """The per-round info flags are cleared on every WAIT entry:
        the first round's tool_calls must not leak into the second."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = [
            ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.run()
        self.assertIsNone(loop.info.get("tool_calls"))
        self.assertEqual(loop.info["assistant"].text(), "done")

    def test_next_state_fails_loudly_without_matching_predicate(self):
        """A table entry whose predicates all fail (no True default)
        must raise instead of stalling the machine."""
        session = RecordingSession()
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        table = dict(AgentLoop.TRANSITIONS)
        table[AgentLoop.TOOL] = ((AgentLoop._cancelled_p, AgentLoop.ABRT),)
        with mock.patch.object(AgentLoop, "TRANSITIONS", table):
            loop.state = AgentLoop.TOOL
            with self.assertRaises(RuntimeError):
                loop._next_state()
