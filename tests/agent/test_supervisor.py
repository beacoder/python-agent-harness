"""End-to-end agent loop tests with a fake client and fake session."""

import os
import sys
import unittest

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

from python_agent_harness.agent import Supervisor


class FakeSupervisorSession:
    def __init__(self, alive=True, tools=True, compacting=False):
        self.alive = alive
        self.tools_enabled = tools
        self.compacting = compacting


class TestSupervisor(unittest.TestCase):
    def test_terminal_agentic_top_level_nudges(self):
        sup = Supervisor(FakeSupervisorSession())
        self.assertTrue(
            sup.supervise(
                terminal=True,
                agentic=True,
                top_level=True,
                pending=False,
            )
        )
        self.assertEqual(sup.nudge_count, 1)

    def test_nudge_budget_exhausted(self):
        sup = Supervisor(FakeSupervisorSession())
        for _ in range(2):
            sup.supervise(terminal=True, agentic=True, top_level=True, pending=False)
        self.assertEqual(sup.nudge_count, 2)
        self.assertFalse(
            sup.supervise(
                terminal=True,
                agentic=True,
                top_level=True,
                pending=False,
            )
        )

    def test_dead_session_fails_closed(self):
        sup = Supervisor(FakeSupervisorSession(alive=False))
        self.assertFalse(
            sup.supervise(
                terminal=True,
                agentic=True,
                top_level=True,
                pending=False,
            )
        )

    def test_reset_nudges_on_tool_calls(self):
        sup = Supervisor(FakeSupervisorSession())
        sup.supervise(terminal=True, agentic=True, top_level=True, pending=False)
        sup.reset_nudges()
        self.assertEqual(sup.nudge_count, 0)

    def test_compacting_blocks_supervision(self):
        sup = Supervisor(FakeSupervisorSession(compacting=True))
        self.assertFalse(
            sup.supervise(
                terminal=True,
                agentic=True,
                top_level=True,
                pending=False,
            )
        )

    def test_non_agentic_does_not_nudge(self):
        sup = Supervisor(FakeSupervisorSession(tools=False))
        self.assertFalse(
            sup.supervise(
                terminal=True,
                agentic=False,
                top_level=True,
                pending=False,
            )
        )

    def test_non_top_level_does_not_nudge(self):
        sup = Supervisor(FakeSupervisorSession())
        self.assertFalse(
            sup.supervise(
                terminal=True,
                agentic=True,
                top_level=False,
                pending=False,
            )
        )

    def test_pending_tools_does_not_nudge(self):
        sup = Supervisor(FakeSupervisorSession())
        self.assertFalse(
            sup.supervise(
                terminal=True,
                agentic=True,
                top_level=True,
                pending=True,
            )
        )

    def test_non_terminal_does_not_nudge(self):
        sup = Supervisor(FakeSupervisorSession())
        self.assertFalse(
            sup.supervise(
                terminal=False,
                agentic=True,
                top_level=True,
                pending=False,
            )
        )
