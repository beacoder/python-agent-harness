"""End-to-end agent loop tests with a fake client and fake session."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from python_agent_harness.core.agent import Supervisor
from tests.support import (
    plan_cleanup,  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
    session_sandbox,  # noqa: F401,E402  (side-effect: redirect SESSION_DIR)
)
from tests.support.agent_test_utils import (  # noqa: E402,F401
    FakeClient,
    ParallelToolSession,
    RealParallelSession,
    RecordingSession,
    SerialPromptSession,
    StaggeredSession,
    agent_call,
)


class FakeSupervisorSession:
    def __init__(self, alive=True, tools=True, compacting=False, config_path=None):
        self.alive = alive
        self.tools_enabled = tools
        self.compacting = compacting
        self.config_path = config_path


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
        budget = sup.max_nudges
        self.assertGreater(budget, 0, "default budget must allow at least one nudge")
        for _ in range(budget):
            sup.supervise(terminal=True, agentic=True, top_level=True, pending=False)
        self.assertEqual(sup.nudge_count, budget)
        self.assertFalse(
            sup.supervise(
                terminal=True,
                agentic=True,
                top_level=True,
                pending=False,
            )
        )

    def test_default_budget_is_one(self):
        """One nudge per tool round: enough for a model that stops early
        occasionally, without re-verifying every correct answer twice."""
        self.assertEqual(Supervisor(FakeSupervisorSession()).max_nudges, 1)

    def test_budget_read_from_config(self):
        """A weaker model that keeps stopping early can raise the budget
        via supervision.max_nudges."""
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"supervision": {"max_nudges": 4}}, f)
            path = f.name
        try:
            sup = Supervisor(FakeSupervisorSession(config_path=path))
            self.assertEqual(sup.max_nudges, 4)
        finally:
            os.unlink(path)

    def test_zero_disables_supervision(self):
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"supervision": {"max_nudges": 0}}, f)
            path = f.name
        try:
            sup = Supervisor(FakeSupervisorSession(config_path=path))
            self.assertFalse(
                sup.supervise(terminal=True, agentic=True, top_level=True, pending=False)
            )
        finally:
            os.unlink(path)

    def test_malformed_budget_raises(self):
        """A config error must surface, not silently fall back."""
        import json
        import tempfile

        for bad in ("two", -1, True, 1.5):
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump({"supervision": {"max_nudges": bad}}, f)
                path = f.name
            try:
                with self.assertRaises(ValueError, msg=f"accepted {bad!r}"):
                    Supervisor(FakeSupervisorSession(config_path=path))
            finally:
                os.unlink(path)

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
