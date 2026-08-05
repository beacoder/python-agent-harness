import unittest

from python_agent_harness.fsm import (
    Fsm, FsmInfo, PendingToolCall, State, Supervisor,
    fail_pending_tool_calls, sanitize_tool_result,
)
from python_agent_harness.models import ToolCall


class FakeSession:
    def __init__(self, alive=True, tools=True, compacting=False):
        self.alive = alive
        self.tools_enabled = tools
        self.compacting = compacting


class TestFsm(unittest.TestCase):
    def test_transitions_to_done(self):
        fsm = Fsm(info=FsmInfo(tools=False))
        fsm.transition()  # INIT -> WAIT
        self.assertEqual(fsm.state, State.WAIT)
        fsm.transition()  # WAIT -> TYPE
        fsm.transition()  # TYPE -> DONE
        self.assertEqual(fsm.state, State.DONE)

    def test_transition_to_tool_on_pending(self):
        info = FsmInfo(tools=True)
        info.pending = [PendingToolCall(ToolCall(id="1", name="read", arguments="{}"))]
        fsm = Fsm(info=info)
        fsm.transition()  # INIT -> WAIT
        fsm.transition()  # WAIT -> TYPE
        fsm.transition()  # TYPE -> TOOL (pending)
        self.assertEqual(fsm.state, State.TOOL)
        fsm.transition()  # TOOL -> TRET
        self.assertEqual(fsm.state, State.TRET)
        info.pending = []
        fsm.transition()  # TRET -> WAIT
        self.assertEqual(fsm.state, State.WAIT)

    def test_terminal_supervision_nudges(self):
        session = FakeSession()
        sup = Supervisor(session)
        fsm = Fsm(info=FsmInfo(tools=True, top_level=True))
        effective = sup.supervise(fsm, State.DONE)
        self.assertEqual(effective, State.WAIT)
        self.assertEqual(sup.nudge_count, 1)

    def test_nudge_budget_exhausted(self):
        session = FakeSession()
        sup = Supervisor(session)
        fsm = Fsm(info=FsmInfo(tools=True, top_level=True))
        for _ in range(2):
            sup.supervise(fsm, State.DONE)
        self.assertEqual(sup.nudge_count, 2)
        effective = sup.supervise(fsm, State.DONE)
        self.assertEqual(effective, State.DONE)

    def test_dead_session_fails_closed(self):
        session = FakeSession(alive=False)
        sup = Supervisor(session)
        fsm = Fsm(info=FsmInfo(tools=True, top_level=True))
        effective = sup.supervise(fsm, State.DONE)
        self.assertEqual(effective, State.DONE)

    def test_reset_nudges_on_tool_calls(self):
        session = FakeSession()
        sup = Supervisor(session)
        fsm = Fsm(info=FsmInfo(tools=True, top_level=True))
        sup.supervise(fsm, State.DONE)
        sup.reset_nudges()
        self.assertEqual(sup.nudge_count, 0)

    def test_compacting_blocks_supervision(self):
        session = FakeSession(compacting=True)
        sup = Supervisor(session)
        fsm = Fsm(info=FsmInfo(tools=True, top_level=True))
        effective = sup.supervise(fsm, State.DONE)
        self.assertEqual(effective, State.DONE)

    def test_sanitize_tool_result(self):
        self.assertEqual(sanitize_tool_result(None), "Error: tool produced no result (it may have been interrupted or failed to return).")
        self.assertEqual(sanitize_tool_result(""), "")
        self.assertEqual(sanitize_tool_result("x"), "x")
        self.assertEqual(sanitize_tool_result(42), "42")

    def test_fail_pending_tool_calls(self):
        info = FsmInfo()
        call = ToolCall(id="1", name="bash", arguments="{}")
        info.pending = [PendingToolCall(call)]
        fail_pending_tool_calls(Fsm(info=info), ValueError("boom"))
        self.assertIn("Error: tool call failed", call.result)


if __name__ == "__main__":
    unittest.main()
