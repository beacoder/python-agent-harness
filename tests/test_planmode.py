import os
import tempfile
import unittest

from python_agent_harness.models import AgentMode
from python_agent_harness.planmode import PlanMode

PROMPTS = {
    "plan": "PLAN-PROMPT",
    "plan-mode": "PLAN-MODE ${planInfo}",
    "build-switch": "BUILD-SWITCH",
}


class TestPlanMode(unittest.TestCase):
    def test_initial_build(self):
        pm = PlanMode("/tmp/proj")
        self.assertEqual(pm.mode, AgentMode.BUILD)
        self.assertFalse(pm.is_plan)

    def test_switch_to_plan(self):
        with tempfile.TemporaryDirectory() as d:
            pm = PlanMode(d)
            pm.set_mode(AgentMode.PLAN, PROMPTS)
            self.assertTrue(pm.is_plan)
            self.assertTrue(pm.plan_file)
            self.assertTrue(os.path.exists(pm.plan_file))
            self.assertTrue(pm.plan_file.endswith("PLAN.md"))
            prompts = pm.consume_prompts()
            self.assertEqual(len(prompts), 2)
            self.assertEqual(prompts[0], "PLAN-PROMPT")
            self.assertIn(pm.plan_file, prompts[1])
            self.assertEqual(pm.consume_prompts(), [])

    def test_switch_to_build(self):
        pm = PlanMode("/tmp/proj")
        pm.set_mode(AgentMode.PLAN, PROMPTS)
        pm.set_mode(AgentMode.BUILD, PROMPTS)
        self.assertFalse(pm.is_plan)
        self.assertEqual(pm.consume_prompts(), ["BUILD-SWITCH"])

    def test_plan_file_truncated_on_switch(self):
        with tempfile.TemporaryDirectory() as d:
            pm = PlanMode(d)
            pm.set_mode(AgentMode.PLAN, PROMPTS)
            with open(pm.plan_file, "w") as f:
                f.write("draft")
            pm.set_mode(AgentMode.PLAN, PROMPTS)
            with open(pm.plan_file) as f:
                self.assertEqual(f.read(), "")

    def test_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            pm = PlanMode(d)
            pm.set_mode(AgentMode.PLAN, PROMPTS)
            path = pm.plan_file
            pm.cleanup_plan_file()
            self.assertFalse(os.path.exists(path))

    def test_plan_reminder(self):
        pm = PlanMode("/tmp/proj")
        pm.set_mode(AgentMode.PLAN, PROMPTS)
        reminder = pm.plan_reminder()
        self.assertIn("READ-ONLY", reminder)
        self.assertIn(pm.plan_file, reminder)


class TestPlanExitConfirm(unittest.TestCase):
    """plan_exit goes through the session confirm hook (a y/n choice UI
    in the TUI) — not through the Question tool's ask_questions."""

    class FakeClient:
        def chat(self, *a, **k):
            return None

        def chat_sync(self, *a, **k):
            return None

        def close(self):
            pass

    def make_session(self):
        from python_agent_harness.agent_session import AgentSession
        from python_agent_harness.tools import default_registry

        s = AgentSession(
            project_dir="/tmp/proj", client=self.FakeClient(), model="m",
            registry=default_registry(),
        )
        s.switch_to_plan()
        return s

    def test_plan_exit_uses_confirm_hook_not_ask_questions(self):
        s = self.make_session()
        seen = {}
        s.confirm_fn = lambda prompt: seen.setdefault("prompt", prompt) or True
        asked = []
        s.ask_fn = lambda questions: asked.append(questions) or "Unanswered"
        result = s.plan_exit()
        self.assertIn("approved", result)
        self.assertFalse(s.plan_mode.is_plan)
        self.assertIn("Plan at", seen["prompt"])
        self.assertIn("Switch to build agent", seen["prompt"])
        # the Question path must NOT be used for the plan approval
        self.assertEqual(asked, [])
        s.close()

    def test_plan_exit_rejected_stays_in_plan(self):
        s = self.make_session()
        s.confirm_fn = lambda prompt: False
        result = s.plan_exit()
        self.assertIn("rejected", result)
        self.assertTrue(s.plan_mode.is_plan)
        s.close()

    def test_plan_exit_noop_outside_plan(self):
        s = self.make_session()
        s.switch_to_build()
        s.confirm_fn = lambda prompt: True
        result = s.plan_exit()
        self.assertIn("Not in plan mode", result)
        self.assertFalse(s.plan_mode.is_plan)
        s.close()


if __name__ == "__main__":
    unittest.main()
