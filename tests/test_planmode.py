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


if __name__ == "__main__":
    unittest.main()
