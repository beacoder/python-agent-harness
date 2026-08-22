import os
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)

from python_agent_harness.models import AgentMode
from python_agent_harness.planmode import PlanMode, _plan_temp_dir
from python_agent_harness.tools.base import Tool

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


class TestPlanTempDir(unittest.TestCase):
    def test_tmpdir_env_wins(self):
        with mock.patch.dict(os.environ, {"TMPDIR": "/custom/tmp"}, clear=False):
            self.assertEqual(_plan_temp_dir(), "/custom/tmp")

    def test_falls_back_to_plain_tmp(self):
        """With no TMPDIR/TMP/TEMP and an empty gettempdir(), the
        hard-coded /tmp fallback is used."""
        saved = {k: os.environ.pop(k, None) for k in ("TMPDIR", "TMP", "TEMP")}
        try:
            with mock.patch(
                "python_agent_harness.planmode.tempfile.gettempdir",
                return_value="",
            ):
                self.assertEqual(_plan_temp_dir(), "/tmp")
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_cleanup_tolerates_oserror(self):
        """A failing os.remove must not propagate: the file is left in
        place but the tracked plan_file is cleared."""
        with tempfile.TemporaryDirectory() as d:
            pm = PlanMode(d)
            pm.set_mode(AgentMode.PLAN, PROMPTS)
            path = pm.plan_file
            with mock.patch(
                "python_agent_harness.planmode.os.remove",
                side_effect=OSError("permission denied"),
            ):
                pm.cleanup_plan_file()
            self.assertIsNone(pm.plan_file)
            self.assertTrue(os.path.exists(path))


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
        from python_agent_harness.session import Session
        from python_agent_harness.tools import default_registry

        s = Session(
            project_dir="/tmp/proj",
            client=self.FakeClient(),
            model="m",
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

    def test_close_cleans_spooled_files(self):
        import python_agent_harness.tools.filesystem as fs

        s = self.make_session()
        with tempfile.TemporaryDirectory() as d:
            orig = fs._spool_dir
            fs._spool_dir = lambda: d
            try:
                big = "x" * (fs.MAX_OUTPUT + 1)
                result = fs._spool(big, "grep")
            finally:
                fs._spool_dir = orig
            m = re.search(r'file_path="([^"]+)"', result)
            self.assertIsNotNone(m, result)
            self.assertTrue(os.path.exists(m.group(1)))
            s.close()
            self.assertFalse(os.path.exists(m.group(1)))
            self.assertEqual(fs._spooled_files, [])


class TestPlanModeMCPGuard(unittest.TestCase):
    """Plan mode must refuse every mcp__ tool too: the harness cannot
    verify what an external server's tool does (the README's example
    server alone exposes write_file), so the read-only guarantee holds
    by blocking all of them."""

    class DummyMCPTool(Tool):
        name = "mcp__server__write_file"

        def run(self, args, ctx):
            return "SHOULD NOT RUN"

    def make_session(self):
        from python_agent_harness.session import Session
        from python_agent_harness.tools import default_registry

        s = Session(
            project_dir="/tmp/proj",
            client=TestPlanExitConfirm.FakeClient(),
            model="m",
            registry=default_registry(),
        )
        s.registry.register(self.DummyMCPTool())
        return s

    def test_plan_mode_blocks_mcp_tool(self):
        s = self.make_session()
        try:
            s.switch_to_plan()
            result = s.execute_tool("mcp__server__write_file", {})
            self.assertIn("blocked by plan mode", result)
            self.assertIn("MCP tools are disabled", result)
            self.assertNotIn("SHOULD NOT RUN", result)
        finally:
            s.close()

    def test_build_mode_allows_mcp_tool(self):
        s = self.make_session()
        try:
            s.switch_to_plan()
            s.switch_to_build()
            self.assertEqual(s.execute_tool("mcp__server__write_file", {}), "SHOULD NOT RUN")
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
