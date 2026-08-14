"""task-completion-rules.md must be auto-injected into the system prompt:
it is the LAST context piece, immediately before the actual agent prompt —
for the main agent, session commands, and the TUI slash path.  Sub-agents
are excluded by design: their system prompt is subagent.md ONLY.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from python_agent_harness import prompts
from python_agent_harness.prompts import (
    assemble_agent_prompt,
    load_task_completion_rules,
)


class TestTaskCompletionRules(unittest.TestCase):
    def test_rules_file_loaded(self):
        rules = load_task_completion_rules()
        self.assertIsNotNone(rules)
        self.assertIn("Task Completion Rules", rules)
        self.assertIn("MUST NOT stop execution", rules)

    def test_assemble_order_context_rules_agent(self):
        """Order: context files -> rules -> agent prompt."""
        with tempfile.TemporaryDirectory() as d:
            ctx = Path(d) / "contexts"
            ctx.mkdir()
            (ctx / "general-rules.md").write_text("GENERAL CONTEXT", encoding="utf-8")
            with mock.patch(
                "python_agent_harness.agent_session.find_context_dir",
                return_value=str(ctx),
            ):
                prompt = assemble_agent_prompt(d, "ACTUAL AGENT PROMPT")
            self.assertIsNotNone(prompt)
            i_ctx = prompt.index("GENERAL CONTEXT")
            i_rules = prompt.index("Task Completion Rules")
            i_agent = prompt.index("ACTUAL AGENT PROMPT")
            self.assertLess(i_ctx, i_rules)
            self.assertLess(i_rules, i_agent)

    def test_assemble_rules_before_agent_without_context(self):
        """include_context=False keeps the rules (this path is NOT used
        for sub-agents — they get subagent.md only)."""
        prompt = assemble_agent_prompt("/tmp", "AGENT PROMPT", include_context=False)
        self.assertLess(prompt.index("Task Completion Rules"), prompt.index("AGENT PROMPT"))

    def test_assemble_no_agent_prompt_still_has_rules(self):
        prompt = assemble_agent_prompt("/tmp", None, include_context=False)
        self.assertIsNotNone(prompt)
        self.assertIn("Task Completion Rules", prompt)

    def test_assemble_missing_rules_falls_back(self):
        with mock.patch.object(prompts, "load_task_completion_rules", return_value=None):
            self.assertEqual(
                assemble_agent_prompt("/tmp", "AGENT", include_context=False),
                "AGENT",
            )
            self.assertIsNone(assemble_agent_prompt("/tmp", None, include_context=False))

    def test_slash_commands_keep_rules_before_command_prompt(self):
        """/init and custom commands (TUI slash path): the command
        prompt is the run's system prompt; rules stay in front of it."""
        import io

        from rich.console import Console

        from python_agent_harness.agent_session import AgentSession
        from python_agent_harness.client import Client
        from python_agent_harness.tools import default_registry
        from python_agent_harness.tui import Tui

        session = AgentSession(
            project_dir="/tmp",
            client=Client(
                base_url="http://127.0.0.1:1/v1",
                api_key="x",
                model="m",
            ),
            model="m",
            registry=default_registry(),
        )
        tui = Tui(session, Console(file=io.StringIO(), width=100, force_terminal=False))
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["system"] = system

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/init")
            self.assertIn("Task Completion Rules", captured["system"])
            self.assertLess(
                captured["system"].index("Task Completion Rules"),
                captured["system"].index("AGENTS.md"),
            )
            tui._handle_slash("/explain client.py")
            self.assertIn("Task Completion Rules", captured["system"])
            self.assertLess(
                captured["system"].index("Task Completion Rules"),
                captured["system"].index("You are a senior engineer"),
            )
        session.close()

    def test_agent_loop_falls_back_to_session_prompt(self):
        """A bare run_agent_loop without a system prompt still uses the
        session's assembled prompt (rules included)."""
        from python_agent_harness.agent import AgentLoop
        from python_agent_harness.agent_session import AgentSession
        from python_agent_harness.models import Message
        from python_agent_harness.tools import default_registry

        s = AgentSession(
            project_dir="/tmp",
            client=object(),
            model="m",
            system_prompt=assemble_agent_prompt("/tmp", "AGENT", include_context=False),
            registry=default_registry(),
        )
        loop = AgentLoop(s, messages=[Message(role="user", content="hi")])
        self.assertIn("Task Completion Rules", loop.system)
        self.assertLess(
            loop.system.index("Task Completion Rules"),
            loop.system.index("AGENT"),
        )
        # explicit system still wins
        loop2 = AgentLoop(s, messages=[Message(role="user", content="hi")], system="EXPLICIT")
        self.assertEqual(loop2.system, "EXPLICIT")


if __name__ == "__main__":
    unittest.main()
