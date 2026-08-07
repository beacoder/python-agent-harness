"""task-completion-rules.txt must be auto-injected into the system prompt:
it is the LAST context piece, immediately before the actual agent prompt —
for the main agent, sub-agents, session commands, and the TUI slash path.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from python_agent_harness import prompts, config
from python_agent_harness.prompts import (
    assemble_agent_prompt,
    load_agent_prompt,
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
        prompt = assemble_agent_prompt("/tmp", "AGENT PROMPT", include_context=False)
        self.assertLess(
            prompt.index("Task Completion Rules"), prompt.index("AGENT PROMPT")
        )

    def test_assemble_no_agent_prompt_still_has_rules(self):
        prompt = assemble_agent_prompt("/tmp", None, include_context=False)
        self.assertIsNotNone(prompt)
        self.assertIn("Task Completion Rules", prompt)

    def test_assemble_missing_rules_falls_back(self):
        with mock.patch.object(
            prompts, "load_task_completion_rules", return_value=None
        ):
            self.assertEqual(
                assemble_agent_prompt("/tmp", "AGENT", include_context=False),
                "AGENT",
            )
            self.assertIsNone(
                assemble_agent_prompt("/tmp", None, include_context=False)
            )

    def test_make_session_injects_rules(self):
        """`run` sessions: rules before the main agent prompt; the
        sub-agent prompt is NOT touched (no rules, no parent context)."""
        from python_agent_harness.cli import make_session

        with tempfile.TemporaryDirectory() as d:
            s = make_session(d)
            try:
                self.assertIsNotNone(s.system_prompt)
                self.assertIn("Task Completion Rules", s.system_prompt)
                marker = load_agent_prompt(config.DEFAULT_AGENT_PROMPT_FILE)
                marker = marker[: marker.index("\n")] if marker else "You are"
                self.assertLess(
                    s.system_prompt.index("Task Completion Rules"),
                    s.system_prompt.index(marker),
                )
                self.assertNotIn("Task Completion Rules", s.subagent_system_prompt)
            finally:
                s.close()

    def test_adopt_keeps_rules_before_command_prompt(self):
        """init/review/custom commands: the command prompt is the agent
        prompt; rules stay in front of it."""
        from python_agent_harness.cli import _adopt

        class Stub:
            project_dir = "/tmp"

        s = Stub()
        _adopt(s, {"system_prompt": "COMMAND PROMPT"})
        self.assertIn("Task Completion Rules", s.system_prompt)
        self.assertLess(
            s.system_prompt.index("Task Completion Rules"),
            s.system_prompt.index("COMMAND PROMPT"),
        )

    def test_command_run_loop_system_includes_rules(self):
        from python_agent_harness.commands import initialize_command

        with mock.patch("python_agent_harness.commands.run_agent_loop") as m:
            with tempfile.TemporaryDirectory() as d:
                cmd = initialize_command()
                cmd.run(
                    session_factory=lambda **kw: object(),
                    project_dir=d,
                    extra=None,
                )
            _, kwargs = m.call_args
            self.assertIn("Task Completion Rules", kwargs["system"])
            self.assertLess(
                kwargs["system"].index("Task Completion Rules"),
                kwargs["system"].index("AGENTS.md"),
            )

    def test_agent_loop_falls_back_to_session_prompt(self):
        """A bare run_agent_loop without a system prompt still uses the
        session's assembled prompt (rules included)."""
        from python_agent_harness.agent import AgentLoop
        from python_agent_harness.agent_session import AgentSession
        from python_agent_harness.models import Message
        from python_agent_harness.tools import default_registry

        s = AgentSession(
            project_dir="/tmp", client=object(), model="m",
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
        loop2 = AgentLoop(
            s, messages=[Message(role="user", content="hi")], system="EXPLICIT"
        )
        self.assertEqual(loop2.system, "EXPLICIT")


if __name__ == "__main__":
    unittest.main()
