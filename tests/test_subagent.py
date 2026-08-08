"""Tests for sub-agent system-prompt selection (subagent.run_subagent)."""

from __future__ import annotations

import tempfile
import unittest

from python_agent_harness.agent_session import AgentSession
from python_agent_harness.models import Message, Usage
from python_agent_harness.session_store import SessionStore
from python_agent_harness.subagent import run_subagent
from python_agent_harness.tools import default_registry


class SpyClient:
    """Records the ``system`` kwarg passed to chat(); returns a fixed reply."""

    def __init__(self):
        self.systems: list[str | None] = []

    def chat(self, messages, tools=None, system=None, temperature=None,
              max_tokens=None, reasoning_effort=None, on_delta=None, stream=True):
        self.systems.append(system)
        return Message(role="assistant", content="sub-agent done"), Usage(input_tokens=10)

    def chat_sync(self, messages, system=None, temperature=None, max_tokens=None,
                  reasoning_effort=None):
        return Message(role="assistant", content="SYNC-OK"), Usage()


def make_session(system_prompt, subagent_system_prompt, session_dir):
    import python_agent_harness.config as cfg
    from pathlib import Path

    cfg.SESSION_DIR = Path(session_dir)
    session = AgentSession(
        project_dir="/tmp/fakeproj",
        client=SpyClient(),
        model="gpt-5-mini",
        system_prompt=system_prompt,
        subagent_system_prompt=subagent_system_prompt,
        registry=default_registry(),
    )
    session.store = SessionStore(
        project_dir="/tmp/fakeproj", model=session.model, backend=session.backend,
        system_prompt=session.system_prompt, temperature=session.temperature,
        max_tokens=session.max_tokens, tool_names=session.store.tool_names,
    )
    return session


class TestSubagentPromptSelection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_subagent_uses_its_own_prompt_not_parent(self):
        session = make_session("MAIN AGENT PROMPT", "SUBAGENT PROMPT", self._tmp.name)
        result = run_subagent(session, "task", "do something")
        self.assertEqual(result, "sub-agent done")
        self.assertEqual(session.client.systems, ["SUBAGENT PROMPT"])

    def test_falls_back_to_default_subagent_prompt_when_missing(self):
        """Without a configured subagent prompt, the DEFAULT bundled
        subagent prompt is used — never the parent's main prompt."""
        session = make_session("MAIN AGENT PROMPT", None, self._tmp.name)
        run_subagent(session, "task", "do something")
        from python_agent_harness import config as cfg
        from python_agent_harness.prompts import load_agent_prompt

        default_sub = load_agent_prompt(cfg.DEFAULT_SUBAGENT_PROMPT_FILE)
        self.assertEqual(session.client.systems, [default_sub])
        self.assertNotIn("MAIN AGENT PROMPT", session.client.systems[0])

    def test_plan_mode_reminder_prepended_but_prompt_still_subagent(self):
        session = make_session("MAIN", "SUB", self._tmp.name)
        session.plan_mode.set_mode(session.plan_mode.mode.PLAN, {
            "plan": "P1", "plan-mode": "P2", "build-switch": "B",
        })
        run_subagent(session, "task", "do something")
        # plan mode changes the *messages*, not which system prompt is used
        self.assertEqual(session.client.systems, ["SUB"])

    def test_errors_are_contained_not_raised(self):
        session = make_session("MAIN", "SUB", self._tmp.name)

        class Boom:
            @property
            def is_plan(self):
                raise RuntimeError("boom")

        session.plan_mode = Boom()
        result = run_subagent(session, "risky task", "do it")
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("risky task", result)

    def test_default_session_does_not_auto_load_prompt(self):
        """A bare AgentSession() must NOT silently auto-load the bundled
        subagent prompt — defaulting is cli.make_session's responsibility
        (covered in test_cli.py); the bundled files themselves are
        checked in test_prompts.py."""
        session = make_session("MAIN AGENT PROMPT", None, self._tmp.name)
        self.assertIsNone(session.subagent_system_prompt)


if __name__ == "__main__":
    unittest.main()
