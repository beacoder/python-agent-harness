"""Tests for cli.make_session's default agent-prompt wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from python_agent_harness import cli, config


class TestMakeSessionPromptDefaults(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cfg_dir = tempfile.TemporaryDirectory()
        # point config file resolution at an empty dir so no real
        # api key / base_url leaks into these tests
        self._config_path = str(Path(self._cfg_dir.name) / "config.toml")

    def tearDown(self):
        self._tmp.cleanup()
        self._cfg_dir.cleanup()

    def test_defaults_to_main_agent_prompt_when_system_not_given(self):
        session = cli.make_session(self._tmp.name, config_path=self._config_path)
        try:
            main = _load(config.DEFAULT_AGENT_PROMPT_FILE, project_dir=self._tmp.name)
            self.assertIn(main, session.system_prompt)          # agent prompt present
            self.assertIn("Task Completion Rules", session.system_prompt)  # rules injected
            self.assertLess(  # rules are the last context piece, before the prompt
                session.system_prompt.index("Task Completion Rules"),
                session.system_prompt.index(main[:20]),
            )
        finally:
            session.close()

    def test_subagent_prompt_always_loaded(self):
        """Sub-agents get ONLY their own system prompt — no parent
        context and no task-completion rules."""
        session = cli.make_session(
            self._tmp.name, config_path=self._config_path,
        )
        try:
            expected = _load(config.DEFAULT_SUBAGENT_PROMPT_FILE, project_dir=self._tmp.name)
            self.assertEqual(session.subagent_system_prompt, expected)
            self.assertNotIn("Task Completion Rules", session.subagent_system_prompt)
            self.assertNotIn("Request context:", session.subagent_system_prompt)
        finally:
            session.close()

    def test_main_and_subagent_prompts_differ(self):
        session = cli.make_session(self._tmp.name, config_path=self._config_path)
        try:
            self.assertIsNotNone(session.system_prompt)
            self.assertIsNotNone(session.subagent_system_prompt)
            self.assertNotEqual(session.system_prompt, session.subagent_system_prompt)
        finally:
            session.close()

    def test_context_dir_files_prepended_to_system_prompt(self):
        """Files in a contexts/ dir are prepended to system_prompt."""
        import unittest.mock as mock

        ctx_dir = Path(self._tmp.name) / "contexts"
        ctx_dir.mkdir()
        (ctx_dir / "notes.md").write_text("# My Notes\nHello world\n", encoding="utf-8")
        with mock.patch(
            "python_agent_harness.harness.find_context_dir",
            return_value=str(ctx_dir),
        ):
            session = cli.make_session(self._tmp.name, config_path=self._config_path)
        try:
            self.assertIn("Request context:", session.system_prompt)
            self.assertIn("In file `", session.system_prompt)
            self.assertIn("# My Notes", session.system_prompt)
            self.assertIn("Hello world", session.system_prompt)
        finally:
            session.close()

    def test_no_context_dir_no_prefix(self):
        """Without a contexts/ dir, system_prompt has no context prefix."""
        import unittest.mock as mock

        with mock.patch(
            "python_agent_harness.harness.find_context_dir",
            return_value=None,
        ):
            session = cli.make_session(self._tmp.name, config_path=self._config_path)
        try:
            self.assertNotIn("Request context:", session.system_prompt)
        finally:
            session.close()


def _load(path, project_dir=None, with_context=False):
    from python_agent_harness.compaction import load_agent_prompt, load_context_files
    from python_agent_harness.harness import find_context_dir, find_skill_dir

    skill_dir = find_skill_dir(project_dir) if project_dir else None
    prompt = load_agent_prompt(path, skill_dir=skill_dir)
    if with_context:
        context_dir = find_context_dir(project_dir) if project_dir else None
        context_block = load_context_files(context_dir)
        if context_block and prompt:
            return context_block + "\n\n" + prompt
        if context_block:
            return context_block
    return prompt


if __name__ == "__main__":
    unittest.main()
