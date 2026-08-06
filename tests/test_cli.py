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
            self.assertEqual(session.system_prompt, main)
        finally:
            session.close()

    def test_subagent_prompt_always_loaded(self):
        session = cli.make_session(
            self._tmp.name, config_path=self._config_path,
        )
        try:
            expected = _load(config.DEFAULT_SUBAGENT_PROMPT_FILE, project_dir=self._tmp.name)
            self.assertEqual(session.subagent_system_prompt, expected)
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


def _load(path, project_dir=None):
    from python_agent_harness.compaction import load_agent_prompt
    from python_agent_harness.harness import find_skill_dir

    skill_dir = find_skill_dir(project_dir) if project_dir else None
    return load_agent_prompt(path, skill_dir=skill_dir)


if __name__ == "__main__":
    unittest.main()
