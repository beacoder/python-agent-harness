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
            "python_agent_harness.agent_session.find_context_dir",
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
            "python_agent_harness.agent_session.find_context_dir",
            return_value=None,
        ):
            session = cli.make_session(self._tmp.name, config_path=self._config_path)
        try:
            self.assertNotIn("Request context:", session.system_prompt)
        finally:
            session.close()

    def test_stream_defaults_true(self):
        """Streaming is the default unless the config file says otherwise."""
        session = cli.make_session(self._tmp.name, config_path=self._config_path)
        try:
            self.assertIs(session.stream, True)
        finally:
            session.close()

    def test_stream_false_from_config_file(self):
        """A config file with stream=false must reach the session."""
        cfg_path = Path(self._cfg_dir.name) / "config.toml"
        cfg_path.write_text('{"llm": {"stream": false}}', encoding="utf-8")
        session = cli.make_session(self._tmp.name, config_path=str(cfg_path))
        try:
            self.assertIs(session.stream, False)
        finally:
            session.close()

    def test_stream_param_overrides_config_file(self):
        """make_session's stream kwarg (--no-stream) beats the config file."""
        cfg_path = Path(self._cfg_dir.name) / "config.toml"
        cfg_path.write_text('{"llm": {"stream": true}}', encoding="utf-8")
        session = cli.make_session(
            self._tmp.name, config_path=str(cfg_path), stream=False
        )
        try:
            self.assertIs(session.stream, False)
        finally:
            session.close()

    def test_custom_commands_not_cli_subcommands(self):
        """Custom commands (e.g. explain) are TUI slash commands only —
        no CLI subcommand is registered for any of them.

        They must still resolve as SessionCommands so the TUI /explain
        keeps working (via commands.find_command).
        """
        from python_agent_harness.commands import (
            find_command, load_custom_commands,
        )

        parser = cli.build_parser()
        subparsers = next(
            a for a in parser._actions
            if a.__class__.__name__ == "_SubParsersAction"
        )
        for cmd in load_custom_commands():
            self.assertNotIn(cmd.name, subparsers.choices)
        self.assertIn("run", subparsers.choices)
        self.assertNotIn("review", subparsers.choices)
        self.assertNotIn("init", subparsers.choices)
        self.assertNotIn("sessions", subparsers.choices)
        self.assertNotIn("restore", subparsers.choices)
        cmd = find_command("explain")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.name, "explain")


class TestCommandToolAvailability(unittest.TestCase):
    """Tool availability per command type.

    - init/review: all tools except PlanExit (allow_planexit=False)
    - custom commands: all tools, incl. PlanExit (allow_planexit=True)
    - compact/summary: no tools (chat_sync without tools — covered by
      agent_session/client; nothing to configure here)
    """

    def test_init_and_review_forbid_planexit(self):
        from python_agent_harness.commands import (
            initialize_command, review_command,
        )

        self.assertFalse(initialize_command().allow_planexit)
        self.assertFalse(review_command().allow_planexit)

    def test_custom_commands_allow_planexit(self):
        from python_agent_harness.commands import (
            find_command, load_custom_commands,
        )

        customs = load_custom_commands()
        self.assertTrue(customs)  # explain.md etc. bundled
        for c in customs:
            self.assertTrue(c.allow_planexit, c.name)
        self.assertTrue(find_command("explain").allow_planexit)

    def test_hide_planexit_noop_without_registry(self):
        """Sessions without a registry (or without PlanExit) are untouched."""
        from python_agent_harness.commands import hide_planexit

        self.assertIsNone(hide_planexit(object()))

        class NoPlanExit:
            registry = None

        self.assertIsNone(hide_planexit(NoPlanExit()))

    def test_hide_planexit_removes_and_restores(self):
        from python_agent_harness.agent_session import AgentSession
        from python_agent_harness.commands import hide_planexit
        from python_agent_harness.tools import default_registry

        s = AgentSession(
            project_dir="/tmp", client=object(), model="m",
            registry=default_registry(),
        )
        try:
            s.switch_to_plan()  # registers PlanExit
            self.assertIsNotNone(s.registry.get("PlanExit"))

            restore = hide_planexit(s)
            self.assertIsNotNone(restore)
            self.assertIsNone(s.registry.get("PlanExit"))

            restore()
            self.assertIsNotNone(s.registry.get("PlanExit"))
        finally:
            s.close()


def _load(path, project_dir=None, with_context=False):
    from python_agent_harness.prompts import load_agent_prompt, load_context_files
    from python_agent_harness.agent_session import find_context_dir, find_skill_dir

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
