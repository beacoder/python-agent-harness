"""Tests for cli.make_session's default agent-prompt wiring."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)

from python_agent_harness import cli, config


class TestMakeSessionPromptDefaults(unittest.TestCase):
    ENV_KEYS = [
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BACKEND",
        "OPENAI_SUBAGENT_BASE_URL",
        "OPENAI_SUBAGENT_API_KEY",
        "OPENAI_SUBAGENT_MODEL",
        "OPENAI_SUBAGENT_BACKEND",
    ]

    def setUp(self):
        self._saved_env = {k: os.environ.get(k) for k in self.ENV_KEYS}
        for k in self.ENV_KEYS:
            os.environ.pop(k, None)
        self._tmp = tempfile.TemporaryDirectory()
        self._cfg_dir = tempfile.TemporaryDirectory()
        # point config file resolution at an empty dir so no real
        # api key / base_url leaks into these tests
        self._config_path = str(Path(self._cfg_dir.name) / "config.toml")

    def tearDown(self):
        self._tmp.cleanup()
        self._cfg_dir.cleanup()
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_defaults_to_main_agent_prompt_when_system_not_given(self):
        session = cli.make_session(self._tmp.name, config_path=self._config_path)
        try:
            main = _load(config.DEFAULT_AGENT_PROMPT_FILE, project_dir=self._tmp.name)
            self.assertIn(main, session.system_prompt)  # agent prompt present
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
            self._tmp.name,
            config_path=self._config_path,
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
        session = cli.make_session(self._tmp.name, config_path=str(cfg_path), stream=False)
        try:
            self.assertIs(session.stream, False)
        finally:
            session.close()

    def test_no_stream_flag_applies_to_subagents_too(self):
        """--no-stream is a session-wide flag: it must beat an explicit
        subagent_llm.stream config as well, mirroring the main agent's
        precedence (CLI flag > config file)."""
        cfg_path = Path(self._cfg_dir.name) / "config.toml"
        cfg_path.write_text(
            '{"llm": {"stream": true}, "subagent_llm": {"model": "sub-m", "stream": true}}',
            encoding="utf-8",
        )
        session = cli.make_session(self._tmp.name, config_path=str(cfg_path), stream=False)
        try:
            self.assertIs(session.stream, False)
            self.assertIs(session.subagent_stream, False)
        finally:
            session.close()

    def test_subagent_stream_inherits_main_without_cli_flag(self):
        """Without --no-stream, an unset subagent_llm.stream inherits the
        main config value."""
        cfg_path = Path(self._cfg_dir.name) / "config.toml"
        cfg_path.write_text(
            '{"llm": {"stream": false}, "subagent_llm": {"model": "sub-m"}}',
            encoding="utf-8",
        )
        session = cli.make_session(self._tmp.name, config_path=str(cfg_path))
        try:
            self.assertIs(session.stream, False)
            self.assertIs(session.subagent_stream, False)
        finally:
            session.close()

    def test_subagent_llm_inherits_main_client_by_default(self):
        """Without subagent_llm overrides, the sub-agent shares the
        main client (mirrors gptel-agent-harness: nil backend/model
        inherit the main agent's)."""
        session = cli.make_session(self._tmp.name, config_path=self._config_path)
        try:
            self.assertIs(session.subagent_client, session.client)
            self.assertIs(session.subagent_temperature, session.temperature)
            self.assertIs(session.subagent_max_tokens, session.max_tokens)
            self.assertIs(session.subagent_reasoning_effort, session.reasoning_effort)
            self.assertIs(session.subagent_stream, session.stream)
        finally:
            session.close()

    def test_subagent_llm_separate_client_from_config_file(self):
        """A subagent_llm block with a different model/base_url creates
        a dedicated sub-agent client; unset per-request options inherit
        the main settings."""
        cfg_path = Path(self._cfg_dir.name) / "config.toml"
        cfg_path.write_text(
            '{"llm": {"base_url": "https://main.example/v1",'
            ' "api_key": "sk-main", "model": "main-model",'
            ' "reasoning_effort": "high"},'
            ' "subagent_llm": {"base_url": "https://sub.example/v1",'
            ' "api_key": "sk-sub", "model": "cheap-model"}}',
            encoding="utf-8",
        )
        session = cli.make_session(self._tmp.name, config_path=str(cfg_path))
        try:
            self.assertIsNot(session.subagent_client, session.client)
            self.assertEqual(session.client.model, "main-model")
            self.assertEqual(session.client.base_url, "https://main.example/v1")
            self.assertEqual(session.subagent_client.model, "cheap-model")
            self.assertEqual(session.subagent_client.base_url, "https://sub.example/v1")
            self.assertEqual(session.subagent_client.api_key, "sk-sub")
            # per-request options unset in subagent_llm inherit main
            self.assertEqual(session.subagent_reasoning_effort, "high")
            self.assertIs(session.subagent_stream, session.stream)
        finally:
            session.close()

    def test_subagent_llm_env_override(self):
        """OPENAI_SUBAGENT_* env vars win over the config file for the
        sub-agent LLM."""
        import unittest.mock as mock

        cfg_path = Path(self._cfg_dir.name) / "config.toml"
        cfg_path.write_text(
            '{"llm": {"model": "main-model"}, "subagent_llm": {"model": "file-sub"}}',
            encoding="utf-8",
        )
        with mock.patch.dict(os.environ, {"OPENAI_SUBAGENT_MODEL": "env-sub"}, clear=False):
            session = cli.make_session(self._tmp.name, config_path=str(cfg_path))
        try:
            self.assertEqual(session.subagent_client.model, "env-sub")
        finally:
            session.close()

    def test_subagent_llm_profile_from_models_section(self):
        """subagent_llm.profile reuses a models profile: the dedicated
        sub-agent client is built from the profile's settings, and the
        main client keeps the main llm settings."""
        cfg_path = Path(self._cfg_dir.name) / "config.toml"
        cfg_path.write_text(
            '{"llm": {"base_url": "https://main.example/v1",'
            ' "api_key": "sk-main", "model": "main-model"},'
            ' "models": {"cheap": {"base_url": "https://cheap.example/v1",'
            ' "api_key": "sk-cheap", "model": "cheap-model"}},'
            ' "subagent_llm": {"profile": "cheap"}}',
            encoding="utf-8",
        )
        session = cli.make_session(self._tmp.name, config_path=str(cfg_path))
        try:
            self.assertEqual(session.client.model, "main-model")
            self.assertEqual(session.client.base_url, "https://main.example/v1")
            self.assertIsNotNone(session.subagent_client)
            self.assertEqual(session.subagent_client.model, "cheap-model")
            self.assertEqual(session.subagent_client.base_url, "https://cheap.example/v1")
            self.assertEqual(session.subagent_client.api_key, "sk-cheap")
        finally:
            session.close()

    def test_custom_commands_not_cli_subcommands(self):
        """Custom commands (e.g. explain) are TUI slash commands only —
        no CLI subcommand is registered for any of them.

        They must still resolve as SessionCommands so the TUI /explain
        keeps working (via commands.find_command).
        """
        from python_agent_harness.commands import (
            find_command,
            load_custom_commands,
        )

        parser = cli.build_parser()
        subparsers = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
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
            initialize_command,
            review_command,
        )

        self.assertFalse(initialize_command().allow_planexit)
        self.assertFalse(review_command().allow_planexit)

    def test_custom_commands_allow_planexit(self):
        from python_agent_harness.commands import (
            find_command,
            load_custom_commands,
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
            project_dir="/tmp",
            client=object(),
            model="m",
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


class TestCliEntryPoints(unittest.TestCase):
    """main()/cmd_run(): TUI startup and exit-path handling, with the
    heavy dependencies (Tui, make_session) mocked out so the tests are
    fast and never start a real TUI."""

    def test_main_run_starts_tui_and_closes_session(self):
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.cli.make_session_with_mcp", return_value=session
            ) as ms,
            mock.patch("python_agent_harness.tui.Tui") as tui_cls,
        ):
            rc = cli.main(["run", "/tmp/proj"])
        self.assertEqual(rc, 0)
        ms.assert_called_once()
        self.assertEqual(ms.call_args.args[0], "/tmp/proj")
        tui_cls.assert_called_once_with(session)
        tui_cls.return_value.run.assert_called_once()
        session.close.assert_called_once()

    def test_main_no_command_defaults_to_run(self):
        """With no subcommand, main() dispatches to cmd_run and the
        project defaults to the current directory (the run subparser
        attrs are absent when argv is empty, so cmd_run must not touch
        args.project blindly)."""
        import unittest.mock as mock

        session = mock.Mock()
        with (
            mock.patch(
                "python_agent_harness.cli.make_session_with_mcp", return_value=session
            ) as ms,
            mock.patch("python_agent_harness.tui.Tui") as tui_cls,
        ):
            rc = cli.main([])
        self.assertEqual(rc, 0)
        self.assertEqual(ms.call_args.args[0], os.getcwd())
        tui_cls.return_value.run.assert_called_once()
        session.close.assert_called_once()

    def test_run_no_stream_flag_forwarded(self):
        """--no-stream on run must reach make_session as stream=False."""
        import unittest.mock as mock

        with (
            mock.patch("python_agent_harness.cli.make_session_with_mcp") as ms,
            mock.patch("python_agent_harness.tui.Tui"),
        ):
            rc = cli.main(["run", "/tmp/proj", "--no-stream"])
        self.assertEqual(rc, 0)
        self.assertIs(ms.call_args.kwargs["stream"], False)

    def test_main_unknown_command_prints_help_and_returns_1(self):
        """An unhandled command must print the parser help and exit 1
        (argparse normally rejects unknown subcommands before this)."""
        import argparse
        import unittest.mock as mock

        parser = mock.Mock()
        parser.parse_args.return_value = argparse.Namespace(command="bogus")
        with mock.patch.object(cli, "build_parser", return_value=parser):
            rc = cli.main([])
        self.assertEqual(rc, 1)
        parser.print_help.assert_called_once()

    def test_module_main_calls_sys_exit(self):
        """`python -m python_agent_harness.cli` must call sys.exit(main()).
        The module is executed as __main__ with main() stubbed out so no
        session or TUI machinery runs; the exit code is the stub's."""
        src = Path(cli.__file__).read_text(encoding="utf-8")
        # stub the module's own main so nothing real is started
        src = src.replace("sys.exit(main())", "sys.exit(__test_main__())")
        calls = []

        def __test_main__(argv=None):
            calls.append(argv)
            return 7

        ns = {
            "__name__": "__main__",
            "__file__": str(cli.__file__),
            "__package__": "python_agent_harness",
            "__test_main__": __test_main__,
        }
        with self.assertRaises(SystemExit) as cm:
            exec(compile(src, str(cli.__file__), "exec"), ns)
        self.assertEqual(cm.exception.code, 7)
        self.assertEqual(calls, [None])


def _load(path, project_dir=None, with_context=False):
    from python_agent_harness.agent_session import find_context_dir, find_skill_dir
    from python_agent_harness.prompts import load_agent_prompt, load_context_files

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
