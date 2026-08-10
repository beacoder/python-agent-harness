"""Tests for session commands (commands.py): project-root discovery,
prompt/kickoff preparation, custom-command loading and lookup."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from python_agent_harness import commands
from python_agent_harness.commands import (
    SessionCommand,
    _project_root,
    find_command,
    initialize_command,
    load_custom_commands,
)


class TestProjectRoot(unittest.TestCase):
    def test_git_root_returned_when_git_succeeds(self):
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="/some/git/root\n")
            self.assertEqual(_project_root("/work/proj"), "/some/git/root")
        run.assert_called_once()

    def test_falls_back_to_dir_with_dot_git_when_git_fails(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, ".git").mkdir()
            with mock.patch("subprocess.run") as run:
                run.return_value = mock.Mock(returncode=1, stdout="")
                self.assertEqual(_project_root(d), str(Path(d).resolve()))

    def test_falls_back_to_dir_with_agents_md_on_oserror(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "AGENTS.md").write_text("# rules\n", encoding="utf-8")
            with mock.patch("subprocess.run", side_effect=OSError("git missing")):
                self.assertEqual(_project_root(d), str(Path(d).resolve()))

    def test_returns_cwd_when_nothing_found(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("git", 10)
            ):
                self.assertEqual(_project_root(d), d)


class TestSessionCommandPrepare(unittest.TestCase):
    def test_prepare_substitutes_path_and_arguments(self):
        cmd = SessionCommand(
            name="explain",
            prompt_file="commands/explain.md",
            kickoff="Proceed with the task described in your instructions.\n",
            buffer_name="*gptel-agent-explain*",
            status=" Running explain...",
        )
        with tempfile.TemporaryDirectory() as d:
            cwd, prompt, kickoff = cmd.prepare(project_dir=d, extra="client.py")
        self.assertEqual(cwd, d)
        self.assertIn(d, prompt)             # ${path} substituted
        self.assertIn("client.py", prompt)   # $ARGUMENTS substituted
        self.assertEqual(
            kickoff, "Proceed with the task described in your instructions.\n"
        )

    def test_prepare_replaces_path_in_kickoff(self):
        cmd = initialize_command()
        with tempfile.TemporaryDirectory() as d:
            cwd, prompt, kickoff = cmd.prepare(project_dir=d)
        self.assertEqual(cwd, d)
        self.assertIn(d, kickoff)
        self.assertNotIn("${path}", kickoff)

    def test_prepare_without_project_dir_uses_project_root(self):
        cmd = initialize_command()
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="/git/root\n")
            cwd, prompt, kickoff = cmd.prepare()
        self.assertEqual(cwd, "/git/root")


class TestLoadCustomCommands(unittest.TestCase):
    def test_missing_commands_dir_returns_empty(self):
        with mock.patch.object(
            commands, "COMMANDS_DIR", Path("/nonexistent/commands")
        ):
            self.assertEqual(load_custom_commands(), [])

    def test_skips_files_with_empty_names_and_builds_commands(self):
        with tempfile.TemporaryDirectory() as d:
            prompts_dir = Path(d) / "prompts"
            cmds_dir = prompts_dir / "commands"
            cmds_dir.mkdir(parents=True)
            (cmds_dir / "explain.md").write_text("explain prompt", encoding="utf-8")
            # "---" strips to an empty name: must be skipped, not crash
            (cmds_dir / "---.md").write_text("ignored", encoding="utf-8")
            with mock.patch.object(commands, "COMMANDS_DIR", cmds_dir), \
                 mock.patch.object(commands, "PROMPTS_DIR", prompts_dir):
                loaded = load_custom_commands()
        self.assertEqual([c.name for c in loaded], ["explain"])
        self.assertEqual(loaded[0].prompt_file, "commands/explain.md")


class TestFindCommand(unittest.TestCase):
    def test_builtin_commands(self):
        self.assertEqual(find_command("init").name, "initialize")
        self.assertEqual(find_command("review").name, "review")

    def test_custom_command_found(self):
        cmd = find_command("explain")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.name, "explain")

    def test_unknown_command_returns_none(self):
        self.assertIsNone(find_command("no-such-command"))


if __name__ == "__main__":
    unittest.main()
