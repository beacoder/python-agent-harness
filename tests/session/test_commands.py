"""Tests for session commands (commands.py): project-root discovery,
prompt/kickoff preparation, custom-command loading and lookup."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import tests  # noqa: F401,E402  (side-effect: hermetic config, see tests/__init__.py)
from python_agent_harness.prompts import _git_toplevel
from python_agent_harness.session import commands
from python_agent_harness.session.commands import (
    SessionCommand,
    _project_root,
    find_command,
    initialize_command,
    load_custom_commands,
)


class TestProjectRoot(unittest.TestCase):
    def test_git_root_returned_when_git_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, ".git").mkdir()
            with mock.patch("subprocess.run") as run:
                run.return_value = mock.Mock(returncode=0, stdout="/some/git/root\n")
                self.assertEqual(_project_root(d), "/some/git/root")
            run.assert_called_once()

    def test_git_not_invoked_outside_a_repository(self):
        # no .git anywhere above: no git subprocess, AGENTS.md/cwd fallback
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch("subprocess.run") as run,
        ):
            self.assertEqual(_project_root(d), d)
        run.assert_not_called()

    def test_nested_agents_md_inside_repo_resolves_to_repo_root(self):
        # agrees with prompts.core._git_toplevel (project-context loading):
        # an AGENTS.md in a subdirectory must not shadow the repo root
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / ".git").mkdir()
            sub = root / "pkg" / "sub"
            sub.mkdir(parents=True)
            (root / "pkg" / "AGENTS.md").write_text("# rules\n", encoding="utf-8")
            with mock.patch("subprocess.run", side_effect=OSError("git missing")):
                self.assertEqual(_project_root(str(sub)), str(root))
                self.assertEqual(_project_root(str(sub)), _git_toplevel(str(sub)))

    def test_agents_md_ancestor_used_outside_repo(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "AGENTS.md").write_text("# rules\n", encoding="utf-8")
            sub = root / "a" / "b"
            sub.mkdir(parents=True)
            self.assertEqual(_project_root(str(sub)), str(root))

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
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 10)),
        ):
            self.assertEqual(_project_root(d), d)


class TestSessionCommandPrepare(unittest.TestCase):
    def test_prepare_substitutes_path_and_arguments(self):
        cmd = SessionCommand(
            name="explain",
            prompt_file="commands/explain.md",
            kickoff="Proceed with the task described in your instructions.\n",
            status=" Running explain...",
        )
        with tempfile.TemporaryDirectory() as d:
            cwd, prompt, kickoff = cmd.prepare(project_dir=d, extra="client.py")
        self.assertEqual(cwd, d)
        self.assertIn(d, prompt)  # ${path} substituted
        self.assertIn("client.py", prompt)  # $ARGUMENTS substituted
        self.assertEqual(kickoff, "Proceed with the task described in your instructions.\n")

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
        with mock.patch.object(commands, "COMMANDS_DIR", Path("/nonexistent/commands")):
            self.assertEqual(load_custom_commands(), [])

    def test_skips_files_with_empty_names_and_builds_commands(self):
        with tempfile.TemporaryDirectory() as d:
            prompts_dir = Path(d) / "prompts"
            cmds_dir = prompts_dir / "commands"
            cmds_dir.mkdir(parents=True)
            (cmds_dir / "explain.md").write_text("explain prompt", encoding="utf-8")
            # "---" strips to an empty name: must be skipped, not crash
            (cmds_dir / "---.md").write_text("ignored", encoding="utf-8")
            with (
                mock.patch.object(commands, "COMMANDS_DIR", cmds_dir),
                mock.patch.object(commands, "PROMPTS_DIR", prompts_dir),
            ):
                loaded = load_custom_commands()
        self.assertEqual([c.name for c in loaded], ["explain"])
        self.assertEqual(loaded[0].prompt_file.replace(os.sep, "/"), "commands/explain.md")

    def test_external_commands_dir_uses_absolute_prompt_file(self):
        """A configured dir outside the package must still resolve its prompt
        file: ``read_prompt_file`` gets an absolute path (relative_to would
        raise ValueError for a file outside PROMPTS_DIR)."""
        with tempfile.TemporaryDirectory() as d:
            cmds_dir = Path(d) / "cmds"
            cmds_dir.mkdir()
            (cmds_dir / "external-cmd.md").write_text("external prompt", encoding="utf-8")
            loaded = load_custom_commands(cmds_dir)
            names = [c.name for c in loaded]
            self.assertIn("external-cmd", names)
            # bundled commands are still available (merge, not replace)
            self.assertIn("explain", names)
            external = next(c for c in loaded if c.name == "external-cmd")
            self.assertTrue(Path(external.prompt_file).is_absolute())
            _, prompt, _ = external.prepare(project_dir=d)
        self.assertEqual(prompt, "external prompt")

    def test_configured_command_overrides_bundled_of_same_name(self):
        with tempfile.TemporaryDirectory() as d:
            cmds_dir = Path(d) / "cmds"
            cmds_dir.mkdir()
            (cmds_dir / "explain.md").write_text("my explain prompt", encoding="utf-8")
            loaded = {c.name: c for c in load_custom_commands(cmds_dir)}
            # exactly one entry, and it points at the configured file
            self.assertEqual([n for n in loaded if n == "explain"], ["explain"])
            self.assertTrue(Path(loaded["explain"].prompt_file).is_absolute())
            _, prompt, _ = loaded["explain"].prepare(project_dir=d)
        self.assertEqual(prompt, "my explain prompt")

    def test_configured_command_overrides_builtin_command(self):
        """A command file replaces the builtin slash command of the same name:
        a configured review.md wins over the builtin /review."""
        with tempfile.TemporaryDirectory() as d:
            cmds_dir = Path(d) / "cmds"
            cmds_dir.mkdir()
            (cmds_dir / "review.md").write_text("my review prompt", encoding="utf-8")
            (cmds_dir / "init.md").write_text("my init prompt", encoding="utf-8")
            self.assertEqual(find_command("review", cmds_dir).name, "review")
            self.assertEqual(find_command("init", cmds_dir).name, "init")
            _, prompt, _ = find_command("review", cmds_dir).prepare(project_dir=d)
            self.assertEqual(prompt, "my review prompt")
            # without the configured dir the builtin prompt commands apply
            self.assertEqual(find_command("review").name, "review")
            self.assertEqual(
                Path(find_command("review").prompt_file).name,
                "review.md",
            )
            self.assertEqual(find_command("init").name, "initialize")

    def test_only_exit_cannot_be_overridden(self):
        """``exit`` is the one name a command file may not claim (it is the
        only way out of a session); /help reports the skipped file."""
        with tempfile.TemporaryDirectory() as d:
            cmds_dir = Path(d) / "cmds"
            cmds_dir.mkdir()
            (cmds_dir / "exit.md").write_text("shadow exit", encoding="utf-8")
            (cmds_dir / "ok-cmd.md").write_text("fine", encoding="utf-8")
            loaded = {c.name for c in load_custom_commands(cmds_dir)}
            self.assertIn("ok-cmd", loaded)
            self.assertNotIn("exit", loaded)
            self.assertIsNone(find_command("exit", cmds_dir))
            self.assertEqual(commands.unoverridable_command_names(cmds_dir), ["exit"])
            self.assertEqual(commands.unoverridable_command_names(str(Path(d) / "missing")), [])


class TestFindCommand(unittest.TestCase):
    def test_builtin_commands(self):
        self.assertEqual(find_command("init").name, "initialize")
        self.assertEqual(find_command("review").name, "review")

    def test_custom_command_found(self):
        cmd = find_command("explain")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.name, "explain")

    def test_configured_command_beats_builtin(self):
        """Resolution order: configured files > bundled files > builtin
        prompt commands."""
        with tempfile.TemporaryDirectory() as d:
            cmds_dir = Path(d) / "cmds"
            cmds_dir.mkdir()
            (cmds_dir / "explain.md").write_text("configured explain", encoding="utf-8")
            (cmds_dir / "review.md").write_text("configured review", encoding="utf-8")
            explain = find_command("explain", cmds_dir)
            self.assertTrue(Path(explain.prompt_file).is_absolute())
            _, prompt, _ = explain.prepare(project_dir=d)
            self.assertEqual(prompt, "configured explain")
            review = find_command("review", cmds_dir)
            self.assertEqual(Path(review.prompt_file).name, "review.md")
            self.assertTrue(Path(review.prompt_file).is_absolute())

    def test_unknown_command_returns_none(self):
        self.assertIsNone(find_command("no-such-command"))


if __name__ == "__main__":
    unittest.main()
