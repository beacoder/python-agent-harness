"""Tests for AGENTS.md discovery and injection.

``AGENTS.md`` is the only instruction file the harness knows about: no
user-global file, no CLAUDE.md/CONTEXT.md fallback.  Discovery collects
every AGENTS.md walking up from the project directory to the git
worktree root (nearest first, mirroring opencode's findUp).

Beyond discovery there is no special handling: the files are passed to
``load_context_files`` as extra context files, so they share the block
format and the ``Request context:`` section with the context directory's
own files.

Covers:
- find_agents_md_files: ancestor stacking, git-root bound, no fallbacks
- assemble_agent_prompt: AGENTS.md rendered as context, ahead of the
  context directory's files
- no per-directory discovery: subdirectories are never scanned
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from python_agent_harness.prompts import (
    assemble_agent_prompt,
    find_agents_md_files,
    load_context_files,
)
from python_agent_harness.tools.base import ToolContext
from python_agent_harness.tools.filesystem import Read


class TestFindAgentsMdFiles(unittest.TestCase):
    """Discovery walks up to the git root, collecting every match."""

    def test_no_agents_md_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(find_agents_md_files(d), [])

    def test_finds_agents_md_in_project_root(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")
            result = find_agents_md_files(d)
            self.assertEqual([Path(p) for p in result], [Path(d) / "AGENTS.md"])

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_finds_agents_md_in_ancestor(self):
        """Walks up from a subdirectory to the git root."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True)
            sub = Path(d) / "sub" / "deep"
            sub.mkdir(parents=True)
            (Path(d) / "AGENTS.md").write_text("# Root rules\n", encoding="utf-8")
            result = find_agents_md_files(str(sub))
            self.assertEqual([Path(p) for p in result], [Path(d) / "AGENTS.md"])

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_stacks_every_ancestor_nearest_first(self):
        """ALL ancestor AGENTS.md files are collected, nearest first.

        This is opencode's findUp behaviour: running the agent inside a
        package still picks up the repo-root instructions.
        """
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True)
            pkg = Path(d) / "pkg"
            sub = pkg / "sub"
            sub.mkdir(parents=True)
            (Path(d) / "AGENTS.md").write_text("# Root\n", encoding="utf-8")
            (pkg / "AGENTS.md").write_text("# Pkg\n", encoding="utf-8")
            result = find_agents_md_files(str(sub))
            self.assertEqual(
                [Path(p) for p in result],
                [pkg / "AGENTS.md", Path(d) / "AGENTS.md"],
            )

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_stops_at_git_root(self):
        """Discovery never escapes the git worktree root."""
        with tempfile.TemporaryDirectory() as outer:
            repo = Path(outer) / "myrepo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            sub = repo / "src"
            sub.mkdir()
            (repo / "AGENTS.md").write_text("# Repo rules\n", encoding="utf-8")
            # above the git root: must NOT be picked up
            (Path(outer) / "AGENTS.md").write_text("# Outer rules\n", encoding="utf-8")
            result = find_agents_md_files(str(sub))
            self.assertEqual([Path(p) for p in result], [repo / "AGENTS.md"])

    def test_non_git_dir_is_its_own_bound(self):
        """Outside a repo the search is bounded by the project dir."""
        with tempfile.TemporaryDirectory() as outer:
            proj = Path(outer) / "proj"
            proj.mkdir()
            (Path(outer) / "AGENTS.md").write_text("# Outer\n", encoding="utf-8")
            self.assertEqual(find_agents_md_files(str(proj)), [])

    def test_unrelated_git_root_does_not_escape(self):
        """A worktree root that is not an ancestor is ignored, not chased.

        Differently-spelled paths (symlinked or automounted checkouts)
        could otherwise make the upward walk run to the filesystem root
        and pick up unrelated ancestors' AGENTS.md files.
        """
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "proj"
            proj.mkdir()
            (Path(d) / "AGENTS.md").write_text("# Outer\n", encoding="utf-8")
            with mock.patch(
                "python_agent_harness.prompts._git_toplevel",
                return_value="/definitely/not/an/ancestor",
            ):
                self.assertEqual(find_agents_md_files(str(proj)), [])

    def test_claude_md_is_ignored(self):
        """No CLAUDE.md fallback: without an AGENTS.md, nothing is found."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "CLAUDE.md").write_text("# Claude rules\n", encoding="utf-8")
            (Path(d) / "CONTEXT.md").write_text("# Context rules\n", encoding="utf-8")
            self.assertEqual(find_agents_md_files(d), [])

    def test_no_global_file_is_injected(self):
        """There is no user-global instruction file tier."""
        with tempfile.TemporaryDirectory() as d:
            prompt = assemble_agent_prompt(d, "AGENT", include_context=True)
            self.assertNotIn("In file `", prompt)


class TestAgentsMdAsContextFiles(unittest.TestCase):
    """AGENTS.md files go through load_context_files like any other file."""

    def test_no_agents_md_yields_no_context(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_context_files(None, extra_files=find_agents_md_files(d)))

    def test_rendered_as_context_block(self):
        with tempfile.TemporaryDirectory() as d:
            agents = Path(d) / "AGENTS.md"
            agents.write_text("# Project Rules\nUse 4 spaces.\n", encoding="utf-8")
            result = load_context_files(None, extra_files=find_agents_md_files(d))
            self.assertEqual(
                result,
                f"Request context:\n\nIn file `{agents}`:\n\n# Project Rules\nUse 4 spaces.",
            )

    def test_empty_file_yields_no_context(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AGENTS.md").write_text("  \n\n", encoding="utf-8")
            self.assertIsNone(load_context_files(None, extra_files=find_agents_md_files(d)))

    def test_content_passed_through_verbatim(self):
        """No fencing or escaping: a file with ``` is injected as-is."""
        with tempfile.TemporaryDirectory() as d:
            body = "# Rules\n\nBuild with:\n\n```bash\nmake test\n```\n\nAlways lint."
            (Path(d) / "AGENTS.md").write_text(body + "\n", encoding="utf-8")
            result = load_context_files(None, extra_files=find_agents_md_files(d))
            self.assertTrue(result.endswith(body))

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_stacked_files_rendered_nearest_first(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True)
            pkg = Path(d) / "pkg"
            pkg.mkdir()
            (Path(d) / "AGENTS.md").write_text("ROOT RULES", encoding="utf-8")
            (pkg / "AGENTS.md").write_text("PKG RULES", encoding="utf-8")
            result = load_context_files(None, extra_files=find_agents_md_files(str(pkg)))
            self.assertLess(result.index("PKG RULES"), result.index("ROOT RULES"))

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_empty_nearest_file_does_not_hide_ancestor(self):
        """An empty AGENTS.md is skipped, the ancestor still applies."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True)
            pkg = Path(d) / "pkg"
            pkg.mkdir()
            (Path(d) / "AGENTS.md").write_text("ROOT RULES", encoding="utf-8")
            (pkg / "AGENTS.md").write_text("   \n", encoding="utf-8")
            result = load_context_files(None, extra_files=find_agents_md_files(str(pkg)))
            self.assertIn("ROOT RULES", result)
            self.assertEqual(result.count("In file `"), 1)


class TestAssembleAgentPromptWithAgentsMd(unittest.TestCase):
    """AGENTS.md shares the context section, ahead of the context dir."""

    def test_agents_md_before_context_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AGENTS.md").write_text("AGENTS RULES", encoding="utf-8")
            ctx = Path(d) / "contexts"
            ctx.mkdir()
            (ctx / "general.md").write_text("GENERAL CONTEXT", encoding="utf-8")
            with mock.patch(
                "python_agent_harness.session.find_context_dir",
                return_value=str(ctx),
            ):
                prompt = assemble_agent_prompt(d, "AGENT PROMPT")
            self.assertIsNotNone(prompt)
            i_agents = prompt.index("AGENTS RULES")
            i_ctx = prompt.index("GENERAL CONTEXT")
            i_rules = prompt.index("Task Completion Rules")
            i_agent = prompt.index("AGENT PROMPT")
            self.assertLess(i_agents, i_ctx)
            self.assertLess(i_ctx, i_rules)
            self.assertLess(i_rules, i_agent)
            # one shared context section, not a separate AGENTS.md one
            self.assertEqual(prompt.count("Request context:"), 1)

    def test_no_agents_md_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            prompt = assemble_agent_prompt(d, "AGENT", include_context=False)
            self.assertIn("Task Completion Rules", prompt)
            self.assertIn("AGENT", prompt)

    def test_agents_md_formatted_as_context_block(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")
            prompt = assemble_agent_prompt(d, "AGENT", include_context=True)
            self.assertIn("In file `", prompt)
            self.assertIn("# Rules", prompt)


class TestNoPerDirectoryDiscovery(unittest.TestCase):
    """Subdirectory AGENTS.md files are deliberately NOT discovered.

    Discovery only ever walks upward, and nothing is injected into tool
    results: the Read tool returns file content and nothing else.
    """

    def test_subdirectory_agents_md_not_injected(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "pkg"
            pkg.mkdir()
            (pkg / "AGENTS.md").write_text("NESTED RULES", encoding="utf-8")
            prompt = assemble_agent_prompt(d, "AGENT", include_context=True)
            self.assertNotIn("NESTED RULES", prompt)
            self.assertNotIn("In file `", prompt)

    def test_read_returns_file_content_only(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "pkg"
            pkg.mkdir()
            (pkg / "AGENTS.md").write_text("NESTED RULES", encoding="utf-8")
            f = pkg / "mod.py"
            f.write_text("content\n", encoding="utf-8")
            result = Read().run({"file_path": str(f)}, ToolContext())
            self.assertEqual(result, "content\n")
            self.assertNotIn("NESTED RULES", result)
            self.assertNotIn("<system-reminder>", result)


if __name__ == "__main__":
    unittest.main()
