"""Tests for the default agent-prompt loader (compaction.load_agent_prompt)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from python_agent_harness import config
from python_agent_harness.compaction import load_agent_prompt, strip_frontmatter


class TestStripFrontmatter(unittest.TestCase):
    def test_strips_leading_yaml_block(self):
        text = (
            "---\nname: foo\ndescription: bar\n---\n"
            "# Role\nYou are foo.\n"
        )
        self.assertEqual(strip_frontmatter(text), "# Role\nYou are foo.\n")

    def test_no_frontmatter_unchanged(self):
        text = "# Role\nYou are foo.\n"
        self.assertEqual(strip_frontmatter(text), text)

    def test_only_leading_frontmatter_stripped(self):
        """A '---' later in the body (not at the very start) must survive."""
        text = "---\nname: foo\n---\nSee the --- separator below.\n---\nmore\n"
        out = strip_frontmatter(text)
        self.assertNotIn("name: foo", out)
        self.assertIn("--- separator", out)


class TestLoadAgentPrompt(unittest.TestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(load_agent_prompt("/nonexistent/path/agent.md"))

    def test_none_path_returns_none(self):
        self.assertIsNone(load_agent_prompt(None))

    def test_strips_frontmatter_and_substitutes_skills_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "agent.md"
            path.write_text(
                "---\nname: x\ntools:\n  - Read\n---\n"
                "# Role\nUse skills: {{SKILLS}}\n",
                encoding="utf-8",
            )
            text = load_agent_prompt(path)
            self.assertIsNotNone(text)
            self.assertNotIn("---", text)
            self.assertNotIn("{{SKILLS}}", text)
            self.assertIn("# Role", text)

    def test_empty_file_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.md"
            path.write_text("---\nname: x\n---\n\n  \n", encoding="utf-8")
            self.assertIsNone(load_agent_prompt(path))

    def test_bundled_agent_files_present(self):
        """The package-bundled prompts (prompts/agent.txt, prompts/subagent.txt)
        must load cleanly and differ from each other."""
        main = load_agent_prompt(config.DEFAULT_AGENT_PROMPT_FILE)
        sub = load_agent_prompt(config.DEFAULT_SUBAGENT_PROMPT_FILE)
        self.assertIsNotNone(main)
        self.assertIsNotNone(sub)
        self.assertNotIn("{{SKILLS}}", main)
        self.assertNotIn("{{SKILLS}}", sub)
        self.assertFalse(main.startswith("---"))
        self.assertFalse(sub.startswith("---"))
        self.assertNotEqual(main, sub)


if __name__ == "__main__":
    unittest.main()
