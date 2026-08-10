"""Tests for the default agent-prompt loader (prompts.load_agent_prompt)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from python_agent_harness import config
from python_agent_harness.prompts import (
    _SKILLS_FALLBACK,
    _parse_skill_frontmatter,
    discover_skills,
    last_user_request,
    load_agent_prompt,
    load_context_files,
    load_task_completion_rules,
    strip_frontmatter,
)


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
        """The package-bundled prompts (prompts/agent.md, prompts/subagent.md)
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


class TestParseSkillFrontmatter(unittest.TestCase):
    def test_unreadable_skill_file_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            skill = Path(d) / "SKILL.md"
            skill.mkdir()  # a directory: read_text raises OSError
            self.assertIsNone(_parse_skill_frontmatter(skill))

    def test_no_frontmatter_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            skill = Path(d) / "SKILL.md"
            skill.write_text("# Just a heading\n", encoding="utf-8")
            self.assertIsNone(_parse_skill_frontmatter(skill))

    def test_frontmatter_without_name_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            skill = Path(d) / "SKILL.md"
            skill.write_text(
                "---\ndescription: only a description\n---\nbody",
                encoding="utf-8",
            )
            self.assertIsNone(_parse_skill_frontmatter(skill))

    def test_valid_frontmatter_returns_name_and_description(self):
        with tempfile.TemporaryDirectory() as d:
            skill = Path(d) / "SKILL.md"
            skill.write_text(
                "---\nname: my-skill\ndescription: does things\n---\nbody",
                encoding="utf-8",
            )
            self.assertEqual(
                _parse_skill_frontmatter(skill), ("my-skill", "does things")
            )


class TestDiscoverSkills(unittest.TestCase):
    def test_none_dir_returns_fallback(self):
        self.assertEqual(discover_skills(None), _SKILLS_FALLBACK)

    def test_missing_dir_returns_fallback(self):
        self.assertEqual(discover_skills("/nonexistent/skills"), _SKILLS_FALLBACK)

    def test_plain_files_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "notes.md").write_text("not a skill dir", encoding="utf-8")
            self.assertEqual(discover_skills(d), _SKILLS_FALLBACK)

    def test_dir_without_valid_skills_returns_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "sub"
            sub.mkdir()
            (sub / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")
            self.assertEqual(discover_skills(d), _SKILLS_FALLBACK)

    def test_valid_skills_formatted_listing(self):
        with tempfile.TemporaryDirectory() as d:
            alpha = Path(d) / "alpha"
            alpha.mkdir()
            (alpha / "SKILL.md").write_text(
                "---\nname: alpha-skill\ndescription: does alpha\n---\nbody",
                encoding="utf-8",
            )
            beta = Path(d) / "beta"
            beta.mkdir()
            (beta / "SKILL.md").write_text(
                "---\nname: beta-skill\ndescription: does beta\n---\nbody",
                encoding="utf-8",
            )
            listing = discover_skills(d)
        self.assertIn("<available-skills>", listing)
        self.assertIn("<name>alpha-skill</name>", listing)
        self.assertIn("<description>does alpha</description>", listing)
        self.assertIn("<name>beta-skill</name>", listing)
        self.assertIn("</available-skills>", listing)


class TestLoadContextFiles(unittest.TestCase):
    def test_none_dir_returns_none(self):
        self.assertIsNone(load_context_files(None))

    def test_missing_dir_returns_none(self):
        self.assertIsNone(load_context_files("/nonexistent/contexts"))

    def test_skips_unreadable_files(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.md"
            bad.write_text("secret", encoding="utf-8")
            good = Path(d) / "good.md"
            good.write_text("visible", encoding="utf-8")
            orig = Path.read_text

            def fake_read(self, *a, **k):
                if str(self) == str(bad):
                    raise OSError("permission denied")
                return orig(self, *a, **k)

            with mock.patch.object(Path, "read_text", fake_read):
                block = load_context_files(d)
        self.assertIn("visible", block)
        self.assertNotIn("secret", block)

    def test_skips_subdirectories(self):
        """Subdirectories inside the context dir are not files and must
        be skipped."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sub").mkdir()
            (Path(d) / "real.md").write_text("real content", encoding="utf-8")
            block = load_context_files(d)
        self.assertIn("real content", block)
        self.assertEqual(block.count("In file `"), 1)

    def test_skips_empty_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "empty.md").write_text("   \n\n", encoding="utf-8")
            (Path(d) / "real.md").write_text("real content", encoding="utf-8")
            block = load_context_files(d)
        self.assertIn("real content", block)
        self.assertNotIn("empty.md", block)

    def test_all_files_empty_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("  ", encoding="utf-8")
            (Path(d) / "b.md").write_text("", encoding="utf-8")
            self.assertIsNone(load_context_files(d))

    def test_returns_blocks_for_files(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "README.md"
            f.write_text("# Notes\n", encoding="utf-8")
            block = load_context_files(d)
        self.assertTrue(block.startswith("Request context:"))
        self.assertIn("In file `", block)
        self.assertIn("# Notes", block)


class TestLoadTaskCompletionRules(unittest.TestCase):
    def test_missing_rules_file_returns_none(self):
        with mock.patch.object(Path, "read_text", side_effect=OSError("missing")):
            self.assertIsNone(load_task_completion_rules())


class TestLastUserRequest(unittest.TestCase):
    def test_str_content(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "last"},
        ]
        self.assertEqual(last_user_request(msgs), "last")

    def test_list_content_joined(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]}]
        self.assertEqual(last_user_request(msgs), "hello world")

    def test_list_content_ignores_non_text_parts(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "only text"},
            "raw-string-part",
            {"type": "image"},
        ]}]
        self.assertEqual(last_user_request(msgs), "only text")

    def test_nudge_messages_skipped(self):
        msgs = [
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": config.NUDGE_MESSAGE},
        ]
        self.assertEqual(last_user_request(msgs), "real question")

    def test_non_string_content_skipped(self):
        msgs = [
            {"role": "user", "content": None},
            {"role": "user", "content": 42},
        ]
        self.assertIsNone(last_user_request(msgs))

    def test_no_user_message_returns_none(self):
        msgs = [{"role": "assistant", "content": "hi"}]
        self.assertIsNone(last_user_request(msgs))

    def test_only_nudge_returns_none(self):
        msgs = [{"role": "user", "content": config.NUDGE_MESSAGE}]
        self.assertIsNone(last_user_request(msgs))


if __name__ == "__main__":
    unittest.main()
