"""Tests for the default agent-prompt loader (prompts.load_agent_prompt)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from python_agent_harness import config
from python_agent_harness.models import Message
from python_agent_harness.prompts import (
    _SKILLS_FALLBACK,
    _parse_skill_frontmatter,
    discover_skills,
    index_skills,
    load_agent_prompt,
    load_context_files,
    load_task_completion_rules,
    strip_frontmatter,
    user_prompt_texts,
)


class TestStripFrontmatter(unittest.TestCase):
    def test_strips_leading_yaml_block(self):
        text = "---\nname: foo\ndescription: bar\n---\n# Role\nYou are foo.\n"
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
                "---\nname: x\ntools:\n  - Read\n---\n# Role\nUse skills: {{SKILLS}}\n",
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
            self.assertEqual(_parse_skill_frontmatter(skill), ("my-skill", "does things"))


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

    def test_nested_skill_discovered(self):
        """SKILL.md files at any depth are discovered (opencode-style)."""
        with tempfile.TemporaryDirectory() as d:
            nested = Path(d) / "a" / "b" / "deep"
            nested.mkdir(parents=True)
            (nested / "SKILL.md").write_text(
                "---\nname: deep-skill\ndescription: nested\n---\nbody",
                encoding="utf-8",
            )
            listing = discover_skills(d)
        self.assertIn("<name>deep-skill</name>", listing)
        self.assertIn("<description>nested</description>", listing)

    def test_duplicate_names_last_in_sorted_path_wins(self):
        """Duplicate frontmatter names keep the last file in sorted-path
        order (deterministic, mirroring opencode's overwrite)."""
        with tempfile.TemporaryDirectory() as d:
            for sub in ("a-first", "z-last"):
                p = Path(d) / sub
                p.mkdir()
                (p / "SKILL.md").write_text(
                    f"---\nname: dup-skill\ndescription: {sub}\n---\nbody",
                    encoding="utf-8",
                )
            index = index_skills(d)
            self.assertEqual(len(index), 1)
            self.assertEqual(index["dup-skill"][1], "z-last")


class TestIndexSkills(unittest.TestCase):
    def test_none_dir_returns_empty(self):
        self.assertEqual(index_skills(None), {})

    def test_missing_dir_returns_empty(self):
        self.assertEqual(index_skills("/nonexistent/skills"), {})

    def test_index_keyed_by_frontmatter_name(self):
        """Directory names are irrelevant: the key is the frontmatter
        name (the mismatched-dir-name bug)."""
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "weather-forecaster"
            sub.mkdir()
            skill = sub / "SKILL.md"
            skill.write_text(
                "---\nname: 天气预报助手\ndescription: does things\n---\nbody",
                encoding="utf-8",
            )
            index = index_skills(d)
            self.assertEqual(index, {"天气预报助手": (str(skill), "does things")})

    def test_no_frontmatter_name_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "sub"
            sub.mkdir()
            (sub / "SKILL.md").write_text("# no frontmatter", encoding="utf-8")
            self.assertEqual(index_skills(d), {})

    def test_symlinked_skill_dir_indexed(self):
        with (
            tempfile.TemporaryDirectory() as d,
            tempfile.TemporaryDirectory(prefix="pah-skills-out-") as outside,
        ):
            sub = Path(outside) / "linked"
            sub.mkdir()
            skill = sub / "SKILL.md"
            skill.write_text("---\nname: linked-skill\n---\nbody", encoding="utf-8")
            os.symlink(sub, Path(d) / "linked")
            self.assertEqual(index_skills(d)["linked-skill"][0], str(skill))


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

    def test_content_not_fenced(self):
        """Context file contents are injected verbatim, without a fence."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("Run:\n\n```sh\nmake\n```\n\nDone.\n", encoding="utf-8")
            block = load_context_files(d)
        self.assertTrue(block.endswith("Done."))
        self.assertIn("```sh", block)
        self.assertEqual(block.count("```"), 2)

    def test_extra_files_come_first(self):
        """*extra_files* are rendered ahead of the context directory's own."""
        with tempfile.TemporaryDirectory() as d:
            ctx = Path(d) / "contexts"
            ctx.mkdir()
            (ctx / "a.md").write_text("DIR FILE", encoding="utf-8")
            extra = Path(d) / "AGENTS.md"
            extra.write_text("EXTRA FILE", encoding="utf-8")
            block = load_context_files(ctx, extra_files=[str(extra)])
        self.assertEqual(block.count("Request context:"), 1)
        self.assertLess(block.index("EXTRA FILE"), block.index("DIR FILE"))

    def test_extra_files_without_context_dir(self):
        with tempfile.TemporaryDirectory() as d:
            extra = Path(d) / "AGENTS.md"
            extra.write_text("EXTRA FILE", encoding="utf-8")
            block = load_context_files(None, extra_files=[str(extra)])
        self.assertIn("EXTRA FILE", block)

    def test_missing_extra_file_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            block = load_context_files(None, extra_files=[str(Path(d) / "nope.md")])
        self.assertIsNone(block)

    def test_file_reachable_both_ways_rendered_once(self):
        """A context dir that also holds an extra file must not duplicate it."""
        with tempfile.TemporaryDirectory() as d:
            agents = Path(d) / "AGENTS.md"
            agents.write_text("ROOT RULES", encoding="utf-8")
            block = load_context_files(d, extra_files=[str(agents)])
        self.assertEqual(block.count("ROOT RULES"), 1)
        self.assertEqual(block.count("In file `"), 1)


class TestLoadTaskCompletionRules(unittest.TestCase):
    def test_missing_rules_file_returns_none(self):
        with mock.patch.object(Path, "read_text", side_effect=OSError("missing")):
            self.assertIsNone(load_task_completion_rules())


class TestUserPromptTexts(unittest.TestCase):
    def test_returns_every_user_prompt_oldest_first(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
        self.assertEqual(user_prompt_texts(msgs), ["first", "second"])

    def test_excludes_nudge(self):
        msgs = [
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": config.NUDGE_MESSAGE},
        ]
        self.assertEqual(user_prompt_texts(msgs), ["real question"])

    def test_keeps_latest_plan_build_reminder_batch(self):
        """Harness-injected plan/build reminders are KEPT, but only the
        most recent batch: after /plan -> /build the old read-only plan
        reminders would contradict the build switch, so earlier batches
        must not survive compaction."""
        msgs = [
            {
                "role": "user",
                "content": "<system-reminder>\nPlan mode ACTIVE — READ-ONLY.",
                "injected": True,
            },
            {"role": "user", "content": "plan the feature", "injected": False},
            {
                "role": "user",
                "content": "<system-reminder>\nMode changed to build.",
                "injected": True,
            },
            {"role": "user", "content": "then implement it", "injected": False},
        ]
        self.assertEqual(
            user_prompt_texts(msgs),
            [
                "plan the feature",
                "<system-reminder>\nMode changed to build.",
                "then implement it",
            ],
        )

    def test_keeps_whole_latest_batch(self):
        """A single /plan injects plan + plan-mode reminders as one
        contiguous batch; the whole batch survives."""
        msgs = [
            {"role": "user", "content": "<system-reminder>\nPlan mode ACTIVE."},
            {"role": "user", "content": "<system-reminder>\nPlan file: /tmp/x/PLAN.md"},
            {"role": "user", "content": "plan the feature"},
        ]
        self.assertEqual(user_prompt_texts(msgs), [m["content"] for m in msgs])

    def test_plan_exit_notice_is_latest_reminder(self):
        """The plan-exit approval notice is a mode reminder: it carries
        the plan->build handoff and supersedes the earlier plan-mode
        batch (which is no longer the current mode state)."""
        notice = (
            "The plan at /tmp/x/PLAN.md has been approved, "
            "you can now edit files. Execute the plan"
        )
        msgs = [
            {"role": "user", "content": "<system-reminder>\nPlan mode ACTIVE."},
            {"role": "user", "content": "<system-reminder>\nPlan file: /tmp/x/PLAN.md"},
            {"role": "user", "content": "approve the plan"},
            {"role": "user", "content": notice, "injected": True},
        ]
        self.assertEqual(user_prompt_texts(msgs), ["approve the plan", notice])

    def test_excludes_previous_summary_frames(self):
        frame = config.COMPACT_HEADER + "old summary" + config.COMPACT_SEPARATOR
        msgs = [
            {"role": "user", "content": frame},
            {"role": "user", "content": "still here"},
        ]
        self.assertEqual(user_prompt_texts(msgs), ["still here"])

    def test_empty_and_non_text_content_skipped(self):
        msgs = [
            {"role": "user", "content": None},
            {"role": "user", "content": 42},
            {"role": "assistant", "content": "hi"},
        ]
        self.assertEqual(user_prompt_texts(msgs), [])

    def test_no_user_messages_returns_empty(self):
        self.assertEqual(user_prompt_texts([]), [])
        self.assertEqual(user_prompt_texts([{"role": "assistant", "content": "hi"}]), [])

    def test_accepts_message_objects(self):
        msgs = [
            Message(role="user", content="first"),
            Message(role="user", content=config.NUDGE_MESSAGE, injected=True),
            Message(role="user", content="second"),
        ]
        self.assertEqual(user_prompt_texts(msgs), ["first", "second"])

    def test_list_content_joined(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
        self.assertEqual(user_prompt_texts(msgs), ["hello world"])


if __name__ == "__main__":
    unittest.main()
