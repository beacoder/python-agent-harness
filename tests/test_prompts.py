"""Tests for the default agent-prompt loader (prompts.load_agent_prompt)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from python_agent_harness import config
from python_agent_harness.models import ImagePart, Message, TextPart
from python_agent_harness.prompts import (
    _SKILLS_FALLBACK,
    _parse_skill_frontmatter,
    agent_exclude_tools,
    assemble_tool_instructions,
    discover_agents,
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
            self.assertEqual(index, {"天气预报助手": (os.path.realpath(skill), "does things")})

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
            self.assertEqual(index_skills(d)["linked-skill"][0], os.path.realpath(skill))


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
            "The plan at /tmp/x/PLAN.md has been approved, you can now edit files. Execute the plan"
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

    def test_image_part_gets_placeholder_note(self):
        """A multimodal user message must survive compaction with a note
        that an image was attached — otherwise the model loses all
        indication the image ever existed."""
        msgs = [
            Message(
                role="user",
                content=[
                    TextPart(text="what is this?"),
                    ImagePart(data=b"\x89PNG", media_type="image/png", path="/tmp/shot.png"),
                ],
            ),
        ]
        self.assertEqual(
            user_prompt_texts(msgs),
            ["what is this?\n[image was attached: /tmp/shot.png]"],
        )

    def test_url_image_part_gets_placeholder_note(self):
        """A URL-based image must survive compaction with its URL in the
        note, so the model knows which image was attached."""
        msgs = [
            Message(
                role="user",
                content=[
                    TextPart(text="what is this?"),
                    ImagePart.from_url("https://example.com/image.jpg"),
                ],
            ),
        ]
        self.assertEqual(
            user_prompt_texts(msgs),
            ["what is this?\n[image was attached: https://example.com/image.jpg]"],
        )

    def test_image_only_message_dropped(self):
        """An image-only prompt has no text; the image's content is
        already captured in the summary, so the prompt is dropped (the
        existing empty-text guard) rather than kept as a bare note."""
        msgs = [
            Message(role="user", content=[ImagePart(data=b"\x89PNG", media_type="image/png")]),
        ]
        self.assertEqual(user_prompt_texts(msgs), [])

    def test_no_image_part_no_note(self):
        msgs = [Message(role="user", content=[TextPart(text="plain text")])]
        self.assertEqual(user_prompt_texts(msgs), ["plain text"])


class TestAssembleToolInstructions(unittest.TestCase):
    """Tests for assemble_tool_instructions and load_agent_prompt's
    {{TOOL_INSTRUCTIONS}} placeholder substitution."""

    def test_none_dict_returns_empty(self):
        self.assertEqual(assemble_tool_instructions(None), "")

    def test_empty_dict_returns_empty(self):
        self.assertEqual(assemble_tool_instructions({}), "")

    def test_single_tool_rendered_as_block(self):
        result = assemble_tool_instructions({"Read": "Use Read to read files."})
        self.assertIn('<tool name="Read">', result)
        self.assertIn("Use Read to read files.", result)
        self.assertIn("</tool>", result)

    def test_multiple_tools_separated_by_blank_lines(self):
        result = assemble_tool_instructions(
            {
                "Read": "Read instructions.",
                "Bash": "Bash instructions.",
            }
        )
        self.assertEqual(result.count('<tool name="'), 2)
        self.assertIn('<tool name="Read">', result)
        self.assertIn('<tool name="Bash">', result)
        # blocks are separated by a blank line
        self.assertIn("</tool>\n\n<tool name=", result)

    def test_preserves_insertion_order(self):
        result = assemble_tool_instructions(
            {
                "Zebra": "z",
                "Alpha": "a",
                "Mid": "m",
            }
        )
        idx_zebra = result.index("Zebra")
        idx_alpha = result.index("Alpha")
        idx_mid = result.index("Mid")
        self.assertLess(idx_zebra, idx_alpha)
        self.assertLess(idx_alpha, idx_mid)

    def test_excluded_tools_dropped(self):
        result = assemble_tool_instructions(
            {"Read": "read", "Bash": "bash", "Agent": "agent"},
            excluded=("Agent", "Bash"),
        )
        self.assertNotIn('<tool name="Agent">', result)
        self.assertNotIn('<tool name="Bash">', result)
        self.assertIn('<tool name="Read">', result)

    def test_excluded_glob_pattern_drops_matching_tools(self):
        """Glob patterns like ``mcp__git__*`` must exclude matching tools,
        mirroring ``session._tool_excluded``."""
        result = assemble_tool_instructions(
            {
                "Read": "read",
                "mcp__git__list": "git list",
                "mcp__git__search": "git search",
                "mcp__fs__read": "fs read",
            },
            excluded=("mcp__git__*",),
        )
        self.assertNotIn('<tool name="mcp__git__list">', result)
        self.assertNotIn('<tool name="mcp__git__search">', result)
        self.assertIn('<tool name="mcp__fs__read">', result)
        self.assertIn('<tool name="Read">', result)

    def test_excluded_prefix_drops_matching_tools(self):
        """``__``-delimited prefix exclusion (``mcp__git`` hides
        ``mcp__git__list``) must work, mirroring ``session._tool_excluded``."""
        result = assemble_tool_instructions(
            {
                "Read": "read",
                "mcp__git__list": "git list",
                "mcp__git__search": "git search",
                "mcp__fs__read": "fs read",
            },
            excluded=("mcp__git",),
        )
        self.assertNotIn('<tool name="mcp__git__list">', result)
        self.assertNotIn('<tool name="mcp__git__search">', result)
        self.assertIn('<tool name="mcp__fs__read">', result)

    def test_excluded_exact_name_does_not_match_prefix(self):
        """Excluding ``Write`` must NOT exclude ``TodoWrite`` — the
        ``__``-delimited prefix rule prevents this false match."""
        result = assemble_tool_instructions(
            {"Write": "w", "TodoWrite": "tw"},
            excluded=("Write",),
        )
        self.assertNotIn('<tool name="Write">', result)
        self.assertIn('<tool name="TodoWrite">', result)

    def test_backslashes_in_instructions_preserved(self):
        """re.sub must not interpret backslashes in tool instruction text
        as escape sequences (e.g. regex examples like \\s, \\w)."""
        result = assemble_tool_instructions({"Grep": r"Use \s+ and \w+ patterns."})
        self.assertIn(r"\s+", result)
        self.assertIn(r"\w+", result)

    def test_placeholder_substituted_when_tool_instructions_given(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "agent.md"
            path.write_text(
                "# Role\n\n{{TOOL_INSTRUCTIONS}}\n\n# End",
                encoding="utf-8",
            )
            result = load_agent_prompt(path, tool_instructions={"Read": "Read files."})
        self.assertIsNotNone(result)
        self.assertNotIn("{{TOOL_INSTRUCTIONS}}", result)
        self.assertIn('<tool name="Read">', result)
        self.assertIn("Read files.", result)

    def test_placeholder_removed_when_no_tool_instructions(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "agent.md"
            path.write_text(
                "# Role\n\n{{TOOL_INSTRUCTIONS}}\n\n# End",
                encoding="utf-8",
            )
            result = load_agent_prompt(path, tool_instructions=None)
        self.assertIsNotNone(result)
        self.assertNotIn("{{TOOL_INSTRUCTIONS}}", result)
        self.assertNotIn("<tool name=", result)

    def test_placeholder_removed_when_empty_dict(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "agent.md"
            path.write_text(
                "# Role\n\n{{TOOL_INSTRUCTIONS}}\n\n# End",
                encoding="utf-8",
            )
            result = load_agent_prompt(path, tool_instructions={})
        self.assertIsNotNone(result)
        self.assertNotIn("{{TOOL_INSTRUCTIONS}}", result)

    def test_excluded_tools_passed_to_load_agent_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "agent.md"
            path.write_text("{{TOOL_INSTRUCTIONS}}", encoding="utf-8")
            result = load_agent_prompt(
                path,
                tool_instructions={"Read": "read", "Agent": "agent"},
                excluded_tools=("Agent",),
            )
        self.assertIsNotNone(result)
        self.assertIn('<tool name="Read">', result)
        self.assertNotIn('<tool name="Agent">', result)

    def test_tool_instructions_substituted_before_skills(self):
        """{{TOOL_INSTRUCTIONS}} is resolved before {{SKILLS}} so that
        a tool whose instructions contain {{SKILLS}} (the Skill tool)
        has the placeholder resolved correctly."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "agent.md"
            path.write_text("{{TOOL_INSTRUCTIONS}}", encoding="utf-8")
            result = load_agent_prompt(
                path,
                skill_dir=None,
                tool_instructions={"Skill": "Skills: {{SKILLS}}"},
            )
        self.assertIsNotNone(result)
        self.assertNotIn("{{TOOL_INSTRUCTIONS}}", result)
        self.assertNotIn("{{SKILLS}}", result)
        self.assertIn("Skills:", result)
        self.assertIn(_SKILLS_FALLBACK, result)

    def test_bundled_agent_md_has_placeholder(self):
        """The bundled agent.md must contain the {{TOOL_INSTRUCTIONS}}
        placeholder (not hardcoded tool blocks)."""
        text = config.DEFAULT_AGENT_PROMPT_FILE.read_text(encoding="utf-8")
        self.assertIn("{{TOOL_INSTRUCTIONS}}", text)

    def test_bundled_subagent_md_has_placeholder(self):
        """The bundled subagent.md must contain the {{TOOL_INSTRUCTIONS}}
        placeholder (not hardcoded tool blocks)."""
        text = config.DEFAULT_SUBAGENT_PROMPT_FILE.read_text(encoding="utf-8")
        self.assertIn("{{TOOL_INSTRUCTIONS}}", text)

    def test_bundled_prompts_no_hardcoded_tool_blocks(self):
        """Neither bundled prompt should contain hardcoded <tool name=...>
        blocks — those are now assembled dynamically."""
        for prompt_file in (
            config.DEFAULT_AGENT_PROMPT_FILE,
            config.DEFAULT_SUBAGENT_PROMPT_FILE,
        ):
            text = prompt_file.read_text(encoding="utf-8")
            self.assertNotIn('<tool name="', text)

    def test_bundled_prompts_load_with_tool_instructions(self):
        """Loading the bundled prompts with realool instructions from
        the default registry must produce a fully resolved prompt with
        all tool blocks and no leftover placeholders."""
        from python_agent_harness.tools import default_registry

        ti = default_registry().tool_instructions()
        for prompt_file in (
            config.DEFAULT_AGENT_PROMPT_FILE,
            config.DEFAULT_SUBAGENT_PROMPT_FILE,
        ):
            result = load_agent_prompt(prompt_file, tool_instructions=ti)
            self.assertIsNotNone(result)
            self.assertNotIn("{{TOOL_INSTRUCTIONS}}", result)
            self.assertNotIn("{{SKILLS}}", result)
            # at least some tool blocks should be present
            self.assertIn('<tool name="', result)

    def test_subagent_prompt_excludes_subagent_tools(self):
        """Loading subagent.md with SUBAGENT_EXCLUDED_TOOLS must not
        include instructions for Agent, TodoWrite, Question, PlanExit."""
        from python_agent_harness.tools import default_registry

        ti = default_registry().tool_instructions()
        result = load_agent_prompt(
            config.DEFAULT_SUBAGENT_PROMPT_FILE,
            tool_instructions=ti,
            excluded_tools=config.SUBAGENT_EXCLUDED_TOOLS,
        )
        self.assertIsNotNone(result)
        for excluded in config.SUBAGENT_EXCLUDED_TOOLS:
            self.assertNotIn(f'<tool name="{excluded}">', result)
        # but Read and Bash should still be present
        self.assertIn('<tool name="Read">', result)
        self.assertIn('<tool name="Bash">', result)

    def test_tool_instructions_consistent_with_tool_specs_main_agent(self):
        """Every tool with instructions in the main agent prompt must
        also have a tool spec in the default registry (no instructions
        for non-existent or excluded tools)."""
        from python_agent_harness.tools import default_registry

        reg = default_registry()
        ti = reg.tool_instructions()
        prompt = load_agent_prompt(
            config.DEFAULT_AGENT_PROMPT_FILE,
            tool_instructions=ti,
        )
        spec_names = {s.name for s in reg.specs()}
        for name in ti:
            if f'<tool name="{name}">' in prompt:
                self.assertIn(
                    name,
                    spec_names,
                    f"Tool {name} has instructions in prompt but no spec",
                )

    def test_tool_instructions_consistent_with_tool_specs_subagent(self):
        """Every tool with instructions in the subagent prompt must NOT
        be in SUBAGENT_EXCLUDED_TOOLS, and every non-excluded tool with
        instructions must appear in the prompt."""
        from python_agent_harness.tools import default_registry

        reg = default_registry()
        ti = reg.tool_instructions()
        excluded = set(config.SUBAGENT_EXCLUDED_TOOLS)
        prompt = load_agent_prompt(
            config.DEFAULT_SUBAGENT_PROMPT_FILE,
            tool_instructions=ti,
            excluded_tools=config.SUBAGENT_EXCLUDED_TOOLS,
        )
        for name in ti:
            has_block = f'<tool name="{name}">' in prompt
            if name in excluded:
                self.assertFalse(
                    has_block,
                    f"Excluded tool {name} should not have instructions in subagent prompt",
                )
            else:
                self.assertTrue(
                    has_block,
                    f"Non-excluded tool {name} should have instructions in subagent prompt",
                )

    def test_tool_instructions_consistent_with_reviewer_exclusions(self):
        """The reviewer agent profile's ``exclude_tools`` must be honored:
        excluded tools must not appear in the assembled prompt, and every
        other tool with instructions must.  Exclusions are read from the
        real ``reviewer.md`` (via ``agent_exclude_tools``) so this test
        tracks the agent file instead of a hardcoded copy."""
        from python_agent_harness.tools import default_registry

        agents = discover_agents()
        self.assertIn("reviewer", agents, "reviewer agent profile not found")
        reviewer_exclusions = agent_exclude_tools(agents["reviewer"])
        self.assertTrue(
            reviewer_exclusions,
            "reviewer.md declares no exclude_tools; test would be vacuous",
        )

        reg = default_registry()
        ti = reg.tool_instructions()
        prompt = load_agent_prompt(
            config.DEFAULT_AGENT_PROMPT_FILE,
            tool_instructions=ti,
            excluded_tools=reviewer_exclusions,
        )
        for name in reviewer_exclusions:
            self.assertNotIn(
                f'<tool name="{name}">',
                prompt,
                f"Excluded tool {name} should not have instructions in reviewer prompt",
            )
        # non-excluded tools with instructions should still be present
        for name in ti:
            if name not in reviewer_exclusions:
                self.assertIn(
                    f'<tool name="{name}">',
                    prompt,
                    f"Non-excluded tool {name} should have instructions in reviewer prompt",
                )


if __name__ == "__main__":
    unittest.main()
