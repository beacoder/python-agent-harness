"""Tests for the custom agent feature: discovery from prompts/agents/ directory,
runtime switching via Session.switch_agent, and default_agent config."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import session_sandbox  # noqa: F401,E402  (side-effect: redirect SESSION_DIR)

from python_agent_harness import config
from python_agent_harness.client import Client
from python_agent_harness.prompts import AGENTS_DIR, discover_agents
from python_agent_harness.session import Session
from python_agent_harness.tools import default_registry

ENV_KEYS = [
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "PYTHON_AGENT_HARNESS_CONFIG",
    "OPENAI_SUBAGENT_BASE_URL",
    "OPENAI_SUBAGENT_API_KEY",
    "OPENAI_SUBAGENT_MODEL",
]


def _make_session(
    system_prompt: str = "default prompt",
    default_agent: str | None = None,
) -> Session:
    client = Client(base_url="http://localhost", api_key="test", model="test-model")
    return Session(
        project_dir="/tmp",
        client=client,
        model="test-model",
        system_prompt=system_prompt,
        registry=default_registry(),
        default_agent=default_agent,
    )


class TestDiscoverAgents(unittest.TestCase):
    """Agent discovery from the prompts/agents/ directory."""

    def test_reviewer_agent_discovered(self):
        """The bundled reviewer.md agent is discovered by name."""
        agents = discover_agents()
        self.assertIn("reviewer", agents)
        self.assertTrue(agents["reviewer"].endswith("reviewer.md"))

    def test_discovered_agent_file_exists(self):
        agents = discover_agents()
        for name, path in agents.items():
            self.assertTrue(Path(path).exists(), f"agent {name}: file missing: {path}")

    def test_empty_directory_returns_empty(self):
        with mock.patch.object(Path, "is_dir", return_value=False):
            self.assertEqual(discover_agents(), {})

    def test_reserved_default_name_skipped(self):
        """A file claiming the reserved name ``default`` is not discovered:
        the built-in agent always wins, so a custom default.md would be
        unreachable and duplicate the TUI listing."""
        default_file = AGENTS_DIR / "default.md"
        default_file.write_text("You are a shadowing agent.", encoding="utf-8")
        try:
            agents = discover_agents()
            self.assertNotIn("default", agents)
        finally:
            default_file.unlink(missing_ok=True)

    def test_reserved_default_name_via_frontmatter_skipped(self):
        """A frontmatter ``name: default`` is reserved the same way."""
        test_file = AGENTS_DIR / "test-reserved-frontmatter.md"
        test_file.write_text(
            "---\nname: default\n---\nYou are a shadowing agent.", encoding="utf-8"
        )
        try:
            agents = discover_agents()
            self.assertNotIn("default", agents)
        finally:
            test_file.unlink(missing_ok=True)

    def test_agent_name_from_frontmatter(self):
        """When a file has YAML frontmatter with name:, that name is used."""
        test_file = AGENTS_DIR / "test-frontmatter-agent.md"
        test_file.write_text(
            "---\nname: my-custom-name\n---\nYou are a custom agent.", encoding="utf-8"
        )
        try:
            agents = discover_agents()
            self.assertIn("my-custom-name", agents)
            self.assertEqual(agents["my-custom-name"], str(test_file.resolve()))
        finally:
            test_file.unlink(missing_ok=True)

    def test_agent_name_from_stem_when_no_frontmatter(self):
        """Without frontmatter, the file stem (lowercased) is the name."""
        test_file = AGENTS_DIR / "test-stem-agent.md"
        test_file.write_text("You are a simple agent with no frontmatter.", encoding="utf-8")
        try:
            agents = discover_agents()
            self.assertIn("test-stem-agent", agents)
        finally:
            test_file.unlink(missing_ok=True)


class TestSwitchAgent(unittest.TestCase):
    """Session.switch_agent: runtime prompt switching."""

    def test_switch_to_named_agent(self):
        session = _make_session()
        ok, msg = session.switch_agent("reviewer")
        self.assertTrue(ok, msg)
        self.assertIn("code reviewer", session.system_prompt)

    def test_switch_to_default_restores_original(self):
        session = _make_session(system_prompt="original prompt")
        session.switch_agent("reviewer")
        self.assertNotEqual(session.system_prompt, "original prompt")
        ok, msg = session.switch_agent("default")
        self.assertTrue(ok)
        self.assertEqual(session.system_prompt, "original prompt")

    def test_switch_to_unknown_agent_fails(self):
        session = _make_session()
        ok, msg = session.switch_agent("nonexistent")
        self.assertFalse(ok)
        self.assertIn("unknown agent", msg)
        self.assertIn("default", msg)
        self.assertIn("reviewer", msg)

    def test_switch_preserves_context_and_rules(self):
        """switch_agent reassembles with assemble_agent_prompt, so
        task-completion rules are still prepended."""
        session = _make_session()
        session.switch_agent("reviewer")
        from python_agent_harness.prompts import load_task_completion_rules

        rules = load_task_completion_rules()
        if rules:
            self.assertIn(rules, session.system_prompt)

    def test_default_system_prompt_saved(self):
        """_default_system_prompt holds the original prompt for restoration."""
        session = _make_session(system_prompt="my original")
        self.assertEqual(session._default_system_prompt, "my original")

    def test_switch_updates_store_system_prompt(self):
        """switch_agent must update store.system_prompt so auto-saved
        sessions record the correct prompt in metadata."""
        session = _make_session(system_prompt="original prompt")
        session.switch_agent("reviewer")
        self.assertEqual(session.store.system_prompt, session.system_prompt)
        session.switch_agent("default")
        self.assertEqual(session.store.system_prompt, "original prompt")

    def test_switch_updates_store_agent(self):
        """switch_agent must keep store.agent in sync so every auto-save
        records the currently active agent in the metadata block."""
        session = _make_session(system_prompt="original prompt")
        self.assertIsNone(session.store.agent)  # no custom agent yet
        session.switch_agent("reviewer")
        self.assertEqual(session.store.agent, "reviewer")
        session.switch_agent("default")
        self.assertIsNone(session.store.agent)  # default = no custom agent

    def test_failed_switch_leaves_store_agent_untouched(self):
        """A failed switch (unknown name) must not change store.agent:
        the metadata must never record an agent that is not active."""
        session = _make_session(system_prompt="original prompt")
        session.switch_agent("reviewer")
        ok, _ = session.switch_agent("nonexistent")
        self.assertFalse(ok)
        self.assertEqual(session.store.agent, "reviewer")


class TestDefaultAgentConfig(unittest.TestCase):
    """config.load_default_agent: reading the default_agent setting."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_missing_file_returns_none(self):
        self.assertIsNone(config.load_default_agent("/no/such/file.json"))

    def test_unset_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"model": "test"}}', encoding="utf-8")
            self.assertIsNone(config.load_default_agent(p))

    def test_valid_string_returned(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"default_agent": "reviewer"}), encoding="utf-8")
            self.assertEqual(config.load_default_agent(p), "reviewer")

    def test_empty_string_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"default_agent": ""}), encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_default_agent(p)

    def test_non_string_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"default_agent": 123}), encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_default_agent(p)


class TestMakeSessionWithDefaultAgent(unittest.TestCase):
    """cli.make_session applies default_agent at session start."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_default_agent_applied(self):
        """When default_agent is set, make_session starts with that agent."""
        from python_agent_harness.cli import make_session

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"default_agent": "reviewer"}), encoding="utf-8")
            session = make_session("/tmp", config_path=str(p))
            self.assertEqual(session.default_agent, "reviewer")
            self.assertIn("code reviewer", session.system_prompt)

    def test_no_default_agent_uses_agent_md(self):
        """Without default_agent, the built-in agent.md prompt is used."""
        from python_agent_harness.cli import make_session

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"llm": {"model": "test-model"}}), encoding="utf-8")
            session = make_session("/tmp", config_path=str(p))
            self.assertIsNone(session.default_agent)

    def test_default_agent_restorable_to_original(self):
        """After default_agent is applied, /agent default restores the
        original agent.md prompt (not the default_agent setting)."""
        from python_agent_harness.cli import make_session

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"default_agent": "reviewer"}), encoding="utf-8")
            session = make_session("/tmp", config_path=str(p))
            self.assertIn("code reviewer", session.system_prompt)
            ok, _ = session.switch_agent("default")
            self.assertTrue(ok)
            self.assertNotIn("code reviewer", session.system_prompt)

    def test_unknown_default_agent_records_warning(self):
        """A typo'd/missing default_agent must not fail silently: the
        session records a startup warning (rendered by the TUI) and
        starts with the built-in agent.md prompt."""
        from python_agent_harness.cli import make_session

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"default_agent": "no-such-agent"}), encoding="utf-8")
            session = make_session("/tmp", config_path=str(p))
            self.assertEqual(session.default_agent, "no-such-agent")
            # failed switch: session starts with the built-in agent.md prompt
            self.assertNotIn("code reviewer", session.system_prompt)
            warnings = session.startup_warnings
        self.assertEqual(len(warnings), 1)
        self.assertIn("default_agent", warnings[0])
        self.assertIn("no-such-agent", warnings[0])

    def test_valid_default_agent_no_warning(self):
        """A valid default_agent records no startup warning."""
        from python_agent_harness.cli import make_session

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"default_agent": "reviewer"}), encoding="utf-8")
            session = make_session("/tmp", config_path=str(p))
            self.assertEqual(session.startup_warnings, [])
            session.close()

    def test_valid_default_agent_recorded_in_store(self):
        """A successfully applied default_agent is recorded in store.agent,
        so saved sessions restore it.  A typo'd one must NOT be recorded:
        the metadata must never name an agent that is not active."""
        from python_agent_harness.cli import make_session

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"default_agent": "reviewer"}), encoding="utf-8")
            session = make_session("/tmp", config_path=str(p))
            try:
                self.assertEqual(session.store.agent, "reviewer")
            finally:
                session.close()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"default_agent": "no-such-agent"}), encoding="utf-8")
            session = make_session("/tmp", config_path=str(p))
            try:
                self.assertIsNone(session.store.agent)
            finally:
                session.close()


class TestMcpFailuresAsStartupWarnings(unittest.TestCase):
    """make_session_with_mcp records MCP connection failures as startup
    warnings so the TUI banner renders them (instead of a stderr print
    that is invisible inside the interface)."""

    def test_mcp_failure_recorded_as_startup_warning(self):
        from python_agent_harness import cli
        from python_agent_harness.mcp.config import MCPConfig, MCPServerConfig
        from python_agent_harness.mcp.manager import MCPManager
        from python_agent_harness.session import Session as RealSession

        class WarnSession(RealSession):
            # swap in a manager whose only server fails to connect
            def __init__(self, *args, **kwargs):
                kwargs.pop("mcp", None)
                super().__init__(*args, **kwargs)
                self.mcp_manager = MCPManager(
                    MCPConfig(
                        servers={
                            "ghost": MCPServerConfig(
                                name="ghost",
                                transport="stdio",
                                command=sys.executable,
                                args=["-c", "import sys; sys.exit(1)"],
                                timeout=15,
                            )
                        }
                    )
                )

        saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        try:
            with (
                mock.patch.object(cli, "Session", WarnSession),
                tempfile.TemporaryDirectory() as d,
            ):
                p = Path(d) / "config.json"
                p.write_text(json.dumps({"llm": {"model": "test-model"}}), encoding="utf-8")
                session = cli.make_session_with_mcp("/tmp", config_path=str(p))
                try:
                    self.assertEqual(len(session.startup_warnings), 1)
                    self.assertIn("MCP [ghost]", session.startup_warnings[0])
                    # built-in tools still work after the failed server
                    self.assertIn("Read", session.registry._tools)
                finally:
                    session.close()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class TestAgentExcludeTools(unittest.TestCase):
    """Agent profile ``exclude_tools`` frontmatter: parsing and effect
    on the tool specs exposed to the model."""

    def _write_agent(
        self, name: str, frontmatter: str, body: str = "You are a test agent."
    ) -> Path:
        p = AGENTS_DIR / f"{name}.md"
        p.write_text(f"---\n{frontmatter}---\n{body}", encoding="utf-8")
        return p

    def test_parse_exclude_tools(self):
        """agent_exclude_tools reads the frontmatter list."""
        from python_agent_harness.prompts import agent_exclude_tools

        p = self._write_agent(
            "test-exclude-parse",
            "name: test-exclude-parse\nexclude_tools:\n  - Bash\n  - Edit\n  - mcp__git__*\n",
        )
        try:
            self.assertEqual(
                agent_exclude_tools(p),
                ("Bash", "Edit", "mcp__git__*"),
            )
        finally:
            p.unlink(missing_ok=True)

    def test_parse_exclude_tools_inline(self):
        """agent_exclude_tools reads an inline ``[a, b]`` list too."""
        from python_agent_harness.prompts import agent_exclude_tools

        p = self._write_agent(
            "test-exclude-inline",
            "name: test-exclude-inline\nexclude_tools: [Bash, mcp__git__*]\n",
        )
        try:
            self.assertEqual(
                agent_exclude_tools(p),
                ("Bash", "mcp__git__*"),
            )
        finally:
            p.unlink(missing_ok=True)

    def test_no_exclude_tools_returns_empty(self):
        """An agent without the key excludes nothing."""
        from python_agent_harness.prompts import agent_exclude_tools

        p = self._write_agent("test-exclude-none", "name: test-exclude-none\n")
        try:
            self.assertEqual(agent_exclude_tools(p), ())
        finally:
            p.unlink(missing_ok=True)

    def test_switch_agent_applies_exclusions(self):
        """switch_agent activates the profile's excluded tools."""
        p = self._write_agent(
            "test-exclude-switch",
            "name: test-exclude-switch\nexclude_tools:\n  - Bash\n  - Edit\n",
        )
        try:
            session = _make_session()
            ok, msg = session.switch_agent("test-exclude-switch")
            self.assertTrue(ok, msg)
            names = {s.name for s in session.tool_specs()}
            self.assertNotIn("Bash", names)
            self.assertNotIn("Edit", names)
            self.assertIn("Read", names)
        finally:
            p.unlink(missing_ok=True)

    def test_switch_to_default_clears_exclusions(self):
        """/agent default restores the full tool set."""
        p = self._write_agent(
            "test-exclude-default",
            "name: test-exclude-default\nexclude_tools:\n  - Bash\n",
        )
        try:
            session = _make_session()
            session.switch_agent("test-exclude-default")
            self.assertNotIn("Bash", {s.name for s in session.tool_specs()})
            ok, msg = session.switch_agent("default")
            self.assertTrue(ok, msg)
            self.assertIn("Bash", {s.name for s in session.tool_specs()})
        finally:
            p.unlink(missing_ok=True)

    def test_glob_pattern_excludes_mcp_tools(self):
        """mcp__<server>__* patterns hide every tool of one server."""
        from python_agent_harness.tools.base import Tool

        class FakeMcpTool(Tool):
            name = "mcp__git__list_repos"
            description = "fake"

            def run(self, args, ctx):
                return ""

        p = self._write_agent(
            "test-exclude-glob",
            "name: test-exclude-glob\nexclude_tools:\n  - mcp__git__*\n",
        )
        try:
            session = _make_session()
            session.registry.register(FakeMcpTool())
            self.assertIn("mcp__git__list_repos", {s.name for s in session.tool_specs()})
            session.switch_agent("test-exclude-glob")
            names = {s.name for s in session.tool_specs()}
            self.assertNotIn("mcp__git__list_repos", names)
        finally:
            p.unlink(missing_ok=True)

    def test_substring_pattern_excludes_mcp_tools(self):
        """A ``__``-delimited prefix (mcp__git) hides every tool of one
        server, but does NOT accidentally match unrelated tools."""
        from python_agent_harness.tools.base import Tool

        class FakeMcpTool(Tool):
            name = "mcp__git__list_repos"
            description = "fake"

            def run(self, args, ctx):
                return ""

        p = self._write_agent(
            "test-exclude-substr",
            "name: test-exclude-substr\nexclude_tools:\n  - mcp__git\n",
        )
        try:
            session = _make_session()
            session.registry.register(FakeMcpTool())
            self.assertIn("mcp__git__list_repos", {s.name for s in session.tool_specs()})
            session.switch_agent("test-exclude-substr")
            names = {s.name for s in session.tool_specs()}
            self.assertNotIn("mcp__git__list_repos", names)
        finally:
            p.unlink(missing_ok=True)

    def test_write_does_not_exclude_todowrite(self):
        """``Write`` must not accidentally hide ``TodoWrite``: matching
        is by exact name or ``__``-delimited prefix, not raw substring."""
        p = self._write_agent(
            "test-exclude-no-false-match",
            "name: test-exclude-no-false-match\nexclude_tools:\n  - Write\n",
        )
        try:
            session = _make_session()
            session.switch_agent("test-exclude-no-false-match")
            names = {s.name for s in session.tool_specs()}
            self.assertNotIn("Write", names)
            self.assertIn("TodoWrite", names)
        finally:
            p.unlink(missing_ok=True)

    def test_explicit_exclude_still_works(self):
        """The caller's exclude argument composes with the profile's."""
        p = self._write_agent(
            "test-exclude-compose",
            "name: test-exclude-compose\nexclude_tools:\n  - Bash\n",
        )
        try:
            session = _make_session()
            session.switch_agent("test-exclude-compose")
            names = {s.name for s in session.tool_specs(exclude=("Edit",))}
            self.assertNotIn("Bash", names)
            self.assertNotIn("Edit", names)
        finally:
            p.unlink(missing_ok=True)

    def test_failed_switch_leaves_exclusions_untouched(self):
        """A failed switch (unknown name) must not change the active
        exclusions: the caller's tool set must never silently shift."""
        p = self._write_agent(
            "test-exclude-fail",
            "name: test-exclude-fail\nexclude_tools:\n  - Bash\n",
        )
        try:
            session = _make_session()
            session.switch_agent("test-exclude-fail")
            self.assertNotIn("Bash", {s.name for s in session.tool_specs()})
            ok, _ = session.switch_agent("nonexistent")
            self.assertFalse(ok)
            self.assertNotIn("Bash", {s.name for s in session.tool_specs()})
        finally:
            p.unlink(missing_ok=True)

    def test_quoted_items_stripped(self):
        """Quoted frontmatter items lose their quotes during parsing."""
        from python_agent_harness.prompts import agent_exclude_tools

        p = self._write_agent(
            "test-exclude-quoted",
            "name: test-exclude-quoted\nexclude_tools:\n  - \"Bash\"\n  - 'Edit'\n",
        )
        try:
            self.assertEqual(agent_exclude_tools(p), ("Bash", "Edit"))
        finally:
            p.unlink(missing_ok=True)

    def test_no_frontmatter_at_all(self):
        """A file with no frontmatter block excludes nothing."""
        from python_agent_harness.prompts import agent_exclude_tools

        p = self._write_agent("test-no-frontmatter", "")
        p.write_text("You are a simple agent with no frontmatter.", encoding="utf-8")
        try:
            self.assertEqual(agent_exclude_tools(p), ())
        finally:
            p.unlink(missing_ok=True)


class TestReviewerAgentContent(unittest.TestCase):
    """Verify the bundled reviewer.md agent has expected content."""

    def test_reviewer_prompt_contains_focus_areas(self):
        agents = discover_agents()
        path = agents["reviewer"]
        from python_agent_harness.prompts import load_agent_prompt

        prompt = load_agent_prompt(path)
        self.assertIsNotNone(prompt)
        self.assertIn("code reviewer", prompt)
        self.assertIn("Correctness", prompt)
        self.assertIn("Security", prompt)


if __name__ == "__main__":
    unittest.main()
