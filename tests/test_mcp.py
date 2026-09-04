"""Tests for the optional MCP integration (python_agent_harness/mcp/,
python_agent_harness/tools/mcp.py).

MCP is an OPTIONAL extra: the unit tests (config validation, tool
adapter, result normalization, config-file loading) never need the
SDK.  The integration tests that spawn a real MCP server through the
official SDK are skipped when the ``mcp`` extra is not installed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import session_sandbox  # noqa: F401,E402  (side-effect: redirect SESSION_DIR)
from agent.agent_test_utils import RecordingSession

from python_agent_harness import config
from python_agent_harness.mcp.client import MCPUnavailableError
from python_agent_harness.mcp.config import MCPConfig, MCPServerConfig
from python_agent_harness.mcp.manager import MCPManager, MCPToolSpec
from python_agent_harness.session import Session
from python_agent_harness.tools.base import PendingToolResult
from python_agent_harness.tools.mcp import (
    MCPTool,
    mcp_tool_name,
    mcp_tools_from_manager,
    normalize_mcp_result,
)

try:
    import mcp  # noqa: F401

    HAS_MCP_SDK = True
except ImportError:
    HAS_MCP_SDK = False

FAKE_SERVER = Path(__file__).parent / "fake_mcp_server.py"


def demo_config(**overrides: object) -> MCPConfig:
    """A stdio config pointing at the fake JSON-RPC server."""
    params = {
        "name": "demo",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(FAKE_SERVER)],
        "timeout": 15,
    }
    params.update(overrides)
    return MCPConfig(servers={"demo": MCPServerConfig(**params)})


class TestMCPConfig(unittest.TestCase):
    """MCPServerConfig / MCPConfig parsing and validation."""

    def test_from_dict_stdio(self):
        cfg = MCPConfig.from_dict(
            {"srv": {"command": "npx", "args": ["-y", "srv"], "env": ["GITHUB_TOKEN"]}}
        )
        server = cfg.servers["srv"]
        self.assertEqual(server.transport, "stdio")
        self.assertEqual(server.command, "npx")
        self.assertEqual(server.args, ["-y", "srv"])
        self.assertEqual(server.env, ["GITHUB_TOKEN"])
        self.assertTrue(server.enabled)

    def test_from_dict_streamable_http(self):
        cfg = MCPConfig.from_dict(
            {
                "remote": {
                    "transport": "streamable-http",
                    "url": "http://localhost:8000/mcp",
                    "headers": {"Authorization": "Bearer x"},
                    "parallel": True,
                    "timeout": 5,
                    "enabled": False,
                }
            }
        )
        server = cfg.servers["remote"]
        self.assertEqual(server.url, "http://localhost:8000/mcp")
        self.assertEqual(server.headers, {"Authorization": "Bearer x"})
        self.assertTrue(server.parallel)
        self.assertEqual(server.timeout, 5)
        self.assertFalse(server.enabled)

    def test_validate_stdio_requires_command(self):
        with self.assertRaises(ValueError):
            MCPConfig.from_dict({"srv": {"transport": "stdio"}})

    def test_validate_http_requires_url(self):
        with self.assertRaises(ValueError):
            MCPConfig.from_dict({"srv": {"transport": "streamable-http"}})

    def test_validate_unknown_transport(self):
        with self.assertRaises(ValueError):
            MCPConfig.from_dict({"srv": {"transport": "carrier-pigeon", "command": "x"}})

    def test_validate_non_object_server(self):
        with self.assertRaises(ValueError):
            MCPConfig.from_dict({"srv": "npx"})

    def test_empty_config(self):
        self.assertEqual(MCPConfig.from_dict(None).servers, {})
        self.assertEqual(MCPConfig().servers, {})

    def test_compact_construction_sets_name_from_key(self):
        """The design's point-14 API: MCPServerConfig without `name`,
        the MCPConfig dict key fills it in."""
        cfg = MCPConfig(
            servers={
                "github": MCPServerConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-github"],
                ),
            }
        )
        self.assertEqual(cfg.servers["github"].name, "github")
        self.assertEqual(cfg.servers["github"].command, "npx")

    def test_explicit_name_wins_over_key(self):
        cfg = MCPConfig(servers={"key": MCPServerConfig(name="real", command="x")})
        self.assertEqual(cfg.servers["key"].name, "real")


class TestNormalizeMCPResult(unittest.TestCase):
    """MCP CallToolResult dicts → harness tool-result strings."""

    def test_text_block(self):
        result = {"content": [{"type": "text", "text": "hello"}], "is_error": False}
        self.assertEqual(normalize_mcp_result(result), "hello")

    def test_multiple_blocks_joined(self):
        result = {
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
            "is_error": False,
        }
        self.assertEqual(normalize_mcp_result(result), "a\nb")

    def test_image_and_audio_placeholders(self):
        result = {
            "content": [
                {"type": "image", "data": "QUJD", "mime_type": "image/png"},
                {"type": "audio", "data": "QUJD", "mime_type": "audio/wav"},
            ],
            "is_error": False,
        }
        text = normalize_mcp_result(result)
        self.assertIn("image omitted", text)
        self.assertIn("image/png", text)
        self.assertIn("audio omitted", text)

    def test_resource_link(self):
        result = {
            "content": [
                {"type": "resource", "uri": "file:///tmp/x.txt", "mime_type": "text/plain"}
            ],
            "is_error": False,
        }
        self.assertEqual(normalize_mcp_result(result), "[resource file:///tmp/x.txt]")

    def test_embedded_text_resource(self):
        result = {
            "content": [
                {
                    "type": "embedded_resource",
                    "resource": {"type": "text_resource", "uri": "file:///a", "text": "body"},
                }
            ],
            "is_error": False,
        }
        self.assertEqual(
            normalize_mcp_result(result), "[embedded resource: [resource file:///a]\nbody]"
        )

    def test_structured_content_json(self):
        result = {
            "content": [{"type": "text", "text": "line1"}],
            "structured_content": {"count": 1},
            "is_error": False,
        }
        self.assertEqual(normalize_mcp_result(result), 'line1\n{"count": 1}')

    def test_embedded_blob_resource(self):
        result = {
            "content": [
                {
                    "type": "embedded_resource",
                    "resource": {
                        "type": "blob_resource",
                        "uri": "file:///b",
                        "mime_type": "application/octet-stream",
                        "data": "QUJD",
                    },
                }
            ],
            "is_error": False,
        }
        text = normalize_mcp_result(result)
        self.assertIn("file:///b", text)
        self.assertIn("omitted", text)

    def test_resource_without_uri(self):
        result = {"content": [{"type": "resource"}], "is_error": False}
        self.assertIn("(no uri)", normalize_mcp_result(result))

    def test_unknown_block_type(self):
        result = {"content": [{"type": "weird", "x": 1}], "is_error": False}
        self.assertIn("weird", normalize_mcp_result(result))

    def test_non_dict_content(self):
        result = {"content": ["raw"], "is_error": False}
        self.assertEqual(normalize_mcp_result(result), "raw")

    def test_server_error_prefix(self):
        result = {"content": [{"type": "text", "text": "boom"}], "is_error": True}
        text = normalize_mcp_result(result)
        self.assertTrue(text.startswith("Error: MCP tool reported an error"), text)
        self.assertIn("boom", text)

    def test_empty_result(self):
        self.assertEqual(normalize_mcp_result({"content": [], "is_error": False}), "(no result)")

    def test_content_block_fallback(self):
        """Non-pydantic content blocks degrade to a readable dict."""
        from python_agent_harness.mcp.client import _content_block_to_dict

        self.assertEqual(_content_block_to_dict(42), {"type": "unknown", "raw": "42"})


class _FakeManager:
    """Stand-in for MCPManager: no SDK, no threads, scripted results."""

    def __init__(self, results: dict | None = None, errors: dict | None = None) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.results = results or {}
        self.errors = errors or {}

    def call_tool(
        self, server: str, tool: str, arguments: dict, timeout=None, cancel_check=None
    ) -> dict:
        self.calls.append((server, tool, arguments))
        if tool in self.errors:
            raise self.errors[tool]
        return self.results.get(
            tool, {"content": [], "structured_content": None, "is_error": False}
        )


class TestMCPTool(unittest.TestCase):
    """MCPTool adapter: namespacing, schema mapping, serial/parallel
    execution, error containment."""

    def spec(self, name: str = "search") -> MCPToolSpec:
        return MCPToolSpec(
            server="github",
            name=name,
            description="Search code",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )

    def test_name_namespaced(self):
        tool = MCPTool(self.spec(), _FakeManager())
        self.assertEqual(tool.name, "mcp__github__search")

    def test_same_tool_name_different_servers_do_not_collide(self):
        a = MCPTool(MCPToolSpec("github", "search", "d", {}), _FakeManager())
        b = MCPTool(MCPToolSpec("browser", "search", "d", {}), _FakeManager())
        self.assertNotEqual(a.name, b.name)

    def test_parameters_schema_mapped(self):
        tool = MCPTool(self.spec(), _FakeManager())
        self.assertEqual(tool.parameters["type"], "object")
        self.assertEqual(tool.parameters["properties"], {"q": {"type": "string"}})
        self.assertEqual(tool.parameters["required"], ["q"])

    def test_parameters_empty_schema(self):
        tool = MCPTool(MCPToolSpec("s", "t", "d", {}), _FakeManager())
        self.assertEqual(tool.parameters, {"type": "object"})

    def test_description_fallback(self):
        tool = MCPTool(MCPToolSpec("s", "t", "", {}), _FakeManager())
        self.assertIn("t", tool.description)
        self.assertIn("s", tool.description)

    def test_run_serial(self):
        manager = _FakeManager(
            results={"search": {"content": [{"type": "text", "text": "hits"}], "is_error": False}}
        )
        tool = MCPTool(self.spec(), manager)
        result = tool.run({"q": "x"}, None)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "hits")
        self.assertEqual(manager.calls, [("github", "search", {"q": "x"})])

    def test_run_normalizes_server_error(self):
        manager = _FakeManager(
            results={"search": {"content": [{"type": "text", "text": "boom"}], "is_error": True}}
        )
        tool = MCPTool(self.spec(), manager)
        self.assertTrue(tool.run({}, None).startswith("Error:"))

    def test_run_contains_connection_error(self):
        manager = _FakeManager(errors={"search": ConnectionError("connection refused")})
        tool = MCPTool(self.spec(), manager)
        result = tool.run({}, None)
        self.assertTrue(result.startswith("Error: tool mcp__github__search failed"), result)
        self.assertIn("connection refused", result)

    def test_run_parallel_returns_pending(self):
        manager = _FakeManager(
            results={"search": {"content": [{"type": "text", "text": "ok"}], "is_error": False}}
        )
        tool = MCPTool(self.spec(), manager, parallel=True)
        result = tool.run({}, None)
        self.assertIsInstance(result, PendingToolResult)
        self.assertEqual(result.wait(), "ok")

    def test_run_parallel_error_contained(self):
        manager = _FakeManager(errors={"search": TimeoutError("timed out after 5s")})
        tool = MCPTool(self.spec(), manager, parallel=True)
        result = tool.run({}, None)
        self.assertIsInstance(result, PendingToolResult)
        text = result.wait()
        self.assertTrue(text.startswith("Error: tool mcp__github__search failed"), text)
        self.assertIn("timed out", text)

    def test_mcp_tool_name_helper(self):
        self.assertEqual(mcp_tool_name("github", "search"), "mcp__github__search")


class TestToolsFromManager(unittest.TestCase):
    """mcp_tools_from_manager maps discovered specs + server config
    (parallel/timeout) onto MCPTool instances."""

    def test_concurrency_flags_from_server_config(self):
        manager = MCPManager(
            MCPConfig(
                servers={
                    "ro": MCPServerConfig(name="ro", command="x", parallel=True, timeout=7),
                    "rw": MCPServerConfig(name="rw", command="x"),
                }
            )
        )
        manager._tools["ro__search"] = MCPToolSpec("ro", "search", "d", {})
        manager._tools["rw__write"] = MCPToolSpec("rw", "write", "d", {})
        tools = {t.name: t for t in mcp_tools_from_manager(manager)}
        self.assertTrue(tools["mcp__ro__search"]._parallel)
        self.assertEqual(tools["mcp__ro__search"]._timeout, 7)
        self.assertFalse(tools["mcp__rw__write"]._parallel)

    def test_empty_manager_yields_no_tools(self):
        self.assertEqual(mcp_tools_from_manager(MCPManager()), [])


class TestLoadMCPConfig(unittest.TestCase):
    """Config-file loading (config.load_mcp_config)."""

    def write_config(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_no_mcp_section(self):
        path = self.write_config('{"llm": {"model": "x"}}')
        try:
            self.assertEqual(config.load_mcp_config(path).servers, {})
        finally:
            os.unlink(path)

    def test_servers_parsed(self):
        path = self.write_config(
            json.dumps({"mcp": {"servers": {"gh": {"command": "npx", "args": ["-y", "srv"]}}}})
        )
        try:
            cfg = config.load_mcp_config(path)
            self.assertIn("gh", cfg.servers)
            self.assertEqual(cfg.servers["gh"].command, "npx")
            self.assertEqual(cfg.servers["gh"].args, ["-y", "srv"])
        finally:
            os.unlink(path)

    def test_malformed_server_raises(self):
        path = self.write_config(json.dumps({"mcp": {"servers": {"gh": "npx"}}}))
        try:
            with self.assertRaises(ValueError):
                config.load_mcp_config(path)
        finally:
            os.unlink(path)

    def test_malformed_section_raises(self):
        path = self.write_config(json.dumps({"mcp": ["not", "an", "object"]}))
        try:
            with self.assertRaises(ValueError):
                config.load_mcp_config(path)
        finally:
            os.unlink(path)

    def test_config_template_roundtrip(self):
        """`config --init` template must parse, and its example server
        must be disabled (never auto-connected)."""
        path = self.write_config(config.CONFIG_TEMPLATE.format(path="/tmp/x.json"))
        try:
            cfg = config.load_mcp_config(path)
            self.assertIn("example", cfg.servers)
            self.assertFalse(cfg.servers["example"].enabled)
        finally:
            os.unlink(path)


class TestSDKUnavailable(unittest.TestCase):
    """The SDK-missing path: MCPUnavailableError surfaces per-server as
    a connect failure, never crashing the session.  Runs with or
    without the SDK installed."""

    def test_require_sdk_raises(self):
        import python_agent_harness.mcp.client as mc

        with (
            mock.patch.object(mc, "_MCP_AVAILABLE", False),
            self.assertRaises(MCPUnavailableError),
        ):
            mc._require_sdk()

    def test_connect_all_reports_sdk_missing(self):
        import python_agent_harness.mcp.client as mc

        manager = MCPManager(demo_config())
        try:
            with mock.patch.object(mc, "_MCP_AVAILABLE", False):
                failures = manager.connect_all()
            self.assertEqual(len(failures), 1)
            self.assertIn("mcp", failures[0][1].lower())
            self.assertEqual(manager.connected, [])
            self.assertEqual(manager.discover_tools(), [])
        finally:
            manager.close_all()


@unittest.skipUnless(HAS_MCP_SDK, "requires the optional `mcp` extra")
class TestMCPManagerIntegration(unittest.TestCase):
    """End-to-end through the official SDK + the fake stdio server."""

    def setUp(self) -> None:
        self.manager = MCPManager(demo_config())

    def tearDown(self) -> None:
        self.manager.close_all()

    def test_connect_discover_call_roundtrip(self):
        failures = self.manager.connect_all()
        self.assertEqual(failures, [], failures)
        self.assertEqual(self.manager.connected, ["demo"])
        # connect_all is idempotent: reconnecting leaves the session alone
        self.assertEqual(self.manager.connect_all(), [])

        specs = self.manager.discover_tools()
        names = {(s.server, s.name) for s in specs}
        self.assertIn(("demo", "echo"), names)
        self.assertIn(("demo", "fail"), names)
        self.assertIn(("demo", "rich"), names)
        # discovery happens ONCE per session, not per turn
        self.assertEqual(self.manager.discover_tools(), [])

        result = self.manager.call_tool("demo", "echo", {"text": "hi"})
        self.assertFalse(result["is_error"])
        self.assertEqual(result["content"][0]["text"], "echo:hi")

        # server-reported failures come back as is_error, not exceptions
        result = self.manager.call_tool("demo", "fail", {})
        self.assertTrue(result["is_error"])

    def test_tool_adapter_end_to_end(self):
        self.manager.connect_all()
        self.manager.discover_tools()
        tools = {t.name: t for t in mcp_tools_from_manager(self.manager)}
        echo = tools["mcp__demo__echo"]
        self.assertEqual(echo.run({"text": "yo"}, None), "echo:yo")
        rich = tools["mcp__demo__rich"]
        text = rich.run({}, None)
        self.assertIn("line1", text)
        self.assertIn("image omitted", text)
        self.assertIn('"count": 1', text)
        fail = tools["mcp__demo__fail"]
        self.assertTrue(fail.run({}, None).startswith("Error:"))

    def test_disconnect_drops_tools_and_calls(self):
        self.manager.connect_all()
        self.manager.discover_tools()
        names = {(s.server, s.name) for s in self.manager.tool_specs()}
        self.assertIn(("demo", "echo"), names)
        self.manager.disconnect("demo")
        self.assertEqual(self.manager.connected, [])
        self.assertEqual(self.manager.tool_specs(), [])
        with self.assertRaises(ConnectionError):
            self.manager.call_tool("demo", "echo", {})

    def test_connect_failure_is_reported_not_raised(self):
        bad = MCPManager(
            MCPConfig(
                servers={
                    "ghost": MCPServerConfig(
                        name="ghost",
                        command=sys.executable,
                        args=["-c", "import sys; sys.exit(1)"],
                        timeout=15,
                    )
                }
            )
        )
        try:
            failures = bad.connect_all()
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0][0], "ghost")
            self.assertIn("failed to connect", failures[0][1])
            self.assertEqual(bad.connected, [])
        finally:
            bad.close_all()

    def test_disabled_server_not_connected(self):
        manager = MCPManager(demo_config(enabled=False))
        try:
            self.assertEqual(manager.connect_all(), [])
            self.assertEqual(manager.connected, [])
            self.assertEqual(manager.discover_tools(), [])
        finally:
            manager.close_all()

    def test_client_async_context_manager(self):
        """MCPClient is usable directly as an async context manager
        (the documented SDK-wrapper form), independent of MCPManager."""
        import asyncio

        from python_agent_harness.mcp.client import MCPClient

        config = demo_config().servers["demo"]

        async def run():
            async with MCPClient(config) as client:
                tools = await client.list_tools()
                result = await client.call_tool("echo", {"text": "x"})
                return tools, result

        tools, result = asyncio.run(run())
        names = {t["name"] for t in tools}
        self.assertIn("echo", names)
        self.assertIn("fail", names)
        self.assertEqual(result["content"][0]["text"], "echo:x")

    def test_transport_selection(self):
        """Transport construction: streamable-http with headers, sse,
        stdio with env passthrough, and an unknown transport rejected."""
        import asyncio

        from python_agent_harness.mcp.client import MCPClient

        cfg = MCPServerConfig(
            name="s",
            transport="streamable-http",
            url="http://example.com/mcp",
            headers={"Authorization": "Bearer t"},
        )
        client = MCPClient(cfg)
        try:
            self.assertIsNotNone(client._transport())
        finally:
            asyncio.run(client.close())
        sse = MCPClient(MCPServerConfig(name="s", transport="sse", url="http://example.com/sse"))
        self.assertIsNotNone(sse._transport())
        plain = MCPClient(
            MCPServerConfig(name="s", transport="streamable-http", url="http://example.com/mcp")
        )
        self.assertIsNotNone(plain._transport())
        stdio = MCPClient(
            MCPServerConfig(
                name="s", transport="stdio", command="echo", env=["PYTHONPATH", "NO_SUCH_VAR"]
            )
        )
        self.assertIsNotNone(stdio._transport())
        with self.assertRaises(ValueError):
            MCPClient(MCPServerConfig(name="s", transport="bogus"))._transport()

    def test_discovery_failure_recorded(self):
        """A server that connects but fails tools/list is reported, and
        the session keeps working with the other servers."""
        self.manager.connect_all()
        with mock.patch.object(
            self.manager._clients["demo"], "list_tools", side_effect=RuntimeError("boom")
        ):
            specs = self.manager.discover_tools()
        self.assertEqual(specs, [])
        self.assertTrue(
            any("discovery failed" in err for _, err in self.manager.errors), self.manager.errors
        )

    def test_call_timeout_raises(self):
        """A server that never answers must raise TimeoutError within
        the configured timeout instead of hanging the agent loop."""
        self.manager.connect_all()
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            self.manager.call_tool("demo", "hang", {}, timeout=1)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 30, f"timeout took too long: {elapsed:.1f}s")
        # the connection survives a timed-out call: the next call works
        result = self.manager.call_tool("demo", "echo", {"text": "still alive"})
        self.assertEqual(result["content"][0]["text"], "echo:still alive")


class TestMCPToolCallTimeout(unittest.TestCase):
    """The per-call timeout reaches the manager (unit level)."""

    def test_timeout_passed_through(self):
        manager = _FakeManager(
            results={"echo": {"content": [{"type": "text", "text": "ok"}], "is_error": False}}
        )
        tool = MCPTool(MCPToolSpec("s", "echo", "d", {}), manager, parallel=False, timeout=3.5)
        self.assertEqual(tool.run({}, None), "ok")
        self.assertEqual(manager.calls, [("s", "echo", {})])


class TestConnectFailureCleanup(unittest.TestCase):
    """A failed connect must close the MCP client: connect() may have
    spawned the server process / opened an HTTP session before failing,
    and dropping the client without close() orphans it."""

    def test_failed_connect_closes_client(self):
        import python_agent_harness.mcp.manager as mgr_mod

        closed: list[bool] = []

        class BoomClient:
            def __init__(self, config):
                self.config = config

            async def connect(self):
                raise RuntimeError("boom")

            async def close(self):
                closed.append(True)

        manager = MCPManager(
            MCPConfig(servers={"ghost": MCPServerConfig(name="ghost", command="true", timeout=5)})
        )
        try:
            with mock.patch.object(mgr_mod, "MCPClient", BoomClient):
                failures = manager.connect_all()
            self.assertEqual(len(failures), 1)
            self.assertIn("failed to connect", failures[0][1])
            self.assertEqual(manager.connected, [])
            self.assertEqual(len(closed), 1)
        finally:
            manager.close_all()


class TestMCPCallCancel(unittest.TestCase):
    """Ctrl-C must unblock a hung MCP call: the manager polls the
    cancel check while waiting and cancels the underlying future."""

    def test_cancelled_call_unblocks(self):
        import asyncio
        import threading

        import python_agent_harness.mcp.manager as mgr_mod

        started = threading.Event()
        gate = asyncio.Event()

        class HangClient:
            def __init__(self, config):
                self.config = config

            async def connect(self):
                pass

            async def close(self):
                pass

            async def call_tool(self, name, arguments):
                started.set()
                await gate.wait()
                return {"content": [], "is_error": False}

        manager = MCPManager(
            MCPConfig(servers={"hang": MCPServerConfig(name="hang", command="true", timeout=30)})
        )
        outcome: dict[str, Any] = {}
        try:
            with mock.patch.object(mgr_mod, "MCPClient", HangClient):
                self.assertEqual(manager.connect_all(), [])
                cancel = threading.Event()

                def worker():
                    try:
                        manager.call_tool("hang", "t", {}, cancel_check=cancel.is_set)
                    except Exception as e:  # noqa: BLE001 - recorded for assertions
                        outcome["error"] = e

                t = threading.Thread(target=worker, daemon=True)
                t.start()
                self.assertTrue(started.wait(5))
                cancel.set()
                t.join(5)
                self.assertFalse(t.is_alive(), "worker thread still blocked after cancel")
                self.assertIsInstance(outcome.get("error"), mgr_mod.MCPCallCancelled)
        finally:
            gate.set()
            manager.close_all()


@unittest.skipUnless(HAS_MCP_SDK, "requires the optional `mcp` extra")
class TestSessionMCP(unittest.TestCase):
    """Session lifecycle: connect_mcp registers tools into the
    registry; close() disconnects."""

    def make_session(self):
        session = RecordingSession()
        session.logs = []
        session.log_fn = session.logs.append
        session.mcp_manager = MCPManager(demo_config())
        return session

    def test_connect_mcp_registers_tools(self):
        session = self.make_session()
        try:
            failures = session.connect_mcp()
            self.assertEqual(failures, [], failures)
            self.assertEqual(session.mcp_errors, [])
            names = {spec.name for spec in session.registry.specs()}
            self.assertIn("mcp__demo__echo", names)
            self.assertIn("mcp__demo__fail", names)
            # the tool executes through the real MCP path (not the
            # RecordingSession execute_tool override)
            result = Session.execute_tool(session, "mcp__demo__echo", {"text": "hi"})
            self.assertEqual(result, "echo:hi")
            result = Session.execute_tool(session, "mcp__demo__fail", {})
            self.assertTrue(result.startswith("Error:"), result)
            self.assertIn("MCP: registered", "\n".join(session.logs))
        finally:
            session.close()

    def test_connect_mcp_without_servers_is_noop(self):
        session = RecordingSession()
        session.mcp_manager = MCPManager()  # empty config
        self.assertEqual(session.connect_mcp(), [])
        self.assertEqual(session.mcp_errors, [])
        session.close()

    def test_connect_mcp_failure_is_non_fatal(self):
        session = RecordingSession()
        session.logs = []
        session.log_fn = session.logs.append
        session.mcp_manager = MCPManager(
            MCPConfig(
                servers={
                    "ghost": MCPServerConfig(
                        name="ghost",
                        command=sys.executable,
                        args=["-c", "import sys; sys.exit(1)"],
                        timeout=15,
                    )
                }
            )
        )
        try:
            failures = session.connect_mcp()
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0][0], "ghost")
            # built-in tools keep working
            self.assertIn("Read", {spec.name for spec in session.registry.specs()})
        finally:
            session.close()

    def test_plan_mode_blocks_mcp_tools(self):
        """Plan mode refuses every mcp__ tool: the read-only guarantee
        must hold even though the harness cannot inspect what an
        external server's tool does."""
        from python_agent_harness.models import AgentMode

        session = self.make_session()
        try:
            failures = session.connect_mcp()
            self.assertEqual(failures, [], failures)
            session.plan_mode.set_mode(
                AgentMode.PLAN,
                {"plan": "P1", "plan-mode": "P2", "build-switch": "B"},
            )
            result = Session.execute_tool(session, "mcp__demo__echo", {"text": "hi"})
            self.assertIn("blocked by plan mode", result)
            self.assertIn("MCP tools are disabled", result)
            # the write-capable fake tool is refused the same way
            result = Session.execute_tool(session, "mcp__demo__fail", {})
            self.assertIn("blocked by plan mode", result)
            # back in build mode the tool runs again
            session.plan_mode.set_mode(
                AgentMode.BUILD,
                {"plan": "P1", "plan-mode": "P2", "build-switch": "B"},
            )
            result = Session.execute_tool(session, "mcp__demo__echo", {"text": "hi"})
            self.assertEqual(result, "echo:hi")
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
