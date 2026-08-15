"""MCP tool adapter: MCP server tools as normal harness tools.

An MCP server's tools become :class:`Tool` instances named
``mcp__<server>__<tool>`` (unambiguous namespacing — two servers can
both expose ``search`` without colliding).  The agent loop never sees
MCP: these tools go through the same ToolRegistry as built-ins, so the
agent's existing retry / supervision / sanitization machinery applies
unchanged.

Results are normalized to the harness's string tool-result form
(:func:`normalize_mcp_result`), and every MCP failure surfaces as a
plain ``Error: ...`` string — never an SDK exception — so the agent
sees::

    Tool mcp__github__search failed: connection refused

Concurrency mirrors the harness policy: servers configured with
``parallel = true`` run their tools in the background (a
``PendingToolResult``, like Bash/Agent); the default is conservative
serial execution.  The harness, not the MCP protocol, retains authority
over this.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from ..mcp.manager import MCPManager, MCPToolSpec
from .base import PendingToolResult, Tool, ToolContext

MCP_PREFIX = "mcp__"


def mcp_tool_name(server: str, tool: str) -> str:
    """The namespaced harness name for an MCP tool."""
    return f"{MCP_PREFIX}{server}__{tool}"


def normalize_mcp_result(result: dict[str, Any]) -> str:
    """MCP CallToolResult (plain dict) → harness tool-result string.

    Maps the MCP content blocks onto the harness's single-string tool
    result: text blocks become text; image/audio/resource blocks become
    readable placeholders (the payload is base64 and must not flood the
    context); structured content is JSON-serialized.  A server-reported
    error becomes an ``Error: ...`` string.
    """
    if result.get("is_error"):
        text = _render_content(result)
        return (
            f"Error: MCP tool reported an error: {text}"
            if text
            else ("Error: MCP tool reported an error")
        )
    text = _render_content(result)
    structured = result.get("structured_content")
    if structured is not None:
        rendered = json.dumps(structured, ensure_ascii=False, default=str)
        text = f"{text}\n{rendered}" if text else rendered
    return text if text else "(no result)"


def _render_content(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in result.get("content") or []:
        rendered = _render_block(block) if isinstance(block, dict) else str(block)
        if rendered:
            parts.append(rendered)
    return "\n".join(parts)


def _render_block(block: dict[str, Any]) -> str:
    kind = block.get("type")
    if kind == "text":
        return str(block.get("text", ""))
    if kind == "image":
        data = block.get("data") or ""
        return (
            f"[image omitted: mimeType={block.get('mime_type')}, {len(data)} chars of base64 data]"
        )
    if kind == "audio":
        data = block.get("data") or ""
        return (
            f"[audio omitted: mimeType={block.get('mime_type')}, {len(data)} chars of base64 data]"
        )
    if kind == "resource":
        return _render_resource(block)
    if kind == "embedded_resource":
        resource = block.get("resource")
        inner = _render_resource(resource) if isinstance(resource, dict) else str(resource)
        return f"[embedded resource: {inner}]"
    return str(block)


def _render_resource(resource: dict[str, Any]) -> str:
    uri = resource.get("uri") or "(no uri)"
    if resource.get("type") == "text_resource" and resource.get("text"):
        return f"[resource {uri}]\n{resource['text']}"
    data = resource.get("data")
    if data:
        return (
            f"[resource {uri} omitted: mimeType={resource.get('mime_type')}, "
            f"{len(str(data))} chars]"
        )
    return f"[resource {uri}]"


class MCPTool(Tool):
    """A tool exposed by an MCP server, namespaced ``mcp__<server>__<tool>``.

    ``run`` executes through the manager: serial by default, or in the
    background (``PendingToolResult``) when the server is configured
    with ``parallel = true``.
    """

    def __init__(
        self,
        spec: MCPToolSpec,
        manager: MCPManager,
        *,
        parallel: bool = False,
        timeout: float | None = None,
    ) -> None:
        self.server_name = spec.server
        self.mcp_name = spec.name
        self._manager = manager
        self._parallel = parallel
        self._timeout = timeout
        self.name = mcp_tool_name(spec.server, spec.name)
        self.description = spec.description or f"MCP tool {spec.name} (server {spec.server})"
        schema = spec.input_schema if isinstance(spec.input_schema, dict) else {}
        self.parameters = _schema_from_input(schema)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str | PendingToolResult:
        if not self._parallel:
            return self._execute(args)
        pending = PendingToolResult()
        threading.Thread(target=lambda: pending.deliver(self._execute(args)), daemon=True).start()
        return pending

    def _execute(self, args: dict[str, Any]) -> str:
        try:
            result = self._manager.call_tool(
                self.server_name, self.mcp_name, args or {}, timeout=self._timeout
            )
        except Exception as e:  # noqa: BLE001 - errors become tool results
            return f"Error: tool {self.name} failed — {e}"
        return normalize_mcp_result(result)


def _schema_from_input(schema: dict[str, Any]) -> dict[str, Any]:
    """MCP input schema → harness JSON-schema parameters.

    The schema is passed through as-is (it already is a JSON object
    schema); only the container is normalized so a schema without
    ``properties``/``required`` still serializes cleanly.
    """
    parameters: dict[str, Any] = {"type": "object"}
    properties = schema.get("properties")
    if isinstance(properties, dict):
        parameters["properties"] = properties
    required = schema.get("required")
    if isinstance(required, list) and required:
        parameters["required"] = [str(r) for r in required]
    return parameters


def mcp_tools_from_manager(manager: MCPManager) -> list[MCPTool]:
    """Build harness tools for every tool discovered by MANAGER.

    Concurrency flags come from the server config (``parallel``); a
    per-server timeout is applied to every call when configured.
    """
    tools: list[MCPTool] = []
    for spec in manager.tool_specs():
        server_config = manager.config.servers.get(spec.server)
        tools.append(
            MCPTool(
                spec,
                manager,
                parallel=bool(server_config and server_config.parallel),
                timeout=server_config.timeout if server_config else None,
            )
        )
    return tools
