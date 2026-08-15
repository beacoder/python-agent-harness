"""Thin wrapper around the official MCP Python SDK (client side).

This module is the ONLY place that touches the SDK.  The rest of the
harness sees plain dicts and strings, so a future SDK API change (the
v1 → v2 line already renamed FastMCP → MCPServer and moved the client
onto ``mcp.Client``) stays contained here.

The SDK is optional: importing this module never fails without it.  The
``mcp`` extra (``pip install -e ".[mcp]"``) provides ``mcp>=2.0,<3``;
without it every operation raises :class:`MCPUnavailableError`.

The wrapper is async (the SDK is async); :class:`MCPManager` drives it
from the harness's synchronous world through a dedicated event-loop
thread.  SDK types never leak out: ``list_tools`` returns plain dicts
and ``call_tool`` returns a plain dict with keys ``content`` (list of
content-block dicts), ``structured_content`` and ``is_error``.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from mcp import Client
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the optional extra
    _MCP_AVAILABLE = False


class MCPUnavailableError(RuntimeError):
    """Raised when MCP is configured but the optional SDK is not installed.

    Fix: ``pip install -e ".[mcp]"``.
    """


def _require_sdk() -> None:
    if not _MCP_AVAILABLE:
        raise MCPUnavailableError(
            "MCP support requires the optional `mcp` extra — install with `pip install -e '.[mcp]'`"
        )


class MCPClient:
    """One connection to one MCP server; hides the official SDK.

    Use as an async context manager::

        async with MCPClient(config) as client:
            tools = await client.list_tools()
            result = await client.call_tool("search", {"q": "x"})

    All SDK exceptions propagate as-is (connection refused, protocol
    errors, ...); the manager/tool layer turns them into normal tool
    error strings.
    """

    def __init__(self, config: Any) -> None:
        from .config import MCPServerConfig

        self.config: MCPServerConfig = config
        self._client: Any = None
        self._http_client: Any = None

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _transport(self) -> Any:
        """Build the SDK transport for this server's config."""
        cfg = self.config
        if cfg.transport not in ("stdio", "streamable-http", "sse"):
            raise ValueError(
                f"MCP server {cfg.name or '(unnamed)'!r}: unknown transport {cfg.transport!r}"
            )
        if cfg.transport == "stdio":
            # command is validated non-None by MCPServerConfig.validate
            assert cfg.command is not None
            env = None
            if cfg.env:
                env = {name: os.environ[name] for name in cfg.env if name in os.environ}
            return stdio_client(StdioServerParameters(command=cfg.command, args=cfg.args, env=env))
        # url is validated non-None by MCPServerConfig.validate for the
        # HTTP transports
        assert cfg.url is not None
        if cfg.transport == "streamable-http":
            if cfg.headers:
                import httpx2  # shipped as part of the mcp SDK

                self._http_client = httpx2.AsyncClient(headers=cfg.headers)
                return streamable_http_client(cfg.url, http_client=self._http_client)
            return streamable_http_client(cfg.url)
        if cfg.transport == "sse":
            return sse_client(cfg.url, headers=cfg.headers or None)
        # unreachable: the transport whitelist at the top rejects
        # everything else
        raise ValueError(
            f"MCP server {cfg.name!r}: unknown transport {cfg.transport!r}"
        )  # pragma: no cover

    async def connect(self) -> None:
        """Establish the connection (spawn process / open HTTP session)."""
        _require_sdk()
        self._client = Client(
            server=self._transport(),
            read_timeout_seconds=self.config.timeout,
        )
        await self._client.__aenter__()

    async def close(self) -> None:
        """Tear down the connection (best effort, never raises)."""
        import contextlib

        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):  # teardown noise is not an error
                await client.__aexit__(None, None, None)
        http_client, self._http_client = self._http_client, None
        if http_client is not None:
            with contextlib.suppress(Exception):  # teardown noise is not an error
                await http_client.aclose()

    async def list_tools(self) -> list[dict[str, Any]]:
        """tools/list — the server's tool descriptors as plain dicts."""
        _require_sdk()
        result = await self._client.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.input_schema if isinstance(t.input_schema, dict) else {},
            }
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """tools/call — returns a plain dict (content blocks, structured
        content, is_error).  Server-reported failures surface as
        ``is_error=True``, NOT as exceptions."""
        _require_sdk()
        result = await self._client.call_tool(name, arguments or {})
        return {
            "content": [_content_block_to_dict(b) for b in result.content],
            "structured_content": result.structured_content,
            "is_error": bool(result.is_error),
        }


def _content_block_to_dict(block: Any) -> dict[str, Any]:
    """One MCP content block → plain dict (text/image/audio/resource/...)."""
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return {"type": "unknown", "raw": str(block)}
