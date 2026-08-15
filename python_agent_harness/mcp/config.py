"""MCP server configuration for python-agent-harness.

MCP support is OPTIONAL: install the harness with ``pip install -e ".[mcp]"``
to get the official ``mcp`` SDK (the ``mcp`` extra).  This module only
defines plain configuration data classes — importing it never imports the
SDK, so the base harness stays dependency-free.

Transports (per the MCP spec):

- ``stdio``: spawn ``command`` with ``args`` as a subprocess; ``env``
  lists environment variable names passed through from the harness
  process (e.g. ``["GITHUB_TOKEN"]``).
- ``streamable-http``: connect to ``url``; ``headers`` (e.g.
  Authorization) are sent with every request.  The direction to target
  for new remote deployments.
- ``sse``: connect to ``url`` over the legacy SSE transport.

Concurrency policy: ``parallel`` marks the server's tools as safe for
concurrent execution (read-only servers); the tool then runs in the
background like Bash/Agent.  The default is conservative serial
execution.  The harness — not the MCP protocol — retains authority
over this.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

TRANSPORTS = ("stdio", "streamable-http", "sse")


@dataclass
class MCPServerConfig:
    """Configuration for one MCP server connection.

    ``name`` is optional: when the server lives in an ``MCPConfig``
    dict the dict key is authoritative (it fills ``name`` on
    construction), so the design's compact form works::

        MCPConfig(servers={"github": MCPServerConfig(command="npx", ...)})

    ``enabled=False`` keeps the server in the config file without
    connecting it (a documented example, or a temporarily-disabled
    server).
    """

    name: str = ""
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    parallel: bool = False
    timeout: float | None = None
    enabled: bool = True

    def validate(self) -> None:
        label = self.name or "(unnamed)"
        if self.transport not in TRANSPORTS:
            raise ValueError(
                f"MCP server {label!r}: unknown transport {self.transport!r} "
                f"(expected one of {', '.join(TRANSPORTS)})"
            )
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"MCP server {label!r}: stdio transport requires `command`")
        elif not self.url:
            raise ValueError(f"MCP server {label!r}: {self.transport} transport requires `url`")


@dataclass
class MCPConfig:
    """The set of MCP servers for one session.

    Usage (the compact form — the dict key IS the server name)::

        config = MCPConfig(
            servers={
                "github": MCPServerConfig(
                    transport="stdio",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-github"],
                    env=["GITHUB_TOKEN"],
                ),
                "remote": MCPServerConfig(
                    transport="streamable-http",
                    url="http://localhost:8000/mcp",
                ),
            }
        )
    """

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The dict key is the authoritative server name: fill in any
        # config whose name was left unset (compact construction).
        for key, server in self.servers.items():
            if not server.name:
                server.name = key

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> MCPConfig:
        """Build from a plain mapping (e.g. the config file's ``mcp.servers``).

        Raises ValueError on malformed entries (unknown transport, missing
        command/url) so config errors surface at session start, not mid-run.
        """
        config = cls()
        for name, raw in (data or {}).items():
            if not isinstance(raw, dict):
                raise ValueError(f"MCP server {name!r}: expected an object")
            server = MCPServerConfig(
                name=name,
                transport=str(raw.get("transport", "stdio")),
                command=raw.get("command"),
                args=[str(a) for a in (raw.get("args") or [])],
                env=[str(e) for e in (raw.get("env") or [])],
                url=raw.get("url"),
                headers={str(k): str(v) for k, v in (raw.get("headers") or {}).items()},
                parallel=bool(raw.get("parallel", False)),
                timeout=raw.get("timeout"),
                enabled=bool(raw.get("enabled", True)),
            )
            server.validate()
            config.servers[name] = server
        return config
