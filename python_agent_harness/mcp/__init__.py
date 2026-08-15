"""MCP (Model Context Protocol) client support.

Optional: requires the ``mcp`` extra (``pip install -e ".[mcp]"``).
Importing this package never requires the SDK — only actually
connecting to an MCP server does.
"""

from __future__ import annotations

from .client import MCPClient, MCPUnavailableError
from .config import MCPConfig, MCPServerConfig
from .manager import MCPManager, MCPToolSpec

__all__ = [
    "MCPClient",
    "MCPConfig",
    "MCPManager",
    "MCPServerConfig",
    "MCPToolSpec",
    "MCPUnavailableError",
]
