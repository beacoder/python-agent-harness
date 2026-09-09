"""Minimal built-in LSP client used by python-agent-harness."""
from .client import LSPClient, LSPError
from .manager import get_client, shutdown_all

__all__ = ["LSPClient", "LSPError", "get_client", "shutdown_all"]
