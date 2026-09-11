"""python-agent-harness: a Python port of the gptel-agent-harness."""

from .client import Client, LLMClient
from .lsp.config import LSPConfig, LSPServerConfig
from .mcp.config import MCPConfig, MCPServerConfig
from .mcp.manager import MCPManager
from .models import AgentMode, Message, ToolCall, ToolSpec
from .session import Session

__version__ = "1.5.5.7"

__all__ = [
    "Client",
    "LLMClient",
    "Session",
    "AgentMode",
    "LSPConfig",
    "LSPServerConfig",
    "MCPConfig",
    "MCPManager",
    "MCPServerConfig",
    "Message",
    "ToolCall",
    "ToolSpec",
    "__version__",
]
