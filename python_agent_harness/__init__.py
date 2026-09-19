"""python-agent-harness: a Python port of the gptel-agent-harness."""

from .core.models import AgentMode, ImagePart, Message, TextPart, ToolCall, ToolSpec
from .entry.cli import main  # noqa: F401  (public lib surface: main entry point)
from .llm.client import Client, LLMClient
from .lsp.config import LSPConfig, LSPServerConfig
from .mcp.config import MCPConfig, MCPServerConfig
from .mcp.manager import MCPManager
from .session.session import Session

__version__ = "1.5.5.8"

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
    "TextPart",
    "ImagePart",
    "ToolCall",
    "ToolSpec",
    "__version__",
]
