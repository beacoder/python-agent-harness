"""python-agent-harness: a Python port of the gptel-agent-harness."""

from .client import Client, LLMClient
from .config import ModelInfo
from .lsp.config import LSPConfig, LSPServerConfig
from .mcp.config import MCPConfig, MCPServerConfig
from .mcp.manager import MCPManager
from .models import AgentMode, ImagePart, Message, TextPart, ToolCall, ToolSpec
from .session import Session

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
    "ModelInfo",
    "Message",
    "TextPart",
    "ImagePart",
    "ToolCall",
    "ToolSpec",
    "__version__",
]
