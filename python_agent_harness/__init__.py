"""python-agent-harness: a Python port of the gptel-agent-harness."""

from .agent_session import AgentSession
from .mcp.config import MCPConfig, MCPServerConfig
from .mcp.manager import MCPManager
from .models import AgentMode, Message, ToolCall, ToolSpec

__version__ = "1.3.0"

__all__ = [
    "AgentSession",
    "AgentMode",
    "MCPConfig",
    "MCPManager",
    "MCPServerConfig",
    "Message",
    "ToolCall",
    "ToolSpec",
    "__version__",
]
