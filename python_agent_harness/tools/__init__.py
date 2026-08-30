"""Tool registry with default tools."""

from __future__ import annotations

import sys

from .agent_tool import AgentTool
from .base import PendingToolResult, Registry, Tool, ToolContext
from .bash import Bash
from .edit import Edit
from .edit_mac import EditMac
from .filesystem import GlobTool, Grep, Insert, Mkdir, Read, Write
from .mcp import MCPTool, mcp_tools_from_manager, normalize_mcp_result
from .planexit import PlanExit
from .question import Question
from .skill import Skill
from .todo import TodoWrite

__all__ = [
    "PendingToolResult",
    "Registry",
    "Tool",
    "ToolContext",
    "AgentTool",
    "Bash",
    "Edit",
    "EditMac",
    "GlobTool",
    "Grep",
    "Insert",
    "MCPTool",
    "Mkdir",
    "PlanExit",
    "Question",
    "Read",
    "Skill",
    "TodoWrite",
    "Write",
    "mcp_tools_from_manager",
    "normalize_mcp_result",
]


def default_registry() -> Registry:
    reg = Registry()
    # macOS's BSD patch rejects well-formed hunks that GNU patch accepts,
    # so macOS uses EditMac (pure-Python diff applier); Linux keeps the
    # patch-binary Edit.  Both register under the name "Edit".
    edit_tool = EditMac() if sys.platform == "darwin" else Edit()
    for tool in (
        AgentTool(),
        TodoWrite(),
        GlobTool(),
        Grep(),
        Read(),
        Insert(),
        edit_tool,
        Write(),
        Mkdir(),
        Bash(),
        Skill(),
        Question(),
    ):
        reg.register(tool)
    return reg
