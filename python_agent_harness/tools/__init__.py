"""Tool registry with default tools."""

from __future__ import annotations

import sys

from .agent_tool import AgentTool
from .base import PendingToolResult, Registry, Tool, ToolContext
from .bash import Bash
from .edit import Edit
from .edit_mac import EditMac
from .filesystem import GlobTool, Grep, Insert, Mkdir, Read, Write
from .glob_mac import GlobMac
from .grep_mac import GrepMac
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
    "GlobMac",
    "GlobTool",
    "Grep",
    "GrepMac",
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
    # macOS lacks GNU `patch`, `tree`, and PCRE-enabled git — use
    # pure-Python / POSIX-compatible Mac variants that register under
    # the same tool names so callers are platform-independent.
    _mac = sys.platform == "darwin"
    edit_tool = EditMac() if _mac else Edit()
    glob_tool = GlobMac() if _mac else GlobTool()
    grep_tool = GrepMac() if _mac else Grep()
    for tool in (
        AgentTool(),
        TodoWrite(),
        glob_tool,
        grep_tool,
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
