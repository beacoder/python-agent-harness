"""Tool registry with default tools."""

from __future__ import annotations

from .agent_tool import AgentTool
from .base import PendingToolResult, Registry, Tool, ToolContext
from .bash import Bash
from .filesystem import Edit, GlobTool, Grep, Insert, Mkdir, Read, Write
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
    for tool in (
        AgentTool(),
        TodoWrite(),
        GlobTool(),
        Grep(),
        Read(),
        Insert(),
        Edit(),
        Write(),
        Mkdir(),
        Bash(),
        Skill(),
        Question(),
    ):
        reg.register(tool)
    return reg
