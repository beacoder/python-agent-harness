"""Tool registry with default tools."""

from __future__ import annotations

from .base import PendingToolResult, Registry, Tool, ToolContext
from .agent_tool import AgentTool
from .bash import Bash
from .filesystem import Edit, GlobTool, Grep, Insert, Mkdir, Read, Write
from .planexit import PlanExit
from .question import Question
from .skill import Skill
from .todo import TodoWrite

__all__ = [
    "PendingToolResult", "Registry", "Tool", "ToolContext",
    "AgentTool", "Bash", "Edit", "GlobTool", "Grep", "Insert",
    "Mkdir", "PlanExit", "Question", "Read", "Skill", "TodoWrite", "Write",
]


def default_registry() -> Registry:
    reg = Registry()
    for tool in (
        AgentTool(), TodoWrite(), GlobTool(), Grep(), Read(), Insert(),
        Edit(), Write(), Mkdir(), Bash(), Skill(), Question(),
    ):
        reg.register(tool)
    return reg
