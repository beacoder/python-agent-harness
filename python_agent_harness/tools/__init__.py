"""Tool registry with default tools."""

from __future__ import annotations

import sys

from .agent_tool import AgentTool
from .base import PendingToolResult, Registry, Tool, ToolContext
from .bash import Bash
from .bash_win import BashWindows
from .edit import Edit
from .edit_mac import EditMac
from .edit_win import EditWindows
from .filesystem import GlobTool, Grep, Insert, Mkdir, Read, Write
from .glob_mac import GlobMac
from .glob_win import GlobWindows
from .grep_mac import GrepMac
from .grep_win import GrepWindows
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
    "BashWindows",
    "Edit",
    "EditMac",
    "EditWindows",
    "GlobMac",
    "GlobTool",
    "GlobWindows",
    "Grep",
    "GrepMac",
    "GrepWindows",
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
    # Windows lacks process groups, select on pipes, SIGKILL, `patch`,
    # `tree`, `find`, and `grep` — Windows variants use pure-Python
    # fallbacks and CREATE_NEW_PROCESS_GROUP + taskkill instead.
    _mac = sys.platform == "darwin"
    _win = sys.platform == "win32"
    edit_tool = EditWindows() if _win else EditMac() if _mac else Edit()
    glob_tool = GlobWindows() if _win else GlobMac() if _mac else GlobTool()
    grep_tool = GrepWindows() if _win else GrepMac() if _mac else Grep()
    bash_tool = BashWindows() if _win else Bash()
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
        bash_tool,
        Skill(),
        Question(),
    ):
        reg.register(tool)
    return reg
