"""Agent tool: spawn sub-agents for delegated work.

Sub-agents run the same agent loop with a fresh FSM; their backend/model
can be overridden (see config).  Results flow back to the parent as a
single tool result string.  Errors are contained: an unexpected sub-agent
response becomes an error string fed to the parent, never a crash.
"""

from __future__ import annotations

import json
import threading

from .base import Tool, ToolContext

DESCRIPTION = (
    "Launch a specialized sub-agent to handle complex, multi-step tasks "
    "autonomously. Sub-agents run independently and return results in one "
    "message. Use for open-ended searches, complex research, or when "
    "uncertain about finding results in the first few tries."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "subagent_type": {
            "type": "string",
            "description": "Type of sub-agent: 'subagent' or 'gptel-opencode-agent'",
        },
        "description": {"type": "string", "description": "Short 3-5 word description of the task"},
        "prompt": {"type": "string", "description": "The detailed task for the sub-agent"},
    },
    "required": ["subagent_type", "description", "prompt"],
}


class AgentTool(Tool):
    name = "Agent"
    description = DESCRIPTION
    parameters = PARAMETERS

    def run(self, args: dict, ctx: ToolContext) -> str:
        prompt = args.get("prompt", "")
        description = args.get("description", "task")
        subagent_type = args.get("subagent_type", "subagent")
        if not prompt:
            return "Error: prompt must not be empty"
        result = ctx.run_subagent(subagent_type, description, prompt)
        return result
