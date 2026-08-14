"""TodoWrite tool: maintain a structured task list during execution."""

from __future__ import annotations

import json

from .base import Tool, ToolContext

DESCRIPTION = (
    "Create and manage a structured task list for your session. Helps track "
    "progress on complex, multi-step tasks.\n\n"
    "Use it when:\n"
    "- The task has 3+ distinct steps or phases\n"
    "- The task is non-trivial and benefits from planning\n"
    "- You start a task (mark it in_progress) or finish a task (mark it completed)\n\n"
    "Task states: pending (not started), in_progress (currently working on), "
    "completed (finished). Only one task can be in_progress at a time. Send "
    "the entire todo list with each call (not just changed items)."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Task description (imperative form)",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                    },
                    "activeForm": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Present continuous form",
                    },
                },
                "required": ["content", "status"],
            },
        }
    },
    "required": ["todos"],
}


class TodoWrite(Tool):
    name = "TodoWrite"
    description = DESCRIPTION
    parameters = PARAMETERS

    def run(self, args: dict, ctx: ToolContext) -> str:
        todos = args.get("todos") or []
        ctx.update_todos(todos)
        return json.dumps({"todos": todos, "count": len(todos)}, ensure_ascii=False)
