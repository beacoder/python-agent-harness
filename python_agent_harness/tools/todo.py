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

INSTRUCTIONS = """\
You MUST create a todo list immediately when:
- Task has 3+ distinct steps or phases
- Task is non-trivial and benefits from planning
- Task will span multiple responses or tool calls
- The user provides multiple tasks (numbered or comma-separated) or explicitly asks for a todo list
- New instructions arrive - capture them as todos
- You start a task - mark it `in_progress` (only one at a time) before working
- You finish a task - mark it `completed` and add any follow-ups discovered during the work

When NOT to use `TodoWrite`:
- Single, straightforward tasks (or <3 trivial steps)
- The request is purely informational or conversational
- Tracking adds no organizational value

Task States:
- `pending`: Task not yet started
- `in_progress`: Currently working on (exactly one at a time)
- `completed`: Task finished successfully

Rules:
- Update status in real time; don't batch completions
- Mark `completed` only after the required work is actually done, including any required verification. Never based on intent.
- If blocked or partial, keep it `in_progress` and add a follow-up todo describing the blocker
- Preserve user-provided commands verbatim (flags, args, order)
- Items should be specific and actionable; break large work into smaller steps

How to use `TodoWrite`:
- Always provide both `content` (imperative: "Run tests") and `activeForm` (present continuous: "Running tests")
- Exactly ONE task must be in_progress at any time when you're executing tasks yourself
- When delegating to subagents in parallel, multiple tasks can be in_progress simultaneously
- Complete current tasks before starting new ones
- Send entire todo list with each call (not just changed items)
- Remove tasks that are no longer relevant

Examples:
**Use it:**
- "Add a dark mode toggle and run the tests" -> multi-step feature + explicit verification
- "Rename getCwd -> getCurrentWorkingDirectory across the repo" -> grep reveals 15 occurrences in 8 files
- "Implement registration, catalog, cart, checkout" -> multiple complex features

**Skip it:**
- "How do I print Hello World in Python?" -> informational
- "Add a comment to calculateTotal" -> single edit
- "Run npm install and tell me what happened" -> one command

When in doubt, use it.
"""

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
    instructions = INSTRUCTIONS
    parameters = PARAMETERS

    def run(self, args: dict, ctx: ToolContext) -> str:
        todos = args.get("todos") or []
        ctx.update_todos(todos)
        return json.dumps({"todos": todos, "count": len(todos)}, ensure_ascii=False)
