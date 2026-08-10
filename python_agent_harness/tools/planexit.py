"""PlanExit tool: ask the user to approve switching from plan to build.

Asynchronous (mirrors ``:async t``): ``run`` returns a
``PendingToolResult`` immediately and a background thread asks for the
approval, delivering the outcome when the user responds — the wait for
user input never occupies a thread-pool slot.
"""

from __future__ import annotations

import threading

from .base import PendingToolResult, Tool, ToolContext

DESCRIPTION = (
    "Use this tool when you have completed the planning phase and are "
    "ready to exit plan mode.\n\n"
    "This tool will ask the user whether they want to switch to the build "
    "agent and start implementing the plan. Do NOT use the Question tool "
    "to ask \"Is this plan okay?\" — that is what this tool is for.\n\n"
    "Call this tool:\n"
    "- After you have written a complete plan to the plan file\n"
    "- After you have clarified any questions with the user\n"
    "- When you are confident the plan is ready for implementation\n\n"
    "Do NOT call this tool:\n"
    "- Before you have created or finalized the plan\n"
    "- If you still have unanswered questions about the implementation\n"
    "- If the user has indicated they want to continue planning\n\n"
    "On approval, the session switches to build mode (file edits become "
    "allowed) and you should proceed to execute the approved plan. On "
    "rejection, you remain in the read-only plan phase and should continue "
    "refining the plan."
)


class PlanExit(Tool):
    name = "PlanExit"
    description = DESCRIPTION
    parameters = {"type": "object", "properties": {}}

    def run(self, args: dict, ctx: ToolContext) -> str | PendingToolResult:
        pending = PendingToolResult()

        def worker() -> None:
            # containment boundary: a failure in the approval prompt
            # becomes an error string for the model, never a crash
            try:
                result = ctx.plan_exit()
            except Exception as e:  # noqa: BLE001 - error string for the model
                result = f"Error: PlanExit failed — {e}"
            pending.deliver(result)

        threading.Thread(target=worker, daemon=True, name="planexit-tool").start()
        return pending
