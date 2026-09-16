"""Agent tool: spawn sub-agents for delegated work.

Asynchronous (mirrors ``:async t``): ``run`` returns a
``PendingToolResult`` immediately and a background thread runs the
sub-agent loop, delivering the result string when it finishes — a
long-running sub-agent never blocks the parent's sequential tool loop.

Sub-agents run the same agent loop with a fresh loop instance; their
model can be overridden (see config).  Results flow back to the
parent as a single tool result string.  Errors are contained: an
unexpected sub-agent response becomes an error string fed to the parent,
never a crash.

Tool execution mirrors gptel: synchronous tools (Read, Edit, ...) run
ONE AT A TIME in model-emitted order, while asynchronous tools — Agent
and Bash — are dispatched in line and run concurrently in the
background.  Each sub-agent is fully isolated (own loop, own history,
own stream), so independent tasks can be delegated in parallel.
"""

from __future__ import annotations

import threading

from .base import PendingToolResult, Tool, ToolContext

DESCRIPTION = (
    "Launch a specialized sub-agent to handle complex, multi-step tasks "
    "autonomously. Sub-agents run independently and return results in one "
    "message. Use for open-ended searches, complex research, or when "
    "uncertain about finding results in the first few tries.\n\n"
    "Multiple Agent calls issued in the same round run concurrently (like "
    "Bash), while other tools execute one by one — delegate independent "
    "tasks in parallel for efficiency."
)

INSTRUCTIONS = """\
**MANDATORY delegation scenarios (use Agent immediately):**
- **Searching codebase for code understanding or information gathering** → DELEGATE to `Agent`
- **Exploring unfamiliar code with uncertain search paths** → DELEGATE to `Agent`
- **Expected to search 3+ files or get many search results** → DELEGATE to `Agent`
- **Well-defined multi-step task that will bloat your context** → DELEGATE to `Agent`
- **Creating/modifying 3+ files with clear requirements** → DELEGATE to `Agent`

**When NOT to use `Agent`:**
- You know exact file paths and just need to read 1-2 specific files → use `Read`
- Searching for ONE specific, well-defined string in known location → use `Grep`
- User provides specific file paths to examine → handle inline
- Simple, focused task with all information available → handle inline
- Quick edits to 1-2 files → handle inline
- Finding a specific item (e.g., "read the config in settings.py") → Handle inline

**How to use the `Agent` tool:**
- Agents run autonomously and return a single summary message
- Review the result, then proactively integrate it into your reply to user

**Context isolation (CRITICAL):**
- Subagents have NO access to prior conversation history
- Include all necessary context in the prompt: file paths, requirements, constraints, coding conventions
- Reference specific file paths rather than "the file we discussed earlier"
- Be detailed and comprehensive in the prompt — the subagent starts from scratch

**Parallel vs Sequential:**
- Use parallel agents for independent tasks (e.g., searching two unrelated areas)
- Use sequential when one result feeds the next (e.g., find files → then edit them)
- Parallel agents cannot communicate with each other

**Result handling:**
- Trust subagent results for information gathering and exploration
- Verify subagent file modifications by reading key files if the change is complex or safety-critical
- If a subagent returns an error or incomplete result, retry with refined instructions

**Examples of good prompts:**
- "Search for all files under src/auth/ that import SessionManager. Read each file and summarize how session expiry is handled."
- "Create unit tests for src/utils/parser.ts. Follow the test patterns in src/utils/__tests__/formatter.test.ts. Use vitest as the test framework."
- "Find all usages of the deprecated `fetchData()` API in the project and replace them with `queryData()`. Preserve all existing arguments."
"""

PARAMETERS = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "description": "Short 3-5 word description of the task"},
        "prompt": {"type": "string", "description": "The detailed task for the sub-agent"},
    },
    "required": ["description", "prompt"],
}


class AgentTool(Tool):
    name = "Agent"
    description = DESCRIPTION
    instructions = INSTRUCTIONS
    parameters = PARAMETERS

    def run(self, args: dict, ctx: ToolContext) -> str | PendingToolResult:
        prompt = args.get("prompt", "")
        description = args.get("description", "task")
        if not prompt:
            return "Error: prompt must not be empty"

        pending = PendingToolResult()

        def worker() -> None:
            # containment boundary: a sub-agent failure becomes an error
            # string for the parent, never a crash in the delivery thread
            try:
                result = ctx.run_subagent(description, prompt)
            except Exception as e:  # noqa: BLE001 - error string for the parent
                result = f"Error: Task {description!r} failed — {e}"
            pending.deliver(result)

        threading.Thread(target=worker, daemon=True, name="subagent-tool").start()
        return pending
