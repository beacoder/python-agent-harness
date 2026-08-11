"""Tool base classes and the tool registry."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any

from ..models import ToolSpec


class PendingToolResult:
    """Handle for an asynchronous tool result (mirrors ``:async t``).

    An async tool's ``run`` returns this handle instead of a string: it
    starts its background work (e.g. a spawned process) and returns
    immediately, then delivers the final result string later via
    ``deliver`` — so the wait never blocks the sequential tool loop.

    ``deliver`` is idempotent (first delivery wins, late duplicates are
    no-ops — mirroring the gptel-agent FSM's idempotent-result advice);
    ``wait`` blocks until the result has been delivered.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: str | None = None

    def deliver(self, result: str) -> None:
        if not self._event.is_set():
            self._result = result
            self._event.set()

    def wait(self) -> str:
        self._event.wait()
        return self._result or ""


class ToolContext:
    """Runtime context handed to tools.

    Tools may call back into the session for user questions,
    plan-mode checks, and sub-agent delegation.  All methods
    proxy to the session when present; defaults are safe no-ops.
    """

    def __init__(self, session: Any = None) -> None:
        self.session = session

    @property
    def cwd(self) -> str:
        return self.session.project_dir if self.session else "."

    def ask_questions(self, questions: list[dict]) -> str:
        if self.session and hasattr(self.session, "ask_questions"):
            return self.session.ask_questions(questions)
        return "Unanswered"

    def record_diff(self, diff_text: str) -> None:
        """Attach a unified diff to the currently-executing tool call."""
        if self.session and hasattr(self.session, "record_diff"):
            self.session.record_diff(diff_text)

    def update_todos(self, todos: list[dict]) -> None:
        if self.session and hasattr(self.session, "update_todos"):
            self.session.update_todos(todos)

    def find_skill(self, name: str) -> str | None:
        if self.session and hasattr(self.session, "find_skill"):
            return self.session.find_skill(name)
        return None

    def run_subagent(self, subagent_type: str, description: str, prompt: str) -> str:
        if self.session and hasattr(self.session, "run_subagent"):
            return self.session.run_subagent(subagent_type, description, prompt)
        return f"Error: Task {description!r} returned an unexpected response — no session"

    def plan_exit(self) -> str:
        if self.session and hasattr(self.session, "plan_exit"):
            return self.session.plan_exit()
        return "Not in plan mode; PlanExit has no effect.  Continue as normal."

    @property
    def cancel_event(self) -> Any:
        """Session cancel event (set when the user presses Ctrl-C)."""
        if self.session and hasattr(self.session, "cancel_event"):
            return self.session.cancel_event
        return None


class Tool(ABC):
    name: str = ""
    description: str = ""
    # NB: Tool is an ABC, not a dataclass, so this is a plain class-level
    # default (never mutated in place — every concrete tool overrides it
    # with its own schema).  It must be a real dict: a dataclasses.field()
    # sentinel here would silently become the "parameters" of any tool
    # that forgot to override it and then fail JSON serialization.
    parameters: dict[str, Any] = {}

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> str | PendingToolResult:
        """Execute the tool and return the result string.

        Async tools return a ``PendingToolResult`` instead (see
        ``Bash``): the background work is spawned here and the final
        string is delivered later via ``PendingToolResult.deliver``.
        """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        wanted = set(names) if names is not None else set(self._tools)
        return [t.spec() for name, t in self._tools.items() if name in wanted]

    def execute(
        self, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> str | PendingToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}"
        try:
            return tool.run(args, ctx)
        except Exception as e:  # noqa: BLE001 - errors become tool results
            return f"Error: tool {name} failed — {e}"
