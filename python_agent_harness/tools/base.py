"""Tool base classes and the tool registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any

from ..models import ToolSpec


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
    parameters: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Execute the tool and return the result string."""

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
    ) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}"
        try:
            return tool.run(args, ctx)
        except Exception as e:  # noqa: BLE001 - errors become tool results
            return f"Error: tool {name} failed — {e}"
