"""Data model classes for the agent harness."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any


class AgentMode(enum.Enum):
    BUILD = "build"
    PLAN = "plan"


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] | str
    result: str | None = None
    diff: str | None = None  # unified diff for Edit/Write, for TUI rendering
    elapsed: float | None = None  # execution wall-time in seconds (TUI display)


@dataclass
class Message:
    """One conversation message in OpenAI-compatible format.

    ``role`` is one of system/user/assistant/tool.
    ``content`` may be a str or a list of parts (multimodal).
    ``tool_calls`` carries requested tool invocations on assistant messages.
    ``tool_call_id`` links a tool message to its assistant tool call.
    ``reasoning`` holds reasoning content if the backend reports it.
    """

    role: str
    content: str | list[Any] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    reasoning: str | None = None
    name: str | None = None
    injected: bool = False  # harness-injected (nudge/plan/build-switch), not user input

    def to_api(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self._api_content()
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                        if isinstance(tc.arguments, str)
                        else json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    def _api_content(self) -> str | list[Any] | None:
        """Content as sent over the wire, with the reasoning preamble removed.

        The client merges streamed ``reasoning_content`` ahead of the
        answer into ``content`` (so the live stream and stored history
        show the model's thinking).  That reasoning is bookkeeping for
        the current turn only — re-sending it on later turns just
        inflates the context (and skews token estimation) and can
        confuse the model, so it is stripped here at the API boundary.
        The stored ``content`` is left untouched (the TUI collapses the
        reasoning for display via its own helper).
        """
        content = self.content
        if self.reasoning and isinstance(content, str):
            if content.startswith(self.reasoning):
                return content[len(self.reasoning) :].lstrip("\n")
            stripped = content.lstrip()
            if stripped.startswith(self.reasoning):
                return stripped[len(self.reasoning) :].lstrip("\n")
        return content

    def text(self) -> str:
        """Plain text of the message; empty when no text parts exist."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts: list[str] = []
            for p in self.content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    if isinstance(p.get("text"), str):
                        parts.append(p["text"])
                    elif isinstance(p.get("thinking"), str):
                        parts.append(p["thinking"])
            return "".join(parts)
        return ""

    def text_without_reasoning(self) -> str:
        """Plain text with the reasoning preamble stripped.

        Use this for one-shot results (compaction, summary, title) where
        the reasoning chain should not leak into the stored output.
        """
        t = self.text()
        if self.reasoning and t:
            if t.startswith(self.reasoning):
                return t[len(self.reasoning) :].lstrip("\n")
            stripped = t.lstrip()
            if stripped.startswith(self.reasoning):
                return stripped[len(self.reasoning) :].lstrip("\n")
            # Fallback: remove the reasoning anywhere in the text
            return t.replace(self.reasoning, "").strip()
        return t


@dataclass
class ToolSpec:
    """A tool exposed to the model (JSON schema)."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_api(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
