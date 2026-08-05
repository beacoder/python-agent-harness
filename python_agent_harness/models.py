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

    def to_api(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
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

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "Message":
        role = raw.get("role", "assistant")
        content = raw.get("content")
        tool_calls = None
        if raw.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", "{}"),
                )
                for tc in raw["tool_calls"]
            ]
        return cls(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=raw.get("tool_call_id"),
            reasoning=raw.get("reasoning_content"),
        )

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


@dataclass
class SessionInfo:
    """Metadata saved with a session file."""

    project_dir: str | None = None
    model: str | None = None
    backend: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tool_names: list[str] = field(default_factory=list)
    title: str | None = None
    file_path: str | None = None
