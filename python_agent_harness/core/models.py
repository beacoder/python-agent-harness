"""Data model classes for the agent harness."""

from __future__ import annotations

import base64
import enum
import json
import re
from dataclasses import dataclass, field
from typing import Any, Union

# Lone surrogates (\ud800-\udfff) reach a str when a file is decoded with
# errors="surrogateescape" — which every tool deliberately does, so a
# non-UTF-8 source file round-trips instead of becoming U+FFFD.  Such a
# string cannot be encoded to UTF-8, so anything that leaves the process
# (a JSON request body, a terminal write) must replace them first.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def api_safe(text: str) -> str:
    """TEXT with lone surrogates replaced by U+FFFD, safe to serialize.

    The tools read files with ``errors="surrogateescape"``, so a
    conversation can carry lone surrogates from a non-UTF-8 (Latin-1,
    GBK, ...) source.  A request body containing one cannot be encoded
    to UTF-8 — ``httpx`` raised ``UnicodeEncodeError`` — so they are
    replaced at the OUTBOUND boundary only: the stored history, the
    session file and the file on disk keep the exact bytes.
    """
    return _SURROGATE_RE.sub("\ufffd", text)


class AgentMode(enum.Enum):
    BUILD = "build"
    PLAN = "plan"


@dataclass
class TextPart:
    """A plain-text content part in a multimodal message."""

    text: str

    def to_api(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass
class ImagePart:
    """A provider-neutral image attachment in a multimodal message.

    The image source is either ``data`` (raw image bytes, with
    ``media_type`` as the MIME type, e.g. ``"image/png"``) or ``url``
    (an http(s) URL the provider fetches).  At least one of the two must
    be set; when both are, ``url`` wins.  The conversion to a provider-specific format (e.g.
    OpenAI's ``image_url`` with a data URL or a plain URL) happens in
    ``to_api()``, which is only called at the API serialization boundary
    (``Message.to_api()`` → ``Client._payload``), never in the agent core.

    ``path`` (when set) is the filesystem path the image was attached
    from.  It is metadata only — never sent to the API — and lets
    session persistence record where the image came from so a restored
    session can re-attach it.  Images that did not originate from a
    path (drag-drop, URLs) leave it None.
    """

    data: bytes | None = None
    media_type: str = "image/png"
    path: str | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        # to_api() runs on every round (token estimation + API payload),
        # so the base64 data URL is encoded once, not per call.
        self._data_url: str | None = None
        if self.data is not None:
            self._data_url = (
                f"data:{self.media_type};base64,"
                f"{base64.b64encode(self.data).decode('ascii')}"
            )

    @classmethod
    def from_url(cls, url: str) -> ImagePart:
        return cls(url=url)

    def to_api(self) -> dict[str, Any]:
        if self.url:
            return {"type": "image_url", "image_url": {"url": self.url}}
        if self._data_url is None:
            raise ValueError("ImagePart requires either data or url")
        return {"type": "image_url", "image_url": {"url": self._data_url}}


# Type alias for the individual parts a Message.content list may contain.
ContentPart = Union[TextPart, ImagePart, str, "dict[str, Any]"]


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] | str
    result: str | None = None
    diff: str | None = None  # unified diff for Edit/Write, for TUI rendering
    elapsed: float | None = None  # execution wall-time in seconds (TUI display)


def arg_repr(value: Any, limit: int = 100) -> str:
    """repr() of a tool-call argument, truncated to LIMIT chars total.

    Long values (a long Bash command, a big JSON payload) are cut with
    an ellipsis instead of being dropped from the label entirely.
    """
    r = repr(value)
    if len(r) <= limit:
        return r
    return r[: limit - 1] + "…"


def display_args(arguments: dict[str, Any] | str, limit: int = 100) -> dict[str, str]:
    """Tool-call arguments as short display strings, keyed by name.

    ``arguments`` arrives either already decoded or as the raw JSON
    string the model emitted (unparseable when the model truncated it).
    ``content`` is skipped: a Write/Edit payload is far too large for a
    one-line label, and it is shown as a diff instead.

    Values are truncated, so the result is a display artifact -- never
    the data a tool is executed with.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return {}
    if not isinstance(arguments, dict):
        return {}
    return {k: arg_repr(v, limit) for k, v in arguments.items() if k != "content"}


@dataclass
class Message:
    """One conversation message in OpenAI-compatible format.

    ``role`` is one of system/user/assistant/tool.
    ``content`` may be a str, a list of parts (multimodal:
    ``TextPart``, ``ImagePart``, plain strings, or dicts), or None.
    ``tool_calls`` carries requested tool invocations on assistant messages.
    ``tool_call_id`` links a tool message to its assistant tool call.
    ``reasoning`` holds reasoning content if the backend reports it.
    """

    role: str
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    reasoning: str | None = None
    name: str | None = None
    injected: bool = False  # harness-injected (nudge/plan/build-switch), not user input

    def to_api(self) -> dict[str, Any]:
        # Not cached: the per-message dict is cheap to rebuild and
        # caching it (returning the same object each call) would make a
        # mutation by any caller corrupt later rounds and token
        # estimation.  The one expensive step — base64-encoding an image
        # — is cached on ``ImagePart`` instead (see ImagePart.__post_init__).
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            content = self._api_content()
            # outbound boundary: a lone surrogate cannot be encoded into
            # the JSON body (see api_safe)
            d["content"] = api_safe(content) if isinstance(content, str) else content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # ensure_ascii=False keeps non-ASCII readable;
                        # api_safe then removes the unencodable surrogates
                        "arguments": api_safe(
                            tc.arguments
                            if isinstance(tc.arguments, str)
                            else json.dumps(tc.arguments, ensure_ascii=False)
                        ),
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

        For multimodal content (a list of parts), each part is
        serialized to its provider-specific dict representation via
        ``to_api()`` when the part is a ``TextPart`` or ``ImagePart``;
        strings and dicts pass through unchanged.
        """
        content = self.content
        if self.reasoning and isinstance(content, str):
            if content.startswith(self.reasoning):
                return content[len(self.reasoning) :].lstrip("\n")
            stripped = content.lstrip()
            if stripped.startswith(self.reasoning):
                return stripped[len(self.reasoning) :].lstrip("\n")
        if isinstance(content, list):
            return [self._part_to_api(p) for p in content]
        return content

    @staticmethod
    def _part_to_api(part: ContentPart) -> Any:
        """Serialize one content part to its provider-specific dict."""
        if isinstance(part, TextPart):
            # text parts are a second carrier of file content into the
            # payload, so they need the same outbound treatment
            return {"type": "text", "text": api_safe(part.text)}
        if isinstance(part, ImagePart):
            return part.to_api()
        return part

    def text(self) -> str:
        """Plain text of the message; empty when no text parts exist."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts: list[str] = []
            for p in self.content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, TextPart):
                    parts.append(p.text)
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
