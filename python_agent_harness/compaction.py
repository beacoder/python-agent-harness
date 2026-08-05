"""Context compaction.

Ported from gptel-agent-harness.el: on high context usage, abort the
current round, summarize the whole conversation (compact.txt as system
prompt, tools disabled), wrap the summary in a compact frame, reset the
cache epoch, and resume with the last user request.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def read_prompt_file(name: str) -> str:
    """Read a prompt file from the package prompts dir."""
    path = Path(__file__).parent / "prompts" / name
    return path.read_text(encoding="utf-8")


def strip_compact_prefix(text: str) -> str:
    """Remove an existing compact frame (header + separator), keeping summary."""
    if text.startswith(config.COMPACT_HEADER):
        text = text[len(config.COMPACT_HEADER):]
    sep = config.COMPACT_SEPARATOR
    idx = text.find(sep)
    if idx != -1:
        text = text[:idx] + "\n\n" + text[idx + len(sep):]
    return text


def insert_compact_frame(summary: str) -> str:
    """Wrap SUMMARY in header + separator."""
    return (
        config.COMPACT_HEADER
        + summary
        + config.COMPACT_SEPARATOR
    )


def last_user_request(messages: list[dict]) -> str | None:
    """Return the last user message text, excluding nudge messages."""
    nudge = config.NUDGE_MESSAGE
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            text = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            )
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        if text and text != nudge:
            return text
    return None
