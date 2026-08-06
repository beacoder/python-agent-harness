"""Context compaction.

Ported from gptel-agent-harness.el: on high context usage, abort the
current round, summarize the whole conversation (compact.txt as system
prompt, tools disabled), wrap the summary in a compact frame, reset the
cache epoch, and resume with the last user request.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import config


def read_prompt_file(name: str) -> str:
    """Read a prompt file from the package prompts dir."""
    path = Path(__file__).parent / "prompts" / name
    return path.read_text(encoding="utf-8")


_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)
_SKILLS_PLACEHOLDER_RE = re.compile(r"\{\{\s*SKILLS\s*\}\}")
_SKILLS_FALLBACK = (
    "Invoke with a skill name and optional args; the tool reports an "
    "error if no matching skill is found."
)


def strip_frontmatter(text: str) -> str:
    """Strip a leading YAML frontmatter block (--- ... ---) if present."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def load_agent_prompt(path: "Path | str | None") -> str | None:
    """Load an opencode-style agent prompt file, or None if unavailable.

    Strips the YAML frontmatter header (name/description/tools) since
    that metadata isn't part of the prompt text, and substitutes the
    ``{{SKILLS}}`` placeholder (no runtime skill listing is generated
    here) with a short static fallback note.  Missing files, unreadable
    files, and empty files all resolve to None so callers can fall back
    cleanly to no system prompt.
    """
    if not path:
        return None
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    text = strip_frontmatter(text)
    text = _SKILLS_PLACEHOLDER_RE.sub(_SKILLS_FALLBACK, text)
    text = text.strip()
    return text or None


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
