"""Prompt loading and assembly.

Ported from gptel-agent-harness.el: loads bundled prompt files
(agent/subagent/commands), strips YAML frontmatter, discovers skills
for the {{SKILLS}} placeholder, assembles the effective system prompt
from project context files + task-completion rules + agent prompt, and
provides last_user_request() for the compaction flow (summarize the
conversation and resume with the last user request).
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


def _parse_skill_frontmatter(skill_file: Path) -> tuple[str, str] | None:
    """Extract (name, description) from a SKILL.md frontmatter block."""
    try:
        text = skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.match(r"\A---\n(.*?\n)---\n?", text, re.DOTALL)
    if not m:
        return None
    name = desc = ""
    for line in m.group(1).splitlines():
        if line.startswith("name:"):
            name = line[len("name:"):].strip()
        elif line.startswith("description:"):
            desc = line[len("description:"):].strip()
    if name:
        return (name, desc)
    return None


def discover_skills(skill_dir: "Path | str | None") -> str:
    """Build a skill listing from a skill directory.

    Looks for subdirectories containing SKILL.md with frontmatter
    (name/description).  Returns a formatted listing string, or the
    static fallback if no skills are found.
    """
    if not skill_dir:
        return _SKILLS_FALLBACK
    d = Path(skill_dir)
    if not d.is_dir():
        return _SKILLS_FALLBACK
    entries: list[tuple[str, str]] = []
    for child in sorted(d.iterdir()):
        if not child.is_dir():
            continue
        skill_file = child / "SKILL.md"
        if skill_file.is_file():
            parsed = _parse_skill_frontmatter(skill_file)
            if parsed:
                entries.append(parsed)
    if not entries:
        return _SKILLS_FALLBACK
    lines = ["<available-skills>"]
    for name, desc in entries:
        lines.append(f"  <skill>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{desc}</description>")
        lines.append(f"  </skill>")
    lines.append("</available-skills>")
    return "\n".join(lines)


def load_agent_prompt(path: "Path | str | None", skill_dir: "Path | str | None" = None) -> str | None:
    """Load an opencode-style agent prompt file, or None if unavailable.

    Strips the YAML frontmatter header (name/description/tools) since
    that metadata isn't part of the prompt text, and substitutes the
    ``{{SKILLS}}`` placeholder with the discovered skill listing from
    *skill_dir* (or a static fallback if no skills are found).

    Missing files, unreadable files, and empty files all resolve to None
    so callers can fall back cleanly to no system prompt.
    """
    if not path:
        return None
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    text = strip_frontmatter(text)
    skills_text = discover_skills(skill_dir)
    text = _SKILLS_PLACEHOLDER_RE.sub(skills_text, text)
    text = text.strip()
    return text or None


def load_context_files(context_dir: "Path | str | None") -> str | None:
    """Read all files in *context_dir* and format them as context blocks.

    Returns a string like:
        Request context:

        In file `~/.emacs.d/contexts/README.md`:

        ```
        <file contents>
        ```

    Returns None if no context directory or no readable files.
    """
    if not context_dir:
        return None
    d = Path(context_dir)
    if not d.is_dir():
        return None
    blocks: list[str] = []
    for child in sorted(d.iterdir()):
        if not child.is_file():
            continue
        try:
            content = child.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not content.strip():
            continue
        blocks.append(f"In file `{child}`:\n\n```\n{content.rstrip()}\n```")
    if not blocks:
        return None
    return "Request context:\n\n" + "\n\n".join(blocks)


def load_task_completion_rules() -> str | None:
    """Load ``prompts/task-completion-rules.txt``, or None if unavailable.

    These rules are injected automatically into every agent system
    prompt (main agent, sub-agents, and session commands), so the model
    never stops before the task is fully completed and verified.
    """
    p = Path(__file__).parent / "prompts" / "task-completion-rules.txt"
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    text = text.strip()
    return text or None


def assemble_agent_prompt(
    project_dir: str,
    agent_prompt: str | None,
    include_context: bool = True,
    context_path: str | None = None,
) -> str | None:
    """Assemble the effective system prompt for an agent run.

    Order: [project context files] -> task-completion-rules.txt ->
    the actual agent prompt.  The completion rules are always the LAST
    context piece, immediately before the agent prompt, so they read as
    global ground rules rather than part of the task instructions.

    ``include_context=False`` drops the project context files (used for
    sub-agents, which get the rules but not the parent's context).
    ``context_path`` overrides the default context directory discovery.
    Returns None if every part is empty/missing.
    """
    parts: list[str] = []
    if include_context:
        # lazy import: harness imports this module at call time
        from .agent_session import find_context_dir

        context_block = load_context_files(find_context_dir(project_dir, context_path))
        if context_block:
            parts.append(context_block)
    rules = load_task_completion_rules()
    if rules:
        parts.append(rules)
    if agent_prompt:
        parts.append(agent_prompt)
    return "\n\n".join(parts) if parts else None


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
