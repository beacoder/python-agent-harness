"""Agent discovery and management.

Custom agents are markdown files (``*.md``) in this directory —
``python_agent_harness/agents/``.  Each file becomes a switchable
agent profile available via the ``/agent`` TUI command.

The file's stem (e.g. ``reviewer.md`` → ``reviewer``) is the agent
name.  An optional YAML frontmatter block (``---\\nname: ...\\n---``)
can override the name.  The file body (after frontmatter) is the
agent's system prompt; project context and task-completion rules are
still prepended at assembly time.

The built-in ``default`` agent (``prompts/agent.md``) is always
available and cannot be overridden by a file in this directory.
"""

from __future__ import annotations

import re
from pathlib import Path

AGENTS_DIR = Path(__file__).parent

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n?", re.DOTALL)


def _agent_name_from_file(path: Path) -> str:
    """Derive agent name from a file: frontmatter ``name:`` if present,
    otherwise the file stem (lowercased, non-alphanumerics → ``-``)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = _FRONTMATTER_RE.match(text)
    if m:
        for line in m.group(1).splitlines():
            if line.startswith("name:"):
                name = line[len("name:") :].strip()
                if name:
                    return name
    base = path.stem.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base)
    return base.strip("-")


def discover_agents() -> dict[str, str]:
    """Discover all custom agents in the agents directory.

    Returns a dict mapping agent name → absolute path to the prompt
    file.  Files without a valid name are skipped.  The built-in
    ``default`` agent is NOT included here — callers add it separately.
    """
    if not AGENTS_DIR.is_dir():
        return {}
    agents: dict[str, str] = {}
    for f in sorted(AGENTS_DIR.glob("*.md")):
        name = _agent_name_from_file(f)
        if not name:
            continue
        agents[name] = str(f.resolve())
    return agents
