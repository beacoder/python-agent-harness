"""Prompt loading and assembly.

Ported from gptel-agent-harness.el: loads bundled prompt files
(agent/subagent/commands), strips YAML frontmatter, discovers skills
for the {{SKILLS}} placeholder, assembles the effective system prompt
from project context files + task-completion rules + agent prompt, and
provides user_prompt_texts() for the compaction flow (summarize the
conversation and rebuild the history with every user prompt preserved
verbatim).
"""

from __future__ import annotations

import os
import re
import subprocess
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
            name = line[len("name:") :].strip()
        elif line.startswith("description:"):
            desc = line[len("description:") :].strip()
    if name:
        return (name, desc)
    return None


def index_skills(skill_dir: Path | str | None) -> dict[str, tuple[str, str]]:
    """Index skills by their frontmatter ``name``.

    Mirrors opencode's skill service: recursively scan *skill_dir* for
    ``SKILL.md`` files (following symlinks), parse each file's YAML
    frontmatter, and return a mapping of frontmatter ``name`` to
    ``(path, description)``.  Files without a frontmatter ``name`` are
    skipped, so the advertised listing and the lookup index always agree
    on the same names.  Duplicate names keep the last file in
    sorted-path order (deterministic, like opencode's overwrite).
    """
    if not skill_dir:
        return {}
    root = Path(skill_dir)
    if not root.is_dir():
        return {}
    skills: dict[str, tuple[str, str]] = {}
    visited: set[str] = set()
    found: list[tuple[str, str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames[:] = []
            continue
        visited.add(real)
        dirnames[:] = [
            d for d in dirnames if os.path.realpath(os.path.join(dirpath, d)) not in visited
        ]
        if "SKILL.md" not in filenames:
            continue
        path = os.path.realpath(os.path.join(dirpath, "SKILL.md"))
        parsed = _parse_skill_frontmatter(Path(path))
        if parsed:
            name, desc = parsed
            found.append((path, name, desc))
    # os.walk yields directories in arbitrary order, so resolve duplicate
    # names by path explicitly: sort first, then insert, and the last file
    # in sorted-path order wins (deterministic, like opencode's overwrite).
    for path, name, desc in sorted(found):
        skills[name] = (path, desc)
    return dict(sorted(skills.items()))


def discover_skills(skill_dir: Path | str | None) -> str:
    """Build a skill listing from a skill directory.

    Recursively indexes ``SKILL.md`` files by their frontmatter
    name/description (same index the Skill tool resolves against).
    Returns a formatted listing string, or the static fallback if no
    skills are found.
    """
    skills = index_skills(skill_dir)
    if not skills:
        return _SKILLS_FALLBACK
    lines = ["<available-skills>"]
    for name, (_, desc) in skills.items():
        lines.append("  <skill>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{desc}</description>")
        lines.append("  </skill>")
    lines.append("</available-skills>")
    return "\n".join(lines)


def load_agent_prompt(path: Path | str | None, skill_dir: Path | str | None = None) -> str | None:
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


def _git_toplevel(directory: str) -> str:
    """Return the git worktree root for *directory*.

    Mirrors opencode's git.repo.discover: walk up for ``.git``, then
    run ``git rev-parse --show-toplevel``.  Falls back to the nearest
    ``.git`` parent when git itself is unusable, and to *directory*
    when no repository is found (so AGENTS.md lookup still works in
    non-git projects, bounded by the project directory).
    """
    d = Path(directory).resolve()
    for parent in [d, *d.parents]:
        if (parent / ".git").exists():
            try:
                proc = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=str(parent),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
            return str(parent)
    return str(d)


def _find_up(filename: str, start: Path, stop: Path) -> list[str]:
    """Every *filename* from *start* up to and including *stop*.

    Mirrors opencode's ``FileSystem.findUp``: collect EVERY match along
    the way, not just the nearest one, and stop after *stop* (or at the
    filesystem root, whichever comes first).
    """
    matches: list[str] = []
    current = start
    while True:
        candidate = current / filename
        if candidate.is_file():
            matches.append(str(candidate))
        if current == stop:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return matches


def find_agents_md_files(project_dir: str) -> list[str]:
    """Locate the project's ``AGENTS.md`` files, nearest first.

    Collects EVERY ``AGENTS.md`` walking up from *project_dir* to the
    git worktree root (mirrors opencode's ``Instruction.systemPaths``),
    so running the agent in a subdirectory still picks up the repo-root
    instructions.

    ``AGENTS.md`` is the ONLY recognized file: there is no user-global
    instruction file and no ``CLAUDE.md``/``CONTEXT.md`` fallback, so a
    project without an ``AGENTS.md`` gets nothing injected.  The walk
    only ever goes upward, bounded by the git worktree root (or
    *project_dir* outside a repo), so it neither escapes into unrelated
    parent directories nor discovers files in subdirectories.
    """
    start = Path(project_dir).resolve()
    stop = Path(_git_toplevel(project_dir)).resolve()
    if not start.is_relative_to(stop):
        # git reported a worktree root that is not an ancestor of the
        # resolved project dir (differently-spelled paths — symlinked or
        # automounted checkouts).  Without this guard _find_up would walk
        # to the filesystem root looking for `stop` and pick up AGENTS.md
        # files from unrelated ancestors.
        stop = start
    return _find_up("AGENTS.md", start, stop)


def load_context_files(
    context_dir: Path | str | None,
    extra_files: list[str] | None = None,
) -> str | None:
    """Format *extra_files* plus every file in *context_dir* as context.

    Returns a string like:
        Request context:

        In file `/path/to/project/AGENTS.md`:

        <file contents>

        In file `/path/to/project/contexts/README.md`:

        <file contents>

    *extra_files* are individual files outside the context directory
    (the project's ``AGENTS.md`` files) and come first, in the order
    given; the context directory's own files follow, sorted by name.
    They are read and rendered identically: an ``In file `path`:``
    header is the only delimiter, and contents are NOT wrapped in a code
    fence (a fence would be closed early by any file containing one).

    Unreadable and empty files are skipped, and a file reachable both
    ways (a *context_dir* that also holds a discovered ``AGENTS.md``) is
    rendered once.  Returns None when there is nothing to inject.
    """
    paths: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        paths.append(path)

    for extra in extra_files or []:
        _add(Path(extra))
    d = Path(context_dir) if context_dir else None
    if d and d.is_dir():
        for child in sorted(d.iterdir()):
            if child.is_file():
                _add(child)
    blocks: list[str] = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not content.strip():
            continue
        blocks.append(f"In file `{path}`:\n\n{content.rstrip()}")
    if not blocks:
        return None
    return "Request context:\n\n" + "\n\n".join(blocks)


def load_task_completion_rules() -> str | None:
    """Load ``prompts/task-completion-rules.md``, or None if unavailable.

    These rules are injected automatically into the main agent and
    session-command system prompts, so the model never stops before the
    task is fully completed and verified.  Sub-agents are intentionally
    excluded: they get ONLY their own prompt (subagent.md) with no
    extra context injected.
    """
    p = Path(__file__).parent / "prompts" / "task-completion-rules.md"
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

    Order: [project context files, AGENTS.md first] ->
    task-completion-rules.md -> the actual agent prompt.  The completion
    rules are always the LAST context piece, immediately before the
    agent prompt, so they read as global ground rules rather than part
    of the task instructions.

    ``include_context=False`` drops the context section (rules are still
    included).  This function is NOT used for sub-agents: their system
    prompt is their own prompt file (subagent.md) only, with no context
    files and no task-completion rules (see ``cli.make_session`` and
    ``subagent._subagent_system_prompt``).  ``context_path`` overrides
    the default context directory discovery.
    Returns None if every part is empty/missing.
    """
    parts: list[str] = []
    if include_context:
        # lazy import: harness imports this module at call time
        from .agent_session import find_context_dir

        # the project's AGENTS.md files are just context files that live
        # outside the context directory — same block format, same section
        context_block = load_context_files(
            find_context_dir(project_dir, context_path),
            extra_files=find_agents_md_files(project_dir),
        )
        if context_block:
            parts.append(context_block)
    rules = load_task_completion_rules()
    if rules:
        parts.append(rules)
    if agent_prompt:
        parts.append(agent_prompt)
    return "\n\n".join(parts) if parts else None


def _message_text(msg: object) -> str:
    """Plain text of a Message object or an OpenAI-style dict."""
    text = getattr(msg, "text", None)
    if callable(text):
        result = text()
        if isinstance(result, str):
            return result
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def _message_role(msg: object) -> str | None:
    """Role of a Message object or an OpenAI-style dict."""
    if isinstance(msg, dict):
        return msg.get("role")
    return getattr(msg, "role", None)


def _is_plan_exit_notice(text: str) -> bool:
    """True for the plan-exit approval notice (PLAN_EXIT_APPROVED_MESSAGE
    with the plan file path substituted in)."""
    template = config.PLAN_EXIT_APPROVED_MESSAGE
    if "%s" not in template:
        return text == template
    prefix, suffix = template.split("%s", 1)
    return text.startswith(prefix) and text.endswith(suffix)


def _is_mode_reminder_text(text: str) -> bool:
    """True for harness-injected plan/build mode reminders.

    Content-based: the plan.md / plan-mode.md / build-switch.md prompts
    all start with the <system-reminder> tag, and the plan-exit approval
    notice has a fixed format — the same checks the TUI uses, which also
    cover restored sessions where the ``injected`` flag is lost.
    """
    if text.startswith("<system-reminder>"):
        return True
    return _is_plan_exit_notice(text)


def user_prompt_texts(messages: list) -> list[str]:
    """Every user prompt in *messages*, oldest first.

    Used by the compaction flows to rebuild the conversation after a
    summary: the compacted frame is followed by every prompt, so the
    actual requests survive compaction.

    Accepts Message objects or OpenAI-style dicts.  Excludes:
    - nudges (the completion-supervision message — never user input)
    - previously compacted summary frames: harness artifacts that the
      new summary supersedes (the content check also covers restored
      sessions, where the ``injected`` flag is lost)

    Plan/build-mode reminders are KEPT, but only the most recent batch:
    they carry the current mode context (read-only plan phase, the plan
    file path, "execute the approved plan"), so dropping them at
    compaction would leave the model unsure whether it is planning or
    building — yet replaying every historical reminder would feed it
    stale, possibly contradictory mode instructions after a /plan ->
    /build switch.  Each mode switch injects its reminders as one
    contiguous batch, so the last batch IS the current mode state.
    (``remember_user_text`` excludes reminders for title generation —
    a different purpose, so the filters differ.)
    """
    nudge = config.NUDGE_MESSAGE
    is_reminder: list[bool] = []
    last_reminder = -1
    for msg in messages:
        if _message_role(msg) != "user":
            is_reminder.append(False)
            continue
        text = _message_text(msg)
        rem = bool(text) and _is_mode_reminder_text(text)
        is_reminder.append(rem)
        if rem:
            last_reminder = len(is_reminder) - 1
    if last_reminder >= 0:
        batch_start = last_reminder
        while batch_start > 0 and is_reminder[batch_start - 1]:
            batch_start -= 1
    else:
        batch_start = -1
    prompts: list[str] = []
    for i, msg in enumerate(messages):
        if _message_role(msg) != "user":
            continue
        text = _message_text(msg)
        if not text or text == nudge:
            continue
        if text.startswith(config.COMPACT_HEADER):
            continue
        if is_reminder[i] and not (batch_start <= i <= last_reminder):
            continue
        prompts.append(text)
    return prompts
