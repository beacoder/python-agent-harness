"""Session commands: init, review, custom commands (TUI slash commands).

Ported from gptel-agent-harness-commands.el.  Commands run inside the
current TUI session (tui._run_slash_command): the command's prompt
file becomes the run's system prompt, the project context and
task-completion rules stay in front of it, and the kickoff message is
the run's user text.

Tool availability per command:
- init/review: all tools EXCEPT PlanExit (they are one-shot runs that
  must not end in a plan/build handoff)
- custom commands (prompts/commands/*.md): all tools, incl. PlanExit
- compact/summary: no tools at all (direct chat_sync calls, like the
  session-title generation)
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .prompts import read_prompt_file

PROMPTS_DIR = Path(__file__).parent / "prompts"
COMMANDS_DIR = PROMPTS_DIR / "commands"


def _substitute(text: str, path: str, extra: str | None) -> str:
    text = text.replace("${path}", path)
    text = text.replace("$ARGUMENTS", extra or "")
    return text


def _project_root(cwd: str) -> str:
    """Best-effort project root (git dir or parent with AGENTS.md)."""
    d = Path(cwd).resolve()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    for parent in [d, *d.parents]:
        if (parent / "AGENTS.md").exists() or (parent / ".git").exists():
            return str(parent)
    return cwd


class SessionCommand:
    """A configured session command (init/review/custom...)."""

    def __init__(
        self,
        name: str,
        prompt_file: str,
        kickoff: str,
        status: str,
        validate_dir: bool = False,
        allow_planexit: bool = True,
    ) -> None:
        self.name = name
        self.prompt_file = prompt_file
        self.kickoff = kickoff
        self.status = status
        self.validate_dir = validate_dir
        self.allow_planexit = allow_planexit

    def prepare(
        self,
        project_dir: str | None = None,
        extra: str | None = None,
    ) -> tuple[str, str, str]:
        """Resolve (cwd, system_prompt, kickoff) without creating a session.

        Used by the TUI slash commands, which run inside the current
        session.
        """
        cwd = project_dir or _project_root(__import__("os").getcwd())
        prompt = _substitute(read_prompt_file(self.prompt_file), cwd, extra)
        kickoff = self.kickoff
        if "${path}" in kickoff:
            kickoff = kickoff.replace("${path}", cwd)
        return cwd, prompt, kickoff


def initialize_command() -> SessionCommand:
    return SessionCommand(
        name="initialize",
        prompt_file="initialize.md",
        kickoff="Analyze the repository at ${path} and create/update AGENTS.md.\n",
        status=" Initializing...",
        validate_dir=True,
        allow_planexit=False,
    )


def review_command() -> SessionCommand:
    return SessionCommand(
        name="review",
        prompt_file="review.md",
        kickoff="Review the requested code changes.",
        status=" Reviewing...",
        allow_planexit=False,
    )


def hide_planexit(session: Any) -> Callable[[], None] | None:
    """Remove the PlanExit tool from SESSION's registry for a command run.

    Used by init/review (``allow_planexit=False``), which may use every
    tool except PlanExit: the run must not end in a plan/build handoff.
    Custom commands keep PlanExit and skip this.  Returns a callable
    that restores the previous registration state (the tool is
    stateless, so a fresh instance is equivalent), or None when there
    was nothing to hide (PlanExit not registered — e.g. a build-mode
    session, or a session without a registry).  Call the returned
    callable when the run finishes, including on cancellation or error.
    """
    registry = getattr(session, "registry", None)
    if registry is None or registry.get("PlanExit") is None:
        return None
    registry.unregister("PlanExit")

    def restore() -> None:
        from .tools import PlanExit

        registry.register(PlanExit())

    return restore


def custom_name(file: str) -> str:
    base = Path(file).stem.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base)
    return base.strip("-")


def load_custom_commands() -> list[SessionCommand]:
    if not COMMANDS_DIR.is_dir():
        return []
    commands = []
    for f in sorted(COMMANDS_DIR.glob("*.md")):
        name = custom_name(f.name)
        if not name:
            continue
        commands.append(
            SessionCommand(
                name=name,
                prompt_file=str(f.relative_to(PROMPTS_DIR)),
                kickoff="Proceed with the task described in your instructions.\n",
                status=f" Running {name}...",
            )
        )
    return commands


def find_command(name: str) -> SessionCommand | None:
    """Look up a command by name: builtins (init/review) then custom."""
    if name == "init":
        return initialize_command()
    if name == "review":
        return review_command()
    for c in load_custom_commands():
        if c.name == name:
            return c
    return None
