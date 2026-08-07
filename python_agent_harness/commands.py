"""Session commands: init, review, summary, custom commands.

Ported from gptel-agent-harness-commands.el.  Each command builds a
fresh session (buffer) with a prompt file as the system prompt and a
kickoff message, then runs the agent loop.
"""

from __future__ import annotations

import re
from pathlib import Path

from .agent import run_agent_loop
from .prompts import read_prompt_file
from .models import Message

PROMPTS_DIR = Path(__file__).parent / "prompts"
COMMANDS_DIR = PROMPTS_DIR / "commands"


def _substitute(text: str, path: str, extra: str | None) -> str:
    text = text.replace("${path}", path)
    text = text.replace("$ARGUMENTS", extra or "")
    return text


def _project_root(cwd: str) -> str:
    """Best-effort project root (git dir or parent with AGENTS.md)."""
    import subprocess

    d = Path(cwd).resolve()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
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
        self, name: str, prompt_file: str, kickoff: str,
        buffer_name: str, status: str, validate_dir: bool = False,
    ) -> None:
        self.name = name
        self.prompt_file = prompt_file
        self.kickoff = kickoff
        self.buffer_name = buffer_name
        self.status = status
        self.validate_dir = validate_dir

    def prepare(
        self,
        project_dir: str | None = None,
        extra: str | None = None,
    ) -> tuple[str, str, str]:
        """Resolve (cwd, system_prompt, kickoff) without creating a session.

        Shared by the CLI (which builds a fresh session) and the TUI
        slash commands (which run inside the current session).
        """
        cwd = project_dir or _project_root(__import__("os").getcwd())
        prompt = _substitute(read_prompt_file(self.prompt_file), cwd, extra)
        kickoff = self.kickoff
        if "${path}" in kickoff:
            kickoff = kickoff.replace("${path}", cwd)
        return cwd, prompt, kickoff

    def run(
        self,
        session_factory,
        project_dir: str | None = None,
        extra: str | None = None,
    ) -> None:
        """Run the command: create a session and start the agent loop."""
        cwd, prompt, kickoff = self.prepare(project_dir, extra)
        session = session_factory(
            project_dir=cwd, system_prompt=prompt, kickoff=kickoff
        )
        # the command prompt is the "actual agent prompt"; the project
        # context and task-completion rules are kept in front of it
        from .prompts import assemble_agent_prompt

        context_path = getattr(session, "_configured_context_path", None)
        system = assemble_agent_prompt(cwd, prompt, context_path=context_path)
        run_agent_loop(
            session,
            messages=[Message(role="user", content=kickoff)],
            top_level=True,
            system=system,
        )


def initialize_command() -> SessionCommand:
    return SessionCommand(
        name="initialize",
        prompt_file="initialize.txt",
        kickoff="Analyze the repository at ${path} and create/update AGENTS.md.\n",
        buffer_name="*gptel-agent-init:*",
        status=" Initializing...",
        validate_dir=True,
    )


def review_command() -> SessionCommand:
    return SessionCommand(
        name="review",
        prompt_file="review.txt",
        kickoff="Review the requested code changes.",
        buffer_name="*gptel-agent-review*",
        status=" Reviewing...",
    )


def custom_name(file: str) -> str:
    base = Path(file).stem.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base)
    return base.strip("-")


def load_custom_commands() -> list[SessionCommand]:
    if not COMMANDS_DIR.is_dir():
        return []
    commands = []
    for f in sorted(COMMANDS_DIR.glob("*.txt")):
        name = custom_name(f.name)
        if not name:
            continue
        commands.append(
            SessionCommand(
                name=name,
                prompt_file=str(f.relative_to(PROMPTS_DIR)),
                kickoff="Proceed with the task described in your instructions.\n",
                buffer_name=f"*gptel-agent-{name}*",
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
