"""Session commands: init, review, custom commands (TUI slash commands).

Ported from gptel-agent-harness-commands.el.  Commands run inside the
current TUI session (tui._run_slash_command): the command's prompt
file becomes the run's system prompt, the project context and
task-completion rules stay in front of it, and the kickoff message is
the run's user text.

Tool availability per command:
- init/review: all tools EXCEPT PlanExit (they are one-shot runs that
  must not end in a plan/build handoff)
- custom commands (prompts/commands/*.md and paths.command_path): all tools, incl. PlanExit
- compact/summary: no tools at all (direct chat_sync calls, like the
  session-title generation)

Precedence: a command file (configured dir first, then bundled) replaces
the builtin slash command of the same name — see ``find_command``.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..prompts import read_prompt_file

PROMPTS_DIR = Path(__file__).parents[1] / "prompts"
COMMANDS_DIR = PROMPTS_DIR / "commands"

# Custom command files take precedence over the builtin slash commands (see
# ``find_command``), so a configured ``review.md`` replaces ``/review``.  The
# one exception is ``/exit``: it is the only way out of an interactive
# session, so a command file may not shadow it.  Such a file is skipped and
# ``/help`` reports it instead of silently dropping it.
UNOVERRIDABLE_COMMAND_NAMES = frozenset({"exit"})


def _prompt_file_ref(f: Path) -> str:
    """Reference for ``read_prompt_file``: relative to the packaged prompts
    dir when the file lives inside it (backward compatible), absolute
    otherwise (a custom command dir configured via ``paths.command_path``)."""
    try:
        return str(f.relative_to(PROMPTS_DIR))
    except ValueError:
        return os.path.abspath(f)


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
        from ..tools import PlanExit

        registry.register(PlanExit())

    return restore


def custom_name(file: str) -> str:
    base = Path(file).stem.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base)
    return base.strip("-")


def _resolve_commands_dir(commands_dir: str | os.PathLike | None) -> Path | None:
    """Absolute path of a configured commands dir, or None when unset."""
    if not commands_dir:
        return None
    return Path(os.path.abspath(os.path.expanduser(commands_dir)))


def _scan_commands_dir(directory: Path) -> list[SessionCommand]:
    """Commands defined by the ``*.md`` files in *directory*.

    Files whose name normalizes to nothing (``---.md``) and files that
    would shadow an unoverridable builtin (see
    ``UNOVERRIDABLE_COMMAND_NAMES``) are skipped, so the advertised
    listing only contains runnable commands.
    """
    if not directory.is_dir():
        return []
    commands = []
    for f in sorted(directory.glob("*.md")):
        name = custom_name(f.name)
        if not name or name in UNOVERRIDABLE_COMMAND_NAMES:
            continue
        commands.append(
            SessionCommand(
                name=name,
                prompt_file=_prompt_file_ref(f),
                kickoff="Proceed with the task described in your instructions.\n",
                status=f" Running {name}...",
            )
        )
    return commands


def unoverridable_command_names(commands_dir: str | os.PathLike | None = None) -> list[str]:
    """Names in *commands_dir* skipped because the builtin cannot be replaced.

    Only ``exit`` is in that set; ``/help`` reports the files so the user
    is not left wondering why their command never ran.
    """
    directory = _resolve_commands_dir(commands_dir)
    if directory is None or not directory.is_dir():
        return []
    return sorted(
        name
        for name in (custom_name(f.name) for f in sorted(directory.glob("*.md")))
        if name in UNOVERRIDABLE_COMMAND_NAMES
    )


def load_custom_commands(
    commands_dir: str | os.PathLike | None = None,
) -> list[SessionCommand]:
    """Load custom commands from the bundled and the configured directory.

    The bundled ``prompts/commands/`` commands are always loaded.  When
    *commands_dir* is set (``paths.command_path`` in the config file),
    its files are layered on top: a configured command overrides a
    bundled one of the same name, and the remaining bundled commands stay
    available.  A missing configured directory contributes nothing.
    """
    merged = {c.name: c for c in _scan_commands_dir(COMMANDS_DIR)}
    configured = _resolve_commands_dir(commands_dir)
    if configured is not None:
        merged.update({c.name: c for c in _scan_commands_dir(configured)})
    return [merged[name] for name in sorted(merged)]


def find_command(name: str, commands_dir: str | os.PathLike | None = None) -> SessionCommand | None:
    """Look up a command by name.

    Command files win over the builtin prompt commands, so a configured
    ``review.md`` replaces ``/review`` and an ``init.md`` replaces
    ``/init``.  Among command files the configured directory wins over
    the bundled one (see ``load_custom_commands``); the remaining bundled
    ``initialize.md``/``review.md`` prompts back the builtins when no
    command file claims the name.
    """
    for c in load_custom_commands(commands_dir):
        if c.name == name:
            return c
    if name == "init":
        return initialize_command()
    if name == "review":
        return review_command()
    return None
