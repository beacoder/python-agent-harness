"""CLI entry points: interactive TUI session and command-line commands.

Commands:
  run [project]            interactive TUI agent session (default)
  config [--init]          show effective LLM config / write a template file
  init [project]           create/update AGENTS.md
  review [args]            review uncommitted changes / commit / branch / PR
  explain <target>         explain code (from prompts/commands/explain.txt)
  <custom>                 any prompt file in prompts/commands/
  summary                  summarize the current (saved) session
  sessions                 list saved sessions
  restore <file>           restore a saved session
  restore-latest           restore the newest session

Configuration (LLM etc.) is read from a TOML file, by default
~/.config/python-agent-harness/config.toml; see `config --init`.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import config
from .client import Client
from .commands import (
    SessionCommand, initialize_command, load_custom_commands, review_command,
)
from .harness import AgentSession
from .models import Message
from .session import SessionStore
from .tools import default_registry


def make_session(
    project_dir: str,
    system_prompt: str | None = None,
    config_path: str | None = None,
    model: str | None = None,
) -> AgentSession:
    """Create an AgentSession from config file + env (no env required)."""
    settings = config.load_llm_config(config_path)
    model = model or settings["model"]
    client = Client(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        model=model,
        timeout=settings["timeout"],
    )
    return AgentSession(
        project_dir=os.path.abspath(project_dir),
        client=client,
        model=model,
        backend=settings["backend"],
        system_prompt=system_prompt,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        reasoning_effort=settings["reasoning_effort"],
        registry=default_registry(),
    )


def cmd_run(args: argparse.Namespace) -> int:
    from .tui import Tui

    project_dir = args.project or os.getcwd()
    session = make_session(
        project_dir, system_prompt=args.system, config_path=args.config
    )
    Tui(session).run()
    session.close()
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    path = config._config_path(args.path)
    if args.init:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.force:
            print(f"config already exists: {path} (use --force to overwrite)")
            return 1
        template = config.CONFIG_TEMPLATE.format(path=path)
        path.write_text(template, encoding="utf-8")
        print(f"wrote config template: {path}")
        return 0
    settings = config.load_llm_config(args.path)
    print(f"config file: {path}")
    if not path.exists():
        print(f"(file does not exist yet — run `python-agent-harness config --init` to create it)")
    for key in ("base_url", "model", "backend"):
        print(f"{key}: {settings[key]}")
    print(f"api_key: {config.mask_secret(settings['api_key'])}")
    print(f"temperature: {settings['temperature']}")
    print(f"max_tokens: {settings['max_tokens']}")
    print(f"reasoning_effort: {settings['reasoning_effort']}")
    print(f"timeout: {settings['timeout']}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    _run_command(initialize_command(), args.project, args.extra, args.config)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    _run_command(review_command(), args.project, args.arguments, args.config)
    return 0


def cmd_custom(args: argparse.Namespace) -> int:
    cmd = _find_custom(args.command_name)
    if cmd is None:
        print(f"unknown custom command: {args.command_name}", file=sys.stderr)
        return 1
    _run_command(cmd, args.project, args.extra, args.config)
    return 0


def _find_custom(name: str) -> SessionCommand | None:
    for c in load_custom_commands():
        if c.name == name:
            return c
    return None


def _run_command(
    cmd: SessionCommand,
    project: str | None,
    extra: str | None,
    config_path: str | None = None,
) -> None:
    project_dir = project or os.getcwd()
    session = make_session(project_dir, config_path=config_path)
    cmd.run(lambda **kw: _adopt(session, kw), project_dir=project_dir, extra=extra)


def _adopt(session: AgentSession, kw: dict) -> AgentSession:
    # SessionCommand.run builds its own session kwargs; reuse ours.
    if kw.get("system_prompt") is not None:
        session.system_prompt = kw["system_prompt"]
    return session


def cmd_sessions(args: argparse.Namespace) -> int:
    files = SessionStore.list_sessions()
    if not files:
        print("no saved sessions")
        return 0
    for f in files:
        meta = SessionStore.parse_metadata(open(f, encoding="utf-8").read())
        print(
            f"{os.path.basename(f):60s} "
            f"model={meta.get('gptel-model', '?'):20s} "
            f"project={meta.get('python-agent-harness--project-dir', '?')}"
        )
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    from .session import SessionStore as Store
    from .session import title_from_filename

    path = args.file
    if not path and args.latest:
        path = Store.latest_session()
    if not path:
        print("no session file given", file=sys.stderr)
        return 1
    text = open(path, encoding="utf-8").read()
    meta = Store.parse_metadata(text)
    body = Store.strip_metadata(text)
    project = meta.get("python-agent-harness--project-dir") or os.getcwd()
    model = meta.get("gptel-model") or config.DEFAULT_MODEL
    session = make_session(
        project,
        system_prompt=meta.get("gptel-system-prompt"),
        config_path=args.config,
        model=model,
    )
    session.store.file_path = path
    title = title_from_filename(path)
    session.store.title = title
    print(f"restored: {path} (project={project}, model={model})")
    print("conversation preview:")
    print("\n".join(body.splitlines()[:20]))
    session.close()
    return 0


def _add_config_arg(
    parser: argparse.ArgumentParser, suppress: bool = False
) -> None:
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS if suppress else None,
        help="path to config.toml (default: ~/.config/python-agent-harness/config.toml)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python-agent-harness",
        description="Python agent execution harness (gptel-agent-harness port)",
    )
    _add_config_arg(parser)
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="interactive TUI agent session")
    _add_config_arg(p_run, suppress=True)
    p_run.add_argument("project", nargs="?", help="project directory (default: cwd)")
    p_run.add_argument("--system", help="system prompt file or text")

    p_config = sub.add_parser(
        "config", help="show effective LLM config or write a template file"
    )
    p_config.add_argument("--init", action="store_true", help="write a config template")
    p_config.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_config.add_argument("--path", metavar="PATH", help="config file path")
    p_config.set_defaults(func=cmd_config)

    p_init = sub.add_parser("init", help="create/update AGENTS.md")
    _add_config_arg(p_init, suppress=True)
    p_init.add_argument("project", nargs="?", help="project directory")
    p_init.add_argument("--extra", help="extra instructions")

    p_review = sub.add_parser("review", help="review code changes")
    _add_config_arg(p_review, suppress=True)
    p_review.add_argument("project", nargs="?")
    p_review.add_argument("arguments", nargs="?", help="commit/branch/PR, or empty")

    for cmd in load_custom_commands():
        p = sub.add_parser(cmd.name, help=f"run custom command {cmd.name}")
        _add_config_arg(p, suppress=True)
        p.add_argument("project", nargs="?")
        p.add_argument("extra", nargs="?", help="arguments for the command")
        p.set_defaults(func=cmd_custom, command_name=cmd.name)

    p_sessions = sub.add_parser("sessions", help="list saved sessions")
    p_sessions.set_defaults(func=cmd_sessions)

    p_restore = sub.add_parser("restore", help="restore a saved session")
    _add_config_arg(p_restore, suppress=True)
    p_restore.add_argument("file", nargs="?", help="session file path")
    p_restore.add_argument("--latest", action="store_true", help="restore newest session")
    p_restore.set_defaults(func=cmd_restore)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "run"):
        return cmd_run(args)
    if args.command == "config":
        return cmd_config(args)
    if args.command == "init":
        return cmd_init(args)
    if args.command == "review":
        return cmd_review(args)
    if hasattr(args, "func"):
        return args.func(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
