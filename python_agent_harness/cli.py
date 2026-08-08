"""CLI entry points: interactive TUI session and configuration.

Commands:
  run [project]            interactive TUI agent session (default)
  config [--init]          show effective LLM config / write a template file

Custom commands (prompts/commands/*.txt) — like init, review,
sessions, restore and summary/explain — are TUI slash commands only;
they are NOT registered as CLI subcommands.

Configuration (LLM etc.) is read from a JSON file, by default
~/.config/python-agent-harness/config.json; see `config --init`.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import config
from .client import Client
from .agent_session import AgentSession
from .tools import default_registry


def make_session(
    project_dir: str,
    config_path: str | None = None,
    model: str | None = None,
    stream: bool | None = None,
) -> AgentSession:
    """Create an AgentSession from config file + env (no env required).

    The system prompt defaults to the ported main-agent prompt
    (config.DEFAULT_AGENT_PROMPT_FILE); the sub-agent prompt always
    defaults to config.DEFAULT_SUBAGENT_PROMPT_FILE.  Either default
    falls back to no system prompt if its file is unavailable.
    """
    settings = config.load_llm_config(config_path)
    paths = config.load_paths_config(config_path)
    model = model or settings["model"]
    client = Client(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        model=model,
        timeout=settings["timeout"],
    )
    from .prompts import assemble_agent_prompt, load_agent_prompt
    from .agent_session import find_skill_dir

    abs_project = os.path.abspath(project_dir)
    skill_dir = find_skill_dir(abs_project, paths.get("skill_path"))
    system_prompt = assemble_agent_prompt(
        abs_project,
        load_agent_prompt(config.DEFAULT_AGENT_PROMPT_FILE, skill_dir=skill_dir),
        context_path=paths.get("context_path"),
    )
    # sub-agents get ONLY their own system prompt — no parent project
    # context and no task-completion rules injected
    subagent_system_prompt = load_agent_prompt(
        config.DEFAULT_SUBAGENT_PROMPT_FILE, skill_dir=skill_dir
    )
    return AgentSession(
        project_dir=abs_project,
        client=client,
        model=model,
        backend=settings["backend"],
        system_prompt=system_prompt,
        subagent_system_prompt=subagent_system_prompt,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        reasoning_effort=settings["reasoning_effort"],
        stream=settings["stream"] if stream is None else stream,
        registry=default_registry(),
        context_path=paths.get("context_path"),
        skill_path=paths.get("skill_path"),
    )


def cmd_run(args: argparse.Namespace) -> int:
    from .tui import Tui

    project_dir = args.project or os.getcwd()
    session = make_session(
        project_dir, config_path=args.config,
        stream=False if getattr(args, "no_stream", False) else None,
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
    paths = config.load_paths_config(args.path)
    print(f"config file: {path}")
    if not path.exists():
        print(f"(file does not exist yet — run `python-agent-harness config --init` to create it)")
    for key in ("base_url", "model", "backend"):
        print(f"{key}: {settings[key]}")
    print(f"api_key: {config.mask_secret(settings['api_key'])}")
    print(f"temperature: {settings['temperature']}")
    print(f"max_tokens: {settings['max_tokens']}")
    print(f"reasoning_effort: {settings['reasoning_effort']}")
    print(f"stream: {settings['stream']}")
    print(f"timeout: {settings['timeout']}")
    print(f"context_path: {paths['context_path'] or '(auto-discover)'}")
    print(f"skill_path: {paths['skill_path'] or '(auto-discover)'}")
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
    p_run.add_argument(
        "--no-stream", action="store_true",
        help="disable streaming (one-shot responses; overrides config file)",
    )
    p_run.add_argument("project", nargs="?", help="project directory (default: cwd)")

    p_config = sub.add_parser(
        "config", help="show effective LLM config or write a template file"
    )
    p_config.add_argument("--init", action="store_true", help="write a config template")
    p_config.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_config.add_argument("--path", metavar="PATH", help="config file path")
    p_config.set_defaults(func=cmd_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "run"):
        return cmd_run(args)
    if args.command == "config":
        return cmd_config(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
