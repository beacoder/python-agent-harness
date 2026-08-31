"""CLI entry points: interactive TUI session and configuration.

Commands:
  run [project]            interactive TUI agent session (default)
  config [--init]          show effective LLM config / write a template file

Custom commands (prompts/commands/*.md) — like init, review,
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
from .session import Session
from .tools import default_registry


def make_session(
    project_dir: str,
    config_path: str | None = None,
    model: str | None = None,
    stream: bool | None = None,
) -> Session:
    """Create a Session from config file + env (no env required).

    The system prompt defaults to the ported main-agent prompt
    (config.DEFAULT_AGENT_PROMPT_FILE); the sub-agent prompt always
    defaults to config.DEFAULT_SUBAGENT_PROMPT_FILE.  Either default
    falls back to no system prompt if its file is unavailable.
    """
    settings = config.load_llm_config(config_path)
    paths = config.load_paths_config(config_path)
    mcp_config = config.load_mcp_config(config_path)
    model = model or settings["model"]
    # resolve sub-agent overrides against the EFFECTIVE main settings
    # (so a CLI/caller model override is inherited too when the
    # subagent_llm model is unset)
    settings["model"] = model
    subagent_settings = config.load_subagent_llm_config(config_path, main=settings)
    client = Client(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        model=model,
        timeout=settings["timeout"],
        config_path=config_path,
    )
    # A separate client for sub-agent requests only when a different
    # LLM is configured (mirrors gptel-agent-harness-subagent-model);
    # otherwise the sub-agent shares the main client.
    subagent_client = None
    if any(
        subagent_settings[k] != settings[k] for k in ("base_url", "api_key", "model", "timeout")
    ):
        subagent_client = Client(
            base_url=subagent_settings["base_url"],
            api_key=subagent_settings["api_key"],
            model=subagent_settings["model"],
            timeout=subagent_settings["timeout"],
            config_path=config_path,
        )
        # keep every request of this session (main + sub-agents) in the
        # same LLM log file — the TUI advertises the main client's log
        # path, and a separate sub-agent log would fragment debugging
        subagent_client.log_path = client.log_path
    from .prompts import assemble_agent_prompt, load_agent_prompt
    from .session import find_skill_dir

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
    # Resolve the effective stream once: the CLI --no-stream flag wins
    # over the config file for the whole session, sub-agents included
    # (same precedence as the main agent's stream).
    effective_stream = settings["stream"] if stream is None else stream
    model_profiles = config.load_models_config(config_path)
    # Base settings for /model switching: the main llm settings as
    # resolved at session start (incl. the CLI --model/--no-stream
    # overrides above), so a profile's unset keys inherit these
    # instead of values drifted by earlier switches.
    llm_settings = dict(settings)
    llm_settings["stream"] = effective_stream
    return Session(
        project_dir=abs_project,
        client=client,
        model=model,
        system_prompt=system_prompt,
        subagent_system_prompt=subagent_system_prompt,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        reasoning_effort=settings["reasoning_effort"],
        stream=effective_stream,
        subagent_client=subagent_client,
        subagent_temperature=subagent_settings["temperature"],
        subagent_max_tokens=subagent_settings["max_tokens"],
        subagent_reasoning_effort=subagent_settings["reasoning_effort"],
        subagent_stream=(effective_stream if stream is not None else subagent_settings["stream"]),
        registry=default_registry(),
        context_path=paths.get("context_path"),
        skill_path=paths.get("skill_path"),
        mcp=mcp_config,
        model_profiles=model_profiles,
        llm_settings=llm_settings,
        config_path=config_path,
    )


def make_session_with_mcp(
    project_dir: str,
    config_path: str | None = None,
    model: str | None = None,
    stream: bool | None = None,
) -> Session:
    """Create a Session and connect its configured MCP servers.

    Wraps ``make_session``: the session's MCP servers are connected and
    their tools registered before the session is returned (discovery
    happens once, at session start).  Per-server failures are printed
    to stderr and never prevent the session from running — the agent
    keeps working with the built-in tools.
    """
    session = make_session(project_dir, config_path=config_path, model=model, stream=stream)
    failures = session.connect_mcp()
    for server, err in failures:
        print(f"python-agent-harness: [{server}] {err}", file=sys.stderr)
    return session


def cmd_run(args: argparse.Namespace) -> int:
    from .tui import Tui

    project_dir = getattr(args, "project", None) or os.getcwd()
    session = make_session_with_mcp(
        project_dir,
        config_path=args.config,
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
    subagent_settings = config.load_subagent_llm_config(args.path, main=settings)
    paths = config.load_paths_config(args.path)
    mcp_config = config.load_mcp_config(args.path)
    print(f"config file: {path}")
    if not path.exists():
        print("(file does not exist yet — run `python-agent-harness config --init` to create it)")
    for key in ("base_url", "model"):
        print(f"{key}: {settings[key]}")
    print(f"api_key: {config.mask_secret(settings['api_key'])}")
    print(f"temperature: {settings['temperature']}")
    print(f"max_tokens: {settings['max_tokens']}")
    print(f"reasoning_effort: {settings['reasoning_effort']}")
    print(f"stream: {settings['stream']}")
    print(f"timeout: {settings['timeout']}")
    print(f"context_path: {paths['context_path'] or '(default: <project>/contexts)'}")
    print(f"skill_path: {paths['skill_path'] or '(default: <project>/skills)'}")
    if mcp_config.servers:
        for name, server in mcp_config.servers.items():
            status = "enabled" if server.enabled else "disabled"
            target = (
                f"command={server.command} {' '.join(server.args)}"
                if server.transport == "stdio"
                else f"url={server.url}"
            )
            print(
                f"mcp server {name}: {status}, transport={server.transport}, "
                f"{target}, parallel={server.parallel}"
            )
    else:
        print('mcp: (none configured — add an "mcp" section to the config file)')
    print(
        "subagent_llm: (inherits main)"
        if subagent_settings == settings
        else f"subagent_llm: model={subagent_settings['model']} "
        f"base_url={subagent_settings['base_url']} "
        f"api_key={config.mask_secret(subagent_settings['api_key'])} "
        f"temperature={subagent_settings['temperature']} "
        f"max_tokens={subagent_settings['max_tokens']} "
        f"reasoning_effort={subagent_settings['reasoning_effort']} "
        f"stream={subagent_settings['stream']} timeout={subagent_settings['timeout']}"
    )
    # Show model profiles for /model command
    model_profiles = config.load_models_config(args.path)
    if model_profiles:
        print("models:")
        for name, profile in sorted(model_profiles.items()):
            model_name = profile.get("model", "(inherited)")
            base_url = profile.get("base_url", "(inherited)")
            cw = profile.get("context_window")
            cw_str = f", context_window={cw}" if cw is not None else ""
            print(f"  {name}: model={model_name}, base_url={base_url}{cw_str}")
    else:
        print("models: (none configured — add a 'models' section to use /model)")
    return 0


def _add_config_arg(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS if suppress else None,
        help="path to config.json (default: ~/.config/python-agent-harness/config.json)",
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
        "--no-stream",
        action="store_true",
        help="disable streaming (one-shot responses; overrides config file)",
    )
    p_run.add_argument("project", nargs="?", help="project directory (default: cwd)")

    p_config = sub.add_parser("config", help="show effective LLM config or write a template file")
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
