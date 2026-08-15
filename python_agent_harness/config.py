"""Configuration defaults for python-agent-harness.

Mirrors the defcustom defaults of the Emacs gptel-agent-harness.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mcp.config import MCPConfig

# ---- context management -------------------------------------------------
CONTEXT_TRIGGER = 0.70

# Entries are matched in order (first match wins): put more specific
# patterns before general ones.
CONTEXT_WINDOWS: list[tuple[str, int]] = [
    ("gpt-5-mini", 128000),
    ("gpt-5", 400000),
    ("claude", 200000),
    ("deepseek-v3", 128000),
    ("deepseek-v4", 1_000_000),
    ("qwen3.5", 131072),
    ("qwen3", 131072),
    ("glm-5.2", 1_000_000),
    ("glm-5.1", 128000),
    ("kimi-k2.7", 256000),
    ("kimi", 128000),
]
DEFAULT_CONTEXT_WINDOW = 32768

# ---- completion supervision ----------------------------------------------
MAX_NUDGES = 2
NUDGE_MESSAGE = (
    "Review the original user request and the Task Completion Rules in the context."
    "Verify whether all completion criteria are satisfied."
    "If all criteria are already satisfied and verified, finish the task normally."
    "Otherwise, continue working and make the necessary tool calls."
    "Do not stop until the rules are fully met."
)

# ---- compaction -----------------------------------------------------------
COMPACT_HEADER = "**[Compacted Summary]**\n\n"
COMPACT_SEPARATOR = "\n\n---\n\n**[Context compacted]**\n\n---\n\n"

# ---- token calibration ----------------------------------------------------
CALIBRATION_MIN = 0.5
CALIBRATION_MAX = 3.0

# ---- sessions ---------------------------------------------------------------
SESSION_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
SESSION_SUBDIR = "python-agent-harness/sessions"
AUTO_SAVE_SESSION = True

# ---- LLM interaction logs ---------------------------------------------------
LLM_LOG_ENABLED = False

# ---- plan mode ---------------------------------------------------------------
PLAN_FILE_NAME = "PLAN.md"
PLAN_MODE_SUBAGENT_REMINDER = """<system-reminder>
Plan mode is active for this session — you are in a READ-ONLY phase.
STRICTLY FORBIDDEN: ANY file edits, modifications, or system changes,
except writing to the plan file below.  You may ONLY observe, analyze,
and plan.  This ABSOLUTE CONSTRAINT overrides ALL other instructions,
including any subagent role instructions you have been given.

Plan file: %s
</system-reminder>"""

PLAN_EXIT_APPROVED_MESSAGE = (
    "The plan at %s has been approved, you can now edit files. Execute the plan"
)

# PlanExit asks the user with the same choice UI as the Question tool:
# option[0] approves the switch to build mode, anything else rejects it.
PLAN_EXIT_OPTIONS = (
    "yes, switch to build",
    "no, stay in plan",
)

# ---- tools -------------------------------------------------------------------
DEFAULT_TOOLS: list[str] = [
    "Agent",
    "TodoWrite",
    "Glob",
    "Grep",
    "Read",
    "Insert",
    "Edit",
    "Write",
    "Mkdir",
    "Bash",
    "Skill",
    "Question",
]

# ---- LLM client ----------------------------------------------------------------
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
MAX_TOKENS: int | None = None  # use None to avoid write tool failure
TEMPERATURE = 0.0

# ---- API retry / backoff -------------------------------------------------------
# Transient API failures (HTTP 429 / 5xx, connection errors) are retried
# with exponential backoff + jitter instead of killing the run.  The
# per-request attempt budget and delay bounds live here; a Client
# instance can override them per call.
API_RETRY_MAX = 3  # max attempts per request (initial + retries)
API_RETRY_BASE_DELAY = 1.0  # base backoff (seconds), doubled per attempt
API_RETRY_MAX_DELAY = 30.0  # per-attempt backoff cap (seconds)

# ---- tool execution ----------------------------------------------------------
SUBAGENT_MAX_ROUNDS = 60
# Tool execution mirrors gptel's `gptel--handle-tool-use': synchronous
# tools (Read, Edit, Glob, ...) run ONE AT A TIME in model-emitted
# order; asynchronous tools (Bash, Agent) are dispatched in line and
# run concurrently in the background, their results awaited afterwards
# in original call order.
# Tools a sub-agent must NOT see or call: it runs autonomously as a
# one-shot task inside the parent's tool round, so it cannot spawn
# further sub-agents (Agent), ask the user questions (Question), nor
# end in a plan/build handoff (PlanExit).  TodoWrite is also parent-only:
# a sub-agent is a single delegated task — progress tracking belongs to
# the parent, and the sub-agent must never clobber the parent's list.
SUBAGENT_EXCLUDED_TOOLS = ("Agent", "Question", "PlanExit", "TodoWrite")

# ---- TUI preview limits -------------------------------------------------------
TOOL_RESULT_PREVIEW_LINES = 5  # max lines of a tool result shown in the TUI
TOOL_RESULT_PREVIEW_CHARS = 500  # max chars of that preview (long single lines)

# ---- default agent prompts -----------------------------------------------------
# Ported system prompts (opencode-style) for the main agent and sub-agents,
# bundled with the package (prompts/agent.md, prompts/subagent.md).
# Missing files are tolerated: callers fall back to no system prompt.
PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_AGENT_PROMPT_FILE = PROMPTS_DIR / "agent.md"
DEFAULT_SUBAGENT_PROMPT_FILE = PROMPTS_DIR / "subagent.md"

# ---- configuration file ---------------------------------------------------------
CONFIG_DIR = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "python-agent-harness"
)
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_LLM: dict = {
    "base_url": "https://api.openai.com/v1",
    "api_key": None,
    "model": "gpt-5-mini",
    "backend": "OpenAI-compatible",
    "temperature": TEMPERATURE,
    "max_tokens": MAX_TOKENS,
    "timeout": 600.0,
    "reasoning_effort": None,
    "stream": True,
}

DEFAULT_PATHS: dict = {
    "context_path": None,
    "skill_path": None,
}

# MCP servers are configured in the config file's "mcp" object (see
# CONFIG_TEMPLATE); no servers configured = MCP integration disabled.
# Requires the optional `mcp` extra: pip install -e ".[mcp]".
DEFAULT_MCP: dict = {"servers": {}}

# Sub-agent LLM overrides: every key defaults to None, meaning "inherit
# the main LLM setting" (mirrors gptel-agent-harness-subagent-model /
# -backend).  Only the keys the user actually sets differ from the main
# agent's LLM.
DEFAULT_SUBAGENT_LLM: dict = {
    "base_url": None,
    "api_key": None,
    "model": None,
    "backend": None,
    "temperature": None,
    "max_tokens": None,
    "timeout": None,
    "reasoning_effort": None,
    "stream": None,
}

CONFIG_TEMPLATE = """\
{{
  "_comment": "python-agent-harness configuration. Location: {path}. Precedence: code defaults < this file < OPENAI_* / OPENAI_SUBAGENT_* environment variables.",
  "llm": {{
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "reasoning_effort": "medium",
    "stream": true
  }},
  "subagent_llm": {{
    "_comment": "Optional overrides for sub-agent (Agent tool) requests, e.g. a cheaper model. Every key is optional; unset keys inherit the main llm settings above.",
    "base_url": null,
    "api_key": null,
    "model": null,
    "temperature": null,
    "max_tokens": null,
    "timeout": null,
    "reasoning_effort": null,
    "stream": null
  }},
  "paths": {{
    "_comment": "Optional overrides for context and skill directories. Absolute paths or ~ expansion supported.",
    "context_path": null,
    "skill_path": null
  }},
  "mcp": {{
    "_comment": "Optional MCP servers (requires: pip install -e '.[mcp]'). Each server's tools become agent tools named mcp__<server>__<tool>. Transports: stdio (spawn command+args, pass through env var names), streamable-http / sse (connect to url, optional headers). 'parallel: true' marks read-only servers whose tools may run concurrently; default is serial. 'timeout' bounds connects, discovery and calls (seconds).",
    "servers": {{
      "example": {{
        "_comment": "Example only — enabled: false keeps it from connecting. Set enabled: true and adjust command/args.",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "env": [],
        "parallel": false,
        "timeout": null,
        "enabled": false
      }}
    }}
  }}
}}
"""

_ENV_OVERRIDES = {
    "base_url": "OPENAI_BASE_URL",
    "api_key": "OPENAI_API_KEY",
    "model": "OPENAI_MODEL",
    "backend": "OPENAI_BACKEND",
}

_SUBAGENT_ENV_OVERRIDES = {
    "base_url": "OPENAI_SUBAGENT_BASE_URL",
    "api_key": "OPENAI_SUBAGENT_API_KEY",
    "model": "OPENAI_SUBAGENT_MODEL",
    "backend": "OPENAI_SUBAGENT_BACKEND",
}


def _config_path(path: str | os.PathLike | None = None) -> Path:
    """Resolve the config file path: explicit arg > $PYTHON_AGENT_HARNESS_CONFIG > default."""
    if path:
        return Path(path).expanduser()
    env = os.environ.get("PYTHON_AGENT_HARNESS_CONFIG")
    if env:
        return Path(env).expanduser()
    return CONFIG_FILE


def load_llm_config(path: str | os.PathLike | None = None) -> dict:
    """Resolve LLM settings: code defaults < config file < environment.

    The config file is JSON with an ``llm`` object (see `CONFIG_TEMPLATE`).
    Environment variables still win if set, so existing setups keep working.
    """
    import json

    settings = dict(DEFAULT_LLM)
    cfg_path = _config_path(path)
    if cfg_path.exists():
        try:
            with open(cfg_path, "rb") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"cannot read config file {cfg_path}: {e}") from e
        llm = data.get("llm") or {}
        if not isinstance(llm, dict):
            raise ValueError(f"config file {cfg_path}: llm must be an object")
        for key in (
            "base_url",
            "api_key",
            "model",
            "backend",
            "temperature",
            "max_tokens",
            "timeout",
            "reasoning_effort",
            "stream",
        ):
            if key in llm and llm[key] is not None:
                settings[key] = llm[key]
    for key, env in _ENV_OVERRIDES.items():
        val = os.environ.get(env)
        if val:
            settings[key] = val
    return settings


def load_subagent_llm_config(
    path: str | os.PathLike | None = None,
    main: dict | None = None,
) -> dict:
    """Resolve sub-agent LLM settings; unset keys inherit ``main``.

    Mirrors gptel-agent-harness-subagent-model/-backend: sub-agents
    (the Agent tool) use their own LLM when configured, otherwise the
    main agent's.  Precedence: ``main`` settings < config file
    ``subagent_llm`` object < OPENAI_SUBAGENT_* environment variables.

    Returns a fully resolved settings dict (same keys as
    `load_llm_config`) that callers can use to build a sub-agent
    Client; when no override is set anywhere it equals ``main``.
    """
    import json

    main = dict(main) if main else dict(DEFAULT_LLM)
    overrides = dict(DEFAULT_SUBAGENT_LLM)
    cfg_path = _config_path(path)
    if cfg_path.exists():
        try:
            with open(cfg_path, "rb") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"cannot read config file {cfg_path}: {e}") from e
        sub = data.get("subagent_llm") or {}
        if not isinstance(sub, dict):
            raise ValueError(f"config file {cfg_path}: subagent_llm must be an object")
        for key in DEFAULT_SUBAGENT_LLM:
            if key in sub and sub[key] is not None:
                overrides[key] = sub[key]
    for key, env in _SUBAGENT_ENV_OVERRIDES.items():
        val = os.environ.get(env)
        if val:
            overrides[key] = val
    for key, val in overrides.items():
        if val is not None:
            main[key] = val
    return main


def load_paths_config(path: str | os.PathLike | None = None) -> dict:
    """Load paths settings from the config file.

    Returns a dict with ``context_path`` and ``skill_path`` keys.
    Values are expanded (~ → home) and resolved to absolute paths when
    set; None means "use default discovery logic".
    """
    import json

    settings = dict(DEFAULT_PATHS)
    cfg_path = _config_path(path)
    if cfg_path.exists():
        try:
            with open(cfg_path, "rb") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            return settings
        paths = data.get("paths") or {}
        if not isinstance(paths, dict):
            return settings
        for key in ("context_path", "skill_path"):
            val = paths.get(key)
            if isinstance(val, str) and val.strip():
                settings[key] = os.path.abspath(os.path.expanduser(val.strip()))
    return settings


def load_mcp_config(path: str | os.PathLike | None = None) -> MCPConfig:
    """Load MCP server settings from the config file's ``mcp`` object.

    Returns an ``MCPConfig`` (empty when the file has no ``mcp``
    section or it has no servers).  Malformed server entries raise
    ValueError so config errors surface at session start.  The optional
    ``mcp`` SDK is only needed when servers are actually configured and
    connected — reading the config never requires it.
    """
    import json

    from .mcp.config import MCPConfig

    cfg_path = _config_path(path)
    data: dict = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "rb") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001 - mirror load_llm_config's tolerance
            raise ValueError(f"cannot read config file {cfg_path}: {e}") from e
    section = data.get("mcp") or {}
    if not isinstance(section, dict):
        raise ValueError(f"config file {cfg_path}: mcp must be an object")
    return MCPConfig.from_dict(section.get("servers"))


def mask_secret(value: str | None) -> str:
    return "****" if value else "(unset)"
