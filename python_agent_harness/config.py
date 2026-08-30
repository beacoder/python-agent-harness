"""Configuration defaults for python-agent-harness.

Mirrors the defcustom defaults of the Emacs gptel-agent-harness.
"""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

from .mcp.config import MCPConfig

# ---- context management -------------------------------------------------
CONTEXT_TRIGGER = 0.70

# Entries are matched in order (first match wins): put more specific
# patterns before general ones. Supports wildcards (*).  Trailing `*`
# on explicit entries preserves prefix matching for suffixed model
# names (e.g. "deepseek-v4-flash" -> deepseek-v4*), matching the
# legacy substring behavior.
CONTEXT_WINDOWS: list[tuple[str, int]] = [
    ("gpt-5-mini*", 128_000),
    ("gpt-5*", 400_000),
    ("gpt-oss-120b*", 128_000),
    ("claude*", 200_000),
    ("deepseek-v3*", 128_000),
    ("deepseek-v4*", 1_000_000),
    ("qwen3.5*", 131_072),
    ("qwen3.6*", 262_144),
    ("qwen3.8*", 262_144),
    ("qwen3*", 131_072),
    ("glm-5.2*", 1_000_000),
    ("glm-5.1*", 128_000),
    ("kimi-k2.7*", 256_000),
    ("kimi*", 128_000),
    # Wildcard fallbacks for unknown models
    ("gpt-*", 128_000),
    ("claude-*", 200_000),
    ("deepseek-*", 128_000),
    ("qwen-*", 128_000),
    ("glm-*", 128_000),
    ("kimi-*", 128_000),
]
DEFAULT_CONTEXT_WINDOW = 128_000

# ---- completion supervision ----------------------------------------------
MAX_NUDGES = 2
NUDGE_MESSAGE = (
    "Review the original user request and the Task Completion Rules in the context. "
    "Verify whether all completion criteria are satisfied. "
    "If all criteria are already satisfied and verified, finish the task normally. "
    "Otherwise, continue working and make the necessary tool calls. "
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

# ---- tool output limits --------------------------------------------------------
# Shared cap for tool results.  Bash truncates output to a head+tail at
# this size; Read/Glob/Grep spill results larger than this to a temp
# file (the model then Reads the file for the full content).  One
# definition, used by all tools, so the limits never drift apart.
MAX_OUTPUT_CHARS = 20_000

# ---- Bash tool timeout --------------------------------------------------------
# Silence-based timeout: a command that produces no output for this
# long (seconds) is killed (SIGTERM, then SIGKILL after 2s) and
# reported as timed out.  Builds that keep printing are never affected
# — only genuinely stuck commands surface.  None disables the check.
BASH_TIMEOUT_SILENCE: float | None = 120.0
# Optional absolute wall-clock cap (seconds): no command may run longer
# than this regardless of output.  None disables the cap (default).
BASH_TIMEOUT_MAX: float | None = None

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

# Sub-agent LLM overrides: every key defaults to None, meaning "inherit
# the main LLM setting" (mirrors gptel-agent-harness-subagent-model /
# -backend).  Only the keys the user actually sets differ from the main
# agent's LLM.  ``profile`` references a named profile from the
# ``models`` section: its settings are applied on top of any explicit
# subagent_llm keys (profile wins), and unset keys still inherit the
# main LLM settings.
DEFAULT_SUBAGENT_LLM: dict = {
    "profile": None,
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
  "models": {{
    "_comment": "Named LLM profiles for /model switching. Each entry is a full set of LLM settings (base_url, api_key, model, etc.). Use /model in the TUI to switch at runtime.",
    "deepseek": {{
      "base_url": "https://api.deepseek.com/v1",
      "model": "deepseek-chat"
    }},
    "openai": {{
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-5-mini"
    }}
  }},
  "context_windows": {{
    "_comment": "Optional per-model context-window overrides (tokens). Keys are model names or fnmatch patterns (e.g. deepseek-v4* = 1000000); matched in file order, first match wins. Overrides the built-in CONTEXT_WINDOWS table in config.py. Remove this section to use the built-in table.",
    "deepseek-v4*": 1000000
  }},
  "subagent_llm": {{
    "_comment": "Optional overrides for sub-agent (Agent tool) requests, e.g. a cheaper model. Every key is optional; unset keys inherit the main llm settings above. Set 'profile' to a name from the 'models' section to reuse a model profile (profile settings win over explicit keys below).",
    "profile": null,
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


def _read_config(path: str | os.PathLike | None = None) -> dict:
    """Read and parse the config file; ``{}`` when it does not exist.

    Raises ValueError on unreadable/invalid JSON so config errors
    surface at session start.  Callers that tolerate a broken file
    (e.g. `load_paths_config`) catch it and fall back to defaults.
    """
    cfg_path = _config_path(path)
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "rb") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"cannot read config file {cfg_path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"config file {cfg_path}: top level must be an object")
    return data


def load_llm_config(path: str | os.PathLike | None = None) -> dict:
    """Resolve LLM settings: code defaults < config file < environment.

    The config file is JSON with an ``llm`` object (see `CONFIG_TEMPLATE`).
    Environment variables still win if set, so existing setups keep working.
    """
    settings = dict(DEFAULT_LLM)
    data = _read_config(path)
    llm = data.get("llm") or {}
    if not isinstance(llm, dict):
        raise ValueError(f"config file {_config_path(path)}: llm must be an object")
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
    ``subagent_llm`` object < referenced ``models`` profile (when
    ``subagent_llm.profile`` is set) < OPENAI_SUBAGENT_* environment
    variables.  A referenced profile's keys win over explicit
    ``subagent_llm`` keys; keys the profile leaves unset still inherit
    the main settings.

    Returns a fully resolved settings dict (same keys as
    `load_llm_config`) that callers can use to build a sub-agent
    Client; when no override is set anywhere it equals ``main``.
    """
    main = dict(main) if main else dict(DEFAULT_LLM)
    overrides = dict(DEFAULT_SUBAGENT_LLM)
    data = _read_config(path)
    sub = data.get("subagent_llm") or {}
    if not isinstance(sub, dict):
        raise ValueError(f"config file {_config_path(path)}: subagent_llm must be an object")
    for key in DEFAULT_SUBAGENT_LLM:
        if key in sub and sub[key] is not None:
            overrides[key] = sub[key]
    profile_name = overrides.get("profile")
    if profile_name:
        models = data.get("models") or {}
        if not isinstance(models, dict):
            raise ValueError(f"config file {_config_path(path)}: models must be an object")
        profile = models.get(profile_name)
        if not isinstance(profile, dict):
            raise ValueError(
                f"config file {_config_path(path)}: subagent_llm.profile references "
                f"unknown models profile {profile_name!r}"
            )
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
            if key in profile and profile[key] is not None:
                overrides[key] = profile[key]
    for key, env in _SUBAGENT_ENV_OVERRIDES.items():
        val = os.environ.get(env)
        if val:
            overrides[key] = val
    overrides.pop("profile", None)
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
    settings = dict(DEFAULT_PATHS)
    try:
        data = _read_config(path)
    except ValueError:
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
    data = _read_config(path)
    section = data.get("mcp") or {}
    if not isinstance(section, dict):
        raise ValueError(f"config file {_config_path(path)}: mcp must be an object")
    return MCPConfig.from_dict(section.get("servers"))


def load_models_config(path: str | os.PathLike | None = None) -> dict[str, dict]:
    """Load named LLM profiles from the config file's ``models`` object.

    Returns a dict mapping profile names to their LLM settings dicts.
    Each profile is a partial set of DEFAULT_LLM keys (base_url, api_key,
    model, etc.); unset keys inherit the main ``llm`` settings when the
    profile is applied.  An empty dict when the file has no ``models``
    section or it is empty.
    """
    data = _read_config(path)
    section = data.get("models") or {}
    if not isinstance(section, dict):
        raise ValueError(f"config file {_config_path(path)}: models must be an object")
    profiles: dict[str, dict] = {}
    for name, val in section.items():
        if name.startswith("_"):
            continue
        if not isinstance(val, dict):
            raise ValueError(f"config file {_config_path(path)}: models.{name} must be an object")
        profiles[name] = val
    return profiles


def mask_secret(value: str | None) -> str:
    return "****" if value else "(unset)"


def _match_context_window(model: str) -> int | None:
    """Match model name against CONTEXT_WINDOWS patterns (supports wildcards).

    Args:
        model: The model ID to match (e.g., "gpt-5-mini", "claude-3-opus")

    Returns:
        The context window size if matched, None otherwise
    """
    lowered = model.lower()
    for pattern, size in CONTEXT_WINDOWS:
        if fnmatch.fnmatch(lowered, pattern.lower()):
            return size
    return None


def load_context_windows_config(
    path: str | os.PathLike | None = None,
) -> list[tuple[str, int]]:
    """Load per-model context-window overrides from the config file.

    Reads the ``context_windows`` object: a mapping of model names or
    fnmatch patterns (matched in file order, first match wins) to token
    counts.  Keys starting with ``_`` are comments and skipped.  A
    missing file, missing section, or unreadable JSON yields ``[]``
    (callers fall back to the built-in table); a malformed section or
    non-integer size raises ValueError so config errors surface.
    """
    try:
        data = _read_config(path)
    except ValueError:
        return []
    section = data.get("context_windows") or {}
    if not isinstance(section, dict):
        raise ValueError(f"config file {_config_path(path)}: context_windows must be an object")
    entries: list[tuple[str, int]] = []
    for pattern, size in section.items():
        if pattern.startswith("_"):
            continue
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError(
                f"config file {_config_path(path)}: context_windows.{pattern} must be an integer"
            )
        entries.append((pattern, size))
    return entries


def get_context_window_for_model(
    model: str,
    config_path: str | os.PathLike | None = None,
) -> int:
    """Get the context window for MODEL: config-file overrides, then
    the built-in table, then the default.

    The config file's ``context_windows`` object (user overrides) is
    consulted first (fnmatch over its patterns, first match wins,
    case-insensitive); then the CONTEXT_WINDOWS table in config.py;
    then DEFAULT_CONTEXT_WINDOW.

    Args:
        model: The model ID to look up (e.g. "deepseek-v4-flash")
        config_path: Optional path to the config file; defaults to the
            standard config location (config.json).

    Returns:
        The context window size as an integer.
    """
    lowered = model.lower()
    for pattern, size in load_context_windows_config(config_path):
        if fnmatch.fnmatch(lowered, pattern.lower()):
            return size
    matched = _match_context_window(model)
    if matched is not None:
        return matched
    return DEFAULT_CONTEXT_WINDOW
