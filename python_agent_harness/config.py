"""Configuration defaults for python-agent-harness.

Mirrors the defcustom defaults of the Emacs gptel-agent-harness.
"""

from __future__ import annotations

import os
from pathlib import Path

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
    "Review the original user request and the Task Completion Rules "
    "in the context. Verify whether all completion criteria are satisfied. "
    "If not, continue by making tool calls. Do not stop until the rules are fully met."
)

# ---- compaction -----------------------------------------------------------
COMPACT_HEADER = "**[Compacted Summary]**\n\n"
COMPACT_SEPARATOR = "\n\n---\n\n**[Context compacted]**\n\n---\n\n"

# ---- token calibration ----------------------------------------------------
CALIBRATION_MIN = 0.5
CALIBRATION_MAX = 3.0

# ---- safety ----------------------------------------------------------------
FORBIDDEN_PATHS: list[str] = [r"^\s*/mnt/"]
BASH_TIMEOUT = 300
BASH_APPROVAL = "confirm"  # "nil" | "confirm" | "block"

DANGEROUS_PATTERNS: list[str] = [
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+|[^\s]*\s+-[a-zA-Z]*r\b)",
    r"\bgit\s+push\s+--force",
    r"\bgit\s+reset\s+--hard",
    r"\bchmod\s+-R\s+[0-7][0-7][0-7]",
    r"\bchown\s+-R\b",
    r"\bsu\s+-",
    r"\btar\s+--remove-files",
]

DESTRUCTIVE_PATTERNS: list[str] = [
    r"\bkillall\b",
    r"\bpkill\b",
    r"\bsudo\b",
]

CATASTROPHIC_PATTERNS: list[str] = [
    r"\brm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{",
    r"\bshutdown\b",
    r"\breboot\b",
    r">\s*/dev/sd[a-z]",
    r"\bmv\s+/\s",
    r"\brmdir\s+/",
]

PLAN_READONLY_COMMANDS: set[str] = {
    "ls", "cat", "head", "tail", "less", "more", "pwd", "echo", "printf",
    "find", "grep", "rg", "git", "git-status", "git-diff", "git-log",
    "git-show", "git-branch", "git-blame", "git-ls-files", "wc", "sort",
    "uniq", "cut", "tr", "sed", "awk", "file", "stat", "du", "df", "env",
    "which", "whereis", "type", "python", "python3", "node", "jq", "yq",
    "date", "cal", "tree", "basename", "dirname", "readlink", "realpath",
    "xargs", "test", "[", "true", "false", "git-grep", "git-log",
}

GIT_MUTATING_SUBCOMMANDS: set[str] = {
    "add", "rm", "mv", "commit", "rebase", "reset", "checkout", "switch",
    "restore", "merge", "cherry-pick", "revert", "push", "pull", "fetch",
    "clone", "init", "clean", "stash", "tag", "branch", "apply", "am",
    "format-patch", "archive", "bundle", "worktree", "maintenance",
    "update-index", "update-ref", "symbolic-ref", "config",
}

BASH_ARG_DENYLIST: list[tuple[str, set[str]]] = [
    ("find", {"-delete", "-exec", "-execdir", "-ok", "-okdir"}),
    ("sort", {"-o", "--output"}),
    ("yq", {"-i", "--inplace"}),
    ("jq", {"-i", "--in-place"}),
]

# ---- sessions ---------------------------------------------------------------
SESSION_DIR = Path(
    os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
)
SESSION_SUBDIR = "python-agent-harness/sessions"
AUTO_SAVE_SESSION = True

# ---- LLM interaction logs ---------------------------------------------------
LLM_LOG_ENABLED = True

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
    "Agent", "TodoWrite", "Glob", "Grep", "Read", "Insert", "Edit",
    "Write", "Mkdir", "Bash", "Skill", "Question",
]

# ---- LLM client ----------------------------------------------------------------
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
MAX_TOKENS = 8192
TEMPERATURE = 0.0

# ---- API retry / backoff -------------------------------------------------------
# Transient API failures (HTTP 429 / 5xx, connection errors) are retried
# with exponential backoff + jitter instead of killing the run.  The
# per-request attempt budget and delay bounds live here; a Client
# instance can override them per call.
API_RETRY_MAX = 3            # max attempts per request (initial + retries)
API_RETRY_BASE_DELAY = 1.0   # base backoff (seconds), doubled per attempt
API_RETRY_MAX_DELAY = 30.0   # per-attempt backoff cap (seconds)

# ---- tool execution ----------------------------------------------------------
SUBAGENT_MAX_ROUNDS = 60
# Max tool calls that may run CONCURRENTLY in one tool round (all tools
# issued together in a round — Agent calls included — execute in
# parallel; excess calls queue).
PARALLEL_TOOL_MAX = 8
# Tools a sub-agent must NOT see or call: it runs autonomously as a
# one-shot task inside the parent's tool round, so it cannot spawn
# further sub-agents (Agent), ask the user questions (Question), nor
# end in a plan/build handoff (PlanExit).  TodoWrite is also parent-only:
# a sub-agent is a single delegated task — progress tracking belongs to
# the parent, and the sub-agent must never clobber the parent's list.
SUBAGENT_EXCLUDED_TOOLS = ("Agent", "Question", "PlanExit", "TodoWrite")

# ---- TUI preview limits -------------------------------------------------------
TOOL_RESULT_PREVIEW_LINES = 5    # max lines of a tool result shown in the TUI
TOOL_RESULT_PREVIEW_CHARS = 500  # max chars of that preview (long single lines)

# ---- default agent prompts -----------------------------------------------------
# Ported system prompts (opencode-style) for the main agent and sub-agents,
# bundled with the package (prompts/agent.txt, prompts/subagent.txt).
# Missing files are tolerated: callers fall back to no system prompt.
PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_AGENT_PROMPT_FILE = PROMPTS_DIR / "agent.txt"
DEFAULT_SUBAGENT_PROMPT_FILE = PROMPTS_DIR / "subagent.txt"

# ---- configuration file ---------------------------------------------------------
CONFIG_DIR = Path(
    os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
) / "python-agent-harness"
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

CONFIG_TEMPLATE = """\
{{
  "_comment": "python-agent-harness configuration. Location: {path}. Precedence: code defaults < this file < OPENAI_* environment variables.",
  "llm": {{
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "reasoning_effort": "medium",
    "stream": true
  }},
  "paths": {{
    "_comment": "Optional overrides for context and skill directories. Absolute paths or ~ expansion supported.",
    "context_path": null,
    "skill_path": null
  }}
}}
"""

_ENV_OVERRIDES = {
    "base_url": "OPENAI_BASE_URL",
    "api_key": "OPENAI_API_KEY",
    "model": "OPENAI_MODEL",
    "backend": "OPENAI_BACKEND",
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
            "base_url", "api_key", "model", "backend",
            "temperature", "max_tokens", "timeout",
            "reasoning_effort", "stream",
        ):
            if key in llm and llm[key] is not None:
                settings[key] = llm[key]
    for key, env in _ENV_OVERRIDES.items():
        val = os.environ.get(env)
        if val:
            settings[key] = val
    return settings


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


def mask_secret(value: str | None) -> str:
    return "****" if value else "(unset)"
