<div align="center">

# python-agent-harness

**A lightweight, hackable mini-OpenCode written in Python.**  
FSM-driven execution · OpenAI-compatible · built for daily use and easy customization

[![CI](https://github.com/beacoder/python-agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/beacoder/python-agent-harness/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/python-agent-harness.svg)](https://pypi.org/project/python-agent-harness/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

A terminal coding agent that reads your codebase, plans changes, edits files, runs commands, and verifies its work.

`python-agent-harness` is inspired by [gptel-agent-harness](https://github.com/beacoder/gptel-agent-harness) and [opencode](https://github.com/anomalyco/opencode). It brings opencode's prompts and core behaviors—such as `AGENTS.md` discovery, plan/build modes, skills, sub-agents, and todo tracking—into a lightweight Python implementation with only **three runtime dependencies**:

- `rich`
- `httpx`
- `prompt_toolkit`

It works with any **OpenAI-compatible API** and is designed to be easy to inspect, customize, and use for everyday software development.

## Demo

![python-agent-harness demo](https://raw.githubusercontent.com/beacoder/python-agent-harness/main/demo.png)

## Quick start

### Install from PyPI

```sh
pip install python-agent-harness

python-agent-harness config --init
python-agent-harness run
```

Optional extras:

```sh
pip install "python-agent-harness[mcp]"   # MCP server integration
pip install "python-agent-harness[dev]"   # development tools
```

### Install from source (GitHub)

```sh
git clone git@github.com:beacoder/python-agent-harness.git
cd python-agent-harness

make install
. venv/bin/activate

python-agent-harness config --init
python-agent-harness run
```

Optional extras:

```sh
pip install -e ".[mcp]"   # MCP server integration
pip install -e ".[dev]"   # development tools
```

Edit `~/.config/python-agent-harness/config.json` and set your `base_url`, `api_key`, and `model`.

## Features

- **FSM-driven execution** — explicit `WAIT` / `TOOL` / `TRET` / `SUPERVISE` / `DONE` / `ERRS` / `ABRT` states. Completion supervision nudges the model when it stops early, while failed tool calls are sanitized so they never strand the agent. Transient API failures (`429` / `5xx`) retry with exponential backoff and jitter.
- **Context management** — CJK-aware token estimation, per-model context windows, and automatic compaction at 70% usage.
- **Coding tools** — `Agent`, `TodoWrite`, `Glob`, `Grep`, `Read`, `Insert`, `Edit` (including unified diffs), `Write`, `Mkdir`, `Bash`, `Skill`, `Question`, and `PlanExit`. Synchronous tools execute sequentially; asynchronous tools such as `Bash` and `Agent` can run concurrently while preserving emitted order.
- **Plan / Build modes** — plan mode is read-only except for the per-session plan file.
- **Persistent sessions** — sessions are automatically saved after every response to `~/.local/share/python-agent-harness/sessions/`, with LLM-generated titles and support for `/restore --latest` and `/sessions`.
- **Focused TUI** — a Rich-based interface with a pinned status bar, Todos panel, inline red/green diff rendering for `Edit` and `Write`, and a `prompt_toolkit` editor with history and completion. `Esc+Enter` submits, `Ctrl-D` quits, and `Ctrl-C` cancels without leaving the application.
- **MCP support** — optional MCP integration through the `[mcp]` extra. MCP tools become ordinary agent tools such as `mcp__<server>__<tool>`. Supports `stdio`, `streamable-http`, and `sse` transports.
- **Slash commands** — built-in `/init`, `/review`, `/explain`, and other commands, plus custom commands loaded from `prompts/commands/*.md`.

## Inspired by opencode

Most of [opencode](https://github.com/anomalyco/opencode)'s prompts and core behaviors have been ported to this project. The goal is to retain its practical coding-agent workflow while keeping the implementation small, dependency-light, and easy to customize.

### Prompt and behavior mapping

The following opencode prompts have corresponding implementations in `python-agent-harness`:

| opencode | python-agent-harness |
|---|---|
| `default.txt` (main agent) | `agent.md` |
| `plan.txt` / `plan-mode.txt` / `build-switch.txt` | `plan.md` / `plan-mode.md` / `build-switch.md` |
| `task.txt` (sub-agent) | `subagent.md` + `Agent` tool |
| `todowrite.txt` / `question.txt` / `skill.txt` | `TodoWrite` / `Question` / `Skill` tools |
| `read.txt` / `write.txt` / `edit.txt` / `grep.txt` / `glob.txt` | `Read` / `Write` / `Edit` / `Grep` / `Glob` tools |
| `shell.txt` | `Bash` tool + `agent.md` Git/GitHub guidance |
| `plan-enter.txt` / `plan-exit.txt` | `PlanExit` tool |
| `initialize.txt` / `review.txt` / `explain` | `initialize.md` / `review.md` / `commands/explain.md` |
| compaction / summary / title | `compact.md` / `summary.md` / `title.md` |
| `AGENTS.md` handling | `prompts.py` (`find_agents_md_files`, `load_context_files`, per-file resolution) |

## Configuration

All LLM settings live in a single JSON configuration file. Environment variables are optional.

```json
{
  "llm": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "gpt-5-mini",
    "reasoning_effort": null,
    "stream": true
  },
  "models": {
    "_comment": "Named LLM profiles for /model switching. Partial settings; unset keys inherit the main llm.",
    "deepseek": {
      "base_url": "https://api.deepseek.com/v1",
      "model": "deepseek-chat"
    },
    "qwen": {
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "model": "qwen3.5-coder"
    }
  },
  "context_windows": {
    "_comment": "Optional per-model context-window overrides (tokens). Keys are model names or substrings (e.g. deepseek-v4 = 1000000); matched in file order, first match wins. Overrides the built-in CONTEXT_WINDOWS table in config.py.",
    "deepseek-v4": 1000000
  },
  "subagent_llm": {
    "profile": null,
    "base_url": null,
    "api_key": null,
    "model": null,
    "temperature": null,
    "max_tokens": null,
    "timeout": null,
    "reasoning_effort": null,
    "stream": null
  },
  "paths": {
    "context_path": null,
    "skill_path": null
  },
  "mcp": {
    "servers": {
      "example": {
        "transport": "stdio",
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-filesystem",
          "/tmp"
        ],
        "env": [],
        "parallel": false,
        "timeout": null,
        "enabled": false
      }
    }
  }
}
```

### Configuration options

- **`llm`** — main LLM configuration. Optional keys include `backend`, `temperature`, `max_tokens`, `timeout`, `reasoning_effort`, and `stream`. Values such as `reasoning_effort` are passed to the API as-is when set. `run --no-stream` overrides `stream`.
- **`models`** — named LLM profiles for runtime switching with `/model`. A profile is a partial settings dictionary; unset keys inherit from the main `llm`. `default` restores the main LLM configuration.
- **`context_windows`** — optional per-model context-window overrides (tokens). Keys are model names or substrings (e.g., `deepseek-v4`); matched in file order, first match wins. Overrides the built-in `CONTEXT_WINDOWS` table in `config.py`.
- **`subagent_llm`** — LLM configuration for `Agent` tool requests. Unset values inherit from the main `llm`. Set `profile` to reuse a profile from `models`. Precedence is: profile settings > explicit `subagent_llm` settings > main `llm` > environment variables.
- **`paths.context_path` / `paths.skill_path`** — locations from which to load context files and skills. When unset, the project-local `<project>/contexts` and `<project>/skills` directories are used.
- **`mcp.servers`** — MCP server configuration. Requires the `[mcp]` extra. Each server supports `transport`, `command`, `args`, `env`, `url`, `headers`, `parallel`, `timeout`, and `enabled`.
- **Configuration precedence** — code defaults < config file < `OPENAI_*` environment variables. Sub-agent settings also support `OPENAI_SUBAGENT_*` (`_BASE_URL`, `_API_KEY`, `_MODEL`, `_BACKEND`).
- **Custom config** — use `--config PATH` or `PYTHON_AGENT_HARNESS_CONFIG`.
- **LLM logging** — request and response bodies are logged as JSON to `/tmp/python-agent-harness-<date>-<id>.json`. Set `LLM_LOG_DIR` to change the directory. The log path is printed at startup.

## Usage

```sh
python-agent-harness run [project-dir]
```

Launches the interactive TUI agent. If `project-dir` is omitted, the current directory is used.

### Slash commands

| Command | Description |
|---|---|
| `/plan` / `/build` | Switch between read-only plan mode and build mode |
| `/init` | Create or update `AGENTS.md` |
| `/review` | Review uncommitted changes, commits, branches, or pull requests |
| `/explain [project] [target]` | Explain code |
| `/compact` | Compact the conversation |
| `/summary` | Append a conversation summary |
| `/save` | Save the current session |
| `/sessions` | List saved sessions |
| `/restore [path\|title\|--latest\|latest]` | Restore a session; title matching uses substring search |
| `/clear` | Start a fresh conversation |
| `/model [name]` | Switch LLM profiles; `default` restores the session's original model |
| `/exit` | Quit |

Custom commands from `prompts/commands/*.md` are registered as slash commands as well (TUI only).

## Project layout

```text
python_agent_harness/
├── agent.py           # Agent FSM core: states, transitions, supervision
├── tool_runner.py     # Tool-call execution/delivery + history salvage
├── context_manager.py # Context-ratio tracking + compaction
├── client.py          # OpenAI-compatible streaming client (httpx)
├── models.py          # Message / ToolCall / ToolSpec data classes
├── token_estimator.py # CJK-aware token estimation + calibration
├── planmode.py        # Plan/build modes + plan-file lifecycle
├── prompts.py         # Prompt loading + system-prompt assembly
├── persistence.py     # Session persistence + titles
├── session.py         # Session wiring hub + MCP lifecycle
├── subagent.py        # Sub-agent runner + error containment
├── commands.py        # Init/review/custom command definitions
├── cli.py             # CLI entry points
├── tui/               # Rich + prompt_toolkit TUI (package)
├── diffrender.py      # Unified diff generation + Rich rendering
├── mcp/               # Optional MCP client
└── tools/             # Tool implementations + registry
```

## Development

Requires Python ≥ 3.11. CI runs against Python 3.11, 3.12, and 3.13 on Linux and macOS.

```sh
make test                           # unit tests
venv/bin/pip install -e ".[dev]"    # development tools
venv/bin/ruff check .               # lint
venv/bin/pyright                    # type checking
venv/bin/python -m build            # build sdist + wheel
venv/bin/pip-audit                  # dependency audit
```

CI blocks on Ruff and Pyright failures.

## Design philosophy

**Keep it intact, not bloated.**

The project aims to provide a capable coding-agent within a lightweight framework.

## Related projects

- [gptel-agent-harness](https://github.com/beacoder/gptel-agent-harness) — the Emacs-based implementation that inspired this project.
- [opencode](https://github.com/anomalyco/opencode) — the primary source of many prompts and coding-agent behaviors.

## License

MIT
