<div align="center">

# python-agent-harness

**A lightweight, hackable mini-OpenCode written in Python.**  
FSM-driven execution · OpenAI-compatible · built for daily use and easy customization

[![CI](https://github.com/beacoder/python-agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/beacoder/python-agent-harness/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/python-agent-harness.svg)](https://pypi.org/project/python-agent-harness/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/python-agent-harness?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/python-agent-harness)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

A terminal coding agent that reads your codebase, plans changes, edits files, runs commands, and verifies its work.

Inspired by [gptel-agent-harness](https://github.com/beacoder/gptel-agent-harness) and [opencode](https://github.com/anomalyco/opencode). It brings opencode's prompts and core behaviors — `AGENTS.md` discovery, plan/build modes, skills, sub-agents, todo tracking — into a lightweight Python implementation with only **three runtime dependencies**: `rich`, `httpx`, `prompt_toolkit`.

Works with any **OpenAI-compatible API**.

## Demo

![python-agent-harness demo](https://raw.githubusercontent.com/beacoder/python-agent-harness/main/demo.png)

## Quick start

```sh
pip install python-agent-harness          # or: pip install -e ".[dev]" from source
python-agent-harness config --init
python-agent-harness run
```

Optional extras: `[mcp]` for MCP server integration, `[dev]` for development tools.

Edit `~/.config/python-agent-harness/config.json` and set your `base_url`, `api_key`, and `model`.

## Features

- **FSM-driven execution** — explicit `WAIT` / `TOOL` / `TRET` / `SUPERVISE` / `DONE` / `ERRS` / `ABRT` states. Completion supervision nudges the model when it stops early; failed tool calls are sanitized. Transient API failures (`429` / `5xx`) retry with exponential backoff. Auth-expired codes (configurable via `AUTH_REFRESH_STATUS_CODES` in `config.py`; defaults to `[401, 502]`) trigger automatic API key re-read from config/env.
- **Context management** — CJK-aware token estimation, per-model context windows, automatic compaction at 70% usage.
- **Image & text attachments** — attach with `@path` (e.g. `@screenshot.png`, `@README.md`): images (PNG/JPEG/GIF/WebP, up to 20 MB) become multimodal `image_url` parts; text files are inlined. Web URLs supported for images.
- **Coding tools** — `Agent`, `TodoWrite`, `Glob`, `Grep`, `Read`, `Insert`, `Edit` (incl. unified diffs), `Write`, `Mkdir`, `Bash`, `Skill`, `Question`, `PlanExit`. Read-only tool rounds (`Read`, `Glob`, `Grep`, `Skill`) dispatch concurrently; async tools (`Bash`, `Agent`) can also run concurrently. Results always delivered in emitted order.
- **Plan / Build modes** — plan mode is read-only except for the per-session plan file.
- **Persistent sessions** — auto-saved after every response to `~/.local/share/python-agent-harness/sessions/`, with LLM-generated titles and `/restore --latest` / `/sessions`.
- **Focused TUI** — Rich-based with pinned status bar, Todos panel, inline diff rendering, and a `prompt_toolkit` editor with history and completion. `Esc+Enter` submits, `Ctrl-D` quits, `Ctrl-C` cancels.
- **MCP support** — optional via `[mcp]` extra. MCP tools become ordinary agent tools (`mcp__<server>__<tool>`). Supports `stdio`, `streamable-http`, and `sse` transports.
- **Slash commands** — built-in `/init`, `/review`, `/explain`, etc., plus custom commands from `prompts/commands/*.md`.
- **Custom agents** — switch the main agent's system prompt at runtime with `/agent`. Agent files live in `prompts/agents/*.md` and support `name:` and `exclude_tools:` frontmatter.
- **Embeddable runtime boundaries** — `headless --json` for one-shot CI/scripting; `serve` for hosting apps (resident process, bidirectional protocol: multi-turn memory, mid-run Q&A, protocol-level cancel).

## Usage

### TUI mode

```sh
python-agent-harness run [--project DIR]
```

### Headless mode

```sh
python-agent-harness headless [prompt] [--project DIR] [--restore [SPEC]] [--model NAME] [--json]
```

Runs a single prompt without the TUI — for CI, scripting, and piping. The assistant's answer goes to stdout; tool/status events go to stderr. Interactive prompts are auto-answered (`confirm` → yes, `ask` → "Unanswered").

- Prompt is a positional argument; when omitted, read from stdin.
- `--restore` continues a saved session. Bare `--restore` uses the most recent; a SPEC may be a file path or title substring.
- `--model NAME` selects a profile from the `models` config section, or a raw model name on the configured endpoint.
- `--json` emits JSON lines on stdout: `start`, `delta`, `notify`, `log`, and a final `result` (with structured `errors`). Each line carries `protocol` (wire schema version) and `seq` (monotonic counter). Exit code 0 on success, 1 on error.

### Serve mode

```sh
python-agent-harness serve [--project DIR] [--answer-timeout SECONDS] [--max-rounds N] [--timeout SECONDS]
```

A persistent, bidirectional runtime for hosting applications. Keeps the `Controller`/`Session` resident so the host can submit multiple prompts, answer mid-run questions, and cancel runs via protocol messages. See [docs/serve-protocol.md](docs/serve-protocol.md) for the full protocol specification.

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
| `/restore [path\|title\|--latest\|latest]` | Restore a session (substring search) |
| `/clear` | Start a fresh conversation |
| `/model [name]` | Switch LLM profiles; `default` restores original |
| `/agent [name]` | Switch agent prompt; `default` restores built-in |
| `/help` | List available commands |
| `/exit` | Quit |

Custom commands from `prompts/commands/*.md` are also registered (TUI only).

### Custom agents

Agent files are markdown (`.md`) in `prompts/agents/`. Each becomes a switchable profile via `/agent`. Optional YAML frontmatter:

- `name:` — override the agent name (defaults to file stem)
- `exclude_tools:` — list of tool names to hide. Matches by exact name, glob (`mcp__git__*`), or `__`-delimited prefix (`mcp__git` hides `mcp__git__list_repos` but `Write` does NOT hide `TodoWrite`).

**Commands vs Agents:** Commands (`/review`) are one-shot with a hardcoded kickoff and reset after run. Agents (`/agent reviewer`) persist until the next `/agent`, let the user type their own prompt, and stay active after `Ctrl-C`.

## Configuration

All LLM settings live in `~/.config/python-agent-harness/config.json`. Environment variables are optional.

```json
{
  "llm": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "gpt-5-mini",
    "reasoning_effort": null,
    "stream": true,
    "supports_image_input": false
  },
  "models": {
    "deepseek": { "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat" },
    "qwen": { "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen3.5-coder" }
  },
  "context_windows": {
    "deepseek-v4": 1000000
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
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "env": [],
        "parallel": false,
        "timeout": null,
        "enabled": false
      }
    }
  }
}
```

### Options

- **`llm`** — main LLM config. Optional keys: `temperature`, `max_tokens`, `timeout`, `reasoning_effort`, `stream`, `supports_image_input`. `run --no-stream` overrides `stream`. `supports_image_input` (default `false`) controls whether `@path` images are sent or stripped.
- **`models`** — named LLM profiles for `/model` switching. Partial settings; unset keys inherit from `llm`. `default` restores the main config.
- **`context_windows`** — per-model context-window overrides (tokens). Keys are model names or substrings; first match wins. Overrides `CONTEXT_WINDOWS` in `config.py`.
- **`subagent_llm`** — LLM config for `Agent` tool. Unset values inherit from `llm`. Set `profile` to reuse a `models` entry. Precedence: profile > `subagent_llm` > `llm` > env vars.
- **`default_agent`** — agent name to use at session start (must exist in `prompts/agents/`). `null` = built-in default.
- **`paths.context_path` / `paths.skill_path`** — locations for context files and skills. Default: project-local `contexts/` and `skills/`.
- **`mcp.servers`** — MCP server config (requires `[mcp]` extra). Each supports `transport`, `command`, `args`, `env`, `url`, `headers`, `parallel`, `timeout`, `enabled`.
- **`headless` / `serve`** — optional budget limits: `max_rounds` (LLM rounds) and `timeout` (wall-clock seconds). `--max-rounds` / `--timeout` override per call; `<= 0` or `null` disables. Unset = unlimited.
- **Precedence** — code defaults < config file < `OPENAI_*` env vars. Sub-agents also support `OPENAI_SUBAGENT_*` (`_BASE_URL`, `_API_KEY`, `_MODEL`).
- **Custom config** — `--config PATH` or `PYTHON_AGENT_HARNESS_CONFIG` env var.
- **LLM logging** — request/response bodies logged as JSON to `/tmp/python-agent-harness-<date>-<id>.json`. Set `LLM_LOG_DIR` to change. Path printed at startup.

## Inspired by opencode

Most of [opencode](https://github.com/anomalyco/opencode)'s prompts and core behaviors have been ported. The goal: retain its practical coding-agent workflow while keeping the implementation small and easy to customize.

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
| `AGENTS.md` handling | `prompts.py` (`find_agents_md_files`, `load_context_files`) |

## Development

Requires Python ≥ 3.11. CI runs against 3.11, 3.12, and 3.13 on Linux and macOS.

```sh
make test                           # unit tests
venv/bin/pip install -e ".[dev]"    # development tools
venv/bin/ruff check .               # lint
venv/bin/ruff format --check .      # formatting (CI blocks on this)
venv/bin/pyright                    # type checking
venv/bin/python -m build            # build sdist + wheel
venv/bin/pip-audit                  # dependency audit
```

## Related projects

- [gptel-agent-harness](https://github.com/beacoder/gptel-agent-harness) — the Emacs-based implementation that inspired this project.
- [opencode](https://github.com/anomalyco/opencode) — the primary source of many prompts and coding-agent behaviors.

## License

MIT
