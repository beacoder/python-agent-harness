<div align="center">

# python-agent-harness

**A minimal-dependency coding agent for your terminal** — FSM-driven execution, OpenAI-compatible, made for daily use and easy customization.

[![CI](https://github.com/beacoder/python-agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/beacoder/python-agent-harness/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

A terminal coding agent inspired by [gptel-agent-harness](https://github.com/beacoder/gptel-agent-harness) and [opencode](https://github.com/anomalyco/opencode): reads your repo, plans, edits files, runs commands, and verifies its own work — with only **three runtime dependencies** (`rich`, `httpx`, `prompt_toolkit`) and any OpenAI-compatible API. It ports opencode's prompts and behaviors (AGENTS.md discovery, plan/build modes, skills, sub-agents, todo tracking) while staying dependency-light.

## Demo

![Demo](demo.png)

## Quick start

```sh
git clone git@github.com:beacoder/python-agent-harness.git
cd python-agent-harness
make install                        # create venv, install deps + package
. venv/bin/activate
python-agent-harness config --init  # write ~/.config/python-agent-harness/config.json
python-agent-harness run            # launch the agent in your project dir
```

Edit `~/.config/python-agent-harness/config.json`, set `base_url`/`api_key`/`model`, and run. Optional: `pip install -e ".[mcp]"` for MCP server integration; `pip install -e ".[dev]"` for dev tooling.

## Features

- **FSM-driven execution** (`WAIT`/`TOOL`/`TRET`/`SUPERVISE`/`DONE`/`ERRS`/`ABRT`) with completion supervision: the model is nudged (max 2) if it stops early; failed tool calls are sanitized and never strand the machine. Transient failures (429/5xx) retry with exponential backoff + jitter.
- **Context management** — CJK-aware token estimation, per-model context windows, automatic compaction at 70% usage.
- **Real coding tools** — Agent (sub-agents), TodoWrite, Glob, Grep, Read, Insert, Edit (incl. unified diffs), Write, Mkdir, Bash, Skill, Question, PlanExit. Synchronous tools run one at a time; asynchronous ones (Bash, Agent) run concurrently in emitted order.
- **Plan / Build modes** — plan mode is read-only except the per-session plan file.
- **Sessions that survive** — auto-saved to `~/.local/share/python-agent-harness/sessions/` after every response, LLM-generated titles, `/restore --latest`, `/sessions`.
- **A TUI built for focus** — rich live interface with pinned status bar, Todos panel, inline red/green diff rendering for Edit/Write, `prompt_toolkit` editor (Esc+Enter to submit, Tab completion, history, Ctrl-D quits, Ctrl-C cancels without leaving the app).
- **MCP servers (optional)** — with the `[mcp]` extra, MCP tools become ordinary agent tools (`mcp__<server>__<tool>`); supports `stdio`, `streamable-http`, `sse` transports.
- **Slash commands** — `/init`, `/review`, `/explain`, plus custom commands from `prompts/commands/*.md`.

## Mini opencode

Most of [opencode](https://github.com/anomalyco/opencode)'s prompts and behaviors have been ported over, so the agent reasons and works like opencode while staying dependency-light.

**Prompts extracted from opencode** (in `python_agent_harness/prompts/`):

| opencode prompt | harness equivalent |
|---|---|
| `default.txt` (main agent) | `agent.md` |
| `plan.txt` / `plan-mode.txt` / `build-switch.txt` | `plan.md` / `plan-mode.md` / `build-switch.md` |
| `task.txt` (subagent) | `subagent.md` + `Agent` tool |
| `todowrite.txt` / `question.txt` / `skill.txt` | `TodoWrite` / `Question` / `Skill` tools |
| `read.txt` / `write.txt` / `edit.txt` / `grep.txt` / `glob.txt` | `Read` / `Write` / `Edit` / `Grep` / `Glob` tools |
| `shell.txt` (git/bash guidance) | `Bash` tool + `agent.md` "Git and GitHub" section |
| `plan-enter.txt` / `plan-exit.txt` | `PlanExit` tool |
| `initialize.txt` / `review.txt` / `explain` commands | `initialize.md` / `review.md` / `commands/explain.md` |
| compaction / summary / title | `compact.md` / `summary.md` / `title.md` |
| AGENTS.md handling | `prompts.py` (`find_agents_md_files`, `load_context_files`, per-file resolution) |

## Configuration

All LLM settings live in one JSON config file (no env vars required):

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
    "deepseek": { "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat" },
    "qwen": { "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen3.5-coder" }
  },
  "subagent_llm": {
    "profile": null,
    "base_url": null, "api_key": null, "model": null,
    "temperature": null, "max_tokens": null, "timeout": null,
    "reasoning_effort": null, "stream": null
  },
  "paths": { "context_path": null, "skill_path": null },
  "mcp": {
    "servers": {
      "example": { "transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"], "env": [], "parallel": false, "timeout": null, "enabled": false }
    }
  }
}
```

- Optional `llm` keys: `backend`, `temperature`, `max_tokens`, `timeout`, `reasoning_effort` (passed to the API as-is when set), `stream` (`run --no-stream` overrides).
- **`models`** — named LLM profiles for runtime switching via `/model` (in the TUI: no arg lists, name or number switches; `default` always restores the main `llm` settings). Each profile is a partial settings dict; unset keys inherit the main `llm`.
- **`subagent_llm`** — LLM for Agent-tool requests; every key optional, unset keys inherit main `llm`. Set `profile` to a name from `models` to reuse a profile; precedence: profile settings > explicit `subagent_llm` keys > main `llm` > env.
- **`paths.context_path` / `paths.skill_path`** — where to load context files / skills from. Unset means the project's own `<project>/contexts` and `<project>/skills`.
- **`mcp.servers`** — requires the `[mcp]` extra; each server is `{transport, command, args, env, url, headers, parallel, timeout, enabled}`.
- **Precedence**: code defaults < config file < `OPENAI_*` env vars. Sub-agent settings honor `OPENAI_SUBAGENT_*` (`_BASE_URL`, `_API_KEY`, `_MODEL`, `_BACKEND`).
- Custom config: `--config PATH` or `PYTHON_AGENT_HARNESS_CONFIG`.
- LLM request/response bodies are logged as JSON to `/tmp/python-agent-harness-<date>-<id>.json` (override dir with `LLM_LOG_DIR`); path printed at startup.

## Usage

```sh
python-agent-harness run [project-dir]   # interactive TUI agent
```

| Command | What it does |
|---|---|
| `/plan` / `/build` | switch between read-only plan and build mode |
| `/init` | create/update `AGENTS.md` |
| `/review` | review uncommitted changes / commit / branch / PR |
| `/explain [project] [target]` | explain code |
| `/compact` | compact the conversation |
| `/summary` | append a conversation summary |
| `/save` | save the session |
| `/sessions` | list saved sessions |
| `/restore [path\|title\|--latest\|latest]` | restore a session (title substring match) |
| `/clear` | start a fresh conversation |
| `/model [name]` | switch LLM model profile (`default` restores the session's original model; no arg: list available) |
| `/exit` | quit |

Custom commands from `prompts/commands/*.md` are registered as slash commands too (TUI-only).

## Project layout

```
python_agent_harness/
├── agent.py        agent FSM (states, transitions, supervision, compaction)
├── client.py       OpenAI-compatible streaming client (httpx)
├── models.py       Message / ToolCall / ToolSpec data classes
├── token_estimator.py  CJK-aware token estimation + calibration
├── planmode.py     build/plan mode + plan file lifecycle
├── prompts.py      prompt loading + system prompt assembly
├── session_store.py  session persistence + titles
├── agent_session.py  AgentSession (wiring hub; MCP lifecycle)
├── subagent.py     sub-agent runner (error containment, plan reminder)
├── commands.py     init/review/custom command definitions
├── cli.py          argparse entry points
├── tui.py          rich + prompt_toolkit TUI
├── diffrender.py   unified diff generation + rich rendering
├── mcp/            optional MCP client (config, SDK wrapper, manager)
└── tools/          tool implementations + registry (incl. MCPTool adapter)
```

## Development

Python ≥ 3.11 (CI runs 3.11 / 3.12 / 3.13).

```sh
make test                           # unit tests (unittest discover)
venv/bin/pip install -e ".[dev]"    # dev tools: ruff, pyright, build, pip-audit
venv/bin/ruff check .               # lint (CI blocks on this)
venv/bin/pyright                    # type check, basic mode (CI blocks on this)
venv/bin/python -m build            # sdist + wheel
venv/bin/pip-audit                  # dependency audit
```

## Principle

Keep it intact, not bloated.

## License

MIT
