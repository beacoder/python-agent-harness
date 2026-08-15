<div align="center">

# python-agent-harness

**A minimal-dependency coding agent for your terminal** — FSM-driven execution, OpenAI-compatible, made for daily use and easy customization.

[![CI](https://github.com/beacoder/python-agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/beacoder/python-agent-harness/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

Python coding-agent inspired by [gptel-agent-harness](https://github.com/beacoder/gptel-agent-harness): a terminal coding agent that reads your repo, plans, edits files, runs commands, and verifies its own work — with only **three runtime dependencies** (`rich`, `httpx`, `prompt_toolkit`) and any OpenAI-compatible API. No heavy frameworks, no vendor lock-in.

## Demo

![Demo](demo.png)

## Quick start

```sh
git clone git@github.com:beacoder/python-agent-harness.git
cd python-agent-harness
make install                        # create venv, install deps + package
. venv/bin/activate                 # add venv/bin to PATH
python-agent-harness config --init  # write ~/.config/python-agent-harness/config.json
python-agent-harness run            # launch the agent in your project dir
```

Optional extras: `pip install -e ".[mcp]"` enables MCP server integration (see below); `pip install -e ".[dev]"` installs the development tooling.

Edit `~/.config/python-agent-harness/config.json`, set your `base_url`, `api_key` and `model`, and you're ready to go:

```sh
python-agent-harness run ~/my-project
```

## Features

### Reliable by design

- **FSM-driven execution with completion supervision** — the run is driven by a finite state machine (`WAIT`/`TOOL`/`TRET`/`SUPERVISE`/`DONE`/`ERRS`/`ABRT`). When the model tries to stop before the task is done, it is nudged (max 2); the nudge counter resets on tool calls, and tool results are sanitized so a failed call never strands the machine.
- **Automatic retry with backoff** — transient failures (HTTP 429 / 5xx, connection errors) retry with exponential backoff + jitter, honoring `Retry-After`; retries never duplicate streamed output, Ctrl-C aborts the wait promptly, and permanent 4xx errors fail fast.

### Long conversations, no babysitting

- **Context management** — CJK-aware token estimation, per-model context windows (deepseek-v4/glm-5.2 1M, gpt-5 400k, kimi-k2.7 256k, claude 200k, ...), self-calibrating estimates from API-reported input tokens, and automatic compaction at 70% usage that summarizes the conversation and resumes with the last user request.

### Real coding tools

All OpenAI-compatible tool schemas: **Agent** (sub-agents), **TodoWrite**, **Glob** (git-aware), **Grep** (git grep → rg → grep), **Read**, **Insert**, **Edit** (incl. unified diffs), **Write**, **Mkdir**, **Bash**, **Skill**, **Question**, and **PlanExit** (plan mode only).

Tool execution mirrors gptel's `gptel--handle-tool-use`: synchronous tools (Read, Edit, Glob, ...) run one at a time in model-emitted order, while asynchronous tools (Bash, Agent) run concurrently in the background — results delivered in original call order.

### Plan before you build

- **Plan / Build modes** — plan mode is read-only except the per-session plan file; `PlanExit` switches back to build with an "execute the plan" prompt; sub-agents in plan mode get the read-only reminder.

### Sessions that survive

- Auto-saved after every response to `~/.local/share/python-agent-harness/sessions/`
- LLM-generated titles (one-shot, when the run finishes; file renamed to `<title>_<TS>.md`)
- `/restore` (with `--latest`) and `/sessions` TUI commands

### A TUI built for focus

- Rich live interface: pinned status bar (mode, context usage, spinner), streaming output, tool-result previews, pinned Todos panel (sub-agent lists labeled `sub:`), and numbered-choice prompts for Question / PlanExit.
- **Inline diff rendering** — Edit/Write calls capture a unified diff and render it red/green in the TUI, so file changes are visible without leaving the app.
- `prompt_toolkit` editor: Enter for newline, **Esc+Enter** (or Alt+Enter) to submit, **Tab completion** for slash commands and paths (`~/` and relative, Shift+Tab to cycle backwards), Up/Down history recall (persisted to `~/.local/share/python-agent-harness/input_history`), **Ctrl-D** quits, **Ctrl-C** cancels the current input or run without leaving the app — history is preserved so you can immediately ask a follow-up, and a cancelled worker can never clobber the next run's state (per-run cancellation identity).

### Extensible

- **MCP servers (optional)** — with `pip install -e ".[mcp]"`, any MCP server's tools become ordinary agent tools, namespaced `mcp__<server>__<tool>` (so `search` from two servers never collides). Supported transports: `stdio` (spawn a command), `streamable-http` and `sse` (remote URLs). Tool discovery happens once at session start; results are normalized into the harness's tool-result format, MCP errors surface as normal tool errors, and `parallel: true` opts a read-only server into concurrent execution (serial by default). Configured in the `mcp` section of the config file (or programmatically via `MCPConfig` + `session.connect_mcp()`); `enabled: false` keeps an entry without connecting it. The `mcp` extra is never required — the base harness works without it.
- **Default agent prompts** — distinct system prompts for the main agent and sub-agents (`prompts/agent.md`, `prompts/subagent.md`), YAML frontmatter stripped, `{{SKILLS}}` filled from the discovered skill directory; the main prompt is prefixed with project context files and task-completion rules.
- **Slash commands** — `/init`, `/review`, `/explain`, plus custom commands from `prompts/commands/*.md` become TUI slash commands automatically.

## Configuration

LLM settings live in a JSON config file — no environment variables required:

```json
{
  "llm": {
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "sk-...",
    "model": "deepseek-chat",
    "reasoning_effort": "medium",
    "stream": true
  },
  "subagent_llm": {
    "base_url": null,
    "api_key": null,
    "model": "deepseek-chat",
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

- `reasoning_effort` is passed to the API as-is (omitted when unset) — whatever your provider accepts ("low"/"medium"/"high").
- Other optional keys: `backend`, `temperature`, `max_tokens`, `timeout`, `stream` (`true` by default; `run --no-stream` overrides on the command line).
- `subagent_llm` configures the LLM for Agent-tool requests: every key is optional and unset keys inherit the main `llm`, so a cheaper/smaller model (or a different provider) can serve delegated work.
- `paths.context_path` / `paths.skill_path` override context/skill discovery — defaults are `<project>/contexts` or `~/.emacs.d/contexts` (skills: `<project>/skills` or `~/.emacs.d/skills`).
- `mcp.servers` configures MCP servers (requires the `[mcp]` extra). Each server is `{transport, command, args, env, url, headers, parallel, timeout, enabled}` — `stdio` needs `command`/`args` (optionally `env` naming environment variables to pass through, e.g. `["GITHUB_TOKEN"]`); `streamable-http`/`sse` need `url` (optionally `headers`). Its tools appear as `mcp__<server>__<tool>`.
- Precedence: code defaults < config file < `OPENAI_*` env vars (env still wins if set, but nothing is required). Sub-agent settings honor `OPENAI_SUBAGENT_*` (`_BASE_URL`, `_API_KEY`, `_MODEL`, `_BACKEND`).
- Use a custom config with `--config PATH` (or `PYTHON_AGENT_HARNESS_CONFIG`).

LLM request/response bodies are logged as JSON to `/tmp/python-agent-harness-<date>-<id>.json` (override the directory with `LLM_LOG_DIR`); the path is printed at TUI startup.

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
| `/restore [path\|title\|--latest]` | restore a session (title substring match) |
| `/clear` | start a fresh conversation |
| `/exit` | quit |

Custom commands from `prompts/commands/*.md` are registered as slash commands too (TUI-only — no CLI subcommand is registered for them). Tool availability differs per command: `/init` and `/review` may use all tools except `PlanExit` (hidden for the run, including for spawned sub-agents); custom commands may use everything; `compact`/`summary` run with no tools (a one-shot `chat_sync` call, like session-title generation).

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

Python ≥ 3.11 required (CI runs 3.11 / 3.12 / 3.13).

```sh
make test                           # unit tests (unittest discover)
venv/bin/pip install -e ".[dev]"    # dev tools: ruff, pyright, build, pip-audit
venv/bin/ruff check .               # lint (CI blocks on this)
venv/bin/pyright                    # type check, basic mode (CI blocks on this)
venv/bin/python -m build            # sdist + wheel
venv/bin/pip-audit                  # dependency audit
```

### Verification checklist (inherited semantics)

- [x] Nudge supervision with fail-closed dead-session budget
- [x] Tool-result sanitization (None → error placeholder)
- [x] Compaction: frame, resume last request
- [x] Plan mode: read-only + plan-file writes only
- [x] Bash: Ctrl-C process-group kill
- [x] Session metadata round-trip and title sanitization
- [x] One-shot LLM title generation after the agent run finishes
- [x] Ctrl-C cancel: stale workers can't clobber the next run's history

## License

MIT
