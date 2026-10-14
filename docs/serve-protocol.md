# Serve mode protocol

`python-agent-harness serve` is a persistent, bidirectional runtime boundary for hosting applications (web backends, IDEs, CI drivers). Unlike `headless --json` (one prompt per process, write-only stream), `serve` keeps the `Controller`/`Session` resident and speaks a request/response protocol over stdin/stdout.

The host can:

- Submit multiple prompts over the process's lifetime — no per-turn interpreter spawn, and conversation history is retained between them (multi-turn memory).
- Answer the agent's mid-run questions (the `Question` tool and plan-exit confirmation) via an `answer` op.
- Cancel a run as a protocol message (no signal semantics).

## Protocol

One JSON object per line.

```
host → agent: {"op": "submit", "prompt": ..., "run_id": ...}
              {"op": "answer", "run_id": ..., "answers": [...], "ask_id": ...}
              {"op": "cancel", "run_id": ...}
              {"op": "hello", "protocol_versions": [...]}
              {"op": "ping"} | {"op": "shutdown"}
              ... any op may carry "op_id", echoed on what it causes

agent → host: {"protocol": V, "type": "ready", "pid": ...,
               "protocol_version": V, "capabilities": [...]}   first line
              {"protocol": V, "type": "hello", "protocol_version": V,
               "capabilities": [...], "op_id": ...}   negotiation reply
              {"protocol": V, "seq": N, "type": "start"|"delta"|"notify"|"log",
               "run_id": ...}
              {"protocol": V, "seq": N, "type": "result", "run_id": ...,
               "answer": ..., "errors": [{"code", "message"}],
               "error_messages": [...], "usage": {...}, "cancelled": bool}
              {"protocol": V, "type": "error",
               "error": {"code": "protocol", "message": ...},
               "message": ...}                      protocol failures
```

## Budget limits

`--max-rounds` caps LLM rounds; `--timeout` caps wall-clock seconds. These are applied per `submit` (the server is resident, so every submit gets a fresh budget). Values `<= 0` or `null` disable. Unset = unlimited. A tripped budget ends the run with a normal `result` line carrying `errors[].code` of `budget` or `timeout` plus the token usage consumed. Enforced in-process so a driving host cannot raise its own ceiling.
