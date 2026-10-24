# SWE-bench Lite (local run)

Run [python-agent-harness](https://github.com/beacoder/python-agent-harness) on
[SWE-bench Lite](https://huggingface.co/SWE-bench/SWE-bench_Lite) on your own machine.

`run-swebench-lite-local.sh` does everything: it creates a driver venv, builds
the harness wheel, runs the agent inside each task's Docker container, writes
`predictions.jsonl`, grades it with `swebench eval`, and prints a summary.

## Requirements

- Docker Engine (the daemon must be reachable from your shell)
- Python 3.11+ on the host
- ~8 GB RAM per concurrent task, ~120 GB disk for all 300 images

## Steps

```bash
cd /path/to/python-agent-harness

# 1. Credentials: either the harness config or an env var
#    (profiles live in ~/.config/python-agent-harness/config.json)
export OPENAI_API_KEY=sk-...

# 2. Smoke test: 1 instance, ~10 min
MODEL=deepseek TASK_TIMEOUT_MINUTES=10 \
  benchmarks/swe-bench-lite/run-swebench-lite-local.sh

# 3. A real run
MODEL=deepseek N_TASKS=10 benchmarks/swe-bench-lite/run-swebench-lite-local.sh

# 4. Full dataset (300 instances)
MODEL=deepseek N_TASKS=0 benchmarks/swe-bench-lite/run-swebench-lite-local.sh
```

`MODEL` accepts a profile name from the harness config or a raw model id. The
key, base URL and real model id are resolved from the profile, so `MODEL=deepseek`
ends up calling `deepseek-flash`. Explicit `OPENAI_*` variables always win.

Results land in `swebench-results/` (override with `RESULTS_DIR`):

- `predictions.jsonl` — the patches the agent produced
- `<model>.<run-id>.json` — the `swebench eval` report
- `summary.md` — pass rate and per-instance table
- `agent-logs/<instance_id>.log` — agent stdout/stderr per instance

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `N_TASKS` | 1 | Instances to run (0 = all 300) |
| `MODEL` | `deepseek-flash` | Config profile name or raw model id |
| `SAMPLE_SEED` | 0 | Seed for instance selection (same seed = same instances) |
| `TASK_TIMEOUT_MINUTES` | 30 | Agent budget per instance |
| `CONCURRENCY` | 1 | Parallel agent runs |
| `MAX_ROUNDS` | 0 | Max LLM rounds per instance (0 = unlimited) |
| `OPENAI_API_KEY` | — | API key (falls back to the harness config) |
| `OPENAI_BASE_URL` | `https://api.deepseek.com/v1` | API endpoint |
| `SKIP_AGENT` | 0 | 1 = re-grade existing predictions without running the agent |
| `SKIP_EVAL` | 0 | 1 = only run the agent, skip grading |
| `SKIP_PULL` | 0 | 1 = do not pre-pull Docker images |
| `VENV_DIR` | `<repo>/.venv-swebench` | Driver venv |
| `RESULTS_DIR` | `<repo>/swebench-results` | Output directory |
| `EXTRA_ARGS` | — | Extra flags passed to `swebench_agent.py` |

## Common variants

```bash
# Re-grade the predictions you already have (no API calls)
MODEL=deepseek SKIP_AGENT=1 benchmarks/swe-bench-lite/run-swebench-lite-local.sh

# Produce patches only, grade later
MODEL=deepseek SKIP_EVAL=1 benchmarks/swe-bench-lite/run-swebench-lite-local.sh
```

## Troubleshooting

- **`docker: permission denied`** — `sudo usermod -aG docker $USER`, then re-login.
  The script can also re-run itself through `sg docker` automatically.
- **Containers OOM-killed** — keep `CONCURRENCY=1` and don't run other heavy jobs
  during evaluation.
- **Agent produced no patch** — check `agent-logs/<instance_id>.log`; usually a
  timeout or a missing API key.
