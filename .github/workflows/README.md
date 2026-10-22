# GitHub Actions Workflows

## DeepSWE Benchmark

The `deepswe-benchmark.yml` workflow runs the python-agent-harness on the
DeepSWE v1.1 benchmark entirely in the cloud — nothing is installed or executed
locally.

### Setup

1. Add GitHub Secrets:
   - `OPENAI_API_KEY`: model API key (required)
   - `OPENAI_BASE_URL`: optional, defaults to `https://api.deepseek.com/v1`

2. Navigate to **Actions** → **DeepSWE Benchmark** → **Run workflow**

3. Configure inputs and run (`workflow_dispatch` only — no spend on push)

The per-task agent budget is derived automatically so the requested number of
tasks fits the 6 h job cap; the derived values are written to the run summary.
For a meaningful score, run 1–2 tasks at the full 180 min instead of many tasks
at a truncated budget.

### Requirements

GitHub-hosted `ubuntu-latest` is enough: it ships Docker Engine (the `Check
Docker` step asserts it) and Python 3.12, which Pier 0.3.x requires
(`requires-python >= 3.12`). The driver venv is therefore created with
`uv venv --python 3.12`; inside the task containers the harness runs on the
image's own interpreter.

### Artifacts

- `deepswe-results-{run_id}`: `pier-results/` (per-task `result.json`, verifier
  scores, agent trajectories) plus the harness wheel; retained 30 days

See [../../benchmarks/deepswe-benchmark.md](../../benchmarks/deepswe-benchmark.md)
for full documentation, including where the score lives and the pass-rate
one-liner.
