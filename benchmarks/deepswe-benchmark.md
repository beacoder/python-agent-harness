# DeepSWE Benchmark Integration

This document explains how to run the [python-agent-harness](https://github.com/beacoder/python-agent-harness) on the [DeepSWE v1.1](https://github.com/datacurve-ai/deep-swe) benchmark using GitHub Actions.

## Overview

DeepSWE is a benchmark for evaluating frontier coding agents on long-horizon software engineering tasks. It uses [Pier](https://pypi.org/project/datacurve-pier/) as the execution framework and provides 113 real-world tasks from various repositories.

## Quick Start

### 1. Configure GitHub Secrets

Add the following secrets to your repository settings:

```
OPENAI_API_KEY = your-api-key-here
OPENAI_BASE_URL = https://api.deepseek.com/v1  # Optional, defaults to DeepSeek
```

`OPENAI_BASE_URL` is optional; the adapter defaults to DeepSeek. For other
providers set it explicitly:
- **DeepSeek** (default): `https://api.deepseek.com/v1`
- **OpenAI**: `https://api.openai.com/v1`
- **Anthropic**: `https://api.anthropic.com/v1`
- **ModelScope**: `https://api-inference.modelscope.cn/v1`

Optionally, add AWS credentials:

```
AWS_ACCESS_KEY_ID = ...
AWS_SECRET_ACCESS_KEY = ...
```

These are used **only** to authenticate to ECR Public, where the prebuilt task
images live. ECR Public meters anonymous pulls per source IP and GitHub runners
share theirs, so without credentials the job can fail with
`429 ... Data limit exceeded` even though the model API key is fine. See
[ECR Public throttles the task image pulls](#ecr-public-throttles-the-task-image-pulls-429).

The request body carries `reasoning_effort: "max"` and a fixed `temperature: 0.0`
(see `DEFAULT_REASONING_EFFORT` in the adapter). Both are overridable without
editing code:

```bash
--ak 'model_kwargs={"reasoning_effort": "high", "temperature": 0.3}'
```

### 2. Run the Benchmark

Navigate to **Actions** → **DeepSWE Benchmark** → **Run workflow**

Configure the inputs:
- **n_tasks**: Number of tasks (1, 2, 10, 25, 50, or 0 for full suite). A fixed
  `--sample-seed 0` is applied so the same subset is picked every run.
  **1–2 is the only choice that keeps a meaningful agent budget** — larger
  values get their per-task budget scaled down to fit the 6 h job cap.
- **model**: Model identifier (default `deepseek-flash`)
- **max_rounds**: Maximum LLM rounds per task (0 = unlimited)
- **task_timeout_minutes**: Agent budget per task (default 180 = DeepSWE's own
  3 h). It is automatically scaled down when the requested number of tasks
  cannot fit the 6 h job cap — see
  [Wall-clock budget](#wall-clock-budget-how-the-workflow-fits-the-6-h-cap).
- **concurrency**: Concurrent trials. Default `1`: each trial is capped at
  2 vCPU / 8 GB, and the runner only has 4 vCPU / 16 GB (see
  [Resource limits](#resource-limits-why-concurrency-defaults-to-1)).

The job timeout is derived from these inputs (it is not a separate knob), and
both the derived per-task budget and the job timeout are written to the run
summary.

The workflow is manual only (`workflow_dispatch`), so no API spend happens on
push.

### 3. View Results

The **Summary** page of the run shows the agent budget the scores were produced
under, a per-task table, the pass rate, and token totals. The budget line
matters: a pass rate at a 20 min budget is not comparable to one at the full
180 min, so record it whenever you quote a score.

Artifacts hold the raw data:

- **deepswe-results-{run_id}**: `pier-results/` (job + trial directories) and
  the built harness wheel
- Retained for 30 days; download it to keep a permanent record

Inside a trial directory:

```
pier-results/<job>/<trial>/
├── result.json          # TrialResult: verifier_result.rewards, token usage
├── config.json          # resolved trial config
├── agent/               # harness JSON event stream (mounted at /logs/agent)
│   └── python-agent-harness.txt
└── verifier/
    ├── reward.txt       # score (DeepSWE is binary: 0 or 1)
    ├── reward.json      # same, JSON form
    └── test-stdout.txt  # held-out test output
```

See [Where the benchmark runs](#where-the-benchmark-runs) for the pass-rate
one-liner and `pier view`.

## Architecture

### Custom Agent Adapter

[`deepswe_adapter.py`](./deepswe_adapter.py) implements Pier's
`BaseInstalledAgent`:

```python
from pier.agents.installed.base import BaseInstalledAgent


class PythonAgentHarness(BaseInstalledAgent):
    def install_spec(self):
        # Runtime deps, or the released package when no wheel is passed
        ...

    async def setup(self, environment):
        # Optionally install the locally built wheel, then write
        # /tmp/python-agent-harness/config.json (selected at run time with
        # PYTHON_AGENT_HARNESS_CONFIG)
        ...

    async def run(self, instruction, environment, context):
        # python -m python_agent_harness headless --json '<instruction>'
        ...

    def populate_context_post_run(self, context):
        # Parse the final `result` event for token usage
        ...
```

Notes:

- **The adapter must declare a network allowlist.** Every DeepSWE task sets
  `[agent] network_mode = "no-network"`, which Pier resolves to
  `allow_internet = False`. When an agent returns a non-empty
  `network_allowlist()`, Pier attaches a filtered Squid egress proxy and injects
  `HTTP(S)_PROXY` into the agent process; when the allowlist is empty it instead
  applies `network_mode: none`, so *no* request — including the LLM call —
  leaves the container. `PythonAgentHarness.network_allowlist()` derives the
  host from `OPENAI_BASE_URL` (default: `api.deepseek.com`) for
  exactly this reason.
- The workflow builds a wheel from the checkout and passes it with
  `--ak wheel=<path>`; the adapter uploads it into the container and installs it
  with `--no-index --no-deps` (the container has no network at run time). Without
  `wheel`, the adapter installs the released package from PyPI at image build
  time instead.
- The harness config is written to `/tmp/python-agent-harness/config.json` and
  selected with `PYTHON_AGENT_HARNESS_CONFIG`. The harness's default location
  follows `XDG_CONFIG_HOME` / `$HOME`, neither of which the task images
  guarantee for the user the adapter writes as — a mismatch leaves the run with
  no API key and fails every task on auth.
- The harness is installed with `pip --target /opt/python-agent-harness`
  (not `--user`): it sidesteps PEP 668's
  `externally-managed-environment` error on modern images and makes the
  interpreter that installed it and the one that runs it agree. Runs use
  `PYTHONPATH=/opt/python-agent-harness python3 -m python_agent_harness`.
- `headless --json` writes one JSON object per line to stdout; the run is teed
  into `/logs/agent/python-agent-harness.txt`, which Pier mounts from the
  trial's `agent/` directory. `populate_context_post_run` reads it back on the
  host for token/round accounting.
- `max_rounds` is **not** a Pier flag: it is forwarded to the adapter with
  `--ak max_rounds=N` and becomes `headless --max-rounds N`.
- **Uncommitted work is committed before grading.** Each task collects the
  solution with `git diff <base_commit> HEAD` (the `[[verifier.collect]]` hook in
  `task.toml`), which ignores working-tree changes. Tasks do end their
  instruction with "commit everything when you are done", but a run that fixed
  the code and forgot to commit would score 0 for reasons unrelated to the task.
  After the run the adapter commits any leftover changes (`git add -A` +
  commit with an inline identity, in the task workdir); it is a no-op when the
  agent already committed or the workdir is not a repo. It runs in a `finally`,
  so partial work is still committed when the run ends badly (round budget
  exhausted, agent timeout, cancellation). Disable it with
  `--ak commit_remaining=false`. The held-out tests still decide pass/fail.

### Execution Flow

```
GitHub Actions Runner
    ↓
Checkout python-agent-harness + DeepSWE tasks
    ↓
Install harness (editable) + Pier >= 0.3.1 into a Python 3.12 venv, build wheel
    ↓
Log in to ECR Public (if AWS secrets are set), pre-pull the task images
    ↓
pier run -p deep-swe/tasks --agent-import-path ...
    ↓
Per task: build an image FROM the task's prebuilt image,
          adding the harness dependencies (build time = has network);
          attach the Squid egress proxy for the LLM host only
    ↓
python -m python_agent_harness headless --json '<instruction>'
    ↓
Commit leftover work (the verifier grades `git diff <base> HEAD`)
    ↓
Verifier scores the resulting workspace in a pristine container
    ↓
Job + trial results saved to pier-results/
```

Because the adapter supplies an `install_spec()`, Pier does **not** use the
prebuilt image as-is: it derives a small child image (`FROM
<task.docker_image>` + `pip install`) and caches it by the install fingerprint.
That is what puts the harness inside an otherwise air-gapped image.

## Task Structure

Each DeepSWE task follows the Harbor task layout:

- `instruction.md`: Prompt shown to the agent
- `task.toml`: Task metadata, limits, environment settings
- `environment/Dockerfile`: Container image (and working directory)
- `tests/`: Held-out tests run by the verifier

Scoring is binary (0/1): the agent's patch must make the fail-to-pass tests pass
without breaking the pass-to-pass tests.

## Cost Management

Running benchmarks can incur significant API costs:

| Tasks | Estimated Cost (GPT-5.5) |
|-------|--------------------------|
| 10    | ~$5-15 USD               |
| 25    | ~$15-40 USD              |
| 50    | ~$30-80 USD              |
| 113   | ~$70-200 USD             |

Recommendations:
- Start with 10 tasks to validate setup
- Use `max_rounds` to control token usage
- Monitor API usage in your provider dashboard

## Troubleshooting

### Workflow file rejected as invalid

GitHub Actions expressions have **no arithmetic operators** (`+ - * / %`).
Compute such values in a shell step and pass them through `outputs`; the
workflow's `prepare` job does exactly this for the job timeout.

### `No solution found when resolving dependencies` (Pier needs Python 3.12)

`datacurve-pier` 0.3.x declares `requires-python >= 3.12`, so the driver venv
must be 3.12 or newer even though the harness itself still supports 3.11:

```
error: Because the current Python version (3.11.17) does not satisfy Python>=3.12
and datacurve-pier==0.3.1 depends on Python>=3.12, we can conclude that
datacurve-pier==0.3.1 cannot be used.
```

The workflow creates its venv with `uv venv .venv --python 3.12`. Only the
host-side Pier driver runs on 3.12; inside the task container the harness runs
on the image's own interpreter (`python3`, 3.11+).

### ECR Public throttles the task image pulls (429)

DeepSWE's prebuilt images come from `public.ecr.aws/d3j8x8q7/swe-bench-202605`
(0.8–1.6 GB compressed each). ECR Public meters **anonymous** pulls per source
IP, and GitHub-hosted runners share their egress addresses, so a lazy pull fails
like this partway through a run:

```
#6 [main internal] load metadata for public.ecr.aws/d3j8x8q7/swe-bench-202605:...:
#6 ERROR: failed to copy: httpReadSeeker: failed open: unexpected status code
   https://public.ecr.aws/v2/...: 429 Too Many Requests
   - Server message: toomanyrequests: Data limit exceeded
```

The workflow handles this in two steps before Pier starts:

1. `Log in to ECR Public` — if `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
   secrets exist, it runs `aws ecr-public get-login-password` and
   `docker login public.ecr.aws`. Authenticated pulls are metered against your
   AWS account (10 pulls/s per region) instead of the 500 GB/month anonymous
   allowance that is shared with everyone behind the runner's IP. Any AWS
   account works; the IAM user needs only `ecr-public:GetAuthorizationToken`
   and `sts:GetServiceBearerToken`.
2. `Pre-pull task images` — resolves the exact image set for the selected
   `n_tasks`/`sample_seed` through Pier's own `DatasetConfig` (so it matches
   the run), pulls each with exponential backoff (15 s → 120 s, five attempts),
   and stops early with an actionable message if the quota is exhausted or the
   runner's disk fills up (see
   [Resource limits](#resource-limits-why-concurrency-defaults-to-1)).

Without AWS credentials the failure is now a two-minute error instead of a
docker-compose stack trace an hour in. Re-running the job may help on its own,
because a new run usually gets a different egress IP — but the quota is monthly
and the address pool is shared, so credentials are the only reliable fix.

Pre-pulling does not change the scores: images are only fetched earlier.

### Docker Issues

GitHub-hosted `ubuntu-latest` **includes Docker Engine** by default. The
workflow's `Check Docker` step verifies it:

```bash
docker version
docker info
docker run --rm hello-world
```

If that step passes but a task container fails, the cause is usually resources,
image pulls, or the task itself — not the Docker daemon.

### Resource limits (why concurrency defaults to 1)

Every DeepSWE task requests `cpus = 2`, `memory_mb = 8192`, and Pier's default
resource policy is `auto` → enforced as a hard Docker **limit**
(`deploy.resources.limits`). A GitHub-hosted `ubuntu-latest` runner has
**4 vCPU / 16 GB RAM**, so two concurrent trials would reserve 4 vCPU and
16 GB of limits on a 16 GB machine — before Docker, Pier, the harness
process, and the egress-proxy container are counted. Hence `concurrency: 1`
by default (2 vCPU / 8 GB in use, leaving headroom).

Disk stays bounded: Pier stops each environment with
`down --rmi all --volumes`, which removes the trial's container and the child
image it built, so per-trial layers do not accumulate. The prebuilt task images
themselves are not compose service images and therefore stay cached for the rest
of the job — but tasks' images share most of their layers, so Docker dedups
them. Measured compressed layer unions: 0.9 GB for 1 task, 1.0 GB for 2,
3.2 GB for the 10-task sample, 7 GB for 25 (a single image alone is ~0.8 GB).
Unpacked, expect roughly 2×. Against the ~14 GB a GitHub-hosted runner has, a
handful of tasks is fine and a large run is not; the `Pre-pull task images` step
checks free space as it goes and fails with an actionable message instead of
letting a trial die on `no space left on device`.

Do not "fix" a memory error with `--ignore` resource modes: Pier warns that
running a task outside its declared resources can disqualify the result. If you
need more parallelism, use a larger or self-hosted runner instead.

### Wall-clock budget (how the workflow fits the 6 h cap)

A GitHub-hosted job cannot exceed **360 minutes**, while DeepSWE declares
`agent.timeout_sec = 10800` (3 h) + verifier and environment-build timeouts of
1800 s each — 240 min of worst case **per task**.

The `prepare` job therefore derives both values from your inputs and passes
`--timeout-multiplier` to Pier (which scales the agent, verifier and build
timeouts together):

- `agent budget = min(task_timeout_minutes, cap-driven ceiling)`
- `job timeout = rounds × (scaled per-task worst case + per-task setup) + overhead`,
  capped at 360

What that means in practice (`task_timeout_minutes: 180`, i.e. DeepSWE's own
budget):

| n_tasks | concurrency | derived agent budget | job timeout | all tasks run? |
| ------- | ----------- | -------------------- | ----------- | -------------- |
| 1       | 1           | 180 min              | 263 min     | yes            |
| 2       | 1           | 123 min              | 359 min     | yes            |
| 10      | 1           | 20 min               | 359 min     | yes            |
| 10      | 2           | 46 min               | 360 min     | yes (if the runner can host 2×8 GB) |
| 25      | 1           | 9 min (floor)        | 360 min     | ~17 of 25     |
| 0 (113) | 1           | 9 min (floor)        | 360 min     | ~17 of 113    |

(8 min per task is reserved outside any Pier timeout for building the agent
child image, starting the container and unpacking whatever the pre-pull step
did not download. Pier deletes each trial's image afterwards, so this cost is
paid per task even though the base images stay cached — see
[Resource limits](#resource-limits-why-concurrency-defaults-to-1).)

The derived budget is echoed to the run summary and a `::warning::` is emitted
whenever it differs from what you asked for.

Trade-off to be aware of: shortening the budget makes runs fit, not scores
meaningful. A 24-minute agent has far less chance on a long-horizon task than
the intended 3 hours.

- Want a **real score**? Run 1–2 tasks at the full 180 min.
- Want **coverage**? Raise `concurrency` on a larger/self-hosted runner (with
  4 vCPU / 16 GB, `concurrency: 1` is the safe maximum), or split the suite
  across runs with `--include-task-name '<glob>'` (add
  `-i '<glob>'` to the `args` array in the workflow).
- `Upload benchmark results` uses `if: always()`, so whatever finished is kept
  even if the job hits the cap; the summary lists only trials with a
  `result.json`.

### Adapter import fails

Pier resolves `--agent-import-path` with `importlib.import_module()`, and a
console script's `sys.path` does not include the current directory. That is why
the workflow copies the adapter next to the job and runs it with `PYTHONPATH=.`.

### Every task fails with a network error

DeepSWE tasks are air-gapped (`[agent] network_mode = "no-network"`). If the
agent returns an empty `network_allowlist()`, Pier applies `network_mode: none`
and the LLM call cannot leave the container. Symptom: `headless --json` ends
with `Error: network error: ...` on the first round. Fix: make sure
`OPENAI_BASE_URL` points at the endpoint you actually call — the adapter allows
exactly that host through Pier's egress proxy.

### A task scores 0 even though the code looked correct

The verifier grades `git diff <base_commit> HEAD`, which ignores working-tree
changes. The adapter commits leftover work after each run, but confirm the
*trial's* `git status` was clean — if `commit_remaining=false` was passed, an
agent that edited files without committing produces an empty patch and fails
every test.

### `externally-managed-environment` from pip

Modern Debian/Ubuntu images refuse `pip install` into the system environment
(PEP 668). The adapter installs with `--target /opt/python-agent-harness` and
prepends that directory via `PYTHONPATH`, which is unaffected.

### API Rate Limits

- Lower `concurrency`
- Use a different model endpoint
- Re-run the job (`pier job resume`) once the limit clears

### `429 Too Many Requests` while pulling task images

That is the ECR Public quota, not your model API key — see
[ECR Public throttles the task image pulls](#ecr-public-throttles-the-task-image-pulls-429).
The workflow now pulls the images up front, retries, and reports which image
failed; add the AWS secrets so the pulls are authenticated.

## Advanced Configuration

### Model Switching

Set the `model` workflow input to a different model:

```
deepseek-flash         (default — adapter's DEFAULT_MODEL)
deepseek-chat          (same endpoint, different model)
gpt-5-mini             (requires OPENAI_BASE_URL=https://api.openai.com/v1)
```

The base URL defaults to DeepSeek; override it with the `OPENAI_BASE_URL`
secret when switching providers.

### Where the benchmark runs

**GitHub Actions only.** The workflow is the single supported way to produce a
score; nothing needs to be installed or executed on a local machine. The
workflow checks out both repositories, builds the harness wheel, runs Pier with
`--env docker`, and publishes the results as artifacts.

Reading an artifact locally is fine and does not run any task — it is just
files:

```bash
unzip deepswe-results-<run_id>.zip -d run-<run_id>
grep -r . run-<run_id>/pier-results/*/*/verifier/reward.txt \
  | awk -F: '{n++; s+=$NF} END {printf "pass rate: %d/%d = %.0f%%\n", s, n, 100*s/n}'
pier view run-<run_id>/pier-results/    # optional web UI, read-only
```

`pier view` serves the collected trajectories/scores; it does not start
containers or call a model.

## References

- [DeepSWE Repository](https://github.com/datacurve-ai/deep-swe)
- [Pier Documentation](https://pypi.org/project/datacurve-pier/)
- [Harbor Task Format](https://www.harborframework.com/docs/tasks)
- [python-agent-harness](https://github.com/beacoder/python-agent-harness)
