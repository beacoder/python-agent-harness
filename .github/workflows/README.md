# GitHub Actions Workflows

## DeepSWE Benchmark

The `deepswe-benchmark.yml` workflow runs the python-agent-harness on the
DeepSWE v1.1 benchmark entirely in the cloud — nothing is installed or executed
locally.

### Setup

1. Add GitHub Secrets:
   - `OPENAI_API_KEY`: model API key (required)
   - `OPENAI_BASE_URL`: optional, defaults to `https://api.deepseek.com/v1`
   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: optional, strongly
     recommended. They are used **only** to authenticate to ECR Public for the
     DeepSWE task images (no other AWS call is made). Anonymous ECR Public
     pulls are throttled per source IP, and GitHub runners share theirs, so
     without these the run can fail with `429 ... Data limit exceeded`.

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

### Task images and the ECR Public quota

DeepSWE's prebuilt task images live in `public.ecr.aws/d3j8x8q7/swe-bench-202605`.
ECR Public meters **anonymous** pulls per source IP (500 GB/month) and a
GitHub-hosted runner's egress address is shared with many other jobs, so the
allowance is frequently exhausted and an unauthenticated run dies with:

```
failed to resolve source metadata for public.ecr.aws/...: 429 Too Many Requests
- Server message: toomanyrequests: Data limit exceeded
```

The workflow therefore pulls every image the run needs *before* Pier starts
(`Pre-pull task images`, resolved through Pier's own sampler), retries each
pull with exponential backoff, and stops early with an actionable error. With
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` set it logs in to ECR Public
first, so the pulls are metered against your AWS account instead of the
runner's shared IP. Any AWS account works; the IAM user needs only
`ecr-public:GetAuthorizationToken` and `sts:GetServiceBearerToken`.

Without those secrets, re-running the job may help on its own — a new run
usually gets a different egress IP — but the quota is monthly and the address
pool is shared, so credentials are the only reliable fix.

### Artifacts

- `deepswe-results-{run_id}`: `pier-results/` (per-task `result.json`, verifier
  scores, agent trajectories) plus the harness wheel; retained 30 days

See [../../benchmarks/deepswe-benchmark.md](../../benchmarks/deepswe-benchmark.md)
for full documentation, including where the score lives and the pass-rate
one-liner.
