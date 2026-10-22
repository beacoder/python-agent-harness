# GitHub Actions Workflows

## DeepSWE Benchmark

The `deepswe-benchmark.yml` workflow enables running the python-agent-harness on the DeepSWE v1.1 benchmark without local setup.

### Setup

1. Add GitHub Secrets:
   - `OPENAI_API_KEY`: Your ModelScope API key (required)
   - `OPENAI_BASE_URL`: Optional, defaults to `https://api-inference.modelscope.cn/v1`

2. Navigate to **Actions** → **DeepSWE Benchmark** → **Run workflow**

3. Configure inputs and run

### Artifacts

- `deepswe-results-{run_id}`: Task results, patches, verifier outputs
- `deepswe-logs-{run_id}`: Detailed execution logs

See [../../benchmarks/deepswe-benchmark.md](../../benchmarks/deepswe-benchmark.md) for full documentation.
