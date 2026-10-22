# DeepSWE Benchmark Integration

This document explains how to run the [python-agent-harness](https://github.com/beacoder/python-agent-harness) on the [DeepSWE v1.1](https://github.com/datacurve-ai/deep-swe) benchmark using GitHub Actions.

## Overview

DeepSWE is a benchmark for evaluating frontier coding agents on long-horizon software engineering tasks. It uses [Pier](https://pypi.org/project/datacurve-pier/) as the execution framework and provides 113 real-world tasks from various repositories.

## Quick Start

### 1. Configure GitHub Secrets

Add the following secrets to your repository settings:

```
OPENAI_API_KEY = your-api-key-here
OPENAI_BASE_URL = https://api-inference.modelscope.cn/v1  # Optional, defaults to ModelScope
```

For other models (OpenAI, Anthropic, DeepSeek), update accordingly:
- **OpenAI**: `https://api.openai.com/v1`
- **Anthropic**: `https://api.anthropic.com/v1`
- **DeepSeek**: `https://api.deepseek.com/v1`

### 2. Run the Benchmark

Navigate to **Actions** → **DeepSWE Benchmark** → **Run workflow**

Configure the inputs:
- **n_tasks**: Number of tasks (10, 25, 50, or 0 for full suite)
- **model**: Model identifier (e.g., `Qwen/Qwen3.5-35B-A3B`)
- **max_rounds**: Maximum LLM rounds per task (0 = unlimited)
- **timeout_minutes**: Max execution time per task (default: 180)

### 3. View Results

After completion, download the artifacts:
- **deepswe-results-{run_id}**: Contains `pier-jobs/` with task results, patches, and verifier outputs
- **deepswe-logs-{run_id}**: Detailed logs from each task execution

## Architecture

### Custom Agent Adapter

The file [`deepswe_adapter.py`](./deepswe_adapter.py) implements the `BaseAgent` interface required by Pier:

```python
from pier.agents.installed.base import BaseInstalledAgent


class PythonAgentHarness(BaseInstalledAgent):
    async def setup(self, environment):
        # Install python-agent-harness and configure LLM
        pass

    async def run(self, instruction, environment, context):
        # Run headless mode and capture output
        pass
```

### Execution Flow

```
GitHub Actions Runner
    ↓
Checkout python-agent-harness + DeepSWE tasks
    ↓
Install dependencies (uv + pip)
    ↓
pier run -p deep-swe/tasks --agent-import-path ...
    ↓
Docker container per task
    ↓
python-agent-harness headless mode
    ↓
Verifier (grader.py) scores solution
    ↓
Results saved to pier-results/
```

## Task Structure

Each DeepSWE task contains:
- `task.toml`: Metadata, limits, collector hooks
- `instruction.md`: Prompt shown to agent
- `environment/Dockerfile`: Container image
- `tests/grader.py`: Shared grading logic
- `tests/config.json`: Per-task test whitelists

The verifier uses a two-phase process:
1. **Prepare**: Apply agent's patch
2. **Grade**: Run held-out tests, compare against whitelist

Scoring is binary (0/1) based on:
- All fail-to-pass tests must now pass
- No pass-to-pass tests should fail

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
- Use `max_rounds` limit to control token usage
- Monitor API usage in your provider dashboard

## Troubleshooting

### Docker Issues

GitHub-hosted `ubuntu-latest` **includes Docker Engine** by default. If you encounter errors:

1. **Check Docker availability**: The workflow includes a `Check Docker` step that verifies:
   ```bash
   docker version
   docker info
   docker run --rm hello-world
   ```

2. **Resource limits**: DeepSWE tasks may need more CPU/memory than the runner provides
   - Use self-hosted runners with more resources if needed
   - Start with `--n-tasks 1` to test single task

3. **Image pull failures**: Large Docker images may timeout
   - Check network connectivity
   - Pier caches images, so first run takes longer

### Memory Limits

Default runner may struggle with large tasks. Consider:
- Reducing `n_tasks` initially
- Using self-hosted runners with more resources
- Setting `--env docker` explicitly

### API Rate Limits

If hitting rate limits:
- Increase timeout between runs
- Use a different model endpoint
- Implement retry logic in adapter

## Advanced Configuration

### Model Switching

Set the `model` workflow input to a different model:
```
Qwen/Qwen3.5-35B-A3B   (default, ModelScope)
deepseek-chat          (requires OPENAI_BASE_URL=https://api.deepseek.com/v1)
gpt-5-mini             (requires OPENAI_BASE_URL=https://api.openai.com/v1)
```

The base URL defaults to ModelScope; override with the `OPENAI_BASE_URL`
secret when switching providers.

### Local Testing

Test locally before running on GitHub Actions:

```bash
pip install datacurve-pier python-agent-harness

# PYTHONPATH is required: Pier resolves --agent-import-path via
# importlib.import_module(), and the pier console script does not
# include the current directory on sys.path.
PYTHONPATH=. pier run \
  -p deep-swe/tasks \
  --agent-import-path deepswe_adapter:PythonAgentHarness \
  --model Qwen/Qwen3.5-35B-A3B \
  --n-tasks 1 \
  --sample-seed 0 \
  --env docker
```

## References

- [DeepSWE Repository](https://github.com/datacurve-ai/deep-swe)
- [Pier Documentation](https://pypi.org/project/datacurve-pier/)
- [Harbor Task Format](https://www.harborframework.com/docs/tasks)
- [python-agent-harness](https://github.com/beacoder/python-agent-harness)
