"""
DeepSWE/Pier adapter for python-agent-harness.

This module provides a custom agent implementation that integrates
python-agent-harness with the DeepSWE benchmark via Pier.

Usage (module must be importable by Pier's process — set PYTHONPATH to
the directory containing this file):
    PYTHONPATH=. pier run -p deep-swe/tasks \
        --agent-import-path deepswe_adapter:PythonAgentHarness \
        --model Qwen/Qwen3.5-35B-A3B
"""

import json
import shlex
from pathlib import Path
from typing import Any

from pier.agents.installed.base import BaseInstalledAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep


class PythonAgentHarness(BaseInstalledAgent):
    """Custom agent using python-agent-harness for DeepSWE benchmark."""

    SUPPORTS_ATIF: bool = False

    def __init__(
        self,
        logs_dir: Path,
        model_kwargs: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(logs_dir, *args, **kwargs)
        self._model_kwargs = model_kwargs or {}

    @staticmethod
    def name() -> str:
        return "python-agent-harness"

    def version(self) -> str | None:
        return "1.5.6"

    def install_spec(self) -> AgentInstallSpec:
        return AgentInstallSpec(
            agent_name=self.name(),
            steps=[
                InstallStep(
                    user="agent",
                    run="pip install python-agent-harness",
                )
            ],
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install agent then configure LLM settings."""
        await super().setup(environment)

        config = {
            "llm": {
                "base_url": self._get_env("OPENAI_BASE_URL")
                or "https://api-inference.modelscope.cn/v1",
                "api_key": self._get_env("OPENAI_API_KEY"),
                "model": self.model_name or "Qwen/Qwen3.5-35B-A3B",
                "reasoning_effort": "high",
                "temperature": 0.0,
                "timeout": 600.0,
                "stream": False,
            }
        }

        if "temperature" in self._model_kwargs:
            config["llm"]["temperature"] = self._model_kwargs["temperature"]

        if "max_tokens" in self._model_kwargs:
            config["llm"]["max_tokens"] = self._model_kwargs["max_tokens"]

        config_json = json.dumps(config)
        await self.exec_as_agent(
            environment,
            command=(
                "mkdir -p ~/.config/python-agent-harness && "
                f"echo '{config_json}' > ~/.config/python-agent-harness/config.json"
            ),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run python-agent-harness headless mode."""
        escaped_instruction = shlex.quote(instruction)

        cmd = (
            f"python-agent-harness headless --json {escaped_instruction} "
            "2>&1 | tee /logs/agent/python-agent-harness.txt"
        )

        await self.exec_as_agent(environment, command=cmd)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Populate context with trajectory and token metrics."""
        trajectory_path = Path("/logs/agent/python-agent-harness.txt")

        try:
            trajectory_content = trajectory_path.read_text()
            trajectory_file = self.logs_dir / "python-agent-harness.log"
            trajectory_file.write_text(trajectory_content)
        except OSError:
            pass

        # Parse the final `result` JSON line from headless --json output
        # to extract token usage; fall back to zeros when unavailable.
        n_input = n_output = n_rounds = 0
        try:
            for line in reversed(trajectory_content.splitlines()):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                event = json.loads(line)
                if event.get("type") == "result":
                    usage = event.get("usage") or {}
                    n_input = int(usage.get("input") or 0)
                    n_output = int(usage.get("output") or 0)
                    n_rounds = int(usage.get("rounds") or 0)
                    break
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass

        context.n_input_tokens = n_input
        context.n_output_tokens = n_output
        context.n_agent_steps = n_rounds
