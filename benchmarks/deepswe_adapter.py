"""DeepSWE/Pier adapter for python-agent-harness.

Integrates python-agent-harness with the DeepSWE v1.1 benchmark through
Pier.  The harness runs inside the task container in ``headless --json``
mode; the resulting JSON event stream is teed into the trial's agent log
directory, which Pier mounts at ``/logs/agent``.

Usage (the module must be importable by Pier's process, so put the
directory holding this file on ``PYTHONPATH``):

    PYTHONPATH=. pier run -p deep-swe/tasks \
        --agent-import-path deepswe_adapter:PythonAgentHarness \
        --model deepseek-flash \
        --ak wheel=/abs/path/to/python_agent_harness-1.5.6.4-py3-none-any.whl

Passing ``wheel`` benchmarks the checkout that produced the wheel; without
it the adapter installs the released package from PyPI.
"""

from __future__ import annotations

import base64
import contextlib
import json
import re
import shlex
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pier.agents.installed.base import BaseInstalledAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist

# DeepSeek's OpenAI-compatible endpoint; override with OPENAI_BASE_URL.
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-flash"
# Passed through to the API as-is (the provider decides what it honours).
DEFAULT_REASONING_EFFORT = "max"

# The harness is installed into this directory (at image build time for the
# dependencies, at run time for the wheel) and reached through PYTHONPATH.
# Deliberately not a --user install: the task images run as root with an
# externally-managed pip, and a fixed --target path keeps the interpreter
# that runs the agent and the one that installed it in agreement.
DEPS_DIR = "/opt/python-agent-harness"

# Where the harness config lives inside the task container. A fixed path
# referenced through PYTHON_AGENT_HARNESS_CONFIG (which the harness resolves
# before ~/.config) rather than $HOME/.config: the harness's default location
# follows XDG_CONFIG_HOME / $HOME, neither of which the task images guarantee
# for the user we write as.
CONFIG_DIR = "/tmp/python-agent-harness"
CONFIG_PATH = f"{CONFIG_DIR}/config.json"

# The trial's agent log directory is bind-mounted at /logs/agent.
AGENT_LOG_DIR = "/logs/agent"
TRAJECTORY_NAME = "python-agent-harness.txt"

# Harness core dependencies (mirrors pyproject.toml); installed up front so
# the wheel can later be dropped in with --no-deps.
RUNTIME_DEPS = "'rich>=13.0' 'httpx>=0.27' 'certifi>=2024.0' 'prompt_toolkit>=3.0'"


def _probe_python() -> str:
    """Shell snippet setting ``$pyname`` to a usable Python interpreter.

    The task images are polyglot, so neither ``python`` nor ``python3`` can be
    assumed to be present (or to be the same interpreter pip installed into).
    """
    return (
        'pyname=""; for cand in python3 python; do '
        'command -v "$cand" >/dev/null 2>&1 && pyname="$cand" && break; done; '
        'if [ -z "$pyname" ]; then '
        "echo 'no python interpreter found' >&2; exit 1; fi; "
    )


def _pip_install(target: str, requirement: str) -> str:
    """A pip invocation that works on polyglot images and offline alike.

    - ``--target`` installs into a plain directory, which sidesteps the
      externally-managed-environment error of modern Debian/Ubuntu pythons
      (PEP 668) and the ambiguity of ``--user`` site paths.
    - ``ensurepip`` is the stdlib fallback for images whose python ships
      without pip (common in Go/Rust/TypeScript task images).
    """
    return (
        _probe_python() + '"$pyname" -m pip --version >/dev/null 2>&1 || '
        '"$pyname" -m ensurepip --upgrade >/dev/null 2>&1 || true; '
        '"$pyname" -m pip install --no-input --disable-pip-version-check '
        f"--target {target} --upgrade {requirement}"
    )


def _host_of(url: str | None) -> str | None:
    """The bare hostname of URL, or None when nothing usable is left.

    Accepts a full URL and a scheme-less ``host[:port]/path`` alike, since
    either form is a plausible ``OPENAI_BASE_URL`` value.  Paths, ports and
    schemes are stripped: Pier's allowlist wants a bare domain.
    """
    if not url:
        return None
    raw = url.strip()
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    host = parts.hostname or ""
    if not host:
        # No scheme: urlsplit left everything in the path component.
        host = parts.path.split("/", 1)[0]
    host = host.split(":", 1)[0].strip().lower().rstrip(".")
    if not host or any(char in host for char in "/:*"):
        return None
    return host


class PythonAgentHarness(BaseInstalledAgent):
    """Custom Pier agent that drives python-agent-harness headless mode."""

    SUPPORTS_ATIF: bool = False

    def __init__(
        self,
        logs_dir: Path | str,
        model_kwargs: dict[str, Any] | None = None,
        wheel: str | None = None,
        max_rounds: int | str | None = None,
        commit_remaining: bool | str = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # logs_dir is the host-side trial agent dir, mounted at /logs/agent.
        # Coerced to Path: BaseInstalledAgent joins it with "/", so a caller
        # passing a string (e.g. a hand-written --config) would raise.
        super().__init__(Path(logs_dir), *args, **kwargs)
        self._model_kwargs = model_kwargs or {}
        self._wheel = Path(wheel).expanduser() if wheel else None
        self._max_rounds = int(max_rounds) if max_rounds not in (None, "") else None
        self._commit_remaining = _as_bool(commit_remaining, default=True)

    @staticmethod
    def name() -> str:
        return "python-agent-harness"

    def version(self) -> str | None:
        if self._version:
            return self._version
        if self._wheel is not None:
            # Wheel filename: python_agent_harness-<version>-py3-none-any.whl
            match = re.search(r"-(\d[^-]*)-.+\.whl$", self._wheel.name)
            if match:
                return match.group(1)
        try:
            return metadata.version("python-agent-harness")
        except metadata.PackageNotFoundError:
            return None

    @property
    def _base_url(self) -> str:
        raw = (self._get_env("OPENAI_BASE_URL") or "").strip()
        return raw or DEFAULT_BASE_URL

    def network_allowlist(self) -> NetworkAllowlist:
        """Allow the LLM endpoint through Pier's egress proxy.

        Every DeepSWE task declares ``[agent] network_mode = "no-network"``,
        which Pier turns into ``allow_internet = False``.  With a non-empty
        allowlist Pier instead attaches a filtered egress proxy and injects
        HTTP(S)_PROXY into the agent process; with an empty one the container
        gets ``network_mode: none`` and every LLM request fails.  Without this
        override the adapter therefore cannot run a single task.
        """
        host = _host_of(self._base_url)
        return NetworkAllowlist(domains=[host] if host else [])

    def install_spec(self) -> AgentInstallSpec:
        """Put the runtime dependencies (or the released package) in place.

        Runs at image build time, where the build has network access.  When a
        local wheel is supplied it is uploaded and unpacked in ``setup()``,
        offline, over these dependencies.
        """
        requirement = RUNTIME_DEPS if self._wheel is not None else "python-agent-harness"
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self.version(),
            steps=[
                InstallStep(
                    user="root",
                    run=_pip_install(DEPS_DIR, requirement),
                )
            ],
            verification_command=self.get_version_command(),
        )

    def get_version_command(self) -> str | None:
        # Runs with the same interpreter and PYTHONPATH the agent uses, so it
        # reports the version actually installed in the container.
        return (
            _probe_python()
            + f"PYTHONPATH={DEPS_DIR} "
            + '"$pyname" -c "from importlib.metadata import version; '
            + "print(version('python-agent-harness'))\""
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install the agent, then write its LLM config."""
        await super().setup(environment)

        if self._wheel is not None:
            if not self._wheel.is_file():
                raise FileNotFoundError(f"harness wheel not found: {self._wheel}")
            remote = f"/tmp/{self._wheel.name}"
            await environment.upload_file(self._wheel, remote)
            # Offline (--no-index --no-deps) over the dependency tree the
            # build installed.  As root: DEPS_DIR is root-owned and the
            # task's agent user is unset, so it may not be writable for an
            # unprivileged exec.
            await self.exec_as_root(
                environment,
                command=_pip_install(DEPS_DIR, f"--no-index --no-deps {shlex.quote(remote)}"),
            )

        await self._write_config(environment)
        await self._preflight(environment)

    async def _preflight(self, environment: BaseEnvironment) -> None:
        """Record the interpreter/harness build in the trial's agent dir.

        The task images are polyglot and the harness needs Python >= 3.11; if a
        future image ships something older, the run fails with a SyntaxError
        deep in a traceback. Logging both versions first makes that diagnosable
        from the trial log instead. Best-effort: never fails the task.
        """
        with contextlib.suppress(BaseException):
            await self.exec_as_agent(
                environment,
                command=(
                    _probe_python()
                    + '{ "$pyname" -V; '
                    + f'PYTHONPATH={DEPS_DIR} "$pyname" -c '
                    + '"import python_agent_harness as p; '
                    + "print('harness', p.__file__)\"; "
                    + f"}} > {AGENT_LOG_DIR}/preflight.txt 2>&1 || true"
                ),
            )

    async def _write_config(self, environment: BaseEnvironment) -> None:
        """Write the harness config to CONFIG_PATH in the container.

        Referenced through PYTHON_AGENT_HARNESS_CONFIG by ``run`` (see
        CONFIG_PATH for why it is not $HOME/.config).  The JSON is
        base64-encoded so no quoting (or a stray quote inside a key or URL) can
        break the shell command.

        The container's own OPENAI_* environment variables are deliberately not
        relied on: the harness lets env vars override the file, so passing the
        key through the file (and only the file) keeps one source of truth.
        """
        config: dict[str, Any] = {
            "llm": {
                "base_url": self._base_url,
                "api_key": self._get_env("OPENAI_API_KEY"),
                "model": self.model_name or DEFAULT_MODEL,
                "reasoning_effort": DEFAULT_REASONING_EFFORT,
                "temperature": 0.0,
                "timeout": 600.0,
                "stream": False,
            }
        }
        config["llm"].update(self._model_kwargs)

        payload = base64.b64encode(json.dumps(config).encode("utf-8")).decode("ascii")
        await self.exec_as_agent(
            environment,
            command=(f"mkdir -p {CONFIG_DIR} && printf %s {payload} | base64 -d > {CONFIG_PATH}"),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run one task through ``python-agent-harness headless --json``."""
        flags = ["--json"]
        if self._max_rounds:
            flags += ["--max-rounds", str(self._max_rounds)]

        # ``-m`` (not the console script) so the --target install is found
        # via PYTHONPATH, through the same probed interpreter that installed
        # it.  ``--`` ends option parsing so an instruction starting with
        # "-" is still the positional prompt.
        cmd = (
            _probe_python()
            + f"PYTHONPATH={DEPS_DIR} PYTHON_AGENT_HARNESS_CONFIG={CONFIG_PATH} "
            + '"$pyname" -m python_agent_harness headless '
            + f"{' '.join(flags)} -- {shlex.quote(instruction)} "
            + f"</dev/null 2>&1 | tee {AGENT_LOG_DIR}/{TRAJECTORY_NAME}"
        )

        try:
            await self.exec_as_agent(environment, command=cmd)
        finally:
            # A run that ends badly -- round budget exhausted, wall-clock
            # timeout, or a Pier-side cancellation -- still leaves real work
            # in the tree, and the collected patch is what gets graded.
            # Committing here (suppressing everything, so the original error
            # or cancellation is what propagates) salvages that partial work
            # instead of reporting an empty patch.
            if self._commit_remaining:
                with contextlib.suppress(BaseException):
                    await self._commit_work(environment)

    async def _commit_work(self, environment: BaseEnvironment) -> None:
        """Commit whatever the agent left uncommitted, so it is graded.

        Every DeepSWE task collects the solution with
        ``git diff <base> HEAD`` (see ``[[verifier.collect]]`` in task.toml),
        which ignores working-tree changes.  Tasks do end their instruction
        with "commit everything when you are done", but a run that fixed the
        code and forgot to commit would score 0 for a reason unrelated to the
        task.  Committing here makes the collected patch reflect the actual
        work; the held-out tests still decide pass/fail.

        Best-effort: on a non-git workdir, or when the agent already committed,
        it is a no-op.  ``-c user.*`` avoids depending on a configured git
        identity; the task images disable commit hooks
        (``core.hooksPath /dev/null``).
        """
        # ``safe.directory=*``: the task image's workdir may be owned by a UID
        # other than the one running git (the same reason each task.toml's
        # collect hook sets it before diffing), which would otherwise make
        # every git call fail with "dubious ownership" and be swallowed below.
        # The value is single-quoted: an unquoted ``*`` is a bash glob and a
        # stray ``safe.directory=<something>`` file in the workdir would
        # silently replace the value.
        git = "git -c 'safe.directory=*' -c user.email=pier@pier.local -c user.name=pier"
        await self.exec_as_root(
            environment,
            command=(
                "git -c 'safe.directory=*' rev-parse --git-dir >/dev/null 2>&1 "
                "|| exit 0; "
                f"{git} add -A >/dev/null 2>&1 || true; "
                f"if ! {git} diff --cached --quiet >/dev/null 2>&1; then "
                f"{git} commit -q -m 'harness: commit remaining work' "
                ">/dev/null 2>&1 || true; fi; "
                "exit 0"
            ),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Read the JSON event stream back from the (host-side) agent log."""
        trajectory_path = self.logs_dir / TRAJECTORY_NAME
        try:
            log_text = trajectory_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            self.logger.debug(f"trajectory not found: {trajectory_path}")
            return

        result = self._last_result_event(log_text)
        if result is None:
            return

        usage = result.get("usage") or {}
        context.n_input_tokens = _as_int(usage.get("input"))
        context.n_output_tokens = _as_int(usage.get("output"))
        context.n_agent_steps = _as_int(usage.get("rounds"))
        context.metadata = {
            "trajectory": str(trajectory_path),
            "answer": result.get("answer"),
            "errors": result.get("error_messages") or [],
            "cancelled": bool(result.get("cancelled")),
        }

    @staticmethod
    def _last_result_event(log_text: str) -> dict[str, Any] | None:
        """The terminal ``result`` event of the JSON-lines stream, if any."""
        for line in reversed(log_text.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                return event
        return None


def _as_int(value: Any) -> int:
    """Coerce a usage counter to int, falling back to 0."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_bool(value: Any, *, default: bool) -> bool:
    """Coerce a `--ak` kwarg (bool, JSON false, or a string) to bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("", "0", "false", "no", "off")
