#!/usr/bin/env python3
"""Run python-agent-harness on SWE-bench Lite instances.

For each instance:
  1. Start the task's Docker container (image from the dataset's `image` field)
  2. Make sure the container has Python >= 3.11 (the images ship 3.9/3.10, so a
     standalone CPython is unpacked when needed)
  3. Install the harness wheel plus its runtime deps into /opt/python-agent-harness
  4. Write the harness config (API key, base URL, model)
  5. Run `python -m python_agent_harness headless --json '<problem_statement>'`
  6. Collect `git diff` as the model patch
  7. Append to predictions.jsonl

Usage:
    python swebench_agent.py --wheel /path/to/harness.whl --model deepseek-flash \
        --predictions predictions.jsonl --n-tasks 1 --concurrency 1 --timeout-minutes 30
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import pathlib
import shlex
import subprocess
import sys
import threading
import time
from typing import Any

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-flash"
DEPS_DIR = "/opt/python-agent-harness"
CONFIG_DIR = "/tmp/python-agent-harness"
CONFIG_PATH = f"{CONFIG_DIR}/config.json"
RUNTIME_DEPS = ("rich", "httpx", "certifi", "prompt_toolkit")
AGENT_SLEEP_SECONDS = 600

# The swe-bench task images ship Python 3.9 (conda env ``testbed``) or 3.10
# (/usr/bin/python3); the harness needs >= 3.11 (it imports ``typing.TypeGuard``
# and other 3.10+ APIs), so an image interpreter can only be used when it is
# new enough.  Otherwise a standalone CPython is unpacked into the container:
# one shared download on the host, copied into each container (~10 s, no
# network round-trip per instance).
PY_STANDALONE_TAG = "20250115"
PY_STANDALONE_NAME = (
    f"cpython-3.12.8+{PY_STANDALONE_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"
)
PY_STANDALONE_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PY_STANDALONE_TAG}/{PY_STANDALONE_NAME}"
)
PY_STANDALONE_DIR = "/opt/py312"

_py_detect = (
    "for cand in /opt/miniconda3/envs/testbed/bin/python "
    "python3.13 python3.12 python3.11 python3; do "
    'p=$(command -v "$cand" 2>/dev/null) || continue; '
    '"$p" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" '
    "2>/dev/null || continue; "
    'echo "$p"; exit 0; '
    "done; exit 3"
)

_predictions_lock = threading.Lock()


def ensure_python_tarball(cache_dir: pathlib.Path) -> pathlib.Path | None:
    """Download the standalone CPython tarball once, into the run cache."""
    tarball = cache_dir / PY_STANDALONE_NAME
    if tarball.is_file() and tarball.stat().st_size > 1_000_000:
        return tarball
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = tarball.with_suffix(tarball.suffix + ".part")
    try:
        import shutil
        import urllib.request

        print(f"Downloading a standalone CPython 3.12 (one-time): {PY_STANDALONE_URL}")
        with (
            urllib.request.urlopen(PY_STANDALONE_URL, timeout=180) as resp,
            open(partial, "wb") as fh,
        ):
            shutil.copyfileobj(resp, fh)
        partial.replace(tarball)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        print(f"warning: could not download {PY_STANDALONE_URL}: {exc}", file=sys.stderr)
        return None
    return tarball


def load_dataset(n_tasks: int, sample_seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    instances = list(ds)
    if n_tasks == 0:
        return instances
    import random

    rng = random.Random(sample_seed)
    rng.shuffle(instances)
    return instances[: min(n_tasks, len(instances))]


def get_image_name(instance: dict[str, Any]) -> str:
    image = instance.get("image", "")
    if image:
        return image
    instance_id = instance["instance_id"]
    repo = instance_id.rsplit("-", 1)[0].replace("/", "_")
    repo_escaped = repo.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{repo_escaped}.{instance_id}:latest"


def docker(*args: str, check: bool = False, **kwargs: Any) -> subprocess.CompletedProcess:
    proc = subprocess.run(["docker", *args], **kwargs)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, ["docker", *args])
    return proc


def resolve_interpreter(container_name: str, instance_id: str, tarball: pathlib.Path | None) -> str:
    """Return a container interpreter that is >= Python 3.11."""
    proc = docker("exec", container_name, "bash", "-c", _py_detect, capture_output=True, text=True)
    py = proc.stdout.strip()
    if proc.returncode == 0 and py:
        return py
    if tarball is None:
        raise RuntimeError(
            "the image has no Python >= 3.11 and the standalone interpreter was not downloaded"
        )
    print(f"[{instance_id}] Image has no Python >= 3.11; unpacking a standalone CPython")
    docker("cp", str(tarball), f"{container_name}:/tmp/{PY_STANDALONE_NAME}", check=True)
    docker(
        "exec",
        container_name,
        "bash",
        "-c",
        f"mkdir -p {PY_STANDALONE_DIR} && "
        f"tar -xzf /tmp/{PY_STANDALONE_NAME} -C {PY_STANDALONE_DIR} --strip-components=1 && "
        f"{PY_STANDALONE_DIR}/bin/python3 -c "
        "'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'",
        check=True,
    )
    return f"{PY_STANDALONE_DIR}/bin/python3"


def _normalize_patch(raw: str) -> str:
    """Normalize a `git diff` dump so GNU patch accepts it.

    Trailing blank lines are dropped, but the final newline must be kept:
    `patch` reports "malformed patch" / "unexpectedly ends in middle of line"
    when the payload does not end with a newline.
    """
    body = raw.replace("\r\n", "\n").rstrip("\n")
    if not body.strip():
        return ""
    return body + "\n"


def run_instance(
    instance: dict[str, Any],
    wheel_path: str,
    model: str,
    api_key: str,
    base_url: str,
    max_rounds: int,
    timeout_minutes: int,
    predictions_path: pathlib.Path,
    python_tarball: pathlib.Path | None = None,
) -> dict[str, Any]:
    instance_id = instance["instance_id"]
    image = get_image_name(instance)
    problem_statement = instance["problem_statement"]
    base_commit = instance["base_commit"]

    container_name = f"swebench-agent-{instance_id}-{int(time.time())}"
    result: dict[str, Any] = {
        "instance_id": instance_id,
        "model_name_or_path": model,
        "model_patch": "",
    }

    try:
        print(f"[{instance_id}] Starting container from {image}")
        docker(
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "-e",
            f"OPENAI_API_KEY={api_key}",
            "-e",
            f"OPENAI_BASE_URL={base_url}",
            "-e",
            f"OPENAI_MODEL={model}",
            image,
            "sleep",
            str(timeout_minutes * 60 + AGENT_SLEEP_SECONDS),
            check=True,
            capture_output=True,
        )

        # The swe-bench task images are squashed into one "SWE-bench" commit
        # whose file modes differ from base_commit, so `git diff <base_commit>`
        # reports a mode change for every file in the repo (thousands of junk
        # hunks).  Diff against the image's own checkout instead: that exposes
        # exactly the changes the agent made.
        start_rev = base_commit
        rev_proc = docker(
            "exec",
            container_name,
            "bash",
            "-c",
            "cd /testbed && git rev-parse HEAD",
            capture_output=True,
            text=True,
        )
        if rev_proc.returncode == 0 and rev_proc.stdout.strip():
            start_rev = rev_proc.stdout.strip()
        else:
            print(
                f"[{instance_id}] warning: could not read the image HEAD; "
                f"diffing against {base_commit}",
                file=sys.stderr,
            )

        print(f"[{instance_id}] Resolving a Python >= 3.11 in the container")
        py = resolve_interpreter(container_name, instance_id, python_tarball)

        print(f"[{instance_id}] Installing harness wheel")
        # pip validates the wheel *filename*, so keep the original name instead
        # of copying the artifact to a generic harness.whl.
        wheel_name = pathlib.Path(wheel_path).name
        wheel_dest = f"/tmp/{wheel_name}"
        docker("cp", wheel_path, f"{container_name}:{wheel_dest}", check=True)
        deps = " ".join(shlex.quote(dep) for dep in RUNTIME_DEPS)
        install_cmd = (
            f"{shlex.quote(py)} -m pip install --no-input --disable-pip-version-check "
            f"--target {DEPS_DIR} {deps} {shlex.quote(wheel_dest)}"
        )
        install_proc = docker(
            "exec", container_name, "bash", "-c", install_cmd, capture_output=True, text=True
        )
        if install_proc.returncode != 0:
            raise RuntimeError(
                "could not install the harness into the container: "
                f"{(install_proc.stderr or install_proc.stdout).strip()[-800:]}"
            )

        print(f"[{instance_id}] Writing harness config")
        # LLM settings in one `llm` object; the config file parsing follows
        # code defaults < config file < environment (env wins here too, which
        # is why the container gets OPENAI_* as well).
        config = {
            "llm": {
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "temperature": 0.0,
                "timeout": 600.0,
                "stream": False,
            }
        }
        config_b64 = base64.b64encode(json.dumps(config).encode()).decode()
        docker(
            "exec",
            container_name,
            "bash",
            "-c",
            f"mkdir -p {CONFIG_DIR} && printf %s {config_b64} | base64 -d > {CONFIG_PATH}",
            check=True,
        )

        # /testbed is the repo checked out at the instance's commit; it is the
        # authoritative workdir, so the agent sees the real project layout.
        print(f"[{instance_id}] Running harness headless (timeout {timeout_minutes} min)")
        flags = ["--json", f"--timeout {timeout_minutes * 60.0:g}"]
        if max_rounds > 0:
            flags.append(f"--max-rounds {max_rounds}")

        # ``--`` ends option parsing so a problem statement that starts with
        # "-" is still the positional prompt.
        cmd = (
            f"cd /testbed && PYTHONPATH={DEPS_DIR} PYTHON_AGENT_HARNESS_CONFIG={CONFIG_PATH} "
            + f"{shlex.quote(py)} -m python_agent_harness headless "
            + f"--model {shlex.quote(model)} {' '.join(flags)} -- {shlex.quote(problem_statement)}"
        )

        timeout_sec = timeout_minutes * 60 + 60
        try:
            proc = docker(
                "exec",
                container_name,
                "bash",
                "-c",
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            print(f"[{instance_id}] Harness exited with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print(f"[{instance_id}] Harness timed out after {timeout_minutes} min")
            proc = None

        # Preserve the harness output on the host: without it a failed run
        # leaves nothing to diagnose beyond the exit code.
        log_dir = predictions_path.parent / "agent-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{instance_id}.log"
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            if proc is not None:
                logf.write(f"exit code: {proc.returncode}\n")
                logf.write("==== stdout ====\n")
                logf.write(proc.stdout or "")
                logf.write("==== stderr ====\n")
                logf.write(proc.stderr or "")
            else:
                logf.write(f"timed out after {timeout_minutes} min\n")

        print(f"[{instance_id}] Collecting git diff")
        # The swebench grader applies model_patch onto a pristine checkout at
        # base_commit and only grades committed work via the patch it collects.
        # Stage any leftover changes and commit them with an inline identity so
        # the patch reflects the real work; grading still folds in the
        # gold test_patch, which the model must not touch.
        docker(
            "exec",
            container_name,
            "bash",
            "-c",
            "cd /testbed && git add -A >/dev/null 2>&1 || true; "
            "if ! git diff --cached --quiet >/dev/null 2>&1; then "
            "git -c user.email=agent@swebench.local -c user.name=SWE-bench-agent "
            "commit -q -m 'harness: commit remaining work' >/dev/null 2>&1 || true; fi",
            check=False,
        )
        diff_proc = docker(
            "exec",
            container_name,
            "bash",
            "-c",
            f"cd /testbed && git diff {start_rev} --ignore-submodules",
            capture_output=True,
            text=True,
        )
        patch = _normalize_patch(diff_proc.stdout)
        if not patch:
            diff_proc2 = docker(
                "exec",
                container_name,
                "bash",
                "-c",
                f"cd /testbed && git add -A >/dev/null 2>&1; "
                f"git diff {start_rev} --cached --ignore-submodules",
                capture_output=True,
                text=True,
            )
            patch = _normalize_patch(diff_proc2.stdout)

        result["model_patch"] = patch
        if patch:
            print(f"[{instance_id}] Patch collected ({len(patch)} chars)")
        else:
            print(f"[{instance_id}] No patch produced")

    except subprocess.CalledProcessError as e:
        print(f"[{instance_id}] Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[{instance_id}] Unexpected error: {e}", file=sys.stderr)
    finally:
        docker("rm", "-f", container_name, capture_output=True)

    with _predictions_lock, open(predictions_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run harness on SWE-bench Lite")
    parser.add_argument("--wheel", required=True, help="Path to harness wheel")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name on the endpoint")
    parser.add_argument(
        "--predictions", required=True, type=pathlib.Path, help="Output predictions JSONL path"
    )
    parser.add_argument("--n-tasks", type=int, default=1, help="Number of tasks (0=all)")
    parser.add_argument("--sample-seed", type=int, default=0, help="Random seed")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel runs")
    parser.add_argument("--max-rounds", type=int, default=0, help="Max LLM rounds (0=unlimited)")
    parser.add_argument("--timeout-minutes", type=int, default=30, help="Agent timeout per task")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        print("error: OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.touch()

    # Downloaded once and reused for every container that ships an old Python.
    python_tarball = ensure_python_tarball(args.predictions.parent / ".cache")

    instances = load_dataset(args.n_tasks, args.sample_seed)
    print(f"Running {len(instances)} instance(s) with concurrency {args.concurrency}")

    if args.concurrency == 1:
        for inst in instances:
            run_instance(
                inst,
                args.wheel,
                args.model,
                api_key,
                base_url,
                args.max_rounds,
                args.timeout_minutes,
                args.predictions,
                python_tarball,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = []
            for inst in instances:
                futures.append(
                    pool.submit(
                        run_instance,
                        inst,
                        args.wheel,
                        args.model,
                        api_key,
                        base_url,
                        args.max_rounds,
                        args.timeout_minutes,
                        args.predictions,
                        python_tarball,
                    )
                )
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"error: {e}", file=sys.stderr)

    print(f"Predictions written to {args.predictions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
