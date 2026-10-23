#!/usr/bin/env bash
#
# Run the DeepSWE v1.1 benchmark on this machine.
#
# The GitHub Actions workflow (.github/workflows/deepswe-benchmark.yml) does the
# same thing in the cloud; this script is the local equivalent, kept next to it
# so both paths share the adapter, the image resolver and the summary. Nothing
# here is used by CI, and running this script does not change the workflow.
#
# Requirements: Docker Engine with the compose v2 plugin, Python 3.12+ (Pier
# 0.3.x needs it), git, and network access to public.ecr.aws (the task images).
#
# Quick start:
#
#     export OPENAI_API_KEY=...            # or keep it in the harness config
#     benchmarks/run-deepswe-local.sh                       # 1 task, full budget
#     N_TASKS=2 benchmarks/run-deepswe-local.sh
#     N_TASKS=0 benchmarks/run-deepswe-local.sh             # all 113 tasks
#
# Environment (defaults in parentheses):
#   N_TASKS (1)                 0 = the whole suite
#   MODEL (deepseek-flash)      model name passed to Pier
#   SAMPLE_SEED (0)             fixed so reruns pick the same tasks
#   TASK_TIMEOUT_MINUTES (180)  agent budget per task; DeepSWE declares 180
#   CONCURRENCY (1)             concurrent trials; each asks for 2 vCPU / 8 GB
#   MAX_ROUNDS (0)              0 = unlimited LLM rounds per task
#   OPENAI_API_KEY              falls back to the harness config file
#   OPENAI_BASE_URL             falls back to the harness config file
#   DEEP_SWE_DIR                tasks checkout (<repo>/benchmarks/deep-swe)
#   VENV_DIR                    driver venv (<repo>/.venv-deepswe)
#   JOBS_DIR                    Pier output (<repo>/pier-results)
#   DISK_RESERVE_MB (20480)     free space to leave for the trials themselves
#   SKIP_PULL (0)               1 = do not pre-pull the task images
#   SKIP_BUILD (0)              1 = reuse the newest wheel in dist/
#
set -euo pipefail

script_src=${BASH_SOURCE[0]:-$0}
# A bare name means the script came from $PATH (or a symlink); resolve it so
# script_dir is the directory holding the real file rather than the caller's
# cwd. readlink -f also follows symlinks, which keeps the docker-group re-exec
# below pointing at this file.
if [ "${script_src##*/}" = "$script_src" ]; then
  script_src=$(command -v -- "$script_src" || printf '%s' "$script_src")
fi
script_path=$(readlink -f -- "$script_src" 2>/dev/null || printf '%s' "$script_src")
script_dir=$(cd -- "$(dirname -- "$script_path")" && pwd)
# Absolute, and valid after the `cd` below: the re-exec has to name this file.
script_path=$script_dir/${script_path##*/}
repo_dir=$(cd -- "$script_dir/.." && pwd)
cd "$repo_dir"

n_tasks=${N_TASKS:-1}
model=${MODEL:-deepseek-flash}
sample_seed=${SAMPLE_SEED:-0}
per_task=${TASK_TIMEOUT_MINUTES:-180}
concurrency=${CONCURRENCY:-1}
max_rounds=${MAX_ROUNDS:-0}
deep_swe_dir=${DEEP_SWE_DIR:-$script_dir/deep-swe}
venv=${VENV_DIR:-$repo_dir/.venv-deepswe}
jobs_dir=${JOBS_DIR:-$repo_dir/pier-results}
reserve_mb=${DISK_RESERVE_MB:-20480}
skip_pull=${SKIP_PULL:-0}
skip_build=${SKIP_BUILD:-0}
extra_args=${EXTRA_ARGS:-}
tasks_dir=$deep_swe_dir/tasks

if [ "$concurrency" -lt 1 ]; then concurrency=1; fi

note() { echo "==> $*"; }
die() {
  echo "error: $*" >&2
  exit 1
}

# ---------------------------------------------------------------- preflight

note "Checking Docker"
command -v docker >/dev/null 2>&1 || die "docker not found. Install Docker Engine (see benchmarks/deepswe-benchmark.md)."

# Is USER a member of GROUP according to the group database? `id -nG` only lists
# the groups applied to the current session, which is exactly the stale state
# the re-exec below is there to work around.
in_group() {
  getent group "$1" 2>/dev/null |
    awk -F: -v user="$(id -un)" '{
      n = split($4, members, ",")
      for (i = 1; i <= n; i++) if (members[i] == user) found = 1
    } END { exit !found }'
}

if ! docker info >/dev/null 2>&1; then
  # `usermod -aG docker` only reaches new login sessions, so a shell opened
  # before it still gets "permission denied" on the socket. `sg docker` applies
  # the recorded membership to this one process, which turns a confusing
  # failure into a working run without logging out of the desktop.
  if [ "${DEEPSWE_REEXEC:-0}" != "1" ] && in_group docker && command -v sg >/dev/null 2>&1; then
    note "The docker socket is not reachable in this shell; re-running inside the 'docker' group"
    DEEPSWE_REEXEC=1 exec sg docker -c "$(printf '%q ' "$script_path" "$@")"
  fi
  die "the Docker daemon is not reachable. Start it, or join the docker group and re-login (newgrp docker)."
fi
docker compose version >/dev/null 2>&1 ||
  die "the 'docker compose' v2 plugin is missing; Pier shells out to 'docker compose'."

# Every DeepSWE task declares cpus = 2 and memory_mb = 8192, and Pier applies
# them as hard limits. Enforcement still works on a smaller host, but a
# container that asks for more RAM than the machine has gets OOM-killed
# mid-task, which looks like a harness failure. Warn instead of failing:
# memory is a cap, not a reservation, so a lighter task may still pass.
mem_total_mb=$(awk '/^MemTotal:/ {printf "%d", $2 / 1024}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$mem_total_mb" -gt 0 ] && [ "$mem_total_mb" -lt 12288 ]; then
  echo "warning: this machine has ${mem_total_mb} MiB of RAM but DeepSWE tasks request 8192 MiB each;" >&2
  echo "warning: expect OOM kills. On WSL raise the limit with memory=16GB in %USERPROFILE%\\.wslconfig," >&2
  echo "warning: or pass EXTRA_ARGS='--override-memory-mb <mb>' (changes the declared resources)." >&2
fi
if [ "$concurrency" -gt 1 ] && [ "$mem_total_mb" -gt 0 ]; then
  need_mb=$((concurrency * 8192))
  if [ "$mem_total_mb" -lt "$need_mb" ]; then
    echo "warning: CONCURRENCY=$concurrency asks for ${need_mb} MiB of task memory on a ${mem_total_mb} MiB machine." >&2
  fi
fi

# Pier 0.3.x declares requires-python >= 3.12, so pick an interpreter that
# satisfies it before anything is installed.
if [ ! -x "$venv/bin/python" ]; then
  python_bin=""
  for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' 2>/dev/null; then
      python_bin=$(command -v "$candidate")
      break
    fi
  done
  [ -n "$python_bin" ] || die "no Python 3.12+ found; Pier 0.3.x requires it."
  note "Creating $venv with $("$python_bin" -V 2>&1)"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$venv" --python "$python_bin"
  else
    "$python_bin" -m venv "$venv"
  fi
fi
"$venv/bin/python" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' ||
  die "$venv is not Python 3.12+; remove it and rerun."

if ! "$venv/bin/python" -c 'import pier, python_agent_harness' >/dev/null 2>&1; then
  note "Installing Pier and python-agent-harness into $venv"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$venv/bin/python" -e "$repo_dir" "datacurve-pier>=0.3.1" build
  else
    "$venv/bin/python" -m pip install --quiet --upgrade pip
    "$venv/bin/python" -m pip install --quiet -e "$repo_dir" "datacurve-pier>=0.3.1" build
  fi
fi
# DeepSWE v1.1 needs Pier newer than 0.3.0 (per-task verifier environment).
"$venv/bin/python" -c 'import pier' || die "Pier is not importable from $venv"

if [ ! -d "$tasks_dir" ]; then
  note "Cloning datacurve-ai/deep-swe into $deep_swe_dir"
  git clone --depth 1 https://github.com/datacurve-ai/deep-swe.git "$deep_swe_dir"
fi
[ -d "$tasks_dir" ] || die "no task directory at $tasks_dir"

# The adapter needs a model API key inside the task containers. The harness
# config under ~/.config already holds one per profile, so reuse it instead of
# asking for the key twice; nothing is printed.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  config_file=${PYTHON_AGENT_HARNESS_CONFIG:-$HOME/.config/python-agent-harness/config.json}
  if [ -f "$config_file" ]; then
    eval "$("$venv/bin/python" - "$config_file" "$model" <<'PY'
import json
import shlex
import sys

path, model = sys.argv[1], sys.argv[2]
data = json.load(open(path, encoding="utf-8"))
profiles = data.get("models") or {}
profile = profiles.get(model)
if profile is None:
    profile = next(
        (p for p in profiles.values() if isinstance(p, dict) and p.get("model") == model),
        None,
    )
if profile is None and (data.get("llm") or {}).get("model") == model:
    profile = data["llm"]
if profile:
    if profile.get("api_key"):
        print(f"export OPENAI_API_KEY={shlex.quote(profile['api_key'])}")
    if profile.get("base_url"):
        print(f"export OPENAI_BASE_URL={shlex.quote(profile['base_url'])}")
PY
)"
    if [ -n "${OPENAI_API_KEY:-}" ]; then
      note "Using the API key from $config_file"
    fi
  fi
fi
[ -n "${OPENAI_API_KEY:-}" ] ||
  die "OPENAI_API_KEY is not set and no matching profile exists in the harness config."
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.deepseek.com/v1}

# ---------------------------------------------------------------- wheel

if [ "$skip_build" = "1" ]; then
  wheel=$(ls -1t "$repo_dir"/dist/python_agent_harness-*.whl 2>/dev/null | head -n 1 || true)
  [ -n "$wheel" ] || die "SKIP_BUILD=1 but no wheel in $repo_dir/dist"
  note "Reusing $wheel"
else
  note "Building the harness wheel"
  rm -rf "$repo_dir/dist"
  "$venv/bin/python" -m build --wheel --outdir "$repo_dir/dist" "$repo_dir" >/dev/null
  wheels=("$repo_dir"/dist/python_agent_harness-*.whl)
  [ -f "${wheels[0]}" ] || die "the wheel build produced no file in $repo_dir/dist"
  wheel=${wheels[0]}
fi

# ---------------------------------------------------------------- images

# The task images live in public.ecr.aws, which throttles *anonymous* pulls per
# source IP (429 toomanyrequests: Data limit exceeded). Pulling them up front
# turns a mid-run failure into an early one and lets the retry loop below run:
# Pier treats a failed pull as a failed trial and does not retry it.
# Authenticated pulls are not throttled the same way -- `docker login
# public.ecr.aws` before running this script (see the docs) if you keep hitting
# the limit.
if [ "$skip_pull" != "1" ]; then
  note "Resolving the images for n_tasks=$n_tasks (seed $sample_seed)"
  if ! resolved=$("$venv/bin/python" "$script_dir/deepswe_images.py" \
    --path "$tasks_dir" --n-tasks "$n_tasks" --sample-seed "$sample_seed"); then
    die "could not resolve the task images for n_tasks=$n_tasks"
  fi
  [ -n "$resolved" ] || die "no task image resolved from $tasks_dir"
  mapfile -t rows <<<"$resolved"

  pull() {
    local image="$1" attempt=1 delay=15
    until docker pull --quiet "$image"; do
      if [ "$attempt" -ge 5 ]; then
        echo "warning: docker pull $image failed after $attempt attempts" >&2
        return 1
      fi
      echo "warning: docker pull $image failed (attempt $attempt); retrying in ${delay}s" >&2
      sleep "$delay"
      attempt=$((attempt + 1))
      delay=$((delay * 2))
    done
  }

  declare -A seen
  total=${#rows[@]}
  pulled=0
  for row in "${rows[@]}"; do
    image="${row#*$'\t'}"
    if [ -n "${seen[$image]:-}" ]; then continue; fi
    seen[$image]=1

    avail=$(df -Pm "$repo_dir" | awk 'NR==2 {print $4}')
    if [ "$avail" -lt "$reserve_mb" ]; then
      die "only ${avail} MiB of disk left after ${pulled}/${total} image(s); free space or request fewer tasks"
    fi
    if ! pull "$image"; then
      die "could not pull $image (see above). Log in to ECR Public to lift the anonymous quota, or retry later."
    fi
    pulled=$((pulled + 1))
  done
  note "All ${pulled} task image(s) are present locally"

  # The egress-proxy container builds FROM ubuntu:24.04 (Docker Hub).
  docker pull --quiet ubuntu:24.04 || echo "warning: could not pre-pull ubuntu:24.04" >&2
  df -h "$repo_dir" | tail -n 1
fi

# ---------------------------------------------------------------- run

# No 6 h job cap locally, so the full DeepSWE budget is available unless
# TASK_TIMEOUT_MINUTES asks for a smaller one. Only the *agent* budget shrinks:
# Pier's generic --timeout-multiplier would also scale the 1800 s environment
# build timeout, and a cold build (install the harness into the task image)
# needs far more than the 150 s that a 15 min budget would leave it -- the
# trial then dies with EnvironmentStartTimeoutError before the agent runs.
multiplier=${TIMEOUT_MULTIPLIER:-}
agent_multiplier=""
if [ -z "$multiplier" ] && [ "$per_task" != "180" ]; then
  agent_multiplier=$(awk -v p="$per_task" 'BEGIN {printf "%.4f", p * 60 / 10800}')
fi

args=(
  run
  -p "$tasks_dir"
  --agent-import-path deepswe_adapter:PythonAgentHarness
  --model "$model"
  --env docker
  --jobs-dir "$jobs_dir"
  --n-concurrent "$concurrency"
  --yes
  --ak "wheel=$wheel"
)
if [ "$n_tasks" != "0" ]; then
  args+=(--n-tasks "$n_tasks" --sample-seed "$sample_seed")
fi
if [ "$max_rounds" != "0" ]; then
  args+=(--ak "max_rounds=$max_rounds")
fi
if [ -n "$multiplier" ]; then
  # Explicit TIMEOUT_MULTIPLIER: scale everything, as Pier documents.
  args+=(--timeout-multiplier "$multiplier")
elif [ -n "$agent_multiplier" ]; then
  args+=(--agent-timeout-multiplier "$agent_multiplier")
fi
# Escape hatch for the remaining Pier options (--include-task-name,
# --override-memory-mb, --max-retries, ...) without teaching this script every
# flag. Word splitting is intentional: EXTRA_ARGS is a command-line fragment.
if [ -n "$extra_args" ]; then
  read -r -a extra_array <<<"$extra_args"
  args+=("${extra_array[@]}")
fi

note "Running $n_tasks task(s) with $model (concurrency $concurrency, agent budget $per_task min)"
# Pier resolves --agent-import-path through importlib.import_module(), and a
# console script's sys.path does not include the cwd, so the directory holding
# the adapter goes on PYTHONPATH.
set +e
PYTHONPATH="$script_dir" "$venv/bin/pier" "${args[@]}"
status=$?
set -e

# ---------------------------------------------------------------- summary

# Report whatever finished, even if Pier exited non-zero.
if [ -d "$jobs_dir" ]; then
  budget=$per_task
  effective_multiplier=${multiplier:-$agent_multiplier}
  if [ -n "$effective_multiplier" ]; then
    budget=$(awk -v m="$effective_multiplier" 'BEGIN {printf "%d", m * 180 + 0.5}')
  fi
  # Pier writes each run into its own timestamped directory and this script
  # reuses one jobs dir across runs, so summarize the newest run only; mixing
  # in older trials would make the pass rate meaningless.
  newest=$(ls -dt "$jobs_dir"/*/ 2>/dev/null | head -n 1 || true)
  "$venv/bin/python" "$script_dir/deepswe_summary.py" \
    --jobs-dir "${newest:-$jobs_dir}" \
    --budget-minutes "$budget" \
    --timeout-multiplier "$effective_multiplier" |
    tee "$jobs_dir/summary.md" || true
  note "Results are in $jobs_dir (summary: $jobs_dir/summary.md)"
fi

exit "$status"
