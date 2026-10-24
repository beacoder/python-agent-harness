#!/usr/bin/env bash
#
# Run the SWE-bench Lite benchmark on this machine.
#
# SWE-bench Lite is graded by the `swebench` evaluation harness. The flow:
#   1. Load the HuggingFace dataset SWE-bench/SWE-bench_Lite
#   2. For each instance: spin up the task's Docker image, run the harness
#      headless inside it to produce a git patch
#   3. Write predictions.jsonl
#   4. Run `swebench eval` to grade patches against held-out tests
#
# Quick start:
#
#     export OPENAI_API_KEY=...
#     benchmarks/swe-bench-lite/run-swebench-lite-local.sh          # 1 task
#     N_TASKS=10 benchmarks/swe-bench-lite/run-swebench-lite-local.sh
#     N_TASKS=0 benchmarks/swe-bench-lite/run-swebench-lite-local.sh   # all 300
#
# Environment (defaults in parentheses):
#   N_TASKS (1)                 0 = all 300 instances
#   MODEL (deepseek-flash)      model name in the harness config
#   SAMPLE_SEED (0)             fixed so reruns pick the same instances
#   TASK_TIMEOUT_MINUTES (30)   agent budget per task (SWE-bench eval has its
#                               own 1800 s per-instance test timeout, separate
#                               from this budget)
#   CONCURRENCY (1)             parallel agent runs
#   MAX_ROUNDS (0)              0 = unlimited LLM rounds per task
#   OPENAI_API_KEY              falls back to the harness config file
#   OPENAI_BASE_URL             falls back to the harness config file
#   VENV_DIR                    driver venv (<repo>/.venv-swebench)
#   RESULTS_DIR                 output directory (<repo>/swebench-results)
#   SKIP_AGENT (0)              1 = skip agent runs, only re-grade existing
#                               predictions
#   SKIP_EVAL (0)               1 = skip swebench eval, only run the agent
#   SKIP_PULL (0)               1 = do not pre-pull task Docker images
#   EXTRA_ARGS                  appended verbatim to the agent runner
#
set -euo pipefail

script_src=${BASH_SOURCE[0]:-$0}
if [ "${script_src##*/}" = "$script_src" ]; then
  script_src=$(command -v -- "$script_src" || printf '%s' "$script_src")
fi
script_path=$(readlink -f -- "$script_src" 2>/dev/null || printf '%s' "$script_src")
script_dir=$(cd -- "$(dirname -- "$script_path")" && pwd)
script_path=$script_dir/${script_path##*/}
repo_dir=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_dir"

n_tasks=${N_TASKS:-1}
model=${MODEL:-deepseek-flash}
sample_seed=${SAMPLE_SEED:-0}
per_task=${TASK_TIMEOUT_MINUTES:-30}
concurrency=${CONCURRENCY:-1}
max_rounds=${MAX_ROUNDS:-0}
venv=${VENV_DIR:-$repo_dir/.venv-swebench}
results_dir=${RESULTS_DIR:-$repo_dir/swebench-results}
skip_agent=${SKIP_AGENT:-0}
skip_eval=${SKIP_EVAL:-0}
skip_pull=${SKIP_PULL:-0}
extra_args=${EXTRA_ARGS:-}

if [ "$concurrency" -lt 1 ]; then concurrency=1; fi

note() { echo "==> $*"; }
die() {
  echo "error: $*" >&2
  exit 1
}

# ---------------------------------------------------------------- preflight

note "Checking Docker"
command -v docker >/dev/null 2>&1 || die "docker not found. Install Docker Engine."

in_group() {
  getent group "$1" 2>/dev/null |
    awk -F: -v user="$(id -un)" '{
      n = split($4, members, ",")
      for (i = 1; i <= n; i++) if (members[i] == user) found = 1
    } END { exit !found }'
}

if ! docker info >/dev/null 2>&1; then
  if [ "${SWEBENCH_REEXEC:-0}" != "1" ] && in_group docker && command -v sg >/dev/null 2>&1; then
    note "The docker socket is not reachable in this shell; re-running inside the 'docker' group"
    SWEBENCH_REEXEC=1 exec sg docker -c "$(printf '%q ' "$script_path" "$@")"
  fi
  die "the Docker daemon is not reachable. Start it, or join the docker group and re-login."
fi

# SWE-bench image evaluations run one full conda environment per worker; a
# machine with less headroom gets OOM-killed containers mid-run, which reads
# like a model failure. Warn instead of failing.
mem_total_mb=$(awk '/^MemTotal:/ {printf "%d", $2 / 1024}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$mem_total_mb" -gt 0 ] && [ "$mem_total_mb" -lt 12288 ]; then
  echo "warning: this machine has ${mem_total_mb} MiB of RAM; swe-bench containers can use several GiB each." >&2
  echo "warning: keep CONCURRENCY at 1 and expect OOM kills on big test suites." >&2
fi
if [ "$concurrency" -gt 1 ] && [ "$mem_total_mb" -gt 0 ]; then
  need_mb=$((concurrency * 8192))
  if [ "$mem_total_mb" -lt "$need_mb" ]; then
    echo "warning: CONCURRENCY=$concurrency asks for ~${need_mb} MiB of container memory on a ${mem_total_mb} MiB machine." >&2
  fi
fi

# ---------------------------------------------------------------- venv

# swebench >= 2.0 needs Python >= 3.10; the harness itself needs >= 3.11.
# Pick 3.11+ to satisfy both.
if [ ! -x "$venv/bin/python" ]; then
  python_bin=""
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' 2>/dev/null; then
      python_bin=$(command -v "$candidate")
      break
    fi
  done
  [ -n "$python_bin" ] || die "no Python 3.11+ found."
  note "Creating $venv with $("$python_bin" -V 2>&1)"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$venv" --python "$python_bin"
  else
    "$python_bin" -m venv "$venv"
  fi
fi
"$venv/bin/python" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' ||
  die "$venv is not Python 3.11+; remove it and rerun."

if ! "$venv/bin/python" -c 'import swebench, python_agent_harness' >/dev/null 2>&1; then
  note "Installing swebench and python-agent-harness into $venv"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$venv/bin/python" -e "$repo_dir" "swebench>=2.0" "datasets" "build"
  else
    "$venv/bin/python" -m pip install --quiet --upgrade pip
    "$venv/bin/python" -m pip install --quiet -e "$repo_dir" "swebench>=2.0" "datasets" "build"
  fi
fi
"$venv/bin/python" -c 'import swebench' || die "swebench is not importable from $venv"

# ---------------------------------------------------------------- credentials

# MODEL can be either a named profile in the harness config ({"models": {
# "deepseek": {...}}}) or a bare model id.  Resolve it through the config so
# the agent gets the endpoint's real model name, base URL and key.  Values
# already present in the environment always win.
config_file=${PYTHON_AGENT_HARNESS_CONFIG:-$HOME/.config/python-agent-harness/config.json}
if [ -f "$config_file" ]; then
  had_key=${OPENAI_API_KEY:-}
  eval "$("$venv/bin/python" - "$config_file" "$model" <<'PY'
import json
import os
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
    if profile.get("api_key") and not os.environ.get("OPENAI_API_KEY"):
        print(f"export OPENAI_API_KEY={shlex.quote(profile['api_key'])}")
    if profile.get("base_url") and not os.environ.get("OPENAI_BASE_URL"):
        print(f"export OPENAI_BASE_URL={shlex.quote(profile['base_url'])}")
    if profile.get("model") and not os.environ.get("OPENAI_MODEL"):
        print(f"export OPENAI_MODEL={shlex.quote(profile['model'])}")
PY
)"
  if [ -z "$had_key" ] && [ -n "${OPENAI_API_KEY:-}" ]; then
    note "Using the API key from $config_file"
  fi
fi
[ -n "${OPENAI_API_KEY:-}" ] ||
  die "OPENAI_API_KEY is not set and no matching profile exists in the harness config."
# A profile name ("deepseek") is not necessarily an id the endpoint accepts
# ("deepseek-flash"); use the resolved id everywhere so the agent, the
# predictions and the report all agree.
if [ -n "${OPENAI_MODEL:-}" ] && [ "$OPENAI_MODEL" != "$model" ]; then
  note "Resolved model profile $model -> $OPENAI_MODEL"
  model=$OPENAI_MODEL
fi
export OPENAI_MODEL=$model
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.deepseek.com/v1}

# ---------------------------------------------------------------- build wheel

# The wheel is only needed by the agent phase; skip the rebuild (and its
# implicit network use) when re-grading existing predictions.
if [ "$skip_agent" != "1" ]; then
  note "Building the harness wheel"
  rm -rf "$repo_dir/dist"
  "$venv/bin/python" -m build --wheel --outdir "$repo_dir/dist" "$repo_dir" >/dev/null
  wheels=("$repo_dir"/dist/python_agent_harness-*.whl)
  [ -f "${wheels[0]}" ] || die "the wheel build produced no file in $repo_dir/dist"
  wheel=${wheels[0]}
fi

# ---------------------------------------------------------------- resolve instances

note "Resolving instances for n_tasks=$n_tasks (seed $sample_seed)"
if ! resolved=$("$venv/bin/python" "$script_dir/swebench_instances.py" \
  --n-tasks "$n_tasks" --sample-seed "$sample_seed"); then
  die "could not resolve instances from the SWE-bench Lite dataset"
fi
[ -n "$resolved" ] || die "no instances resolved"
mapfile -t instance_ids <<<"$resolved"
total=${#instance_ids[@]}
note "Selected $total instance(s)"

# ---------------------------------------------------------------- pre-pull images

if [ "$skip_pull" != "1" ]; then
  note "Pre-pulling Docker images for $total instance(s)"
  # One dataset load for the whole selection: a per-instance lookup would
  # re-download the HF dataset for every id.  The helper prints one
  # "instance_id<TAB>image" line per instance.
  images_out=""
  if ! images_out=$("$venv/bin/python" "$script_dir/swebench_instances.py" \
    --n-tasks "$n_tasks" --sample-seed "$sample_seed" --images); then
    echo "warning: could not list images up front; falling back to per-instance lookup" >&2
    images_out=""
  fi
  declare -A image_map=()
  if [ -n "$images_out" ]; then
    while IFS=$'\t' read -r iid img; do
      if [ -n "$iid" ] && [ -n "$img" ]; then
        image_map["$iid"]="$img"
      fi
    done <<<"$images_out"
  fi
  for id in "${instance_ids[@]}"; do
    image=${image_map["$id"]:-}
    if [ -z "$image" ]; then
      # A broken image lookup must not abort the whole run under `set -e`.
      set +e
      image=$("$venv/bin/python" "$script_dir/swebench_instances.py" --image "$id" 2>/dev/null)
      set -e
    fi
    if [ -n "$image" ]; then
      note "Pulling $image"
      docker pull --quiet "$image" || echo "warning: could not pre-pull $image" >&2
    else
      echo "warning: no image resolvable for $id" >&2
    fi
  done
fi

# ---------------------------------------------------------------- run agent

predictions_file="$results_dir/predictions.jsonl"
mkdir -p "$results_dir"

if [ "$skip_agent" != "1" ]; then
  # fresh run: reset the predictions file so a rerun cannot leave stale
  # entries behind and every instance graded was written by this run.
  : > "$predictions_file"
  note "Running agent on $total instance(s) with $model (concurrency $concurrency, budget ${per_task} min)"

  agent_args=(
    "$script_dir/swebench_agent.py"
    --wheel "$wheel"
    --model "$model"
    --predictions "$predictions_file"
    --concurrency "$concurrency"
    --timeout-minutes "$per_task"
  )
  # There is no sentinel for "whole suite": pass the resolved
  # count (0 = every instance) explicitly so the agent does not fall back to
  # its own default of 1.
  agent_args+=(--n-tasks "$n_tasks" --sample-seed "$sample_seed")
  if [ "$max_rounds" != "0" ]; then
    agent_args+=(--max-rounds "$max_rounds")
  fi
  if [ -n "$extra_args" ]; then
    read -r -a extra_array <<<"$extra_args"
    agent_args+=("${extra_array[@]}")
  fi

  set +e
  "$venv/bin/python" "${agent_args[@]}"
  status=$?
  set -e
  if [ "$status" -ne 0 ]; then
    echo "warning: agent runner exited with status $status; continuing to eval" >&2
  fi
else
  note "Skipping agent runs (SKIP_AGENT=1)"
  if [ "$skip_eval" != "1" ]; then
    [ -s "$predictions_file" ] ||
      die "SKIP_AGENT=1 needs an existing predictions file at $predictions_file, and it is missing or empty."
  fi
  status=0
fi

# ---------------------------------------------------------------- eval

if [ "$skip_eval" != "1" ]; then
  note "Running swebench eval"
  # Swebench names its per-run log directory with the run id, and the final
  # report file as <model>.<run_id>.json.  A model name containing "/" would
  # therefore corrupt both paths, so map it to a filesafe form (the harness
  # also writes predictions under the same name, keeping them comparable).
  run_id=$(printf '%s' "$model" | tr '/ ' '__')-$(date +%Y%m%d-%H%M%S)

  set +e
  "$venv/bin/swebench" eval SWE-bench/SWE-bench_Lite \
    -p "$predictions_file" \
    --run-id "$run_id" \
    -j "$concurrency" \
    -t 1800 \
    --report-dir "$results_dir"
  eval_status=$?
  set -e

  if [ "$eval_status" -ne 0 ]; then
    echo "warning: swebench eval exited with status $eval_status" >&2
    # A failed eval means the numbers below are incomplete; surface it in the
    # exit code instead of reporting success.
    status=$eval_status
  fi

  # ---------------------------------------------------------------- summary

  note "Generating summary"
  "$venv/bin/python" "$script_dir/swebench_summary.py" \
    --run-id "$run_id" \
    --report-dir "$results_dir" \
    --log-dir "$repo_dir/logs/run_evaluation" |
    tee "$results_dir/summary.md" || true
  note "Results are in $results_dir (summary: $results_dir/summary.md)"
else
  note "Skipping swebench eval (SKIP_EVAL=1)"
fi

exit "$status"
