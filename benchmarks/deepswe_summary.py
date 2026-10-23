"""Summarize a finished DeepSWE/Pier run as GitHub-flavoured markdown.

Shared by the GitHub Actions workflow (which appends the output to the run
summary) and by ``run-deepswe-local.sh`` (which prints it and stores it next to
the results), so a local run reports scores exactly like a cloud run: per-task
rewards, pass rate, token totals and the agent budget the scores were produced
under.

Usage:

    python benchmarks/deepswe_summary.py --jobs-dir pier-results \
        --budget-minutes 180 --timeout-multiplier 1.0000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def summarize(jobs_dir: pathlib.Path, budget: str = "", multiplier: str = "") -> str:
    """Render every completed trial under ``jobs_dir`` as markdown."""
    rows: list[tuple[str, float | int | None, dict[str, object], str]] = []
    tokens_in = tokens_out = 0

    for path in sorted(jobs_dir.rglob("result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        task = data.get("task_name")
        if not task:
            # the job-level result.json carries no task_name
            continue
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        values = [v for v in rewards.values() if isinstance(v, (int, float))]
        score = min(values) if values else None
        exception = (data.get("exception_info") or {}).get("exception_type") or ""
        rows.append((task, score, rewards, exception))
        agent = data.get("agent_result") or {}
        tokens_in += agent.get("n_input_tokens") or 0
        tokens_out += agent.get("n_output_tokens") or 0

    lines = ["## DeepSWE results", ""]
    # The budget the scores were produced under: a pass rate at a 9 min budget
    # is not comparable to one at the full 180 min, so it belongs next to the
    # number.
    if budget:
        lines.append(
            f"Agent budget: **{budget} min/task** (timeout multiplier {multiplier or 'n/a'})"
        )
        lines.append("")
    if not rows:
        lines.append("No trials completed - check the job logs.")
        return "\n".join(lines) + "\n"

    lines.append("| Task | Score | Rewards | Exception |")
    lines.append("| --- | --- | --- | --- |")
    for task, score, rewards, exception in rows:
        lines.append(f"| {task} | {score} | `{json.dumps(rewards)}` | {exception} |")
    passed = sum(1 for _, score, _, _ in rows if score == 1)
    lines.append("")
    lines.append(f"**Pass rate: {passed}/{len(rows)} ({100 * passed / len(rows):.0f}%)**")
    lines.append(f"**Tokens: {tokens_in:,} in / {tokens_out:,} out**")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a DeepSWE/Pier run.")
    parser.add_argument(
        "--jobs-dir",
        type=pathlib.Path,
        default=pathlib.Path("pier-results"),
        help="Pier --jobs-dir of the run (default: pier-results)",
    )
    parser.add_argument(
        "--budget-minutes",
        default="",
        help="Agent budget per task the scores were produced under",
    )
    parser.add_argument(
        "--timeout-multiplier",
        default="",
        help="Pier --timeout-multiplier used for the run",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Also write the markdown to this file (default: stdout only)",
    )
    args = parser.parse_args(argv)

    if not args.jobs_dir.is_dir():
        print(f"error: no jobs directory at {args.jobs_dir}", file=sys.stderr)
        return 1

    markdown = summarize(args.jobs_dir, args.budget_minutes, args.timeout_multiplier)
    sys.stdout.write(markdown)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
