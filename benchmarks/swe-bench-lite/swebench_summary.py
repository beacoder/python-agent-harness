#!/usr/bin/env python3
"""Generate a SWE-bench Lite benchmark summary from eval results.

Reads the final report (`<model>.<run_id>.json` in the report dir) and the
per-instance `report.json` files under `logs/run_evaluation/<run_id>/...`,
then renders a markdown summary.

Usage:
    python swebench_summary.py --run-id my-eval --report-dir ./swebench-results \
        --log-dir ./logs/run_evaluation
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def find_summary_report(report_dir: pathlib.Path, run_id: str) -> dict | None:
    """The final run report, named `<model>.<run_id>.json` by swebench."""
    for path in sorted(report_dir.glob(f"*.{run_id}.json")):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def find_per_instance_reports(log_dir: pathlib.Path, run_id: str) -> list[tuple[str, dict]]:
    """All per-instance report.json files for the run, keyed by instance_id."""
    reports: list[tuple[str, dict]] = []
    run_dir = log_dir / run_id
    for report_path in sorted(run_dir.rglob("report.json")):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and len(data) == 1:
            instance_id = next(iter(data))
            payload = data[instance_id]
            if isinstance(payload, dict):
                reports.append((instance_id, payload))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SWE-bench Lite results")
    parser.add_argument("--run-id", required=True, help="Eval run ID")
    parser.add_argument(
        "--report-dir",
        type=pathlib.Path,
        required=True,
        help="Directory swebench wrote its final report to",
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=pathlib.Path("logs/run_evaluation"),
        help="Directory with per-instance report.json files",
    )
    args = parser.parse_args()

    summary = find_summary_report(args.report_dir, args.run_id)
    per_instance = find_per_instance_reports(args.log_dir, args.run_id)

    lines = ["## SWE-bench Lite results", ""]

    if summary:
        dataset_total = summary.get("total_instances", 0)
        submitted = summary.get("submitted_instances", 0)
        completed = summary.get("completed_instances", 0)
        resolved = summary.get("resolved_instances", 0)
        unresolved = summary.get("unresolved_instances", 0)
        error = summary.get("error_instances", 0)
        empty_patch = summary.get("empty_patch_instances", 0)

        # `total_instances` is the whole dataset (300); with N_TASKS < 300 the
        # meaningful denominator is what this run actually graded.
        graded = completed or submitted
        if graded:
            lines.append(f"**Pass rate: {resolved}/{graded} ({100 * resolved / graded:.0f}%)**")
        else:
            lines.append("**No instances graded**")
        lines.append("")
        if dataset_total and graded and graded < dataset_total:
            lines.append(f"- Run covered {graded} of {dataset_total} dataset instances")
        lines.append(f"- Instances completed: {completed}")
        lines.append(f"- Resolved: {resolved}")
        lines.append(f"- Unresolved: {unresolved}")
        lines.append(f"- Errors: {error}")
        lines.append(f"- Empty patches: {empty_patch}")
        lines.append("")

    if per_instance:
        total = len(per_instance)
        resolved = sum(1 for _, r in per_instance if r.get("resolved"))
        if not summary:
            lines.append(f"**Pass rate: {resolved}/{total} ({100 * resolved / total:.0f}%)**")
            lines.append("")
        lines.append("| Instance | Resolved | Status |")
        lines.append("| --- | --- | --- |")
        for instance_id, r in per_instance:
            resolved_str = "PASS" if r.get("resolved") else "FAIL"
            status = r.get("status", "")
            lines.append(f"| {instance_id} | {resolved_str} | {status} |")
    elif summary and summary.get("resolved_ids"):
        lines.append("### Resolved instances")
        for id_ in summary["resolved_ids"]:
            lines.append(f"- {id_}")
    elif not summary:
        lines.append("No results found. Check that the eval completed.")
        print("\n".join(lines))
        return 0

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
