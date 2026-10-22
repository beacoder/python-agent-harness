"""List the container images a DeepSWE/Pier run will need.

The DeepSWE task images live in a public ECR registry
(``public.ecr.aws/d3j8x8q7/swe-bench-202605``), which throttles *anonymous*
pulls per source IP.  GitHub-hosted runners share their egress addresses with
everyone else on the platform, so a run that pulls lazily -- one image at a
time, as each trial builds its child image -- can die mid-run with::

    429 Too Many Requests - toomanyrequests: Data limit exceeded

Printing the image set first lets the workflow pull everything up front: with
a retry loop, and (when AWS credentials are configured) with authenticated
pulls, which are not subject to the anonymous quota.

The selection deliberately goes through Pier's own ``DatasetConfig`` so that
the printed set is exactly what ``pier run -p ... --n-tasks N --sample-seed S``
will execute.  Reimplementing the sampling (``random.Random(seed).shuffle``
over ``Path.iterdir()``) would drift from Pier's behaviour.

Usage:

    python benchmarks/deepswe_images.py --path deep-swe/tasks \
        --n-tasks 10 --sample-seed 0

Prints one ``<task-name>\t<image>`` line per image.  ``--n-tasks 0`` means the
whole dataset, matching the workflow's convention; ``--sample-seed`` is only
applied when a task limit is set, again matching the workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tomllib
from pathlib import Path

from pier.models.job.config import DatasetConfig


def _images_of(task_toml: Path) -> list[str]:
    """Return every Docker image a task declares, top level and per step."""
    data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    environments = [data.get("environment") or {}]
    environments += [step.get("environment") or {} for step in data.get("steps") or []]

    images: list[str] = []
    for environment in environments:
        image = environment.get("docker_image")
        if isinstance(image, str) and image and image not in images:
            images.append(image)
    return images


async def _collect(
    path: Path, n_tasks: int | None, sample_seed: int | None
) -> list[tuple[str, str]]:
    dataset = DatasetConfig(path=path, n_tasks=n_tasks, sample_seed=sample_seed)
    rows: list[tuple[str, str]] = []
    for task in await dataset.get_task_configs():
        for image in _images_of(task.path / "task.toml"):
            rows.append((task.path.name, image))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List the container images a DeepSWE run will pull.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("deep-swe/tasks"),
        help="DeepSWE task directory (default: deep-swe/tasks)",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=0,
        help="Number of tasks Pier will sample; 0 means all of them",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Sampling seed; ignored when --n-tasks is 0",
    )
    args = parser.parse_args(argv)

    if not args.path.is_dir():
        parser.error(f"task directory not found: {args.path}")

    n_tasks = args.n_tasks if args.n_tasks > 0 else None
    sample_seed = args.sample_seed if n_tasks is not None else None

    rows = asyncio.run(_collect(args.path, n_tasks, sample_seed))
    for name, image in rows:
        print(f"{name}\t{image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
