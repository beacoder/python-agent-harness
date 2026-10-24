#!/usr/bin/env python3
"""Resolve SWE-bench Lite instance IDs and Docker image names.

Used by run-swebench-lite-local.sh to:
  - List instance IDs for a given n_tasks/sample_seed
  - Look up the Docker image name for a specific instance (--image)
  - List image names for the whole selection (--images, one load)

Usage:
    python swebench_instances.py --n-tasks 10 --sample-seed 0
    python swebench_instances.py --image astropy__astropy-12907
    python swebench_instances.py --n-tasks 10 --sample-seed 0 --images
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def load_dataset() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    return list(ds)


def get_image_name(instance: dict[str, Any]) -> str:
    image = instance.get("image", "")
    if image:
        return image
    instance_id = instance["instance_id"]
    repo = instance_id.rsplit("-", 1)[0].replace("/", "_")
    repo_escaped = repo.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{repo_escaped}.{instance_id}:latest"


def select_instances(
    instances: list[dict[str, Any]], n_tasks: int, sample_seed: int
) -> list[dict[str, Any]]:
    if n_tasks == 0:
        return instances
    import random

    rng = random.Random(sample_seed)
    shuffled = instances[:]
    rng.shuffle(shuffled)
    return shuffled[: min(n_tasks, len(shuffled))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve SWE-bench Lite instances")
    parser.add_argument("--n-tasks", type=int, default=1, help="Number of tasks (0=all)")
    parser.add_argument("--sample-seed", type=int, default=0, help="Random seed")
    parser.add_argument("--image", default=None, help="Get image name for this instance ID")
    parser.add_argument(
        "--images", action="store_true", help="List image names for the selected set"
    )
    args = parser.parse_args()

    instances = load_dataset()

    if args.image:
        for inst in instances:
            if inst["instance_id"] == args.image:
                print(get_image_name(inst))
                return 0
        print(f"error: instance {args.image} not found", file=sys.stderr)
        return 1

    selected = select_instances(instances, args.n_tasks, args.sample_seed)

    if args.images:
        for inst in selected:
            print(f"{inst['instance_id']}\t{get_image_name(inst)}")
        return 0

    for inst in selected:
        print(inst["instance_id"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
