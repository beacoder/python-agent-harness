# GitHub Actions Workflows

- `ci.yml`: lint, typecheck, tests and package validation on every push/PR
  (Ubuntu + macOS, Python 3.11-3.13).
- `release.yml`: build and publish the PyPI release.

## Benchmarks

Benchmarking runs on your own machine, not in Actions:

- `benchmarks/swe-bench-lite/run-swebench-lite-local.sh` runs the harness on the
  SWE-bench Lite dataset and grades the result with the `swebench` package.
  See `benchmarks/swe-bench-lite/README.md`.