#!/usr/bin/env bash
# Model-check every TLA+ model in formal/models.py. Writes results/m9/<model>.out and
# results/m9/summary.json. Exit status is non-zero if any model's outcome differs from its
# expectation (fixed protocols must pass; buggy variants must produce a counterexample).
set -euo pipefail
cd "$(dirname "$0")/.."
bash formal/get_tools.sh
uv run python -m formal.run_tlc "$@"
