#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "ERROR: legacy MPS FP32 physics used electrode masks as material holes and is disabled in v7.9.0." >&2
echo "Use: bash tools/run_nsga2_m2_cpp_fp64.sh \"$ROOT/runs/nsga2_m2_cpp_fp64\"" >&2
exit 2
