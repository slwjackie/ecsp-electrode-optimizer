#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "ERROR: MPS benchmark is archived because it does not implement v7.9.0 surface-contact physics." >&2
echo "Use: bash tools/benchmark_m2_cpp_fp64.sh \"$ROOT/runs/m2_cpp_fp64_benchmark\"" >&2
exit 2
