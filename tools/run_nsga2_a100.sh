#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/nsga2_cpp_fp64_on_a100_host}"
echo "NOTICE: v7.9.0 condensed physics uses the native C++ CPU FP64 backend; the A100 GPU is not used by this wrapper." >&2
exec bash "$ROOT/tools/run_nsga2_m2_cpp_fp64.sh" "$WORKDIR"
