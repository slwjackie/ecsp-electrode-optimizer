#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "ERROR: v7.9.0 surface-contact physics is not implemented in the legacy MPS FP32 backend." >&2
echo "Use: bash tools/run_m2_cpp_fp64_preflight.sh \"$ROOT/runs/m2_cpp_fp64_preflight\"" >&2
exit 2
