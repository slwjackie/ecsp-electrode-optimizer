#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/tools/run_nsga2_m2_cpp_fp64.sh" "${1:-$ROOT/runs/nsga2_cpp_fp64}"
