#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/cpp_python_fp64_parity}"
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
python "$ROOT/python/compare_cpp_python_fp64.py" \
  --package-root "$ROOT" \
  --workdir "$WORKDIR"
