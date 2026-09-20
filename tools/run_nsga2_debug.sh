#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/nsga2_debug}"
mkdir -p "$(dirname "$WORKDIR")"
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --package-root "$ROOT" \
  --config "$ROOT/config/nsga2_condensed_phase_no_f_debug.yaml" \
  --workdir "$WORKDIR" \
  --allow-debug-physics
