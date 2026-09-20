#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/config/nsga2_bc_reactive_debug.yaml}"
WORKDIR="${1:-$ROOT/runs/bc_reactive_debug}"
if [[ -e "$WORKDIR" || -L "$WORKDIR" || -e "${WORKDIR}.log" || -L "${WORKDIR}.log" ]]; then echo "ERROR: output exists: $WORKDIR" >&2; exit 2; fi
mkdir -p "$(dirname "$WORKDIR")"
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --package-root "$ROOT" --config "$CONFIG" --workdir "$WORKDIR" 2>&1 | tee "${WORKDIR}.log"
