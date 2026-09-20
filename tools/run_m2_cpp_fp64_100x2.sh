#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/m2_cpp_fp64_100x2}"
POPULATION="${2:-100}"
GENERATIONS="${3:-2}"
CONFIG="${CONFIG:-$ROOT/config/nsga2_condensed_phase_no_f_m2_cpp_fp64.yaml}"
if [[ -e "$WORKDIR" ]]; then
  echo "ERROR: workdir already exists: $WORKDIR" >&2
  echo "Use a new path so old rejection/fitness files cannot mix with this run." >&2
  exit 2
fi
"$ROOT/tools/build_cpp_cpu.sh"
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --package-root "$ROOT" \
  --config "$CONFIG" \
  --device cpu \
  --population-size "$POPULATION" \
  --generations "$GENERATIONS" \
  --allow-no-feasible \
  --workdir "$WORKDIR"
