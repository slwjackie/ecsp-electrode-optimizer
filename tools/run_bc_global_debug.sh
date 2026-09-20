#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/bc_global_debug}"
mkdir -p "$(dirname "$WORKDIR")"
if [[ -e "$WORKDIR" ]]; then
  echo "ERROR: debug workdir already exists; no files are deleted: $WORKDIR" >&2
  echo "Choose a new path so previous and current results cannot be mixed." >&2
  exit 2
fi
PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --config "$ROOT/config/nsga2_bc_global_preflame_propagation_debug.yaml" \
  --workdir "$WORKDIR" \
  --package-root "$ROOT" \
  --allow-no-feasible
