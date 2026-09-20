#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/surface_contact_validation}"
if [[ -e "$WORKDIR" ]]; then
  echo "ERROR: validation workdir exists: $WORKDIR" >&2
  exit 2
fi
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
cd "$ROOT"
python -m pytest -q \
  python/tests/test_surface_contact_multicomponent_geometry.py \
  python/tests/test_cpp_surface_contact_physics.py
python python/validate_multicomponent_surface_pairs.py \
  --package-root "$ROOT" \
  --workdir "${WORKDIR}_pairs"
bash tools/run_m2_cpp_fp64_preflight.sh "$WORKDIR"
