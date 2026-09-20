#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/m2_cpp_fp64_preflight}"
CONFIG="${CONFIG:-$ROOT/config/nsga2_condensed_phase_no_f_m2_cpp_fp64.yaml}"
if [[ -e "$WORKDIR" ]]; then
  echo "ERROR: preflight workdir exists: $WORKDIR" >&2
  exit 2
fi
"$ROOT/tools/build_cpp_cpu.sh"
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
python - <<'PY'
import platform
print('architecture :', platform.machine())
print('physics      : C++ CPU FP64')
print('Metal/MPS    : not used by condensed physics')
PY
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --package-root "$ROOT" \
  --config "$CONFIG" \
  --device cpu \
  --population-size 4 \
  --generations 1 \
  --grid-size 193 \
  --end-time-s 0.00025 \
  --allow-no-feasible \
  --workdir "$WORKDIR"
SUCCESS=$(find "$WORKDIR/generation_000/physics" -name condensed_metrics.json 2>/dev/null | wc -l | tr -d ' ')
REJECTED=$(find "$WORKDIR/generation_000/physics" -name physics_rejection.txt 2>/dev/null | wc -l | tr -d ' ')
echo "C++ FP64 preflight: successful=$SUCCESS rejected=$REJECTED"
if [[ "$SUCCESS" -ne 4 || "$REJECTED" -ne 0 ]]; then
  find "$WORKDIR/generation_000/physics" -name physics_rejection.txt -exec cat {} \; 2>/dev/null || true
  exit 3
fi
