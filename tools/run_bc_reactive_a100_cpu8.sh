#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/config/nsga2_bc_dual_a100_cpu8.yaml}"
WORKDIR="${1:-$ROOT/runs/bc_reactive_a100_cpu8}"
GEOMETRY_CHECK="${WORKDIR}_geometry_check"
if [[ -e "$GEOMETRY_CHECK" || -L "$GEOMETRY_CHECK" ]]; then
  echo "ERROR: geometry preflight output exists; choose a fresh workdir." >&2; exit 2
fi
if [[ -e "$WORKDIR" || -L "$WORKDIR" || -e "${WORKDIR}.log" || -L "${WORKDIR}.log" || -e "${WORKDIR}_preflight" || -L "${WORKDIR}_preflight" ]]; then
  echo "ERROR: output/log/preflight exists; choose a fresh workdir (nothing is deleted)." >&2; exit 2
fi
mkdir -p "$(dirname "$WORKDIR")"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST=8.0
# Geometry is validated first on both design and actual solver grids, before
# any numerical preflight or expensive candidate PDE. The main loop reuses this
# source/seed/grid-bound cache and rechecks every mask, without area re-fitting.
GEOMETRY_ARGS=()
if [[ "${ECSP_REQUIRE_POST_ONSET_POWER_OFF:-0}" == "1" ]]; then
  GEOMETRY_ARGS+=(--require-power-off)
fi
python "$ROOT/python/validate_nsga2_geometry.py" --config "$CONFIG" --output "$GEOMETRY_CHECK" "${GEOMETRY_ARGS[@]}"
# No skip-preflight switch. Explicit NVIDIA A100 identity + actual GPU tests.
python "$ROOT/python/preflight_reactive_a100.py" --output "${WORKDIR}_preflight"
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" --package-root "$ROOT" --config "$CONFIG" --workdir "$WORKDIR" --bootstrap-cache "$GEOMETRY_CHECK" 2>&1 | tee "${WORKDIR}.log"
