#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "WARNING: this is the preserved legacy v7.9.5 non-B/C hybrid launcher." >&2
echo "WARNING: it does NOT run the corrected v8.2 surface-overlay B/C model." >&2
echo "WARNING: use tools/run_bc_native_a100_cpu8.sh for v8.2 production." >&2
WORKDIR="${1:-$ROOT/runs/a100_cpu48_hybrid}"
POPULATION="${2:-200}"
GENERATIONS="${3:-3}"
CONFIG="${CONFIG:-$ROOT/config/nsga2_condensed_phase_no_f_a100_cpu48_hybrid.yaml}"

if [[ -e "$WORKDIR" ]]; then
  echo "ERROR: workdir already exists: $WORKDIR" >&2
  echo "Use a new directory so old CPU/GPU results cannot be mixed." >&2
  exit 2
fi
mkdir -p "$(dirname "$WORKDIR")"

export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
# Each C++ worker is intentionally single-threaded. Candidate-level parallelism
# is controlled by hybrid_cpu_worker_cases; this prevents 40 x BLAS/OpenMP
# oversubscription on a 48-vCPU allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$ROOT/tools/build_cpp_cpu.sh"
"$ROOT/tools/build_cpp_hybrid_cpu.sh"

NCPU=$(python - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)
if (( NCPU < 48 )); then
  echo "WARNING: only $NCPU logical CPUs are visible; the supplied profile targets 48 vCPUs." >&2
fi
nvidia-smi -L
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('ERROR: torch.cuda.is_available() is false')
print('CUDA device:', torch.cuda.get_device_name(0))
PY

if [[ "${ECSP_SKIP_HYBRID_PREFLIGHT:-0}" != "1" ]]; then
  PREFLIGHT="${WORKDIR}_preflight"
  CONFIG="$CONFIG" bash "$ROOT/tools/run_a100_cpu48_hybrid_preflight.sh" "$PREFLIGHT"
fi

set -o pipefail
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --package-root "$ROOT" \
  --config "$CONFIG" \
  --device cuda \
  --population-size "$POPULATION" \
  --generations "$GENERATIONS" \
  --allow-no-feasible \
  --workdir "$WORKDIR" \
  2>&1 | tee "${WORKDIR}.log"
