#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/a100_cpu48_hybrid_benchmark}"
TASKS="${2:-192}"
HORIZON_S="${3:-0.02}"
CONFIG="${CONFIG:-$ROOT/config/nsga2_condensed_phase_no_f_a100_cpu48_hybrid.yaml}"
if [[ -e "$WORKDIR" ]]; then
  echo "ERROR: benchmark workdir exists: $WORKDIR" >&2
  exit 2
fi
mkdir -p "$(dirname "$WORKDIR")"

export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$ROOT/tools/build_cpp_cpu.sh"
"$ROOT/tools/build_cpp_hybrid_cpu.sh"
python - <<'PY'
import os, torch
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA is unavailable')
print('benchmark GPU:', torch.cuda.get_device_name(0))
print('CPU workers override:', os.environ.get('ECSP_HYBRID_CPU_WORKERS', 'config default'))
print('CUDA batch override:', os.environ.get('ECSP_HYBRID_CUDA_BATCH_SIZE', 'config default'))
PY

# The default 192 candidates provides enough work for a 64-case CUDA batch,
# 40 CPU workers and a remaining queue. The short horizon exercises repeated
# electrical/species/thermal updates without a production-scale wait.
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --package-root "$ROOT" \
  --config "$CONFIG" \
  --device cuda \
  --population-size "$TASKS" \
  --generations 1 \
  --end-time-s "$HORIZON_S" \
  --allow-no-feasible \
  --workdir "$WORKDIR"

echo '--- hybrid runtime stats ---'
cat "$WORKDIR/adapter/hybrid_runtime_stats.json"
echo '--- hybrid evaluator diagnostics ---'
cat "$WORKDIR/adapter/hybrid_evaluator_diagnostics.json"
