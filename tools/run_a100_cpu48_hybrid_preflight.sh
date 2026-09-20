#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/a100_cpu48_hybrid_preflight}"
CONFIG="${CONFIG:-$ROOT/config/nsga2_condensed_phase_no_f_a100_cpu48_hybrid.yaml}"
if [[ -e "$WORKDIR" ]]; then
  echo "ERROR: preflight workdir exists: $WORKDIR" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
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

python - <<'PY'
import os, platform, torch
print('architecture  :', platform.machine())
print('logical CPUs :', os.cpu_count())
print('torch        :', torch.__version__)
print('CUDA ready   :', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA is unavailable')
p = torch.cuda.get_device_properties(0)
print('GPU          :', p.name)
print('VRAM GiB     :', round(p.total_memory / 2**30, 2))
print('FP64 dtype   : enabled')
PY

python "$ROOT/python/validate_hybrid_a100_cpu48.py" \
  --package-root "$ROOT" \
  --config "$CONFIG" \
  --workdir "$WORKDIR" \
  --task-count 80 \
  --grid-size 193 \
  --cpu-workers 16
