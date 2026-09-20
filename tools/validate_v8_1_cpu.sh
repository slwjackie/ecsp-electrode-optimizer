#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
OUT="${1:-$ROOT/runs/v8_1_cpu_validation}"
mkdir -p "$OUT"
python -c "from ecsp_native import load_native; from ecsp_native.standalone import build_standalone; load_native(False, True); build_standalone(True)"
cd "$ROOT"
python -m pytest -q --durations=20 --junitxml="$OUT/full_suite.xml"
python "$ROOT/python/compare_bc_native_python.py" --output "$OUT/native_python_parity_strict.json" --strict-common-tolerances
python "$ROOT/python/audit_v800_preservation.py" --output-dir "$OUT/preservation"
python "$ROOT/python/check_bc_native_cuda.py" --output "$OUT/cuda_readiness.json"
