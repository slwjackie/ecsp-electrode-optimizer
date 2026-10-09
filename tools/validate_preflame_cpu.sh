#!/usr/bin/env bash
# Current source/test gate. CUDA runtime validation is a separate hardware task.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
cd "$ROOT"
"$PYTHON_BIN" -m compileall -q python
while IFS= read -r script; do
  bash -n "$script"
done < <(find "$ROOT/tools" -type f -name '*.sh' -print | sort)
"$PYTHON_BIN" -m pytest -q python/tests tests "$@"
# Optional CUDA source/toolchain report (does not replace CUDA runtime tests):
# python python/check_preflame_cpp_cuda.py --output <fresh-report.json>
