#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/generated_bc_audit}"
RUN="${2:-$ROOT/tmp_bc_global_audit_run}"
if [[ -e "$RUN" ]]; then
  echo "ERROR: runtime audit workdir already exists; no files are deleted: $RUN" >&2
  echo "Choose a new path so prior runtime evidence cannot be reused." >&2
  exit 2
fi
PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
python "$ROOT/python/validate_bc_global_pipeline.py" \
  --package-root "$ROOT" \
  --output-dir "$OUT" \
  --runtime-workdir "$RUN"
