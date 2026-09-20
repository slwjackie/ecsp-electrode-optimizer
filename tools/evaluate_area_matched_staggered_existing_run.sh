#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:?usage: $0 /absolute/path/to/existing/run [optional-config.yaml]}"
CONFIG="${2:-}"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  python "$ROOT/python/evaluate_area_matched_staggered.py"
  --package-root "$ROOT"
  --workdir "$WORKDIR"
)
if [[ -n "$CONFIG" ]]; then
  ARGS+=(--config "$CONFIG")
fi
"${ARGS[@]}"
