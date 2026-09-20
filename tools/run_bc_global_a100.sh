#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

WORKDIR="${1:-$ROOT/runs/bc_global_a100}"
POPULATION="${2:-1000}"
GENERATIONS="${3:-5}"
PREFLIGHT_DIR="${WORKDIR}.a100_preflight"
PREFLIGHT_REPORT="$PREFLIGHT_DIR/A100_PREFLIGHT_COMPLETE.json"

if [[ -e "$WORKDIR" || -L "$WORKDIR" ]]; then
  echo "ERROR: production workdir must not already exist (nothing is deleted or reused): $WORKDIR" >&2
  exit 2
fi

case "${ECSP_UNSAFE_BYPASS_REQUIRED_A100_PREFLIGHT:-0}" in
  0|"") BYPASS_PREFLIGHT=0 ;;
  1)
    BYPASS_PREFLIGHT=1
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "WARNING: REQUIRED A100 PREFLIGHT WAS EXPLICITLY BYPASSED." >&2
    echo "WARNING: this run is non-certifying; the bypass is recorded in run metadata." >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    ;;
  *)
    echo "ERROR: ECSP_UNSAFE_BYPASS_REQUIRED_A100_PREFLIGHT must be 0 or 1." >&2
    exit 2
    ;;
esac

if [[ "$BYPASS_PREFLIGHT" -eq 0 ]]; then
  bash "$ROOT/tools/run_bc_global_a100_preflight.sh" "$PREFLIGHT_DIR"
  python "$ROOT/python/a100_device_gate.py" verify-preflight \
    --report "$PREFLIGHT_REPORT" \
    --expected-kind corrected_v8_2_python_bc_global
fi

# Close the time-of-check/time-of-use window before the production launcher.
if [[ -e "$WORKDIR" || -L "$WORKDIR" ]]; then
  echo "ERROR: production workdir appeared during preflight; refusing to reuse it: $WORKDIR" >&2
  exit 2
fi

set +e
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --config "$ROOT/config/nsga2_bc_global_preflame_propagation_a100.yaml" \
  --workdir "$WORKDIR" \
  --package-root "$ROOT" \
  --population-size "$POPULATION" \
  --generations "$GENERATIONS"
RUN_STATUS=$?
set -e

if [[ -d "$WORKDIR" ]]; then
  RECORD_ARGS=(
    record
    --workdir "$WORKDIR"
    --launcher tools/run_bc_global_a100.sh
    --launcher-exit-status "$RUN_STATUS"
  )
  if [[ "$BYPASS_PREFLIGHT" -eq 1 ]]; then
    RECORD_ARGS+=(--unsafe-bypass)
  else
    RECORD_ARGS+=(
      --preflight-report "$PREFLIGHT_REPORT"
      --expected-kind corrected_v8_2_python_bc_global
    )
  fi
  if ! python "$ROOT/python/a100_device_gate.py" "${RECORD_ARGS[@]}"; then
    echo "ERROR: failed to record A100 production-gate metadata." >&2
    if [[ "$RUN_STATUS" -eq 0 ]]; then
      exit 3
    fi
  fi
elif [[ "$RUN_STATUS" -eq 0 ]]; then
  echo "ERROR: successful production launcher did not create its workdir." >&2
  exit 3
fi

exit "$RUN_STATUS"
