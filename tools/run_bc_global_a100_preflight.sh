#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/bc_global_a100_preflight}"

if [[ -e "$WORKDIR" || -L "$WORKDIR" ]]; then
  echo "ERROR: preflight workdir already exists; choose a new path (nothing is deleted): $WORKDIR" >&2
  exit 2
fi
mkdir -p "$(dirname "$WORKDIR")"
mkdir "$WORKDIR"

DEVICE_ARGS=()
case "${ECSP_ALLOW_NON_A100_NONCERTIFYING_PREFLIGHT:-0}" in
  0|"") ;;
  1)
    echo "WARNING: ECSP_ALLOW_NON_A100_NONCERTIFYING_PREFLIGHT=1" >&2
    echo "WARNING: results from this device MUST NOT be described as A100 validation or certification." >&2
    DEVICE_ARGS+=(--allow-non-a100)
    ;;
  *)
    echo "ERROR: ECSP_ALLOW_NON_A100_NONCERTIFYING_PREFLIGHT must be 0 or 1." >&2
    exit 2
    ;;
esac

PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
python "$ROOT/python/a100_device_gate.py" check \
  --output "$WORKDIR/a100_device_report.json" \
  "${DEVICE_ARGS[@]}"

TMP_CFG="$(mktemp "${TMPDIR:-/tmp}/ecsp_bc_cuda_preflight.XXXXXX.yaml")"
trap 'rm -f "$TMP_CFG"' EXIT
python - "$ROOT/config/nsga2_bc_global_preflame_propagation_debug.yaml" "$TMP_CFG" <<'PY'
from pathlib import Path
import sys
import yaml

src, dst = (Path(value) for value in sys.argv[1:])
data = yaml.safe_load(src.read_text(encoding="utf-8"))
data["project"]["name"] = "ECSP_v8_2_BC_Global_CUDA_Preflight"
data["project"]["device"] = "cuda"
data["evaluator"]["device"] = "cuda"
data["evaluator"]["internal_batch_size"] = 4
data["evaluator"]["base_overrides"]["numerics"]["physicsDevice"] = "cuda"
data["evaluator"]["base_overrides"]["numerics"]["physicsDtype"] = "float64"
data["evaluator"]["base_overrides"]["numerics"]["coupledBatchSize"] = 4
data["propagation_refinement"]["execution"]["parallel_cases"] = 2
dst.write_text(
    yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
PY
PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --config "$TMP_CFG" \
  --workdir "$WORKDIR" \
  --package-root "$ROOT" \
  --allow-no-feasible
python - "$WORKDIR" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import sys

root = Path(sys.argv[1])
rows = list(csv.DictReader((root / "final" / "recommended_design_metrics.csv").open()))
if not rows or rows[0].get("physicsDevice") != "cuda":
    raise SystemExit("CUDA preflight did not report physicsDevice=cuda")
complete = json.loads((root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
if not complete.get("bc_global_preflame_used") or complete.get("gas_phase_cfd_used"):
    raise SystemExit("Unexpected B/C preflight scope")
device = json.loads((root / "a100_device_report.json").read_text(encoding="utf-8"))
verified = device.get("a100_device_identity_verified") is True
override = device.get("non_certifying_override_used") is True
if verified == override:
    raise SystemExit("Device report has inconsistent A100/override status")
summary = {
    "schema": "ecsp.a100-production-gate/v1",
    "report_type": "completed_functional_preflight",
    "preflight_kind": "corrected_v8_2_python_bc_global",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "functional_preflight_passed": True,
    "a100_device_identity_verified": verified,
    "non_certifying_override_used": override,
    "device_report": "a100_device_report.json",
    "physics_device_reported_by_workflow": "cuda",
    "bc_global_preflame_used": True,
    "gas_phase_cfd_used": False,
    "production_throughput_measured": False,
    "peak_vram_measured": False,
    "performance_or_peak_vram_certified": False,
}
(root / "A100_PREFLIGHT_COMPLETE.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
python "$ROOT/python/a100_device_gate.py" verify-preflight \
  --report "$WORKDIR/A100_PREFLIGHT_COMPLETE.json" \
  --expected-kind corrected_v8_2_python_bc_global
