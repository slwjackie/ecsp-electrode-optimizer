#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
OUT="${1:-$ROOT/runs/native_a100_preflight}"

if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo "ERROR: preflight output path already exists; choose a new path (nothing is deleted): $OUT" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"

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

python "$ROOT/python/a100_device_gate.py" check \
  --output "$OUT/a100_device_report.json" \
  "${DEVICE_ARGS[@]}"
python "$ROOT/python/check_bc_native_cuda.py" \
  --output "$OUT/cuda_build.json" \
  --require-compile
python - "$OUT/cuda_build.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("compile_status") != "compiled_and_loaded":
    raise SystemExit(f"CUDA extension was not compiled and loaded: {report.get('compile_status')!r}")
if report.get("extension_has_cuda") is not True:
    raise SystemExit("Loaded native extension does not expose CUDA entrypoints")
PY

cd "$ROOT"
python -m pytest -q python/tests/test_bc_native.py -k 'cuda' \
  --junitxml="$OUT/cuda_parity.xml"
python - "$OUT/cuda_parity.xml" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root = ET.parse(Path(sys.argv[1])).getroot()
suites = [root] if root.tag.endswith("testsuite") else [
    child for child in root if child.tag.endswith("testsuite")
]
counts = {
    key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
test_ids = sorted(
    f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
    for case in root.iter("testcase")
)
expected_ids = [
    "python.tests.test_bc_native::test_native_cuda32_64_batch_size_invariance",
    "python.tests.test_bc_native::test_native_cuda_cpu_full_field_and_mixed_voltage_parity",
]
if (
    counts["tests"] != len(expected_ids)
    or any(counts[key] for key in ("failures", "errors", "skipped"))
    or test_ids != expected_ids
):
    raise SystemExit(
        "CUDA parity/batch suite did not execute the exact required tests: "
        f"counts={counts}, test_ids={test_ids}"
    )
PY

# Exercise the corrected v8.2 scheduler with real short-horizon B/C work on
# both the CUDA owner and one spawned standalone CPU worker. This is a
# functional smoke test, not a production throughput/VRAM benchmark.
HYBRID_CONFIG="$OUT/bc_native_hybrid_debug.yaml"
HYBRID_WORK="$OUT/hybrid_smoke"
if [[ -e "$HYBRID_WORK" || -L "$HYBRID_WORK" ]]; then
  echo "ERROR: hybrid preflight workdir already exists: $HYBRID_WORK" >&2
  exit 2
fi
python - "$ROOT/config/nsga2_bc_global_native_debug.yaml" "$HYBRID_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
cfg["project"]["name"] = "ECSP_v8_2_A100_CPU_hybrid_preflight"
cfg["project"]["device"] = "cuda"
cfg["optimization"].update(
    population_size=4,
    generations=1,
    initial_topologies=2,
    variants_per_topology=2,
    physics_batch_size=2,
)
cfg["evaluator"]["backend"] = "bc_global_native_hybrid"
cfg["evaluator"]["device"] = "cuda"
cfg["evaluator"]["internal_batch_size"] = 2
cfg["evaluator"]["base_overrides"]["numerics"]["physicsDevice"] = "cuda"
cfg["evaluator"]["native"].update(
    cpu_budget=3,
    host_reserve=1,
    cpu_workers=1,
    cpu_threads_per_worker=1,
    cpu_runtime="standalone",
)
cfg["propagation_refinement"]["execution"]["parallel_cases"] = 1
target.write_text(
    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
PY
python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
  --package-root "$ROOT" \
  --config "$HYBRID_CONFIG" \
  --workdir "$HYBRID_WORK" \
  --population-size 4 \
  --generations 1 \
  --allow-no-feasible
python - "$HYBRID_WORK" "$OUT/hybrid_smoke_report.json" <<'PY'
from pathlib import Path
import json
import sys

work = Path(sys.argv[1])
schedule = work / "adapter" / "hybrid_schedule.jsonl"
if not schedule.is_file():
    raise SystemExit("Corrected hybrid scheduler did not write a schedule log")
rows = [
    json.loads(line)
    for line in schedule.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
devices = {str(row.get("device")) for row in rows}
if not {"cpu", "cuda"}.issubset(devices):
    raise SystemExit(
        f"Corrected hybrid preflight did not use both CPU and CUDA: {sorted(devices)}"
    )
complete = json.loads((work / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
report = {
    "status": "passed",
    "scope": "corrected_v8_2_short_horizon_functional_hybrid_smoke_not_performance_benchmark",
    "devices_observed": sorted(devices),
    "schedule_records": len(rows),
    "elapsed_s": float(complete["elapsed_s"]),
    "peak_vram_not_measured": True,
    "production_throughput_not_measured": True,
}
Path(sys.argv[2]).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

python - "$OUT" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
device = json.loads((root / "a100_device_report.json").read_text(encoding="utf-8"))
build = json.loads((root / "cuda_build.json").read_text(encoding="utf-8"))
hybrid = json.loads((root / "hybrid_smoke_report.json").read_text(encoding="utf-8"))
xml_root = ET.parse(root / "cuda_parity.xml").getroot()
suites = [xml_root] if xml_root.tag.endswith("testsuite") else [
    child for child in xml_root if child.tag.endswith("testsuite")
]
counts = {
    key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
test_ids = sorted(
    f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
    for case in xml_root.iter("testcase")
)
verified = device.get("a100_device_identity_verified") is True
override = device.get("non_certifying_override_used") is True
if verified == override:
    raise SystemExit("Device report has inconsistent A100/override status")
summary = {
    "schema": "ecsp.a100-production-gate/v1",
    "report_type": "completed_functional_preflight",
    "preflight_kind": "corrected_v8_2_native_hybrid",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "functional_preflight_passed": True,
    "a100_device_identity_verified": verified,
    "non_certifying_override_used": override,
    "device_report": "a100_device_report.json",
    "cuda_extension_compiled_and_loaded": build.get("compile_status") == "compiled_and_loaded",
    "cuda_extension_has_cuda_entrypoints": build.get("extension_has_cuda") is True,
    "cuda_test_counts": counts,
    "cuda_test_ids": test_ids,
    "cpu_and_cuda_hybrid_observed": set(hybrid.get("devices_observed", [])) >= {"cpu", "cuda"},
    "production_throughput_measured": False,
    "peak_vram_measured": False,
    "performance_or_peak_vram_certified": False,
}
(root / "A100_PREFLIGHT_COMPLETE.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
python "$ROOT/python/a100_device_gate.py" verify-preflight \
  --report "$OUT/A100_PREFLIGHT_COMPLETE.json" \
  --expected-kind corrected_v8_2_native_hybrid
