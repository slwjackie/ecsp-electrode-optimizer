#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/v8_2_cpu_validation}"
PYTHON_BIN="${PYTHON:-python}"
MODE="release"
if [[ "${ECSP_V82_PREPACKAGE:-0}" == "1" ]]; then
  MODE="prepackage"
fi

BASELINE_ZIP="${ECSP_V810_BASELINE_ZIP:-}"
if [[ -z "$BASELINE_ZIP" || ! -f "$BASELINE_ZIP" || -L "$BASELINE_ZIP" ]]; then
  echo "ECSP_V810_BASELINE_ZIP must name the immutable v8.1.0 ZIP regular file" >&2
  exit 2
fi
BASELINE_ZIP="$($PYTHON_BIN - <<'PY' "$BASELINE_ZIP"
from pathlib import Path
import sys

path = Path(sys.argv[1]).resolve(strict=True)
if not path.is_file():
    raise SystemExit(f"Baseline ZIP is not a regular file: {path}")
print(path)
PY
)"

# Validation evidence is meaningful only when it is assembled by this exact
# invocation.  Refuse every pre-existing path, including a dangling symlink,
# rather than mixing new results with stale files.
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo "Validation output must be a new, nonexistent path: $OUT" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
OUT="$(cd "$OUT" && pwd -P)"

export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$OUT/python_cache"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

$PYTHON_BIN - <<'PY' "$OUT/environment.json" "$MODE" "$BASELINE_ZIP"
from __future__ import annotations
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import torch

baseline = Path(sys.argv[3])
payload = {
    "mode": sys.argv[2],
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_device_available": torch.cuda.is_available(),
    "mps_built": bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()
    ),
    "mps_available": bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ),
    "compiler": shutil.which("c++"),
    "nvcc": shutil.which("nvcc"),
    "baseline_zip_name": baseline.name,
    "baseline_zip_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

$PYTHON_BIN - <<'PY' "$ROOT/python" "$OUT/python_compile.json"
from __future__ import annotations
import compileall
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
sources = sorted(
    path for path in source.rglob("*.py")
    if not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts)
)
passed = compileall.compile_dir(source, quiet=1, force=True)
payload = {
    "status": "passed" if passed else "failed",
    "python_source_count": len(sources),
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if not passed:
    raise SystemExit("Python compileall failed")
PY

SHELL_SCRIPT_COUNT=0
while IFS= read -r script; do
  bash -n "$script"
  SHELL_SCRIPT_COUNT=$((SHELL_SCRIPT_COUNT + 1))
done < <(find "$ROOT/tools" -type f -name '*.sh' -print | sort)
bash -n "$ROOT/RUN_NSGA2_ONLY.sh"
SHELL_SCRIPT_COUNT=$((SHELL_SCRIPT_COUNT + 1))
$PYTHON_BIN - <<'PY' "$OUT/shell_syntax.json" "$SHELL_SCRIPT_COUNT"
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    json.dumps(
        {"status": "passed", "shell_script_count": int(sys.argv[2])}, indent=2
    ) + "\n",
    encoding="utf-8",
)
PY

$PYTHON_BIN -c "from ecsp_native import load_native; from ecsp_native.standalone import build_standalone; load_native(False, True); build_standalone(True)"
$PYTHON_BIN - <<'PY' "$OUT/native_cpu_build.json"
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "passed",
            "cpu_extension_compiled_and_loaded": True,
            "standalone_executable_compiled": True,
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY

cd "$ROOT"
$PYTHON_BIN -m pytest -q --durations=25 \
  --junitxml="$OUT/full_suite.xml" | tee "$OUT/full_suite.log"
$PYTHON_BIN "$ROOT/python/compare_bc_native_python.py" \
  --output "$OUT/native_python_parity_strict.json" --strict-common-tolerances
$PYTHON_BIN "$ROOT/python/check_bc_native_cuda.py" \
  --output "$OUT/cuda_readiness.json"

# Execute the end-to-end B/C pipeline into a fresh runtime directory and keep
# its complete evidence outside the package's checked-in audit directory.
$PYTHON_BIN "$ROOT/python/validate_bc_global_pipeline.py" \
  --package-root "$ROOT" \
  --output-dir "$OUT/pipeline_audit" \
  --runtime-workdir "$OUT/pipeline_runtime" \
  | tee "$OUT/pipeline_audit.log"

# A freshly passing 28/28 run is necessary but not sufficient: compare the
# semantic check payload to the checked-in release evidence so stale checked-in
# claims cannot survive.  Only host/run volatility is removed.
$PYTHON_BIN - <<'PY' \
  "$OUT/pipeline_audit/bc_global_pipeline_audit.json" \
  "$ROOT/docs/generated_bc_audit/bc_global_pipeline_audit.json" \
  "$OUT/pipeline_semantic_comparison.json"
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
from typing import Any

fresh_path, checked_path, output_path = map(Path, sys.argv[1:])
fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
checked = json.loads(checked_path.read_text(encoding="utf-8"))

volatile_keys = {
    "elapsed_s",
    "created_at_utc",
    "started_at_utc",
    "completed_at_utc",
    "generated_at_utc",
    "timestamp",
    "path",
    "package_root",
    "runtime_workdir",
    "workdir",
    "output_dir",
}

def normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalise(child)
            for key, child in sorted(value.items())
            if key not in volatile_keys and not key.endswith("_absolute_path")
        }
    if isinstance(value, list):
        return [normalise(child) for child in value]
    if isinstance(value, str):
        # Sanitisation in the pipeline tool handles normal paths.  This catches
        # platform-specific temporary roots in older checked-in evidence.
        value = re.sub(r"/(?:private/)?tmp/[^\s\"']+", "<TEMP_PATH>", value)
        return value
    return value

def checks_by_id(payload: dict[str, Any], label: str) -> dict[str, Any]:
    rows = payload.get("checks")
    if not isinstance(rows, list):
        raise SystemExit(f"{label} audit has no check list")
    result: dict[str, Any] = {}
    for row in rows:
        key = row.get("id") if isinstance(row, dict) else None
        if not isinstance(key, str) or key in result:
            raise SystemExit(f"{label} audit has an invalid/duplicate check id: {key!r}")
        result[key] = normalise(row)
    return result

fresh_checks = checks_by_id(fresh, "fresh")
checked_checks = checks_by_id(checked, "checked-in")
fresh_passed = (
    fresh.get("status") == "passed"
    and fresh.get("checks_passed") == 28
    and fresh.get("checks_total") == 28
    and len(fresh_checks) == 28
    and all(row.get("passed") is True for row in fresh_checks.values())
)
checked_passed = (
    checked.get("status") == "passed"
    and checked.get("checks_passed") == 28
    and checked.get("checks_total") == 28
    and len(checked_checks) == 28
    and all(row.get("passed") is True for row in checked_checks.values())
)
all_ids = sorted(set(fresh_checks) | set(checked_checks))
mismatched = [
    key for key in all_ids if fresh_checks.get(key) != checked_checks.get(key)
]
report = {
    "status": (
        "passed" if fresh_passed and checked_passed and not mismatched else "failed"
    ),
    "fresh_checks_passed": fresh.get("checks_passed"),
    "fresh_checks_total": fresh.get("checks_total"),
    "checked_in_checks_passed": checked.get("checks_passed"),
    "checked_in_checks_total": checked.get("checks_total"),
    "ignored_volatile_keys": sorted(volatile_keys),
    "mismatched_check_ids": mismatched,
}
output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if report["status"] != "passed":
    raise SystemExit(
        "Fresh and checked-in B/C pipeline audits are not the same passing "
        f"28/28 semantic check set: {mismatched}"
    )
PY

AUDIT_ARGS=(
  --baseline-zip "$BASELINE_ZIP"
  --output-dir "$OUT/preservation"
  --require-declared-changes
)
if [[ "$MODE" == "prepackage" ]]; then
  AUDIT_ARGS+=(--allow-generated-release-artifacts-missing)
fi
$PYTHON_BIN "$ROOT/python/audit_v810_preservation.py" "${AUDIT_ARGS[@]}"

if [[ "$MODE" == "release" ]]; then
  if [[ ! -f "$ROOT/PACKAGE_SHA256_MANIFEST_V8_2.txt" ]]; then
    echo "Missing authoritative PACKAGE_SHA256_MANIFEST_V8_2.txt" >&2
    exit 2
  fi
  $PYTHON_BIN "$ROOT/python/verify_release_manifest.py" \
    --package-root "$ROOT" \
    --output "$OUT/release_manifest_verification.json"
else
  $PYTHON_BIN - <<'PY' "$OUT/release_manifest_verification.json"
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "deferred_prepackage",
            "reason": "authoritative manifest is generated only after pre-package validation",
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
fi

# This marker is written last and atomically.  Its hashes bind the completion
# claim to the exact evidence files produced by this invocation.
$PYTHON_BIN - <<'PY' "$OUT" "$MODE" "$ROOT"
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

out = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
package_root = Path(sys.argv[3]).resolve()

def load(relative: str) -> dict:
    return json.loads((out / relative).read_text(encoding="utf-8"))

def digest(relative: str) -> dict[str, str]:
    path = out / relative
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

xml_root = ET.parse(out / "full_suite.xml").getroot()
xml_suites = (
    [xml_root]
    if xml_root.tag.rsplit("}", 1)[-1] == "testsuite"
    else list(xml_root.findall(".//testsuite"))
)
pytest_counts = {
    key: sum(int(float(suite.attrib.get(key, 0))) for suite in xml_suites)
    for key in ("tests", "failures", "errors", "skipped")
}
testcases = [
    element
    for element in xml_root.iter()
    if element.tag.rsplit("}", 1)[-1] == "testcase"
]
skipped_test_ids = sorted(
    f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
    for case in testcases
    if any(child.tag.rsplit("}", 1)[-1] == "skipped" for child in case)
)
expected_no_cuda_skips = [
    "python.tests.test_bc_native::test_native_cuda32_64_batch_size_invariance",
    "python.tests.test_bc_native::test_native_cuda_cpu_full_field_and_mixed_voltage_parity",
]
parity = load("native_python_parity_strict.json")
parity_comparisons = sum(
    len(case.get("comparisons", [])) for case in parity.get("cases", [])
)
parity_failures = sum(
    int(not comparison.get("passed", False))
    for case in parity.get("cases", [])
    for comparison in case.get("comparisons", [])
)
cuda = load("cuda_readiness.json")
pipeline = load("pipeline_audit/bc_global_pipeline_audit.json")
pipeline_comparison = load("pipeline_semantic_comparison.json")
preservation = load("preservation/audit.json")
manifest = load("release_manifest_verification.json")
python_compile = load("python_compile.json")
shell_syntax = load("shell_syntax.json")
native_build = load("native_cpu_build.json")
environment = load("environment.json")

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)

require(pytest_counts["tests"] > 0, "Pytest evidence contains no tests")
require(
    pytest_counts["failures"] == 0 and pytest_counts["errors"] == 0,
    f"Pytest evidence records failures/errors: {pytest_counts}",
)
if environment.get("cuda_device_available") is True:
    require(
        pytest_counts["skipped"] == 0 and not skipped_test_ids,
        f"CUDA-capable validation unexpectedly skipped tests: {skipped_test_ids}",
    )
else:
    require(
        pytest_counts["skipped"] == len(expected_no_cuda_skips)
        and skipped_test_ids == expected_no_cuda_skips,
        "A non-CUDA validation may skip exactly the two declared CUDA tests; "
        f"observed {skipped_test_ids}",
    )
require(
    parity.get("status") == "passed"
    and len(parity.get("cases", [])) > 0
    and parity_comparisons > 0
    and parity_failures == 0,
    "Strict Python/native parity did not pass",
)
require(
    cuda.get("all_source_static_checks_passed") is True
    and len(cuda.get("source_static_checks", {})) > 0
    and all(cuda.get("source_static_checks", {}).values()),
    "CUDA source static checks did not all pass",
)
require(pipeline.get("status") == "passed", "Pipeline audit did not pass")
require(
    pipeline.get("checks_passed") == pipeline.get("checks_total") == 28,
    "Pipeline audit is not a passing 28/28 result",
)
require(
    pipeline_comparison.get("status") == "passed",
    "Fresh pipeline checks do not match checked-in evidence",
)
require(preservation.get("status") == "passed", "v8.1 preservation audit failed")
require(
    preservation.get("baseline_zip_verification", {}).get("status")
    == "verified",
    "Immutable baseline ZIP was not verified",
)
require(
    preservation.get("unsupported_symlink_count") == 0,
    "Preservation audit found unsupported symbolic links",
)
require(python_compile.get("status") == "passed", "Python compileall failed")
require(shell_syntax.get("status") == "passed", "Shell syntax checks failed")
require(native_build.get("status") == "passed", "Native CPU builds failed")
require(
    int(python_compile.get("python_source_count", 0)) > 0,
    "Python compile evidence has no source files",
)
require(
    int(shell_syntax.get("shell_script_count", 0)) > 0,
    "Shell syntax evidence has no scripts",
)
if mode == "release":
    require(manifest.get("status") == "passed", "Release manifest verification failed")
    require(
        preservation.get("deferred_declared_change_count") == 0,
        "Release validation may not defer generated artifacts",
    )
else:
    require(
        manifest.get("status") == "deferred_prepackage",
        "Unexpected pre-package manifest-verification state",
    )
    require(
        preservation.get("deferred_declared_change_count") == 3,
        "Pre-package validation must defer exactly three generated artifacts",
    )

evidence_names = [
    "environment.json",
    "python_compile.json",
    "shell_syntax.json",
    "native_cpu_build.json",
    "full_suite.xml",
    "full_suite.log",
    "native_python_parity_strict.json",
    "cuda_readiness.json",
    "pipeline_audit/bc_global_pipeline_audit.json",
    "pipeline_semantic_comparison.json",
    "preservation/audit.json",
    "release_manifest_verification.json",
]
payload = {
    "schema": "ecsp.v8.2-cpu-validation/v1",
    "status": "passed",
    "mode": mode,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "counts": {
        "python_sources_compiled": python_compile["python_source_count"],
        "shell_scripts_syntax_checked": shell_syntax["shell_script_count"],
        "pytest": pytest_counts,
        "pytest_skipped_test_ids": skipped_test_ids,
        "native_python_parity_cases": len(parity.get("cases", [])),
        "native_python_parity_comparisons": parity_comparisons,
        "native_python_parity_failures": parity_failures,
        "cuda_static_checks_passed": sum(
            int(value) for value in cuda.get("source_static_checks", {}).values()
        ),
        "cuda_static_checks_total": len(cuda.get("source_static_checks", {})),
        "pipeline_checks_passed": pipeline["checks_passed"],
        "pipeline_checks_total": pipeline["checks_total"],
        "preservation_byte_identical_files": preservation["byte_identical_count"],
        "preservation_modified_files": preservation["modified_count"],
        "preservation_added_files": preservation["added_count"],
        "preservation_deleted_files": preservation["deleted_count"],
        "prepackage_deferred_release_artifacts": preservation[
            "deferred_declared_change_count"
        ],
    },
    "results": {
        "cpu_native_build": native_build["status"],
        "python_native_parity": parity["status"],
        "cuda_source_static": (
            "passed" if cuda["all_source_static_checks_passed"] else "failed"
        ),
        "cuda_compile": cuda.get("compile_status"),
        "pipeline_audit": pipeline["status"],
        "pipeline_checked_in_semantic_match": pipeline_comparison["status"],
        "v810_preservation": preservation["status"],
        "baseline_zip_verification": preservation["baseline_zip_verification"][
            "status"
        ],
        "release_manifest": manifest["status"],
    },
    "evidence": [digest(relative) for relative in evidence_names],
    "validated_package": {
        "version_json_sha256": hashlib.sha256(
            (package_root / "VERSION.json").read_bytes()
        ).hexdigest(),
        "expected_changes_sha256": hashlib.sha256(
            (package_root / "docs" / "V820_EXPECTED_CHANGES.json").read_bytes()
        ).hexdigest(),
        "checked_in_pipeline_audit_sha256": hashlib.sha256(
            (
                package_root
                / "docs"
                / "generated_bc_audit"
                / "bc_global_pipeline_audit.json"
            ).read_bytes()
        ).hexdigest(),
        "authoritative_manifest_sha256": (
            hashlib.sha256(
                (package_root / "PACKAGE_SHA256_MANIFEST_V8_2.txt").read_bytes()
            ).hexdigest()
            if mode == "release"
            else None
        ),
    },
}
target = out / "VALIDATION_COMPLETE.json"
temporary = out / f".{target.name}.tmp-{os.getpid()}"
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(temporary, target)
PY

echo "CPU validation completed: $OUT"
