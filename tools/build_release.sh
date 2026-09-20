#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
PYTHON_BIN="${PYTHON:-python}"
EXPECTED_NAME="$($PYTHON_BIN - <<'PY' "$ROOT/VERSION.json"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["release_directory_name"])
PY
)"
if [[ "$NAME" != "$EXPECTED_NAME" ]]; then
  echo "Release root must be named $EXPECTED_NAME (received $NAME)" >&2
  exit 2
fi

BASELINE_ZIP="${ECSP_V820_BASELINE_ZIP:-}"
if [[ -z "$BASELINE_ZIP" || ! -f "$BASELINE_ZIP" || -L "$BASELINE_ZIP" ]]; then
  echo "ECSP_V820_BASELINE_ZIP must name the immutable regular-file v8.2.0 ZIP" >&2
  exit 2
fi

VERSION="$($PYTHON_BIN - <<'PY' "$ROOT/VERSION.json"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["version"])
PY
)"
MANIFEST_TAG="$($PYTHON_BIN - <<'PY' "$ROOT/VERSION.json"
import json, sys
parts = str(json.load(open(sys.argv[1], encoding="utf-8"))["version"]).split("-", 1)[0].split(".")
if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
    raise SystemExit("VERSION.json version must begin with numeric major.minor.patch")
# Preserve the historical v8.2.0 V8_2 name.  Patch releases include the patch
# component so their authoritative manifest cannot replace the baseline one.
tag_parts = parts[:2] if int(parts[2]) == 0 else parts[:3]
print("V" + "_".join(tag_parts))
PY
)"
SCHEMA="$($PYTHON_BIN - <<'PY' "$ROOT/VERSION.json"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["geometry_schema_version"])
PY
)"

ZIP="$PARENT/${NAME}.zip"
ZIP_CHECKSUM="$ZIP.sha256.txt"
if [[ -e "$ZIP" || -L "$ZIP" || -e "$ZIP_CHECKSUM" || -L "$ZIP_CHECKSUM" ]]; then
  echo "Release archive target already exists; refusing to overwrite:" >&2
  echo "  $ZIP" >&2
  echo "  $ZIP_CHECKSUM" >&2
  exit 2
fi

VALIDATION_OUT="${ECSP_RELEASE_VALIDATION_OUT:-$PARENT/${NAME}_release_validation}"
$PYTHON_BIN - <<'PY' "$ROOT" "$VALIDATION_OUT"
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
if out.exists() or out.is_symlink():
    raise SystemExit(f"Release validation output must be a new path: {out}")
resolved = out.resolve(strict=False)
if resolved == root or resolved.is_relative_to(root):
    raise SystemExit("Release validation output must be outside the package tree")
PY

# A final archive may not be created from pending claims, cached binaries, runtime
# output, or symbolic links.  The build operates on a disposable staging tree and
# fails instead of mutating user data to clean it.
$PYTHON_BIN - <<'PY' "$ROOT"
from pathlib import Path
import json, sys

root = Path(sys.argv[1]).resolve()
version = json.loads((root / "VERSION.json").read_text(encoding="utf-8"))
pending = []
def walk(value, path="test_status"):
    if isinstance(value, dict):
        for key, child in value.items():
            walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk(child, f"{path}[{index}]")
    elif isinstance(value, str) and "pending_final" in value.lower():
        pending.append(f"{path}={value}")
walk(version.get("test_status", {}))
if pending:
    raise SystemExit("Final validation metadata is still pending: " + "; ".join(pending))

report = (root / "docs" / "V8_2_1_VALIDATION_REPORT_KR.md").read_text(encoding="utf-8")
if "| pending final" in report.lower():
    raise SystemExit("Validation report still contains pending-final result rows")

forbidden_dirs = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "runs", "build", ".native_build"}
bad = []
for path in root.rglob("*"):
    rel = path.relative_to(root).as_posix()
    if path.is_symlink():
        bad.append(f"symlink:{rel}")
    elif path.is_dir() and path.name in forbidden_dirs:
        bad.append(f"directory:{rel}")
    elif path.is_file() and path.suffix in {".pyc", ".pyo"}:
        bad.append(f"bytecode:{rel}")
if bad:
    raise SystemExit("Release staging tree contains forbidden artifacts: " + ", ".join(sorted(bad)))
PY

MANIFEST="$ROOT/PACKAGE_MANIFEST.txt"
SHA_MANIFEST="$ROOT/PACKAGE_SHA256_MANIFEST_${MANIFEST_TAG}.txt"
GENERIC_SHA_MANIFEST="$ROOT/PACKAGE_SHA256_MANIFEST.txt"
$PYTHON_BIN - <<'PY' "$ROOT" "$MANIFEST" "$SHA_MANIFEST" "$GENERIC_SHA_MANIFEST" "$VERSION" "$SCHEMA"
from __future__ import annotations
import datetime
import hashlib
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2]).resolve()
sha_manifest = Path(sys.argv[3]).resolve()
generic_sha_manifest = Path(sys.argv[4]).resolve()
version = sys.argv[5]
schema = sys.argv[6]

def included(path: Path) -> bool:
    rel = path.relative_to(root)
    if not path.is_file() or path.is_symlink() or path.name == ".DS_Store":
        return False
    if path in {sha_manifest, generic_sha_manifest}:
        return False
    excluded = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "runs", "build", ".native_build"}
    if any(part in excluded or part.startswith("tmp_") or part.startswith(".tmp_") for part in rel.parts):
        return False
    return path.suffix not in {".pyc", ".pyo"}

def rows(exclude: set[Path]) -> list[str]:
    result = []
    for path in sorted(root.rglob("*")):
        if path in exclude or not included(path):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result.append(f"{digest}  {path.relative_to(root).as_posix()}")
    return result

def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".tmp_{path.name}.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)

stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
body = [
    "ECSP release manifest",
    f"Version: {version}",
    f"Geometry schema: {schema}",
    "MATLAB required: false",
    "Current authoritative SHA manifest and generic alias excluded from payload inventory: true",
    f"Generated UTC: {stamp}",
    "",
    "SHA256  relative_path",
    *rows({manifest}),
]
atomic_write(manifest, "\n".join(body) + "\n")
sha_body = "\n".join(rows(set())) + "\n"
atomic_write(sha_manifest, sha_body)
atomic_write(generic_sha_manifest, sha_body)
PY

# The normal (not prepackage) validator now sees the exact manifest-bearing
# tree that will be archived.  It reruns the full CPU suite, native builds,
# strict parity, CUDA source checks, executable pipeline audit,
# immutable-v8.2.0 preservation audit, and release-manifest verification.
ECSP_V820_BASELINE_ZIP="$BASELINE_ZIP" \
  bash "$ROOT/tools/validate_v8_2_1_cpu.sh" "$VALIDATION_OUT"
$PYTHON_BIN - <<'PY' "$VALIDATION_OUT/VALIDATION_COMPLETE.json" "$ROOT"
from pathlib import Path
import hashlib, json, sys
path = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
if not path.is_file():
    raise SystemExit(f"Validator did not create completion evidence: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "passed" or payload.get("mode") != "release":
    raise SystemExit(f"Release validation did not pass: {payload!r}")
evidence_root = path.parent.resolve()
evidence_rows = payload.get("evidence")
expected_evidence = {
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
}
if not isinstance(evidence_rows, list) or {
    str(row.get("path", "")) for row in evidence_rows if isinstance(row, dict)
} != expected_evidence:
    raise SystemExit("Validator completion marker has an incomplete evidence inventory")
for row in evidence_rows:
    relative = Path(str(row.get("path", "")))
    unresolved_evidence = evidence_root / relative
    evidence = unresolved_evidence.resolve()
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or not evidence.is_relative_to(evidence_root)
        or not evidence.is_file()
        or unresolved_evidence.is_symlink()
    ):
        raise SystemExit(f"Unsafe/missing validation evidence: {relative}")
    observed = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if observed != row.get("sha256"):
        raise SystemExit(f"Validation evidence hash mismatch: {relative}")
validated = payload.get("validated_package", {})
current_hashes = {
    "version_json_sha256": hashlib.sha256(
        (root / "VERSION.json").read_bytes()
    ).hexdigest(),
    "expected_changes_sha256": hashlib.sha256(
        (root / "docs" / "V821_EXPECTED_CHANGES.json").read_bytes()
    ).hexdigest(),
    "checked_in_pipeline_audit_sha256": hashlib.sha256(
        (root / "docs" / "generated_bc_audit" / "bc_global_pipeline_audit.json").read_bytes()
    ).hexdigest(),
    "authoritative_manifest_sha256": hashlib.sha256(
        (root / "PACKAGE_SHA256_MANIFEST_V8_2_1.txt").read_bytes()
    ).hexdigest(),
}
if validated != current_hashes:
    raise SystemExit(
        "Package metadata/manifest changed after validation: "
        f"validated={validated!r}, current={current_hashes!r}"
    )
PY

TEMP_RELEASE_DIR="$(mktemp -d "$PARENT/.${NAME}.release.XXXXXX")"
trap 'rm -rf -- "$TEMP_RELEASE_DIR"' EXIT
TEMP_ZIP="$TEMP_RELEASE_DIR/${NAME}.zip"
TEMP_CHECKSUM="$TEMP_RELEASE_DIR/${NAME}.zip.sha256.txt"
VERIFY_DIR="$TEMP_RELEASE_DIR/extracted"

cd "$PARENT"
zip -qr "$TEMP_ZIP" "$NAME" \
  -x '*/.git/*' '*/.pytest_cache/*' '*/.ruff_cache/*' '*/__pycache__/*' \
     '*/runs/*' '*/build/*' '*/.native_build/*' '*/tmp_*' '*/tmp_*/*' \
     '*/.tmp_*' '*/.tmp_*/*' '*.pyc' '*.pyo' '*/.DS_Store'
unzip -tq "$TEMP_ZIP"
mkdir "$VERIFY_DIR"
unzip -q "$TEMP_ZIP" -d "$VERIFY_DIR"
$PYTHON_BIN - <<'PY' "$VERIFY_DIR" "$NAME"
from pathlib import Path
import sys
root = Path(sys.argv[1])
expected = sys.argv[2]
entries = sorted(path.name for path in root.iterdir())
if entries != [expected] or not (root / expected).is_dir():
    raise SystemExit(f"Archive must contain exactly one top-level {expected!r} directory: {entries!r}")
PY
$PYTHON_BIN "$VERIFY_DIR/$NAME/python/verify_release_manifest.py" \
  --package-root "$VERIFY_DIR/$NAME" \
  --output "$VALIDATION_OUT/fresh_extract_manifest_verification.json"
$PYTHON_BIN - <<'PY' "$TEMP_ZIP" "$TEMP_CHECKSUM"
from pathlib import Path
import hashlib, sys
archive = Path(sys.argv[1])
Path(sys.argv[2]).write_text(
    f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
    encoding="utf-8",
)
PY

# Publish without overwriting: hard links provide an atomic no-clobber operation
# on the shared parent filesystem.  If the second link loses a race, only the
# first link created by this process is rolled back.
$PYTHON_BIN - <<'PY' "$TEMP_ZIP" "$TEMP_CHECKSUM" "$ZIP" "$ZIP_CHECKSUM"
from pathlib import Path
import os, sys
source_zip, source_sum, target_zip, target_sum = map(Path, sys.argv[1:])
created_zip = False
try:
    os.link(source_zip, target_zip)
    created_zip = True
    os.link(source_sum, target_sum)
except Exception:
    if created_zip:
        try:
            if target_zip.stat().st_ino == source_zip.stat().st_ino:
                target_zip.unlink()
        except FileNotFoundError:
            pass
    raise
PY

$PYTHON_BIN - <<'PY' "$ROOT" "$BASELINE_ZIP" "$ZIP" "$ZIP_CHECKSUM" "$VALIDATION_OUT"
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, sys

root, baseline, archive, checksum, out = map(Path, sys.argv[1:])
completion = out / "RELEASE_BUILD_COMPLETE.json"
validation = out / "VALIDATION_COMPLETE.json"
payload = {
    "schema": "ecsp.release-build/v1",
    "status": "passed",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "release_directory_name": root.name,
    "release_version": json.loads((root / "VERSION.json").read_text(encoding="utf-8"))["version"],
    "archive_name": archive.name,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "checksum_file": checksum.name,
    "baseline_zip_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
    "validation_complete_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
    "fresh_extract_manifest_verification": "fresh_extract_manifest_verification.json",
}
temporary = completion.with_name(f".{completion.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, completion)
PY

echo "$ZIP"
echo "$ZIP_CHECKSUM"
echo "$VALIDATION_OUT/RELEASE_BUILD_COMPLETE.json"
