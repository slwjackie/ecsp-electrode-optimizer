#!/usr/bin/env python3
"""Fail if v8.3 changes a v8.2.1 parent file outside the declared scope."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
IGNORED_NAMES = {
    ".DS_Store",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".native_build",
}
IGNORED_PACKAGE_FILES = {
    "PACKAGE_MANIFEST.txt",
    "PACKAGE_SHA256_MANIFEST.txt",
    "PACKAGE_SHA256_MANIFEST_V8_2_1.txt",
    "PACKAGE_SHA256_MANIFEST_V8_3_0.txt",
}
ALLOWED_MODIFIED = {
    "README_KR.md",
    "VERSION.json",
    "python/tests/test_v82_release_contract.py",
}
ALLOWED_ADDED_EXACT = {
    "README_V8_3_0_KR.md",
    "config/ecsp_extended_reactive_v8_3_0.yaml",
    "config/paper_faithful_assumed_v8_3_0.yaml",
    "config/paper_faithful_required_inputs_v8_3_0.yaml",
    "tools/audit_v821_parent_v830.py",
    "tools/benchmark_reactive_v8_3.py",
    "tools/build_reactive_v8_3_release.py",
    "tools/run_reactive_reference.sh",
    "tools/validate_reactive_v8_3.py",
}
ALLOWED_ADDED_PREFIXES = (
    "python/ecsp_reactive/",
    "python/tests/test_reactive_",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--candidate", default=ROOT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_NAMES for part in relative.parts):
            continue
        text = relative.as_posix()
        if text in IGNORED_PACKAGE_FILES or path.suffix == ".pyc":
            continue
        inventory[text] = hashlib.sha256(path.read_bytes()).hexdigest()
    return inventory


def _allowed_added(path: str) -> bool:
    if path in ALLOWED_ADDED_EXACT:
        return True
    if path.startswith("docs/") and "V8_3_0" in Path(path).name.upper():
        return True
    return any(path.startswith(prefix) for prefix in ALLOWED_ADDED_PREFIXES)


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
        Path(temporary_name).replace(destination)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def main() -> int:
    args = _args()
    parent = args.parent.expanduser().resolve(strict=True)
    candidate = args.candidate.expanduser().resolve(strict=True)
    parent_files = _inventory(parent)
    candidate_files = _inventory(candidate)
    shared = parent_files.keys() & candidate_files.keys()
    modified = sorted(path for path in shared if parent_files[path] != candidate_files[path])
    added = sorted(candidate_files.keys() - parent_files.keys())
    deleted = sorted(parent_files.keys() - candidate_files.keys())
    unexpected_modified = sorted(set(modified) - ALLOWED_MODIFIED)
    unexpected_added = sorted(path for path in added if not _allowed_added(path))
    unexpected_deleted = deleted
    status = (
        "PASS"
        if not unexpected_modified and not unexpected_added and not unexpected_deleted
        else "FAIL"
    )
    report: dict[str, object] = {
        "schema": "ecsp.v821-parent-preservation/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "parent": str(parent),
        "candidate": str(candidate),
        "parent_file_count_excluding_package_manifests": len(parent_files),
        "candidate_file_count_excluding_package_manifests": len(candidate_files),
        "modified": modified,
        "added": added,
        "deleted": deleted,
        "unexpected_modified": unexpected_modified,
        "unexpected_added": unexpected_added,
        "unexpected_deleted": unexpected_deleted,
        "scope": {
            "allowed_modified": sorted(ALLOWED_MODIFIED),
            "allowed_added_exact": sorted(ALLOWED_ADDED_EXACT),
            "allowed_added_prefixes": list(ALLOWED_ADDED_PREFIXES),
            "docs_rule": "new docs whose filename contains V8_3_0",
            "deletion_policy": "no parent payload deletion",
        },
    }
    _atomic_write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
