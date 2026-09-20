#!/usr/bin/env python3
"""Audit the v8.2 tree against the immutable v8.1.0 input release."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".native_build",
    "build",
    "runs",
}

# These are the only declared changes that cannot exist until the release
# manifest generation step.  Pre-package validation may defer precisely these
# three observations; it must still require every other declared change.
GENERATED_RELEASE_MODIFICATIONS = frozenset(
    {"PACKAGE_MANIFEST.txt", "PACKAGE_SHA256_MANIFEST.txt"}
)
GENERATED_RELEASE_ADDITIONS = frozenset(
    {"PACKAGE_SHA256_MANIFEST_V8_2.txt"}
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _included(root: Path, path: Path) -> bool:
    if not path.is_file() or path.is_symlink() or path.name == ".DS_Store":
        return False
    parts = path.relative_to(root).parts
    if any(
        part in EXCLUDED_PARTS
        or part.startswith("tmp_")
        or part.startswith(".tmp_")
        for part in parts
    ):
        return False
    return path.suffix not in {".pyc", ".pyo"}


def _load_mapping(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result: dict[str, str] = {}
    for raw_path, raw_reason in value.items():
        path = PurePosixPath(str(raw_path))
        if path.is_absolute() or ".." in path.parts or str(path) in result:
            raise ValueError(f"Unsafe or duplicate {field} path: {raw_path!r}")
        reason = str(raw_reason).strip()
        if not reason:
            raise ValueError(f"Missing reason for {field} path {raw_path!r}")
        result[str(path)] = reason
    return result


def _baseline_zip_files(
    archive_path: Path, expected_archive_sha256: str
) -> dict[str, bytes]:
    if _sha256_file(archive_path) != expected_archive_sha256:
        raise ValueError("Provided ZIP SHA-256 does not match the v8.1 baseline")
    result: dict[str, bytes] = {}
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Baseline ZIP CRC failure: {bad_member}")
        members = [name for name in archive.namelist() if not name.endswith("/")]
        roots = {PurePosixPath(name).parts[0] for name in members}
        if len(roots) != 1:
            raise ValueError("Baseline ZIP must contain exactly one package root")
        root_name = next(iter(roots))
        for name in members:
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != root_name
            ):
                raise ValueError(f"Unsafe baseline ZIP member: {name}")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if not relative or relative in result:
                raise ValueError(f"Duplicate baseline ZIP path: {relative}")
            result[relative] = archive.read(name)
    return result


def audit(
    root: Path,
    baseline_manifest_path: Path,
    expected_changes_path: Path,
    baseline_zip: Path | None = None,
    *,
    require_declared_changes: bool = False,
    allow_generated_release_artifacts_missing: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if allow_generated_release_artifacts_missing and not require_declared_changes:
        raise ValueError(
            "--allow-generated-release-artifacts-missing is valid only with "
            "--require-declared-changes"
        )
    root = root.resolve()
    baseline = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_changes_path.read_text(encoding="utf-8"))
    baseline_files = {
        str(PurePosixPath(path)): str(digest).lower()
        for path, digest in baseline["files"].items()
    }
    if int(baseline.get("file_count", -1)) != len(baseline_files):
        raise ValueError("Baseline manifest file_count does not match its inventory")
    allowed_modified = _load_mapping(
        expected.get("allowed_modified", {}), "allowed_modified"
    )
    allowed_added = _load_mapping(expected.get("allowed_added", {}), "allowed_added")
    allowed_deleted = _load_mapping(
        expected.get("allowed_deleted", {}), "allowed_deleted"
    )
    if allow_generated_release_artifacts_missing:
        missing_modified_declarations = sorted(
            GENERATED_RELEASE_MODIFICATIONS - set(allowed_modified)
        )
        missing_added_declarations = sorted(
            GENERATED_RELEASE_ADDITIONS - set(allowed_added)
        )
        if missing_modified_declarations or missing_added_declarations:
            raise ValueError(
                "Generated-release deferral policy does not match the declared "
                "change inventory: "
                f"missing modified={missing_modified_declarations}, "
                f"missing added={missing_added_declarations}"
            )
    if expected.get("baseline_release") != baseline.get("release"):
        raise ValueError("Expected-change file references a different baseline release")
    if expected.get("baseline_archive_sha256") != baseline.get("archive_sha256"):
        raise ValueError("Expected-change file references a different baseline archive")
    current_version = json.loads(
        (root / "VERSION.json").read_text(encoding="utf-8")
    ).get("version")
    if expected.get("target_release") != current_version:
        raise ValueError("Expected-change file target does not match VERSION.json")

    baseline_zip_files: dict[str, bytes] = {}
    baseline_zip_verification: dict[str, Any] = {
        "status": "not_provided",
        "provided": False,
        "archive_sha256_matches": False,
        "inventory_matches_embedded_manifest": False,
    }
    if baseline_zip is not None:
        resolved_baseline_zip = baseline_zip.resolve(strict=True)
        baseline_zip_files = _baseline_zip_files(
            resolved_baseline_zip, str(baseline["archive_sha256"])
        )
        observed = {
            path: _sha256_bytes(content)
            for path, content in baseline_zip_files.items()
        }
        if observed != baseline_files:
            raise ValueError("Baseline ZIP inventory does not match embedded manifest")

        baseline_zip_verification = {
            "status": "verified",
            "provided": True,
            "archive_name": resolved_baseline_zip.name,
            "archive_sha256_matches": True,
            "inventory_matches_embedded_manifest": True,
        }

    unsupported_symlinks = []
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            target = os.readlink(path)
        except OSError as error:
            target = f"<unreadable: {error}>"
        unsupported_symlinks.append(
            {
                "path": relative,
                "target": target,
                "target_exists": path.exists(),
            }
        )
    unsupported_symlinks.sort(key=lambda row: row["path"])

    current_files = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in root.rglob("*")
        if _included(root, path)
    }
    baseline_paths = set(baseline_files)
    current_paths = set(current_files)
    same = sorted(
        path
        for path in baseline_paths & current_paths
        if baseline_files[path] == current_files[path]
    )
    modified_paths = sorted(
        path
        for path in baseline_paths & current_paths
        if baseline_files[path] != current_files[path]
    )
    deleted_paths = sorted(baseline_paths - current_paths)
    added_paths = sorted(current_paths - baseline_paths)

    unexpected_modified = sorted(set(modified_paths) - set(allowed_modified))
    unexpected_deleted = sorted(set(deleted_paths) - set(allowed_deleted))
    unexpected_added = sorted(set(added_paths) - set(allowed_added))
    declared_but_unchanged = sorted(set(allowed_modified) & set(same))
    declared_additions_missing = sorted(set(allowed_added) - set(added_paths))
    declared_deletions_missing = sorted(set(allowed_deleted) - set(deleted_paths))

    deferred_declared_changes: list[dict[str, str]] = []
    undeferred_declared_but_unchanged = list(declared_but_unchanged)
    undeferred_declared_additions_missing = list(declared_additions_missing)
    if allow_generated_release_artifacts_missing:
        deferred_modified = sorted(
            set(declared_but_unchanged) & GENERATED_RELEASE_MODIFICATIONS
        )
        deferred_added = sorted(
            set(declared_additions_missing) & GENERATED_RELEASE_ADDITIONS
        )
        undeferred_declared_but_unchanged = sorted(
            set(declared_but_unchanged) - GENERATED_RELEASE_MODIFICATIONS
        )
        undeferred_declared_additions_missing = sorted(
            set(declared_additions_missing) - GENERATED_RELEASE_ADDITIONS
        )
        deferred_declared_changes.extend(
            {
                "path": path,
                "declaration": "allowed_modified",
                "observed_state": "byte_identical_to_baseline",
            }
            for path in deferred_modified
        )
        deferred_declared_changes.extend(
            {
                "path": path,
                "declaration": "allowed_added",
                "observed_state": "not_yet_generated",
            }
            for path in deferred_added
        )

    exact_failures = []
    if require_declared_changes:
        exact_failures = (
            undeferred_declared_but_unchanged
            + undeferred_declared_additions_missing
            + declared_deletions_missing
        )
    status = (
        "passed"
        if not (
            unexpected_modified
            or unexpected_deleted
            or unexpected_added
            or exact_failures
            or unsupported_symlinks
        )
        else "failed"
    )
    report: dict[str, Any] = {
        "status": status,
        "baseline_release": baseline["release"],
        "baseline_archive_sha256": baseline["archive_sha256"],
        "baseline_file_count": len(baseline_files),
        "current_file_count": len(current_files),
        "byte_identical_count": len(same),
        "modified_count": len(modified_paths),
        "added_count": len(added_paths),
        "deleted_count": len(deleted_paths),
        "require_declared_changes": require_declared_changes,
        "allow_generated_release_artifacts_missing": (
            allow_generated_release_artifacts_missing
        ),
        "baseline_zip_verification": baseline_zip_verification,
        "unsupported_symlink_count": len(unsupported_symlinks),
        "unsupported_symlinks": unsupported_symlinks,
        "modified": [
            {
                "path": path,
                "sha256_before": baseline_files[path],
                "sha256_after": current_files[path],
                "reason": allowed_modified.get(path),
            }
            for path in modified_paths
        ],
        "added": [
            {
                "path": path,
                "sha256": current_files[path],
                "reason": allowed_added.get(path),
            }
            for path in added_paths
        ],
        "deleted": [
            {"path": path, "reason": allowed_deleted.get(path)}
            for path in deleted_paths
        ],
        "unexpected_modified": unexpected_modified,
        "unexpected_added": unexpected_added,
        "unexpected_deleted": unexpected_deleted,
        "declared_but_unchanged": declared_but_unchanged,
        "declared_additions_missing": declared_additions_missing,
        "declared_deletions_missing": declared_deletions_missing,
        "undeferred_declared_but_unchanged": (
            undeferred_declared_but_unchanged
        ),
        "undeferred_declared_additions_missing": (
            undeferred_declared_additions_missing
        ),
        "deferred_declared_changes": deferred_declared_changes,
        "deferred_declared_change_count": len(deferred_declared_changes),
        "byte_identical_files": same,
        "meaning": (
            "Hash/inventory preservation proves declared file scope only; "
            "functional and numerical correctness require separate tests. "
            "Any listed generated-release deferral is limited to the three "
            "manifest artifacts created after pre-package validation."
        ),
    }
    return report, baseline_zip_files


def _write_report(
    output_dir: Path,
    report: dict[str, Any],
    root: Path,
    baseline_zip_files: dict[str, bytes],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# v8.1.0 → v8.2.0 파일 범위 감사",
        "",
        (
            f"상태: **{report['status']}** — 기준 {report['baseline_file_count']}개, "
            f"동일 {report['byte_identical_count']}개, 수정 {report['modified_count']}개, "
            f"추가 {report['added_count']}개, 삭제 {report['deleted_count']}개."
        ),
        "",
        "| 구분 | 파일 | 선언된 사유 |",
        "|---|---|---|",
    ]
    for kind in ("modified", "added", "deleted"):
        for row in report[kind]:
            lines.append(
                f"| {kind} | {row['path']} | {row.get('reason') or 'UNDECLARED'} |"
            )
    if report["deferred_declared_changes"]:
        lines.extend(
            [
                "",
                "## Pre-package 생성물 유예",
                "",
                "아래 항목만 release manifest 생성 직전 상태로 유예되었다.",
                "",
            ]
        )
        for row in report["deferred_declared_changes"]:
            lines.append(
                f"- `{row['path']}` ({row['declaration']}: "
                f"{row['observed_state']})"
            )
    if report["unsupported_symlinks"]:
        lines.extend(["", "## 지원하지 않는 symbolic link", ""])
        for row in report["unsupported_symlinks"]:
            lines.append(f"- `{row['path']}` → `{row['target']}`")
    lines.extend(
        [
            "",
            "해시와 inventory 감사는 변경 범위를 검증한다. 기능·수치 정확성은 "
            "별도 회귀시험, 보존식 시험 및 parity 결과를 확인해야 한다.",
        ]
    )
    (output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if baseline_zip_files:
        for row in report["modified"]:
            path = row["path"]
            before = baseline_zip_files[path]
            try:
                before_lines = before.decode("utf-8").splitlines(keepends=True)
                after_lines = (root / path).read_text(encoding="utf-8").splitlines(
                    keepends=True
                )
            except (UnicodeDecodeError, OSError):
                continue
            diff = "".join(
                difflib.unified_diff(
                    before_lines,
                    after_lines,
                    fromfile=f"v8.1.0/{path}",
                    tofile=f"v8.2.0/{path}",
                )
            )
            destination = output_dir / "diffs" / f"{path}.patch"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(diff, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path)
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=root / "docs" / "V810_BASELINE_MANIFEST.json",
    )
    parser.add_argument(
        "--expected-changes",
        type=Path,
        default=root / "docs" / "V820_EXPECTED_CHANGES.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs" / "v8_2_cpu_validation" / "preservation",
    )
    parser.add_argument("--require-declared-changes", action="store_true")
    parser.add_argument(
        "--allow-generated-release-artifacts-missing",
        action="store_true",
        help=(
            "With --require-declared-changes, defer only the two generic "
            "manifest modifications and the v8.2 authoritative manifest addition"
        ),
    )
    args = parser.parse_args()

    if (
        args.allow_generated_release_artifacts_missing
        and not args.require_declared_changes
    ):
        parser.error(
            "--allow-generated-release-artifacts-missing requires "
            "--require-declared-changes"
        )

    report, baseline_zip_files = audit(
        root,
        args.baseline_manifest.resolve(),
        args.expected_changes.resolve(),
        args.baseline_zip,
        require_declared_changes=args.require_declared_changes,
        allow_generated_release_artifacts_missing=(
            args.allow_generated_release_artifacts_missing
        ),
    )
    _write_report(args.output_dir.resolve(), report, root, baseline_zip_files)
    summary = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "modified",
            "added",
            "deleted",
            "byte_identical_files",
        }
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
