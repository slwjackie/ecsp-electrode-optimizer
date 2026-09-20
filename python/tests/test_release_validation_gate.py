from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from audit_v820_preservation import audit


ROOT = Path(__file__).resolve().parents[2]


def _preservation_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    current = tmp_path / "current"
    metadata = tmp_path / "metadata"
    current.mkdir()
    metadata.mkdir()

    baseline_files = {
        "VERSION.json": json.dumps({"version": "8.2.0"}).encode(),
        "PACKAGE_MANIFEST.txt": b"v8.2.0 manifest\n",
        "PACKAGE_SHA256_MANIFEST.txt": b"v8.2.0 hashes\n",
        "payload.txt": b"preserved payload\n",
    }
    baseline_zip = metadata / "baseline.zip"
    with zipfile.ZipFile(baseline_zip, "w") as archive:
        for relative, content in baseline_files.items():
            archive.writestr(f"v8.2.0-root/{relative}", content)
    archive_sha256 = hashlib.sha256(baseline_zip.read_bytes()).hexdigest()
    baseline_manifest = metadata / "baseline_manifest.json"
    baseline_manifest.write_text(
        json.dumps(
            {
                "release": "8.2.0",
                "archive_sha256": archive_sha256,
                "file_count": len(baseline_files),
                "files": {
                    relative: hashlib.sha256(content).hexdigest()
                    for relative, content in baseline_files.items()
                },
            }
        ),
        encoding="utf-8",
    )

    for relative, content in baseline_files.items():
        (current / relative).write_bytes(content)
    (current / "VERSION.json").write_text(
        json.dumps({"version": "8.2.1"}), encoding="utf-8"
    )
    expected_changes = metadata / "expected_changes.json"
    expected_changes.write_text(
        json.dumps(
            {
                "baseline_release": "8.2.0",
                "baseline_archive_sha256": archive_sha256,
                "target_release": "8.2.1",
                "allowed_modified": {
                    "VERSION.json": "version transition",
                    "PACKAGE_MANIFEST.txt": "generated release inventory",
                    "PACKAGE_SHA256_MANIFEST.txt": "generated hash alias",
                },
                "allowed_added": {
                    "PACKAGE_SHA256_MANIFEST_V8_2_1.txt": (
                        "generated authoritative hash manifest"
                    )
                },
                "allowed_deleted": {},
            }
        ),
        encoding="utf-8",
    )
    return current, baseline_manifest, expected_changes, baseline_zip


def test_prepackage_audit_defers_only_three_generated_manifest_changes(
    tmp_path: Path,
) -> None:
    current, baseline_manifest, expected_changes, baseline_zip = (
        _preservation_fixture(tmp_path)
    )

    strict, _ = audit(
        current,
        baseline_manifest,
        expected_changes,
        baseline_zip,
        require_declared_changes=True,
    )
    assert strict["status"] == "failed"
    assert strict["declared_but_unchanged"] == [
        "PACKAGE_MANIFEST.txt",
        "PACKAGE_SHA256_MANIFEST.txt",
    ]
    assert strict["declared_additions_missing"] == [
        "PACKAGE_SHA256_MANIFEST_V8_2_1.txt"
    ]

    prepackage, _ = audit(
        current,
        baseline_manifest,
        expected_changes,
        baseline_zip,
        require_declared_changes=True,
        allow_generated_release_artifacts_missing=True,
    )
    assert prepackage["status"] == "passed"
    assert prepackage["baseline_zip_verification"] == {
        "status": "verified",
        "provided": True,
        "archive_name": "baseline.zip",
        "archive_sha256_matches": True,
        "inventory_matches_embedded_manifest": True,
    }
    assert [row["path"] for row in prepackage["deferred_declared_changes"]] == [
        "PACKAGE_MANIFEST.txt",
        "PACKAGE_SHA256_MANIFEST.txt",
        "PACKAGE_SHA256_MANIFEST_V8_2_1.txt",
    ]

    changed = json.loads(expected_changes.read_text(encoding="utf-8"))
    changed["allowed_added"]["undeferred-required.txt"] = "must already exist"
    expected_changes.write_text(json.dumps(changed), encoding="utf-8")
    missing_other, _ = audit(
        current,
        baseline_manifest,
        expected_changes,
        baseline_zip,
        require_declared_changes=True,
        allow_generated_release_artifacts_missing=True,
    )
    assert missing_other["status"] == "failed"
    assert missing_other["undeferred_declared_additions_missing"] == [
        "undeferred-required.txt"
    ]


def test_generated_artifact_deferral_requires_strict_declared_change_mode(
    tmp_path: Path,
) -> None:
    current, baseline_manifest, expected_changes, baseline_zip = (
        _preservation_fixture(tmp_path)
    )
    with pytest.raises(ValueError, match="valid only with"):
        audit(
            current,
            baseline_manifest,
            expected_changes,
            baseline_zip,
            allow_generated_release_artifacts_missing=True,
        )


def test_preservation_audit_reports_and_rejects_symlinks(tmp_path: Path) -> None:
    current, baseline_manifest, expected_changes, baseline_zip = (
        _preservation_fixture(tmp_path)
    )
    link = current / "payload-alias.txt"
    link.symlink_to(current / "payload.txt")

    report, _ = audit(
        current,
        baseline_manifest,
        expected_changes,
        baseline_zip,
        require_declared_changes=True,
        allow_generated_release_artifacts_missing=True,
    )
    assert report["status"] == "failed"
    assert report["unsupported_symlink_count"] == 1
    assert report["unsupported_symlinks"] == [
        {
            "path": "payload-alias.txt",
            "target": str(current / "payload.txt"),
            "target_exists": True,
        }
    ]


def test_validator_fails_before_work_for_missing_baseline_or_occupied_output(
    tmp_path: Path,
) -> None:
    script = ROOT / "tools" / "validate_v8_2_1_cpu.sh"
    environment = dict(os.environ)
    environment["PYTHON"] = sys.executable
    environment.pop("ECSP_V820_BASELINE_ZIP", None)
    missing_baseline = subprocess.run(
        ["bash", str(script), str(tmp_path / "unused-output")],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert missing_baseline.returncode == 2
    assert "ECSP_V820_BASELINE_ZIP" in missing_baseline.stderr
    assert not (tmp_path / "unused-output").exists()

    baseline = tmp_path / "baseline.zip"
    baseline.write_bytes(b"test-only; output guard runs before ZIP audit")
    environment["ECSP_V820_BASELINE_ZIP"] = str(baseline)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    occupied_result = subprocess.run(
        ["bash", str(script), str(occupied)],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert occupied_result.returncode == 2
    assert "new, nonexistent path" in occupied_result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"

    dangling = tmp_path / "dangling-output"
    dangling.symlink_to(tmp_path / "missing-target")
    dangling_result = subprocess.run(
        ["bash", str(script), str(dangling)],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert dangling_result.returncode == 2
    assert "new, nonexistent path" in dangling_result.stderr
    assert dangling.is_symlink()


def test_validator_rejects_symlink_baseline_and_has_atomic_completion_contract(
    tmp_path: Path,
) -> None:
    script = ROOT / "tools" / "validate_v8_2_1_cpu.sh"
    baseline = tmp_path / "baseline.zip"
    baseline.write_bytes(b"placeholder")
    baseline_link = tmp_path / "baseline-link.zip"
    baseline_link.symlink_to(baseline)
    environment = dict(os.environ)
    environment.update(
        PYTHON=sys.executable,
        ECSP_V820_BASELINE_ZIP=str(baseline_link),
    )
    rejected = subprocess.run(
        ["bash", str(script), str(tmp_path / "output")],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "regular file" in rejected.stderr
    assert not (tmp_path / "output").exists()

    source = script.read_text(encoding="utf-8")
    assert 'PYTHON_BIN="${PYTHON:-python}"' in source
    assert '--allow-generated-release-artifacts-missing' in source
    assert 'pipeline.get("checks_passed") == pipeline.get("checks_total") == 28' in source
    assert "expected_no_cuda_skips = [" in source
    assert '"pytest_skipped_test_ids": skipped_test_ids' in source
    assert '"validated_package": {' in source
    assert '"authoritative_manifest_sha256"' in source
    assert "def require(condition: bool, message: str)" in source
    assert 'os.replace(temporary, target)' in source
    assert '"VALIDATION_COMPLETE.json"' in source
