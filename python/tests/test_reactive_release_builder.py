from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import build_reactive_v8_3_release as builder  # noqa: E402

from verify_release_manifest import verify  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_parent(root: Path) -> Path:
    parent = root / builder.PARENT_NAME
    parent.mkdir()
    _write_json(
        parent / "VERSION.json",
        {
            "version": builder.PARENT_VERSION,
            "release_directory_name": builder.PARENT_NAME,
        },
    )
    (parent / "payload.txt").write_text("immutable parent\n", encoding="utf-8")
    rows = [
        f"{_sha256(parent / relative)}  {relative}"
        for relative in ("VERSION.json", "payload.txt")
    ]
    payload = "\n".join(rows) + "\n"
    (parent / builder.PARENT_MANIFEST).write_text(payload, encoding="utf-8")
    (parent / builder.GENERIC_MANIFEST).write_text(payload, encoding="utf-8")
    return parent


def _make_source(root: Path) -> Path:
    source = root / builder.RELEASE_NAME
    source.mkdir()
    _write_json(
        source / "VERSION.json",
        {
            "version": builder.RELEASE_VERSION,
            "release_directory_name": builder.RELEASE_NAME,
            # An unrelated scientific caveat may contain the word pending;
            # only v8_3_test_status is a release gate.
            "predictive_status": "calibration_pending_external_measurements",
            "paper_reactive_v8_3": {"final_verdict": "NOT_FULLY_VERIFIED"},
            "v8_3_test_status": {
                "reactive_cpu_fp64": "PASS",
                "legacy_v8_2_1_full_suite": "PASS",
                "grid_and_time_convergence": "PASS",
                "determinism_50_repeats": "PASS",
                "actual_MPI_rank_1_2_4_8": "NOT_TESTED_NO_MPI_RUNTIME",
            },
        },
    )
    for relative in builder.REQUIRED_DOCS:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("release document\n", encoding="utf-8")
    _write_json(
        source / "docs/PAPER_ASSUMPTIONS_V8_3_0.json",
        {
            "schema": "ecsp.paper-assumptions/v1",
            "verdict": "NOT_FULLY_VERIFIED",
        },
    )
    _write_json(
        source / "docs/REACTIVE_CONVERGENCE_V8_3_0.json",
        {
            "schema": "ecsp.reactive-validation/v1",
            "verdict": "NOT_FULLY_VERIFIED",
            "runnable_cpu_verification": {
                "status": "PASS",
                "failed_check_count": 0,
                "passed_check_count": 1,
            },
            "checks": {"manufactured": {"status": "PASS"}},
        },
    )
    _write_json(
        source / "docs/V821_PARENT_PRESERVATION_V8_3_0.json",
        {
            "schema": "ecsp.v821-parent-preservation/v1",
            "status": "PASS",
            "unexpected_added": [],
            "unexpected_deleted": [],
            "unexpected_modified": [],
        },
    )
    (source / "payload.py").write_text("VALUE = 83\n", encoding="utf-8")
    return source


def _make_validation(root: Path, *, source: Path, parent: Path) -> Path:
    validation = root / "validation"
    validation.mkdir()
    junit = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite name="pytest" errors="0" failures="0" '
        'skipped="0" tests="1"><testcase name="ok" /></testsuite></testsuites>'
    )
    (validation / "full_suite.xml").write_text(junit, encoding="utf-8")
    (validation / "reactive_suite.xml").write_text(junit, encoding="utf-8")
    _write_json(
        validation / "parent_preservation.json",
        {
            "schema": "ecsp.v821-parent-preservation/v1",
            "status": "PASS",
            "candidate": str(source),
            "parent": str(parent),
            "unexpected_added": [],
            "unexpected_deleted": [],
            "unexpected_modified": [],
        },
    )
    _write_json(
        validation / "reactive_benchmark.json",
        {
            "schema": "ecsp.paper-reactive-benchmark/v1",
            "overall_status": "PASS",
            "runs": [{"bitwise_final_state_deterministic": True}],
        },
    )
    smoke_npz = validation / "reactive_assumed_smoke.npz"
    smoke_npz.write_bytes(b"synthetic deterministic npz")
    _write_json(
        validation / "reactive_assumed_smoke.json",
        {
            "schema": "ecsp.paper-reactive-result/v1",
            "accepted_steps": 1,
            "npz_sha256": _sha256(smoke_npz),
            "metadata": {
                "geometry_ranking_eligible": False,
                "paper_reproduction_verdict": (
                    "NOT_FULLY_VERIFIED_MISSING_PAPER_INPUTS"
                ),
            },
        },
    )
    _write_json(
        validation / "VALIDATED_SOURCE_TREE.json",
        {
            "schema": "ecsp.validated-source-tree/v1",
            "status": "PASS",
            "verdict": "NOT_FULLY_VERIFIED",
            "release": builder.RELEASE_VERSION,
            "source_tree_sha256": builder._source_tree_digest(source),
        },
    )
    return validation


def _case(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    parent = _make_parent(inputs)
    source = _make_source(inputs)
    validation = _make_validation(inputs, source=source, parent=parent)
    output = tmp_path / "output"
    output.mkdir()
    return source, parent, validation, output


def _assert_no_published_artifacts(output: Path, validation: Path) -> None:
    assert not (output / builder.RELEASE_NAME).exists()
    assert not (output / f"{builder.RELEASE_NAME}.zip").exists()
    assert not (output / f"{builder.RELEASE_NAME}.zip.sha256.txt").exists()
    assert not (validation / "RELEASE_BUILD_COMPLETE.json").exists()


def test_release_builder_semantically_validates_then_atomically_publishes(
    tmp_path: Path,
) -> None:
    source, parent, validation, output = _case(tmp_path)
    report = builder.build_release(
        source=source,
        parent=parent,
        output_root=output,
        validation_dir=validation,
    )

    release = output / builder.RELEASE_NAME
    archive = output / f"{builder.RELEASE_NAME}.zip"
    assert release.is_dir() and archive.is_file()
    assert report["zip_crc"] == "PASS"
    assert report["verdict"] == "NOT_FULLY_VERIFIED"
    assert report["validated_source_tree_sha256"] == builder._source_tree_digest(source)
    assert (release / builder.AUTHORITATIVE_MANIFEST).is_file()
    assert (release / builder.SUPPLEMENTARY_MANIFEST).is_file()
    assert (release / builder.GENERIC_MANIFEST).read_bytes() == (
        release / builder.AUTHORITATIVE_MANIFEST
    ).read_bytes()
    generic_report = verify(release, release / builder.AUTHORITATIVE_MANIFEST)
    assert generic_report["status"] == "passed"
    assert generic_report["unlisted"] == []


def test_fresh_extract_failure_leaves_no_partial_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, parent, validation, output = _case(tmp_path)

    def fail_fresh_extract(path: Path) -> dict[str, object]:
        del path
        raise RuntimeError("synthetic fresh extraction failure")

    monkeypatch.setattr(builder, "_verify_extracted", fail_fresh_extract)
    with pytest.raises(RuntimeError, match="synthetic fresh extraction failure"):
        builder.build_release(
            source=source,
            parent=parent,
            output_root=output,
            validation_dir=validation,
        )
    _assert_no_published_artifacts(output, validation)


def test_publication_rename_failure_rolls_back_every_created_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, parent, validation, output = _case(tmp_path)
    real_rename = builder.os.rename
    archive_target = output / f"{builder.RELEASE_NAME}.zip"

    def fail_second_publish(staged: Path, target: Path) -> None:
        if Path(target) == archive_target:
            raise OSError("synthetic publication failure")
        real_rename(staged, target)

    monkeypatch.setattr(builder.os, "rename", fail_second_publish)
    with pytest.raises(OSError, match="synthetic publication failure"):
        builder.build_release(
            source=source,
            parent=parent,
            output_root=output,
            validation_dir=validation,
        )
    _assert_no_published_artifacts(output, validation)


@pytest.mark.parametrize("failure_kind", ("junit", "source_binding", "parent_hash"))
def test_release_builder_rejects_invalid_evidence_or_parent_before_publication(
    tmp_path: Path, failure_kind: str
) -> None:
    source, parent, validation, output = _case(tmp_path)
    if failure_kind == "junit":
        (validation / "full_suite.xml").write_text(
            '<testsuite tests="1" failures="1" errors="0">'
            '<testcase><failure /></testcase></testsuite>',
            encoding="utf-8",
        )
        match = "JUnit evidence is not passing"
    elif failure_kind == "source_binding":
        binding = json.loads(
            (validation / "VALIDATED_SOURCE_TREE.json").read_text(encoding="utf-8")
        )
        binding["source_tree_sha256"] = "0" * 64
        _write_json(validation / "VALIDATED_SOURCE_TREE.json", binding)
        match = "source_tree_sha256"
    else:
        (parent / "payload.txt").write_text("tampered\n", encoding="utf-8")
        match = "hash mismatch"

    with pytest.raises(RuntimeError, match=match):
        builder.build_release(
            source=source,
            parent=parent,
            output_root=output,
            validation_dir=validation,
        )
    _assert_no_published_artifacts(output, validation)


def test_pending_gate_is_scoped_to_v8_3_test_status(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    builder._validate_version(source)
    version_path = source / "VERSION.json"
    version = json.loads(version_path.read_text(encoding="utf-8"))
    version["v8_3_test_status"]["reactive_cpu_fp64"] = "pending_final_run"
    _write_json(version_path, version)
    with pytest.raises(RuntimeError, match="reactive_cpu_fp64"):
        builder._validate_version(source)
