#!/usr/bin/env python3
"""Build and independently verify the immutable v8.3 experimental archive.

The builder deliberately treats validation reports as untrusted input.  It
checks their schemas and pass/fail fields, binds them to the exact source tree,
and verifies a temporary ZIP before publishing any release artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE_NAME = "ECSP_v8_3_0_PaperReactiveEuler_Experimental"
RELEASE_VERSION = "8.3.0-PaperReactiveEuler-Experimental"
PARENT_NAME = "ECSP_v8_2_1_P0P1Fixed_BCNative_Hybrid_A100_CPU8"
PARENT_VERSION = "8.2.1-P0P1Fixed-BCNative-Hybrid-A100-CPU8"
PARENT_MANIFEST = "PACKAGE_SHA256_MANIFEST_V8_2_1.txt"
AUTHORITATIVE_MANIFEST = "PACKAGE_SHA256_MANIFEST_V8_3.txt"
SUPPLEMENTARY_MANIFEST = "PACKAGE_SHA256_MANIFEST_V8_3_0.txt"
GENERIC_MANIFEST = "PACKAGE_SHA256_MANIFEST.txt"
PACKAGE_INVENTORY = "PACKAGE_MANIFEST.txt"
BUILD_METADATA = "PACKAGE_BUILD_METADATA_V8_3_0.json"

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".native_build",
    "__pycache__",
    "build",
    "runs",
}
REGENERATED_FILES = {
    PACKAGE_INVENTORY,
    GENERIC_MANIFEST,
    AUTHORITATIVE_MANIFEST,
    SUPPLEMENTARY_MANIFEST,
    BUILD_METADATA,
}
REGENERATED_MANIFESTS = {
    PACKAGE_INVENTORY,
    GENERIC_MANIFEST,
    AUTHORITATIVE_MANIFEST,
    SUPPLEMENTARY_MANIFEST,
}
REQUIRED_DOCS = {
    "docs/CHANGELOG_V8_3_0_KR.md",
    "docs/PAPER_ASSUMPTIONS_V8_3_0.json",
    "docs/PAPER_CODE_TRACEABILITY_V8_3_0_KR.md",
    "docs/PAPER_REPRODUCTION_V8_3_0_KR.md",
    "docs/REACTIVE_CONVERGENCE_V8_3_0.json",
    "docs/REACTIVE_PARITY_PERFORMANCE_V8_3_0_KR.md",
    "docs/REACTIVE_VALIDATION_V8_3_0_KR.md",
    "docs/V821_PARENT_PRESERVATION_V8_3_0.json",
}
REQUIRED_EVIDENCE = {
    "VALIDATED_SOURCE_TREE.json",
    "full_suite.xml",
    "parent_preservation.json",
    "reactive_assumed_smoke.json",
    "reactive_assumed_smoke.npz",
    "reactive_benchmark.json",
    "reactive_suite.xml",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included(relative: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.as_posix() in REGENERATED_FILES:
        return False
    if relative.suffix in {".pyc", ".pyo"} or relative.name == ".DS_Store":
        return False
    return not any(
        part.startswith("tmp_") or part.startswith(".tmp_")
        for part in relative.parts
    )


def _copy_source(source: Path, staging: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink():
            raise RuntimeError(f"Release source contains an unsupported symlink: {relative}")
        if not _included(relative):
            continue
        target = staging / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _source_tree_digest(root: Path) -> str:
    """Hash exactly the non-generated source payload copied by this builder."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root)
        if path.is_symlink():
            raise RuntimeError(
                f"Release source contains an unsupported symlink: {relative_path}"
            )
        if not path.is_file() or not _included(relative_path):
            continue
        relative = relative_path.as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _read_json_object(path: Path) -> dict[str, object]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"Non-finite JSON number {token!r} is forbidden")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Invalid JSON evidence {path.name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"JSON evidence must contain an object: {path.name}")
    return parsed


def _require_field(
    report: dict[str, object], path: Path, field: str, expected: object
) -> None:
    if report.get(field) != expected:
        raise RuntimeError(
            f"{path.name} must declare {field}={expected!r}; "
            f"found {report.get(field)!r}"
        )


def _parse_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a nonnegative integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a nonnegative integer") from exc
    if str(parsed) != str(value) or parsed < 0:
        raise RuntimeError(f"{label} must be a nonnegative integer")
    return parsed


def _validate_junit(path: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Cannot read JUnit evidence {path.name}: {exc}") from exc
    upper = payload.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise RuntimeError(f"Unsafe XML declaration in JUnit evidence: {path.name}")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(f"Invalid JUnit XML {path.name}: {exc}") from exc

    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    suites = [element for element in root.iter() if local_name(element) == "testsuite"]
    if not suites:
        raise RuntimeError(f"JUnit evidence contains no testsuite: {path.name}")
    tests = failures = errors = skipped = 0
    for index, suite in enumerate(suites):
        tests += _parse_nonnegative_int(
            suite.attrib.get("tests"), label=f"{path.name} testsuite[{index}].tests"
        )
        failures += _parse_nonnegative_int(
            suite.attrib.get("failures"),
            label=f"{path.name} testsuite[{index}].failures",
        )
        errors += _parse_nonnegative_int(
            suite.attrib.get("errors"), label=f"{path.name} testsuite[{index}].errors"
        )
        skipped += _parse_nonnegative_int(
            suite.attrib.get("skipped", "0"),
            label=f"{path.name} testsuite[{index}].skipped",
        )
    failure_nodes = sum(
        1 for element in root.iter() if local_name(element) in {"failure", "error"}
    )
    if tests <= 0 or failures != 0 or errors != 0 or failure_nodes != 0:
        raise RuntimeError(
            f"JUnit evidence is not passing: {path.name} "
            f"tests={tests} failures={failures} errors={errors} "
            f"failure_nodes={failure_nodes}"
        )
    return {
        "schema": "junit-testsuite",
        "status": "PASS",
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _read_sha_manifest(root: Path, manifest: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Cannot read SHA manifest {manifest}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        pieces = raw.split("  ", 1)
        if len(pieces) != 2:
            raise RuntimeError(f"Malformed SHA manifest line {line_number}: {raw!r}")
        digest, relative = pieces
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise RuntimeError(f"Invalid SHA-256 on manifest line {line_number}")
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in entries
        ):
            raise RuntimeError(f"Unsafe or duplicate manifest path: {relative!r}")
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise RuntimeError(f"Manifest path escapes package root: {relative}")
        entries[relative] = digest
    return entries


def _parent_included(root: Path, path: Path, manifest: Path, generic: Path) -> bool:
    if not path.is_file() or path.is_symlink() or path in {manifest, generic}:
        return False
    relative = path.relative_to(root)
    if relative.name == ".DS_Store" or relative.suffix in {".pyc", ".pyo"}:
        return False
    return not any(
        part in EXCLUDED_PARTS
        or part.startswith("tmp_")
        or part.startswith(".tmp_")
        for part in relative.parts
    )


def _verify_parent_release(parent: Path) -> dict[str, object]:
    if parent.name != PARENT_NAME:
        raise RuntimeError(f"Parent directory must be named exactly {PARENT_NAME}")
    version_path = parent / "VERSION.json"
    version = _read_json_object(version_path)
    _require_field(version, version_path, "version", PARENT_VERSION)
    _require_field(version, version_path, "release_directory_name", PARENT_NAME)
    manifest = parent / PARENT_MANIFEST
    generic = parent / GENERIC_MANIFEST
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError(f"Parent authoritative manifest is missing: {manifest}")
    if not generic.is_file() or generic.is_symlink():
        raise RuntimeError("Parent generic SHA manifest is missing or is a symlink")
    if generic.read_bytes() != manifest.read_bytes():
        raise RuntimeError("Parent generic and authoritative SHA manifests differ")
    symlinks = [path.relative_to(parent).as_posix() for path in parent.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"Parent package contains unsupported symlinks: {symlinks}")
    entries = _read_sha_manifest(parent, manifest)
    actual = {
        path.relative_to(parent).as_posix()
        for path in parent.rglob("*")
        if _parent_included(parent, path, manifest, generic)
    }
    if set(entries) != actual:
        raise RuntimeError(
            "Parent authoritative manifest inventory mismatch: "
            f"missing={sorted(set(entries) - actual)} "
            f"unlisted={sorted(actual - set(entries))}"
        )
    mismatched = [
        relative
        for relative, expected in entries.items()
        if _sha256(parent / relative) != expected
    ]
    if mismatched:
        raise RuntimeError(f"Parent authoritative manifest hash mismatch: {mismatched}")
    return {
        "status": "PASS",
        "version": PARENT_VERSION,
        "manifest": PARENT_MANIFEST,
        "manifest_sha256": _sha256(manifest),
        "verified_file_count": len(entries),
    }


def _validate_release_documents(source: Path) -> dict[str, object]:
    convergence_path = source / "docs/REACTIVE_CONVERGENCE_V8_3_0.json"
    convergence = _read_json_object(convergence_path)
    _require_field(convergence, convergence_path, "schema", "ecsp.reactive-validation/v1")
    _require_field(convergence, convergence_path, "verdict", "NOT_FULLY_VERIFIED")
    runnable = convergence.get("runnable_cpu_verification")
    if not isinstance(runnable, dict) or runnable.get("status") != "PASS":
        raise RuntimeError("Reactive convergence CPU verification is not PASS")
    if runnable.get("failed_check_count") != 0:
        raise RuntimeError("Reactive convergence report contains failed checks")
    checks = convergence.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise RuntimeError("Reactive convergence report contains no checks")
    failed_checks = sorted(
        name
        for name, report in checks.items()
        if not isinstance(report, dict) or report.get("status") != "PASS"
    )
    if failed_checks:
        raise RuntimeError(f"Reactive convergence checks are not PASS: {failed_checks}")

    assumptions_path = source / "docs/PAPER_ASSUMPTIONS_V8_3_0.json"
    assumptions = _read_json_object(assumptions_path)
    _require_field(assumptions, assumptions_path, "schema", "ecsp.paper-assumptions/v1")
    _require_field(assumptions, assumptions_path, "verdict", "NOT_FULLY_VERIFIED")

    preservation_path = source / "docs/V821_PARENT_PRESERVATION_V8_3_0.json"
    preservation = _read_json_object(preservation_path)
    _require_field(
        preservation, preservation_path, "schema", "ecsp.v821-parent-preservation/v1"
    )
    _require_field(preservation, preservation_path, "status", "PASS")
    for field in ("unexpected_added", "unexpected_deleted", "unexpected_modified"):
        if preservation.get(field) != []:
            raise RuntimeError(f"{preservation_path.name} declares nonempty {field}")
    return {
        "status": "PASS",
        "verdict": "NOT_FULLY_VERIFIED",
        "convergence_passed_check_count": runnable.get("passed_check_count"),
    }


def _validate_evidence(
    validation_dir: Path,
    *,
    source: Path,
    parent: Path,
    source_tree_sha256: str,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    available = {path.name for path in _files(validation_dir)}
    missing = sorted(REQUIRED_EVIDENCE - available)
    if missing:
        raise RuntimeError(f"Required validation evidence is missing: {missing}")

    semantic: dict[str, object] = {
        "full_suite.xml": _validate_junit(validation_dir / "full_suite.xml"),
        "reactive_suite.xml": _validate_junit(validation_dir / "reactive_suite.xml"),
    }

    preservation_path = validation_dir / "parent_preservation.json"
    preservation = _read_json_object(preservation_path)
    _require_field(
        preservation, preservation_path, "schema", "ecsp.v821-parent-preservation/v1"
    )
    _require_field(preservation, preservation_path, "status", "PASS")
    for field in ("unexpected_added", "unexpected_deleted", "unexpected_modified"):
        if preservation.get(field) != []:
            raise RuntimeError(f"{preservation_path.name} declares nonempty {field}")
    try:
        evidence_candidate = Path(str(preservation["candidate"])).resolve(strict=True)
        evidence_parent = Path(str(preservation["parent"])).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise RuntimeError("Parent-preservation evidence has invalid source paths") from exc
    if evidence_candidate != source or evidence_parent != parent:
        raise RuntimeError("Parent-preservation evidence is for a different source or parent")
    semantic[preservation_path.name] = {
        "schema": preservation["schema"],
        "status": preservation["status"],
    }

    benchmark_path = validation_dir / "reactive_benchmark.json"
    benchmark = _read_json_object(benchmark_path)
    _require_field(benchmark, benchmark_path, "schema", "ecsp.paper-reactive-benchmark/v1")
    _require_field(benchmark, benchmark_path, "overall_status", "PASS")
    runs = benchmark.get("runs")
    if not isinstance(runs, list) or not runs:
        raise RuntimeError("Reactive benchmark contains no runs")
    if any(
        not isinstance(run, dict) or run.get("bitwise_final_state_deterministic") is not True
        for run in runs
    ):
        raise RuntimeError("Reactive benchmark did not report bitwise-deterministic runs")
    semantic[benchmark_path.name] = {
        "schema": benchmark["schema"],
        "status": benchmark["overall_status"],
        "run_count": len(runs),
    }

    smoke_path = validation_dir / "reactive_assumed_smoke.json"
    smoke = _read_json_object(smoke_path)
    _require_field(smoke, smoke_path, "schema", "ecsp.paper-reactive-result/v1")
    if not isinstance(smoke.get("accepted_steps"), int) or smoke["accepted_steps"] <= 0:
        raise RuntimeError("Reactive smoke evidence contains no accepted steps")
    metadata = smoke.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("Reactive smoke evidence has no metadata object")
    if metadata.get("geometry_ranking_eligible") is not False:
        raise RuntimeError("Experimental reactive smoke must not claim ranking eligibility")
    paper_verdict = metadata.get("paper_reproduction_verdict")
    if not isinstance(paper_verdict, str) or not paper_verdict.startswith(
        "NOT_FULLY_VERIFIED"
    ):
        raise RuntimeError("Reactive smoke has an invalid paper-reproduction verdict")
    smoke_npz = validation_dir / "reactive_assumed_smoke.npz"
    if smoke.get("npz_sha256") != _sha256(smoke_npz):
        raise RuntimeError("Reactive smoke JSON is not bound to its NPZ payload")
    semantic[smoke_path.name] = {
        "schema": smoke["schema"],
        "status": "PASS",
        "verdict": paper_verdict,
    }

    binding_path = validation_dir / "VALIDATED_SOURCE_TREE.json"
    binding = _read_json_object(binding_path)
    _require_field(binding, binding_path, "schema", "ecsp.validated-source-tree/v1")
    _require_field(binding, binding_path, "status", "PASS")
    _require_field(binding, binding_path, "verdict", "NOT_FULLY_VERIFIED")
    _require_field(binding, binding_path, "release", RELEASE_VERSION)
    _require_field(binding, binding_path, "source_tree_sha256", source_tree_sha256)
    semantic[binding_path.name] = {
        "schema": binding["schema"],
        "status": binding["status"],
        "verdict": binding["verdict"],
        "source_tree_sha256": binding["source_tree_sha256"],
    }

    evidence = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in _files(validation_dir)
        if path.name != "RELEASE_BUILD_COMPLETE.json"
    }
    return evidence, semantic


def _write_manifests(staging: Path) -> tuple[int, str]:
    manifest_names = {
        PACKAGE_INVENTORY,
        GENERIC_MANIFEST,
        AUTHORITATIVE_MANIFEST,
        SUPPLEMENTARY_MANIFEST,
    }
    paths = [path.relative_to(staging).as_posix() for path in _files(staging)]
    paths.extend(sorted(manifest_names))
    paths = sorted(set(paths))
    _atomic_bytes(
        staging / PACKAGE_INVENTORY,
        ("\n".join(paths) + "\n").encode("utf-8"),
    )

    # Keep the explicit SemVer spelling as a supplementary audit manifest.
    # It is written first so the historical two-component authoritative name
    # can hash it and remain compatible with verify_release_manifest.py.
    supplementary_paths = [
        path for path in paths if path not in {
            GENERIC_MANIFEST,
            AUTHORITATIVE_MANIFEST,
            SUPPLEMENTARY_MANIFEST,
        }
    ]
    supplementary_rows = [
        f"{_sha256(staging / relative)}  {relative}"
        for relative in supplementary_paths
    ]
    _atomic_bytes(
        staging / SUPPLEMENTARY_MANIFEST,
        ("\n".join(supplementary_rows) + "\n").encode("utf-8"),
    )

    authoritative_paths = [
        path for path in paths if path not in {GENERIC_MANIFEST, AUTHORITATIVE_MANIFEST}
    ]
    authoritative_rows = [
        f"{_sha256(staging / relative)}  {relative}"
        for relative in authoritative_paths
    ]
    payload = ("\n".join(authoritative_rows) + "\n").encode("utf-8")
    _atomic_bytes(staging / AUTHORITATIVE_MANIFEST, payload)
    _atomic_bytes(staging / GENERIC_MANIFEST, payload)
    return len(paths), hashlib.sha256(payload).hexdigest()


def _write_deterministic_zip(release: Path, archive: Path) -> None:
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as output:
        for path in _files(release):
            relative = Path(RELEASE_NAME) / path.relative_to(release)
            info = zipfile.ZipInfo(
                relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o100755 if os.access(path, os.X_OK) else 0o100644
            info.external_attr = mode << 16
            output.writestr(info, path.read_bytes(), compresslevel=9)


def _verify_manifest_inventory(
    root: Path, manifest_name: str, expected_paths: set[str]
) -> int:
    manifest = root / manifest_name
    rows = _read_sha_manifest(root, manifest)
    if set(rows) != expected_paths:
        raise RuntimeError(
            f"Fresh extraction inventory mismatch for {manifest_name}: "
            f"missing={sorted(set(rows) - expected_paths)} "
            f"unlisted={sorted(expected_paths - set(rows))}"
        )
    for relative, expected in rows.items():
        if _sha256(root / relative) != expected:
            raise RuntimeError(f"Fresh extraction hash mismatch: {relative}")
    return len(rows)


def _verify_extracted(extracted_release: Path) -> dict[str, object]:
    listed = (extracted_release / PACKAGE_INVENTORY).read_text(
        encoding="utf-8"
    ).splitlines()
    actual = [
        path.relative_to(extracted_release).as_posix()
        for path in _files(extracted_release)
    ]
    if len(listed) != len(set(listed)) or sorted(listed) != sorted(actual):
        raise RuntimeError("Fresh extraction does not match PACKAGE_MANIFEST.txt")
    actual_set = set(actual)
    authoritative_expected = actual_set - {AUTHORITATIVE_MANIFEST, GENERIC_MANIFEST}
    authoritative_count = _verify_manifest_inventory(
        extracted_release, AUTHORITATIVE_MANIFEST, authoritative_expected
    )
    supplementary_expected = actual_set - {
        AUTHORITATIVE_MANIFEST,
        GENERIC_MANIFEST,
        SUPPLEMENTARY_MANIFEST,
    }
    supplementary_count = _verify_manifest_inventory(
        extracted_release, SUPPLEMENTARY_MANIFEST, supplementary_expected
    )
    if (extracted_release / GENERIC_MANIFEST).read_bytes() != (
        extracted_release / AUTHORITATIVE_MANIFEST
    ).read_bytes():
        raise RuntimeError("Generic and authoritative SHA manifests differ")
    return {
        "status": "PASS",
        "listed_file_count": len(listed),
        "verified_authoritative_hash_count": authoritative_count,
        "verified_supplementary_hash_count": supplementary_count,
        "authoritative_manifest": AUTHORITATIVE_MANIFEST,
    }


def _validate_version(source: Path) -> dict[str, object]:
    version_path = source / "VERSION.json"
    version = _read_json_object(version_path)
    _require_field(version, version_path, "version", RELEASE_VERSION)
    _require_field(version, version_path, "release_directory_name", RELEASE_NAME)
    paper = version.get("paper_reactive_v8_3")
    if not isinstance(paper, dict) or paper.get("final_verdict") != "NOT_FULLY_VERIFIED":
        raise RuntimeError("VERSION.json has an invalid v8.3 final verdict")
    statuses = version.get("v8_3_test_status")
    if not isinstance(statuses, dict) or not statuses:
        raise RuntimeError("VERSION.json has no v8_3_test_status object")
    pending = sorted(
        str(key)
        for key, value in statuses.items()
        if isinstance(value, str) and "pending" in value.lower()
    )
    if pending:
        raise RuntimeError(
            f"VERSION.json v8_3_test_status still contains pending fields: {pending}"
        )
    return version


def _rollback_published(paths: list[Path]) -> None:
    for path in reversed(paths):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def _publish_verified(
    pairs: list[tuple[Path, Path]], *, all_targets: tuple[Path, ...]
) -> None:
    for target in all_targets:
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"Refusing to overwrite existing release artifact: {target}")
    published: list[Path] = []
    try:
        for staged, target in pairs:
            os.rename(staged, target)
            published.append(target)
    except BaseException:
        _rollback_published(published)
        raise


def build_release(
    *, source: Path, parent: Path, output_root: Path, validation_dir: Path
) -> dict[str, object]:
    source = source.expanduser().resolve(strict=True)
    parent = parent.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=True)
    validation_dir = validation_dir.expanduser().resolve(strict=True)
    if source.name != RELEASE_NAME:
        raise RuntimeError(f"Source directory must be named exactly {RELEASE_NAME}")
    destination = output_root / RELEASE_NAME
    archive = output_root / f"{RELEASE_NAME}.zip"
    checksum = output_root / f"{RELEASE_NAME}.zip.sha256.txt"
    completion = validation_dir / "RELEASE_BUILD_COMPLETE.json"
    final_targets = (destination, archive, checksum, completion)
    for path in final_targets:
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Refusing to overwrite existing release artifact: {path}")

    missing_docs = sorted(path for path in REQUIRED_DOCS if not (source / path).is_file())
    if missing_docs:
        raise RuntimeError(f"Required release documents are missing: {missing_docs}")
    _validate_version(source)
    document_validation = _validate_release_documents(source)
    parent_validation = _verify_parent_release(parent)
    source_tree_sha256 = _source_tree_digest(source)
    evidence, evidence_semantics = _validate_evidence(
        validation_dir,
        source=source,
        parent=parent,
        source_tree_sha256=source_tree_sha256,
    )

    source_parent_manifest = source / PARENT_MANIFEST
    parent_manifest = parent / PARENT_MANIFEST
    if source_parent_manifest.exists() and (
        source_parent_manifest.is_symlink()
        or _sha256(source_parent_manifest) != _sha256(parent_manifest)
    ):
        raise RuntimeError("Source contains a non-historical v8.2.1 parent manifest")

    with tempfile.TemporaryDirectory(prefix="ecsp-v830-build-", dir=output_root) as temporary:
        transaction = Path(temporary)
        staging = transaction / RELEASE_NAME
        staging.mkdir()
        _copy_source(source, staging)
        shutil.copy2(parent_manifest, staging / PARENT_MANIFEST)
        build_metadata = {
            "schema": "ecsp.release-build-input/v2",
            "release": RELEASE_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": "NOT_FULLY_VERIFIED",
            "validated_source_tree_sha256": source_tree_sha256,
            "source_tree_digest_scope": (
                "non-generated regular files copied by build_reactive_v8_3_release.py"
            ),
            "validation_evidence": evidence,
            "validation_evidence_semantics": evidence_semantics,
            "release_document_validation": document_validation,
            "parent_validation": parent_validation,
            "actual_cuda_or_a100_claim": "NOT_TESTED_NO_HARDWARE_OR_TOOLCHAIN",
            "actual_mpi_claim": "NOT_TESTED_NO_MPI_RUNTIME",
        }
        _atomic_bytes(staging / BUILD_METADATA, _json_bytes(build_metadata))
        file_count, manifest_sha256 = _write_manifests(staging)

        temporary_archive = transaction / archive.name
        temporary_checksum = transaction / checksum.name
        temporary_completion = transaction / completion.name
        _write_deterministic_zip(staging, temporary_archive)
        archive_sha256 = _sha256(temporary_archive)
        _atomic_bytes(
            temporary_checksum,
            f"{archive_sha256}  {archive.name}\n".encode("utf-8"),
        )
        with zipfile.ZipFile(temporary_archive) as package:
            bad_member = package.testzip()
            if bad_member is not None:
                raise RuntimeError(f"ZIP CRC failure: {bad_member}")
            extract_root = transaction / "fresh-extract"
            package.extractall(extract_root)
        fresh = _verify_extracted(extract_root / RELEASE_NAME)
        report = {
            "schema": "ecsp.release-build-complete/v2",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "release": RELEASE_VERSION,
            "verdict": "NOT_FULLY_VERIFIED",
            "release_directory": str(destination),
            "archive": str(archive),
            "archive_sha256": archive_sha256,
            "checksum_file": str(checksum),
            "package_file_count": file_count,
            "authoritative_manifest": AUTHORITATIVE_MANIFEST,
            "authoritative_manifest_sha256": manifest_sha256,
            "validated_source_tree_sha256": source_tree_sha256,
            "zip_crc": "PASS",
            "fresh_extract_manifest": fresh,
            "validation_evidence": evidence,
            "validation_evidence_semantics": evidence_semantics,
            "parent_validation": parent_validation,
            "unverified": [
                "paper quantitative reproduction: missing unpublished inputs",
                "reactive MPI ranks: no runtime and no integrated MPI solver",
                "reactive CUDA/A100: not implemented or tested",
                "Linux GCC and sanitizer matrix: not executed on this macOS host",
                "experimental predictive calibration: not performed",
            ],
        }
        _atomic_bytes(temporary_completion, _json_bytes(report))

        # All validation above operates only on transaction-local artifacts.
        # If any publication rename fails, remove only the targets created by
        # this transaction so a failed build never leaves a partial release.
        _publish_verified(
            [
                (staging, destination),
                (temporary_archive, archive),
                (temporary_checksum, checksum),
                (temporary_completion, completion),
            ],
            all_targets=final_targets,
        )
    return report


def main() -> int:
    args = _args()
    try:
        report = build_release(
            source=args.source,
            parent=args.parent,
            output_root=args.output_root,
            validation_dir=args.validation_dir,
        )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
