#!/usr/bin/env python3
"""Verify the current release SHA-256 manifest and package file inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "runs",
    "build",
    ".native_build",
}


def _included(root: Path, path: Path, manifest: Path, generic_alias: Path) -> bool:
    if (
        not path.is_file()
        or path.is_symlink()
        or path == manifest
        or path == generic_alias
        or path.name == ".DS_Store"
    ):
        return False
    parts = path.relative_to(root).parts
    if any(
        part in EXCLUDED_DIRS
        or part.startswith("tmp_")
        or part.startswith(".tmp_")
        for part in parts
    ):
        return False
    return path.suffix not in {".pyc", ".pyo"}


def _default_manifest(root: Path) -> Path:
    version = json.loads((root / "VERSION.json").read_text(encoding="utf-8"))[
        "version"
    ]
    parts = str(version).split("-", 1)[0].split(".")
    if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
        raise ValueError(f"Cannot derive manifest tag from version {version!r}")
    # v8.2.0 shipped before patch-level manifest names were introduced and its
    # historical authoritative name must remain verifiable.  Patch releases
    # use all three SemVer components so v8.2.1 cannot overwrite or masquerade
    # as the immutable v8.2.0 manifest.
    tag_parts = parts[:2] if int(parts[2]) == 0 else parts[:3]
    return root / f"PACKAGE_SHA256_MANIFEST_V{'_'.join(tag_parts)}.txt"


def _read_manifest(root: Path, manifest: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        pieces = raw.split(maxsplit=1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise ValueError(f"Malformed manifest line {line_number}: {raw!r}")
        digest, relative = pieces
        if any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise ValueError(f"Invalid SHA-256 on manifest line {line_number}")
        relative = relative.removeprefix("./")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in entries:
            raise ValueError(f"Unsafe or duplicate path on manifest line {line_number}")
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Manifest path escapes package root: {relative}")
        entries[relative] = digest.lower()
    return entries


def verify(root: Path, manifest: Path) -> dict:
    root = root.resolve()
    manifest = manifest.resolve()
    if not root.is_dir():
        raise ValueError(f"Package root is not a directory: {root}")
    expected_manifest = _default_manifest(root).resolve()
    if manifest != expected_manifest or manifest.parent != root:
        raise ValueError(
            "Authoritative manifest must be the VERSION-derived file inside "
            f"the package root: {expected_manifest.name}"
        )
    generic_alias = root / "PACKAGE_SHA256_MANIFEST.txt"
    entries = _read_manifest(root, manifest)
    unsupported_symlinks = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    )
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if _included(root, path, manifest, generic_alias)
    }
    listed_paths = set(entries)
    missing = sorted(listed_paths - actual_paths)
    unlisted = sorted(actual_paths - listed_paths)
    mismatched = []
    for relative in sorted(listed_paths & actual_paths):
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if digest != entries[relative]:
            mismatched.append(
                {"path": relative, "expected_sha256": entries[relative], "actual_sha256": digest}
            )
    generic_alias_is_regular = generic_alias.is_file() and not generic_alias.is_symlink()
    generic_alias_matches = bool(
        generic_alias_is_regular
        and generic_alias.read_bytes() == manifest.read_bytes()
    )
    return {
        "status": (
            "passed"
            if (
                not missing
                and not unlisted
                and not mismatched
                and not unsupported_symlinks
                and generic_alias_is_regular
                and generic_alias_matches
            )
            else "failed"
        ),
        "manifest": manifest.name,
        "authoritative_manifest_matches_version": True,
        "generic_alias": generic_alias.name,
        "generic_alias_is_regular_file": generic_alias_is_regular,
        "generic_alias_matches_authoritative": generic_alias_matches,
        "listed_file_count": len(entries),
        "verified_file_count": len(entries) - len(missing) - len(mismatched),
        "missing": missing,
        "unlisted": unlisted,
        "mismatched": mismatched,
        "unsupported_symlinks": unsupported_symlinks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    manifest = (args.manifest or _default_manifest(root)).resolve()
    report = verify(root, manifest)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
