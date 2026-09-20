"""Atomic artifacts, content identity and candidate-level restart records."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Mapping
import csv
import hashlib
import io
import json
import math
import os
import platform
import tempfile

import numpy as np
from PIL import Image, ImageDraw


def json_safe(value):
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def mask_hash(anode, cathode) -> str:
    h = hashlib.sha256()
    for mask in (anode, cathode):
        a = np.ascontiguousarray(mask, dtype=np.uint8)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def atomic_bytes(path: Path, content: bytes, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Immutable selection/config artifact differs: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value, *, immutable=False):
    atomic_bytes(Path(path), (canonical_json(value) + "\n").encode(), immutable=immutable)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, *, fieldnames=(), immutable=False):
    rows = list(rows)
    columns = list(fieldnames)
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: canonical_json(value) if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()})
    atomic_bytes(Path(path), stream.getvalue().encode(), immutable=immutable)


class RejectionLog:
    """Append and flush every rejected sample, including interrupted generation."""
    FIELDS = ("topology_id", "attempt", "reason", "details")

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            write_csv(self.path, [], fieldnames=self.FIELDS)

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Mapping):
            data = dict(args[0])
            data.update(kwargs)
        else:
            data = dict(zip(self.FIELDS, args))
            data.update(kwargs)
        row = {key: data.get(key, "") for key in self.FIELDS}
        details = data.get("details", {})
        extras = {k: v for k, v in data.items() if k not in self.FIELDS}
        row["details"] = canonical_json({"validation": details, **extras})
        with self.path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.FIELDS).writerow(row)
            stream.flush()
            os.fsync(stream.fileno())


def render_mask(anode, cathode, size=320):
    rgb = np.full((*np.shape(anode), 3), 248, dtype=np.uint8)
    rgb[np.asarray(anode, bool)] = (211, 68, 55)
    rgb[np.asarray(cathode, bool)] = (39, 99, 178)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


def save_record(path, candidate, geometry_id, stage):
    """Persist CAD metadata and both grids; return the solver-facing record."""
    path = Path(path)
    metadata = dict(candidate.metadata)
    metadata.update(geometry_id=geometry_id, stage=stage)
    metadata["physics_mask_hash"] = mask_hash(candidate.anode_mask, candidate.cathode_mask)
    metadata.setdefault("geometry_hash", digest({"metadata": metadata,
        "mask": metadata["physics_mask_hash"]}))
    polygons = getattr(candidate, "polygons", None)
    if polygons is not None:
        metadata["cad_polygons"] = json_safe(polygons)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, anode_mask=np.asarray(candidate.anode_mask, bool),
                        cathode_mask=np.asarray(candidate.cathode_mask, bool),
                        design_anode_mask=np.asarray(candidate.design_anode_mask, bool),
                        design_cathode_mask=np.asarray(candidate.design_cathode_mask, bool))
    atomic_bytes(path.with_suffix(".npz"), buffer.getvalue())
    image_buffer = io.BytesIO()
    render_mask(candidate.design_anode_mask, candidate.design_cathode_mask).save(image_buffer, format="PNG")
    atomic_bytes(path.with_suffix(".png"), image_buffer.getvalue())
    write_json(path.with_suffix(".json"), metadata)
    return load_record(path)


def load_record(path):
    path = Path(path)
    metadata = read_json(path.with_suffix(".json"))
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        a, c = arrays["anode_mask"].copy(), arrays["cathode_mask"].copy()
    if mask_hash(a, c) != metadata["physics_mask_hash"]:
        raise RuntimeError(f"Persisted raster hash mismatch: {path}")
    return {"metadata": metadata, "anode_mask": a, "cathode_mask": c, "artifact_path": str(path)}


def contact_sheet(paths, output, columns=10, tile=160):
    paths = list(paths)
    if not paths:
        return
    canvas = Image.new("RGB", (columns * tile, math.ceil(len(paths) / columns) * (tile + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        path = Path(path)
        x, y = (index % columns) * tile, (index // columns) * (tile + 24)
        with Image.open(path) as img:
            canvas.paste(img.resize((tile, tile)), (x, y))
        draw.text((x + 4, y + tile + 3), path.stem, fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    atomic_bytes(Path(output), buffer.getvalue())


def source_fingerprint(package_root):
    root = Path(package_root)
    paths = []
    for directory in ("python/ecsp_doe", "python/ecsp_nsga2", "python/ecsp_v6", "cpp"):
        paths.extend(p for p in (root / directory).rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp"))
    for name in ("tools/run_phidl_doe600.py", "python/requirements-phidl-doe.txt"):
        if (root / name).exists():
            paths.append(root / name)
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    versions = {"python": platform.python_version()}
    for package in ("phidl", "gdspy", "shapely", "networkx", "scikit-image", "numpy", "scipy", "Pillow", "torch"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


@contextmanager
def run_lock(path):
    import fcntl
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".doe.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another DOE process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
