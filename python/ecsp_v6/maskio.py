from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def save_geometry(root: Path, geometry_id: str, anode: np.ndarray, cathode: np.ndarray) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{geometry_id}.npz"
    mat_path = root / f"{geometry_id}.mat"
    np.savez_compressed(npz_path, anodeMask=anode.astype(np.uint8), cathodeMask=cathode.astype(np.uint8))
    savemat(mat_path, {"anodeMask": anode.astype(bool), "cathodeMask": cathode.astype(bool)})
    return {"npz_path": str(npz_path), "mat_path": str(mat_path)}


def load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return data["anodeMask"].astype(bool), data["cathodeMask"].astype(bool)
    if path.suffix.lower() == ".mat":
        data = loadmat(path)
        return data["anodeMask"].astype(bool), data["cathodeMask"].astype(bool)
    raise ValueError(f"Unsupported geometry format: {path}")


def load_library(root: Path) -> Iterator[tuple[pd.Series, np.ndarray, np.ndarray]]:
    manifest = pd.read_csv(root / "geometry_manifest.csv")
    for _, row in manifest.iterrows():
        anode, cathode = load_geometry(Path(row["npz_path"]))
        yield row, anode, cathode
