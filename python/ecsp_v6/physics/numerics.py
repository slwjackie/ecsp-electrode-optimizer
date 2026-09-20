from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def resolve_device(value: str | None) -> torch.device:
    requested = (value or "auto").lower()
    mps_backend = getattr(torch.backends, "mps", None)
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if device.type == "mps" and (
        mps_backend is None or not mps_backend.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is unavailable")
    return device


def resolve_dtype(value: str | None) -> torch.dtype:
    text = (value or "float64").lower()
    if text in {"float64", "double", "fp64"}:
        return torch.float64
    if text in {"float32", "single", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported physics dtype: {value}")


def validate_device_dtype(device: torch.device, dtype: torch.dtype) -> None:
    """Reject precision/backend combinations that PyTorch cannot execute safely."""
    if device.type == "mps" and dtype == torch.float64:
        raise RuntimeError(
            "MPS physics requires numerics.physicsDtype=float32 because PyTorch MPS "
            "does not support float64 tensors. Use config/m2_mps.yaml (or an "
            "equivalent float32 override) for MPS, and keep CPU/CUDA FP64 for "
            "reference/final verification runs."
        )


def configure_torch(device: torch.device, deterministic: bool = True) -> None:
    """Configure numerical backends without mutating thread-local grad state."""
    if device.type == "cuda":
        # Physics kernels use explicit FP64 by default. Disable TF32 so that
        # accidental FP32 matrix operations do not silently alter parity.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def as_tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def ensure_batch(field: torch.Tensor) -> torch.Tensor:
    if field.ndim == 2:
        return field.unsqueeze(0)
    if field.ndim != 3:
        raise ValueError(f"Expected [H,W] or [B,H,W], got {tuple(field.shape)}")
    return field


def physical_floor(config: dict | None, key: str, default: float) -> float:
    """Return an explicit, dtype-independent physical/numerical floor.

    Machine epsilon is a property of the storage type, not of the ECSP model.
    Quantities that enter reported metrics or constitutive closures therefore
    use named floors from ``numerics.physicalFloors`` so CPU FP64 and MPS FP32
    differ only through arithmetic accuracy, not through nine orders of
    magnitude of implicit clipping.
    """
    if config is None:
        return float(default)
    floors = config.get("numerics", {}).get("physicalFloors", {})
    return float(floors.get(key, default))


def harmonic_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Overflow-safe physical harmonic mean with explicit invalid propagation."""
    finite_nonnegative = (
        torch.isfinite(a) & torch.isfinite(b) & (a >= 0.0) & (b >= 0.0)
    )
    low = torch.minimum(a, b)
    high = torch.maximum(a, b)
    positive = finite_nonnegative & (low > 0.0)
    safe_high = torch.where(positive, high, torch.ones_like(high))
    # low * 2/(1+low/high) avoids both a*b and a+b overflow while preserving
    # the smaller operand even when low/high underflows for extreme ratios.
    value = low * (2.0 / (1.0 + low / safe_high))
    zero_or_value = torch.where(positive, value, torch.zeros_like(value))
    # A non-finite/negative constitutive input is not an insulating face.
    # Preserve it as NaN so the caller's convergence/physics gate fails closed.
    return torch.where(
        finite_nonnegative,
        zero_or_value,
        torch.full_like(zero_or_value, torch.nan),
    )


def apply_neumann_boundary(field: torch.Tensor) -> torch.Tensor:
    """Match the MATLAB boundary-copy convention on a cloned tensor."""
    result = field.clone()
    result[..., 0, :] = result[..., 1, :]
    result[..., -1, :] = result[..., -2, :]
    result[..., :, 0] = result[..., :, 1]
    result[..., :, -1] = result[..., :, -2]
    return result


def matlab_gradient(field: torch.Tensor, spacing_x: float, spacing_y: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return derivatives in MATLAB gradient order: d/dx(columns), d/dy(rows).

    Interior points use centered differences; boundaries use first-order
    one-sided differences, matching MATLAB's default gradient convention.
    """
    spacing_y = spacing_x if spacing_y is None else spacing_y
    f = ensure_batch(field)
    dx = torch.empty_like(f)
    dy = torch.empty_like(f)
    dx[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / (2.0 * spacing_x)
    dx[..., :, 0] = (f[..., :, 1] - f[..., :, 0]) / spacing_x
    dx[..., :, -1] = (f[..., :, -1] - f[..., :, -2]) / spacing_x
    dy[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2.0 * spacing_y)
    dy[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / spacing_y
    dy[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / spacing_y
    return dx, dy


def laplacian_neumann(field: torch.Tensor, spacing: float) -> torch.Tensor:
    f = ensure_batch(field)
    padded = F.pad(f.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    return (
        padded[..., 1:-1, 2:]
        + padded[..., 1:-1, :-2]
        + padded[..., 2:, 1:-1]
        + padded[..., :-2, 1:-1]
        - 4.0 * padded[..., 1:-1, 1:-1]
    ) / (spacing * spacing)


def _positive_half_up_indices(source_size: int, target_size: int) -> np.ndarray:
    """Nearest-neighbour indices with positive half-up rounding.

    MATLAB's positive-coordinate ``round`` convention rounds exact half ties
    upward, whereas ``numpy.rint`` uses bankers' rounding.  Explicitly using
    floor(x+0.5) removes this otherwise rare mask-boundary discrepancy.
    """
    coordinates = np.linspace(0.0, float(source_size - 1), int(target_size))
    return np.clip(np.floor(coordinates + 0.5).astype(np.int64), 0, source_size - 1)


def resize_nearest_numpy(mask: np.ndarray, target: int | tuple[int, int]) -> np.ndarray:
    if isinstance(target, int):
        target = (target, target)
    rows, columns = mask.shape
    row_index = _positive_half_up_indices(rows, target[0])
    column_index = _positive_half_up_indices(columns, target[1])
    # A NumPy array obtained from a Pillow mode-1 image can have dtype bool
    # while retaining 0xff for true pixels.  Fancy indexing copies those raw
    # bytes unchanged.  Convert through uint8 by value so every downstream
    # NumPy/Torch mask owns a canonical 0/1 bool representation.
    canonical = np.asarray(mask, dtype=np.uint8) != 0
    return canonical[np.ix_(row_index, column_index)]


def resize_linear_tensor(field: torch.Tensor, target: int | tuple[int, int]) -> torch.Tensor:
    if isinstance(target, int):
        target = (target, target)
    f = ensure_batch(field).unsqueeze(1)
    return F.interpolate(f, size=target, mode="bilinear", align_corners=True).squeeze(1)


def resize_nearest_tensor(field: torch.Tensor, target: int | tuple[int, int]) -> torch.Tensor:
    if isinstance(target, int):
        target = (target, target)
    f = ensure_batch(field)
    row_index = torch.as_tensor(
        _positive_half_up_indices(f.shape[-2], target[0]),
        device=f.device,
        dtype=torch.long,
    )
    column_index = torch.as_tensor(
        _positive_half_up_indices(f.shape[-1], target[1]),
        device=f.device,
        dtype=torch.long,
    )
    return f.index_select(-2, row_index).index_select(-1, column_index) > 0.5


def percentile_linear(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile, method="linear"))


def batch_masked_mean(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    maskf = mask.to(field.dtype)
    return (field * maskf).sum(dim=(-2, -1)) / torch.clamp(maskf.sum(dim=(-2, -1)), min=1.0)


def batch_masked_max(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg = torch.full_like(field, -torch.inf)
    return torch.where(mask, field, neg).amax(dim=(-2, -1))


def batch_quantile_masked(field: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    """Fixed-shape masked linear quantile suitable for Apple MPS.

    Boolean advanced indexing creates a different tensor shape for every
    candidate and can force repeated synchronization on MPS.  Invalid entries
    are instead moved to the end of a fixed-size sorted row, after which the
    two linear-interpolation order statistics are gathered.
    """
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError(f"q must lie in [0,1], got {q}")
    values_batch = ensure_batch(field)
    valid_batch = ensure_batch(mask)
    if values_batch.shape != valid_batch.shape:
        raise ValueError(
            "field and mask must have the same batched shape; "
            f"got {tuple(values_batch.shape)} and {tuple(valid_batch.shape)}"
        )
    values = values_batch.reshape(values_batch.shape[0], -1)
    valid = valid_batch.reshape(valid_batch.shape[0], -1).to(torch.bool)
    counts = valid.sum(dim=1)
    infinity = torch.full_like(values, torch.inf)
    ordered = torch.sort(torch.where(valid, values, infinity), dim=1).values
    safe_counts = torch.clamp(counts, min=1)
    position = float(q) * (safe_counts.to(field.dtype) - 1.0)
    lower = torch.floor(position).to(torch.long)
    upper = torch.ceil(position).to(torch.long)
    lower_value = ordered.gather(1, lower[:, None]).squeeze(1)
    upper_value = ordered.gather(1, upper[:, None]).squeeze(1)
    weight = position - lower.to(field.dtype)
    result = lower_value + weight * (upper_value - lower_value)
    nan = torch.full_like(result, torch.nan)
    return torch.where(counts > 0, result, nan)


@dataclass
class SolverDiagnostics:
    iterations: torch.Tensor
    relative_residual: torch.Tensor
    converged: torch.Tensor
    method: str
    restarts: torch.Tensor | None = None
    refinement_rounds: torch.Tensor | None = None
    fallback_used: torch.Tensor | None = None

    def to_cpu_dicts(self) -> list[dict[str, Any]]:
        iterations = self.iterations.detach().cpu().numpy()
        residual = self.relative_residual.detach().cpu().numpy()
        converged = self.converged.detach().cpu().numpy()
        restarts = (
            self.restarts.detach().cpu().numpy() if self.restarts is not None else None
        )
        rounds = (
            self.refinement_rounds.detach().cpu().numpy()
            if self.refinement_rounds is not None
            else None
        )
        fallback = (
            self.fallback_used.detach().cpu().numpy()
            if self.fallback_used is not None
            else None
        )
        result: list[dict[str, Any]] = []
        for i in range(len(iterations)):
            row: dict[str, Any] = {
                "iterations": int(iterations[i]),
                "relative_residual": float(residual[i]),
                "converged": bool(converged[i]),
                "method": self.method,
            }
            if restarts is not None:
                row["restarts"] = int(restarts[i])
            if rounds is not None:
                row["refinement_rounds"] = int(rounds[i])
            if fallback is not None:
                row["fallback_used"] = bool(fallback[i])
            result.append(row)
        return result
