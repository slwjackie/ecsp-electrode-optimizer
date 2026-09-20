"""Electrical source terms shared by paper-reproduction and ECSP-extended modes.

The paper specifies ``q_J = sigma |grad(V)|^2`` but does not publish a
potential equation or the measured fields.  Consequently this module accepts a
prescribed potential/conductivity field in paper mode.  It never calls that a
solved electrical field.  Extended mode may supply a separately solved heat
field from the existing surface-contact/BV implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .provenance import ReactiveConfigurationError, ReactiveNumericalError


def canonical_bool_mask(mask: Any, name: str) -> np.ndarray:
    """Return a C/CUDA-safe, contiguous NumPy bool mask with raw bytes 0/1."""
    try:
        raw = np.asarray(mask)
    except (TypeError, ValueError) as exc:
        raise ReactiveConfigurationError(
            f"{name} must be convertible to a numeric two-dimensional mask"
        ) from exc
    if raw.ndim != 2:
        raise ReactiveConfigurationError(f"{name} must be a two-dimensional mask")
    if raw.dtype.kind == "b":
        # A NumPy/Torch bool may still contain a noncanonical Pillow mode-1
        # byte such as 0xff.  Inspect its storage as bytes before any C++ bool
        # interpretation, then materialise owned canonical 0/1 storage.
        truth = raw.view(np.uint8) != 0
    elif raw.dtype.kind in {"i", "u", "f"}:
        if raw.dtype.kind == "f" and not np.isfinite(raw).all():
            raise ReactiveConfigurationError(f"{name} contains NaN or infinity")
        truth = raw != 0
    else:
        raise ReactiveConfigurationError(
            f"{name} dtype {raw.dtype} is not a supported bool/integer/real mask"
        )
    result = np.ascontiguousarray(truth, dtype=np.bool_)
    bytes_view = result.view(np.uint8)
    if not np.all((bytes_view == 0) | (bytes_view == 1)):
        raise ReactiveConfigurationError(f"{name} did not canonicalise to raw 0/1 bytes")
    return result


def _immutable_bool_mask(mask: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(mask, dtype=np.bool_)
    # Keep the canonical raw 0/1 representation on an immutable bytes base.
    # Unlike ndarray.writeable=False, this cannot be reversed by setflags().
    return np.frombuffer(canonical.view(np.uint8).tobytes(), dtype=np.bool_).reshape(
        canonical.shape
    )


@dataclass(frozen=True)
class SurfaceContactGeometry:
    propellant: np.ndarray
    anode_contact: np.ndarray
    cathode_contact: np.ndarray

    @classmethod
    def build(cls, anode_contact: Any, cathode_contact: Any) -> "SurfaceContactGeometry":
        anode = canonical_bool_mask(anode_contact, "anode_contact")
        cathode = canonical_bool_mask(cathode_contact, "cathode_contact")
        if anode.shape != cathode.shape:
            raise ReactiveConfigurationError("Anode and cathode contact masks must have one shape")
        if np.any(anode & cathode):
            raise ReactiveConfigurationError("Anode and cathode contact masks overlap")
        if not np.any(anode) or not np.any(cathode):
            raise ReactiveConfigurationError("Both surface-contact polarities must be nonempty")
        # Contacts are labels on the top surface.  They do not remove condensed
        # propellant cells or any thermochemical/electrical state below them.
        propellant = np.ones(anode.shape, dtype=np.bool_)
        return cls(
            propellant=_immutable_bool_mask(propellant),
            anode_contact=_immutable_bool_mask(anode),
            cathode_contact=_immutable_bool_mask(cathode),
        )


def gradient_2d(
    field: np.ndarray,
    dx: float,
    dy: float,
    *,
    boundary_scheme: str,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(field, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 3:
        raise ReactiveConfigurationError("gradient_2d requires a 2-D field with at least 3 cells per axis")
    if not np.isfinite(value).all() or not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        raise ReactiveConfigurationError("Invalid field or spacing for electrical gradient")
    if boundary_scheme != "FIRST_ORDER_ONE_SIDED":
        raise ReactiveConfigurationError(
            "electrical gradient boundary_scheme must be FIRST_ORDER_ONE_SIDED"
        )
    gx = np.empty_like(value)
    gy = np.empty_like(value)
    gx[:, 1:-1] = (value[:, 2:] - value[:, :-2]) / (2.0 * dx)
    gx[:, 0] = (value[:, 1] - value[:, 0]) / dx
    gx[:, -1] = (value[:, -1] - value[:, -2]) / dx
    gy[1:-1, :] = (value[2:, :] - value[:-2, :]) / (2.0 * dy)
    gy[0, :] = (value[1, :] - value[0, :]) / dy
    gy[-1, :] = (value[-1, :] - value[-2, :]) / dy
    return gx, gy


def joule_heating_sigma_e2(
    electric_potential_V: np.ndarray,
    conductivity_S_per_m: np.ndarray | float,
    dx_m: float,
    dy_m: float,
    *,
    gradient_boundary_scheme: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate paper Eq.(1)'s Joule source in W/m^3."""
    potential = np.asarray(electric_potential_V, dtype=np.float64)
    sigma = np.broadcast_to(np.asarray(conductivity_S_per_m, dtype=np.float64), potential.shape)
    if not np.isfinite(sigma).all() or np.any(sigma < 0.0):
        raise ReactiveConfigurationError("Electrical conductivity must be finite and non-negative")
    gx, gy = gradient_2d(
        potential,
        dx_m,
        dy_m,
        boundary_scheme=gradient_boundary_scheme,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        heat = sigma * (gx * gx + gy * gy)
    if not np.isfinite(heat).all() or np.any(heat < 0.0):
        raise ReactiveNumericalError(
            "nonfinite_joule_heating",
            "sigma*|grad(V)|^2 produced a non-finite or negative source",
            {"maximum_abs_gradient_V_per_m": float(np.nanmax(np.hypot(gx, gy)))},
        )
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_heat = heat * (dx_m * dy_m)
        integral = np.sum(scaled_heat)
    if not np.isfinite(integral):
        raise ReactiveNumericalError(
            "nonfinite_joule_integral",
            "Joule heat integral overflowed after cell-wise area scaling",
        )
    return heat, {
        "electric_gradient_boundary_scheme": gradient_boundary_scheme,
        "joule_heat_min_W_per_m3": float(np.min(heat)),
        "joule_heat_max_W_per_m3": float(np.max(heat)),
        "joule_heat_integral_per_depth_W_per_m": float(integral),
    }


def surface_heat_flux_to_volume(
    contact_mask: np.ndarray,
    heat_flux_W_per_m2: np.ndarray | float,
    active_layer_thickness_m: float,
) -> np.ndarray:
    """Convert a footprint source to W/m^3 without using its perimeter."""
    mask = canonical_bool_mask(contact_mask, "contact_mask")
    flux = np.broadcast_to(np.asarray(heat_flux_W_per_m2, dtype=np.float64), mask.shape)
    if (
        not np.isfinite(flux).all()
        or np.any(flux < 0.0)
        or not np.isfinite(active_layer_thickness_m)
        or active_layer_thickness_m <= 0.0
    ):
        raise ReactiveConfigurationError("Invalid surface heat flux or active-layer thickness")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.where(mask, flux / active_layer_thickness_m, 0.0)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ReactiveNumericalError(
            "nonfinite_surface_volume_source",
            "Surface heat-flux to volume conversion overflowed or became invalid",
        )
    if int(np.count_nonzero(result)) != int(np.count_nonzero(mask & (flux > 0.0))):
        raise ReactiveNumericalError(
            "surface_source_footprint_mismatch",
            "Surface heat source was not applied on the complete nonzero contact footprint",
        )
    return result


@dataclass(frozen=True)
class ElectricalSourceFields:
    joule_heat_W_per_m3: np.ndarray
    electrochemical_heat_W_per_m3: np.ndarray
    electric_potential_V: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def total_heat_W_per_m3(self) -> np.ndarray:
        total = self.joule_heat_W_per_m3 + self.electrochemical_heat_W_per_m3
        if not np.isfinite(total).all():
            raise ReactiveNumericalError(
                "nonfinite_electrical_heat_sum",
                "Joule plus electrochemical heat became non-finite",
            )
        return total
