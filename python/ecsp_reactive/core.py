"""FP64 conservative reactive-Euler state and Park & Yoh Eq. (1) fluxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .eos import EquationOfState, TaitEOS
from .provenance import ReactiveConfigurationError, ReactiveNumericalError


NCONS = 5
RHO = 0
MOMENTUM_X = 1
MOMENTUM_Y = 2
TOTAL_ENERGY = 3
REACTION_PROGRESS = 4


@dataclass(frozen=True)
class PrimitiveState:
    density_kg_per_m3: np.ndarray
    velocity_x_m_per_s: np.ndarray
    velocity_y_m_per_s: np.ndarray
    pressure_Pa: np.ndarray
    temperature_K: np.ndarray
    reaction_progress: np.ndarray
    specific_internal_energy_J_per_kg: np.ndarray


def _state_fp64(state: Any, *, require_finite: bool) -> np.ndarray:
    try:
        result = np.asarray(state, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReactiveConfigurationError(
            "conservative state must be convertible to FP64"
        ) from exc
    if result.ndim < 1 or result.shape[-1] != NCONS:
        raise ReactiveConfigurationError(
            f"conservative state must have last dimension {NCONS}, received {result.shape}"
        )
    if require_finite and not np.isfinite(result).all():
        raise ReactiveNumericalError(
            "nonfinite_conservative_state",
            "conservative state contains NaN or infinity",
        )
    return result


def _broadcast_fp64(*values: Any) -> tuple[np.ndarray, ...]:
    try:
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
        return tuple(np.broadcast_arrays(*arrays))
    except (TypeError, ValueError) as exc:
        raise ReactiveConfigurationError(
            "primitive fields must be FP64-broadcastable"
        ) from exc


def primitive_to_conservative(
    density_kg_per_m3: Any,
    velocity_x_m_per_s: Any,
    velocity_y_m_per_s: Any,
    pressure_Pa: Any | None,
    reaction_progress: Any,
    eos: EquationOfState,
    *,
    temperature_K: Any | None = None,
    eos_consistency_rtol: float = 1.0e-10,
) -> np.ndarray:
    """Build ``U=[rho,rho*u,rho*v,rho*E,rho*lambda]``.

    Tait pressure does not determine temperature, so ``temperature_K`` is
    mandatory for :class:`TaitEOS`.  For an ideal gas, pressure alone is
    sufficient; if temperature is also supplied, both must agree.
    """
    if not np.isfinite(eos_consistency_rtol) or eos_consistency_rtol < 0.0:
        raise ReactiveConfigurationError(
            "eos_consistency_rtol must be finite and non-negative"
        )

    if pressure_Pa is None and temperature_K is None:
        raise ReactiveConfigurationError(
            "at least pressure_Pa or temperature_K is required"
        )

    common = [
        density_kg_per_m3,
        velocity_x_m_per_s,
        velocity_y_m_per_s,
        reaction_progress,
    ]
    if pressure_Pa is not None:
        common.append(pressure_Pa)
    if temperature_K is not None:
        common.append(temperature_K)
    broadcast = _broadcast_fp64(*common)
    rho, velocity_x, velocity_y, progress = broadcast[:4]
    offset = 4
    pressure = broadcast[offset] if pressure_Pa is not None else None
    offset += int(pressure_Pa is not None)
    temperature = broadcast[offset] if temperature_K is not None else None

    if not all(np.isfinite(field).all() for field in broadcast):
        raise ReactiveNumericalError(
            "nonfinite_primitive_state", "primitive state contains NaN or infinity"
        )
    if np.any(rho <= 0.0):
        raise ReactiveNumericalError(
            "invalid_density", "density must be strictly positive"
        )
    if np.any((progress < 0.0) | (progress > 1.0)):
        raise ReactiveNumericalError(
            "reaction_progress_out_of_range", "reaction progress must be in [0, 1]"
        )

    if temperature is not None:
        specific_internal = np.asarray(
            eos.specific_internal_energy_from_temperature(rho, temperature),
            dtype=np.float64,
        )
        implied_pressure = np.asarray(
            eos.pressure_from_rho_e(rho, specific_internal), dtype=np.float64
        )
        if pressure is not None and not np.allclose(
            pressure, implied_pressure, rtol=eos_consistency_rtol, atol=1.0e-8
        ):
            mismatch = float(np.max(np.abs(pressure - implied_pressure)))
            raise ReactiveConfigurationError(
                f"pressure and temperature violate the selected EOS (max mismatch {mismatch:.6e} Pa)"
            )
        pressure = implied_pressure
    else:
        if isinstance(eos, TaitEOS):
            raise ReactiveConfigurationError(
                "temperature_K is required for TaitEOS because the paper gives no caloric closure"
            )
        assert pressure is not None
        specific_internal = np.asarray(
            eos.specific_internal_energy_from_pressure(rho, pressure), dtype=np.float64
        )

    assert pressure is not None
    if np.any(pressure <= 0.0):
        raise ReactiveNumericalError(
            "invalid_pressure", "pressure must be strictly positive"
        )
    total_specific_energy = specific_internal + 0.5 * (
        velocity_x**2 + velocity_y**2
    )
    state = np.stack(
        (
            rho,
            rho * velocity_x,
            rho * velocity_y,
            rho * total_specific_energy,
            rho * progress,
        ),
        axis=-1,
    )
    if not np.isfinite(state).all():
        raise ReactiveNumericalError(
            "nonfinite_conservative_state",
            "primitive-to-conservative conversion overflowed",
        )
    return np.ascontiguousarray(state, dtype=np.float64)


def physical_state_mask(
    conservative_state: Any,
    eos: EquationOfState,
    *,
    density_floor_kg_per_m3: float = 0.0,
    pressure_floor_Pa: float = 0.0,
    temperature_floor_K: float = 0.0,
) -> np.ndarray:
    """Return a mask; never clip or silently repair a conservative state."""
    state = _state_fp64(conservative_state, require_finite=False)
    floors = (density_floor_kg_per_m3, pressure_floor_Pa, temperature_floor_K)
    if not all(np.isfinite(value) and value >= 0.0 for value in floors):
        raise ReactiveConfigurationError(
            "physical-state floors must be finite and non-negative"
        )

    finite_state = np.isfinite(state).all(axis=-1)
    rho = state[..., RHO]
    rho_safe = np.where(finite_state & (rho > 0.0), rho, 1.0)
    velocity_x = state[..., MOMENTUM_X] / rho_safe
    velocity_y = state[..., MOMENTUM_Y] / rho_safe
    total_specific = state[..., TOTAL_ENERGY] / rho_safe
    internal = total_specific - 0.5 * (velocity_x**2 + velocity_y**2)
    progress = state[..., REACTION_PROGRESS] / rho_safe

    pressure = np.asarray(eos.pressure_from_rho_e(rho_safe, internal, validate=False))
    temperature = np.asarray(
        eos.temperature_from_rho_e(rho_safe, internal, validate=False)
    )
    return (
        finite_state
        & np.isfinite(internal)
        & np.isfinite(pressure)
        & np.isfinite(temperature)
        & (rho > density_floor_kg_per_m3)
        & (pressure > pressure_floor_Pa)
        & (temperature > temperature_floor_K)
        & (progress >= 0.0)
        & (progress <= 1.0)
    )


def conservative_to_primitive(
    conservative_state: Any,
    eos: EquationOfState,
    *,
    density_floor_kg_per_m3: float = 0.0,
    pressure_floor_Pa: float = 0.0,
    temperature_floor_K: float = 0.0,
) -> PrimitiveState:
    state = _state_fp64(conservative_state, require_finite=True)
    valid = physical_state_mask(
        state,
        eos,
        density_floor_kg_per_m3=density_floor_kg_per_m3,
        pressure_floor_Pa=pressure_floor_Pa,
        temperature_floor_K=temperature_floor_K,
    )
    if not np.all(valid):
        rho_raw = state[..., RHO]
        raise ReactiveNumericalError(
            "nonphysical_conservative_state",
            "conservative-to-primitive conversion found a nonphysical state; no floor was applied",
            {
                "invalid_cell_count": int(np.size(valid) - np.count_nonzero(valid)),
                "total_cell_count": int(np.size(valid)),
                "minimum_density_kg_per_m3": float(np.min(rho_raw)),
            },
        )

    rho = state[..., RHO]
    velocity_x = state[..., MOMENTUM_X] / rho
    velocity_y = state[..., MOMENTUM_Y] / rho
    specific_total = state[..., TOTAL_ENERGY] / rho
    specific_internal = specific_total - 0.5 * (velocity_x**2 + velocity_y**2)
    pressure = np.asarray(
        eos.pressure_from_rho_e(rho, specific_internal), dtype=np.float64
    )
    temperature = np.asarray(
        eos.temperature_from_rho_e(rho, specific_internal), dtype=np.float64
    )
    progress = state[..., REACTION_PROGRESS] / rho
    return PrimitiveState(
        density_kg_per_m3=rho,
        velocity_x_m_per_s=velocity_x,
        velocity_y_m_per_s=velocity_y,
        pressure_Pa=pressure,
        temperature_K=temperature,
        reaction_progress=progress,
        specific_internal_energy_J_per_kg=specific_internal,
    )


def flux_x(conservative_state: Any, eos: EquationOfState) -> np.ndarray:
    """Return the x-flux in paper Eq. (1)."""
    state = _state_fp64(conservative_state, require_finite=True)
    primitive = conservative_to_primitive(state, eos)
    rho = primitive.density_kg_per_m3
    u = primitive.velocity_x_m_per_s
    v = primitive.velocity_y_m_per_s
    pressure = primitive.pressure_Pa
    result = np.stack(
        (
            rho * u,
            rho * u * u + pressure,
            rho * u * v,
            u * (state[..., TOTAL_ENERGY] + pressure),
            state[..., REACTION_PROGRESS] * u,
        ),
        axis=-1,
    )
    if not np.isfinite(result).all():
        raise ReactiveNumericalError(
            "nonfinite_euler_flux", "x-directed Euler flux is non-finite"
        )
    return result


def flux_y(conservative_state: Any, eos: EquationOfState) -> np.ndarray:
    """Return the y-flux in paper Eq. (1)."""
    state = _state_fp64(conservative_state, require_finite=True)
    primitive = conservative_to_primitive(state, eos)
    rho = primitive.density_kg_per_m3
    u = primitive.velocity_x_m_per_s
    v = primitive.velocity_y_m_per_s
    pressure = primitive.pressure_Pa
    result = np.stack(
        (
            rho * v,
            rho * u * v,
            rho * v * v + pressure,
            v * (state[..., TOTAL_ENERGY] + pressure),
            state[..., REACTION_PROGRESS] * v,
        ),
        axis=-1,
    )
    if not np.isfinite(result).all():
        raise ReactiveNumericalError(
            "nonfinite_euler_flux", "y-directed Euler flux is non-finite"
        )
    return result
