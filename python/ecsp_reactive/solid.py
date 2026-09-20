"""Condensed thermal decomposition equation printed as paper Eq. (2).

The article's printed units do not close: it defines ``r_alpha=dalpha/dt`` but
labels it as a surface molar rate, then multiplies it by a mass-specific heat.
This executable closure takes alpha and r_alpha as dimensionless and 1/s and
therefore uses ``-rho*Q_r*r_alpha`` in W/m^3.  That dimensional repair is not a
paper value and strict-paper configuration must reject it as an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .provenance import ReactiveConfigurationError, ReactiveNumericalError


@dataclass(frozen=True)
class SolidThermalModel:
    density_kg_per_m3: float
    heat_capacity_J_per_kgK: float
    conductivity_W_per_mK: float
    preexponential_per_s: float
    activation_energy_J_per_mol: float
    gas_constant_J_per_molK: float
    heat_of_decomposition_J_per_kg: float
    reaction_order: float = 1.0

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not all(math.isfinite(value) for value in values):
            raise ReactiveConfigurationError("Solid thermal parameters must be finite")
        if self.density_kg_per_m3 <= 0 or self.heat_capacity_J_per_kgK <= 0:
            raise ReactiveConfigurationError("Solid density and heat capacity must be positive")
        if self.conductivity_W_per_mK < 0 or self.preexponential_per_s < 0:
            raise ReactiveConfigurationError("Conductivity and pre-exponential factor cannot be negative")
        if self.activation_energy_J_per_mol < 0 or self.gas_constant_J_per_molK <= 0:
            raise ReactiveConfigurationError("Invalid activation energy or gas constant")
        if self.reaction_order < 0:
            raise ReactiveConfigurationError("Solid reaction order cannot be negative")

    @property
    def thermal_diffusivity_m2_per_s(self) -> float:
        return self.conductivity_W_per_mK / (
            self.density_kg_per_m3 * self.heat_capacity_J_per_kgK
        )


def laplacian_2d(field: np.ndarray, dx_m: float, dy_m: float, boundary: str) -> np.ndarray:
    value = np.asarray(field, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 3:
        raise ReactiveConfigurationError("Thermal field must be 2-D with at least 3 cells per axis")
    if not np.isfinite(value).all() or dx_m <= 0 or dy_m <= 0:
        raise ReactiveConfigurationError("Invalid thermal field or cell spacing")
    if boundary == "PERIODIC":
        padded = np.pad(value, 1, mode="wrap")
    elif boundary == "OUTFLOW":
        # Constant extrapolation is a zero-normal-gradient thermal boundary.
        padded = np.pad(value, 1, mode="edge")
    else:
        raise ReactiveConfigurationError(f"Unsupported thermal boundary {boundary!r}")
    return (
        (padded[1:-1, 2:] - 2.0 * value + padded[1:-1, :-2]) / dx_m**2
        + (padded[2:, 1:-1] - 2.0 * value + padded[:-2, 1:-1]) / dy_m**2
    )


def solid_arrhenius_rate(
    temperature_K: np.ndarray,
    alpha: np.ndarray,
    model: SolidThermalModel,
    progress_tolerance: float = 0.0,
) -> np.ndarray:
    """Return ``d(alpha)/dt`` with a caller-declared terminal tolerance.

    The state is never changed or clipped.  A cell whose positive unreacted
    remainder is no larger than ``progress_tolerance`` simply has zero future
    rate, which makes zero/sublinear-order completion finite in an explicit
    integrator.  The default remains the historical exact terminal at alpha=1.
    """
    temperature = np.asarray(temperature_K, dtype=np.float64)
    progress = np.asarray(alpha, dtype=np.float64)
    if temperature.shape != progress.shape or temperature.ndim != 2:
        raise ReactiveConfigurationError("Solid temperature and alpha must have one 2-D shape")
    if not np.isfinite(temperature).all() or not np.isfinite(progress).all():
        raise ReactiveNumericalError("nonfinite_solid_state", "Solid T/alpha contains NaN or Inf")
    if np.any(temperature <= 0):
        raise ReactiveNumericalError("nonpositive_temperature", "Arrhenius temperature must be positive")
    if not math.isfinite(progress_tolerance) or progress_tolerance < 0.0:
        raise ReactiveConfigurationError("progress_tolerance must be finite and non-negative")
    if np.any(progress < 0.0) or np.any(progress > 1.0):
        raise ReactiveNumericalError("alpha_out_of_bounds", "Solid alpha left [0,1]")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rate = (
            model.preexponential_per_s
            * np.power(1.0 - progress, model.reaction_order)
            * np.exp(
                -model.activation_energy_J_per_mol
                / (model.gas_constant_J_per_molK * temperature)
            )
        )
    rate = np.where((1.0 - progress) <= progress_tolerance, 0.0, rate)
    if not np.isfinite(rate).all() or np.any(rate < 0):
        raise ReactiveNumericalError("nonfinite_solid_rate", "Solid Arrhenius rate is invalid")
    return rate


def solid_thermal_rhs(
    temperature_K: np.ndarray,
    alpha: np.ndarray,
    model: SolidThermalModel,
    dx_m: float,
    dy_m: float,
    external_heat_W_per_m3: np.ndarray | float,
    *,
    boundary: str = "OUTFLOW",
    progress_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    temperature = np.asarray(temperature_K, dtype=np.float64)
    heat = np.broadcast_to(np.asarray(external_heat_W_per_m3, dtype=np.float64), temperature.shape)
    if not np.isfinite(heat).all():
        raise ReactiveNumericalError("nonfinite_external_heat", "External solid heat contains NaN or Inf")
    rate = solid_arrhenius_rate(temperature, alpha, model, progress_tolerance)
    conduction = model.conductivity_W_per_mK * laplacian_2d(
        temperature, dx_m, dy_m, boundary
    )
    reaction_heat = -model.density_kg_per_m3 * model.heat_of_decomposition_J_per_kg * rate
    total_heat = conduction + reaction_heat + heat
    with np.errstate(over="ignore", invalid="ignore"):
        temperature_rate = total_heat / (
            model.density_kg_per_m3 * model.heat_capacity_J_per_kgK
        )
    if not np.isfinite(temperature_rate).all():
        raise ReactiveNumericalError("nonfinite_solid_rhs", "Solid thermal RHS overflowed")
    cell_area = dx_m * dy_m
    with np.errstate(over="ignore", invalid="ignore"):
        integrals = tuple(
            float(np.sum(field * cell_area))
            for field in (conduction, reaction_heat, heat)
        )
    if not all(math.isfinite(value) for value in integrals):
        raise ReactiveNumericalError(
            "nonfinite_solid_budget_integral",
            "Solid heat budget overflowed after cell-wise area scaling",
        )
    return temperature_rate, rate, {
        "conduction_integral_W_per_m": integrals[0],
        "reaction_heat_integral_W_per_m": integrals[1],
        "external_heat_integral_W_per_m": integrals[2],
        "maximum_alpha_rate_per_s": float(np.max(rate)),
    }


def solid_stable_time_step(
    model: SolidThermalModel,
    dx_m: float,
    dy_m: float,
    maximum_alpha_rate_per_s: float,
    *,
    diffusion_safety: float = 0.9,
    maximum_progress_increment: float = 0.05,
) -> float:
    inputs = (
        dx_m,
        dy_m,
        maximum_alpha_rate_per_s,
        diffusion_safety,
        maximum_progress_increment,
    )
    if not all(math.isfinite(value) for value in inputs):
        raise ReactiveConfigurationError("Solid stability inputs must be finite")
    if dx_m <= 0 or dy_m <= 0 or maximum_alpha_rate_per_s < 0:
        raise ReactiveConfigurationError("Invalid solid stability input")
    if not 0.0 < diffusion_safety <= 1.0:
        raise ReactiveConfigurationError("diffusion_safety must lie in (0,1]")
    if not 0.0 < maximum_progress_increment <= 1.0:
        raise ReactiveConfigurationError(
            "maximum_progress_increment must lie in (0,1]"
        )
    diffusivity = model.thermal_diffusivity_m2_per_s
    diffusion_dt = math.inf
    if diffusivity > 0:
        diffusion_dt = diffusion_safety / (
            2.0 * diffusivity * (1.0 / dx_m**2 + 1.0 / dy_m**2)
        )
    if maximum_alpha_rate_per_s == 0:
        reaction_dt = math.inf
    else:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            reaction_dt = float(
                np.float64(maximum_progress_increment)
                / np.float64(maximum_alpha_rate_per_s)
            )
    result = min(diffusion_dt, reaction_dt)
    if result <= 0 or math.isnan(result):
        raise ReactiveNumericalError("invalid_solid_time_step", "Solid stability limit is invalid")
    return result


def advance_solid_ssprk3(
    temperature_K: np.ndarray,
    alpha: np.ndarray,
    dt_s: float,
    rhs,
    *,
    minimum_temperature_K: float,
) -> tuple[np.ndarray, np.ndarray]:
    """SSPRK(3,3) update with stage validation and no physical clipping."""
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ReactiveConfigurationError("Solid SSPRK time step must be positive and finite")
    t0 = np.asarray(temperature_K, dtype=np.float64)
    a0 = np.asarray(alpha, dtype=np.float64)

    def valid(t: np.ndarray, a: np.ndarray, stage: int) -> None:
        if not np.isfinite(t).all() or not np.isfinite(a).all():
            raise ReactiveNumericalError("nonfinite_solid_stage", f"Solid RK stage {stage} is non-finite")
        if np.any(t < minimum_temperature_K):
            raise ReactiveNumericalError("solid_temperature_below_limit", f"Solid RK stage {stage} is too cold")
        if np.any(a < 0.0) or np.any(a > 1.0):
            raise ReactiveNumericalError("solid_alpha_stage_out_of_bounds", f"Solid RK stage {stage} alpha left [0,1]")

    k_t, k_a = rhs(t0, a0)
    t1, a1 = t0 + dt_s * k_t, a0 + dt_s * k_a
    valid(t1, a1, 1)
    k_t, k_a = rhs(t1, a1)
    t2 = 0.75 * t0 + 0.25 * (t1 + dt_s * k_t)
    a2 = 0.75 * a0 + 0.25 * (a1 + dt_s * k_a)
    valid(t2, a2, 2)
    k_t, k_a = rhs(t2, a2)
    tn = (1.0 / 3.0) * t0 + (2.0 / 3.0) * (t2 + dt_s * k_t)
    an = (1.0 / 3.0) * a0 + (2.0 / 3.0) * (a2 + dt_s * k_a)
    valid(tn, an, 3)
    return tn, an
