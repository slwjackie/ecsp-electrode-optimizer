"""Equation-of-state closures for the experimental reactive Euler solver.

The mechanical Tait law is transcribed from Eq. (3) of Park & Yoh (2024)::

    p = B * ((rho / rho0)**N - 1) + A
    c**2 = B*N/rho0 * (rho/rho0)**(N - 1)

The paper does not give a caloric EOS.  The Tait implementation therefore uses
an explicitly labelled constant-``cv`` thermal closure plus the cold energy
obtained by integrating ``de_cold/drho = p/rho**2``.  This is required to map
the paper's conservative energy to temperature, but it is not a paper result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .provenance import ReactiveConfigurationError, ReactiveNumericalError


PAPER_TAIT_METADATA: dict[str, str] = {
    "pressure_law": "PAPER_EQ_3",
    "sound_speed_law": "PAPER_EQ_3",
    "caloric_temperature_closure": "ASSUMED_NOT_FROM_PAPER",
    "caloric_closure_detail": (
        "constant-cv thermal energy plus thermodynamically compatible Tait cold energy"
    ),
}

IDEAL_GAS_METADATA: dict[str, str] = {
    "equation_of_state": "ASSUMED_NOT_FROM_PAPER",
    "intended_use": "verification_only_ideal_gas_shock_tube",
}


def _as_fp64(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReactiveConfigurationError(f"{name} must be convertible to FP64") from exc
    return result


def _finite_positive_parameter(
    value: float, name: str, *, strictly: bool = True
) -> float:
    result = float(value)
    invalid_sign = result <= 0.0 if strictly else result < 0.0
    if not np.isfinite(result) or invalid_sign:
        qualifier = "positive" if strictly else "non-negative"
        raise ReactiveConfigurationError(f"{name} must be finite and {qualifier}")
    return result


def _maybe_scalar(array: np.ndarray) -> np.ndarray | float:
    return float(array) if array.ndim == 0 else array


def _require_positive_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        finite = array[np.isfinite(array)]
        diagnostics = {
            "nonfinite_count": int(
                np.size(array) - np.count_nonzero(np.isfinite(array))
            ),
            "minimum_finite": float(np.min(finite)) if finite.size else float("nan"),
        }
        raise ReactiveNumericalError(
            f"invalid_{name}",
            f"{name} must be finite and strictly positive",
            diagnostics,
        )


@dataclass(frozen=True)
class TaitEOS:
    """Paper Eq. (3) with an explicit, non-paper caloric closure.

    All inputs are SI: ``rho0`` in kg/m^3, ``B``/``A`` in Pa, ``cv`` in
    J/(kg K), ``T_ref`` in K and ``e_ref`` in J/kg.  The paper publishes the
    form but not the values required for its ECSP case, so callers must supply
    every dimensional parameter explicitly.
    """

    rho0_kg_per_m3: float
    B_Pa: float
    N: float
    A_Pa: float
    cv_J_per_kg_K: float
    T_ref_K: float
    e_ref_J_per_kg: float = 0.0
    provenance: dict[str, str] = field(
        default_factory=lambda: dict(PAPER_TAIT_METADATA), init=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rho0_kg_per_m3",
            _finite_positive_parameter(self.rho0_kg_per_m3, "rho0"),
        )
        object.__setattr__(self, "B_Pa", _finite_positive_parameter(self.B_Pa, "B"))
        n_value = _finite_positive_parameter(self.N, "N")
        object.__setattr__(self, "N", n_value)
        a_value = float(self.A_Pa)
        if not np.isfinite(a_value):
            raise ReactiveConfigurationError("Tait offset A must be finite")
        object.__setattr__(self, "A_Pa", a_value)
        object.__setattr__(
            self,
            "cv_J_per_kg_K",
            _finite_positive_parameter(
                self.cv_J_per_kg_K, "constant-volume heat capacity"
            ),
        )
        object.__setattr__(
            self,
            "T_ref_K",
            _finite_positive_parameter(self.T_ref_K, "reference temperature"),
        )
        e_ref = float(self.e_ref_J_per_kg)
        if not np.isfinite(e_ref):
            raise ReactiveConfigurationError("reference specific energy must be finite")
        object.__setattr__(self, "e_ref_J_per_kg", e_ref)

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "TaitEOS",
            "mechanical_equation": "Park_and_Yoh_2024_Eq_3",
            **self.provenance,
            "parameters_are_caller_supplied": True,
        }

    def pressure_from_density(
        self, density_kg_per_m3: Any, *, validate: bool = True
    ) -> np.ndarray | float:
        rho = _as_fp64(density_kg_per_m3, "density")
        if validate:
            _require_positive_finite(rho, "density")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pressure = (
                self.B_Pa * (np.power(rho / self.rho0_kg_per_m3, self.N) - 1.0)
                + self.A_Pa
            )
        if validate and (not np.isfinite(pressure).all() or np.any(pressure <= 0.0)):
            finite = pressure[np.isfinite(pressure)]
            raise ReactiveNumericalError(
                "invalid_tait_pressure",
                "Tait Eq. (3) produced non-finite or non-positive pressure",
                {
                    "minimum_finite_pressure_Pa": float(np.min(finite))
                    if finite.size
                    else float("nan")
                },
            )
        return _maybe_scalar(pressure)

    def pressure(
        self, density_kg_per_m3: Any, *, validate: bool = True
    ) -> np.ndarray | float:
        """Public Eq. (3) alias for :meth:`pressure_from_density`."""
        return self.pressure_from_density(density_kg_per_m3, validate=validate)

    def pressure_from_rho_e(
        self,
        density_kg_per_m3: Any,
        specific_internal_energy_J_per_kg: Any,
        *,
        validate: bool = True,
    ) -> np.ndarray | float:
        # Tait Eq. (3) is barotropic.  Still broadcast/check e so shape and
        # non-finite energy errors cannot be hidden by the mechanical closure.
        rho, energy = np.broadcast_arrays(
            _as_fp64(density_kg_per_m3, "density"),
            _as_fp64(specific_internal_energy_J_per_kg, "specific internal energy"),
        )
        if validate:
            _require_positive_finite(rho, "density")
            if not np.isfinite(energy).all():
                raise ReactiveNumericalError(
                    "nonfinite_internal_energy",
                    "specific internal energy must be finite",
                )
        return self.pressure_from_density(rho, validate=validate)

    def sound_speed(
        self,
        density_kg_per_m3: Any,
        pressure_Pa: Any | None = None,
        *,
        validate: bool = True,
    ) -> np.ndarray | float:
        del pressure_Pa  # Eq. (3) gives c directly from density.
        rho = _as_fp64(density_kg_per_m3, "density")
        if validate:
            _require_positive_finite(rho, "density")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            c_squared = (
                self.B_Pa
                * self.N
                / self.rho0_kg_per_m3
                * np.power(rho / self.rho0_kg_per_m3, self.N - 1.0)
            )
            sound = np.sqrt(c_squared)
        if validate and (not np.isfinite(sound).all() or np.any(c_squared <= 0.0)):
            raise ReactiveNumericalError(
                "invalid_tait_sound_speed",
                "Tait Eq. (3) produced invalid squared sound speed",
            )
        return _maybe_scalar(sound)

    def cold_specific_energy(
        self, density_kg_per_m3: Any, *, validate: bool = True
    ) -> np.ndarray | float:
        """Return integral of ``p/rho**2`` relative to ``rho0`` in J/kg."""
        rho = _as_fp64(density_kg_per_m3, "density")
        if validate:
            _require_positive_finite(rho, "density")
        rho0 = self.rho0_kg_per_m3
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            exponent_delta = self.N - 1.0
            log_density_ratio = np.log(rho / rho0)
            if exponent_delta == 0.0:
                power_term = self.B_Pa / rho0 * log_density_ratio
            else:
                # expm1 preserves the analytic N->1 limit when N is only a few
                # ulps from one; pow(...)-1 would catastrophically cancel.
                power_term = self.B_Pa / rho0 * (
                    np.expm1(exponent_delta * log_density_ratio)
                    / exponent_delta
                )
            offset_term = (self.A_Pa - self.B_Pa) * (
                (rho - rho0) / (rho0 * rho)
            )
            energy = power_term + offset_term
        if validate and not np.isfinite(energy).all():
            raise ReactiveNumericalError(
                "nonfinite_tait_cold_energy", "Tait cold energy became non-finite"
            )
        return _maybe_scalar(energy)

    def specific_internal_energy_from_temperature(
        self, density_kg_per_m3: Any, temperature_K: Any, *, validate: bool = True
    ) -> np.ndarray | float:
        rho, temperature = np.broadcast_arrays(
            _as_fp64(density_kg_per_m3, "density"),
            _as_fp64(temperature_K, "temperature"),
        )
        if validate:
            _require_positive_finite(rho, "density")
            _require_positive_finite(temperature, "temperature")
        cold = np.asarray(
            self.cold_specific_energy(rho, validate=validate), dtype=np.float64
        )
        with np.errstate(over="ignore", invalid="ignore"):
            energy = (
                cold
                + self.e_ref_J_per_kg
                + self.cv_J_per_kg_K * (temperature - self.T_ref_K)
            )
        if validate and not np.isfinite(energy).all():
            raise ReactiveNumericalError(
                "nonfinite_caloric_energy",
                "assumed Tait caloric closure produced non-finite energy",
            )
        return _maybe_scalar(energy)

    def temperature_from_rho_e(
        self,
        density_kg_per_m3: Any,
        specific_internal_energy_J_per_kg: Any,
        *,
        validate: bool = True,
    ) -> np.ndarray | float:
        rho, energy = np.broadcast_arrays(
            _as_fp64(density_kg_per_m3, "density"),
            _as_fp64(specific_internal_energy_J_per_kg, "specific internal energy"),
        )
        if validate:
            _require_positive_finite(rho, "density")
            if not np.isfinite(energy).all():
                raise ReactiveNumericalError(
                    "nonfinite_internal_energy",
                    "specific internal energy must be finite",
                )
        cold = np.asarray(
            self.cold_specific_energy(rho, validate=validate), dtype=np.float64
        )
        with np.errstate(over="ignore", invalid="ignore"):
            temperature = (
                self.T_ref_K
                + (energy - cold - self.e_ref_J_per_kg) / self.cv_J_per_kg_K
            )
        if validate:
            _require_positive_finite(temperature, "temperature")
        return _maybe_scalar(temperature)

    def temperature(
        self,
        density_kg_per_m3: Any,
        specific_internal_energy_J_per_kg: Any,
        *,
        validate: bool = True,
    ) -> np.ndarray | float:
        return self.temperature_from_rho_e(
            density_kg_per_m3, specific_internal_energy_J_per_kg, validate=validate
        )

    def specific_internal_energy_from_pressure(
        self, density_kg_per_m3: Any, pressure_Pa: Any, *, validate: bool = True
    ) -> np.ndarray | float:
        del density_kg_per_m3, pressure_Pa, validate
        raise ReactiveConfigurationError(
            "Tait pressure is barotropic and cannot determine temperature/internal energy; "
            "temperature_K is required (ASSUMED_NOT_FROM_PAPER caloric closure)"
        )


@dataclass(frozen=True)
class IdealGasEOS:
    """Calorically perfect ideal gas used only for verification problems."""

    gamma: float
    gas_constant_J_per_kg_K: float
    provenance: dict[str, str] = field(
        default_factory=lambda: dict(IDEAL_GAS_METADATA), init=False, repr=False
    )

    def __post_init__(self) -> None:
        gamma = _finite_positive_parameter(self.gamma, "ideal-gas gamma")
        if gamma <= 1.0:
            raise ReactiveConfigurationError("ideal-gas gamma must be greater than one")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(
            self,
            "gas_constant_J_per_kg_K",
            _finite_positive_parameter(
                self.gas_constant_J_per_kg_K, "specific gas constant"
            ),
        )

    @property
    def cv_J_per_kg_K(self) -> float:
        return self.gas_constant_J_per_kg_K / (self.gamma - 1.0)

    def metadata(self) -> dict[str, Any]:
        return {"model": "IdealGasEOS", **self.provenance}

    def pressure_from_rho_e(
        self,
        density_kg_per_m3: Any,
        specific_internal_energy_J_per_kg: Any,
        *,
        validate: bool = True,
    ) -> np.ndarray | float:
        rho, energy = np.broadcast_arrays(
            _as_fp64(density_kg_per_m3, "density"),
            _as_fp64(specific_internal_energy_J_per_kg, "specific internal energy"),
        )
        if validate:
            _require_positive_finite(rho, "density")
            _require_positive_finite(energy, "specific_internal_energy")
        with np.errstate(over="ignore", invalid="ignore"):
            pressure = (self.gamma - 1.0) * rho * energy
        if validate:
            _require_positive_finite(pressure, "pressure")
        return _maybe_scalar(pressure)

    def pressure(
        self,
        density_kg_per_m3: Any,
        specific_internal_energy_J_per_kg: Any,
        *,
        validate: bool = True,
    ) -> np.ndarray | float:
        return self.pressure_from_rho_e(
            density_kg_per_m3, specific_internal_energy_J_per_kg, validate=validate
        )

    def sound_speed(
        self, density_kg_per_m3: Any, pressure_Pa: Any, *, validate: bool = True
    ) -> np.ndarray | float:
        rho, pressure = np.broadcast_arrays(
            _as_fp64(density_kg_per_m3, "density"), _as_fp64(pressure_Pa, "pressure")
        )
        if validate:
            _require_positive_finite(rho, "density")
            _require_positive_finite(pressure, "pressure")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            c_squared = self.gamma * pressure / rho
            sound = np.sqrt(c_squared)
        if validate and (not np.isfinite(sound).all() or np.any(c_squared <= 0.0)):
            raise ReactiveNumericalError(
                "invalid_ideal_gas_sound_speed",
                "ideal-gas squared sound speed is invalid",
            )
        return _maybe_scalar(sound)

    def specific_internal_energy_from_pressure(
        self, density_kg_per_m3: Any, pressure_Pa: Any, *, validate: bool = True
    ) -> np.ndarray | float:
        rho, pressure = np.broadcast_arrays(
            _as_fp64(density_kg_per_m3, "density"), _as_fp64(pressure_Pa, "pressure")
        )
        if validate:
            _require_positive_finite(rho, "density")
            _require_positive_finite(pressure, "pressure")
        with np.errstate(divide="ignore", invalid="ignore"):
            energy = pressure / ((self.gamma - 1.0) * rho)
        if validate:
            _require_positive_finite(energy, "specific_internal_energy")
        return _maybe_scalar(energy)

    def specific_internal_energy_from_temperature(
        self, density_kg_per_m3: Any, temperature_K: Any, *, validate: bool = True
    ) -> np.ndarray | float:
        rho, temperature = np.broadcast_arrays(
            _as_fp64(density_kg_per_m3, "density"),
            _as_fp64(temperature_K, "temperature"),
        )
        if validate:
            _require_positive_finite(rho, "density")
            _require_positive_finite(temperature, "temperature")
        energy = self.cv_J_per_kg_K * temperature
        return _maybe_scalar(energy)

    def temperature_from_rho_e(
        self,
        density_kg_per_m3: Any,
        specific_internal_energy_J_per_kg: Any,
        *,
        validate: bool = True,
    ) -> np.ndarray | float:
        rho, energy = np.broadcast_arrays(
            _as_fp64(density_kg_per_m3, "density"),
            _as_fp64(specific_internal_energy_J_per_kg, "specific internal energy"),
        )
        if validate:
            _require_positive_finite(rho, "density")
            _require_positive_finite(energy, "specific_internal_energy")
        temperature = energy / self.cv_J_per_kg_K
        if validate:
            _require_positive_finite(temperature, "temperature")
        return _maybe_scalar(temperature)

    def temperature(
        self,
        density_kg_per_m3: Any,
        specific_internal_energy_J_per_kg: Any,
        *,
        validate: bool = True,
    ) -> np.ndarray | float:
        return self.temperature_from_rho_e(
            density_kg_per_m3, specific_internal_energy_J_per_kg, validate=validate
        )


EquationOfState = TaitEOS | IdealGasEOS
