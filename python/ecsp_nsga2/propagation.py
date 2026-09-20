from __future__ import annotations

"""Post-onset condensed-phase reaction-progress/refinement model.

This is not a gas-phase CFD solver.  It consumes the complete B/C onset fields
and continues heat conduction, the same two-channel global chemistry and a
level-set diagnostic for the condensed reaction front.  Its purpose is to rank
pre-flame Pareto/near-Pareto electrode geometries by spatial reaction
propagation and regression without re-running or double-counting the B/C
pre-flame interval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json
import math

import numpy as np
from scipy import ndimage

from ecsp_v6.physics.bc_global import (
    BC_REQUIRED_AUTHORIZED_PROPAGATION_HANDOFF_FIELDS,
)


_FLOAT64_MAX = float(np.finfo(np.float64).max)
_FLOAT64_MIN_SUBNORMAL = float(np.nextafter(np.float64(0.0), np.float64(1.0)))


class PropagationError(Exception):
    """Base class for classified condensed-propagation failures."""


class PropagationCandidateError(PropagationError):
    """A candidate-scoped input, numerical, or model-validity failure."""


class PropagationCandidateInputError(PropagationCandidateError, ValueError):
    """A physically invalid or inconsistent per-candidate handoff."""


class PropagationCandidateNumericalError(PropagationCandidateError, RuntimeError):
    """A per-candidate numerical failure that may safely receive a penalty."""


class PropagationCandidateModelValidityError(PropagationCandidateError):
    """A configured model domain was exceeded; not a solver instability."""


class ConfiguredModelTemperatureRangeExceeded(PropagationCandidateModelValidityError):
    """Terminal local chemistry endpoint outside the configured caloric range.

    Positional exception args preserve the structured data across spawned CPU
    workers. The index is lane-local, C-order over the original (y, x) grid,
    never an index into the chemistry integrator's compacted active-cell set.
    No claim is made about budget checks that follow the endpoint guard.
    """

    code = "configured_model_temperature_range_exceeded"

    def __init__(self, configured_maximum, observed_maximum, cell_index,
                 roundoff_tolerance, diagnostic=""):
        super().__init__(float(configured_maximum), float(observed_maximum),
                         int(cell_index), float(roundoff_tolerance), diagnostic)
        self.configured_maximum = float(configured_maximum)
        self.observed_maximum = float(observed_maximum)
        self.cell_index = int(cell_index)
        self.roundoff_tolerance = float(roundoff_tolerance)
        self.diagnostic = diagnostic

    def __str__(self):
        return (f"{self.code}: configuredMaximumTemperature_K="
                f"{self.configured_maximum!r}, observedMaximumTemperature_K="
                f"{self.observed_maximum!r}, offendingCellIndex={self.cell_index}, "
                f"temperatureRoundoffTolerance_K={self.roundoff_tolerance!r}"
                + (f"; {self.diagnostic}" if self.diagnostic else ""))

    def validity_metrics(self):
        return {
            "status": "post_onset_model_invalid",
            "failureCode": self.code,
            "propagationSucceeded": False,
            "postOnsetModelValid": False,
            "postOnsetValidityConstraintViolation": 1.0,
            "postOnsetValidityReason": self.code,
            "configuredMaximumTemperature_K": self.configured_maximum,
            "observedMaximumTemperature_K": self.observed_maximum,
            "offendingCellIndex": self.cell_index,
            "offendingCellIndexOrder": "C(y,x), lane-local original grid",
            "temperatureRoundoffTolerance_K": self.roundoff_tolerance,
            "temperatureValidationStage": "local_chemistry_endpoint",
        }


def candidate_failure_payload(error):
    """Serialize typed validity data without parsing human diagnostics."""
    result = {"status": "failed", "errorType": type(error).__name__,
              "message": str(error)}
    if isinstance(error, ConfiguredModelTemperatureRangeExceeded):
        result.update(error.validity_metrics())
    return result


class PropagationConfigurationError(PropagationError, ValueError):
    """A global propagation-model/configuration error; never candidate-scoped."""


def _require_finite_json_numbers(value: Any, *, context: str) -> None:
    """Reject non-finite numbers before they cross a JSON/output boundary."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_json_numbers(
                item, context=f"{context}.{key}"
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json_numbers(
                item, context=f"{context}[{index}]"
            )
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and np.any(~np.isfinite(value)):
            raise PropagationCandidateNumericalError(
                f"{context} contains a non-finite number"
            )
        return
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise PropagationCandidateNumericalError(
            f"{context} is non-finite: {value!r}"
        )


def _validated_cell_area(dx: float) -> float:
    """Return a representable positive FP64 cell area without overflowing."""
    if not math.isfinite(dx) or dx <= 0.0:
        raise PropagationConfigurationError(
            "Propagation grid spacing dx must be finite and positive"
        )
    maximum_spacing = math.sqrt(_FLOAT64_MAX)
    minimum_spacing = math.sqrt(_FLOAT64_MIN_SUBNORMAL)
    if dx > maximum_spacing or dx < minimum_spacing:
        raise PropagationConfigurationError(
            "Propagation cell area is not representable as a positive finite "
            f"FP64 value: dx={dx!r}"
        )
    cell_area = dx * dx
    if not math.isfinite(cell_area) or cell_area <= 0.0:
        raise PropagationConfigurationError(
            "Propagation cell area is not representable as a positive finite "
            f"FP64 value: dx={dx!r}"
        )
    return cell_area


def _checked_scaled_measure(count: int, scale: float, *, context: str) -> float:
    """Multiply a discrete count by a physical scale without silent overflow."""
    if count < 0 or not math.isfinite(scale) or scale < 0.0:
        raise PropagationConfigurationError(
            f"Invalid {context} inputs: count={count}, scale={scale!r}"
        )
    if count and scale > _FLOAT64_MAX / count:
        raise PropagationConfigurationError(
            f"{context} is not representable as finite FP64"
        )
    value = float(count) * scale
    if not math.isfinite(value):
        raise PropagationConfigurationError(
            f"{context} is not representable as finite FP64"
        )
    return value


def _safe_harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Overflow-safe harmonic mean with invalid operands propagated as NaN."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    finite_nonnegative = (
        np.isfinite(left)
        & np.isfinite(right)
        & (left >= 0.0)
        & (right >= 0.0)
    )
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    positive = finite_nonnegative & (low > 0.0)
    safe_high = np.where(positive, high, 1.0)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        value = low * (2.0 / (1.0 + low / safe_high))
    return np.where(
        finite_nonnegative,
        np.where(positive, value, 0.0),
        np.nan,
    )


def _scaled_diffusion_cfl_term(
    face_conductivity: np.ndarray,
    capacity: np.ndarray,
    step_dt: float,
    dx: float,
) -> np.ndarray:
    """Evaluate ``k*dt/(capacity*dx**2)`` without intermediate overflow."""
    face = np.asarray(face_conductivity, dtype=np.float64)
    cap = np.asarray(capacity, dtype=np.float64)
    if (
        not math.isfinite(step_dt)
        or step_dt <= 0.0
        or not math.isfinite(dx)
        or dx <= 0.0
        or np.any(~np.isfinite(face))
        or np.any(face < 0.0)
        or np.any(~np.isfinite(cap))
        or np.any(cap <= 0.0)
    ):
        raise PropagationCandidateNumericalError(
            "Invalid finite-positive inputs to propagation thermal CFL"
        )

    face_mantissa, face_exponent = np.frexp(face)
    capacity_mantissa, capacity_exponent = np.frexp(cap)
    dt_mantissa, dt_exponent = math.frexp(step_dt)
    dx_mantissa, dx_exponent = math.frexp(dx)
    # Every non-zero mantissa is in [0.5, 1), so this expression cannot
    # overflow.  ldexp then overflows only when the true CFL is unrepresentable.
    scaled_mantissa = (
        face_mantissa
        * dt_mantissa
        / capacity_mantissa
        / dx_mantissa
        / dx_mantissa
    )
    normalized_mantissa, normalization_exponent = np.frexp(scaled_mantissa)
    total_exponent = (
        face_exponent.astype(np.int64)
        + int(dt_exponent)
        - capacity_exponent.astype(np.int64)
        - 2 * int(dx_exponent)
        + normalization_exponent.astype(np.int64)
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.ldexp(normalized_mantissa, total_exponent)
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class PropagationMetrics:
    final_unreacted_area_fraction: float
    mean_effective_regression_velocity_m_per_s: float
    maximum_effective_regression_velocity_m_per_s: float
    established_time_after_onset_s: float
    reaction_front_nonuniformity: float
    final_mean_global_progress: float
    final_maximum_temperature_K: float
    onset_succeeded: bool
    continued_electrical_heating: bool
    front_arrival_coverage_fraction: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "finalUnreactedAreaFraction": self.final_unreacted_area_fraction,
            "meanEffectiveRegressionVelocity_m_per_s": self.mean_effective_regression_velocity_m_per_s,
            "maximumEffectiveRegressionVelocity_m_per_s": self.maximum_effective_regression_velocity_m_per_s,
            "establishedTimeAfterOnset_s": self.established_time_after_onset_s,
            "reactionFrontNonuniformity": self.reaction_front_nonuniformity,
            "finalMeanGlobalProgress": self.final_mean_global_progress,
            "finalMaximumTemperature_K": self.final_maximum_temperature_K,
            "onsetSucceeded": self.onset_succeeded,
            "continuedElectricalHeating": self.continued_electrical_heating,
            "frontArrivalCoverageFraction": self.front_arrival_coverage_fraction,
        }


def _table_property(
    temperature: np.ndarray,
    raw: Mapping[str, Any] | float,
    gas_constant: float,
) -> np.ndarray:
    if isinstance(raw, (float, int)):
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise PropagationConfigurationError(
                "Propagation property constants must be finite and non-negative"
            )
        return np.full_like(temperature, value, dtype=np.float64)
    mode = str(raw.get("mode", "table")).lower()
    if mode == "table":
        temperature_grid = np.asarray(raw["temperature_K"], dtype=float)
        values = np.asarray(raw["values"], dtype=float)
        if (
            temperature_grid.ndim != 1
            or values.ndim != 1
            or temperature_grid.size < 2
            or temperature_grid.size != values.size
            or np.any(~np.isfinite(temperature_grid))
            or np.any(~np.isfinite(values))
            or np.any(np.diff(temperature_grid) <= 0.0)
            or np.any(values < 0.0)
        ):
            raise PropagationConfigurationError(
                "Propagation property tables require finite, non-negative values "
                "on a strictly increasing temperature grid"
            )
        return np.interp(
            temperature,
            temperature_grid,
            values,
        )
    if mode == "constant":
        value = float(raw["value"])
        if not math.isfinite(value) or value < 0.0:
            raise PropagationConfigurationError(
                "Propagation property constants must be finite and non-negative"
            )
        return np.full_like(temperature, value, dtype=np.float64)
    if mode in {"arrhenius", "reference_arrhenius"}:
        ref = float(raw["reference_value"])
        tref = float(raw["reference_temperature_K"])
        ea = float(raw.get("activation_energy_J_per_mol", 0.0))
        if (
            not all(math.isfinite(value) for value in (ref, tref, ea))
            or ref < 0.0
            or tref <= 0.0
            or ea < 0.0
        ):
            raise PropagationConfigurationError(
                "Invalid propagation Arrhenius property parameters"
            )
        # Overflow/non-finite results are handled by the caller's explicit
        # fail-closed property check; do not leak incidental NumPy warnings.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            return ref * np.exp(
                -ea
                / gas_constant
                * (1.0 / np.maximum(temperature, 1.0) - 1.0 / tref)
            )
    raise PropagationConfigurationError(
        f"Unsupported propagation property mode: {mode}"
    )


def _div_k_grad(field: np.ndarray, k: np.ndarray, mask: np.ndarray, dx: float) -> np.ndarray:
    """Conservative 2-D div(k grad(T)) with zero normal boundary flux."""
    f = np.asarray(field, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if f.shape != k.shape or f.shape != m.shape:
        raise ValueError("Propagation conduction fields must have identical shapes")
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("Propagation conduction dx must be finite and positive")
    if (
        np.any(~np.isfinite(f[m]))
        or np.any(~np.isfinite(k[m]))
        or np.any(k[m] < 0.0)
    ):
        raise PropagationCandidateNumericalError(
            "Propagation conduction inputs are non-finite or non-physical"
        )
    result = np.zeros_like(f)
    valid = m[:, :-1] & m[:, 1:]
    kf = _safe_harmonic_mean(k[:, :-1], k[:, 1:])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        flux = np.where(valid, kf * ((f[:, 1:] - f[:, :-1]) / dx), 0.0)
        result[:, :-1] += flux / dx
        result[:, 1:] -= flux / dx
    valid = m[:-1, :] & m[1:, :]
    kf = _safe_harmonic_mean(k[:-1, :], k[1:, :])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        flux = np.where(valid, kf * ((f[1:, :] - f[:-1, :]) / dx), 0.0)
        result[:-1, :] += flux / dx
        result[1:, :] -= flux / dx
    result[~m] = 0.0
    if np.any(~np.isfinite(result[m])):
        raise PropagationCandidateNumericalError(
            "Propagation conduction update became non-finite"
        )
    return result


def _thermal_diffusive_cfl(
    conductivity: np.ndarray,
    volumetric_heat_capacity: np.ndarray,
    mask: np.ndarray,
    dx: float,
    step_dt: float,
) -> np.ndarray:
    """Local explicit-Euler FV diffusion diagonal for a square grid."""
    k = np.asarray(conductivity, dtype=np.float64)
    capacity = np.asarray(volumetric_heat_capacity, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if k.shape != capacity.shape or k.shape != m.shape:
        raise ValueError("Propagation thermal-CFL fields must have identical shapes")
    if (
        not math.isfinite(dx)
        or dx <= 0.0
        or not math.isfinite(step_dt)
        or step_dt <= 0.0
        or np.any(~np.isfinite(k[m]))
        or np.any(k[m] < 0.0)
        or np.any(~np.isfinite(capacity[m]))
        or np.any(capacity[m] <= 0.0)
    ):
        raise PropagationCandidateNumericalError(
            "Invalid propagation thermal-CFL inputs"
        )
    cfl = np.zeros_like(k)
    valid_x = m[:, :-1] & m[:, 1:]
    kx = _safe_harmonic_mean(k[:, :-1], k[:, 1:])
    kx = np.where(valid_x, kx, 0.0)
    x_left = np.where(
        valid_x,
        _scaled_diffusion_cfl_term(kx, capacity[:, :-1], step_dt, dx),
        0.0,
    )
    x_right = np.where(
        valid_x,
        _scaled_diffusion_cfl_term(kx, capacity[:, 1:], step_dt, dx),
        0.0,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        cfl[:, :-1] += x_left
        cfl[:, 1:] += x_right
    valid_y = m[:-1, :] & m[1:, :]
    ky = _safe_harmonic_mean(k[:-1, :], k[1:, :])
    ky = np.where(valid_y, ky, 0.0)
    y_top = np.where(
        valid_y,
        _scaled_diffusion_cfl_term(ky, capacity[:-1, :], step_dt, dx),
        0.0,
    )
    y_bottom = np.where(
        valid_y,
        _scaled_diffusion_cfl_term(ky, capacity[1:, :], step_dt, dx),
        0.0,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        cfl[:-1, :] += y_top
        cfl[1:, :] += y_bottom
    cfl = np.where(m, cfl, 0.0)
    if np.any(~np.isfinite(cfl[m])) or np.any(cfl[m] < 0.0):
        raise PropagationCandidateNumericalError(
            "Propagation thermal CFL became non-finite or negative"
        )
    return cfl


def _interp_time_history(
    query_s: float,
    times_s: np.ndarray,
    values: np.ndarray,
    *,
    continued: bool,
) -> np.ndarray:
    if not continued or values.size == 0 or times_s.size == 0:
        return np.zeros(values.shape[-2:] if values.ndim >= 2 else (1, 1), dtype=float)
    if query_s <= times_s[0]:
        return np.asarray(values[0], dtype=float)
    if query_s >= times_s[-1]:
        return np.asarray(values[-1], dtype=float)
    upper = int(np.searchsorted(times_s, query_s, side="left"))
    lower = upper - 1
    weight = (query_s - times_s[lower]) / max(times_s[upper] - times_s[lower], 1e-30)
    return (1.0 - weight) * values[lower] + weight * values[upper]


def _kinetic_rate(
    alpha: np.ndarray,
    temperature: np.ndarray,
    channel: Mapping[str, Any],
    gas_constant: float,
) -> np.ndarray:
    grid = np.asarray(channel["alpha_grid"], dtype=float)
    ea = np.interp(alpha, grid, np.asarray(channel["activation_energy_J_per_mol"], dtype=float))
    ln_af = np.interp(alpha, grid, np.asarray(channel["ln_Af_per_s"], dtype=float))
    exponent = np.clip(
        ln_af - ea / (gas_constant * np.maximum(temperature, 1.0)),
        -100.0,
        60.0,
    )
    return np.where(alpha < 1.0, np.exp(exponent), 0.0)


def _accepted_channel_step(
    alpha: np.ndarray,
    temperature: np.ndarray,
    channel: Mapping[str, Any],
    maximum_rate: float,
    mask: np.ndarray,
    dt: float,
    gas_constant: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance one channel and return its physically accepted rate.

    The raw Arrhenius rate is first bounded by ``maximum_rate`` and the
    remaining conversion.  The returned rate is ``delta(alpha) / dt`` rather
    than the pre-clipping proposal, so integrating the associated heat source
    gives exactly the enthalpy of the accepted reaction progress.
    """
    raw_rate = _kinetic_rate(alpha, temperature, channel, gas_constant)
    bounded_rate = np.clip(raw_rate, 0.0, maximum_rate)
    proposed = np.clip(alpha + dt * bounded_rate, 0.0, 1.0)
    accepted_delta = np.where(mask, np.maximum(proposed - alpha, 0.0), 0.0)
    updated = np.where(mask, alpha + accepted_delta, 0.0)
    accepted_rate = accepted_delta / dt
    tolerance = np.finfo(np.float64).eps * np.maximum(
        1.0, np.maximum(np.abs(alpha), np.abs(proposed))
    )
    limited = mask & (
        (raw_rate > maximum_rate)
        | (dt * bounded_rate > accepted_delta + tolerance)
    )
    return updated, accepted_rate, limited


def _positive_time_step_count(
    duration: float,
    step_size: float,
    *,
    maximum_steps: int = 100_000,
) -> int:
    """Return a ceil-like count without a floating-point zero final step."""
    if (
        not math.isfinite(duration)
        or not math.isfinite(step_size)
        or duration <= 0.0
        or step_size <= 0.0
    ):
        raise PropagationConfigurationError(
            "Propagation duration_s and time_step_s must be finite and positive"
        )
    ratio = duration / step_size
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise PropagationConfigurationError(
            "Propagation duration_s/time_step_s ratio must be finite and positive"
        )
    nearest = round(ratio)
    tolerance = 64.0 * math.ulp(1.0) * max(1.0, abs(ratio))
    if abs(ratio - nearest) <= tolerance:
        count = max(1, int(nearest))
    else:
        count = max(1, int(math.floor(ratio)) + 1)
    if count > maximum_steps:
        raise PropagationConfigurationError(
            "Propagation time-step count exceeds maximum_time_steps: "
            f"count={count}, maximum={maximum_steps}"
        )
    return count


def _configuration_integer(
    raw: Any, *, name: str, minimum: int, maximum: int
) -> int:
    if isinstance(raw, bool):
        raise PropagationConfigurationError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        )
    try:
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PropagationConfigurationError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        ) from exc
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or numeric < minimum
        or numeric > maximum
    ):
        raise PropagationConfigurationError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        )
    return int(numeric)


def _arrival_time_statistics(
    arrival: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Return arrival-time CV and coverage as independent quantities."""
    domain = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(domain))
    if count == 0:
        return 0.0, 0.0
    arrived = np.asarray(arrival, dtype=float)[np.isfinite(arrival) & domain]
    coverage = float(arrived.size / count)
    if arrived.size < 2:
        return 0.0, coverage
    scale = float(np.max(arrived))
    if scale <= 0.0:
        return 0.0, coverage
    scaled = arrived / scale
    mean_arrival = float(np.mean(scaled))
    if mean_arrival <= 0.0:
        return 0.0, coverage
    return float(np.std(scaled) / mean_arrival), coverage


def _front_edge_count(mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool)
    horizontal = np.count_nonzero(m[:, 1:] != m[:, :-1])
    vertical = np.count_nonzero(m[1:, :] != m[:-1, :])
    return int(horizontal + vertical)


def _front_perimeter(mask: np.ndarray, dx: float) -> float:
    return _checked_scaled_measure(
        _front_edge_count(mask), dx, context="propagation front perimeter"
    )


def _effective_regression_velocity(
    previous_unreacted_cells: int,
    unreacted_cells: int,
    previous_front_edges: int,
    front_edges: int,
    dx: float,
    step_dt: float,
) -> float:
    """Compute swept-area/front-length speed after analytically cancelling dx.

    ``delta_cells*dx**2 / (dt*mean_edges*dx)`` is mathematically equal to the
    expression below, but the latter never creates the potentially overflowing
    intermediate ``dx**2``.
    """
    if (
        min(
            previous_unreacted_cells,
            unreacted_cells,
            previous_front_edges,
            front_edges,
        )
        < 0
        or unreacted_cells > previous_unreacted_cells
        or not math.isfinite(dx)
        or dx <= 0.0
        or not math.isfinite(step_dt)
        or step_dt <= 0.0
    ):
        raise PropagationCandidateNumericalError(
            "Invalid propagation regression-speed inputs"
        )
    representative_front_edges = 0.5 * (
        previous_front_edges + front_edges
    )
    if representative_front_edges == 0.0:
        return 0.0
    swept_cells = previous_unreacted_cells - unreacted_cells
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        velocity = (
            (float(swept_cells) / representative_front_edges) * dx / step_dt
        )
    if not math.isfinite(velocity) or velocity < 0.0:
        raise PropagationCandidateNumericalError(
            "Propagation effective regression velocity became non-finite or negative"
        )
    return velocity


def _reaction_level_set(
    progress: np.ndarray,
    propellant_mask: np.ndarray,
    threshold: float,
    dx: float,
) -> np.ndarray:
    """Return a signed-distance level set driven by reaction progress.

    Negative values denote reacted material and positive values denote
    unreacted material.  The progress equation advances the physics; this
    function reinitialises its interface as a signed-distance field for front
    tracking, regression metrics and handoff/output.  No gas-phase velocity or
    uncalibrated front-speed law is introduced.
    """
    mask = np.asarray(propellant_mask, dtype=bool)
    progress_array = np.asarray(progress, dtype=float)
    if progress_array.shape != mask.shape:
        raise ValueError("Propagation level-set fields must have identical shapes")
    if (
        not math.isfinite(dx)
        or dx <= 0.0
        or not math.isfinite(threshold)
        or np.any(~np.isfinite(progress_array[mask]))
    ):
        raise PropagationCandidateNumericalError(
            "Propagation level-set inputs must be finite and physical"
        )
    reacted = (progress_array >= threshold) & mask
    unreacted = (~reacted) & mask
    with np.errstate(over="ignore", invalid="ignore"):
        if np.any(reacted) and np.any(unreacted):
            phi = (
                ndimage.distance_transform_edt(unreacted, sampling=dx)
                - ndimage.distance_transform_edt(reacted, sampling=dx)
            )
        elif np.any(reacted):
            phi = -ndimage.distance_transform_edt(reacted, sampling=dx)
        else:
            phi = ndimage.distance_transform_edt(mask, sampling=dx)
    if np.any(~np.isfinite(phi[mask])):
        raise PropagationCandidateNumericalError(
            "Propagation level-set distance became non-finite"
        )
    if np.all(mask):
        return np.asarray(phi, dtype=np.float64)
    maximum_distance = max(float(np.max(np.abs(phi[mask]))), dx)
    if maximum_distance > _FLOAT64_MAX - dx:
        raise PropagationCandidateNumericalError(
            "Propagation outside-domain level-set value overflowed"
        )
    result = np.where(mask, phi, maximum_distance + dx)
    if np.any(~np.isfinite(result)):
        raise PropagationCandidateNumericalError(
            "Propagation level-set output became non-finite"
        )
    return np.asarray(result, dtype=np.float64)


def run_condensed_propagation(
    handoff: Mapping[str, Any],
    config: Mapping[str, Any],
    bc_global_config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Continue one candidate from its B/C onset state and write diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prop_cfg = dict(config)
    temperature = np.asarray(handoff["temperatureAtOnset_K"], dtype=float).copy()
    alpha_1 = np.asarray(handoff["alphaChannel1AtOnset"], dtype=float).copy()
    alpha_2 = np.asarray(handoff["alphaChannel2AtOnset"], dtype=float).copy()
    progress = np.asarray(handoff["globalProgressAtOnset"], dtype=float).copy()
    mask = np.asarray(handoff["propellantMask"], dtype=bool)
    if temperature.ndim != 2:
        raise PropagationCandidateInputError(
            "Propagation handoff fields must be single-candidate 2-D arrays"
        )
    if any(field.shape != temperature.shape for field in (alpha_1, alpha_2, progress, mask)):
        raise PropagationCandidateInputError(
            "All propagation handoff fields must have the same 2-D shape"
        )
    if not np.any(mask):
        raise PropagationCandidateInputError(
            "Propagation propellantMask must contain at least one active cell"
        )
    if not np.all(mask):
        raise PropagationCandidateInputError(
            "Corrected B/C propagation requires a full-domain propellantMask"
        )
    if temperature.shape[0] != temperature.shape[1] or temperature.shape[0] < 2:
        raise PropagationCandidateInputError(
            "Propagation currently requires a square grid with at least 2 cells "
            "per side because one spacing is used for both axes"
        )
    if not all(
        np.all(np.isfinite(field[mask]))
        for field in (temperature, alpha_1, alpha_2, progress)
    ):
        raise PropagationCandidateInputError(
            "Propagation handoff fields must be finite inside propellantMask"
        )
    if np.any(temperature[mask] <= 0.0):
        raise PropagationCandidateInputError(
            "Propagation temperatures must be positive Kelvin values"
        )

    requested_continued_heating = bool(
        prop_cfg.get("continued_electrical_heating", False)
    )
    configured_heating_mode = str(
        prop_cfg.get("electrical_heating_mode", "off")
    ).strip().lower()
    off_modes = {"off", "none", "disabled"}
    if not requested_continued_heating and configured_heating_mode not in off_modes:
        raise PropagationConfigurationError(
            "continued_electrical_heating=false requires "
            "electrical_heating_mode=off"
        )
    if not requested_continued_heating:
        heating_mode = "off"
    elif configured_heating_mode in off_modes:
        raise PropagationConfigurationError(
            "continued_electrical_heating=true contradicts "
            "electrical_heating_mode=off"
        )
    elif configured_heating_mode in {
        "provided_post_onset_history",
        "post_onset_history",
    }:
        raise PropagationConfigurationError(
            "provided_post_onset_history is not a closed electrochemical "
            "energy/species model and is disabled in the corrected solver; "
            "supply a coupled post-onset electrical solver before enabling it"
        )
    elif configured_heating_mode in {
        "legacy_preflame_history_replay",
        "legacy_replay",
    }:
        raise PropagationConfigurationError(
            "Stale pre-flame electrical-heat replay is forbidden in the "
            "corrected production solver"
        )
    elif configured_heating_mode in {"recomputed", "recompute"}:
        raise PropagationConfigurationError(
            "electrical_heating_mode=recomputed requires a coupled post-onset "
            "electrical solver; no such solver was supplied to this API"
        )
    else:
        raise PropagationConfigurationError(
            f"Unsupported propagation electrical_heating_mode: {configured_heating_mode}"
        )

    duration = float(prop_cfg["duration_s"])
    dt = float(prop_cfg["time_step_s"])
    if not math.isfinite(duration) or duration <= 0.0:
        raise PropagationConfigurationError(
            "Propagation duration_s must be finite and positive"
        )
    if not math.isfinite(dt) or dt <= 0.0:
        raise PropagationConfigurationError(
            "Propagation time_step_s must be finite and positive"
        )
    maximum_time_steps = _configuration_integer(
        prop_cfg.get("maximum_time_steps", 100_000),
        name="Propagation maximum_time_steps",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    maximum_history_bytes = _configuration_integer(
        prop_cfg.get("maximum_history_allocation_bytes", 16 * 1024**3),
        name="Propagation maximum_history_allocation_bytes",
        minimum=8,
        maximum=(1 << 63) - 1,
    )
    snapshot_interval = float(
        prop_cfg.get("snapshot_interval_s", max(duration / 10.0, dt))
    )
    if not math.isfinite(snapshot_interval) or snapshot_interval <= 0.0:
        raise PropagationConfigurationError(
            "Propagation snapshot_interval_s must be finite and positive"
        )
    thermal_cfl_limit = float(
        prop_cfg.get(
            "maximum_thermal_stability_cfl",
            prop_cfg.get("maximum_thermal_diffusive_cfl", 1.0),
        )
    )
    if (
        not math.isfinite(thermal_cfl_limit)
        or not 0.0 < thermal_cfl_limit <= 1.0
    ):
        raise PropagationConfigurationError(
            "maximum_thermal_stability_cfl must lie in (0, 1]"
        )
    temperature_cap_limit = float(
        prop_cfg.get("maximum_temperature_cap_fraction", 0.0)
    )
    chemical_rate_cap_limit = float(
        prop_cfg.get("maximum_chemical_rate_cap_fraction", 0.0)
    )
    if (
        not math.isfinite(temperature_cap_limit)
        or not 0.0 <= temperature_cap_limit <= 1.0
    ):
        raise PropagationConfigurationError(
            "maximum_temperature_cap_fraction must lie in [0, 1]"
        )
    if (
        not math.isfinite(chemical_rate_cap_limit)
        or not 0.0 <= chemical_rate_cap_limit <= 1.0
    ):
        raise PropagationConfigurationError(
            "maximum_chemical_rate_cap_fraction must lie in [0, 1]"
        )
    try:
        gas_constant = float(prop_cfg["gas_constant_J_per_molK"])
    except ValueError as exc:
        raise PropagationConfigurationError(
            "Propagation requires the resolved pre-flame "
            "gas_constant_J_per_molK"
        ) from exc
    if not math.isfinite(gas_constant) or gas_constant <= 0.0:
        raise PropagationConfigurationError(
            "Propagation gas_constant_J_per_molK must be finite and positive"
        )

    schema = handoff.get("handoffSchemaVersion")
    if schema != "ecsp_bc_surface_onset_v8.2.0":
        raise PropagationCandidateInputError(
            "Propagation requires handoffSchemaVersion="
            "ecsp_bc_surface_onset_v8.2.0"
        )
    onset_authorized = bool(handoff.get("onsetSucceeded", False))
    onset_reported = bool(handoff.get("onsetReportedByPhysics", False))
    numerically_valid = bool(
        handoff.get("numericallyValidForPropagationHandoff", False)
    )
    authorization_reason = str(
        handoff.get("propagationHandoffAuthorizationReason", "")
    )
    expected_authorized = bool(
        onset_reported
        and numerically_valid
        and authorization_reason == "ignition_and_numerics_valid"
    )
    if onset_authorized != expected_authorized:
        raise PropagationCandidateInputError(
            "Propagation handoff onset authorization metadata is inconsistent"
        )
    if not onset_authorized:
        metrics = PropagationMetrics(
            final_unreacted_area_fraction=1.0,
            mean_effective_regression_velocity_m_per_s=0.0,
            maximum_effective_regression_velocity_m_per_s=0.0,
            established_time_after_onset_s=float(prop_cfg["duration_s"]),
            reaction_front_nonuniformity=0.0,
            final_mean_global_progress=float(np.mean(progress[mask])) if np.any(mask) else 0.0,
            final_maximum_temperature_K=float(np.max(temperature[mask])) if np.any(mask) else float(np.max(temperature)),
            onset_succeeded=False,
            continued_electrical_heating=False,
            front_arrival_coverage_fraction=0.0,
        )
        payload = {
            **metrics.as_dict(),
            "status": (
                "skipped_no_bc_onset"
                if not onset_reported
                else "skipped_unauthorized_bc_onset"
            ),
            "electricalHeatingPolicy": "disabled_without_authorized_bc_onset",
            "arrivalTimeCoefficientOfVariation": 0.0,
        }
        _require_finite_json_numbers(
            payload, context="Propagation no-onset metrics"
        )
        (output_dir / "propagation_metrics.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        return payload

    try:
        ignition_delay = float(handoff["ignitionDelay_s"])
    except ValueError as exc:
        raise PropagationCandidateInputError(
            "A successful propagation handoff requires a finite ignitionDelay_s"
        ) from exc
    preflame_end_time = float(bc_global_config.get("endTime_s", math.nan))
    if not math.isfinite(preflame_end_time) or preflame_end_time <= 0.0:
        raise PropagationConfigurationError(
            "Propagation requires a finite positive pre-flame endTime_s"
        )
    delay_tolerance = 64.0 * math.ulp(abs(preflame_end_time))
    if (
        not math.isfinite(ignition_delay)
        or ignition_delay < 0.0
        or ignition_delay > preflame_end_time + delay_tolerance
    ):
        raise PropagationCandidateInputError(
            "A successful propagation handoff requires ignitionDelay_s within "
            "the configured pre-flame horizon"
        )
    onset = bc_global_config.get("onsetCriterion")
    if not isinstance(onset, Mapping):
        raise PropagationConfigurationError(
            "Propagation requires the exact pre-flame onsetCriterion"
        )
    onset_temperature = float(onset["temperature_K"])
    onset_progress = float(onset["minimum_progress"])
    onset_area = float(onset["minimum_area_fraction"])
    if (
        not math.isfinite(onset_temperature)
        or onset_temperature <= 0.0
        or not math.isfinite(onset_progress)
        or not 0.0 < onset_progress <= 1.0
        or not math.isfinite(onset_area)
        or not 0.0 < onset_area <= 1.0
    ):
        raise PropagationConfigurationError("Invalid propagation onsetCriterion")
    qualified_onset = (
        (temperature >= onset_temperature)
        & (progress >= onset_progress)
        & mask
    )
    observed_onset_area = float(
        np.count_nonzero(qualified_onset) / np.count_nonzero(mask)
    )
    if observed_onset_area + 64.0 * np.finfo(np.float64).eps < onset_area:
        raise PropagationCandidateInputError(
            "Handoff claims onset but its temperature/progress fields do not "
            "satisfy the configured onset area criterion"
        )

    steps = _positive_time_step_count(
        duration, dt, maximum_steps=maximum_time_steps
    )
    snapshot_count_upper_bound = min(
        steps,
        1
        + _positive_time_step_count(
            duration,
            max(snapshot_interval, dt),
            maximum_steps=maximum_time_steps,
        ),
    )
    retained_history_bytes = 11 * steps * 8
    # Three full-grid snapshots are retained and then stacked for NPZ output.
    retained_history_bytes += (
        2
        * 3
        * snapshot_count_upper_bound
        * temperature.shape[-2]
        * temperature.shape[-1]
        * 8
    )
    if retained_history_bytes > maximum_history_bytes:
        raise PropagationConfigurationError(
            "Propagation retained-history allocation exceeds "
            "maximum_history_allocation_bytes: "
            f"required={retained_history_bytes}, maximum={maximum_history_bytes}"
        )
    domain_size = float(prop_cfg["domain_size_m"])
    if not math.isfinite(domain_size) or domain_size <= 0.0:
        raise PropagationConfigurationError(
            "Propagation domain_size_m must be finite and positive"
        )
    # The handoff fields are cell-centred finite-volume pixels.  N cells span
    # the declared domain, so equal-area integration uses dx=L/N.
    dx = domain_size / temperature.shape[-1]
    cell_area = _validated_cell_area(dx)
    active_cell_count = int(np.count_nonzero(mask))
    _checked_scaled_measure(
        active_cell_count,
        cell_area,
        context="propagation active-domain area",
    )
    density = float(prop_cfg["density_kg_per_m3"])
    weights = np.asarray(bc_global_config["kinetics"]["mass_conversion_weights"], dtype=float)
    if (
        weights.shape != (2,)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or not math.isclose(
            float(np.sum(weights)), 1.0, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise PropagationConfigurationError(
            "Propagation mass_conversion_weights must be two finite, non-negative values summing to one"
        )
    channels = list(bc_global_config["kinetics"]["channels"])
    maximum_rate = float(bc_global_config["kinetics"].get("maximum_rate_per_s", 1e6))
    if len(channels) != 2:
        raise PropagationConfigurationError(
            "Propagation requires exactly two B/C kinetic channels"
        )
    if not math.isfinite(maximum_rate) or maximum_rate <= 0.0:
        raise PropagationConfigurationError(
            "Propagation maximum_rate_per_s must be finite and positive"
        )
    for index, channel in enumerate(channels, start=1):
        alpha_grid = np.asarray(channel["alpha_grid"], dtype=float)
        activation_energy = np.asarray(
            channel["activation_energy_J_per_mol"], dtype=float
        )
        ln_af = np.asarray(channel["ln_Af_per_s"], dtype=float)
        heat_release = float(channel["heat_release_J_per_kg"])
        if (
            alpha_grid.ndim != 1
            or alpha_grid.size < 2
            or activation_energy.shape != alpha_grid.shape
            or ln_af.shape != alpha_grid.shape
            or np.any(~np.isfinite(alpha_grid))
            or np.any(~np.isfinite(activation_energy))
            or np.any(~np.isfinite(ln_af))
            or np.any(activation_energy < 0.0)
            or np.any(np.diff(alpha_grid) <= 0.0)
            or not math.isclose(float(alpha_grid[0]), 0.0, abs_tol=1e-12)
            or not math.isclose(float(alpha_grid[-1]), 1.0, abs_tol=1e-12)
            or not math.isfinite(heat_release)
            or heat_release < 0.0
        ):
            raise PropagationConfigurationError(
                f"Invalid propagation kinetic channel {index}"
            )
    thermal = bc_global_config["thermal"]
    ambient = float(thermal.get("ambientTemperature_K", 298.15))
    t_min = float(thermal.get("minimumTemperature_K", 200.0))
    t_max = float(thermal.get("maximumTemperature_K", 3000.0))
    h = float(thermal.get("convectionCoefficient_W_per_m2K", 0.0))
    emissivity = float(thermal.get("emissivity", 0.0))
    sigma_sb = float(thermal.get("stefanBoltzmann_W_per_m2K4", 5.670374419e-8))
    thickness = float(prop_cfg["surface_layer_thickness_m"])
    front_threshold = float(prop_cfg.get("front_progress_threshold", 0.5))
    established_fraction = float(prop_cfg.get("established_reacted_area_fraction", 0.5))
    if not math.isfinite(density) or density <= 0.0:
        raise PropagationConfigurationError(
            "Propagation density_kg_per_m3 must be finite and positive"
        )
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise PropagationConfigurationError(
            "Propagation surface_layer_thickness_m must be finite and positive"
        )
    if not 0.0 < front_threshold <= 1.0:
        raise PropagationConfigurationError(
            "front_progress_threshold must lie in (0, 1]"
        )
    if not 0.0 < established_fraction <= 1.0:
        raise PropagationConfigurationError(
            "established_reacted_area_fraction must lie in (0, 1]"
        )
    if (
        not all(
            math.isfinite(value)
            for value in (ambient, t_min, t_max, h, emissivity, sigma_sb)
        )
        or t_min <= 0.0
        or t_max <= t_min
        or not t_min <= ambient <= t_max
        or h < 0.0
        or not 0.0 <= emissivity <= 1.0
        or sigma_sb < 0.0
    ):
        raise PropagationConfigurationError(
            "Invalid propagation thermal bounds or loss parameters"
        )
    alpha_tolerance = 1e-10
    if (
        np.any(alpha_1[mask] < -alpha_tolerance)
        or np.any(alpha_1[mask] > 1.0 + alpha_tolerance)
        or np.any(alpha_2[mask] < -alpha_tolerance)
        or np.any(alpha_2[mask] > 1.0 + alpha_tolerance)
        or np.any(progress[mask] < -alpha_tolerance)
        or np.any(progress[mask] > 1.0 + alpha_tolerance)
    ):
        raise PropagationCandidateInputError(
            "Propagation alpha/progress fields must lie in [0, 1]"
        )
    expected_progress = np.clip(weights[0] * alpha_1 + weights[1] * alpha_2, 0.0, 1.0)
    if not np.allclose(
        progress[mask], expected_progress[mask], rtol=1e-9, atol=1e-10
    ):
        raise PropagationCandidateInputError(
            "globalProgressAtOnset is inconsistent with the two channel alpha fields"
        )
    alpha_1 = np.where(mask, np.clip(alpha_1, 0.0, 1.0), 0.0)
    alpha_2 = np.where(mask, np.clip(alpha_2, 0.0, 1.0), 0.0)
    progress = np.where(mask, expected_progress, 0.0)
    continued = heating_mode != "off"
    absolute_onset = ignition_delay
    if heating_mode == "provided_post_onset_history":
        q_times = np.asarray(handoff.get("postOnsetTimes_s", []), dtype=float)
        qj_history = np.asarray(
            handoff.get("postOnsetQJ_W_per_m3", []), dtype=float
        )
        qe_history = np.asarray(
            handoff.get("postOnsetQEchem_W_per_m3", []), dtype=float
        )
        salt_sink_history = np.asarray(
            handoff.get("postOnsetSaltSink_mol_per_m3_s", []), dtype=float
        )
        water_sink_history = np.asarray(
            handoff.get("postOnsetWaterSink_mol_per_m3_s", []), dtype=float
        )
        heating_time_origin = "relative_to_onset"
    elif heating_mode == "legacy_preflame_history_replay":
        q_times = np.asarray(handoff.get("times_s", []), dtype=float)
        qj_history = np.asarray(handoff.get("qJ_W_per_m3", []), dtype=float)
        qe_history = np.asarray(handoff.get("qEchem_W_per_m3", []), dtype=float)
        salt_sink_history = np.empty((0, *temperature.shape), dtype=float)
        water_sink_history = np.empty((0, *temperature.shape), dtype=float)
        heating_time_origin = "absolute_preflame_time_legacy"
    else:
        q_times = np.empty((0,), dtype=float)
        qj_history = np.empty((0, *temperature.shape), dtype=float)
        qe_history = np.empty((0, *temperature.shape), dtype=float)
        salt_sink_history = np.empty((0, *temperature.shape), dtype=float)
        water_sink_history = np.empty((0, *temperature.shape), dtype=float)
        heating_time_origin = "disabled"

    if continued:
        if q_times.ndim != 1 or q_times.size < 2:
            raise PropagationCandidateInputError(
                f"{heating_mode} requires at least two electrical-heat time samples"
            )
        expected_history_shape = (q_times.size, *temperature.shape)
        if qj_history.shape != expected_history_shape or qe_history.shape != expected_history_shape:
            raise PropagationCandidateInputError(
                "Electrical-heat histories must have shape "
                f"{expected_history_shape}; received {qj_history.shape} and {qe_history.shape}"
            )
        if heating_mode == "provided_post_onset_history" and (
            salt_sink_history.shape != expected_history_shape
            or water_sink_history.shape != expected_history_shape
        ):
            raise PropagationCandidateInputError(
                "Provided post-onset electrical history also requires salt/water "
                f"sink histories with shape {expected_history_shape}"
            )
        if (
            np.any(~np.isfinite(q_times))
            or np.any(np.diff(q_times) <= 0.0)
            or np.any(~np.isfinite(qj_history))
            or np.any(~np.isfinite(qe_history))
            or (
                heating_mode == "provided_post_onset_history"
                and (
                    np.any(~np.isfinite(salt_sink_history))
                    or np.any(~np.isfinite(water_sink_history))
                    or np.any(salt_sink_history < 0.0)
                    or np.any(water_sink_history < 0.0)
                )
            )
        ):
            raise PropagationCandidateInputError(
                "Electrical-heat histories must be finite with strictly increasing times"
            )
        history_time_tolerance = 64.0 * max(
            math.ulp(abs(float(q_times[0]))),
            math.ulp(abs(float(q_times[-1]))),
            math.ulp(abs(duration)),
        )
        if heating_mode == "provided_post_onset_history" and (
            q_times[0] > history_time_tolerance
            or q_times[-1] < duration - history_time_tolerance
        ):
            raise PropagationCandidateInputError(
                "Provided post-onset electrical-heat history must cover [0, duration_s]"
            )

    inventory_keys = (
        "mobileLPAtOnset_mol_per_m3",
        "pvaReactiveRepeatAtOnset_mol_per_m3",
        "generatedWaterProductAtOnset_mol_per_m3",
        "mobileWaterAtOnset_mol_per_m3",
        "electrochemicalLPConsumedAtOnset_mol_per_m3",
        "xiMax_mol_per_m3",
        "molarMassLP_kg_per_mol",
        "molarMassPVARepeat_kg_per_mol",
        "initialReactiveMass_kg_per_m3",
        "initialMobileLP_mol_per_m3",
        "initialPVARepeat_mol_per_m3",
        "initialMobileWater_mol_per_m3",
    )
    inventory_present = [handoff.get(key) is not None for key in inventory_keys]
    if any(inventory_present) and not all(inventory_present):
        missing = [
            key for key, present in zip(inventory_keys, inventory_present) if not present
        ]
        raise PropagationCandidateInputError(
            "Partial reaction inventory in propagation handoff; missing "
            + ", ".join(missing)
        )
    reaction_inventory_tracked = all(inventory_present)
    if not any(inventory_present):
        raise PropagationCandidateInputError(
            "Corrected propagation requires the complete explicit LP/PVA, "
            "mobile-water, product-water and electrochemical-consumption "
            "inventory handoff."
        )
    if reaction_inventory_tracked:
        xi_max = float(handoff["xiMax_mol_per_m3"])
        mobile_lp = np.asarray(
            handoff["mobileLPAtOnset_mol_per_m3"], dtype=float
        ).copy()
        reactive_pva = np.asarray(
            handoff["pvaReactiveRepeatAtOnset_mol_per_m3"], dtype=float
        ).copy()
        generated_water_product = np.asarray(
            handoff["generatedWaterProductAtOnset_mol_per_m3"], dtype=float
        ).copy()
        if (
            not math.isfinite(xi_max)
            or xi_max <= 0.0
            or any(
                field.shape != temperature.shape
                for field in (mobile_lp, reactive_pva, generated_water_product)
            )
            or any(
                np.any(~np.isfinite(field[mask])) or np.any(field[mask] < 0.0)
                for field in (mobile_lp, reactive_pva, generated_water_product)
            )
        ):
            raise PropagationCandidateInputError(
                "Invalid explicit reaction inventory in propagation handoff"
            )
        expected_generated_water = 2.0 * xi_max * progress
        if not np.allclose(
            generated_water_product[mask],
            expected_generated_water[mask],
            rtol=1e-9,
            atol=1e-10,
        ):
            raise PropagationCandidateInputError(
                "generatedWaterProductAtOnset_mol_per_m3 is inconsistent with "
                "2*xiMax*globalProgressAtOnset"
            )
    else:  # pragma: no cover - all-or-none validation above makes this unreachable.
        raise AssertionError("Complete propagation inventory contract was bypassed")

    if heating_mode == "provided_post_onset_history" and not reaction_inventory_tracked:
        raise PropagationCandidateInputError(
            "Provided post-onset electrochemistry requires the complete explicit "
            "LP/PVA/mobile-water inventory handoff"
        )

    if handoff.get("mobileWaterAtOnset_mol_per_m3") is not None:
        mobile_water = np.asarray(
            handoff["mobileWaterAtOnset_mol_per_m3"], dtype=float
        ).copy()
        if (
            mobile_water.shape != temperature.shape
            or np.any(~np.isfinite(mobile_water[mask]))
            or np.any(mobile_water[mask] < 0.0)
        ):
            raise PropagationCandidateInputError(
                "Invalid mobileWaterAtOnset_mol_per_m3 handoff field"
            )
    else:
        mobile_water = np.full_like(temperature, np.nan)

    if handoff.get("electrochemicalLPConsumedAtOnset_mol_per_m3") is not None:
        electrochemical_lp_consumed = np.asarray(
            handoff["electrochemicalLPConsumedAtOnset_mol_per_m3"], dtype=float
        ).copy()
        if (
            electrochemical_lp_consumed.shape != temperature.shape
            or np.any(~np.isfinite(electrochemical_lp_consumed[mask]))
            or np.any(electrochemical_lp_consumed[mask] < 0.0)
        ):
            raise PropagationCandidateInputError(
                "Invalid electrochemicalLPConsumedAtOnset_mol_per_m3 handoff field"
            )
    else:
        electrochemical_lp_consumed = np.full_like(temperature, np.nan)
    initial_lp = float(handoff["initialMobileLP_mol_per_m3"])
    initial_pva = float(handoff["initialPVARepeat_mol_per_m3"])
    initial_mobile_water = float(handoff["initialMobileWater_mol_per_m3"])
    molar_mass_lp = float(handoff["molarMassLP_kg_per_mol"])
    molar_mass_pva = float(handoff["molarMassPVARepeat_kg_per_mol"])
    initial_reactive_mass = float(handoff["initialReactiveMass_kg_per_m3"])
    if (
        not all(
            math.isfinite(value)
            for value in (
                initial_lp,
                initial_pva,
                initial_mobile_water,
                molar_mass_lp,
                molar_mass_pva,
                initial_reactive_mass,
            )
        )
        or initial_lp <= 0.0
        or initial_pva <= 0.0
        or initial_mobile_water < 0.0
        or molar_mass_lp <= 0.0
        or molar_mass_pva <= 0.0
        or initial_reactive_mass <= 0.0
        or not math.isclose(
            initial_reactive_mass,
            molar_mass_lp * initial_lp + molar_mass_pva * initial_pva,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise PropagationCandidateInputError(
            "Invalid initial reactive-inventory contract in handoff"
        )
    expected_pva_onset = initial_pva - xi_max * progress
    # Check conserved inventories in dimensionless per-cell form.  Raw sums
    # can overflow for otherwise finite, deliberately adversarial FP64 input
    # fields and must never turn a malformed handoff into an accepted case.
    normalized_lp_balance = (
        mobile_lp / initial_lp
        + electrochemical_lp_consumed / initial_lp
        + (1.45 * xi_max / initial_lp) * progress
    )
    normalized_mobile_water = (
        mobile_water / initial_mobile_water
        if initial_mobile_water > 0.0
        else np.zeros_like(mobile_water)
    )
    invalid_normalized_inventory = (
        np.any(~np.isfinite(normalized_lp_balance[mask]))
        or (
            initial_mobile_water > 0.0
            and np.any(~np.isfinite(normalized_mobile_water[mask]))
        )
    )
    mean_lp_balance = (
        math.nan
        if invalid_normalized_inventory
        else float(np.mean(normalized_lp_balance[mask]))
    )
    mean_mobile_water_fraction = (
        0.0
        if initial_mobile_water == 0.0 and np.all(mobile_water[mask] == 0.0)
        else (
            math.inf
            if initial_mobile_water == 0.0
            else float(np.mean(normalized_mobile_water[mask]))
        )
    )
    if (
        not np.allclose(
            reactive_pva[mask],
            expected_pva_onset[mask],
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        or invalid_normalized_inventory
        or not math.isclose(mean_lp_balance, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9)
        or not math.isfinite(mean_mobile_water_fraction)
        or mean_mobile_water_fraction > 1.0 + 1.0e-9
    ):
        raise PropagationCandidateInputError(
            "Propagation onset inventory violates B/C conservation"
        )

    missing_handoff_fields = [
        key
        for key in BC_REQUIRED_AUTHORIZED_PROPAGATION_HANDOFF_FIELDS
        if key not in handoff or handoff[key] is None
    ]
    if missing_handoff_fields:
        raise PropagationCandidateInputError(
            "Corrected authorized propagation requires the complete solver "
            "snapshot, inventory and authorization handoff; missing "
            + ", ".join(missing_handoff_fields)
        )
    continued_from_handoff = handoff["continuedElectricalHeating"]
    if not isinstance(continued_from_handoff, (bool, np.bool_)) or bool(
        continued_from_handoff
    ):
        raise PropagationCandidateInputError(
            "Corrected propagation requires continuedElectricalHeating=false; "
            "pre-flame electrical sources cannot be replayed"
        )
    diagnostic_snapshots = {
        key: np.asarray(handoff[key], dtype=float)
        for key in (
            "cationAtOnset_mol_per_m3",
            "anionAtOnset_mol_per_m3",
            "potentialAtOnset_V",
            "qJAtOnset_W_per_m3",
            "qEchemAtOnset_W_per_m3",
        )
    }
    if any(
        field.shape != temperature.shape
        or np.any(~np.isfinite(field[mask]))
        for field in diagnostic_snapshots.values()
    ):
        raise PropagationCandidateInputError(
            "Propagation solver snapshot fields must match the 2-D onset "
            "state and be finite inside propellantMask"
        )
    cation_onset = diagnostic_snapshots["cationAtOnset_mol_per_m3"]
    anion_onset = diagnostic_snapshots["anionAtOnset_mol_per_m3"]
    if (
        np.any(cation_onset[mask] < 0.0)
        or np.any(anion_onset[mask] < 0.0)
        or not np.allclose(
            mobile_lp[mask],
            0.5 * (cation_onset[mask] + anion_onset[mask]),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    ):
        raise PropagationCandidateInputError(
            "Propagation mobile LP is inconsistent with the non-negative "
            "cation/anion onset snapshot"
        )
    base_progress = progress.copy()
    base_mobile_lp = mobile_lp.copy()
    base_reactive_pva = reactive_pva.copy()
    base_generated_water_product = generated_water_product.copy()
    base_electrochemical_lp_consumed = electrochemical_lp_consumed.copy()
    arrival = np.full_like(progress, np.nan, dtype=float)
    arrival[(progress >= front_threshold) & mask] = 0.0
    initial_unreacted = float(np.mean((progress < front_threshold)[mask]))
    if not math.isfinite(initial_unreacted) or not 0.0 <= initial_unreacted <= 1.0:
        raise PropagationCandidateNumericalError(
            "Propagation initial unreacted-area fraction is invalid"
        )
    initial_level_set = _reaction_level_set(progress, mask, front_threshold, dx)

    time_history = np.zeros(steps, dtype=float)
    unreacted_history = np.zeros(steps, dtype=float)
    mean_progress_history = np.zeros(steps, dtype=float)
    max_temperature_history = np.zeros(steps, dtype=float)
    regression_velocity_history = np.zeros(steps, dtype=float)
    step_duration_history = np.zeros(steps, dtype=float)
    chemical_limiter_history = np.zeros(steps, dtype=float)
    chemical_rate_cap_history = np.zeros(steps, dtype=float)
    thermal_cfl_history = np.zeros(steps, dtype=float)
    thermal_stability_cfl_history = np.zeros(steps, dtype=float)
    temperature_cap_history = np.zeros(steps, dtype=float)
    established_time = (
        0.0 if 1.0 - initial_unreacted >= established_fraction else math.nan
    )
    previous_unreacted_cells = int(
        np.count_nonzero((progress < front_threshold) & mask)
    )
    _checked_scaled_measure(
        previous_unreacted_cells,
        cell_area,
        context="propagation initial unreacted area",
    )
    previous_front_edges = _front_edge_count(
        (progress < front_threshold) & mask
    )
    _front_perimeter((progress < front_threshold) & mask, dx)

    next_snapshot = 0.0
    snapshot_times: list[float] = []
    temperature_snapshots: list[np.ndarray] = []
    progress_snapshots: list[np.ndarray] = []
    level_set_snapshots: list[np.ndarray] = []

    # v8.4 comparison-only energy ledger. The baseline still advances T with
    # the original explicit cp(T) update; its caloric truncation residual is
    # measured, not corrected or hidden.
    from ecsp_reactive.condensed.thermo import BCHeatCapacity
    try:
        comparison_caloric = BCHeatCapacity(
            thermal["heat_capacity"], thermal.get("initialTemperature_K", 298.15)
        )
    except PropagationConfigurationError:
        # Do not change the legacy solver's supported constitutive laws or
        # typed physical-property failures merely to add comparison diagnostics.
        comparison_caloric = None
    comparison_volume = cell_area * thickness
    comparison_initial_energy = (
        float(np.sum(density * comparison_caloric.sensible_energy(temperature)[mask])
              * comparison_volume) if comparison_caloric is not None else None
    )
    comparison_initial_mass = density * active_cell_count * comparison_volume
    comparison_qchem = comparison_qj = comparison_qe = comparison_loss = 0.0
    comparison_max_temperature = float(np.max(temperature[mask]))

    for step in range(steps):
        step_start = step * dt
        step_dt = duration - step_start if step == steps - 1 else dt
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise PropagationCandidateNumericalError(
                "Propagation integrator generated a non-positive time step; "
                f"step={step}, step_dt={step_dt}, duration={duration}, dt={dt}"
            )
        relative_time = duration if step == steps - 1 else step_start + step_dt
        source_sample_time = step_start + 0.5 * step_dt
        if not all(
            math.isfinite(value)
            for value in (step_start, relative_time, source_sample_time)
        ):
            raise PropagationCandidateNumericalError(
                "Propagation time history became non-finite"
            )
        if heating_mode == "provided_post_onset_history":
            heating_query_time = source_sample_time
        elif heating_mode == "legacy_preflame_history_replay":
            heating_query_time = absolute_onset + source_sample_time
        else:
            heating_query_time = relative_time
        qj = _interp_time_history(
            heating_query_time, q_times, qj_history, continued=continued
        )
        qe = _interp_time_history(
            heating_query_time, q_times, qe_history, continued=continued
        )
        if heating_mode == "provided_post_onset_history":
            salt_sink = _interp_time_history(
                heating_query_time,
                q_times,
                salt_sink_history,
                continued=True,
            )
            water_sink = _interp_time_history(
                heating_query_time,
                q_times,
                water_sink_history,
                continued=True,
            )
            requested_lp_consumption = np.where(
                mask, np.maximum(salt_sink, 0.0) * step_dt, 0.0
            )
            requested_water_consumption = np.where(
                mask, np.maximum(water_sink, 0.0) * step_dt, 0.0
            )
            lp_tolerance = 1e-12 + 1e-10 * np.maximum(
                requested_lp_consumption, mobile_lp
            )
            water_tolerance = 1e-12 + 1e-10 * np.maximum(
                requested_water_consumption, mobile_water
            )
            if np.any(
                mask
                & (requested_lp_consumption > mobile_lp + lp_tolerance)
            ) or np.any(
                mask
                & (
                    requested_water_consumption
                    > mobile_water + water_tolerance
                )
            ):
                raise PropagationCandidateNumericalError(
                    "Provided post-onset electrochemical history consumes more "
                    "LP or mobile water than the explicit handoff inventory"
                )
            accepted_lp_consumption = np.minimum(
                requested_lp_consumption, np.maximum(mobile_lp, 0.0)
            )
            accepted_water_consumption = np.minimum(
                requested_water_consumption, np.maximum(mobile_water, 0.0)
            )
            mobile_lp = np.where(
                mask, np.maximum(mobile_lp - accepted_lp_consumption, 0.0), 0.0
            )
            mobile_water = np.where(
                mask,
                np.maximum(mobile_water - accepted_water_consumption, 0.0),
                0.0,
            )
            electrochemical_lp_consumed = np.where(
                mask,
                electrochemical_lp_consumed + accepted_lp_consumption,
                0.0,
            )
        old_alpha_1 = alpha_1
        old_alpha_2 = alpha_2
        raw_rate_cap = mask & (
            (
                _kinetic_rate(
                    old_alpha_1, temperature, channels[0], gas_constant
                )
                > maximum_rate
            )
            | (
                _kinetic_rate(
                    old_alpha_2, temperature, channels[1], gas_constant
                )
                > maximum_rate
            )
        )
        raw_rate_cap_fraction = float(
            np.count_nonzero(raw_rate_cap) / max(np.count_nonzero(mask), 1)
        )
        cap_fraction_tolerance = 64.0 * np.finfo(np.float64).eps
        if raw_rate_cap_fraction > chemical_rate_cap_limit + cap_fraction_tolerance:
            raise PropagationCandidateNumericalError(
                "Propagation chemical rate cap exceeds the configured "
                f"fraction {chemical_rate_cap_limit}: "
                f"actual={raw_rate_cap_fraction}. Revise the validated "
                "kinetics or time-resolution contract."
            )
        proposed_alpha_1, accepted_rate_1, limited_1 = _accepted_channel_step(
            old_alpha_1,
            temperature,
            channels[0],
            maximum_rate,
            mask,
            step_dt,
            gas_constant,
        )
        proposed_alpha_2, accepted_rate_2, limited_2 = _accepted_channel_step(
            old_alpha_2,
            temperature,
            channels[1],
            maximum_rate,
            mask,
            step_dt,
            gas_constant,
        )
        if reaction_inventory_tracked:
            proposed_delta_alpha_1 = proposed_alpha_1 - old_alpha_1
            proposed_delta_alpha_2 = proposed_alpha_2 - old_alpha_2
            proposed_progress_increment = np.maximum(
                weights[0] * proposed_delta_alpha_1
                + weights[1] * proposed_delta_alpha_2,
                0.0,
            )
            proposed_xi_increment = xi_max * proposed_progress_increment
            maximum_xi_increment = np.minimum(
                np.maximum(mobile_lp, 0.0) / 1.45,
                np.maximum(reactive_pva, 0.0),
            )
            inventory_scale = np.ones_like(progress)
            positive_proposal = proposed_xi_increment > 0.0
            inventory_scale[positive_proposal] = np.clip(
                maximum_xi_increment[positive_proposal]
                / proposed_xi_increment[positive_proposal],
                0.0,
                1.0,
            )
            inventory_scale[~mask] = 0.0
            alpha_1 = old_alpha_1 + proposed_delta_alpha_1 * inventory_scale
            alpha_2 = old_alpha_2 + proposed_delta_alpha_2 * inventory_scale
            accepted_rate_1 = (
                proposed_delta_alpha_1 * inventory_scale / step_dt
            )
            accepted_rate_2 = (
                proposed_delta_alpha_2 * inventory_scale / step_dt
            )
            inventory_limited = mask & (inventory_scale < 1.0 - 1e-14)
        else:
            alpha_1 = proposed_alpha_1
            alpha_2 = proposed_alpha_2
            inventory_limited = np.zeros_like(mask)
        progress_new = np.clip(weights[0] * alpha_1 + weights[1] * alpha_2, 0.0, 1.0)
        if reaction_inventory_tracked:
            accepted_progress_increment = np.maximum(progress_new - progress, 0.0)
            accepted_xi_increment = xi_max * accepted_progress_increment
            mobile_lp = np.where(
                mask, np.maximum(mobile_lp - 1.45 * accepted_xi_increment, 0.0), 0.0
            )
            reactive_pva = np.where(
                mask, np.maximum(reactive_pva - accepted_xi_increment, 0.0), 0.0
            )
            generated_water_product = np.where(
                mask,
                generated_water_product + 2.0 * accepted_xi_increment,
                0.0,
            )
        crossed = np.isnan(arrival) & (progress_new >= front_threshold) & mask
        progress_delta_for_crossing = progress_new - progress
        crossing_fraction = np.zeros_like(progress)
        advancing = crossed & (progress_delta_for_crossing > 0.0)
        crossing_fraction[advancing] = np.clip(
            (front_threshold - progress[advancing])
            / progress_delta_for_crossing[advancing],
            0.0,
            1.0,
        )
        arrival[crossed] = step_start + crossing_fraction[crossed] * step_dt
        progress = progress_new

        cp = _table_property(
            temperature, thermal["heat_capacity"], gas_constant
        )
        k = _table_property(
            temperature, thermal["thermal_conductivity"], gas_constant
        )
        thermal_capacity = density * cp
        if (
            np.any(~np.isfinite(cp[mask]))
            or np.any(cp[mask] <= 0.0)
            or np.any(~np.isfinite(thermal_capacity[mask]))
            or np.any(thermal_capacity[mask] <= 0.0)
            or np.any(~np.isfinite(k[mask]))
            or np.any(k[mask] < 0.0)
        ):
            raise PropagationCandidateNumericalError(
                "Propagation thermal properties became non-finite or non-physical"
            )
        thermal_diffusive_cfl = _thermal_diffusive_cfl(
            k, thermal_capacity, mask, dx, step_dt
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            loss_jacobian = (
                h / thickness
                + 4.0
                * emissivity
                * sigma_sb
                / thickness
                * np.maximum(temperature, ambient) ** 3
            )
            thermal_loss_cfl = step_dt * loss_jacobian / thermal_capacity
            thermal_stability_cfl = np.where(
                mask, thermal_diffusive_cfl + thermal_loss_cfl, 0.0
            )
        if (
            np.any(~np.isfinite(loss_jacobian[mask]))
            or np.any(loss_jacobian[mask] < 0.0)
            or np.any(~np.isfinite(thermal_loss_cfl[mask]))
            or np.any(thermal_loss_cfl[mask] < 0.0)
            or np.any(~np.isfinite(thermal_stability_cfl[mask]))
        ):
            raise PropagationCandidateNumericalError(
                "Propagation thermal loss/stability CFL became non-finite or negative"
            )
        maximum_diffusive_cfl = float(
            np.max(thermal_diffusive_cfl[mask])
        )
        maximum_thermal_cfl = float(np.max(thermal_stability_cfl[mask]))
        cfl_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0, thermal_cfl_limit
        )
        if (
            not math.isfinite(maximum_thermal_cfl)
            or maximum_thermal_cfl > thermal_cfl_limit + cfl_tolerance
        ):
            raise PropagationCandidateNumericalError(
                "Propagation total explicit thermal stability CFL "
                "(diffusion plus convection/radiation loss) exceeds the "
                f"configured limit {thermal_cfl_limit}: "
                f"maximum={maximum_thermal_cfl}. Reduce time_step_s."
            )
        chemical_heat = density * (
            float(channels[0]["heat_release_J_per_kg"])
            * accepted_rate_1
            + float(channels[1]["heat_release_J_per_kg"])
            * accepted_rate_2
        )
        chemical_energy_density = density * (
            float(channels[0]["heat_release_J_per_kg"])
            * (alpha_1 - old_alpha_1)
            + float(channels[1]["heat_release_J_per_kg"])
            * (alpha_2 - old_alpha_2)
        )
        loss = (
            h / thickness * (temperature - ambient)
            + emissivity
            * sigma_sb
            / thickness
            * (temperature**4 - ambient**4)
        )
        conduction = _div_k_grad(temperature, k, mask, dx)
        thermal_energy_density = (
            step_dt * (conduction + qj + qe - loss)
            + chemical_energy_density
        )
        raw_temperature = temperature + thermal_energy_density / thermal_capacity
        comparison_qchem += float(np.sum(chemical_energy_density[mask]) * comparison_volume)
        comparison_qj += float(np.sum(np.broadcast_to(qj, mask.shape)[mask]) * step_dt * comparison_volume)
        comparison_qe += float(np.sum(np.broadcast_to(qe, mask.shape)[mask]) * step_dt * comparison_volume)
        comparison_loss += float(np.sum(loss[mask]) * step_dt * comparison_volume)
        comparison_max_temperature = max(comparison_max_temperature, float(np.max(raw_temperature[mask])))
        if (
            np.any(~np.isfinite(chemical_heat[mask]))
            or np.any(~np.isfinite(thermal_energy_density[mask]))
            or np.any(~np.isfinite(raw_temperature[mask]))
        ):
            raise PropagationCandidateNumericalError(
                "Propagation energy update produced a non-finite state"
            )
        temperature_cap = mask & (
            (raw_temperature < t_min) | (raw_temperature > t_max)
        )
        temperature_cap_fraction = float(
            np.count_nonzero(temperature_cap)
            / max(np.count_nonzero(mask), 1)
        )
        cap_tolerance = 64.0 * np.finfo(np.float64).eps
        if temperature_cap_fraction > temperature_cap_limit + cap_tolerance:
            raise PropagationCandidateNumericalError(
                "Propagation temperature clipping exceeds the configured "
                f"fraction {temperature_cap_limit}: "
                f"actual={temperature_cap_fraction}. Reduce time_step_s or "
                "revise the validated temperature range."
            )
        accepted_temperature = np.where(
            mask,
            np.clip(raw_temperature, t_min, t_max),
            ambient,
        )
        accepted_cp = _table_property(
            accepted_temperature, thermal["heat_capacity"], gas_constant
        )
        accepted_k = _table_property(
            accepted_temperature,
            thermal["thermal_conductivity"],
            gas_constant,
        )
        if (
            np.any(~np.isfinite(accepted_cp[mask]))
            or np.any(accepted_cp[mask] <= 0.0)
            or np.any(~np.isfinite(accepted_k[mask]))
            or np.any(accepted_k[mask] < 0.0)
        ):
            raise PropagationCandidateNumericalError(
                "Propagation thermal properties became non-finite or "
                "non-physical at the accepted updated temperature"
            )
        temperature = accepted_temperature
        expected_progress_now = np.clip(
            weights[0] * alpha_1 + weights[1] * alpha_2, 0.0, 1.0
        )
        invariant_fields = (temperature, alpha_1, alpha_2, progress)
        if (
            any(np.any(~np.isfinite(field[mask])) for field in invariant_fields)
            or np.any(alpha_1[mask] < 0.0)
            or np.any(alpha_1[mask] > 1.0)
            or np.any(alpha_2[mask] < 0.0)
            or np.any(alpha_2[mask] > 1.0)
            or np.any(progress[mask] < 0.0)
            or np.any(progress[mask] > 1.0)
            or not np.allclose(
                progress[mask], expected_progress_now[mask], rtol=1e-11, atol=1e-12
            )
        ):
            raise PropagationCandidateNumericalError(
                "Propagation reaction/temperature state invariant failed"
            )
        if reaction_inventory_tracked:
            inventory_fields = (
                mobile_lp,
                mobile_water,
                reactive_pva,
                generated_water_product,
                electrochemical_lp_consumed,
            )
            if any(
                np.any(~np.isfinite(field[mask]))
                or np.any(field[mask] < 0.0)
                for field in inventory_fields
            ):
                raise PropagationCandidateNumericalError(
                    "Propagation reaction inventory became non-finite or negative"
                )
            progress_change = progress - base_progress
            expected_mobile_lp = (
                base_mobile_lp - 1.45 * xi_max * progress_change
            )
            expected_reactive_pva = (
                base_reactive_pva - xi_max * progress_change
            )
            expected_generated_water_product = (
                base_generated_water_product + 2.0 * xi_max * progress_change
            )
            if (
                not np.allclose(
                    mobile_lp[mask],
                    expected_mobile_lp[mask],
                    rtol=1.0e-10,
                    atol=1.0e-10,
                )
                or not np.allclose(
                    reactive_pva[mask],
                    expected_reactive_pva[mask],
                    rtol=1.0e-10,
                    atol=1.0e-10,
                )
                or not np.allclose(
                    generated_water_product[mask],
                    expected_generated_water_product[mask],
                    rtol=1.0e-10,
                    atol=1.0e-10,
                )
                or not np.array_equal(
                    electrochemical_lp_consumed[mask],
                    base_electrochemical_lp_consumed[mask],
                )
            ):
                raise PropagationCandidateNumericalError(
                    "Propagation step violated LP/PVA/product-water conservation"
                )

        unreacted = (progress < front_threshold) & mask
        unreacted_cells = int(np.count_nonzero(unreacted))
        unreacted_fraction = float(unreacted_cells / active_cell_count)
        if not math.isfinite(unreacted_fraction) or not 0.0 <= unreacted_fraction <= 1.0:
            raise PropagationCandidateNumericalError(
                "Propagation unreacted-area fraction became invalid"
            )
        _checked_scaled_measure(
            unreacted_cells,
            cell_area,
            context="propagation unreacted area",
        )
        front_edges = _front_edge_count(unreacted)
        _front_perimeter(unreacted, dx)
        # A spatially uniform bulk conversion has no identifiable front and
        # therefore no finite regression speed.  Do not manufacture an
        # enormous value by dividing its swept area by an epsilon perimeter.
        regression_velocity = _effective_regression_velocity(
            previous_unreacted_cells,
            unreacted_cells,
            previous_front_edges,
            front_edges,
            dx,
            step_dt,
        )
        previous_unreacted_cells = unreacted_cells
        previous_front_edges = front_edges
        if math.isnan(established_time) and 1.0 - unreacted_fraction >= established_fraction:
            established_time = relative_time

        time_history[step] = relative_time
        unreacted_history[step] = unreacted_fraction
        mean_progress_history[step] = float(np.mean(progress[mask]))
        max_temperature_history[step] = float(np.max(temperature[mask]))
        regression_velocity_history[step] = regression_velocity
        step_duration_history[step] = step_dt
        chemical_limiter_history[step] = float(
            np.count_nonzero((limited_1 | limited_2 | inventory_limited) & mask)
            / max(np.count_nonzero(mask), 1)
        )
        chemical_rate_cap_history[step] = raw_rate_cap_fraction
        thermal_cfl_history[step] = maximum_diffusive_cfl
        thermal_stability_cfl_history[step] = maximum_thermal_cfl
        temperature_cap_history[step] = temperature_cap_fraction
        snapshot_time_tolerance = 64.0 * max(
            math.ulp(abs(relative_time)), math.ulp(abs(next_snapshot))
        )
        if (
            relative_time >= next_snapshot - snapshot_time_tolerance
            or step == steps - 1
        ):
            snapshot_times.append(relative_time)
            temperature_snapshots.append(temperature.copy())
            progress_snapshots.append(progress.copy())
            level_set_snapshots.append(
                _reaction_level_set(progress, mask, front_threshold, dx)
            )
            next_snapshot += max(snapshot_interval, step_dt)

    signed_distance = _reaction_level_set(progress, mask, front_threshold, dx)
    history_arrays = {
        "time": time_history,
        "unreacted area fraction": unreacted_history,
        "mean progress": mean_progress_history,
        "maximum temperature": max_temperature_history,
        "regression velocity": regression_velocity_history,
        "step duration": step_duration_history,
        "chemical limiter": chemical_limiter_history,
        "chemical rate cap": chemical_rate_cap_history,
        "thermal diffusive CFL": thermal_cfl_history,
        "thermal stability CFL": thermal_stability_cfl_history,
        "temperature cap": temperature_cap_history,
    }
    invalid_histories = [
        name
        for name, values in history_arrays.items()
        if np.any(~np.isfinite(values))
    ]
    if invalid_histories:
        raise PropagationCandidateNumericalError(
            "Propagation histories contain non-finite values: "
            + ", ".join(invalid_histories)
        )
    if np.any(np.isinf(arrival)):
        raise PropagationCandidateNumericalError(
            "Propagation arrival-time field contains infinity"
        )
    snapshot_arrays = {
        "times": np.asarray(snapshot_times, dtype=np.float64),
        "temperature": np.asarray(temperature_snapshots, dtype=np.float64),
        "progress": np.asarray(progress_snapshots, dtype=np.float64),
        "level set": np.asarray(level_set_snapshots, dtype=np.float64),
    }
    invalid_snapshots = [
        name
        for name, values in snapshot_arrays.items()
        if np.any(~np.isfinite(values))
    ]
    if invalid_snapshots:
        raise PropagationCandidateNumericalError(
            "Propagation snapshots contain non-finite values: "
            + ", ".join(invalid_snapshots)
        )
    timing_cv, arrival_coverage = _arrival_time_statistics(arrival, mask)
    # An empty or sparsely observed front must not receive the ideal zero-CV
    # objective.  Keep the raw CV as a diagnostic and add an explicit missing-
    # coverage penalty for ranking.
    coverage_adjusted_nonuniformity = timing_cv + (1.0 - arrival_coverage)
    final_unreacted = float(unreacted_history[-1])
    final_remaining_reactive_mass_fraction: float | None = None
    if reaction_inventory_tracked:
        initial_lp_mass_fraction = molar_mass_lp * initial_lp / initial_reactive_mass
        initial_pva_mass_fraction = (
            molar_mass_pva * initial_pva / initial_reactive_mass
        )
        normalized_final_reactive_mass = (
            initial_lp_mass_fraction * (mobile_lp / initial_lp)
            + initial_pva_mass_fraction * (reactive_pva / initial_pva)
        )
        final_remaining_reactive_mass_fraction = float(
            np.mean(normalized_final_reactive_mass[mask])
        )
        mass_tolerance = 128.0 * np.finfo(np.float64).eps
        if (
            not math.isfinite(final_remaining_reactive_mass_fraction)
            or final_remaining_reactive_mass_fraction < -mass_tolerance
            or final_remaining_reactive_mass_fraction > 1.0 + mass_tolerance
        ):
            raise PropagationCandidateNumericalError(
                "Propagation remaining reactive-mass fraction is outside [0, 1]"
            )
        final_remaining_reactive_mass_fraction = float(
            np.clip(final_remaining_reactive_mass_fraction, 0.0, 1.0)
        )
    metrics = PropagationMetrics(
        final_unreacted_area_fraction=final_unreacted,
        mean_effective_regression_velocity_m_per_s=float(
            np.average(regression_velocity_history, weights=step_duration_history)
        ),
        maximum_effective_regression_velocity_m_per_s=float(np.max(regression_velocity_history)),
        established_time_after_onset_s=(
            float(established_time) if math.isfinite(established_time) else duration
        ),
        reaction_front_nonuniformity=coverage_adjusted_nonuniformity,
        final_mean_global_progress=float(mean_progress_history[-1]),
        final_maximum_temperature_K=float(max_temperature_history[-1]),
        onset_succeeded=True,
        continued_electrical_heating=continued,
        front_arrival_coverage_fraction=arrival_coverage,
    )
    _require_finite_json_numbers(
        metrics.as_dict(), context="Propagation final metrics"
    )
    payload = {
        **metrics.as_dict(),
        "status": "complete",
        "postOnsetBackend": "condensed_propagation",
        "initialMass_kg": comparison_initial_mass,
        "finalMass_kg": comparison_initial_mass,
        "finalMassBudgetResidual_kg": 0.0,
        "maximumMassBudgetRelativeResidual": 0.0,
        "integratedJouleHeat_J": comparison_qj,
        "integratedElectrochemicalHeat_J": comparison_qe,
        "integratedChemicalHeat_J": comparison_qchem,
        "integratedHeatLoss_J": comparison_loss,
        "maximumTemperatureDuringPropagation_K": comparison_max_temperature,
        "establishedTimeCensored": not math.isfinite(established_time),
        "finalEnergyBudgetResidual_J": (
            float(np.sum(density * comparison_caloric.sensible_energy(temperature)[mask])
                  * comparison_volume - comparison_initial_energy
                  - comparison_qj - comparison_qe - comparison_qchem + comparison_loss)
            if comparison_caloric is not None else None
        ),
        "energyBudgetInterpretation": "BC_cp_integral_residual_of_original_explicit_temperature_update_not_corrected",
        "modelScope": "post_onset_condensed_phase_reaction_progress_and_level_set_not_gas_cfd",
        "frontProgressThreshold": front_threshold,
        "establishedReactedAreaFraction": established_fraction,
        "arrivalTimeCoefficientOfVariation": timing_cv,
        "arrivalCoveragePenalty": 1.0 - arrival_coverage,
        "reactionFrontNonuniformityDefinition": (
            "arrival_time_CV_plus_one_minus_arrival_coverage"
        ),
        "maximumChemicalRateLimiterFraction": float(
            np.max(chemical_limiter_history)
        ),
        "maximumChemicalRateCapFraction": float(
            np.max(chemical_rate_cap_history)
        ),
        "allowedMaximumChemicalRateCapFraction": chemical_rate_cap_limit,
        "maximumThermalDiffusiveCFL": float(np.max(thermal_cfl_history)),
        "maximumThermalStabilityCFL": float(
            np.max(thermal_stability_cfl_history)
        ),
        "allowedMaximumThermalStabilityCFL": thermal_cfl_limit,
        "maximumTemperatureCapFraction": float(
            np.max(temperature_cap_history)
        ),
        "allowedMaximumTemperatureCapFraction": temperature_cap_limit,
        "reactionInventoryTracked": reaction_inventory_tracked,
        "legacyHeatOnlyWithoutReactionInventory": False,
        "finalRemainingReactiveMassFraction": (
            final_remaining_reactive_mass_fraction
        ),
        "generatedWaterIsSeparateFromMobileWater": True,
        "electricalHeatingPolicy": heating_mode,
        "electricalHeatingTimeOrigin": heating_time_origin,
        "handoffPolicy": (
            "authorized_v8.2_full_onset_state; post-onset electrical heating "
            "off; preflame heat replay forbidden"
        ),
        "levelSetTracking": "signed_distance_reinitialised_from_the_reaction_progress_front",
    }
    _require_finite_json_numbers(payload, context="Propagation metrics payload")
    (output_dir / "propagation_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "propagation_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time_after_onset_s",
            "unreacted_area_fraction",
            "mean_global_progress",
            "maximum_temperature_K",
            "effective_regression_velocity_m_per_s",
        ])
        writer.writerows(
            zip(
                time_history,
                unreacted_history,
                mean_progress_history,
                max_temperature_history,
                regression_velocity_history,
            )
        )
    np.savez_compressed(
        output_dir / "propagation_fields.npz",
        time_after_onset_s=time_history,
        unreacted_area_fraction=unreacted_history,
        mean_global_progress=mean_progress_history,
        maximum_temperature_K=max_temperature_history,
        effective_regression_velocity_m_per_s=regression_velocity_history,
        chemical_rate_limiter_fraction=chemical_limiter_history,
        chemical_rate_cap_fraction=chemical_rate_cap_history,
        thermal_diffusive_cfl=thermal_cfl_history,
        thermal_stability_cfl=thermal_stability_cfl_history,
        temperature_cap_fraction=temperature_cap_history,
        snapshot_times_s=snapshot_arrays["times"],
        temperature_snapshots_K=snapshot_arrays["temperature"],
        progress_snapshots=snapshot_arrays["progress"],
        level_set_snapshots_m=snapshot_arrays["level set"],
        initial_level_set_m=initial_level_set,
        final_temperature_K=temperature,
        final_alpha_channel1=alpha_1,
        final_alpha_channel2=alpha_2,
        final_global_progress=progress,
        final_mobile_lp_mol_per_m3=mobile_lp,
        final_mobile_water_mol_per_m3=mobile_water,
        final_pva_reactive_repeat_mol_per_m3=reactive_pva,
        final_generated_water_product_mol_per_m3=generated_water_product,
        electrochemical_lp_consumed_mol_per_m3=electrochemical_lp_consumed,
        reaction_inventory_tracked=np.asarray(reaction_inventory_tracked),
        # Censor not-yet-arrived cells at the finite analysis horizon and
        # persist the observation mask separately; the NPZ never relies on a
        # NaN sentinel to encode front coverage.
        front_arrival_time_s=np.where(np.isfinite(arrival), arrival, duration),
        front_arrival_observed_mask=np.isfinite(arrival) & mask,
        final_level_set_m=signed_distance,
        propellant_mask=mask,
    )
    return payload


def final_refinement_objectives(preflame: Sequence[float], metrics: Mapping[str, Any]) -> np.ndarray:
    """Eight-objective post-refinement vector, all expressed as minimisation."""
    preflame_objectives = np.asarray(preflame, dtype=np.float64)
    if preflame_objectives.shape != (4,) or np.any(
        ~np.isfinite(preflame_objectives)
    ):
        raise ValueError(
            "Propagation refinement requires four finite pre-flame objectives"
        )
    propagation_objectives = np.asarray(
        [
            float(metrics["finalUnreactedAreaFraction"]),
            float(metrics["establishedTimeAfterOnset_s"]),
            -float(metrics["meanEffectiveRegressionVelocity_m_per_s"]),
            float(metrics["reactionFrontNonuniformity"]),
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(propagation_objectives)):
        raise ValueError("Propagation refinement objectives must all be finite")
    objectives = np.concatenate((preflame_objectives, propagation_objectives))
    if objectives.shape != (8,) or np.any(~np.isfinite(objectives)):
        raise RuntimeError("Propagation refinement objective vector is invalid")
    return objectives
