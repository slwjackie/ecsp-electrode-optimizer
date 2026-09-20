"""Fail-closed CPU FP64 orchestration for the experimental paper model.

This module advances four deliberately distinct fields:

* the conservative reactive-Euler state printed as paper Eq. (1),
* a condensed temperature and decomposition progress governed by Eq. (2),
* an independently transported material level set governed by Eq. (4), and
* electrical heat supplied either as prescribed fields (paper-faithful mode)
  or by an explicit corrected-BC callback (ECSP-extended mode).

The paper does not publish an interface law coupling Eq. (1) and Eq. (2).
Consequently electrical heat is added to Eq. (1), where it is printed, and is
not silently duplicated into Eq. (2).  The solid state is an explicitly
separate diagnostic solution until such a coupling law is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping
import zipfile

import numpy as np

from .configuration import ReactiveCaseConfig, load_reactive_config
from .core import (
    NCONS,
    REACTION_PROGRESS,
    RHO,
    TOTAL_ENERGY,
    conservative_to_primitive,
    physical_state_mask,
    primitive_to_conservative,
)
from .electrical import (
    ElectricalSourceFields,
    SurfaceContactGeometry,
    joule_heating_sigma_e2,
)
from .eos import TaitEOS
from .front import axis_aligned_sampling_frame, evaluate_species_threshold_sensitivity
from .level_set import (
    BoundaryCondition2D,
    ReinitializationConfig,
    level_set_advection_rhs,
    recommended_level_set_timestep_s,
    reinitialize_transported_level_set,
    signed_distance_rms_error,
    zero_contour_interior_area_m2,
)
from .numerics import BoundaryType, cfl_time_step, finite_volume_rhs
from .provenance import Provenance, ReactiveConfigurationError, ReactiveNumericalError
from .reaction import (
    ArrheniusReaction,
    arrhenius_rate,
    reaction_source,
)
from .solid import (
    SolidThermalModel,
    solid_arrhenius_rate,
    solid_stable_time_step,
    solid_thermal_rhs,
)


RK_WEIGHTS = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)
STAGE_NAMES = ("stage_1", "stage_2", "stage_3")


def _surface_geometry_sha256(geometry: SurfaceContactGeometry) -> str:
    """Hash solver-owned canonical masks without trusting callback metadata."""

    digest = hashlib.sha256()
    digest.update(b"ecsp.surface-contact-overlay/v1\0")
    digest.update(np.asarray(geometry.propellant.shape, dtype="<i8").tobytes())
    for name, mask in (
        ("propellant", geometry.propellant),
        ("anode_contact", geometry.anode_contact),
        ("cathode_contact", geometry.cathode_contact),
    ):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(mask).view(np.uint8).tobytes())
    return digest.hexdigest()


def _terminal_remainder_summary(
    progress: np.ndarray, completion_tolerance: float
) -> dict[str, int | float]:
    values = np.asarray(progress, dtype=np.float64)
    remaining = 1.0 - values
    tolerance_terminal = (remaining > 0.0) & (
        remaining <= completion_tolerance
    )
    return {
        "positive_remainder_terminal_cell_count": int(
            np.count_nonzero(tolerance_terminal)
        ),
        "maximum_unreacted_remainder_in_terminal_cells": (
            float(np.max(remaining[tolerance_terminal]))
            if np.any(tolerance_terminal)
            else 0.0
        ),
        "exact_complete_cell_count": int(np.count_nonzero(remaining == 0.0)),
    }


def _immutable_fp64(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ReactiveConfigurationError(
            f"{name} must have shape {shape}, received {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ReactiveNumericalError(
            f"nonfinite_{name}", f"{name} contains NaN or infinity"
        )
    owned = np.array(array, dtype=np.float64, order="C", copy=True)
    # A read-only flag on an owning ndarray is advisory: callers can re-enable
    # writes with setflags(write=True).  A bytes-backed view has an actually
    # immutable base and therefore keeps completed-result hashes stable.
    output = np.frombuffer(owned.tobytes(order="C"), dtype=np.float64).reshape(shape)
    return output


def _immutable_state(value: Any, shape: tuple[int, int, int], name: str) -> np.ndarray:
    return _immutable_fp64(value, shape, name)


def _json_value(value: Any) -> Any:
    """Build a deterministic JSON tree or reject the value explicitly.

    In particular, never fall back to ``str(object)``: an object's repr may
    include an address and would make a supposedly deterministic result depend
    on process layout.  Mapping keys are required to be strings so sorting is
    total and distinct callback keys cannot collapse after coercion.
    """

    active_containers: set[int] = set()

    def convert(item: Any, path: str) -> Any:
        if isinstance(item, np.ndarray):
            return convert(item.tolist(), path)
        if isinstance(item, np.generic):
            return convert(item.item(), path)
        if isinstance(item, Mapping):
            identifier = id(item)
            if identifier in active_containers:
                raise ReactiveConfigurationError(
                    f"{path} contains a cyclic mapping and is not valid JSON metadata"
                )
            if any(not isinstance(key, str) for key in item):
                raise ReactiveConfigurationError(
                    f"{path} mapping keys must all be strings"
                )
            active_containers.add(identifier)
            try:
                return {
                    key: convert(item[key], f"{path}.{key}")
                    for key in sorted(item)
                }
            finally:
                active_containers.remove(identifier)
        if isinstance(item, (tuple, list)):
            identifier = id(item)
            if identifier in active_containers:
                raise ReactiveConfigurationError(
                    f"{path} contains a cyclic sequence and is not valid JSON metadata"
                )
            active_containers.add(identifier)
            try:
                return [
                    convert(child, f"{path}[{index}]")
                    for index, child in enumerate(item)
                ]
            finally:
                active_containers.remove(identifier)
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ReactiveNumericalError(
                    "nonfinite_result_metadata",
                    f"{path} contains NaN or infinity",
                )
            return item
        if isinstance(item, (str, int, bool)) or item is None:
            return item
        raise ReactiveConfigurationError(
            f"{path} contains unsupported JSON metadata type "
            f"{type(item).__name__}"
        )

    return convert(value, "metadata")


def _deep_freeze_json(value: Any) -> Any:
    """Return an immutable, ownership-independent JSON-compatible tree.

    ``ReactiveResult`` is frozen, but a shallow ``MappingProxyType`` still lets a
    caller mutate nested dictionaries and lists after the run.  Normalize first
    (which also rejects non-finite floats) and then freeze every container so the
    saved result and its digest cannot be changed through an aliased object.
    """

    normalized = _json_value(value)

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {str(key): freeze(child) for key, child in item.items()}
            )
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(normalized)


def _json_resource_charge_bytes(value: Any) -> tuple[int, int, int]:
    """Measure canonical JSON bytes and the current interpreter's deep size."""

    canonical = _json_value(value)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    seen: set[int] = set()

    def deep_size(item: Any) -> int:
        identifier = id(item)
        if identifier in seen:
            return 0
        seen.add(identifier)
        size = sys.getsizeof(item)
        if isinstance(item, Mapping):
            size += sum(deep_size(key) + deep_size(child) for key, child in item.items())
        elif isinstance(item, (tuple, list)):
            size += sum(deep_size(child) for child in item)
        return size

    measured_deep_size = deep_size(canonical)
    # Charge both the retained canonical tree and its compact serialized audit
    # representation.  This is a measured CPython/PyPy object-size guard for
    # this process, not a promise about OS RSS or allocator fragmentation.
    return len(payload), measured_deep_size, measured_deep_size + len(payload)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scaled_sum(
    values: np.ndarray,
    scale: float,
    *,
    axis: tuple[int, ...] | None,
    name: str,
) -> np.ndarray | float:
    if not math.isfinite(scale):
        raise ReactiveNumericalError(
            "nonfinite_budget_scale", f"{name} integration scale is non-finite"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.asarray(values, dtype=np.float64) * scale
    if not np.isfinite(scaled).all():
        raise ReactiveNumericalError(
            "nonfinite_budget_integrand",
            f"{name} overflowed while scaling individual cells",
        )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.sum(scaled, axis=axis)
    if not np.isfinite(result).all():
        raise ReactiveNumericalError(
            "nonfinite_budget_reduction", f"{name} reduction overflowed"
        )
    return float(result) if np.ndim(result) == 0 else result


def _finite_accumulate(current: Any, increment: Any, name: str) -> Any:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(current, dtype=np.float64) + np.asarray(
            increment, dtype=np.float64
        )
    if not np.isfinite(result).all():
        raise ReactiveNumericalError(
            "nonfinite_budget_accumulator", f"{name} accumulation overflowed"
        )
    return float(result) if result.ndim == 0 else result


def _nullable_finite(value: float) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _minimum_positive_ratio(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    *,
    name: str,
) -> float:
    """Return min(numerator/denominator), accepting +inf as unbounded.

    This helper is deliberately NumPy based: ordinary scalar division of a
    finite numerator by a positive subnormal rate can overflow.  Positive
    infinity is the correct limiting interpretation in that case; NaN and
    non-positive ratios remain hard failures.
    """

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = np.asarray(numerator, dtype=np.float64) / np.asarray(
            denominator, dtype=np.float64
        )
    if ratio.size == 0 or np.isnan(ratio).any() or np.any(ratio <= 0.0):
        raise ReactiveNumericalError(
            "invalid_source_timestep_ratio", f"{name} produced an invalid ratio"
        )
    return float(np.min(ratio))


def _strict_headroom_limit(limit: float, headroom: float) -> float:
    """Apply headroom and keep a finite bound strictly below its event."""

    with np.errstate(over="ignore", invalid="ignore"):
        bounded = float(np.float64(limit) * np.float64(headroom))
    if math.isnan(bounded) or bounded <= 0.0:
        raise ReactiveNumericalError(
            "invalid_source_timestep_limit", "Source stability limit is invalid"
        )
    if math.isfinite(bounded):
        bounded = float(np.nextafter(bounded, 0.0))
        if bounded <= 0.0:
            raise ReactiveNumericalError(
                "source_timestep_underflow",
                "A positive source stability limit underflowed in FP64",
            )
    return bounded


def _front_step_summary(sensitivity: Any, end_time_s: float) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for threshold, result in sensitivity.by_threshold.items():
        previous_positions = result.previous_snapshot.normal_iso_position_m
        current_positions = result.current_snapshot.normal_iso_position_m
        previous_unique = result.previous_snapshot.unique_intersection_mask
        current_unique = result.current_snapshot.unique_intersection_mask
        previous_mean = (
            float(np.mean(previous_positions[previous_unique]))
            if np.any(previous_unique)
            else math.nan
        )
        current_mean = (
            float(np.mean(current_positions[current_unique]))
            if np.any(current_unique)
            else math.nan
        )
        transition = result.diagnostics
        rows[f"{threshold:.17g}"] = {
            "threshold": float(threshold),
            "threshold_provenance": result.threshold_provenance.value,
            "previous_mean_positive_x_ray_position_m": _nullable_finite(previous_mean),
            "current_mean_positive_x_ray_position_m": _nullable_finite(current_mean),
            "mean_positive_x_ray_speed_m_per_s": _nullable_finite(
                result.mean_normal_regression_speed_m_s
            ),
            "median_positive_x_ray_speed_m_per_s": _nullable_finite(
                result.median_normal_regression_speed_m_s
            ),
            "positive_x_ray_speed_standard_deviation_m_per_s": _nullable_finite(
                result.normal_regression_speed_standard_deviation_m_s
            ),
            "per_ray_positive_x_speed_m_per_s": [
                _nullable_finite(value)
                for value in result.local_normal_regression_speed_m_s
            ],
            "persistent_unique_ray_count": transition.persistent_unique_ray_count,
            "front_birth_ray_count": transition.front_birth_ray_count,
            "front_death_ray_count": transition.front_death_ray_count,
            "branching_candidate_ray_count": (
                transition.unique_to_multiple_branching_candidate_ray_count
            ),
            "merging_candidate_ray_count": (
                transition.multiple_to_unique_merging_candidate_ray_count
            ),
            "topology_ambiguous_ray_count": transition.topology_ambiguous_ray_count,
        }
    return {
        "end_time_s": float(end_time_s),
        "ray_axis": "positive_x",
        "ray_axis_provenance": "ASSUMED_NOT_FROM_PAPER",
        "tracking_method": "fixed_positive_x_ray_linear_subcell_interpolation",
        "metric_limitation": (
            "FIXED_X_DIRECTION_DISPLACEMENT_NOT_LOCAL_CONTOUR_NORMAL_SPEED;"
            "MULTIPLE_INTERSECTION_RAYS_ARE_EXCLUDED"
        ),
        "by_threshold": rows,
    }


def _projected_history_memory(
    config: ReactiveCaseConfig,
    *,
    electrical_stage_metadata_charge_per_record: int = 8192,
    electrical_charge_basis: str = "CALLBACK_UNKNOWN_RUNTIME_MEASURED",
) -> dict[str, int | str]:
    """Versioned planning estimate for retained Python result history.

    The estimate includes an explicit measured/baseline charge for electrical
    metadata in addition to versioned fixed/container allowances.  It is a
    deterministic guard, not a promise about exact allocator use or OS RSS.
    """

    retained_front_samples = (
        math.ceil(config.maximum_steps / config.front_history_stride_steps) + 1
    )
    fixed_step_bytes = config.maximum_steps * (
        1024 + 2 * np.dtype(np.float64).itemsize
    )
    if electrical_stage_metadata_charge_per_record <= 0:
        raise ReactiveConfigurationError(
            "electrical stage metadata charge must be positive"
        )
    electrical_stage_allowance_bytes = (
        config.maximum_steps * 3 * electrical_stage_metadata_charge_per_record
    )
    per_front_sample_bytes = 1024 + len(config.front_thresholds) * (
        2048 + config.ny * 32
    )
    front_bytes = retained_front_samples * per_front_sample_bytes
    return {
        "estimator": "ECSP_REACTIVE_PYTHON_HISTORY_V2_PLANNING_ESTIMATE",
        "maximum_steps": config.maximum_steps,
        "front_history_stride_steps": config.front_history_stride_steps,
        "retained_front_sample_upper_bound": retained_front_samples,
        "fixed_step_history_bytes": fixed_step_bytes,
        "electrical_stage_metadata_charge_per_record_bytes": (
            electrical_stage_metadata_charge_per_record
        ),
        "electrical_stage_metadata_charge_basis": electrical_charge_basis,
        "electrical_stage_metadata_preflight_allowance_bytes": (
            electrical_stage_allowance_bytes
        ),
        "front_history_bytes": front_bytes,
        "projected_without_electrical_stage_metadata_bytes": (
            fixed_step_bytes + front_bytes
        ),
        "projected_total_bytes": (
            fixed_step_bytes + electrical_stage_allowance_bytes + front_bytes
        ),
        "configured_maximum_bytes": config.maximum_history_bytes,
    }


def _projected_working_set_memory(config: ReactiveCaseConfig) -> dict[str, int | str]:
    """Return a checked planning bound for grid-sized arrays and artifact staging.

    NumPy temporaries, three retained SSPRK derivatives, face reconstructions,
    immutable callback snapshots, and the in-memory deterministic NPZ writer all
    coexist at different points.  The versioned 4096-byte cell allowance is
    intentionally generous for this implementation, while the report remains
    explicit that Python/NumPy allocator fragmentation and third-party-library
    workspace are not measurable in advance.
    """

    cell_count = int(config.nx) * int(config.ny)
    per_cell_allowance = 4096
    fixed_allowance = 16 * 1024 * 1024
    time_history_array_allowance = int(config.maximum_steps) * 2 * 8
    grid_array_allowance = cell_count * per_cell_allowance
    projected = (
        fixed_allowance
        + grid_array_allowance
        + time_history_array_allowance
        + int(config.maximum_history_bytes)
    )
    return {
        "estimator": "ECSP_REACTIVE_NUMPY_WORKING_SET_V1_PLANNING_BOUND",
        "cell_count": cell_count,
        "per_cell_array_and_scratch_allowance_bytes": per_cell_allowance,
        "grid_array_and_scratch_allowance_bytes": grid_array_allowance,
        "fixed_runtime_and_npz_staging_allowance_bytes": fixed_allowance,
        "time_history_array_allowance_bytes": time_history_array_allowance,
        "retained_history_budget_included_bytes": int(config.maximum_history_bytes),
        "projected_total_bytes": projected,
        "configured_maximum_bytes": int(config.maximum_working_set_bytes),
        "scope_limitation": (
            "VERSIONED_PREALLOCATION_PLANNING_BOUND_NOT_AN_OS_RSS_OR_"
            "THIRD_PARTY_ALLOCATOR_GUARANTEE"
        ),
    }


@dataclass(frozen=True)
class ElectricalStageRequest:
    """Read-only state passed to an ECSP-extended electrical callback."""

    time_s: float
    stage_name: str
    conservative_state: np.ndarray
    solid_temperature_K: np.ndarray
    solid_decomposition_alpha: np.ndarray
    material_level_set_m: np.ndarray
    dx_m: float
    dy_m: float
    surface_geometry: SurfaceContactGeometry | None


ElectricalSourceCallback = Callable[[ElectricalStageRequest], ElectricalSourceFields]


@dataclass(frozen=True)
class StepDiagnostics:
    step: int
    start_time_s: float
    dt_s: float
    proposed_time_step_limit: str
    selected_time_step_limit: str
    flow_cfl_limit_s: float
    flow_reaction_limit_s: float
    solid_stability_limit_s: float
    level_set_cfl_limit_s: float
    positivity_face_fallback_count: int
    positivity_face_fallback_max_abs_state_correction: float
    reconstructed_face_count: int
    electrical_source_evaluation_count: int
    rejected_trial_electrical_source_evaluation_count: int
    stage_retry_count: int
    material_level_set_reinitialized: bool
    reinitialization_sign_changed_cell_count: int
    reinitialization_area_drift_m2: float
    reinitialization_relative_area_drift: float

    def as_dict(self) -> dict[str, Any]:
        result = dict(vars(self))
        for field_name in (
            "flow_cfl_limit_s",
            "flow_reaction_limit_s",
            "solid_stability_limit_s",
            "level_set_cfl_limit_s",
        ):
            value = float(result[field_name])
            if math.isnan(value) or value < 0.0:
                raise ReactiveNumericalError(
                    "invalid_timestep_diagnostic",
                    f"{field_name} is not a valid non-negative limit",
                )
            status_name = field_name.removesuffix("_s") + "_status"
            if math.isinf(value):
                result[field_name] = None
                result[status_name] = "UNBOUNDED"
            else:
                result[status_name] = "FINITE"
        return _json_value(result)


@dataclass(frozen=True)
class ConservationBudget:
    """Domain-integrated per-unit-depth balance for the five Euler variables."""

    component_names: tuple[str, ...]
    initial_integral: tuple[float, ...]
    final_integral: tuple[float, ...]
    flux_divergence_increment: tuple[float, ...]
    reaction_increment: tuple[float, ...]
    electrical_increment: tuple[float, ...]
    closure_residual: tuple[float, ...]
    reaction_progress_increment: float
    reaction_heat_increment_J_per_m: float
    configured_heat_per_progress_J_per_kg: float
    reaction_bookkeeping_residual_J_per_m: float
    electrical_energy_increment_J_per_m: float

    def as_dict(self) -> dict[str, Any]:
        result = dict(vars(self))
        result["reaction_progress_increment_unit"] = (
            "kg/m (two-dimensional domain integral per unit out-of-plane depth)"
        )
        return _json_value(result)


@dataclass(frozen=True)
class SolidBudget:
    initial_sensible_energy_J_per_m: float
    final_sensible_energy_J_per_m: float
    conduction_increment_J_per_m: float
    decomposition_heat_increment_J_per_m: float
    externally_coupled_heat_increment_J_per_m: float
    closure_residual_J_per_m: float
    initial_alpha_area_integral_m2: float
    final_alpha_area_integral_m2: float
    rate_increment_area_integral_m2: float
    alpha_closure_residual_m2: float

    def as_dict(self) -> dict[str, Any]:
        return _json_value(vars(self))


@dataclass(frozen=True)
class ReactiveResult:
    conservative_state: np.ndarray
    solid_temperature_K: np.ndarray
    solid_decomposition_alpha: np.ndarray
    material_level_set_m: np.ndarray
    final_time_s: float
    accepted_steps: int
    step_time_s: np.ndarray
    step_dt_s: np.ndarray
    steps: tuple[StepDiagnostics, ...]
    conservation: ConservationBudget
    solid_budget: SolidBudget
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        ny, nx = self.solid_temperature_K.shape
        object.__setattr__(
            self,
            "conservative_state",
            _immutable_state(self.conservative_state, (ny, nx, NCONS), "conservative_state"),
        )
        object.__setattr__(
            self,
            "solid_temperature_K",
            _immutable_fp64(self.solid_temperature_K, (ny, nx), "solid_temperature_K"),
        )
        object.__setattr__(
            self,
            "solid_decomposition_alpha",
            _immutable_fp64(
                self.solid_decomposition_alpha,
                (ny, nx),
                "solid_decomposition_alpha",
            ),
        )
        object.__setattr__(
            self,
            "material_level_set_m",
            _immutable_fp64(
                self.material_level_set_m, (ny, nx), "material_level_set_m"
            ),
        )
        time = np.asarray(self.step_time_s, dtype=np.float64)
        dt = np.asarray(self.step_dt_s, dtype=np.float64)
        if time.shape != (self.accepted_steps,) or dt.shape != (self.accepted_steps,):
            raise ReactiveConfigurationError("Result time-history lengths do not match accepted_steps")
        if not np.isfinite(time).all() or not np.isfinite(dt).all() or np.any(dt <= 0.0):
            raise ReactiveNumericalError("invalid_result_history", "Result time history is invalid")
        object.__setattr__(
            self,
            "step_time_s",
            _immutable_fp64(time, (self.accepted_steps,), "step_time_s"),
        )
        object.__setattr__(
            self,
            "step_dt_s",
            _immutable_fp64(dt, (self.accepted_steps,), "step_dt_s"),
        )
        object.__setattr__(self, "metadata", _deep_freeze_json(self.metadata))

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "ecsp.paper-reactive-result/v1",
            "final_time_s": self.final_time_s,
            "accepted_steps": self.accepted_steps,
            "array_shapes": {
                "conservative_state": list(self.conservative_state.shape),
                "solid_temperature_K": list(self.solid_temperature_K.shape),
                "solid_decomposition_alpha": list(self.solid_decomposition_alpha.shape),
                "material_level_set_m": list(self.material_level_set_m.shape),
            },
            "steps": [step.as_dict() for step in self.steps],
            "conservation": self.conservation.as_dict(),
            "solid_budget": self.solid_budget.as_dict(),
            "metadata": _json_value(self.metadata),
        }

    def save(self, output_prefix: str | Path) -> tuple[Path, Path]:
        """Write deterministic NPZ/JSON artifacts and return their paths."""

        prefix = Path(output_prefix).expanduser()
        if prefix.suffix in {".npz", ".json"}:
            prefix = prefix.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        npz_path = Path(f"{prefix}.npz")
        json_path = Path(f"{prefix}.json")
        arrays = {
            "conservative_state": self.conservative_state,
            "material_level_set_m": self.material_level_set_m,
            "solid_decomposition_alpha": self.solid_decomposition_alpha,
            "solid_temperature_K": self.solid_temperature_K,
            "step_dt_s": self.step_dt_s,
            "step_time_s": self.step_time_s,
        }
        npz_bytes = _deterministic_npz(arrays)
        summary = self.summary()
        summary["npz_sha256"] = hashlib.sha256(npz_bytes).hexdigest()
        json_bytes = (
            json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        _atomic_write(npz_path, npz_bytes)
        _atomic_write(json_path, json_bytes)
        return npz_path, json_path


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(
                payload,
                np.ascontiguousarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


@dataclass
class _Accumulator:
    flux: np.ndarray
    reaction: np.ndarray
    electrical: np.ndarray
    solid_conduction_J_per_m: float = 0.0
    solid_reaction_J_per_m: float = 0.0
    solid_external_J_per_m: float = 0.0
    solid_alpha_area_m2: float = 0.0
    fallback_count: int = 0
    fallback_max_abs_state_correction: float = 0.0
    interface_count: int = 0
    electrical_evaluations: int = 0
    flow_completion_tolerance_zeroed_cell_evaluations: int = 0
    solid_completion_tolerance_zeroed_cell_evaluations: int = 0
    electrical_source_stage_records: list[dict[str, Any]] = field(default_factory=list)
    electrical_source_stage_metadata_json_bytes: int = 0
    electrical_source_stage_metadata_measured_deep_bytes: int = 0
    electrical_source_stage_metadata_resource_charge_bytes: int = 0


@dataclass(frozen=True)
class _StageDerivative:
    flow_total: np.ndarray
    flow_flux: np.ndarray
    flow_reaction: np.ndarray
    flow_electrical: np.ndarray
    solid_temperature: np.ndarray
    solid_alpha: np.ndarray
    level_set: np.ndarray
    fallback_count: int
    fallback_max_abs_state_correction: float
    interface_count: int
    solid_diagnostics: Mapping[str, float]
    electrical_metadata: Mapping[str, Any]
    flow_completion_tolerance_zeroed_cell_count: int
    solid_completion_tolerance_zeroed_cell_count: int


class ReactiveSolver:
    """Single-process reference solver; deterministic for deterministic sources."""

    def _provider_authentication_status(self) -> str:
        provider = self.config.electrical_source_provider
        if provider == "CONFIG_UNIFORM_FIELD":
            return "CONFIG_DECLARED_FIELD_NO_GEOMETRY_SOLVE"
        if provider == "PRESCRIBED_ARRAY_FIELDS":
            return "ARRAY_METADATA_SELF_ATTESTED_UNVERIFIED"
        return "CALLBACK_SELF_ATTESTED_UNVERIFIED"

    def __init__(
        self,
        config: ReactiveCaseConfig,
        *,
        prescribed_electrical: ElectricalSourceFields | None = None,
        electrical_callback: ElectricalSourceCallback | None = None,
        anode_contact: Any | None = None,
        cathode_contact: Any | None = None,
        reinitialization: ReinitializationConfig | None = None,
    ) -> None:
        if not isinstance(config, ReactiveCaseConfig):
            raise ReactiveConfigurationError("config must be a validated ReactiveCaseConfig")
        config.validate_runtime_contract()
        self.working_set_projection = _projected_working_set_memory(config)
        if int(self.working_set_projection["projected_total_bytes"]) > (
            config.maximum_working_set_bytes
        ):
            raise ReactiveNumericalError(
                "projected_working_set_exceeds_limit",
                "Configured grid and solver/result scratch can exceed the explicit "
                "working-set memory limit before grid allocation",
                self.working_set_projection,
            )
        self.runtime_configuration_audit = config.runtime_provenance_audit()
        self.effective_configuration_sha256 = _canonical_json_sha256(
            {
                "merged_source_configuration": config.raw,
                "effective_runtime_configuration": (
                    self.runtime_configuration_audit[
                        "effective_runtime_configuration"
                    ]
                ),
            }
        )
        runtime_algorithms = {
            "spatial_reconstruction": "WENO5_JS",
            "riemann_solver": "HLL",
            "time_integrator": "SSPRK33",
            "source_coupling": "SSPRK_STAGE_COUPLED",
            "material_interface": "PASSIVE_SINGLE_EOS",
            "level_set_gradient": "WENO5_JS_UPWIND",
            "electrical_gradient_boundary": "FIRST_ORDER_ONE_SIDED",
            "mpi_decomposition": "SERIAL_SINGLE_PROCESS",
        }
        for algorithm_name, implemented_choice in runtime_algorithms.items():
            declared_choice = config.algorithms.get(algorithm_name)
            if declared_choice != implemented_choice:
                raise ReactiveConfigurationError(
                    f"ReactiveSolver executes {algorithm_name}={implemented_choice}; "
                    f"received declaration {declared_choice!r}"
                )
        self.config = config
        self.shape = (config.ny, config.nx)
        if config.algorithms["mpi_decomposition"] != "SERIAL_SINGLE_PROCESS":
            raise ReactiveConfigurationError(
                "ReactiveSolver is a single-process runner and requires "
                "mpi_decomposition=SERIAL_SINGLE_PROCESS; ROWWISE_WENO3_HALO "
                "must use a separately integrated distributed runner"
            )
        self.eos = TaitEOS(
            rho0_kg_per_m3=config.tait_rho0_kg_per_m3,
            B_Pa=config.tait_B_Pa,
            N=config.tait_N,
            A_Pa=config.tait_A_Pa,
            cv_J_per_kg_K=config.caloric_cv_J_per_kgK,
            T_ref_K=config.caloric_reference_temperature_K,
            e_ref_J_per_kg=config.caloric_reference_energy_J_per_kg,
        )
        self.reaction = ArrheniusReaction(
            pre_exponential_s_inv=config.reaction_preexponential_per_s,
            activation_energy_J_per_mol=config.reaction_activation_energy_J_per_mol,
            reaction_order=config.reaction_order,
            heat_release_J_per_kg=config.reaction_heat_J_per_kg,
            gas_constant_J_per_mol_K=config.gas_constant_J_per_molK,
        )
        self.solid_model = SolidThermalModel(
            density_kg_per_m3=config.solid_density_kg_per_m3,
            heat_capacity_J_per_kgK=config.solid_heat_capacity_J_per_kgK,
            conductivity_W_per_mK=config.solid_conductivity_W_per_mK,
            preexponential_per_s=config.decomposition_preexponential_per_s,
            activation_energy_J_per_mol=config.decomposition_activation_energy_J_per_mol,
            gas_constant_J_per_molK=config.gas_constant_J_per_molK,
            heat_of_decomposition_J_per_kg=config.decomposition_heat_J_per_kg,
            reaction_order=config.decomposition_order,
        )
        flow_boundary = config.algorithms["flow_boundary"]
        if flow_boundary not in {"PERIODIC", "OUTFLOW"}:
            raise ReactiveConfigurationError(
                f"Unsupported runtime flow boundary {flow_boundary!r}"
            )
        self.flow_boundary = (
            BoundaryType.PERIODIC
            if flow_boundary == "PERIODIC"
            else BoundaryType.TRANSMISSIVE
        )
        solid_boundary = config.algorithms["solid_boundary"]
        if solid_boundary not in {"PERIODIC", "OUTFLOW"}:
            raise ReactiveConfigurationError(
                f"Unsupported runtime solid boundary {solid_boundary!r}"
            )
        self.solid_boundary = solid_boundary
        level_set_boundary_x = config.algorithms["level_set_boundary_x"]
        level_set_boundary_y = config.algorithms["level_set_boundary_y"]
        if level_set_boundary_x not in {"PERIODIC", "OUTFLOW"} or (
            level_set_boundary_y not in {"PERIODIC", "OUTFLOW"}
        ):
            raise ReactiveConfigurationError(
                "Runtime level-set boundaries must be OUTFLOW or PERIODIC"
            )
        if level_set_boundary_x == "PERIODIC":
            raise ReactiveConfigurationError(
                "level_set_boundary_x=PERIODIC is incompatible with the built-in "
                "single-plane phi=x-interface_x initializer; an explicit periodic "
                "paired-interface initializer is not implemented"
            )
        self.level_set_boundary = BoundaryCondition2D(
            x="periodic" if level_set_boundary_x == "PERIODIC" else "outflow",
            y="periodic" if level_set_boundary_y == "PERIODIC" else "outflow",
        )
        if (anode_contact is None) != (cathode_contact is None):
            raise ReactiveConfigurationError(
                "anode_contact and cathode_contact must be supplied together"
            )
        self.surface_geometry = (
            None
            if anode_contact is None
            else SurfaceContactGeometry.build(anode_contact, cathode_contact)
        )
        if self.surface_geometry is not None and self.surface_geometry.propellant.shape != self.shape:
            raise ReactiveConfigurationError(
                f"Surface contact masks must have solver shape {self.shape}"
            )
        if self.surface_geometry is not None:
            for mask in (
                self.surface_geometry.propellant,
                self.surface_geometry.anode_contact,
                self.surface_geometry.cathode_contact,
            ):
                mask.setflags(write=False)
        self.surface_geometry_sha256 = (
            _surface_geometry_sha256(self.surface_geometry)
            if self.surface_geometry is not None
            else None
        )
        if config.model_mode == "paper_faithful" and self.surface_geometry is not None:
            raise ReactiveConfigurationError(
                "paper_faithful prescribed-field mode forbids electrode surface masks: "
                "the paper does not publish a geometry-to-field solve, so geometry ranking "
                "is available only through an explicit ecsp_extended electrical callback"
            )

        declared_reinitialization = config.algorithms["level_set_reinitialization"]
        expected_reinitialization = (
            "NONE"
            if config.reinitialization_interval_steps == 0
            else "SUSSMAN_FIRST_ORDER"
        )
        if declared_reinitialization != expected_reinitialization:
            raise ReactiveConfigurationError(
                "Runtime reinitialization declaration contradicts its interval: "
                f"expected {expected_reinitialization}, received "
                f"{declared_reinitialization}"
            )
        if config.reinitialization_interval_steps == 0:
            if reinitialization is not None and reinitialization.enabled:
                raise ReactiveConfigurationError(
                    "Reinitialization is enabled but its configured interval is zero"
                )
            self.reinitialization = ReinitializationConfig(enabled=False)
        else:
            if reinitialization is None or not reinitialization.enabled:
                raise ReactiveConfigurationError(
                    "A nonzero reinitialization interval requires an explicit enabled "
                    "ReinitializationConfig; pseudo-time defaults are not hidden"
                )
            reinitialization.validate()
            self.reinitialization = reinitialization

        if config.model_mode == "paper_faithful":
            if electrical_callback is not None:
                raise ReactiveConfigurationError(
                    "paper_faithful mode does not accept the non-paper BV callback"
                )
            self.electrical_callback = None
            if config.electrical_source_provider == "CONFIG_UNIFORM_FIELD":
                if prescribed_electrical is not None:
                    raise ReactiveConfigurationError(
                        "CONFIG_UNIFORM_FIELD rejects supplied electrical arrays; "
                        "select PRESCRIBED_ARRAY_FIELDS to consume them"
                    )
                self.prescribed_electrical = self._validate_electrical_fields(
                    self._constant_prescribed_electrical()
                )
                consumed = [
                    "config.electrical.conductivity",
                    "config.electrical.electric_field_x",
                    "config.electrical.electric_field_y",
                    "config.electrical.electrochemical_heat",
                ]
            elif config.electrical_source_provider == "PRESCRIBED_ARRAY_FIELDS":
                if prescribed_electrical is None:
                    raise ReactiveConfigurationError(
                        "PRESCRIBED_ARRAY_FIELDS requires explicit ElectricalSourceFields"
                    )
                self.prescribed_electrical = self._validate_electrical_fields(
                    prescribed_electrical
                )
                consumed = [
                    "prescribed_electrical.joule_heat_W_per_m3",
                    "prescribed_electrical.electrochemical_heat_W_per_m3",
                    "prescribed_electrical.electric_potential_V",
                    "prescribed_electrical.metadata",
                ]
            else:
                raise ReactiveConfigurationError(
                    "Unsupported paper_faithful electrical source provider"
                )
        else:
            if self.surface_geometry is None:
                raise ReactiveConfigurationError(
                    "ecsp_extended geometry-source contract requires explicit anode and "
                    "cathode surface masks; geometry-free callbacks are not ranking inputs"
                )
            if prescribed_electrical is not None:
                raise ReactiveConfigurationError(
                    "ecsp_extended mode requires callback fields and rejects prescribed substitution"
                )
            if electrical_callback is None or not callable(electrical_callback):
                raise ReactiveConfigurationError(
                    "ecsp_extended requires an explicit electrical_callback; BV is never faked"
                )
            self.electrical_callback = electrical_callback
            self.prescribed_electrical = None
            consumed = [
                "electrical_callback.stage_fields",
                "surface_geometry.propellant",
                "surface_geometry.anode_contact",
                "surface_geometry.cathode_contact",
            ]
        uniform_scalars = {
            "config.electrical.conductivity": config.conductivity_S_per_m,
            "config.electrical.electric_field_x": config.electric_field_x_V_per_m,
            "config.electrical.electric_field_y": config.electric_field_y_V_per_m,
            "config.electrical.electrochemical_heat": (
                config.electrochemical_heat_W_per_m3
            ),
        }
        self.consumed_electrical_inputs = _deep_freeze_json(
            {
                "provider": config.electrical_source_provider,
                "consumed_inputs": consumed,
                "ignored_config_uniform_scalars": [
                    name
                    for name, value in uniform_scalars.items()
                    if value is not None and name not in consumed
                ],
                "config_uniform_scalars_absent": [
                    name for name, value in uniform_scalars.items() if value is None
                ],
                "configuration_reported_not_consumed_inputs": (
                    config.provenance_report.get("not_consumed_inputs", {})
                ),
            }
        )
        if self.prescribed_electrical is not None:
            # Prescribed metadata is immutable, so its complete retained stage
            # record can be measured before a run.  Use the longest finite FP64
            # time rendering and every stage name; object sizes of floats do not
            # depend on their value in supported Python interpreters.
            preflight_charges = []
            for stage_name in STAGE_NAMES:
                record = self._electrical_stage_record(
                    self.prescribed_electrical,
                    stage_name,
                    np.finfo(np.float64).max,
                )
                _, _, charge = _json_resource_charge_bytes(record)
                preflight_charges.append(charge)
            self._electrical_stage_metadata_preflight_charge_bytes = max(
                preflight_charges
            )
            self._electrical_stage_metadata_preflight_basis = (
                "PRESCRIBED_COMPLETE_STAGE_RECORD_MEASURED_DEEP_PLUS_JSON"
            )
        else:
            # A callback's metadata shape is unknowable before invocation.  The
            # baseline prevents a zero-cost projection; every returned record is
            # measured and charged against the hard runtime history limit.
            self._electrical_stage_metadata_preflight_charge_bytes = 8192
            self._electrical_stage_metadata_preflight_basis = (
                "CALLBACK_BASELINE_ONLY_DYNAMIC_RECORDS_MEASURED_AT_RUNTIME"
            )
        self._runtime_electrical_source_evaluations = 0

    def _constant_prescribed_electrical(self) -> ElectricalSourceFields:
        config = self.config
        scalar_values = (
            config.conductivity_S_per_m,
            config.electric_field_x_V_per_m,
            config.electric_field_y_V_per_m,
            config.electrochemical_heat_W_per_m3,
        )
        if any(value is None for value in scalar_values):
            raise ReactiveConfigurationError(
                "CONFIG_UNIFORM_FIELD requires all four configured electrical scalars"
            )
        provider_record = config.provenance_report["algorithms"][
            "electrical_source_provider"
        ]
        if provider_record.get("choice") != config.electrical_source_provider:
            provider_record = {
                "provenance": "ASSUMED_NOT_FROM_PAPER",
                "source": "non-provenance-bearing runtime source-provider override",
            }
        x = (np.arange(config.nx, dtype=np.float64) + 0.5) * config.dx_m
        y = (np.arange(config.ny, dtype=np.float64) + 0.5) * config.dy_m
        potential = -(
            config.electric_field_x_V_per_m * x[None, :]
            + config.electric_field_y_V_per_m * y[:, None]
        )
        joule, diagnostics = joule_heating_sigma_e2(
            potential,
            config.conductivity_S_per_m,
            config.dx_m,
            config.dy_m,
            gradient_boundary_scheme=config.algorithms[
                "electrical_gradient_boundary"
            ],
        )
        electrochemical = np.full(
            self.shape, config.electrochemical_heat_W_per_m3, dtype=np.float64
        )
        return ElectricalSourceFields(
            joule_heat_W_per_m3=joule,
            electrochemical_heat_W_per_m3=electrochemical,
            electric_potential_V=potential,
            metadata={
                "provider": "CONFIG_UNIFORM_FIELD",
                "provenance": provider_record["provenance"],
                "source": provider_record["source"],
                "potential_status": "constructed_from_configured_uniform_field",
                **diagnostics,
            },
        )

    def _validate_electrical_fields(
        self, fields: ElectricalSourceFields
    ) -> ElectricalSourceFields:
        if not isinstance(fields, ElectricalSourceFields):
            raise ReactiveConfigurationError(
                "Electrical source provider must return ElectricalSourceFields"
            )
        joule = _immutable_fp64(fields.joule_heat_W_per_m3, self.shape, "joule_heat")
        if np.any(joule < 0.0):
            raise ReactiveNumericalError(
                "negative_joule_heat", "Joule heat must be non-negative"
            )
        electrochemical = _immutable_fp64(
            fields.electrochemical_heat_W_per_m3,
            self.shape,
            "electrochemical_heat",
        )
        potential = _immutable_fp64(
            fields.electric_potential_V, self.shape, "electric_potential"
        )
        total = joule + electrochemical
        if not np.isfinite(total).all():
            raise ReactiveNumericalError(
                "nonfinite_electrical_heat_sum",
                "Joule plus electrochemical heat became non-finite",
            )
        if np.any(total < 0.0):
            raise ReactiveNumericalError(
                "negative_total_electrical_heat",
                "Joule plus electrochemical source must be non-negative; this "
                "reference contract supports electrical heating, not an "
                "unbounded cooling callback",
            )
        if not isinstance(fields.metadata, Mapping):
            raise ReactiveConfigurationError("Electrical source metadata must be a mapping")
        canonical_metadata = _json_value(fields.metadata)
        if not isinstance(canonical_metadata, dict):
            raise ReactiveConfigurationError(
                "Electrical source metadata must canonicalize to a JSON object"
            )
        metadata = canonical_metadata
        provider = str(metadata.get("provider", "")).strip()
        source = str(metadata.get("source", "")).strip()
        try:
            provenance = Provenance(str(metadata.get("provenance", "")))
        except ValueError as exc:
            raise ReactiveConfigurationError(
                "Electrical source metadata requires a recognized provenance"
            ) from exc
        if not source:
            raise ReactiveConfigurationError(
                "Electrical source metadata requires a nonempty source"
            )
        expected_provider = self.config.electrical_source_provider
        if provider != expected_provider:
            raise ReactiveConfigurationError(
                f"Electrical source provider must identify as {expected_provider}, "
                f"received {provider or '<missing>'}"
            )
        if self.config.strict_paper and provenance is Provenance.ASSUMED_NOT_FROM_PAPER:
            raise ReactiveConfigurationError(
                "Strict paper mode refuses assumed electrical source fields"
            )
        metadata["provenance"] = provenance.value
        metadata["provider"] = provider
        metadata["source"] = source
        metadata["field_sha256"] = {
            "joule_heat_W_per_m3": hashlib.sha256(joule.view(np.uint8)).hexdigest(),
            "electrochemical_heat_W_per_m3": hashlib.sha256(
                electrochemical.view(np.uint8)
            ).hexdigest(),
            "electric_potential_V": hashlib.sha256(potential.view(np.uint8)).hexdigest(),
        }
        metadata["field_shape"] = list(self.shape)
        metadata["field_dtype"] = "float64"
        json_bytes, measured_deep_bytes, resource_charge = (
            _json_resource_charge_bytes(metadata)
        )
        if resource_charge > self.config.maximum_history_bytes:
            raise ReactiveNumericalError(
                "electrical_metadata_record_exceeds_history_limit",
                "One electrical callback metadata record exceeds the configured "
                "result-history resource limit",
                {
                    "canonical_json_bytes": json_bytes,
                    "measured_deep_bytes": measured_deep_bytes,
                    "resource_charge_bytes": resource_charge,
                    "configured_maximum_bytes": self.config.maximum_history_bytes,
                },
            )
        return ElectricalSourceFields(
            joule_heat_W_per_m3=joule,
            electrochemical_heat_W_per_m3=electrochemical,
            electric_potential_V=potential,
            metadata=_deep_freeze_json(metadata),
        )

    def _electrical_stage_record(
        self,
        electrical: ElectricalSourceFields,
        stage_name: str,
        time_s: float,
    ) -> dict[str, Any]:
        """Construct the one canonical metadata record retained for a stage."""

        return {
            **dict(electrical.metadata),
            "stage_name": stage_name,
            "stage_time_s": float(time_s),
            "canonical_surface_geometry_sha256": self.surface_geometry_sha256,
            "provider_authentication_status": self._provider_authentication_status(),
            "geometry_ranking_eligible": False,
        }

    def initial_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        config = self.config
        rho = np.full(self.shape, config.initial_rho_kg_per_m3, dtype=np.float64)
        velocity_x = np.full(self.shape, config.initial_u_m_per_s, dtype=np.float64)
        velocity_y = np.full(self.shape, config.initial_v_m_per_s, dtype=np.float64)
        temperature = np.full(self.shape, config.initial_temperature_K, dtype=np.float64)
        progress = np.full(self.shape, config.initial_reaction_progress, dtype=np.float64)
        conservative = primitive_to_conservative(
            rho,
            velocity_x,
            velocity_y,
            pressure_Pa=None,
            reaction_progress=progress,
            eos=self.eos,
            temperature_K=temperature,
        )
        solid_temperature = temperature.copy()
        solid_alpha = np.full(self.shape, config.initial_alpha, dtype=np.float64)
        x = (np.arange(config.nx, dtype=np.float64) + 0.5) * config.dx_m
        material_level_set = np.broadcast_to(
            x[None, :] - config.interface_x_m, self.shape
        ).copy()
        self._validate_stage(conservative, solid_temperature, solid_alpha, material_level_set, "initial")
        return conservative, solid_temperature, solid_alpha, material_level_set

    def _validate_stage(
        self,
        conservative: np.ndarray,
        solid_temperature: np.ndarray,
        solid_alpha: np.ndarray,
        material_level_set: np.ndarray,
        stage: str,
    ) -> None:
        if conservative.shape != (*self.shape, NCONS):
            raise ReactiveConfigurationError(f"{stage} conservative shape changed")
        valid = physical_state_mask(
            conservative,
            self.eos,
            density_floor_kg_per_m3=self.config.density_reject_below_kg_per_m3,
            pressure_floor_Pa=self.config.pressure_reject_below_Pa,
            temperature_floor_K=self.config.temperature_reject_below_K,
        )
        if not np.all(valid):
            raise ReactiveNumericalError(
                "nonphysical_rk_stage",
                f"{stage} contains {int(valid.size - np.count_nonzero(valid))} invalid Euler cells; no clipping was applied",
            )
        if (
            solid_temperature.shape != self.shape
            or solid_alpha.shape != self.shape
            or material_level_set.shape != self.shape
        ):
            raise ReactiveConfigurationError(f"{stage} auxiliary-state shape changed")
        if not (
            np.isfinite(solid_temperature).all()
            and np.isfinite(solid_alpha).all()
            and np.isfinite(material_level_set).all()
        ):
            raise ReactiveNumericalError(
                "nonfinite_rk_stage", f"{stage} contains a non-finite auxiliary state"
            )
        if np.any(solid_temperature < self.config.temperature_reject_below_K):
            raise ReactiveNumericalError(
                "solid_temperature_below_limit", f"{stage} solid temperature is below the reject limit"
            )
        if np.any(solid_alpha < 0.0) or np.any(solid_alpha > 1.0):
            raise ReactiveNumericalError(
                "solid_alpha_stage_out_of_bounds",
                f"{stage} solid alpha left [0,1]; no clipping was applied",
            )

    def _electrical_at_stage(
        self,
        time_s: float,
        stage_name: str,
        conservative: np.ndarray,
        solid_temperature: np.ndarray,
        solid_alpha: np.ndarray,
        material_level_set: np.ndarray,
    ) -> ElectricalSourceFields:
        if self.prescribed_electrical is not None:
            fields = self.prescribed_electrical
        else:
            assert self.electrical_callback is not None
            request = ElectricalStageRequest(
                time_s=float(time_s),
                stage_name=stage_name,
                conservative_state=_immutable_state(
                    conservative, (*self.shape, NCONS), "callback_conservative_state"
                ),
                solid_temperature_K=_immutable_fp64(
                    solid_temperature, self.shape, "callback_solid_temperature"
                ),
                solid_decomposition_alpha=_immutable_fp64(
                    solid_alpha, self.shape, "callback_solid_alpha"
                ),
                material_level_set_m=_immutable_fp64(
                    material_level_set, self.shape, "callback_material_level_set"
                ),
                dx_m=self.config.dx_m,
                dy_m=self.config.dy_m,
                # SurfaceContactGeometry.build stores every mask on an
                # immutable bytes base.  Passing that canonical snapshot is
                # safer than an owning ndarray copy whose writeable flag a
                # callback could re-enable and mutate after the stamped hash.
                surface_geometry=self.surface_geometry,
            )
            try:
                callback_fields = self.electrical_callback(request)
            except (ReactiveConfigurationError, ReactiveNumericalError):
                raise
            except Exception as exc:
                raise ReactiveNumericalError(
                    "electrical_callback_failed",
                    f"Electrical callback failed at {stage_name}: {type(exc).__name__}: {exc}",
                ) from exc
            fields = self._validate_electrical_fields(callback_fields)
        self._runtime_electrical_source_evaluations += 1
        return fields

    def _stage_derivative(
        self,
        time_s: float,
        stage_name: str,
        dt_s: float,
        conservative: np.ndarray,
        solid_temperature: np.ndarray,
        solid_alpha: np.ndarray,
        material_level_set: np.ndarray,
    ) -> _StageDerivative:
        self._validate_stage(
            conservative, solid_temperature, solid_alpha, material_level_set, stage_name
        )
        self._validate_stage_time_limits(
            dt_s, conservative, solid_temperature, solid_alpha, stage_name
        )
        flux_x_rhs, diag_x = finite_volume_rhs(
            conservative,
            self.config.dx_m,
            self.eos,
            spatial_axis=1,
            direction="x",
            boundary=self.flow_boundary,
            weno_epsilon=self.config.weno_epsilon,
        )
        flux_y_rhs, diag_y = finite_volume_rhs(
            conservative,
            self.config.dy_m,
            self.eos,
            spatial_axis=0,
            direction="y",
            boundary=self.flow_boundary,
            weno_epsilon=self.config.weno_epsilon,
        )
        flow_flux = flux_x_rhs + flux_y_rhs
        primitive = conservative_to_primitive(
            conservative,
            self.eos,
            density_floor_kg_per_m3=self.config.density_reject_below_kg_per_m3,
            pressure_floor_Pa=self.config.pressure_reject_below_Pa,
            temperature_floor_K=self.config.temperature_reject_below_K,
        )
        flow_reaction = reaction_source(
            conservative,
            primitive.temperature_K,
            self.reaction,
            completion_tolerance=self.config.progress_tolerance,
        )
        flow_remaining = 1.0 - primitive.reaction_progress
        flow_terminal_count = int(
            np.count_nonzero(
                (flow_remaining > 0.0)
                & (flow_remaining <= self.config.progress_tolerance)
            )
        )
        solid_remaining = 1.0 - solid_alpha
        solid_terminal_count = int(
            np.count_nonzero(
                (solid_remaining > 0.0)
                & (solid_remaining <= self.config.progress_tolerance)
            )
        )
        electrical = self._electrical_at_stage(
            time_s,
            stage_name,
            conservative,
            solid_temperature,
            solid_alpha,
            material_level_set,
        )
        flow_electrical = np.zeros_like(conservative)
        flow_electrical[..., TOTAL_ENERGY] = electrical.total_heat_W_per_m3
        flow_total = flow_flux + flow_reaction + flow_electrical
        if not np.isfinite(flow_total).all():
            raise ReactiveNumericalError(
                "nonfinite_flow_rhs", f"Flow RHS became non-finite at {stage_name}"
            )

        # Eq. (2) is advanced independently.  The paper does not define how its
        # temperature exchanges energy with Eq. (1), so electrical heat is not
        # duplicated here.  The zero external term is explicit and budgeted.
        solid_temperature_rhs, solid_alpha_rhs, solid_diagnostics = solid_thermal_rhs(
            solid_temperature,
            solid_alpha,
            self.solid_model,
            self.config.dx_m,
            self.config.dy_m,
            np.zeros(self.shape, dtype=np.float64),
            boundary=self.solid_boundary,
            progress_tolerance=self.config.progress_tolerance,
        )
        level_set_rhs = level_set_advection_rhs(
            material_level_set,
            primitive.velocity_x_m_per_s,
            primitive.velocity_y_m_per_s,
            dx_m=self.config.dx_m,
            dy_m=self.config.dy_m,
            boundary=self.level_set_boundary,
            weno_relative_epsilon=self.config.level_set_weno_relative_epsilon,
            weno_absolute_epsilon_m2=self.config.level_set_weno_absolute_epsilon_m2,
        )
        return _StageDerivative(
            flow_total=flow_total,
            flow_flux=flow_flux,
            flow_reaction=flow_reaction,
            flow_electrical=flow_electrical,
            solid_temperature=solid_temperature_rhs,
            solid_alpha=solid_alpha_rhs,
            level_set=level_set_rhs,
            fallback_count=(
                diag_x.positivity_face_fallback_count
                + diag_y.positivity_face_fallback_count
            ),
            fallback_max_abs_state_correction=max(
                diag_x.positivity_face_fallback_max_abs_state_correction,
                diag_y.positivity_face_fallback_max_abs_state_correction,
            ),
            interface_count=diag_x.interface_count + diag_y.interface_count,
            solid_diagnostics=solid_diagnostics,
            electrical_metadata=self._electrical_stage_record(
                electrical, stage_name, time_s
            ),
            flow_completion_tolerance_zeroed_cell_count=flow_terminal_count,
            solid_completion_tolerance_zeroed_cell_count=solid_terminal_count,
        )

    def _time_limits(
        self,
        conservative: np.ndarray,
        solid_temperature: np.ndarray,
        solid_alpha: np.ndarray,
    ) -> tuple[float, float, float, float]:
        flow = cfl_time_step(
            conservative,
            self.eos,
            self.config.dx_m,
            cfl=self.config.cfl,
            dy_m=self.config.dy_m,
        )
        primitive = conservative_to_primitive(conservative, self.eos)
        flow_rate = np.asarray(
            arrhenius_rate(
                primitive.temperature_K,
                primitive.reaction_progress,
                self.reaction,
                completion_tolerance=self.config.progress_tolerance,
            ),
            dtype=np.float64,
        )
        maximum_flow_rate = float(np.max(flow_rate))
        if maximum_flow_rate == 0.0:
            flow_reaction = math.inf
        else:
            increment_limit = _minimum_positive_ratio(
                self.config.maximum_flow_progress_increment,
                maximum_flow_rate,
                name="flow maximum progress increment",
            )
            active = flow_rate > 0.0
            remaining = 1.0 - primitive.reaction_progress
            if np.any(active & (remaining <= 0.0)):
                raise ReactiveNumericalError(
                    "positive_rate_at_complete_progress",
                    "Flow reaction closure returned a positive rate at lambda=1",
                )
            bound_limit = _strict_headroom_limit(
                _minimum_positive_ratio(
                    remaining[active],
                    flow_rate[active],
                    name="flow remaining progress",
                ),
                self.config.flow_progress_headroom,
            )
            flow_reaction = min(increment_limit, bound_limit)
            if self.reaction.heat_release_J_per_kg < 0.0:
                cooling_rate = (
                    -self.reaction.heat_release_J_per_kg
                    / self.eos.cv_J_per_kg_K
                    * flow_rate[active]
                )
                thermal_margin = (
                    np.asarray(primitive.temperature_K)[active]
                    - self.config.temperature_reject_below_K
                )
                thermal_limit = _strict_headroom_limit(
                    _minimum_positive_ratio(
                        thermal_margin,
                        cooling_rate,
                        name="flow cooling temperature headroom",
                    ),
                    self.config.flow_progress_headroom,
                )
                flow_reaction = min(flow_reaction, thermal_limit)
            if math.isnan(flow_reaction) or flow_reaction <= 0.0:
                raise ReactiveNumericalError(
                    "invalid_flow_reaction_timestep",
                    "Flow reaction progress stability limit is invalid",
                )
        solid_rate = solid_arrhenius_rate(
            solid_temperature,
            solid_alpha,
            self.solid_model,
            self.config.progress_tolerance,
        )
        solid = solid_stable_time_step(
            self.solid_model,
            self.config.dx_m,
            self.config.dy_m,
            float(np.max(solid_rate)),
            diffusion_safety=self.config.solid_diffusion_safety,
            maximum_progress_increment=self.config.maximum_solid_progress_increment,
        )
        solid_active = solid_rate > 0.0
        if np.any(solid_active):
            solid_remaining = 1.0 - solid_alpha
            if np.any(solid_active & (solid_remaining <= 0.0)):
                raise ReactiveNumericalError(
                    "positive_rate_at_complete_alpha",
                    "Solid reaction closure returned a positive rate at alpha=1",
                )
            solid_bound_limit = _strict_headroom_limit(
                _minimum_positive_ratio(
                    solid_remaining[solid_active],
                    solid_rate[solid_active],
                    name="solid remaining progress",
                ),
                self.config.solid_progress_headroom,
            )
            solid = min(solid, solid_bound_limit)
            if self.solid_model.heat_of_decomposition_J_per_kg > 0.0:
                solid_cooling_rate = (
                    self.solid_model.heat_of_decomposition_J_per_kg
                    / self.solid_model.heat_capacity_J_per_kgK
                    * solid_rate[solid_active]
                )
                solid_thermal_margin = (
                    solid_temperature[solid_active]
                    - self.config.temperature_reject_below_K
                )
                solid_thermal_limit = _strict_headroom_limit(
                    _minimum_positive_ratio(
                        solid_thermal_margin,
                        solid_cooling_rate,
                        name="solid cooling temperature headroom",
                    ),
                    self.config.solid_progress_headroom,
                )
                solid = min(solid, solid_thermal_limit)
            if math.isnan(solid) or solid <= 0.0:
                raise ReactiveNumericalError(
                    "invalid_solid_reaction_timestep",
                    "Solid reaction progress stability limit is invalid",
                )
        level = recommended_level_set_timestep_s(
            primitive.velocity_x_m_per_s,
            primitive.velocity_y_m_per_s,
            shape=self.shape,
            dx_m=self.config.dx_m,
            dy_m=self.config.dy_m,
            target_cfl=self.config.cfl,
        )
        return flow, flow_reaction, solid, level

    def _validate_stage_time_limits(
        self,
        dt_s: float,
        conservative: np.ndarray,
        solid_temperature: np.ndarray,
        solid_alpha: np.ndarray,
        stage_name: str,
    ) -> None:
        limits = self._time_limits(conservative, solid_temperature, solid_alpha)
        stage_limit = min(limits)
        tolerance = 64.0 * np.finfo(np.float64).eps
        if dt_s > stage_limit * (1.0 + tolerance):
            raise ReactiveNumericalError(
                "rk_stage_stability_violation",
                f"{stage_name} state reduced a stability limit below the proposed dt; "
                "the caller may perform an explicitly counted transactional retry",
                {
                    "dt_s": dt_s,
                    "flow_limit_s": limits[0],
                    "flow_reaction_limit_s": limits[1],
                    "solid_limit_s": limits[2],
                    "level_set_limit_s": limits[3],
                },
            )

    def _ssprk_trial(
        self,
        time_s: float,
        dt_s: float,
        states: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        list[_StageDerivative],
    ]:
        """Evaluate one transaction; no caller-owned state/budget is mutated."""

        derivatives: list[_StageDerivative] = []
        d1 = self._stage_derivative(time_s, STAGE_NAMES[0], dt_s, *states)
        derivatives.append(d1)
        stage_1 = tuple(
            state + dt_s * derivative
            for state, derivative in zip(
                states,
                (d1.flow_total, d1.solid_temperature, d1.solid_alpha, d1.level_set),
            )
        )
        self._validate_stage(*stage_1, STAGE_NAMES[0])

        d2 = self._stage_derivative(
            time_s + dt_s, STAGE_NAMES[1], dt_s, *stage_1
        )
        derivatives.append(d2)
        stage_2 = tuple(
            0.75 * initial + 0.25 * (first + dt_s * derivative)
            for initial, first, derivative in zip(
                states,
                stage_1,
                (d2.flow_total, d2.solid_temperature, d2.solid_alpha, d2.level_set),
            )
        )
        self._validate_stage(*stage_2, STAGE_NAMES[1])

        d3 = self._stage_derivative(
            time_s + 0.5 * dt_s, STAGE_NAMES[2], dt_s, *stage_2
        )
        derivatives.append(d3)
        output = tuple(
            (1.0 / 3.0) * initial
            + (2.0 / 3.0) * (second + dt_s * derivative)
            for initial, second, derivative in zip(
                states,
                stage_2,
                (d3.flow_total, d3.solid_temperature, d3.solid_alpha, d3.level_set),
            )
        )
        self._validate_stage(*output, STAGE_NAMES[2])
        return output, derivatives

    @staticmethod
    def _reduced_retry_timestep(
        current_dt_s: float, failure: ReactiveNumericalError
    ) -> float:
        names = (
            "flow_limit_s",
            "flow_reaction_limit_s",
            "solid_limit_s",
            "level_set_limit_s",
        )
        finite_limits = [
            float(failure.diagnostics[name])
            for name in names
            if name in failure.diagnostics
            and math.isfinite(float(failure.diagnostics[name]))
            and float(failure.diagnostics[name]) > 0.0
        ]
        if not finite_limits:
            raise ReactiveNumericalError(
                "stage_retry_missing_limit",
                "Rejected RK stage did not report a positive finite stability limit",
                failure.diagnostics,
            ) from failure
        stage_limit = min(finite_limits)
        reduced = float(np.nextafter(min(current_dt_s, stage_limit), 0.0))
        if reduced <= 0.0 or not math.isfinite(reduced) or reduced >= current_dt_s:
            raise ReactiveNumericalError(
                "stage_retry_timestep_underflow",
                "Rejected RK stage could not produce a smaller positive FP64 time step",
                {"current_dt_s": current_dt_s, "reported_stage_limit_s": stage_limit},
            ) from failure
        return reduced

    def _accumulate_stage(
        self,
        accumulator: _Accumulator,
        derivative: _StageDerivative,
        weighted_dt_s: float,
    ) -> None:
        cell_area = self.config.dx_m * self.config.dy_m
        integration_scale = weighted_dt_s * cell_area
        accumulator.flux = _finite_accumulate(
            accumulator.flux,
            _scaled_sum(
                derivative.flow_flux,
                integration_scale,
                axis=(0, 1),
                name="Euler flux-divergence budget",
            ),
            "Euler flux-divergence budget",
        )
        accumulator.reaction = _finite_accumulate(
            accumulator.reaction,
            _scaled_sum(
                derivative.flow_reaction,
                integration_scale,
                axis=(0, 1),
                name="Euler reaction budget",
            ),
            "Euler reaction budget",
        )
        accumulator.electrical = _finite_accumulate(
            accumulator.electrical,
            _scaled_sum(
                derivative.flow_electrical,
                integration_scale,
                axis=(0, 1),
                name="Euler electrical budget",
            ),
            "Euler electrical budget",
        )
        diagnostics = derivative.solid_diagnostics
        accumulator.solid_conduction_J_per_m = _finite_accumulate(
            accumulator.solid_conduction_J_per_m,
            weighted_dt_s * diagnostics["conduction_integral_W_per_m"],
            "solid conduction budget",
        )
        accumulator.solid_reaction_J_per_m = _finite_accumulate(
            accumulator.solid_reaction_J_per_m,
            weighted_dt_s * diagnostics["reaction_heat_integral_W_per_m"],
            "solid reaction budget",
        )
        accumulator.solid_external_J_per_m = _finite_accumulate(
            accumulator.solid_external_J_per_m,
            weighted_dt_s * diagnostics["external_heat_integral_W_per_m"],
            "solid external-heat budget",
        )
        accumulator.solid_alpha_area_m2 = _finite_accumulate(
            accumulator.solid_alpha_area_m2,
            _scaled_sum(
                derivative.solid_alpha,
                integration_scale,
                axis=None,
                name="solid alpha budget",
            ),
            "solid alpha budget",
        )
        accumulator.fallback_count += derivative.fallback_count
        accumulator.fallback_max_abs_state_correction = max(
            accumulator.fallback_max_abs_state_correction,
            derivative.fallback_max_abs_state_correction,
        )
        accumulator.interface_count += derivative.interface_count
        accumulator.electrical_evaluations += 1
        accumulator.flow_completion_tolerance_zeroed_cell_evaluations += (
            derivative.flow_completion_tolerance_zeroed_cell_count
        )
        accumulator.solid_completion_tolerance_zeroed_cell_evaluations += (
            derivative.solid_completion_tolerance_zeroed_cell_count
        )
        electrical_record = dict(derivative.electrical_metadata)
        json_bytes, measured_deep_bytes, resource_charge = (
            _json_resource_charge_bytes(electrical_record)
        )
        prospective_charge = (
            accumulator.electrical_source_stage_metadata_resource_charge_bytes
            + resource_charge
        )
        projected_nonmetadata = self._projected_nonmetadata_history_bytes
        if projected_nonmetadata + prospective_charge > self.config.maximum_history_bytes:
            raise ReactiveNumericalError(
                "electrical_metadata_history_limit_exceeded",
                "Accepted electrical stage metadata would exceed the configured "
                "result-history resource limit",
                {
                    "projected_nonmetadata_bytes": projected_nonmetadata,
                    "accepted_stage_metadata_charge_before_bytes": (
                        accumulator.electrical_source_stage_metadata_resource_charge_bytes
                    ),
                    "candidate_record_canonical_json_bytes": json_bytes,
                    "candidate_record_measured_deep_bytes": measured_deep_bytes,
                    "candidate_record_resource_charge_bytes": resource_charge,
                    "configured_maximum_bytes": self.config.maximum_history_bytes,
                },
            )
        accumulator.electrical_source_stage_metadata_json_bytes += json_bytes
        accumulator.electrical_source_stage_metadata_measured_deep_bytes += (
            measured_deep_bytes
        )
        accumulator.electrical_source_stage_metadata_resource_charge_bytes = (
            prospective_charge
        )
        accumulator.electrical_source_stage_records.append(electrical_record)

    def run(self) -> ReactiveResult:
        config = self.config
        history_projection = _projected_history_memory(
            config,
            electrical_stage_metadata_charge_per_record=(
                self._electrical_stage_metadata_preflight_charge_bytes
            ),
            electrical_charge_basis=self._electrical_stage_metadata_preflight_basis,
        )
        self._projected_nonmetadata_history_bytes = int(
            history_projection["projected_without_electrical_stage_metadata_bytes"]
        )
        if (
            int(history_projection["projected_total_bytes"])
            > config.maximum_history_bytes
        ):
            raise ReactiveNumericalError(
                "projected_result_history_exceeds_limit",
                "Configured grid/step/history stride can exceed the explicit result-history memory budget",
                history_projection,
            )
        conservative, solid_temperature, solid_alpha, material_level_set = self.initial_state()
        cell_area = config.dx_m * config.dy_m
        initial_integral = _scaled_sum(
            conservative,
            cell_area,
            axis=(0, 1),
            name="initial Euler integral",
        )
        initial_solid_energy = _scaled_sum(
            solid_temperature,
            self.solid_model.density_kg_per_m3
            * self.solid_model.heat_capacity_J_per_kgK
            * cell_area,
            axis=None,
            name="initial solid sensible energy",
        )
        initial_alpha_integral = _scaled_sum(
            solid_alpha,
            cell_area,
            axis=None,
            name="initial solid alpha integral",
        )
        initial_level_set_area = zero_contour_interior_area_m2(
            material_level_set,
            dx_m=config.dx_m,
            dy_m=config.dy_m,
            boundary=self.level_set_boundary,
        )
        accumulator = _Accumulator(
            flux=np.zeros(NCONS, dtype=np.float64),
            reaction=np.zeros(NCONS, dtype=np.float64),
            electrical=np.zeros(NCONS, dtype=np.float64),
        )
        time_s = 0.0
        diagnostics: list[StepDiagnostics] = []
        step_times: list[float] = []
        step_sizes: list[float] = []
        total_reinitialization_drift = 0.0
        total_reinitialization_sign_changes = 0
        reinitialization_count = 0
        total_stage_retries = 0
        self._runtime_electrical_source_evaluations = 0
        front_frame = axis_aligned_sampling_frame(
            self.shape,
            dx_m=config.dx_m,
            dy_m=config.dy_m,
            grid_first_cell_xy_m=(0.5 * config.dx_m, 0.5 * config.dy_m),
            normal_axis="x",
        )
        front_history: list[dict[str, Any]] = []

        for step_index in range(1, config.maximum_steps + 1):
            if time_s >= config.end_time_s:
                break
            flow_limit, flow_reaction_limit, solid_limit, level_limit = self._time_limits(
                conservative, solid_temperature, solid_alpha
            )
            remaining = config.end_time_s - time_s
            choices = (
                (flow_limit, "flow_cfl"),
                (flow_reaction_limit, "flow_reaction"),
                (solid_limit, "solid_stability"),
                (level_limit, "level_set_cfl"),
                (remaining, "end_time"),
            )
            dt_s, selected = min(choices, key=lambda item: item[0])
            proposed_selected = selected
            if not math.isfinite(dt_s) or dt_s <= 0.0 or time_s + dt_s <= time_s:
                raise ReactiveNumericalError(
                    "invalid_global_timestep", "Adaptive global time step is invalid or underflowed"
                )

            states = (conservative, solid_temperature, solid_alpha, material_level_set)
            previous_progress = (
                conservative[..., REACTION_PROGRESS] / conservative[..., RHO]
            ).copy()
            source_evaluations_before_trial = (
                self._runtime_electrical_source_evaluations
            )
            stage_retry_count = 0
            while True:
                try:
                    output, derivatives = self._ssprk_trial(time_s, dt_s, states)
                    break
                except ReactiveNumericalError as failure:
                    if failure.category != "rk_stage_stability_violation":
                        raise
                    if stage_retry_count >= config.maximum_stage_retries:
                        raise ReactiveNumericalError(
                            "stage_retry_budget_exhausted",
                            "SSPRK stage stability retry budget was exhausted without "
                            "committing a partial state",
                            {
                                "maximum_stage_retries": config.maximum_stage_retries,
                                "last_dt_s": dt_s,
                                "last_failure": failure.diagnostics,
                            },
                        ) from failure
                    dt_s = self._reduced_retry_timestep(dt_s, failure)
                    retry_limits = [
                        (float(failure.diagnostics[name]), name)
                        for name in (
                            "flow_limit_s",
                            "flow_reaction_limit_s",
                            "solid_limit_s",
                            "level_set_limit_s",
                        )
                        if name in failure.diagnostics
                        and math.isfinite(float(failure.diagnostics[name]))
                        and float(failure.diagnostics[name]) > 0.0
                    ]
                    if not retry_limits:
                        # _reduced_retry_timestep already rejects this case;
                        # retain a defensive branch if its contract changes.
                        raise ReactiveNumericalError(
                            "stage_retry_missing_limit",
                            "Retry label could not identify a finite stage limiter",
                        ) from failure
                    retry_name = min(retry_limits, key=lambda item: item[0])[1]
                    selected = "rk_stage_retry_" + retry_name.removesuffix("_limit_s")
                    stage_retry_count += 1
            total_stage_retries += stage_retry_count
            conservative, solid_temperature, solid_alpha, transported_level_set = output
            current_progress = (
                conservative[..., REACTION_PROGRESS] / conservative[..., RHO]
            )
            retain_front_sample = (
                step_index % config.front_history_stride_steps == 0
                or time_s + dt_s >= config.end_time_s
            )
            if retain_front_sample:
                front_sensitivity = evaluate_species_threshold_sensitivity(
                    previous_progress,
                    current_progress,
                    front_frame,
                    elapsed_time_s=dt_s,
                    dx_m=config.dx_m,
                    dy_m=config.dy_m,
                    grid_first_cell_xy_m=(0.5 * config.dx_m, 0.5 * config.dy_m),
                    species_thresholds=config.front_thresholds,
                )
                front_history.append(
                    _front_step_summary(front_sensitivity, time_s + dt_s)
                )

            reinitialized = (
                config.reinitialization_interval_steps > 0
                and step_index % config.reinitialization_interval_steps == 0
            )
            reinit_drift = 0.0
            reinit_relative_drift = 0.0
            reinit_sign_changes = 0
            if reinitialized:
                transported_area = zero_contour_interior_area_m2(
                    transported_level_set,
                    dx_m=config.dx_m,
                    dy_m=config.dy_m,
                    boundary=self.level_set_boundary,
                )
                reinitialized_level_set = reinitialize_transported_level_set(
                    transported_level_set,
                    dx_m=config.dx_m,
                    dy_m=config.dy_m,
                    boundary=self.level_set_boundary,
                    config=self.reinitialization,
                )
                reinitialized_area = zero_contour_interior_area_m2(
                    reinitialized_level_set,
                    dx_m=config.dx_m,
                    dy_m=config.dy_m,
                    boundary=self.level_set_boundary,
                )
                reinit_drift = reinitialized_area - transported_area
                reinit_relative_drift = reinit_drift / max(
                    abs(transported_area), cell_area
                )
                reinit_sign_changes = int(
                    np.count_nonzero(
                        np.signbit(reinitialized_level_set)
                        != np.signbit(transported_level_set)
                    )
                )
                material_level_set = reinitialized_level_set
                total_reinitialization_drift += reinit_drift
                total_reinitialization_sign_changes += reinit_sign_changes
                reinitialization_count += 1
            else:
                material_level_set = transported_level_set

            step_fallback = 0
            step_fallback_max_correction = 0.0
            step_interfaces = 0
            step_electrical = 0
            for weight, derivative in zip(RK_WEIGHTS, derivatives):
                self._accumulate_stage(accumulator, derivative, weight * dt_s)
                step_fallback += derivative.fallback_count
                step_fallback_max_correction = max(
                    step_fallback_max_correction,
                    derivative.fallback_max_abs_state_correction,
                )
                step_interfaces += derivative.interface_count
                step_electrical += 1
            rejected_source_evaluations = (
                self._runtime_electrical_source_evaluations
                - source_evaluations_before_trial
                - step_electrical
            )
            if rejected_source_evaluations < 0:
                raise ReactiveNumericalError(
                    "invalid_source_evaluation_accounting",
                    "Electrical source evaluation accounting became negative",
                )

            start_time = time_s
            time_s = min(config.end_time_s, time_s + dt_s)
            step_times.append(time_s)
            step_sizes.append(dt_s)
            diagnostics.append(
                StepDiagnostics(
                    step=step_index,
                    start_time_s=start_time,
                    dt_s=dt_s,
                    proposed_time_step_limit=proposed_selected,
                    selected_time_step_limit=selected,
                    flow_cfl_limit_s=flow_limit,
                    flow_reaction_limit_s=flow_reaction_limit,
                    solid_stability_limit_s=solid_limit,
                    level_set_cfl_limit_s=level_limit,
                    positivity_face_fallback_count=step_fallback,
                    positivity_face_fallback_max_abs_state_correction=(
                        step_fallback_max_correction
                    ),
                    reconstructed_face_count=step_interfaces,
                    electrical_source_evaluation_count=step_electrical,
                    rejected_trial_electrical_source_evaluation_count=(
                        rejected_source_evaluations
                    ),
                    stage_retry_count=stage_retry_count,
                    material_level_set_reinitialized=reinitialized,
                    reinitialization_sign_changed_cell_count=reinit_sign_changes,
                    reinitialization_area_drift_m2=reinit_drift,
                    reinitialization_relative_area_drift=reinit_relative_drift,
                )
            )

        if time_s < config.end_time_s:
            raise ReactiveNumericalError(
                "maximum_steps_exhausted",
                f"Reached maximum_steps={config.maximum_steps} at t={time_s:.17g} s before end time",
            )

        final_integral = _scaled_sum(
            conservative,
            cell_area,
            axis=(0, 1),
            name="final Euler integral",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            closure = (
                final_integral
                - initial_integral
                - accumulator.flux
                - accumulator.reaction
                - accumulator.electrical
            )
        if not np.isfinite(closure).all():
            raise ReactiveNumericalError(
                "nonfinite_conservation_closure",
                "Euler conservation closure became non-finite",
            )
        reaction_progress_increment = float(accumulator.reaction[REACTION_PROGRESS])
        reaction_heat_increment = float(accumulator.reaction[TOTAL_ENERGY])
        with np.errstate(over="ignore", invalid="ignore"):
            reaction_bookkeeping_residual = (
                reaction_heat_increment
                - config.reaction_heat_J_per_kg * reaction_progress_increment
            )
        if not math.isfinite(reaction_bookkeeping_residual):
            raise ReactiveNumericalError(
                "nonfinite_reaction_bookkeeping",
                "Reaction heat/progress bookkeeping became non-finite",
            )
        conservation = ConservationBudget(
            component_names=("rho", "rho_u", "rho_v", "rho_E", "rho_lambda"),
            initial_integral=tuple(float(value) for value in initial_integral),
            final_integral=tuple(float(value) for value in final_integral),
            flux_divergence_increment=tuple(float(value) for value in accumulator.flux),
            reaction_increment=tuple(float(value) for value in accumulator.reaction),
            electrical_increment=tuple(float(value) for value in accumulator.electrical),
            closure_residual=tuple(float(value) for value in closure),
            reaction_progress_increment=reaction_progress_increment,
            reaction_heat_increment_J_per_m=reaction_heat_increment,
            configured_heat_per_progress_J_per_kg=config.reaction_heat_J_per_kg,
            reaction_bookkeeping_residual_J_per_m=reaction_bookkeeping_residual,
            electrical_energy_increment_J_per_m=float(
                accumulator.electrical[TOTAL_ENERGY]
            ),
        )

        final_solid_energy = _scaled_sum(
            solid_temperature,
            self.solid_model.density_kg_per_m3
            * self.solid_model.heat_capacity_J_per_kgK
            * cell_area,
            axis=None,
            name="final solid sensible energy",
        )
        final_alpha_integral = _scaled_sum(
            solid_alpha,
            cell_area,
            axis=None,
            name="final solid alpha integral",
        )
        solid_energy_closure = (
            final_solid_energy
            - initial_solid_energy
            - accumulator.solid_conduction_J_per_m
            - accumulator.solid_reaction_J_per_m
            - accumulator.solid_external_J_per_m
        )
        solid_alpha_closure = (
            final_alpha_integral
            - initial_alpha_integral
            - accumulator.solid_alpha_area_m2
        )
        if not math.isfinite(solid_energy_closure) or not math.isfinite(
            solid_alpha_closure
        ):
            raise ReactiveNumericalError(
                "nonfinite_solid_closure", "Solid conservation closure became non-finite"
            )
        solid_budget = SolidBudget(
            initial_sensible_energy_J_per_m=initial_solid_energy,
            final_sensible_energy_J_per_m=final_solid_energy,
            conduction_increment_J_per_m=accumulator.solid_conduction_J_per_m,
            decomposition_heat_increment_J_per_m=accumulator.solid_reaction_J_per_m,
            externally_coupled_heat_increment_J_per_m=accumulator.solid_external_J_per_m,
            closure_residual_J_per_m=solid_energy_closure,
            initial_alpha_area_integral_m2=initial_alpha_integral,
            final_alpha_area_integral_m2=final_alpha_integral,
            rate_increment_area_integral_m2=accumulator.solid_alpha_area_m2,
            alpha_closure_residual_m2=solid_alpha_closure,
        )
        final_level_set_area = zero_contour_interior_area_m2(
            material_level_set,
            dx_m=config.dx_m,
            dy_m=config.dy_m,
            boundary=self.level_set_boundary,
        )
        final_flow_progress = conservative_to_primitive(
            conservative,
            self.eos,
            density_floor_kg_per_m3=config.density_reject_below_kg_per_m3,
            pressure_floor_Pa=config.pressure_reject_below_Pa,
            temperature_floor_K=config.temperature_reject_below_K,
        ).reaction_progress
        flow_terminal_summary = _terminal_remainder_summary(
            final_flow_progress, config.progress_tolerance
        )
        solid_terminal_summary = _terminal_remainder_summary(
            solid_alpha, config.progress_tolerance
        )
        final_flow_remaining = 1.0 - final_flow_progress
        final_flow_terminal_mask = (final_flow_remaining > 0.0) & (
            final_flow_remaining <= config.progress_tolerance
        )
        flow_suppressed_inventory = _scaled_sum(
            np.where(
                final_flow_terminal_mask,
                conservative[..., RHO] * final_flow_remaining,
                0.0,
            ),
            cell_area,
            axis=None,
            name="flow completion-tolerance suppressed inventory",
        )
        final_solid_remaining = 1.0 - solid_alpha
        final_solid_terminal_mask = (final_solid_remaining > 0.0) & (
            final_solid_remaining <= config.progress_tolerance
        )
        solid_suppressed_inventory = _scaled_sum(
            np.where(final_solid_terminal_mask, final_solid_remaining, 0.0),
            self.solid_model.density_kg_per_m3 * cell_area,
            axis=None,
            name="solid completion-tolerance suppressed inventory",
        )
        flow_terminal_summary.update(
            {
                "accepted_stage_zeroed_cell_evaluation_count": (
                    accumulator.flow_completion_tolerance_zeroed_cell_evaluations
                ),
                "final_in_domain_terminal_remainder_inventory_kg_per_m": (
                    flow_suppressed_inventory
                ),
                "final_in_domain_terminal_remainder_heat_magnitude_J_per_m": (
                    abs(config.reaction_heat_J_per_kg)
                    * flow_suppressed_inventory
                ),
            }
        )
        solid_terminal_summary.update(
            {
                "accepted_stage_zeroed_cell_evaluation_count": (
                    accumulator.solid_completion_tolerance_zeroed_cell_evaluations
                ),
                "final_in_domain_terminal_remainder_inventory_kg_per_m": (
                    solid_suppressed_inventory
                ),
                "final_in_domain_terminal_remainder_heat_magnitude_J_per_m": (
                    abs(config.decomposition_heat_J_per_kg)
                    * solid_suppressed_inventory
                ),
            }
        )
        metadata: dict[str, Any] = {
            "solver": "CPU_FP64_WENO5_JS_HLL_SSPRK33_REFERENCE",
            "model_mode": config.model_mode,
            "source_config_identifier": (
                config.source_path
                if config.source_path.startswith("<")
                else Path(config.source_path).name
            ),
            "effective_configuration_sha256": self.effective_configuration_sha256,
            "source_configuration_provenance": config.provenance_report,
            "runtime_configuration_provenance_audit": (
                self.runtime_configuration_audit
            ),
            "solid_eq2_coupling_status": (
                "SEPARATE_DIAGNOSTIC_STATE_NO_PUBLISHED_EQ1_EQ2_INTERFACE_CLOSURE"
            ),
            "electrical_heat_destination": "EULER_EQ1_ONLY_NO_DUPLICATION_IN_SOLID_EQ2",
            "electrical_source_provider": config.electrical_source_provider,
            "electrical_source_evaluation_count": accumulator.electrical_evaluations,
            "electrical_source_evaluation_count_including_rejected_trials": (
                self._runtime_electrical_source_evaluations
            ),
            "rejected_trial_electrical_source_evaluation_count": (
                self._runtime_electrical_source_evaluations
                - accumulator.electrical_evaluations
            ),
            "electrical_callback_evaluation_count": (
                self._runtime_electrical_source_evaluations
                if self.electrical_callback is not None
                else 0
            ),
            "accepted_step_electrical_callback_evaluation_count": (
                accumulator.electrical_evaluations
                if self.electrical_callback is not None
                else 0
            ),
            "electrical_source_stage_records": (
                accumulator.electrical_source_stage_records
            ),
            "electrical_source_stage_metadata_resource_accounting": {
                "canonical_json_bytes": (
                    accumulator.electrical_source_stage_metadata_json_bytes
                ),
                "measured_deep_bytes": (
                    accumulator.electrical_source_stage_metadata_measured_deep_bytes
                ),
                "resource_charge_bytes": (
                    accumulator.electrical_source_stage_metadata_resource_charge_bytes
                ),
                "charge_formula": (
                    "MEASURED_RECURSIVE_PYTHON_OBJECT_SIZE_PLUS_COMPACT_UTF8_JSON_"
                    "BYTES_PER_RECORD"
                ),
                "scope_limitation": (
                    "PROCESS_INTERPRETER_OBJECT_MEASUREMENT_NOT_OS_RSS_OR_ALLOCATOR_"
                    "FRAGMENTATION_GUARANTEE"
                ),
                "runtime_limit_enforced": True,
            },
            "consumed_electrical_inputs": self.consumed_electrical_inputs,
            "provider_authentication_status": (
                self._provider_authentication_status()
            ),
            "canonical_surface_geometry_sha256": self.surface_geometry_sha256,
            "surface_geometry_passed_to_callback": (
                config.model_mode == "ecsp_extended"
                and self.surface_geometry is not None
            ),
            "geometry_ranking_eligible": False,
            "geometry_ranking_gate": (
                "HARD_GATED_UNTIL_CONCRETE_BC_ADAPTER_ENFORCES_CONVERGENCE_"
                "CURRENT_BALANCE_AND_BV_RESIDUALS"
            ),
            "electrical_callback_retry_contract": (
                "CALLBACK_MUST_BE_PURE_AND_DETERMINISTIC;REJECTED_TRIAL_CALLS_HAVE_"
                "NO_SOLVER_STATE_COMMIT_BUT_EXTERNAL_CALLBACK_SIDE_EFFECTS_CANNOT_BE_ROLLED_BACK"
                if self.electrical_callback is not None
                else "NOT_APPLICABLE_TO_PRESCRIBED_FIELDS"
            ),
            "surface_contact_model": (
                "overlay_on_full_propellant_domain"
                if self.surface_geometry is not None
                else "no_surface_masks_supplied"
            ),
            "propellant_cell_count": (
                int(np.count_nonzero(self.surface_geometry.propellant))
                if self.surface_geometry is not None
                else config.nx * config.ny
            ),
            "anode_contact_cell_count": (
                int(np.count_nonzero(self.surface_geometry.anode_contact))
                if self.surface_geometry is not None
                else 0
            ),
            "cathode_contact_cell_count": (
                int(np.count_nonzero(self.surface_geometry.cathode_contact))
                if self.surface_geometry is not None
                else 0
            ),
            "positivity_face_fallback_count": accumulator.fallback_count,
            "positivity_face_fallback_max_abs_state_correction": (
                accumulator.fallback_max_abs_state_correction
            ),
            "positivity_face_fallback_correction_metric": (
                "MAX_ABSOLUTE_RAW_CONSERVATIVE_COMPONENT_CORRECTION;COMPONENTS_"
                "HAVE_MIXED_PHYSICAL_UNITS;NOT_A_DIMENSIONLESS_NORM"
            ),
            "cell_state_clipping_count": 0,
            "density_pressure_temperature_floor_application_count": 0,
            "reaction_rate_cap_application_count": 0,
            "completion_tolerance_terminal_policy": {
                "completion_tolerance": config.progress_tolerance,
                "provenance": "ASSUMED_NOT_FROM_PAPER",
                "exact_chemistry": False,
                "deliberate_terminal_cutoff": True,
                "semantics": (
                    "RATE_SET_EXACTLY_ZERO_WHEN_POSITIVE_UNREACTED_REMAINDER_"
                    "IS_AT_MOST_TOLERANCE;STORED_PROGRESS_IS_NOT_CHANGED"
                ),
                "classification": (
                    "EXPLICIT_NUMERICAL_TERMINAL_SEMANTICS_NOT_STATE_CLIPPING_"
                    "AND_NOT_A_MAGNITUDE_RATE_CAP"
                ),
                "heat_progress_bookkeeping": (
                    "BOTH_HEAT_AND_PROGRESS_SOURCES_USE_THE_SAME_TERMINATED_RATE"
                ),
                "accepted_stage_counting_scope": (
                    "ONLY_ACCEPTED_SSPRK_STAGE_CELL_EVALUATIONS_ARE_ACCUMULATED;"
                    "REJECTED_TRIAL_TERMINAL_ZEROING_COUNTS_ARE_NOT_RETAINED"
                ),
                "final_remainder_scope": (
                    "FINAL_IN_DOMAIN_TERMINAL_REMAINDER_ONLY;NOT_A_CUMULATIVE_"
                    "SUPPRESSION_BOUND_UNDER_ADVECTION_OR_OUTFLOW"
                ),
                "flow_final": flow_terminal_summary,
                "solid_final": solid_terminal_summary,
            },
            "completion_tolerance_rate_zeroing_flow": flow_terminal_summary,
            "completion_tolerance_rate_zeroing_solid": solid_terminal_summary,
            "stage_retry_count": total_stage_retries,
            "maximum_stage_retries_per_step": config.maximum_stage_retries,
            "reconstructed_face_count": accumulator.interface_count,
            "level_set_sign_convention": "phi<0 material interior; phi>0 exterior",
            "level_set_initial_interior_area_m2": initial_level_set_area,
            "level_set_final_interior_area_m2": final_level_set_area,
            "level_set_reinitialization_count": reinitialization_count,
            "level_set_reinitialization_configuration": {
                "enabled": self.reinitialization.enabled,
                "method": self.reinitialization.method,
                "interval_steps": config.reinitialization_interval_steps,
                "pseudo_steps": (
                    self.reinitialization.pseudo_steps
                    if self.reinitialization.enabled
                    else None
                ),
                "pseudo_cfl": (
                    self.reinitialization.pseudo_cfl
                    if self.reinitialization.enabled
                    else None
                ),
                "narrow_band_half_width_m": (
                    self.reinitialization.narrow_band_half_width_m
                    if self.reinitialization.enabled
                    else None
                ),
                "provenance": self.reinitialization.provenance.value,
                "source": self.reinitialization.source,
                "parameter_origin": (
                    "EXPLICIT_REQUIRED_SOLVER_ARGUMENT"
                    if self.reinitialization.enabled
                    else "CANONICAL_DISABLED_CONFIGURATION"
                ),
            },
            "level_set_total_reinitialization_area_drift_m2": total_reinitialization_drift,
            "level_set_total_reinitialization_sign_changes": total_reinitialization_sign_changes,
            "level_set_final_signed_distance_rms_error": signed_distance_rms_error(
                material_level_set,
                dx_m=config.dx_m,
                dy_m=config.dy_m,
                boundary=self.level_set_boundary,
            ),
            "species_front_tracking": {
                "status": "computed_from_reaction_progress_not_material_level_set",
                "ranking_status": (
                    "DIAGNOSTIC_FIXED_POSITIVE_X_RAYS_ONLY_NOT_TRUE_LOCAL_NORMAL_SPEED"
                ),
                "legacy_area_over_front_length_used": False,
                "thresholds": list(config.front_thresholds),
                "threshold_provenance": "ASSUMED_NOT_FROM_PAPER",
                "history_stride_steps": config.front_history_stride_steps,
                "retained_sample_count": len(front_history),
                "accepted_step_count": len(diagnostics),
                "history_thinned": config.front_history_stride_steps > 1,
                "history": front_history,
            },
            "result_history_resource_preflight": history_projection,
            "working_set_resource_preflight": self.working_set_projection,
            "finite_conservation_budget_policy": {
                "status": "DIAGNOSTIC_ONLY_NOT_ENFORCED",
                "reason": (
                    "NO_PAPER_OR_CALIBRATED_ABSOLUTE_AND_SCALE_AWARE_RESIDUAL_"
                    "TOLERANCES_ARE_AVAILABLE"
                ),
                "nonfinite_residuals": "FAIL_CLOSED",
                "finite_residuals": "RECORDED_WITHOUT_PASS_FAIL_CLASSIFICATION",
                "geometry_ranking_eligible": False,
            },
            "stage_failure_policy": (
                "TRANSACTIONAL_RETRY_TO_REPORTED_STAGE_LIMIT;_NO_PARTIAL_COMMIT;"
                "FAIL_CLOSED_ON_EXHAUSTION;NO_PHYSICAL_CLIPPING"
            ),
            "flow_progress_time_step_policy": {
                "maximum_progress_increment": config.maximum_flow_progress_increment,
                "remaining_progress_headroom": config.flow_progress_headroom,
                "provenance": "ASSUMED_NOT_FROM_PAPER",
            },
            "solid_time_step_policy": {
                "maximum_progress_increment": config.maximum_solid_progress_increment,
                "remaining_progress_headroom": config.solid_progress_headroom,
                "diffusion_safety": config.solid_diffusion_safety,
            },
            "weno_epsilon": config.weno_epsilon,
            "level_set_weno_relative_epsilon": config.level_set_weno_relative_epsilon,
            "level_set_weno_absolute_epsilon_m2": (
                config.level_set_weno_absolute_epsilon_m2
            ),
            "minimum_pressure_reject_Pa": config.pressure_reject_below_Pa,
            "paper_reproduction_verdict": "NOT_FULLY_VERIFIED_MISSING_PAPER_INPUTS",
        }
        return ReactiveResult(
            conservative_state=conservative,
            solid_temperature_K=solid_temperature,
            solid_decomposition_alpha=solid_alpha,
            material_level_set_m=material_level_set,
            final_time_s=time_s,
            accepted_steps=len(diagnostics),
            step_time_s=np.asarray(step_times, dtype=np.float64),
            step_dt_s=np.asarray(step_sizes, dtype=np.float64),
            steps=tuple(diagnostics),
            conservation=conservation,
            solid_budget=solid_budget,
            metadata=metadata,
        )


def run_reactive_case(
    config: ReactiveCaseConfig | str | Path,
    **solver_options: Any,
) -> ReactiveResult:
    validated = load_reactive_config(config) if isinstance(config, (str, Path)) else config
    return ReactiveSolver(validated, **solver_options).run()


__all__ = [
    "ConservationBudget",
    "ElectricalSourceCallback",
    "ElectricalStageRequest",
    "ReactiveResult",
    "ReactiveSolver",
    "SolidBudget",
    "StepDiagnostics",
    "run_reactive_case",
]
