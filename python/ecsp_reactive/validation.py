"""Executable verification matrix for the experimental reactive solver.

The checks in this module are verification problems with analytic or discrete
reference answers.  They do *not* reproduce the ECSP experiment reported by
Park & Yoh (2024): the paper omits material constants, kinetic coefficients,
measured electrical fields, and several numerical choices required for that
calculation.  The report contract keeps those two statements separate so a
passing manufactured test can never be presented as paper validation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Callable

import numpy as np

from .core import (
    MOMENTUM_X,
    MOMENTUM_Y,
    REACTION_PROGRESS,
    RHO,
    TOTAL_ENERGY,
    conservative_to_primitive,
    primitive_to_conservative,
)
from .configuration import ReactiveCaseConfig, load_reactive_config
from .electrical import ElectricalSourceFields, joule_heating_sigma_e2
from .eos import IdealGasEOS, TaitEOS
from .front import (
    ASSUMED_SPECIES_THRESHOLD_SET,
    axis_aligned_sampling_frame,
    evaluate_species_threshold_sensitivity,
)
from .level_set import advect_material_level_set
from .numerics import finite_volume_rhs, ssprk3_step, weno5_js_reconstruct
from .parallel import add_outflow_row_halos, all_row_partitions
from .reaction import ArrheniusReaction, reaction_source
from .solver import ReactiveSolver
from .solid import laplacian_2d


VALIDATION_SCHEMA = "ecsp.reactive-validation/v1"
PASS = "PASS"
FAIL = "FAIL"
FULL_VERIFICATION_VERDICT = "NOT_FULLY_VERIFIED"


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _observed_orders(errors: list[float], refinement_ratio: float = 2.0) -> list[float]:
    _require(len(errors) >= 3, "at least three error levels are required")
    _require(all(math.isfinite(value) and value > 0.0 for value in errors), "errors must be finite and positive")
    return [
        float(math.log(errors[index] / errors[index + 1], refinement_ratio))
        for index in range(len(errors) - 1)
    ]


def _tait_identity_check() -> dict[str, Any]:
    eos = TaitEOS(
        rho0_kg_per_m3=1000.0,
        B_Pa=3.0e8,
        N=7.0,
        A_Pa=3.0e8,
        cv_J_per_kg_K=2000.0,
        T_ref_K=300.0,
        e_ref_J_per_kg=0.0,
    )
    densities = np.array([1000.0, 1002.0, 1010.0], dtype=np.float64)
    relative_errors: list[float] = []
    for density in densities:
        step = density * 1.0e-5
        derivative = (
            float(eos.pressure_from_density(density + step))
            - float(eos.pressure_from_density(density - step))
        ) / (2.0 * step)
        sound_speed_squared = float(eos.sound_speed(density)) ** 2
        relative_errors.append(abs(derivative - sound_speed_squared) / sound_speed_squared)
    maximum_error = float(max(relative_errors))
    tolerance = 2.0e-8
    _require(maximum_error <= tolerance, "Tait finite-difference dp/drho does not equal c^2")
    return {
        "description": "Paper Eq. (3) mechanical identity dp/drho = c^2",
        "metrics": {
            "densities_kg_per_m3": densities.tolist(),
            "maximum_relative_error": maximum_error,
        },
        "criteria": {"maximum_relative_error_lte": tolerance},
    }


def _primitive_roundtrip_check() -> dict[str, Any]:
    gas = IdealGasEOS(gamma=1.4, gas_constant_J_per_kg_K=287.05)
    density = np.array([0.9, 1.2, 1.8], dtype=np.float64)
    velocity_x = np.array([-2.0, 0.0, 4.0], dtype=np.float64)
    velocity_y = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    pressure = np.array([8.0e4, 1.01325e5, 1.4e5], dtype=np.float64)
    progress = np.array([0.0, 0.35, 0.9], dtype=np.float64)
    state = primitive_to_conservative(
        density, velocity_x, velocity_y, pressure, progress, gas
    )
    recovered = conservative_to_primitive(state, gas)
    fields = (
        (density, recovered.density_kg_per_m3),
        (velocity_x, recovered.velocity_x_m_per_s),
        (velocity_y, recovered.velocity_y_m_per_s),
        (pressure, recovered.pressure_Pa),
        (progress, recovered.reaction_progress),
    )
    relative_scale = np.maximum(
        1.0,
        np.concatenate([np.abs(reference).ravel() for reference, _ in fields]),
    )
    absolute_difference = np.concatenate(
        [np.abs(reference - actual).ravel() for reference, actual in fields]
    )
    maximum_scaled_error = float(np.max(absolute_difference / relative_scale))

    tait = TaitEOS(1000.0, 3.0e8, 7.0, 3.0e8, 1800.0, 300.0, 20.0)
    tait_density = np.array([1000.0, 1001.0], dtype=np.float64)
    tait_temperature = np.array([310.0, 750.0], dtype=np.float64)
    tait_pressure = np.asarray(tait.pressure_from_density(tait_density))
    tait_state = primitive_to_conservative(
        tait_density,
        np.array([0.0, 0.1]),
        0.0,
        tait_pressure,
        np.array([0.1, 0.6]),
        tait,
        temperature_K=tait_temperature,
    )
    recovered_tait = conservative_to_primitive(tait_state, tait)
    tait_temperature_error = float(
        np.max(np.abs(recovered_tait.temperature_K - tait_temperature))
    )
    tolerance = 2.0e-11
    _require(maximum_scaled_error <= tolerance, "ideal-gas primitive roundtrip failed")
    _require(tait_temperature_error <= tolerance, "Tait caloric roundtrip failed")
    return {
        "description": "Primitive/conservative FP64 roundtrip for verification EOS closures",
        "metrics": {
            "ideal_gas_maximum_scaled_error": maximum_scaled_error,
            "tait_maximum_temperature_error_K": tait_temperature_error,
        },
        "criteria": {"both_errors_lte": tolerance},
    }


def _free_stream_conservation_check() -> dict[str, Any]:
    eos = IdealGasEOS(gamma=1.4, gas_constant_J_per_kg_K=287.05)
    ny, nx = 18, 20
    dx, dy = 1.0 / nx, 1.0 / ny
    one_state = primitive_to_conservative(1.2, 10.0, -3.0, 101325.0, 0.2, eos)
    uniform = np.broadcast_to(one_state, (ny, nx, 5)).copy()
    rhs_x, diagnostics_x = finite_volume_rhs(
        uniform, dx, eos, spatial_axis=1, direction="x", boundary="periodic"
    )
    rhs_y, diagnostics_y = finite_volume_rhs(
        uniform, dy, eos, spatial_axis=0, direction="y", boundary="periodic"
    )
    free_stream_maximum = float(np.max(np.abs(rhs_x + rhs_y)))

    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    x_grid, y_grid = np.meshgrid(x, y)
    density = 1.2 + 0.05 * np.sin(2.0 * np.pi * x_grid) * np.cos(2.0 * np.pi * y_grid)
    velocity_x = 10.0 + 0.2 * np.cos(2.0 * np.pi * x_grid)
    velocity_y = -3.0 + 0.1 * np.sin(2.0 * np.pi * y_grid)
    pressure = 101325.0 + 100.0 * np.sin(2.0 * np.pi * (x_grid + y_grid))
    progress = 0.2 + 0.02 * np.cos(2.0 * np.pi * x_grid)
    smooth = primitive_to_conservative(
        density, velocity_x, velocity_y, pressure, progress, eos
    )
    smooth_x, _ = finite_volume_rhs(
        smooth, dx, eos, spatial_axis=1, direction="x", boundary="periodic"
    )
    smooth_y, _ = finite_volume_rhs(
        smooth, dy, eos, spatial_axis=0, direction="y", boundary="periodic"
    )
    smooth_rhs = smooth_x + smooth_y
    integrated = np.sum(smooth_rhs, axis=(0, 1)) * dx * dy
    component_scale = np.sum(np.abs(smooth_rhs), axis=(0, 1)) * dx * dy
    normalized = np.abs(integrated) / np.maximum(component_scale, 1.0)
    major_indices = (RHO, MOMENTUM_X, MOMENTUM_Y, TOTAL_ENERGY)
    maximum_normalized_major_residual = float(np.max(normalized[list(major_indices)]))
    tolerance = 2.0e-13
    _require(free_stream_maximum <= 1.0e-13, "uniform free stream was not preserved")
    _require(
        maximum_normalized_major_residual <= tolerance,
        "periodic mass/momentum/energy residual does not telescope",
    )
    _require(
        diagnostics_x.positivity_face_fallback_count == 0
        and diagnostics_y.positivity_face_fallback_count == 0,
        "uniform state unexpectedly activated positivity fallback",
    )
    return {
        "description": "Free-stream preservation and periodic global conservation",
        "metrics": {
            "free_stream_maximum_abs_rhs": free_stream_maximum,
            "integrated_residual_by_conserved_component": integrated.tolist(),
            "normalized_residual_by_conserved_component": normalized.tolist(),
            "maximum_normalized_mass_momentum_energy_residual": maximum_normalized_major_residual,
        },
        "criteria": {
            "free_stream_maximum_abs_rhs_lte": 1.0e-13,
            "normalized_major_residual_lte": tolerance,
        },
    }


def _reaction_bookkeeping_check() -> dict[str, Any]:
    eos = IdealGasEOS(gamma=1.4, gas_constant_J_per_kg_K=287.05)
    model = ArrheniusReaction(
        pre_exponential_s_inv=2.0e3,
        activation_energy_J_per_mol=4.0e4,
        reaction_order=1.5,
        heat_release_J_per_kg=3.0e6,
    )
    progress = np.array([0.2, 0.7], dtype=np.float64)
    state = primitive_to_conservative(
        np.array([1.0, 0.8]), 0.0, 0.0, np.array([1.0e5, 9.0e4]), progress, eos
    )
    source = reaction_source(state, np.array([900.0, 1200.0]), model)
    active = source[..., REACTION_PROGRESS] > 0.0
    ratio = source[..., TOTAL_ENERGY][active] / source[..., REACTION_PROGRESS][active]
    maximum_relative_error = float(
        np.max(np.abs(ratio - model.heat_release_J_per_kg))
        / model.heat_release_J_per_kg
    )
    tolerance = 3.0e-15
    _require(maximum_relative_error <= tolerance, "reaction heat/progress ratio is inconsistent")
    return {
        "description": "Eq. (1) reaction heat and progress source bookkeeping",
        "metrics": {
            "configured_heat_release_J_per_kg": model.heat_release_J_per_kg,
            "source_ratio_J_per_kg": ratio.tolist(),
            "maximum_relative_error": maximum_relative_error,
        },
        "criteria": {"maximum_relative_error_lte": tolerance},
    }


def _joule_linear_field_check() -> dict[str, Any]:
    ny, nx = 7, 9
    dx, dy = 2.0e-4, 3.0e-4
    sigma = 3.2
    electric_field_x = 2.5e4
    electric_field_y = -1.2e4
    x = np.arange(nx, dtype=np.float64) * dx
    y = np.arange(ny, dtype=np.float64) * dy
    x_grid, y_grid = np.meshgrid(x, y)
    potential = -(electric_field_x * x_grid + electric_field_y * y_grid)
    heat, diagnostics = joule_heating_sigma_e2(
        potential,
        sigma,
        dx,
        dy,
        gradient_boundary_scheme="FIRST_ORDER_ONE_SIDED",
    )
    expected_density = sigma * (electric_field_x**2 + electric_field_y**2)
    expected_integral = expected_density * nx * ny * dx * dy
    maximum_relative_error = float(np.max(np.abs(heat - expected_density)) / expected_density)
    integral_relative_error = abs(
        diagnostics["joule_heat_integral_per_depth_W_per_m"] - expected_integral
    ) / expected_integral
    tolerance = 2.0e-14
    _require(maximum_relative_error <= tolerance, "linear-potential Joule density is wrong")
    _require(integral_relative_error <= tolerance, "linear-potential Joule integral is wrong")
    return {
        "description": "Paper Joule source sigma*|grad(V)|^2 for a linear potential",
        "metrics": {
            "expected_heat_density_W_per_m3": expected_density,
            "maximum_relative_error": maximum_relative_error,
            "integral_relative_error": integral_relative_error,
        },
        "criteria": {"both_relative_errors_lte": tolerance},
    }


def _periodic_thermal_integral_check() -> dict[str, Any]:
    rng = np.random.default_rng(71031)
    field = rng.normal(size=(17, 19))
    dx, dy = 0.2, 0.3
    laplacian = laplacian_2d(field, dx, dy, "PERIODIC")
    integral = float(np.sum(laplacian) * dx * dy)
    l1_integral = float(np.sum(np.abs(laplacian)) * dx * dy)
    normalized = abs(integral) / max(l1_integral, 1.0)
    tolerance = 2.0e-15
    _require(normalized <= tolerance, "periodic thermal conduction has a nonzero global integral")
    return {
        "description": "Discrete periodic thermal Laplacian conserves the field integral",
        "metrics": {
            "signed_laplacian_integral": integral,
            "l1_laplacian_integral": l1_integral,
            "normalized_integral_residual": normalized,
        },
        "criteria": {"normalized_integral_residual_lte": tolerance},
    }


def _level_set_translation_check() -> dict[str, Any]:
    # A periodic smooth field exercises every cell.  This avoids making a
    # false pass by discarding the outflow-boundary stencil region.
    count = 96
    spacing = 1.0 / count
    coordinates = np.arange(count, dtype=np.float64) * spacing
    x_grid, _ = np.meshgrid(coordinates, coordinates)
    phase = 0.2
    initial = np.sin(2.0 * np.pi * (x_grid - phase))
    velocity = 0.7
    dt = 0.4 * spacing / velocity
    result = advect_material_level_set(
        initial,
        velocity,
        0.0,
        dt_s=dt,
        dx_m=spacing,
        boundary="periodic",
    )
    expected = np.sin(2.0 * np.pi * (x_grid - phase - velocity * dt))
    full_domain_rms_error = float(
        np.sqrt(np.mean((result.material_level_set.values_m - expected) ** 2))
    )
    tolerance = 3.0e-8
    _require(
        full_domain_rms_error <= tolerance,
        "periodic level set did not translate at the supplied velocity",
    )
    return {
        "description": "WENO5/SSPRK3 smooth periodic material-level-set translation",
        "metrics": {
            "cfl_number": result.diagnostics.cfl_number,
            "full_domain_rms_error_m": full_domain_rms_error,
            "evaluated_cell_count": count * count,
            "excluded_boundary_cell_count": 0,
            "reinitialized": result.diagnostics.reinitialized,
        },
        "criteria": {"full_domain_rms_error_m_lte": tolerance},
    }


def _translated_planar_species(
    rows: int, columns: int, spacing_m: float, position_m: float, width_m: float
) -> np.ndarray:
    x = np.arange(columns, dtype=np.float64) * spacing_m
    profile = np.clip(0.5 + (position_m - x) / width_m, 0.0, 1.0)
    return np.tile(profile, (rows, 1))


def _front_regression_check() -> dict[str, Any]:
    rows, columns = 12, 72
    spacing = 0.08
    elapsed = 0.25
    expected_speed = 0.36
    previous_position = 2.1
    previous = _translated_planar_species(rows, columns, spacing, previous_position, 0.45)
    current = _translated_planar_species(
        rows,
        columns,
        spacing,
        previous_position + expected_speed * elapsed,
        0.45,
    )
    frame = axis_aligned_sampling_frame(previous.shape, dx_m=spacing)
    sensitivity = evaluate_species_threshold_sensitivity(
        previous,
        current,
        frame,
        elapsed_time_s=elapsed,
        dx_m=spacing,
        species_thresholds=ASSUMED_SPECIES_THRESHOLD_SET,
    )
    speeds = {
        str(threshold): float(result.mean_normal_regression_speed_m_s)
        for threshold, result in sensitivity.by_threshold.items()
    }
    maximum_error = float(max(abs(value - expected_speed) for value in speeds.values()))
    comparable_counts = {
        str(threshold): result.diagnostics.persistent_unique_ray_count
        for threshold, result in sensitivity.by_threshold.items()
    }
    tolerance = 2.0e-13
    _require(tuple(sensitivity.by_threshold) == (0.3, 0.5, 0.7), "threshold set changed")
    _require(maximum_error <= tolerance, "subcell planar front speed is wrong")
    _require(all(value == rows for value in comparable_counts.values()), "front rays were lost")
    return {
        "description": "Subcell planar Species-front regression at assumed thresholds",
        "metrics": {
            "thresholds": [0.3, 0.5, 0.7],
            "threshold_provenance": sensitivity.threshold_provenance.value,
            "expected_speed_m_per_s": expected_speed,
            "measured_speed_m_per_s": speeds,
            "maximum_abs_speed_error_m_per_s": maximum_error,
            "comparable_ray_count": comparable_counts,
        },
        "criteria": {"maximum_abs_speed_error_m_per_s_lte": tolerance},
    }


def _weno_spatial_convergence_check() -> dict[str, Any]:
    grids = [20, 40, 80]
    errors: list[float] = []
    for count in grids:
        dx = 1.0 / count
        centre = (np.arange(count, dtype=np.float64) + 0.5) * dx
        wave_number = 2.0 * np.pi
        cell_average = (
            np.cos(wave_number * (centre - 0.5 * dx))
            - np.cos(wave_number * (centre + 0.5 * dx))
        ) / (wave_number * dx)
        reconstructed, _ = weno5_js_reconstruct(
            cell_average[:, None], boundary="periodic"
        )
        exact = np.sin(wave_number * np.arange(count + 1, dtype=np.float64) * dx)
        errors.append(float(np.mean(np.abs(reconstructed[:, 0] - exact))))
    orders = _observed_orders(errors)
    minimum_required_order = 4.5
    _require(min(orders) >= minimum_required_order, "WENO smooth spatial order is below threshold")
    return {
        "description": "WENO5-JS smooth periodic face-reconstruction convergence",
        "metrics": {
            "grid_cell_counts": grids,
            "mean_absolute_errors": errors,
            "observed_orders": orders,
            "minimum_observed_order": float(min(orders)),
        },
        "criteria": {"minimum_observed_order_gte": minimum_required_order},
    }


def _ssprk_temporal_convergence_check() -> dict[str, Any]:
    step_counts = [10, 20, 40]
    errors: list[float] = []
    dt_values: list[float] = []
    cfl_like_values: list[float] = []
    decay_rate_per_s = 1.0
    for step_count in step_counts:
        dt = 1.0 / step_count
        value = np.array([1.0], dtype=np.float64)
        time = 0.0
        for _ in range(step_count):
            value = ssprk3_step(
                value,
                time,
                dt,
                lambda _time, state: -decay_rate_per_s * state,
            )
            time += dt
        dt_values.append(dt)
        cfl_like_values.append(decay_rate_per_s * dt)
        errors.append(abs(float(value[0]) - math.exp(-1.0)))
    orders = _observed_orders(errors)
    minimum_required_order = 2.85
    _require(min(orders) >= minimum_required_order, "SSPRK3 temporal order is below threshold")
    return {
        "description": "SSPRK(3,3) temporal convergence for y'=-y",
        "metrics": {
            "step_counts": step_counts,
            "dt_s": dt_values,
            "cfl_like_abs_lambda_dt": cfl_like_values,
            "absolute_errors": errors,
            "observed_orders": orders,
            "minimum_observed_order": float(min(orders)),
        },
        "criteria": {"minimum_observed_order_gte": minimum_required_order},
    }


class _ManufacturedPassiveContactSolver(ReactiveSolver):
    """Full-orchestrator manufactured solution with no unpublished source fit.

    Constant Tait density/pressure/velocity transport independent smooth
    temperature and Species fields.  Reaction, electrical heating, solid
    decomposition, and thermal conduction are exactly zero.  This exercises
    the complete stage coupling without inventing an ECSP reaction closure.
    """

    _temperature_centre_m = 0.62
    _temperature_width_m = 0.10
    _temperature_base_K = 2.0
    _temperature_amplitude_K = 0.4
    _front_position_m = 0.35
    _front_width_m = 0.08

    def analytic_fields(
        self, time_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        config = self.config
        x = (np.arange(config.nx, dtype=np.float64) + 0.5) * config.dx_m
        x_grid = np.broadcast_to(x[None, :], self.shape)
        displacement = config.initial_u_m_per_s * float(time_s)
        temperature = self._temperature_base_K + self._temperature_amplitude_K * np.exp(
            -(
                (x_grid - (self._temperature_centre_m + displacement))
                / self._temperature_width_m
            )
            ** 2
        )
        progress = 0.5 * (
            1.0
            - np.tanh(
                (x_grid - (self._front_position_m + displacement))
                / self._front_width_m
            )
        )
        material_level_set = x_grid - (self._front_position_m + displacement)
        return temperature, progress, material_level_set

    def initial_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        config = self.config
        temperature, progress, material_level_set = self.analytic_fields(0.0)
        density = np.full(self.shape, config.initial_rho_kg_per_m3, dtype=np.float64)
        velocity_x = np.full(self.shape, config.initial_u_m_per_s, dtype=np.float64)
        velocity_y = np.full(self.shape, config.initial_v_m_per_s, dtype=np.float64)
        conservative = primitive_to_conservative(
            density,
            velocity_x,
            velocity_y,
            pressure_Pa=None,
            reaction_progress=progress,
            eos=self.eos,
            temperature_K=temperature,
        )
        # Eq. (2) is deliberately a separate, constant diagnostic state here.
        solid_temperature = np.full(
            self.shape, self._temperature_base_K, dtype=np.float64
        )
        solid_alpha = np.ones(self.shape, dtype=np.float64)
        self._validate_stage(
            conservative,
            solid_temperature,
            solid_alpha,
            material_level_set,
            "manufactured_initial",
        )
        return conservative, solid_temperature, solid_alpha, material_level_set


def _manufactured_coupled_config(nx: int, cfl: float) -> ReactiveCaseConfig:
    package_root = Path(__file__).resolve().parents[2]
    base = load_reactive_config(
        package_root / "config" / "paper_faithful_assumed_v8_3_0.yaml"
    )
    return replace(
        base,
        source_path="<manufactured-passive-contact-validation>",
        strict_paper=False,
        nx=nx,
        ny=8,
        lx_m=1.0,
        ly_m=8.0,
        end_time_s=0.2,
        cfl=cfl,
        maximum_steps=1000,
        initial_rho_kg_per_m3=1.0,
        initial_u_m_per_s=0.2,
        initial_v_m_per_s=0.0,
        initial_temperature_K=2.0,
        initial_reaction_progress=0.0,
        initial_alpha=1.0,
        tait_rho0_kg_per_m3=1.0,
        tait_A_Pa=1.0,
        tait_B_Pa=0.05,
        tait_N=2.0,
        caloric_cv_J_per_kgK=1.0,
        caloric_reference_temperature_K=1.0,
        caloric_reference_energy_J_per_kg=0.0,
        reaction_preexponential_per_s=0.0,
        reaction_activation_energy_J_per_mol=0.0,
        reaction_heat_J_per_kg=0.0,
        reaction_order=1.0,
        solid_density_kg_per_m3=1.0,
        solid_heat_capacity_J_per_kgK=1.0,
        solid_conductivity_W_per_mK=0.0,
        decomposition_preexponential_per_s=0.0,
        decomposition_activation_energy_J_per_mol=0.0,
        decomposition_heat_J_per_kg=0.0,
        decomposition_order=1.0,
        conductivity_S_per_m=None,
        electric_field_x_V_per_m=None,
        electric_field_y_V_per_m=None,
        electrochemical_heat_W_per_m3=None,
        interface_x_m=_ManufacturedPassiveContactSolver._front_position_m,
        reinitialization_interval_steps=0,
        density_reject_below_kg_per_m3=1.0e-12,
        pressure_reject_below_Pa=0.0,
        temperature_reject_below_K=1.0e-12,
        progress_tolerance=1.0e-12,
        butler_volmer_enabled=False,
        electrical_source_provider="PRESCRIBED_ARRAY_FIELDS",
        algorithms={**base.algorithms, "flow_boundary": "OUTFLOW"},
    )


def _manufactured_coupled_run(
    nx: int, cfl: float, *, include_solution_fields: bool = False
) -> dict[str, Any]:
    config = _manufactured_coupled_config(nx, cfl)
    zeros = np.zeros((config.ny, config.nx), dtype=np.float64)
    electrical = ElectricalSourceFields(
        joule_heat_W_per_m3=zeros.copy(),
        electrochemical_heat_W_per_m3=zeros.copy(),
        electric_potential_V=zeros.copy(),
        metadata={
            "provider": "PRESCRIBED_ARRAY_FIELDS",
            "provenance": "ASSUMED_NOT_FROM_PAPER",
            "source": "manufactured validation source fixed to exact zero",
        },
    )
    solver = _ManufacturedPassiveContactSolver(
        config, prescribed_electrical=electrical
    )
    initial_conservative, _, _, _ = solver.initial_state()
    result = solver.run()
    primitive = conservative_to_primitive(result.conservative_state, solver.eos)
    exact_temperature, exact_progress, exact_level_set = solver.analytic_fields(
        result.final_time_s
    )
    exact_conservative = primitive_to_conservative(
        np.ones(solver.shape, dtype=np.float64),
        config.initial_u_m_per_s,
        config.initial_v_m_per_s,
        pressure_Pa=None,
        reaction_progress=exact_progress,
        eos=solver.eos,
        temperature_K=exact_temperature,
    )
    temperature_rms_error = float(
        np.sqrt(np.mean((primitive.temperature_K - exact_temperature) ** 2))
    )
    energy_rms_error = float(
        np.sqrt(
            np.mean(
                (
                    result.conservative_state[..., TOTAL_ENERGY]
                    - exact_conservative[..., TOTAL_ENERGY]
                )
                ** 2
            )
        )
    )
    pressure_linf_error = float(np.max(np.abs(primitive.pressure_Pa - 1.0)))
    level_set_rms_error = float(
        np.sqrt(np.mean((result.material_level_set_m - exact_level_set) ** 2))
    )

    grid_first_cell = (0.5 * config.dx_m, 0.5 * config.dy_m)
    frame = axis_aligned_sampling_frame(
        solver.shape,
        dx_m=config.dx_m,
        dy_m=config.dy_m,
        grid_first_cell_xy_m=grid_first_cell,
    )
    initial_progress = conservative_to_primitive(
        initial_conservative, solver.eos
    ).reaction_progress
    sensitivity = evaluate_species_threshold_sensitivity(
        initial_progress,
        primitive.reaction_progress,
        frame,
        elapsed_time_s=result.final_time_s,
        dx_m=config.dx_m,
        dy_m=config.dy_m,
        grid_first_cell_xy_m=grid_first_cell,
        species_thresholds=ASSUMED_SPECIES_THRESHOLD_SET,
    )
    speeds = {
        str(threshold): float(row.mean_normal_regression_speed_m_s)
        for threshold, row in sensitivity.by_threshold.items()
    }
    expected_speed = config.initial_u_m_per_s
    maximum_speed_error = float(
        max(abs(value - expected_speed) for value in speeds.values())
    )
    energy_index = TOTAL_ENERGY
    energy_scale = max(
        abs(result.conservation.initial_integral[energy_index]),
        abs(result.conservation.final_integral[energy_index]),
        abs(result.conservation.flux_divergence_increment[energy_index]),
        1.0e-30,
    )
    energy_balance_residual = abs(
        result.conservation.closure_residual[energy_index]
    ) / energy_scale
    all_component_scales = np.maximum.reduce(
        (
            np.abs(np.asarray(result.conservation.initial_integral)),
            np.abs(np.asarray(result.conservation.final_integral)),
            np.abs(np.asarray(result.conservation.flux_divergence_increment)),
            np.full(5, 1.0e-30),
        )
    )
    maximum_conservation_residual = float(
        np.max(
            np.abs(np.asarray(result.conservation.closure_residual))
            / all_component_scales
        )
    )
    payload: dict[str, Any] = {
        "nx": nx,
        "ny": config.ny,
        "dx_m": config.dx_m,
        "dy_m": config.dy_m,
        "cfl": cfl,
        "accepted_steps": result.accepted_steps,
        "expected_regression_speed_m_per_s": expected_speed,
        "regression_speed_m_per_s_by_threshold": speeds,
        "maximum_abs_regression_speed_error_m_per_s": maximum_speed_error,
        "flow_temperature_rms_error_K": temperature_rms_error,
        "flow_pressure_linf_error_Pa": pressure_linf_error,
        "flow_total_energy_density_rms_error_J_per_m3": energy_rms_error,
        "normalized_total_energy_balance_residual": float(energy_balance_residual),
        "maximum_normalized_conservation_residual": maximum_conservation_residual,
        "material_level_set_rms_error_m": level_set_rms_error,
        "positivity_face_fallback_count": int(
            result.metadata["positivity_face_fallback_count"]
        ),
        "positivity_face_fallback_max_abs_state_correction": float(
            result.metadata["positivity_face_fallback_max_abs_state_correction"]
        ),
        "cell_state_clipping_count": int(
            result.metadata["cell_state_clipping_count"]
        ),
        "density_pressure_temperature_floor_application_count": int(
            result.metadata["density_pressure_temperature_floor_application_count"]
        ),
        "reaction_rate_cap_application_count": int(
            result.metadata["reaction_rate_cap_application_count"]
        ),
        "reinitialization_count": int(
            result.metadata["level_set_reinitialization_count"]
        ),
        "stage_failure_policy": result.metadata["stage_failure_policy"],
    }
    if include_solution_fields:
        # Private in-memory comparison data.  The caller removes this key
        # before constructing the strict-JSON report.
        payload["_solution_fields"] = {
            "temperature_K": np.array(primitive.temperature_K, copy=True),
            "pressure_Pa": np.array(primitive.pressure_Pa, copy=True),
            "total_energy_density_J_per_m3": np.array(
                result.conservative_state[..., TOTAL_ENERGY], copy=True
            ),
        }
    return payload


def _coupled_solver_grid_cfl_check() -> dict[str, Any]:
    # These are deliberately inexpensive manufactured resolutions, not the
    # paper's 50^2--200^2 ECSP grids.  The latter remain blocked below.
    grid_counts = [20, 40, 80]
    fixed_cfl = 0.25
    grid_runs = [_manufactured_coupled_run(count, fixed_cfl) for count in grid_counts]
    # Full-solver temporal refinement is compared at one fixed spatial
    # operator against a separately computed small-CFL reference.  Comparing
    # directly with the continuum answer here would mix spatial and temporal
    # truncation and can yield a false temporal-order claim.
    temporal_nx = 20
    cfl_values = [0.4, 0.2, 0.1]
    temporal_reference_cfl = 0.025
    temporal_runs_private = [
        _manufactured_coupled_run(
            temporal_nx, cfl, include_solution_fields=True
        )
        for cfl in cfl_values
    ]
    temporal_reference = _manufactured_coupled_run(
        temporal_nx,
        temporal_reference_cfl,
        include_solution_fields=True,
    )
    reference_fields = temporal_reference["_solution_fields"]
    temporal_temperature_errors = [
        float(
            np.sqrt(
                np.mean(
                    (
                        row["_solution_fields"]["temperature_K"]
                        - reference_fields["temperature_K"]
                    )
                    ** 2
                )
            )
        )
        for row in temporal_runs_private
    ]
    temporal_energy_errors = [
        float(
            np.sqrt(
                np.mean(
                    (
                        row["_solution_fields"]["total_energy_density_J_per_m3"]
                        - reference_fields["total_energy_density_J_per_m3"]
                    )
                    ** 2
                )
            )
        )
        for row in temporal_runs_private
    ]
    temporal_pressure_errors = [
        float(
            np.max(
                np.abs(
                    row["_solution_fields"]["pressure_Pa"]
                    - reference_fields["pressure_Pa"]
                )
            )
        )
        for row in temporal_runs_private
    ]
    temporal_front_errors_by_threshold = {
        str(threshold): [
            abs(
                row["regression_speed_m_per_s_by_threshold"][str(threshold)]
                - temporal_reference["regression_speed_m_per_s_by_threshold"][
                    str(threshold)
                ]
            )
            for row in temporal_runs_private
        ]
        for threshold in ASSUMED_SPECIES_THRESHOLD_SET
    }
    temporal_temperature_orders = _observed_orders(temporal_temperature_errors)
    temporal_energy_orders = _observed_orders(temporal_energy_errors)
    temporal_front_orders_by_threshold = {
        threshold: _observed_orders(errors)
        for threshold, errors in temporal_front_errors_by_threshold.items()
    }
    cfl_runs: list[dict[str, Any]] = []
    for row in temporal_runs_private:
        public_row = dict(row)
        del public_row["_solution_fields"]
        cfl_runs.append(public_row)
    temporal_reference_public = dict(temporal_reference)
    del temporal_reference_public["_solution_fields"]

    temperature_errors = [row["flow_temperature_rms_error_K"] for row in grid_runs]
    energy_field_errors = [
        row["flow_total_energy_density_rms_error_J_per_m3"] for row in grid_runs
    ]
    front_errors = [
        row["maximum_abs_regression_speed_error_m_per_s"] for row in grid_runs
    ]
    temperature_orders = _observed_orders(temperature_errors)
    energy_field_orders = _observed_orders(energy_field_errors)
    front_orders = _observed_orders(front_errors)
    minimum_temperature_order = 2.5
    minimum_energy_field_order = 2.5
    minimum_front_order = 1.0
    _require(
        min(temperature_orders) >= minimum_temperature_order,
        "manufactured flow-temperature grid convergence is below the threshold",
    )
    _require(
        min(energy_field_orders) >= minimum_energy_field_order,
        "manufactured total-energy field grid convergence is below the threshold",
    )
    _require(
        min(front_orders) >= minimum_front_order,
        "manufactured Species-front grid convergence is below the threshold",
    )

    all_runs = grid_runs + cfl_runs + [temporal_reference_public]
    maximum_pressure_error = float(
        max(row["flow_pressure_linf_error_Pa"] for row in all_runs)
    )
    maximum_energy_balance_residual = float(
        max(row["normalized_total_energy_balance_residual"] for row in all_runs)
    )
    maximum_conservation_residual = float(
        max(row["maximum_normalized_conservation_residual"] for row in all_runs)
    )
    fallback_count = sum(row["positivity_face_fallback_count"] for row in all_runs)
    fallback_maximum_correction = float(
        max(
            row["positivity_face_fallback_max_abs_state_correction"]
            for row in all_runs
        )
    )
    clipping_count = sum(row["cell_state_clipping_count"] for row in all_runs)
    floor_count = sum(
        row["density_pressure_temperature_floor_application_count"]
        for row in all_runs
    )
    reaction_cap_count = sum(
        row["reaction_rate_cap_application_count"] for row in all_runs
    )
    reinitialization_count = sum(row["reinitialization_count"] for row in all_runs)
    _require(maximum_pressure_error <= 2.0e-13, "constant Tait pressure was not preserved")
    _require(
        maximum_energy_balance_residual <= 1.0e-11,
        "manufactured total-energy balance did not close",
    )
    _require(
        maximum_conservation_residual <= 1.0e-11,
        "manufactured conservative-state budget did not close",
    )
    _require(fallback_count == 0, "manufactured smooth runs activated positivity fallback")
    _require(
        fallback_maximum_correction == 0.0,
        "zero-count positivity fallback reported a nonzero correction",
    )
    _require(
        clipping_count == 0 and floor_count == 0 and reaction_cap_count == 0,
        "manufactured convergence used clipping, a physical-state floor, or a rate cap",
    )
    _require(reinitialization_count == 0, "manufactured runs unexpectedly reinitialized")

    minimum_temporal_temperature_order = 2.8
    minimum_temporal_energy_order = 2.8
    minimum_temporal_front_order = 2.5
    _require(
        min(temporal_temperature_orders) >= minimum_temporal_temperature_order,
        "full-solver flow-temperature temporal order is below threshold",
    )
    _require(
        min(temporal_energy_orders) >= minimum_temporal_energy_order,
        "full-solver energy-field temporal order is below threshold",
    )
    _require(
        min(
            min(orders) for orders in temporal_front_orders_by_threshold.values()
        )
        >= minimum_temporal_front_order,
        "full-solver Species-front temporal order is below threshold",
    )
    _require(
        max(temporal_pressure_errors) <= 2.0e-13,
        "full-solver temporal refinement changed invariant Tait pressure",
    )
    return {
        "description": (
            "Full ReactiveSolver passive-contact manufactured grid and temporal convergence"
        ),
        "scope": {
            "status": "SYNTHETIC_MANUFACTURED_NOT_PAPER_REPRODUCTION",
            "flow_sources": "reaction_and_electrical_exactly_zero",
            "solid_state": "separate_constant_diagnostic",
            "paper_parameter_fitting": False,
            "physical_state_clipping": "NONE",
            "positivity_fallback_required": False,
            "quantity_scope": {
                "regression_rate": "passively_advected_Species_front",
                "temperature": "flow_temperature_smooth_contact_advection",
                "pressure": "constant_Tait_pressure_invariance",
                "energy": "no_source_transport_field_and_conservative_budget",
            },
            "limitations": {
                "kinetically_generated_reaction_front": (
                    "NOT_VERIFIED_MISSING_PAPER_KINETICS"
                ),
                "nontrivial_solid_Eq2_temperature_convergence": (
                    "NOT_VERIFIED_IN_THIS_MANUFACTURED_CHECK"
                ),
                "nonuniform_Tait_pressure_wave_convergence": (
                    "NOT_VERIFIED_IN_THIS_MANUFACTURED_CHECK"
                ),
                "source_active_energy_grid_convergence": (
                    "NOT_VERIFIED_MISSING_PAPER_SOURCE_CLOSURES; source ratios and "
                    "Joule integrals are checked separately"
                ),
            },
        },
        "grid_refinement": {
            "runs": grid_runs,
            "flow_temperature_observed_orders": temperature_orders,
            "flow_total_energy_field_observed_orders": energy_field_orders,
            "species_front_speed_observed_orders": front_orders,
            "pressure_order_status": "EXACT_INVARIANT_ZERO_ERROR_ORDER_UNDEFINED",
            "energy_balance_order_status": "ROUNDOFF_BOUNDED_ORDER_NOT_MEANINGFUL",
        },
        "cfl_refinement": {
            "runs": cfl_runs,
            "fixed_spatial_grid_nx_ny": [temporal_nx, 8],
            "reference_run": temporal_reference_public,
            "reference_cfl": temporal_reference_cfl,
            "reference_status": (
                "SMALL_CFL_SAME_SPATIAL_OPERATOR_NUMERICAL_REFERENCE_NOT_EXACT_SOLUTION"
            ),
            "temperature_rms_difference_from_reference_K": temporal_temperature_errors,
            "temperature_observed_orders": temporal_temperature_orders,
            "total_energy_density_rms_difference_from_reference_J_per_m3": (
                temporal_energy_errors
            ),
            "total_energy_field_observed_orders": temporal_energy_orders,
            "pressure_linf_difference_from_reference_Pa": temporal_pressure_errors,
            "pressure_order_status": "EXACT_INVARIANT_ZERO_ERROR_ORDER_UNDEFINED",
            "regression_speed_abs_difference_from_reference_m_per_s_by_threshold": (
                temporal_front_errors_by_threshold
            ),
            "regression_speed_observed_orders_by_threshold": (
                temporal_front_orders_by_threshold
            ),
            "paper_cfl_equivalence": "NONE_MANUFACTURED_VALUES_ONLY",
        },
        "metrics": {
            "maximum_pressure_linf_error_Pa_all_runs": maximum_pressure_error,
            "maximum_normalized_total_energy_balance_residual_all_runs": (
                maximum_energy_balance_residual
            ),
            "maximum_normalized_conservation_residual_all_runs": (
                maximum_conservation_residual
            ),
            "positivity_face_fallback_count_all_runs": fallback_count,
            "positivity_face_fallback_max_abs_state_correction_all_runs": (
                fallback_maximum_correction
            ),
            "cell_state_clipping_count_all_runs": clipping_count,
            "density_pressure_temperature_floor_application_count_all_runs": (
                floor_count
            ),
            "reaction_rate_cap_application_count_all_runs": reaction_cap_count,
            "reinitialization_count_all_runs": reinitialization_count,
        },
        "criteria": {
            "minimum_temperature_grid_order_gte": minimum_temperature_order,
            "minimum_total_energy_field_grid_order_gte": minimum_energy_field_order,
            "minimum_species_front_speed_grid_order_gte": minimum_front_order,
            "maximum_pressure_linf_error_Pa_lte": 2.0e-13,
            "maximum_normalized_energy_balance_residual_lte": 1.0e-11,
            "minimum_full_solver_temperature_temporal_order_gte": (
                minimum_temporal_temperature_order
            ),
            "minimum_full_solver_energy_temporal_order_gte": (
                minimum_temporal_energy_order
            ),
            "minimum_full_solver_front_temporal_order_gte": (
                minimum_temporal_front_order
            ),
            "fallback_clipping_floor_cap_and_reinitialization_count_eq": 0,
            "tolerance_rationale": (
                "orders test refinement rather than one-grid magnitude; invariant pressure "
                "and budgets use FP64 roundoff bounds; full-solver CFL order uses a "
                "separate small-CFL result with the identical spatial operator"
            ),
        },
    }


def _deterministic_kernel_signature() -> str:
    eos = IdealGasEOS(gamma=1.4, gas_constant_J_per_kg_K=287.05)
    ny, nx = 10, 12
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    x_grid, y_grid = np.meshgrid(x, y)
    state = primitive_to_conservative(
        1.1 + 0.01 * np.sin(2.0 * np.pi * x_grid),
        2.0 + 0.02 * np.cos(2.0 * np.pi * y_grid),
        -0.4,
        101325.0 + 10.0 * np.cos(2.0 * np.pi * x_grid),
        0.25 + 0.01 * np.sin(2.0 * np.pi * y_grid),
        eos,
    )
    rhs_x, _ = finite_volume_rhs(
        state, 1.0 / nx, eos, spatial_axis=1, direction="x", boundary="periodic"
    )
    rhs_y, _ = finite_volume_rhs(
        state, 1.0 / ny, eos, spatial_axis=0, direction="y", boundary="periodic"
    )
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(rhs_x).view(np.uint8).tobytes())
    digest.update(np.ascontiguousarray(rhs_y).view(np.uint8).tobytes())
    return digest.hexdigest()


def _determinism_check(repeats: int) -> dict[str, Any]:
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError("deterministic_repeats must be an integer of at least two")
    signatures = [_deterministic_kernel_signature() for _ in range(repeats)]
    unique = sorted(set(signatures))
    _require(len(unique) == 1, "repeated FP64 validation kernels were not bitwise deterministic")
    return {
        "description": "Bitwise repeatability of a periodic 2-D Euler RHS",
        "metrics": {
            "repeat_count": repeats,
            "unique_sha256_count": len(unique),
            "sha256": unique[0],
        },
        "criteria": {"unique_sha256_count_eq": 1},
    }


def _row_decomposition_parity_check() -> dict[str, Any]:
    rng = np.random.default_rng(83100)
    global_rows, columns, components = 24, 13, 5
    rank_count, halo = 3, 3
    field = rng.normal(size=(global_rows, columns, components))
    serial_halo = add_outflow_row_halos(field, halo)
    weights = np.array([-0.03, 0.12, -0.31, 0.44, -0.31, 0.12, -0.03])

    def stencil(haloed: np.ndarray, interior_rows: int) -> np.ndarray:
        return sum(
            weights[offset + halo]
            * haloed[offset + halo : offset + halo + interior_rows]
            for offset in range(-halo, halo + 1)
        )

    serial = stencil(serial_halo, global_rows)
    local_outputs: list[np.ndarray] = []
    reconstructed_blocks: list[np.ndarray] = []
    partitions = all_row_partitions(global_rows, rank_count)
    for rank, (start, stop) in enumerate(partitions):
        local = np.ascontiguousarray(field[start:stop])
        emulated = add_outflow_row_halos(local, halo)
        if rank > 0:
            emulated[:halo] = field[start - halo : start]
        if rank + 1 < rank_count:
            emulated[-halo:] = field[stop : stop + halo]
        local_outputs.append(stencil(emulated, stop - start))
        reconstructed_blocks.append(local)
    decomposed = np.concatenate(local_outputs, axis=0)
    reconstructed = np.concatenate(reconstructed_blocks, axis=0)
    maximum_stencil_error = float(np.max(np.abs(decomposed - serial)))
    maximum_gather_error = float(np.max(np.abs(reconstructed - field)))
    _require(maximum_stencil_error == 0.0, "emulated row halos changed the stencil result")
    _require(maximum_gather_error == 0.0, "row partition/gather changed field values")
    return {
        "description": "Balanced row decomposition and emulated three-row halo parity",
        "metrics": {
            "global_shape": [global_rows, columns, components],
            "rank_count": rank_count,
            "halo_rows": halo,
            "partitions": [list(pair) for pair in partitions],
            "maximum_stencil_abs_error": maximum_stencil_error,
            "maximum_gather_abs_error": maximum_gather_error,
        },
        "criteria": {
            "maximum_stencil_abs_error_eq": 0.0,
            "maximum_gather_abs_error_eq": 0.0,
        },
    }


def _execute_check(function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        payload = function()
        return {"status": PASS, **payload}
    except Exception as exc:  # each check must leave a serializable audit row
        return {
            "status": FAIL,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _mpi_status() -> dict[str, Any]:
    mpi4py_available = importlib.util.find_spec("mpi4py") is not None
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if not mpi4py_available or launcher is None:
        return {
            "status": "NOT_TESTED_NO_MPI_RUNTIME",
            "mpi4py_available": mpi4py_available,
            "mpi_launcher": launcher,
            "reason": "Actual multi-process MPI parity requires both mpi4py and an MPI launcher.",
        }
    return {
        "status": "NOT_TESTED_MULTI_RANK_NOT_RUN",
        "mpi4py_available": True,
        "mpi_launcher": launcher,
        "reason": "This in-process harness does not claim a multi-rank run; execute a dedicated mpiexec validation.",
    }


def run_validation(*, deterministic_repeats: int = 50) -> dict[str, Any]:
    """Run all CPU manufactured checks and return a strict-JSON-ready report.

    A ``PASS`` in ``checks`` only establishes the stated analytic/discrete
    identity.  The top-level verdict remains ``NOT_FULLY_VERIFIED`` while the
    paper inputs, actual MPI execution, and CUDA/A100 implementation evidence
    are absent.
    """

    if not isinstance(deterministic_repeats, int) or deterministic_repeats < 2:
        raise ValueError("deterministic_repeats must be an integer of at least two")
    checks: dict[str, dict[str, Any]] = {
        "tait_dpdrho_equals_c2": _execute_check(_tait_identity_check),
        "primitive_conservative_roundtrip": _execute_check(_primitive_roundtrip_check),
        "free_stream_and_conservation": _execute_check(_free_stream_conservation_check),
        "reaction_heat_progress_ratio": _execute_check(_reaction_bookkeeping_check),
        "linear_potential_joule_energy": _execute_check(_joule_linear_field_check),
        "periodic_thermal_integral": _execute_check(_periodic_thermal_integral_check),
        "level_set_translation": _execute_check(_level_set_translation_check),
        "planar_front_regression_thresholds": _execute_check(_front_regression_check),
        "weno_smooth_spatial_convergence": _execute_check(_weno_spatial_convergence_check),
        "ssprk3_temporal_convergence": _execute_check(_ssprk_temporal_convergence_check),
        "coupled_solver_grid_and_cfl": _execute_check(
            _coupled_solver_grid_cfl_check
        ),
        "deterministic_repeats": _execute_check(
            lambda: _determinism_check(deterministic_repeats)
        ),
        "row_decomposition_emulated_halo_parity": _execute_check(
            _row_decomposition_parity_check
        ),
    }
    passed = sum(row["status"] == PASS for row in checks.values())
    failed = len(checks) - passed
    runnable_status = PASS if failed == 0 else FAIL
    report: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "floating_point": "numpy_float64_cpu",
        },
        "runnable_cpu_verification": {
            "status": runnable_status,
            "passed_check_count": passed,
            "failed_check_count": failed,
            "total_check_count": len(checks),
            "scope": "manufactured_and_discrete_reference_checks_only",
        },
        "checks": checks,
        "convergence_scope": {
            "synthetic_coupled_solver": {
                "status": checks["coupled_solver_grid_and_cfl"]["status"],
                "spatial_grid_cell_counts_x": [20, 40, 80],
                "fixed_grid_cfl_values": [0.4, 0.2, 0.1],
                "small_cfl_numerical_reference": 0.025,
                "quantities": [
                    "species_front_regression_speed",
                    "flow_temperature",
                    "flow_pressure",
                    "flow_total_energy_field",
                    "conservative_energy_balance",
                ],
                "scope": "manufactured_passive_contact_no_ECSP_parameter_fit",
            },
            "paper_ecsp_case": {
                "status": "NOT_VERIFIED_MISSING_PAPER_INPUTS",
                "paper_reported_grid_family": [50, 100, 150, 200],
                "paper_selected_grid": 100,
                "paper_cfl_values": "NOT_PUBLISHED",
                "regression_rate": "NOT_VERIFIED",
                "temperature": "NOT_VERIFIED",
                "pressure": "NOT_VERIFIED",
                "energy_balance": "NOT_VERIFIED",
                "reason": (
                    "The paper grid family is known, but its executable input deck, "
                    "error norm, time history, material closures, and CFL values are not."
                ),
            },
        },
        "numerical_safety": {
            "physical_state_clipping": "NONE_IN_REACTIVE_SOLVER",
            "density_pressure_temperature_limits": "REJECTION_ONLY_NOT_FLOORS",
            "stage_retry_policy": (
                "EXPLICIT_COUNTED_TRANSACTIONAL_STAGE_RETRY;NO_PARTIAL_STATE_OR_"
                "BUDGET_COMMIT;FAIL_CLOSED_ON_CONFIGURED_LIMIT"
            ),
            "positivity_fallback": (
                "FACE_LOCAL_AND_COUNTED; coupled manufactured convergence requires zero activations"
            ),
            "validation_failure_policy": "FAILED_CHECK_RECORDED_AND_OVERALL_CPU_STATUS_FAIL",
        },
        "paper_reproduction": {
            "status": "BLOCKED_MISSING_INPUTS",
            "target_regression_speed_m_per_s": 0.0036094,
            "target_temperature_K": 1804.8,
            "fitting_attempted": False,
            "requested_grid_and_cfl_convergence": {
                "status": "NOT_VERIFIED_MISSING_INPUTS",
                "regression_rate": "NOT_VERIFIED",
                "temperature": "NOT_VERIFIED",
                "pressure": "NOT_VERIFIED",
                "energy_balance": "NOT_VERIFIED",
                "note": (
                    "The passing 20/40/80 and CFL 0.4/0.2/0.1 study is a "
                    "manufactured solver check, not the paper ECSP case."
                ),
            },
            "missing_inputs": [
                "ECSP Tait A/B/N/rho0 values and a caloric EOS",
                "gas/progress and condensed decomposition kinetic coefficients",
                "dimensionally closed reaction/electrochemical heat conversions",
                "complete initial, external-boundary, and material-interface conditions",
                "measured conductivity and electric-potential fields",
                "WENO variant, flux split/Riemann solver, CFL, and reinitialization details",
                "MPI rank topology, halo width, and exchange order",
            ],
            "reason": "The article does not publish a uniquely executable input deck; synthetic checks were not tuned to its reported outputs.",
        },
        "capability_validation": {
            "actual_mpi": _mpi_status(),
            "cuda_backend": {
                "status": "NOT_IMPLEMENTED",
                "reason": "ecsp_reactive is currently a NumPy CPU reference implementation.",
            },
            "a100_execution": {
                "status": "NOT_TESTED_NO_HARDWARE",
                "reason": "No reactive CUDA backend exists, so no A100 execution claim is made.",
            },
        },
        "verdict": FULL_VERIFICATION_VERDICT,
        "verdict_reason": (
            "CPU manufactured checks do not replace missing paper inputs, actual multi-rank MPI validation, "
            "or a CUDA/A100 implementation and hardware run."
        ),
    }
    # Refuse NaN/Infinity and accidental non-JSON objects before returning an
    # audit payload that downstream release tooling may persist.
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def write_validation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a report atomically enough for a single-process release check."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return destination


__all__ = [
    "FAIL",
    "FULL_VERIFICATION_VERDICT",
    "PASS",
    "VALIDATION_SCHEMA",
    "run_validation",
    "write_validation_report",
]
