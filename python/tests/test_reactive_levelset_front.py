from __future__ import annotations

import math

import numpy as np
import pytest

from ecsp_reactive.front import (
    ASSUMED_SPECIES_THRESHOLD_PROVENANCE,
    ASSUMED_SPECIES_THRESHOLD_SET,
    NormalSamplingFrame,
    axis_aligned_sampling_frame,
    evaluate_species_threshold_sensitivity,
    extract_species_isofront,
    measure_normal_regression,
    track_species_isofront_motion,
)
from ecsp_reactive.level_set import (
    MATERIAL_LEVEL_SET_SIGN_CONVENTION,
    BoundaryCondition2D,
    MaterialLevelSet,
    ReinitializationConfig,
    advect_material_level_set,
    reinitialize_transported_level_set,
    signed_distance_rms_error,
    weno5_one_sided_derivatives,
    zero_contour_interior_area_m2,
)
from ecsp_reactive.provenance import (
    Provenance,
    ReactiveConfigurationError,
    ReactiveNumericalError,
)


def _cell_centres(count: int, spacing: float) -> np.ndarray:
    return (np.arange(count, dtype=np.float64) + 0.5) * spacing


def test_material_level_set_sign_is_fixed_and_independent_of_reaction_progress() -> (
    None
):
    phi = np.array([[-1.0, 0.0, 2.0], [-0.2, 0.3, 1.0]])
    level_set = MaterialLevelSet(phi, dx_m=0.1, dy_m=0.2)
    reaction_progress_a = np.zeros_like(phi)
    reaction_progress_b = np.ones_like(phi)

    assert MATERIAL_LEVEL_SET_SIGN_CONVENTION.startswith("phi<0 material_interior")
    assert np.array_equal(
        level_set.interior_mask, np.array([[True, False, False], [True, False, False]])
    )
    # Reaction fields are deliberately not an input to MaterialLevelSet and do
    # not alter its material classification.
    assert not np.array_equal(reaction_progress_a, reaction_progress_b)
    assert np.array_equal(level_set.interior_mask, phi < 0.0)
    assert not level_set.values_m.flags.writeable
    with pytest.raises(ValueError):
        level_set.values_m.setflags(write=True)


def test_periodic_weno5_derivative_has_high_order_smooth_convergence() -> None:
    errors = []
    for count in (32, 64):
        spacing = 1.0 / count
        x = np.arange(count, dtype=np.float64) * spacing
        values = np.tile(np.sin(2.0 * np.pi * x), (8, 1))
        expected = np.tile(2.0 * np.pi * np.cos(2.0 * np.pi * x), (8, 1))
        backward, forward = weno5_one_sided_derivatives(
            values, spacing, axis=1, boundary="periodic"
        )
        errors.append(float(np.sqrt(np.mean((backward - expected) ** 2))))
        assert np.sqrt(np.mean((forward - expected) ** 2)) < 1.1 * errors[-1]
    assert errors[0] / errors[1] > 20.0


def test_level_set_weno_is_local_to_its_stencil() -> None:
    count = 100
    spacing = 1.0 / count
    x = np.arange(count, dtype=np.float64) * spacing
    values = np.tile(np.sin(2.0 * np.pi * x), (8, 1))
    reference = weno5_one_sided_derivatives(
        values, spacing, axis=1, boundary="periodic"
    )
    changed = values.copy()
    changed[0, 80] = 1.0e8
    perturbed = weno5_one_sided_derivatives(
        changed, spacing, axis=1, boundary="periodic"
    )
    for derivative_reference, derivative_perturbed in zip(reference, perturbed):
        np.testing.assert_array_equal(
            derivative_perturbed[0, 10:50], derivative_reference[0, 10:50]
        )


def test_ssprk3_weno_periodic_constant_velocity_translates_smooth_level_set() -> None:
    count = 96
    spacing = 1.0 / count
    x = np.arange(count, dtype=np.float64) * spacing
    x_grid, _ = np.meshgrid(x, x)
    initial = np.sin(2.0 * np.pi * (x_grid - 0.2))
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
    expected = np.sin(2.0 * np.pi * (x_grid - 0.2 - velocity * dt))

    assert result.diagnostics.cfl_number == pytest.approx(0.4)
    assert result.diagnostics.boundary_x == "periodic"
    assert result.diagnostics.boundary_y == "periodic"
    assert (
        result.diagnostics.spatial_scheme_provenance
        is Provenance.ASSUMED_NOT_FROM_PAPER
    )
    assert (
        result.diagnostics.time_integrator_provenance
        is Provenance.ASSUMED_NOT_FROM_PAPER
    )
    assert (
        result.diagnostics.boundary_closure_provenance
        is Provenance.ASSUMED_NOT_FROM_PAPER
    )
    assert (
        np.sqrt(np.mean((result.material_level_set.values_m - expected) ** 2)) < 3.0e-8
    )


def test_outflow_planar_level_set_moves_at_prescribed_velocity() -> None:
    count = 48
    spacing = 0.02
    x = _cell_centres(count, spacing)
    x_grid, _ = np.meshgrid(x, x)
    initial = x_grid - 0.47
    velocity = 0.13
    dt = 0.35 * spacing / velocity
    result = advect_material_level_set(
        initial,
        velocity,
        0.0,
        dt_s=dt,
        dx_m=spacing,
        boundary=BoundaryCondition2D(x="outflow", y="outflow"),
    )
    expected = initial - velocity * dt
    # Constant extrapolation deliberately reduces the stencil order at the
    # physical boundary.  The uncontaminated interior transports a plane
    # exactly to floating-point accuracy.
    # Three Runge--Kutta stages each evaluate a five-point upwind stencil, so
    # exclude the full nine-cell numerical domain of dependence.
    assert (
        np.max(np.abs(result.material_level_set.values_m[:, 9:-9] - expected[:, 9:-9]))
        < 2.0e-13
    )


def test_constant_velocity_circle_advection_preserves_location_and_area() -> None:
    count = 64
    spacing = 1.0 / count
    coordinate = _cell_centres(count, spacing)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    radius = 0.15
    initial = np.sqrt((x_grid - 0.35) ** 2 + (y_grid - 0.45) ** 2) - radius
    initial_area = zero_contour_interior_area_m2(initial, dx_m=spacing)
    velocity_x, velocity_y = 0.2, -0.1
    final_time = 0.2
    nominal_dt = 0.35 * spacing / (abs(velocity_x) + abs(velocity_y))
    step_count = math.ceil(final_time / nominal_dt)
    dt = final_time / step_count
    transported = initial
    for _ in range(step_count):
        transported = advect_material_level_set(
            transported,
            velocity_x,
            velocity_y,
            dt_s=dt,
            dx_m=spacing,
            boundary="outflow",
        ).material_level_set.values_m
    expected = (
        np.sqrt(
            (x_grid - (0.35 + velocity_x * final_time)) ** 2
            + (y_grid - (0.45 + velocity_y * final_time)) ** 2
        )
        - radius
    )
    near_interface = np.abs(expected) <= 3.0 * spacing
    final_area = zero_contour_interior_area_m2(transported, dx_m=spacing)

    assert (
        np.sqrt(np.mean((transported[near_interface] - expected[near_interface]) ** 2))
        < 1.0e-4 * spacing
    )
    assert abs(final_area - initial_area) / initial_area < 2.0e-4


def test_level_set_cfl_violation_is_not_silently_advanced() -> None:
    phi = np.tile(np.linspace(-1.0, 1.0, 16), (16, 1))
    with pytest.raises(ReactiveNumericalError) as captured:
        advect_material_level_set(
            phi,
            1.0,
            1.0,
            dt_s=1.0,
            dx_m=0.1,
            cfl_limit=0.6,
        )
    assert captured.value.category == "level_set_cfl_violation"


def test_reinitialization_requires_explicit_nonpaper_provenance() -> None:
    phi = np.tile(np.linspace(-1.0, 1.0, 16), (16, 1))
    config = ReinitializationConfig(enabled=True, provenance=Provenance.PAPER)
    with pytest.raises(ReactiveConfigurationError, match="ASSUMED_NOT_FROM_PAPER"):
        reinitialize_transported_level_set(phi, dx_m=0.1, config=config)


def test_reinitialization_uses_transported_phi_and_measures_zero_contour_drift() -> (
    None
):
    count = 72
    spacing = 1.0 / count
    coordinate = _cell_centres(count, spacing)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    signed_distance = np.sqrt((x_grid - 0.5) ** 2 + (y_grid - 0.5) ** 2) - 0.22
    distorted = signed_distance * (
        1.0 + 0.45 * np.sin(2.0 * np.pi * x_grid) * np.cos(2.0 * np.pi * y_grid)
    )
    before_error = signed_distance_rms_error(distorted, dx_m=spacing)
    config = ReinitializationConfig(enabled=True, pseudo_steps=70, pseudo_cfl=0.3)
    result = advect_material_level_set(
        distorted,
        0.0,
        0.0,
        dt_s=1.0e-4,
        dx_m=spacing,
        reinitialization=config,
    )
    diagnostics = result.diagnostics

    assert diagnostics.reinitialized
    assert diagnostics.reinitialization_provenance is Provenance.ASSUMED_NOT_FROM_PAPER
    assert diagnostics.signed_distance_rms_error_before == pytest.approx(before_error)
    assert diagnostics.signed_distance_rms_error_after < 0.08 * before_error
    assert abs(diagnostics.reinitialization_zero_contour_relative_drift) < 0.015
    assert diagnostics.reinitialization_zero_contour_area_drift_m2 != 0.0
    assert diagnostics.sign_changed_cell_count_during_reinitialization == 0


def _translated_planar_species(
    rows: int, columns: int, spacing: float, position_m: float, width_m: float
) -> np.ndarray:
    x = np.arange(columns, dtype=np.float64) * spacing
    profile = np.clip(0.5 + (position_m - x) / width_m, 0.0, 1.0)
    return np.tile(profile, (rows, 1))


def test_species_front_uses_subcell_threshold_interpolation() -> None:
    rows, columns = 9, 64
    spacing = 0.1
    position = 2.35
    progress = _translated_planar_species(rows, columns, spacing, position, 0.5)
    frame = axis_aligned_sampling_frame(progress.shape, dx_m=spacing, normal_axis="x")
    snapshot = extract_species_isofront(
        progress,
        frame,
        species_threshold=0.5,
        dx_m=spacing,
    )

    assert snapshot.diagnostics.unique_intersection_ray_count == rows
    assert snapshot.diagnostics.absent_intersection_ray_count == 0
    assert snapshot.diagnostics.multiple_intersection_ray_count == 0
    assert snapshot.tracking_method_provenance is Provenance.ASSUMED_NOT_FROM_PAPER
    assert np.allclose(snapshot.normal_iso_position_m, position, atol=2.0e-15)
    assert np.allclose(snapshot.isofront_coordinates_xy_m[:, 0], position)


def test_planar_species_front_local_and_mean_normal_speed_are_exact() -> None:
    rows, columns = 12, 72
    spacing = 0.08
    elapsed_time = 0.25
    expected_speed = 0.36
    previous_position = 2.1
    current_position = previous_position + expected_speed * elapsed_time
    previous = _translated_planar_species(
        rows, columns, spacing, previous_position, 0.45
    )
    current = _translated_planar_species(rows, columns, spacing, current_position, 0.45)
    frame = axis_aligned_sampling_frame(previous.shape, dx_m=spacing)
    result = track_species_isofront_motion(
        previous,
        current,
        frame,
        elapsed_time_s=elapsed_time,
        species_threshold=0.5,
        dx_m=spacing,
    )

    assert result.diagnostics.persistent_unique_ray_count == rows
    assert np.allclose(
        result.local_normal_displacement_m, expected_speed * elapsed_time
    )
    assert np.allclose(result.local_normal_regression_speed_m_s, expected_speed)
    assert result.mean_normal_regression_speed_m_s == pytest.approx(expected_speed)
    assert result.normal_regression_speed_standard_deviation_m_s < 1.0e-14
    # No consumed-area or front-length field exists in the new result contract.
    assert not hasattr(result, "consumed_area_m2")
    assert not hasattr(result, "front_length_m")


def test_planar_front_speed_is_axis_orientation_independent() -> None:
    spacing = 0.08
    elapsed_time = 0.25
    expected_speed = 0.36
    previous_x = _translated_planar_species(12, 72, spacing, 2.1, 0.45)
    current_x = _translated_planar_species(
        12, 72, spacing, 2.1 + expected_speed * elapsed_time, 0.45
    )
    frame_x = axis_aligned_sampling_frame(
        previous_x.shape, dx_m=spacing, normal_axis="x"
    )
    result_x = track_species_isofront_motion(
        previous_x,
        current_x,
        frame_x,
        elapsed_time_s=elapsed_time,
        species_threshold=0.5,
        dx_m=spacing,
    )

    previous_y = previous_x.T
    current_y = current_x.T
    frame_y = axis_aligned_sampling_frame(
        previous_y.shape, dx_m=spacing, dy_m=spacing, normal_axis="y"
    )
    result_y = track_species_isofront_motion(
        previous_y,
        current_y,
        frame_y,
        elapsed_time_s=elapsed_time,
        species_threshold=0.5,
        dx_m=spacing,
        dy_m=spacing,
    )
    assert result_x.mean_normal_regression_speed_m_s == pytest.approx(expected_speed)
    assert result_y.mean_normal_regression_speed_m_s == pytest.approx(expected_speed)
    assert result_x.mean_normal_regression_speed_m_s == pytest.approx(
        result_y.mean_normal_regression_speed_m_s, abs=2.0e-14
    )


def test_species_threshold_0p3_0p5_0p7_sensitivity_is_assumed_and_reported() -> None:
    rows, columns = 8, 60
    spacing = 0.1
    elapsed_time = 0.2
    expected_speed = 0.5
    previous = _translated_planar_species(rows, columns, spacing, 2.3, 0.6)
    current = _translated_planar_species(
        rows, columns, spacing, 2.3 + expected_speed * elapsed_time, 0.6
    )
    frame = axis_aligned_sampling_frame(previous.shape, dx_m=spacing)
    sensitivity = evaluate_species_threshold_sensitivity(
        previous,
        current,
        frame,
        elapsed_time_s=elapsed_time,
        dx_m=spacing,
    )

    assert tuple(sensitivity.by_threshold) == ASSUMED_SPECIES_THRESHOLD_SET
    assert sensitivity.threshold_provenance is ASSUMED_SPECIES_THRESHOLD_PROVENANCE
    for result in sensitivity.by_threshold.values():
        assert result.threshold_provenance is Provenance.ASSUMED_NOT_FROM_PAPER
        assert result.mean_normal_regression_speed_m_s == pytest.approx(expected_speed)
    assert sensitivity.mean_speed_span_m_s < 5.0e-14
    assert sensitivity.maximum_absolute_deviation_from_threshold_0p5_m_s < 5.0e-14


def test_front_birth_death_and_multiple_intersections_are_diagnosed() -> None:
    spacing = 0.1
    previous = np.array(
        [
            [0.0] * 9,
            np.linspace(1.0, 0.0, 9),
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            np.linspace(1.0, 0.0, 9),
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    current = np.array(
        [
            np.linspace(1.0, 0.0, 9),
            [0.0] * 9,
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            np.linspace(1.0, 0.0, 9),
        ],
        dtype=np.float64,
    )
    frame = axis_aligned_sampling_frame(previous.shape, dx_m=spacing, dy_m=spacing)
    previous_snapshot = extract_species_isofront(
        previous, frame, species_threshold=0.5, dx_m=spacing
    )
    current_snapshot = extract_species_isofront(
        current, frame, species_threshold=0.5, dx_m=spacing
    )
    transition = measure_normal_regression(
        previous_snapshot, current_snapshot, elapsed_time_s=0.1
    )

    assert transition.diagnostics.front_birth_ray_count == 1
    assert transition.diagnostics.front_death_ray_count == 1
    assert transition.diagnostics.unique_to_multiple_branching_candidate_ray_count == 1
    assert transition.diagnostics.multiple_to_unique_merging_candidate_ray_count == 1
    assert transition.diagnostics.previous_multiple_intersection_ray_count == 2
    assert transition.diagnostics.current_multiple_intersection_ray_count == 2
    assert transition.diagnostics.topology_ambiguous_ray_count == 3
    assert transition.diagnostics.persistent_unique_ray_count == 0
    assert math.isnan(transition.mean_normal_regression_speed_m_s)


def test_front_comparison_refuses_different_coordinates_or_normals() -> None:
    progress = _translated_planar_species(5, 20, 0.1, 0.8, 0.4)
    frame = axis_aligned_sampling_frame(progress.shape, dx_m=0.1)
    shifted = NormalSamplingFrame(
        frame.anchor_points_xy_m + np.array((0.01, 0.0)),
        frame.unit_normals_xy,
        frame.signed_sample_offsets_m,
    )
    snapshot_a = extract_species_isofront(
        progress, frame, species_threshold=0.5, dx_m=0.1
    )
    snapshot_b = extract_species_isofront(
        progress, shifted, species_threshold=0.5, dx_m=0.1
    )
    with pytest.raises(
        ReactiveConfigurationError, match="identical physical coordinates"
    ):
        measure_normal_regression(snapshot_a, snapshot_b, elapsed_time_s=0.1)
