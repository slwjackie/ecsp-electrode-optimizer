from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from ecsp_nsga2.baselines import (
    BaselineGeometryError,
    generate_area_matched_staggered,
    write_comparison_image,
    write_objective_comparison_csv,
)
from ecsp_nsga2.evaluator import CppCondensedFp64Evaluator, EvaluatorError
from ecsp_nsga2.geometry import GeometryLimits, save_geometry
from ecsp_v6.physics.numerics import resize_nearest_numpy


def _boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    labels, count = ndimage.label(mask, np.ones((3, 3), int))
    result = []
    for label in range(1, count + 1):
        yy, xx = np.nonzero(labels == label)
        result.append((yy.min(), yy.max() + 1, xx.min(), xx.max() + 1))
    return sorted(result, key=lambda box: box[2])


def _assert_reference(limits: GeometryLimits, physics_grid: int) -> tuple[dict, object, object]:
    genome, raster, params = generate_area_matched_staggered(
        limits,
        physics_grid_size=physics_grid,
        fingers_per_polarity=2,
        target_interdigitation_overlap_fraction=0.50,
        minimum_interdigitation_overlap_fraction=0.45,
    )
    assert genome["included_in_nsga2_population"] is False
    assert genome["source_role"] == "post_optimization_reference_only"
    assert genome["baseline_layout_revision"] == "v7.9.2_hidden_bus_vertical_2a2c"
    assert genome["horizontal_contact_order"] == ["anode", "cathode", "anode", "cathode"]
    assert genome["hidden_bus_in_contact_mask"] is False
    assert genome["hidden_bus_contact_area_fraction"] == 0.0
    assert len(genome["anode"]["components"]) == 2
    assert len(genome["cathode"]["components"]) == 2
    assert params.fingers_per_polarity == 2
    assert raster.constraint_violation == 0.0
    assert ndimage.label(raster.anode_mask, np.ones((3, 3), int))[1] == 2
    assert ndimage.label(raster.cathode_mask, np.ones((3, 3), int))[1] == 2

    anode_boxes = _boxes(raster.anode_mask)
    cathode_boxes = _boxes(raster.cathode_mask)
    assert len({(b[1] - b[0], b[3] - b[2]) for b in anode_boxes}) == 1
    assert len({(b[1] - b[0], b[3] - b[2]) for b in cathode_boxes}) == 1
    assert {(b[1] - b[0], b[3] - b[2]) for b in anode_boxes} == {
        (b[1] - b[0], b[3] - b[2]) for b in cathode_boxes
    }
    assert all(box[0] == 0 for box in anode_boxes)
    assert all(box[1] == limits.grid_size for box in cathode_boxes)
    # No same-polarity contact bus exists in the propellant-contact mask.
    assert not np.any(raster.anode_mask[-1])
    assert not np.any(raster.cathode_mask[0])
    # Left-to-right visible order must be A-C-A-C.
    ordered = sorted(
        [(box[2], "anode") for box in anode_boxes]
        + [(box[2], "cathode") for box in cathode_boxes]
    )
    assert [polarity for _, polarity in ordered] == ["anode", "cathode", "anode", "cathode"]

    assert limits.minimum_width_mm <= params.common_finger_width_mm <= limits.maximum_width_mm
    assert params.design_minimum_gap_mm >= limits.minimum_gap_mm
    assert params.physics_minimum_gap_mm >= limits.minimum_gap_mm
    assert params.design_vertical_overlap_fraction >= 0.45
    assert params.physics_vertical_overlap_fraction >= 0.45
    assert (
        abs(params.physics_area_fraction_per_polarity - limits.target_area_fraction_per_polarity)
        / limits.target_area_fraction_per_polarity
        <= limits.area_tolerance_fraction
    )

    anode = resize_nearest_numpy(raster.anode_mask, physics_grid)
    cathode = resize_nearest_numpy(raster.cathode_mask, physics_grid)
    assert ndimage.label(anode, np.ones((3, 3), int))[1] == 2
    assert ndimage.label(cathode, np.ones((3, 3), int))[1] == 2
    physics_dims = {
        (box[1] - box[0], box[3] - box[2])
        for box in _boxes(anode) + _boxes(cathode)
    }
    assert len(physics_dims) == 1
    spacing = limits.domain_mm / (physics_grid - 1)
    gap = ndimage.distance_transform_edt(~cathode)[anode].min() * spacing
    assert gap >= limits.minimum_gap_mm
    assert abs(float(anode.mean()) - limits.target_area_fraction_per_polarity) <= (
        limits.target_area_fraction_per_polarity * limits.area_tolerance_fraction
    )
    return genome, raster, params


def test_area_matched_staggered_user_20mm_constraints():
    _assert_reference(
        GeometryLimits(
            domain_mm=20.0,
            grid_size=96,
            margin_mm=0.75,
            minimum_gap_mm=2.0,
            minimum_width_mm=1.0,
            maximum_width_mm=5.0,
            target_area_fraction_per_polarity=0.20,
            area_tolerance_fraction=0.02,
            maximum_components_per_polarity=1,
            maximum_total_components=2,
        ),
        physics_grid=193,
    )


def test_area_matched_staggered_user_25mm_constraints():
    _assert_reference(
        GeometryLimits(
            domain_mm=25.0,
            grid_size=120,
            margin_mm=0.75,
            minimum_gap_mm=2.0,
            minimum_width_mm=1.0,
            maximum_width_mm=5.0,
            target_area_fraction_per_polarity=0.20,
            area_tolerance_fraction=0.02,
            maximum_components_per_polarity=3,
            maximum_total_components=4,
        ),
        physics_grid=241,
    )


def test_exact_two_finger_layout_is_not_silently_replaced_when_infeasible():
    with pytest.raises(BaselineGeometryError, match="No two-anode/two-cathode"):
        generate_area_matched_staggered(
            GeometryLimits(
                domain_mm=20.0,
                grid_size=96,
                minimum_gap_mm=0.5,
                minimum_width_mm=0.3,
                maximum_width_mm=1.2,
                target_area_fraction_per_polarity=0.175,
                area_tolerance_fraction=0.035,
            ),
            physics_grid_size=193,
        )


def test_cpp_reference_component_override_is_baseline_only(tmp_path: Path):
    limits = GeometryLimits(
        domain_mm=20.0,
        grid_size=96,
        margin_mm=0.75,
        minimum_gap_mm=2.0,
        minimum_width_mm=1.0,
        maximum_width_mm=5.0,
        target_area_fraction_per_polarity=0.20,
        area_tolerance_fraction=0.02,
        maximum_components_per_polarity=1,
        maximum_total_components=2,
    )
    genome, raster, _ = generate_area_matched_staggered(
        limits, physics_grid_size=193
    )
    evaluator = CppCondensedFp64Evaluator.__new__(CppCondensedFp64Evaluator)
    evaluator.grid_size = 193
    evaluator.domain_size_m = 0.020
    evaluator.minimum_gap_m = 0.002
    evaluator.maximum_components_per_polarity = 1
    evaluator.maximum_total_components = 2
    evaluator.target_area_fraction_per_polarity = 0.20
    evaluator.area_tolerance_fraction = 0.02

    base_metadata = {
        "geometry_id": "AREA_MATCHED_STAGGERED",
        "intended_anode_components": 2,
        "intended_cathode_components": 2,
    }
    with pytest.raises(EvaluatorError, match="Component cap violated"):
        evaluator._prepare_cpp_item(
            (raster.anode_mask, raster.cathode_mask, base_metadata, tmp_path)
        )

    reference_metadata = {
        **base_metadata,
        "baseline_type": "area_matched_staggered",
        "source_role": "post_optimization_area_matched_staggered_reference",
        "allow_hidden_bus_reference_component_override": True,
        "reference_maximum_components_per_polarity": 2,
        "reference_maximum_total_components": 4,
        "reference_target_area_fraction_per_polarity": 0.20,
    }
    prepared = evaluator._prepare_cpp_item(
        (raster.anode_mask, raster.cathode_mask, reference_metadata, tmp_path)
    )
    assert prepared[0].shape == (193, 193)
    assert ndimage.label(prepared[0], np.ones((3, 3), int))[1] == 2
    assert ndimage.label(prepared[1], np.ones((3, 3), int))[1] == 2


def test_baseline_output_helpers(tmp_path: Path):
    limits = GeometryLimits(
        domain_mm=20.0,
        grid_size=96,
        minimum_gap_mm=2.0,
        minimum_width_mm=1.0,
        maximum_width_mm=5.0,
        target_area_fraction_per_polarity=0.20,
        area_tolerance_fraction=0.02,
        maximum_components_per_polarity=1,
        maximum_total_components=2,
    )
    genome, raster, _ = generate_area_matched_staggered(
        limits, physics_grid_size=193
    )
    save_geometry(genome, raster, tmp_path)
    baseline_png = tmp_path / "AREA_MATCHED_STAGGERED.png"
    recommended_png = tmp_path / "recommended.png"
    recommended_png.write_bytes(baseline_png.read_bytes())
    names = (
        "ignition_delay_s",
        "area_undecomposed_fraction_at_2s",
        "minimum_ignition_voltage_V",
        "current_congestion",
    )
    recommended = {
        "geometry_id": "AI",
        "topology_id": "AI_TOP",
        "constraint_violation": 0.0,
        "ignition_success": True,
        "objectives": dict(zip(names, (1.0, 0.9, 10.0, 20.0))),
    }
    baseline = {
        "geometry_id": "AREA_MATCHED_STAGGERED",
        "topology_id": "BASELINE_HIDDEN_BUS_STAGGERED_2A2C",
        "constraint_violation": 0.0,
        "ignition_success": True,
        "objectives": dict(zip(names, (1.2, 0.92, 11.0, 25.0))),
    }
    write_objective_comparison_csv(
        tmp_path / "comparison.csv", recommended, baseline, names
    )
    write_comparison_image(
        tmp_path / "comparison.png",
        recommended_png,
        baseline_png,
        recommended,
        baseline,
        names,
    )
    assert (tmp_path / "comparison.csv").is_file()
    assert (tmp_path / "comparison.png").is_file()


def test_area_matched_staggered_explicit_recommended_area_override():
    limits = GeometryLimits(
        domain_mm=20.0,
        grid_size=96,
        margin_mm=0.75,
        minimum_gap_mm=2.0,
        minimum_width_mm=1.0,
        maximum_width_mm=5.0,
        target_area_fraction_per_polarity=0.20,
        area_tolerance_fraction=0.02,
        maximum_components_per_polarity=1,
        maximum_total_components=2,
    )
    requested_actual_area = 0.1975
    _, _, params = generate_area_matched_staggered(
        limits,
        physics_grid_size=193,
        target_area_fraction_per_polarity=requested_actual_area,
    )
    assert params.target_area_fraction_per_polarity == requested_actual_area
    assert (
        abs(params.physics_area_fraction_per_polarity - requested_actual_area)
        / requested_actual_area
        <= limits.area_tolerance_fraction
    )


def test_area_matched_staggered_streaming_best_memory_regression():
    import inspect

    import ecsp_nsga2.baselines as baseline_module

    source = inspect.getsource(baseline_module.generate_area_matched_staggered)
    assert "candidates.append" not in source
    assert "best_candidate" in source

    # Keep the behavioural parity check deliberately small/fast; the source
    # invariant above is what prevents the large-N mask-accumulation regression.
    limits = GeometryLimits(
        domain_mm=20.0,
        grid_size=96,
        margin_mm=0.75,
        minimum_gap_mm=2.0,
        minimum_width_mm=1.0,
        maximum_width_mm=5.0,
        target_area_fraction_per_polarity=0.20,
        area_tolerance_fraction=0.02,
        maximum_components_per_polarity=2,
        maximum_total_components=4,
    )

    _, raster_a, params_a = generate_area_matched_staggered(
        limits,
        physics_grid_size=193,
        target_area_fraction_per_polarity=0.20,
    )
    _, raster_b, params_b = generate_area_matched_staggered(
        limits,
        physics_grid_size=193,
        target_area_fraction_per_polarity=0.20,
    )

    assert params_a == params_b
    np.testing.assert_array_equal(raster_a.anode_mask, raster_b.anode_mask)
    np.testing.assert_array_equal(raster_a.cathode_mask, raster_b.cathode_mask)
