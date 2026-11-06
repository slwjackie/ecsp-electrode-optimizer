"""Fixed-end geometry invariants checked independently of the solver."""
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.ndimage import label
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import electrode_central_gap_geometry as geom


def case(gap=8., sid="TEST"):
    return dict(id=sid, central_gap_mm=gap, groups=["central_gap"])


@pytest.fixture
def spec():
    return yaml.safe_load((ROOT / "config/electrode_central_gap_study_12x25mm.yaml").read_text())


def polygon_area(vertices):
    xy = np.asarray(vertices)
    return abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) -
               np.dot(xy[:, 1], np.roll(xy[:, 0], -1))) / 2


@pytest.mark.parametrize("n", [100, 150, 200, 400])
@pytest.mark.parametrize("gap", [2, 4, 6, 8])
def test_area_symmetry_endpoints_central_gap_and_full_domain(n, gap):
    a, c, meta = geom.make_central_gap_case(case(gap), n)
    h, nx = 25 / n, round(12 * n / 25)
    assert a.shape == c.shape == (n, nx)
    assert a.dtype == c.dtype == np.bool_
    assert not (a & c).any()
    assert np.array_equal(a, np.flipud(a))
    assert np.array_equal(c, np.fliplr(a))
    assert np.all(a.sum(axis=1) == round(2 / h))
    assert np.all(c.sum(axis=1) == round(2 / h))
    assert label(a)[1] == label(c)[1] == 1
    assert a.sum() * h**2 == pytest.approx(50)
    assert c.sum() * h**2 == pytest.approx(50)
    assert a.size * h**2 == pytest.approx(300)
    assert meta["propellant_domain_area_mm2"] == 300
    assert meta["electrode_masks_remove_propellant"] is False
    assert meta["anode_area_mm2"] == meta["cathode_area_mm2"] == 50
    assert meta["target_anode_area_fraction"] == pytest.approx(1 / 6)
    for polarity in ("anode", "cathode"):
        assert polygon_area(meta[f"{polarity}_polygon_mm"]) == pytest.approx(50)
    # Independent expected face coordinates: the endpoints never move.
    for mask, bounds in ((a, (3, 5)), (c, (7, 9))):
        for index in (0, -1):
            occupied = np.flatnonzero(mask[index])
            assert occupied[0] * h == pytest.approx(bounds[0])
            assert (occupied[-1] + 1) * h == pytest.approx(bounds[1])
    a_centre = np.flatnonzero(a[n // 2])
    c_centre = np.flatnonzero(c[n // 2])
    assert a_centre[0] * h == pytest.approx(4 - gap / 2)
    assert (a_centre[-1] + 1) * h == pytest.approx(6 - gap / 2)
    assert c_centre[0] * h == pytest.approx(6 + gap / 2)
    assert (c_centre[-1] + 1) * h == pytest.approx(8 + gap / 2)
    assert meta["central_outer_side_clearance_mm"] == 4 - gap / 2
    assert meta["end_outer_side_clearance_mm"] == 3
    assert meta["central_side_clearance_mm"] == 4 - gap / 2
    assert meta["end_side_clearance_mm"] == 3
    assert meta["central_gap_mm"] == meta["gap_mm"] == gap
    assert meta["end_gap_mm"] == meta["minimum_edge_gap_mm"] == 2
    raster_gap = (np.where(c)[1].min() - np.where(a)[1].max() - 1) * h
    assert raster_gap == pytest.approx(2)
    assert meta["raster_minimum_gap_mm"] == pytest.approx(2)
    angle = math.atan((gap - 2) / 25)
    assert meta["inclination_deg"] == pytest.approx(math.degrees(angle))
    assert meta["inclination_is_derived"] is True
    assert meta["anode_normal_width_mm"] == pytest.approx(2 * math.cos(angle))
    assert meta["electrode_centerline_length_mm"] == pytest.approx(25 / math.cos(angle))


@pytest.mark.parametrize("gap", [2, 4, 6, 8])
def test_exact_polygons_match_requested_centre_and_end_coordinates(gap):
    _, _, meta = geom.make_central_gap_case(case(gap), 200)
    assert meta["anode_polygon_mm"] == [
        [3., 0.], [5., 0.], [6 - gap / 2, 12.5],
        [5., 25.], [3., 25.], [4 - gap / 2, 12.5]]
    assert meta["cathode_polygon_mm"] == [
        [7., 0.], [9., 0.], [8 + gap / 2, 12.5],
        [9., 25.], [7., 25.], [6 + gap / 2, 12.5]]


def test_gap_two_is_exactly_the_straight_control():
    a, c, meta = geom.make_central_gap_case(case(2), 200)
    x = (np.arange(96) + .5) * .125
    assert np.array_equal(a, np.broadcast_to((x > 3) & (x < 5), (200, 96)))
    assert np.array_equal(c, np.broadcast_to((x > 7) & (x < 9), (200, 96)))
    assert meta["inclination_deg"] == 0
    assert meta["raster_max_boundary_error_mm"] == 0


def test_default_study_and_subsets_do_not_mutate_input(spec):
    before = copy.deepcopy(spec)
    validated = geom.validate_central_gap_spec(spec)
    assert spec == before
    assert [row["central_gap_mm"] for row in validated["cases"]] == [2, 4, 6, 8]
    assert [row["id"] for row in validated["cases"]] == ["CENTRAL_G2", "CENTRAL_G4", "CENTRAL_G6", "CENTRAL_G8"]
    spec["cases"] = [spec["cases"][2]]
    assert len(geom.validate_central_gap_spec(spec)["cases"]) == 1


@pytest.mark.parametrize("gap", [0, 1, 3, 10, -2, True, float("nan"), float("inf"), "4"])
def test_only_requested_discrete_central_gaps_are_accepted(gap):
    with pytest.raises(ValueError):
        geom.make_central_gap_case(case(gap), 200)


@pytest.mark.parametrize("row", [dict(case(), inclination_deg=10), dict(case(), anode_width_mm=4),
    dict(case(), end_gap_mm=4), dict(case(), id="../outside"), dict(case(), groups=["inclination"])])
def test_angles_widths_endpoints_and_unsafe_ids_are_not_case_inputs(row):
    with pytest.raises(ValueError):
        geom.make_central_gap_case(row, 200)


@pytest.mark.parametrize("n", [True, 99, 193, 201, 200.0])
def test_bad_grid_cannot_stretch_the_domain(n):
    with pytest.raises(ValueError):
        geom.make_central_gap_case(case(), n)


@pytest.mark.parametrize("key,value", [("domain_width_mm", 25), ("domain_height_mm", 12),
    ("electrode_length_mm", 12), ("electrode_width_mm", 4), ("end_gap_mm", 4),
    ("placement", "edge_anchored_pair"), ("experimental_replicates", 4)])
def test_fixed_study_contract_cannot_be_silently_changed(spec, key, value):
    spec[key] = value
    with pytest.raises(ValueError):
        geom.validate_central_gap_spec(spec)


@pytest.mark.parametrize("mutation", ["duplicate_id", "duplicate_gap", "empty"])
def test_reject_duplicate_or_empty_cases(spec, mutation):
    if mutation == "duplicate_id":
        spec["cases"][-1]["id"] = spec["cases"][0]["id"]
    elif mutation == "duplicate_gap":
        spec["cases"].append(dict(spec["cases"][0], id="ANOTHER"))
    else:
        spec["cases"] = []
    with pytest.raises(ValueError):
        geom.validate_central_gap_spec(spec)


def test_raster_boundary_error_bounds_faces_and_decreases():
    errors = []
    gap = 8
    slope = (gap - 2) / 25
    for n in (100, 200, 400):
        a, _, meta = geom.make_central_gap_case(case(gap), n)
        h = 25 / n
        actual_left = np.argmax(a, axis=1) * h
        face_y = np.arange(n + 1) * h
        analytical_faces = 4 - gap / 2 + np.abs(face_y - 12.5) * slope
        observed = np.max(np.maximum(np.abs(actual_left - analytical_faces[:-1]),
                                     np.abs(actual_left - analytical_faces[1:])))
        assert observed <= meta["raster_max_boundary_error_mm"] + 1e-12
        errors.append(meta["raster_max_boundary_error_mm"])
    assert errors[0] > errors[1] > errors[2]
