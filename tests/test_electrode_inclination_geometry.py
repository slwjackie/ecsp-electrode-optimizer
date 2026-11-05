"""Geometry invariants for angle changes independent of production physics."""
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import electrode_inclination_geometry as geom
import run_electrode_parameter_study as runner


def case(angle=20., wa=4., wc=4., gap=17., sid="ANGLE20"):
    return dict(id=sid, anode_width_mm=wa, cathode_width_mm=wc,
                gap_mm=gap, inclination_deg=angle, groups=["inclination"])


def spec():
    return dict(schema=geom.INCLINED_SCHEMA, domain_mm=25., electrode_length_mm=25.,
                grid_size=200, placement="centered_assembly", experimental_replicates=5,
                cases=[case(0., sid="STRAIGHT"), case()])


def polygon_area(vertices):
    xy = np.asarray(vertices)
    return abs(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) -
                      xy[:, 1] * np.roll(xy[:, 0], -1))) / 2


@pytest.mark.parametrize("n", [100, 150, 200, 400])
@pytest.mark.parametrize("angle", [0., 10., 20., 25.])
def test_area_and_symmetry_are_preserved(n, angle):
    a, c, m = geom.make_inclined_case(case(angle), n)
    h = 25 / n
    assert a.dtype == c.dtype == np.bool_
    assert not np.any(a & c)
    assert a.sum() * h**2 == pytest.approx(100.)
    assert c.sum() * h**2 == pytest.approx(100.)
    assert np.array_equal(a, np.flipud(a))
    assert np.array_equal(c, np.fliplr(a))
    assert np.all(a.sum(axis=1) == round(4 / h))
    assert np.all(np.any(a[:-1] & a[1:], axis=1))
    assert np.all(np.any(c[:-1] & c[1:], axis=1))
    assert polygon_area(m["anode_polygon_mm"]) == pytest.approx(100.)
    assert polygon_area(m["cathode_polygon_mm"]) == pytest.approx(100.)
    # The bend's analytic midpoint remains identical to the straight reference.
    assert m["anode_polygon_mm"][2] == [4., 12.5]
    assert m["cathode_polygon_mm"][-1] == [21., 12.5]
    assert m["central_gap_mm"] == 17
    assert m["end_gap_mm"] == pytest.approx(17 - 25 * math.tan(math.radians(angle)))
    assert m["anode_normal_width_mm"] == pytest.approx(4 * math.cos(math.radians(angle)))
    assert m["electrode_centerline_length_mm"] == pytest.approx(25 / math.cos(math.radians(angle)))
    amin = np.where(a)[1].max()
    cmin = np.where(c)[1].min()
    assert m["raster_minimum_gap_mm"] == pytest.approx((cmin - amin - 1) * h)


@pytest.mark.parametrize("sid", list(runner.CASES))
def test_zero_degree_reproduces_every_legacy_straight_mask(sid):
    wa, wc, gap, _ = runner.CASES[sid]
    a, c, m = geom.make_inclined_case(case(0, wa, wc, gap), 200)
    old_a, old_c, _ = runner.make_case(sid, 200)
    assert np.array_equal(a, old_a)
    assert np.array_equal(c, old_c)
    assert m["inclined_edges_exact_on_grid"]
    assert m["raster_max_boundary_error_mm"] == 0


def test_unequal_areas_remain_matched_to_their_own_straight_controls():
    a, c, m = geom.make_inclined_case(case(15, wa=2., wc=4., gap=19.), 200)
    assert a.sum() * .125**2 == 50
    assert c.sum() * .125**2 == 100
    assert m["anode_to_cathode_area_ratio"] == .5


def test_error_bound_includes_full_cell_height_and_shrinks_with_grid():
    bounds = []
    for n in [100, 200, 400]:
        a, _, m = geom.make_inclined_case(case(), n)
        h = 25 / n
        xleft = np.argmax(a, axis=1) * h
        slope = math.tan(math.radians(20))
        # Check both ends of each stair-step cell row against exact left edge.
        error = max(np.max(abs(xleft - abs(np.arange(n) * h - 12.5) * slope)),
                    np.max(abs(xleft - abs((np.arange(n) + 1) * h - 12.5) * slope)))
        assert m["raster_max_boundary_error_mm"] == pytest.approx(error)
        assert error <= h * (1 + slope) / 2 + 1e-12
        bounds.append(error)
    assert bounds[2] < bounds[1] < bounds[0]


@pytest.mark.parametrize("key,value", [
    ("inclination_deg", float("nan")), ("inclination_deg", float("inf")),
    ("inclination_deg", -1), ("inclination_deg", 90), ("inclination_deg", 45),
    ("inclination_deg", True), ("anode_width_mm", float("nan")),
    ("anode_width_mm", 0), ("gap_mm", 30), ("gap_mm", .5),
    ("cathode_width_mm", 4.01), ("id", "../bad"), ("groups", ["spacing"]),
])
def test_invalid_case_never_produces_masks(key, value):
    row = case()
    row[key] = value
    with pytest.raises(ValueError):
        geom.make_inclined_case(row, 200)


@pytest.mark.parametrize("width", [.25, 1.])
def test_inclined_normal_width_respects_existing_one_mm_minimum(width):
    with pytest.raises(ValueError, match="normal width"):
        geom.make_inclined_case(case(20, wa=width, wc=width, gap=25-2*width), 100)


def test_spec_requires_matched_control_and_valid_unique_cases():
    original = spec()
    assert geom.validate_inclined_spec(original) == original
    variants = []
    changed = copy.deepcopy(original)
    changed["cases"][0]["inclination_deg"] = 10
    variants.append(changed)
    changed = copy.deepcopy(original)
    changed["cases"][1]["anode_width_mm"] = 3
    variants.append(changed)
    changed = copy.deepcopy(original)
    changed["cases"][1]["id"] = "STRAIGHT"
    variants.append(changed)
    changed = copy.deepcopy(original)
    changed["cases"][1]["inclination_deg"] = 0
    variants.append(changed)
    for changed in variants:
        with pytest.raises(ValueError):
            geom.validate_inclined_spec(changed)
    assert original == spec()


def test_unresolved_angle_is_rejected_instead_of_simulating_duplicate_masks():
    requested = spec()
    requested["cases"][1]["inclination_deg"] = .001
    with pytest.raises(ValueError, match="identical raster masks.*reduce grid spacing"):
        geom.validate_inclined_spec(requested)
