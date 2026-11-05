"""Independent geometry checks for 12 mm horizontal x 25 mm vertical contacts."""
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
import electrode_rectangular_geometry as geom


def case(width=2., angle=15., sid="W2_TEST"):
    return dict(id=sid, anode_width_mm=width, cathode_width_mm=width,
                gap_mm=12 - 2 * width, inclination_deg=angle, groups=["inclination"])


@pytest.fixture
def spec():
    return yaml.safe_load((ROOT / "config/electrode_inclination_study_12x25mm.yaml").read_text())


def polygon_area(vertices):
    xy = np.asarray(vertices)
    return abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) -
               np.dot(xy[:, 1], np.roll(xy[:, 0], -1))) / 2


@pytest.mark.parametrize("n", [100, 150, 200, 400])
@pytest.mark.parametrize("width,angle", [(2, 0), (2, 5), (2, 10), (2, 15), (4, 0), (4, 5)])
def test_true_rectangular_area_symmetry_full_height_and_connectivity(n, width, angle):
    a, c, meta = geom.make_rectangular_case(case(width, angle), n)
    h = 25 / n
    assert a.shape == c.shape == (n, round(12 / h))
    assert a.dtype == c.dtype == np.bool_
    assert meta["grid_rows"] == n and meta["grid_columns"] == round(12 / h)
    assert meta["domain_width_mm"] == 12 and meta["domain_height_mm"] == 25
    assert a.size * h**2 == pytest.approx(300)
    assert meta["propellant_domain_area_mm2"] == 300
    assert not (a & c).any()
    assert np.array_equal(a, np.flipud(a))
    assert np.array_equal(c, np.fliplr(a))
    assert np.all(a.sum(axis=1) == round(width / h))
    assert label(a)[1] == label(c)[1] == 1
    assert a.sum() * h**2 == pytest.approx(25 * width)
    assert c.sum() * h**2 == pytest.approx(25 * width)
    assert meta["anode_area_mm2"] == meta["cathode_area_mm2"] == 25 * width
    assert meta["target_anode_area_fraction"] == pytest.approx(width / 12)
    assert meta["target_cathode_area_fraction"] == pytest.approx(width / 12)
    assert polygon_area(meta["anode_polygon_mm"]) == pytest.approx(25 * width)
    assert polygon_area(meta["cathode_polygon_mm"]) == pytest.approx(25 * width)
    assert meta["central_gap_mm"] == 12 - 2 * width
    assert meta["end_gap_mm"] == pytest.approx(12 - 2 * width - 25 * math.tan(math.radians(angle)))
    assert meta["anode_normal_width_mm"] == pytest.approx(width * math.cos(math.radians(angle)))
    assert meta["raster_minimum_gap_mm"] >= 1
    # Extremal x edges coincide at both ends by y-symmetry, so this is the global gap.
    raster_gap = (np.where(c)[1].min() - np.where(a)[1].max() - 1) * h
    assert raster_gap == pytest.approx(meta["raster_minimum_gap_mm"])


@pytest.mark.parametrize("width", [2, 4])
def test_straight_control_exactly_matches_two_full_height_edge_rectangles(width):
    a, c, meta = geom.make_rectangular_case(case(width, 0), 200)
    x = (np.arange(96) + .5) * .125
    assert np.array_equal(a, np.broadcast_to(x < width, (200, 96)))
    assert np.array_equal(c, np.broadcast_to(x > 12 - width, (200, 96)))
    assert meta["raster_max_boundary_error_mm"] == 0
    assert meta["anode_polygon_mm"][2] == [width, 12.5]
    assert meta["cathode_polygon_mm"][-1] == [12 - width, 12.5]


def test_default_spec_and_single_width_selections_preserve_input(spec):
    before = copy.deepcopy(spec)
    result = geom.validate_rectangular_spec(spec)
    assert spec == before
    assert len(result["cases"]) == 6
    for width, expected_angles in [(2, [0, 5, 10, 15]), (4, [0, 5])]:
        selected = copy.deepcopy(spec)
        selected["cases"] = [r for r in selected["cases"] if r["anode_width_mm"] == width]
        assert [r["inclination_deg"] for r in geom.validate_rectangular_spec(selected)["cases"]] == expected_angles


@pytest.mark.parametrize("row", [
    case(2, 20), case(4, 10), case(3, 0), case(2, -1), case(2, 90),
    case(2, float("nan")), case(2, float("inf")), case(2, True),
    dict(case(), cathode_width_mm=4), dict(case(), gap_mm=4),
    dict(case(), id="../outside"), dict(case(), groups=["width"]),
])
def test_invalid_requested_contacts_fail_before_rasterization(row):
    with pytest.raises(ValueError):
        geom.make_rectangular_case(row, 200)


@pytest.mark.parametrize("n", [True, 99, 193, 201, 200.0])
def test_grid_contract_cannot_stretch_cells_or_round_domain(n):
    with pytest.raises(ValueError):
        geom.make_rectangular_case(case(), n)


@pytest.mark.parametrize("mutation", ["width", "height", "length", "placement", "duplicate_id", "duplicate_mask", "missing_control", "only_straight"])
def test_study_contract_rejects_geometry_or_control_changes(spec, mutation):
    if mutation == "width":
        spec["domain_width_mm"] = 25
    elif mutation == "height":
        spec["domain_height_mm"] = 12
    elif mutation == "length":
        spec["electrode_length_mm"] = 12
    elif mutation == "placement":
        spec["placement"] = "centered_assembly"
    elif mutation == "duplicate_id":
        spec["cases"][-1]["id"] = spec["cases"][0]["id"]
    elif mutation == "duplicate_mask":
        spec["cases"].append(dict(spec["cases"][0], id="OTHER_ID"))
    elif mutation == "missing_control":
        spec["cases"] = spec["cases"][1:]
    else:
        spec["cases"] = [r for r in spec["cases"] if r["inclination_deg"] == 0]
    with pytest.raises(ValueError):
        geom.validate_rectangular_spec(spec)


def test_raster_boundary_error_is_bounded_and_decreases_with_refinement():
    errors = []
    for n in (100, 200, 400):
        a, _, meta = geom.make_rectangular_case(case(2, 15), n)
        h = 25 / n
        actual_left = np.argmax(a, axis=1) * h
        face_y = np.arange(n + 1) * h
        analytical_faces = np.abs(face_y - 12.5) * math.tan(math.radians(15))
        observed = np.max(np.maximum(np.abs(actual_left - analytical_faces[:-1]),
                                     np.abs(actual_left - analytical_faces[1:])))
        assert observed <= meta["raster_max_boundary_error_mm"] + 1e-12
        errors.append(meta["raster_max_boundary_error_mm"])
    assert errors[0] > errors[1] > errors[2]
