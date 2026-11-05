"""Analytical angle limits and feasible-only sweep planning."""
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from electrode_inclination_geometry import make_inclined_case
from electrode_inclination_sweep import inclination_limits, plan_inclination_sweep


@pytest.fixture
def template():
    return yaml.safe_load((ROOT / "config/electrode_inclination_study_25mm.yaml").read_text())


def test_default_sweep_records_every_request_and_keeps_fixed_area(template):
    before = copy.deepcopy(template)
    study, reports = plan_inclination_sweep(template)
    assert template == before
    assert [r["inclination_deg"] for r in reports] == list(range(0, 61, 5))
    assert [r["inclination_deg"] for r in study["cases"]] == list(range(0, 31, 5))
    assert [r["status"] for r in reports] == ["accepted"] * 7 + ["rejected"] * 6
    assert all("end gap" in r["reason"] for r in reports[7:])
    assert reports[-1]["end_gap_mm"] < 0
    for row in study["cases"]:
        a, c, meta = make_inclined_case(row, study["grid_size"])
        assert a.sum() == c.sum() == 6400
        assert meta["actual_anode_area_mm2"] == meta["actual_cathode_area_mm2"] == 100
        assert not np.any(a & c)


def test_custom_decimal_requests_deduplicate_and_add_control(template):
    study, rows = plan_inclination_sweep(template, [15, 5.5, 15.0, 60])
    assert [r["inclination_deg"] for r in rows] == [0, 5.5, 15, 60]
    assert [r["id"] for r in study["cases"]] == ["ANGLE_00", "ANGLE_5p5", "ANGLE_15"]
    assert rows[-1]["status"] == "rejected"


def test_duplicate_raster_is_reported_without_losing_other_angles(template):
    study, rows = plan_inclination_sweep(template, [1e-6, 20])
    assert [r["inclination_deg"] for r in study["cases"]] == [0, 20]
    assert rows[1]["status"] == "rejected"
    assert "Identical raster mask" in rows[1]["reason"]


@pytest.mark.parametrize("angles", [[], [0], [35, 60], [1e-6]])
def test_no_feasible_nonzero_case_is_an_error(template, angles):
    with pytest.raises(ValueError, match="nonzero"):
        plan_inclination_sweep(template, angles)


@pytest.mark.parametrize("angles", [[float("nan")], [float("inf")], [True], ["5"],
                                   [-5], [90], [100], "5,10", 5, {5: 10}])
def test_malformed_input_is_not_silently_reported_as_infeasible(template, angles):
    with pytest.raises(ValueError):
        plan_inclination_sweep(template, angles)


def test_rectangular_limit_depends_on_full_height():
    square = inclination_limits(25, 25, 4, 4, 17)
    short = inclination_limits(25, 12, 4, 4, 17)
    assert square["maximum_inclination_deg"] == pytest.approx(math.degrees(math.atan(16 / 25)))
    assert short["maximum_inclination_deg"] == pytest.approx(53.13010235415598)
    assert short["geometry_only"] and not short["raster_checked"]
    theta = math.radians(short["maximum_inclination_deg"])
    assert 17 - 12 * math.tan(theta) == pytest.approx(1)
    assert 4 * math.cos(theta) >= 1
    assert 17 - 12 * math.tan(math.radians(55)) < 1
    assert 17 - 12 * math.tan(math.radians(50)) > 1


def test_normal_width_can_be_the_limiting_constraint():
    limits = inclination_limits(25, 1, 2, 3, 20)
    assert limits["maximum_inclination_deg"] == pytest.approx(60.)
    assert limits["gap_limit_deg"] > 60


def test_preserving_100_square_mm_area_on_short_domain_changes_limit():
    width = 100 / 12
    gap = 25 - 2 * width
    limits = inclination_limits(25, 12, width, width, gap)
    assert limits["maximum_inclination_deg"] == pytest.approx(math.degrees(math.atan((gap - 1) / 12)))
    assert 30 < limits["maximum_inclination_deg"] < 35


@pytest.mark.parametrize("args", [
    (25, 0, 4, 4, 17), (25, 12, 4, 4, 18), (25, 12, .5, 4, 17),
    (25, 12, 4, 4, .5), (25, True, 4, 4, 17),
    (25, float("nan"), 4, 4, 17), (25, 12, 4, 4, 17, 0),
    (25, 12, 4, 4, 17, 1, 0),
])
def test_invalid_limit_dimensions_are_rejected(args):
    with pytest.raises(ValueError):
        inclination_limits(*args)
