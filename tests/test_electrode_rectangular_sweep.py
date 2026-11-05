"""Width-specific feasible angle sets on the 12 mm by 25 mm domain."""
import copy
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from electrode_inclination_sweep import plan_inclination_sweep
from electrode_rectangular_geometry import make_rectangular_case


@pytest.fixture
def template():
    return yaml.safe_load((ROOT / "config/electrode_inclination_study_12x25mm.yaml").read_text())


def test_default_rectangular_sweep_reports_both_widths_without_changing_area(template):
    before = copy.deepcopy(template)
    study, reports = plan_inclination_sweep(template)
    assert template == before
    assert len(reports) == 14
    assert len(study["cases"]) == 6
    assert sum(row["status"] == "accepted" for row in reports) == 6
    assert sum(row["status"] == "rejected" for row in reports) == 8
    for width, angles in ((2, [0, 5, 10, 15]), (4, [0, 5])):
        rows = [r for r in study["cases"] if r["anode_width_mm"] == width]
        assert [r["inclination_deg"] for r in rows] == angles
        for row in rows:
            a, c, meta = make_rectangular_case(row, study["grid_size"])
            assert a.shape == c.shape == (200, 96)
            assert meta["actual_anode_area_mm2"] == meta["actual_cathode_area_mm2"] == 25 * width
            assert meta["gap_mm"] == 12 - 2 * width
        requests = [r for r in reports if r["anode_width_mm"] == width]
        assert len(requests) == 7
        assert all(r["domain_width_mm"] == 12 and r["domain_height_mm"] == 25 for r in requests)
        assert all(r["reason"] for r in requests if r["status"] == "rejected")
    assert len({row["id"] for row in study["cases"]}) == 6


@pytest.mark.parametrize("width,angles", [(2, [0, 5, 10, 15]), (4, [0, 5])])
def test_single_width_sweep_preserves_only_selected_width(template, width, angles):
    template["cases"] = [row for row in template["cases"] if row["anode_width_mm"] == width]
    study, reports = plan_inclination_sweep(template)
    assert [row["inclination_deg"] for row in study["cases"]] == angles
    assert all(row["anode_width_mm"] == width for row in reports)
    assert all(row["id"].startswith(f"W{width}_") for row in study["cases"])


def test_partial_group_failure_keeps_its_control_and_other_width_valid_angles(template):
    study, reports = plan_inclination_sweep(template, [10, 15])
    assert [(r["anode_width_mm"], r["inclination_deg"]) for r in study["cases"]] == [
        (2, 0), (2, 10), (2, 15), (4, 0)]
    assert all(r["status"] == "rejected" for r in reports
               if r["anode_width_mm"] == 4 and r["inclination_deg"] > 0)


def test_two_straight_controls_do_not_satisfy_nonzero_requirement(template):
    with pytest.raises(ValueError, match="No feasible nonzero"):
        plan_inclination_sweep(template, [20, 30])


def test_rectangular_duplicate_raster_angles_are_reported_per_width(template):
    study, reports = plan_inclination_sweep(template, [1e-6, 5, 5])
    assert len(study["cases"]) == 4
    duplicates = [r for r in reports if r["inclination_deg"] == 1e-6]
    assert len(duplicates) == 2
    assert all(r["status"] == "rejected" and "Identical raster" in r["reason"] for r in duplicates)
