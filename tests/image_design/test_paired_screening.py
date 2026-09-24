"""Postprocessing tests use synthetic observations, not new PDE measurements."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ecsp_image_design.paired_screening import ScreeningPolicy, screen_pair
from ecsp_image_design.paired_workflow import write_reports, reclassify


def context():
    return dict(domain_mm=25.0, grid_size=101, minimum_gap_mm=3.0, minimum_width_mm=2.0,
                anode_area_mm2=70.0, cathode_area_mm2=70.0, electrode_area_fraction=140/625,
                area_basis="solver_mask", physics_config_hash="synthetic-common-config",
                reference_voltage_V=260.0, evaluation_time_s=2.0, initial_temperature_K=298.15)


def row(ignited=True, t=.2, v=100., temperature=650., j=10.):
    return dict(external_numerical_valid=True, converged=True, allElectricalLinearSolvesConverged=True,
                allNonlinearRobinSolvesConverged=True, ignitionSucceeded=ignited,
                ignitionDelay_s=t if ignited else None, minimumIgnitionVoltageSearchValid=True,
                minimumIgnitionVoltageLeftCensored=False, minimumIgnitionVoltageRightCensored=not ignited,
                minimumIgnitionVoltage_V=v if ignited else None, minimumIgnitionVoltageBracketWidth_V=5.,
                peakMaximumTemperature_K=temperature, peakCurrentCongestionToEvaluationTime=j,
                nativeExecution={"completedFullHorizon": True})


def pair(c=None, b=None, sid="E114"):
    return dict(source_id=sid, candidate_raw=c or row(), baseline_raw=b or row(t=.4, v=160., j=20.),
                context=context(), baseline_context=context())


def test_both_ignited_resolved_improvement():
    r = screen_pair(pair())
    assert r["screening_valid"]
    assert r["comparison_state"] == "both_ignited"
    assert r["R_t"] == .5 and r["R_V"] == .625 and r["R_J"] == .5
    assert r["ignition_screen"] == "promising"
    assert "ignition_superior_candidates" in r["labels"]
    assert r["R_T"] is None


@pytest.mark.parametrize("ci,bi,state,label", [
    (True, False, "candidate_only_ignited", "ignition_superior_candidates"),
    (False, True, "baseline_only_ignited", "ignition_reject_at_reference_condition"),
])
def test_one_sided_ignition(ci, bi, state, label):
    r = screen_pair(pair(row(ci), row(bi)))
    assert r["screening_valid"] and r["comparison_state"] == state
    assert label in r["labels"] and r["R_t"] is None and r["R_V"] is None


def test_user_e114_scalar_regression():
    # Supplied scalar values; other fields are synthetic validity/context fixtures.
    c = row(False, temperature=366.67743983097023, j=8.185091617650894)
    b = row(False, temperature=394.2977746283119, j=35.40469891561951)
    r = screen_pair(pair(c, b))
    assert r["screening_valid"] and r["comparison_state"] == "neither_ignited"
    assert r["R_T"] == pytest.approx((366.67743983097023-298.15)/(394.2977746283119-298.15))
    assert r["R_J"] == pytest.approx(8.185091617650894/35.40469891561951)
    assert r["R_t"] is None and r["R_V"] is None
    assert r["ignition_screen"] == "not_demonstrated" and r["thermal_screen"] == "inferior"
    assert r["labels"] == ["congestion_superior_candidates"]


def test_thermal_gain_is_not_ignition_superiority():
    r = screen_pair(pair(row(False, temperature=400.), row(False, temperature=350.)))
    assert "thermal_promising_candidates" in r["labels"]
    assert "ignition_superior_candidates" not in r["labels"]
    assert r["R_V"] is None and r["R_t"] is None


@pytest.mark.parametrize("key", ["domain_mm", "grid_size", "minimum_gap_mm", "minimum_width_mm",
                                  "reference_voltage_V", "evaluation_time_s", "initial_temperature_K"])
def test_context_mismatch_fails_closed(key):
    p = pair(); p["baseline_context"][key] += 1
    r = screen_pair(p)
    assert not r["screening_valid"] and r["comparison_state"] == "invalid_pair"
    assert r["R_T"] is None and r["R_t"] is None


@pytest.mark.parametrize("key", ["anode_area_mm2", "cathode_area_mm2", "electrode_area_fraction", "physics_config_hash", "area_basis"])
def test_wrong_area_coverage_or_provenance_fails_closed(key):
    p = pair(); p["baseline_context"][key] = None
    assert screen_pair(p)["comparison_state"] == "invalid_pair"


def test_actual_area_more_than_one_percent_rejected():
    p = pair(); p["baseline_context"]["anode_area_mm2"] = 71.
    p["baseline_context"]["electrode_area_fraction"] = 141/625
    assert screen_pair(p)["comparison_state"] == "invalid_pair"


@pytest.mark.parametrize("key", ["converged", "allElectricalLinearSolvesConverged", "allNonlinearRobinSolvesConverged", "external_numerical_valid"])
def test_numerical_failure_never_becomes_no_ignition(key):
    p = pair(row(False), row(False)); p["candidate_raw"][key] = False
    assert screen_pair(p)["comparison_state"] == "numerically_invalid"


def test_censored_vmin_does_not_erase_fixed_condition_metrics():
    p = pair(row(False, temperature=370.), row(False, temperature=350.))
    p["candidate_raw"]["minimumIgnitionVoltageObjective_V"] = 999999.
    r = screen_pair(p)
    assert r["R_T"] > 1 and r["R_V"] is None
    assert r["voltage_status"]["candidate"] == "right_censored"


def test_invalid_search_does_not_invalidate_converged_reference():
    p = pair(row(False, temperature=370.), row(False, temperature=350.))
    p["candidate_raw"]["minimumIgnitionVoltageSearchValid"] = False
    r = screen_pair(p)
    assert r["screening_valid"] and r["R_T"] > 1 and r["R_V"] is None


def test_left_censored_values_not_used_as_exact_voltages():
    p = pair(); p["candidate_raw"]["minimumIgnitionVoltageLeftCensored"] = True
    assert screen_pair(p)["R_V"] is None


def test_overlapping_voltage_brackets_not_called_lower_vmin():
    p = pair(row(v=157.), row(v=160.))
    r = screen_pair(p)
    assert r["R_V"] < 1 and r["voltage_comparison"] == "overlapping_brackets"
    assert "ignition_superior_candidates" not in r["labels"]


def test_missing_bracket_width_does_not_claim_resolved_gain():
    p = pair(); del p["candidate_raw"]["minimumIgnitionVoltageBracketWidth_V"]
    r = screen_pair(p)
    assert r["R_V"] is not None
    assert r["ignition_screen"] != "promising"


@pytest.mark.parametrize("temp", [298.15, 298.15000000001, None, float("nan"), float("inf")])
def test_bad_baseline_temperature_denominator(temp):
    r = screen_pair(pair(row(False, temperature=370.), row(False, temperature=temp)))
    assert r["R_T"] is None and "R_T" in r["ratio_reasons"]
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("j", [0., None, float("nan"), float("inf")])
def test_bad_congestion_denominator(j):
    r = screen_pair(pair(row(j=1.), row(j=j)))
    assert r["R_J"] is None and "R_J" in r["ratio_reasons"]
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("t", [None, 0., -1., 3., float("nan")])
def test_inconsistent_ignition_time(t):
    p = pair(); p["candidate_raw"]["ignitionDelay_s"] = t
    assert screen_pair(p)["comparison_state"] == "incomplete_observation"


def test_nonignition_does_not_imply_full_horizon_if_marked_incomplete():
    p = pair(row(False), row(False))
    p["candidate_raw"]["nativeExecution"]["completedFullHorizon"] = False
    assert screen_pair(p)["comparison_state"] == "incomplete_observation"


def test_policy_margin_and_no_mutation():
    p = pair(row(False, temperature=350.1), row(False, temperature=350.))
    old = copy.deepcopy(p)
    r = screen_pair(p, ScreeningPolicy(thermal_gain=.05))
    assert r["thermal_screen"] == "no_resolved_difference" and p == old


@pytest.mark.parametrize("kwargs", [{"time_gain":1}, {"thermal_gain":-1}, {"area_rtol":.02}, {"voltage_gain":float("nan")}])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):ScreeningPolicy(**kwargs)


def test_reports_have_no_global_rank_and_groups_overlap(tmp_path):
    r = screen_pair(pair())
    write_reports(tmp_path, [r])
    groups = json.loads((tmp_path/"screening_groups.json").read_text())
    assert groups["ignition_superior_candidates"] == groups["congestion_superior_candidates"] == ["E114"]
    manifest = json.loads((tmp_path/"fixed_25mm_revalidation_manifest.json").read_text())
    assert manifest[0]["status"] == "requires_redesign_and_revalidation"
    assert manifest[0]["feasibility_not_guaranteed"]
    assert not {"rank", "score", "pareto_front", "objective_vector"}.intersection(r)
    assert not list(tmp_path.glob("*pareto*"))


def test_reclassification_is_read_only(tmp_path):
    original=tmp_path/"result.json"
    original.write_text(json.dumps(pair(row(False, temperature=370.), row(False, temperature=350.))))
    before=original.read_bytes()
    out=tmp_path/"report"
    rows=reclassify([original], out)
    assert rows[0]["screening_valid"]
    assert original.read_bytes()==before
    assert json.loads((out/"reclassification_provenance.json").read_text())["pde_executed"] is False


def test_duplicate_result_ids_are_not_silently_overwritten(tmp_path):
    a=tmp_path/"a.json";b=tmp_path/"b.json"
    a.write_text(json.dumps(pair()));b.write_text(json.dumps(pair()))
    with pytest.raises(ValueError,match="Duplicate"):
        reclassify([a,b],tmp_path/"report")
