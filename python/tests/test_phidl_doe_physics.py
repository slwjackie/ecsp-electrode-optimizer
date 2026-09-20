"""DOE service integration tests: production equations are never executed."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ecsp_doe.physics import ProductionPhysicsAdapter
from ecsp_nsga2.propagation import ConfiguredModelTemperatureRangeExceeded


def raw_metrics():
    return {
        "ignitionDelay_s": 0.5,
        "remainingReactiveMassFractionAtEvaluationTime": 0.4,
        "minimumIgnitionVoltage_V": 100.0,
        "peakCurrentCongestionToEvaluationTime": 1.5,
        "ignitionSucceeded": True,
        "minimumIgnitionVoltageSearchValid": True,
        "maximumTemperatureCapFraction": 0.0,
        "maximumSpeciesLimiterFraction": 0.0,
        "maximumGasCapFraction": 0.0,
        "maximumChemicalRateCapFraction": 0.0,
        "solverConverged": True,
        "modelStatus": "bc_global_preflame",
    }


def config():
    return {
        "project": {"seed": 123, "device": "cpu"},
        "geometry": {"grid_size": 96, "maximum_width_mm": 5.0},
        "physics": {"voltage_V": 260.0, "end_time_s": 2.0},
        "evaluator": {"backend": "bc_global_preflame"},
        "bc_global": {"endTime_s": 2.0, "thermal": {"maximumTemperature_K": 2500.0}},
        "propagation_refinement": {"enabled": False, "duration_s": 0.1},
    }


class FakeEvaluator:
    grid_size = 193

    def __init__(self):
        self.calls = []
        self.handoff_calls = []
        self.config = {
            "bcGlobal": {"endTime_s": 2.0, "thermal": {
                "maximumTemperature_K": 2500.0, "density_kg_per_m3": 1000.0,
            }},
            "geometry": {"domainSize_m": 0.02},
            "transport": {"gasConstant_J_per_molK": 8.314},
        }
        self.result = raw_metrics()

    def evaluate(self, anode, cathode, metadata, output_dir):
        self.calls.append((anode.copy(), cathode.copy(), dict(metadata), output_dir))
        return copy.deepcopy(self.result)

    def evaluate_handoff(self, anode, cathode, metadata, output_dir):
        self.handoff_calls.append((dict(metadata), output_dir))
        return {"handoff": {"onsetSucceeded": True}, "metrics": self.result.copy()}


def adapter(tmp_path):
    evaluator = FakeEvaluator()
    return ProductionPhysicsAdapter(config(), Path(__file__).parents[2], tmp_path,
                                    evaluator=evaluator), evaluator


def record(geometry_id="S1_T000_V000"):
    a = np.zeros((8, 8), dtype=bool)
    a[1:6, 1:3] = True
    c = np.fliplr(a)
    from ecsp_nsga2.evaluator import canonicalise_metrics
    metrics = canonicalise_metrics(raw_metrics(), 2.0, 2.0)
    return {
        "metadata": {"geometry_id": geometry_id, "topology_id": "T000",
                     "component_pair": [1, 1]},
        "anode_mask": a, "cathode_mask": c, "metrics": metrics,
    }


def baseline_pair():
    r = record("AREA_MATCHED_STAGGERED")
    r["metadata"].update(topology_id="BASELINE", component_pair=[2, 2])
    return ProductionPhysicsAdapter._individual(r)


def frozen_selection(tmp_path, records):
    final = tmp_path / "final"
    final.mkdir()
    path = final / "preflame_top20.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["geometry_id", "topology_id"])
        writer.writeheader()
        writer.writerows({key: row["metadata"][key] for key in writer.fieldnames}
                         for row in records)
    return final, path.read_bytes()


def success_post(*args, **kwargs):
    return {
        "status": "complete", "onsetSucceeded": True,
        "finalUnreactedAreaFraction": 0.1,
        "establishedTimeAfterOnset_s": 0.01,
        "meanEffectiveRegressionVelocity_m_per_s": 0.002,
        "reactionFrontNonuniformity": 0.1,
    }


def test_preflame_exactly_one_call_each_and_preserves_metrics(tmp_path):
    service, evaluator = adapter(tmp_path)
    r = record()
    for index in range(5):
        out = service.evaluate(r["anode_mask"], r["cathode_mask"], r["metadata"], tmp_path / str(index))
        assert out["physics_success"] is True
        assert out["objective_vector"] == [0.5, 0.4, 100.0, 1.5]
        for key, value in raw_metrics().items():
            assert out[key] == value
    assert len(evaluator.calls) == 5
    assert evaluator.handoff_calls == []
    assert all(call[2]["intended_anode_components"] == 1 for call in evaluator.calls)
    assert all(call[2]["intended_cathode_components"] == 1 for call in evaluator.calls)
    assert "intended_anode_components" not in r["metadata"]


def test_asymmetric_or_missing_component_metadata_fails_before_physics(tmp_path):
    service, evaluator = adapter(tmp_path)
    r = record()
    for pair in ([1, 2], "1A2C", None):
        r["metadata"]["component_pair"] = pair
        with pytest.raises(Exception, match="symmetric component_pair"):
            service.evaluate(r["anode_mask"], r["cathode_mask"], r["metadata"], tmp_path)
    assert evaluator.calls == []


def test_numerical_failure_is_explicit_nonignition_keeps_existing_penalty(tmp_path):
    service, evaluator = adapter(tmp_path)
    r = record()
    evaluator.result["solverConverged"] = False
    failed = service.evaluate(r["anode_mask"], r["cathode_mask"], r["metadata"], tmp_path)
    assert failed["physics_success"] is False
    assert "production_numerical_rejection" in failed["doe_failure_reason"]
    evaluator.result = raw_metrics()
    evaluator.result["ignitionSucceeded"] = False
    out = service.evaluate(r["anode_mask"], r["cathode_mask"], r["metadata"], tmp_path)
    assert out["physics_success"] is True
    assert out["ignition_delay_s"] == 4.0
    assert out["ignition_success"] is False


def test_invalid_voltage_search_is_a_failed_result(tmp_path):
    service, evaluator = adapter(tmp_path)
    evaluator.result["minimumIgnitionVoltageSearchValid"] = False
    r = record()
    out = service.evaluate(r["anode_mask"], r["cathode_mask"], r["metadata"], tmp_path)
    assert out["physics_success"] is False
    assert out["doe_failure_reason"] == "invalid_minimum_ignition_voltage_search"


def test_baseline_uses_existing_generator_target_and_resumes(tmp_path):
    service, evaluator = adapter(tmp_path)
    final = tmp_path / "final"
    baseline, raster = service.evaluate_baseline(final)
    again, persisted = service.evaluate_baseline(final)
    assert len(evaluator.calls) == 1
    assert baseline.geometry_id == again.geometry_id
    assert baseline.metrics["area_match_target_per_polarity"] == service.limits.target_area_fraction_per_polarity
    assert baseline.metrics["included_in_final_pareto_front"] is False
    assert evaluator.calls[0][2]["baseline_type"] == "area_matched_staggered"
    assert evaluator.calls[0][2]["reference_target_area_fraction_per_polarity"] == service.limits.target_area_fraction_per_polarity
    assert evaluator.calls[0][2]["propagation_refinement"] is False
    np.testing.assert_array_equal(raster.anode_mask, persisted.anode_mask)
    geometry = json.loads((final / "area_matched_staggered.json").read_text())
    assert geometry["anode"]["components"]
    assert geometry["geometry_hash"]
    for suffix in ("json", "npz", "png"):
        assert (final / f"area_matched_staggered.{suffix}").is_file()
    service.write_preflame_comparison([record()], (baseline, raster), final)
    with (final / "preflame_top20_vs_staggered.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 4


def test_refines_only_frozen_candidates_plus_one_baseline(tmp_path, monkeypatch):
    service, evaluator = adapter(tmp_path)
    selected = [record("A"), record("B")]
    final, before = frozen_selection(tmp_path, selected)
    monkeypatch.setattr("ecsp_nsga2.propagation.run_condensed_propagation", success_post)
    rows = service.refine_selected(selected, baseline_pair(), final)
    assert len(rows) == 3
    assert [m["geometry_id"] for m, _ in evaluator.handoff_calls] == ["A", "B", "AREA_MATCHED_STAGGERED"]
    assert evaluator.handoff_calls[-1][0]["baseline_type"] == "area_matched_staggered"
    assert (final / "preflame_top20.csv").read_bytes() == before
    assert "propagationSucceeded" not in selected[0]["metrics"]
    with (final / "post_onset_valid_refined_ranking.csv").open() as stream:
        assert {r["geometry_id"] for r in csv.DictReader(stream)} == {"A", "B"}
    with (final / "propagation_comparison.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 8
    repeated = service.refine_selected(selected, baseline_pair(), final)
    assert len(evaluator.handoff_calls) == 3
    assert all(row["reused_post_onset_result"] is True for row in repeated)
    assert (final / "preflame_top20.csv").read_bytes() == before


def test_interrupted_post_resume_skips_completed_candidates(tmp_path, monkeypatch):
    service, evaluator = adapter(tmp_path)
    selected = [record("A"), record("B")]
    final, before = frozen_selection(tmp_path, selected)
    count = []

    def interrupted(*args, **kwargs):
        count.append(1)
        if len(count) == 2:
            raise RuntimeError("simulated process interruption")
        return success_post()

    monkeypatch.setattr("ecsp_nsga2.propagation.run_condensed_propagation", interrupted)
    with pytest.raises(RuntimeError, match="process interruption"):
        service.refine_selected(selected, baseline_pair(), final)
    monkeypatch.setattr("ecsp_nsga2.propagation.run_condensed_propagation", success_post)
    rows = service.refine_selected(selected, baseline_pair(), final)
    assert [metadata["geometry_id"] for metadata, _ in evaluator.handoff_calls] == ["A", "B", "B", "AREA_MATCHED_STAGGERED"]
    assert [row["reused_post_onset_result"] for row in rows] == [True, False, False]
    assert (final / "preflame_top20.csv").read_bytes() == before


@pytest.mark.parametrize("onset", [True, False])
def test_unestablished_post_checkpoint_preserves_nonfinite_diagnostics(tmp_path, monkeypatch, onset):
    service, evaluator = adapter(tmp_path)
    selected = [record("A")]
    final, before = frozen_selection(tmp_path, selected)

    def unestablished(*args, **kwargs):
        result = success_post()
        result.update(
            onsetSucceeded=onset, establishedTimeCensored=True,
            # Successful onset with no establishment retains the existing
            # finite-duration censor. A no-onset diagnostic can be nonfinite.
            establishedTimeAfterOnset_s=0.1 if onset else float("nan"),
            diagnostics={"unset": None, "values": [float("nan"), float("inf"), -float("inf")]},
        )
        return result

    monkeypatch.setattr("ecsp_nsga2.propagation.run_condensed_propagation", unestablished)
    first = service.refine_selected(selected, baseline_pair(), final)
    previous_comparison = (final / "propagation_comparison.csv").read_bytes()
    resumed = service.refine_selected(selected, baseline_pair(), final)
    assert len(evaluator.handoff_calls) == 2
    assert (final / "propagation_comparison.csv").read_bytes() == previous_comparison
    assert (final / "preflame_top20.csv").read_bytes() == before
    for original, repeated in zip(first, resumed):
        assert original["propagationSucceeded"] == repeated["propagationSucceeded"]
        assert repeated["reused_post_onset_result"] is True
        assert repeated["diagnostics"]["unset"] is None
        assert np.isnan(repeated["diagnostics"]["values"][0])
        assert repeated["diagnostics"]["values"][1:] == [float("inf"), -float("inf")]
        if onset:
            assert repeated["establishedTimeAfterOnset_s"] == 0.1
        else:
            assert np.isnan(repeated["establishedTimeAfterOnset_s"])
    saved = json.loads((final / "propagation/A/result_checkpoint.json").read_text())
    assert saved["nonfinite_metric_values"]
    assert saved["metrics"]["diagnostics"]["values"] == [None, None, None]


def test_extra_post_candidate_fails_closed_before_any_solver_call(tmp_path):
    service, evaluator = adapter(tmp_path)
    selected = [record("A")]
    final, before = frozen_selection(tmp_path, selected)
    with pytest.raises(ValueError, match="immutable"):
        service.refine_selected(selected + [record("PROMOTED")], baseline_pair(), final)
    assert evaluator.handoff_calls == []
    assert (final / "preflame_top20.csv").read_bytes() == before


def test_same_production_model_validity_for_candidate_and_baseline(tmp_path, monkeypatch):
    service, evaluator = adapter(tmp_path)
    selected = [record("A"), record("B")]
    final, before = frozen_selection(tmp_path, selected)
    observed_caps = []

    def invalid_post(handoff, propagation, bc, output):
        observed_caps.append(bc["thermal"]["maximumTemperature_K"])
        raise ConfiguredModelTemperatureRangeExceeded(2500.0, 2500.001, 7, 1.4e-10)

    monkeypatch.setattr("ecsp_nsga2.propagation.run_condensed_propagation", invalid_post)
    rows = service.refine_selected(selected, baseline_pair(), final)
    assert observed_caps == [2500.0, 2500.0, 2500.0]
    assert len(evaluator.handoff_calls) == 3
    assert all(row["postOnsetModelValid"] is False for row in rows)
    assert all(row["postOnsetValidityReason"] == "configured_model_temperature_range_exceeded" for row in rows)
    assert (final / "preflame_top20.csv").read_bytes() == before
    with (final / "post_onset_valid_refined_ranking.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
    with (final / "propagation_rejections.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    service.refine_selected(selected, baseline_pair(), final)
    assert len(evaluator.handoff_calls) == 3
    assert observed_caps == [2500.0, 2500.0, 2500.0]


def test_mock_injection_never_constructs_production_workflow(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("DOE must never initialize NSGA-II orchestration")
    monkeypatch.setattr("ecsp_nsga2.workflow.NSGA2ElectricalSolidWorkflow.__init__", forbidden)
    service, evaluator = adapter(tmp_path)
    assert service.evaluator is evaluator


def test_production_factory_receives_unmodified_physics_config(tmp_path, monkeypatch):
    cfg = config()
    original = copy.deepcopy(cfg)
    fake = FakeEvaluator()
    captured = []

    def factory(package_root, evaluator_cfg, workdir, allow_debug):
        captured.append((evaluator_cfg, allow_debug))
        return fake

    monkeypatch.setattr("ecsp_doe.physics.create_evaluator", factory)
    service = ProductionPhysicsAdapter(cfg, Path(__file__).parents[2], tmp_path)
    assert service.evaluator is fake
    assert cfg == original
    assert captured[0][0]["physics_config"] == original
    assert captured[0][1] is False


def test_approximate_backend_is_forbidden(tmp_path):
    cfg = config()
    cfg["evaluator"]["backend"] = "analytic_debug"
    with pytest.raises(Exception, match="production B/C"):
        ProductionPhysicsAdapter(cfg, Path(__file__).parents[2], tmp_path)
