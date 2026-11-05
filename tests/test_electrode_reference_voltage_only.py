"""Skipping Vmin preserves reference physics and never creates a voltage score."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import pytest
import yaml

import run_electrode_parameter_study as study
from ecsp_preflame.evaluator import ElectrochemicalThermalDecompositionEvaluator, EvaluatorError


ROOT = Path(__file__).resolve().parents[1]


def base_config():
    raw = yaml.safe_load((ROOT / study.PHYSICS_DEFAULT).read_text())
    dt = raw["preflame_model"]["timeStep_s"]
    for key in ("physics", "evaluator"):
        raw[key].update(end_time_s=dt, metric_evaluation_time_s=dt)
    raw["condensed_ignition"]["reference_time_s"] = dt
    raw["preflame_model"].update(endTime_s=dt, evaluationTime_s=dt,
                                  electricalUpdateInterval_s=dt)
    raw["evaluator"]["cpp_cuda"]["cpu_runtime"] = "standalone"
    return raw


def test_skip_changes_only_search_and_active_objective():
    base = base_config()
    original = copy.deepcopy(base)
    default = study.runtime_config(base, 100, "cpu", 1)
    skipped = study.runtime_config(base, 100, "cpu", 1, skip_vmin=True)
    expected = copy.deepcopy(default)
    expected["minimum_ignition_voltage_search"]["enabled"] = False
    expected["evaluation"]["objectives_minimise"].remove("minimum_ignition_voltage_V")
    assert skipped == expected
    assert base == original
    assert default["minimum_ignition_voltage_search"]["enabled"] is True


def test_general_workflow_still_requires_voltage_search(tmp_path):
    cfg = study.runtime_config(base_config(), 100, "cpu", 1, skip_vmin=True)
    adapter = dict(cfg["evaluator"], physics_config=cfg)
    with pytest.raises(EvaluatorError, match="enabled must be true"):
        ElectrochemicalThermalDecompositionEvaluator(ROOT, adapter, tmp_path)


def test_study_cannot_disable_an_active_voltage_objective(tmp_path):
    cfg = study.runtime_config(base_config(), 100, "cpu", 1)
    cfg["minimum_ignition_voltage_search"]["enabled"] = False
    with pytest.raises(EvaluatorError, match="active objective"):
        study.create_study_evaluator(ROOT, cfg, tmp_path)


@pytest.mark.parametrize("ignited,invalid", [(True, False), (False, False), (False, True)])
def test_skip_never_enters_scheduler_or_keeps_penalty(tmp_path, monkeypatch, ignited, invalid):
    import ecsp_preflame.cpp_cuda_evaluator as native
    cfg = study.runtime_config(base_config(), 100, "cpu", 1, skip_vmin=True)
    ev = study.create_study_evaluator(ROOT, cfg, tmp_path / "adapter")
    spec = yaml.safe_load((ROOT / "config/electrode_inclination_study_25mm.yaml").read_text())
    item = (*study.make_inclined_case(spec["cases"][0], 100), tmp_path / "case")
    def fail(*args, **kwargs):
        raise AssertionError("Skipped Vmin must never enter the scheduler")
    monkeypatch.setattr(native, "batched_voltage_search", fail)
    calls = []
    def reference(items, volts):
        calls.append(volts)
        return [dict(geometry_id=item[2]["geometry_id"], ignitionSucceeded=ignited,
                     physicsRejected=invalid, ignitionDelay_s=.001 if ignited else None,
                     peakMaximumTemperature_K=530., minimumIgnitionVoltageObjective_V=1e12)], []
    monkeypatch.setattr(ev, "_safe_run", reference)
    row, = ev.evaluate_batch([item])
    assert calls == [[260.]]
    assert row["minimumIgnitionVoltageTrialCount"] == 0
    assert row["minimumIgnitionVoltageTrials"] == []
    assert row["minimumIgnitionVoltageSearchStatus"] == "not_requested_reference_voltage_only"
    assert row["minimumIgnitionVoltageSearchValid"] is None
    assert row["minimumIgnitionVoltageObjective_V"] is None
    assert row["peakMaximumTemperature_K"] == 530.
    row["external_numerical_valid"] = not invalid
    summary, = study.summary_rows([dict(geometry=item[2], raw=row)])
    assert summary["minimumIgnitionVoltage_V"] is None
    assert summary["result_state"] == ("numerically_invalid" if invalid else
        "condensed_onset" if ignited else "no_onset_within_horizon")
    assert summary["ignitionDelay_s"] == (.001 if ignited and not invalid else None)
    assert not (item[3] / "vmin_search").exists()


@pytest.mark.parametrize("rectangular", [False, True])
def test_real_reference_parity_outputs_and_resume(tmp_path, rectangular):
    filename = ("electrode_inclination_study_12x25mm.yaml" if rectangular
                else "electrode_inclination_study_25mm.yaml")
    spec = yaml.safe_load((ROOT / "config" / filename).read_text())
    spec["grid_size"] = 100
    spec["cases"] = spec["cases"][-2:] if rectangular else spec["cases"][:2]
    physics = tmp_path / "one_step.yaml"
    physics.write_text(yaml.safe_dump(base_config()))
    output = {}
    for skip in (False, True):
        folder = tmp_path / str(skip)
        study.generate(spec, folder)
        rows = study.evaluate(ROOT, folder, physics, "cpu", 2, False, skip_vmin=skip)
        output[skip] = rows
        if not skip:
            continue
        assert study.evaluate(ROOT, folder, physics, "cpu", 2, True, skip_vmin=True) == rows
        with pytest.raises(ValueError, match="Existing/changed"):
            study.evaluate(ROOT, folder, physics, "cpu", 2, True, skip_vmin=False)
        waves = [json.loads(line) for line in
                 (folder / "adapter/cpp_cuda_batch_waves.jsonl").read_text().splitlines()]
        assert len(waves) == 1 and waves[0]["role"] == "reference"
        assert waves[0]["voltages_V"] == [260., 260.]
        assert not list(folder.glob("cases/*/vmin_search"))
        diagnostic = json.loads((folder / "adapter/evaluator_adapter_diagnostics.json").read_text())
        assert diagnostic["vmin_search"] == "not_requested_reference_voltage_only"
        with (folder / "inclination_summary.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert all(row["minimumIgnitionVoltage_V"] == "" for row in csv_rows)
        assert all(row["minimumIgnitionVoltageSearchValid"] == "" for row in csv_rows)
        assert all(row["referenceAppliedVoltage_V"] == "260.0" for row in csv_rows)
    for default, skipped in zip(output[False], output[True]):
        assert default["external_numerical_valid"] and skipped["external_numerical_valid"]
        for key in ("peakMaximumTemperature_K", "peakCurrentCongestionToEvaluationTime",
                    "inputElectricalEnergyAtEvaluationTime_J",
                    "areaAveragedUndecomposedFractionAtEvaluationTime", "result_state"):
            assert default[key] == skipped[key]


def test_cli_forwards_explicit_skip_flag(tmp_path, monkeypatch):
    seen = {}
    def evaluate(*args, **kwargs):
        seen.update(kwargs)
        return []
    monkeypatch.setattr(study, "evaluate", evaluate)
    monkeypatch.setattr(sys, "argv", ["study", "evaluate", "--out", str(tmp_path), "--skip-vmin"])
    assert study.main() == 0
    assert seen["skip_vmin"] is True


def test_cli_rejects_skip_during_geometry_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["study", "generate", "--out", str(tmp_path), "--skip-vmin"])
    with pytest.raises(SystemExit) as exc:
        study.main()
    assert exc.value.code == 2
