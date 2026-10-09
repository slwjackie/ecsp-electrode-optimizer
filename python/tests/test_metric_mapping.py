from __future__ import annotations

import numpy as np
import pytest

from ecsp_preflame.evaluator import canonicalise_metrics


def test_canonical_four_objectives() -> None:
    result = canonicalise_metrics(
        {
            "condensedPhaseIgnitionDelay_s": 0.31,
            "ignitionSucceeded": True,
            "areaAveragedUndecomposedFractionAt2s": 0.42,
            "minimumIgnitionVoltageObjective_V": 112.5,
            "minimumIgnitionVoltage_V": 112.5,
            "minimumIgnitionVoltageSearchValid": True,
            "inputElectricalEnergyToIgnition_J": 7.5,
            "peakCurrentCongestion": 2.25,
        },
        end_time_s=2.0,
        no_ignition_penalty_s=2.0,
    )
    assert result["objective_vector"] == [0.31, 0.42, 112.5, 2.25]
    assert result["ignition_success"] is True
    assert result["minimumIgnitionVoltageSearchValid"] is True
    assert result["inputElectricalEnergyToIgnition_J"] == 7.5
    assert result["objective_definition"]["flame_progress_used"] is False


def test_no_ignition_is_finite_and_right_censored_not_nan() -> None:
    result = canonicalise_metrics(
        {
            "condensedPhaseIgnitionDelay_s": np.nan,
            "ignitionSucceeded": False,
            "areaAveragedUndecomposedFractionAt2s": 0.95,
            "minimumIgnitionVoltageObjective_V": 310.0,
            "minimumIgnitionVoltage_V": None,
            "minimumIgnitionVoltageSearchValid": True,
            "minimumIgnitionVoltageRightCensored": True,
            "peakCurrentCongestion": 3.0,
        },
        end_time_s=2.0,
        no_ignition_penalty_s=2.0,
    )
    assert result["ignition_delay_s"] == 4.0
    assert np.all(np.isfinite(result["objective_vector"]))
    assert result["objective_vector"][2] == 310.0
    assert result["minimumIgnitionVoltageRightCensored"] is True
    assert result["ignition_success"] is False


@pytest.mark.parametrize("model_status", [
    "electrochemical_thermal_decomposition_literature_nominal_uncalibrated",
    "bc_global_corrected_surface_overlay_preflame_literature_nominal_uncalibrated",
    "preflame_torch_corrected_surface_overlay_preflame_literature_nominal_uncalibrated",
    "preflame_model_corrected_surface_overlay_preflame_literature_nominal_uncalibrated",
])
def test_model_rename_preserves_reactive_mass_objective_definition(model_status):
    result = canonicalise_metrics(
        {"modelStatus": model_status, "ignitionSucceeded": False,
         "remainingReactiveMassFractionAtEvaluationTime": 0.73,
         "minimumIgnitionVoltageObjective_V": 310.0,
         "peakCurrentCongestionToEvaluationTime": 2.0},
        end_time_s=2.0, no_ignition_penalty_s=2.0)
    definition = result["objective_definition"]
    assert definition["preflame_model_species_mass_objective"] is True
    assert "remaining reactive LP-plus-PVA" in definition["area_undecomposed_fraction_at_evaluation_time"]
    assert result["objective_vector"][1] == 0.73


@pytest.mark.parametrize("model_status", [
    "electrochemical_thermal_decomposition_literature_nominal_uncalibrated",
    "preflame_torch_literature_nominal_uncalibrated",
    "bc_global_literature_nominal_uncalibrated",
])
def test_model_rename_preserves_conjunctive_onset_area_constraint(model_status):
    from types import SimpleNamespace
    from ecsp_preflame.evaluation_workflow import EvaluationWorkflow
    from ecsp_preflame.candidate_ranking import CandidateEvaluation
    workflow = SimpleNamespace(
        end_time_s=2.0, no_ignition_penalty_s=2.0, opt={},
        minimum_no_ignition_violation=1e-6, no_ignition_constraint_violation=1.0,
        ignition_onset_temperature_K=523.15, initial_temperature_K=298.15,
        vmin_invalid_search_constraint_violation=1.0,
        _numerical_violation=lambda _: 0.0)
    candidate = CandidateEvaluation("unreacted_hotspot", {})
    EvaluationWorkflow._apply_raw_metrics(workflow, candidate, {
        "modelStatus": model_status, "ignitionSucceeded": False,
        "peakMaximumTemperature_K": 600.0, "temperatureOnsetAreaFractionAt2s": 0.0025,
        "ignitionMinimumAreaFraction": 0.01, "minimumIgnitionVoltageSearchValid": True,
        "remainingReactiveMassFractionAtEvaluationTime": 1.0,
        "minimumIgnitionVoltageObjective_V": 310.0,
        "peakCurrentCongestionToEvaluationTime": 2.0})
    assert candidate.metrics["ignition_constraint_violation"] == 0.75
