from __future__ import annotations

import numpy as np

from ecsp_nsga2.evaluator import canonicalise_metrics


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
