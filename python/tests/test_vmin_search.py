from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecsp_nsga2.evaluator import CppCondensedFp64Evaluator


@dataclass
class _Prepared:
    voltage: float
    output_dir: Path


class _Process:
    def poll(self):
        return 0


class _FakeRunner:
    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def prepare_case(self, anode, cathode, config_values, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return _Prepared(float(config_values["voltage"]), Path(output_dir))

    def launch_prepared_case(self, prepared):
        return _Process()

    def collect_prepared_case(self, prepared, process, started):
        ignited = prepared.voltage >= self.threshold
        return {
            "appliedVoltage_V": prepared.voltage,
            "ignitionSucceeded": ignited,
            "ignitionDelay_s": 1.5 if ignited else None,
            "converged": True,
            "finalElectricalConverged": True,
            "maximumTemperatureCapFraction": 0.0,
            "maximumSpeciesLimiterFraction": 0.0,
            "maximumGasCapFraction": 0.0,
            "maximumChemicalRateCapFraction": 0.0,
            "simulatedTime_s": 1.5 if ignited else 2.0,
            "terminatedAtIgnition": ignited,
            "wallClockTime_s": 0.001,
        }


def _reference_row(ignited: bool = True):
    return {
        "ignitionSucceeded": ignited,
        "ignitionDelay_s": 1.0 if ignited else None,
        "converged": True,
        "finalElectricalConverged": True,
        "maximumTemperatureCapFraction": 0.0,
        "maximumSpeciesLimiterFraction": 0.0,
        "maximumGasCapFraction": 0.0,
        "maximumChemicalRateCapFraction": 0.0,
    }


def test_vmin_bracketed_search_reports_conservative_igniting_bound(tmp_path: Path):
    evaluator = CppCondensedFp64Evaluator.__new__(CppCondensedFp64Evaluator)
    evaluator.vmin_enabled = True
    evaluator.vmin_lower_bound_V = 20.0
    evaluator.vmin_upper_bound_V = 260.0
    evaluator.vmin_tolerance_V = 5.0
    evaluator.vmin_max_iterations = 8
    evaluator.vmin_right_censor_penalty_V = 50.0
    evaluator.vmin_invalid_penalty_V = 100.0
    evaluator.vmin_stop_successful_trials_at_ignition = True
    evaluator.vmin_numerical_thresholds = {
        "temperature": 0.02,
        "species": 0.02,
        "gas": 0.02,
        "chemical_rate": 0.02,
    }
    evaluator.voltage = 260.0
    evaluator.cpp_parallel_cases = 2
    evaluator.cpp_config_values = {"voltage": 260.0}
    evaluator.cpp_runner = _FakeRunner(threshold=123.0)

    mask = np.zeros((8, 8), dtype=bool)
    prepared = {
        0: (mask, mask, {"geometry_id": "G0"}, tmp_path / "case")
    }
    results = [_reference_row(True)]
    evaluator._attach_minimum_ignition_voltage_search(prepared, results)

    row = results[0]
    assert row["minimumIgnitionVoltageSearchValid"] is True
    assert row["minimumIgnitionVoltageSearchStatus"] == "bracketed_converged"
    assert row["minimumIgnitionVoltageLowerNonIgnitingBound_V"] < 123.0
    assert row["minimumIgnitionVoltageUpperIgnitingBound_V"] >= 123.0
    assert row["minimumIgnitionVoltageBracketWidth_V"] <= 5.0 + 1e-12
    assert row["minimumIgnitionVoltageObjective_V"] == row["minimumIgnitionVoltageUpperIgnitingBound_V"]
    assert row["minimumIgnitionVoltageMonotonicityObserved"] is True
    assert (tmp_path / "case" / "condensed_metrics.json").is_file()


def test_vmin_right_censor_when_reference_does_not_ignite(tmp_path: Path):
    evaluator = CppCondensedFp64Evaluator.__new__(CppCondensedFp64Evaluator)
    evaluator.vmin_enabled = True
    evaluator.vmin_lower_bound_V = 20.0
    evaluator.vmin_upper_bound_V = 260.0
    evaluator.vmin_tolerance_V = 5.0
    evaluator.vmin_max_iterations = 8
    evaluator.vmin_right_censor_penalty_V = 50.0
    evaluator.vmin_invalid_penalty_V = 100.0
    evaluator.vmin_stop_successful_trials_at_ignition = True
    evaluator.vmin_numerical_thresholds = {
        "temperature": 0.02,
        "species": 0.02,
        "gas": 0.02,
        "chemical_rate": 0.02,
    }
    evaluator.voltage = 260.0
    evaluator.cpp_parallel_cases = 1
    evaluator.cpp_config_values = {"voltage": 260.0}
    evaluator.cpp_runner = _FakeRunner(threshold=300.0)

    mask = np.zeros((8, 8), dtype=bool)
    prepared = {0: (mask, mask, {"geometry_id": "G0"}, tmp_path / "case")}
    results = [_reference_row(False)]
    evaluator._attach_minimum_ignition_voltage_search(prepared, results)
    row = results[0]
    assert row["minimumIgnitionVoltageRightCensored"] is True
    assert row["minimumIgnitionVoltageSearchValid"] is True
    assert row["minimumIgnitionVoltage_V"] is None
    assert row["minimumIgnitionVoltageObjective_V"] == 310.0
