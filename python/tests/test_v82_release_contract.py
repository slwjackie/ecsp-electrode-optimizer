from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from ecsp_preflame.voltage_search import (
    _monotonicity_violation,
    batched_voltage_search,
    validate_voltage_search_contract,
)
from ecsp_preflame.evaluator import (
    PreflameCandidateGeometryError,
    ElectrochemicalThermalDecompositionEvaluator,
    EvaluatorError,
    _bc_solver_converged,
    canonicalise_metrics,
)
from ecsp_preflame.candidate_ranking import CandidateEvaluation
from ecsp_preflame.evaluation_workflow import EvaluationWorkflow
from ecsp_v6.config import load_config
from ecsp_preflame.electrochemical_thermal_decomposition import PreflameCandidateBatchError
from ecsp_v6.physics.composition_model import build_composition
from ecsp_v6.physics.electrochem import initial_state
from ecsp_v6.physics.geometry import load_geometry_batch
from verify_release_manifest import verify


ROOT = Path(__file__).resolve().parents[2]
PREFLAME_PROFILE_NAMES = (
    "preflame_torch_propagation_a100.yaml",
    "preflame_torch_propagation_debug.yaml",
    "preflame_cpp_cuda_cpu8.yaml",
    "preflame_cpp_cuda_a100_batch32.yaml",
    "preflame_cpp_cuda_a100_batch64.yaml",
    "preflame_cpp_cuda_debug.yaml",
)


def _profile(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def _geometry_batch() -> object:
    """Build through the production resize path, without constructing a solver."""
    evaluator = object.__new__(ElectrochemicalThermalDecompositionEvaluator)
    evaluator.grid_size = 17
    evaluator.domain_size_m = 0.020
    evaluator.minimum_gap_m = 0.002
    evaluator.device = torch.device("cpu")
    anode = np.zeros((17, 17), dtype=bool)
    cathode = np.zeros((17, 17), dtype=bool)
    anode[3:7, 2:5] = True
    cathode[10:14, 12:15] = True
    return evaluator._build_geometry_batch(
        [(anode, cathode, {"geometry_id": "overlay-contract"}, ROOT / "unused")]
    )


def _synthetic_bc_batch_evaluator(tmp_path: Path) -> ElectrochemicalThermalDecompositionEvaluator:
    evaluator = object.__new__(ElectrochemicalThermalDecompositionEvaluator)
    evaluator.internal_batch_size = 8
    evaluator._failed_geometry_row = lambda metadata, output, exc: {
        "geometry_id": metadata["geometry_id"],
        "physicsRejected": True,
        "physicsRejectionReason": str(exc),
    }
    return evaluator


def _synthetic_bc_items(tmp_path: Path, count: int = 4) -> list[tuple]:
    return [
        (None, None, {"geometry_id": f"candidate-{index}"}, tmp_path / str(index))
        for index in range(count)
    ]


def test_reference_bc_batch_propagates_unclassified_device_failure(
    tmp_path: Path,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)

    def fail_device(items):
        del items
        raise RuntimeError("simulated device loss")

    evaluator._evaluate_chunk = fail_device
    with pytest.raises(RuntimeError, match="simulated device loss"):
        evaluator.evaluate_batch(_synthetic_bc_items(tmp_path))


@pytest.mark.parametrize(
    "exception_factory",
    (
        lambda index: PreflameCandidateBatchError(
            "typed candidate physics failure", (index,), "test_physics"
        ),
        lambda index: PreflameCandidateGeometryError(
            f"typed candidate geometry failure at lane {index}"
        ),
    ),
)
def test_reference_bc_batch_isolates_only_typed_candidate_failure(
    tmp_path: Path, exception_factory,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)

    def isolate(items):
        bad = [
            index
            for index, item in enumerate(items)
            if item[2]["geometry_id"] == "candidate-2"
        ]
        if bad:
            raise exception_factory(bad[0])
        return [
            {"geometry_id": item[2]["geometry_id"], "physicsRejected": False}
            for item in items
        ]

    evaluator._evaluate_chunk = isolate
    rows = evaluator.evaluate_batch(_synthetic_bc_items(tmp_path))
    assert [row["geometry_id"] for row in rows] == [
        "candidate-0",
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]
    assert [row["physicsRejected"] for row in rows] == [False, False, True, False]


def test_reference_bc_batch_oom_splits_but_singleton_oom_is_fatal(
    tmp_path: Path,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)
    visits: list[str] = []

    def split_oom(items):
        if len(items) > 1:
            raise torch.OutOfMemoryError("synthetic allocation exhaustion")
        visits.append(items[0][2]["geometry_id"])
        return [{"geometry_id": visits[-1], "physicsRejected": False}]

    evaluator._evaluate_chunk = split_oom
    rows = evaluator.evaluate_batch(_synthetic_bc_items(tmp_path))
    assert [row["geometry_id"] for row in rows] == [
        "candidate-0",
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]

    def singleton_oom(items):
        del items
        raise torch.OutOfMemoryError("synthetic singleton exhaustion")

    evaluator._evaluate_chunk = singleton_oom
    with pytest.raises(torch.OutOfMemoryError, match="singleton exhaustion"):
        evaluator.evaluate_batch(_synthetic_bc_items(tmp_path, count=1))


def test_reference_handoff_batch_isolates_typed_candidate_failure(
    tmp_path: Path,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)
    evaluator.voltage = 260.0

    def run_model(items, voltage, **kwargs):
        del voltage, kwargs
        bad = [
            index
            for index, item in enumerate(items)
            if item[2]["geometry_id"] == "candidate-2"
        ]
        if bad:
            raise PreflameCandidateBatchError(
                "typed handoff failure", bad, "handoff_test"
            )
        return ([{"geometry_id": item[2]["geometry_id"]} for item in items], {})

    evaluator._run_model = run_model
    evaluator._extract_handoff_batch = lambda rows, output, items: [
        {
            "metrics": row,
            "handoff": {"marker": item[2]["geometry_id"]},
        }
        for row, item in zip(rows, items)
    ]
    results = evaluator.evaluate_handoff_batch(_synthetic_bc_items(tmp_path))
    assert [
        result.get("handoffPreparationFailed", False) for result in results
    ] == [False, False, True, False]
    assert results[0]["handoff"]["marker"] == "candidate-0"
    assert results[3]["handoff"]["marker"] == "candidate-3"


def test_reference_handoff_batch_propagates_infrastructure_failure(
    tmp_path: Path,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)
    evaluator.voltage = 260.0

    def fail(items, voltage, **kwargs):
        del items, voltage, kwargs
        raise RuntimeError("simulated handoff device loss")

    evaluator._run_model = fail
    with pytest.raises(RuntimeError, match="handoff device loss"):
        evaluator.evaluate_handoff_batch(_synthetic_bc_items(tmp_path))


def test_all_bc_profiles_use_one_surface_overlay_contract() -> None:
    for name in PREFLAME_PROFILE_NAMES:
        cfg = _profile(name)
        assert cfg["geometry"]["surface_contact_model"] == (
            "overlay_on_full_propellant_domain"
        ), name
        assert cfg["evaluator"]["base_overrides"]["interface"][
            "boundaryCouplingModel"
        ] == "surface_overlay_bv", name
        assert cfg["preflame_model"]["continuedElectricalHeatingAfterOnset"] is False, name
        assert cfg["propagation_refinement"]["continued_electrical_heating"] is False, name


def test_debug_profiles_label_their_actual_evaluation_horizon() -> None:
    for name in (
        "preflame_cpp_cuda_debug.yaml",
        "preflame_torch_propagation_debug.yaml",
    ):
        cfg = _profile(name)
        horizon = float(cfg["preflame_model"]["evaluationTime_s"])
        assert float(cfg["physics"]["end_time_s"]) == horizon, name
        assert float(cfg["condensed_ignition"]["reference_time_s"]) == horizon, name
        assert "within_2s" not in cfg["minimum_ignition_voltage_search"][
            "interpretation"
        ], name














def test_onset_and_initial_temperature_have_one_effective_value() -> None:
    base = load_config(ROOT / "config" / "default_lp_pva.yaml")
    base_t0 = float(base["thermal"]["initialTemperature_K"])
    assert float(base["electrical"]["initialTemperature_K"]) == base_t0
    for name in PREFLAME_PROFILE_NAMES:
        cfg = _profile(name)
        assert float(cfg["condensed_ignition"]["onset_temperature_K"]) == float(
            cfg["preflame_model"]["onsetCriterion"]["temperature_K"]
        ), name
        assert float(
            cfg["condensed_ignition"]["minimum_conversion_numerical_guard"]
        ) == float(cfg["preflame_model"]["onsetCriterion"]["minimum_progress"]), name
        assert float(cfg["preflame_model"]["thermal"]["initialTemperature_K"]) == base_t0, name


def test_conflicting_bc_initial_temperature_fails_closed(tmp_path: Path) -> None:
    cfg = copy.deepcopy(_profile("preflame_cpp_cuda_debug.yaml"))
    cfg["evaluator"]["base_overrides"].setdefault("thermal", {})[
        "initialTemperature_K"
    ] = 310.0
    adapter = copy.deepcopy(cfg["evaluator"])
    adapter["physics_config"] = cfg
    with pytest.raises(Exception, match="Conflicting initial temperatures"):
        ElectrochemicalThermalDecompositionEvaluator(
            ROOT, adapter, tmp_path / "conflicting-temperature"
        )


def test_solver_geometry_keeps_propellant_and_species_under_contacts() -> None:
    geometry = _geometry_batch()
    assert bool(torch.all(geometry.propellant))
    assert int(geometry.propellant.sum()) == geometry.grid_size**2

    config = load_config(ROOT / "config" / "default_lp_pva.yaml")
    composition = build_composition(config)
    state = initial_state(geometry, config, composition, 260.0, torch.float64)
    contacts = geometry.anode | geometry.cathode
    assert bool(torch.all(state["cation"][contacts] > 0.0))
    assert bool(torch.all(state["anion"][contacts] > 0.0))
    assert bool(torch.all(state["water"][contacts] > 0.0))
    assert bool(
        torch.all(
            state["temperature"][contacts]
            == float(config["thermal"]["initialTemperature_K"])
        )
    )


def test_legacy_direct_geometry_loader_keeps_embedded_electrode_semantics(
    tmp_path: Path,
) -> None:
    anode = np.zeros((17, 17), dtype=np.uint8)
    cathode = np.zeros((17, 17), dtype=np.uint8)
    anode[3:7, 2:5] = 1
    cathode[10:14, 12:15] = 1
    path = tmp_path / "legacy_geometry.npz"
    np.savez_compressed(path, anodeMask=anode, cathodeMask=cathode)
    rows = pd.DataFrame(
        [{"geometry_id": "legacy-direct", "npz_path": str(path)}]
    )
    geometry, _ = load_geometry_batch(
        rows,
        grid_size=17,
        domain_size_m=0.020,
        minimum_gap_m=0.002,
        reject_resize_shorts=True,
        device=torch.device("cpu"),
        manufacturability_cfg={"projectConstraintsAfterResize": False},
    )
    contacts = geometry.anode | geometry.cathode
    assert torch.equal(geometry.fixed, contacts)
    assert torch.equal(geometry.propellant, ~contacts)


def _objective_row(**extra: float | bool) -> dict:
    row: dict = {
        "condensedPhaseIgnitionDelay_s": 0.2,
        "ignitionSucceeded": True,
        "minimumIgnitionVoltageObjective_V": 100.0,
        "minimumIgnitionVoltageSearchValid": True,
        "peakCurrentCongestion": 2.0,
        "evaluationTime_s": 0.25,
        "modelStatus": "preflame_model_corrected",
    }
    row.update(extra)
    return row


def test_evaluation_time_metric_is_canonical_and_at2s_is_deprecated_alias() -> None:
    result = canonicalise_metrics(
        _objective_row(
            remainingReactiveMassFractionAtEvaluationTime=0.37,
            remainingReactiveMassFractionAt2s=0.91,
        ),
        end_time_s=0.25,
        no_ignition_penalty_s=0.25,
    )
    assert result["area_undecomposed_fraction_at_evaluation_time"] == 0.37
    assert result["area_undecomposed_fraction_at_2s"] == 0.37
    assert result["objective_vector"][1] == 0.37
    assert (
        result["objective_definition"]["deprecated_aliases"][
            "area_undecomposed_fraction_at_2s"
        ]
        == "area_undecomposed_fraction_at_evaluation_time"
    )

    legacy = canonicalise_metrics(
        _objective_row(remainingReactiveMassFractionAt2s=0.42),
        end_time_s=0.25,
        no_ignition_penalty_s=0.25,
    )
    assert legacy["area_undecomposed_fraction_at_evaluation_time"] == 0.42
    assert legacy["area_undecomposed_fraction_at_2s"] == 0.42


def test_congestion_objective_prefers_evaluation_horizon_over_legacy_peak() -> None:
    result = canonicalise_metrics(
        _objective_row(
            remainingReactiveMassFractionAtEvaluationTime=0.37,
            peakCurrentCongestion=9.0,
            peakCurrentCongestionToEvaluationTime=2.5,
            current_congestion=7.0,
        ),
        end_time_s=0.5,
        no_ignition_penalty_s=0.5,
    )
    assert result["peakCurrentCongestion"] == 9.0
    assert result["peakCurrentCongestionToEvaluationTime"] == 2.5
    assert result["current_congestion"] == 2.5
    assert result["objective_vector"][3] == 2.5


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("no_ignition_penalty_s", float("nan")),
        ("no_ignition_constraint_violation", -1.0),
        ("minimum_no_ignition_violation", 1.0e-12),
        ("vmin_invalid_search_constraint_violation", 0.0),
    ),
)
def test_workflow_rejects_invalid_failure_penalty_contract_before_evaluator(
    tmp_path: Path, field: str, value: float,
) -> None:
    cfg=copy.deepcopy(_profile("preflame_cpp_cuda_debug.yaml"))
    cfg["evaluation"][field]=value
    with pytest.raises(EvaluatorError,match="failure-penalty contract"):
        EvaluationWorkflow(ROOT,cfg,tmp_path/field)


def test_required_noncontinuous_ignition_failure_cannot_become_feasible() -> None:
    workflow=object.__new__(EvaluationWorkflow)
    workflow.opt={
        "require_ignition_for_feasibility":True,
        "continuous_ignition_constraint":False,
        "numerical_cap_thresholds":{
            "temperature":0.02,"species":0.02,"gas":0.02,"chemical_rate":0.02,
        },
    }
    workflow.end_time_s=1.0
    workflow.no_ignition_penalty_s=1.0
    workflow.no_ignition_constraint_violation=0.0
    workflow.minimum_no_ignition_violation=1.0e-6
    workflow.vmin_invalid_search_constraint_violation=1.0
    workflow.vmin_upper_bound_V=260.0
    workflow.ignition_onset_temperature_K=523.15
    workflow.initial_temperature_K=298.15
    candidate=CandidateEvaluation("missing-onset",{})
    workflow._apply_raw_metrics(
        candidate,
        _objective_row(
            condensedPhaseIgnitionDelay_s=None,
            ignitionSucceeded=False,
            remainingReactiveMassFractionAtEvaluationTime=1.0,
            peakMaximumTemperature_K=298.15,
            converged=True,
            maximumTemperatureCapFraction=0.0,
            maximumSpeciesLimiterFraction=0.0,
            maximumGasCapFraction=0.0,
            maximumChemicalRateCapFraction=0.0,
        ),
    )
    assert candidate.metrics["ignition_constraint_violation"]==1.0e-6
    assert candidate.constraint_violation>1.0e-12


def test_vmin_reports_observed_monotonicity_for_consistent_trials() -> None:
    rows = [{"ignitionSucceeded": True, "converged": True}]

    def run(indices: list[int], volts: list[float], role: str) -> list[dict]:
        del indices, role
        return [
            {"ignitionSucceeded": voltage >= 125.0, "converged": True}
            for voltage in volts
        ]

    batched_voltage_search(
        rows,
        run,
        lambda row: (bool(row["converged"]), "valid"),
        enabled=True,
        low_voltage=20.0,
        high_voltage=260.0,
        tolerance=5.0,
        max_iterations=8,
        invalid_penalty=100.0,
        censor_penalty=50.0,
        verify_final=True,
    )
    assert rows[0]["minimumIgnitionVoltageMonotonicityObserved"] is True
    assert rows[0]["minimumIgnitionVoltageSearchValid"] is True


def test_vmin_does_not_claim_unobserved_monotonicity_when_disabled() -> None:
    rows = [{"ignitionSucceeded": True, "converged": True}]
    batched_voltage_search(
        rows,
        lambda indices, volts, role: [],
        lambda row: (bool(row["converged"]), "valid"),
        enabled=False,
        low_voltage=20.0,
        high_voltage=260.0,
        tolerance=5.0,
        max_iterations=8,
        invalid_penalty=100.0,
        censor_penalty=50.0,
        verify_final=True,
    )
    assert rows[0]["minimumIgnitionVoltageMonotonicityChecked"] is False
    assert rows[0]["minimumIgnitionVoltageMonotonicityObserved"] is None
    assert rows[0]["minimumIgnitionVoltageSearchValid"] is False
    assert rows[0]["minimumIgnitionVoltage_V"] is None
    assert rows[0]["minimumIgnitionVoltageSearchStatus"] == "disabled_search_invalid"


@pytest.mark.parametrize(
    "overrides",
    (
        {"invalid_penalty": -1.0},
        {"censor_penalty": float("nan")},
        {"invalid_penalty": 1.0e308, "high_voltage": 1.0e308},
        {"low_voltage": float("nan")},
        {"tolerance": 0.0},
        {"max_iterations": 0},
        {"max_iterations": 2.5},
        {"tolerance": 1.0, "max_iterations": 1},
    ),
)
def test_vmin_contract_rejects_nonfinite_or_favourable_failure_scores(
    overrides: dict,
) -> None:
    values = {
        "low_voltage": 20.0,
        "high_voltage": 260.0,
        "tolerance": 5.0,
        "max_iterations": 8,
        "invalid_penalty": 100.0,
        "censor_penalty": 50.0,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        validate_voltage_search_contract(**values)


def test_vmin_contract_rejects_tolerance_below_fp64_bracket_spacing() -> None:
    low_voltage = 1.0e308
    high_voltage = np.nextafter(low_voltage, np.inf)
    representable_spacing = max(math.ulp(low_voltage), math.ulp(high_voltage))

    with pytest.raises(ValueError, match="below the FP64 representable spacing"):
        validate_voltage_search_contract(
            low_voltage=low_voltage,
            high_voltage=high_voltage,
            tolerance=representable_spacing * 0.5,
            max_iterations=2,
            invalid_penalty=0.0,
            censor_penalty=0.0,
        )


def test_vmin_search_keeps_extreme_finite_bracket_representable() -> None:
    high_voltage = 1.0e308
    tolerance = math.ulp(high_voltage)
    ignition_threshold = 7.5e307
    rows = [{"ignitionSucceeded": True, "converged": True}]

    def run(indices: list[int], volts: list[float], role: str) -> list[dict]:
        del indices, role
        return [
            {
                "ignitionSucceeded": voltage >= ignition_threshold,
                "converged": True,
            }
            for voltage in volts
        ]

    batched_voltage_search(
        rows,
        run,
        lambda row: (bool(row["converged"]), "valid"),
        enabled=True,
        low_voltage=0.0,
        high_voltage=high_voltage,
        tolerance=tolerance,
        max_iterations=64,
        invalid_penalty=0.0,
        censor_penalty=0.0,
    )

    result = rows[0]
    assert result["minimumIgnitionVoltageSearchValid"] is True
    assert result["minimumIgnitionVoltageBracketWidth_V"] <= tolerance
    assert result["minimumIgnitionVoltage_V"] >= ignition_threshold
    assert all(
        math.isfinite(trial["voltage_V"])
        for trial in result["minimumIgnitionVoltageTrials"]
    )


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("lower_bound_V", 0.0, "must exceed the cathode voltage"),
        (
            "maximum_bisection_iterations",
            2.5,
            "maximum iterations must be an integer",
        ),
        ("right_censor_objective_penalty_V", -1.0, "penalties must be non-negative"),
    ),
)
def test_bc_evaluator_rejects_invalid_vmin_contract_at_initialisation(
    tmp_path: Path, field: str, value: float, match: str,
) -> None:
    cfg = copy.deepcopy(_profile("preflame_cpp_cuda_debug.yaml"))
    cfg["minimum_ignition_voltage_search"]["enabled"] = True
    cfg["minimum_ignition_voltage_search"][field] = value
    adapter = copy.deepcopy(cfg["evaluator"])
    adapter["physics_config"] = cfg
    with pytest.raises(Exception, match=match):
        ElectrochemicalThermalDecompositionEvaluator(ROOT, adapter, tmp_path / field)


def test_bc_evaluator_forbids_disabling_active_vmin_objective(tmp_path: Path) -> None:
    cfg = copy.deepcopy(_profile("preflame_cpp_cuda_debug.yaml"))
    cfg["minimum_ignition_voltage_search"]["enabled"] = False
    adapter = copy.deepcopy(cfg["evaluator"])
    adapter["physics_config"] = cfg
    with pytest.raises(Exception, match="must be true"):
        ElectrochemicalThermalDecompositionEvaluator(ROOT, adapter, tmp_path / "disabled-vmin")


def test_vmin_repeated_voltage_contradiction_is_invalid_not_a_threshold() -> None:
    rows = [{"ignitionSucceeded": True, "converged": True}]

    def run(indices: list[int], volts: list[float], role: str) -> list[dict]:
        del indices
        return [
            {
                "ignitionSucceeded": False
                if role == "final_upper_full_horizon_verification"
                else voltage >= 125.0,
                "converged": True,
            }
            for voltage in volts
        ]

    batched_voltage_search(
        rows,
        run,
        lambda row: (bool(row["converged"]), "valid"),
        enabled=True,
        low_voltage=20.0,
        high_voltage=260.0,
        tolerance=5.0,
        max_iterations=8,
        invalid_penalty=100.0,
        censor_penalty=50.0,
        verify_final=True,
    )
    assert rows[0]["minimumIgnitionVoltageMonotonicityObserved"] is False
    assert rows[0]["minimumIgnitionVoltageSearchValid"] is False
    assert rows[0]["minimumIgnitionVoltage_V"] is None
    assert "nonmonotonic" in rows[0]["minimumIgnitionVoltageSearchStatus"]


def test_bc_trial_rejects_nonlinear_robin_nonconvergence() -> None:
    solver = {"converged": True}
    assert _bc_solver_converged(
        {
            "allElectricalLinearSolvesConverged": True,
            "allNonlinearRobinSolvesConverged": True,
        },
        solver,
    )
    assert not _bc_solver_converged(
        {
            "allElectricalLinearSolvesConverged": True,
            "allNonlinearRobinSolvesConverged": False,
        },
        solver,
    )
    assert not _bc_solver_converged(
        {
            "allElectricalLinearSolvesConverged": False,
            "allNonlinearRobinSolvesConverged": True,
        },
        solver,
    )


def test_vmin_same_voltage_contradiction_is_order_independent() -> None:
    trials = [
        {"voltage_V": 100.0, "ignited": False, "numerically_valid": True},
        {"voltage_V": 100.0, "ignited": True, "numerically_valid": True},
    ]
    forward = _monotonicity_violation(trials)
    reverse = _monotonicity_violation(list(reversed(trials)))
    assert forward == reverse
    assert forward == {
        "lowerIgnitingVoltage_V": 100.0,
        "higherNonIgnitingVoltage_V": 100.0,
        "sameVoltageContradiction": True,
    }


def test_reference_vmin_uses_shared_monotonicity_and_full_horizon_contract(
    tmp_path: Path,
) -> None:
    evaluator = object.__new__(ElectrochemicalThermalDecompositionEvaluator)
    evaluator.vmin_enabled = True
    evaluator.vmin_lower_bound_V = 20.0
    evaluator.vmin_upper_bound_V = 260.0
    evaluator.vmin_tolerance_V = 5.0
    evaluator.vmin_max_iterations = 8
    evaluator.vmin_invalid_penalty_V = 100.0
    evaluator.vmin_right_censor_penalty_V = 50.0
    evaluator.vmin_verify_final = True
    evaluator.vmin_stop_successful_trials_at_ignition = True
    evaluator._trial_valid = lambda row: (bool(row["converged"]), "valid")
    calls: list[tuple[str, bool]] = []

    def fake_run_model(items, voltage, *, write_metrics, stop_on_onset):
        del write_metrics
        directory = str(items[0][3])
        calls.append((directory, stop_on_onset))
        final_verification = "final_upper_full_horizon_verification" in directory
        return ([{
            "ignitionSucceeded": (
                False if final_verification else float(voltage) >= 125.0
            ),
            "ignitionDelay_s": 0.1,
            "converged": True,
        }], None)

    evaluator._run_model = fake_run_model
    row = {"ignitionSucceeded": True, "converged": True}
    mask = np.ones((2, 2), dtype=bool)
    evaluator._attach_vmin(
        (mask, mask.copy(), {"geometry_id": "shared-vmin"}, tmp_path), row
    )

    assert row["minimumIgnitionVoltageSearchValid"] is False
    assert row["minimumIgnitionVoltageMonotonicityObserved"] is False
    assert "nonmonotonic" in row["minimumIgnitionVoltageSearchStatus"]
    assert any(
        "final_upper_full_horizon_verification" in directory and not early
        for directory, early in calls
    )


def test_reference_vmin_typed_trial_failure_preserves_reference_objectives(
    tmp_path: Path,
) -> None:
    evaluator = object.__new__(ElectrochemicalThermalDecompositionEvaluator)
    evaluator.vmin_enabled = True
    evaluator.vmin_lower_bound_V = 20.0
    evaluator.vmin_upper_bound_V = 260.0
    evaluator.vmin_tolerance_V = 5.0
    evaluator.vmin_max_iterations = 8
    evaluator.vmin_invalid_penalty_V = 100.0
    evaluator.vmin_right_censor_penalty_V = 50.0
    evaluator.vmin_verify_final = True
    evaluator.vmin_stop_successful_trials_at_ignition = True
    evaluator._trial_valid = lambda row: (bool(row["converged"]), "valid")

    def typed_trial_failure(items, voltage, *, write_metrics, stop_on_onset):
        del items, voltage, write_metrics, stop_on_onset
        raise PreflameCandidateBatchError(
            "typed trial CFL failure", (0,), "thermal_stability_cfl"
        )

    evaluator._run_model = typed_trial_failure
    evaluator._failed_geometry_row = lambda metadata, output, exc: {
        "geometry_id": metadata["geometry_id"],
        "ignitionSucceeded": False,
        "converged": False,
        "physicsRejected": True,
        "physicsRejectionReason": str(exc),
    }
    row = {
        "ignitionSucceeded": True,
        "converged": True,
        "referenceObjectiveSentinel": 12.5,
    }
    mask = np.ones((2, 2), dtype=bool)
    evaluator._attach_vmin(
        (mask, mask.copy(), {"geometry_id": "trial-failure"}, tmp_path), row
    )

    assert row["referenceObjectiveSentinel"] == 12.5
    assert row["minimumIgnitionVoltageSearchValid"] is False
    assert row["minimumIgnitionVoltageSearchStatus"] == "invalid_lower_bound_trial"
    assert row["minimumIgnitionVoltageObjective_V"] == 360.0
    assert row.get("physicsRejected", False) is False
    trial = row["minimumIgnitionVoltageTrials"][-1]
    assert trial["numerically_valid"] is False


def test_release_manifest_verifier_checks_hashes_and_complete_inventory(
    tmp_path: Path,
) -> None:
    import hashlib

    (tmp_path / "VERSION.json").write_text(
        json.dumps({"version": "8.2.0-test"}), encoding="utf-8"
    )
    payload = tmp_path / "payload.txt"
    payload.write_text("validated\n", encoding="utf-8")
    manifest = tmp_path / "PACKAGE_SHA256_MANIFEST_V8_2.txt"
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    version_digest = hashlib.sha256(
        (tmp_path / "VERSION.json").read_bytes()
    ).hexdigest()
    manifest.write_text(
        f"{digest}  payload.txt\n{version_digest}  VERSION.json\n",
        encoding="utf-8",
    )
    generic = tmp_path / "PACKAGE_SHA256_MANIFEST.txt"
    generic.write_bytes(manifest.read_bytes())
    assert verify(tmp_path, manifest)["status"] == "passed"

    generic.write_text("tampered alias\n", encoding="utf-8")
    alias_report = verify(tmp_path, manifest)
    assert alias_report["status"] == "failed"
    assert alias_report["generic_alias_matches_authoritative"] is False
    generic.write_bytes(manifest.read_bytes())

    # The generic same-content alias is excluded to avoid self-reference, but
    # historical or invented versioned manifests remain payload and may not
    # evade complete-inventory verification.
    historical = tmp_path / "PACKAGE_SHA256_MANIFEST_V8_1.txt"
    historical.write_text(
        "historical\n", encoding="utf-8"
    )
    report = verify(tmp_path, manifest)
    assert report["status"] == "failed"
    assert report["unlisted"] == ["PACKAGE_SHA256_MANIFEST_V8_1.txt"]
    historical_digest = hashlib.sha256(historical.read_bytes()).hexdigest()
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + f"{historical_digest}  PACKAGE_SHA256_MANIFEST_V8_1.txt\n",
        encoding="utf-8",
    )
    generic.write_bytes(manifest.read_bytes())
    assert verify(tmp_path, manifest)["status"] == "passed"

    extra = tmp_path / "unlisted.txt"
    extra.write_text("not listed\n", encoding="utf-8")
    report = verify(tmp_path, manifest)
    assert report["status"] == "failed"
    assert report["unlisted"] == ["unlisted.txt"]

    extra.unlink()
    payload.write_text("mutated\n", encoding="utf-8")
    report = verify(tmp_path, manifest)
    assert report["status"] == "failed"
    assert [row["path"] for row in report["mismatched"]] == ["payload.txt"]


def test_release_manifest_verifier_separates_v821_from_historical_v820(
    tmp_path: Path,
) -> None:
    import hashlib

    version = tmp_path / "VERSION.json"
    version.write_text(json.dumps({"version": "8.2.1-test"}), encoding="utf-8")
    historical = tmp_path / "PACKAGE_SHA256_MANIFEST_V8_2.txt"
    historical.write_text("immutable v8.2.0 manifest\n", encoding="utf-8")
    payload = tmp_path / "payload.txt"
    payload.write_text("v8.2.1 payload\n", encoding="utf-8")
    authoritative = tmp_path / "PACKAGE_SHA256_MANIFEST_V8_2_1.txt"
    rows = []
    for path in (historical, payload, version):
        rows.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        )
    authoritative.write_text("\n".join(rows) + "\n", encoding="utf-8")
    generic = tmp_path / "PACKAGE_SHA256_MANIFEST.txt"
    generic.write_bytes(authoritative.read_bytes())

    report = verify(tmp_path, authoritative)
    assert report["status"] == "passed"
    assert report["manifest"] == "PACKAGE_SHA256_MANIFEST_V8_2_1.txt"
    assert report["generic_alias_matches_authoritative"] is True
