from __future__ import annotations

import ast
import copy
import inspect
import math
from pathlib import Path
import textwrap

import numpy as np
import pytest
import torch
import yaml

import ecsp_v6.physics.bc_global as bc_global
from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
from ecsp_v6.physics.bc_global import run_bc_global_batch


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def continuation_case(tmp_path_factory: pytest.TempPathFactory):
    raw_config = yaml.safe_load(
        (ROOT / "config/nsga2_bc_global_native_debug.yaml").read_text(
            encoding="utf-8"
        )
    )
    workflow = NSGA2ElectricalSolidWorkflow(
        ROOT,
        raw_config,
        tmp_path_factory.mktemp("continuation_async") / "workflow",
    )
    config = copy.deepcopy(workflow.evaluator.config)
    composition = workflow.evaluator.composition
    bc = config["bcGlobal"]
    bc.update(
        timeStep_s=0.001,
        endTime_s=0.003,
        evaluationTime_s=0.003,
        electricalUpdateInterval_s=0.001,
    )
    bc["thermal"].update(
        heat_capacity={"mode": "constant", "value": 1000.0},
        thermal_conductivity={"mode": "constant", "value": 0.0},
        ambientTemperature_K=300.0,
        minimumTemperature_K=1.0,
        maximumTemperature_K=1.0e6,
        convectionCoefficient_W_per_m2K=0.0,
        emissivity=0.0,
    )
    for channel in bc["kinetics"]["channels"]:
        point_count = len(channel["alpha_grid"])
        channel["activation_energy_J_per_mol"] = [0.0] * point_count
        channel["ln_Af_per_s"] = [math.log(0.1)] * point_count
        channel["heat_release_J_per_kg"] = 0.0

    shape = (2, 3, 3)
    dtype = torch.float64
    onset_fields = {
        "temperatureAtOnset_K": torch.full(shape, 300.0, dtype=dtype),
        "globalProgressAtOnset": torch.zeros(shape, dtype=dtype),
        "alphaChannel1AtOnset": torch.zeros(shape, dtype=dtype),
        "alphaChannel2AtOnset": torch.zeros(shape, dtype=dtype),
        "mobileLPAtOnset_mol_per_m3": torch.full(
            shape, composition.initial_lp_mol_per_m3, dtype=dtype
        ),
        "mobileWaterAtOnset_mol_per_m3": torch.full(
            shape, composition.initial_water_mol_per_m3, dtype=dtype
        ),
        "generatedWaterProductAtOnset_mol_per_m3": torch.zeros(
            shape, dtype=dtype
        ),
        "electrochemicalLPConsumedAtOnset_mol_per_m3": torch.zeros(
            shape, dtype=dtype
        ),
        "pvaReactiveRepeatAtOnset_mol_per_m3": torch.full(
            shape, composition.initial_pva_repeat_mol_per_m3, dtype=dtype
        ),
        "propellantMask": torch.ones(shape, dtype=torch.bool),
    }
    delays = torch.tensor([0.0, 0.002], dtype=dtype)
    return onset_fields, config, composition, delays


def _continue(continuation_case):
    onset_fields, config, composition, delays = continuation_case
    return bc_global.continue_bc_onset_batch_to_evaluation(
        onset_fields,
        config,
        composition,
        delays,
        0.003,
        torch.float64,
    )


def _assert_candidate_failure(
    caught: pytest.ExceptionInfo[bc_global.BCCandidateBatchError],
    category: str,
    indices: tuple[int, ...] = (0, 1),
) -> None:
    assert caught.value.category == category
    assert caught.value.candidate_indices == indices


def test_continuation_timestep_loop_has_no_host_synchronizing_calls() -> None:
    source = textwrap.dedent(
        inspect.getsource(bc_global.continue_bc_onset_batch_to_evaluation)
    )
    function = ast.parse(source).body[0]
    loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "steps"
    ]
    assert len(loops) == 1

    forbidden: list[str] = []
    for node in ast.walk(loops[0]):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "cpu",
                "item",
                "numpy",
                "tolist",
            }:
                forbidden.append(node.func.attr)
            if isinstance(node.func, ast.Name) and node.func.id in {
                "bool",
                "float",
                "int",
            }:
                forbidden.append(node.func.id)
    assert forbidden == []


def test_continuation_preserves_mixed_lane_durations(continuation_case) -> None:
    result = _continue(continuation_case)

    torch.testing.assert_close(
        result["meanGlobalProgressAtEvaluationTime"],
        torch.tensor([3.0e-4, 1.0e-4], dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        result["evaluationStateTime_s"],
        torch.tensor([0.003, 0.003], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_continuation_integrates_valid_sub_femtosecond_steps(
    continuation_case,
) -> None:
    onset_fields, config, composition, _ = continuation_case
    fields = {
        name: value[:1].clone() if isinstance(value, torch.Tensor) else value
        for name, value in onset_fields.items()
    }
    initial_progress = 1.0e-8
    xi_max = min(
        composition.initial_lp_mol_per_m3 / 1.45,
        composition.initial_pva_repeat_mol_per_m3,
    )
    fields["alphaChannel1AtOnset"].fill_(initial_progress)
    fields["alphaChannel2AtOnset"].fill_(initial_progress)
    fields["globalProgressAtOnset"].fill_(initial_progress)
    fields["mobileLPAtOnset_mol_per_m3"].fill_(
        composition.initial_lp_mol_per_m3 - 1.45 * xi_max * initial_progress
    )
    fields["generatedWaterProductAtOnset_mol_per_m3"].fill_(
        2.0 * xi_max * initial_progress
    )
    fields["pvaReactiveRepeatAtOnset_mol_per_m3"].fill_(
        composition.initial_pva_repeat_mol_per_m3 - xi_max * initial_progress
    )
    cfg = copy.deepcopy(config)
    cfg["bcGlobal"].update(
        timeStep_s=1.0e-15,
        endTime_s=3.0e-14,
        evaluationTime_s=3.0e-14,
        electricalUpdateInterval_s=1.0e-15,
        handoffSnapshotInterval_s=1.0e-15,
    )
    cfg["bcGlobal"]["kinetics"]["maximum_rate_per_s"] = 1.0e6
    for channel in cfg["bcGlobal"]["kinetics"]["channels"]:
        point_count = len(channel["alpha_grid"])
        channel["activation_energy_J_per_mol"] = [0.0] * point_count
        channel["ln_Af_per_s"] = [math.log(1.0e6)] * point_count
        channel["heat_release_J_per_kg"] = 0.0

    result = bc_global.continue_bc_onset_batch_to_evaluation(
        fields,
        cfg,
        composition,
        torch.tensor([1.0e-14], dtype=torch.float64),
        3.0e-14,
        torch.float64,
    )
    assert result["continuedMask"].item()
    assert result["evaluationStateTime_s"].item() == 3.0e-14
    assert result["meanGlobalProgressAtEvaluationTime"].item() > initial_progress


def test_deferred_inventory_failure_still_fails_closed(
    continuation_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never_close(left: torch.Tensor, right: torch.Tensor, **_: object):
        return torch.zeros_like(left, dtype=torch.bool)

    monkeypatch.setattr(bc_global.torch, "isclose", never_close)
    with pytest.raises(
        bc_global.BCCandidateBatchError, match="inventory invariant failed"
    ) as caught:
        _continue(continuation_case)
    _assert_candidate_failure(caught, "continuation_inventory_invariant")


def test_deferred_property_failure_runs_all_steps_then_fails_closed(
    continuation_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def invalid_property(
        temperature: torch.Tensor, raw: object, *, gas_constant: float
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.full_like(temperature, torch.nan)

    monkeypatch.setattr(bc_global, "_property_from_config", invalid_property)
    with pytest.raises(
        bc_global.BCCandidateBatchError, match="thermal properties are invalid"
    ) as caught:
        _continue(continuation_case)
    _assert_candidate_failure(caught, "continuation_thermal_properties")
    # cp and k are checked at both the old and candidate accepted temperature.
    assert calls == 12


def test_final_continuation_step_rejects_invalid_updated_properties(
    continuation_case,
) -> None:
    onset_fields, config, composition, delays = continuation_case
    config = copy.deepcopy(config)
    for channel in config["bcGlobal"]["kinetics"]["channels"]:
        channel["heat_release_J_per_kg"] = 1.0e5
    config["bcGlobal"]["thermal"]["thermal_conductivity"] = {
        "mode": "reference_arrhenius",
        "reference_value": 0.0,
        "reference_temperature_K": 300.0,
        "activation_energy_J_per_mol": 1.0e12,
    }
    with pytest.raises(
        bc_global.BCCandidateBatchError, match="thermal properties are invalid"
    ) as caught:
        bc_global.continue_bc_onset_batch_to_evaluation(
            onset_fields,
            config,
            composition,
            torch.zeros_like(delays),
            0.001,
            torch.float64,
        )
    _assert_candidate_failure(caught, "continuation_thermal_properties")


def test_deferred_cfl_failure_reports_original_batch_indices(
    continuation_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def unstable_cfl(
        conductivity: torch.Tensor,
        capacity: torch.Tensor,
        propellant: torch.Tensor,
        spacing: float,
        step_dt: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        active = step_dt > 0.0
        return torch.where(
            active,
            torch.full_like(conductivity, 2.0),
            torch.zeros_like(conductivity),
        )

    monkeypatch.setattr(bc_global, "_thermal_diffusive_cfl", unstable_cfl)
    with pytest.raises(
        bc_global.BCCandidateBatchError,
        match=r"stability CFL exceeded; batch_indices=\[0, 1\]",
    ) as caught:
        _continue(continuation_case)
    _assert_candidate_failure(caught, "continuation_thermal_stability_cfl")
    assert calls == 3


def test_continuation_rejects_nonfinite_cfl_from_finite_large_properties(
    continuation_case,
) -> None:
    onset_fields, config, composition, _ = continuation_case
    fields = {
        name: value[:1].clone() if isinstance(value, torch.Tensor) else value
        for name, value in onset_fields.items()
    }
    cfg = copy.deepcopy(config)
    cfg["bcGlobal"]["thermal"].update(
        density_kg_per_m3=1.0e308,
        heat_capacity={"mode": "constant", "value": 2.0},
        thermal_conductivity={"mode": "constant", "value": 1.0e308},
    )
    with pytest.raises(
        bc_global.BCCandidateBatchError, match="thermal properties are invalid"
    ) as caught:
        bc_global.continue_bc_onset_batch_to_evaluation(
            fields,
            cfg,
            composition,
            torch.zeros(1, dtype=torch.float64),
            0.001,
            torch.float64,
        )
    _assert_candidate_failure(
        caught,
        "continuation_thermal_properties",
        indices=(0,),
    )


def test_deferred_nonfinite_temperature_still_fails_closed(
    continuation_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def nonfinite_conduction(
        temperature: torch.Tensor,
        conductivity: torch.Tensor,
        propellant: torch.Tensor,
        spacing: float,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.full_like(temperature, torch.inf)

    monkeypatch.setattr(bc_global, "_divergence_k_grad", nonfinite_conduction)
    with pytest.raises(
        bc_global.BCCandidateBatchError, match="produced non-finite temperature"
    ) as caught:
        _continue(continuation_case)
    _assert_candidate_failure(caught, "continuation_nonfinite_temperature")
    assert calls == 3


def test_invalid_onset_state_reports_only_bad_lane(continuation_case) -> None:
    onset_fields, config, composition, delays = continuation_case
    bad_fields = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in onset_fields.items()
    }
    bad_fields["generatedWaterProductAtOnset_mol_per_m3"][1, 0, 0] = 1.0
    with pytest.raises(
        bc_global.BCCandidateBatchError, match="Invalid or non-conservative"
    ) as caught:
        bc_global.continue_bc_onset_batch_to_evaluation(
            bad_fields, config, composition, delays, 0.003, torch.float64
        )
    _assert_candidate_failure(
        caught, "continuation_invalid_onset_state", indices=(1,)
    )


def test_invalid_ignition_delay_reports_only_bad_lane(continuation_case) -> None:
    onset_fields, config, composition, delays = continuation_case
    bad_delays = delays.clone()
    bad_delays[0] = -0.001
    with pytest.raises(
        bc_global.BCCandidateBatchError, match="outside the configured B/C horizon"
    ) as caught:
        bc_global.continue_bc_onset_batch_to_evaluation(
            onset_fields, config, composition, bad_delays, 0.003, torch.float64
        )
    _assert_candidate_failure(
        caught, "continuation_invalid_ignition_delay", indices=(0,)
    )


def test_invalid_remaining_mass_reports_only_bad_lane(
    continuation_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_remaining(*_: object, **__: object) -> torch.Tensor:
        return torch.tensor([0.5, 1.5], dtype=torch.float64)

    monkeypatch.setattr(bc_global, "_masked_integral_ratio", invalid_remaining)
    with pytest.raises(
        bc_global.BCCandidateBatchError, match="remaining mass is outside"
    ) as caught:
        _continue(continuation_case)
    _assert_candidate_failure(
        caught, "continuation_remaining_reactive_mass", indices=(1,)
    )


def test_invalid_common_evaluation_state_reports_only_bad_lane(
    continuation_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    onset_fields, config, composition, _ = continuation_case

    def invalid_continuation(*_: object, **__: object) -> dict[str, torch.Tensor]:
        return {
            "continuedMask": torch.ones(2, dtype=torch.bool),
            "remainingReactiveMassFractionAtEvaluationTime": torch.tensor(
                [0.5, 1.5], dtype=torch.float64
            ),
            "meanGlobalProgressAtEvaluationTime": torch.tensor(
                [0.5, 0.5], dtype=torch.float64
            ),
            "temperatureOnsetAreaFractionAtEvaluationTime": torch.zeros(
                2, dtype=torch.float64
            ),
        }

    monkeypatch.setattr(
        bc_global, "continue_bc_onset_batch_to_evaluation", invalid_continuation
    )
    output: dict[str, object] = {
        "evaluationTime_s": 0.003,
        "ignitionDelay_s": torch.zeros(2, dtype=torch.float64),
    }
    for key in (
        "areaAveragedUndecomposedFractionAt2s",
        "areaAveragedUndecomposedFractionAtEvaluationTime",
        "remainingReactiveMassFractionAt2s",
        "remainingReactiveMassFractionAtEvaluationTime",
        "meanGlobalProgressAt2s",
        "meanGlobalProgressAtEvaluationTime",
        "temperatureOnsetAreaFractionAt2s",
    ):
        output[key] = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(
        bc_global.BCCandidateBatchError, match="common evaluation state"
    ) as caught:
        bc_global.complete_bc_evaluation_output(
            output, onset_fields, config, composition, torch.float64
        )
    _assert_candidate_failure(
        caught, "continuation_common_evaluation_state", indices=(1,)
    )


def test_python_mixed_voltage_batch_matches_serial_after_one_lane_freezes(
    tmp_path: Path,
) -> None:
    raw_config = yaml.safe_load(
        (ROOT / "config/nsga2_bc_global_native_debug.yaml").read_text(
            encoding="utf-8"
        )
    )
    evaluator = NSGA2ElectricalSolidWorkflow(
        ROOT, raw_config, tmp_path / "workflow"
    ).evaluator
    evaluator.config["bcGlobal"].update(
        endTime_s=0.02,
        evaluationTime_s=0.02,
    )
    for channel in evaluator.config["bcGlobal"]["kinetics"]["channels"]:
        channel["heat_release_J_per_kg"] = 0.0
    evaluator.config["bcGlobal"]["onsetCriterion"]["temperature_K"] = (
        298.1502
    )

    items = []
    for index in range(2):
        anode = np.zeros((32, 32), dtype=bool)
        cathode = np.zeros_like(anode)
        anode[5:8, 5:26] = True
        cathode[24:27, 5:26] = True
        items.append(
            (
                anode,
                cathode,
                {"geometry_id": f"mixed_voltage_{index}"},
                tmp_path / f"case_{index}",
            )
        )

    voltages = [260.0, 1.0]
    geometry = evaluator._build_geometry_batch(items)
    batched = run_bc_global_batch(
        geometry,
        evaluator.config,
        evaluator.composition,
        voltages,
        torch.float64,
        save_fields=True,
        stop_on_onset=True,
    )
    assert batched["ignitionDelay_s"][0].item() == pytest.approx(0.002)
    assert torch.isnan(batched["ignitionDelay_s"][1])
    # electricalUpdateInterval=0.01 s and dt=0.002 s: lane 1 is re-solved as
    # a one-lane active subset at step 5 after lane 0 has frozen at step 0.
    assert batched["preflameTermination"]["candidateStatesFrozenAtFirstOnset"]

    for index, voltage in enumerate(voltages):
        serial_geometry = evaluator._build_geometry_batch([items[index]])
        serial = run_bc_global_batch(
            serial_geometry,
            evaluator.config,
            evaluator.composition,
            voltage,
            torch.float64,
            save_fields=True,
            stop_on_onset=True,
        )
        for name, values in batched["finalFields"].items():
            torch.testing.assert_close(
                values[index],
                serial["finalFields"][name][0],
                rtol=1.0e-12,
                atol=1.0e-12,
                equal_nan=True,
            )
        for name, values in batched["histories"].items():
            torch.testing.assert_close(
                values[:, index],
                serial["histories"][name][:, 0],
                rtol=1.0e-12,
                atol=1.0e-12,
                equal_nan=True,
            )
