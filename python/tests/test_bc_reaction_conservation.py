from __future__ import annotations

import math
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from ecsp_nsga2.propagation import (
    PropagationCandidateNumericalError,
    PropagationConfigurationError,
    _accepted_channel_step,
    _arrival_time_statistics,
    _div_k_grad as _propagation_div_k_grad,
    _effective_regression_velocity,
    _positive_time_step_count as _propagation_step_count,
    _safe_harmonic_mean as _propagation_harmonic_mean,
    _thermal_diffusive_cfl as _propagation_thermal_cfl,
    final_refinement_objectives,
    run_condensed_propagation,
)
from ecsp_v6.physics.bc_global import _reaction_inventory
from ecsp_v6.physics.bc_global import (
    _is_time_grid_aligned,
    _masked_integral_ratio,
    _positive_time_step_count as _preflame_step_count,
    _require_unclipped_faradaic_inventory,
    _thermal_diffusive_cfl as _preflame_thermal_cfl,
    _validate_bc_config,
)
from ecsp_v6.physics.bc_global import run_bc_global_batch
from ecsp_v6.physics.geometry import GeometryBatch


ROOT = Path(__file__).resolve().parents[2]


def _channel(*, rate_per_s: float, heat_j_per_kg: float) -> dict:
    return {
        "alpha_grid": [0.0, 1.0],
        "activation_energy_J_per_mol": [0.0, 0.0],
        "ln_Af_per_s": [math.log(rate_per_s), math.log(rate_per_s)],
        "heat_release_J_per_kg": heat_j_per_kg,
    }


def _propagation_config(**overrides: object) -> dict:
    config = {
        "duration_s": 1.0,
        "time_step_s": 1.0,
        "domain_size_m": 1.0,
        "density_kg_per_m3": 1.0,
        "gas_constant_J_per_molK": 8.31446261815324,
        "surface_layer_thickness_m": 1.0,
        "front_progress_threshold": 0.5,
        "established_reacted_area_fraction": 0.5,
        "continued_electrical_heating": False,
        "electrical_heating_mode": "off",
    }
    config.update(overrides)
    return config


def _bc_config(channel_1: dict, channel_2: dict, weights=(1.0, 0.0)) -> dict:
    return {
        "endTime_s": 3.0,
        "onsetCriterion": {
            "temperature_K": 300.0,
            "minimum_progress": 0.001,
            "minimum_area_fraction": 0.5,
        },
        "kinetics": {
            "mass_conversion_weights": list(weights),
            "maximum_rate_per_s": 1.0e6,
            "channels": [channel_1, channel_2],
        },
        "thermal": {
            "heat_capacity": 1.0,
            "thermal_conductivity": 0.0,
            "ambientTemperature_K": 300.0,
            "minimumTemperature_K": 1.0,
            "maximumTemperature_K": 1.0e9,
            "convectionCoefficient_W_per_m2K": 0.0,
            "emissivity": 0.0,
        },
    }


def _handoff(shape=(2, 2)) -> dict:
    molar_mass_lp = 0.1
    molar_mass_pva = 0.05
    initial_lp = 10.0 + 1.45 * 0.9
    initial_pva = 10.0 + 0.9
    return {
        "handoffSchemaVersion": "ecsp_bc_surface_onset_v8.2.0",
        "temperatureAtOnset_K": np.full(shape, 300.0),
        "alphaChannel1AtOnset": np.full(shape, 0.9),
        "alphaChannel2AtOnset": np.ones(shape),
        "globalProgressAtOnset": np.full(shape, 0.9),
        "propellantMask": np.ones(shape, dtype=bool),
        "onsetSucceeded": True,
        "continuedElectricalHeating": False,
        "ignitionDelay_s": 2.0,
        "mobileLPAtOnset_mol_per_m3": np.full(shape, 10.0),
        "cationAtOnset_mol_per_m3": np.full(shape, 10.0),
        "anionAtOnset_mol_per_m3": np.full(shape, 10.0),
        "pvaReactiveRepeatAtOnset_mol_per_m3": np.full(shape, 10.0),
        "generatedWaterProductAtOnset_mol_per_m3": np.full(shape, 1.8),
        "mobileWaterAtOnset_mol_per_m3": np.full(shape, 10.0),
        "electrochemicalLPConsumedAtOnset_mol_per_m3": np.zeros(shape),
        "potentialAtOnset_V": np.zeros(shape),
        "qJAtOnset_W_per_m3": np.zeros(shape),
        "qEchemAtOnset_W_per_m3": np.zeros(shape),
        "xiMax_mol_per_m3": 1.0,
        "molarMassLP_kg_per_mol": molar_mass_lp,
        "molarMassPVARepeat_kg_per_mol": molar_mass_pva,
        "initialReactiveMass_kg_per_m3": (
            molar_mass_lp * initial_lp + molar_mass_pva * initial_pva
        ),
        "initialMobileLP_mol_per_m3": initial_lp,
        "initialPVARepeat_mol_per_m3": initial_pva,
        "initialMobileWater_mol_per_m3": 10.0,
        "onsetReportedByPhysics": True,
        "numericallyValidForPropagationHandoff": True,
        "propagationHandoffAuthorizationReason": (
            "ignition_and_numerics_valid"
        ),
        "postOnsetElectricalPolicy": "recompute_required_no_preflame_replay",
        "preflameElectricalHistoryMayBeReplayed": False,
    }


def test_inventory_uses_actual_mobile_lp_and_reports_electrochemical_loss() -> None:
    composition = SimpleNamespace(
        initial_lp_mol_per_m3=14.5,
        initial_pva_repeat_mol_per_m3=20.0,
        molar_mass_lp_kg_per_mol=0.1,
        molar_mass_pva_repeat_kg_per_mol=0.05,
    )
    progress = torch.tensor([0.5], dtype=torch.float64)
    inventory = _reaction_inventory(
        progress,
        composition,
        mobile_lp_mol_per_m3=torch.tensor([5.0], dtype=torch.float64),
        electrochemical_lp_consumed_mol_per_m3=torch.tensor(
            [2.25], dtype=torch.float64
        ),
        generated_water_product_mol_per_m3=torch.tensor(
            [10.0], dtype=torch.float64
        ),
    )

    assert inventory["lpRemainingFromChemicalProgressOnly_mol_per_m3"].item() == pytest.approx(7.25)
    assert inventory["lpReactive_mol_per_m3"].item() == pytest.approx(5.0)
    assert inventory["electrochemicalLPConsumed_mol_per_m3"].item() == pytest.approx(2.25)
    assert inventory["waterProduct_mol_per_m3"].item() == pytest.approx(10.0)
    assert inventory["reactiveMass_kg_per_m3"].item() == pytest.approx(
        0.1 * 5.0 + 0.05 * 15.0
    )
    reconstructed = _reaction_inventory(
        progress,
        composition,
        electrochemical_lp_consumed_mol_per_m3=torch.tensor(
            [2.25], dtype=torch.float64
        ),
    )
    assert reconstructed["lpReactive_mol_per_m3"].item() == pytest.approx(5.0)


def test_masked_integral_ratio_scales_before_reduction_to_avoid_fp64_overflow() -> None:
    numerator = torch.full((1, 2, 2), 1.0e308, dtype=torch.float64)
    denominator = torch.full_like(numerator, 1.0e308)
    mask = torch.ones_like(numerator, dtype=torch.bool)

    # Either unscaled reduction is +inf, so the ratio must be formed only
    # after a common per-candidate normalization.
    assert bool(torch.isinf(numerator.sum()))
    assert bool(torch.isinf(denominator.sum()))
    torch.testing.assert_close(
        _masked_integral_ratio(numerator, denominator, mask),
        torch.ones(1, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "composition",
    (
        SimpleNamespace(
            initial_lp_mol_per_m3=1.0e308,
            initial_pva_repeat_mol_per_m3=1.0e308,
            molar_mass_lp_kg_per_mol=2.0,
            molar_mass_pva_repeat_kg_per_mol=2.0,
        ),
        SimpleNamespace(
            initial_lp_mol_per_m3=1.0,
            initial_pva_repeat_mol_per_m3=1.0,
            molar_mass_lp_kg_per_mol=float("nan"),
            molar_mass_pva_repeat_kg_per_mol=1.0,
        ),
    ),
    ids=("finite_operands_overflow", "nonfinite_total"),
)
def test_reaction_inventory_rejects_unrepresentable_initial_mass(
    composition: SimpleNamespace,
) -> None:
    with pytest.raises(
        ValueError, match="Initial reactive mass must be representable"
    ):
        _reaction_inventory(
            torch.zeros(1, dtype=torch.float64),
            composition,
        )


def test_accepted_channel_rate_is_delta_alpha_over_dt() -> None:
    alpha = np.full((2, 2), 0.9)
    temperature = np.full((2, 2), 300.0)
    mask = np.ones((2, 2), dtype=bool)

    updated, accepted_rate, limited = _accepted_channel_step(
        alpha,
        temperature,
        _channel(rate_per_s=100.0, heat_j_per_kg=1.0),
        maximum_rate=1000.0,
        mask=mask,
        dt=1.0,
        gas_constant=8.31446261815324,
    )

    np.testing.assert_allclose(updated, 1.0)
    np.testing.assert_allclose(accepted_rate, 0.1)
    assert np.all(limited)


@pytest.mark.parametrize(
    ("duration", "step_size", "expected_steps"),
    [(0.07, 0.01, 7), (0.075, 0.01, 8), (2.0, 0.002, 1000), (0.005, 0.01, 1)],
)
def test_time_step_count_has_no_zero_length_tail(
    duration: float, step_size: float, expected_steps: int
) -> None:
    assert _preflame_step_count(duration, step_size) == expected_steps
    assert _propagation_step_count(duration, step_size) == expected_steps


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"duration_s": 2.0, "time_step_s": 1.0, "maximum_time_steps": 1},
         "maximum_time_steps"),
        ({"maximum_history_allocation_bytes": 8},
         "maximum_history_allocation_bytes"),
    ),
)
def test_propagation_resource_limits_fail_closed_before_integration(
    tmp_path: Path, overrides: dict[str, object], match: str,
) -> None:
    with pytest.raises(PropagationConfigurationError, match=match):
        run_condensed_propagation(
            _handoff(),
            _propagation_config(**overrides),
            _bc_config(
                _channel(rate_per_s=1.0e-100, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0e-100, heat_j_per_kg=0.0),
            ),
            tmp_path / match,
        )


def test_faradaic_inventory_clipping_fails_closed() -> None:
    clear = torch.zeros((2, 2, 2), dtype=torch.bool)
    _require_unclipped_faradaic_inventory(
        clear, clear, step=0, step_dt=0.01
    )
    limited = clear.clone()
    limited[1, 0, 1] = True
    with pytest.raises(
        RuntimeError,
        match="refusing an inconsistent state.*batch_indices=\\[1\\]",
    ):
        _require_unclipped_faradaic_inventory(
            limited, clear, step=3, step_dt=0.01
        )


def test_internal_evaluation_time_must_align_with_time_grid() -> None:
    assert _is_time_grid_aligned(0.02, 0.01)
    assert not _is_time_grid_aligned(0.015, 0.01)


@pytest.mark.parametrize("maximum_rate", [0.0, -1.0, float("nan")])
def test_preflame_rejects_nonpositive_or_nonfinite_maximum_rate(
    maximum_rate: float,
) -> None:
    bc = _bc_config(
        _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
        _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
    )
    bc["kinetics"]["maximum_rate_per_s"] = maximum_rate
    bc["onsetCriterion"] = {
        "temperature_K": 500.0,
        "minimum_progress": 0.01,
        "minimum_area_fraction": 0.01,
    }
    with pytest.raises(ValueError, match="maximum_rate_per_s"):
        _validate_bc_config({"bcGlobal": bc})


def test_preflame_thermal_cfl_uses_finite_volume_face_diagonal() -> None:
    conductivity = torch.full((1, 3, 3), 100.0, dtype=torch.float64)
    capacity = torch.ones_like(conductivity)
    mask = torch.ones_like(conductivity, dtype=torch.bool)
    cfl = _preflame_thermal_cfl(
        conductivity, capacity, mask, spacing=0.5, step_dt=1.0
    )
    assert cfl[0, 1, 1].item() == pytest.approx(1600.0)
    assert cfl[0, 0, 0].item() == pytest.approx(800.0)


@pytest.mark.parametrize("duration", [0.07, 0.075])
def test_preflame_time_grid_is_strict_and_conversion_is_finite(
    tmp_path: Path, duration: float
) -> None:
    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow

    raw_config = yaml.safe_load(
        (ROOT / "config/nsga2_bc_global_native_debug.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw_config = copy.deepcopy(raw_config)
    raw_bc = raw_config["bc_global"]
    raw_bc["timeStep_s"] = 0.01
    raw_bc["endTime_s"] = duration
    raw_bc["evaluationTime_s"] = duration
    raw_bc["electricalUpdateInterval_s"] = 1.0
    raw_config["physics"]["end_time_s"] = duration
    raw_config["evaluator"]["end_time_s"] = duration
    raw_bc["onsetCriterion"].update(
        temperature_K=1.0e9,
        minimum_progress=1.0,
        minimum_area_fraction=1.0,
    )
    raw_config["condensed_ignition"]["onset_temperature_K"] = 1.0e9
    raw_config["condensed_ignition"][
        "minimum_conversion_numerical_guard"
    ] = 1.0
    for channel in raw_bc["kinetics"]["channels"]:
        channel["activation_energy_J_per_mol"] = [
            0.0 for _ in channel["alpha_grid"]
        ]
        channel["ln_Af_per_s"] = [
            math.log(0.1) for _ in channel["alpha_grid"]
        ]
        channel["heat_release_J_per_kg"] = 0.0

    evaluator = NSGA2ElectricalSolidWorkflow(
        ROOT, raw_config, tmp_path / "preflame"
    ).evaluator
    n = 9
    anode = torch.zeros((1, n, n), dtype=torch.bool)
    cathode = torch.zeros_like(anode)
    anode[0, 1:3, 1:8] = True
    cathode[0, 6:8, 1:8] = True
    geometry = GeometryBatch(
        geometry_ids=["time-grid"],
        anode=anode,
        cathode=cathode,
        fixed=torch.zeros_like(anode),
        propellant=torch.ones_like(anode),
        grid_size=n,
        domain_size_m=float(evaluator.config["geometry"]["domainSize_m"]),
        minimum_gap_m=float(
            evaluator.config["geometry"]["minimumElectrodeGap_m"]
        ),
    )
    result = run_bc_global_batch(
        geometry,
        evaluator.config,
        evaluator.composition,
        20.0,
        torch.float64,
        save_fields=True,
    )

    times = result["time_s"].detach().cpu().numpy()
    assert result["preflameTermination"]["configuredSteps"] == (
        _preflame_step_count(duration, 0.01)
    )
    assert result["preflameTermination"]["executedSteps"] == len(times)
    assert np.all(np.diff(times) > 0.0)
    assert times[-1] == pytest.approx(duration, rel=0.0, abs=0.0)
    for values in result["histories"].values():
        assert bool(torch.all(torch.isfinite(values)))
    for field in result["finalFields"].values():
        if torch.is_tensor(field):
            assert bool(torch.all(torch.isfinite(field)))
    torch.testing.assert_close(
        result["finalFields"]["alphaChannel1"],
        torch.full_like(
            result["finalFields"]["alphaChannel1"], 0.1 * duration
        ),
        rtol=2e-13,
        atol=2e-15,
    )


@pytest.mark.parametrize("duration", [0.07, 0.075])
def test_propagation_time_grid_is_strict_and_conversion_is_finite(
    tmp_path: Path, duration: float
) -> None:
    handoff = _handoff()
    result = run_condensed_propagation(
        handoff,
        _propagation_config(
            duration_s=duration,
            time_step_s=0.01,
            continued_electrical_heating=False,
        ),
        _bc_config(
            _channel(rate_per_s=0.1, heat_j_per_kg=0.0),
            _channel(rate_per_s=0.1, heat_j_per_kg=0.0),
            weights=(1.0, 0.0),
        ),
        tmp_path,
    )
    fields = np.load(tmp_path / "propagation_fields.npz")
    times = fields["time_after_onset_s"]
    assert len(times) == _propagation_step_count(duration, 0.01)
    assert np.all(np.diff(times) > 0.0)
    assert times[-1] == pytest.approx(duration, rel=0.0, abs=0.0)
    assert np.sum(np.diff(np.concatenate(([0.0], times)))) == pytest.approx(
        duration, rel=0.0, abs=2e-17
    )
    assert np.all(np.isfinite(fields["final_temperature_K"]))
    assert np.all(np.isfinite(fields["final_global_progress"]))
    np.testing.assert_allclose(
        fields["final_global_progress"], 0.9 + 0.1 * duration, rtol=2e-14
    )
    assert math.isfinite(result["finalMaximumTemperature_K"])


def test_propagation_rejects_zero_extent_inventory(tmp_path: Path) -> None:
    handoff = _handoff()
    handoff["xiMax_mol_per_m3"] = 0.0
    with pytest.raises(ValueError, match="Invalid explicit reaction inventory"):
        run_condensed_propagation(
            handoff,
            _propagation_config(),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    [
        "mobileWaterAtOnset_mol_per_m3",
        "electrochemicalLPConsumedAtOnset_mol_per_m3",
    ],
)
def test_propagation_rejects_negative_handoff_inventory(
    tmp_path: Path, field: str
) -> None:
    handoff = _handoff()
    handoff[field] = np.full((2, 2), -1.0)
    with pytest.raises(ValueError, match="Invalid .* handoff field"):
        run_condensed_propagation(
            handoff,
            _propagation_config(),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path,
        )


def test_missing_reaction_inventory_is_rejected_even_with_legacy_flag(
    tmp_path: Path,
) -> None:
    handoff = _handoff()
    inventory_keys = (
        "mobileLPAtOnset_mol_per_m3",
        "pvaReactiveRepeatAtOnset_mol_per_m3",
        "generatedWaterProductAtOnset_mol_per_m3",
        "mobileWaterAtOnset_mol_per_m3",
        "electrochemicalLPConsumedAtOnset_mol_per_m3",
        "xiMax_mol_per_m3",
        "molarMassLP_kg_per_mol",
        "molarMassPVARepeat_kg_per_mol",
        "initialReactiveMass_kg_per_m3",
        "initialMobileLP_mol_per_m3",
        "initialPVARepeat_mol_per_m3",
        "initialMobileWater_mol_per_m3",
    )
    for key in inventory_keys:
        handoff.pop(key)
    with pytest.raises(ValueError, match="Corrected propagation requires"):
        run_condensed_propagation(
            handoff,
            _propagation_config(),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path / "rejected",
        )

    with pytest.raises(ValueError, match="Corrected propagation requires"):
        run_condensed_propagation(
            handoff,
            _propagation_config(
                allow_legacy_heat_only_without_reaction_inventory=True
            ),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path / "legacy",
        )


def test_authorized_propagation_requires_complete_solver_snapshot(
    tmp_path: Path,
) -> None:
    handoff = _handoff()
    handoff.pop("qJAtOnset_W_per_m3")
    with pytest.raises(ValueError, match="complete solver snapshot.*qJAtOnset"):
        run_condensed_propagation(
            handoff,
            _propagation_config(),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path,
        )


def test_propagation_rejects_mobile_lp_cation_anion_mismatch(
    tmp_path: Path,
) -> None:
    handoff = _handoff()
    handoff["cationAtOnset_mol_per_m3"] = np.full((2, 2), 12.0)
    with pytest.raises(ValueError, match="mobile LP is inconsistent"):
        run_condensed_propagation(
            handoff,
            _propagation_config(),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path,
        )

def test_propagation_rejects_rectangular_grid_and_inconsistent_product(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires a square grid"):
        run_condensed_propagation(
            _handoff(shape=(2, 3)),
            _propagation_config(),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path / "rectangular",
        )

    handoff = _handoff()
    handoff["generatedWaterProductAtOnset_mol_per_m3"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="inconsistent with 2\\*xiMax"):
        run_condensed_propagation(
            handoff,
            _propagation_config(),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path / "stoichiometry",
        )


def test_propagation_rejects_nonphysical_thermal_properties(
    tmp_path: Path,
) -> None:
    bc_config = _bc_config(
        _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
        _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
    )
    bc_config["thermal"]["heat_capacity"] = 0.0
    with pytest.raises(RuntimeError, match="thermal properties"):
        run_condensed_propagation(
            _handoff(), _propagation_config(), bc_config, tmp_path
        )


def test_propagation_final_step_rejects_invalid_updated_properties(
    tmp_path: Path,
) -> None:
    bc_config = _bc_config(
        _channel(rate_per_s=0.1, heat_j_per_kg=10.0),
        _channel(rate_per_s=0.1, heat_j_per_kg=0.0),
    )
    bc_config["thermal"]["thermal_conductivity"] = {
        "mode": "reference_arrhenius",
        "reference_value": 0.0,
        "reference_temperature_K": 300.0,
        "activation_energy_J_per_mol": 1.0e12,
    }
    with pytest.raises(RuntimeError, match="accepted updated temperature"):
        run_condensed_propagation(
            _handoff(),
            _propagation_config(duration_s=1.0, time_step_s=1.0),
            bc_config,
            tmp_path,
        )


def test_propagation_fails_closed_on_unstable_total_thermal_cfl(
    tmp_path: Path,
) -> None:
    handoff = _handoff(shape=(3, 3))
    handoff["temperatureAtOnset_K"][1, 1] = 400.0
    bc_config = _bc_config(
        _channel(rate_per_s=1.0e-100, heat_j_per_kg=0.0),
        _channel(rate_per_s=1.0e-100, heat_j_per_kg=0.0),
    )
    bc_config["thermal"]["thermal_conductivity"] = 100.0
    with pytest.raises(RuntimeError, match="total explicit thermal stability CFL"):
        run_condensed_propagation(
            handoff,
            _propagation_config(duration_s=1.0, time_step_s=1.0),
            bc_config,
            tmp_path,
        )


def test_propagation_temperature_cap_is_not_silent(tmp_path: Path) -> None:
    bc_config = _bc_config(
        _channel(rate_per_s=0.1, heat_j_per_kg=1000.0),
        _channel(rate_per_s=0.1, heat_j_per_kg=0.0),
        weights=(1.0, 0.0),
    )
    bc_config["thermal"]["maximumTemperature_K"] = 301.0
    with pytest.raises(RuntimeError, match="temperature clipping"):
        run_condensed_propagation(
            _handoff(), _propagation_config(), bc_config, tmp_path
        )


def test_workflow_does_not_mark_skipped_no_onset_as_propagation_success(
    tmp_path: Path,
) -> None:
    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow

    workflow = object.__new__(NSGA2ElectricalSolidWorkflow)
    workflow.propagation_cfg = _propagation_config()
    workflow.evaluator = SimpleNamespace(
        config={
            "bcGlobal": _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            )
        }
    )
    workflow._propagation_metadata = lambda ind, raster, baseline=False: {}
    workflow._resolved_propagation_config = lambda: _propagation_config()
    individual = SimpleNamespace(metrics={})
    handoff = _handoff()
    handoff.update(
        onsetSucceeded=False,
        onsetReportedByPhysics=False,
        numericallyValidForPropagationHandoff=True,
        propagationHandoffAuthorizationReason="no_condensed_onset",
    )

    metrics = workflow._run_propagation_refinement(
        individual,
        None,
        tmp_path,
        handoff_result={"handoff": handoff, "metrics": {}},
    )

    assert metrics["status"] == "skipped_no_bc_onset"
    assert metrics["onsetSucceeded"] is False
    assert metrics["propagationSucceeded"] is False


def _propagation_boundary_workflow() -> tuple[object, SimpleNamespace, dict]:
    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow

    workflow = object.__new__(NSGA2ElectricalSolidWorkflow)
    workflow.propagation_cfg = _propagation_config()
    workflow.evaluator = SimpleNamespace(
        config={
            "bcGlobal": _bc_config(
                _channel(rate_per_s=0.1, heat_j_per_kg=0.0),
                _channel(rate_per_s=0.1, heat_j_per_kg=0.0),
            )
        }
    )
    workflow._propagation_metadata = lambda ind, raster, baseline=False: {}
    workflow._resolved_propagation_config = lambda: _propagation_config()
    workflow._validate_handoff_reevaluation = lambda ind, result: {
        "preflameHandoffReevaluationConsistent": True
    }
    individual = SimpleNamespace(metrics={})
    return workflow, individual, {"handoff": _handoff(), "metrics": {}}


@pytest.mark.parametrize(
    "exception_type",
    [MemoryError, OSError, KeyError, TypeError, AssertionError],
)
def test_workflow_propagation_infrastructure_and_programming_errors_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception_type: type[Exception]
) -> None:
    import ecsp_nsga2.propagation as propagation

    workflow, individual, handoff_result = _propagation_boundary_workflow()

    def fail(*args: object, **kwargs: object) -> dict:
        del args, kwargs
        raise exception_type("injected non-candidate failure")

    monkeypatch.setattr(propagation, "run_condensed_propagation", fail)
    with pytest.raises(exception_type, match="injected non-candidate failure"):
        workflow._run_propagation_refinement(
            individual,
            None,
            tmp_path / exception_type.__name__,
            handoff_result=handoff_result,
        )
    assert individual.metrics == {}


def test_workflow_converts_only_typed_candidate_propagation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ecsp_nsga2.propagation as propagation

    workflow, individual, handoff_result = _propagation_boundary_workflow()

    def reject_candidate(*args: object, **kwargs: object) -> dict:
        del args, kwargs
        raise PropagationCandidateNumericalError("candidate CFL failure")

    monkeypatch.setattr(
        propagation, "run_condensed_propagation", reject_candidate
    )
    metrics = workflow._run_propagation_refinement(
        individual,
        None,
        tmp_path / "candidate",
        handoff_result=handoff_result,
    )

    assert metrics["status"] == "propagation_failed"
    assert metrics["propagationSucceeded"] is False
    assert metrics["errorType"] == "PropagationCandidateNumericalError"
    assert metrics["message"] == "candidate CFL failure"
    assert individual.metrics["finalUnreactedAreaFraction"] == 1.0
    failure = json.loads(
        (tmp_path / "candidate" / "PROPAGATION_FAILED.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["finalMaximumTemperature_K"] is None


def test_workflow_invalid_global_propagation_config_escapes(
    tmp_path: Path,
) -> None:
    workflow, individual, handoff_result = _propagation_boundary_workflow()
    invalid_config = _propagation_config(time_step_s=0.0)
    workflow._resolved_propagation_config = lambda: invalid_config

    with pytest.raises(PropagationConfigurationError, match="time_step_s"):
        workflow._run_propagation_refinement(
            individual,
            None,
            tmp_path / "invalid-config",
            handoff_result=handoff_result,
        )
    assert individual.metrics == {}
    assert not (
        tmp_path / "invalid-config" / "PROPAGATION_FAILED.json"
    ).exists()


def test_propagation_rejects_falsely_deauthorized_valid_onset(
    tmp_path: Path,
) -> None:
    handoff = _handoff()
    handoff.update(
        onsetSucceeded=False,
        onsetReportedByPhysics=True,
        numericallyValidForPropagationHandoff=True,
        propagationHandoffAuthorizationReason="ignition_and_numerics_valid",
    )
    with pytest.raises(ValueError, match="authorization metadata is inconsistent"):
        run_condensed_propagation(
            handoff,
            _propagation_config(),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path,
        )


def test_preflame_heat_history_is_not_replayed_by_boolean_flags(
    tmp_path: Path,
) -> None:
    handoff = _handoff()
    handoff.update(
        {
            "times_s": np.asarray([0.0, 3.0]),
            "qJ_W_per_m3": np.full((2, 2, 2), 1.0e9),
            "qEchem_W_per_m3": np.full((2, 2, 2), 1.0e9),
        }
    )
    result = run_condensed_propagation(
        handoff,
        _propagation_config(),
        _bc_config(
            _channel(rate_per_s=100.0, heat_j_per_kg=100.0),
            _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
        ),
        tmp_path,
    )
    fields = np.load(tmp_path / "propagation_fields.npz")

    assert result["continuedElectricalHeating"] is False
    assert result["electricalHeatingPolicy"] == "off"
    np.testing.assert_allclose(fields["final_temperature_K"], 310.0)


def test_propagation_inventory_limits_heat_and_keeps_product_water_separate(
    tmp_path: Path,
) -> None:
    handoff = _handoff()
    handoff.update(
        {
            "alphaChannel1AtOnset": np.full((2, 2), 0.01),
            "globalProgressAtOnset": np.full((2, 2), 0.01),
            "mobileLPAtOnset_mol_per_m3": np.full((2, 2), 0.145),
            "cationAtOnset_mol_per_m3": np.full((2, 2), 0.145),
            "anionAtOnset_mol_per_m3": np.full((2, 2), 0.145),
            "pvaReactiveRepeatAtOnset_mol_per_m3": np.full((2, 2), 10.0),
            "generatedWaterProductAtOnset_mol_per_m3": np.full((2, 2), 0.02),
            "mobileWaterAtOnset_mol_per_m3": np.full((2, 2), 50.0),
            "electrochemicalLPConsumedAtOnset_mol_per_m3": np.zeros((2, 2)),
            "xiMax_mol_per_m3": 1.0,
            "initialMobileLP_mol_per_m3": 0.145 + 1.45 * 0.01,
            "initialPVARepeat_mol_per_m3": 10.0 + 0.01,
            "initialMobileWater_mol_per_m3": 50.0,
            "initialReactiveMass_kg_per_m3": (
                0.1 * (0.145 + 1.45 * 0.01)
                + 0.05 * (10.0 + 0.01)
            ),
        }
    )
    result = run_condensed_propagation(
        handoff,
        _propagation_config(continued_electrical_heating=False),
        _bc_config(
            _channel(rate_per_s=100.0, heat_j_per_kg=100.0),
            _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
        ),
        tmp_path,
    )
    fields = np.load(tmp_path / "propagation_fields.npz")

    assert result["reactionInventoryTracked"] is True
    np.testing.assert_allclose(fields["final_global_progress"], 0.11)
    np.testing.assert_allclose(fields["final_temperature_K"], 310.0)
    np.testing.assert_allclose(fields["final_mobile_lp_mol_per_m3"], 0.0, atol=1e-15)
    np.testing.assert_allclose(fields["final_mobile_water_mol_per_m3"], 50.0)
    np.testing.assert_allclose(
        fields["final_generated_water_product_mol_per_m3"], 0.22
    )


def test_propagation_inventory_ratios_do_not_overflow_raw_domain_sums(
    tmp_path: Path,
) -> None:
    handoff = _handoff()
    shape = (2, 2)
    initial_lp = 5.0e307
    initial_pva = 5.0e307
    xi_max = initial_lp / 1.45
    progress = 0.5
    mobile_lp = initial_lp - 1.45 * xi_max * progress
    reactive_pva = initial_pva - xi_max * progress
    generated_water = 2.0 * xi_max * progress
    initial_reactive_mass = initial_lp + initial_pva
    handoff.update(
        {
            "alphaChannel1AtOnset": np.full(shape, progress),
            "alphaChannel2AtOnset": np.ones(shape),
            "globalProgressAtOnset": np.full(shape, progress),
            "mobileLPAtOnset_mol_per_m3": np.full(shape, mobile_lp),
            "cationAtOnset_mol_per_m3": np.full(shape, mobile_lp),
            "anionAtOnset_mol_per_m3": np.full(shape, mobile_lp),
            "pvaReactiveRepeatAtOnset_mol_per_m3": np.full(
                shape, reactive_pva
            ),
            "generatedWaterProductAtOnset_mol_per_m3": np.full(
                shape, generated_water
            ),
            "mobileWaterAtOnset_mol_per_m3": np.ones(shape),
            "electrochemicalLPConsumedAtOnset_mol_per_m3": np.zeros(shape),
            "xiMax_mol_per_m3": xi_max,
            "molarMassLP_kg_per_mol": 1.0,
            "molarMassPVARepeat_kg_per_mol": 1.0,
            "initialReactiveMass_kg_per_m3": initial_reactive_mass,
            "initialMobileLP_mol_per_m3": initial_lp,
            "initialPVARepeat_mol_per_m3": initial_pva,
            "initialMobileWater_mol_per_m3": 1.0,
        }
    )

    # These are deliberately valid per-cell inventories whose old raw domain
    # sums overflowed to inf/inf during both conservation and U_rem checks.
    with np.errstate(over="ignore"):
        assert np.isinf(
            np.sum(
                handoff["mobileLPAtOnset_mol_per_m3"]
                + 1.45 * xi_max * handoff["globalProgressAtOnset"]
            )
        )
        assert math.isinf(initial_reactive_mass * np.prod(shape))

    result = run_condensed_propagation(
        handoff,
        _propagation_config(),
        _bc_config(
            _channel(rate_per_s=1.0e-100, heat_j_per_kg=0.0),
            _channel(rate_per_s=1.0e-100, heat_j_per_kg=0.0),
        ),
        tmp_path,
    )

    expected_remaining = (mobile_lp + reactive_pva) / initial_reactive_mass
    assert result["status"] == "complete"
    assert math.isfinite(result["finalRemainingReactiveMassFraction"])
    assert result["finalRemainingReactiveMassFraction"] == pytest.approx(
        expected_remaining, rel=2.0e-15
    )


def test_arrival_cv_and_unarrived_coverage_are_independent() -> None:
    arrival = np.asarray([[1.0, 2.0], [np.nan, np.nan]])
    timing_cv, coverage = _arrival_time_statistics(
        arrival, np.ones_like(arrival, dtype=bool)
    )

    assert timing_cv == pytest.approx(1.0 / 3.0)
    assert coverage == pytest.approx(0.5)


def test_propagation_extreme_finite_domain_fails_closed_before_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "unrepresentable-domain"
    with pytest.raises(ValueError, match="cell area is not representable"):
        run_condensed_propagation(
            _handoff(),
            _propagation_config(domain_size_m=1.0e307),
            _bc_config(
                _channel(rate_per_s=0.1, heat_j_per_kg=0.0),
                _channel(rate_per_s=0.1, heat_j_per_kg=0.0),
            ),
            output_dir,
        )
    assert not (output_dir / "propagation_metrics.json").exists()
    assert not (output_dir / "propagation_fields.npz").exists()


def test_propagation_regression_speed_cancels_cell_area_before_arithmetic() -> None:
    velocity = _effective_regression_velocity(
        previous_unreacted_cells=2,
        unreacted_cells=1,
        previous_front_edges=2,
        front_edges=2,
        dx=1.0e307,
        step_dt=1.0,
    )
    assert math.isfinite(velocity)
    assert velocity == pytest.approx(5.0e306)


def test_propagation_harmonic_and_cfl_algebra_avoid_intermediate_overflow() -> None:
    extreme = np.asarray([[1.0e308]], dtype=np.float64)
    harmonic = _propagation_harmonic_mean(extreme, extreme)
    assert np.isfinite(harmonic).all()
    np.testing.assert_array_equal(harmonic, extreme)

    conductivity = np.full((3, 3), 1.0e308, dtype=np.float64)
    capacity = np.ones_like(conductivity)
    mask = np.ones_like(conductivity, dtype=bool)
    cfl = _propagation_thermal_cfl(
        conductivity,
        capacity,
        mask,
        dx=1.0e160,
        step_dt=1.0,
    )
    assert np.isfinite(cfl).all()
    assert cfl[1, 1] == pytest.approx(4.0e-12, rel=2.0e-15)

    constant_temperature = np.full((3, 3), 300.0)
    conduction = _propagation_div_k_grad(
        constant_temperature, conductivity, mask, dx=1.0e160
    )
    np.testing.assert_array_equal(conduction, np.zeros_like(conduction))


@pytest.mark.parametrize(
    ("preflame", "metric_key"),
    [
        ([0.0, 1.0, float("inf"), 1.0], None),
        ([0.0, 1.0, 2.0, 1.0], "meanEffectiveRegressionVelocity_m_per_s"),
    ],
)
def test_propagation_refinement_rejects_nonfinite_objectives(
    preflame: list[float], metric_key: str | None
) -> None:
    metrics = {
        "finalUnreactedAreaFraction": 0.5,
        "establishedTimeAfterOnset_s": 1.0,
        "meanEffectiveRegressionVelocity_m_per_s": 0.1,
        "reactionFrontNonuniformity": 0.2,
    }
    if metric_key is not None:
        metrics[metric_key] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        final_refinement_objectives(preflame, metrics)


def test_explicit_post_onset_history_is_rejected_without_closed_coupled_solver(
    tmp_path: Path,
) -> None:
    handoff = _handoff()
    handoff.update(
        {
            "alphaChannel1AtOnset": np.ones((2, 2)),
            "globalProgressAtOnset": np.ones((2, 2)),
            "continuedElectricalHeating": False,
            "mobileLPAtOnset_mol_per_m3": np.ones((2, 2)),
            "pvaReactiveRepeatAtOnset_mol_per_m3": np.ones((2, 2)),
            "generatedWaterProductAtOnset_mol_per_m3": np.full((2, 2), 2.0),
            "mobileWaterAtOnset_mol_per_m3": np.full((2, 2), 2.0),
            "electrochemicalLPConsumedAtOnset_mol_per_m3": np.zeros((2, 2)),
            "xiMax_mol_per_m3": 1.0,
            "postOnsetTimes_s": np.asarray([0.0, 1.0]),
            "postOnsetQJ_W_per_m3": np.full((2, 2, 2), 10.0),
            "postOnsetQEchem_W_per_m3": np.full((2, 2, 2), 20.0),
            "postOnsetSaltSink_mol_per_m3_s": np.full((2, 2, 2), 0.1),
            "postOnsetWaterSink_mol_per_m3_s": np.full((2, 2, 2), 0.2),
        }
    )
    with pytest.raises(ValueError, match="not a closed electrochemical"):
        run_condensed_propagation(
            handoff,
            _propagation_config(
                continued_electrical_heating=True,
                electrical_heating_mode="provided_post_onset_history",
            ),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path,
        )


def test_legacy_stale_replay_requires_explicit_escape_hatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Stale pre-flame"):
        run_condensed_propagation(
            _handoff(),
            _propagation_config(
                continued_electrical_heating=True,
                electrical_heating_mode="legacy_preflame_history_replay"
            ),
            _bc_config(
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
                _channel(rate_per_s=1.0, heat_j_per_kg=0.0),
            ),
            tmp_path,
        )


def test_preflame_candidate_freezes_at_first_onset_and_history_tail_is_filled(
    tmp_path: Path,
) -> None:
    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow

    config = yaml.safe_load(
        (ROOT / "config/nsga2_bc_global_native_debug.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = copy.deepcopy(config)
    config["bc_global"]["endTime_s"] = 0.01
    config["bc_global"]["evaluationTime_s"] = 0.01
    config["bc_global"]["onsetCriterion"].update(
        temperature_K=1.0,
        minimum_progress=1.0e-3,
        minimum_area_fraction=1.0,
    )
    config["condensed_ignition"]["onset_temperature_K"] = 1.0
    config["condensed_ignition"]["minimum_conversion_numerical_guard"] = 1.0e-3
    for channel in config["bc_global"]["kinetics"]["channels"]:
        channel["activation_energy_J_per_mol"] = [
            0.0 for _ in channel["alpha_grid"]
        ]
        channel["ln_Af_per_s"] = [0.0 for _ in channel["alpha_grid"]]
        channel["heat_release_J_per_kg"] = 0.0

    workflow = NSGA2ElectricalSolidWorkflow(ROOT, config, tmp_path / "workflow")
    evaluator = workflow.evaluator
    anode = np.zeros((32, 32), dtype=bool)
    cathode = np.zeros_like(anode)
    anode[5:8, 5:26] = True
    cathode[24:27, 5:26] = True
    geometry = evaluator._build_geometry_batch(
        [
            (
                anode,
                cathode,
                {"geometry_id": "freeze_regression"},
                tmp_path / "case",
            )
        ]
    )
    result = run_bc_global_batch(
        geometry,
        evaluator.config,
        evaluator.composition,
        20.0,
        torch.float64,
        save_fields=True,
        stop_on_onset=True,
    )

    termination = result["preflameTermination"]
    assert termination["stopOnOnsetRequested"] is True
    assert termination["terminatedAfterAllCandidatesReachedOnset"] is True
    assert termination["executedSteps"] == 1
    assert result["postIgnitionClosure"].startswith("candidate_state_power_and_heat_frozen")
    instantaneous = {
        "current_A",
        "power_W",
        "maximumChannel1Rate_per_s",
        "maximumChannel2Rate_per_s",
        "currentCongestion",
        "speciesLimiterFraction",
        "temperatureCapFraction",
        "chemicalRateCapFraction",
        "equation32ElectricalHeatRate_W",
        "meanJouleHeat_W_per_m3",
        "meanElectrochemicalHeat_W_per_m3",
        "meanChemicalHeat_W_per_m3",
        "anodeCathodeCurrentMismatch",
        "thermalDiffusiveCFL",
        "thermalStabilityCFL",
    }
    for name, history in result["histories"].items():
        if name in instantaneous:
            torch.testing.assert_close(
                history[1:], torch.zeros_like(history[1:]), rtol=0.0, atol=0.0
            )
        else:
            torch.testing.assert_close(
                history,
                history[0:1].expand_as(history),
                rtol=0.0,
                atol=0.0,
            )

    completed = run_bc_global_batch(
        geometry,
        evaluator.config,
        evaluator.composition,
        20.0,
        torch.float64,
        save_fields=True,
        stop_on_onset=False,
    )
    assert completed["evaluationStateCompletion"] == {
        "completedAtCommonEvaluationTime": True,
        "earlyOnsetLaneCount": 1,
        "electricalHeatingAfterOnset": False,
        "onsetFieldsRemainImmutable": True,
    }
    torch.testing.assert_close(
        completed["evaluationStateTime_s"],
        torch.full_like(completed["evaluationStateTime_s"], 0.01),
        rtol=0.0,
        atol=0.0,
    )
    assert completed["ignitionDelay_s"].item() < 0.01
    assert completed["preflameHistorySemantics"][
        "representsObjectiveContinuation"
    ] is False

    onset = completed["onsetFields"]
    evaluated = completed["evaluationFields"]
    xi_max = float(onset["xiMax_mol_per_m3"])
    progress_increment = (
        evaluated["globalProgress"] - onset["globalProgressAtOnset"]
    )
    assert bool(torch.all(progress_increment > 0.0))
    torch.testing.assert_close(
        evaluated["cation_mol_per_m3"],
        onset["mobileLPAtOnset_mol_per_m3"]
        - 1.45 * xi_max * progress_increment,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    torch.testing.assert_close(
        evaluated["anion_mol_per_m3"],
        evaluated["cation_mol_per_m3"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        evaluated["pvaReactiveRepeat_mol_per_m3"],
        onset["pvaReactiveRepeatAtOnset_mol_per_m3"]
        - xi_max * progress_increment,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    torch.testing.assert_close(
        evaluated["generatedWaterProduct_mol_per_m3"],
        onset["generatedWaterProductAtOnset_mol_per_m3"]
        + 2.0 * xi_max * progress_increment,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    torch.testing.assert_close(
        evaluated["water_mol_per_m3"],
        onset["mobileWaterAtOnset_mol_per_m3"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        evaluated["electrochemicalLPConsumed_mol_per_m3"],
        onset["electrochemicalLPConsumedAtOnset_mol_per_m3"],
        rtol=0.0,
        atol=0.0,
    )

    initial_mass = float(onset["initialReactiveMass_kg_per_m3"])
    reactive_mass = (
        float(onset["molarMassLP_kg_per_mol"])
        * evaluated["cation_mol_per_m3"]
        + float(onset["molarMassPVARepeat_kg_per_mol"])
        * evaluated["pvaReactiveRepeat_mol_per_m3"]
    )
    expected_remaining = reactive_mass.mean() / initial_mass
    torch.testing.assert_close(
        completed["remainingReactiveMassFractionAtEvaluationTime"],
        expected_remaining.reshape(1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert completed[
        "remainingReactiveMassFractionAtEvaluationTime"
    ].item() < completed["histories"]["remainingReactiveMassFraction"][-1].item()
    torch.testing.assert_close(
        completed["equation32ElectricalHeatRateAtEvaluationTime_W"],
        torch.zeros_like(
            completed["equation32ElectricalHeatRateAtEvaluationTime_W"]
        ),
        rtol=0.0,
        atol=0.0,
    )
