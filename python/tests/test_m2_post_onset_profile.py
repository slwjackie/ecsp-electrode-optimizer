from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ecsp_preflame.geometry import RasterizedGeometry
from ecsp_preflame.candidate_ranking import CandidateEvaluation
from ecsp_preflame.post_onset import validate_post_onset_config
from ecsp_preflame.propagation import PropagationCandidateInputError
from ecsp_preflame.evaluation_workflow import (
    EvaluationWorkflow,OBJECTIVE_NAMES,REFINED_OBJECTIVE_NAMES,
)
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case
from ecsp_reactive.condensed.handoff import PreflameReactiveHandoffAdapter, model_digest


ROOT = Path(__file__).resolve().parents[2]
M2_CONFIG = ROOT / "config" / "preflame_reactive_m2cpu.yaml"
A100_CONFIG = ROOT / "config" / "preflame_reactive_a100_gpuonly.yaml"

PRE_FLAME_KEYS = (
    "project",
    "workflow",
    "evaluation",
    "geometry",
    "physics",
    "condensed_ignition",
    "minimum_ignition_voltage_search",
    "baselines",
    "preflame_model",
    "evaluator",
)

# Config-name/metadata migration only; model and propagation digests below
# retain the exact pre-refactor reference values.
EXPECTED_PRE_FLAME_SHA256 = {
    M2_CONFIG.name: "e5ba4d4ac51f821dbd884f7875af7144b12de7a3d24f34a558842a3099f17ec9",
    A100_CONFIG.name: "2819503ef5fbaebe1b2d7bb80b5d626329e79072b24aff3ee91d7a06d56a1a64",
}
EXPECTED_PREFLAME_MODEL_SHA256 = (
    "b481bfe7308f917b7fe6559a01bb4a4f41b76aab38823f3f9a2a22cb37521730"
)
EXPECTED_PROPAGATION_REFINEMENT_SHA256 = (
    "5365d9b0ddcb5e123ad9d364dc5ac2388c3ca5f45c81211853930b2aecb6e96a"
)

EXPECTED_CHEMISTRY_CONTROLS = {
    "chemistry_integration_mode": "local_adaptive_thermochemical",
    "chemistry_relative_tolerance": 1.0e-7,
    "chemistry_absolute_tolerance": 1.0e-10,
    "chemistry_temperature_tolerance_K": 1.0e-4,
    "maximum_chemistry_corrector_iterations": 8,
    "maximum_chemistry_depletion_iterations": 40,
    "maximum_chemistry_local_refinements": 10,
    "maximum_chemistry_reaction_coordinate_steps": 64,
    "progress_log_interval_steps": 250,
    "progress_log_interval_wall_s": 30.0,
}

REMOVED_POST_ONSET_CAP_OR_SUBCYCLE_KEYS = {
    "maximum_chemical_rate_cap_fraction",
    "maximum_channel_increment",
    "maximum_chemistry_subcycles_per_pde_step",
    "maximum_chemistry_panel_attempts_per_cell",
}


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("path", [M2_CONFIG, A100_CONFIG])
def test_production_preflame_sections_are_immutable(path: Path) -> None:
    config = _load(path)
    protected = {key: config[key] for key in PRE_FLAME_KEYS}

    assert _digest(protected) == EXPECTED_PRE_FLAME_SHA256[path.name]
    assert model_digest(config["preflame_model"]) == EXPECTED_PREFLAME_MODEL_SHA256
    assert config["evaluation"]["population_size"] == 200
    assert "generations" not in config["evaluation"]
    assert "algorithm" not in config["evaluation"]
    assert tuple(config["evaluation"]["objectives_minimise"]) == OBJECTIVE_NAMES
    assert config["condensed_ignition"]["onset_temperature_K"] == 523.15
    assert config["preflame_model"]["kinetics"]["maximum_rate_per_s"] == 1000.0


def test_m2_propagation_objectives_and_settings_are_immutable() -> None:
    config = _load(M2_CONFIG)
    propagation = config["propagation_refinement"]

    assert _digest(propagation) == EXPECTED_PROPAGATION_REFINEMENT_SHA256
    assert tuple(propagation["combined_final_objectives"]) == REFINED_OBJECTIVE_NAMES
    assert propagation["duration_s"] == 1.0
    assert propagation["time_step_s"] == 0.00025
    assert propagation["continued_electrical_heating"] is False
    assert propagation["electrical_heating_mode"] == "off"


@pytest.mark.parametrize("path", [M2_CONFIG, A100_CONFIG])
def test_production_uses_only_local_adaptive_chemistry_controls(path: Path) -> None:
    config = _load(path)
    reactive = config["post_onset"]["reactive_euler"]

    assert {
        key: reactive[key] for key in EXPECTED_CHEMISTRY_CONTROLS
    } == EXPECTED_CHEMISTRY_CONTROLS
    assert REMOVED_POST_ONSET_CAP_OR_SUBCYCLE_KEYS.isdisjoint(reactive)
    assert reactive["eos"]["A_Pa"] == 101325.0
    assert reactive["eos"]["B_Pa"] == 2.0e9
    assert reactive["eos"]["N"] == 7.0
    assert reactive["caloric_closure"] == "heat_capacity_integral_plus_tait_cold_energy"
    assert reactive["boundary"] == "reflective"
    assert reactive["riemann_solver"] == "hllc"
    assert reactive["cfl"] == 0.35
    assert reactive["thermal_cfl"] == 0.7

    validated = validate_post_onset_config(
        config["post_onset"],
        for_optimization=True,
    )
    assert validated["reactive_euler"]["chemistry_integration_mode"] == (
        "local_adaptive_thermochemical"
    )


def test_m2_and_a100_keep_only_the_intended_runtime_differences() -> None:
    m2 = _load(M2_CONFIG)["post_onset"]
    a100 = _load(A100_CONFIG)["post_onset"]

    assert m2["reactive_euler"] == a100["reactive_euler"]
    assert m2["compare_backends"] is False
    assert a100["compare_backends"] is True
    assert m2["execution"]["backend"] == "numpy_cpu"
    assert m2["execution"]["device"] == "cpu"
    assert m2["execution"]["batch_size"] == 1
    assert m2["execution"]["cpu_workers"] == 6
    assert a100["execution"]["backend"] == "torch_batch"
    assert a100["execution"]["device"] == "cuda"
    assert a100["execution"]["batch_size"] == 8


def _compatible_handoff(preflame_model: dict) -> tuple[dict, dict]:
    shape = (8, 8)
    field = lambda value: np.full(shape, value, dtype=np.float64)
    anode = np.zeros(shape, dtype=bool)
    cathode = np.zeros(shape, dtype=bool)
    anode[:, 0] = True
    cathode[:, -1] = True

    weights = np.asarray(preflame_model["kinetics"]["mass_conversion_weights"])
    alpha = np.asarray([0.2, 0.2])
    progress = float(weights @ alpha)
    xi_max = 100.0
    extent = xi_max * progress
    initial_lp = 145.0
    initial_pva = 100.0
    molar_mass_lp = 0.1
    molar_mass_pva = 0.05

    handoff = {
        "handoffSchemaVersion": "ecsp_bc_surface_onset_v8.2.0",
        "onsetSucceeded": True,
        "onsetReportedByPhysics": True,
        "numericallyValidForPropagationHandoff": True,
        "propagationHandoffAuthorizationReason": "ignition_and_numerics_valid",
        "ignitionDelay_s": 0.1,
        "continuedElectricalHeating": False,
        "preflameModelConfigSHA256": EXPECTED_PREFLAME_MODEL_SHA256,
        "temperatureAtOnset_K": field(523.15),
        "propellantMask": np.ones(shape, dtype=bool),
        "anodeContactMask": anode,
        "cathodeContactMask": cathode,
        "alphaChannel1AtOnset": field(alpha[0]),
        "alphaChannel2AtOnset": field(alpha[1]),
        "globalProgressAtOnset": field(progress),
        "xiMax_mol_per_m3": xi_max,
        "initialMobileLP_mol_per_m3": initial_lp,
        "initialPVARepeat_mol_per_m3": initial_pva,
        "initialMobileWater_mol_per_m3": 50.0,
        "molarMassLP_kg_per_mol": molar_mass_lp,
        "molarMassPVARepeat_kg_per_mol": molar_mass_pva,
        "initialReactiveMass_kg_per_m3": (
            molar_mass_lp * initial_lp + molar_mass_pva * initial_pva
        ),
        "cationAtOnset_mol_per_m3": field(initial_lp - 1.45 * extent),
        "anionAtOnset_mol_per_m3": field(initial_lp - 1.45 * extent),
        "mobileLPAtOnset_mol_per_m3": field(initial_lp - 1.45 * extent),
        "pvaReactiveRepeatAtOnset_mol_per_m3": field(initial_pva - extent),
        "generatedWaterProductAtOnset_mol_per_m3": field(2.0 * extent),
        "mobileWaterAtOnset_mol_per_m3": field(50.0),
        "electrochemicalLPConsumedAtOnset_mol_per_m3": field(0.0),
        "potentialAtOnset_V": field(0.0),
        "qJAtOnset_W_per_m3": field(0.0),
        "qEchemAtOnset_W_per_m3": field(0.0),
    }
    propagation = {
        "duration_s": 0.001,
        "time_step_s": 0.001,
        "snapshot_interval_s": 0.001,
        "domain_size_m": 0.02,
        "density_kg_per_m3": 1000.0,
        "surface_layer_thickness_m": 0.001,
        "gas_constant_J_per_molK": 8.314462618,
        "continued_electrical_heating": False,
        "electrical_heating_mode": "off",
    }
    return handoff, propagation


def test_authorized_v82_handoff_digest_remains_compatible() -> None:
    config = _load(M2_CONFIG)
    preflame_model = config["preflame_model"]
    reactive = config["post_onset"]["reactive_euler"]
    handoff, propagation = _compatible_handoff(preflame_model)

    adapted = PreflameReactiveHandoffAdapter(
        propagation,
        preflame_model,
        reactive,
    ).adapt(handoff)
    assert adapted.audit["preflame_model_config_sha256"] == EXPECTED_PREFLAME_MODEL_SHA256
    assert adapted.audit["onset_time_s"] == 0.1

    changed_bc = copy.deepcopy(preflame_model)
    changed_bc["kinetics"]["channels"][0]["ln_Af_per_s"][0] += 1.0e-12
    with pytest.raises(PropagationCandidateInputError, match="hash mismatch"):
        PreflameReactiveHandoffAdapter(
            propagation,
            changed_bc,
            reactive,
        ).adapt(handoff)

    wrong_schema = copy.deepcopy(handoff)
    wrong_schema["handoffSchemaVersion"] = "changed"
    with pytest.raises(PropagationCandidateInputError, match="authorized v8.2"):
        PreflameReactiveHandoffAdapter(
            propagation,
            preflame_model,
            reactive,
        ).adapt(wrong_schema)


@pytest.mark.parametrize("hashes,valid", [
    ({"bcGlobalConfigSHA256": EXPECTED_PREFLAME_MODEL_SHA256}, True),
    ({"bcGlobalConfigSHA256": "wrong"}, False),
    ({"preflameModelConfigSHA256": EXPECTED_PREFLAME_MODEL_SHA256,
      "bcGlobalConfigSHA256": "wrong"}, False),
    ({"preflameModelConfigSHA256": "wrong",
      "bcGlobalConfigSHA256": EXPECTED_PREFLAME_MODEL_SHA256}, False),
])
def test_handoff_model_hash_aliases_never_hide_a_mismatch(hashes, valid):
    config = _load(M2_CONFIG)
    model = config["preflame_model"]
    handoff, propagation = _compatible_handoff(model)
    handoff.pop("preflameModelConfigSHA256")
    handoff.update(hashes)
    adapter = PreflameReactiveHandoffAdapter(
        propagation, model, config["post_onset"]["reactive_euler"])
    if valid:
        assert adapter.adapt(handoff).audit["preflame_model_config_sha256"] == EXPECTED_PREFLAME_MODEL_SHA256
    else:
        with pytest.raises(PropagationCandidateInputError, match="hash mismatch"):
            adapter.adapt(handoff)
