from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
from ecsp_v6.composition import composition_summary


ROOT = Path(__file__).resolve().parents[2]


def _debug_config() -> dict:
    return yaml.safe_load(
        (ROOT / "config/nsga2_bc_global_preflame_propagation_debug.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_cured_water_mass_fraction_is_final_specimen_basis() -> None:
    cfg = yaml.safe_load((ROOT / "config/default_lp_pva.yaml").read_text(encoding="utf-8"))
    # Exact third-paper wet recipe, normalised to the legacy 35 g recipe basis.
    cfg["composition"]["masses_g"] = {
        "LP": 35.0 * 0.3158,
        "water": 35.0 * 0.5842,
        "PVA": 35.0 * 0.09,
        "glycerol": 0.0,
        "boric_acid": 35.0 * 0.01,
    }
    cfg["composition"]["cured_water_mass_fraction"] = 0.20
    summary = composition_summary(cfg)
    assert np.isclose(summary["cured_water_mass_fraction"], 0.20, atol=1e-12)
    assert np.isclose(sum(summary["analyzed_mass_fractions"].values()), 1.0)
    assert summary["analyzed_mass_fractions"]["glycerol"] == 0.0
    dry = summary["dry_mass_fractions"]
    assert np.isclose(dry["LP"], 31.58 / (31.58 + 9.0 + 1.0))
    assert np.isclose(dry["PVA"], 9.0 / (31.58 + 9.0 + 1.0))


def test_bc_global_full_debug_pipeline_and_staggered_reference(tmp_path: Path) -> None:
    config = _debug_config()
    workflow = NSGA2ElectricalSolidWorkflow(
        package_root=ROOT,
        config=config,
        workdir=tmp_path / "run",
        allow_debug_physics=False,
    )
    recommendation = workflow.run()
    assert recommendation.objectives is not None
    assert recommendation.objectives.shape == (8,)
    assert recommendation.metrics["modelStatus"].startswith("bc_global_")
    assert recommendation.metrics["undecomposedMetricType"] == (
        "remaining_reactive_LP_plus_PVA_mass_fraction"
    )
    assert recommendation.metrics["paperEquationUse"]["used"] == [
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        13,
        15,
        16,
        17,
        22,
        23,
        24,
    ]
    assert recommendation.metrics["paperEquationUse"]["equation32_role"].startswith(
        "postprocessing"
    )
    assert recommendation.metrics["equation32ElectricalHeatEnergy_J"] >= 0.0
    assert recommendation.metrics["equation32ElectricalHeatRateAtEvaluationTime_W"] >= 0.0
    final = tmp_path / "run" / "final"
    required = [
        "preflame_final_pareto_designs.csv",
        "propagation_selection.csv",
        "all_propagation_refined_designs.csv",
        "final_pareto_designs.csv",
        "recommended_design.json",
        "area_matched_staggered.json",
        "area_matched_staggered_propagation.json",
        "recommended_vs_area_matched_staggered_propagation.json",
    ]
    assert all((final / name).is_file() for name in required)
    baseline_propagation = yaml.safe_load(
        (final / "area_matched_staggered_propagation.json").read_text(encoding="utf-8")
    )
    assert baseline_propagation["propagation_metrics"]["propagationSucceeded"] is True
    run_complete = yaml.safe_load(
        (tmp_path / "run" / "RUN_COMPLETE.json").read_text(encoding="utf-8")
    )
    assert run_complete["bc_global_preflame_used"] is True
    assert run_complete["post_onset_condensed_propagation_used"] is True
    assert run_complete["gas_phase_cfd_used"] is False


def test_bc_global_release_contract_is_complete() -> None:
    import json

    from ecsp_v6.physics.bc_global import bc_model_contract

    cfg = yaml.safe_load(
        (ROOT / "config/nsga2_bc_global_preflame_propagation_a100.yaml").read_text(
            encoding="utf-8"
        )
    )
    contract = bc_model_contract()
    assert contract["paper_equations_used"] == [
        2, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 22, 23, 24
    ]
    assert contract["paper_equations_not_used_in_preflame"] == [
        3, 18, 25, 26, 27, 28, 29, 30, 31
    ]
    assert contract["equation_32_role"] == (
        "postprocessing_identity_only_not_added_as_a_second_heat_source"
    )
    assert contract["global_reaction"].startswith("1.45 LiClO4")
    assert contract["legacy_proxy_closures_used"] is False
    assert contract["gas_phase_cfd_used"] is False
    assert len(contract["solver_onset_snapshot_fields"]) == 18
    required_handoff = set(
        contract["required_authorized_propagation_handoff_fields"]
    )
    assert set(contract["solver_onset_snapshot_fields"]) < required_handoff
    version = json.loads((ROOT / "VERSION.json").read_text(encoding="utf-8"))
    assert version["post_onset_condensed_propagation"][
        "required_propagation_handoff_fields"
    ] == contract["required_authorized_propagation_handoff_fields"]
    assert {
        "propellantMask",
        "initialMobileLP_mol_per_m3",
        "initialPVARepeat_mol_per_m3",
        "initialMobileWater_mol_per_m3",
        "handoffSchemaVersion",
        "onsetReportedByPhysics",
        "numericallyValidForPropagationHandoff",
        "propagationHandoffAuthorizationReason",
        "continuedElectricalHeating",
    } <= required_handoff

    comp = cfg["physics"]["composition"]
    assert np.isclose(comp["lithium_perchlorate_mass_fraction"], 0.3158)
    assert np.isclose(comp["water_mass_fraction"], 0.5842)
    assert np.isclose(comp["pva_mass_fraction"], 0.09)
    assert np.isclose(comp["boric_acid_mass_fraction"], 0.01)
    assert comp["glycerol_mass_fraction"] == 0.0
    assert comp["tungsten_mass_fraction"] == 0.0
    assert np.isclose(cfg["physics"]["cured_water_mass_fraction"], 0.20)

    assert cfg["optimization"]["objectives_minimise"] == [
        "ignition_delay_s",
        "area_undecomposed_fraction_at_evaluation_time",
        "minimum_ignition_voltage_V",
        "current_congestion",
    ]
    onset = cfg["bc_global"]["onsetCriterion"]
    assert np.isclose(onset["temperature_K"], 523.15)
    assert np.isclose(onset["minimum_progress"], 0.01)
    assert np.isclose(onset["minimum_area_fraction"], 0.01)
    assert cfg["evaluator"]["base_overrides"]["coupled"][
        "usePaperMassTransferSaturation"
    ] is False
    assert cfg["evaluator"]["base_overrides"]["interface"]["blocking"][
        "passivation"
    ]["enabled"] is False
    assert cfg["evaluator"]["base_overrides"]["interface"]["blocking"][
        "gasCoverage"
    ]["enabled"] is False
    assert cfg["evaluator"]["base_overrides"]["interface"][
        "includeActivationHeat"
    ] is False
    assert cfg["propagation_refinement"]["enabled"] is True
    assert 0.10 <= cfg["propagation_refinement"]["selection"][
        "additional_fraction"
    ] <= 0.20
    assert cfg["workflow"]["gas_phase_cfd_enabled"] is False


def test_bc_global_production_staggered_is_constructible() -> None:
    from ecsp_nsga2.baselines import generate_area_matched_staggered
    from ecsp_nsga2.geometry import GeometryLimits

    cfg = yaml.safe_load(
        (ROOT / "config/nsga2_bc_global_preflame_propagation_a100.yaml").read_text(
            encoding="utf-8"
        )
    )
    limits = GeometryLimits(**cfg["geometry"])
    _, raster, parameters = generate_area_matched_staggered(
        limits,
        physics_grid_size=int(cfg["evaluator"]["grid_size"]),
        target_area_fraction_per_polarity=float(
            cfg["geometry"]["target_area_fraction_per_polarity"]
        ),
        fingers_per_polarity=2,
        target_interdigitation_overlap_fraction=0.5,
        minimum_interdigitation_overlap_fraction=0.45,
        maximum_gap_safety_pixels=8,
    )
    assert raster.constraint_violation <= 1.0e-12
    assert parameters.design_vertical_overlap_fraction >= 0.45
    assert parameters.physics_vertical_overlap_fraction >= 0.45
