from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from ecsp_nsga2.evaluator import DirectCondensedV772NoFEvaluator
from ecsp_v6.config import load_config
from ecsp_v6.physics.composition_model import build_composition
from ecsp_v6.physics.electrochem import (
    _balance_nonlinear_robin_gauge,
    initial_state,
    transport_fields,
)
from ecsp_v6.physics.geometry import GeometryBatch


ROOT = Path(__file__).resolve().parents[2]


def _two_case_geometry(config: dict, grid_size: int = 17) -> GeometryBatch:
    anode = torch.zeros((2, grid_size, grid_size), dtype=torch.bool)
    cathode = torch.zeros_like(anode)

    # Case 0: left/right electrodes. Case 1: bottom/top electrodes.
    anode[0, 2:-2, 2] = True
    cathode[0, 2:-2, -3] = True
    anode[1, 2, 2:-2] = True
    cathode[1, -3, 2:-2] = True
    fixed = anode | cathode
    return GeometryBatch(
        geometry_ids=["vertical", "horizontal"],
        anode=anode,
        cathode=cathode,
        fixed=fixed,
        propellant=~fixed,
        grid_size=grid_size,
        domain_size_m=float(config["geometry"]["domainSize_m"]),
        minimum_gap_m=float(config["geometry"]["minimumElectrodeGap_m"]),
    )


def test_exact_global_gauge_balances_batched_bv_currents_fp32() -> None:
    config = load_config(ROOT / "config/m2_mps.yaml")
    geometry = _two_case_geometry(config)
    composition = build_composition(config)
    state = initial_state(geometry, config, composition, 260.0, torch.float32)
    transport = transport_fields(state, geometry, config, composition)

    # Deliberately force opposite one-sided starting gauges.  The production
    # defect produced dphi=dI=0 with balance=1 from states of this kind.
    state["potential"][0] = torch.where(
        geometry.propellant[0],
        torch.zeros_like(state["potential"][0]),
        state["potential"][0],
    )
    state["potential"][1] = torch.where(
        geometry.propellant[1],
        torch.full_like(state["potential"][1], 260.0),
        state["potential"][1],
    )

    spacing = geometry.domain_size_m / (geometry.grid_size - 1)
    _, boundary, diagnostics = _balance_nonlinear_robin_gauge(
        state["potential"],
        state,
        transport,
        geometry,
        config,
        composition,
        260.0,
        spacing,
    )

    assert bool(torch.all(diagnostics["converged"]))
    assert bool(torch.all(diagnostics["currentBalanceCombinedResidual"] <= 1.0))
    assert bool(torch.all(diagnostics["currentBalanceMismatch"] <= 5.0e-3))
    assert bool(torch.all(diagnostics["anodeCurrent_A"] > 0.0))
    assert bool(torch.all(diagnostics["cathodeCurrent_A"] > 0.0))

    reaction = boundary["reaction"]
    assert bool(torch.all(reaction["localRobinAllFacesConverged"]))
    assert bool(torch.all(reaction["maximumLocalRobinCombinedResidual"] <= 1.0))
    # This small FP32 case intentionally exercises the representability floor.
    assert bool(torch.all(reaction["roundoffLimitedLocalRobinFaceCount"] > 0))


def test_mps_fp32_profile_one_step_on_cpu_reference(tmp_path: Path) -> None:
    """Exercise the exact MPS/FP32 equations without requiring Apple hardware."""
    nsga = yaml.safe_load(
        (ROOT / "config/nsga2_condensed_phase_no_f_m2_mps.yaml").read_text()
    )
    nsga["project"]["device"] = "cpu"
    nsga["physics"]["end_time_s"] = 1.0e-4
    nsga["condensed_ignition"]["reference_time_s"] = 1.0e-4
    evaluator_config = {
        "base_config": "config/m2_mps.yaml",
        "device": "cpu",
        "voltage_V": 260.0,
        "end_time_s": 1.0e-4,
        "grid_size": 17,
        "internal_batch_size": 1,
        "metric_evaluation_time_s": 1.0e-4,
        "physics_config": nsga,
        "base_overrides": {
            "coupled": {
                "timeStep_s": 1.0e-4,
                "endTime_s": 1.0e-4,
                "electricalUpdateInterval_s": 1.0e-4,
                "snapshotTimes_s": [],
            },
            "condensedPhaseMetrics": {"evaluationTime_s": 1.0e-4},
            "numerics": {"coupledBatchSize": 1},
        },
    }
    evaluator = DirectCondensedV772NoFEvaluator(
        ROOT, evaluator_config, tmp_path / "adapter"
    )
    anode = np.zeros((32, 32), dtype=bool)
    cathode = np.zeros((32, 32), dtype=bool)
    anode[:, 3:5] = True
    cathode[:, 27:29] = True

    result = evaluator.evaluate(
        anode,
        cathode,
        {"geometry_id": "mps-fp32-reference", "geometry_descriptors": {}},
        tmp_path / "case",
    )

    assert not bool(result.get("physicsRejected", False))
    assert result["physicsDtype"] == "float32"
    assert result["finalNonlinearRobinConverged"] is True
    assert result["finalNonlinearRobinGaugeConverged"] is True
    assert result["finalNonlinearRobinCurrentBalanceCombinedResidual"] <= 1.0
    assert result["finalAnodeCathodeCurrentMismatch"] <= 5.0e-3
    assert result["maximumLocalRobinCombinedResidual"] <= 1.0
    assert result["maximumUnresolvedLocalRobinFaceCount"] == 0.0
