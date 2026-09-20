from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ecsp_nsga2.evaluator import DirectCondensedV772NoFEvaluator, canonicalise_metrics


ROOT = Path(__file__).resolve().parents[2]


def test_direct_no_f_physics_one_step_cpu(tmp_path: Path) -> None:
    nsga = yaml.safe_load((ROOT / "config/nsga2_condensed_phase_no_f.yaml").read_text())
    nsga["project"]["device"] = "cpu"
    nsga["physics"]["end_time_s"] = 1.0e-4
    nsga["condensed_ignition"]["reference_time_s"] = 1.0e-4
    evaluator_cfg = {
        "base_config": "config/default_lp_pva.yaml",
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
    evaluator = DirectCondensedV772NoFEvaluator(ROOT, evaluator_cfg, tmp_path / "adapter")
    anode = np.zeros((32, 32), dtype=bool)
    cathode = np.zeros((32, 32), dtype=bool)
    anode[:, 3:5] = True
    cathode[:, 27:29] = True
    raw = evaluator.evaluate(
        anode,
        cathode,
        {"geometry_id": "smoke", "geometry_descriptors": {}},
        tmp_path / "case",
    )
    # Legacy direct/MPS path does not implement the v7.9.4 inner voltage search.
    # Supply a finite audit placeholder only to exercise generic canonical mapping.
    raw["minimumIgnitionVoltageObjective_V"] = 260.0
    raw["minimumIgnitionVoltageSearchValid"] = False
    canonical = canonicalise_metrics(raw, end_time_s=1.0e-4, no_ignition_penalty_s=2.0)
    assert raw["empiricalSurfaceReactionProgressUsed"] is False
    assert raw["backend"] == "direct_condensed_v772_no_f"
    assert raw["solverRevision"] == "v7_7_5_mps_robust_spd_pcg_gap_bv"
    assert raw["finalNonlinearRobinGaugeConverged"] is True
    assert raw["finalAnodeCathodeCurrentMismatch"] <= 5.0e-3
    assert 0.0 <= canonical["area_undecomposed_fraction_at_2s"] <= 1.0
    assert np.all(np.isfinite(canonical["objective_vector"]))
