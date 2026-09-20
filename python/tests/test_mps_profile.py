from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from ecsp_nsga2.evaluator import DirectCondensedV772NoFEvaluator, EvaluatorError
from ecsp_v6.physics.numerics import resolve_device, validate_device_dtype


ROOT = Path(__file__).resolve().parents[2]


def _mock_mps_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        pytest.skip("This PyTorch build does not expose torch.backends.mps")
    monkeypatch.setattr(backend, "is_available", lambda: True)


def test_auto_prefers_mps_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps_available(monkeypatch)
    assert resolve_device("auto").type == "mps"


def test_mps_rejects_fp64() -> None:
    with pytest.raises(RuntimeError, match="requires numerics.physicsDtype=float32"):
        validate_device_dtype(torch.device("mps"), torch.float64)


def test_m2_profile_contains_float32_pilot_tolerances() -> None:
    cfg = yaml.safe_load((ROOT / "config/m2_mps.yaml").read_text())
    numerics = cfg["numerics"]
    solver = numerics["potentialSolver"]
    assert numerics["physicsDevice"] == "mps"
    assert numerics["physicsDtype"] == "float32"
    assert solver["relativeToleranceStatic"] == pytest.approx(1.0e-6)
    assert solver["relativeToleranceCoupled"] == pytest.approx(1.0e-6)
    assert solver["absoluteTolerance"] == pytest.approx(1.0e-8)
    assert solver["methodStatic"] == "pcg"
    assert solver["methodCoupled"] == "pcg"
    assert solver["preconditioner"] == "jacobi"
    robin = cfg["interface"]["nonlinearRobin"]
    assert robin["localRobinResidualTolerance_A_per_m2"] == pytest.approx(0.01)
    assert robin["localRobinRelativeResidualTolerance"] == pytest.approx(1.0e-4)
    assert robin["potentialUnderRelaxation"] == pytest.approx(0.35)
    assert robin["gaugeMaximumIterations"] == 24
    assert robin["gaugeMaximumStep_V"] == pytest.approx(65.0)
    assert robin["status"] == "implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5"
    assert solver["symmetricEquilibration"] is True
    assert solver["correctionForm"] is True
    assert solver["iterativeRefinementRounds"] == 3
    assert solver["fallbackMethod"] == "equilibrated_minimal_residual"
    assert numerics["physicalFloors"]["current_A"] == pytest.approx(1.0e-15)


def test_direct_evaluator_accepts_mps_fp32_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_mps_available(monkeypatch)
    nsga = yaml.safe_load(
        (ROOT / "config/nsga2_condensed_phase_no_f_m2_mps.yaml").read_text()
    )
    evaluator = DirectCondensedV772NoFEvaluator(
        ROOT,
        {
            "base_config": "config/m2_mps.yaml",
            "device": "mps",
            "physics_config": nsga,
        },
        tmp_path / "mps_adapter",
    )
    assert evaluator.device.type == "mps"
    assert evaluator.dtype == torch.float32


def test_direct_evaluator_rejects_mps_with_default_fp64(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_mps_available(monkeypatch)
    nsga = yaml.safe_load((ROOT / "config/nsga2_condensed_phase_no_f.yaml").read_text())
    with pytest.raises(EvaluatorError, match="requires numerics.physicsDtype=float32"):
        DirectCondensedV772NoFEvaluator(
            ROOT,
            {
                "base_config": "config/default_lp_pva.yaml",
                "device": "mps",
                "physics_config": nsga,
            },
            tmp_path / "bad_mps_adapter",
        )


def test_float32_physics_path_smoke_on_cpu(tmp_path: Path) -> None:
    """Exercise the same FP32 physics code path when MPS hardware is unavailable in CI."""
    import numpy as np
    from ecsp_nsga2.evaluator import canonicalise_metrics

    nsga = yaml.safe_load(
        (ROOT / "config/nsga2_condensed_phase_no_f_m2_mps.yaml").read_text()
    )
    nsga["project"]["device"] = "cpu"
    nsga["physics"]["end_time_s"] = 1.0e-4
    nsga["condensed_ignition"]["reference_time_s"] = 1.0e-4
    evaluator = DirectCondensedV772NoFEvaluator(
        ROOT,
        {
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
        },
        tmp_path / "fp32_adapter",
    )
    anode = np.zeros((32, 32), dtype=bool)
    cathode = np.zeros((32, 32), dtype=bool)
    anode[:, 3:5] = True
    cathode[:, 27:29] = True
    raw = evaluator.evaluate(
        anode,
        cathode,
        {"geometry_id": "fp32_smoke", "geometry_descriptors": {}},
        tmp_path / "fp32_case",
    )
    # Legacy direct/MPS path does not implement the v7.9.4 inner voltage search.
    # Supply a finite audit placeholder only to exercise generic canonical mapping.
    raw["minimumIgnitionVoltageObjective_V"] = 260.0
    raw["minimumIgnitionVoltageSearchValid"] = False
    canonical = canonicalise_metrics(raw, end_time_s=1.0e-4, no_ignition_penalty_s=2.0)
    assert raw["physicsDtype"] == "float32"
    assert raw["physicsDevice"] == "cpu"
    assert raw["finalNonlinearRobinGaugeConverged"] is True
    assert raw["finalAnodeCathodeCurrentMismatch"] <= 5.0e-3
    assert np.all(np.isfinite(canonical["objective_vector"]))
