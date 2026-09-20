from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import subprocess

import numpy as np

from ecsp_cpp.backend import CppPhysicsRunner, serialise_cpp_config
from ecsp_cuda import CudaBatchCase, TorchCudaSurfaceContactBatchRunner
from ecsp_nsga2.hybrid import HybridCudaCpuFp64Evaluator, _HybridTask
from ecsp_v6.config import load_config
from ecsp_v6.physics.composition_model import build_composition


def _mask(n: int) -> tuple[np.ndarray, np.ndarray]:
    anode = np.zeros((n, n), dtype=bool)
    cathode = np.zeros((n, n), dtype=bool)
    anode[3:-3, 3:6] = True
    cathode[3:-3, -6:-3] = True
    return anode, cathode


def test_legacy_cpu_solver_source_is_preserved_byte_for_byte():
    root = Path(__file__).resolve().parents[2]
    assert (root / "cpp/ecsp_cpp_solver.cpp").read_bytes() == (
        root / "cpp/reference/ecsp_cpp_solver_v7_9_4_reference.cpp"
    ).read_bytes()


def test_hybrid_scheduler_reserves_work_for_both_devices(tmp_path: Path):
    tasks = deque(
        _HybridTask(
            task_id=i,
            anode=np.zeros((3, 3), dtype=bool),
            cathode=np.zeros((3, 3), dtype=bool),
            metadata={"geometry_id": f"G{i}"},
            output_dir=tmp_path / str(i),
            config_values={},
        )
        for i in range(40)
    )
    selected = HybridCudaCpuFp64Evaluator._pop_gpu_batch(
        tasks,
        batch_size=64,
        minimum_size=8,
        allow_partial=True,
        cpu_running_count=0,
        cpu_worker_capacity=40,
        cpu_reserve_per_wave=16,
    )
    assert len(selected) == 24
    assert len(tasks) == 16

    large = deque(
        _HybridTask(
            task_id=i,
            anode=np.zeros((3, 3), dtype=bool),
            cathode=np.zeros((3, 3), dtype=bool),
            metadata={"geometry_id": f"L{i}"},
            output_dir=tmp_path / f"L{i}",
            config_values={},
        )
        for i in range(192)
    )
    selected_large = HybridCudaCpuFp64Evaluator._pop_gpu_batch(
        large,
        batch_size=64,
        minimum_size=8,
        allow_partial=True,
        cpu_running_count=0,
        cpu_worker_capacity=40,
        cpu_reserve_per_wave=16,
    )
    assert len(selected_large) == 64
    assert len(large) == 128


def test_torch_batched_equations_match_buffered_cpp_short_horizon(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "ecsp_cpp_solver_hybrid"
    subprocess.run(
        [
            "c++", "-std=c++17", "-O2", "-DNDEBUG",
            str(root / "cpp/ecsp_cpp_solver_hybrid.cpp"),
            "-o", str(executable),
        ],
        check=True,
    )
    cfg = load_config(root / "config/default_lp_pva.yaml")
    composition = build_composition(cfg)
    values = serialise_cpp_config(cfg, composition, grid_size=17, voltage=260.0)
    values["end_time"] = 10.0 * float(values["dt"])
    values["eval_time"] = values["end_time"]
    values["stop_on_ignition"] = False
    anode, cathode = _mask(17)

    cpp = CppPhysicsRunner(root, executable).run_case(
        anode, cathode, values, tmp_path / "cpp"
    )
    torch_runner = TorchCudaSurfaceContactBatchRunner(
        "cpu", allow_cpu_for_test=True, dtype_name="float64"
    )
    torch_row = torch_runner.run_batch(
        [
            CudaBatchCase(
                anode=anode,
                cathode=cathode,
                config=values,
                output_dir=tmp_path / "torch",
                geometry_id="G0",
            )
        ]
    )[0]

    assert not bool(torch_row.get("physicsRejected", False))
    for key, rtol, atol in (
        ("peakCurrent_A", 3.0e-5, 1.0e-12),
        ("inputElectricalEnergyAt2s_J", 3.0e-5, 1.0e-12),
        ("peakCurrentCongestion", 3.0e-5, 1.0e-10),
        ("peakMaximumTemperature_K", 1.0e-10, 1.0e-8),
        ("areaAveragedUndecomposedFractionAt2s", 1.0e-10, 1.0e-12),
    ):
        assert np.isclose(float(torch_row[key]), float(cpp[key]), rtol=rtol, atol=atol), (
            key, torch_row[key], cpp[key]
        )
    assert float(torch_row["finalElectricalRelativeResidual"]) < 2.0e-9
    assert float(cpp["finalElectricalRelativeResidual"]) < 2.0e-9


def test_create_evaluator_routes_hybrid_backend(monkeypatch, tmp_path: Path):
    import ecsp_nsga2.hybrid as hybrid_module
    from ecsp_nsga2.evaluator import create_evaluator

    sentinel = object()
    monkeypatch.setattr(
        hybrid_module,
        "HybridCudaCpuFp64Evaluator",
        lambda package_root, config, workdir: sentinel,
    )
    result = create_evaluator(
        Path("."),
        {"backend": "hybrid_cuda_cpu"},
        tmp_path,
        allow_debug=False,
    )
    assert result is sentinel


def _valid_trial_row(voltage: float, threshold: float) -> dict:
    ignited = voltage >= threshold
    return {
        "appliedVoltage_V": voltage,
        "ignitionSucceeded": ignited,
        "ignitionDelay_s": 1.0 if ignited else None,
        "converged": True,
        "finalElectricalConverged": True,
        "maximumTemperatureCapFraction": 0.0,
        "maximumSpeciesLimiterFraction": 0.0,
        "maximumGasCapFraction": 0.0,
        "maximumChemicalRateCapFraction": 0.0,
        "simulatedTime_s": 1.0 if ignited else 2.0,
        "terminatedAtIgnition": ignited,
    }


def test_hybrid_vmin_search_uses_wave_scheduler(tmp_path: Path):
    evaluator = HybridCudaCpuFp64Evaluator.__new__(HybridCudaCpuFp64Evaluator)
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
    evaluator.cpp_config_values = {"voltage": 260.0}
    threshold = 123.0
    stages: list[str] = []

    def fake_execute(tasks, *, stage):
        stages.append(stage)
        return {
            t.task_id: _valid_trial_row(float(t.config_values["voltage"]), threshold)
            for t in tasks
        }

    evaluator._execute_hybrid_tasks = fake_execute
    mask = np.zeros((8, 8), dtype=bool)
    out = tmp_path / "G0"
    prepared = {0: (mask, mask, {"geometry_id": "G0"}, out)}
    results = [_valid_trial_row(260.0, threshold)]
    evaluator._attach_minimum_ignition_voltage_search_hybrid(prepared, results)
    row = results[0]
    assert row["minimumIgnitionVoltageSearchValid"] is True
    assert row["minimumIgnitionVoltageLowerNonIgnitingBound_V"] < threshold
    assert row["minimumIgnitionVoltageUpperIgnitingBound_V"] >= threshold
    assert row["minimumIgnitionVoltageBracketWidth_V"] <= 5.0 + 1e-12
    assert stages and stages[0] == "vmin-wave-0"
    assert (out / "condensed_metrics.json").is_file()


def test_hybrid_diagnostics_serialises_effective_partition(tmp_path: Path):
    evaluator = HybridCudaCpuFp64Evaluator.__new__(HybridCudaCpuFp64Evaluator)
    evaluator.workdir = tmp_path
    evaluator.cpp_version = "ecsp_cpp_solver_hybrid 7.9.5"
    evaluator.cpp_runner = type("Runner", (), {"executable": tmp_path / "solver"})()
    evaluator.cpu_worker_cases = 40
    evaluator.requested_cpu_workers = 40
    evaluator.visible_cpu_count = 48
    evaluator.reserved_host_vcpus = 8
    evaluator.cuda_runner = None
    evaluator.cuda_device = "cuda:0"
    evaluator.cuda_batch_size = 64
    evaluator.cuda_min_batch_size = 8
    evaluator.cuda_fallback_to_cpu = True
    evaluator.cpu_reserve_per_wave = 16
    evaluator._cuda_runtime_disabled_reason = "test-no-cuda"
    evaluator.cuda_info = {}
    evaluator._write_hybrid_diagnostics({"hybrid_cpu_worker_cases": 40})
    payload = json.loads((tmp_path / "hybrid_evaluator_diagnostics.json").read_text())
    assert payload["requested_cpu_worker_cases"] == 40
    assert payload["configured_48vcpu_target_partition"] == {
        "cpu_solver_processes": 40,
        "reserved_for_python_cuda_host_io_os": 8,
    }
    assert payload["effective_runtime_partition"]["remaining_visible_cpus"] == 8
