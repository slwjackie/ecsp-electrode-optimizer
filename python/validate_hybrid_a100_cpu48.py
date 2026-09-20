#!/usr/bin/env python3
from __future__ import annotations

"""A100 + CPU hybrid preflight.

This gate intentionally executes the same one-step physics case on both
backends and also submits enough cases through the work-stealing evaluator to
prove that the A100 and C++ CPU pool are both used in one stage.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml


def _make_mask(
    n: int, target_area: float, variant: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    # Single-component rectangles sized to the active per-polarity area target.
    anode = np.zeros((n, n), dtype=bool)
    cathode = np.zeros((n, n), dtype=bool)
    r0 = 5 + (variant % 2)
    r1 = n - 5 + (variant % 2)
    r1 = min(r1, n)
    rows = r1 - r0
    target_cells = max(2 * rows, int(round(float(target_area) * n * n)))
    width = max(2, target_cells // rows)
    extra = max(0, min(rows, target_cells - width * rows))
    anode[r0:r1, 5 : 5 + width] = True
    cathode[r0:r1, n - 5 - width : n - 5] = True
    if extra:
        extra_start = r0 + (rows - extra) // 2
        anode[extra_start : extra_start + extra, 5 + width] = True
        cathode[extra_start : extra_start + extra, n - 6 - width] = True
    return anode, cathode


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=80)
    parser.add_argument("--grid-size", type=int, default=193)
    parser.add_argument("--cpu-workers", type=int, default=16)
    args = parser.parse_args()

    root = args.package_root.resolve()
    workdir = args.workdir.resolve()
    if workdir.exists():
        raise SystemExit(f"Preflight workdir already exists: {workdir}")
    workdir.mkdir(parents=True)
    if str(root / "python") not in sys.path:
        sys.path.insert(0, str(root / "python"))

    from ecsp_nsga2.hybrid import HybridCudaCpuFp64Evaluator

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    dt = 2.5e-4
    cfg.setdefault("physics", {})["end_time_s"] = dt
    cfg.setdefault("condensed_ignition", {})["reference_time_s"] = dt
    cfg.setdefault("minimum_ignition_voltage_search", {})["upper_bound_V"] = float(
        cfg.get("physics", {}).get("voltage_V", 260.0)
    )
    evaluator_cfg = copy.deepcopy(cfg.setdefault("evaluator", {}))
    evaluator_cfg.update(
        {
            "backend": "hybrid_cuda_cpu",
            "device": "cuda",
            "end_time_s": dt,
            "grid_size": int(args.grid_size),
            "hybrid_cpu_worker_cases": int(args.cpu_workers),
            "hybrid_cuda_batch_size": 64,
            "hybrid_cuda_min_batch_size": 8,
            "hybrid_cpu_reserve_per_wave": 16,
            "hybrid_require_cuda": True,
            "hybrid_cuda_enabled": True,
            "physics_config": cfg,
        }
    )

    evaluator = HybridCudaCpuFp64Evaluator(root, evaluator_cfg, workdir / "adapter")
    design_n = int(cfg.get("geometry", {}).get("grid_size", 96))
    items = []
    for i in range(int(args.task_count)):
        a, c = _make_mask(
            design_n,
            float(cfg.get("geometry", {}).get("target_area_fraction_per_polarity", 0.20)),
            i,
        )
        out = workdir / "cases" / f"G{i:04d}"
        items.append(
            (
                a,
                c,
                {
                    "geometry_id": f"HYBRID_PREFLIGHT_{i:04d}",
                    "topology_id": "HYBRID_PREFLIGHT_1A1C",
                    "source_role": "hybrid_preflight",
                    "intended_anode_components": 1,
                    "intended_cathode_components": 1,
                },
                out,
            )
        )

    rows = evaluator.evaluate_batch(items)
    valid = [r for r in rows if not bool(r.get("physicsRejected", False))]
    cpu = [r for r in valid if str(r.get("backend", "")).startswith("cpp_fp64_cpu")]
    gpu = [r for r in valid if str(r.get("backend", "")).startswith("torch_cuda")]

    metrics = (
        "peakCurrent_A",
        "inputElectricalEnergyAt2s_J",
        "peakCurrentCongestion",
        "peakMaximumTemperature_K",
    )
    comparisons = {}
    for key in metrics:
        cpu_med = float(np.median([float(r[key]) for r in cpu])) if cpu else None
        gpu_med = float(np.median([float(r[key]) for r in gpu])) if gpu else None
        comparisons[key] = {
            "cpu_median": cpu_med,
            "gpu_median": gpu_med,
            "relative_difference": None if cpu_med is None or gpu_med is None else _rel(cpu_med, gpu_med),
        }

    stats_path = workdir / "adapter" / "hybrid_runtime_stats.json"
    runtime_stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    gates = {
        "all_rows_returned": len(rows) == len(items),
        "all_physics_successful": len(valid) == len(items),
        "gpu_used": len(gpu) > 0 and int(runtime_stats.get("cuda_completed", 0)) > 0,
        "cpu_used": len(cpu) > 0 and int(runtime_stats.get("cpu_completed", 0)) > 0,
        "fp64_on_gpu": all(str(r.get("physicsDtype")) == "float64" for r in gpu),
        "fp64_on_cpu": all(str(r.get("physicsDtype")) == "float64" for r in cpu),
        "cpu_gpu_current_agreement": (
            comparisons["peakCurrent_A"]["relative_difference"] is not None
            and comparisons["peakCurrent_A"]["relative_difference"] <= 5.0e-4
        ),
        "cpu_gpu_energy_agreement": (
            comparisons["inputElectricalEnergyAt2s_J"]["relative_difference"] is not None
            and comparisons["inputElectricalEnergyAt2s_J"]["relative_difference"] <= 5.0e-4
        ),
    }
    payload = {
        "status": "passed" if all(gates.values()) else "failed",
        "task_count": len(items),
        "successful": len(valid),
        "rejected": len(items) - len(valid),
        "cpu_rows": len(cpu),
        "gpu_rows": len(gpu),
        "runtime_stats": runtime_stats,
        "comparisons": comparisons,
        "gates": gates,
    }
    summary = workdir / "hybrid_preflight_summary.json"
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
