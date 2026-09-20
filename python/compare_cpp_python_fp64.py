#!/usr/bin/env python3
"""Short-horizon numerical parity check: Python FP64 vs C++ FP64.

The same deterministic NSGA-II population is evaluated by both backends on a
small grid and short evolving-state horizon.  This checks model-port parity,
not experimental calibration and not production 2-s runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

ABSOLUTE_TOLERANCES = {
    # Both implementations satisfy the 0.5% physical gate.  Near zero, a
    # relative error is ill-conditioned, so compare this diagnostic absolutely.
    "finalAnodeCathodeCurrentMismatch": 5.0e-6,
}

METRICS = [
    "peakMaximumTemperature_K",
    "peakCurrent_A",
    "peakCurrentCongestion",
    "inputElectricalEnergyAt2s_J",
    "areaAveragedUndecomposedFractionAt2s",
    "finalEffectiveResistance_ohm",
    "finalNonlinearRobinGaugeOffset_V",
    "finalAnodeCathodeCurrentMismatch",
]


def _run(
    root: Path,
    config: Path,
    workdir: Path,
    grid: int,
    end_time: float,
    population_size: int,
) -> None:
    cmd = [
        sys.executable,
        str(root / "python" / "run_nsga2_electrical_solid_loop.py"),
        "--package-root", str(root),
        "--config", str(config),
        "--device", "cpu",
        "--population-size", str(population_size),
        "--generations", "1",
        "--grid-size", str(grid),
        "--end-time-s", str(end_time),
        "--allow-no-feasible",
        "--workdir", str(workdir),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "python") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(cmd, cwd=root, env=env, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"backend command failed ({completed.returncode}): {' '.join(cmd)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--grid-size", type=int, default=33)
    parser.add_argument("--end-time-s", type=float, default=0.003)
    parser.add_argument("--population-size", type=int, default=1)
    parser.add_argument("--max-relative-error", type=float, default=5e-4)
    args = parser.parse_args()
    root = args.package_root.resolve()
    workdir = (
        args.workdir or (root / "runs" / "cpp_python_fp64_parity")
    ).resolve()
    if workdir.exists():
        raise SystemExit(f"workdir already exists: {workdir}")
    py_dir = workdir / "python_fp64"
    cpp_dir = workdir / "cpp_fp64"
    workdir.mkdir(parents=True)

    build = subprocess.run(
        ["bash", str(root / "tools" / "build_cpp_cpu.sh")], cwd=root
    )
    if build.returncode != 0:
        return build.returncode

    _run(
        root,
        root / "config" / "nsga2_condensed_phase_no_f.yaml",
        py_dir,
        args.grid_size,
        args.end_time_s,
        args.population_size,
    )
    _run(
        root,
        root / "config" / "nsga2_condensed_phase_no_f_m2_cpp_fp64.yaml",
        cpp_dir,
        args.grid_size,
        args.end_time_s,
        args.population_size,
    )

    py_frame = pd.read_csv(
        py_dir / "generation_000" / "population_evaluated.csv"
    ).set_index("geometry_id", drop=False)
    cpp_frame = pd.read_csv(
        cpp_dir / "generation_000" / "population_evaluated.csv"
    ).set_index("geometry_id", drop=False)
    py_ids = set(map(str, py_frame.index))
    cpp_ids = set(map(str, cpp_frame.index))
    if py_ids != cpp_ids:
        raise RuntimeError(
            "deterministic geometry IDs differ between backends: "
            f"python_only={sorted(py_ids-cpp_ids)}, cpp_only={sorted(cpp_ids-py_ids)}"
        )

    rows: list[dict[str, float | str]] = []
    for geometry_id in sorted(py_ids):
        py_row = py_frame.loc[geometry_id]
        cpp_row = cpp_frame.loc[geometry_id]
        for metric in METRICS:
            if metric not in py_row or metric not in cpp_row:
                continue
            a = float(py_row[metric])
            b = float(cpp_row[metric])
            denom = max(abs(a), abs(b), 1e-14)
            rel = abs(a - b) / denom
            absolute_error = abs(a - b)
            absolute_tolerance = ABSOLUTE_TOLERANCES.get(metric)
            passed = rel <= args.max_relative_error or (
                absolute_tolerance is not None
                and absolute_error <= absolute_tolerance
            )
            rows.append(
                {
                    "geometry_id": geometry_id,
                    "metric": metric,
                    "python_fp64": a,
                    "cpp_fp64": b,
                    "absolute_error": absolute_error,
                    "absolute_tolerance": absolute_tolerance,
                    "relative_error": rel,
                    "passed": passed,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(workdir / "cpp_vs_python_fp64_metrics.csv", index=False)
    if frame.empty:
        max_rel = float("inf")
        worst_geometry = None
        worst_metric = None
        all_passed = False
    else:
        relative_only = frame[~frame["metric"].isin(ABSOLUTE_TOLERANCES)]
        if relative_only.empty:
            max_rel = 0.0
            worst_geometry = None
            worst_metric = None
        else:
            worst_index = relative_only["relative_error"].astype(float).idxmax()
            worst = relative_only.loc[worst_index]
            max_rel = float(worst["relative_error"])
            worst_geometry = str(worst["geometry_id"])
            worst_metric = str(worst["metric"])
        all_passed = bool(frame["passed"].all())
    report = {
        "status": "passed" if all_passed else "failed",
        "case_count": len(py_ids),
        "geometry_ids": sorted(py_ids),
        "grid_size": args.grid_size,
        "end_time_s": args.end_time_s,
        "maximum_relative_error": max_rel,
        "worst_geometry_id": worst_geometry,
        "worst_metric": worst_metric,
        "relative_error_threshold": args.max_relative_error,
        "metric_specific_absolute_tolerances": ABSOLUTE_TOLERANCES,
        "note": (
            "Short-horizon evolving-state port-parity gate; it is not a full "
            "2-s validation or experimental calibration."
        ),
    }
    (workdir / "cpp_vs_python_fp64_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = (
        frame.groupby("metric", as_index=False)
        .agg(
            maximum_relative_error=("relative_error", "max"),
            maximum_absolute_error=("absolute_error", "max"),
            passed=("passed", "all"),
        )
        .sort_values("maximum_relative_error", ascending=False)
    )
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
