#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OBJECTIVES = [
    "ignition_delay_s",
    "area_undecomposed_fraction_at_2s",
    "minimum_ignition_voltage_V",
    "current_congestion",
]


def _pareto_ids(frame: pd.DataFrame, suffix: str) -> set[str]:
    cols = [f"{name}_{suffix}" for name in OBJECTIVES]
    cv = f"constraint_violation_{suffix}"
    valid = frame.copy()
    if cv in valid.columns:
        valid = valid[pd.to_numeric(valid[cv], errors="coerce").fillna(np.inf) <= 1e-12]
    values = valid[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.all(np.isfinite(values), axis=1)
    valid = valid.loc[finite].copy()
    values = values[finite]
    if len(valid) == 0:
        return set()

    dominated = np.zeros(len(valid), dtype=bool)
    for i in range(len(valid)):
        if dominated[i]:
            continue
        # All objectives are minimised. j dominates i if j <= i componentwise
        # and is strictly better in at least one objective.
        le = np.all(values <= values[i], axis=1)
        lt = np.any(values < values[i], axis=1)
        if np.any(le & lt):
            dominated[i] = True
    return set(valid.loc[~dominated, "geometry_id"].astype(str))


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 3:
        return None
    if np.all(a[finite] == a[finite][0]) or np.all(b[finite] == b[finite][0]):
        return None
    value = spearmanr(a[finite], b[finite]).statistic
    return None if not np.isfinite(value) else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare identical ECSP geometries evaluated with CPU FP64 and MPS FP32."
    )
    parser.add_argument("--cpu", type=Path, required=True, help="CPU population_evaluated.csv")
    parser.add_argument("--mps", type=Path, required=True, help="MPS population_evaluated.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cpu = pd.read_csv(args.cpu)
    mps = pd.read_csv(args.mps)
    if cpu["geometry_id"].duplicated().any() or mps["geometry_id"].duplicated().any():
        raise SystemExit("geometry_id must be unique in both validation CSV files")

    merged = cpu.merge(mps, on="geometry_id", suffixes=("_cpu", "_mps"), how="inner")
    if len(merged) == 0:
        raise SystemExit("No common geometry_id values were found")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame({"geometry_id": merged["geometry_id"].astype(str)})
    objective_summary: dict[str, dict[str, float | int | None]] = {}

    for name in OBJECTIVES:
        cpu_values = pd.to_numeric(merged[f"{name}_cpu"], errors="coerce").to_numpy(float)
        mps_values = pd.to_numeric(merged[f"{name}_mps"], errors="coerce").to_numpy(float)
        finite = np.isfinite(cpu_values) & np.isfinite(mps_values)
        scale = np.maximum(np.abs(cpu_values), 1e-12)
        rel = np.full_like(cpu_values, np.nan, dtype=float)
        rel[finite] = np.abs(mps_values[finite] - cpu_values[finite]) / scale[finite]

        detail[f"{name}_cpu"] = cpu_values
        detail[f"{name}_mps"] = mps_values
        detail[f"{name}_relative_error"] = rel

        valid_rel = rel[np.isfinite(rel)]
        objective_summary[name] = {
            "n_compared": int(valid_rel.size),
            "median_relative_error": float(np.median(valid_rel)) if valid_rel.size else None,
            "mean_relative_error": float(np.mean(valid_rel)) if valid_rel.size else None,
            "max_relative_error": float(np.max(valid_rel)) if valid_rel.size else None,
            "fraction_within_1pct": float(np.mean(valid_rel <= 0.01)) if valid_rel.size else None,
            "fraction_within_3pct": float(np.mean(valid_rel <= 0.03)) if valid_rel.size else None,
            "spearman": _safe_spearman(cpu_values, mps_values),
        }

    # Use CPU ranges as a fixed scale so both devices receive the same aggregate score.
    cpu_matrix = np.column_stack(
        [pd.to_numeric(merged[f"{name}_cpu"], errors="coerce").to_numpy(float) for name in OBJECTIVES]
    )
    mps_matrix = np.column_stack(
        [pd.to_numeric(merged[f"{name}_mps"], errors="coerce").to_numpy(float) for name in OBJECTIVES]
    )
    finite_rows = np.all(np.isfinite(cpu_matrix), axis=1) & np.all(np.isfinite(mps_matrix), axis=1)
    aggregate_spearman: float | None = None
    if np.count_nonzero(finite_rows) >= 3:
        ref = cpu_matrix[finite_rows]
        lo = np.min(ref, axis=0)
        hi = np.max(ref, axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        cpu_score = np.linalg.norm((cpu_matrix[finite_rows] - lo) / span, axis=1)
        mps_score = np.linalg.norm((mps_matrix[finite_rows] - lo) / span, axis=1)
        aggregate_spearman = _safe_spearman(cpu_score, mps_score)

    cpu_pareto = _pareto_ids(merged, "cpu")
    mps_pareto = _pareto_ids(merged, "mps")
    union = cpu_pareto | mps_pareto
    intersection = cpu_pareto & mps_pareto
    pareto_jaccard = 1.0 if not union else len(intersection) / len(union)
    pareto_recall = 1.0 if not cpu_pareto else len(intersection) / len(cpu_pareto)

    median_errors = [
        item["median_relative_error"]
        for item in objective_summary.values()
        if item["median_relative_error"] is not None
    ]
    screening_ready = bool(
        median_errors
        and max(median_errors) <= 0.03
        and aggregate_spearman is not None
        and aggregate_spearman >= 0.95
        and pareto_recall >= 0.90
    )

    report = {
        "comparison": "CPU_FP64_vs_MPS_FP32",
        "common_geometry_count": int(len(merged)),
        "objective_summary": objective_summary,
        "aggregate_utopia_score_spearman": aggregate_spearman,
        "cpu_pareto_count": len(cpu_pareto),
        "mps_pareto_count": len(mps_pareto),
        "pareto_intersection_count": len(intersection),
        "pareto_jaccard": float(pareto_jaccard),
        "cpu_pareto_recall_by_mps": float(pareto_recall),
        "pilot_screening_criteria": {
            "maximum_objective_median_relative_error": 0.03,
            "minimum_aggregate_spearman": 0.95,
            "minimum_cpu_pareto_recall": 0.90,
            "note": (
                "These are pilot screening criteria, not universal publication standards. "
                "Final reported designs should still be recomputed with CPU/CUDA FP64."
            ),
        },
        "pilot_screening_ready": screening_ready,
        "cpu_pareto_geometry_ids": sorted(cpu_pareto),
        "mps_pareto_geometry_ids": sorted(mps_pareto),
    }

    detail.to_csv(output_dir / "cpu_fp64_vs_mps_fp32_objective_errors.csv", index=False)
    (output_dir / "cpu_fp64_vs_mps_fp32_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
