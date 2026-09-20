#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def _load_config(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    # effective_config.json contains audit-only fields that the workflow ignores,
    # but remove them to keep the reconstructed configuration clean.
    data.pop("resolved_geometry_limits", None)
    data.pop("workflow_invariants", None)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the post-optimization area-matched staggered reference for an "
            "existing v7.9.x run without rerunning NSGA-II."
        )
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config. Default: <workdir>/effective_config.json",
    )
    parser.add_argument(
        "--allow-debug-physics",
        action="store_true",
        help="Permit analytic debug physics; production comparisons should omit this flag.",
    )
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    workdir = args.workdir.resolve()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else workdir / "effective_config.json"
    )
    if not config_path.is_file():
        parser.error(f"configuration not found: {config_path}")

    python_root = package_root / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

    from ecsp_nsga2.nsga2 import Individual
    from ecsp_nsga2.geometry import RasterizedGeometry
    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow, OBJECTIVE_NAMES

    config = _load_config(config_path)
    runtime_dir = workdir / ".area_matched_staggered_runtime"
    workflow = NSGA2ElectricalSolidWorkflow(
        package_root=package_root,
        config=config,
        workdir=runtime_dir,
        allow_debug_physics=args.allow_debug_physics,
    )
    final_dir = workdir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    recommendation_path = final_dir / "recommended_design.json"
    recommendation = None
    recommendation_recomputed = False
    if recommendation_path.is_file():
        data = json.loads(recommendation_path.read_text(encoding="utf-8"))
        objective_map = data.get("objectives", {})
        if all(name in objective_map for name in OBJECTIVE_NAMES):
            recommendation = Individual(
                geometry_id=str(data.get("geometry_id", "RECOMMENDED_AI")),
                topology_id=str(data.get("topology_id", "UNKNOWN")),
                genome=dict(data.get("genome", {})),
                objectives=np.asarray(
                    [float(objective_map[name]) for name in OBJECTIVE_NAMES],
                    dtype=float,
                ),
                constraint_violation=float(data.get("constraint_violation", 0.0)),
                metrics=dict(data.get("metrics", {})),
                source_role="existing_recommended_design",
            )
        else:
            # Any pre-v7.9.4 recommendation lacks the true bracketed V_min objective.
            # Never reinterpret an energy metric as voltage.  If saved contact masks
            # are available, re-evaluate only this already-selected geometry with the
            # v7.9.4 reference-voltage + V_min search layer.
            npz_path = final_dir / "recommended_design.npz"
            genome = dict(data.get("genome", {}))
            if npz_path.is_file() and genome:
                with np.load(npz_path) as npz:
                    anode_mask = np.asarray(
                        npz[
                            "anode_contact_mask"
                            if "anode_contact_mask" in npz
                            else "anode_mask"
                        ],
                        dtype=bool,
                    )
                    cathode_mask = np.asarray(
                        npz[
                            "cathode_contact_mask"
                            if "cathode_contact_mask" in npz
                            else "cathode_mask"
                        ],
                        dtype=bool,
                    )
                old_metrics = dict(data.get("metrics", {}))
                descriptors = dict(old_metrics.get("geometry_descriptors", {}))
                raster = RasterizedGeometry(
                    anode_mask=anode_mask,
                    cathode_mask=cathode_mask,
                    descriptors=descriptors,
                    constraint_violation=0.0,
                    violation_details={},
                )
                recommendation = Individual(
                    geometry_id=str(data.get("geometry_id", "RECOMMENDED_AI")),
                    topology_id=str(data.get("topology_id", "UNKNOWN")),
                    genome=genome,
                    constraint_violation=0.0,
                    metrics=old_metrics,
                    source_role="existing_recommended_design_recomputed_v794",
                )
                metadata = workflow._metadata(recommendation, raster, generation=-2)
                physics_dir = final_dir / "recommended_design_v794_recomputed" / "physics"
                physics_dir.mkdir(parents=True, exist_ok=True)
                raw = workflow.evaluator.evaluate(
                    anode_mask, cathode_mask, metadata, physics_dir
                )
                workflow._apply_raw_metrics(recommendation, raw)
                recommendation_recomputed = True
                (final_dir / "recommended_design_v794_recomputed.json").write_text(
                    json.dumps(
                        {
                            "geometry_id": recommendation.geometry_id,
                            "topology_id": recommendation.topology_id,
                            "source": "legacy_recommended_design_re_evaluated_for_minimum_ignition_voltage",
                            "objectives": dict(
                                zip(
                                    OBJECTIVE_NAMES,
                                    [float(x) for x in recommendation.metrics["objective_vector"]],
                                )
                            ),
                            "constraint_violation": float(recommendation.constraint_violation),
                            "metrics": recommendation.metrics,
                        },
                        indent=2,
                        default=lambda x: (
                            x.tolist() if isinstance(x, np.ndarray)
                            else x.item() if isinstance(x, np.generic)
                            else str(x)
                        ),
                    ),
                    encoding="utf-8",
                )


    baseline = workflow._evaluate_area_matched_staggered(
        final_dir, recommendation=recommendation
    )
    if baseline is None:
        print(json.dumps({"status": "disabled"}, indent=2))
        return 0

    if recommendation is not None:
        workflow._write_area_matched_staggered_comparison(
            recommendation, baseline, final_dir
        )

    raw_baseline = baseline.metrics.get("objective_vector", baseline.objectives)
    result = {
        "status": "passed",
        "workdir": str(workdir),
        "baseline_geometry_id": baseline.geometry_id,
        "baseline_objectives_raw_unpenalized": dict(
            zip(OBJECTIVE_NAMES, [float(x) for x in raw_baseline])
        ),
        "baseline_constraint_violation": float(baseline.constraint_violation),
        "baseline_numerical_constraint_violation": float(
            baseline.metrics.get("numerical_constraint_violation", 0.0)
        ),
        "recommended_design_found": recommendation is not None,
        "recommended_design_recomputed_for_minimum_ignition_voltage": recommendation_recomputed,
        "comparison_written": bool(
            (final_dir / "recommended_vs_area_matched_staggered.csv").is_file()
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
