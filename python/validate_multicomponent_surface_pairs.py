#!/usr/bin/env python3
"""Validate every supported (anode, cathode) component-count pair.

This is a one-step plumbing/numerical gate, not an ignition validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import yaml

from ecsp_nsga2.evaluator import create_evaluator
from ecsp_nsga2.geometry import (
    GeometryLimits,
    instantiate_variant,
    make_topology_templates,
    rasterize_and_validate,
)

REQUIRED_PAIRS = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    root = args.package_root.resolve()
    workdir = args.workdir.resolve()
    if workdir.exists():
        raise SystemExit(f"validation workdir already exists: {workdir}")
    config = yaml.safe_load(
        (root / "config" / "nsga2_condensed_phase_no_f_m2_cpp_fp64.yaml").read_text(
            encoding="utf-8"
        )
    )
    limits = GeometryLimits(**config["geometry"])
    seed = int(config["project"]["seed"])
    templates = make_topology_templates(20, seed, limits)

    items = []
    chosen_rows = []
    for na, nc in REQUIRED_PAIRS:
        template = next(
            item
            for item in templates
            if (
                len(item["anode"]["components"]),
                len(item["cathode"]["components"]),
            )
            == (na, nc)
        )
        selected = None
        for attempt in range(500):
            genome = instantiate_variant(template, attempt, seed, limits)
            raster = rasterize_and_validate(genome, limits)
            if raster.constraint_violation <= 1.0e-12:
                selected = (genome, raster, attempt)
                break
        if selected is None:
            raise RuntimeError(f"no feasible variant found for ({na},{nc})")
        genome, raster, attempt = selected
        geometry_id = f"pair_{na}A_{nc}C"
        genome["geometry_id"] = geometry_id
        metadata = {
            "geometry_id": geometry_id,
            "intended_anode_components": na,
            "intended_cathode_components": nc,
        }
        items.append(
            (
                raster.anode_mask,
                raster.cathode_mask,
                metadata,
                workdir / "physics" / geometry_id,
            )
        )
        chosen_rows.append(
            {
                "geometry_id": geometry_id,
                "variant_attempt": attempt,
                "design_grid_anode_contact_fraction": raster.descriptors[
                    "anode_area_fraction"
                ],
                "design_grid_cathode_contact_fraction": raster.descriptors[
                    "cathode_area_fraction"
                ],
            }
        )

    evaluator_config = dict(config["evaluator"])
    evaluator_config["physics_config"] = config
    evaluator_config["end_time_s"] = 0.00025
    evaluator_config["grid_size"] = 193
    evaluator_config["base_overrides"] = {
        "coupled": {
            "timeStep_s": 0.00025,
            "endTime_s": 0.00025,
            "electricalUpdateInterval_s": 0.00025,
            "snapshotTimes_s": [],
        },
        "condensedPhaseMetrics": {"evaluationTime_s": 0.00025},
    }
    evaluator = create_evaluator(
        root, evaluator_config, workdir / "adapter", allow_debug=False
    )
    results = evaluator.evaluate_batch(items)

    output_rows = []
    by_id = {row["geometry_id"]: row for row in chosen_rows}
    for result in results:
        row = dict(by_id[result["geometry_id"]])
        row.update(
            {
                "physics_rejected": bool(result.get("physicsRejected", False)),
                "electrical_converged": bool(
                    result.get("finalElectricalConverged", False)
                ),
                "propellant_domain_fraction": result.get(
                    "propellantDomainAreaFraction"
                ),
                "physics_grid_anode_contact_fraction": result.get(
                    "anodeContactAreaFraction"
                ),
                "physics_grid_cathode_contact_fraction": result.get(
                    "cathodeContactAreaFraction"
                ),
                "current_balance_mismatch": result.get(
                    "finalAnodeCathodeCurrentMismatch"
                ),
                "potential_relative_residual": result.get(
                    "finalElectricalRelativeResidual"
                ),
            }
        )
        output_rows.append(row)

    passed = all(
        not row["physics_rejected"]
        and row["electrical_converged"]
        and row["propellant_domain_fraction"] == 1.0
        for row in output_rows
    )
    summary = {
        "status": "passed" if passed else "failed",
        "required_component_pairs": [list(pair) for pair in REQUIRED_PAIRS],
        "cases": output_rows,
    }
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "multi_component_pair_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        return 2
    if not args.keep:
        # Preserve only the compact summary unless the caller requests all case files.
        summary_path = workdir / "multi_component_pair_validation.json"
        payload = summary_path.read_text(encoding="utf-8")
        for child in list(workdir.iterdir()):
            if child != summary_path:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        summary_path.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
