#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except Exception as exc:
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Physics code may require algorithms without deterministic CUDA kernels;
        # warn_only preserves the run while recording the deterministic intent.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "v8.0 surface-contact multi-component NSGA-II closed loop. Legacy v7.9.5 backends remain available; the opt-in B/C backend adds paper-based pre-flame species/electrochemical/thermal physics and post-onset condensed reaction-progress/level-set refinement. No OpenFOAM or gas-phase CFD is called."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "nsga2_condensed_phase_no_f.yaml",
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--bootstrap-cache", type=Path, default=None,
                        help="Reuse the geometry-only preflight; code/seed/grids and every mask are revalidated")
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of the v8 package (including all preserved v7.9.5 functionality)",
    )
    parser.add_argument(
        "--allow-debug-physics",
        action="store_true",
        help="Allow the non-physical analytic backend for smoke testing only",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Override project/evaluator device. Use cpu for the preserved C++ backend and "
            "cuda for the A100/CPU hybrid backend. Legacy direct MPS physics is not a production path."
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override generation count",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Override population size; non-1000 pilot values automatically rebalance topology/variant counts",
    )
    parser.add_argument(
        "--end-time-s",
        type=float,
        default=None,
        help="Override the condensed-phase horizon (useful for a one-step smoke test)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Override the physics grid size without changing the design-mask grid",
    )
    parser.add_argument(
        "--allow-no-feasible",
        action="store_true",
        help=(
            "Return exit status 0 after a numerically successful smoke/benchmark run "
            "even when no tested candidate ignites; production recommendations remain strict."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.bootstrap_cache is not None:
        config["_bootstrap_cache_directory"] = str(args.bootstrap_cache.resolve())
    if args.device is not None:
        requested_device = str(args.device).lower()
        config.setdefault("project", {})["device"] = requested_device
        config.setdefault("evaluator", {})["device"] = requested_device
    if args.generations is not None:
        config.setdefault("optimization", {})["generations"] = args.generations
    if args.end_time_s is not None:
        if args.end_time_s <= 0.0:
            parser.error("--end-time-s must be positive")
        config.setdefault("physics", {})["end_time_s"] = float(args.end_time_s)
        config.setdefault("condensed_ignition", {})["reference_time_s"] = float(
            args.end_time_s
        )
        config.setdefault("evaluator", {})["end_time_s"] = float(args.end_time_s)
        config["evaluator"]["metric_evaluation_time_s"] = float(args.end_time_s)
        if isinstance(config.get("bc_global"), dict):
            config["bc_global"]["endTime_s"] = float(args.end_time_s)
            config["bc_global"]["evaluationTime_s"] = float(args.end_time_s)
    if args.grid_size is not None:
        if args.grid_size < 9:
            parser.error("--grid-size must be at least 9")
        config.setdefault("evaluator", {})["grid_size"] = int(args.grid_size)
    if args.population_size is not None:
        config.setdefault("optimization", {})["population_size"] = args.population_size
        # Debug overrides use one topology per 10 variants unless explicitly set.
        if args.population_size != 1000:
            n_top = min(20, max(1, args.population_size // 10))
            while args.population_size % n_top != 0 and n_top > 1:
                n_top -= 1
            config["optimization"]["initial_topologies"] = n_top
            config["optimization"]["variants_per_topology"] = args.population_size // n_top
            total_mating = min(100, args.population_size)
            perf = max(1, int(round(total_mating * 0.70)))
            topo = min(n_top, max(0, int(round(total_mating * 0.20))))
            div = max(0, total_mating - perf - topo)
            config["optimization"]["mating_performance_count"] = perf
            config["optimization"]["mating_topology_count"] = topo
            config["optimization"]["mating_diversity_count"] = div

    seed = int(config.get("project", {}).get("seed", 20260827))
    set_reproducible_seed(seed)

    python_root = args.package_root.resolve() / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow

    workflow = NSGA2ElectricalSolidWorkflow(
        package_root=args.package_root,
        config=config,
        workdir=args.workdir,
        allow_debug_physics=args.allow_debug_physics,
    )
    try:
        result = workflow.run()
    except RuntimeError:
        diagnostic_path = args.workdir.resolve() / "final" / "NO_FEASIBLE_IGNITING_DESIGN.json"
        if args.allow_no_feasible and diagnostic_path.is_file():
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            diagnostic.update(
                {
                    "workdir": str(args.workdir.resolve()),
                    "smoke_or_benchmark_completed": True,
                    "exit_status_overridden_to_success": True,
                }
            )
            print(json.dumps(diagnostic, indent=2))
            return 0
        raise
    print(
        json.dumps(
            {
                "recommended_geometry_id": result.geometry_id,
                "topology_id": result.topology_id,
                "objectives": result.objectives.tolist() if result.objectives is not None else None,
                "workdir": str(args.workdir.resolve()),
                "openfoam_used": False,
                "gas_phase_cfd_used": False,
                "model_scope": (
                    "bc_global_preflame_plus_post_onset_condensed_reaction_propagation"
                    if bool(config.get("propagation_refinement", {}).get("enabled", False))
                    else "condensed_phase_no_empirical_surface_progress"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
