#!/usr/bin/env python3
from __future__ import annotations

"""Convert any v7.9.4-compatible NSGA-II YAML to the v7.9.5 hybrid backend."""

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create an A100 CUDA-FP64 + C++ CPU-FP64 hybrid configuration while "
            "preserving geometry, composition, constraints and objectives from the input YAML."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu-workers", type=int, default=40)
    parser.add_argument("--host-reserve-vcpus", type=int, default=8)
    parser.add_argument("--cuda-batch-size", type=int, default=64)
    parser.add_argument("--cuda-min-batch-size", type=int, default=8)
    parser.add_argument("--cpu-reserve-per-wave", type=int, default=16)
    parser.add_argument("--physics-batch-size", type=int, default=192)
    parser.add_argument("--vmin-tolerance-V", type=float, default=None)
    parser.add_argument("--vmin-max-iterations", type=int, default=None)
    args = parser.parse_args()

    if args.cpu_workers < 1:
        parser.error("--cpu-workers must be >= 1")
    if args.host_reserve_vcpus < 1:
        parser.error("--host-reserve-vcpus must be >= 1")
    if args.cuda_batch_size < 1 or args.cuda_min_batch_size < 1:
        parser.error("CUDA batch sizes must be >= 1")
    if args.cuda_min_batch_size > args.cuda_batch_size:
        parser.error("--cuda-min-batch-size cannot exceed --cuda-batch-size")
    if args.physics_batch_size < args.cuda_batch_size + 1:
        parser.error("--physics-batch-size should exceed --cuda-batch-size")

    cfg = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"YAML root must be a mapping: {args.input}")

    project = cfg.setdefault("project", {})
    project["name"] = f"{project.get('name', 'ECSP')}_Hybrid_A100_CPU48"
    project["device"] = "cuda"

    opt = cfg.setdefault("optimization", {})
    opt["physics_batch_size"] = int(args.physics_batch_size)

    evaluator = cfg.setdefault("evaluator", {})
    evaluator.update(
        {
            "backend": "hybrid_cuda_cpu",
            "device": "cuda",
            "cpp_auto_build": True,
            "cpp_parallel_cases": 1,
            "hybrid_cpu_executable": "build/ecsp_cpp_solver_hybrid",
            "hybrid_cpu_worker_cases": int(args.cpu_workers),
            "hybrid_reserved_host_vcpus": int(args.host_reserve_vcpus),
            "hybrid_cpu_reserve_per_wave": int(args.cpu_reserve_per_wave),
            "hybrid_cuda_enabled": True,
            "hybrid_require_cuda": True,
            "hybrid_cuda_device": "cuda:0",
            "hybrid_cuda_batch_size": int(args.cuda_batch_size),
            "hybrid_cuda_min_batch_size": int(args.cuda_min_batch_size),
            "hybrid_cuda_allow_partial_final_batch": True,
            "hybrid_cuda_fallback_to_cpu": True,
            "hybrid_cuda_max_retries": 1,
            "hybrid_cuda_deterministic": True,
            "hybrid_poll_interval_s": 0.05,
        }
    )
    base_overrides = evaluator.setdefault("base_overrides", {})
    numerics = base_overrides.setdefault("numerics", {})
    numerics["physicsDevice"] = "cpu"
    numerics["physicsDtype"] = "float64"

    vmin = cfg.setdefault("minimum_ignition_voltage_search", {})
    if args.vmin_tolerance_V is not None:
        if args.vmin_tolerance_V <= 0.0:
            parser.error("--vmin-tolerance-V must be positive")
        vmin["tolerance_V"] = float(args.vmin_tolerance_V)
    if args.vmin_max_iterations is not None:
        if args.vmin_max_iterations < 1:
            parser.error("--vmin-max-iterations must be >= 1")
        vmin["maximum_bisection_iterations"] = int(args.vmin_max_iterations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
