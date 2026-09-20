#!/usr/bin/env python3
"""Compare CPU FP32 and CPU FP64 potential solutions on the 193-grid comb case."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from ecsp_v6.config import load_config
from ecsp_v6.physics.potential import _apply_operator, _build_linear_system, _dot_batch, _solve_pcg_system
from mps_numerical_preflight import (
    _representative_interdigitated_geometry,
    _stress_conductivity,
)


def _solve(dtype: torch.dtype, config: dict, grid_size: int) -> tuple[torch.Tensor, dict]:
    device = torch.device("cpu")
    geometry = _representative_interdigitated_geometry(
        grid_size,
        device=device,
        domain_size_m=float(config["geometry"]["domainSize_m"]),
        minimum_gap_m=float(config["geometry"]["minimumElectrodeGap_m"]),
    )
    sigma = _stress_conductivity(
        geometry,
        dtype=dtype,
        minimum=float(config["electrical"]["conductivityMinimum_S_per_m"]),
        maximum=float(config["electrical"]["conductivityMaximum_S_per_m"]),
    )
    spacing = geometry.domain_size_m / float(grid_size - 1)
    system = _build_linear_system(
        sigma,
        torch.zeros_like(sigma),
        geometry,
        float(config["coupled"]["voltage_V"]),
        float(config["electrical"]["cathodeVoltage_V"]),
        spacing,
    )
    initial = torch.where(
        system["active"],
        torch.full_like(sigma, 130.0),
        torch.zeros_like(sigma),
    )
    solver = copy.deepcopy(config["numerics"]["potentialSolver"])
    solver["methodStatic"] = "pcg"
    solver["methodCoupled"] = "pcg"
    solver["preconditioner"] = "jacobi"
    solver["symmetricEquilibration"] = True
    solver["correctionForm"] = True
    if dtype == torch.float64:
        rtol, atol = 1.0e-10, 1.0e-12
    else:
        rtol = float(solver["relativeToleranceCoupled"])
        atol = float(solver["absoluteTolerance"])
    solution, diagnostics = _solve_pcg_system(
        initial,
        system,
        solver,
        maximum_iterations=max(3000, int(solver["maximumIterationsStatic"])),
        relative_tolerance=rtol,
        absolute_tolerance=atol,
    )
    residual = system["b"] - _apply_operator(solution, system)
    relative = torch.sqrt(torch.clamp(_dot_batch(residual, residual), min=0.0)) / torch.clamp(
        torch.sqrt(torch.clamp(_dot_batch(system["b"], system["b"]), min=0.0)), min=1.0
    )
    return solution, {
        "method": diagnostics.method,
        "iterations": int(diagnostics.iterations[0]),
        "restarts": int(diagnostics.restarts[0]) if diagnostics.restarts is not None else 0,
        "refinement_rounds": int(diagnostics.refinement_rounds[0]) if diagnostics.refinement_rounds is not None else 0,
        "fallback_used": bool(diagnostics.fallback_used[0]) if diagnostics.fallback_used is not None else False,
        "true_relative_residual": float(relative[0]),
        "converged": bool(diagnostics.converged[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/m2_mps.yaml"))
    parser.add_argument("--grid-size", type=int, default=193)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    solution32, diag32 = _solve(torch.float32, config, args.grid_size)
    solution64, diag64 = _solve(torch.float64, config, args.grid_size)
    difference = solution32.to(torch.float64) - solution64
    max_error = float(torch.abs(difference).max())
    rms_error = float(torch.sqrt(torch.mean(difference * difference)))
    alpha = float(config["interface"]["lp"]["anode"]["chargeTransferCoefficient"])
    n = float(config["interface"]["lp"]["anode"]["electronNumber"])
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    gas_constant = float(config["transport"]["gasConstant_J_per_molK"])
    temperature = float(config["thermal"]["initialTemperature_K"])
    bv_amplification = math.exp(alpha * n * faraday / (gas_constant * temperature) * rms_error)
    result = {
        "grid_size": args.grid_size,
        "fp32": diag32,
        "fp64": diag64,
        "maximum_fp32_vs_fp64_potential_error_V": max_error,
        "rms_fp32_vs_fp64_potential_error_V": rms_error,
        "estimated_local_bv_multiplicative_error_from_rms": bv_amplification,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if diag32["converged"] and diag64["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
