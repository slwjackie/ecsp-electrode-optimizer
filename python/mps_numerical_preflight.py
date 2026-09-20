#!/usr/bin/env python3
"""Numerical preflight for the ECSP potential solve on Apple MPS.

The check deliberately exercises the 193 x 193 free-form electrode stencil,
MPS-sensitive reductions, the symmetric Jacobi equilibration, and the exact
PCG implementation selected by the M2 profile.  It fails before the expensive
physics loop if the backend silently routes to another method or if the
operator/solver assumptions are violated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from ecsp_v6.config import load_config
from ecsp_v6.physics.electrochem import _batch_scatter_max, _batch_scatter_sum
from ecsp_v6.physics.geometry import GeometryBatch
from ecsp_v6.physics.numerics import (
    batch_quantile_masked,
    configure_torch,
    resolve_device,
    resolve_dtype,
    validate_device_dtype,
)
from ecsp_v6.physics.potential import (
    _apply_operator,
    _build_linear_system,
    _dot_batch,
    _equilibrate_linear_system,
    _norm_batch,
    _solve_pcg_system,
)


def _representative_interdigitated_geometry(
    grid_size: int,
    *,
    device: torch.device,
    domain_size_m: float,
    minimum_gap_m: float,
) -> GeometryBatch:
    """Connected interdigitated combs with approximately 8% area per polarity."""
    if grid_size < 33 or grid_size % 2 == 0:
        raise ValueError("The representative preflight grid must be odd and >= 33")

    n = int(grid_size)
    anode = torch.zeros((1, n, n), device=device, dtype=torch.bool)
    cathode = torch.zeros_like(anode)

    # The dimensions below are scaled from the audited 193-grid construction:
    # six-pixel buses and six four-pixel fingers per polarity.  On other odd
    # grids they retain the same approximate area and topology.
    margin = max(2, round(5 * (n - 1) / 192))
    bus_width = max(2, round(6 * (n - 1) / 192))
    finger_width = max(2, round(4 * (n - 1) / 192))
    middle = n // 2
    central_half_gap = max(3, round(7.5 * (n - 1) / 192))
    anode_tip = middle - central_half_gap
    cathode_tip = n - anode_tip
    left_bus_end = margin + bus_width
    right_bus_start = n - margin - bus_width

    anode[:, margin : n - margin, margin:left_bus_end] = True
    cathode[:, margin : n - margin, right_bus_start : n - margin] = True

    usable = max(n - 2 * margin - finger_width, 1)
    count = 6
    pitch = usable / float(2 * count)
    for k in range(count):
        row_a = margin + int(round((2 * k + 0.55) * pitch))
        row_c = margin + int(round((2 * k + 1.55) * pitch))
        row_a = min(max(row_a, margin), n - margin - finger_width)
        row_c = min(max(row_c, margin), n - margin - finger_width)
        anode[:, row_a : row_a + finger_width, left_bus_end:anode_tip] = True
        cathode[:, row_c : row_c + finger_width, cathode_tip:right_bus_start] = True

    fixed = anode | cathode
    return GeometryBatch(
        geometry_ids=["mps_preflight_interdigitated_comb"],
        anode=anode,
        cathode=cathode,
        fixed=fixed,
        propellant=~fixed,
        grid_size=n,
        domain_size_m=float(domain_size_m),
        minimum_gap_m=float(minimum_gap_m),
    )


def _stress_conductivity(
    geometry: GeometryBatch,
    *,
    dtype: torch.dtype,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Smooth field spanning the configured transport conductivity bounds."""
    n = geometry.grid_size
    coordinate = torch.linspace(
        0.0, 1.0, n, device=geometry.fixed.device, dtype=dtype
    )
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    raw_phase = (
        0.42
        + 0.23 * torch.sin(2.0 * math.pi * xx) * torch.cos(3.0 * math.pi * yy)
        + 0.25 * xx
        + 0.10 * yy
    )
    phase = (raw_phase - raw_phase.amin()) / torch.clamp(
        raw_phase.amax() - raw_phase.amin(), min=torch.finfo(dtype).eps
    )
    log_minimum = math.log(float(minimum))
    log_maximum = math.log(float(maximum))
    sigma = torch.exp(log_minimum + (log_maximum - log_minimum) * phase)[None]
    return torch.where(
        geometry.fixed,
        torch.full_like(sigma, float(maximum)),
        sigma,
    )


def _mps_sensitive_operation_smoke(device: torch.device) -> dict[str, bool]:
    batch_index = torch.tensor([0, 0, 1, 2, 2, 2], device=device, dtype=torch.int64)
    integer_values = torch.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=torch.int64)
    floating_values = integer_values.to(torch.float32) * 0.5

    summed_int = _batch_scatter_sum(integer_values, batch_index, 4)
    summed_float = _batch_scatter_sum(floating_values, batch_index, 4)
    maximum = _batch_scatter_max(floating_values, batch_index, 4)

    field = torch.tensor(
        [[[1.0, 7.0, 3.0], [5.0, 9.0, 11.0]], [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]],
        device=device,
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[[True, False, True], [True, False, True]], [[False, True, False], [True, True, False]]],
        device=device,
        dtype=torch.bool,
    )
    quantile = batch_quantile_masked(field, mask, 0.5)

    checks = {
        "int64_batch_sum": bool(
            torch.equal(summed_int.cpu(), torch.tensor([3, 3, 15, 0], dtype=torch.int64))
        ),
        "float_batch_sum": bool(
            torch.allclose(
                summed_float.cpu(), torch.tensor([1.5, 1.5, 7.5, 0.0]), atol=0.0, rtol=0.0
            )
        ),
        "batch_amax": bool(
            torch.allclose(
                maximum.cpu(), torch.tensor([1.0, 1.5, 3.0, 0.0]), atol=0.0, rtol=0.0
            )
        ),
        "fixed_shape_quantile": bool(
            torch.allclose(
                quantile.cpu(), torch.tensor([4.0, 8.0]), atol=2.0e-6, rtol=0.0
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"MPS-sensitive operation smoke failed: {checks}")
    return checks


def _known_solution(grid_size: int, active: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    coordinate = torch.linspace(
        0.0, 1.0, grid_size, device=active.device, dtype=dtype
    )
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    exact = (
        torch.sin(math.pi * xx) * torch.sin(2.0 * math.pi * yy)
        + 0.2 * torch.cos(3.0 * math.pi * xx) * torch.sin(math.pi * yy)
    )[None]
    return torch.where(active, exact, torch.zeros_like(exact))


def run_preflight(
    *,
    config_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    grid_size: int,
    spd_probes: int,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_device_dtype(device, dtype)
    configure_torch(device, bool(config["numerics"].get("deterministic", True)))

    geometry = _representative_interdigitated_geometry(
        grid_size,
        device=device,
        domain_size_m=float(config["geometry"]["domainSize_m"]),
        minimum_gap_m=float(config["geometry"]["minimumElectrodeGap_m"]),
    )
    electrical = config["electrical"]
    sigma = _stress_conductivity(
        geometry,
        dtype=dtype,
        minimum=float(electrical["conductivityMinimum_S_per_m"]),
        maximum=float(electrical["conductivityMaximum_S_per_m"]),
    )
    spacing = geometry.domain_size_m / float(grid_size - 1)
    zeros = torch.zeros_like(sigma)
    system = _build_linear_system(
        sigma,
        zeros,
        geometry,
        float(config["coupled"]["voltage_V"]),
        float(config["electrical"]["cathodeVoltage_V"]),
        spacing,
    )
    active = system["active"]
    eps = torch.finfo(dtype).eps

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260830)
    x_cpu = torch.randn((1, grid_size, grid_size), generator=generator, dtype=dtype)
    y_cpu = torch.randn((1, grid_size, grid_size), generator=generator, dtype=dtype)
    x = torch.where(active, x_cpu.to(device), torch.zeros_like(sigma))
    y = torch.where(active, y_cpu.to(device), torch.zeros_like(sigma))
    ax = _apply_operator(x, system)
    ay = _apply_operator(y, system)
    y_ax = _dot_batch(y, ax)
    x_ay = _dot_batch(x, ay)
    row_sum_bound = (
        system["diagonal"]
        + system["ce"]
        + system["cw"]
        + system["cn"]
        + system["cs"]
    ).amax(dim=(-2, -1))
    symmetry_scale = torch.clamp(
        _norm_batch(x) * _norm_batch(y) * row_sum_bound, min=1.0
    )
    symmetry_absolute_error = torch.abs(y_ax - x_ay)
    symmetry_tolerance = 32.0 * eps * symmetry_scale
    symmetry_relative_error = symmetry_absolute_error / torch.clamp(
        torch.maximum(torch.abs(y_ax), torch.abs(x_ay)), min=1.0
    )
    symmetry_passed = bool(torch.all(symmetry_absolute_error <= symmetry_tolerance))

    probes_cpu = torch.randn(
        (int(spd_probes), grid_size, grid_size), generator=generator, dtype=dtype
    )
    probes = torch.where(active, probes_cpu.to(device), torch.zeros_like(probes_cpu, device=device))
    a_probes = _apply_operator(probes, system)
    rayleigh_numerator = _dot_batch(probes, a_probes)
    rayleigh_denominator = torch.clamp(_dot_batch(probes, probes), min=1.0)
    rayleigh = rayleigh_numerator / rayleigh_denominator
    spd_passed = bool(torch.all(torch.isfinite(rayleigh) & (rayleigh > 0.0)))

    equilibrated, _ = _equilibrate_linear_system(system)
    unit_diagonal_error = torch.abs(
        equilibrated["diagonal"][active] - torch.ones_like(equilibrated["diagonal"][active])
    ).amax()
    unit_diagonal_tolerance = 4.0 * eps
    unit_diagonal_passed = bool(unit_diagonal_error <= unit_diagonal_tolerance)

    exact = _known_solution(grid_size, active, dtype)
    known_system = dict(system)
    known_system["b"] = _apply_operator(exact, system)
    solver_cfg = config["numerics"]["potentialSolver"]
    relative_tolerance = float(solver_cfg["relativeToleranceCoupled"])
    absolute_tolerance = float(solver_cfg["absoluteTolerance"])
    numerical_solution, diagnostics = _solve_pcg_system(
        torch.zeros_like(exact),
        known_system,
        solver_cfg,
        maximum_iterations=int(solver_cfg["maximumIterationsCoupled"]),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    true_residual = torch.where(
        active,
        known_system["b"] - _apply_operator(numerical_solution, known_system),
        torch.zeros_like(numerical_solution),
    )
    true_relative_residual = _norm_batch(true_residual) / torch.clamp(
        _norm_batch(known_system["b"]), min=1.0
    )
    error = (numerical_solution - exact)[active]
    max_error = torch.abs(error).amax()
    rms_error = torch.sqrt(torch.mean(error * error))
    method_passed = diagnostics.method.startswith(
        "pcg_symmetric_equilibrated_correction_"
    )
    known_solution_passed = bool(
        torch.all(diagnostics.converged)
        and torch.all(true_relative_residual <= relative_tolerance)
        and torch.isfinite(max_error)
        and float(max_error.detach().cpu()) <= 2.0e-2
        and method_passed
    )

    operation_checks = _mps_sensitive_operation_smoke(device)
    gates = {
        "symmetry": symmetry_passed,
        "spd": spd_passed,
        "equilibrated_unit_diagonal": unit_diagonal_passed,
        "known_solution_pcg": known_solution_passed,
        "actual_method_is_pcg": method_passed,
        "mps_sensitive_operations": all(operation_checks.values()),
    }
    passed = all(gates.values())

    result: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "grid_size": int(grid_size),
        "unknown_count": int(active.sum().detach().cpu()),
        "anode_area_fraction": float(geometry.anode.to(torch.float32).mean().cpu()),
        "cathode_area_fraction": float(geometry.cathode.to(torch.float32).mean().cpu()),
        "conductivity_minimum_S_per_m": float(sigma.min().detach().cpu()),
        "conductivity_maximum_S_per_m": float(sigma.max().detach().cpu()),
        "symmetry_absolute_error": float(symmetry_absolute_error.max().detach().cpu()),
        "symmetry_relative_error": float(symmetry_relative_error.max().detach().cpu()),
        "symmetry_tolerance": float(symmetry_tolerance.max().detach().cpu()),
        "minimum_random_probe_rayleigh_quotient": float(rayleigh.min().detach().cpu()),
        "maximum_equilibrated_diagonal_error": float(unit_diagonal_error.detach().cpu()),
        "known_solution": {
            "method": diagnostics.method,
            "iterations": int(diagnostics.iterations[0].detach().cpu()),
            "restarts": int(diagnostics.restarts[0].detach().cpu()) if diagnostics.restarts is not None else 0,
            "refinement_rounds": int(diagnostics.refinement_rounds[0].detach().cpu()) if diagnostics.refinement_rounds is not None else 0,
            "fallback_used": bool(diagnostics.fallback_used[0].detach().cpu()) if diagnostics.fallback_used is not None else False,
            "true_relative_residual": float(true_relative_residual[0].detach().cpu()),
            "maximum_solution_error_V": float(max_error.detach().cpu()),
            "rms_solution_error_V": float(rms_error.detach().cpu()),
        },
        "operation_smoke": operation_checks,
        "gates": gates,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/m2_mps.yaml"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--grid-size", type=int, default=193)
    parser.add_argument("--spd-probes", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    result = run_preflight(
        config_path=args.config.resolve(),
        device=device,
        dtype=dtype,
        grid_size=args.grid_size,
        spd_probes=args.spd_probes,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
