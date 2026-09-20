from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ecsp_v6.config import load_config
from ecsp_v6.physics.potential import solve_potential
from mps_numerical_preflight import run_preflight


ROOT = Path(__file__).resolve().parents[2]


def test_representative_spd_pcg_preflight_cpu_fp32() -> None:
    result = run_preflight(
        config_path=ROOT / "config/m2_mps.yaml",
        device=torch.device("cpu"),
        dtype=torch.float32,
        grid_size=65,
        spd_probes=4,
    )
    assert result["status"] == "passed"
    assert result["gates"]["symmetry"] is True
    assert result["gates"]["spd"] is True
    assert result["gates"]["actual_method_is_pcg"] is True
    assert result["known_solution"]["method"].startswith(
        "pcg_symmetric_equilibrated_correction_"
    )
    assert result["known_solution"]["true_relative_residual"] <= 1.0e-6


def test_pcg_with_legacy_multigrid_is_not_silently_rerouted() -> None:
    config = load_config(ROOT / "config/m2_mps.yaml")
    config = copy.deepcopy(config)
    config["numerics"]["potentialSolver"]["methodCoupled"] = "pcg"
    config["numerics"]["potentialSolver"]["preconditioner"] = "multigrid"

    size = 9
    anode = torch.zeros((1, size, size), dtype=torch.bool)
    cathode = torch.zeros_like(anode)
    anode[:, 2:-2, 2] = True
    cathode[:, 2:-2, -3] = True
    fixed = anode | cathode

    from ecsp_v6.physics.geometry import GeometryBatch

    geometry = GeometryBatch(
        geometry_ids=["routing"],
        anode=anode,
        cathode=cathode,
        fixed=fixed,
        propellant=~fixed,
        grid_size=size,
        domain_size_m=0.02,
        minimum_gap_m=5.0e-4,
    )
    shape = fixed.shape
    potential = torch.full(shape, 130.0)
    cation = torch.ones(shape)
    anion = torch.ones(shape)
    diffusion = torch.zeros(shape)
    sigma = torch.ones(shape)

    with pytest.raises(ValueError, match="Invalid potential solver combination"):
        solve_potential(
            potential,
            cation,
            anion,
            diffusion,
            diffusion,
            sigma,
            geometry,
            config,
            260.0,
            0.02 / (size - 1),
            static=False,
        )


def test_gpu_resident_residual_minimising_fallback_recovers_small_spd_case() -> None:
    import copy

    from ecsp_v6.physics.potential import (
        _apply_operator,
        _build_linear_system,
        _norm_batch,
        _solve_pcg_system,
    )
    from mps_numerical_preflight import (
        _known_solution,
        _representative_interdigitated_geometry,
        _stress_conductivity,
    )

    config = load_config(ROOT / "config/m2_mps.yaml")
    n = 33
    geometry = _representative_interdigitated_geometry(
        n,
        device=torch.device("cpu"),
        domain_size_m=float(config["geometry"]["domainSize_m"]),
        minimum_gap_m=float(config["geometry"]["minimumElectrodeGap_m"]),
    )
    sigma = _stress_conductivity(
        geometry,
        dtype=torch.float32,
        minimum=float(config["electrical"]["conductivityMinimum_S_per_m"]),
        maximum=float(config["electrical"]["conductivityMaximum_S_per_m"]),
    )
    spacing = geometry.domain_size_m / float(n - 1)
    system = _build_linear_system(
        sigma,
        torch.zeros_like(sigma),
        geometry,
        float(config["coupled"]["voltage_V"]),
        float(config["electrical"]["cathodeVoltage_V"]),
        spacing,
    )
    exact = _known_solution(n, system["active"], torch.float32)
    known_system = dict(system)
    known_system["b"] = _apply_operator(exact, system)
    solver = copy.deepcopy(config["numerics"]["potentialSolver"])
    solver["iterativeRefinementRounds"] = 1
    solver["fallbackMaximumIterations"] = 5000
    solver["fallbackCheckInterval"] = 5

    solution, diagnostics = _solve_pcg_system(
        torch.zeros_like(exact),
        known_system,
        solver,
        maximum_iterations=1,  # force the MPS-resident fallback path
        relative_tolerance=1.0e-5,
        absolute_tolerance=1.0e-8,
    )
    residual = known_system["b"] - _apply_operator(solution, known_system)
    relative = _norm_batch(residual) / torch.clamp(
        _norm_batch(known_system["b"]), min=1.0
    )
    assert diagnostics.fallback_used is not None
    assert bool(diagnostics.fallback_used[0])
    assert bool(diagnostics.converged[0])
    assert float(relative[0]) <= 1.0e-5
