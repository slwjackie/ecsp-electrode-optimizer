from __future__ import annotations

from pathlib import Path

import torch

from ecsp_v6.config import load_config
from ecsp_v6.physics.electrochem import _reaction_candidate_and_deta
from ecsp_v6.physics.numerics import physical_floor


ROOT = Path(__file__).resolve().parents[2]


def test_physical_floors_do_not_depend_on_dtype() -> None:
    config = load_config(ROOT / "config/m2_mps.yaml")
    expected = 1.0e-12
    assert physical_floor(config, "currentDensity_A_per_m2", 0.0) == expected
    assert expected != torch.finfo(torch.float32).eps
    assert expected != torch.finfo(torch.float64).eps


def test_mass_transfer_derivative_does_not_flush_to_zero_at_large_kinetic_current() -> None:
    config = load_config(ROOT / "config/m2_mps.yaml")
    channel = config["interface"]["lp"]["anode"]
    temperature = torch.tensor([298.15], dtype=torch.float32)
    k_forward = (
        float(channel["chargeTransferCoefficient"])
        * float(channel["electronNumber"])
        * float(config["transport"]["faradayConstant_C_per_mol"])
        / (float(config["transport"]["gasConstant_J_per_molK"]) * 298.15)
    )
    # Stay just below the exponential clamp while driving j_kinetic far above
    # the mass-transfer limit.  The algebraically stable derivative must remain
    # finite and positive on FP32/Metal rather than underflowing to zero.
    overpotential = torch.tensor([44.0 / k_forward], dtype=torch.float32)
    current, derivative = _reaction_candidate_and_deta(
        torch.ones(1, dtype=torch.float32),
        torch.full((1,), 3000.0, dtype=torch.float32),
        torch.full((1,), 4.5e-9, dtype=torch.float32),
        overpotential,
        temperature,
        channel,
        config,
        torch.full((1,), 40.0, dtype=torch.float32),
    )
    assert bool(torch.all(torch.isfinite(current)))
    assert bool(torch.all(torch.isfinite(derivative)))
    assert bool(torch.all(current > 0.0))
    assert bool(torch.all(derivative > 0.0))
