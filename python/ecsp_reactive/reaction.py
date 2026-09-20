"""Explicit Arrhenius progress and heat sources for the reactive Euler state.

Park & Yoh (2024) include ``rho*omega_dot`` and a reaction-energy term in
Eq. (1), but do not publish a gas/mixture progress-rate closure or the molar-to-
mass conversion needed to evaluate it.  The closure in this module is therefore
always labelled ``ASSUMED_NOT_FROM_PAPER`` and has no nominal defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import NCONS, REACTION_PROGRESS, RHO, TOTAL_ENERGY
from .provenance import ReactiveConfigurationError, ReactiveNumericalError


UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324


@dataclass(frozen=True)
class ArrheniusReaction:
    pre_exponential_s_inv: float
    activation_energy_J_per_mol: float
    reaction_order: float
    heat_release_J_per_kg: float
    gas_constant_J_per_mol_K: float = UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K
    provenance: dict[str, str] = field(
        default_factory=lambda: {
            "progress_rate_closure": "ASSUMED_NOT_FROM_PAPER",
            "reaction_parameters": "CALLER_SUPPLIED_NOT_PAPER_DEFAULTS",
            "specific_heat_release_conversion": "ASSUMED_NOT_FROM_PAPER",
            "paper_note": (
                "Eq. (1) gives source structure, not kinetic coefficients or the molar-mass conversion"
            ),
        },
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        checks = {
            "pre_exponential_s_inv": self.pre_exponential_s_inv,
            "activation_energy_J_per_mol": self.activation_energy_J_per_mol,
            "reaction_order": self.reaction_order,
        }
        for name, raw in checks.items():
            value = float(raw)
            if not np.isfinite(value) or value < 0.0:
                raise ReactiveConfigurationError(
                    f"{name} must be finite and non-negative"
                )
            object.__setattr__(self, name, value)
        heat = float(self.heat_release_J_per_kg)
        if not np.isfinite(heat):
            raise ReactiveConfigurationError("heat_release_J_per_kg must be finite")
        object.__setattr__(self, "heat_release_J_per_kg", heat)
        gas_constant = float(self.gas_constant_J_per_mol_K)
        if not np.isfinite(gas_constant) or gas_constant <= 0.0:
            raise ReactiveConfigurationError(
                "gas_constant_J_per_mol_K must be finite and positive"
            )
        object.__setattr__(self, "gas_constant_J_per_mol_K", gas_constant)

    def metadata(self) -> dict[str, Any]:
        return {"model": "one_step_irreversible_arrhenius", **self.provenance}


def arrhenius_rate(
    temperature_K: Any,
    reaction_progress: Any,
    model: ArrheniusReaction,
    *,
    completion_tolerance: float = 0.0,
) -> np.ndarray | float:
    """Return ``d(lambda)/dt`` with an explicit tolerance-terminal state.

    ``completion_tolerance=0`` preserves the historical exact-terminal API.
    For a positive caller-supplied tolerance, cells with
    ``0 <= 1-lambda <= completion_tolerance`` have exactly zero rate without
    changing ``lambda``.  This prevents sublinear/zero-order closures from
    taking an infinite geometric sequence of explicit headroom steps.  It is a
    declared numerical terminal semantics, not state clipping or a rate cap.
    """
    try:
        temperature, progress = np.broadcast_arrays(
            np.asarray(temperature_K, dtype=np.float64),
            np.asarray(reaction_progress, dtype=np.float64),
        )
    except (TypeError, ValueError) as exc:
        raise ReactiveConfigurationError(
            "temperature and reaction progress must broadcast"
        ) from exc
    if not np.isfinite(temperature).all() or np.any(temperature <= 0.0):
        raise ReactiveNumericalError(
            "invalid_reaction_temperature",
            "Arrhenius temperature must be finite and positive",
        )
    if not np.isfinite(progress).all() or np.any((progress < 0.0) | (progress > 1.0)):
        raise ReactiveNumericalError(
            "reaction_progress_out_of_range",
            "Arrhenius reaction progress must be in [0, 1]",
        )
    terminal_tolerance = float(completion_tolerance)
    if not np.isfinite(terminal_tolerance) or terminal_tolerance < 0.0:
        raise ReactiveConfigurationError(
            "completion_tolerance must be finite and non-negative"
        )
    exponent = -model.activation_energy_J_per_mol / (
        model.gas_constant_J_per_mol_K * temperature
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rate = (
            model.pre_exponential_s_inv
            * np.power(1.0 - progress, model.reaction_order)
            * np.exp(exponent)
        )
    # In particular, x**0 would otherwise return one at x=0 and spuriously
    # react completed cells.  A positive tolerance also supplies a finite
    # terminal event for order<1 while leaving the stored state untouched.
    rate = np.where((1.0 - progress) <= terminal_tolerance, 0.0, rate)
    if not np.isfinite(rate).all() or np.any(rate < 0.0):
        raise ReactiveNumericalError(
            "nonfinite_arrhenius_rate", "Arrhenius rate became non-finite or negative"
        )
    return float(rate) if rate.ndim == 0 else rate


def reaction_source(
    conservative_state: Any,
    temperature_K: Any,
    model: ArrheniusReaction,
    *,
    completion_tolerance: float = 0.0,
) -> np.ndarray:
    """Return Eq. (1)-shaped source with consistent progress/heat bookkeeping.

    ``S[rho*E] = rho*q*omega`` and ``S[rho*lambda] = rho*omega``.
    Therefore their nonzero ratio is exactly the configured specific heat
    release in J/kg; no independent heat cap or progress clipping is applied.
    """
    state = np.asarray(conservative_state, dtype=np.float64)
    if state.ndim < 1 or state.shape[-1] != NCONS:
        raise ReactiveConfigurationError(f"state last dimension must be {NCONS}")
    if not np.isfinite(state).all():
        raise ReactiveNumericalError(
            "nonfinite_conservative_state", "reaction source received non-finite state"
        )
    rho = state[..., RHO]
    if np.any(rho <= 0.0):
        raise ReactiveNumericalError(
            "invalid_density", "reaction source requires positive density"
        )
    progress = state[..., REACTION_PROGRESS] / rho
    omega = np.asarray(
        arrhenius_rate(
            temperature_K,
            progress,
            model,
            completion_tolerance=completion_tolerance,
        ),
        dtype=np.float64,
    )
    try:
        omega = np.broadcast_to(omega, rho.shape)
    except ValueError as exc:
        raise ReactiveConfigurationError(
            "temperature does not broadcast to the state shape"
        ) from exc
    progress_source = rho * omega
    heat_source = model.heat_release_J_per_kg * progress_source
    result = np.zeros_like(state, dtype=np.float64)
    result[..., TOTAL_ENERGY] = heat_source
    result[..., REACTION_PROGRESS] = progress_source
    if not np.isfinite(result).all():
        raise ReactiveNumericalError(
            "nonfinite_reaction_source", "reaction progress/energy source overflowed"
        )
    return result
