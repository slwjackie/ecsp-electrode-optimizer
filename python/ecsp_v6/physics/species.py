from __future__ import annotations

import math

import torch

from .composition_model import CompositionModel
from .geometry import GeometryBatch
from .numerics import harmonic_mean, physical_floor


def _flux_divergence_charged(
    concentration: torch.Tensor,
    diffusivity: torch.Tensor,
    potential: torch.Tensor,
    temperature: torch.Tensor,
    charge_number: float,
    gas_constant: float,
    faraday: float,
    propellant: torch.Tensor,
    spacing: float,
) -> torch.Tensor:
    """Finite-volume divergence of Nernst--Planck charged-species flux."""
    batch, rows, columns = concentration.shape
    flux_x = torch.zeros(
        (batch, rows, columns + 1),
        device=concentration.device,
        dtype=concentration.dtype,
    )
    flux_y = torch.zeros(
        (batch, rows + 1, columns),
        device=concentration.device,
        dtype=concentration.dtype,
    )

    valid_x = propellant[..., :, :-1] & propellant[..., :, 1:]
    d_face = harmonic_mean(diffusivity[..., :, :-1], diffusivity[..., :, 1:])
    c_face = 0.5 * (concentration[..., :, :-1] + concentration[..., :, 1:])
    t_face = torch.clamp(
        0.5 * (temperature[..., :, :-1] + temperature[..., :, 1:]), min=1.0
    )
    value_x = (
        -d_face * (concentration[..., :, 1:] - concentration[..., :, :-1]) / spacing
        - charge_number
        * d_face
        * faraday
        / (gas_constant * t_face)
        * c_face
        * (potential[..., :, 1:] - potential[..., :, :-1])
        / spacing
    )
    flux_x[..., :, 1:columns] = torch.where(
        valid_x, value_x, torch.zeros_like(value_x)
    )

    valid_y = propellant[..., :-1, :] & propellant[..., 1:, :]
    d_face = harmonic_mean(diffusivity[..., :-1, :], diffusivity[..., 1:, :])
    c_face = 0.5 * (concentration[..., :-1, :] + concentration[..., 1:, :])
    t_face = torch.clamp(
        0.5 * (temperature[..., :-1, :] + temperature[..., 1:, :]), min=1.0
    )
    value_y = (
        -d_face * (concentration[..., 1:, :] - concentration[..., :-1, :]) / spacing
        - charge_number
        * d_face
        * faraday
        / (gas_constant * t_face)
        * c_face
        * (potential[..., 1:, :] - potential[..., :-1, :])
        / spacing
    )
    flux_y[..., 1:rows, :] = torch.where(
        valid_y, value_y, torch.zeros_like(value_y)
    )

    divergence = (
        (flux_x[..., :, 1:] - flux_x[..., :, :-1]) / spacing
        + (flux_y[..., 1:, :] - flux_y[..., :-1, :]) / spacing
    )
    return torch.where(propellant, divergence, torch.zeros_like(divergence))


def _flux_divergence_neutral(
    concentration: torch.Tensor,
    diffusivity: torch.Tensor,
    propellant: torch.Tensor,
    spacing: float,
) -> torch.Tensor:
    """Finite-volume divergence of Fickian neutral-species flux."""
    batch, rows, columns = concentration.shape
    flux_x = torch.zeros(
        (batch, rows, columns + 1),
        device=concentration.device,
        dtype=concentration.dtype,
    )
    flux_y = torch.zeros(
        (batch, rows + 1, columns),
        device=concentration.device,
        dtype=concentration.dtype,
    )
    valid_x = propellant[..., :, :-1] & propellant[..., :, 1:]
    d_face = harmonic_mean(diffusivity[..., :, :-1], diffusivity[..., :, 1:])
    value_x = -d_face * (
        concentration[..., :, 1:] - concentration[..., :, :-1]
    ) / spacing
    flux_x[..., :, 1:columns] = torch.where(
        valid_x, value_x, torch.zeros_like(value_x)
    )
    valid_y = propellant[..., :-1, :] & propellant[..., 1:, :]
    d_face = harmonic_mean(diffusivity[..., :-1, :], diffusivity[..., 1:, :])
    value_y = -d_face * (
        concentration[..., 1:, :] - concentration[..., :-1, :]
    ) / spacing
    flux_y[..., 1:rows, :] = torch.where(
        valid_y, value_y, torch.zeros_like(value_y)
    )
    divergence = (
        (flux_x[..., :, 1:] - flux_x[..., :, :-1]) / spacing
        + (flux_y[..., 1:, :] - flux_y[..., :-1, :]) / spacing
    )
    return torch.where(propellant, divergence, torch.zeros_like(divergence))


def _limited_update(
    field: torch.Tensor,
    rate: torch.Tensor,
    dt: float,
    config: dict,
    reference: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Positivity-preserving bounded explicit concentration update.

    Limiter use is exposed as a diagnostic.  Research runs should keep its
    fraction small by reducing the outer time step or increasing species
    subcycles rather than interpreting heavily clipped solutions as converged.
    """
    raw_delta = dt * rate
    maximum_fraction = float(
        config["transport"]["maximumRelativeConcentrationChangePerStep"]
    )
    minimum_fraction = float(config["transport"]["concentrationMinimumFraction"])
    maximum_multiple = float(config["transport"]["concentrationMaximumMultiple"])
    lower_bound = reference * minimum_fraction
    upper_bound = reference * maximum_multiple
    limit_base = torch.maximum(field, torch.full_like(field, lower_bound))
    limit = maximum_fraction * limit_base
    limited_delta = torch.clamp(raw_delta, min=-limit, max=limit)
    comparison_rtol = float(
        config.get("numerics", {}).get(
            "limiterComparisonRelativeTolerance", 1e-7
        )
    )
    concentration_floor = physical_floor(
        config, "concentration_mol_per_m3", 1e-12
    )
    delta_scale = torch.maximum(
        torch.maximum(raw_delta.abs(), limited_delta.abs()),
        torch.maximum(field.abs(), torch.full_like(field, lower_bound)),
    )
    delta_threshold = concentration_floor + comparison_rtol * delta_scale
    delta_limited = torch.abs(limited_delta - raw_delta) > delta_threshold
    raw_updated = field + limited_delta
    updated = torch.clamp(
        raw_updated,
        min=lower_bound,
        max=upper_bound,
    )
    bound_scale = torch.maximum(
        torch.maximum(updated.abs(), raw_updated.abs()),
        torch.full_like(field, lower_bound),
    )
    bound_threshold = concentration_floor + comparison_rtol * bound_scale
    bound_error = torch.abs(updated - raw_updated)
    bound_limited = bound_error > bound_threshold

    # Match the native fail-closed rule: clipping must not turn overflow in a
    # proposed update or its limiter tolerances into an apparently valid
    # concentration.  NaN is propagated to the existing state/energy guards.
    scalar_inputs_finite = all(
        math.isfinite(float(value))
        for value in (
            dt,
            reference,
            maximum_fraction,
            minimum_fraction,
            maximum_multiple,
            comparison_rtol,
            concentration_floor,
            lower_bound,
            upper_bound,
        )
    )
    invalid = (
        ~torch.isfinite(field)
        | ~torch.isfinite(rate)
        | ~torch.isfinite(raw_delta)
        | ~torch.isfinite(limit_base)
        | ~torch.isfinite(limit)
        | ~torch.isfinite(limited_delta)
        | ~torch.isfinite(delta_scale)
        | ~torch.isfinite(delta_threshold)
        | ~torch.isfinite(torch.abs(limited_delta - raw_delta))
        | ~torch.isfinite(raw_updated)
        | ~torch.isfinite(updated)
        | ~torch.isfinite(bound_scale)
        | ~torch.isfinite(bound_threshold)
        | ~torch.isfinite(bound_error)
    )
    if not scalar_inputs_finite:
        invalid = torch.ones_like(field, dtype=torch.bool)
    updated = torch.where(invalid, torch.full_like(updated, torch.nan), updated)
    return updated, (invalid | delta_limited | bound_limited)


def _species_rates(
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    reaction: dict[str, torch.Tensor],
    chemical_rate: torch.Tensor,
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    spacing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gas_constant = float(config["transport"]["gasConstant_J_per_molK"])
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    if bool(config["coupled"]["useFullNernstPlanckTransport"]):
        rate_cation = -_flux_divergence_charged(
            state["cation"],
            transport["Dcation"],
            state["potential"],
            state["temperature"],
            float(config["transport"]["chargeNumberCation"]),
            gas_constant,
            faraday,
            geometry.propellant,
            spacing,
        )
        rate_anion = -_flux_divergence_charged(
            state["anion"],
            transport["Danion"],
            state["potential"],
            state["temperature"],
            float(config["transport"]["chargeNumberAnion"]),
            gas_constant,
            faraday,
            geometry.propellant,
            spacing,
        )
        rate_water = -_flux_divergence_neutral(
            state["water"],
            transport["Dwater"],
            geometry.propellant,
            spacing,
        )
    else:
        rate_cation = torch.zeros_like(state["cation"])
        rate_anion = torch.zeros_like(state["anion"])
        rate_water = torch.zeros_like(state["water"])

    chemical_salt_sink = (
        chemical_rate
        * composition.initial_lp_mol_per_m3
        * float(config["chemical"]["oxidizerConsumptionFractionPerUnitProgress"])
    )
    rate_cation = rate_cation - reaction["saltSink_mol_per_m3_s"] - chemical_salt_sink
    rate_anion = rate_anion - reaction["saltSink_mol_per_m3_s"] - chemical_salt_sink
    rate_water = rate_water - reaction["waterSink_mol_per_m3_s"]
    return rate_cation, rate_anion, rate_water


def _choose_substeps(
    state: dict[str, torch.Tensor],
    rates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    references: tuple[float, float, float],
    config: dict,
    dt: float,
) -> int:
    numerics = config.get("numerics", {})
    if not bool(numerics.get("speciesSubcycling", True)):
        return 1
    maximum = max(1, int(numerics.get("speciesMaximumSubsteps", 32)))
    mode = str(numerics.get("speciesSubcyclingMode", "fixed")).lower()
    if mode == "fixed":
        return min(max(1, int(numerics.get("speciesFixedSubsteps", 2))), maximum)
    if mode != "dynamic":
        raise ValueError(f"Unknown numerics.speciesSubcyclingMode: {mode}")

    # Dynamic mode performs one host synchronization per outer time step.
    # Fixed mode is recommended for long A100 production runs.
    target = max(float(numerics.get("speciesRelativeChangeTarget", 0.01)), 1e-12)
    minimum_fraction = float(config["transport"]["concentrationMinimumFraction"])
    ratio = torch.zeros((), device=state["cation"].device, dtype=state["cation"].dtype)
    for name, rate, reference in zip(("cation", "anion", "water"), rates, references):
        denominator = torch.maximum(
            state[name], torch.full_like(state[name], reference * minimum_fraction)
        )
        ratio = torch.maximum(
            ratio,
            torch.amax(torch.abs(dt * rate) / torch.clamp(denominator, min=1e-30)),
        )
    required = int(math.ceil(float(ratio.detach().cpu()) / target))
    return min(max(1, required), maximum)


def update_species(
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    reaction: dict[str, torch.Tensor],
    chemical_rate: torch.Tensor,
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    spacing: float,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Advance Li+, ClO4- and water with optional operator subcycling.

    The original MATLAB implementation used one limited explicit update per
    outer step.  Setting ``speciesSubcycling: false`` reproduces that ordering.
    The A100 profiles use fixed subcycles to reduce concentration clipping and
    improve time integration without changing the governing equations.
    """
    references = (
        composition.initial_cation_mol_per_m3,
        composition.initial_anion_mol_per_m3,
        composition.initial_water_mol_per_m3,
    )
    first_rates = _species_rates(
        state,
        transport,
        reaction,
        chemical_rate,
        geometry,
        config,
        composition,
        spacing,
    )
    substeps = _choose_substeps(state, first_rates, references, config, dt)
    sub_dt = dt / substeps
    prop = geometry.propellant
    count = torch.clamp(prop.sum(dim=(-2, -1)).to(state["cation"].dtype), min=1.0)
    accumulated = {
        "cation": torch.zeros_like(count),
        "anion": torch.zeros_like(count),
        "water": torch.zeros_like(count),
        "combined": torch.zeros_like(count),
    }
    working_transport = transport
    recompute_transport = bool(
        config.get("numerics", {}).get("recomputeTransportEachSpeciesSubstep", True)
    )

    for substep in range(substeps):
        if substep > 0 and recompute_transport:
            # Local import avoids a module-level circular dependency.
            from .electrochem import transport_fields  # noqa: PLC0415

            working_transport = transport_fields(state, geometry, config, composition)
        rates = (
            first_rates
            if substep == 0
            else _species_rates(
                state,
                working_transport,
                reaction,
                chemical_rate,
                geometry,
                config,
                composition,
                spacing,
            )
        )
        state["cation"], limited_cation = _limited_update(
            state["cation"], rates[0], sub_dt, config, references[0]
        )
        state["anion"], limited_anion = _limited_update(
            state["anion"], rates[1], sub_dt, config, references[1]
        )
        state["water"], limited_water = _limited_update(
            state["water"], rates[2], sub_dt, config, references[2]
        )

        relax = float(config["transport"]["electroneutralRelaxation"])
        neutral = 0.5 * (state["cation"] + state["anion"])
        state["cation"] = (1.0 - relax) * state["cation"] + relax * neutral
        state["anion"] = (1.0 - relax) * state["anion"] + relax * neutral
        for name in ("cation", "anion", "water"):
            state[name] = torch.where(
                geometry.fixed, torch.zeros_like(state[name]), state[name]
            )

        combined = (limited_cation | limited_anion | limited_water) & prop
        accumulated["cation"] += (
            (limited_cation & prop).sum(dim=(-2, -1)).to(count.dtype) / count
        )
        accumulated["anion"] += (
            (limited_anion & prop).sum(dim=(-2, -1)).to(count.dtype) / count
        )
        accumulated["water"] += (
            (limited_water & prop).sum(dim=(-2, -1)).to(count.dtype) / count
        )
        accumulated["combined"] += (
            combined.sum(dim=(-2, -1)).to(count.dtype) / count
        )

    divisor = float(substeps)
    return {
        "cationLimiterFraction": accumulated["cation"] / divisor,
        "anionLimiterFraction": accumulated["anion"] / divisor,
        "waterLimiterFraction": accumulated["water"] / divisor,
        "combinedLimiterFraction": accumulated["combined"] / divisor,
        "speciesSubsteps": torch.full_like(count, float(substeps)),
    }
