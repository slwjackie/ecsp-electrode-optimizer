from __future__ import annotations

"""Condensed-phase ECSP solver used by the no-F NSGA-II workflow.

This module intentionally contains no flame-progress variable and no
reaction--diffusion post-ignition closure.  It solves the existing v7.7.2
Electrical + electrochemical + solid thermal/decomposition equations and
returns four optimisation metrics:

1. condensed-phase ignition delay;
2. area-averaged undecomposed fraction at the configured evaluation time;
3. electrical input energy at the same evaluation time;
4. peak current-congestion ratio.

The ignition delay is an operational *condensed-phase onset* metric, not a
claim of a self-sustained gas flame.  By default its temperature threshold is
the 622.15 K non-metallized M0 decomposition-onset reference stored in the
configuration provenance file.
"""

from typing import Any
import time as wall_time

import torch

from .composition_model import CompositionModel
from .electrochem import (
    initial_state,
    solve_electrical_and_reaction,
    transport_fields,
    update_interfacial_blocking,
)
from .geometry import GeometryBatch
from .numerics import (
    batch_masked_max,
    batch_masked_mean,
    batch_quantile_masked,
    laplacian_neumann,
    physical_floor,
)
from .species import update_species


def _density(config: dict, composition: CompositionModel) -> float:
    value = config["thermal"].get("density_kg_per_m3")
    return composition.density_kg_per_m3 if value is None else float(value)


def _validate_explicit_stability(
    config: dict, grid_size: int, density: float
) -> dict[str, float]:
    spacing = float(config["geometry"]["domainSize_m"]) / max(grid_size - 1, 1)
    dt = float(config["coupled"]["timeStep_s"])
    alpha_thermal = float(config["thermal"]["thermalConductivity_W_per_mK"]) / (
        density * float(config["thermal"]["heatCapacity_J_per_kgK"])
    )
    # F has been removed.  Only solid thermal diffusion and the reduced
    # interfacial gas-concentration diffusion constrain this explicit update.
    maximum_diffusivity = max(
        alpha_thermal,
        float(config["gas"]["diffusivity_m2_per_s"]),
    )
    stable_dt = 0.24 * spacing * spacing / max(maximum_diffusivity, 1e-30)
    if dt > stable_dt:
        raise ValueError(
            f"timeStep_s={dt:.6g} violates explicit heat/gas stability; "
            f"use dt <= {stable_dt:.6g} for grid {grid_size}"
        )
    return {
        "spacing_m": spacing,
        "thermal_diffusivity_m2_per_s": alpha_thermal,
        "explicit_stable_dt_s": stable_dt,
        "configured_dt_s": dt,
        "diffusive_CFL_fraction": dt / stable_dt,
    }


def _history_at_time(
    time: torch.Tensor,
    history: torch.Tensor,
    target_time_s: float,
) -> torch.Tensor:
    """Linearly interpolate a [time,batch] history at a physical time."""
    if history.ndim != 2 or history.shape[0] != time.numel():
        raise ValueError("history must have shape [time,batch]")
    target = float(target_time_s)
    if target <= float(time[0]):
        return history[0]
    if target >= float(time[-1]):
        return history[-1]
    query = torch.as_tensor(target, device=time.device, dtype=time.dtype)
    upper = int(torch.searchsorted(time, query, right=False).item())
    upper = max(1, min(upper, time.numel() - 1))
    lower = upper - 1
    t0 = time[lower]
    t1 = time[upper]
    weight = (query - t0) / torch.clamp(t1 - t0, min=1e-30)
    return history[lower] + weight * (history[upper] - history[lower])


def run_coupled_batch(
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float,
    dtype: torch.dtype,
    *,
    save_fields: bool = False,
    external_surface_feedback_W_per_m2: torch.Tensor | None = None,
    external_surface_feedback_schedule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the v7.7.2 condensed-phase electro-thermo-chemical model.

    The legacy function name is retained so existing runners can call the
    solver.  F, gas-phase flame heat and CFD feedback are deliberately absent.
    """
    if (
        external_surface_feedback_W_per_m2 is not None
        or external_surface_feedback_schedule is not None
    ):
        raise ValueError(
            "The no-F condensed-phase solver does not accept gas-to-solid CFD "
            "feedback. Use a separate high-fidelity validation workflow."
        )

    batch = geometry.batch_size
    grid_size = geometry.grid_size
    density = _density(config, composition)
    stability = _validate_explicit_stability(config, grid_size, density)
    spacing = stability["spacing_m"]
    dt = float(config["coupled"]["timeStep_s"])
    end_time = float(config["coupled"]["endTime_s"])
    steps = int(round(end_time / dt))
    if steps <= 0:
        raise ValueError("Condensed-phase simulation must contain at least one time step")
    time = torch.arange(
        1, steps + 1, device=geometry.fixed.device, dtype=dtype
    ) * dt

    metric_cfg = config.get("condensedPhaseMetrics", {})
    evaluation_time_s = float(metric_cfg.get("evaluationTime_s", 2.0))
    if evaluation_time_s <= 0.0 or evaluation_time_s > end_time + 1e-12:
        raise ValueError(
            "condensedPhaseMetrics.evaluationTime_s must lie in (0, coupled.endTime_s]"
        )

    ignition_cfg = config.get("condensedIgnition", {})
    onset_temperature_K = float(
        ignition_cfg.get("decompositionOnsetTemperature_K", 622.15)
    )
    minimum_progress = float(
        ignition_cfg.get("minimumChemicalProgressNumericalGuard", 0.0)
    )
    minimum_rate = float(
        ignition_cfg.get("minimumChemicalRateNumericalGuard_per_s", 0.0)
    )

    state = initial_state(geometry, config, composition, voltage, dtype)
    cp = float(config["thermal"]["heatCapacity_J_per_kgK"])
    thermal_conductivity = float(
        config["thermal"]["thermalConductivity_W_per_mK"]
    )
    alpha_thermal = thermal_conductivity / (density * cp)
    solve_every = max(
        1,
        int(
            round(
                float(config["coupled"]["electricalUpdateInterval_s"]) / dt
            )
        ),
    )
    numerics_cfg = config.get("numerics", {})
    progress_enabled = bool(numerics_cfg.get("progressEnabled", False))
    progress_interval = max(
        1, int(numerics_cfg.get("progressIntervalSteps", max(steps // 20, 1)))
    )
    progress_start = wall_time.perf_counter()

    transport = transport_fields(state, geometry, config, composition)
    electrical, reaction, solver_diagnostics = solve_electrical_and_reaction(
        state,
        transport,
        geometry,
        config,
        composition,
        voltage,
        spacing,
        static=False,
    )
    state["potential"] = electrical["potential"]
    current_density_floor = physical_floor(
        config, "currentDensity_A_per_m2", 1e-12
    )
    current_floor = physical_floor(config, "current_A", 1e-15)
    concentration_floor = physical_floor(
        config, "concentration_mol_per_m3", 1e-12
    )
    values_mean_j = batch_masked_mean(electrical["Jmagnitude"], geometry.propellant)
    p99_j = batch_quantile_masked(
        electrical["Jmagnitude"], geometry.propellant, 0.99
    )
    current_congestion = p99_j / torch.clamp(
        values_mean_j, min=current_density_floor
    )

    history_names = [
        "maxTemperature_K",
        "meanTemperature_K",
        "current_A",
        "power_W",
        "energy_J",
        "meanChemicalProgress",
        "maxChemicalProgress",
        "areaAveragedUndecomposedFraction",
        "temperatureOnsetAreaFraction",
        "maximumChemicalRate_per_s",
        "meanGasConcentration_mol_per_m3",
        "meanLiquidFraction",
        "meanWaterFraction",
        "meanIonicCurrentFraction",
        "waterReactionFraction",
        "lpReactionFraction",
        "limitingCurrentUtilization",
        "currentCongestion",
        "anodeCathodeCurrentMismatch",
        "speciesLimiterFraction",
        "speciesSubsteps",
        "temperatureCapFraction",
        "gasCapFraction",
        "chemicalRateCapFraction",
        "meanPassivation",
        "meanGasCoverage",
        "maximumNernstShift_V",
        "meanInterfacialActiveAreaFraction",
        "maximumLocalRobinResidual_A_per_m2",
        "maximumLocalRobinRelativeResidual",
        "maximumLocalRobinCombinedResidual",
        "maximumLocalRobinRoundoffFloor_A_per_m2",
        "roundoffLimitedLocalRobinFaceCount",
        "maximumLocalRobinIterations",
        "unresolvedLocalRobinFaceCount",
        "localRobinAllFacesConvergedFraction",
    ]
    histories = {
        name: torch.zeros(
            (steps, batch), device=geometry.fixed.device, dtype=dtype
        )
        for name in history_names
    }
    ignition_delay = torch.full(
        (batch,), torch.nan, device=geometry.fixed.device, dtype=dtype
    )

    snapshot_times = [
        float(v) for v in config["coupled"].get("snapshotTimes_s", [])
    ]
    snapshot_indices = {
        min(max(int(round(value / dt)) - 1, 0), steps - 1): value
        for value in snapshot_times
        if value <= end_time + 1e-12
    }
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    prop = geometry.propellant
    prop_count = torch.clamp(prop.sum(dim=(-2, -1)).to(dtype), min=1.0)
    last_solver_diagnostics = solver_diagnostics

    for step in range(steps):
        current_time = (step + 1) * dt

        liquid_equilibrium = 1.0 / (
            1.0
            + torch.exp(
                -(
                    state["temperature"]
                    - composition.effective_softening_temperature_K
                )
                / float(config["phase"]["transitionWidth_K"])
            )
        )
        liquid_rate = (
            liquid_equilibrium - state["liquidFraction"]
        ) / max(float(config["phase"]["relaxationTime_s"]), 1e-30)

        transport = transport_fields(state, geometry, config, composition)
        # The t=0 electrical state was solved immediately before the loop.
        # Re-solving it at step 0 is identical and doubles startup cost.
        if step > 0 and step % solve_every == 0:
            electrical, reaction, last_solver_diagnostics = (
                solve_electrical_and_reaction(
                    state,
                    transport,
                    geometry,
                    config,
                    composition,
                    voltage,
                    spacing,
                    static=False,
                )
            )
            state["potential"] = electrical["potential"]
            # J is unchanged between electrical solves.  Computing the masked
            # p99 only here avoids 8,000 dynamic quantile/sort calls per case.
            values_mean_j = batch_masked_mean(
                electrical["Jmagnitude"], prop
            )
            p99_j = batch_quantile_masked(
                electrical["Jmagnitude"], prop, 0.99
            )
            current_congestion = p99_j / torch.clamp(
                values_mean_j, min=current_density_floor
            )

        if bool(config["chemical"]["enabled"]):
            oxidizer = torch.minimum(state["cation"], state["anion"]) / max(
                composition.initial_lp_mol_per_m3, concentration_floor
            )
            temperature_gate = 1.0 / (
                1.0
                + torch.exp(
                    -(
                        state["temperature"]
                        - float(config["chemical"]["activationTemperature_K"])
                    )
                    / 20.0
                )
            )
            chemical_rate = (
                float(config["chemical"]["preExponentialFactor_per_s"])
                * torch.exp(
                    -float(config["chemical"]["activationEnergy_J_per_mol"])
                    / (
                        float(config["transport"]["gasConstant_J_per_molK"])
                        * torch.clamp(state["temperature"], min=1.0)
                    )
                )
                * torch.clamp(1.0 - state["chemicalProgress"], min=0.0)
                ** float(config["chemical"]["reactionOrder"])
                * torch.clamp(oxidizer, min=0.0)
                ** float(config["chemical"]["oxidizerOrder"])
                * temperature_gate
            )
            chemical_rate = torch.clamp(
                chemical_rate,
                min=0.0,
                max=float(config["chemical"]["maximumRate_per_s"]),
            )
            chemical_rate = torch.where(
                geometry.fixed,
                torch.zeros_like(chemical_rate),
                chemical_rate,
            )
        else:
            chemical_rate = torch.zeros_like(state["temperature"])

        species_diagnostics = update_species(
            state,
            transport,
            reaction,
            chemical_rate,
            geometry,
            config,
            composition,
            spacing,
            dt,
        )

        chemical_gas_source = (
            density
            * float(config["chemical"]["gasYieldMassFraction"])
            * chemical_rate
            / max(
                float(config["chemical"]["effectiveGasMolarMass_kg_per_mol"]),
                1e-30,
            )
        )
        blocking_diagnostics = update_interfacial_blocking(
            state, reaction, geometry, config, dt
        )
        gas_rate = (
            float(config["gas"]["diffusivity_m2_per_s"])
            * laplacian_neumann(state["gasConcentration"], spacing)
            + reaction["gasSource_mol_per_m3_s"]
            + chemical_gas_source
            - float(config["gas"]["lossRate_per_s"])
            * state["gasConcentration"]
        )

        chemical_heat = (
            density
            * float(config["chemical"]["heatRelease_J_per_kg"])
            * chemical_rate
        )
        gas_coverage_insulation = float(
            config["interface"]
            .get("blocking", {})
            .get("gasCoverage", {})
            .get("thermalInsulationFraction", 0.0)
        )
        convection_factor = torch.clamp(
            1.0 - gas_coverage_insulation * state["gasCoverage"],
            min=0.05,
            max=1.0,
        )
        heat_loss = (
            convection_factor
            * float(config["thermal"]["convectionCoefficient_W_per_m2K"])
            / float(config["geometry"]["surfaceLayerThickness_m"])
            * (
                state["temperature"]
                - float(config["thermal"]["ambientTemperature_K"])
            )
            + float(config["thermal"]["emissivity"])
            * float(config["thermal"]["stefanBoltzmann_W_per_m2K4"])
            / float(config["geometry"]["surfaceLayerThickness_m"])
            * (
                state["temperature"] ** 4
                - float(config["thermal"]["ambientTemperature_K"]) ** 4
            )
        )
        latent_term = (
            density * float(config["phase"]["latentHeat_J_per_kg"]) * liquid_rate
        )
        temperature_rate = (
            alpha_thermal * laplacian_neumann(state["temperature"], spacing)
            + (
                electrical["jouleHeat_W_per_m3"]
                + reaction["qTotal_W_per_m3"]
                + chemical_heat
                - heat_loss
                - latent_term
            )
            / (density * cp)
        )

        raw_temperature = state["temperature"] + dt * temperature_rate
        raw_gas = state["gasConcentration"] + dt * gas_rate
        temperature_min = float(config["thermal"]["minimumTemperature_K"])
        temperature_max = float(config["thermal"]["maximumTemperature_K"])
        gas_maximum = float(config["gas"]["maximumConcentration_mol_per_m3"])
        temperature_cap_mask = (
            (raw_temperature < temperature_min)
            | (raw_temperature > temperature_max)
        ) & prop
        gas_cap_mask = ((raw_gas < 0.0) | (raw_gas > gas_maximum)) & prop

        state["temperature"] = torch.clamp(
            raw_temperature, min=temperature_min, max=temperature_max
        )
        state["liquidFraction"] = torch.clamp(
            state["liquidFraction"] + dt * liquid_rate, 0.0, 1.0
        )
        state["chemicalProgress"] = torch.clamp(
            state["chemicalProgress"] + dt * chemical_rate, 0.0, 1.0
        )
        state["gasConcentration"] = torch.clamp(
            raw_gas, 0.0, gas_maximum
        )
        for name in [
            "liquidFraction",
            "chemicalProgress",
            "gasConcentration",
            "passivation",
            "gasCoverage",
        ]:
            state[name] = torch.where(
                geometry.fixed, torch.zeros_like(state[name]), state[name]
            )

        mean_progress = batch_masked_mean(state["chemicalProgress"], prop)
        max_progress = batch_masked_max(state["chemicalProgress"], prop)
        undecomposed_fraction = batch_masked_mean(
            1.0 - state["chemicalProgress"], prop
        )
        onset_area = (
            (state["temperature"] >= onset_temperature_K) & prop
        ).sum(dim=(-2, -1)).to(dtype) / prop_count

        histories["maxTemperature_K"][step] = batch_masked_max(
            state["temperature"], prop
        )
        histories["meanTemperature_K"][step] = batch_masked_mean(
            state["temperature"], prop
        )
        histories["current_A"][step] = reaction["totalCurrent_A"]
        histories["power_W"][step] = reaction["totalCurrent_A"] * (
            voltage - float(config["electrical"]["cathodeVoltage_V"])
        )
        if step == 0:
            histories["energy_J"][step] = histories["power_W"][step] * dt
        else:
            histories["energy_J"][step] = histories["energy_J"][step - 1] + 0.5 * dt * (
                histories["power_W"][step]
                + histories["power_W"][step - 1]
            )
        histories["meanChemicalProgress"][step] = mean_progress
        histories["maxChemicalProgress"][step] = max_progress
        histories["areaAveragedUndecomposedFraction"][step] = (
            undecomposed_fraction
        )
        histories["temperatureOnsetAreaFraction"][step] = onset_area
        histories["maximumChemicalRate_per_s"][step] = batch_masked_max(
            chemical_rate, prop
        )
        histories["meanGasConcentration_mol_per_m3"][step] = batch_masked_mean(
            state["gasConcentration"], prop
        )
        histories["meanLiquidFraction"][step] = batch_masked_mean(
            state["liquidFraction"], prop
        )
        histories["meanWaterFraction"][step] = batch_masked_mean(
            state["water"], prop
        ) / max(composition.initial_water_mol_per_m3, concentration_floor)
        histories["meanIonicCurrentFraction"][step] = batch_masked_mean(
            electrical["ionicCurrentFraction"], prop
        )
        histories["waterReactionFraction"][step] = reaction[
            "waterReactionFraction"
        ]
        histories["lpReactionFraction"][step] = reaction["lpReactionFraction"]
        histories["limitingCurrentUtilization"][step] = reaction[
            "limitingCurrentUtilization"
        ]
        histories["currentCongestion"][step] = current_congestion
        histories["anodeCathodeCurrentMismatch"][step] = reaction[
            "currentBalanceMismatchBeforeCoupling"
        ]
        histories["speciesLimiterFraction"][step] = species_diagnostics[
            "combinedLimiterFraction"
        ]
        histories["speciesSubsteps"][step] = species_diagnostics[
            "speciesSubsteps"
        ]
        histories["temperatureCapFraction"][step] = temperature_cap_mask.sum(
            dim=(-2, -1)
        ).to(dtype) / prop_count
        histories["gasCapFraction"][step] = gas_cap_mask.sum(
            dim=(-2, -1)
        ).to(dtype) / prop_count
        histories["chemicalRateCapFraction"][step] = (
            (
                chemical_rate
                >= float(config["chemical"]["maximumRate_per_s"])
                * (1.0 - 1e-12)
            )
            & prop
        ).sum(dim=(-2, -1)).to(dtype) / prop_count
        histories["meanPassivation"][step] = batch_masked_mean(
            state["passivation"], prop
        )
        histories["meanGasCoverage"][step] = batch_masked_mean(
            state["gasCoverage"], prop
        )
        histories["maximumNernstShift_V"][step] = reaction.get(
            "maximumNernstShift_V",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        )
        histories["meanInterfacialActiveAreaFraction"][step] = (
            batch_masked_mean(blocking_diagnostics["activeAreaFraction"], prop)
        )
        histories["maximumLocalRobinResidual_A_per_m2"][step] = reaction.get(
            "maximumLocalRobinResidual_A_per_m2",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        )
        histories["maximumLocalRobinRelativeResidual"][step] = reaction.get(
            "maximumLocalRobinRelativeResidual",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        )
        histories["maximumLocalRobinCombinedResidual"][step] = reaction.get(
            "maximumLocalRobinCombinedResidual",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        )
        histories["maximumLocalRobinRoundoffFloor_A_per_m2"][step] = reaction.get(
            "maximumLocalRobinRoundoffFloor_A_per_m2",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        )
        histories["roundoffLimitedLocalRobinFaceCount"][step] = reaction.get(
            "roundoffLimitedLocalRobinFaceCount",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        )
        histories["maximumLocalRobinIterations"][step] = reaction.get(
            "maximumLocalRobinIterations",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        )
        histories["unresolvedLocalRobinFaceCount"][step] = reaction.get(
            "unresolvedLocalRobinFaceCount",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        )
        histories["localRobinAllFacesConvergedFraction"][step] = reaction.get(
            "localRobinAllFacesConverged",
            torch.ones(
                batch, device=geometry.fixed.device, dtype=torch.bool
            ),
        ).to(dtype)

        # Operational condensed-phase onset.  The temperature criterion is
        # literature-traceable; the optional progress/rate guards default to
        # zero and exist only to suppress numerical-noise triggers.
        ignition_cells = (
            (state["temperature"] >= onset_temperature_K)
            & (state["chemicalProgress"] >= minimum_progress)
            & (chemical_rate >= minimum_rate)
            & prop
        )
        ignition_now = torch.isnan(ignition_delay) & ignition_cells.any(
            dim=(-2, -1)
        )
        ignition_delay = torch.where(
            ignition_now,
            torch.full_like(ignition_delay, current_time),
            ignition_delay,
        )

        if save_fields and step in snapshot_indices:
            snapshots[step] = {
                "time_s": torch.full(
                    (batch,),
                    current_time,
                    device=geometry.fixed.device,
                    dtype=dtype,
                ),
                "temperature": state["temperature"].clone(),
                "liquidFraction": state["liquidFraction"].clone(),
                "chemicalProgress": state["chemicalProgress"].clone(),
                "undecomposedFraction": (
                    1.0 - state["chemicalProgress"]
                ).clone(),
                "cation": state["cation"].clone(),
                "anion": state["anion"].clone(),
                "water": state["water"].clone(),
                "gasConcentration": state["gasConcentration"].clone(),
                "potential": state["potential"].clone(),
                "currentDensity": electrical["Jmagnitude"].clone(),
                "jouleHeat": electrical["jouleHeat_W_per_m3"].clone(),
                "electrochemicalHeat": reaction["qTotal_W_per_m3"].clone(),
                "chemicalHeat": chemical_heat.clone(),
                "passivation": state["passivation"].clone(),
                "gasCoverage": state["gasCoverage"].clone(),
            }

        if progress_enabled and (
            step == 0
            or (step + 1) % progress_interval == 0
            or step + 1 == steps
        ):
            elapsed = wall_time.perf_counter() - progress_start
            print(
                f"[condensed] batch={batch} step={step + 1}/{steps} "
                f"t={current_time:.6g}s elapsed={elapsed:.1f}s",
                flush=True,
            )

    current_end = histories["current_A"][-1]
    undecomposed_at_eval = _history_at_time(
        time,
        histories["areaAveragedUndecomposedFraction"],
        evaluation_time_s,
    )
    progress_at_eval = _history_at_time(
        time, histories["meanChemicalProgress"], evaluation_time_s
    )
    energy_at_eval = _history_at_time(
        time, histories["energy_J"], evaluation_time_s
    )
    # Ignition is registered on the discrete timestep at which the local
    # condensed-phase onset criterion is first satisfied.  Energy-to-ignition
    # therefore uses the cumulative trapezoidal electrical energy at that same
    # timestep.  Non-igniting cases remain NaN here and are handled as censored
    # infeasible cases by the NSGA-II canonicalisation layer.
    energy_to_ignition = torch.full_like(ignition_delay, torch.nan)
    ignited_mask = torch.isfinite(ignition_delay)
    if bool(ignited_mask.any()):
        ignition_index = torch.clamp(
            torch.round(ignition_delay / dt).to(torch.long) - 1,
            min=0,
            max=steps - 1,
        )
        batch_index = torch.arange(batch, device=geometry.fixed.device)
        energy_to_ignition[ignited_mask] = histories["energy_J"][
            ignition_index[ignited_mask], batch_index[ignited_mask]
        ]
    onset_area_at_eval = _history_at_time(
        time, histories["temperatureOnsetAreaFraction"], evaluation_time_s
    )
    congestion_at_eval = _history_at_time(
        time, histories["currentCongestion"], evaluation_time_s
    )
    evaluation_prefix_end = int(
        torch.searchsorted(
            time,
            torch.as_tensor(
                evaluation_time_s, device=time.device, dtype=time.dtype
            ),
            right=False,
        ).item()
    )
    congestion_to_evaluation = torch.cat(
        (
            histories["currentCongestion"][:evaluation_prefix_end],
            congestion_at_eval.unsqueeze(0),
        ),
        dim=0,
    ).amax(dim=0)
    ignition_succeeded = torch.isfinite(ignition_delay)

    result = {
        "time_s": time,
        "ignitionDelay_s": ignition_delay,
        "condensedPhaseIgnitionDelay_s": ignition_delay,
        "ignitionSucceeded": ignition_succeeded,
        "areaAveragedUndecomposedFractionAt2s": undecomposed_at_eval,
        "areaAveragedUndecomposedFractionAtEvaluationTime": undecomposed_at_eval,
        "meanChemicalProgressAt2s": progress_at_eval,
        "meanChemicalProgressAtEvaluationTime": progress_at_eval,
        "temperatureOnsetAreaFractionAt2s": onset_area_at_eval,
        "inputElectricalEnergyToIgnition_J": energy_to_ignition,
        "inputElectricalEnergyAt2s_J": energy_at_eval,
        "inputElectricalEnergyAtEvaluationTime_J": energy_at_eval,
        # Backward-compatible alias remains evaluation-horizon energy; the
        # Retained as a diagnostic; v7.9.4 NSGA-II uses minimum ignition voltage as objective 3.
        # evaluation time, not energy to a flame-establishment event.
        "inputElectricalEnergy_J": energy_at_eval,
        "peakMaximumTemperature_K": histories["maxTemperature_K"].amax(dim=0),
        "peakCurrent_A": histories["current_A"].amax(dim=0),
        "peakCurrentCongestion": histories["currentCongestion"].amax(dim=0),
        "peakCurrentCongestionToEvaluationTime": congestion_to_evaluation,
        "peakCurrentCongestionTo2s": congestion_to_evaluation,
        "finalEffectiveResistance_ohm": (
            voltage - float(config["electrical"]["cathodeVoltage_V"])
        )
        / torch.clamp(current_end, min=current_floor),
        "meanWaterReactionFraction": histories["waterReactionFraction"].mean(
            dim=0
        ),
        "meanLPReactionFraction": histories["lpReactionFraction"].mean(dim=0),
        "meanIonicCurrentFraction": histories["meanIonicCurrentFraction"].mean(
            dim=0
        ),
        "meanLimitingCurrentUtilization": histories[
            "limitingCurrentUtilization"
        ].mean(dim=0),
        "meanAnodeCathodeCurrentMismatch": histories[
            "anodeCathodeCurrentMismatch"
        ].mean(dim=0),
        "finalAnodeCathodeCurrentMismatch": histories[
            "anodeCathodeCurrentMismatch"
        ][-1],
        "maximumAnodeCathodeCurrentMismatch": histories[
            "anodeCathodeCurrentMismatch"
        ].amax(dim=0),
        "finalNonlinearRobinOuterIterations": reaction.get(
            "nonlinearRobinIterations",
            torch.zeros(batch, device=geometry.fixed.device, dtype=torch.int64),
        ),
        "finalNonlinearRobinConverged": reaction.get(
            "nonlinearRobinConverged",
            torch.ones(batch, device=geometry.fixed.device, dtype=torch.bool),
        ),
        "finalNonlinearRobinGaugeConverged": reaction.get(
            "nonlinearRobinGaugeConverged",
            torch.ones(batch, device=geometry.fixed.device, dtype=torch.bool),
        ),
        "finalNonlinearRobinCurrentBalanceCombinedResidual": reaction.get(
            "nonlinearRobinCurrentBalanceCombinedResidual",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        ),
        "finalNonlinearRobinGaugeIterations": reaction.get(
            "nonlinearRobinGaugeIterations",
            torch.zeros(batch, device=geometry.fixed.device, dtype=torch.int64),
        ),
        "finalNonlinearRobinGaugeTotalIterations": reaction.get(
            "nonlinearRobinGaugeTotalIterations",
            torch.zeros(batch, device=geometry.fixed.device, dtype=torch.int64),
        ),
        "finalNonlinearRobinGaugeOffset_V": reaction.get(
            "nonlinearRobinGaugeOffset_V",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        ),
        "finalNonlinearRobinGaugeBracketWidth_V": reaction.get(
            "nonlinearRobinGaugeBracketWidth_V",
            torch.zeros(batch, device=geometry.fixed.device, dtype=dtype),
        ),
        "finalNonlinearRobinAnodeCurrent_A": reaction.get(
            "nonlinearRobinAnodeCurrent_A", reaction["rawAnodeCurrent_A"]
        ),
        "finalNonlinearRobinCathodeCurrent_A": reaction.get(
            "nonlinearRobinCathodeCurrent_A", reaction["rawCathodeCurrent_A"]
        ),
        "finalNonlinearRobinAbsoluteCurrentDifference_A": reaction.get(
            "nonlinearRobinAbsoluteCurrentDifference_A",
            torch.abs(reaction["rawAnodeCurrent_A"] - reaction["rawCathodeCurrent_A"]),
        ),
        "finalWaterFraction": histories["meanWaterFraction"][-1],
        "finalLiquidFraction": histories["meanLiquidFraction"][-1],
        "finalMeanChemicalProgress": histories["meanChemicalProgress"][-1],
        "finalAreaAveragedUndecomposedFraction": histories[
            "areaAveragedUndecomposedFraction"
        ][-1],
        "maximumSpeciesLimiterFraction": histories[
            "speciesLimiterFraction"
        ].amax(dim=0),
        "meanSpeciesSubsteps": histories["speciesSubsteps"].mean(dim=0),
        "maximumSpeciesSubsteps": histories["speciesSubsteps"].amax(dim=0),
        "maximumTemperatureCapFraction": histories[
            "temperatureCapFraction"
        ].amax(dim=0),
        "maximumGasCapFraction": histories["gasCapFraction"].amax(dim=0),
        "maximumChemicalRateCapFraction": histories[
            "chemicalRateCapFraction"
        ].amax(dim=0),
        "finalMeanPassivation": histories["meanPassivation"][-1],
        "peakMeanPassivation": histories["meanPassivation"].amax(dim=0),
        "finalMeanGasCoverage": histories["meanGasCoverage"][-1],
        "peakMeanGasCoverage": histories["meanGasCoverage"].amax(dim=0),
        "maximumNernstShift_V": histories["maximumNernstShift_V"].amax(dim=0),
        "meanInterfacialActiveAreaFraction": histories[
            "meanInterfacialActiveAreaFraction"
        ].mean(dim=0),
        "finalMeanInterfacialActiveAreaFraction": histories[
            "meanInterfacialActiveAreaFraction"
        ][-1],
        "minimumMeanInterfacialActiveAreaFraction": histories[
            "meanInterfacialActiveAreaFraction"
        ].amin(dim=0),
        "maximumLocalRobinResidual_A_per_m2": histories[
            "maximumLocalRobinResidual_A_per_m2"
        ].amax(dim=0),
        "maximumLocalRobinRelativeResidual": histories[
            "maximumLocalRobinRelativeResidual"
        ].amax(dim=0),
        "maximumLocalRobinCombinedResidual": histories[
            "maximumLocalRobinCombinedResidual"
        ].amax(dim=0),
        "maximumLocalRobinRoundoffFloor_A_per_m2": histories[
            "maximumLocalRobinRoundoffFloor_A_per_m2"
        ].amax(dim=0),
        "maximumRoundoffLimitedLocalRobinFaceCount": histories[
            "roundoffLimitedLocalRobinFaceCount"
        ].amax(dim=0),
        "maximumLocalRobinIterations": histories[
            "maximumLocalRobinIterations"
        ].amax(dim=0),
        "maximumUnresolvedLocalRobinFaceCount": histories[
            "unresolvedLocalRobinFaceCount"
        ].amax(dim=0),
        "minimumLocalRobinAllFacesConvergedFraction": histories[
            "localRobinAllFacesConvergedFraction"
        ].amin(dim=0),
        "histories": histories,
        "finalState": state,
        "finalElectrical": electrical,
        "finalReaction": reaction,
        "snapshots": snapshots,
        "solverDiagnostics": last_solver_diagnostics,
        "stabilityDiagnostics": stability,
        "evaluationTime_s": evaluation_time_s,
        "ignitionCriterion": {
            "type": "first_local_literature_decomposition_onset_temperature",
            "decompositionOnsetTemperature_K": onset_temperature_K,
            "minimumChemicalProgressNumericalGuard": minimum_progress,
            "minimumChemicalRateNumericalGuard_per_s": minimum_rate,
            "interpretation": "condensed_phase_onset_not_gas_flame",
        },
        "modelStatus": "v7_7_5_condensed_phase_no_F_gap_robin_spd_pcg_nominal_unvalidated",
        "postIgnitionClosure": "none_F_removed",
        "externalCfdFeedbackApplied": False,
    }
    return result
