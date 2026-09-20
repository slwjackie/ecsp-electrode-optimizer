from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .composition_model import CompositionModel
from .errors import BCCandidateBatchError
from .geometry import GeometryBatch
from .numerics import batch_masked_mean, harmonic_mean, matlab_gradient, physical_floor
from .potential import solve_potential


def _batch_voltage_vector(
    voltage: float | Sequence[float] | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Normalize scalar/per-candidate voltage without a device-host transfer."""
    value = torch.as_tensor(
        voltage, device=reference.device, dtype=reference.dtype
    )
    batch_size = reference.shape[0]
    if value.ndim == 0:
        return value.expand(batch_size)
    if value.ndim == 1 and value.numel() == batch_size:
        return value
    if value.shape == (batch_size, 1, 1):
        return value[:, 0, 0]
    raise ValueError("Voltage must be scalar or contain one value per batch lane")


def initial_state(
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float | Sequence[float] | torch.Tensor,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    shape = geometry.fixed.shape
    device = geometry.fixed.device
    t0 = float(config["thermal"]["initialTemperature_K"])
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])
    voltage_vector = _batch_voltage_vector(
        voltage,
        torch.empty(shape, device=device, dtype=dtype),
    )
    voltage_field = voltage_vector[:, None, None].expand(shape)
    state = {
        "temperature": torch.full(shape, t0, device=device, dtype=dtype),
        "cation": torch.full(shape, composition.initial_cation_mol_per_m3, device=device, dtype=dtype),
        "anion": torch.full(shape, composition.initial_anion_mol_per_m3, device=device, dtype=dtype),
        "water": torch.full(shape, composition.initial_water_mol_per_m3, device=device, dtype=dtype),
        "liquidFraction": torch.zeros(shape, device=device, dtype=dtype),
        "chemicalProgress": torch.zeros(shape, device=device, dtype=dtype),
        "gasConcentration": torch.zeros(shape, device=device, dtype=dtype),
        "passivation": torch.zeros(shape, device=device, dtype=dtype),
        "gasCoverage": torch.zeros(shape, device=device, dtype=dtype),
        "potential": 0.5 * (voltage_field + cathode_voltage),
    }
    boundary_model = str(
        config.get("interface", {}).get("boundaryCouplingModel", "legacy_posthoc")
    ).lower()
    if boundary_model not in {"surface_overlay_bv", "surface_contact", "surface_bv"}:
        state["potential"] = torch.where(geometry.anode, voltage_field, state["potential"])
        state["potential"] = torch.where(geometry.cathode, torch.as_tensor(cathode_voltage, device=device, dtype=dtype), state["potential"])
    for name in ["cation", "anion", "water", "liquidFraction", "chemicalProgress", "gasConcentration", "passivation", "gasCoverage"]:
        state[name] = torch.where(geometry.fixed, torch.zeros_like(state[name]), state[name])
    return state


def transport_fields(
    state: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
) -> dict[str, torch.Tensor]:
    cfg = config["transport"]
    phase = config["phase"]
    electrical = config["electrical"]
    gas_constant = float(cfg["gasConstant_J_per_molK"])
    faraday = float(cfg["faradayConstant_C_per_mol"])
    tref = float(cfg["referenceTemperature_K"])
    temperature = torch.clamp(state["temperature"], min=1.0)
    concentration_floor = physical_floor(config, "concentration_mol_per_m3", 1e-12)
    water_activity = torch.clamp(
        state["water"] / max(composition.initial_water_mol_per_m3, concentration_floor),
        0.0,
        1.0,
    )
    water_factor = water_activity ** float(cfg["waterActivityExponent"])
    dplus = float(cfg["cationDiffusivityDry_m2_per_s"]) + (
        float(cfg["cationDiffusivityWet_m2_per_s"]) - float(cfg["cationDiffusivityDry_m2_per_s"])
    ) * water_factor
    dminus = float(cfg["anionDiffusivityDry_m2_per_s"]) + (
        float(cfg["anionDiffusivityWet_m2_per_s"]) - float(cfg["anionDiffusivityDry_m2_per_s"])
    ) * water_factor
    plus_arrhenius = torch.exp(
        -float(cfg["cationTransportActivationEnergy_J_per_mol"]) / gas_constant * (1.0 / temperature - 1.0 / tref)
    )
    minus_arrhenius = torch.exp(
        -float(cfg["anionTransportActivationEnergy_J_per_mol"]) / gas_constant * (1.0 / temperature - 1.0 / tref)
    )
    threshold = float(phase["minimumLiquidFractionForFastIonTransport"])
    fast = torch.clamp((state["liquidFraction"] - threshold) / max(1.0 - threshold, 1e-30), 0.0, 1.0)
    phase_gain = 1.0 + float(cfg["liquidDiffusivityGain"]) * fast
    plasticizer = 1.0 + float(cfg["glycerolPlasticizationGain"]) * composition.glycerol_to_pva_mass_ratio
    crosslink = 1.0 / (
        1.0 + float(cfg["crosslinkTransportPenalty"]) * composition.boric_acid_to_pva_repeat_molar_ratio
    )
    dcation = dplus * plus_arrhenius * phase_gain * plasticizer * crosslink
    danion = dminus * minus_arrhenius * phase_gain * plasticizer * crosslink
    dwater = float(cfg["waterDiffusivity_m2_per_s"]) * torch.exp(
        -12000.0 / gas_constant * (1.0 / temperature - 1.0 / tref)
    ) * (1.0 + 4.0 * fast)
    z_plus = float(cfg["chargeNumberCation"])
    z_minus = float(cfg["chargeNumberAnion"])
    sigma_ionic = faraday * faraday / (gas_constant * temperature) * (
        z_plus * z_plus * dcation * torch.clamp(state["cation"], min=0.0)
        + z_minus * z_minus * danion * torch.clamp(state["anion"], min=0.0)
    )
    solid_path = torch.clamp(
        1.0 - float(electrical["electronicLiquidSuppression"]) * state["liquidFraction"], min=0.01
    )
    sigma_electronic = (
        float(electrical["electronicConductivity0_S_per_m"])
        * torch.exp(
            float(electrical["electronicConductivityTemperatureCoefficient_per_K"])
            * (temperature - float(electrical["initialTemperature_K"]))
        )
        * solid_path
    )
    sigma_total = torch.clamp(
        sigma_electronic + sigma_ionic,
        min=float(electrical["conductivityMinimum_S_per_m"]),
        max=float(electrical["conductivityMaximum_S_per_m"]),
    )
    dcation = torch.where(geometry.fixed, torch.zeros_like(dcation), dcation)
    danion = torch.where(geometry.fixed, torch.zeros_like(danion), danion)
    dwater = torch.where(geometry.fixed, torch.zeros_like(dwater), dwater)
    sigma_ionic = torch.where(
        geometry.fixed,
        torch.full_like(sigma_ionic, float(electrical["conductivityMinimum_S_per_m"])),
        sigma_ionic,
    )
    sigma_electronic = torch.where(
        geometry.fixed,
        torch.full_like(sigma_electronic, float(electrical["conductivityMaximum_S_per_m"])),
        sigma_electronic,
    )
    sigma_total = torch.where(
        geometry.fixed,
        torch.full_like(sigma_total, float(electrical["conductivityMaximum_S_per_m"])),
        sigma_total,
    )
    return {
        "Dcation": dcation,
        "Danion": danion,
        "Dwater": dwater,
        "sigmaIonic": sigma_ionic,
        "sigmaElectronic": sigma_electronic,
        "sigmaTotal": sigma_total,
        "waterActivity": water_activity,
        "fastIonFraction": fast,
    }



def _masked_gradient_propellant(
    field: torch.Tensor,
    propellant: torch.Tensor,
    spacing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell-centred gradient using only electrolyte neighbours.

    At metal interfaces the bulk gradient is taken one-sided from the
    propellant.  The unresolved electrochemical interfacial drop is handled by
    the nonlinear Robin condition rather than by differentiating through a
    metal cell held at the electrode potential.
    """
    east = torch.zeros_like(field); east[..., :, :-1] = field[..., :, 1:]
    west = torch.zeros_like(field); west[..., :, 1:] = field[..., :, :-1]
    north = torch.zeros_like(field); north[..., :-1, :] = field[..., 1:, :]
    south = torch.zeros_like(field); south[..., 1:, :] = field[..., :-1, :]
    pe = torch.zeros_like(propellant); pe[..., :, :-1] = propellant[..., :, 1:]
    pw = torch.zeros_like(propellant); pw[..., :, 1:] = propellant[..., :, :-1]
    pn = torch.zeros_like(propellant); pn[..., :-1, :] = propellant[..., 1:, :]
    ps = torch.zeros_like(propellant); ps[..., 1:, :] = propellant[..., :-1, :]
    both_x = pe & pw & propellant
    only_e = pe & ~pw & propellant
    only_w = pw & ~pe & propellant
    both_y = pn & ps & propellant
    only_n = pn & ~ps & propellant
    only_s = ps & ~pn & propellant
    gx = torch.zeros_like(field)
    gy = torch.zeros_like(field)
    gx = torch.where(both_x, (east - west) / (2.0 * spacing), gx)
    gx = torch.where(only_e, (east - field) / spacing, gx)
    gx = torch.where(only_w, (field - west) / spacing, gx)
    gy = torch.where(both_y, (north - south) / (2.0 * spacing), gy)
    gy = torch.where(only_n, (north - field) / spacing, gy)
    gy = torch.where(only_s, (field - south) / spacing, gy)
    return gx, gy


def _surface_finite_volume_power_density(
    potential: torch.Tensor,
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    spacing: float,
) -> dict[str, torch.Tensor]:
    """Distribute each in-plane electrical face power to its two control volumes.

    Cell-centred gradients are useful current/congestion diagnostics, but their
    squared norm does not preserve the power of the finite-volume network.  A
    face with field ``E_f`` dissipates ``sigma_f E_f**2`` per adjacent cell
    volume; assigning half of that density to each endpoint makes the volume
    integral exactly equal to the sum of all internal face powers.  The same
    face fluxes used by the potential equation are used for the optional
    diffusion-current diagnostics.
    """
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Electrical grid spacing must be finite and positive")

    propellant = geometry.propellant
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    z_plus = float(config["transport"]["chargeNumberCation"])
    z_minus = float(config["transport"]["chargeNumberAnion"])
    diffusion_enabled = bool(
        config["electrical"].get("diffusionPotentialEnabled", True)
    )
    conductive = torch.zeros_like(potential)
    total = torch.zeros_like(potential)
    diffusion = torch.zeros_like(potential)
    legacy_magnitude = torch.zeros_like(potential)

    valid_x = propellant[..., :, :-1] & propellant[..., :, 1:]
    sigma_x = harmonic_mean(
        transport["sigmaTotal"][..., :, :-1],
        transport["sigmaTotal"][..., :, 1:],
    )
    electric_x = -(
        potential[..., :, 1:] - potential[..., :, :-1]
    ) / spacing
    if diffusion_enabled:
        dplus_x = harmonic_mean(
            transport["Dcation"][..., :, :-1],
            transport["Dcation"][..., :, 1:],
        )
        dminus_x = harmonic_mean(
            transport["Danion"][..., :, :-1],
            transport["Danion"][..., :, 1:],
        )
        jdiff_x = -faraday * (
            z_plus
            * dplus_x
            * (state["cation"][..., :, 1:] - state["cation"][..., :, :-1])
            / spacing
            + z_minus
            * dminus_x
            * (state["anion"][..., :, 1:] - state["anion"][..., :, :-1])
            / spacing
        )
    else:
        jdiff_x = torch.zeros_like(electric_x)
    johm_x = sigma_x * electric_x
    conductive_x = torch.where(
        valid_x, johm_x * electric_x, torch.zeros_like(electric_x)
    )
    diffusion_x = torch.where(
        valid_x, jdiff_x * electric_x, torch.zeros_like(electric_x)
    )
    total_x = conductive_x + diffusion_x
    legacy_x = torch.where(
        valid_x,
        electric_x.abs() * (johm_x + jdiff_x).abs(),
        torch.zeros_like(electric_x),
    )
    conductive[..., :, :-1] += 0.5 * conductive_x
    conductive[..., :, 1:] += 0.5 * conductive_x
    diffusion[..., :, :-1] += 0.5 * diffusion_x
    diffusion[..., :, 1:] += 0.5 * diffusion_x
    total[..., :, :-1] += 0.5 * total_x
    total[..., :, 1:] += 0.5 * total_x
    legacy_magnitude[..., :, :-1] += 0.5 * legacy_x
    legacy_magnitude[..., :, 1:] += 0.5 * legacy_x

    valid_y = propellant[..., :-1, :] & propellant[..., 1:, :]
    sigma_y = harmonic_mean(
        transport["sigmaTotal"][..., :-1, :],
        transport["sigmaTotal"][..., 1:, :],
    )
    electric_y = -(
        potential[..., 1:, :] - potential[..., :-1, :]
    ) / spacing
    if diffusion_enabled:
        dplus_y = harmonic_mean(
            transport["Dcation"][..., :-1, :],
            transport["Dcation"][..., 1:, :],
        )
        dminus_y = harmonic_mean(
            transport["Danion"][..., :-1, :],
            transport["Danion"][..., 1:, :],
        )
        jdiff_y = -faraday * (
            z_plus
            * dplus_y
            * (state["cation"][..., 1:, :] - state["cation"][..., :-1, :])
            / spacing
            + z_minus
            * dminus_y
            * (state["anion"][..., 1:, :] - state["anion"][..., :-1, :])
            / spacing
        )
    else:
        jdiff_y = torch.zeros_like(electric_y)
    johm_y = sigma_y * electric_y
    conductive_y = torch.where(
        valid_y, johm_y * electric_y, torch.zeros_like(electric_y)
    )
    diffusion_y = torch.where(
        valid_y, jdiff_y * electric_y, torch.zeros_like(electric_y)
    )
    total_y = conductive_y + diffusion_y
    legacy_y = torch.where(
        valid_y,
        electric_y.abs() * (johm_y + jdiff_y).abs(),
        torch.zeros_like(electric_y),
    )
    conductive[..., :-1, :] += 0.5 * conductive_y
    conductive[..., 1:, :] += 0.5 * conductive_y
    diffusion[..., :-1, :] += 0.5 * diffusion_y
    diffusion[..., 1:, :] += 0.5 * diffusion_y
    total[..., :-1, :] += 0.5 * total_y
    total[..., 1:, :] += 0.5 * total_y
    legacy_magnitude[..., :-1, :] += 0.5 * legacy_y
    legacy_magnitude[..., 1:, :] += 0.5 * legacy_y

    zero = torch.zeros_like(potential)
    return {
        "conductive": torch.where(propellant, conductive, zero),
        "total": torch.where(propellant, total, zero),
        "diffusion": torch.where(propellant, diffusion, zero),
        "legacyMagnitude": torch.where(propellant, legacy_magnitude, zero),
    }

def compute_current(
    potential: torch.Tensor,
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    spacing: float,
) -> dict[str, torch.Tensor]:
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    z_plus = float(config["transport"]["chargeNumberCation"])
    z_minus = float(config["transport"]["chargeNumberAnion"])
    boundary_model = str(
        config.get("interface", {}).get("boundaryCouplingModel", "legacy_posthoc")
    ).lower()
    robin_like = {
        "nonlinear_robin", "robin", "bv_robin",
        "surface_overlay_bv", "surface_contact", "surface_bv",
    }
    if boundary_model in robin_like:
        dphi_dx, dphi_dy = _masked_gradient_propellant(
            potential, geometry.propellant, spacing
        )
    else:
        dphi_dx, dphi_dy = matlab_gradient(potential, spacing)
    ex, ey = -dphi_dx, -dphi_dy
    if bool(config["electrical"].get("diffusionPotentialEnabled", True)):
        if boundary_model in robin_like:
            dcplus_dx, dcplus_dy = _masked_gradient_propellant(
                state["cation"], geometry.propellant, spacing
            )
            dcminus_dx, dcminus_dy = _masked_gradient_propellant(
                state["anion"], geometry.propellant, spacing
            )
        else:
            dcplus_dx, dcplus_dy = matlab_gradient(state["cation"], spacing)
            dcminus_dx, dcminus_dy = matlab_gradient(state["anion"], spacing)
        jdiff_x = -faraday * (
            z_plus * transport["Dcation"] * dcplus_dx + z_minus * transport["Danion"] * dcminus_dx
        )
        jdiff_y = -faraday * (
            z_plus * transport["Dcation"] * dcplus_dy + z_minus * transport["Danion"] * dcminus_dy
        )
    else:
        jdiff_x = torch.zeros_like(potential)
        jdiff_y = torch.zeros_like(potential)
    johm_x = transport["sigmaTotal"] * ex
    johm_y = transport["sigmaTotal"] * ey
    jx, jy = johm_x + jdiff_x, johm_y + jdiff_y
    jmag = torch.sqrt(jx * jx + jy * jy)
    emag = torch.sqrt(ex * ex + ey * ey)

    # v7.3.1: keep irreversible conductive heating separate from the
    # diffusion-current contribution.  |E||J_total| incorrectly treats a
    # concentration-gradient current that is not parallel to E as Joule heat.
    # The default active heat source is J_cond·E = sigma |E|^2.  J_total·E and
    # J_diff·E are retained as diagnostics/sensitivity outputs.
    conductive_power = johm_x * ex + johm_y * ey
    total_electrical_power = jx * ex + jy * ey
    diffusion_electrical_power = jdiff_x * ex + jdiff_y * ey
    legacy_power = emag * jmag
    conservative_surface_heat = boundary_model in {
        "surface_overlay_bv", "surface_contact", "surface_bv"
    }
    if conservative_surface_heat:
        face_power = _surface_finite_volume_power_density(
            potential, state, transport, geometry, config, spacing
        )
        conductive_power = face_power["conductive"]
        total_electrical_power = face_power["total"]
        diffusion_electrical_power = face_power["diffusion"]
        legacy_power = face_power["legacyMagnitude"]
    joule_model = str(config["electrical"].get("jouleHeatModel", "conductive_sigma_E2")).lower()
    if joule_model in {"conductive_sigma_e2", "conductive", "sigma_e2", "jcond_dot_e"}:
        qj = torch.clamp(conductive_power, min=0.0)
    elif joule_model in {"total_j_dot_e", "j_dot_e", "total"}:
        # J_total·E is a signed electrical-energy transfer.  Clipping local
        # negative diffusion cross terms would violate both this model's name
        # and the finite-volume integral.  Callers requiring non-negative
        # irreversible heat must select conductive_sigma_E2 instead.
        qj = total_electrical_power
    elif joule_model in {"legacy_magnitude_e_times_j", "legacy_magnitude", "e_times_jmag"}:
        qj = torch.clamp(legacy_power, min=0.0)
    else:
        raise ValueError(f"Unknown electrical.jouleHeatModel: {joule_model}")

    ionic_mag = transport["sigmaIonic"] * emag
    electronic_mag = transport["sigmaElectronic"] * emag
    current_density_floor = physical_floor(
        config, "currentDensity_A_per_m2", 1e-12
    )
    ionic_fraction = ionic_mag / torch.clamp(
        ionic_mag + electronic_mag, min=current_density_floor
    )
    zero = torch.zeros_like(jmag)
    jmag = torch.where(geometry.fixed, zero, jmag)
    qj = torch.where(geometry.fixed, zero, qj)
    johm_x = torch.where(geometry.fixed, zero, johm_x)
    johm_y = torch.where(geometry.fixed, zero, johm_y)
    jdiff_x = torch.where(geometry.fixed, zero, jdiff_x)
    jdiff_y = torch.where(geometry.fixed, zero, jdiff_y)
    conductive_power = torch.where(geometry.fixed, zero, conductive_power)
    total_electrical_power = torch.where(geometry.fixed, zero, total_electrical_power)
    diffusion_electrical_power = torch.where(geometry.fixed, zero, diffusion_electrical_power)
    ionic_fraction = torch.where(geometry.fixed, zero, ionic_fraction)
    return {
        "potential": potential,
        "Ex": ex,
        "Ey": ey,
        "Emagnitude": emag,
        "Jx": jx,
        "Jy": jy,
        "Jmagnitude": jmag,
        "JohmX": johm_x,
        "JohmY": johm_y,
        "JdiffX": jdiff_x,
        "JdiffY": jdiff_y,
        "jouleHeat_W_per_m3": qj,
        "inPlaneJouleHeat_W_per_m3": qj.clone(),
        "jouleHeatConductive_W_per_m3": conductive_power,
        "electricalPowerDensityJdotE_W_per_m3": total_electrical_power,
        "diffusionElectricalPowerDensity_W_per_m3": diffusion_electrical_power,
        "jouleHeatModel": joule_model,
        "jouleHeatDiscretization": (
            "finite_volume_internal_faces_half_to_each_control_volume"
            if conservative_surface_heat
            else "legacy_cell_centred_gradient"
        ),
        "ionicCurrentFraction": ionic_fraction,
    }


def interface_normal_current(
    electrical: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    current_direction_model: str = "signed_normal",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return physically directed current available to each electrode reaction.

    The old implementation summed |Jx|/|Jy| at every adjacent face.  That can
    count a locally reversed flux as if it fed the intended Faradaic reaction.
    Here the face normal is defined from the propellant cell toward the metal.

    * anode: conventional current must leave the anode and enter propellant,
      therefore available flux is max(-J·n_prop_to_anode, 0).
    * cathode: conventional current must leave propellant and enter cathode,
      therefore available flux is max(+J·n_prop_to_cathode, 0).

    Cell-centred J is projected face-by-face; multiple adjacent faces (corners)
    contribute independently.
    """
    prop = geometry.propellant
    jx, jy = electrical["Jx"], electrical["Jy"]
    anode_e = torch.zeros_like(prop); anode_e[..., :, :-1] = geometry.anode[..., :, 1:]
    anode_w = torch.zeros_like(prop); anode_w[..., :, 1:] = geometry.anode[..., :, :-1]
    anode_n = torch.zeros_like(prop); anode_n[..., :-1, :] = geometry.anode[..., 1:, :]
    anode_s = torch.zeros_like(prop); anode_s[..., 1:, :] = geometry.anode[..., :-1, :]
    cathode_e = torch.zeros_like(prop); cathode_e[..., :, :-1] = geometry.cathode[..., :, 1:]
    cathode_w = torch.zeros_like(prop); cathode_w[..., :, 1:] = geometry.cathode[..., :, :-1]
    cathode_n = torch.zeros_like(prop); cathode_n[..., :-1, :] = geometry.cathode[..., 1:, :]
    cathode_s = torch.zeros_like(prop); cathode_s[..., 1:, :] = geometry.cathode[..., :-1, :]
    dtype = jx.dtype
    ae, aw, an, ass = (anode_e.to(dtype), anode_w.to(dtype), anode_n.to(dtype), anode_s.to(dtype))
    ce, cw, cn, cs = (cathode_e.to(dtype), cathode_w.to(dtype), cathode_n.to(dtype), cathode_s.to(dtype))
    anode_faces = ae + aw + an + ass
    cathode_faces = ce + cw + cn + cs

    model = str(current_direction_model).lower()
    if model in {"legacy_magnitude", "magnitude_legacy", "absolute_components"}:
        anode_available = torch.abs(jx) * (ae + aw) + torch.abs(jy) * (an + ass)
        cathode_available = torch.abs(jx) * (ce + cw) + torch.abs(jy) * (cn + cs)
        zero = torch.zeros_like(anode_available)
        return (
            torch.where(prop, anode_available, zero),
            torch.where(prop, cathode_available, zero),
            torch.where(prop, anode_faces, zero),
            torch.where(prop, cathode_faces, zero),
        )
    if model not in {"signed_normal", "directional", "signed_j_dot_n"}:
        raise ValueError(f"Unknown interface.currentDirectionModel: {current_direction_model}")

    # n_prop_to_electrode = (+x,-x,+y,-y) for (east,west,north,south).
    # Anode admissible direction is opposite that normal; cathode is along it.
    anode_available = (
        torch.clamp(-jx, min=0.0) * ae
        + torch.clamp(jx, min=0.0) * aw
        + torch.clamp(-jy, min=0.0) * an
        + torch.clamp(jy, min=0.0) * ass
    )
    cathode_available = (
        torch.clamp(jx, min=0.0) * ce
        + torch.clamp(-jx, min=0.0) * cw
        + torch.clamp(jy, min=0.0) * cn
        + torch.clamp(-jy, min=0.0) * cs
    )
    zero = torch.zeros_like(anode_available)
    return (
        torch.where(prop, anode_available, zero),
        torch.where(prop, cathode_available, zero),
        torch.where(prop, anode_faces, zero),
        torch.where(prop, cathode_faces, zero),
    )



def _activity_coefficient(
    salt_concentration: torch.Tensor,
    temperature: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Bounded non-ideal activity-coefficient proxy for the 1:1 LP electrolyte.

    This is a calibration closure for concentrated LP/PVA, not a claim that
    dilute Debye-Huckel theory is quantitatively valid in the propellant.
    """
    cfg = config["interface"].get("nernst", {})
    model = str(cfg.get("activityCoefficientModel", "unity")).lower()
    if model in {"unity", "ideal", "ideal_concentration"}:
        return torch.ones_like(salt_concentration)
    if model != "extended_debye_huckel_proxy":
        raise ValueError(f"Unknown Nernst activityCoefficientModel: {model}")
    ionic_strength = torch.clamp(salt_concentration / 1000.0, min=0.0)
    sqrt_i = torch.sqrt(ionic_strength)
    a = float(cfg.get("activityA", 0.51))
    b = float(cfg.get("activityB", 1.0))
    k = float(cfg.get("activityLinearPerMolL", 0.0))
    log10_gamma = -a * sqrt_i / torch.clamp(1.0 + b * sqrt_i, min=1e-12) + k * ionic_strength
    limit = abs(float(cfg.get("maximumLog10ActivityCoefficientMagnitude", 2.0)))
    log10_gamma = torch.clamp(log10_gamma, -limit, limit)
    return torch.pow(torch.full_like(log10_gamma, 10.0), log10_gamma)


def _equilibrium_potential(
    channel: dict,
    activity: torch.Tensor,
    temperature: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concentration/activity-dependent Nernst equilibrium-potential closure.

    E_eq = E0 + (R T / nF) * nu_Q * ln(a/a_ref).
    ``nu_Q`` is configurable because unresolved products are not explicit state
    variables in the reduced chemistry.
    """
    e0 = float(channel["equilibriumPotential_V"])
    nernst_cfg = config["interface"].get("nernst", {})
    if not bool(nernst_cfg.get("enabled", False)):
        base = torch.full_like(temperature, e0)
        return base, torch.zeros_like(base)
    gas_constant = float(config["transport"]["gasConstant_J_per_molK"])
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    electron_number = float(channel["electronNumber"])
    activity_floor = max(float(nernst_cfg.get("activityFloor", 1e-6)), 1e-30)
    reference_activity = max(float(channel.get("referenceActivity", 1.0)), activity_floor)
    quotient_exponent = float(channel.get("nernstReactionQuotientExponent", -1.0))
    ratio = torch.clamp(activity / reference_activity, min=activity_floor)
    shift = gas_constant * temperature / (electron_number * faraday) * quotient_exponent * torch.log(ratio)
    maximum_shift = abs(float(nernst_cfg.get("maximumAbsoluteShift_V", 0.3)))
    shift = torch.clamp(shift, -maximum_shift, maximum_shift)
    return e0 + shift, shift


def update_interfacial_blocking(
    state: dict[str, torch.Tensor],
    reaction: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Advance reduced passivation and gas/bubble-coverage state variables."""
    blocking = config["interface"].get("blocking", {})
    pass_cfg = blocking.get("passivation", {})
    gas_cfg = blocking.get("gasCoverage", {})
    temperature = torch.clamp(state["temperature"], min=1.0)
    interfacial_zone = reaction["anodeZone"] | reaction["cathodeZone"]
    total_j = reaction["jAnode_A_per_m2"] + reaction["jCathode_A_per_m2"]
    if bool(pass_cfg.get("enabled", False)):
        j_ref = max(float(pass_cfg.get("referenceCurrentDensity_A_per_m2", 1000.0)), 1e-30)
        exponent = float(pass_cfg.get("currentExponent", 1.0))
        t0 = float(pass_cfg.get("formationActivationTemperature_K", 350.0))
        tw = max(float(pass_cfg.get("formationActivationWidth_K", 25.0)), 1e-12)
        temp_gate = torch.sigmoid((temperature - t0) / tw)
        formation = (
            float(pass_cfg.get("formationRate_per_s", 0.0))
            * torch.clamp(total_j / j_ref, min=0.0) ** exponent
            * temp_gate
            * (1.0 - state["passivation"])
        )
        removal = float(pass_cfg.get("removalRate_per_s", 0.0)) * state["passivation"]
        passivation = torch.clamp(state["passivation"] + dt * (formation - removal), 0.0, 1.0)
        state["passivation"] = torch.where(
            interfacial_zone & geometry.propellant, passivation, torch.zeros_like(passivation)
        )
    if bool(gas_cfg.get("enabled", False)):
        source_ref = max(float(gas_cfg.get("referenceGasSource_mol_per_m3_s", 100.0)), 1e-30)
        source = torch.clamp(reaction["gasSource_mol_per_m3_s"] / source_ref, min=0.0)
        formation = (
            float(gas_cfg.get("formationRate_per_s", 0.0))
            * source
            * (1.0 - state["gasCoverage"])
        )
        detach = float(gas_cfg.get("detachmentRate_per_s", 0.0)) * state["gasCoverage"]
        gas_coverage = torch.clamp(state["gasCoverage"] + dt * (formation - detach), 0.0, 1.0)
        state["gasCoverage"] = torch.where(
            interfacial_zone & geometry.propellant, gas_coverage, torch.zeros_like(gas_coverage)
        )
    active_fraction = torch.clamp(
        (1.0 - state["passivation"]) * (1.0 - state["gasCoverage"]),
        min=float(blocking.get("minimumActiveAreaFraction", 0.02)),
        max=1.0,
    )
    return {
        "activeAreaFraction": active_fraction,
        "passivation": state["passivation"],
        "gasCoverage": state["gasCoverage"],
    }

def _reaction_candidate(
    concentration_factor: torch.Tensor,
    concentration: torch.Tensor,
    diffusivity: torch.Tensor,
    overpotential: torch.Tensor,
    temperature: torch.Tensor,
    channel: dict,
    config: dict,
    kinetic_multiplier: torch.Tensor,
) -> torch.Tensor:
    interface = config["interface"]
    coupled = config["coupled"]
    gas_constant = float(config["transport"]["gasConstant_J_per_molK"])
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    alpha = float(channel["chargeTransferCoefficient"])
    electron_number = float(channel["electronNumber"])
    forward = (
        alpha * electron_number * faraday * overpotential
        / (gas_constant * temperature)
    )
    reverse = (
        -(1.0 - alpha) * electron_number * faraday * overpotential
        / (gas_constant * temperature)
    )
    limit = float(interface["exponentialArgumentLimit"])
    forward = torch.clamp(forward, -limit, limit)
    reverse = torch.clamp(reverse, -limit, limit)
    reverse_factor = float(channel.get("reverseAvailabilityFraction", 1.0))
    if bool(interface["useFullButlerVolmer"]):
        j_bv = (
            float(channel["exchangeCurrentDensity_A_per_m2"])
            * kinetic_multiplier
            * (
                concentration_factor * torch.exp(forward)
                - reverse_factor * torch.exp(reverse)
            )
        )
    else:
        j_bv = (
            float(channel["exchangeCurrentDensity_A_per_m2"])
            * kinetic_multiplier
            * concentration_factor
            * torch.exp(forward)
        )
    j_kinetic = torch.clamp(j_bv, min=0.0)
    if bool(coupled["usePaperMassTransferSaturation"]):
        if bool(interface["deriveMassTransferCoefficientFromDiffusivity"]):
            km = diffusivity / max(
                float(interface["physicalDiffusionLayerThickness_m"]), 1e-30
            )
        else:
            km = torch.full_like(
                diffusivity, float(channel["massTransferCoefficient_m_per_s"])
            )
        j_limit = (
            electron_number * faraday * km * torch.clamp(concentration, min=0.0)
        )
        current_floor = physical_floor(config, "currentDensity_A_per_m2", 1e-12)
        positive_limit = j_limit > current_floor
        safe_limit = torch.where(
            positive_limit, j_limit, torch.ones_like(j_limit)
        )
        ratio = j_kinetic / safe_limit
        inverse_saturation = 1.0 / (1.0 + ratio)
        candidate = torch.where(
            positive_limit,
            j_kinetic * inverse_saturation,
            torch.zeros_like(j_kinetic),
        )
    else:
        candidate = j_kinetic
    # Non-finite BV arithmetic is a model failure, not zero current.  Preserve
    # it so the enclosing nonlinear solve fails closed instead of silently
    # classifying an invalid high-current case as non-ignition.
    return candidate


def _reaction_candidate_and_deta(
    concentration_factor: torch.Tensor,
    concentration: torch.Tensor,
    diffusivity: torch.Tensor,
    overpotential: torch.Tensor,
    temperature: torch.Tensor,
    channel: dict,
    config: dict,
    kinetic_multiplier: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mass-transfer-limited BV current and d(current)/d(eta).

    The saturation derivative is evaluated as ``dkin * inv * inv`` with
    ``inv = 1/(1+jkin/jlimit)``.  This is algebraically identical to
    ``(jlimit/(jkin+jlimit))**2*dkin`` but avoids an O(1e-37) intermediate
    that Metal may flush to zero in FP32.
    """
    interface = config["interface"]
    coupled = config["coupled"]
    gas_constant = float(config["transport"]["gasConstant_J_per_molK"])
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    alpha = float(channel["chargeTransferCoefficient"])
    electron_number = float(channel["electronNumber"])
    eta = torch.clamp(overpotential, min=0.0)
    k_forward = alpha * electron_number * faraday / (gas_constant * temperature)
    k_reverse = (
        (1.0 - alpha) * electron_number * faraday / (gas_constant * temperature)
    )
    raw_forward = k_forward * eta
    raw_reverse = -k_reverse * eta
    limit = float(interface["exponentialArgumentLimit"])
    forward = torch.clamp(raw_forward, -limit, limit)
    reverse = torch.clamp(raw_reverse, -limit, limit)
    forward_active = (
        (raw_forward > -limit) & (raw_forward < limit) & (overpotential > 0.0)
    )
    reverse_active = (
        (raw_reverse > -limit) & (raw_reverse < limit) & (overpotential > 0.0)
    )
    reverse_factor = float(channel.get("reverseAvailabilityFraction", 1.0))
    prefactor = (
        float(channel["exchangeCurrentDensity_A_per_m2"]) * kinetic_multiplier
    )
    exp_forward = torch.exp(forward)
    exp_reverse = torch.exp(reverse)
    if bool(interface["useFullButlerVolmer"]):
        raw_bv = prefactor * (
            concentration_factor * exp_forward - reverse_factor * exp_reverse
        )
        raw_derivative = prefactor * (
            concentration_factor
            * exp_forward
            * k_forward
            * forward_active.to(eta.dtype)
            + reverse_factor
            * exp_reverse
            * k_reverse
            * reverse_active.to(eta.dtype)
        )
    else:
        raw_bv = prefactor * concentration_factor * exp_forward
        raw_derivative = (
            prefactor
            * concentration_factor
            * exp_forward
            * k_forward
            * forward_active.to(eta.dtype)
        )
    positive = raw_bv > 0.0
    j_kinetic = torch.clamp(raw_bv, min=0.0)
    dkin_deta = torch.where(
        positive, raw_derivative, torch.zeros_like(raw_derivative)
    )
    if bool(coupled["usePaperMassTransferSaturation"]):
        if bool(interface["deriveMassTransferCoefficientFromDiffusivity"]):
            km = diffusivity / max(
                float(interface["physicalDiffusionLayerThickness_m"]), 1e-30
            )
        else:
            km = torch.full_like(
                diffusivity, float(channel["massTransferCoefficient_m_per_s"])
            )
        j_limit = (
            electron_number * faraday * km * torch.clamp(concentration, min=0.0)
        )
        current_floor = physical_floor(config, "currentDensity_A_per_m2", 1e-12)
        positive_limit = j_limit > current_floor
        safe_limit = torch.where(
            positive_limit, j_limit, torch.ones_like(j_limit)
        )
        ratio = j_kinetic / safe_limit
        inverse_saturation = 1.0 / (1.0 + ratio)
        candidate = torch.where(
            positive_limit,
            j_kinetic * inverse_saturation,
            torch.zeros_like(j_kinetic),
        )
        # Multiplication order is intentional: dkin*inv is normally O(jlimit),
        # so the second multiply never forms a denormal ratio-squared first.
        derivative = torch.where(
            positive_limit,
            (dkin_deta * inverse_saturation) * inverse_saturation,
            torch.zeros_like(dkin_deta),
        )
    else:
        candidate = j_kinetic
        derivative = dkin_deta
    # Never turn overflow/NaN into an electrochemically inactive contact.  The
    # local and global nonlinear convergence gates must observe the invalid
    # arithmetic and reject the candidate.
    return candidate, derivative


def _finish_heat(
    current_density: torch.Tensor,
    overpotential: torch.Tensor,
    channel: dict,
    config: dict,
    reaction_layer: float,
) -> torch.Tensor:
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    molar_rate = current_density / (float(channel["electronNumber"]) * faraday)
    heat = molar_rate * float(channel["reactionEnthalpy_J_per_mol"]) / reaction_layer
    if bool(config["interface"]["includeActivationHeat"]):
        heat = heat + float(config["interface"]["activationHeatFraction"]) * current_density * overpotential / reaction_layer
    return torch.clamp(heat, min=0.0)


def _interface_kinetic_candidates(
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float,
    spacing: float,
    potential: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | dict]:
    """Evaluate BV/mass-transfer candidates without post-hoc bulk-current clipping."""
    phi = state["potential"] if potential is None else potential
    dummy = {"Jx": torch.zeros_like(phi), "Jy": torch.zeros_like(phi)}
    _, _, anode_faces, cathode_faces = interface_normal_current(
        dummy, geometry, "signed_normal"
    )
    anode_zone, cathode_zone = anode_faces > 0, cathode_faces > 0
    reaction_layer = max(
        float(config["interface"]["physicalDiffusionLayerThickness_m"]),
        float(config["interface"]["numericalReactionLayerMinimum_m"]),
        spacing,
    )
    temperature = torch.clamp(state["temperature"], min=1.0)
    water_factor = torch.clamp(
        state["water"] / max(composition.initial_water_mol_per_m3, 1e-30), 0.0, 1.0
    )
    salt_concentration = torch.minimum(state["cation"], state["anion"])
    salt_factor = torch.clamp(
        salt_concentration / max(composition.initial_lp_mol_per_m3, 1e-30), 0.0, 1.0
    )
    threshold = float(config["phase"]["minimumLiquidFractionForFastIonTransport"])
    fast = torch.clamp(
        (state["liquidFraction"] - threshold) / max(1.0 - threshold, 1e-30), 0.0, 1.0
    )
    kinetic_multiplier = 1.0 + float(config["interface"]["liquidKineticsGain"]) * fast
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])
    water_a_cfg = config["interface"]["water"]["anode"]
    water_c_cfg = config["interface"]["water"]["cathode"]
    lp_a_cfg = config["interface"]["lp"]["anode"]
    lp_c_cfg = config["interface"]["lp"]["cathode"]
    nernst_cfg = config["interface"].get("nernst", {})
    activity_floor = float(nernst_cfg.get("activityFloor", 1e-6))
    water_activity = torch.clamp(water_factor, min=activity_floor)
    salt_gamma = _activity_coefficient(salt_concentration, temperature, config)
    salt_activity = torch.clamp(salt_factor * salt_gamma, min=activity_floor)
    eeq_aw, shift_aw = _equilibrium_potential(water_a_cfg, water_activity, temperature, config)
    eeq_cw, shift_cw = _equilibrium_potential(water_c_cfg, water_activity, temperature, config)
    eeq_alp, shift_alp = _equilibrium_potential(lp_a_cfg, salt_activity, temperature, config)
    eeq_clp, shift_clp = _equilibrium_potential(lp_c_cfg, salt_activity, temperature, config)
    eta_aw = torch.clamp(voltage - phi - eeq_aw, min=0.0)
    eta_cw = torch.clamp(phi - cathode_voltage - eeq_cw, min=0.0)
    eta_alp = torch.clamp(voltage - phi - eeq_alp, min=0.0)
    eta_clp = torch.clamp(phi - cathode_voltage - eeq_clp, min=0.0)
    blocking_cfg = config["interface"].get("blocking", {})
    active_area = torch.clamp(
        (1.0 - state.get("passivation", torch.zeros_like(temperature)))
        * (1.0 - state.get("gasCoverage", torch.zeros_like(temperature))),
        min=float(blocking_cfg.get("minimumActiveAreaFraction", 0.02)),
        max=1.0,
    )
    ones = torch.ones_like(temperature)
    base_aw, d_aw = _reaction_candidate_and_deta(
        water_factor, state["water"], transport["Dwater"], eta_aw, temperature,
        water_a_cfg, config, ones
    )
    base_cw, d_cw = _reaction_candidate_and_deta(
        water_factor, state["water"], transport["Dwater"], eta_cw, temperature,
        water_c_cfg, config, ones
    )
    minimum_ion_d = torch.minimum(transport["Dcation"], transport["Danion"])
    base_alp, d_alp = _reaction_candidate_and_deta(
        salt_factor, salt_concentration, minimum_ion_d, eta_alp, temperature,
        lp_a_cfg, config, kinetic_multiplier
    )
    base_clp, d_clp = _reaction_candidate_and_deta(
        salt_factor, salt_concentration, minimum_ion_d, eta_clp, temperature,
        lp_c_cfg, config, kinetic_multiplier
    )
    cand_aw, cand_cw = active_area * base_aw, active_area * base_cw
    cand_alp, cand_clp = active_area * base_alp, active_area * base_clp
    d_aw, d_cw = active_area * d_aw, active_area * d_cw
    d_alp, d_clp = active_area * d_alp, active_area * d_clp
    zero = torch.zeros_like(temperature)
    cand_aw = torch.where(anode_zone, cand_aw, zero)
    cand_alp = torch.where(anode_zone, cand_alp, zero)
    cand_cw = torch.where(cathode_zone, cand_cw, zero)
    cand_clp = torch.where(cathode_zone, cand_clp, zero)
    d_aw = torch.where(anode_zone, d_aw, zero)
    d_alp = torch.where(anode_zone, d_alp, zero)
    d_cw = torch.where(cathode_zone, d_cw, zero)
    d_clp = torch.where(cathode_zone, d_clp, zero)
    return {
        "candWaterAnode": cand_aw,
        "candWaterCathode": cand_cw,
        "candLPAnode": cand_alp,
        "candLPCathode": cand_clp,
        "totalAnodeCandidate": cand_aw + cand_alp,
        "totalCathodeCandidate": cand_cw + cand_clp,
        "dTotalAnode_dEta_S_per_m2": d_aw + d_alp,
        "dTotalCathode_dEta_S_per_m2": d_cw + d_clp,
        "anodeZone": anode_zone,
        "cathodeZone": cathode_zone,
        "anodeFaceCount": anode_faces,
        "cathodeFaceCount": cathode_faces,
        "etaWaterAnode": eta_aw,
        "etaWaterCathode": eta_cw,
        "etaLPAnode": eta_alp,
        "etaLPCathode": eta_clp,
        "equilibriumPotentialWaterAnode_V": eeq_aw,
        "equilibriumPotentialWaterCathode_V": eeq_cw,
        "equilibriumPotentialLPAnode_V": eeq_alp,
        "equilibriumPotentialLPCathode_V": eeq_clp,
        "shiftWaterAnode_V": shift_aw,
        "shiftWaterCathode_V": shift_cw,
        "shiftLPAnode_V": shift_alp,
        "shiftLPCathode_V": shift_clp,
        "saltGamma": salt_gamma,
        "activeArea": active_area,
        "reactionLayerThickness_m": torch.full(
            (phi.shape[0],), reaction_layer, device=phi.device, dtype=phi.dtype
        ),
        "fastIonFraction": fast,
        "channelConfigs": {
            "water_a": water_a_cfg, "water_c": water_c_cfg,
            "lp_a": lp_a_cfg, "lp_c": lp_c_cfg,
        },
    }


def _robin_face_values(
    potential: torch.Tensor,
    sigma: torch.Tensor,
    geometry: GeometryBatch,
    spacing: float,
    j_anode: torch.Tensor,
    j_cathode: torch.Tensor,
    config: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Directional electrolyte ghost potentials enforcing J·n = j_BV.

    The outward normal is from propellant to metal.  Conventional current is
    -j_anode at an anode face and +j_cathode at a cathode face.  With zero
    explicit normal diffusion flux at the metal face, -sigma dphi/dn = J·n.
    """
    prop = geometry.propellant
    anode_e = torch.zeros_like(prop); anode_e[..., :, :-1] = geometry.anode[..., :, 1:]
    anode_w = torch.zeros_like(prop); anode_w[..., :, 1:] = geometry.anode[..., :, :-1]
    anode_n = torch.zeros_like(prop); anode_n[..., :-1, :] = geometry.anode[..., 1:, :]
    anode_s = torch.zeros_like(prop); anode_s[..., 1:, :] = geometry.anode[..., :-1, :]
    cathode_e = torch.zeros_like(prop); cathode_e[..., :, :-1] = geometry.cathode[..., :, 1:]
    cathode_w = torch.zeros_like(prop); cathode_w[..., :, 1:] = geometry.cathode[..., :, :-1]
    cathode_n = torch.zeros_like(prop); cathode_n[..., :-1, :] = geometry.cathode[..., 1:, :]
    cathode_s = torch.zeros_like(prop); cathode_s[..., 1:, :] = geometry.cathode[..., :-1, :]

    s_e = torch.zeros_like(sigma); s_e[..., :, :-1] = harmonic_mean(sigma[..., :, :-1], sigma[..., :, 1:])
    s_w = torch.zeros_like(sigma); s_w[..., :, 1:] = harmonic_mean(sigma[..., :, 1:], sigma[..., :, :-1])
    s_n = torch.zeros_like(sigma); s_n[..., :-1, :] = harmonic_mean(sigma[..., :-1, :], sigma[..., 1:, :])
    s_s = torch.zeros_like(sigma); s_s[..., 1:, :] = harmonic_mean(sigma[..., 1:, :], sigma[..., :-1, :])
    conductivity_floor = physical_floor(config, "conductivity_S_per_m", 1e-12)

    def ghost(a_mask: torch.Tensor, c_mask: torch.Tensor, s_face: torch.Tensor) -> torch.Tensor:
        # g = J_total · n_prop_to_metal = -j_a at anode, +j_c at cathode.
        g = -j_anode * a_mask.to(sigma.dtype) + j_cathode * c_mask.to(sigma.dtype)
        return potential - spacing * g / torch.clamp(
            s_face, min=conductivity_floor
        )

    return {
        "east": ghost(anode_e, cathode_e, s_e),
        "west": ghost(anode_w, cathode_w, s_w),
        "north": ghost(anode_n, cathode_n, s_n),
        "south": ghost(anode_s, cathode_s, s_s),
    }


def _relative_rms_change(new: torch.Tensor, old: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(new.dtype)
    count = torch.clamp(weight.sum(dim=(-2, -1)), min=1.0)
    delta = torch.sqrt((((new - old) ** 2) * weight).sum(dim=(-2, -1)) / count)
    scale = torch.sqrt(((torch.maximum(new.abs(), old.abs()) ** 2) * weight).sum(dim=(-2, -1)) / count)
    return delta / torch.clamp(scale, min=1.0)


def _gauge_mask_for_geometry(geometry: GeometryBatch) -> torch.Tensor:
    """Choose one interior propellant cell per case to fix the Neumann gauge."""
    mask = torch.zeros_like(geometry.propellant)
    interior = geometry.propellant.clone()
    interior[..., 0, :] = False
    interior[..., -1, :] = False
    interior[..., :, 0] = False
    interior[..., :, -1] = False
    for batch_index in range(geometry.batch_size):
        coords = torch.nonzero(interior[batch_index], as_tuple=False)
        if coords.numel() == 0:
            raise RuntimeError("No interior propellant cell available for Robin gauge fixing")
        # Prefer a cell close to the geometric centre to reduce conditioning bias.
        coord_dtype = torch.float32 if coords.device.type == "mps" else torch.float64
        centre = torch.tensor(
            [(geometry.grid_size - 1) / 2.0, (geometry.grid_size - 1) / 2.0],
            device=coords.device,
            dtype=coord_dtype,
        )
        distance2 = ((coords.to(coord_dtype) - centre) ** 2).sum(dim=1)
        selected = coords[torch.argmin(distance2)]
        mask[batch_index, int(selected[0]), int(selected[1])] = True
    return mask


def _apply_electrolyte_offset(
    relative_potential: torch.Tensor,
    offset: torch.Tensor,
    geometry: GeometryBatch,
    voltage: float,
    cathode_voltage: float,
) -> torch.Tensor:
    actual = torch.where(
        geometry.propellant,
        relative_potential + offset[:, None, None],
        relative_potential,
    )
    actual = torch.where(
        geometry.anode,
        torch.as_tensor(voltage, device=actual.device, dtype=actual.dtype),
        actual,
    )
    actual = torch.where(
        geometry.cathode,
        torch.as_tensor(cathode_voltage, device=actual.device, dtype=actual.dtype),
        actual,
    )
    return actual




def _interface_context_fields(
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    config: dict,
    composition: CompositionModel,
) -> dict[str, torch.Tensor | dict]:
    """Precompute state-dependent fields used by the implicit BV Robin faces."""
    temperature = torch.clamp(state["temperature"], min=1.0)
    water_reference = float(composition.initial_water_mol_per_m3)
    water_factor = (
        torch.clamp(state["water"] / water_reference, 0.0, 1.0)
        if water_reference > 0.0
        else torch.zeros_like(state["water"])
    )
    salt_concentration = torch.minimum(state["cation"], state["anion"])
    salt_factor = torch.clamp(
        salt_concentration / composition.initial_lp_mol_per_m3, 0.0, 1.0
    )
    threshold = float(config["phase"]["minimumLiquidFractionForFastIonTransport"])
    fast = torch.clamp(
        (state["liquidFraction"] - threshold) / max(1.0 - threshold, 1e-30), 0.0, 1.0
    )
    kinetic_multiplier = 1.0 + float(config["interface"]["liquidKineticsGain"]) * fast
    blocking_cfg = config["interface"].get("blocking", {})
    active_area = torch.clamp(
        (1.0 - state.get("passivation", torch.zeros_like(temperature)))
        * (1.0 - state.get("gasCoverage", torch.zeros_like(temperature))),
        min=float(blocking_cfg.get("minimumActiveAreaFraction", 0.02)),
        max=1.0,
    )
    nernst_cfg = config["interface"].get("nernst", {})
    activity_floor = float(nernst_cfg.get("activityFloor", 1e-6))
    water_activity = torch.clamp(water_factor, min=activity_floor)
    salt_gamma = _activity_coefficient(salt_concentration, temperature, config)
    salt_activity = torch.clamp(salt_factor * salt_gamma, min=activity_floor)
    water_a_cfg = config["interface"]["water"]["anode"]
    water_c_cfg = config["interface"]["water"]["cathode"]
    lp_a_cfg = config["interface"]["lp"]["anode"]
    lp_c_cfg = config["interface"]["lp"]["cathode"]
    eeq_aw, shift_aw = _equilibrium_potential(water_a_cfg, water_activity, temperature, config)
    eeq_cw, shift_cw = _equilibrium_potential(water_c_cfg, water_activity, temperature, config)
    eeq_alp, shift_alp = _equilibrium_potential(lp_a_cfg, salt_activity, temperature, config)
    eeq_clp, shift_clp = _equilibrium_potential(lp_c_cfg, salt_activity, temperature, config)
    return {
        "temperature": temperature,
        "waterFactor": water_factor,
        "saltConcentration": salt_concentration,
        "saltFactor": salt_factor,
        "kineticMultiplier": kinetic_multiplier,
        "activeArea": active_area,
        "saltGamma": salt_gamma,
        "eeqWaterAnode": eeq_aw,
        "eeqWaterCathode": eeq_cw,
        "eeqLPAnode": eeq_alp,
        "eeqLPCathode": eeq_clp,
        "shiftWaterAnode": shift_aw,
        "shiftWaterCathode": shift_cw,
        "shiftLPAnode": shift_alp,
        "shiftLPCathode": shift_clp,
        "waterAnodeConfig": water_a_cfg,
        "waterCathodeConfig": water_c_cfg,
        "lpAnodeConfig": lp_a_cfg,
        "lpCathodeConfig": lp_c_cfg,
    }


def _directional_interface_geometry(
    geometry: GeometryBatch,
    sigma: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return propellant-cell indexed electrode-face masks and face conductivities."""
    prop = geometry.propellant
    result: dict[str, dict[str, torch.Tensor]] = {}
    for direction in ("east", "west", "north", "south"):
        result[direction] = {
            "anode": torch.zeros_like(prop),
            "cathode": torch.zeros_like(prop),
            "sigma": torch.zeros_like(sigma),
        }
    result["east"]["anode"][..., :, :-1] = geometry.anode[..., :, 1:]
    result["east"]["cathode"][..., :, :-1] = geometry.cathode[..., :, 1:]
    result["east"]["sigma"][..., :, :-1] = harmonic_mean(
        sigma[..., :, :-1], sigma[..., :, 1:]
    )
    result["west"]["anode"][..., :, 1:] = geometry.anode[..., :, :-1]
    result["west"]["cathode"][..., :, 1:] = geometry.cathode[..., :, :-1]
    result["west"]["sigma"][..., :, 1:] = harmonic_mean(
        sigma[..., :, 1:], sigma[..., :, :-1]
    )
    result["north"]["anode"][..., :-1, :] = geometry.anode[..., 1:, :]
    result["north"]["cathode"][..., :-1, :] = geometry.cathode[..., 1:, :]
    result["north"]["sigma"][..., :-1, :] = harmonic_mean(
        sigma[..., :-1, :], sigma[..., 1:, :]
    )
    result["south"]["anode"][..., 1:, :] = geometry.anode[..., :-1, :]
    result["south"]["cathode"][..., 1:, :] = geometry.cathode[..., :-1, :]
    result["south"]["sigma"][..., 1:, :] = harmonic_mean(
        sigma[..., 1:, :], sigma[..., :-1, :]
    )
    for entry in result.values():
        entry["anode"] &= prop
        entry["cathode"] &= prop
    return result


def _local_bv_channels(
    interface_potential: torch.Tensor,
    index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    polarity: str,
    context: dict[str, torch.Tensor | dict],
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    config: dict,
    voltage: float,
) -> dict[str, torch.Tensor]:
    """Evaluate the two parallel electrochemical channels on selected faces."""
    temperature = context["temperature"][index]
    active = context["activeArea"][index]
    if polarity == "anode":
        eta_water = torch.clamp(
            voltage - interface_potential - context["eeqWaterAnode"][index], min=0.0
        )
        eta_lp = torch.clamp(
            voltage - interface_potential - context["eeqLPAnode"][index], min=0.0
        )
        water_cfg = context["waterAnodeConfig"]
        lp_cfg = context["lpAnodeConfig"]
    elif polarity == "cathode":
        cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])
        eta_water = torch.clamp(
            interface_potential - cathode_voltage - context["eeqWaterCathode"][index], min=0.0
        )
        eta_lp = torch.clamp(
            interface_potential - cathode_voltage - context["eeqLPCathode"][index], min=0.0
        )
        water_cfg = context["waterCathodeConfig"]
        lp_cfg = context["lpCathodeConfig"]
    else:
        raise ValueError(f"Unknown electrode polarity: {polarity}")
    water, dwater = _reaction_candidate_and_deta(
        context["waterFactor"][index],
        state["water"][index],
        transport["Dwater"][index],
        eta_water,
        temperature,
        water_cfg,
        config,
        torch.ones_like(temperature),
    )
    minimum_ion_d = torch.minimum(transport["Dcation"], transport["Danion"])[index]
    lp, dlp = _reaction_candidate_and_deta(
        context["saltFactor"][index],
        context["saltConcentration"][index],
        minimum_ion_d,
        eta_lp,
        temperature,
        lp_cfg,
        config,
        context["kineticMultiplier"][index],
    )
    return {
        "water": active * water,
        "lp": active * lp,
        "dWater_dEta": active * dwater,
        "dLP_dEta": active * dlp,
        "etaWater": eta_water,
        "etaLP": eta_lp,
    }


def _local_bv_channels_from_electrode_gap(
    electrode_gap: torch.Tensor,
    index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    polarity: str,
    context: dict[str, torch.Tensor | dict],
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    config: dict,
) -> dict[str, torch.Tensor]:
    """Evaluate BV channels directly from |V_electrode-phi_face|.

    Computing eta as ``260 - phi_face`` in FP32 loses the small overpotential
    in a subtraction of O(10^2 V) values.  The local implicit solve already
    owns the small electrode gap, so using it directly removes that cancellation.
    """
    temperature = context["temperature"][index]
    active = context["activeArea"][index]
    if polarity == "anode":
        eta_water = torch.clamp(
            electrode_gap - context["eeqWaterAnode"][index], min=0.0
        )
        eta_lp = torch.clamp(
            electrode_gap - context["eeqLPAnode"][index], min=0.0
        )
        water_cfg = context["waterAnodeConfig"]
        lp_cfg = context["lpAnodeConfig"]
    elif polarity == "cathode":
        eta_water = torch.clamp(
            electrode_gap - context["eeqWaterCathode"][index], min=0.0
        )
        eta_lp = torch.clamp(
            electrode_gap - context["eeqLPCathode"][index], min=0.0
        )
        water_cfg = context["waterCathodeConfig"]
        lp_cfg = context["lpCathodeConfig"]
    else:
        raise ValueError(f"Unknown electrode polarity: {polarity}")
    water, dwater = _reaction_candidate_and_deta(
        context["waterFactor"][index],
        state["water"][index],
        transport["Dwater"][index],
        eta_water,
        temperature,
        water_cfg,
        config,
        torch.ones_like(temperature),
    )
    minimum_ion_d = torch.minimum(
        transport["Dcation"], transport["Danion"]
    )[index]
    lp, dlp = _reaction_candidate_and_deta(
        context["saltFactor"][index],
        context["saltConcentration"][index],
        minimum_ion_d,
        eta_lp,
        temperature,
        lp_cfg,
        config,
        context["kineticMultiplier"][index],
    )
    return {
        "water": active * water,
        "lp": active * lp,
        "dWater_dEta": active * dwater,
        "dLP_dEta": active * dlp,
        "etaWater": eta_water,
        "etaLP": eta_lp,
    }


def _series_contact_slope(
    kinetic_slope: torch.Tensor,
    conductance: torch.Tensor,
    slope_floor: float,
    active: torch.Tensor,
) -> torch.Tensor:
    """Return ``kG/(k+G)`` without overflowing for large finite slopes."""
    valid = (
        torch.isfinite(kinetic_slope)
        & torch.isfinite(conductance)
        & (kinetic_slope >= 0.0)
        & (conductance >= 0.0)
    )
    smaller = torch.minimum(kinetic_slope, conductance)
    larger = torch.maximum(kinetic_slope, conductance)
    safe_larger = torch.where(
        larger > 0.0, larger, torch.ones_like(larger)
    )
    value = smaller / (1.0 + smaller / safe_larger)
    value = torch.where(smaller == 0.0, torch.zeros_like(value), value)
    active_value = torch.where(
        active & (larger > slope_floor), value, torch.zeros_like(value)
    )
    return torch.where(
        valid,
        active_value,
        torch.full_like(value, float("nan")),
    )


def _implicit_bv_surface_contacts(
    potential: torch.Tensor,
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float | Sequence[float] | torch.Tensor,
    spacing: float,
) -> dict[str, Any]:
    """Evaluate nonlinear BV reactions on the complete top-contact footprint.

    Each true mask cell is one surface element of area ``spacing**2``.  The
    propellant cell below it remains part of every bulk equation.  A local
    through-thickness conductance is eliminated together with the BV law, and
    the resulting non-negative series slope is used by the SPD bulk solve.
    """
    if bool(torch.any(geometry.anode & geometry.cathode)):
        raise ValueError("Surface anode/cathode contact masks overlap")
    if bool(torch.any((geometry.anode | geometry.cathode) & ~geometry.propellant)):
        raise ValueError("Surface contacts must overlay, not replace, propellant cells")

    robin = config["interface"].get("nonlinearRobin", {})
    minimum_iterations = max(
        1, int(robin.get("localInterfaceMinimumIterations", 8))
    )
    maximum_iterations = max(
        minimum_iterations,
        int(robin.get("localInterfaceMaximumIterations", 60)),
    )
    absolute_tolerance = max(
        0.0, float(robin.get("localRobinResidualTolerance_A_per_m2", 0.01))
    )
    relative_tolerance = max(
        0.0, float(robin.get("localRobinRelativeResidualTolerance", 1e-4))
    )
    newton_iterations = max(0, int(robin.get("localNewtonPolishIterations", 2)))
    roundoff_safety = max(
        1.0, float(robin.get("localRoundoffSafetyFactor", 2.0))
    )
    contact_length = float(config["interface"]["contactNormalConductionLength_m"])
    surface_layer = float(config["geometry"]["surfaceLayerThickness_m"])
    if not math.isfinite(contact_length) or contact_length <= 0.0:
        raise ValueError("contactNormalConductionLength_m must be finite and positive")
    if not math.isfinite(surface_layer) or surface_layer <= 0.0:
        raise ValueError("surfaceLayerThickness_m must be finite and positive")
    conductivity_floor = physical_floor(config, "conductivity_S_per_m", 1e-12)
    current_floor = physical_floor(config, "currentDensity_A_per_m2", 1e-12)
    slope_floor = physical_floor(config, "currentDensitySlope_A_per_m2_V", 1e-12)
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])
    context = _interface_context_fields(state, transport, config, composition)
    zero = torch.zeros_like(potential)
    voltage_vector = _batch_voltage_vector(voltage, potential)
    fields = {
        "jWaterAnode_A_per_m2": zero.clone(),
        "jLPAnode_A_per_m2": zero.clone(),
        "jWaterCathode_A_per_m2": zero.clone(),
        "jLPCathode_A_per_m2": zero.clone(),
        "etaWaterAnode_V": zero.clone(),
        "etaLPAnode_V": zero.clone(),
        "etaWaterCathode_V": zero.clone(),
        "etaLPCathode_V": zero.clone(),
        "anodeSlope_S_per_m2": zero.clone(),
        "cathodeSlope_S_per_m2": zero.clone(),
        "normalOhmicDropAnode_V": zero.clone(),
        "normalOhmicDropCathode_V": zero.clone(),
    }
    batch_size = geometry.batch_size
    maximum_residual = torch.zeros(batch_size, device=potential.device, dtype=potential.dtype)
    maximum_relative = torch.zeros_like(maximum_residual)
    maximum_combined = torch.zeros_like(maximum_residual)
    maximum_local_iterations = torch.zeros(
        batch_size, device=potential.device, dtype=torch.int64
    )
    unresolved_count = torch.zeros_like(maximum_local_iterations)
    maximum_roundoff_floor = torch.zeros_like(maximum_residual)
    roundoff_limited_count = torch.zeros_like(maximum_local_iterations)
    machine_eps = torch.finfo(potential.dtype).eps
    numerical_tiny = torch.finfo(potential.dtype).tiny

    for polarity, contact_mask in (
        ("anode", geometry.anode),
        ("cathode", geometry.cathode),
    ):
        if not bool(torch.any(contact_mask)):
            raise ValueError(f"Surface {polarity} contact mask is empty")
        index = torch.nonzero(contact_mask, as_tuple=True)
        batch_index = index[0]
        phi_cell = potential[index]
        conductance = torch.clamp(
            transport["sigmaTotal"][index] / contact_length,
            min=conductivity_floor / contact_length,
        )
        electrode_voltage = voltage_vector.index_select(0, batch_index)
        maximum_drop = (
            torch.clamp(electrode_voltage - phi_cell, min=0.0)
            if polarity == "anode"
            else torch.clamp(phi_cell - cathode_voltage, min=0.0)
        )
        forward = maximum_drop > 0.0
        lower = torch.zeros_like(maximum_drop)
        upper = maximum_drop.clone()
        gap = 0.5 * maximum_drop
        converged = ~forward
        iterations_used = torch.zeros_like(batch_index)

        def evaluate(current_gap: torch.Tensor):
            channels = _local_bv_channels_from_electrode_gap(
                current_gap, index, polarity, context, state, transport, config
            )
            kinetic_current = channels["water"] + channels["lp"]
            kinetic_slope = channels["dWater_dEta"] + channels["dLP_dEta"]
            normal_current = conductance * torch.clamp(
                maximum_drop - current_gap, min=0.0
            )
            residual = kinetic_current - normal_current
            residual_slope = kinetic_slope + conductance
            scale = torch.clamp(
                torch.maximum(kinetic_current.abs(), normal_current.abs()), min=1.0
            )
            if absolute_tolerance > 0.0 and relative_tolerance > 0.0:
                physical_threshold = torch.minimum(
                    torch.full_like(scale, absolute_tolerance),
                    relative_tolerance * scale,
                )
            elif absolute_tolerance > 0.0:
                physical_threshold = torch.full_like(scale, absolute_tolerance)
            elif relative_tolerance > 0.0:
                physical_threshold = relative_tolerance * scale
            else:
                physical_threshold = torch.full_like(scale, machine_eps)
            roundoff_floor = 0.5 * roundoff_safety * machine_eps * (
                torch.clamp(current_gap.abs(), min=1.0)
                * residual_slope.abs()
                + torch.clamp(maximum_drop.abs(), min=1.0)
                * conductance.abs()
            )
            threshold = torch.clamp(
                torch.maximum(physical_threshold, roundoff_floor),
                min=numerical_tiny,
            )
            return (
                channels,
                kinetic_current,
                kinetic_slope,
                residual,
                scale,
                threshold,
                physical_threshold,
                roundoff_floor,
                residual_slope,
            )

        for iteration in range(1, maximum_iterations + 1):
            (
                channels,
                kinetic_current,
                kinetic_slope,
                residual,
                scale,
                threshold,
                physical_threshold,
                roundoff_floor,
                residual_slope,
            ) = evaluate(gap)
            newly = (
                (~converged)
                & (iteration >= minimum_iterations)
                & (residual.abs() <= threshold)
            )
            iterations_used = torch.where(
                newly, torch.full_like(iterations_used, iteration), iterations_used
            )
            converged = converged | newly
            unresolved = ~converged
            if not bool(torch.any(unresolved)):
                break
            lower = torch.where(unresolved & (residual < 0.0), gap, lower)
            upper = torch.where(unresolved & (residual >= 0.0), gap, upper)
            gap = torch.where(unresolved, 0.5 * (lower + upper), gap)

        # Match the native contact-cell order exactly: an unresolved midpoint
        # created by the final bisection update is first consumed by the
        # safeguarded Newton polish, not classified in a separate pre-polish
        # convergence check.
        unresolved = ~converged
        for _ in range(newton_iterations):
            if not bool(torch.any(unresolved)):
                break
            (
                channels,
                kinetic_current,
                kinetic_slope,
                residual,
                scale,
                threshold,
                physical_threshold,
                roundoff_floor,
                residual_slope,
            ) = evaluate(gap)
            safe = (
                unresolved
                & torch.isfinite(residual_slope)
                & (residual_slope > slope_floor)
            )
            candidate = gap - residual / torch.where(
                safe, residual_slope, torch.ones_like(residual_slope)
            )
            inside_bracket = (
                safe
                & torch.isfinite(candidate)
                & (candidate >= lower)
                & (candidate <= upper)
            )
            gap = torch.where(inside_bracket, candidate, gap)
            (
                channels,
                kinetic_current,
                kinetic_slope,
                residual,
                scale,
                threshold,
                physical_threshold,
                roundoff_floor,
                residual_slope,
            ) = evaluate(gap)
            move_lower = unresolved & (residual < 0.0)
            lower = torch.where(move_lower, gap, lower)
            upper = torch.where(unresolved & ~move_lower, gap, upper)
            newly = unresolved & (residual.abs() <= threshold)
            converged = converged | newly
            unresolved = unresolved & ~newly

        (
            channels,
            kinetic_current,
            kinetic_slope,
            residual,
            scale,
            threshold,
            physical_threshold,
            roundoff_floor,
            residual_slope,
        ) = evaluate(gap)
        converged = converged | (residual.abs() <= threshold)
        iterations_used = torch.where(
            (iterations_used == 0) & forward,
            torch.full_like(iterations_used, maximum_iterations),
            iterations_used,
        )
        series_slope = _series_contact_slope(
            kinetic_slope,
            conductance,
            slope_floor,
            forward,
        )
        normal_drop = torch.clamp(maximum_drop - gap, min=0.0)
        if polarity == "anode":
            fields["jWaterAnode_A_per_m2"][index] = channels["water"]
            fields["jLPAnode_A_per_m2"][index] = channels["lp"]
            fields["etaWaterAnode_V"][index] = channels["etaWater"]
            fields["etaLPAnode_V"][index] = channels["etaLP"]
            fields["anodeSlope_S_per_m2"][index] = series_slope
            fields["normalOhmicDropAnode_V"][index] = normal_drop
        else:
            fields["jWaterCathode_A_per_m2"][index] = channels["water"]
            fields["jLPCathode_A_per_m2"][index] = channels["lp"]
            fields["etaWaterCathode_V"][index] = channels["etaWater"]
            fields["etaLPCathode_V"][index] = channels["etaLP"]
            fields["cathodeSlope_S_per_m2"][index] = series_slope
            fields["normalOhmicDropCathode_V"][index] = normal_drop

        maximum_residual = torch.maximum(
            maximum_residual,
            _batch_scatter_max(residual.abs(), batch_index, batch_size),
        )
        maximum_relative = torch.maximum(
            maximum_relative,
            _batch_scatter_max(residual.abs() / torch.clamp(scale, min=current_floor), batch_index, batch_size),
        )
        maximum_combined = torch.maximum(
            maximum_combined,
            _batch_scatter_max(residual.abs() / threshold, batch_index, batch_size),
        )
        maximum_roundoff_floor = torch.maximum(
            maximum_roundoff_floor,
            _batch_scatter_max(
                torch.where(forward, roundoff_floor, torch.zeros_like(roundoff_floor)),
                batch_index,
                batch_size,
            ),
        )
        roundoff_limited = (
            forward
            & (roundoff_floor > physical_threshold)
            & (residual.abs() <= roundoff_floor)
        )
        roundoff_limited_count = roundoff_limited_count + _batch_scatter_sum(
            roundoff_limited.to(torch.int64), batch_index, batch_size
        )
        maximum_local_iterations = torch.maximum(
            maximum_local_iterations,
            _batch_scatter_max(iterations_used, batch_index, batch_size),
        )
        unresolved_count = unresolved_count + _batch_scatter_sum(
            (~converged).to(torch.int64), batch_index, batch_size
        )

    j_anode = fields["jWaterAnode_A_per_m2"] + fields["jLPAnode_A_per_m2"]
    j_cathode = fields["jWaterCathode_A_per_m2"] + fields["jLPCathode_A_per_m2"]
    contact_area = spacing * spacing
    raw_anode = j_anode.sum(dim=(-2, -1)) * contact_area
    raw_cathode = j_cathode.sum(dim=(-2, -1)) * contact_area
    absolute_difference = torch.abs(raw_anode - raw_cathode)
    current_scale = torch.maximum(raw_anode.abs(), raw_cathode.abs())
    balance_absolute = float(robin.get("currentBalanceAbsoluteTolerance_A", 1e-12))
    balance_relative = float(robin.get("currentBalanceTolerance", 5e-3))
    balance_threshold = torch.maximum(
        torch.full_like(current_scale, balance_absolute),
        balance_relative * current_scale,
    )
    balance_combined = absolute_difference / torch.clamp(
        balance_threshold, min=torch.finfo(potential.dtype).tiny
    )
    mismatch = absolute_difference / torch.clamp(
        current_scale, min=physical_floor(config, "current_A", 1e-15)
    )
    reaction = {
        **fields,
        "jAnode_A_per_m2": j_anode,
        "jCathode_A_per_m2": j_cathode,
        "anodeZone": geometry.anode,
        "cathodeZone": geometry.cathode,
        "anodeFaceCount": geometry.anode.to(potential.dtype),
        "cathodeFaceCount": geometry.cathode.to(potential.dtype),
        "rawAnodeCurrent_A": raw_anode,
        "rawCathodeCurrent_A": raw_cathode,
        "maximumLocalRobinResidual_A_per_m2": maximum_residual,
        "maximumLocalRobinRelativeResidual": maximum_relative,
        "maximumLocalRobinCombinedResidual": maximum_combined,
        "maximumLocalRobinRoundoffFloor_A_per_m2": maximum_roundoff_floor,
        "roundoffLimitedLocalRobinFaceCount": roundoff_limited_count,
        "maximumLocalRobinIterations": maximum_local_iterations,
        "unresolvedLocalRobinFaceCount": unresolved_count,
        "localRobinAllFacesConverged": unresolved_count == 0,
        "surfaceContactAreaElement_m2": torch.full_like(raw_anode, contact_area),
        "surfaceLayerThickness_m": torch.full_like(raw_anode, surface_layer),
        "currentBalanceMismatch": mismatch,
        "currentBalanceCombinedResidual": balance_combined,
        "currentBalanceThreshold_A": balance_threshold,
        "channelConfigs": {
            "water_a": context["waterAnodeConfig"],
            "water_c": context["waterCathodeConfig"],
            "lp_a": context["lpAnodeConfig"],
            "lp_c": context["lpCathodeConfig"],
        },
        "saltGamma": context["saltGamma"],
        "activeArea": context["activeArea"],
        "fastIonFraction": context["fastIonFraction"] if "fastIonFraction" in context else transport["fastIonFraction"],
        "equilibriumPotentialWaterAnode_V": context["eeqWaterAnode"],
        "equilibriumPotentialWaterCathode_V": context["eeqWaterCathode"],
        "equilibriumPotentialLPAnode_V": context["eeqLPAnode"],
        "equilibriumPotentialLPCathode_V": context["eeqLPCathode"],
        "shiftWaterAnode_V": context["shiftWaterAnode"],
        "shiftWaterCathode_V": context["shiftWaterCathode"],
        "shiftLPAnode_V": context["shiftLPAnode"],
        "shiftLPCathode_V": context["shiftLPCathode"],
    }
    return {
        "surfaceLinearization": {
            "model": "surface_overlay_bv",
            "anodeCurrentDensity_A_per_m2": j_anode,
            "cathodeCurrentDensity_A_per_m2": j_cathode,
            "anodeSlope_S_per_m2": fields["anodeSlope_S_per_m2"],
            "cathodeSlope_S_per_m2": fields["cathodeSlope_S_per_m2"],
            "potential_V": potential,
            "surfaceLayerThickness_m": surface_layer,
        },
        "reaction": reaction,
    }


def _batch_membership(
    batch_index: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Fixed-shape [faces,batch] membership matrix without scatter atomics."""
    labels = torch.arange(batch_size, device=batch_index.device, dtype=batch_index.dtype)
    return batch_index[:, None] == labels[None, :]


def _batch_scatter_max(
    values: torch.Tensor,
    batch_index: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Return per-batch maxima using only fixed-shape elementwise reduction.

    MPS support for ``scatter_reduce_(amax)`` is version-dependent.  The batch
    size is small (normally 4), so a [faces,batch] boolean membership matrix is
    both cheap and deterministic for the current use.
    """
    result = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
    if values.numel() == 0:
        return result
    membership = _batch_membership(batch_index, batch_size)
    fill_value = (
        -torch.inf
        if values.dtype.is_floating_point
        else torch.iinfo(values.dtype).min
    )
    negative_infinity = torch.full(
        (values.shape[0], batch_size),
        fill_value,
        device=values.device,
        dtype=values.dtype,
    )
    candidates = torch.where(
        membership,
        values[:, None].expand(-1, batch_size),
        negative_infinity,
    )
    maximum = candidates.amax(dim=0)
    present = membership.any(dim=0)
    return torch.where(present, maximum, result)


def _batch_scatter_sum(
    values: torch.Tensor,
    batch_index: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Return per-batch sums without int64 ``scatter_add_`` on MPS."""
    if values.numel() == 0:
        return torch.zeros(batch_size, device=values.device, dtype=values.dtype)
    membership = _batch_membership(batch_index, batch_size)
    if values.dtype.is_floating_point:
        accumulation = values
        output_dtype = values.dtype
    else:
        # Face counts are far below the exact-integer range of FP32.  Accumulating
        # them as FP32 avoids the partially supported int64 scatter path on MPS.
        accumulation = values.to(torch.float32)
        output_dtype = values.dtype
    summed = (
        accumulation[:, None]
        * membership.to(accumulation.dtype)
    ).sum(dim=0)
    if output_dtype.is_floating_point:
        return summed.to(output_dtype)
    return torch.round(summed).to(output_dtype)


def _implicit_bv_robin_faces(
    potential: torch.Tensor,
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float,
    spacing: float,
) -> dict[str, Any]:
    """Eliminate each nonlinear BV Robin face in an overpotential-scale variable.

    The solved scalar is the electrode-to-interface gap
    ``gap = |V_electrode-phi_face|`` rather than the absolute interface
    potential near 260 V.  It is the equilibrium-potential shift away from the
    physical overpotential and is normally O(0.1--1 V), even when the bulk
    cell-to-electrode drop is O(10--100 V).  For both polarities

        R(gap) = j_BV(gap) - G*(maximum_drop-gap)

    is monotonically increasing on ``0 <= gap <= maximum_drop``.
    """
    robin = config["interface"].get("nonlinearRobin", {})
    legacy_iterations = max(1, int(robin.get("localInterfaceIterations", 60)))
    minimum_iterations = max(
        1,
        int(robin.get("localInterfaceMinimumIterations", min(8, legacy_iterations))),
    )
    maximum_iterations = max(
        minimum_iterations,
        int(
            robin.get(
                "localInterfaceMaximumIterations",
                max(legacy_iterations, 60),
            )
        ),
    )
    absolute_tolerance = max(
        0.0, float(robin.get("localRobinResidualTolerance_A_per_m2", 0.01))
    )
    relative_tolerance = max(
        0.0, float(robin.get("localRobinRelativeResidualTolerance", 1e-4))
    )
    newton_iterations = max(0, int(robin.get("localNewtonPolishIterations", 2)))
    roundoff_safety = max(
        1.0, float(robin.get("localRoundoffSafetyFactor", 2.0))
    )
    conductivity_floor = physical_floor(config, "conductivity_S_per_m", 1e-12)
    current_floor = physical_floor(config, "currentDensity_A_per_m2", 1e-12)
    slope_floor = physical_floor(
        config, "currentDensitySlope_A_per_m2_V", 1e-12
    )

    context = _interface_context_fields(state, transport, config, composition)
    directional = _directional_interface_geometry(geometry, transport["sigmaTotal"])
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])
    face_values = {
        name: potential.clone() for name in ("east", "west", "north", "south")
    }
    zero = torch.zeros_like(potential)
    sums = {
        "waterAnode": zero.clone(),
        "lpAnode": zero.clone(),
        "waterCathode": zero.clone(),
        "lpCathode": zero.clone(),
        "etaWaterAnodeTimesJ": zero.clone(),
        "etaLPAnodeTimesJ": zero.clone(),
        "etaWaterCathodeTimesJ": zero.clone(),
        "etaLPCathodeTimesJ": zero.clone(),
        "waterAnodeJWeight": zero.clone(),
        "lpAnodeJWeight": zero.clone(),
        "waterCathodeJWeight": zero.clone(),
        "lpCathodeJWeight": zero.clone(),
    }
    anode_face_count = torch.zeros_like(potential)
    cathode_face_count = torch.zeros_like(potential)
    batch_size = geometry.batch_size
    max_local_residual = torch.zeros(
        batch_size, device=potential.device, dtype=potential.dtype
    )
    max_local_relative_residual = torch.zeros_like(max_local_residual)
    max_local_combined_residual = torch.zeros_like(max_local_residual)
    max_local_roundoff_floor = torch.zeros_like(max_local_residual)
    roundoff_limited_face_count = torch.zeros(
        batch_size, device=potential.device, dtype=torch.int64
    )
    max_local_iterations = torch.zeros(
        batch_size, device=potential.device, dtype=torch.int64
    )
    unresolved_face_count = torch.zeros_like(max_local_iterations)
    face_area = spacing * float(config["geometry"]["surfaceLayerThickness_m"])
    integrated_anode = torch.zeros_like(max_local_residual)
    integrated_cathode = torch.zeros_like(max_local_residual)
    integrated_anode_gauge_sensitivity = torch.zeros_like(max_local_residual)
    integrated_cathode_gauge_sensitivity = torch.zeros_like(max_local_residual)
    machine_eps = torch.finfo(potential.dtype).eps
    numerical_tiny = torch.finfo(potential.dtype).tiny

    def indexed_contribution(
        reference: torch.Tensor,
        index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        values: torch.Tensor,
    ) -> torch.Tensor:
        contribution = torch.zeros_like(reference)
        contribution[index] = values
        return contribution

    def evaluate_gap(
        gap: torch.Tensor,
        index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        polarity: str,
        conductance: torch.Tensor,
        phi_cell: torch.Tensor,
        maximum_drop: torch.Tensor,
        forward: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        phi_face = voltage - gap if polarity == "anode" else cathode_voltage + gap
        channels = _local_bv_channels_from_electrode_gap(
            gap, index, polarity, context, state, transport, config
        )
        j_bv = channels["water"] + channels["lp"]
        kinetic_slope = channels["dWater_dEta"] + channels["dLP_dEta"]
        j_bulk = conductance * torch.clamp(maximum_drop - gap, min=0.0)
        signed = j_bv - j_bulk
        residual_slope_magnitude = kinetic_slope + conductance
        absolute = torch.where(forward, signed.abs(), torch.zeros_like(signed))
        residual_scale = torch.clamp(
            torch.maximum(j_bv.abs(), j_bulk.abs()), min=1.0
        )

        infinity = torch.full_like(absolute, torch.inf)
        absolute_threshold = (
            torch.full_like(absolute, absolute_tolerance)
            if absolute_tolerance > 0.0
            else infinity
        )
        relative_threshold = (
            relative_tolerance * residual_scale
            if relative_tolerance > 0.0
            else infinity
        )
        physical_threshold = torch.minimum(
            absolute_threshold, relative_threshold
        )
        # gap is overpotential-scale, so this ULP is about 250x smaller than
        # the former eps*|phi_face| estimate at an absolute 260 V potential.
        gap_ulp_bound = machine_eps * torch.clamp(gap.abs(), min=1.0)
        maximum_drop_ulp_bound = machine_eps * torch.clamp(
            maximum_drop.abs(), min=1.0
        )
        # R depends on both the overpotential-scale gap and the bulk drop
        # maximum_drop-gap.  The first term is now small; the second records
        # the unavoidable FP32 representation of the O(10--100 V) bulk field.
        roundoff_floor = 0.5 * roundoff_safety * (
            gap_ulp_bound * residual_slope_magnitude.abs()
            + maximum_drop_ulp_bound * conductance.abs()
        )
        effective_threshold = torch.maximum(physical_threshold, roundoff_floor)
        effective_threshold = torch.clamp(
            effective_threshold, min=numerical_tiny
        )
        combined = torch.where(
            forward, absolute / effective_threshold, torch.zeros_like(absolute)
        )
        return (
            channels,
            j_bv,
            j_bulk,
            signed,
            combined,
            roundoff_floor,
            physical_threshold,
            phi_face,
            residual_slope_magnitude,
        )

    for direction, data in directional.items():
        for polarity in ("anode", "cathode"):
            mask = data[polarity]
            if not bool(mask.any()):
                continue
            index = torch.nonzero(mask, as_tuple=True)
            batch_index = index[0]
            phi_cell = potential[index]
            conductance = torch.clamp(
                data["sigma"][index] / spacing,
                min=conductivity_floor / max(spacing, 1e-30),
            )
            if polarity == "anode":
                maximum_drop = torch.clamp(voltage - phi_cell, min=0.0)
            else:
                maximum_drop = torch.clamp(
                    phi_cell - cathode_voltage, min=0.0
                )
            forward = maximum_drop > 0.0
            lo = torch.zeros_like(maximum_drop)
            hi = maximum_drop
            gap = torch.where(forward, 0.5 * maximum_drop, torch.zeros_like(maximum_drop))
            unresolved = forward.clone()
            iterations_used = torch.zeros_like(batch_index, dtype=torch.int64)

            for iteration in range(1, maximum_iterations + 1):
                (
                    channels,
                    j_bv,
                    j_bulk,
                    signed,
                    combined,
                    roundoff_floor,
                    physical_threshold,
                    phi_face,
                    residual_slope_magnitude,
                ) = evaluate_gap(
                    gap, index, polarity, conductance, phi_cell, maximum_drop, forward
                )
                newly_resolved = (
                    unresolved
                    & (iteration >= minimum_iterations)
                    & (combined <= 1.0)
                )
                iterations_used = torch.where(
                    newly_resolved,
                    torch.full_like(iterations_used, iteration),
                    iterations_used,
                )
                unresolved = unresolved & ~newly_resolved
                if not bool(unresolved.any()):
                    break

                # R(gap)=j_BV-G*(maximum_drop-gap) increases for both polarities.
                move_lo = unresolved & (signed < 0.0)
                move_hi = unresolved & ~move_lo
                lo = torch.where(move_lo, gap, lo)
                hi = torch.where(move_hi, gap, hi)
                gap = torch.where(unresolved, 0.5 * (lo + hi), gap)

            # Safeguarded Newton polishing in the same small drop variable.
            for _ in range(newton_iterations):
                if not bool(unresolved.any()):
                    break
                (
                    channels,
                    j_bv,
                    j_bulk,
                    signed,
                    combined,
                    roundoff_floor,
                    physical_threshold,
                    phi_face,
                    residual_slope_magnitude,
                ) = evaluate_gap(
                    gap, index, polarity, conductance, phi_cell, maximum_drop, forward
                )
                derivative = residual_slope_magnitude
                safe = (
                    unresolved
                    & torch.isfinite(derivative)
                    & (derivative.abs() > slope_floor)
                )
                candidate = gap - signed / torch.where(
                    safe, derivative, torch.ones_like(derivative)
                )
                candidate = torch.minimum(torch.maximum(candidate, lo), hi)
                gap = torch.where(safe, candidate, gap)
                (
                    _,
                    _,
                    _,
                    signed_after,
                    combined_after,
                    _,
                    _,
                    _,
                    _,
                ) = evaluate_gap(
                    gap, index, polarity, conductance, phi_cell, maximum_drop, forward
                )
                move_lo = unresolved & (signed_after < 0.0)
                lo = torch.where(move_lo, gap, lo)
                hi = torch.where(unresolved & ~move_lo, gap, hi)
                newly_resolved = unresolved & (combined_after <= 1.0)
                iterations_used = torch.where(
                    newly_resolved,
                    torch.full_like(iterations_used, maximum_iterations + 1),
                    iterations_used,
                )
                unresolved = unresolved & ~newly_resolved

            iterations_used = torch.where(
                forward & (iterations_used == 0),
                torch.full_like(
                    iterations_used, maximum_iterations + newton_iterations
                ),
                iterations_used,
            )
            (
                channels,
                j_bv,
                j_bulk,
                signed,
                combined,
                roundoff_floor,
                physical_threshold,
                phi_face,
                residual_slope_magnitude,
            ) = evaluate_gap(
                gap, index, polarity, conductance, phi_cell, maximum_drop, forward
            )
            absolute_residual = torch.where(
                forward, signed.abs(), torch.zeros_like(signed)
            )
            relative_residual = absolute_residual / torch.clamp(
                torch.maximum(j_bv.abs(), j_bulk.abs()), min=1.0
            )

            max_local_residual = torch.maximum(
                max_local_residual,
                _batch_scatter_max(absolute_residual, batch_index, batch_size),
            )
            max_local_relative_residual = torch.maximum(
                max_local_relative_residual,
                _batch_scatter_max(relative_residual, batch_index, batch_size),
            )
            max_local_combined_residual = torch.maximum(
                max_local_combined_residual,
                _batch_scatter_max(combined, batch_index, batch_size),
            )
            max_local_roundoff_floor = torch.maximum(
                max_local_roundoff_floor,
                _batch_scatter_max(
                    torch.where(
                        forward, roundoff_floor, torch.zeros_like(roundoff_floor)
                    ),
                    batch_index,
                    batch_size,
                ),
            )
            roundoff_limited = (
                forward
                & (roundoff_floor > physical_threshold)
                & (absolute_residual <= roundoff_floor)
            )
            roundoff_limited_face_count += _batch_scatter_sum(
                roundoff_limited.to(torch.int64), batch_index, batch_size
            )
            max_local_iterations = torch.maximum(
                max_local_iterations,
                _batch_scatter_max(
                    iterations_used.to(potential.dtype), batch_index, batch_size
                ).to(torch.int64),
            )
            unresolved_face_count += _batch_scatter_sum(
                unresolved.to(torch.int64), batch_index, batch_size
            )

            j_water = torch.where(
                forward, channels["water"], torch.zeros_like(phi_face)
            )
            j_lp = torch.where(
                forward, channels["lp"], torch.zeros_like(phi_face)
            )
            j_bv = j_water + j_lp

            kinetic_slope = channels["dWater_dEta"] + channels["dLP_dEta"]
            larger = torch.maximum(kinetic_slope, conductance)
            smaller = torch.minimum(kinetic_slope, conductance)
            safe_larger = torch.where(
                larger > slope_floor, larger, torch.ones_like(larger)
            )
            gauge_sensitivity = torch.where(
                forward & (larger > slope_floor),
                smaller / (1.0 + smaller / safe_larger),
                torch.zeros_like(kinetic_slope),
            )

            face_grid = indexed_contribution(potential, index, phi_face)
            face_values[direction] = torch.where(
                mask, face_grid, face_values[direction]
            )
            face_ones = indexed_contribution(
                potential, index, torch.ones_like(phi_face)
            )
            if polarity == "anode":
                anode_face_count = anode_face_count + face_ones
                integrated_anode = integrated_anode + _batch_scatter_sum(
                    j_bv * face_area, batch_index, batch_size
                )
                integrated_anode_gauge_sensitivity = (
                    integrated_anode_gauge_sensitivity
                    + _batch_scatter_sum(
                        gauge_sensitivity * face_area, batch_index, batch_size
                    )
                )
                channel_names = (
                    (
                        "waterAnode",
                        "etaWaterAnodeTimesJ",
                        "waterAnodeJWeight",
                        "water",
                        "etaWater",
                    ),
                    (
                        "lpAnode",
                        "etaLPAnodeTimesJ",
                        "lpAnodeJWeight",
                        "lp",
                        "etaLP",
                    ),
                )
            else:
                cathode_face_count = cathode_face_count + face_ones
                integrated_cathode = integrated_cathode + _batch_scatter_sum(
                    j_bv * face_area, batch_index, batch_size
                )
                integrated_cathode_gauge_sensitivity = (
                    integrated_cathode_gauge_sensitivity
                    + _batch_scatter_sum(
                        gauge_sensitivity * face_area, batch_index, batch_size
                    )
                )
                channel_names = (
                    (
                        "waterCathode",
                        "etaWaterCathodeTimesJ",
                        "waterCathodeJWeight",
                        "water",
                        "etaWater",
                    ),
                    (
                        "lpCathode",
                        "etaLPCathodeTimesJ",
                        "lpCathodeJWeight",
                        "lp",
                        "etaLP",
                    ),
                )
            for (
                current_name,
                eta_name,
                weight_name,
                source_current,
                source_eta,
            ) in channel_names:
                current = torch.where(
                    forward,
                    channels[source_current],
                    torch.zeros_like(phi_face),
                )
                eta = channels[source_eta]
                sums[current_name] = sums[current_name] + indexed_contribution(
                    potential, index, current
                )
                sums[eta_name] = sums[eta_name] + indexed_contribution(
                    potential, index, current * eta
                )
                sums[weight_name] = sums[weight_name] + indexed_contribution(
                    potential, index, current
                )

    def average_current(name: str, face_count: torch.Tensor) -> torch.Tensor:
        return torch.where(
            face_count > 0,
            sums[name] / torch.clamp(face_count, min=1.0),
            torch.zeros_like(face_count),
        )

    def weighted_eta(numerator: str, weight: str) -> torch.Tensor:
        return torch.where(
            sums[weight] > current_floor,
            sums[numerator] / torch.clamp(sums[weight], min=current_floor),
            torch.zeros_like(potential),
        )

    reaction = {
        "jWaterAnode_A_per_m2": average_current(
            "waterAnode", anode_face_count
        ),
        "jLPAnode_A_per_m2": average_current("lpAnode", anode_face_count),
        "jWaterCathode_A_per_m2": average_current(
            "waterCathode", cathode_face_count
        ),
        "jLPCathode_A_per_m2": average_current(
            "lpCathode", cathode_face_count
        ),
        "etaWaterAnode_V": weighted_eta(
            "etaWaterAnodeTimesJ", "waterAnodeJWeight"
        ),
        "etaLPAnode_V": weighted_eta("etaLPAnodeTimesJ", "lpAnodeJWeight"),
        "etaWaterCathode_V": weighted_eta(
            "etaWaterCathodeTimesJ", "waterCathodeJWeight"
        ),
        "etaLPCathode_V": weighted_eta(
            "etaLPCathodeTimesJ", "lpCathodeJWeight"
        ),
        "anodeFaceCount": anode_face_count,
        "cathodeFaceCount": cathode_face_count,
        "rawAnodeCurrent_A": integrated_anode,
        "rawCathodeCurrent_A": integrated_cathode,
        "rawAnodeGaugeSensitivity_A_per_V": integrated_anode_gauge_sensitivity,
        "rawCathodeGaugeSensitivity_A_per_V": integrated_cathode_gauge_sensitivity,
        "currentDifferenceGaugeDerivative_A_per_V": -(
            integrated_anode_gauge_sensitivity
            + integrated_cathode_gauge_sensitivity
        ),
        "maximumLocalRobinResidual_A_per_m2": max_local_residual,
        "maximumLocalRobinRelativeResidual": max_local_relative_residual,
        "maximumLocalRobinCombinedResidual": max_local_combined_residual,
        "maximumLocalRobinRoundoffFloor_A_per_m2": max_local_roundoff_floor,
        "roundoffLimitedLocalRobinFaceCount": roundoff_limited_face_count,
        "maximumLocalRobinIterations": max_local_iterations,
        "unresolvedLocalRobinFaceCount": unresolved_face_count,
        "localRobinAllFacesConverged": unresolved_face_count == 0,
        "localRobinSolveVariable": "electrode_to_interface_gap_overpotential_scale_V",
        "context": context,
    }
    reaction["jAnode_A_per_m2"] = (
        reaction["jWaterAnode_A_per_m2"] + reaction["jLPAnode_A_per_m2"]
    )
    reaction["jCathode_A_per_m2"] = (
        reaction["jWaterCathode_A_per_m2"] + reaction["jLPCathode_A_per_m2"]
    )
    reaction["anodeZone"] = anode_face_count > 0
    reaction["cathodeZone"] = cathode_face_count > 0
    return {"faceFixedValues": face_values, "reaction": reaction}


def _balance_nonlinear_robin_gauge(
    potential: torch.Tensor,
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float,
    spacing: float,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    """Balance integrated anode/cathode BV currents by a uniform gauge shift.

    The bulk Robin problem fixes electrolyte-potential *differences*, but a
    spatially uniform offset is not determined by the Neumann flux equations.
    Butler--Volmer kinetics do depend on that absolute offset.  Therefore the
    physically admissible gauge is the scalar offset satisfying

        I_anode(offset) - I_cathode(offset) = 0.

    Each interface face is first eliminated by ``_implicit_bv_robin_faces``.
    A safeguarded Newton step uses the exact series BV/bulk sensitivity
    k*G/(k+G); a bracketed midpoint is used on plateaus or whenever Newton would
    leave the physical bracket.  This removes the former deadlock
    dphi=dI=0, balance=1 without relaxing charge conservation.
    """
    robin = config["interface"].get("nonlinearRobin", {})
    balance_tolerance = max(
        0.0, float(robin.get("currentBalanceTolerance", 5e-3))
    )
    absolute_tolerance = max(
        0.0, float(robin.get("currentBalanceAbsoluteTolerance_A", 1e-12))
    )
    maximum_iterations = max(
        1,
        int(
            robin.get(
                "gaugeMaximumIterations",
                min(int(robin.get("gaugeBisectionIterations", 60)), 24),
            )
        ),
    )
    minimum_iterations = max(1, int(robin.get("gaugeMinimumIterations", 1)))
    under_relaxation = min(
        max(float(robin.get("gaugeUnderRelaxation", 1.0)), 1e-3), 1.0
    )
    maximum_step = max(
        1e-6,
        float(
            robin.get(
                "gaugeMaximumStep_V",
                max(5.0, 0.25 * abs(voltage - float(config["electrical"]["cathodeVoltage_V"]))),
            )
        ),
    )
    derivative_floor = max(
        0.0, float(robin.get("gaugeDerivativeFloor_A_per_V", 1e-14))
    )
    margin = max(0.0, float(robin.get("gaugeBracketMargin_V", 25.0)))
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])

    prop = geometry.propellant
    weight = prop.to(potential.dtype)
    count = torch.clamp(weight.sum(dim=(-2, -1)), min=1.0)
    initial_offset = (potential * weight).sum(dim=(-2, -1)) / count
    relative = torch.where(
        prop,
        potential - initial_offset[:, None, None],
        potential,
    )
    relative_min = torch.where(
        prop, relative, torch.full_like(relative, torch.inf)
    ).amin(dim=(-2, -1))
    relative_max = torch.where(
        prop, relative, torch.full_like(relative, -torch.inf)
    ).amax(dim=(-2, -1))

    # These bounds guarantee, respectively, a cathode-inactive/anode-active
    # state and an anode-inactive/cathode-active state, apart from a possible
    # zero-current plateau.  They therefore bracket the monotone global root.
    low = torch.full_like(initial_offset, cathode_voltage - margin) - relative_max
    high = torch.full_like(initial_offset, voltage + margin) - relative_min
    offset = torch.minimum(torch.maximum(initial_offset, low), high)

    def evaluate(
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        actual = _apply_electrolyte_offset(
            relative, value, geometry, voltage, cathode_voltage
        )
        boundary = _implicit_bv_robin_faces(
            actual,
            state,
            transport,
            geometry,
            config,
            composition,
            voltage,
            spacing,
        )
        reaction = boundary["reaction"]
        ia = reaction["rawAnodeCurrent_A"]
        ic = reaction["rawCathodeCurrent_A"]
        sensitivity = (
            reaction["rawAnodeGaugeSensitivity_A_per_V"]
            + reaction["rawCathodeGaugeSensitivity_A_per_V"]
        )
        return actual, boundary, ia, ic, sensitivity

    actual, boundary, ia, ic, sensitivity = evaluate(offset)
    iterations = torch.ones(
        geometry.batch_size,
        device=potential.device,
        dtype=torch.int64,
    )
    converged = torch.zeros(
        geometry.batch_size, device=potential.device, dtype=torch.bool
    )
    absolute_difference = torch.abs(ia - ic)
    current_scale = torch.maximum(ia.abs(), ic.abs())
    threshold = torch.maximum(
        balance_tolerance * current_scale,
        torch.full_like(current_scale, absolute_tolerance),
    )

    for iteration in range(1, maximum_iterations + 1):
        difference = ia - ic
        absolute_difference = difference.abs()
        current_scale = torch.maximum(ia.abs(), ic.abs())
        threshold = torch.maximum(
            balance_tolerance * current_scale,
            torch.full_like(current_scale, absolute_tolerance),
        )
        newly_converged = (
            (iteration >= minimum_iterations) & (absolute_difference <= threshold)
        )
        converged = converged | newly_converged
        iterations = torch.where(
            (~converged) | newly_converged,
            torch.full_like(iterations, iteration),
            iterations,
        )
        unresolved = ~converged
        if not bool(torch.any(unresolved)):
            break

        # f=Ia-Ic is monotone decreasing with a positive electrolyte offset.
        low = torch.where(unresolved & (difference > 0.0), offset, low)
        high = torch.where(unresolved & (difference < 0.0), offset, high)
        midpoint = 0.5 * (low + high)

        safe_derivative = (
            unresolved
            & torch.isfinite(sensitivity)
            & (sensitivity > derivative_floor)
        )
        newton = offset + difference / torch.where(
            safe_derivative, sensitivity, torch.ones_like(sensitivity)
        )
        newton_inside = (
            safe_derivative
            & torch.isfinite(newton)
            & (newton > low)
            & (newton < high)
        )
        target = torch.where(newton_inside, newton, midpoint)
        step = torch.clamp(target - offset, min=-maximum_step, max=maximum_step)
        candidate = offset + under_relaxation * step
        candidate = torch.minimum(torch.maximum(candidate, low), high)

        # Avoid a finite-precision stall at a bracket endpoint.
        scale = torch.maximum(offset.abs(), torch.ones_like(offset))
        stalled = unresolved & (
            torch.abs(candidate - offset)
            <= 8.0 * torch.finfo(offset.dtype).eps * scale
        )
        candidate = torch.where(stalled, midpoint, candidate)
        offset = torch.where(unresolved, candidate, offset)
        actual, boundary, ia, ic, sensitivity = evaluate(offset)

    difference = ia - ic
    absolute_difference = difference.abs()
    current_scale = torch.maximum(ia.abs(), ic.abs())
    threshold = torch.maximum(
        balance_tolerance * current_scale,
        torch.full_like(current_scale, absolute_tolerance),
    )
    mismatch = absolute_difference / torch.clamp(current_scale, min=1e-12)
    combined_residual = absolute_difference / torch.clamp(
        threshold, min=torch.finfo(absolute_difference.dtype).tiny
    )
    converged = converged | (combined_residual <= 1.0)
    return actual, boundary, {
        "converged": converged,
        "iterations": iterations,
        "offset_V": offset,
        "bracketWidth_V": high - low,
        "anodeCurrent_A": ia,
        "cathodeCurrent_A": ic,
        "absoluteCurrentDifference_A": absolute_difference,
        "currentBalanceMismatch": mismatch,
        "currentBalanceCombinedResidual": combined_residual,
        "currentBalanceThreshold_A": threshold,
        "currentDifferenceGaugeSensitivity_A_per_V": sensitivity,
    }

def _solve_nonlinear_robin_potential(
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float,
    spacing: float,
    *,
    static: bool,
) -> tuple[torch.Tensor, Any, dict[str, Any]]:
    """Solve bulk potential, the BV faces, and the global current gauge."""
    robin = config["interface"].get("nonlinearRobin", {})
    maximum_iterations = max(2, int(robin.get("maximumIterations", 120)))
    minimum_key = "minimumIterationsStatic" if static else "minimumIterationsCoupled"
    minimum_iterations = max(
        1, int(robin.get(minimum_key, robin.get("minimumIterations", 2)))
    )
    potential_tolerance = float(robin.get("potentialTolerance_V", 0.01))
    current_tolerance = float(robin.get("relativeReactionCurrentTolerance", 5e-3))
    balance_tolerance = float(robin.get("currentBalanceTolerance", 5e-3))
    relaxation = float(
        robin.get("potentialUnderRelaxation", robin.get("underRelaxation", 0.35))
    )
    relaxation = min(max(relaxation, 1e-3), 1.0)
    fail = bool(robin.get("failOnNonConvergence", True))
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])

    # Warm start from the previous nonlinear state.  Only a spatially uniform
    # initial state receives one legacy Dirichlet solve as an initial shape.
    current = state["potential"].clone()
    prop = geometry.propellant
    prop_count = torch.clamp(prop.to(current.dtype).sum(dim=(-2, -1)), min=1.0)
    prop_mean = (current * prop.to(current.dtype)).sum(dim=(-2, -1)) / prop_count
    prop_variance = (
        ((current - prop_mean[:, None, None]) ** 2) * prop.to(current.dtype)
    ).sum(dim=(-2, -1)) / prop_count
    needs_initialization = torch.sqrt(torch.clamp(prop_variance, min=0.0)) < 1e-10
    if bool(needs_initialization.any()):
        legacy_config = dict(config)
        legacy_interface = dict(config["interface"])
        legacy_interface["boundaryCouplingModel"] = "legacy_posthoc"
        legacy_config["interface"] = legacy_interface
        initialized, diagnostics, _ = solve_potential(
            current,
            state["cation"], state["anion"],
            transport["Dcation"], transport["Danion"], transport["sigmaTotal"],
            geometry, legacy_config, voltage, spacing, static=static,
        )
        current = torch.where(needs_initialization[:, None, None], initialized, current)
    else:
        diagnostics = None

    # The legacy Dirichlet initialisation gives a useful spatial shape but not
    # the absolute electrolyte gauge required by BV kinetics.  Balance that
    # gauge before the first bulk Robin solve.
    current, boundary_cache, gauge_diag = _balance_nonlinear_robin_gauge(
        current,
        state,
        transport,
        geometry,
        config,
        composition,
        voltage,
        spacing,
    )
    gauge_iterations_total = gauge_diag["iterations"].clone()

    # The gauge-balanced boundary provides a valid current at the warm-start
    # state.  Comparing the first updated current against it permits a single
    # outer iteration during slowly varying transient steps, while the initial
    # static-shaped solve still continues until all residual tests pass.
    previous_total = 0.5 * (
        gauge_diag["anodeCurrent_A"] + gauge_diag["cathodeCurrent_A"]
    )
    potential_change = torch.full(
        (geometry.batch_size,), torch.inf, device=current.device, dtype=current.dtype
    )
    current_change = torch.full_like(potential_change, torch.inf)
    current_balance = gauge_diag["currentBalanceMismatch"]
    current_balance_combined = gauge_diag["currentBalanceCombinedResidual"]
    local_combined_residual = torch.full_like(potential_change, torch.inf)
    converged = torch.zeros(geometry.batch_size, device=current.device, dtype=torch.bool)
    final_boundary: dict[str, Any] | None = boundary_cache
    used_iterations = 0

    for iteration in range(1, maximum_iterations + 1):
        boundary = boundary_cache
        proposed, diagnostics, _ = solve_potential(
            current,
            state["cation"], state["anion"],
            transport["Dcation"], transport["Danion"], transport["sigmaTotal"],
            geometry, config, voltage, spacing, static=static,
            face_fixed_values=boundary["faceFixedValues"],
        )
        updated = torch.where(
            prop,
            current + relaxation * (proposed - current),
            current,
        )
        updated = torch.where(
            geometry.anode,
            torch.as_tensor(voltage, device=current.device, dtype=current.dtype),
            updated,
        )
        updated = torch.where(
            geometry.cathode,
            torch.as_tensor(cathode_voltage, device=current.device, dtype=current.dtype),
            updated,
        )

        # Re-establish the physically admissible absolute electrolyte gauge for
        # the updated bulk-potential shape.  This is the step missing in v7.7.2:
        # without it the outer iteration can freeze at dphi=dI=0 while one
        # electrode carries zero current and balance remains exactly one.
        updated, final_boundary, gauge_diag = _balance_nonlinear_robin_gauge(
            updated,
            state,
            transport,
            geometry,
            config,
            composition,
            voltage,
            spacing,
        )
        boundary_cache = final_boundary
        gauge_iterations_total = gauge_iterations_total + gauge_diag["iterations"]
        reaction_boundary = final_boundary["reaction"]
        ia = reaction_boundary["rawAnodeCurrent_A"]
        ic = reaction_boundary["rawCathodeCurrent_A"]
        total = 0.5 * (ia + ic)
        potential_change = torch.amax(
            torch.where(prop, torch.abs(updated - current), torch.zeros_like(updated)),
            dim=(-2, -1),
        )
        current_change = torch.abs(total - previous_total) / torch.clamp(
            torch.maximum(total.abs(), previous_total.abs()), min=1e-12
        )
        current_balance = gauge_diag["currentBalanceMismatch"]
        current_balance_combined = gauge_diag["currentBalanceCombinedResidual"]
        local_combined_residual = reaction_boundary[
            "maximumLocalRobinCombinedResidual"
        ]
        linear_converged = (
            diagnostics.converged if diagnostics is not None else torch.ones_like(converged)
        )
        local_converged = reaction_boundary["localRobinAllFacesConverged"]
        gauge_converged = gauge_diag["converged"]
        converged = (
            (iteration >= minimum_iterations)
            & (potential_change <= potential_tolerance)
            & (current_change <= current_tolerance)
            & (current_balance_combined <= 1.0)
            & gauge_converged
            & (local_combined_residual <= 1.0)
            & local_converged
            & linear_converged
        )
        current = updated
        previous_total = total
        used_iterations = iteration
        if bool(torch.all(converged)):
            break

    if final_boundary is None:
        raise RuntimeError("Nonlinear Robin iteration did not execute")
    reaction_boundary = final_boundary["reaction"]
    if fail and not bool(torch.all(converged)):
        bad = torch.nonzero(~converged).flatten().detach().cpu().tolist()
        mask = ~converged
        raise RuntimeError(
            "Nonlinear Butler-Volmer Robin potential solve did not converge for "
            f"batch indices {bad}; dphi={potential_change[mask].detach().cpu().tolist()}, "
            f"dI={current_change[mask].detach().cpu().tolist()}, "
            f"balance={current_balance[mask].detach().cpu().tolist()}, "
            "balanceCombined="
            f"{current_balance_combined[mask].detach().cpu().tolist()}, "
            f"Ia_A={gauge_diag['anodeCurrent_A'][mask].detach().cpu().tolist()}, "
            f"Ic_A={gauge_diag['cathodeCurrent_A'][mask].detach().cpu().tolist()}, "
            f"gaugeOffset_V={gauge_diag['offset_V'][mask].detach().cpu().tolist()}, "
            f"gaugeIterations={gauge_diag['iterations'][mask].detach().cpu().tolist()}, "
            f"gaugeBracketWidth_V={gauge_diag['bracketWidth_V'][mask].detach().cpu().tolist()}, "
            "localRobinAbsoluteResidual="
            f"{reaction_boundary['maximumLocalRobinResidual_A_per_m2'][mask].detach().cpu().tolist()}, "
            "localRobinRelativeResidual="
            f"{reaction_boundary['maximumLocalRobinRelativeResidual'][mask].detach().cpu().tolist()}, "
            "localRobinCombinedResidual="
            f"{local_combined_residual[mask].detach().cpu().tolist()}, "
            "localRobinRoundoffFloor="
            f"{reaction_boundary['maximumLocalRobinRoundoffFloor_A_per_m2'][mask].detach().cpu().tolist()}, "
            "roundoffLimitedLocalFaces="
            f"{reaction_boundary['roundoffLimitedLocalRobinFaceCount'][mask].detach().cpu().tolist()}, "
            "unresolvedLocalFaces="
            f"{reaction_boundary['unresolvedLocalRobinFaceCount'][mask].detach().cpu().tolist()}"
        )
    return current, diagnostics, {
        "nonlinearRobinIterations": torch.full(
            (geometry.batch_size,), used_iterations, device=current.device, dtype=torch.int64
        ),
        "nonlinearRobinConverged": converged,
        "nonlinearRobinPotentialChange_V": potential_change,
        "nonlinearRobinRelativeCurrentChange": current_change,
        "nonlinearRobinCurrentBalanceMismatch": current_balance,
        "nonlinearRobinCurrentBalanceCombinedResidual": current_balance_combined,
        "nonlinearRobinGaugeConverged": gauge_diag["converged"],
        "nonlinearRobinGaugeIterations": gauge_diag["iterations"],
        "nonlinearRobinGaugeTotalIterations": gauge_iterations_total,
        "nonlinearRobinGaugeOffset_V": gauge_diag["offset_V"],
        "nonlinearRobinGaugeBracketWidth_V": gauge_diag["bracketWidth_V"],
        "nonlinearRobinAnodeCurrent_A": gauge_diag["anodeCurrent_A"],
        "nonlinearRobinCathodeCurrent_A": gauge_diag["cathodeCurrent_A"],
        "nonlinearRobinAbsoluteCurrentDifference_A": gauge_diag[
            "absoluteCurrentDifference_A"
        ],
        "nonlinearRobinCurrentDifferenceGaugeSensitivity_A_per_V": gauge_diag[
            "currentDifferenceGaugeSensitivity_A_per_V"
        ],
        "nonlinearRobinMaximumLocalResidual_A_per_m2": reaction_boundary[
            "maximumLocalRobinResidual_A_per_m2"
        ],
        "nonlinearRobinMaximumLocalRelativeResidual": reaction_boundary[
            "maximumLocalRobinRelativeResidual"
        ],
        "nonlinearRobinMaximumLocalCombinedResidual": local_combined_residual,
        "nonlinearRobinMaximumLocalRoundoffFloor_A_per_m2": reaction_boundary[
            "maximumLocalRobinRoundoffFloor_A_per_m2"
        ],
        "nonlinearRobinRoundoffLimitedLocalFaceCount": reaction_boundary[
            "roundoffLimitedLocalRobinFaceCount"
        ],
        "nonlinearRobinMaximumLocalIterations": reaction_boundary[
            "maximumLocalRobinIterations"
        ],
        "nonlinearRobinUnresolvedLocalFaceCount": reaction_boundary[
            "unresolvedLocalRobinFaceCount"
        ],
        "nonlinearRobinBoundaryReaction": reaction_boundary,
        "nonlinearRobinDiscretization": (
            "adaptive_implicit_face_bv_robin_with_exact_global_gauge_balance"
        ),
    }

def _solve_surface_overlay_potential(
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float,
    spacing: float,
    *,
    static: bool,
) -> tuple[torch.Tensor, Any, dict[str, Any]]:
    """Couple full-footprint surface BV sources to the bulk SPD equation."""
    if bool(torch.any(geometry.fixed)) or not bool(torch.all(geometry.propellant)):
        raise ValueError(
            "surface_overlay_bv requires fixed=false and propellant=true on the full domain"
        )
    robin = config["interface"].get("nonlinearRobin", {})
    maximum_iterations = max(2, int(robin.get("maximumIterations", 120)))
    minimum_key = "minimumIterationsStatic" if static else "minimumIterationsCoupled"
    minimum_iterations = max(
        1, int(robin.get(minimum_key, robin.get("minimumIterations", 2)))
    )
    potential_tolerance = float(robin.get("potentialTolerance_V", 0.01))
    current_tolerance = float(robin.get("relativeReactionCurrentTolerance", 5e-3))
    relaxation = float(
        robin.get("potentialUnderRelaxation", robin.get("underRelaxation", 0.35))
    )
    if not math.isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
        raise ValueError("potentialUnderRelaxation must lie in (0, 1]")
    fail = bool(robin.get("failOnNonConvergence", True))
    current = state["potential"].clone()
    boundary = _implicit_bv_surface_contacts(
        current, state, transport, geometry, config, composition, voltage, spacing
    )
    reaction_boundary = boundary["reaction"]
    previous_total = 0.5 * (
        reaction_boundary["rawAnodeCurrent_A"]
        + reaction_boundary["rawCathodeCurrent_A"]
    )
    potential_change = torch.full(
        (geometry.batch_size,), torch.inf, device=current.device, dtype=current.dtype
    )
    current_change = torch.full_like(potential_change, torch.inf)
    current_balance = reaction_boundary["currentBalanceMismatch"]
    current_balance_combined = reaction_boundary["currentBalanceCombinedResidual"]
    local_combined = reaction_boundary["maximumLocalRobinCombinedResidual"]
    converged = torch.zeros(
        geometry.batch_size, device=current.device, dtype=torch.bool
    )
    diagnostics = None
    frozen_diagnostics = None
    outer_iterations = torch.zeros(
        geometry.batch_size, device=current.device, dtype=torch.int64
    )

    def retain_latest_active(latest: Any, prior: Any, active: torch.Tensor) -> Any:
        """Update only active lanes in nested surface-solver diagnostics."""
        if latest is None:
            return prior
        if prior is None:
            return latest
        if isinstance(latest, dict) and isinstance(prior, dict):
            return {
                key: retain_latest_active(value, prior[key], active)
                for key, value in latest.items()
            }
        if (
            isinstance(latest, torch.Tensor)
            and isinstance(prior, torch.Tensor)
            and latest.ndim > 0
            and latest.shape[0] == geometry.batch_size
            and prior.shape == latest.shape
        ):
            mask = active.reshape(
                (geometry.batch_size,) + (1,) * (latest.ndim - 1)
            )
            return torch.where(mask, latest, prior)
        return latest

    for iteration in range(1, maximum_iterations + 1):
        working = ~converged
        if not bool(torch.any(working)):
            break
        proposed, diagnostics, _ = solve_potential(
            current,
            state["cation"],
            state["anion"],
            transport["Dcation"],
            transport["Danion"],
            transport["sigmaTotal"],
            geometry,
            config,
            voltage,
            spacing,
            static=static,
            robin_linearization=boundary["surfaceLinearization"],
        )
        candidate = current + relaxation * (proposed - current)
        working_field = working[:, None, None]
        updated = torch.where(working_field, candidate, current)
        latest_boundary = _implicit_bv_surface_contacts(
            updated, state, transport, geometry, config, composition, voltage, spacing
        )
        boundary = retain_latest_active(latest_boundary, boundary, working)
        reaction_boundary = boundary["reaction"]
        ia = reaction_boundary["rawAnodeCurrent_A"]
        ic = reaction_boundary["rawCathodeCurrent_A"]
        total = 0.5 * (ia + ic)
        latest_potential_change = torch.amax(
            torch.abs(updated - current), dim=(-2, -1)
        )
        latest_current_change = torch.abs(total - previous_total) / torch.clamp(
            torch.maximum(total.abs(), previous_total.abs()), min=1e-12
        )
        potential_change = torch.where(
            working, latest_potential_change, potential_change
        )
        current_change = torch.where(
            working, latest_current_change, current_change
        )
        current_balance = reaction_boundary["currentBalanceMismatch"]
        current_balance_combined = reaction_boundary["currentBalanceCombinedResidual"]
        local_combined = reaction_boundary["maximumLocalRobinCombinedResidual"]
        linear_converged = (
            diagnostics.converged
            if diagnostics is not None
            else torch.ones_like(converged)
        )
        local_converged = reaction_boundary["localRobinAllFacesConverged"]
        reached = (
            (iteration >= minimum_iterations)
            & (potential_change <= potential_tolerance)
            & (current_change <= current_tolerance)
            & (current_balance_combined <= 1.0)
            & (local_combined <= 1.0)
            & local_converged
            & linear_converged
        )
        converged = converged | (working & reached)
        current = updated
        previous_total = torch.where(working, total, previous_total)
        outer_iterations = torch.where(
            working, torch.full_like(outer_iterations, iteration), outer_iterations
        )
        if frozen_diagnostics is None:
            frozen_diagnostics = diagnostics
        else:
            frozen_diagnostics = type(diagnostics)(
                iterations=retain_latest_active(
                    diagnostics.iterations, frozen_diagnostics.iterations, working
                ),
                relative_residual=retain_latest_active(
                    diagnostics.relative_residual,
                    frozen_diagnostics.relative_residual,
                    working,
                ),
                converged=retain_latest_active(
                    diagnostics.converged, frozen_diagnostics.converged, working
                ),
                method=diagnostics.method,
                restarts=retain_latest_active(
                    diagnostics.restarts, frozen_diagnostics.restarts, working
                ),
                refinement_rounds=retain_latest_active(
                    diagnostics.refinement_rounds,
                    frozen_diagnostics.refinement_rounds,
                    working,
                ),
                fallback_used=retain_latest_active(
                    diagnostics.fallback_used,
                    frozen_diagnostics.fallback_used,
                    working,
                ),
            )
        if bool(torch.all(converged)):
            break

    if diagnostics is None or frozen_diagnostics is None:
        raise RuntimeError("Surface-contact nonlinear iteration did not execute")
    diagnostics = frozen_diagnostics
    if fail and not bool(torch.all(converged)):
        bad_mask = ~converged
        bad = torch.nonzero(bad_mask).flatten().detach().cpu().tolist()
        raise BCCandidateBatchError(
            "Surface-footprint Butler-Volmer solve did not converge for "
            f"batch indices {bad}; "
            f"dphi={potential_change[bad_mask].detach().cpu().tolist()}, "
            f"dI={current_change[bad_mask].detach().cpu().tolist()}, "
            f"balance={current_balance[bad_mask].detach().cpu().tolist()}, "
            f"balanceCombined={current_balance_combined[bad_mask].detach().cpu().tolist()}, "
            f"localCombined={local_combined[bad_mask].detach().cpu().tolist()}",
            bad,
            "surface_bv_nonconvergence",
        )

    zeros_i64 = torch.zeros(
        geometry.batch_size, device=current.device, dtype=torch.int64
    )
    zeros = torch.zeros_like(potential_change)
    return current, diagnostics, {
        "nonlinearRobinIterations": outer_iterations,
        "nonlinearRobinConverged": converged,
        "nonlinearRobinPotentialChange_V": potential_change,
        "nonlinearRobinRelativeCurrentChange": current_change,
        "nonlinearRobinCurrentBalanceMismatch": current_balance,
        "nonlinearRobinCurrentBalanceCombinedResidual": current_balance_combined,
        "nonlinearRobinGaugeConverged": torch.ones_like(converged),
        "nonlinearRobinGaugeIterations": zeros_i64,
        "nonlinearRobinGaugeTotalIterations": zeros_i64,
        "nonlinearRobinGaugeOffset_V": zeros,
        "nonlinearRobinGaugeBracketWidth_V": zeros,
        "nonlinearRobinAnodeCurrent_A": reaction_boundary["rawAnodeCurrent_A"],
        "nonlinearRobinCathodeCurrent_A": reaction_boundary["rawCathodeCurrent_A"],
        "nonlinearRobinAbsoluteCurrentDifference_A": torch.abs(
            reaction_boundary["rawAnodeCurrent_A"]
            - reaction_boundary["rawCathodeCurrent_A"]
        ),
        "nonlinearRobinCurrentDifferenceGaugeSensitivity_A_per_V": zeros,
        "nonlinearRobinMaximumLocalResidual_A_per_m2": reaction_boundary[
            "maximumLocalRobinResidual_A_per_m2"
        ],
        "nonlinearRobinMaximumLocalRelativeResidual": reaction_boundary[
            "maximumLocalRobinRelativeResidual"
        ],
        "nonlinearRobinMaximumLocalCombinedResidual": local_combined,
        "nonlinearRobinMaximumLocalRoundoffFloor_A_per_m2": reaction_boundary[
            "maximumLocalRobinRoundoffFloor_A_per_m2"
        ],
        "nonlinearRobinRoundoffLimitedLocalFaceCount": reaction_boundary[
            "roundoffLimitedLocalRobinFaceCount"
        ],
        "nonlinearRobinMaximumLocalIterations": reaction_boundary[
            "maximumLocalRobinIterations"
        ],
        "nonlinearRobinUnresolvedLocalFaceCount": reaction_boundary[
            "unresolvedLocalRobinFaceCount"
        ],
        "nonlinearRobinBoundaryReaction": reaction_boundary,
        "nonlinearRobinDiscretization": (
            "full_domain_top_surface_footprint_bv_robin_spd_no_posthoc_gauge"
        ),
    }


def interface_reactions(
    state: dict[str, torch.Tensor],
    electrical: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float,
    spacing: float,
    *,
    robin_boundary: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    boundary_model = str(
        config["interface"].get("boundaryCouplingModel", "legacy_posthoc")
    ).lower()
    surface_models = {"surface_overlay_bv", "surface_contact", "surface_bv"}
    robin_models = {"nonlinear_robin", "robin", "bv_robin"} | surface_models
    if boundary_model in surface_models:
        if robin_boundary is None:
            raise ValueError(
                "surface_overlay_bv reactions require the coupled surface boundary solution"
            )
        surface_layer = float(config["geometry"]["surfaceLayerThickness_m"])
        kinetics = {
            "candWaterAnode": robin_boundary["jWaterAnode_A_per_m2"],
            "candWaterCathode": robin_boundary["jWaterCathode_A_per_m2"],
            "candLPAnode": robin_boundary["jLPAnode_A_per_m2"],
            "candLPCathode": robin_boundary["jLPCathode_A_per_m2"],
            "anodeZone": robin_boundary["anodeZone"],
            "cathodeZone": robin_boundary["cathodeZone"],
            "anodeFaceCount": robin_boundary["anodeFaceCount"],
            "cathodeFaceCount": robin_boundary["cathodeFaceCount"],
            "etaWaterAnode": robin_boundary["etaWaterAnode_V"],
            "etaWaterCathode": robin_boundary["etaWaterCathode_V"],
            "etaLPAnode": robin_boundary["etaLPAnode_V"],
            "etaLPCathode": robin_boundary["etaLPCathode_V"],
            "reactionLayerThickness_m": torch.full(
                (geometry.batch_size,), surface_layer,
                device=state["temperature"].device, dtype=state["temperature"].dtype,
            ),
            "channelConfigs": robin_boundary["channelConfigs"],
            "shiftWaterAnode_V": robin_boundary["shiftWaterAnode_V"],
            "shiftWaterCathode_V": robin_boundary["shiftWaterCathode_V"],
            "shiftLPAnode_V": robin_boundary["shiftLPAnode_V"],
            "shiftLPCathode_V": robin_boundary["shiftLPCathode_V"],
            "saltGamma": robin_boundary["saltGamma"],
            "activeArea": robin_boundary["activeArea"],
            "equilibriumPotentialWaterAnode_V": robin_boundary[
                "equilibriumPotentialWaterAnode_V"
            ],
            "equilibriumPotentialWaterCathode_V": robin_boundary[
                "equilibriumPotentialWaterCathode_V"
            ],
            "equilibriumPotentialLPAnode_V": robin_boundary[
                "equilibriumPotentialLPAnode_V"
            ],
            "equilibriumPotentialLPCathode_V": robin_boundary[
                "equilibriumPotentialLPCathode_V"
            ],
            "fastIonFraction": robin_boundary["fastIonFraction"],
        }
    else:
        kinetics = _interface_kinetic_candidates(
            state, transport, geometry, config, composition, voltage, spacing
        )
    cand_aw = kinetics["candWaterAnode"]
    cand_cw = kinetics["candWaterCathode"]
    cand_alp = kinetics["candLPAnode"]
    cand_clp = kinetics["candLPCathode"]
    anode_zone = kinetics["anodeZone"]
    cathode_zone = kinetics["cathodeZone"]
    anode_faces = kinetics["anodeFaceCount"]
    cathode_faces = kinetics["cathodeFaceCount"]
    eta_aw = kinetics["etaWaterAnode"]
    eta_cw = kinetics["etaWaterCathode"]
    eta_alp = kinetics["etaLPAnode"]
    eta_clp = kinetics["etaLPCathode"]
    reaction_layer = float(kinetics["reactionLayerThickness_m"][0].detach().cpu())
    cfgs = kinetics["channelConfigs"]
    zero = torch.zeros_like(state["temperature"])
    current_density_floor = physical_floor(
        config, "currentDensity_A_per_m2", 1e-12
    )
    current_floor = physical_floor(config, "current_A", 1e-15)
    if boundary_model in robin_models and robin_boundary is not None:
        # Use the currents from the converged *face* Robin solve.  These are
        # already mutually consistent with the bulk potential equation, so no
        # bulk-current clipping or global consistency scaling is allowed here.
        water_a = robin_boundary["jWaterAnode_A_per_m2"]
        lp_a = robin_boundary["jLPAnode_A_per_m2"]
        water_c = robin_boundary["jWaterCathode_A_per_m2"]
        lp_c = robin_boundary["jLPCathode_A_per_m2"]
        available_anode = water_a + lp_a
        available_cathode = water_c + lp_c
        anode_faces = robin_boundary["anodeFaceCount"]
        cathode_faces = robin_boundary["cathodeFaceCount"]
        anode_zone = anode_faces > 0
        cathode_zone = cathode_faces > 0
        eta_aw = robin_boundary["etaWaterAnode_V"]
        eta_alp = robin_boundary["etaLPAnode_V"]
        eta_cw = robin_boundary["etaWaterCathode_V"]
        eta_clp = robin_boundary["etaLPCathode_V"]
    elif boundary_model in robin_models:
        # Fallback only for direct unit-level calls that do not supply the
        # nonlinear face solution.  Production solve_electrical_and_reaction
        # always provides robin_boundary.
        water_a, lp_a, water_c, lp_c = cand_aw, cand_alp, cand_cw, cand_clp
        available_anode = water_a + lp_a
        available_cathode = water_c + lp_c
    else:
        available_anode, available_cathode, _, _ = interface_normal_current(
            electrical, geometry, config["interface"].get("currentDirectionModel", "signed_normal")
        )
        total_a = cand_aw + cand_alp
        total_c = cand_cw + cand_clp
        scale_a_local = torch.clamp(available_anode / torch.clamp(total_a, min=current_density_floor), max=1.0)
        scale_c_local = torch.clamp(available_cathode / torch.clamp(total_c, min=current_density_floor), max=1.0)
        scale_a_local = torch.where(anode_zone, scale_a_local, zero)
        scale_c_local = torch.where(cathode_zone, scale_c_local, zero)
        water_a, lp_a = cand_aw * scale_a_local, cand_alp * scale_a_local
        water_c, lp_c = cand_cw * scale_c_local, cand_clp * scale_c_local

    face_area = (
        spacing * spacing
        if boundary_model in surface_models
        else spacing * float(config["geometry"]["surfaceLayerThickness_m"])
    )
    raw_a = ((water_a + lp_a) * anode_faces).sum(dim=(-2, -1)) * face_area
    raw_c = ((water_c + lp_c) * cathode_faces).sum(dim=(-2, -1)) * face_area
    if boundary_model in robin_models:
        if robin_boundary is not None:
            raw_a = robin_boundary["rawAnodeCurrent_A"]
            raw_c = robin_boundary["rawCathodeCurrent_A"]
        cell_current = 0.5 * (raw_a + raw_c)
        consistency_scale = torch.ones_like(cell_current)
    else:
        cell_current = torch.minimum(raw_a, raw_c)
        scale_a = torch.clamp(cell_current / torch.clamp(raw_a, min=current_floor), max=1.0)
        scale_c = torch.clamp(cell_current / torch.clamp(raw_c, min=current_floor), max=1.0)
        water_a, lp_a = water_a * scale_a[:, None, None], lp_a * scale_a[:, None, None]
        water_c, lp_c = water_c * scale_c[:, None, None], lp_c * scale_c[:, None, None]

    q_aw = _finish_heat(water_a, eta_aw, cfgs["water_a"], config, reaction_layer)
    q_cw = _finish_heat(water_c, eta_cw, cfgs["water_c"], config, reaction_layer)
    q_alp = _finish_heat(lp_a, eta_alp, cfgs["lp_a"], config, reaction_layer)
    q_clp = _finish_heat(lp_c, eta_clp, cfgs["lp_c"], config, reaction_layer)
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    water_sink = (
        float(cfgs["water_a"]["waterStoichiometry_mol_per_molElectron"]) * water_a / (float(cfgs["water_a"]["electronNumber"]) * faraday * reaction_layer)
        + float(cfgs["water_c"]["waterStoichiometry_mol_per_molElectron"]) * water_c / (float(cfgs["water_c"]["electronNumber"]) * faraday * reaction_layer)
    )
    salt_sink = (
        float(cfgs["lp_a"]["saltStoichiometry_mol_per_molElectron"]) * lp_a / (float(cfgs["lp_a"]["electronNumber"]) * faraday * reaction_layer)
        + float(cfgs["lp_c"]["saltStoichiometry_mol_per_molElectron"]) * lp_c / (float(cfgs["lp_c"]["electronNumber"]) * faraday * reaction_layer)
    )
    gas_source = (
        float(cfgs["water_a"]["gasYield_mol_per_molElectron"]) * water_a / (float(cfgs["water_a"]["electronNumber"]) * faraday * reaction_layer)
        + float(cfgs["water_c"]["gasYield_mol_per_molElectron"]) * water_c / (float(cfgs["water_c"]["electronNumber"]) * faraday * reaction_layer)
        + float(cfgs["lp_a"]["gasYield_mol_per_molElectron"]) * lp_a / (float(cfgs["lp_a"]["electronNumber"]) * faraday * reaction_layer)
        + float(cfgs["lp_c"]["gasYield_mol_per_molElectron"]) * lp_c / (float(cfgs["lp_c"]["electronNumber"]) * faraday * reaction_layer)
    )
    bulk_a = (available_anode * anode_faces).sum(dim=(-2, -1)) * face_area
    bulk_c = (available_cathode * cathode_faces).sum(dim=(-2, -1)) * face_area
    bulk_cell = torch.minimum(bulk_a, bulk_c)
    if boundary_model not in robin_models:
        consistency_scale = torch.clamp(cell_current / torch.clamp(bulk_cell, min=current_floor), max=1.0)
    total_reaction_density = (water_a + water_c + lp_a + lp_c).sum(dim=(-2, -1))
    total_candidate_density = (cand_aw + cand_cw + cand_alp + cand_clp).sum(dim=(-2, -1))
    water_fraction = (water_a + water_c).sum(dim=(-2, -1)) / torch.clamp(total_reaction_density, min=current_density_floor)
    lp_fraction = (lp_a + lp_c).sum(dim=(-2, -1)) / torch.clamp(total_reaction_density, min=current_density_floor)
    utilization = total_reaction_density / torch.clamp(total_candidate_density, min=current_density_floor)
    mismatch = torch.abs(raw_a - raw_c) / torch.clamp(torch.maximum(raw_a, raw_c), min=current_floor)
    max_eta = torch.stack([
        eta_aw.amax(dim=(-2, -1)), eta_cw.amax(dim=(-2, -1)),
        eta_alp.amax(dim=(-2, -1)), eta_clp.amax(dim=(-2, -1))
    ], dim=1).amax(dim=1)
    shifts = [
        kinetics["shiftWaterAnode_V"], kinetics["shiftWaterCathode_V"],
        kinetics["shiftLPAnode_V"], kinetics["shiftLPCathode_V"],
    ]
    return {
        "qAnode_W_per_m3": q_aw + q_alp,
        "qCathode_W_per_m3": q_cw + q_clp,
        "qWater_W_per_m3": q_aw + q_cw,
        "qLP_W_per_m3": q_alp + q_clp,
        "qTotal_W_per_m3": q_aw + q_cw + q_alp + q_clp,
        "waterSink_mol_per_m3_s": water_sink,
        "saltSink_mol_per_m3_s": salt_sink,
        "gasSource_mol_per_m3_s": gas_source,
        "jWaterAnode_A_per_m2": water_a,
        "jWaterCathode_A_per_m2": water_c,
        "jLPAnode_A_per_m2": lp_a,
        "jLPCathode_A_per_m2": lp_c,
        "jAnode_A_per_m2": water_a + lp_a,
        "jCathode_A_per_m2": water_c + lp_c,
        "anodeZone": anode_zone,
        "cathodeZone": cathode_zone,
        "anodeFaceCount": anode_faces,
        "cathodeFaceCount": cathode_faces,
        "availableAnodeCurrentDensity_A_per_m2": available_anode,
        "availableCathodeCurrentDensity_A_per_m2": available_cathode,
        "totalCurrent_A": cell_current,
        "rawAnodeCurrent_A": raw_a,
        "rawCathodeCurrent_A": raw_c,
        "currentBalanceMismatchBeforeCoupling": mismatch,
        "bulkAnodeCurrent_A": bulk_a,
        "bulkCathodeCurrent_A": bulk_c,
        "currentConsistencyScale": consistency_scale,
        "waterReactionFraction": water_fraction,
        "lpReactionFraction": lp_fraction,
        "limitingCurrentUtilization": utilization,
        "maximumOverpotential_V": max_eta,
        "maximumNernstShift_V": torch.stack(
            [x.abs().amax(dim=(-2, -1)) for x in shifts], dim=1
        ).amax(dim=1),
        "meanSaltActivityCoefficient": batch_masked_mean(kinetics["saltGamma"], geometry.propellant),
        "meanInterfacialActiveAreaFraction": batch_masked_mean(
            kinetics["activeArea"], anode_zone | cathode_zone
        ),
        "equilibriumPotentialWaterAnode_V": kinetics["equilibriumPotentialWaterAnode_V"],
        "equilibriumPotentialWaterCathode_V": kinetics["equilibriumPotentialWaterCathode_V"],
        "equilibriumPotentialLPAnode_V": kinetics["equilibriumPotentialLPAnode_V"],
        "equilibriumPotentialLPCathode_V": kinetics["equilibriumPotentialLPCathode_V"],
        "reactionLayerThickness_m": kinetics["reactionLayerThickness_m"],
        "fastIonFraction": kinetics["fastIonFraction"],
        "boundaryCouplingModel": boundary_model,
        "surfaceContactAreaElement_m2": torch.full_like(cell_current, face_area),
        "sourceVolumeScaling": (
            "j_times_dx2_for_current_and_j_over_surface_layer_for_volumetric_sources"
            if boundary_model in surface_models
            else "legacy_lateral_face_scaling"
        ),
        "maximumLocalRobinResidual_A_per_m2": (
            robin_boundary["maximumLocalRobinResidual_A_per_m2"]
            if robin_boundary is not None and "maximumLocalRobinResidual_A_per_m2" in robin_boundary
            else torch.zeros_like(cell_current)
        ),
        "maximumLocalRobinRelativeResidual": (
            robin_boundary["maximumLocalRobinRelativeResidual"]
            if robin_boundary is not None and "maximumLocalRobinRelativeResidual" in robin_boundary
            else torch.zeros_like(cell_current)
        ),
        "maximumLocalRobinCombinedResidual": (
            robin_boundary["maximumLocalRobinCombinedResidual"]
            if robin_boundary is not None and "maximumLocalRobinCombinedResidual" in robin_boundary
            else torch.zeros_like(cell_current)
        ),
        "maximumLocalRobinRoundoffFloor_A_per_m2": (
            robin_boundary["maximumLocalRobinRoundoffFloor_A_per_m2"]
            if robin_boundary is not None and "maximumLocalRobinRoundoffFloor_A_per_m2" in robin_boundary
            else torch.zeros_like(cell_current)
        ),
        "roundoffLimitedLocalRobinFaceCount": (
            robin_boundary["roundoffLimitedLocalRobinFaceCount"].to(cell_current.dtype)
            if robin_boundary is not None and "roundoffLimitedLocalRobinFaceCount" in robin_boundary
            else torch.zeros_like(cell_current)
        ),
        "maximumLocalRobinIterations": (
            robin_boundary["maximumLocalRobinIterations"].to(cell_current.dtype)
            if robin_boundary is not None and "maximumLocalRobinIterations" in robin_boundary
            else torch.zeros_like(cell_current)
        ),
        "unresolvedLocalRobinFaceCount": (
            robin_boundary["unresolvedLocalRobinFaceCount"].to(cell_current.dtype)
            if robin_boundary is not None and "unresolvedLocalRobinFaceCount" in robin_boundary
            else torch.zeros_like(cell_current)
        ),
        "localRobinAllFacesConverged": (
            robin_boundary["localRobinAllFacesConverged"]
            if robin_boundary is not None and "localRobinAllFacesConverged" in robin_boundary
            else torch.ones_like(cell_current, dtype=torch.bool)
        ),
    }

def solve_electrical_and_reaction(
    state: dict[str, torch.Tensor],
    transport: dict[str, torch.Tensor],
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float,
    spacing: float,
    *,
    static: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], Any]:
    boundary_model = str(
        config["interface"].get("boundaryCouplingModel", "legacy_posthoc")
    ).lower()
    if boundary_model in {"surface_overlay_bv", "surface_contact", "surface_bv"}:
        potential, diagnostics, robin_diag = _solve_surface_overlay_potential(
            state, transport, geometry, config, composition, voltage, spacing,
            static=static,
        )
        state["potential"] = potential
        electrical = compute_current(
            potential, state, transport, geometry, config, spacing
        )
        reaction_boundary = robin_diag["nonlinearRobinBoundaryReaction"]
        surface_layer = float(config["geometry"]["surfaceLayerThickness_m"])
        contact_normal_heat_anode = (
            reaction_boundary["jAnode_A_per_m2"]
            * reaction_boundary["normalOhmicDropAnode_V"]
            / surface_layer
        )
        contact_normal_heat_cathode = (
            reaction_boundary["jCathode_A_per_m2"]
            * reaction_boundary["normalOhmicDropCathode_V"]
            / surface_layer
        )
        contact_normal_heat = (
            contact_normal_heat_anode + contact_normal_heat_cathode
        )
        invalid_contact_heat = (~torch.isfinite(contact_normal_heat)) | (
            contact_normal_heat < 0.0
        )
        invalid_contact_heat_lanes = invalid_contact_heat.flatten(1).any(dim=1)
        if bool(torch.any(invalid_contact_heat_lanes).item()):
            bad = (
                torch.nonzero(invalid_contact_heat_lanes)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            raise BCCandidateBatchError(
                "Surface-contact normal Joule heat must be finite and non-negative",
                bad,
                "surface_contact_normal_heat",
            )
        electrical["contactNormalJouleHeat_W_per_m3"] = contact_normal_heat
        electrical["contactNormalJouleHeatAnode_W_per_m3"] = (
            contact_normal_heat_anode
        )
        electrical["contactNormalJouleHeatCathode_W_per_m3"] = (
            contact_normal_heat_cathode
        )
        # The local through-thickness resistor is part of electrical
        # dissipation, but is distinct from electrochemical reaction enthalpy.
        # It contributes to every electrical-power diagnostic except the
        # diffusion-only cross term.
        electrical["jouleHeat_W_per_m3"] = (
            electrical["jouleHeat_W_per_m3"] + contact_normal_heat
        )
        electrical["jouleHeatConductive_W_per_m3"] = (
            electrical["jouleHeatConductive_W_per_m3"] + contact_normal_heat
        )
        electrical["electricalPowerDensityJdotE_W_per_m3"] = (
            electrical["electricalPowerDensityJdotE_W_per_m3"]
            + contact_normal_heat
        )
        reaction = interface_reactions(
            state, electrical, transport, geometry, config, composition, voltage,
            spacing, robin_boundary=reaction_boundary,
        )
        robin_diag = dict(robin_diag)
        robin_diag.pop("nonlinearRobinBoundaryReaction", None)
        reaction.update(robin_diag)
        return electrical, reaction, diagnostics
    if boundary_model in {"nonlinear_robin", "robin", "bv_robin"}:
        potential, diagnostics, robin_diag = _solve_nonlinear_robin_potential(
            state, transport, geometry, config, composition, voltage, spacing, static=static
        )
        state["potential"] = potential
        electrical = compute_current(potential, state, transport, geometry, config, spacing)
        reaction = interface_reactions(
            state, electrical, transport, geometry, config, composition, voltage, spacing,
            robin_boundary=robin_diag["nonlinearRobinBoundaryReaction"],
        )
        robin_diag = dict(robin_diag)
        robin_diag.pop("nonlinearRobinBoundaryReaction", None)
        reaction.update(robin_diag)
        return electrical, reaction, diagnostics

    potential, diagnostics, _ = solve_potential(
        state["potential"],
        state["cation"],
        state["anion"],
        transport["Dcation"],
        transport["Danion"],
        transport["sigmaTotal"],
        geometry,
        config,
        voltage,
        spacing,
        static=static,
    )
    state["potential"] = potential
    electrical = compute_current(potential, state, transport, geometry, config, spacing)
    reaction = interface_reactions(
        state, electrical, transport, geometry, config, composition, voltage, spacing
    )
    scale = reaction["currentConsistencyScale"][:, None, None]
    for name in [
        "Jx", "Jy", "Jmagnitude", "JohmX", "JohmY", "JdiffX", "JdiffY",
        "jouleHeat_W_per_m3", "jouleHeatConductive_W_per_m3",
        "electricalPowerDensityJdotE_W_per_m3",
        "diffusionElectricalPowerDensity_W_per_m3",
    ]:
        if name in electrical and torch.is_tensor(electrical[name]):
            electrical[name] = electrical[name] * scale
    reaction = interface_reactions(
        state, electrical, transport, geometry, config, composition, voltage, spacing
    )
    return electrical, reaction, diagnostics

def static_solver(
    geometry: GeometryBatch,
    config: dict,
    composition: CompositionModel,
    voltage: float,
    dtype: torch.dtype,
) -> dict[str, Any]:
    spacing = float(config["geometry"]["domainSize_m"]) / max(geometry.grid_size - 1, 1)
    state = initial_state(geometry, config, composition, voltage, dtype)
    width = float(config["phase"]["transitionWidth_K"])
    state["liquidFraction"] = 1.0 / (
        1.0 + torch.exp(-(state["temperature"] - composition.effective_softening_temperature_K) / width)
    )
    state["liquidFraction"] = torch.where(geometry.fixed, torch.zeros_like(state["liquidFraction"]), state["liquidFraction"])
    transport = transport_fields(state, geometry, config, composition)
    electrical, reaction, diagnostics = solve_electrical_and_reaction(
        state, transport, geometry, config, composition, voltage, spacing, static=True
    )
    current_floor = physical_floor(config, "current_A", 1e-15)
    effective_resistance = (
        voltage - float(config["electrical"]["cathodeVoltage_V"])
    ) / torch.clamp(reaction["totalCurrent_A"], min=current_floor)
    return {
        "state": state,
        "transport": transport,
        "electrical": electrical,
        "reaction": reaction,
        "solverDiagnostics": diagnostics,
        "totalCurrent_A": reaction["totalCurrent_A"],
        "effectiveResistance_ohm": effective_resistance,
        "initialLiquidFraction": batch_masked_mean(state["liquidFraction"], geometry.propellant),
        "ionicCurrentFraction": batch_masked_mean(electrical["ionicCurrentFraction"], geometry.propellant),
        "waterReactionFraction": reaction["waterReactionFraction"],
        "lpReactionFraction": reaction["lpReactionFraction"],
        "paperLimitingCurrentUtilization": reaction["limitingCurrentUtilization"],
        "anodeCathodeCurrentMismatch": reaction["currentBalanceMismatchBeforeCoupling"],
        "modelStatus": "v7_7_5_static_np_bv_gap_robin_spd_pcg_nominal_unvalidated",
    }
