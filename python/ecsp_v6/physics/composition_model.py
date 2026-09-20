from __future__ import annotations

from dataclasses import dataclass

from ..composition import composition_summary


@dataclass(frozen=True)
class CompositionModel:
    # Original v7.x positional field order is preserved for external callers.
    density_kg_per_m3: float
    initial_cation_mol_per_m3: float
    initial_anion_mol_per_m3: float
    initial_water_mol_per_m3: float
    glycerol_to_pva_mass_ratio: float
    boric_acid_to_pva_repeat_molar_ratio: float
    effective_softening_temperature_K: float
    summary: dict
    # v8 B/C global-reaction inventory fields.  Defaults preserve backwards
    # compatibility for any external code that instantiated the legacy class.
    initial_lp_mol_per_m3: float = 0.0
    initial_pva_repeat_mol_per_m3: float = 0.0
    molar_mass_lp_kg_per_mol: float = 0.10639
    molar_mass_pva_repeat_kg_per_mol: float = 0.04405256
    cured_water_mass_fraction: float = 0.0


def build_composition(config: dict) -> CompositionModel:
    summary = composition_summary(config)
    concentrations = summary["initial_concentrations_mol_per_m3"]
    phase = config["phase"]
    effective_softening = (
        float(phase["softeningTemperature_K"])
        - float(phase["glycerolSofteningShift_K_perMassRatio"])
        * float(summary["glycerol_to_PVA_mass_ratio"])
        + float(phase["boricAcidSofteningShift_K_perMolarRatio"])
        * float(summary["boric_acid_to_PVA_repeat_molar_ratio"])
    )
    return CompositionModel(
        density_kg_per_m3=float(summary["density_kg_per_m3"]),
        initial_cation_mol_per_m3=float(concentrations["Li_plus"]),
        initial_anion_mol_per_m3=float(concentrations["ClO4_minus"]),
        initial_water_mol_per_m3=float(concentrations["water"]),
        initial_lp_mol_per_m3=float(concentrations["LP"]),
        initial_pva_repeat_mol_per_m3=float(concentrations["PVA_repeat"]),
        molar_mass_lp_kg_per_mol=float(config["composition"]["molar_masses_g_per_mol"]["LP"]) / 1000.0,
        molar_mass_pva_repeat_kg_per_mol=float(config["composition"]["molar_masses_g_per_mol"]["PVA_repeat"]) / 1000.0,
        cured_water_mass_fraction=float(summary["cured_water_mass_fraction"]),
        glycerol_to_pva_mass_ratio=float(summary["glycerol_to_PVA_mass_ratio"]),
        boric_acid_to_pva_repeat_molar_ratio=float(summary["boric_acid_to_PVA_repeat_molar_ratio"]),
        effective_softening_temperature_K=float(effective_softening),
        summary=summary,
    )
