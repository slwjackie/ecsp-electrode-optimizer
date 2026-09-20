from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_retained_water(comp: dict[str, Any], masses: dict[str, float]) -> tuple[float, float, str]:
    """Return (retained fraction of recipe water, cured-water mass fraction, basis).

    Legacy v7.x configurations specify ``retained_water_fraction`` as the
    fraction of the *mixing water* that remains after cure.  The B/C paper-based
    configuration instead specifies the experimentally meaningful
    ``cured_water_mass_fraction`` in the final cured specimen.  Supporting both
    definitions keeps every legacy profile byte-for-byte usable while removing
    an otherwise easy initial-condition ambiguity in the new model.
    """
    initial_water = float(masses.get("water", 0.0))
    dry_mass = sum(float(v) for k, v in masses.items() if k != "water")
    cured_fraction_value = comp.get("cured_water_mass_fraction")
    if cured_fraction_value is not None:
        cured_fraction = float(cured_fraction_value)
        if not 0.0 <= cured_fraction < 1.0:
            raise ValueError("composition.cured_water_mass_fraction must lie in [0,1)")
        retained_mass = (
            cured_fraction * dry_mass / max(1.0 - cured_fraction, 1e-30)
        )
        if retained_mass > initial_water + 1e-12:
            raise ValueError(
                "Requested cured-water mass fraction requires more water than "
                "the wet recipe contains"
            )
        retained_fraction = (
            retained_mass / initial_water if initial_water > 0.0 else 0.0
        )
        basis = str(
            comp.get(
                "cured_water_mass_fraction_basis",
                comp.get(
                    "retained_water_fraction_basis",
                    "cured_water_mass_fraction",
                ),
            )
        )
        return retained_fraction, cured_fraction, basis

    retained_fraction = float(comp.get("retained_water_fraction", 1.0))
    retained_fraction = max(0.0, min(1.0, retained_fraction))
    retained_mass = initial_water * retained_fraction
    total = dry_mass + retained_mass
    cured_fraction = retained_mass / max(total, 1e-30)
    basis = str(
        comp.get("retained_water_fraction_basis", "unspecified_assumption")
    )
    return retained_fraction, cured_fraction, basis


def composition_summary(config: dict[str, Any]) -> dict[str, Any]:
    comp = config["composition"]
    masses = {k: float(v) for k, v in comp["masses_g"].items()}
    retained, cured_water_fraction, retained_basis = _resolve_retained_water(
        comp, masses
    )
    adjusted = dict(masses)
    adjusted["water"] = masses.get("water", 0.0) * retained
    total_wet = sum(masses.values())
    total_analyzed = sum(adjusted.values())
    dry_total = total_analyzed - adjusted["water"]
    molar = comp["molar_masses_g_per_mol"]
    # Optional ingredients remain supported.  Missing ingredients are treated
    # as zero rather than forcing the non-glycerol B/C recipe to invent a mass.
    moles = {
        "LP": adjusted.get("LP", 0.0) / float(molar["LP"]),
        "water": adjusted.get("water", 0.0) / float(molar["water"]),
        "PVA_repeat": adjusted.get("PVA", 0.0) / float(molar["PVA_repeat"]),
        "glycerol": adjusted.get("glycerol", 0.0)
        / float(molar.get("glycerol", 92.09382)),
        "boric_acid": adjusted.get("boric_acid", 0.0)
        / float(molar["boric_acid"]),
    }
    if comp.get("density_mode") == "specified":
        density = float(comp["mixture_density_kg_per_m3"])
    else:
        densities = comp["component_densities_kg_per_m3"]
        volume_m3 = sum(
            (adjusted[k] / 1000.0) / float(densities[k])
            for k in adjusted
            if adjusted[k] > 0.0
        )
        density = (total_analyzed / 1000.0) / max(volume_m3, 1e-30)
    volume_m3 = (total_analyzed / 1000.0) / density
    concentrations = {name: mol / volume_m3 for name, mol in moles.items()}
    concentrations["Li_plus"] = concentrations["LP"]
    concentrations["ClO4_minus"] = concentrations["LP"]
    wet_fractions = {k: v / total_wet for k, v in masses.items()}
    analyzed_fractions = {k: v / total_analyzed for k, v in adjusted.items()}
    dry_fractions = {
        k: v / dry_total for k, v in adjusted.items() if k != "water"
    }
    pva_mass = max(adjusted.get("PVA", 0.0), 1e-30)
    pva_repeat_moles = max(moles["PVA_repeat"], 1e-30)
    return {
        "total_input_mass_g": total_wet,
        "total_analyzed_mass_g": total_analyzed,
        "retained_water_fraction": retained,
        "cured_water_mass_fraction": cured_water_fraction,
        "retained_water_fraction_basis": retained_basis,
        "retained_water_measurement_id": comp.get(
            "retained_water_measurement_id"
        ),
        "retained_water_is_measured": (
            retained_basis == "measured_cure_mass_balance"
            and bool(str(comp.get("retained_water_measurement_id") or "").strip())
        ),
        "density_kg_per_m3": density,
        "estimated_volume_m3": volume_m3,
        "masses_g": masses,
        "analyzed_masses_g": adjusted,
        "wet_mass_fractions": wet_fractions,
        "analyzed_mass_fractions": analyzed_fractions,
        "dry_mass_fractions": dry_fractions,
        "moles": moles,
        "initial_concentrations_mol_per_m3": concentrations,
        "water_to_LP_mass_ratio": masses.get("water", 0.0)
        / max(masses.get("LP", 0.0), 1e-30),
        "glycerol_to_PVA_mass_ratio": adjusted.get("glycerol", 0.0) / pva_mass,
        "boric_acid_to_PVA_repeat_molar_ratio": moles["boric_acid"]
        / pva_repeat_moles,
        "calibration_status": config["project"].get(
            "calibrationStatus", "unknown"
        ),
    }


def save_composition_summary(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = composition_summary(config)
    (output_dir / "composition_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    for ingredient, mass in summary["masses_g"].items():
        rows.append(
            {
                "ingredient": ingredient,
                "input_mass_g": mass,
                "analyzed_mass_g": summary["analyzed_masses_g"][ingredient],
                "wet_mass_fraction": summary["wet_mass_fractions"][ingredient],
                "analyzed_mass_fraction": summary["analyzed_mass_fractions"][ingredient],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "composition_summary.csv", index=False)
    return summary
