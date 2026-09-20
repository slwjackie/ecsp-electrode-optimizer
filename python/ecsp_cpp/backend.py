from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import os
import struct
import subprocess
import time

import numpy as np




@dataclass(frozen=True)
class PreparedCppCase:
    command: tuple[str, ...]
    metrics_path: Path
    stdout_path: Path
    stderr_path: Path

def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def write_mask_binary(anode: np.ndarray, cathode: np.ndarray, path: Path) -> None:
    anode = np.ascontiguousarray(np.asarray(anode, dtype=np.uint8))
    cathode = np.ascontiguousarray(np.asarray(cathode, dtype=np.uint8))
    if anode.ndim != 2 or cathode.shape != anode.shape or anode.shape[0] != anode.shape[1]:
        raise ValueError("C++ physics masks must be matching square 2-D arrays")
    if np.any((anode != 0) & (cathode != 0)):
        raise ValueError("C++ physics mask contains polarity overlap")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<I", int(anode.shape[0])))
        handle.write(anode.tobytes(order="C"))
        handle.write(cathode.tobytes(order="C"))


def _channel_values(channel: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "Eeq": channel["equilibriumPotential_V"],
        "j0": channel["exchangeCurrentDensity_A_per_m2"],
        "alpha": channel["chargeTransferCoefficient"],
        "reverse": channel["reverseAvailabilityFraction"],
        "n": channel["electronNumber"],
        "km": channel["massTransferCoefficient_m_per_s"],
        "dH": channel["reactionEnthalpy_J_per_mol"],
        "water_stoich": channel.get("waterStoichiometry_mol_per_molElectron", 0.0),
        "salt_stoich": channel.get("saltStoichiometry_mol_per_molElectron", 0.0),
        "gas_yield": channel["gasYield_mol_per_molElectron"],
        "ref_activity": channel["referenceActivity"],
        "nernst_exp": channel["nernstReactionQuotientExponent"],
    }


def serialise_cpp_config(
    config: Mapping[str, Any],
    composition: Any,
    *,
    grid_size: int,
    voltage: float,
) -> dict[str, Any]:
    """Project the resolved Python configuration onto the standalone C++ engine.

    The keys intentionally correspond one-to-one with ``cpp/ecsp_cpp_solver.cpp``.
    No MPS/PyTorch numerical setting is forwarded; the C++ engine is always CPU
    IEEE-754 binary64 and uses a full-propellant surface-contact model with matrix-free PCG + deterministic SOR fallback.
    """
    g = config["geometry"]
    e = config["electrical"]
    t = config["transport"]
    p = config["phase"]
    interface = config["interface"]
    chemical = config["chemical"]
    thermal = config["thermal"]
    gas = config["gas"]
    coupled = config["coupled"]
    ignition = config["condensedIgnition"]
    numerics = config["numerics"]
    robin = interface["nonlinearRobin"]
    blocking = interface["blocking"]
    ps = numerics["potentialSolver"]

    values: dict[str, Any] = {
        "n": int(grid_size),
        "domain": float(g["domainSize_m"]),
        "layer": float(g["surfaceLayerThickness_m"]),
        "voltage": float(voltage),
        "cathode_voltage": float(e["cathodeVoltage_V"]),
        "dt": float(coupled["timeStep_s"]),
        "end_time": float(coupled["endTime_s"]),
        "eval_time": float(min(2.0, coupled["endTime_s"])),
        "electrical_interval": float(coupled["electricalUpdateInterval_s"]),
        "ignition_T": float(ignition["decompositionOnsetTemperature_K"]),
        "ignition_progress_guard": float(ignition["minimumChemicalProgressNumericalGuard"]),
        "ignition_rate_guard": float(ignition["minimumChemicalRateNumericalGuard_per_s"]),
        "density": float(composition.density_kg_per_m3),
        "cation0": float(composition.initial_cation_mol_per_m3),
        "anion0": float(composition.initial_anion_mol_per_m3),
        "water0": float(composition.initial_water_mol_per_m3),
        "lp0": float(composition.initial_lp_mol_per_m3),
        "glycerol_pva_ratio": float(composition.glycerol_to_pva_mass_ratio),
        "boric_pva_ratio": float(composition.boric_acid_to_pva_repeat_molar_ratio),
        "effective_softening_T": float(composition.effective_softening_temperature_K),
        "R": float(t["gasConstant_J_per_molK"]),
        "F": float(t["faradayConstant_C_per_mol"]),
        "zplus": float(t["chargeNumberCation"]),
        "zminus": float(t["chargeNumberAnion"]),
        "Dp_dry": float(t["cationDiffusivityDry_m2_per_s"]),
        "Dm_dry": float(t["anionDiffusivityDry_m2_per_s"]),
        "Dp_wet": float(t["cationDiffusivityWet_m2_per_s"]),
        "Dm_wet": float(t["anionDiffusivityWet_m2_per_s"]),
        "Dw0": float(t["waterDiffusivity_m2_per_s"]),
        "Ea_Dp": float(t["cationTransportActivationEnergy_J_per_mol"]),
        "Ea_Dm": float(t["anionTransportActivationEnergy_J_per_mol"]),
        "Tref": float(t["referenceTemperature_K"]),
        "liquid_D_gain": float(t["liquidDiffusivityGain"]),
        "water_activity_exp": float(t["waterActivityExponent"]),
        "glycerol_gain": float(t["glycerolPlasticizationGain"]),
        "crosslink_penalty": float(t["crosslinkTransportPenalty"]),
        "concentration_min_fraction": float(t["concentrationMinimumFraction"]),
        "concentration_max_multiple": float(t["concentrationMaximumMultiple"]),
        "electroneutral_relax": float(t["electroneutralRelaxation"]),
        "max_relative_concentration_change": float(t["maximumRelativeConcentrationChangePerStep"]),
        "species_substeps": int(numerics.get("speciesFixedSubsteps", 2)),
        "full_np": bool(coupled["useFullNernstPlanckTransport"]),
        "sigma_e0": float(e["electronicConductivity0_S_per_m"]),
        "sigma_e_temp_coeff": float(e["electronicConductivityTemperatureCoefficient_per_K"]),
        "sigma_e_liquid_suppression": float(e["electronicLiquidSuppression"]),
        "sigma_min": float(e["conductivityMinimum_S_per_m"]),
        "sigma_max": float(e["conductivityMaximum_S_per_m"]),
        "diffusion_potential": bool(e["diffusionPotentialEnabled"]),
        "transition_width": float(p["transitionWidth_K"]),
        "phase_relax_time": float(p["relaxationTime_s"]),
        "latent_heat": float(p["latentHeat_J_per_kg"]),
        "fast_ion_liquid_threshold": float(p["minimumLiquidFractionForFastIonTransport"]),
        "full_bv": bool(interface["useFullButlerVolmer"]),
        "mass_transfer_saturation": bool(coupled["usePaperMassTransferSaturation"]),
        "derive_km": bool(interface["deriveMassTransferCoefficientFromDiffusivity"]),
        "diffusion_layer": float(interface["physicalDiffusionLayerThickness_m"]),
        "reaction_layer_min": float(interface["numericalReactionLayerMinimum_m"]),
        "contact_normal_length": float(
            interface.get(
                "contactNormalConductionLength_m",
                interface["numericalReactionLayerMinimum_m"],
            )
        ),
        "initial_contact_drop": float(interface.get("initialContactPotentialDrop_V", 5.0)),
        "surface_contact_model": str(interface.get("contactModel", "")).lower()
        == "surface_overlay_full_propellant",
        "hidden_bus_assumed": bool(interface.get("hiddenBusConnectionAssumed", True)),
        "activation_heat": bool(interface["includeActivationHeat"]),
        "activation_heat_fraction": float(interface["activationHeatFraction"]),
        "exp_limit": float(interface["exponentialArgumentLimit"]),
        "liquid_kinetics_gain": float(interface["liquidKineticsGain"]),
        "min_active_area": float(blocking["minimumActiveAreaFraction"]),
        "nernst_enabled": bool(interface["nernst"]["enabled"]),
        "activity_floor": float(interface["nernst"]["activityFloor"]),
        "activity_A": float(interface["nernst"]["activityA"]),
        "activity_B": float(interface["nernst"]["activityB"]),
        "activity_linear": float(interface["nernst"]["activityLinearPerMolL"]),
        "max_log10_gamma": float(interface["nernst"]["maximumLog10ActivityCoefficientMagnitude"]),
        "max_nernst_shift": float(interface["nernst"]["maximumAbsoluteShift_V"]),
        "passivation_enabled": bool(blocking["passivation"]["enabled"]),
        "pass_form": float(blocking["passivation"]["formationRate_per_s"]),
        "pass_remove": float(blocking["passivation"]["removalRate_per_s"]),
        "pass_jref": float(blocking["passivation"]["referenceCurrentDensity_A_per_m2"]),
        "pass_jexp": float(blocking["passivation"]["currentExponent"]),
        "pass_T0": float(blocking["passivation"]["formationActivationTemperature_K"]),
        "pass_Tw": float(blocking["passivation"]["formationActivationWidth_K"]),
        "gas_coverage_enabled": bool(blocking["gasCoverage"]["enabled"]),
        "gas_cov_form": float(blocking["gasCoverage"]["formationRate_per_s"]),
        "gas_cov_detach": float(blocking["gasCoverage"]["detachmentRate_per_s"]),
        "gas_source_ref": float(blocking["gasCoverage"]["referenceGasSource_mol_per_m3_s"]),
        "gas_thermal_insulation": float(blocking["gasCoverage"]["thermalInsulationFraction"]),
        "robin_min_iter": int(robin["minimumIterations"]),
        "robin_max_iter": int(robin["maximumIterations"]),
        "robin_phi_tol": float(robin["potentialTolerance_V"]),
        "robin_current_tol": float(robin["relativeReactionCurrentTolerance"]),
        "robin_balance_tol": float(robin["currentBalanceTolerance"]),
        "robin_relax": float(robin.get("potentialUnderRelaxation", robin.get("underRelaxation", 0.35))),
        "robin_balance_abs": float(robin.get("currentBalanceAbsoluteTolerance_A", 1e-12)),
        "gauge_margin": float(robin.get("gaugeBracketMargin_V", 25.0)),
        "gauge_max_step": float(robin.get("gaugeMaximumStep_V", 65.0)),
        "gauge_deriv_floor": float(robin.get("gaugeDerivativeFloor_A_per_V", 1e-14)),
        "gauge_min_iter": int(robin.get("gaugeMinimumIterations", 1)),
        "gauge_max_iter": int(robin.get("gaugeMaximumIterations", 24)),
        "gauge_relax": float(robin.get("gaugeUnderRelaxation", 1.0)),
        "local_min_iter": int(robin.get("localInterfaceMinimumIterations", 8)),
        "local_max_iter": int(robin.get("localInterfaceMaximumIterations", 60)),
        "local_newton": int(robin.get("localNewtonPolishIterations", 2)),
        "local_abs_tol": float(robin.get("localRobinResidualTolerance_A_per_m2", 0.01)),
        "local_rel_tol": float(robin.get("localRobinRelativeResidualTolerance", 1e-4)),
        "chemical_enabled": bool(chemical["enabled"]),
        "chem_A": float(chemical["preExponentialFactor_per_s"]),
        "chem_Ea": float(chemical["activationEnergy_J_per_mol"]),
        "chem_order": float(chemical["reactionOrder"]),
        "oxidizer_order": float(chemical["oxidizerOrder"]),
        "oxidizer_consume": float(chemical["oxidizerConsumptionFractionPerUnitProgress"]),
        "chem_max_rate": float(chemical["maximumRate_per_s"]),
        "chem_heat": float(chemical["heatRelease_J_per_kg"]),
        "gas_molar_mass": float(chemical["effectiveGasMolarMass_kg_per_mol"]),
        "gas_yield_mass": float(chemical["gasYieldMassFraction"]),
        "chem_activation_T": float(chemical["activationTemperature_K"]),
        "cp": float(thermal["heatCapacity_J_per_kgK"]),
        "k": float(thermal["thermalConductivity_W_per_mK"]),
        "T0": float(thermal["initialTemperature_K"]),
        "Tamb": float(thermal["ambientTemperature_K"]),
        "hconv": float(thermal["convectionCoefficient_W_per_m2K"]),
        "emissivity": float(thermal["emissivity"]),
        "sigma_sb": float(thermal["stefanBoltzmann_W_per_m2K4"]),
        "Tmin": float(thermal["minimumTemperature_K"]),
        "Tmax": float(thermal["maximumTemperature_K"]),
        "gas_D": float(gas["diffusivity_m2_per_s"]),
        "gas_loss": float(gas["lossRate_per_s"]),
        "gas_max": float(gas["maximumConcentration_mol_per_m3"]),
        "pcg_max_iter": int(max(ps.get("maximumIterationsStatic", 3000), ps.get("maximumIterationsCoupled", 1500))),
        # C++ CPU FP64 keeps the publication/reference tolerances, independent of MPS profiles.
        "pcg_rtol_static": float(ps.get("relativeToleranceStatic", 1e-10)),
        "pcg_rtol_coupled": float(ps.get("relativeToleranceCoupled", 1e-9)),
        "pcg_atol": float(ps.get("absoluteTolerance", 1e-12)),
        "current_floor": float(numerics["physicalFloors"]["current_A"]),
        "current_density_floor": float(numerics["physicalFloors"]["currentDensity_A_per_m2"]),
    }
    for prefix, family, polarity in (
        ("water_a", "water", "anode"),
        ("water_c", "water", "cathode"),
        ("lp_a", "lp", "anode"),
        ("lp_c", "lp", "cathode"),
    ):
        for name, value in _channel_values(interface[family][polarity]).items():
            values[f"{prefix}.{name}"] = value
    return values


def write_cpp_config(values: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = _bool_text(value)
        else:
            rendered = str(value)
        lines.append(f"{key}={rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CppPhysicsRunner:
    def __init__(self, package_root: Path, executable: Path | None = None) -> None:
        self.package_root = Path(package_root).resolve()
        self.executable = (
            Path(executable).resolve()
            if executable is not None
            else (self.package_root / "build" / "ecsp_cpp_solver").resolve()
        )

    def verify(self) -> str:
        if not self.executable.is_file():
            raise FileNotFoundError(
                f"C++ physics executable not found: {self.executable}. "
                "Run tools/build_cpp_cpu.sh first."
            )
        completed = subprocess.run(
            [str(self.executable), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def prepare_case(
        self,
        anode: np.ndarray,
        cathode: np.ndarray,
        config_values: Mapping[str, Any],
        output_dir: Path,
    ) -> PreparedCppCase:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / "cpp_geometry_mask.bin"
        config_path = output_dir / "cpp_physics_config.kv"
        metrics_path = output_dir / "condensed_metrics.json"
        stdout_path = output_dir / "cpp_solver_stdout.log"
        stderr_path = output_dir / "cpp_solver_stderr.log"
        write_mask_binary(anode, cathode, mask_path)
        write_cpp_config(config_values, config_path)
        return PreparedCppCase(
            command=(
                str(self.executable),
                "--mask",
                str(mask_path),
                "--config",
                str(config_path),
                "--output",
                str(metrics_path),
            ),
            metrics_path=metrics_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    @staticmethod
    def launch_prepared_case(prepared: PreparedCppCase) -> subprocess.Popen[str]:
        # Files are opened in the parent only for process creation.  The child
        # keeps its descriptors after the context exits, so no pipe can fill and
        # block a long-running solver.
        with prepared.stdout_path.open("w", encoding="utf-8") as stdout_handle, \
             prepared.stderr_path.open("w", encoding="utf-8") as stderr_handle:
            return subprocess.Popen(
                list(prepared.command),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )

    @staticmethod
    def collect_prepared_case(
        prepared: PreparedCppCase,
        process: subprocess.Popen[str],
        started: float,
    ) -> dict[str, Any]:
        returncode = process.wait()
        if returncode != 0:
            stderr = (
                prepared.stderr_path.read_text(encoding="utf-8").strip()
                if prepared.stderr_path.exists()
                else ""
            )
            stdout = (
                prepared.stdout_path.read_text(encoding="utf-8").strip()
                if prepared.stdout_path.exists()
                else ""
            )
            detail = stderr or stdout
            raise RuntimeError(
                f"C++ FP64 physics failed with exit code {returncode}: {detail}"
            )
        if not prepared.metrics_path.is_file():
            raise RuntimeError(
                f"C++ FP64 physics exited successfully but produced no metrics: "
                f"{prepared.metrics_path}"
            )
        row = json.loads(prepared.metrics_path.read_text(encoding="utf-8"))
        row["cppSubprocessWallClock_s"] = time.perf_counter() - started
        return row

    def run_case(
        self,
        anode: np.ndarray,
        cathode: np.ndarray,
        config_values: Mapping[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        prepared = self.prepare_case(anode, cathode, config_values, output_dir)
        started = time.perf_counter()
        process = self.launch_prepared_case(prepared)
        return self.collect_prepared_case(prepared, process, started)
