"""Validated, conservative conversion of the actual BC onset snapshot.

No pre-flame heat history is replayed. BC density is the total condensed
continuum density (reactants plus products); decomposition does not remove it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import json
import math
import numpy as np
from ecsp_nsga2.propagation import PropagationConfigurationError, PropagationCandidateInputError
from .chemistry import *
from .thermo import BCTaitThermodynamics


def model_digest(bc_config):
    return hashlib.sha256(json.dumps(bc_config,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


@dataclass
class AdaptedBCState:
    U: np.ndarray
    anode_mask: np.ndarray
    cathode_mask: np.ndarray
    propellant_mask: np.ndarray
    potential: np.ndarray
    qj_at_onset: np.ndarray
    qe_at_onset: np.ndarray
    dx: float
    thickness: float
    absolute_onset_s: float
    thermo: BCTaitThermodynamics
    chemistry: BCTwoChannelChemistry
    audit: dict[str, Any]


class BCReactiveHandoffAdapter:
    def __init__(self, propagation_config: Mapping[str, Any], bc_config: Mapping[str, Any], reactive_config: Mapping[str, Any]):
        self.prop = dict(propagation_config)
        self.bc = dict(bc_config)
        self.reactive = dict(reactive_config)

    def adapt(self, handoff: Mapping[str, Any]) -> AdaptedBCState:
        def scalar(key, *, nonnegative=False):
            try:
                value = float(handoff[key])
            except (KeyError, TypeError, ValueError) as e:
                raise PropagationCandidateInputError(f"BC handoff needs scalar {key}") from e
            if not math.isfinite(value) or (nonnegative and value<0):
                raise PropagationCandidateInputError(f"Invalid BC handoff scalar {key}")
            return value

        if handoff.get("handoffSchemaVersion")!="ecsp_bc_surface_onset_v8.2.0":
            raise PropagationCandidateInputError("BC Reactive requires the authorized v8.2 onset schema")
        for key in ("onsetSucceeded","onsetReportedByPhysics","numericallyValidForPropagationHandoff"):
            if not isinstance(handoff.get(key), (bool,np.bool_)) or not bool(handoff[key]):
                raise PropagationCandidateInputError(f"BC Reactive cannot advance an unauthorized onset: {key}")
        if handoff.get("propagationHandoffAuthorizationReason")!="ignition_and_numerics_valid":
            raise PropagationCandidateInputError("BC onset authorization reason is inconsistent")
        onset = scalar("ignitionDelay_s", nonnegative=True)
        if onset > float(self.bc["endTime_s"])+1e-12:
            raise PropagationCandidateInputError("BC onset lies beyond the pre-flame horizon")
        if bool(handoff.get("preflameElectricalHistoryMayBeReplayed",False)):
            raise PropagationCandidateInputError("Pre-flame electrical replay is forbidden")
        # The flag authorizes no heating model; the selected post-onset policy
        # controls power, not this old snapshot metadata.
        if not isinstance(handoff.get("continuedElectricalHeating"), (bool,np.bool_)):
            raise PropagationCandidateInputError("Missing continuedElectricalHeating metadata")
        try:
            temperature = np.array(handoff["temperatureAtOnset_K"],dtype=np.float64,copy=True)
        except (KeyError,ValueError,TypeError) as e:
            raise PropagationCandidateInputError("Missing onset temperature") from e
        shape = temperature.shape
        if temperature.ndim!=2 or min(shape)<5 or shape[0]!=shape[1]:
            raise PropagationCandidateInputError("BC Reactive requires a square cell-centered grid with at least 5 cells per side")

        def field(key, nonnegative=False):
            try:
                value = np.array(handoff[key],dtype=np.float64,copy=True)
            except (KeyError,TypeError,ValueError) as e:
                raise PropagationCandidateInputError(f"BC Reactive requires field {key}") from e
            if value.shape!=shape or not np.isfinite(value).all() or (nonnegative and np.any(value<0)):
                raise PropagationCandidateInputError(f"Invalid BC field {key}: shape/finite/nonnegative contract")
            return value

        def mask(key):
            raw = field(key)
            if not np.all((raw==0)|(raw==1)):
                raise PropagationCandidateInputError(f"{key} must be binary")
            return raw.astype(bool)

        prop_mask = mask("propellantMask")
        anode, cathode = mask("anodeContactMask"),mask("cathodeContactMask")
        if not np.all(prop_mask) or np.any(anode & cathode) or not anode.any() or not cathode.any():
            raise PropagationCandidateInputError("Full-domain surface overlay and disjoint nonempty contacts are required")
        if np.any(temperature<=0):
            raise PropagationCandidateInputError("Temperature must be finite positive Kelvin")
        rho = float(self.prop["density_kg_per_m3"])
        domain = float(self.prop["domain_size_m"])
        depth = float(self.prop["surface_layer_thickness_m"])
        if not all(math.isfinite(v) and v>0 for v in (rho,domain,depth)):
            raise PropagationConfigurationError("Resolved BC density/domain/thickness must be positive")
        for key,expected in (("domainSize_m",domain),("surfaceLayerThickness_m",depth),("condensedDensity_kg_per_m3",rho)):
            if key in handoff and not math.isclose(scalar(key),expected,rel_tol=1e-12,abs_tol=1e-15):
                raise PropagationCandidateInputError(f"Handoff and resolved BC config disagree: {key}")
        digest = model_digest(self.bc)
        if "bcGlobalConfigSHA256" in handoff and handoff["bcGlobalConfigSHA256"]!=digest:
            raise PropagationCandidateInputError("BC chemistry/thermal config changed since onset; handoff hash mismatch")
        for key in ("xiMax_mol_per_m3","initialMobileLP_mol_per_m3","initialPVARepeat_mol_per_m3",
                    "molarMassLP_kg_per_mol","molarMassPVARepeat_kg_per_mol","initialReactiveMass_kg_per_m3"):
            if scalar(key,nonnegative=True)<=0:
                raise PropagationCandidateInputError(f"{key} must be positive")
        scalar("initialMobileWater_mol_per_m3",nonnegative=True)
        chemistry = BCTwoChannelChemistry(self.bc,handoff,rho,self.prop["gas_constant_J_per_molK"])
        thermo = BCTaitThermodynamics(self.reactive["eos"],self.bc["thermal"],rho)
        P = np.zeros((*shape,NCONS),dtype=np.float64)
        P[...,RHO] = rho
        P[...,3] = temperature
        # BC has no mechanics. We do not fabricate a pressure pulse or velocity.
        P[...,A1] = field("alphaChannel1AtOnset",True)
        P[...,A2] = field("alphaChannel2AtOnset",True)
        supplied_progress = field("globalProgressAtOnset",True)
        X = P[...,A1:A2+1] @ chemistry.weights
        if np.any(P[...,A1:A2+1]>1) or not np.allclose(X,supplied_progress,atol=1e-10,rtol=1e-9):
            raise PropagationCandidateInputError("BC two-channel progress is inconsistent")
        inventory_fields = ((CATION,"cationAtOnset_mol_per_m3"),(ANION,"anionAtOnset_mol_per_m3"),
                            (WATER,"mobileWaterAtOnset_mol_per_m3"),(PVA,"pvaReactiveRepeatAtOnset_mol_per_m3"),
                            (PRODUCT_WATER,"generatedWaterProductAtOnset_mol_per_m3"),
                            (EC_LP,"electrochemicalLPConsumedAtOnset_mol_per_m3"))
        for i,key in inventory_fields:
            P[...,i] = field(key,True)/rho
        U = thermo.conservative(P)
        lp = field("mobileLPAtOnset_mol_per_m3",True)
        if not np.allclose(lp,.5*(U[...,CATION]+U[...,ANION]),rtol=1e-10,atol=1e-10):
            raise PropagationCandidateInputError("Mobile LP differs from the BC ion pair average")
        if not np.allclose(U[...,CATION],U[...,ANION],rtol=1e-10,atol=1e-10):
            raise PropagationCandidateInputError("Onset is not the BC electroneutral ion-pair state")
        residuals = chemistry.conserved_inventory_residuals(U)
        for key in ("pva_plus_extent","product_water_minus_extent"):
            if np.max(np.abs(residuals[key])) > 1e-9*max(1.,rho*chemistry.initial_pva_per_kg):
                raise PropagationCandidateInputError(f"Onset reaction inventory is inconsistent: {key}")
        if abs(float(residuals["salt_plus_ec_plus_extent"].mean())) > 1e-9*max(1.,rho*chemistry.initial_lp_per_kg):
            raise PropagationCandidateInputError("Onset integrated LP/Faradaic/global-reaction budget is inconsistent")
        initial_reactive = scalar("initialReactiveMass_kg_per_m3")
        expected_reactive = rho*(chemistry.m_lp*chemistry.initial_lp_per_kg+chemistry.m_pva*chemistry.initial_pva_per_kg)
        if not math.isclose(initial_reactive,expected_reactive,rel_tol=1e-12,abs_tol=1e-12):
            raise PropagationCandidateInputError("Invalid initial LP+PVA mass normalization")
        if U[...,WATER].mean() > rho*chemistry.initial_water_per_kg+1e-8:
            raise PropagationCandidateInputError("Onset mobile water exceeds initial inventory")
        potential = field("potentialAtOnset_V")
        qj,qe = field("qJAtOnset_W_per_m3",True),field("qEchemAtOnset_W_per_m3",True)
        if np.any(np.abs(qe[~(anode|cathode)])>1e-12):
            raise PropagationCandidateInputError("Onset q_echem has support outside the surface contact footprints")
        thermo.validate(U)
        recovered = thermo.primitive(U)
        dx=domain/shape[1]
        vol=dx*dx*depth
        audit = {
            "handoff_policy":"single_continuum_identity_mass_and_energy_transfer_not_solid_to_gas",
            "bc_config_sha256":digest,"onset_time_s":onset,
            "density_source":"resolved_BC_bulk_density", "velocity_initialization":"zero_BC_has_no_momentum",
            "energy_definition":"rho*(BC_cp_integral+Tait_cold_energy+kinetic_energy)",
            "temperature_roundtrip_max_error_K":float(np.max(np.abs(recovered[...,3]-temperature))),
            "channel1_roundtrip_max_error":float(np.max(np.abs(recovered[...,A1]-P[...,A1]))),
            "channel2_roundtrip_max_error":float(np.max(np.abs(recovered[...,A2]-P[...,A2]))),
            "initial_continuum_mass_kg":float(np.sum(U[...,RHO])*vol),
            "initial_total_energy_J":float(np.sum(U[...,ENERGY])*vol),
            "qj_onset_policy":"diagnostic_only_never_replayed",
            "qe_onset_policy":"diagnostic_only_never_replayed",
            "caloric_closure":"explicit_thermodynamic_completion_not_specified_in_attached_paper",
            "same_eos_for_reactant_and_product":True,
            "material_interface_scope":"unreacted_to_reacted_internal_front_not_gas_ablation_boundary",
        }
        return AdaptedBCState(U,anode,cathode,prop_mask,potential,qj,qe,dx,depth,onset,thermo,chemistry,audit)
