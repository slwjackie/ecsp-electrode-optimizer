"""BC -> two-channel condensed Euler/Fourier continuation, CPU FP64.

One energy, one density, two advected reaction channels, six inventory ledgers.
The legacy mode advances all sources with SSPRK33.  The stiff local mode uses a
chemistry half-step, a nonchemical SSPRK33 step, and a second chemistry
half-step. Invalid attempts are retried transactionally with smaller ``dt``;
failed runs never silently become baseline runs or receive completed metrics.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import csv
import json
import math
import time
import numpy as np
from ecsp_nsga2.propagation import (PropagationConfigurationError, PropagationCandidateNumericalError,
    ConfiguredModelTemperatureRangeExceeded,candidate_failure_payload,
    _table_property,_div_k_grad,_arrival_time_statistics,_reaction_level_set,
    _front_edge_count,_effective_regression_velocity,_configuration_integer,_thermal_diffusive_cfl)
from .chemistry import *
from .handoff import BCReactiveHandoffAdapter
from .finite_volume import flux_divergence, mechanically_stationary
from .electrical import BCElectricalAdapter


def _write_json(path,payload):
    Path(path).write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")


def _positive_configuration_float(raw,*,name):
    if isinstance(raw,bool) or not isinstance(raw,(int,float)):
        raise PropagationConfigurationError(name+" must be a positive finite number")
    value=float(raw)
    if not math.isfinite(value) or value<=0:
        raise PropagationConfigurationError(name+" must be a positive finite number")
    return value


class BCReactiveSolver:
    def __init__(self,adapted,propagation_config,bc_config,reactive_config,*,full_bc_config=None,initialize_electrical=True):
        self.a=adapted
        self.prop=dict(propagation_config); self.bc=dict(bc_config); self.cfg=dict(reactive_config)
        self.U=adapted.U.copy(); self.thermo=adapted.thermo; self.chem=adapted.chemistry
        self.duration=float(self.prop["duration_s"]); self.dtmax=float(self.prop["time_step_s"])
        self.snapshot_interval=float(self.prop.get("snapshot_interval_s",self.duration/10))
        self.max_steps=_configuration_integer(self.cfg.get("maximum_time_steps",self.prop.get("maximum_time_steps",100000)),
                    name="reactive maximum_time_steps",minimum=1,maximum=2**31-1)
        self.max_retries=_configuration_integer(self.cfg.get("maximum_step_retries",12),name="maximum_step_retries",minimum=0,maximum=30)
        self.cfl=float(self.cfg.get("cfl",.35))
        self.thermal_cfl=float(self.cfg.get("thermal_cfl",.7))
        self.chemistry_integration_mode=str(
            self.cfg.get("chemistry_integration_mode", "legacy_cap")
        ).lower()
        if self.chemistry_integration_mode == "subcycle_raw":
            raise PropagationConfigurationError(
                "subcycle_raw is retired from the production Reactive Euler "
                "solver; use local_adaptive_thermochemical"
            )
        if self.chemistry_integration_mode not in {
                "legacy_cap", "local_adaptive_thermochemical"}:
            raise PropagationConfigurationError(
                "chemistry_integration_mode must be legacy_cap or "
                "local_adaptive_thermochemical"
            )
        legacy_controls = {
            "maximum_channel_increment",
            "maximum_chemistry_subcycles_per_pde_step",
            "maximum_chemical_rate_cap_fraction",
        }
        if self.chemistry_integration_mode == "local_adaptive_thermochemical":
            supplied_legacy = sorted(legacy_controls.intersection(self.cfg))
            if supplied_legacy:
                raise PropagationConfigurationError(
                    "Local adaptive chemistry does not accept retired legacy "
                    "cap/subcycle controls: "+", ".join(supplied_legacy)
                )
            self.alpha_step=None
            self.max_chemistry_subcycles=None
        else:
            self.alpha_step=float(self.cfg.get("maximum_channel_increment",.02))
            self.max_chemistry_subcycles=_configuration_integer(
                self.cfg.get("maximum_chemistry_subcycles_per_pde_step", 4096),
                name="maximum_chemistry_subcycles_per_pde_step",
                minimum=1, maximum=1000000,
            )
        self.max_chemistry_subcycles_used=0
        self.chemistry_relative_tolerance=_positive_configuration_float(
            self.cfg.get("chemistry_relative_tolerance",1e-7),
            name="chemistry_relative_tolerance")
        self.chemistry_absolute_tolerance=_positive_configuration_float(
            self.cfg.get("chemistry_absolute_tolerance",1e-10),
            name="chemistry_absolute_tolerance")
        self.chemistry_temperature_tolerance_K=_positive_configuration_float(
            self.cfg.get("chemistry_temperature_tolerance_K",1e-4),
            name="chemistry_temperature_tolerance_K")
        # Keep the scalar call-site name explicit while exposing the ``_K``
        # spelling shared by the tensor kernel.
        self.chemistry_temperature_tolerance=self.chemistry_temperature_tolerance_K
        self.max_chemistry_corrector_iterations=_configuration_integer(
            self.cfg.get("maximum_chemistry_corrector_iterations",8),
            name="maximum_chemistry_corrector_iterations",minimum=1,maximum=1000000)
        self.max_chemistry_depletion_iterations=_configuration_integer(
            self.cfg.get("maximum_chemistry_depletion_iterations",40),
            name="maximum_chemistry_depletion_iterations",minimum=1,maximum=1000000)
        self.max_chemistry_local_refinements=_configuration_integer(
            self.cfg.get("maximum_chemistry_local_refinements",10),
            name="maximum_chemistry_local_refinements",minimum=1,maximum=20)
        self.max_chemistry_reaction_coordinate_steps=_configuration_integer(
            self.cfg.get("maximum_chemistry_reaction_coordinate_steps",64),
            name="maximum_chemistry_reaction_coordinate_steps",
            minimum=1,maximum=1000000)
        self.progress_log_interval_steps=_configuration_integer(
            self.cfg.get("progress_log_interval_steps",250),
            name="progress_log_interval_steps",minimum=1,maximum=2**31-1)
        self.progress_log_interval_wall_s=_positive_configuration_float(
            self.cfg.get("progress_log_interval_wall_s",30.),
            name="progress_log_interval_wall_s")
        self.min_dt=float(self.cfg.get("minimum_time_step_s",1e-14))
        self.boundary=str(self.cfg.get("boundary","reflective"))
        self.riemann=str(self.cfg.get("riemann_solver","hllc"))
        fast_raw=self.cfg.get("stationary_mechanics_fast_path",True)
        if not isinstance(fast_raw,bool):
            raise PropagationConfigurationError("stationary_mechanics_fast_path must be boolean")
        self.fast=fast_raw
        self.epsilon=float(self.cfg.get("weno_epsilon",1e-6))
        self.front_threshold=float(self.prop.get("front_progress_threshold",.5))
        self.established_fraction=float(self.prop.get("established_reacted_area_fraction",.5))
        self.mass_tolerance=float(self.cfg.get("mass_budget_relative_tolerance",1e-10))
        self.energy_tolerance=float(self.cfg.get("energy_budget_relative_tolerance",1e-9))
        if (not all(math.isfinite(v) and v>0 for v in (self.duration,self.dtmax,self.snapshot_interval,self.min_dt,
                self.cfl,self.thermal_cfl,self.epsilon,self.mass_tolerance,self.energy_tolerance,
                self.chemistry_relative_tolerance,self.chemistry_absolute_tolerance,
                self.chemistry_temperature_tolerance_K))
                or not 0<self.cfl<=.5 or not 0<self.thermal_cfl<=1
                or not 0<self.front_threshold<=1 or not 0<self.established_fraction<=1):
            raise PropagationConfigurationError("Invalid condensed Reactive time/quality/front parameters")
        if (self.chemistry_integration_mode == "legacy_cap"
                and not 0<float(self.alpha_step)<=1):
            raise PropagationConfigurationError(
                "Invalid legacy maximum_channel_increment"
            )
        if self.boundary not in {"reflective","periodic","transmissive"} or self.riemann not in {"hll","hllc"}:
            raise PropagationConfigurationError("Unsupported condensed boundary/Riemann solver")
        if self.fast and self.riemann!="hllc":
            raise PropagationConfigurationError("Exact stationary fast path requires contact-preserving HLLC")
        if int(math.ceil(self.duration/self.dtmax)) > self.max_steps:
            raise PropagationConfigurationError("Reactive minimum required steps exceeds maximum_time_steps")
        maxbytes=int(self.prop.get("maximum_history_allocation_bytes",2*1024**3))
        snapshots=2+math.ceil(self.duration/self.snapshot_interval)
        if snapshots*np.prod(self.U.shape[:2])*8*10*2 + self.max_steps*8*28 > maxbytes:
            raise PropagationConfigurationError("Reactive retained-history budget exceeds maximum_history_allocation_bytes")
        self.thermal=self.bc["thermal"]
        self.ambient=float(self.thermal.get("ambientTemperature_K",298.15))
        self.h=float(self.thermal.get("convectionCoefficient_W_per_m2K",0))
        self.emissivity=float(self.thermal.get("emissivity",0))
        self.sigma_sb=float(self.thermal.get("stefanBoltzmann_W_per_m2K4",5.670374419e-8))
        if (not all(math.isfinite(v) for v in (self.ambient,self.h,self.emissivity,self.sigma_sb))
                or self.h<0 or not 0<=self.emissivity<=1 or self.sigma_sb<0 or self.ambient<=0):
            raise PropagationConfigurationError("Invalid BC thermal-loss inputs")
        self.vol=self.a.dx**2*self.a.thickness
        heating_raw=self.prop.get("continued_electrical_heating",False)
        if not isinstance(heating_raw,bool):
            raise PropagationConfigurationError("continued_electrical_heating must be boolean")
        self.heating=heating_raw
        mode=str(self.prop.get("electrical_heating_mode","off")).lower()
        if not self.heating and mode not in {"off","none","disabled"}:
            raise PropagationConfigurationError("Power-off continuation contradicts electrical_heating_mode")
        if self.heating and mode not in {"recomputed","recompute"}:
            raise PropagationConfigurationError("Power-on Reactive requires recomputed BC NP/BV; frozen sources/history replay are forbidden")
        self.electrical=BCElectricalAdapter(adapted,full_bc_config) if self.heating and initialize_electrical else None
        self.concentration_floor=0.0
        if self.electrical is not None:
            tr=self.electrical.cfg["transport"]
            frac=float(tr["concentrationMinimumFraction"])
            self.concentration_floor=max(float(self.electrical.cfg.get("physicalFloors",{}).get("concentration_mol_per_m3",1e-12)),
                                         frac*max(self.electrical.composition.initial_cation_mol_per_m3,
                                                  self.electrical.composition.initial_anion_mol_per_m3))
        if self.chemistry_integration_mode == "legacy_cap":
            self.allowed_cap=float(
                self.cfg.get(
                    "maximum_chemical_rate_cap_fraction",
                    self.prop.get("maximum_chemical_rate_cap_fraction", 0),
                )
            )
            if not 0<=self.allowed_cap<=1:
                raise PropagationConfigurationError(
                    "Invalid chemical-rate cap tolerance"
                )
        else:
            self.allowed_cap=None
        self.step_number=0; self.rejected_steps=0; self.first_order_steps=0
        self.face_fallbacks=0; self.faces=0; self.hllc_fallbacks=0; self.exact_skipped_steps=0
        self.last_sources={}
        self.max_rate_cap=0.; self.max_inventory_limiter=0.
        self.max_rate_threshold_exceedance=0.
        self.max_inventory_depleted_fraction=0.
        self.chemistry_half_step_attempts=0; self.chemistry_half_steps_accepted=0
        self.chemistry_failed_half_steps=0
        self.post_chemistry_timestep_rechecks=0; self.post_chemistry_timestep_recheck_rejections=0
        self.minimum_post_chemistry_timestep_safety_ratio=math.inf
        self.minimum_accepted_post_chemistry_timestep_safety_ratio=math.inf
        self.max_chemistry_corrector_iterations_used=0
        self.max_chemistry_depletion_iterations_used=0
        self.max_chemistry_local_refinements_used=0
        self.maximum_chemistry_residual=0.; self.maximum_chemistry_temperature_residual_K=0.
        self.maximum_normalized_embedded_alpha_residual=0.
        self.maximum_normalized_embedded_temperature_residual=0.
        self.maximum_rejected_normalized_embedded_residual=0.
        self.maximum_normalized_corrector_residual=0.
        self.maximum_temperature_feedback_alpha_correction=0.
        self.maximum_temperature_feedback_endpoint_temperature_correction_K=0.
        self.maximum_raw_chemical_rate_per_s=0.
        self.maximum_raw_chemical_rate_initial_per_s=0.
        self.maximum_raw_chemical_rate_endpoint_per_s=0.
        self.chemistry_raw_rate_observation_scope=None
        self.maximum_chemistry_temperature_rise_K=0.
        self.maximum_chemistry_caloric_inverse_residual_J_per_kg=0.
        self.maximum_chemistry_heat_closure_residual_J_per_m3=0.
        self.maximum_chemistry_inventory_event_residual_mol_per_m3=0.
        self.maximum_local_refined_cell_fraction=0.
        self.minimum_inventory_depletion_time_fraction=math.inf
        self.maximum_inventory_depletion_time_fraction=0.
        self.maximum_nonconverged_chemistry_cell_count=0
        self.maximum_attempted_chemistry_corrector_iterations=0
        self.maximum_attempted_chemistry_depletion_iterations=0
        self.maximum_attempted_chemistry_local_refinements=0
        self.maximum_attempted_chemistry_residual=0.
        self.maximum_attempted_chemistry_temperature_residual_K=0.
        self.maximum_attempted_normalized_embedded_alpha_residual=0.
        self.maximum_attempted_normalized_embedded_temperature_residual=0.
        self.maximum_attempted_rejected_normalized_embedded_residual=0.
        self.maximum_attempted_normalized_corrector_residual=0.
        self.maximum_attempted_temperature_feedback_alpha_correction=0.
        self.maximum_attempted_temperature_feedback_endpoint_temperature_correction_K=0.
        self.maximum_attempted_raw_chemical_rate_per_s=0.
        self.maximum_attempted_raw_chemical_rate_initial_per_s=0.
        self.maximum_attempted_raw_chemical_rate_endpoint_per_s=0.
        self.attempted_chemistry_raw_rate_observation_scope=None
        self.maximum_attempted_rate_threshold_exceedance_fraction=0.
        self.maximum_attempted_inventory_depleted_fraction=0.
        self.maximum_attempted_chemistry_temperature_rise_K=0.
        self.maximum_attempted_chemistry_caloric_inverse_residual_J_per_kg=0.
        self.maximum_attempted_chemistry_heat_closure_residual_J_per_m3=0.
        self.maximum_attempted_chemistry_inventory_event_residual_mol_per_m3=0.
        self.maximum_attempted_local_refined_cell_fraction=0.
        self.minimum_attempted_inventory_depletion_time_fraction=math.inf
        self.maximum_attempted_inventory_depletion_time_fraction=0.
        self.maximum_attempted_nonconverged_chemistry_cell_count=0
        self.chemistry_wall_clock_time_s=0.; self.nonchemical_wall_clock_time_s=0.
        self.accepted_chemistry_endpoint_evaluations=0
        self.attempted_chemistry_endpoint_evaluations=0
        self.accepted_chemistry_corrector_iteration_sum=0
        self.accepted_chemistry_depletion_iteration_sum=0
        self.accepted_chemistry_evaluated_cell_count=0
        self.attempted_chemistry_corrector_iteration_sum=0
        self.attempted_chemistry_depletion_iteration_sum=0
        self.attempted_chemistry_evaluated_cell_count=0
        self.accepted_reaction_coordinate_cell_count=0
        self.accepted_reaction_coordinate_rate_evaluations=0
        self.accepted_reaction_coordinate_quadrature_panel_evaluations=0
        self.accepted_reaction_coordinate_root_iterations=0
        self.maximum_reaction_coordinate_root_iterations=0
        self.maximum_reaction_coordinate_quadrature_refinement_depth=0
        self.maximum_reaction_coordinate_normalized_quadrature_residual=0.
        self.attempted_reaction_coordinate_cell_count=0
        self.attempted_reaction_coordinate_rate_evaluations=0
        self.attempted_reaction_coordinate_quadrature_panel_evaluations=0
        self.attempted_reaction_coordinate_root_iterations=0
        self.maximum_attempted_reaction_coordinate_root_iterations=0
        self.maximum_attempted_reaction_coordinate_quadrature_refinement_depth=0
        self.maximum_attempted_reaction_coordinate_normalized_quadrature_residual=0.
        self.accepted_coupled_reaction_coordinate_attempts=0
        self.accepted_coupled_reaction_coordinate_cells=0
        self.accepted_coupled_reaction_coordinate_fallbacks=0
        self.accepted_coupled_reaction_coordinate_rate_evaluations=0
        self.accepted_coupled_reaction_coordinate_steps=0
        self.accepted_coupled_reaction_coordinate_rejected_steps=0
        self.accepted_coupled_reaction_coordinate_driver_switches=0
        self.maximum_coupled_reaction_coordinate_refinement_depth=0
        self.maximum_coupled_reaction_coordinate_normalized_residual=0.
        self.maximum_rejected_coupled_reaction_coordinate_residual=0.
        self.attempted_coupled_reaction_coordinate_attempts=0
        self.attempted_coupled_reaction_coordinate_cells=0
        self.attempted_coupled_reaction_coordinate_fallbacks=0
        self.attempted_coupled_reaction_coordinate_rate_evaluations=0
        self.attempted_coupled_reaction_coordinate_steps=0
        self.attempted_coupled_reaction_coordinate_rejected_steps=0
        self.attempted_coupled_reaction_coordinate_driver_switches=0
        self.maximum_attempted_coupled_reaction_coordinate_refinement_depth=0
        self.maximum_attempted_coupled_reaction_coordinate_normalized_residual=0.
        self.maximum_attempted_rejected_coupled_reaction_coordinate_residual=0.
        self._initial_sums=None; self._initial_reservoir=None

    def _thermal_terms(self,U):
        P=self.thermo.primitive(U); T=P[...,3]
        cp=self.thermo.heat.cp(T)
        k=_table_property(T,self.thermal["thermal_conductivity"],self.chem.R)
        if not np.isfinite(k).all() or np.any(k<0):
            raise PropagationCandidateNumericalError("Nonphysical BC thermal conductivity")
        conduction=_div_k_grad(T,k,self.a.propellant_mask,self.a.dx)
        loss=self.h/self.a.thickness*(T-self.ambient)+self.emissivity*self.sigma_sb/self.a.thickness*(T**4-self.ambient**4)
        return P,cp,k,conduction,loss

    def _nonchemical_time_step_limit(self,U):
        """Return the explicit Euler/Fourier stability limit at ``U``.

        Local chemistry is intentionally absent.  Split-mode callers evaluate
        this once before a trial and again after the first chemistry half-step,
        because heat release can change the explicit Fourier/loss limit.
        """
        P,cp,k,_,_=self._thermal_terms(U)
        rho=P[...,RHO]; T=P[...,3]
        loss_jac=self.h/self.a.thickness+4*self.emissivity*self.sigma_sb/self.a.thickness*np.maximum(T,self.ambient)**3
        inverse_thermal=np.max(_thermal_diffusive_cfl(k,rho*cp,self.a.propellant_mask,self.a.dx,1.)+loss_jac/(rho*cp))
        thermal_dt=self.thermal_cfl/inverse_thermal if inverse_thermal>0 else math.inf
        stationary=self.fast and self.riemann=="hllc" and mechanically_stationary(U,self.boundary)
        if stationary:
            acoustic_dt=math.inf
        else:
            sound=self.thermo.sound_speed(rho)
            acoustic_dt=self.cfl/np.max((np.abs(P[...,MX])+np.abs(P[...,MY])+2*sound)/self.a.dx)
        return min(thermal_dt,acoustic_dt)

    def time_step(self,U,remaining):
        nonchemical_dt=self._nonchemical_time_step_limit(U)
        if self.chemistry_integration_mode == "legacy_cap":
            P=self.thermo.primitive(U); T=P[...,3]
            rates=self.chem.raw_rates(U,T)
            has_stock=(.5*(U[...,CATION]+U[...,ANION])>self.concentration_floor)&(U[...,PVA]>0)
            effective_rates=np.minimum(rates,self.chem.maximum_rate)
            maxrate=float(np.max(np.where(has_stock[...,None],effective_rates,0.)))
            reaction_dt=self.alpha_step/maxrate if maxrate>0 else math.inf
        else:
            # Stiff chemistry is integrated by a local state map; it must
            # not force the entire Euler/Fourier PDE onto the chemistry scale.
            reaction_dt=math.inf
        dt=min(self.dtmax,remaining,nonchemical_dt,reaction_dt)
        if not math.isfinite(dt) or dt<=0 or dt<self.min_dt:
            raise PropagationCandidateNumericalError("Reactive timestep fell below minimum_time_step_s; no artificial sound-speed reduction used")
        return dt

    def rhs(self,t,U,dt,first_order=False):
        """Legacy coupled SSPRK RHS, retained without changing its physics."""
        hydro,boundary,fd=flux_divergence(U,self.thermo,self.a.dx,self.a.thickness,boundary=self.boundary,
                epsilon=self.epsilon,kind=self.riemann,first_order=first_order,stationary_fast_path=self.fast)
        P,cp,k,conduction,loss=self._thermal_terms(U)
        T=P[...,3]
        jac=self.h/self.a.thickness+4*self.emissivity*self.sigma_sb/self.a.thickness*np.maximum(T,self.ambient)**3
        local_cfl=_thermal_diffusive_cfl(k,P[...,0]*cp,self.a.propellant_mask,self.a.dx,dt)+dt*jac/(P[...,0]*cp)
        if np.max(local_cfl)>self.thermal_cfl*(1+1e-12):
            raise PropagationCandidateNumericalError("Reactive explicit thermal CFL exceeded at an RK stage")
        if self.electrical is not None:
            electric,qj,qe,mismatch=self.electrical.evaluate(U,T,dt)
        else:
            electric=np.zeros_like(U); qj=np.zeros(U.shape[:2]); qe=qj; mismatch=0.
        available=U+dt*(hydro+electric)
        if np.any(available[...,CATION:WATER+1]<-1e-9) or np.any(available[...,PVA]<-1e-9):
            raise PropagationCandidateNumericalError("Flux/NP stage would make an inventory negative")
        chemistry,cd=self.chem.source(
            U,T,dt,available=available,
            concentration_floor=self.concentration_floor,
            allowed_rate_cap_fraction=self.allowed_cap,
            integration_mode=self.chemistry_integration_mode,
            maximum_channel_increment=self.alpha_step,
            maximum_subcycles=self.max_chemistry_subcycles,
        )
        derivative=hydro+electric+chemistry
        derivative[...,ENERGY]+=conduction-loss
        if not np.isfinite(derivative).all():
            raise PropagationCandidateNumericalError("Nonfinite condensed Euler source")
        # Source terms use exactly the accepted rates. No second solid update.
        ledger=np.r_[boundary, [np.sum(v)*self.vol for v in
                    (qj,qe,chemistry[...,ENERGY],loss,conduction)]]
        source_fields={"qJ_W_per_m3":qj,"qEchem_W_per_m3":qe,
                       "qChem_W_per_m3":chemistry[...,ENERGY],"conduction_W_per_m3":conduction,
                       "loss_W_per_m3":loss}
        return derivative,ledger,{**fd,**cd,"current_mismatch":mismatch},source_fields

    def _advance_coupled_ssprk(self,t,U,dt,first_order=False):
        """SSPRK(3,3), with source/boundary quadrature weights 1/6,1/6,2/3."""
        k0,l0,d0,f0=self.rhs(t,U,dt,first_order)
        u1=U+dt*k0; self.thermo.validate(u1)
        k1,l1,d1,f1=self.rhs(t+dt,u1,dt,first_order)
        u2=.75*U+.25*(u1+dt*k1); self.thermo.validate(u2)
        k2,l2,d2,f2=self.rhs(t+.5*dt,u2,dt,first_order)
        result=U/3+(2/3)*(u2+dt*k2); self.thermo.validate(result)
        # Restore exactly constant mechanically inactive fields only if they
        # were analytically unchanged: avoid RK convex-combination roundoff
        # breaking the uniform-rho invariant and triggering acoustic work.
        if all(d["mechanics_skipped_exact"] for d in (d0,d1,d2)):
            result[...,:3]=U[...,:3]
        integrated=dt*(l0/6+l1/6+2*l2/3)
        sources={k:f0[k]/6+f1[k]/6+2*f2[k]/3 for k in f0}
        return result,integrated,(d0,d1,d2),sources

    def _nonchemical_rhs(self,t,U,dt,first_order=False):
        """Explicit Euler/Fourier/electrical RHS used by the split mode."""
        hydro,boundary,fd=flux_divergence(U,self.thermo,self.a.dx,self.a.thickness,boundary=self.boundary,
                epsilon=self.epsilon,kind=self.riemann,first_order=first_order,stationary_fast_path=self.fast)
        P,cp,k,conduction,loss=self._thermal_terms(U)
        T=P[...,3]
        jac=self.h/self.a.thickness+4*self.emissivity*self.sigma_sb/self.a.thickness*np.maximum(T,self.ambient)**3
        local_cfl=_thermal_diffusive_cfl(k,P[...,0]*cp,self.a.propellant_mask,self.a.dx,dt)+dt*jac/(P[...,0]*cp)
        if np.max(local_cfl)>self.thermal_cfl*(1+1e-12):
            raise PropagationCandidateNumericalError("Reactive explicit thermal CFL exceeded at an RK stage")
        if self.electrical is not None:
            electric,qj,qe,mismatch=self.electrical.evaluate(U,T,dt)
        else:
            electric=np.zeros_like(U); qj=np.zeros(U.shape[:2]); qe=qj; mismatch=0.
        available=U+dt*(hydro+electric)
        if np.any(available[...,CATION:WATER+1]<-1e-9) or np.any(available[...,PVA]<-1e-9):
            raise PropagationCandidateNumericalError("Flux/NP stage would make an inventory negative")
        derivative=hydro+electric
        derivative[...,ENERGY]+=conduction-loss
        if not np.isfinite(derivative).all():
            raise PropagationCandidateNumericalError("Nonfinite condensed Euler source")
        zero=np.zeros_like(qj)
        ledger=np.r_[boundary, [np.sum(v)*self.vol for v in
                    (qj,qe,zero,loss,conduction)]]
        source_fields={"qJ_W_per_m3":qj,"qEchem_W_per_m3":qe,
                       "qChem_W_per_m3":zero,"conduction_W_per_m3":conduction,
                       "loss_W_per_m3":loss}
        return derivative,ledger,{**fd,"current_mismatch":mismatch},source_fields

    def _advance_nonchemical_ssprk(self,t,U,dt,first_order=False):
        """Advance only nonchemical terms with SSPRK(3,3)."""
        k0,l0,d0,f0=self._nonchemical_rhs(t,U,dt,first_order)
        u1=U+dt*k0; self.thermo.validate(u1)
        k1,l1,d1,f1=self._nonchemical_rhs(t+dt,u1,dt,first_order)
        u2=.75*U+.25*(u1+dt*k1); self.thermo.validate(u2)
        k2,l2,d2,f2=self._nonchemical_rhs(t+.5*dt,u2,dt,first_order)
        result=U/3+(2/3)*(u2+dt*k2); self.thermo.validate(result)
        if all(d["mechanics_skipped_exact"] for d in (d0,d1,d2)):
            result[...,:3]=U[...,:3]
        integrated=dt*(l0/6+l1/6+2*l2/3)
        sources={k:f0[k]/6+f1[k]/6+2*f2[k]/3 for k in f0}
        return result,integrated,(d0,d1,d2),sources

    def _chemistry_half_step(self,U,dt):
        """Apply one tentative local chemistry map and return its exact heat."""
        self.chemistry_half_step_attempts+=1
        before=np.array(U,copy=True)
        chemistry_start=time.perf_counter()
        try:
            result,diagnostics=self.chem.advance_local(
                before,self.thermo,dt,
                concentration_floor=self.concentration_floor,
                relative_tolerance=self.chemistry_relative_tolerance,
                absolute_tolerance=self.chemistry_absolute_tolerance,
                temperature_tolerance_K=self.chemistry_temperature_tolerance,
                maximum_corrector_iterations=self.max_chemistry_corrector_iterations,
                maximum_depletion_iterations=self.max_chemistry_depletion_iterations,
                maximum_local_refinements=self.max_chemistry_local_refinements,
                maximum_reaction_coordinate_steps=(
                    self.max_chemistry_reaction_coordinate_steps
                ),
            )
        except ConfiguredModelTemperatureRangeExceeded:
            self.chemistry_failed_half_steps+=1
            raise
        except PropagationCandidateNumericalError as error:
            self.chemistry_failed_half_steps+=1
            error.retry_reason="local_chemistry"
            raise
        finally:
            self.chemistry_wall_clock_time_s+=time.perf_counter()-chemistry_start
        result=np.asarray(result,dtype=np.float64)
        if result.shape!=before.shape or not isinstance(diagnostics,Mapping):
            self.chemistry_failed_half_steps+=1
            error=PropagationCandidateNumericalError(
                "Local chemistry returned an invalid state/diagnostics contract")
            error.retry_reason="local_chemistry"
            raise error
        try:
            self.thermo.validate(result)
        except PropagationCandidateNumericalError as error:
            self.chemistry_failed_half_steps+=1
            error.retry_reason="local_chemistry"
            raise
        heat_increment=result[...,ENERGY]-before[...,ENERGY]
        if not np.isfinite(heat_increment).all():
            self.chemistry_failed_half_steps+=1
            error=PropagationCandidateNumericalError("Local chemistry returned nonfinite heat")
            error.retry_reason="local_chemistry"
            raise error
        diagnostics=dict(diagnostics)
        self._observe_local_chemistry(diagnostics,accepted=False)
        return result,heat_increment,diagnostics

    @staticmethod
    def _diagnostic_max(diagnostics,*names):
        for name in names:
            if name in diagnostics:
                try:
                    values=np.asarray(diagnostics[name],dtype=np.float64)
                except (TypeError,ValueError):
                    continue
                if values.size and np.isfinite(values).all():
                    return float(np.max(values))
        return 0.

    @staticmethod
    def _diagnostic_min(diagnostics,*names):
        for name in names:
            if name in diagnostics:
                try:
                    values=np.asarray(diagnostics[name],dtype=np.float64)
                except (TypeError,ValueError):
                    continue
                if values.size and np.isfinite(values).all():
                    return float(np.min(values))
        return math.inf

    def _observe_local_chemistry(self,diagnostics,*,accepted):
        raw_rate_scope=diagnostics.get("raw_rate_observation_scope")
        if not isinstance(raw_rate_scope,str):
            raw_rate_scope=None
        corrector=self._diagnostic_max(diagnostics,"maximum_corrector_iterations_used",
            "maximum_corrector_iterations","corrector_iterations")
        depletion=self._diagnostic_max(diagnostics,"maximum_depletion_iterations_used",
            "maximum_depletion_iterations","depletion_iterations")
        refinements=self._diagnostic_max(diagnostics,"maximum_local_refinement_depth",
            "maximum_local_refinements","local_refinements")
        residual=self._diagnostic_max(diagnostics,"maximum_normalized_embedded_residual",
            "maximum_residual","maximum_normalized_residual")
        embedded_alpha_residual=self._diagnostic_max(diagnostics,
            "maximum_normalized_embedded_alpha_residual")
        embedded_temperature_residual=self._diagnostic_max(diagnostics,
            "maximum_normalized_embedded_temperature_residual")
        # Embedded temperature error and the fixed-temperature feedback
        # correction are distinct diagnostics.  Older code accidentally used
        # the latter as the former's dimensional value.  Prefer an explicit
        # Kelvin residual when supplied; the normalized certificate times the
        # configured global tolerance is a conservative dimensional fallback.
        temperature_residual=max(
            self._diagnostic_max(
                diagnostics,"maximum_embedded_temperature_residual_K",
                "maximum_temperature_residual_K","temperature_residual_K"),
            embedded_temperature_residual
                *self.chemistry_temperature_tolerance_K,
        )
        rejected_embedded_residual=self._diagnostic_max(diagnostics,
            "maximum_rejected_normalized_embedded_residual")
        corrector_residual=self._diagnostic_max(diagnostics,
            "maximum_normalized_corrector_residual")
        feedback_alpha_correction=self._diagnostic_max(diagnostics,
            "maximum_temperature_feedback_alpha_correction")
        feedback_temperature_correction=self._diagnostic_max(diagnostics,
            "maximum_temperature_feedback_endpoint_temperature_correction_K")
        raw_rate_initial=self._diagnostic_max(
            diagnostics,"maximum_raw_rate_per_s_initial")
        raw_rate_endpoint=self._diagnostic_max(
            diagnostics,"maximum_raw_rate_per_s_endpoint")
        raw_rate=max(raw_rate_initial,raw_rate_endpoint,
            self._diagnostic_max(diagnostics,"maximum_raw_rate_per_s",
                "maximum_raw_chemical_rate_per_s"))
        threshold_fraction=self._diagnostic_max(diagnostics,
            "chemical_rate_threshold_exceedance_fraction","chemical_rate_cap_fraction",
            "chemical_rate_threshold_fraction")
        depleted_fraction=self._diagnostic_max(diagnostics,
            "inventory_depleted_fraction","inventory_limiter_fraction")
        temperature_rise=self._diagnostic_max(diagnostics,"maximum_temperature_rise_K")
        caloric_residual=self._diagnostic_max(diagnostics,
            "maximum_caloric_inverse_residual_J_per_kg")
        heat_residual=self._diagnostic_max(diagnostics,
            "maximum_heat_closure_residual_J_per_m3")
        inventory_residual=self._diagnostic_max(diagnostics,
            "maximum_inventory_event_residual_mol_per_m3")
        refined_fraction=self._diagnostic_max(diagnostics,"local_refined_cell_fraction")
        depletion_time=self._diagnostic_min(diagnostics,
            "minimum_inventory_depletion_time_fraction")
        maximum_depletion_time=self._diagnostic_max(diagnostics,
            "maximum_inventory_depletion_time_fraction")
        nonconverged=int(self._diagnostic_max(diagnostics,"nonconverged_cell_count"))
        endpoint_evaluations=int(self._diagnostic_max(
            diagnostics,"endpoint_evaluation_count"))
        corrector_iteration_sum=int(self._diagnostic_max(
            diagnostics,"corrector_iteration_sum"))
        depletion_iteration_sum=int(self._diagnostic_max(
            diagnostics,"depletion_iteration_sum"))
        evaluated_cell_count=int(self._diagnostic_max(
            diagnostics,"evaluated_cell_count"))
        reaction_coordinate_cells=int(self._diagnostic_max(
            diagnostics,"one_active_reaction_coordinate_cell_count"))
        reaction_coordinate_rate_evaluations=int(self._diagnostic_max(
            diagnostics,"one_active_reaction_coordinate_rate_evaluation_count"))
        reaction_coordinate_quadrature_evaluations=int(self._diagnostic_max(
            diagnostics,
            "one_active_reaction_coordinate_quadrature_panel_evaluation_count"))
        reaction_coordinate_root_iteration_sum=int(self._diagnostic_max(
            diagnostics,"one_active_reaction_coordinate_root_iteration_count"))
        reaction_coordinate_root_iterations=int(self._diagnostic_max(
            diagnostics,"maximum_one_active_reaction_coordinate_root_iterations"))
        reaction_coordinate_quadrature_depth=int(self._diagnostic_max(
            diagnostics,
            "maximum_one_active_reaction_coordinate_quadrature_refinement_depth"))
        reaction_coordinate_quadrature_residual=self._diagnostic_max(
            diagnostics,
            "maximum_one_active_reaction_coordinate_normalized_quadrature_residual")
        coupled_attempts=int(self._diagnostic_max(
            diagnostics,"coupled_reaction_coordinate_attempt_count"))
        coupled_cells=int(self._diagnostic_max(
            diagnostics,"coupled_reaction_coordinate_cell_count"))
        coupled_fallbacks=int(self._diagnostic_max(
            diagnostics,"coupled_reaction_coordinate_fallback_count"))
        coupled_rate_evaluations=int(self._diagnostic_max(
            diagnostics,"coupled_reaction_coordinate_rate_evaluation_count"))
        coupled_steps=int(self._diagnostic_max(
            diagnostics,"coupled_reaction_coordinate_accepted_step_count"))
        coupled_rejected_steps=int(self._diagnostic_max(
            diagnostics,"coupled_reaction_coordinate_rejected_step_count"))
        coupled_driver_switches=int(self._diagnostic_max(
            diagnostics,"coupled_reaction_coordinate_driver_switch_count"))
        coupled_depth=int(self._diagnostic_max(
            diagnostics,"maximum_coupled_reaction_coordinate_refinement_depth"))
        coupled_residual=self._diagnostic_max(
            diagnostics,"maximum_coupled_reaction_coordinate_normalized_residual")
        coupled_rejected_residual=self._diagnostic_max(
            diagnostics,"maximum_rejected_coupled_reaction_coordinate_residual")
        if not accepted:
            if raw_rate_scope is not None:
                previous=self.attempted_chemistry_raw_rate_observation_scope
                self.attempted_chemistry_raw_rate_observation_scope=(
                    raw_rate_scope if previous in {None,raw_rate_scope} else "mixed")
            self.maximum_attempted_chemistry_corrector_iterations=max(
                self.maximum_attempted_chemistry_corrector_iterations,int(corrector))
            self.maximum_attempted_chemistry_depletion_iterations=max(
                self.maximum_attempted_chemistry_depletion_iterations,int(depletion))
            self.maximum_attempted_chemistry_local_refinements=max(
                self.maximum_attempted_chemistry_local_refinements,int(refinements))
            self.maximum_attempted_chemistry_residual=max(
                self.maximum_attempted_chemistry_residual,residual)
            self.maximum_attempted_chemistry_temperature_residual_K=max(
                self.maximum_attempted_chemistry_temperature_residual_K,temperature_residual)
            self.maximum_attempted_normalized_embedded_alpha_residual=max(
                self.maximum_attempted_normalized_embedded_alpha_residual,
                embedded_alpha_residual)
            self.maximum_attempted_normalized_embedded_temperature_residual=max(
                self.maximum_attempted_normalized_embedded_temperature_residual,
                embedded_temperature_residual)
            self.maximum_attempted_rejected_normalized_embedded_residual=max(
                self.maximum_attempted_rejected_normalized_embedded_residual,
                rejected_embedded_residual)
            self.maximum_attempted_normalized_corrector_residual=max(
                self.maximum_attempted_normalized_corrector_residual,corrector_residual)
            self.maximum_attempted_temperature_feedback_alpha_correction=max(
                self.maximum_attempted_temperature_feedback_alpha_correction,
                feedback_alpha_correction)
            self.maximum_attempted_temperature_feedback_endpoint_temperature_correction_K=max(
                self.maximum_attempted_temperature_feedback_endpoint_temperature_correction_K,
                feedback_temperature_correction)
            self.maximum_attempted_raw_chemical_rate_per_s=max(
                self.maximum_attempted_raw_chemical_rate_per_s,raw_rate)
            self.maximum_attempted_raw_chemical_rate_initial_per_s=max(
                self.maximum_attempted_raw_chemical_rate_initial_per_s,raw_rate_initial)
            self.maximum_attempted_raw_chemical_rate_endpoint_per_s=max(
                self.maximum_attempted_raw_chemical_rate_endpoint_per_s,raw_rate_endpoint)
            self.maximum_attempted_rate_threshold_exceedance_fraction=max(
                self.maximum_attempted_rate_threshold_exceedance_fraction,threshold_fraction)
            self.maximum_attempted_inventory_depleted_fraction=max(
                self.maximum_attempted_inventory_depleted_fraction,depleted_fraction)
            self.maximum_attempted_chemistry_temperature_rise_K=max(
                self.maximum_attempted_chemistry_temperature_rise_K,temperature_rise)
            self.maximum_attempted_chemistry_caloric_inverse_residual_J_per_kg=max(
                self.maximum_attempted_chemistry_caloric_inverse_residual_J_per_kg,
                caloric_residual)
            self.maximum_attempted_chemistry_heat_closure_residual_J_per_m3=max(
                self.maximum_attempted_chemistry_heat_closure_residual_J_per_m3,heat_residual)
            self.maximum_attempted_chemistry_inventory_event_residual_mol_per_m3=max(
                self.maximum_attempted_chemistry_inventory_event_residual_mol_per_m3,inventory_residual)
            self.maximum_attempted_local_refined_cell_fraction=max(
                self.maximum_attempted_local_refined_cell_fraction,refined_fraction)
            self.minimum_attempted_inventory_depletion_time_fraction=min(
                self.minimum_attempted_inventory_depletion_time_fraction,depletion_time)
            self.maximum_attempted_inventory_depletion_time_fraction=max(
                self.maximum_attempted_inventory_depletion_time_fraction,maximum_depletion_time)
            self.maximum_attempted_nonconverged_chemistry_cell_count=max(
                self.maximum_attempted_nonconverged_chemistry_cell_count,nonconverged)
            self.attempted_chemistry_endpoint_evaluations+=endpoint_evaluations
            self.attempted_chemistry_corrector_iteration_sum+=corrector_iteration_sum
            self.attempted_chemistry_depletion_iteration_sum+=depletion_iteration_sum
            self.attempted_chemistry_evaluated_cell_count+=evaluated_cell_count
            self.attempted_reaction_coordinate_cell_count+=reaction_coordinate_cells
            self.attempted_reaction_coordinate_rate_evaluations+=(
                reaction_coordinate_rate_evaluations)
            self.attempted_reaction_coordinate_quadrature_panel_evaluations+=(
                reaction_coordinate_quadrature_evaluations)
            self.attempted_reaction_coordinate_root_iterations+=(
                reaction_coordinate_root_iteration_sum)
            self.maximum_attempted_reaction_coordinate_root_iterations=max(
                self.maximum_attempted_reaction_coordinate_root_iterations,
                reaction_coordinate_root_iterations)
            self.maximum_attempted_reaction_coordinate_quadrature_refinement_depth=max(
                self.maximum_attempted_reaction_coordinate_quadrature_refinement_depth,
                reaction_coordinate_quadrature_depth)
            self.maximum_attempted_reaction_coordinate_normalized_quadrature_residual=max(
                self.maximum_attempted_reaction_coordinate_normalized_quadrature_residual,
                reaction_coordinate_quadrature_residual)
            self.attempted_coupled_reaction_coordinate_attempts+=coupled_attempts
            self.attempted_coupled_reaction_coordinate_cells+=coupled_cells
            self.attempted_coupled_reaction_coordinate_fallbacks+=coupled_fallbacks
            self.attempted_coupled_reaction_coordinate_rate_evaluations+=(
                coupled_rate_evaluations)
            self.attempted_coupled_reaction_coordinate_steps+=coupled_steps
            self.attempted_coupled_reaction_coordinate_rejected_steps+=(
                coupled_rejected_steps)
            self.attempted_coupled_reaction_coordinate_driver_switches+=(
                coupled_driver_switches)
            self.maximum_attempted_coupled_reaction_coordinate_refinement_depth=max(
                self.maximum_attempted_coupled_reaction_coordinate_refinement_depth,
                coupled_depth)
            self.maximum_attempted_coupled_reaction_coordinate_normalized_residual=max(
                self.maximum_attempted_coupled_reaction_coordinate_normalized_residual,
                coupled_residual)
            self.maximum_attempted_rejected_coupled_reaction_coordinate_residual=max(
                self.maximum_attempted_rejected_coupled_reaction_coordinate_residual,
                coupled_rejected_residual)
            return
        if raw_rate_scope is not None:
            previous=self.chemistry_raw_rate_observation_scope
            self.chemistry_raw_rate_observation_scope=(
                raw_rate_scope if previous in {None,raw_rate_scope} else "mixed")
        self.max_chemistry_corrector_iterations_used=max(
            self.max_chemistry_corrector_iterations_used,int(corrector))
        self.max_chemistry_depletion_iterations_used=max(
            self.max_chemistry_depletion_iterations_used,int(depletion))
        self.max_chemistry_local_refinements_used=max(
            self.max_chemistry_local_refinements_used,int(refinements))
        self.maximum_chemistry_residual=max(self.maximum_chemistry_residual,residual)
        self.maximum_chemistry_temperature_residual_K=max(
            self.maximum_chemistry_temperature_residual_K,temperature_residual)
        self.maximum_normalized_embedded_alpha_residual=max(
            self.maximum_normalized_embedded_alpha_residual,embedded_alpha_residual)
        self.maximum_normalized_embedded_temperature_residual=max(
            self.maximum_normalized_embedded_temperature_residual,
            embedded_temperature_residual)
        self.maximum_rejected_normalized_embedded_residual=max(
            self.maximum_rejected_normalized_embedded_residual,
            rejected_embedded_residual)
        self.maximum_normalized_corrector_residual=max(
            self.maximum_normalized_corrector_residual,corrector_residual)
        self.maximum_temperature_feedback_alpha_correction=max(
            self.maximum_temperature_feedback_alpha_correction,
            feedback_alpha_correction)
        self.maximum_temperature_feedback_endpoint_temperature_correction_K=max(
            self.maximum_temperature_feedback_endpoint_temperature_correction_K,
            feedback_temperature_correction)
        self.maximum_raw_chemical_rate_per_s=max(
            self.maximum_raw_chemical_rate_per_s,raw_rate)
        self.maximum_raw_chemical_rate_initial_per_s=max(
            self.maximum_raw_chemical_rate_initial_per_s,raw_rate_initial)
        self.maximum_raw_chemical_rate_endpoint_per_s=max(
            self.maximum_raw_chemical_rate_endpoint_per_s,raw_rate_endpoint)
        self.max_rate_threshold_exceedance=max(
            self.max_rate_threshold_exceedance, threshold_fraction
        )
        self.max_inventory_depleted_fraction=max(
            self.max_inventory_depleted_fraction, depleted_fraction
        )
        self.maximum_chemistry_temperature_rise_K=max(
            self.maximum_chemistry_temperature_rise_K,temperature_rise)
        self.maximum_chemistry_caloric_inverse_residual_J_per_kg=max(
            self.maximum_chemistry_caloric_inverse_residual_J_per_kg,caloric_residual)
        self.maximum_chemistry_heat_closure_residual_J_per_m3=max(
            self.maximum_chemistry_heat_closure_residual_J_per_m3,heat_residual)
        self.maximum_chemistry_inventory_event_residual_mol_per_m3=max(
            self.maximum_chemistry_inventory_event_residual_mol_per_m3,inventory_residual)
        self.maximum_local_refined_cell_fraction=max(
            self.maximum_local_refined_cell_fraction,refined_fraction)
        self.minimum_inventory_depletion_time_fraction=min(
            self.minimum_inventory_depletion_time_fraction,depletion_time)
        self.maximum_inventory_depletion_time_fraction=max(
            self.maximum_inventory_depletion_time_fraction,maximum_depletion_time)
        self.maximum_nonconverged_chemistry_cell_count=max(
            self.maximum_nonconverged_chemistry_cell_count,nonconverged)
        self.accepted_chemistry_endpoint_evaluations+=endpoint_evaluations
        self.accepted_chemistry_corrector_iteration_sum+=corrector_iteration_sum
        self.accepted_chemistry_depletion_iteration_sum+=depletion_iteration_sum
        self.accepted_chemistry_evaluated_cell_count+=evaluated_cell_count
        self.accepted_reaction_coordinate_cell_count+=reaction_coordinate_cells
        self.accepted_reaction_coordinate_rate_evaluations+=(
            reaction_coordinate_rate_evaluations)
        self.accepted_reaction_coordinate_quadrature_panel_evaluations+=(
            reaction_coordinate_quadrature_evaluations)
        self.accepted_reaction_coordinate_root_iterations+=(
            reaction_coordinate_root_iteration_sum)
        self.maximum_reaction_coordinate_root_iterations=max(
            self.maximum_reaction_coordinate_root_iterations,
            reaction_coordinate_root_iterations)
        self.maximum_reaction_coordinate_quadrature_refinement_depth=max(
            self.maximum_reaction_coordinate_quadrature_refinement_depth,
            reaction_coordinate_quadrature_depth)
        self.maximum_reaction_coordinate_normalized_quadrature_residual=max(
            self.maximum_reaction_coordinate_normalized_quadrature_residual,
            reaction_coordinate_quadrature_residual)
        self.accepted_coupled_reaction_coordinate_attempts+=coupled_attempts
        self.accepted_coupled_reaction_coordinate_cells+=coupled_cells
        self.accepted_coupled_reaction_coordinate_fallbacks+=coupled_fallbacks
        self.accepted_coupled_reaction_coordinate_rate_evaluations+=(
            coupled_rate_evaluations)
        self.accepted_coupled_reaction_coordinate_steps+=coupled_steps
        self.accepted_coupled_reaction_coordinate_rejected_steps+=(
            coupled_rejected_steps)
        self.accepted_coupled_reaction_coordinate_driver_switches+=(
            coupled_driver_switches)
        self.maximum_coupled_reaction_coordinate_refinement_depth=max(
            self.maximum_coupled_reaction_coordinate_refinement_depth,
            coupled_depth)
        self.maximum_coupled_reaction_coordinate_normalized_residual=max(
            self.maximum_coupled_reaction_coordinate_normalized_residual,
            coupled_residual)
        self.maximum_rejected_coupled_reaction_coordinate_residual=max(
            self.maximum_rejected_coupled_reaction_coordinate_residual,
            coupled_rejected_residual)

    def _advance_local_split(self,t,U,dt,first_order=False):
        """C(dt/2) -> nonchemical SSPRK33(dt) -> C(dt/2)."""
        half=.5*dt
        chemistry0,heat0,cd0=self._chemistry_half_step(U,half)
        stability_limit=self._nonchemical_time_step_limit(chemistry0)
        self.post_chemistry_timestep_rechecks+=1
        if math.isnan(stability_limit) or stability_limit<=0:
            self.post_chemistry_timestep_recheck_rejections+=1
            error=PropagationCandidateNumericalError(
                "Invalid nonchemical stability limit after the first chemistry half-step")
            error.retry_reason="post_chemistry_timestep_recheck"
            error.chemistry_diagnostics=(cd0,)
            raise error
        safety_ratio=stability_limit/dt
        self.minimum_post_chemistry_timestep_safety_ratio=min(
            self.minimum_post_chemistry_timestep_safety_ratio,safety_ratio)
        if dt>stability_limit*(1+1e-12):
            self.post_chemistry_timestep_recheck_rejections+=1
            error=PropagationCandidateNumericalError(
                "Reactive timestep became unstable after the first chemistry half-step")
            error.retry_time_step_cap_s=float(stability_limit)
            error.retry_reason="post_chemistry_timestep_recheck"
            error.chemistry_diagnostics=(cd0,)
            raise error
        nonchemical_start=time.perf_counter()
        try:
            nonchemical,integrated,transport_diagnostics,sources=self._advance_nonchemical_ssprk(
                t,chemistry0,dt,first_order)
        finally:
            self.nonchemical_wall_clock_time_s+=time.perf_counter()-nonchemical_start
        result,heat1,cd1=self._chemistry_half_step(nonchemical,half)
        integrated=np.array(integrated,copy=True)
        total_heat=heat0+heat1
        integrated[NCONS+2]+=float(np.sum(total_heat)*self.vol)
        sources=dict(sources)
        sources["qChem_W_per_m3"]=total_heat/dt
        diagnostics={
            "transport_stages":transport_diagnostics,
            "chemistry_half_steps":(cd0,cd1),
            "post_first_half_stability_limit_s":float(stability_limit),
            "post_first_half_stability_safety_ratio":float(safety_ratio),
            "timestep_recheck_passed":True,
        }
        return result,integrated,diagnostics,sources

    def advance(self,t,U,dt,first_order=False):
        if self.chemistry_integration_mode=="local_adaptive_thermochemical":
            return self._advance_local_split(t,U,dt,first_order)
        return self._advance_coupled_ssprk(t,U,dt,first_order)

    @staticmethod
    def _copy_electrical_fields(fields):
        return {name:(np.array(value,copy=True) if isinstance(value,np.ndarray) else value)
                for name,value in fields.items()}

    def _electrical_checkpoint(self):
        """Snapshot physical electrical warm-start state for one PDE attempt."""
        if self.electrical is None:
            return None
        return (np.array(self.electrical.potential,copy=True),
                float(self.electrical.maximum_mismatch),
                self._copy_electrical_fields(self.electrical.last_fields))

    def _restore_electrical_checkpoint(self,checkpoint):
        if checkpoint is None:
            return
        potential,maximum_mismatch,last_fields=checkpoint
        self.electrical.potential=np.array(potential,copy=True)
        self.electrical.maximum_mismatch=maximum_mismatch
        self.electrical.last_fields=self._copy_electrical_fields(last_fields)

    def _record_accepted_diagnostics(self,diagnostics):
        if self.chemistry_integration_mode!="local_adaptive_thermochemical":
            self.exact_skipped_steps+=int(all(d["mechanics_skipped_exact"] for d in diagnostics))
            for d in diagnostics:
                self.face_fallbacks+=d["face_fallbacks"]; self.faces+=d["faces"]
                self.hllc_fallbacks+=d["hllc_fallbacks"]
                self.max_rate_cap=max(self.max_rate_cap,d["chemical_rate_cap_fraction"])
                self.max_inventory_limiter=max(self.max_inventory_limiter,d["inventory_limiter_fraction"])
            # Preserve the legacy report contract, including its stage-local
            # counter semantics. The split mode reports half-step iterations.
            self.max_chemistry_subcycles_used=max(
                self.max_chemistry_subcycles_used,int(d.get("chemistry_subcycles",0)))
            return
        transport=diagnostics["transport_stages"]
        chemistry=diagnostics["chemistry_half_steps"]
        self.exact_skipped_steps+=int(all(d["mechanics_skipped_exact"] for d in transport))
        for d in transport:
            self.face_fallbacks+=d["face_fallbacks"]; self.faces+=d["faces"]
            self.hllc_fallbacks+=d["hllc_fallbacks"]
        for d in chemistry:
            self._observe_local_chemistry(d,accepted=True)
        self.chemistry_half_steps_accepted+=len(chemistry)
        self.minimum_accepted_post_chemistry_timestep_safety_ratio=min(
            self.minimum_accepted_post_chemistry_timestep_safety_ratio,
            float(diagnostics["post_first_half_stability_safety_ratio"]))

    def _record(self,t,U,initial,budget):
        P=self.thermo.primitive(U); X=self.chem.progress(U)
        sums=np.sum(U,axis=(0,1))*self.vol
        if self._initial_sums is None:
            self._initial_sums=np.sum(initial,axis=(0,1))*self.vol
            self._initial_reservoir=float(np.sum(self.chem.chemical_reservoir(initial))*self.vol)
        initial_sums=self._initial_sums
        b=budget[:NCONS]; qj,qe,qchem,loss,conduction=budget[NCONS:]
        mass_res=sums[RHO]-initial_sums[RHO]-b[RHO]
        energy_res=sums[ENERGY]-initial_sums[ENERGY]-b[ENERGY]-qj-qe-qchem+loss-conduction
        energy_scale=max(abs(initial_sums[ENERGY]),abs(sums[ENERGY]),abs(qj)+abs(qe)+abs(qchem)+abs(loss),1.)
        mass_rel=abs(mass_res)/max(abs(initial_sums[RHO]),1e-300)
        energy_rel=abs(energy_res)/energy_scale
        if mass_rel>self.mass_tolerance or energy_rel>self.energy_tolerance:
            raise PropagationCandidateNumericalError("Reactive mass/energy budget tolerance exceeded")
        reservoir=float(np.sum(self.chem.chemical_reservoir(U))*self.vol)
        reservoir0=self._initial_reservoir
        rb=self.chem.Q.sum()*b[RHO]-self.chem.Q[0]*b[A1]-self.chem.Q[1]*b[A2]
        total_res=(sums[ENERGY]+reservoir)-(initial_sums[ENERGY]+reservoir0)-b[ENERGY]-rb-qj-qe+loss-conduction
        residuals=self.chem.conserved_inventory_residuals(U)
        max_local=max(float(np.max(np.abs(residuals[k]))) for k in ("pva_plus_extent","product_water_minus_extent"))
        salt_boundary=(.5*(b[CATION]+b[ANION])+b[EC_LP]
                       +1.45*self.chem.xi_per_kg*(self.chem.weights[0]*b[A1]+self.chem.weights[1]*b[A2])
                       -self.chem.initial_lp_per_kg*b[RHO])
        salt_res=float(np.sum(residuals["salt_plus_ec_plus_extent"])*self.vol-salt_boundary)
        stock_scale=max(float(np.max(U[...,RHO]))*self.chem.initial_lp_per_kg,1.)
        if max_local>1e-8*stock_scale or abs(salt_res)>1e-8*stock_scale*U.shape[0]*U.shape[1]*self.vol:
            raise PropagationCandidateNumericalError("Two-channel transported inventory invariant failed")
        return {"time_after_onset_s":float(t),"unreacted_area_fraction":float(np.mean(X<self.front_threshold)),
                "mean_global_progress":float(np.mean(X)),"maximum_temperature_K":float(np.max(P[...,3])),
                "maximum_speed_m_per_s":float(np.max(np.hypot(P[...,MX],P[...,MY]))),
                "minimum_pressure_Pa":float(np.min(self.thermo.pressure(P[...,RHO]))),
                "maximum_pressure_Pa":float(np.max(self.thermo.pressure(P[...,RHO]))),
                "mass_kg":float(sums[RHO]),"total_energy_J":float(sums[ENERGY]),
                "chemical_reservoir_J":reservoir,"mass_budget_residual_kg":float(mass_res),
                "mass_budget_relative_residual":float(mass_rel),"energy_budget_residual_J":float(energy_res),
                "energy_budget_relative_residual":float(energy_rel),"energy_plus_chemical_budget_residual_J":float(total_res),
                "joule_energy_J":float(qj),"electrochemical_energy_J":float(qe),"chemical_energy_J":float(qchem),
                "heat_loss_J":float(loss),"net_conduction_energy_J":float(conduction),
                "boundary_mass_in_kg":float(b[RHO]),"boundary_energy_in_J":float(b[ENERGY]),
                "maximum_local_inventory_residual_mol_per_m3":max_local,"integrated_lp_inventory_residual_mol":salt_res}

    def run(self,output_dir):
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter(); t=0.; U=self.U.copy(); initial=U.copy()
        last_progress_wall=start
        self._initial_sums=None;self._initial_reservoir=None
        self.thermo.validate(U)
        _write_json(out/"handoff_audit.json",self.a.audit)
        X=self.chem.progress(U); mask=self.a.propellant_mask
        arrival=np.full(X.shape,np.nan); arrival[X>=self.front_threshold]=0.
        unreacted0=float(np.mean(X<self.front_threshold))
        established=0. if 1-unreacted0>=self.established_fraction else math.nan
        budget=np.zeros(NCONS+5); history=[]; snapshots=[]; snapshot_t=[]
        velocities=[]; dts=[]
        row0=self._record(0,U,initial,budget); row0["effective_regression_velocity_m_per_s"]=0.; history.append(row0)
        def save_snapshot(time_value,state):
            p=self.thermo.primitive(state); x=self.chem.progress(state)
            # phi=threshold-X is the actual implicit reaction interface. The
            # separate distance is a grid-resolved reinitialisation, not a new
            # material velocity or an EOS selector and never feeds chemistry.
            phi=self.front_threshold-x
            distance=_reaction_level_set(x,mask,self.front_threshold,self.a.dx)
            snapshots.append(np.stack((p[...,3],p[...,A1],p[...,A2],x,p[...,RHO],p[...,MX],p[...,MY],
                                        self.thermo.pressure(p[...,RHO]),phi,distance),axis=-1))
            snapshot_t.append(float(time_value))
        save_snapshot(0,U); next_snapshot=min(self.duration,self.snapshot_interval)
        while t<self.duration:
            if self.step_number>=self.max_steps:
                self.U=U
                np.savez_compressed(out/"INCOMPLETE_STATE.npz",U=U,time_after_onset_s=t)
                raise PropagationCandidateNumericalError("maximum_time_steps reached before requested duration; saved INCOMPLETE_STATE, not a completed run")
            remaining=min(self.duration-t,next_snapshot-t)
            if remaining<=64*np.finfo(float).eps*max(self.duration,1e-12):
                next_snapshot=min(self.duration,next_snapshot+self.snapshot_interval)
                remaining=min(self.duration-t,next_snapshot-t)
            dt=self.time_step(U,remaining)
            first_order=False
            for retry in range(self.max_retries+1):
                electrical_checkpoint=self._electrical_checkpoint()
                try:
                    candidate,integrated,diagnostics,sources=self.advance(t,U,dt,first_order)
                    record=self._record(t+dt,candidate,initial,budget+integrated)
                    break
                except ConfiguredModelTemperatureRangeExceeded as error:
                    # A model-domain boundary is terminal, never a timestep
                    # retry or a geometry/numerical-instability diagnosis.
                    self._restore_electrical_checkpoint(electrical_checkpoint)
                    self.rejected_steps+=1
                    self.U=U
                    np.savez_compressed(out/"INCOMPLETE_STATE.npz",
                                        U=U,time_after_onset_s=t)
                    failure=candidate_failure_payload(error)
                    failure.update(time_after_onset_s=float(t),
                                   attempted_time_step_s=float(dt),retry_count=retry)
                    _write_json(out/"PROPAGATION_FAILED.json",failure)
                    raise
                except PropagationCandidateNumericalError as error:
                    # State, ledgers and accepted electrical warm starts commit
                    # only after both the split trial and budget audit succeed.
                    # ``calls`` intentionally remains actual work telemetry.
                    self._restore_electrical_checkpoint(electrical_checkpoint)
                    self.rejected_steps+=1
                    retry_cap=float(getattr(error,"retry_time_step_cap_s",math.inf))
                    next_dt=min(dt*.5,retry_cap)
                    retry_reason=str(getattr(
                        error,"retry_reason",type(error).__name__
                    ))
                    # A failed local nonlinear/quadrature solve is not a PDE
                    # stability signal.  Halving the global PDE timestep would
                    # mix accuracy/work budgets with physical time stepping and
                    # could silently make a candidate arbitrarily expensive.
                    # Only the explicit nonchemical stability recheck supplies
                    # a mathematically valid local-mode retry timestep.
                    terminal_local_chemistry=(
                        self.chemistry_integration_mode
                        =="local_adaptive_thermochemical"
                        and retry_reason=="local_chemistry"
                    )
                    terminal=(terminal_local_chemistry
                              or retry==self.max_retries
                              or not math.isfinite(next_dt)
                              or next_dt<self.min_dt)
                    now=time.perf_counter()
                    if now-last_progress_wall>=self.progress_log_interval_wall_s:
                        progress={"event":"ecsp_reactive_progress","compute_backend":"numpy_cpu",
                            "chemistry_integration_mode":self.chemistry_integration_mode,
                            "trial_status":"rejected_terminal" if terminal else "rejected_retry_pending",
                            "accepted_time_steps":self.step_number,
                            "rejected_time_steps":self.rejected_steps,
                            "time_after_onset_s":float(t),
                            "duration_after_onset_s":self.duration,
                            "last_attempted_time_step_s":float(dt),
                            "next_time_step_s":float(next_dt) if not terminal else None,
                            "retry_reason":retry_reason,
                            "maximum_temperature_K":history[-1]["maximum_temperature_K"],
                            "chemistry_half_step_attempts":self.chemistry_half_step_attempts,
                            "chemistry_wall_clock_time_s":self.chemistry_wall_clock_time_s,
                            "nonchemical_ssprk_wall_clock_time_s":self.nonchemical_wall_clock_time_s,
                            "wall_clock_time_s":now-start}
                        print(json.dumps(progress,sort_keys=True,allow_nan=False),flush=True)
                        last_progress_wall=now
                    if terminal:
                        raise
                    dt=next_dt
                    if (self.chemistry_integration_mode!="local_adaptive_thermochemical"
                            or getattr(error,"retry_reason",None) not in {
                                "local_chemistry","post_chemistry_timestep_recheck"}):
                        first_order=retry>=1
                except Exception:
                    self._restore_electrical_checkpoint(electrical_checkpoint)
                    raise
            previous_X=X; X=self.chem.progress(candidate)
            crossed=np.isnan(arrival)&(X>=self.front_threshold)
            fraction=np.zeros_like(X); moving=crossed&(X>previous_X)
            fraction[moving]=np.clip((self.front_threshold-previous_X[moving])/(X[moving]-previous_X[moving]),0,1)
            arrival[crossed]=t+fraction[crossed]*dt
            prev_cells=int(np.count_nonzero(previous_X<self.front_threshold)); new_cells=int(np.count_nonzero(X<self.front_threshold))
            prev_edges=_front_edge_count(previous_X<self.front_threshold); new_edges=_front_edge_count(X<self.front_threshold)
            mean_edges=.5*(prev_edges+new_edges)
            velocity=(prev_cells-new_cells)*self.a.dx/(mean_edges*dt) if mean_edges>0 else 0.
            t+=dt
            if abs(t-self.duration)<64*np.finfo(float).eps*max(self.duration,1e-12): t=self.duration
            if math.isnan(established) and 1-record["unreacted_area_fraction"]>=self.established_fraction:
                established=t
            record["time_after_onset_s"]=t
            record["effective_regression_velocity_m_per_s"]=velocity
            history.append(record); velocities.append(velocity); dts.append(dt)
            U=candidate; budget+=integrated; self.last_sources=sources
            self.step_number+=1; self.first_order_steps+=int(first_order)
            self._record_accepted_diagnostics(diagnostics)
            now=time.perf_counter()
            if (self.step_number%self.progress_log_interval_steps==0
                    or now-last_progress_wall>=self.progress_log_interval_wall_s
                    or t==self.duration):
                progress={"event":"ecsp_reactive_progress","compute_backend":"numpy_cpu",
                    "chemistry_integration_mode":self.chemistry_integration_mode,
                    "trial_status":"accepted",
                    "accepted_time_steps":self.step_number,"rejected_time_steps":self.rejected_steps,
                    "time_after_onset_s":float(t),"duration_after_onset_s":self.duration,
                    "last_time_step_s":float(dt),
                    "minimum_accepted_time_step_s":float(min(dts)),
                    "maximum_accepted_time_step_s":float(max(dts)),
                    "maximum_temperature_K":record["maximum_temperature_K"],
                    "chemistry_half_step_attempts":self.chemistry_half_step_attempts,
                    "chemistry_endpoint_evaluation_count":
                        self.accepted_chemistry_endpoint_evaluations,
                    "mean_chemistry_corrector_iterations_per_evaluated_cell":
                        self.accepted_chemistry_corrector_iteration_sum
                        /self.accepted_chemistry_evaluated_cell_count
                        if self.accepted_chemistry_evaluated_cell_count else None,
                    "chemistry_wall_clock_time_s":self.chemistry_wall_clock_time_s,
                    "nonchemical_ssprk_wall_clock_time_s":self.nonchemical_wall_clock_time_s,
                    "wall_clock_time_s":now-start}
                print(json.dumps(progress,sort_keys=True,allow_nan=False),flush=True)
                last_progress_wall=now
            if t>=next_snapshot-64*np.finfo(float).eps*max(self.duration,1e-12) or t==self.duration:
                save_snapshot(t,U); next_snapshot=min(self.duration,next_snapshot+self.snapshot_interval)
        return self.finish(out,start,U,initial,budget,history,snapshots,snapshot_t,arrival,velocities,dts,established)

    def finish(self,out,start,U,initial,budget,history,snapshots,snapshot_t,arrival,velocities,dts,established):
        """Common NumPy/torch report and serialization; no time integration here."""
        self.U=U
        mask=self.a.propellant_mask
        X=self.chem.progress(U)
        t=history[-1]["time_after_onset_s"]
        row0=history[0]
        cv,coverage=_arrival_time_statistics(arrival,mask)
        last=history[-1]; P=self.thermo.primitive(U)
        reactive_mass=np.sum(self.chem.m_lp*.5*(U[...,CATION]+U[...,ANION])+self.chem.m_pva*U[...,PVA])*self.vol
        initial_reactive=self.a.audit["initial_continuum_mass_kg"]*(self.chem.m_lp*self.chem.initial_lp_per_kg+self.chem.m_pva*self.chem.initial_pva_per_kg)
        wall_clock=time.perf_counter()-start
        metrics={
            "status":"complete","onsetSucceeded":True,"postOnsetBackend":"reactive_euler",
            "postOnsetModelValid":True,"postOnsetValidityConstraintViolation":0.0,
            "postOnsetValidityReason":"",
            "modelScope":"single_condensed_continuum_two_channel_Euler_Fourier_Tait_not_gas_CFD",
            "modelValidationStatus":"experimental_uncalibrated_not_experimentally_validated",
            "finalUnreactedAreaFraction":last["unreacted_area_fraction"],
            "finalRemainingReactiveMassFraction":float(reactive_mass/initial_reactive),
            "meanEffectiveRegressionVelocity_m_per_s":float(np.average(velocities,weights=dts)),
            "maximumEffectiveRegressionVelocity_m_per_s":float(np.max(velocities)),
            "establishedTimeAfterOnset_s":float(established) if math.isfinite(established) else self.duration,
            "establishedTimeCensored":not math.isfinite(established),
            "reactionFrontNonuniformity":cv+(1-coverage),"arrivalTimeCoefficientOfVariation":cv,
            "frontArrivalCoverageFraction":coverage,"arrivalCoveragePenalty":1-coverage,
            "finalMeanGlobalProgress":last["mean_global_progress"],
            "finalMaximumTemperature_K":last["maximum_temperature_K"],
            "maximumTemperatureDuringPropagation_K":max(r["maximum_temperature_K"] for r in history),
            "maximumSpeedDuringPropagation_m_per_s":max(r["maximum_speed_m_per_s"] for r in history),
            "continuedElectricalHeating":self.heating,"electricalHeatingPolicy":"recomputed_BC_NP_BV" if self.heating else "off",
            "postOnsetElectricalSolveCalls":self.electrical.calls if self.electrical else 0,
            "maximumPostOnsetCurrentMismatch":self.electrical.maximum_mismatch if self.electrical else 0.,
            "acceptedTimeSteps":self.step_number,"rejectedTimeSteps":self.rejected_steps,
            "firstOrderRetrySteps":self.first_order_steps,"positivityFaceFallbackCount":self.face_fallbacks,
            "positivityFaceFallbackFraction":self.face_fallbacks/max(self.faces,1),"HLLCtoHLLFallbackCount":self.hllc_fallbacks,
            "exactStationaryMechanicsSkippedSteps":self.exact_skipped_steps,
            "stationaryMechanicsFastPath":"exact_invariant_subspace_of_barotropic_Tait_not_an_added_damping_model",
            "barotropicThermalPressureCoupling":False,
            "maximumChemicalRateCapFraction":self.max_rate_cap,
            "maximumChemicalInventoryLimiterFraction":self.max_inventory_limiter,
            "maximumChemistrySubcyclesPerPDEStage":self.max_chemistry_subcycles_used,
            "chemicalRateThresholdDiagnosticOnly":
                self.chemistry_integration_mode=="local_adaptive_thermochemical",
            "inheritedBCMaximumRateThreshold_per_s":float(self.chem.maximum_rate),
            "inheritedBCMaximumRateThresholdUsage":
                "diagnostic_only_not_used_for_kinetic_integration"
                if self.chemistry_integration_mode=="local_adaptive_thermochemical"
                else "legacy_rate_clipping",
            "inheritedChemicalRateDiagnosticThreshold_per_s":
                float(self.chem.maximum_rate),
            "maximumFractionAboveInheritedChemicalRateDiagnosticThreshold":
                (self.max_rate_threshold_exceedance
                 if self.chemistry_integration_mode
                    =="local_adaptive_thermochemical"
                 else self.max_rate_cap),
            "physicalChemicalRatesClipped":self.chemistry_integration_mode=="legacy_cap",
            "chemistryIntegrationMode":self.chemistry_integration_mode,
            "chemistryPhysicalKinetics":
                "raw_uncapped_two_channel_BC_tables"
                if self.chemistry_integration_mode=="local_adaptive_thermochemical"
                else "legacy_threshold_clipped_two_channel_BC_tables",
            "chemistryTrajectoryProvenance":
                "same_raw_two_channel_kinetics; inherited_maximum_rate_per_s_"
                "is_diagnostic_only; not_the_same_physical_trajectory_as_"
                "legacy_capped_post_onset"
                if self.chemistry_integration_mode=="local_adaptive_thermochemical"
                else "legacy_capped_post_onset_trajectory",
            "operatorSplitting":"chemistry_half_nonchemical_full_chemistry_half"
                if self.chemistry_integration_mode=="local_adaptive_thermochemical"
                else "none_legacy_coupled_sources",
            "nonchemicalTimeIntegrator":"SSPRK33",
            "chemistryTimeIntegrator":
                "bulk_midpoint_with_dominant_driver_reaction_coordinate_and_"
                "error_controlled_midpoint_fallback"
                if self.chemistry_integration_mode=="local_adaptive_thermochemical"
                else self.chemistry_integration_mode,
            "coupledReactionCoordinateWorkCounterScope":
                "rate_step_depth_residual_counters_are_successful_endpoint_"
                "only; attempt_and_fallback_counts_cover_all_attempts; "
                "fallback_work_is_included_in_chemistry_wall_clock"
                if self.chemistry_integration_mode=="local_adaptive_thermochemical"
                else "not_applicable",
            "oneActiveReactionCoordinateWorkCounterScope":
                "successful_coordinate_endpoints_only; fallback_work_is_"
                "included_in_chemistry_wall_clock"
                if self.chemistry_integration_mode=="local_adaptive_thermochemical"
                else "not_applicable",
            "transactionalStepRetries":True,
            "chemistryHalfStepAttempts":self.chemistry_half_step_attempts,
            "acceptedChemistryHalfSteps":self.chemistry_half_steps_accepted,
            "failedChemistryHalfSteps":self.chemistry_failed_half_steps,
            "postChemistryTimestepRechecks":self.post_chemistry_timestep_rechecks,
            "postChemistryTimestepRecheckRejections":self.post_chemistry_timestep_recheck_rejections,
            "minimumPostChemistryTimestepSafetyRatio":
                float(self.minimum_post_chemistry_timestep_safety_ratio)
                if math.isfinite(self.minimum_post_chemistry_timestep_safety_ratio) else None,
            "minimumAcceptedPostChemistryTimestepSafetyRatio":
                float(self.minimum_accepted_post_chemistry_timestep_safety_ratio)
                if math.isfinite(self.minimum_accepted_post_chemistry_timestep_safety_ratio) else None,
            "maximumChemistryCorrectorIterationsPerHalfStep":self.max_chemistry_corrector_iterations_used,
            "maximumChemistryDepletionIterationsPerHalfStep":self.max_chemistry_depletion_iterations_used,
            "maximumChemistryLocalRefinementsPerHalfStep":self.max_chemistry_local_refinements_used,
            "maximumChemistryResidual":self.maximum_chemistry_residual,
            "maximumChemistryTemperatureResidual_K":self.maximum_chemistry_temperature_residual_K,
            "maximumNormalizedChemistryEmbeddedResidual":self.maximum_chemistry_residual,
            "maximumNormalizedChemistryEmbeddedAlphaResidual":
                self.maximum_normalized_embedded_alpha_residual,
            "maximumNormalizedChemistryEmbeddedTemperatureResidual":
                self.maximum_normalized_embedded_temperature_residual,
            "maximumRejectedNormalizedChemistryEmbeddedResidual":
                self.maximum_rejected_normalized_embedded_residual,
            "maximumNormalizedChemistryCorrectorResidual":
                self.maximum_normalized_corrector_residual,
            "maximumChemistryTemperatureFeedbackAlphaCorrection":
                self.maximum_temperature_feedback_alpha_correction,
            "maximumChemistryTemperatureFeedbackEndpointTemperatureCorrection_K":
                self.maximum_temperature_feedback_endpoint_temperature_correction_K,
            "maximumRawChemicalRate_per_s":self.maximum_raw_chemical_rate_per_s,
            "maximumRawChemicalRateAtHalfStepInitialState_per_s":
                self.maximum_raw_chemical_rate_initial_per_s,
            "maximumRawChemicalRateAtHalfStepEndpoint_per_s":
                self.maximum_raw_chemical_rate_endpoint_per_s,
            "chemicalRawRateObservationScope":self.chemistry_raw_rate_observation_scope,
            "maximumChemicalRateThresholdExceedanceFraction":(
                self.max_rate_threshold_exceedance
                if self.chemistry_integration_mode
                   =="local_adaptive_thermochemical"
                else self.max_rate_cap
            ),
            "maximumChemicalInventoryDepletedFraction":(
                self.max_inventory_depleted_fraction
                if self.chemistry_integration_mode
                   =="local_adaptive_thermochemical"
                else self.max_inventory_limiter
            ),
            "maximumChemistryTemperatureRise_K":self.maximum_chemistry_temperature_rise_K,
            "maximumChemistryCaloricInverseResidual_J_per_kg":
                self.maximum_chemistry_caloric_inverse_residual_J_per_kg,
            "maximumChemistryHeatClosureResidual_J_per_m3":
                self.maximum_chemistry_heat_closure_residual_J_per_m3,
            "maximumChemistryInventoryEventResidual_mol_per_m3":
                self.maximum_chemistry_inventory_event_residual_mol_per_m3,
            "maximumLocalRefinedCellFraction":self.maximum_local_refined_cell_fraction,
            "minimumInventoryDepletionTimeFraction":
                float(self.minimum_inventory_depletion_time_fraction)
                if math.isfinite(self.minimum_inventory_depletion_time_fraction) else None,
            "maximumInventoryDepletionTimeFraction":
                self.maximum_inventory_depletion_time_fraction
                if math.isfinite(self.minimum_inventory_depletion_time_fraction) else None,
            "maximumNonconvergedChemistryCellCount":self.maximum_nonconverged_chemistry_cell_count,
            "maximumAttemptedChemistryCorrectorIterationsPerHalfStep":self.maximum_attempted_chemistry_corrector_iterations,
            "maximumAttemptedChemistryDepletionIterationsPerHalfStep":self.maximum_attempted_chemistry_depletion_iterations,
            "maximumAttemptedChemistryLocalRefinementsPerHalfStep":self.maximum_attempted_chemistry_local_refinements,
            "maximumAttemptedChemistryResidual":self.maximum_attempted_chemistry_residual,
            "maximumAttemptedChemistryTemperatureResidual_K":self.maximum_attempted_chemistry_temperature_residual_K,
            "maximumAttemptedNormalizedChemistryEmbeddedResidual":
                self.maximum_attempted_chemistry_residual,
            "maximumAttemptedNormalizedChemistryEmbeddedAlphaResidual":
                self.maximum_attempted_normalized_embedded_alpha_residual,
            "maximumAttemptedNormalizedChemistryEmbeddedTemperatureResidual":
                self.maximum_attempted_normalized_embedded_temperature_residual,
            "maximumAttemptedRejectedNormalizedChemistryEmbeddedResidual":
                self.maximum_attempted_rejected_normalized_embedded_residual,
            "maximumAttemptedNormalizedChemistryCorrectorResidual":
                self.maximum_attempted_normalized_corrector_residual,
            "maximumAttemptedChemistryTemperatureFeedbackAlphaCorrection":
                self.maximum_attempted_temperature_feedback_alpha_correction,
            "maximumAttemptedChemistryTemperatureFeedbackEndpointTemperatureCorrection_K":
                self.maximum_attempted_temperature_feedback_endpoint_temperature_correction_K,
            "maximumAttemptedRawChemicalRate_per_s":self.maximum_attempted_raw_chemical_rate_per_s,
            "maximumAttemptedRawChemicalRateAtHalfStepInitialState_per_s":
                self.maximum_attempted_raw_chemical_rate_initial_per_s,
            "maximumAttemptedRawChemicalRateAtHalfStepEndpoint_per_s":
                self.maximum_attempted_raw_chemical_rate_endpoint_per_s,
            "attemptedChemicalRawRateObservationScope":
                self.attempted_chemistry_raw_rate_observation_scope,
            "maximumAttemptedChemicalRateThresholdExceedanceFraction":
                self.maximum_attempted_rate_threshold_exceedance_fraction,
            "maximumAttemptedChemicalInventoryDepletedFraction":
                self.maximum_attempted_inventory_depleted_fraction,
            "maximumAttemptedChemistryTemperatureRise_K":
                self.maximum_attempted_chemistry_temperature_rise_K,
            "maximumAttemptedChemistryCaloricInverseResidual_J_per_kg":
                self.maximum_attempted_chemistry_caloric_inverse_residual_J_per_kg,
            "maximumAttemptedChemistryHeatClosureResidual_J_per_m3":
                self.maximum_attempted_chemistry_heat_closure_residual_J_per_m3,
            "maximumAttemptedChemistryInventoryEventResidual_mol_per_m3":
                self.maximum_attempted_chemistry_inventory_event_residual_mol_per_m3,
            "maximumAttemptedLocalRefinedCellFraction":
                self.maximum_attempted_local_refined_cell_fraction,
            "minimumAttemptedInventoryDepletionTimeFraction":
                float(self.minimum_attempted_inventory_depletion_time_fraction)
                if math.isfinite(self.minimum_attempted_inventory_depletion_time_fraction) else None,
            "maximumAttemptedInventoryDepletionTimeFraction":
                self.maximum_attempted_inventory_depletion_time_fraction
                if math.isfinite(self.minimum_attempted_inventory_depletion_time_fraction) else None,
            "maximumAttemptedNonconvergedChemistryCellCount":
                self.maximum_attempted_nonconverged_chemistry_cell_count,
            "chemistryRelativeTolerance":self.chemistry_relative_tolerance,
            "chemistryAbsoluteTolerance":self.chemistry_absolute_tolerance,
            "chemistryTemperatureTolerance_K":self.chemistry_temperature_tolerance_K,
            "configuredMaximumChemistryCorrectorIterations":self.max_chemistry_corrector_iterations,
            "configuredMaximumChemistryDepletionIterations":self.max_chemistry_depletion_iterations,
            "configuredMaximumChemistryLocalRefinements":self.max_chemistry_local_refinements,
            "configuredMaximumChemistryReactionCoordinateSteps":(
                self.max_chemistry_reaction_coordinate_steps
            ),
            "derivedMaximumChemistryPanelAttemptsPerCell":
                4*(2**self.max_chemistry_local_refinements),
            "progressLogIntervalSteps":self.progress_log_interval_steps,
            "progressLogIntervalWall_s":self.progress_log_interval_wall_s,
            "chemistryWallClockTime_s":self.chemistry_wall_clock_time_s,
            "nonchemicalSSPRKWallClockTime_s":self.nonchemical_wall_clock_time_s,
            "chemistryWallClockFraction":self.chemistry_wall_clock_time_s/max(wall_clock,1e-300),
            "nonchemicalSSPRKWallClockFraction":
                self.nonchemical_wall_clock_time_s/max(wall_clock,1e-300),
            "chemistryEndpointEvaluationCount":self.accepted_chemistry_endpoint_evaluations,
            "chemistryEvaluatedCellCount":self.accepted_chemistry_evaluated_cell_count,
            "meanChemistryCorrectorIterationsPerEvaluatedCell":
                self.accepted_chemistry_corrector_iteration_sum
                /self.accepted_chemistry_evaluated_cell_count
                if self.accepted_chemistry_evaluated_cell_count else None,
            "meanChemistryDepletionIterationsPerEvaluatedCell":
                self.accepted_chemistry_depletion_iteration_sum
                /self.accepted_chemistry_evaluated_cell_count
                if self.accepted_chemistry_evaluated_cell_count else None,
            "meanChemistryEndpointEvaluationsPerEvaluatedCell":
                self.accepted_chemistry_endpoint_evaluations
                /self.accepted_chemistry_evaluated_cell_count
                if self.accepted_chemistry_evaluated_cell_count else None,
            "attemptedChemistryEndpointEvaluationCount":
                self.attempted_chemistry_endpoint_evaluations,
            "attemptedChemistryEvaluatedCellCount":
                self.attempted_chemistry_evaluated_cell_count,
            "oneActiveReactionCoordinateCellCount":
                self.accepted_reaction_coordinate_cell_count,
            "oneActiveReactionCoordinateRateEvaluationCount":
                self.accepted_reaction_coordinate_rate_evaluations,
            "oneActiveReactionCoordinateQuadraturePanelEvaluationCount":
                self.accepted_reaction_coordinate_quadrature_panel_evaluations,
            "oneActiveReactionCoordinateRootIterationCount":
                self.accepted_reaction_coordinate_root_iterations,
            "maximumOneActiveReactionCoordinateRootIterations":
                self.maximum_reaction_coordinate_root_iterations,
            "maximumOneActiveReactionCoordinateQuadratureRefinementDepth":
                self.maximum_reaction_coordinate_quadrature_refinement_depth,
            "maximumOneActiveReactionCoordinateNormalizedQuadratureResidual":
                self.maximum_reaction_coordinate_normalized_quadrature_residual,
            "coupledReactionCoordinateAttemptCount":
                self.accepted_coupled_reaction_coordinate_attempts,
            "coupledReactionCoordinateCellCount":
                self.accepted_coupled_reaction_coordinate_cells,
            "coupledReactionCoordinateFallbackCount":
                self.accepted_coupled_reaction_coordinate_fallbacks,
            "coupledReactionCoordinateRateEvaluationCount":
                self.accepted_coupled_reaction_coordinate_rate_evaluations,
            "coupledReactionCoordinateAcceptedStepCount":
                self.accepted_coupled_reaction_coordinate_steps,
            "coupledReactionCoordinateRejectedStepCount":
                self.accepted_coupled_reaction_coordinate_rejected_steps,
            "coupledReactionCoordinateDriverSwitchCount":
                self.accepted_coupled_reaction_coordinate_driver_switches,
            "maximumCoupledReactionCoordinateRefinementDepth":
                self.maximum_coupled_reaction_coordinate_refinement_depth,
            "maximumCoupledReactionCoordinateNormalizedResidual":
                self.maximum_coupled_reaction_coordinate_normalized_residual,
            "maximumRejectedCoupledReactionCoordinateResidual":
                self.maximum_rejected_coupled_reaction_coordinate_residual,
            "attemptedOneActiveReactionCoordinateCellCount":
                self.attempted_reaction_coordinate_cell_count,
            "attemptedOneActiveReactionCoordinateRateEvaluationCount":
                self.attempted_reaction_coordinate_rate_evaluations,
            "attemptedOneActiveReactionCoordinateQuadraturePanelEvaluationCount":
                self.attempted_reaction_coordinate_quadrature_panel_evaluations,
            "attemptedOneActiveReactionCoordinateRootIterationCount":
                self.attempted_reaction_coordinate_root_iterations,
            "maximumAttemptedOneActiveReactionCoordinateRootIterations":
                self.maximum_attempted_reaction_coordinate_root_iterations,
            "maximumAttemptedOneActiveReactionCoordinateQuadratureRefinementDepth":
                self.maximum_attempted_reaction_coordinate_quadrature_refinement_depth,
            "maximumAttemptedOneActiveReactionCoordinateNormalizedQuadratureResidual":
                self.maximum_attempted_reaction_coordinate_normalized_quadrature_residual,
            "attemptedCoupledReactionCoordinateAttemptCount":
                self.attempted_coupled_reaction_coordinate_attempts,
            "attemptedCoupledReactionCoordinateCellCount":
                self.attempted_coupled_reaction_coordinate_cells,
            "attemptedCoupledReactionCoordinateFallbackCount":
                self.attempted_coupled_reaction_coordinate_fallbacks,
            "attemptedCoupledReactionCoordinateRateEvaluationCount":
                self.attempted_coupled_reaction_coordinate_rate_evaluations,
            "attemptedCoupledReactionCoordinateAcceptedStepCount":
                self.attempted_coupled_reaction_coordinate_steps,
            "attemptedCoupledReactionCoordinateRejectedStepCount":
                self.attempted_coupled_reaction_coordinate_rejected_steps,
            "attemptedCoupledReactionCoordinateDriverSwitchCount":
                self.attempted_coupled_reaction_coordinate_driver_switches,
            "maximumAttemptedCoupledReactionCoordinateRefinementDepth":
                self.maximum_attempted_coupled_reaction_coordinate_refinement_depth,
            "maximumAttemptedCoupledReactionCoordinateNormalizedResidual":
                self.maximum_attempted_coupled_reaction_coordinate_normalized_residual,
            "maximumAttemptedRejectedCoupledReactionCoordinateResidual":
                self.maximum_attempted_rejected_coupled_reaction_coordinate_residual,
            "meanAttemptedChemistryCorrectorIterationsPerEvaluatedCell":
                self.attempted_chemistry_corrector_iteration_sum
                /self.attempted_chemistry_evaluated_cell_count
                if self.attempted_chemistry_evaluated_cell_count else None,
            "meanAttemptedChemistryDepletionIterationsPerEvaluatedCell":
                self.attempted_chemistry_depletion_iteration_sum
                /self.attempted_chemistry_evaluated_cell_count
                if self.attempted_chemistry_evaluated_cell_count else None,
            "meanAttemptedChemistryEndpointEvaluationsPerEvaluatedCell":
                self.attempted_chemistry_endpoint_evaluations
                /self.attempted_chemistry_evaluated_cell_count
                if self.attempted_chemistry_evaluated_cell_count else None,
            "minimumAcceptedTimeStep_s":float(min(dts)),
            "maximumAcceptedTimeStep_s":float(max(dts)),
            "meanAcceptedTimeStep_s":float(np.mean(dts)),
            "maximumMassBudgetRelativeResidual":max(r["mass_budget_relative_residual"] for r in history),
            "maximumEnergyBudgetRelativeResidual":max(r["energy_budget_relative_residual"] for r in history),
            "maximumAbsoluteMassBudgetResidual_kg":
                max(abs(r["mass_budget_residual_kg"]) for r in history),
            "maximumAbsoluteEnergyBudgetResidual_J":
                max(abs(r["energy_budget_residual_J"]) for r in history),
            "maximumAbsoluteEnergyPlusChemicalBudgetResidual_J":
                max(abs(r["energy_plus_chemical_budget_residual_J"]) for r in history),
            "maximumLocalInventoryResidual_mol_per_m3":
                max(r["maximum_local_inventory_residual_mol_per_m3"] for r in history),
            "maximumAbsoluteIntegratedLPInventoryResidual_mol":
                max(abs(r["integrated_lp_inventory_residual_mol"]) for r in history),
            "finalMassBudgetResidual_kg":last["mass_budget_residual_kg"],
            "finalEnergyBudgetResidual_J":last["energy_budget_residual_J"],
            "finalEnergyPlusChemicalBudgetResidual_J":last["energy_plus_chemical_budget_residual_J"],
            "integratedJouleHeat_J":last["joule_energy_J"],"integratedElectrochemicalHeat_J":last["electrochemical_energy_J"],
            "integratedChemicalHeat_J":last["chemical_energy_J"],"integratedHeatLoss_J":last["heat_loss_J"],
            "initialMass_kg":row0["mass_kg"],"finalMass_kg":last["mass_kg"],
            "frontProgressThreshold":self.front_threshold,"establishedReactedAreaFraction":self.established_fraction,
            "levelSetTracking":"phi_reaction=X_front-(w1*alpha1+w2*alpha2); distance_reconstructed_only_for_geometry",
            "materialInterface":"internal_unreacted_reacted_partition_same_Tait_no_material_removed",
            "regressionVelocityDefinition":"same_area_change_over_front_length_lab_frame_effective_metric_not_ablation_speed",
            "reactionInventoryTracked":True,"generatedWaterIsSeparateFromMobileWater":True,
            "separateSolidEnergyEquationUsed":False,"gasMassSourceUsed":False,
            "numericalFlux":"WENO5_JS_shared_scalar_weights_"+self.riemann.upper(),
            "timeIntegrator":"chemistry_half__SSPRK33_nonchemical__chemistry_half"
                if self.chemistry_integration_mode=="local_adaptive_thermochemical" else "SSPRK33",
            "bcGlobalConfigSHA256":self.a.audit["bc_config_sha256"],"durationAfterOnset_s":t,
            "wallClockTime_s":wall_clock,
            "computeBackend":"numpy_cpu", "floatingPointDtype":"float64",
        }
        if self.chemistry_integration_mode=="local_adaptive_thermochemical":
            # These names describe the retired clipping/subcycling algorithm.
            # Keep them for legacy outputs only; the local solver reports the
            # physical rate-threshold and inventory-depletion diagnostics above.
            metrics.pop("maximumChemicalRateCapFraction")
            metrics.pop("maximumChemicalInventoryLimiterFraction")
            metrics.pop("maximumChemistrySubcyclesPerPDEStage")
        _write_json(out/"propagation_metrics.json",metrics)
        with (out/"propagation_history.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        s=np.asarray(snapshots)
        fields={"U_initial":initial,"U_final":U,"state_names":np.asarray(STATE_NAMES),
                "snapshot_times_s":np.asarray(snapshot_t),"temperature_snapshots_K":s[...,0],
                "alpha1_snapshots":s[...,1],"alpha2_snapshots":s[...,2],"progress_snapshots":s[...,3],
                "density_snapshots_kg_per_m3":s[...,4],"velocity_x_snapshots_m_per_s":s[...,5],
                "velocity_y_snapshots_m_per_s":s[...,6],"pressure_snapshots_Pa":s[...,7],
                "reaction_level_set_snapshots":s[...,8],"level_set_snapshots_m":s[...,9],
                "final_temperature_K":P[...,3],"final_alpha_channel1":P[...,A1],"final_alpha_channel2":P[...,A2],
                "final_global_progress":X,"final_mobile_lp_mol_per_m3":.5*(U[...,CATION]+U[...,ANION]),
                "final_mobile_water_mol_per_m3":U[...,WATER],"final_pva_reactive_repeat_mol_per_m3":U[...,PVA],
                "final_generated_water_product_mol_per_m3":U[...,PRODUCT_WATER],
                "electrochemical_lp_consumed_mol_per_m3":U[...,EC_LP],
                "front_arrival_time_s":np.where(np.isfinite(arrival),arrival,self.duration),
                "front_arrival_observed_mask":np.isfinite(arrival),
                "final_reaction_level_set":self.front_threshold-X,"final_level_set_m":s[-1,...,9],
                "reacted_material_mask":X>=self.front_threshold,"unreacted_material_mask":X<self.front_threshold,
                "propellant_mask":mask,"anode_contact_mask":self.a.anode_mask,"cathode_contact_mask":self.a.cathode_mask,
                "potential_at_onset_V":self.a.potential,"qJ_at_onset_diagnostic_only_W_per_m3":self.a.qj_at_onset,
                "qEchem_at_onset_diagnostic_only_W_per_m3":self.a.qe_at_onset,
                **{"last_step_average_"+k:v for k,v in self.last_sources.items()}}
        if self.electrical:
            fields.update({"last_electrical_stage_"+k:v for k,v in self.electrical.last_fields.items()})
        np.savez_compressed(out/"propagation_fields.npz",**fields)
        return metrics


def run_reactive_propagation(handoff,config,bc_global_config,output_dir,*,reactive_config,full_bc_config=None):
    adapted=BCReactiveHandoffAdapter(config,bc_global_config,reactive_config).adapt(handoff)
    solver=BCReactiveSolver(adapted,config,bc_global_config,reactive_config,full_bc_config=full_bc_config)
    return solver.run(output_dir)
