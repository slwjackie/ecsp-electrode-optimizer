"""BC -> two-channel condensed Euler/Fourier continuation, CPU FP64.

One energy, one density, two advected reaction channels, six inventory ledgers.
SSPRK33 advances conservative fluxes and accepted source terms together. The
same RK weights integrate all mass/energy boundary and source ledgers. Invalid
stages are retried with smaller dt; failed runs never silently become baseline
runs or receive completed/ranking-eligible metrics.
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
    _table_property,_div_k_grad,_arrival_time_statistics,_reaction_level_set,
    _front_edge_count,_effective_regression_velocity,_configuration_integer,_thermal_diffusive_cfl)
from .chemistry import *
from .handoff import BCReactiveHandoffAdapter
from .finite_volume import flux_divergence, mechanically_stationary
from .electrical import BCElectricalAdapter


def _write_json(path,payload):
    Path(path).write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")


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
        self.alpha_step=float(self.cfg.get("maximum_channel_increment",.02))
        # ECSP_POST_ONSET_SUBCYCLING_V1
        self.chemistry_integration_mode=str(
            self.cfg.get("chemistry_integration_mode", "legacy_cap")
        ).lower()
        if self.chemistry_integration_mode not in {"legacy_cap", "subcycle_raw"}:
            raise PropagationConfigurationError(
                "chemistry_integration_mode must be legacy_cap or subcycle_raw"
            )
        self.max_chemistry_subcycles=_configuration_integer(
            self.cfg.get("maximum_chemistry_subcycles_per_pde_step", 4096),
            name="maximum_chemistry_subcycles_per_pde_step",
            minimum=1, maximum=1000000,
        )
        self.max_chemistry_subcycles_used=0
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
                self.cfl,self.thermal_cfl,self.alpha_step,self.epsilon,self.mass_tolerance,self.energy_tolerance))
                or not 0<self.cfl<=.5 or not 0<self.thermal_cfl<=1 or not 0<self.alpha_step<=1
                or not 0<self.front_threshold<=1 or not 0<self.established_fraction<=1):
            raise PropagationConfigurationError("Invalid condensed Reactive time/quality/front parameters")
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
        self.allowed_cap=float(
            self.cfg.get(
                "maximum_chemical_rate_cap_fraction",
                self.prop.get("maximum_chemical_rate_cap_fraction", 0),
            )
        )
        if not 0<=self.allowed_cap<=1:
            raise PropagationConfigurationError("Invalid chemical-rate cap tolerance")
        self.step_number=0; self.rejected_steps=0; self.first_order_steps=0
        self.face_fallbacks=0; self.faces=0; self.hllc_fallbacks=0; self.exact_skipped_steps=0
        self.last_sources={}
        self.max_rate_cap=0.; self.max_inventory_limiter=0.
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

    def time_step(self,U,remaining):
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
        if self.chemistry_integration_mode == "legacy_cap":
            rates=self.chem.raw_rates(U,T)
            has_stock=(.5*(U[...,CATION]+U[...,ANION])>self.concentration_floor)&(U[...,PVA]>0)
            effective_rates=np.minimum(rates,self.chem.maximum_rate)
            maxrate=float(np.max(np.where(has_stock[...,None],effective_rates,0.)))
            reaction_dt=self.alpha_step/maxrate if maxrate>0 else math.inf
        else:
            # Stiff chemistry is integrated locally inside source(); it must
            # not force the entire Euler/Fourier PDE onto the chemistry scale.
            reaction_dt=math.inf
        dt=min(self.dtmax,remaining,thermal_dt,acoustic_dt,reaction_dt)
        if not math.isfinite(dt) or dt<=0 or dt<self.min_dt:
            raise PropagationCandidateNumericalError("Reactive timestep fell below minimum_time_step_s; no artificial sound-speed reduction used")
        return dt

    def rhs(self,t,U,dt,first_order=False):
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

    def advance(self,t,U,dt,first_order=False):
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
                try:
                    candidate,integrated,diagnostics,sources=self.advance(t,U,dt,first_order)
                    record=self._record(t+dt,candidate,initial,budget+integrated)
                    break
                except PropagationCandidateNumericalError:
                    self.rejected_steps+=1
                    if retry==self.max_retries or dt*.5<self.min_dt:
                        raise
                    dt*=.5; first_order=retry>=1
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
            self.exact_skipped_steps+=int(all(d["mechanics_skipped_exact"] for d in diagnostics))
            for d in diagnostics:
                self.face_fallbacks+=d["face_fallbacks"]; self.faces+=d["faces"]; self.hllc_fallbacks+=d["hllc_fallbacks"]
                self.max_rate_cap=max(self.max_rate_cap,d["chemical_rate_cap_fraction"])
                self.max_inventory_limiter=max(self.max_inventory_limiter,d["inventory_limiter_fraction"])
            self.max_chemistry_subcycles_used=max(self.max_chemistry_subcycles_used,int(d.get("chemistry_subcycles",0)))
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
        metrics={
            "status":"complete","onsetSucceeded":True,"postOnsetBackend":"reactive_euler",
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
            "maximumChemicalRateCapFraction":self.max_rate_cap,"maximumChemicalInventoryLimiterFraction":self.max_inventory_limiter,"maximumChemistrySubcyclesPerPDEStage":self.max_chemistry_subcycles_used,"chemicalRateThresholdDiagnosticOnly":self.chemistry_integration_mode=="subcycle_raw",
            "maximumMassBudgetRelativeResidual":max(r["mass_budget_relative_residual"] for r in history),
            "maximumEnergyBudgetRelativeResidual":max(r["energy_budget_relative_residual"] for r in history),
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
            "numericalFlux":"WENO5_JS_shared_scalar_weights_"+self.riemann.upper(),"timeIntegrator":"SSPRK33",
            "bcGlobalConfigSHA256":self.a.audit["bc_config_sha256"],"durationAfterOnset_s":t,
            "wallClockTime_s":time.perf_counter()-start,
            "computeBackend":"numpy_cpu", "floatingPointDtype":"float64",
        }
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
