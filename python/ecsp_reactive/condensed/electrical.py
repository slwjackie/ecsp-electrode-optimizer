"""Recompute the existing BC NP/BV closures on the *current* Euler state.

There is no stale heat replay and no second thermal solve. This adapter uses
CPU FP64 Torch, even if the pre-onset evaluator used native CPU or CUDA. Ionic
relative diffusion/migration is added to conservative material advection.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from ecsp_nsga2.propagation import PropagationConfigurationError, PropagationCandidateNumericalError
from ecsp_v6.physics.bc_global import bc_transport_fields
from ecsp_v6.physics.electrochem import initial_state, solve_electrical_and_reaction
from ecsp_v6.physics.geometry import GeometryBatch
from ecsp_v6.physics.composition_model import build_composition
from ecsp_v6.physics.species import _flux_divergence_charged, _flux_divergence_neutral
from .chemistry import *
from .handoff import model_digest


class BCElectricalAdapter:
    def __init__(self, adapted, full_bc_config):
        if not full_bc_config or "bcGlobal" not in full_bc_config:
            raise PropagationConfigurationError("recomputed electrical heating needs the complete resolved BC configuration")
        self.cfg=copy.deepcopy(full_bc_config)
        if model_digest(self.cfg["bcGlobal"])!=adapted.audit["bc_config_sha256"]:
            raise PropagationConfigurationError("Electrical callback must use the same BC config as the onset handoff")
        if self.cfg.get("interface",{}).get("boundaryCouplingModel")!="surface_overlay_bv":
            raise PropagationConfigurationError("BC Reactive electrical callback requires surface_overlay_bv")
        self.cfg["numerics"]["physicsDevice"]="cpu"
        self.cfg["numerics"]["physicsDtype"]="float64"
        self.composition=build_composition(self.cfg)
        shape=adapted.U.shape[:2]
        def bt(a): return torch.from_numpy(np.array(a,copy=True))[None].to(torch.bool)
        self.geometry=GeometryBatch(["bc_reactive_continuation"],bt(adapted.anode_mask),bt(adapted.cathode_mask),
                                    bt(np.zeros(shape,bool)),bt(adapted.propellant_mask),shape[1],
                                    shape[1]*adapted.dx,0.0)
        self.adapted=adapted
        self.voltage=float(self.cfg["coupled"]["voltage_V"])
        self.potential=np.array(adapted.potential,copy=True)
        self.calls=0
        self.last_fields={}
        self.maximum_mismatch=0.0

    @torch.inference_mode()
    def evaluate(self,U,temperature,dt):
        g=self.geometry; cfg=self.cfg
        state=initial_state(g,cfg,self.composition,self.voltage,torch.float64)
        def tensor(v): return torch.from_numpy(np.ascontiguousarray(v,dtype=np.float64))[None]
        def array(v): return v.detach().cpu().numpy()[0].copy()
        state["temperature"]=tensor(temperature)
        state["cation"]=tensor(U[...,CATION]); state["anion"]=tensor(U[...,ANION])
        state["water"]=tensor(U[...,WATER])
        X=self.adapted.chemistry.progress(U)
        state["chemicalProgress"]=tensor(X); state["globalProgress"]=tensor(X)
        state["alphaChannel1"]=tensor(U[...,A1]/U[...,RHO])
        state["alphaChannel2"]=tensor(U[...,A2]/U[...,RHO])
        state["potential"]=tensor(self.potential)
        transport=bc_transport_fields(state,g,cfg,self.composition)
        electrical,reaction,diagnostics=solve_electrical_and_reaction(
            state,transport,g,cfg,self.composition,self.voltage,self.adapted.dx,static=False)
        self.calls+=1
        linear_ok=bool(torch.all(diagnostics.converged).item())
        nonlinear_ok=bool(torch.all(reaction.get("nonlinearRobinConverged",torch.tensor([False]))).item())
        mismatch=float(reaction["currentBalanceMismatchBeforeCoupling"].max().item())
        max_mismatch=float(cfg["interface"]["nonlinearRobin"].get("currentBalanceTolerance",5e-3))
        if not linear_ok or not nonlinear_ok or not np.isfinite(mismatch) or mismatch>max_mismatch:
            raise PropagationCandidateNumericalError("Post-onset BC electrical solve failed linear/BV/current-balance gates")
        self.maximum_mismatch=max(self.maximum_mismatch,mismatch)
        self.potential=array(state["potential"])
        qj=array(electrical["jouleHeat_W_per_m3"])
        qe=array(reaction["qTotal_W_per_m3"])
        salt_sink=array(reaction["saltSink_mol_per_m3_s"])
        water_sink=array(reaction["waterSink_mol_per_m3_s"])
        if any(not np.isfinite(v).all() for v in (qj,qe,salt_sink,water_sink)):
            raise PropagationCandidateNumericalError("Nonfinite recomputed electrical source")
        contacts=self.adapted.anode_mask|self.adapted.cathode_mask
        if np.any(np.abs(qe[~contacts])>1e-12):
            raise PropagationCandidateNumericalError("Recomputed q_echem leaks outside contact footprints")
        if any(np.any(v < -1e-12) for v in (qj,qe,salt_sink,water_sink)):
            raise PropagationCandidateNumericalError("BC electrical source is negative")
        prop=g.propellant
        R=float(cfg["transport"]["gasConstant_J_per_molK"])
        F=float(cfg["transport"]["faradayConstant_C_per_mol"])
        if cfg["coupled"].get("useFullNernstPlanckTransport",True):
            cp_rate=-_flux_divergence_charged(state["cation"],transport["Dcation"],state["potential"],state["temperature"],
                        float(cfg["transport"]["chargeNumberCation"]),R,F,prop,self.adapted.dx)
            cm_rate=-_flux_divergence_charged(state["anion"],transport["Danion"],state["potential"],state["temperature"],
                        float(cfg["transport"]["chargeNumberAnion"]),R,F,prop,self.adapted.dx)
            w_rate=-_flux_divergence_neutral(state["water"],transport["Dwater"],prop,self.adapted.dx)
            neutral_rate=.5*array(cp_rate+cm_rate)
            water_rate=array(w_rate)
        else:
            neutral_rate=np.zeros_like(qj); water_rate=np.zeros_like(qj)
        S=np.zeros_like(U)
        S[...,CATION]=S[...,ANION]=neutral_rate-salt_sink
        S[...,WATER]=water_rate-water_sink
        S[...,EC_LP]=salt_sink
        # Reject/retry rather than clip a Faradaic extent but retain its heat.
        available=U+dt*S
        if np.any(available[...,CATION]<-1e-10) or np.any(available[...,WATER]<-1e-10):
            raise PropagationCandidateNumericalError("Post-onset NP/Faradaic step exhausts inventory; reduce dt")
        S[...,ENERGY]=qj+qe
        self.last_fields={"potential_V":self.potential.copy(),"qJ_W_per_m3":qj,
                          "qEchem_W_per_m3":qe,
                          "inPlaneJouleHeat_W_per_m3":array(electrical["inPlaneJouleHeat_W_per_m3"]),
                          "contactNormalJouleHeat_W_per_m3":array(electrical["contactNormalJouleHeat_W_per_m3"])}
        return S,qj,qe,mismatch
