"""Batched current-state BC NP/BV callback on the selected torch device.

No Euler fields are copied through NumPy. Existing BC nonlinear/linear solver
convergence gates remain in force; this is not a frozen onset heat replay.
The BC elliptic solver itself still contains convergence synchronizations, so
power-on runs cannot use CUDA graph capture of the whole Reactive RK step.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from ecsp_nsga2.propagation import PropagationConfigurationError
from ecsp_v6.physics.bc_global import bc_transport_fields
from ecsp_v6.physics.electrochem import initial_state,solve_electrical_and_reaction
from ecsp_v6.physics.geometry import GeometryBatch
from ecsp_v6.physics.composition_model import build_composition
from ecsp_v6.physics.species import _flux_divergence_charged,_flux_divergence_neutral
from .handoff import model_digest
from .chemistry import *


class TensorBCElectricalAdapter:
    def __init__(self,solvers,full_bc_config,kernel):
        if not full_bc_config or 'bcGlobal' not in full_bc_config:
            raise PropagationConfigurationError('Tensor power-on continuation needs full resolved BC configuration')
        self.cfg=copy.deepcopy(full_bc_config)
        if any(model_digest(self.cfg['bcGlobal'])!=s.a.audit['bc_config_sha256'] for s in solvers):
            raise PropagationConfigurationError('Tensor electrical and handoff BC hashes differ')
        if self.cfg.get('interface',{}).get('boundaryCouplingModel')!='surface_overlay_bv':
            raise PropagationConfigurationError('Tensor electrical callback requires surface_overlay_bv')
        self.cfg['numerics']['physicsDevice']=str(kernel.device)
        self.cfg['numerics']['physicsDtype']='float64'
        self.k=kernel;self.device=kernel.device
        self.composition=build_composition(self.cfg)
        bt=lambda arrays:kernel.tensor(np.stack(arrays),dtype=torch.bool)
        self.geometry=GeometryBatch([f'bc_reactive_{i}' for i in range(len(solvers))],
            bt([s.a.anode_mask for s in solvers]),bt([s.a.cathode_mask for s in solvers]),
            bt([np.zeros(s.U.shape[:2],bool) for s in solvers]),bt([s.a.propellant_mask for s in solvers]),
            kernel.nx,kernel.nx*kernel.dx,0.)
        self.voltage=float(self.cfg['coupled']['voltage_V'])
        self.state=initial_state(self.geometry,self.cfg,self.composition,self.voltage,torch.float64)
        self.potential=kernel.tensor(np.stack([s.a.potential for s in solvers]))
        self.calls=0;self.maximum_mismatch=kernel.tensor(np.zeros(len(solvers)))
        self.last_fields={}
        tr=self.cfg['transport'];frac=float(tr['concentrationMinimumFraction'])
        self.floor=max(float(self.cfg.get('physicalFloors',{}).get('concentration_mol_per_m3',1e-12)),
                       frac*max(self.composition.initial_cation_mol_per_m3,self.composition.initial_anion_mol_per_m3))

    @torch.inference_mode()
    def evaluate(self,U,T,dt):
        state=self.state;cfg=self.cfg;g=self.geometry;k=self.k
        state['temperature']=T
        state['cation']=U[...,CATION];state['anion']=U[...,ANION];state['water']=U[...,WATER]
        X=k.progress(U);state['chemicalProgress']=X;state['globalProgress']=X
        state['alphaChannel1']=U[...,A1]/U[...,RHO];state['alphaChannel2']=U[...,A2]/U[...,RHO]
        state['potential']=self.potential
        transport=bc_transport_fields(state,g,cfg,self.composition)
        electrical,reaction,diagnostics=solve_electrical_and_reaction(state,transport,g,cfg,self.composition,
                                             self.voltage,k.dx,static=False)
        self.calls+=1
        linear=diagnostics.converged
        nonlinear=reaction.get('nonlinearRobinConverged',torch.zeros_like(linear,dtype=torch.bool))
        mismatch=reaction['currentBalanceMismatchBeforeCoupling']
        limit=float(cfg['interface']['nonlinearRobin'].get('currentBalanceTolerance',5e-3))
        valid=linear&nonlinear&torch.isfinite(mismatch)&(mismatch<=limit)
        self.maximum_mismatch=torch.maximum(self.maximum_mismatch,mismatch)
        self.potential=state['potential']
        qj,qe=electrical['jouleHeat_W_per_m3'],reaction['qTotal_W_per_m3']
        salt,water=reaction['saltSink_mol_per_m3_s'],reaction['waterSink_mol_per_m3_s']
        for field in (qj,qe,salt,water):
            valid=valid&torch.isfinite(field).all((1,2))&(field>=-1e-12).all((1,2))
        valid=valid&torch.where(k.contacts,torch.ones_like(k.contacts),abs(qe)<=1e-12).all((1,2))
        R=float(cfg['transport']['gasConstant_J_per_molK']);F=float(cfg['transport']['faradayConstant_C_per_mol'])
        if cfg['coupled'].get('useFullNernstPlanckTransport',True):
            cp=-_flux_divergence_charged(state['cation'],transport['Dcation'],state['potential'],T,
                        float(cfg['transport']['chargeNumberCation']),R,F,g.propellant,k.dx)
            cm=-_flux_divergence_charged(state['anion'],transport['Danion'],state['potential'],T,
                        float(cfg['transport']['chargeNumberAnion']),R,F,g.propellant,k.dx)
            wr=-_flux_divergence_neutral(state['water'],transport['Dwater'],g.propellant,k.dx)
            neutral=.5*(cp+cm)
        else:
            neutral=torch.zeros_like(qj);wr=neutral
        zero=torch.zeros_like(qj)
        S=torch.stack((zero,zero,zero,qj+qe,zero,zero,neutral-salt,neutral-salt,wr-water,zero,zero,salt),-1)
        available=U+dt[:,None,None,None]*S
        valid=valid&(available[...,CATION]>=-1e-10).all((1,2))&(available[...,WATER]>=-1e-10).all((1,2))
        self.last_fields={'potential_V':self.potential,'qJ_W_per_m3':qj,'qEchem_W_per_m3':qe,
          'inPlaneJouleHeat_W_per_m3':electrical['inPlaneJouleHeat_W_per_m3'],
          'contactNormalJouleHeat_W_per_m3':electrical['contactNormalJouleHeat_W_per_m3']}
        return S,qj,qe,valid
