"""Batched FP64 torch/CUDA continuation with independent candidate clocks.

One GPU owner, resident 12-field states/tables, per-case timestep/rejection,
small packed host diagnostics, and snapshot-only field transfers. No silent
CPU fallback. CPU uses the same tensor arithmetic for cross-backend tests.
"""
from __future__ import annotations
import copy
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ecsp_nsga2.propagation import (PropagationConfigurationError,PropagationCandidateError,
    ConfiguredModelTemperatureRangeExceeded,candidate_failure_payload,
    PropagationCandidateNumericalError,_reaction_level_set,_configuration_integer)
from .handoff import BCReactiveHandoffAdapter
from .solver import BCReactiveSolver,_write_json
from .tensor_math import (TensorCondensedKernel,RECORD_NAMES,NCONS,
                          LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES)


def validate_execution_config(raw=None):
    cfg=copy.deepcopy(dict(raw or {}))
    allowed={'backend','device','batch_size','cuda_graph','cpu_budget','host_reserve','cpu_workers','torch_threads',
             'maximum_batch_working_bytes','maximum_batch_history_bytes','oom_split_retry'}
    if set(cfg)-allowed:
        raise PropagationConfigurationError('Unknown post_onset.execution keys: '+', '.join(sorted(set(cfg)-allowed)))
    cfg.setdefault('backend','numpy_cpu');cfg.setdefault('device','cpu')
    if cfg['backend'] not in {'numpy_cpu','torch_batch'}:
        raise PropagationConfigurationError('post_onset.execution.backend must be numpy_cpu or torch_batch')
    device=str(cfg['device'])
    if device!='cpu' and device!='cuda' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise PropagationConfigurationError('Reactive device must be cpu, cuda, or cuda:<index>; FP32 MPS is not used')
    if cfg['backend']=='numpy_cpu' and device!='cpu':
        raise PropagationConfigurationError('numpy_cpu cannot request a CUDA device')
    for name,default,minimum,maximum in (
        ('batch_size',8,1,256),('cpu_budget',8,1,65536),('host_reserve',2,0,65535),
        ('cpu_workers',6,1,65536),('torch_threads',1,1,256),
        ('maximum_batch_working_bytes',8*1024**3,1024,2**63-1),
        ('maximum_batch_history_bytes',2*1024**3,1024,2**63-1)):
        cfg[name]=_configuration_integer(cfg.get(name,default),name='post_onset.execution.'+name,minimum=minimum,maximum=maximum)
    for name,default in (('cuda_graph',False),('oom_split_retry',True)):
        v=cfg.get(name,default)
        if not isinstance(v,bool):raise PropagationConfigurationError(name+' must be boolean')
        cfg[name]=v
    if cfg['cuda_graph'] and (cfg['backend']!='torch_batch' or not device.startswith('cuda')):
        raise PropagationConfigurationError('cuda_graph requires torch_batch on CUDA')
    if cfg['host_reserve']>=cfg['cpu_budget']:
        raise PropagationConfigurationError('host_reserve must be below cpu_budget')
    if cfg['cpu_workers']+cfg['host_reserve']>cfg['cpu_budget']:
        raise PropagationConfigurationError('CPU workers + host reserve exceeds requested CPU budget')
    return cfg


class CapturedTrial:
    """Optional fixed-shape CUDA graph; dt is a tensor, not a captured constant."""
    def __init__(self,kernel,U,dt,first_order):
        if kernel.electrical is not None:
            raise PropagationConfigurationError('CUDA graph capture is disabled for iterative NP/BV power-on callbacks')
        self.U=U.clone();self.dt=dt.clone();self.first=first_order.clone()
        stream=torch.cuda.Stream(device=U.device)
        stream.wait_stream(torch.cuda.current_stream(U.device))
        with torch.cuda.stream(stream):
            for _ in range(3): kernel.trial(self.U,self.dt,self.first)
        torch.cuda.current_stream(U.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output=kernel.trial(self.U,self.dt,self.first)
    def __call__(self,U,dt,first_order):
        self.U.copy_(U);self.dt.copy_(dt);self.first.copy_(first_order)
        self.graph.replay()
        return self.output


def _snapshot(s,state,time_value,snapshots,snapshot_times):
    p=s.thermo.primitive(state);x=s.chem.progress(state)
    distance=_reaction_level_set(x,s.a.propellant_mask,s.front_threshold,s.a.dx)
    snapshots.append(np.stack((p[...,3],p[...,4],p[...,5],x,p[...,0],p[...,1],p[...,2],
                               s.thermo.pressure(p[...,0]),s.front_threshold-x,distance),-1))
    snapshot_times.append(float(time_value))


def _local_chemistry_diagnostics(row):
    diagnostics=dict(zip(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES,map(float,row)))
    diagnostics["local_chemistry_method"]=(
        "bulk_midpoint_with_dominant_driver_reaction_coordinate_and_"
        "error_controlled_midpoint_fallback"
    )
    diagnostics["coupled_reaction_coordinate_rule"]="dormand_prince_5_4"
    diagnostics["coupled_reaction_coordinate_safety_factor"]=2.0
    diagnostics["coupled_reaction_coordinate_work_counter_scope"]=(
        "rate_step_depth_residual_counters_are_successful_endpoint_only; "
        "attempt_and_fallback_counts_cover_all_attempts; fallback_work_is_"
        "included_in_chemistry_wall_clock"
    )
    diagnostics["one_active_reaction_coordinate_quadrature_rule"]=(
        "gauss_legendre_8_16"
    )
    diagnostics["one_active_reaction_coordinate_work_counter_scope"]=(
        "successful_coordinate_endpoints_only; fallback_work_is_included_in_"
        "chemistry_wall_clock"
    )
    diagnostics["inventory_depleted_fraction"]=diagnostics["inventory_limiter_fraction"]
    diagnostics["chemical_rate_threshold_diagnostic_only"]=True
    diagnostics["raw_rate_observation_scope"]=(
        "chemistry_half_step_initial_endpoint_and_monotone_temperature_"
        "traversed_knot_upper_bound"
    )
    return diagnostics


@torch.inference_mode()
def run_tensor_solvers(solvers,output_dirs,execution_config,*,full_bc_config=None):
    """Run already-validated solvers. Return metrics OR a candidate error per case.

    Numerical failure of one lane does not cause its siblings to fail. Global
    CUDA OOM is handled by the scheduler via deterministic split/retry, not by
    swapping solvers or changing grids, coefficients, tolerances, or precision.
    """
    ex=validate_execution_config(execution_config)
    device=torch.device(ex['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise PropagationConfigurationError('CUDA explicitly requested but unavailable; CPU fallback is forbidden')
    if not solvers or len(solvers)!=len(output_dirs):
        raise PropagationConfigurationError('Tensor batch requires equal nonempty solver/output lists')
    s=solvers[0];B=len(solvers)
    key=lambda v:(v.U.shape,v.a.dx,v.a.thickness,json.dumps(v.prop,sort_keys=True),json.dumps(v.bc,sort_keys=True),json.dumps(v.cfg,sort_keys=True))
    if any(key(v)!=key(s) for v in solvers[1:]):
        raise PropagationConfigurationError('One tensor batch requires identical grids, BC laws and numerical settings')
    if s.chemistry_integration_mode not in {
            'legacy_cap','local_adaptive_thermochemical'}:
        raise PropagationConfigurationError(
            'torch_batch supports legacy_cap or local_adaptive_thermochemical; '
            'the retired explicit subcycle_raw tensor path is unavailable'
        )
    local_mode=s.chemistry_integration_mode=='local_adaptive_thermochemical'
    if local_mode and ex['cuda_graph']:
        raise PropagationConfigurationError(
            'cuda_graph is disabled for local_adaptive_thermochemical dynamic refinement'
        )
    if s.heating and ex['cuda_graph']:
        raise PropagationConfigurationError('Use cuda_graph:false for recomputed NP/BV electrical heating')
    state_bytes=sum(v.U.nbytes for v in solvers)
    # Planning bound, not a claim about measured allocator/RSS usage.
    work_estimate=state_bytes*(24 if all(v.fast and v.riemann=="hllc" and np.all(v.U[...,:3]==v.U[0,0,:3]) and np.all(v.U[...,1:3]==0) for v in solvers) else 220)
    if ex['cuda_graph']:work_estimate*=2
    snapshot_count=2+math.ceil(s.duration/s.snapshot_interval)
    history_estimate=B*(snapshot_count*s.U.shape[0]*s.U.shape[1]*10*8*2+s.max_steps*4096)
    if work_estimate>ex['maximum_batch_working_bytes'] or history_estimate>ex['maximum_batch_history_bytes']:
        raise PropagationConfigurationError('Reactive tensor batch exceeds estimated memory budget; reduce batch_size explicitly')
    k=TensorCondensedKernel(solvers,device)
    if s.heating:
        from .tensor_electrical import TensorBCElectricalAdapter
        k.electrical=TensorBCElectricalAdapter(solvers,full_bc_config,k)
        k.floor=k.electrical.floor
    outs=[Path(o) for o in output_dirs]
    for i,o in enumerate(outs):
        o.mkdir(parents=True,exist_ok=True);_write_json(o/'handoff_audit.json',solvers[i].a.audit)
    if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter();U=k.initial.clone();budget=torch.zeros((B,NCONS+5),dtype=torch.float64,device=device)
    X=k.progress(U);arrival=torch.where(X>=s.front_threshold,torch.zeros_like(X),torch.full_like(X,math.nan))
    ts=np.zeros(B);nextsnap=np.full(B,min(s.duration,s.snapshot_interval));active=np.ones(B,bool)
    retry=np.zeros(B,np.int64);transport_retry=np.zeros(B,np.int64)
    caps=np.full(B,math.inf);counts=np.zeros(B,np.int64)
    errors=[None]*B;established=np.full(B,math.nan)
    history=[[] for _ in range(B)];snaps=[[] for _ in range(B)];snap_times=[[] for _ in range(B)]
    velocities=[[] for _ in range(B)];dts=[[] for _ in range(B)]
    record0,valid0=k.record(k.tensor(ts),U,budget)
    record0=record0.cpu().numpy()
    for i in range(B):
        row=dict(zip(RECORD_NAMES,map(float,record0[i])));row['effective_regression_velocity_m_per_s']=0.
        history[i].append(row);_snapshot(solvers[i],solvers[i].U,0.,snaps[i],snap_times[i])
        if 1-row['unreacted_area_fraction']>=s.established_fraction:established[i]=0.
    last_sources=torch.zeros((*U.shape[:3],5),dtype=torch.float64,device=device)
    captured=None;attempts=0;last_progress_wall=start
    transport_diagnostic_size=3*6
    local_diagnostic_size=2*len(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES)
    record_start=2;transport_start=record_start+len(RECORD_NAMES)
    velocity_column=transport_start+transport_diagnostic_size
    budget_column=velocity_column+1
    local_start=budget_column+1;auxiliary_start=local_start+local_diagnostic_size
    eps=64*np.finfo(float).eps*max(s.duration,1e-12)
    while np.any(active):
        for i in np.where(active&(counts>=s.max_steps))[0]:
            errors[i]=PropagationCandidateNumericalError('maximum_time_steps reached before requested duration')
            active[i]=False
        if not active.any():break
        remaining=np.minimum(s.duration-ts,nextsnap-ts)
        advance_snap=active&(remaining<=eps)&(s.duration-ts>eps)
        nextsnap[advance_snap]=np.minimum(s.duration,nextsnap[advance_snap]+s.snapshot_interval)
        remaining=np.minimum(s.duration-ts,nextsnap-ts)
        remaining=np.where(active,np.maximum(remaining,0.),0.)
        dt=torch.minimum(k.step_size(U,k.tensor(remaining)),k.tensor(caps))
        first=k.tensor((transport_retry if local_mode else retry)>=2,dtype=torch.bool)
        if ex['cuda_graph'] and captured is None:
            captured=CapturedTrial(k,U,dt,first)
        electrical_checkpoint=None
        if k.electrical is not None:
            electrical_checkpoint=(k.electrical.potential.clone(),
                                   k.electrical.maximum_mismatch.clone(),
                                   {name:value.clone() for name,value in k.electrical.last_fields.items()})
        candidate,integrated,diag,fields,valid=(captured or k.trial)(U,dt,first)
        rec,budget_ok=k.record(k.tensor(ts)+dt,candidate,budget+integrated)
        valid=valid&budget_ok&torch.isfinite(dt)&(dt>=s.min_dt)&k.tensor(active,dtype=torch.bool)
        if electrical_checkpoint is not None:
            old_potential,old_mismatch,old_fields=electrical_checkpoint
            lane=valid[:,None,None]
            k.electrical.potential=torch.where(lane,k.electrical.potential,old_potential)
            k.electrical.state['potential']=k.electrical.potential
            k.electrical.maximum_mismatch=torch.where(
                valid,k.electrical.maximum_mismatch,old_mismatch)
            for name,value in tuple(k.electrical.last_fields.items()):
                previous=old_fields.get(name,torch.zeros_like(value))
                k.electrical.last_fields[name]=torch.where(lane,value,previous)
        newX=k.progress(candidate)
        prevcells=(X<s.front_threshold).sum((1,2));newcells=(newX<s.front_threshold).sum((1,2))
        edges=.5*(k.front_edges(X)+k.front_edges(newX))
        velocity=torch.where(edges>0,(prevcells-newcells)*s.a.dx/torch.clamp(edges*dt,min=1e-300),torch.zeros_like(dt))
        if local_mode:
            local_diagnostics=k.trial_local_diagnostics.reshape(B,-1)
            auxiliary=torch.stack((k.trial_post_stability_limit,
                k.trial_post_rechecked.double(),k.trial_post_rejected.double(),
                k.trial_failure_reason.double()),-1)
        else:
            local_diagnostics=torch.zeros((B,local_diagnostic_size),dtype=torch.float64,device=device)
            auxiliary=torch.zeros((B,4),dtype=torch.float64,device=device)
        packed=torch.cat((dt[:,None],valid[:,None].double(),rec,diag.reshape(B,-1),
            velocity[:,None],budget_ok[:,None].double(),
            local_diagnostics,auxiliary),-1).cpu().numpy()
        accepted=packed[:,1].astype(bool)&active;dt_cpu=packed[:,0]
        vel_cpu=packed[:,velocity_column]
        budget_pass_cpu=packed[:,budget_column].astype(bool)
        local_cpu=packed[:,local_start:auxiliary_start].reshape(
            B,2,len(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES))
        auxiliary_cpu=packed[:,auxiliary_start:auxiliary_start+4]
        if local_mode:
            chemistry_wall,nonchemical_wall=k.local_trial_phase_wall_times()
            for i in np.where(active)[0]:
                v=solvers[i]
                # Each active lane experiences the whole batch latency.  Count
                # attempted work even when the trial is later rejected, just as
                # the scalar runner's phase timers do.
                v.chemistry_wall_clock_time_s+=chemistry_wall
                v.nonchemical_wall_clock_time_s+=nonchemical_wall
                v.post_chemistry_timestep_rechecks+=int(auxiliary_cpu[i,1])
                v.post_chemistry_timestep_recheck_rejections+=int(auxiliary_cpu[i,2])
                if auxiliary_cpu[i,1]:
                    ratio=auxiliary_cpu[i,0]/max(dt_cpu[i],1e-300)
                    v.minimum_post_chemistry_timestep_safety_ratio=min(
                        v.minimum_post_chemistry_timestep_safety_ratio,float(ratio))
                for half_row in local_cpu[i]:
                    if half_row[0] <= 0:
                        continue
                    v.chemistry_half_step_attempts+=1
                    if half_row[1] <= 0:
                        v.chemistry_failed_half_steps+=1
                    v._observe_local_chemistry(
                        _local_chemistry_diagnostics(half_row),accepted=False)
        failed=active&~accepted

        def failure_for_lane(index,reason):
            if local_mode and reason==4:
                for row in local_cpu[index]:
                    diagnostic=dict(zip(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES,row))
                    if diagnostic["configured_model_temperature_range_exceeded"]>0:
                        return ConfiguredModelTemperatureRangeExceeded(
                            diagnostic["configured_maximum_temperature_K"],
                            diagnostic["observed_maximum_temperature_K"],
                            diagnostic["offending_temperature_cell_index"],
                            diagnostic["offending_temperature_roundoff_K"],
                        )
                raise RuntimeError("Missing structured temperature-range diagnostic")
            g=packed[index,transport_start:velocity_column].reshape(3,6)
            cap_max=float(g[:,4].max())
            limiter_max=float(g[:,5].max())
            budget_pass=bool(budget_pass_cpu[index])
            local_message=""
            threshold_reference=(
                "" if local_mode else
                f" configured_threshold_fraction_reference={s.allowed_cap:.12g}"
            )
            if local_mode:
                local_message=(
                    f" chemistry_failure_reason={reason}"
                    f" maximum_local_refinement_depth="
                    f"{local_cpu[index,:,7].max():.12g}"
                    f" maximum_embedded_residual="
                    f"{local_cpu[index,:,9].max():.12g}"
                )
            return PropagationCandidateNumericalError(
                f"Reactive tensor failure: "
                f"time={ts[index]:.12g}s "
                f"dt_attempt={dt_cpu[index]:.12g}s "
                f"retry={retry[index]} "
                f"budget_ok={budget_pass} "
                f"chemical_rate_threshold_fraction_diagnostic={cap_max:.12g} "
                f"{threshold_reference} "
                f"inventory_limiter_fraction={limiter_max:.12g}"
                f"{local_message}"
            )

        for i in np.where(failed)[0]:
            solvers[i].rejected_steps+=1
            reason=int(auxiliary_cpu[i,3]) if local_mode else 0

            # A failed cell-local chemistry certification is independent of
            # the global PDE step. Halving that step cannot repair the local
            # map and silently turns its numerical work guard into a trajectory
            # control. Fail this lane immediately while siblings remain fully
            # transactional. Reason 2 is different: the accepted chemistry
            # endpoint tightened the transport stability limit, so a smaller
            # global step is the prescribed retry.
            if local_mode and reason in (1,4):
                errors[i]=failure_for_lane(i,reason)
                active[i]=False
                continue

            retry[i]+=1
            retry_cap=auxiliary_cpu[i,0] if local_mode and reason==2 else math.inf
            caps[i]=min(dt_cpu[i]*.5,retry_cap)
            if not local_mode or reason not in {1,2}:
                transport_retry[i]+=1
            if (retry[i]>s.max_retries or not math.isfinite(caps[i]) or caps[i]<s.min_dt):
                errors[i]=failure_for_lane(i,reason)
                active[i]=False
        # These comparisons/reductions remain on-device; only scalar velocity is transferred.
        crossed=torch.isnan(arrival)&(newX>=s.front_threshold)&valid[:,None,None]
        fraction=torch.where(newX>X,torch.clamp((s.front_threshold-X)/torch.clamp(newX-X,min=1e-300),0.,1.),torch.zeros_like(X))
        arrival=torch.where(crossed,k.tensor(ts)[:,None,None]+fraction*dt[:,None,None],arrival)
        U=torch.where(valid[:,None,None,None],candidate,U)
        budget=torch.where(valid[:,None],budget+integrated,budget)
        last_sources=torch.where(valid[:,None,None,None],fields,last_sources)
        X=torch.where(valid[:,None,None],newX,X)
        for i in np.where(accepted)[0]:
            v=solvers[i];duration=float(dt_cpu[i]);ts[i]+=duration
            if abs(ts[i]-s.duration)<=eps:ts[i]=s.duration
            row=dict(zip(RECORD_NAMES,map(float,packed[i,2:2+len(RECORD_NAMES)])))
            row['time_after_onset_s']=float(ts[i]);row['effective_regression_velocity_m_per_s']=float(vel_cpu[i])
            history[i].append(row);velocities[i].append(float(vel_cpu[i]));dts[i].append(duration)
            g=packed[i,transport_start:velocity_column].reshape(3,6)
            v.step_number+=1
            v.first_order_steps+=int((transport_retry[i] if local_mode else retry[i])>=2)
            v.face_fallbacks+=int(g[:,0].sum());v.faces+=int(g[:,1].sum());v.hllc_fallbacks+=int(g[:,2].sum())
            v.exact_skipped_steps+=int(np.all(g[:,3]>0))
            if local_mode:
                for half_row in local_cpu[i]:
                    if half_row[0] > 0 and half_row[1] > 0:
                        v._observe_local_chemistry(
                            _local_chemistry_diagnostics(half_row),accepted=True)
                        v.chemistry_half_steps_accepted+=1
            else:
                v.max_rate_cap=max(v.max_rate_cap,float(g[:,4].max()))
                v.max_inventory_limiter=max(v.max_inventory_limiter,float(g[:,5].max()))
            if math.isnan(established[i]) and 1-row['unreacted_area_fraction']>=s.established_fraction:established[i]=ts[i]
            counts[i]+=1;retry[i]=0;transport_retry[i]=0;caps[i]=math.inf
            if ts[i]>=nextsnap[i]-eps or ts[i]==s.duration:
                _snapshot(v,U[i].cpu().numpy(),ts[i],snaps[i],snap_times[i])
                nextsnap[i]=min(s.duration,nextsnap[i]+s.snapshot_interval)
            if ts[i]>=s.duration:active[i]=False
        attempts+=1
        now=time.perf_counter()
        wall_due=now-last_progress_wall>=s.progress_log_interval_wall_s
        step_due=any(counts[i]>0 and counts[i]%s.progress_log_interval_steps==0
                     for i in np.where(accepted)[0])
        completion_due=bool(np.any(accepted&~active))
        if wall_due or step_due or completion_due:
            lanes=[]
            for i,v in enumerate(solvers):
                lanes.append({"lane":i,"active":bool(active[i]),
                    "time_after_onset_s":float(ts[i]),
                    "accepted_time_steps":int(v.step_number),
                    "rejected_time_steps":int(v.rejected_steps),
                    "last_time_step_s":float(dt_cpu[i]) if math.isfinite(dt_cpu[i]) else None,
                    "maximum_temperature_K":float(packed[i,record_start+3]),
                    "chemistry_half_step_attempts":int(v.chemistry_half_step_attempts),
                    "accepted_chemistry_half_steps":int(v.chemistry_half_steps_accepted),
                    "failed_chemistry_half_steps":int(v.chemistry_failed_half_steps),
                    "maximum_local_refinement_depth":int(
                        v.maximum_attempted_chemistry_local_refinements),
                    "maximum_raw_chemical_rate_per_s":float(
                        v.maximum_attempted_raw_chemical_rate_per_s)})
            print(json.dumps({"event":"ecsp_reactive_tensor_progress",
                "compute_backend":"torch_cuda_batch" if device.type=='cuda' else "torch_cpu_batch",
                "chemistry_integration_mode":s.chemistry_integration_mode,
                "tensor_batch_trials":attempts,"batch_size":B,
                "wall_clock_time_s":now-start,"lanes":lanes},
                sort_keys=True,allow_nan=False),flush=True)
            last_progress_wall=now
    if device.type=='cuda':torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    Ufinal=U.cpu().numpy();initial=k.initial.cpu().numpy();bud=budget.cpu().numpy();arr=arrival.cpu().numpy()
    sources=last_sources.cpu().numpy();source_names=('qJ_W_per_m3','qEchem_W_per_m3','qChem_W_per_m3','loss_W_per_m3','conduction_W_per_m3')
    result=[]
    peak=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None
    for i,v in enumerate(solvers):
        v.U=Ufinal[i]
        if errors[i] is not None:
            np.savez_compressed(outs[i]/'INCOMPLETE_STATE.npz',U=Ufinal[i],time_after_onset_s=ts[i])
            failure={'status':'failed','message':str(errors[i]),'time_after_onset_s':float(ts[i])}
            if isinstance(errors[i],ConfiguredModelTemperatureRangeExceeded):
                failure.update(candidate_failure_payload(errors[i]))
            _write_json(outs[i]/'PROPAGATION_FAILED.json',failure)
            result.append(errors[i]);continue
        v.last_sources={name:sources[i,...,j] for j,name in enumerate(source_names)}
        if k.electrical:
            e=k.electrical
            v.electrical=SimpleNamespace(calls=e.calls,maximum_mismatch=float(e.maximum_mismatch[i].cpu()),
                         last_fields={name:value[i].cpu().numpy() for name,value in e.last_fields.items()})
        metrics=v.finish(outs[i],start,Ufinal[i],initial[i],bud[i],history[i],snaps[i],snap_times[i],arr[i],velocities[i],dts[i],established[i])
        metrics.update({'computeBackend':'torch_cuda_batch' if device.type=='cuda' else 'torch_cpu_batch',
             'computeDevice':str(device),'floatingPointDtype':'float64','postOnsetBatchSize':B,
             'independentCandidateTimesteps':True,'cudaGraphReplayUsed':bool(captured),
             'tensorBatchWallClockTime_s':elapsed,'tensorBatchTrials':attempts,'peakAllocatedGPUBytes':peak,
             'estimatedBatchWorkingBytes':work_estimate,'estimatedBatchHistoryBytes':history_estimate,
             'postOnsetExecutedOnGPU':device.type=='cuda',
             'fieldTransferPolicy':'initial_upload_then_snapshot_and_final_downloads_only; small_per_case_diagnostics_each_trial'})
        _write_json(outs[i]/'propagation_metrics.json',metrics);result.append(metrics)
    return result


def run_tensor_propagation_batch(handoffs,propagation_config,bc_config,output_dirs,*,reactive_config,execution_config,full_bc_config=None):
    solvers=[]
    for h in handoffs:
        a=BCReactiveHandoffAdapter(propagation_config,bc_config,reactive_config).adapt(h)
        solvers.append(BCReactiveSolver(a,propagation_config,bc_config,reactive_config,
                                       full_bc_config=full_bc_config,initialize_electrical=False))
    return run_tensor_solvers(solvers,output_dirs,execution_config,full_bc_config=full_bc_config)
