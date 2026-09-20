"""Single CUDA owner + bounded spawned CPU workers for post-onset refinement.

CPU workers run independent NumPy cases/baseline comparisons. All requested
Reactive tensor cases stay on one GPU; a CUDA failure is never replaced by a
CPU result. Candidate order and the same-handoff comparison are preserved.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from .bc_native import available_cpu_budget
from .propagation import PropagationCandidateError,PropagationConfigurationError,run_condensed_propagation
from .post_onset import validate_post_onset_config,handoff_digest,_comparison,_json,run_post_onset


def _cpu_init():
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    import torch
    torch.set_num_threads(1)
    # threadpoolctl also caps already-imported BLAS libraries after spawn.
    try:
        from threadpoolctl import threadpool_limits
        global _thread_limiter
        _thread_limiter=threadpool_limits(limits=1)
    except ImportError:
        pass


def _cpu_case(handoff,prop,bc,out,post,full):
    try:
        metrics=run_post_onset(handoff,prop,bc,out,post_onset_config=post,full_bc_config=full)
        metrics['postOnsetCPUWorkerPid']=os.getpid()
        return metrics
    except PropagationCandidateError as exc:
        return exc


def _cpu_baseline(handoff,prop,bc,out):
    try:
        result=run_condensed_propagation(handoff,prop,bc,Path(out)/'condensed_propagation')
        result['postOnsetCPUWorkerPid']=os.getpid()
        return result
    except PropagationCandidateError as exc:
        return exc


def run_post_onset_batch(handoffs,propagation_config,bc_config,output_dirs,*,post_onset_config,full_bc_config=None):
    from ecsp_reactive.condensed.tensor_solver import validate_execution_config,run_tensor_propagation_batch
    import torch
    cfg=validate_post_onset_config(post_onset_config)
    ex=validate_execution_config(cfg.get('execution',{}))
    if len(handoffs)!=len(output_dirs):raise PropagationConfigurationError('Post-onset batch input/output length mismatch')
    if not handoffs:return []
    if cfg['compare_backends'] and propagation_config.get('continued_electrical_heating',False):
        raise PropagationConfigurationError('Paired comparison requires power-off on both models')
    n=len(handoffs);out=[Path(o) for o in output_dirs]
    if len({str(p.resolve()) for p in out})!=n:
        raise PropagationConfigurationError('Post-onset output directories must be unique')
    hashes=[handoff_digest(h) for h in handoffs];results=[None]*n
    budget=available_cpu_budget(ex['cpu_budget']);reserve=min(ex['host_reserve'],max(0,budget-1))
    workers=max(1,min(ex['cpu_workers'],max(1,budget-reserve),n))
    cpu_post=copy.deepcopy(cfg);cpu_post.pop('execution',None)
    if ex['backend']=='torch_batch' and str(ex['device']).startswith('cuda') and not torch.cuda.is_available():
        raise PropagationConfigurationError('CUDA requested for Reactive batch but unavailable; no implicit CPU fallback')
    torch.set_num_threads(min(ex['torch_threads'],max(1,reserve or budget)))
    metadata={'schema':'ecsp.post-onset-batch/v1','requested_cpu_budget':ex['cpu_budget'],'detected_cpu_budget':budget,
              'cpu_workers':workers,'host_reserve':reserve,'process_start_method':'spawn',
              'execution':ex,'case_count':n,'candidate_order_preserved':True,'cuda_oom_splits':0,
              'note':'execution evidence, not material/experimental validation'}
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_cpu_init) as pool:
        if ex['backend']=='numpy_cpu':
            # A bounded queue avoids pickling every large handoff at once.
            pending={};next_index=0
            while next_index<n or pending:
                while next_index<n and len(pending)<2*workers:
                    i=next_index;next_index+=1
                    pending[i]=pool.submit(_cpu_case,handoffs[i],propagation_config,bc_config,str(out[i]),cpu_post,full_bc_config)
                i=min(pending);results[i]=pending.pop(i).result()
        else:
            if cfg['backend']!='reactive_euler' and not cfg['compare_backends']:
                raise PropagationConfigurationError('torch_batch requested without an active Reactive solver')
            baseline_futures={};baseline_results={};baseline_cursor=0
            successful=[i for i,h in enumerate(handoffs) if bool(h.get('onsetSucceeded',False))]
            if cfg['compare_backends']:
                def fill_baselines():
                    nonlocal baseline_cursor
                    ready=[i for i,f in baseline_futures.items() if f.done()]
                    for i in ready:baseline_results[i]=baseline_futures.pop(i).result()
                    while baseline_cursor<len(successful) and len(baseline_futures)<2*workers:
                        i=successful[baseline_cursor];baseline_cursor+=1
                        baseline_futures[i]=pool.submit(_cpu_baseline,handoffs[i],propagation_config,bc_config,str(out[i]))
                fill_baselines()
            for i,h in enumerate(handoffs):
                if not bool(h.get('onsetSucceeded',False)):
                    results[i]=_cpu_case(h,propagation_config,bc_config,str(out[i]),cpu_post,full_bc_config)
            reactive={}
            def group(indices):
                if not indices:return
                try:
                    rr=run_tensor_propagation_batch([handoffs[i] for i in indices],propagation_config,bc_config,
                             [out[i]/'reactive_euler' for i in indices],reactive_config=cfg['reactive_euler'],
                             execution_config=ex,full_bc_config=full_bc_config)
                    reactive.update(zip(indices,rr))
                except torch.cuda.OutOfMemoryError:
                    if not ex['oom_split_retry'] or len(indices)==1:raise
                    metadata['cuda_oom_splits']+=1
                    torch.cuda.empty_cache();mid=len(indices)//2;group(indices[:mid]);group(indices[mid:])
                except PropagationConfigurationError as exc:
                    if 'exceeds estimated memory budget' not in str(exc) or len(indices)==1:raise
                    metadata['estimated_memory_splits']=metadata.get('estimated_memory_splits',0)+1
                    mid=len(indices)//2;group(indices[:mid]);group(indices[mid:])
                except PropagationCandidateError as exc:
                    # Handoff/nonlinear callback failures can originate in just
                    # one lane. Isolate by rerunning same physics on the SAME device.
                    if len(indices)==1:reactive[indices[0]]=exc
                    else:
                        metadata['candidate_isolation_splits']=metadata.get('candidate_isolation_splits',0)+1
                        mid=len(indices)//2;group(indices[:mid]);group(indices[mid:])
            for start_i in range(0,len(successful),ex['batch_size']):
                if cfg['compare_backends']:fill_baselines()
                group(successful[start_i:start_i+ex['batch_size']])
            if cfg['compare_backends']:
                while baseline_cursor<len(successful) or baseline_futures:
                    fill_baselines()
                    if baseline_futures:
                        i=min(baseline_futures);baseline_results[i]=baseline_futures.pop(i).result()
            for i in successful:
                out[i].mkdir(parents=True,exist_ok=True)
                paired={};errors={}
                for name,value in [('reactive_euler',reactive[i])]+([('condensed_propagation',baseline_results[i])] if cfg['compare_backends'] else []):
                    if isinstance(value,PropagationCandidateError):
                        errors[name]=value;paired[name]={'status':'failed','errorType':type(value).__name__,'message':str(value)}
                        (out[i]/name).mkdir(exist_ok=True);_json(out[i]/name/'PROPAGATION_FAILED.json',paired[name])
                    else:paired[name]=dict(value)
                comparison=_comparison(paired,hashes[i],out[i]) if cfg['compare_backends'] else None
                selected=cfg['backend']
                if selected in errors:results[i]=errors[selected]
                else:
                    result=dict(paired[selected]);result.update(postOnsetBackend=selected,
                        propagationDirectory=str(out[i]/selected),postOnsetSameHandoffSHA256=hashes[i],
                        postOnsetBackendComparison=comparison,experimentalReactiveRankingRequested=selected=='reactive_euler')
                    results[i]=result
    metadata['wall_clock_s']=time.perf_counter()-start
    for i,h in enumerate(handoffs):
        if handoff_digest(h)!=hashes[i]:raise RuntimeError('Post-onset batch mutated input handoff')
        out[i].mkdir(parents=True,exist_ok=True)
        _json(out[i]/'post_onset_execution.json',metadata)
    return results
