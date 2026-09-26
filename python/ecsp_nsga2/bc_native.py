"""Native B/C FP64 evaluators and candidate-level CPU/CUDA hybrid scheduling.

Existing v7.9.5 and v8.0.0 backends remain selectable, unchanged. Full trial
state stays device-resident; the native loop transfers only compact status at
the configured host-check cadence, in addition to synchronization required by
iterative convergence checks. A CPU worker owns an entire reference integration
+ voltage search; the GPU owns independent batched candidates with mixed
per-candidate trial voltages.
"""
from __future__ import annotations
import atexit
import copy
import gc
import json
import math
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .evaluator import (
    BCGlobalPreflameEvaluator,
    BCCandidateGeometryError,
    EvaluatorError,
    _strict_json_value,
)
from .bc_vmin import batched_voltage_search
from ecsp_v6.physics.bc_global import BCCandidateBatchError


def available_cpu_budget(requested:int|None=None) -> int:
    """Respect affinity AND Linux cgroup quota (Kubernetes), not host CPU count."""
    values=[os.cpu_count() or 1]
    try:values.append(len(os.sched_getaffinity(0)))
    except (AttributeError,OSError):pass
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().strip().split()
        if quota!='max':values.append(max(1,math.floor(int(quota)/int(period))))
    except (OSError,ValueError):pass
    try:
        quota=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
        period=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
        if quota>0:values.append(max(1,quota//period))
    except (OSError,ValueError):pass
    if requested is not None:values.append(max(1,int(requested)))
    return max(1,min(values))


def _write(path:Path,value:Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(_strict_json_value(value),indent=2,allow_nan=False),encoding='utf-8')


def _candidate_failure(exc:Exception) -> bool:
    # Toolchain, device, OOM and other infrastructure errors must fail loudly,
    # not be converted into apparent physical non-ignition.  Human-readable
    # messages are not an API: only errors carrying an explicit candidate scope
    # may be isolated by the batch scheduler.
    return isinstance(exc,(BCCandidateBatchError,BCCandidateGeometryError))


class NativeBCGlobalEvaluator(BCGlobalPreflameEvaluator):
    def __init__(self,package_root:Path,config:Mapping[str,Any],workdir:Path):
        import torch
        from ecsp_native import load_native,build_info
        self.native_options=dict(config.get('native',{}))
        self.representative_fields=bool(config.get('save_representative_fields',False))
        super().__init__(package_root,config,workdir)
        if self.dtype!=torch.float64 or self.device.type not in ('cuda','cpu'):
            raise EvaluatorError('Native B/C requires CPU/CUDA FP64. M2 Pro uses CPU, not MPS.')
        torch.set_num_threads(max(1,int(self.native_options.get('torch_threads',1))))
        # The public minimum_ignition_voltage_search block is the one source
        # of truth shared by Python, CPU-native, CUDA and hybrid execution.
        # Retain the old native keys only as equality assertions so a stale
        # performance block can never silently change Vmin semantics.
        native_aliases = {
            'successful_trial_early_stop': self.vmin_stop_successful_trials_at_ignition,
            'verify_final_upper_full_horizon': self.vmin_verify_final,
        }
        for key, canonical in native_aliases.items():
            if key not in self.native_options:
                continue
            raw = self.native_options[key]
            if not isinstance(raw, bool):
                raise EvaluatorError(f'native.{key} must be a boolean')
            if raw != canonical:
                raise EvaluatorError(
                    f'native.{key} contradicts the canonical '
                    'minimum_ignition_voltage_search setting'
                )
        self.early_stop=self.vmin_stop_successful_trials_at_ignition
        self.verify_final=self.vmin_verify_final
        self.handoff_batch=max(1,int(self.native_options.get('handoff_batch_size',4)))
        self.wave_log=self.workdir/'native_batch_waves.jsonl'
        self.actual_batch_limit=self.internal_batch_size
        # Validate options before compilation, before candidate errors can be caught.
        from ecsp_native.adapter import pack_config
        pack_config(self.config,self.composition,self.grid_size)
        runtime=str(self.native_options.get('cpu_runtime','extension'))
        if runtime not in {'extension','standalone'}:
            raise EvaluatorError('native.cpu_runtime must be extension or standalone')
        if self.device.type=='cpu' and runtime=='standalone':
            from ecsp_native.standalone import build_standalone
            build_standalone(bool(self.native_options.get('verbose_build',False)))
        else:
            load_native(self.device.type=='cuda',bool(self.native_options.get('verbose_build',False)))
        path=self.workdir/'evaluator_adapter_diagnostics.json'
        record=json.loads(path.read_text())
        native_build=build_info(self.device.type=='cuda')
        standard_tag=str(native_build['compiler_language_standard']).replace('+','p')
        record.update(backend='bc_global_native',native_build=native_build,
            numerical_engine=f'compiled_{standard_tag}_fp64_with_custom_cuda_kernels',batch_size=self.internal_batch_size,
            vmin_search='candidate_parallel_bisection',successful_trial_early_stop=self.early_stop,
            final_upper_full_horizon_verification=self.verify_final,
            geometry_semantics='full_domain_propellant_with_separate_surface_contact_masks',
            electrode_masks_remove_propellant=False,
            core_source_language_floor='c++17',
            compiler_language_standard=native_build['compiler_language_standard'])
        _write(path,record)

    def _run_physics(self,geometry,voltage,*,save_handoff=False,stop_on_onset=False):
        from ecsp_native import run_native_batch
        capture=bool(getattr(self,'_capture_representative_fields',False))
        return run_native_batch(geometry,self.config,self.composition,voltage,self.dtype,
            save_fields=bool(save_handoff or capture),save_handoff=save_handoff,stop_on_onset=stop_on_onset,
            native_options=self.native_options)

    def _safe_run(self,items,volts,*,role='reference',save_handoff=False,early=False):
        """Split OOMs/stragglers; retain alignment, do not change physics/tolerances."""
        import torch
        items=list(items);volts=list(volts)
        if not items:return [],[]
        if len(items)>self.actual_batch_limit:
            allrows=[];allhands=[]
            pos=0
            while pos<len(items):
                end=min(len(items),pos+self.actual_batch_limit)
                rows,hands=self._safe_run(items[pos:end],volts[pos:end],
                                         role=role,save_handoff=save_handoff,early=early)
                allrows.extend(rows);allhands.extend(hands);pos=end
            return allrows,allhands
        started=time.perf_counter()
        try:
            capture_fields=bool(self.representative_fields and role=='reference' and not early and not save_handoff)
            previous_capture=bool(getattr(self,'_capture_representative_fields',False))
            self._capture_representative_fields=capture_fields
            try:
                rows,out=self._run_model(items,volts,save_handoff=save_handoff,write_metrics=False,stop_on_onset=early)
            finally:
                self._capture_representative_fields=previous_capture
            if capture_fields:
                from .field_diagnostics import save_representative_field_artifacts
                field_rows=save_representative_field_artifacts(
                    items,rows,out,config=self.config,composition=self.composition,
                    grid_size=self.grid_size,domain_size_m=self.domain_size_m)
                if len(field_rows)!=len(rows):
                    raise RuntimeError('Representative field capture lost candidate alignment')
                for row,field_row in zip(rows,field_rows):
                    row.update(field_row)
            hands=self._extract_handoff_batch(rows,out,items) if save_handoff else []
        except Exception as exc:
            oom=isinstance(exc,torch.OutOfMemoryError) or 'cuda out of memory' in str(exc).lower()
            if oom and len(items)==1:raise RuntimeError('Native single-candidate CUDA OOM; reduce grid/handoff storage explicitly') from exc
            if not oom and not _candidate_failure(exc):raise
            if len(items)>1:
                split=len(items)//2
                if oom:
                    exc.__traceback__=None  # release failed CUDA tensors before retry
                    gc.collect()
                    self.actual_batch_limit=min(self.actual_batch_limit,max(1,split))
                    torch.cuda.empty_cache()
                a,ha=self._safe_run(items[:split],volts[:split],role=role,save_handoff=save_handoff,early=early)
                b,hb=self._safe_run(items[split:],volts[split:],role=role,save_handoff=save_handoff,early=early)
                return a+b,ha+hb
            row=self._failed_geometry_row(items[0][2],Path(items[0][3]),exc)
            row['backend']='bc_global_native';row['nativeExecution']={'trialOnly':early,'validity_window':'numerical_failure'}
            rows=[row];hands=[None] if save_handoff else []
        log={'role':role,'count':len(items),'voltages_V':volts,'elapsed_s':time.perf_counter()-started,
             'device':str(self.device),'early_stop_requested':early,'save_handoff':save_handoff,
             'pid':os.getpid(),'actual_batch_limit':self.actual_batch_limit,
             'steps_per_candidate':[r.get('nativeExecution',{}).get('stepsExecutedPerCandidate') for r in rows]}
        with self.wave_log.open('a',encoding='utf-8') as f:f.write(json.dumps(_strict_json_value(log),allow_nan=False)+'\n')
        return rows,hands

    def _evaluate_chunk(self,items):
        rows,_=self._safe_run(items,[self.voltage]*len(items))
        def trials(indices,voltages,role):
            batch=[]
            for idx,v in zip(indices,voltages):
                a,c,meta,folder=items[idx]
                directory=Path(folder)/'vmin_search'/f'{role}_V_{v:.6f}'.replace('.','p')
                batch.append((a,c,meta,directory))
            got,_=self._safe_run(batch,voltages,role=role,
                early=self.early_stop and role!='final_upper_full_horizon_verification')
            for item,row in zip(batch,got):_write(Path(item[3])/'condensed_metrics.json',row)
            return got
        batched_voltage_search(rows,trials,self._trial_valid,enabled=self.vmin_enabled,
            low_voltage=self.vmin_lower_bound_V,high_voltage=self.vmin_upper_bound_V,
            tolerance=self.vmin_tolerance_V,max_iterations=self.vmin_max_iterations,
            invalid_penalty=self.vmin_invalid_penalty_V,censor_penalty=self.vmin_right_censor_penalty_V,
            verify_final=self.verify_final)
        for item,row in zip(items,rows):
            row['nativeWorkerPid']=os.getpid()
            _write(Path(item[3])/'condensed_metrics.json',row)
        return rows

    def evaluate_batch(self,items):
        # Native path intentionally does not inherit reference's broad exception
        # handler, which would hide a missing compiler as a geometry rejection.
        results=[]
        for start in range(0,len(items),self.internal_batch_size):
            part=list(items[start:start+self.internal_batch_size]);t=time.perf_counter()
            results.extend(self._evaluate_chunk(part))
            print(f'[bc-native] device={self.device} candidates={len(part)} elapsed={time.perf_counter()-t:.3f}s',flush=True)
        return results

    def evaluate_handoff_batch(self,items):
        result=[]
        for start in range(0,len(items),self.handoff_batch):
            part=list(items[start:start+self.handoff_batch])
            rows,hands=self._safe_run(part,[self.voltage]*len(part),role='full_horizon_handoff',save_handoff=True,early=False)
            for item,row,hand in zip(part,rows,hands):
                if hand is None:
                    failure=RuntimeError(str(row.get('physicsRejectionReason','Native candidate handoff failed')))
                    result.append(self._failed_handoff_result(item,failure))
                else:result.append(hand)
        return result


_CPU_EVALUATOR=None

def _cpu_init(root,config,folder,threads):
    global _CPU_EVALUATOR
    import torch
    os.environ['OMP_NUM_THREADS']=str(threads);os.environ['MKL_NUM_THREADS']=str(threads)
    torch.set_num_threads(threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    _CPU_EVALUATOR=NativeBCGlobalEvaluator(Path(root),config,Path(folder)/f'worker_{os.getpid()}')

def _cpu_job(index,item):
    if _CPU_EVALUATOR is None:raise RuntimeError('CPU native worker not initialized')
    return index,_CPU_EVALUATOR.evaluate_batch([item])[0]


class NativeBCHybridEvaluator:
    """One GPU owner + spawned CPU workers, each with the same B/C equations.

    CPU-only pool is also supported explicitly, including Apple Silicon.
    Source compilation occurs once in parent BEFORE workers are spawned.
    """
    def __init__(self,package_root:Path,config:Mapping[str,Any],workdir:Path):
        import torch
        from ecsp_native import load_native
        self.package_root=Path(package_root);self.workdir=Path(workdir);self.workdir.mkdir(parents=True,exist_ok=True)
        self.adapter=copy.deepcopy(dict(config));native=dict(config.get('native',{}))
        self.mode=str(config.get('backend','bc_global_native_hybrid'))
        self.use_cuda=self.mode not in {'bc_native_cpu_pool','bc_global_native_cpu_pool'}
        if self.use_cuda and not torch.cuda.is_available():
            raise EvaluatorError('bc_global_native_hybrid requires CUDA; use bc_native_cpu_pool explicitly for CPU/M2.')
        threads=max(1,int(native.get('cpu_threads_per_worker',1)))
        budget=available_cpu_budget(native.get('cpu_budget',8))
        reserve=max(0,int(native.get('host_reserve',2 if self.use_cuda else 1)))
        requested=max(0,int(native.get('cpu_workers',6)))
        available=max(0,(budget-reserve)//threads)
        self.workers=min(requested,available)
        if not self.use_cuda:self.workers=max(1,self.workers)
        self.threads=threads;self.pool=None;self.schedule_log=self.workdir/'hybrid_schedule.jsonl'
        main=copy.deepcopy(self.adapter);main['backend']='bc_global_native';main['device']='cuda' if self.use_cuda else 'cpu'
        main.setdefault('base_overrides',{}).setdefault('numerics',{})['physicsDevice']=main['device']
        self.gpu_or_main=NativeBCGlobalEvaluator(package_root,main,self.workdir/'primary')
        # Expose the unchanged workflow contract for propagation and model metadata.
        self.config=self.gpu_or_main.config;self.composition=self.gpu_or_main.composition
        self.device=self.gpu_or_main.device;self.dtype=self.gpu_or_main.dtype
        self.voltage=self.gpu_or_main.voltage;self.grid_size=self.gpu_or_main.grid_size
        self.internal_batch_size=self.gpu_or_main.internal_batch_size
        self.cpu_config=copy.deepcopy(self.adapter);self.cpu_config.update(backend='bc_global_native',device='cpu',internal_batch_size=1)
        self.cpu_config.setdefault('base_overrides',{}).setdefault('numerics',{})['physicsDevice']='cpu'
        self.cpu_config['native']=dict(native,torch_threads=threads)
        if self.workers:
            if native.get('cpu_runtime','extension')=='standalone':
                from ecsp_native.standalone import build_standalone
                build_standalone(bool(native.get('verbose_build',False)))
            else:load_native(False,bool(native.get('verbose_build',False)))
        from ecsp_native.loader import required_cpp_standard
        _write(self.workdir/'hybrid_runtime.json',{'cpu_budget_detected':budget,'host_reserve':reserve,
            'cpu_workers':self.workers,'cpu_threads_per_worker':threads,'gpu_enabled':self.use_cuda,
            'gpu_batch':self.internal_batch_size,'start_method':'spawn',
            'cpu_engine':f"same_{required_cpp_standard().replace('+','p')}_BC_FP64",
            'core_source_language_floor':'c++17',
            'allocation_unit':'whole_candidate_reference_plus_voltage_search'})
        atexit.register(self.close)

    def _start_pool(self):
        if self.pool is None and self.workers:
            os.environ["OMP_NUM_THREADS"]=str(self.threads)
            os.environ["MKL_NUM_THREADS"]=str(self.threads)
            self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context('spawn'),
                initializer=_cpu_init,initargs=(str(self.package_root),self.cpu_config,str(self.workdir/'cpu_pool'),self.threads))
        return self.pool

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True,cancel_futures=True);self.pool=None

    def evaluate_batch(self,items):
        if not items:return []
        if not self.workers:return self.gpu_or_main.evaluate_batch(items)
        pool=self._start_pool();pending=deque(enumerate(items));futures={};out=[None]*len(items)
        def submit():
            while pending and len(futures)<self.workers:
                idx,item=pending.popleft();future=pool.submit(_cpu_job,idx,item);futures[future]=idx
        def collect(block=False):
            ready=wait(list(futures),return_when=FIRST_COMPLETED).done if block and futures else {f for f in futures if f.done()}
            for f in ready:
                idx,row=f.result();out[idx]=row;del futures[f]
                with self.schedule_log.open('a') as log:log.write(json.dumps({'index':idx,'device':'cpu','pid':row.get('nativeWorkerPid')})+'\n')
        try:
            submit()
            while pending or futures:
                collect();submit()
                if self.use_cuda and pending:
                    part=[pending.popleft() for _ in range(min(self.internal_batch_size,len(pending)))]
                    rows=self.gpu_or_main.evaluate_batch([x[1] for x in part])
                    for (idx,_),row in zip(part,rows):out[idx]=row
                    with self.schedule_log.open('a') as log:log.write(json.dumps({'indices':[p[0] for p in part],'device':'cuda','pid':os.getpid()})+'\n')
                elif futures:collect(block=True)
            if any(r is None for r in out):raise RuntimeError('Hybrid scheduler lost a candidate result')
            return out
        except BaseException:
            self.close();raise

    def evaluate(self,a,c,metadata,output_dir):return self.evaluate_batch([(a,c,metadata,output_dir)])[0]
    def evaluate_handoff_batch(self,items):
        self.close()
        return self.gpu_or_main.evaluate_handoff_batch(items)
    def evaluate_handoff(self,a,c,metadata,output_dir):return self.evaluate_handoff_batch([(a,c,metadata,output_dir)])[0]
