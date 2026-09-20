#!/usr/bin/env python3
"""Reproducible synthetic FP64 arithmetic benchmark, not a material/NSGA run.

Measures matched RHS+SSPRK trial work, excluding file I/O, handoff and NP/BV.
Always includes independent NumPy reference values. GPU wall timing explicitly
synchronizes. Report throughput is kernel-only, not a full-pipeline speedup.
"""
from __future__ import annotations
import argparse,copy,json,os,platform,statistics,time
from pathlib import Path
import numpy as np
import torch
from ecsp_reactive.condensed import BCReactiveHandoffAdapter,BCReactiveSolver
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case
from ecsp_reactive.condensed.tensor_math import TensorCondensedKernel
from ecsp_reactive.condensed.tensor_solver import CapturedTrial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--grids',default='32,96,193');p.add_argument('--batch-sizes',default='1,4,8')
    p.add_argument('--steps',type=int,default=10);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cuda-graph',action='store_true');p.add_argument('--dynamic',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.steps<1 or a.repeats<1:raise SystemExit('positive steps/repeats required')
    if a.output.exists() or a.output.is_symlink():raise SystemExit('Output exists; choose fresh path')
    grids=[int(x) for x in a.grids.split(',')];batches=[int(x) for x in a.batch_sizes.split(',')]
    if any(x<5 or x>1024 for x in grids) or any(x<1 or x>256 for x in batches):raise SystemExit('Invalid grid/batch bounds')
    if a.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable; no CPU fallback')
    if a.cuda_graph and a.device!='cuda':raise SystemExit('CUDA graphs require CUDA')
    torch.set_num_threads(1)
    rows=[]
    for n in grids:
      for B in batches:
        h,prop,bc,r=synthetic_condensed_case(shape=(n,n))
        h['temperatureAtOnset_K']+=np.cos(np.pi*(np.arange(n)[None,:]+.5)/n)
        ss=[BCReactiveSolver(BCReactiveHandoffAdapter(prop,bc,r).adapt(h),prop,bc,r) for _ in range(B)]
        if a.dynamic:
            for s in ss:
                P=s.thermo.primitive(s.U);P[...,0]*=1+1e-7*np.cos(2*np.pi*(np.arange(n)[None,:]+.5)/n);s.U=s.thermo.conservative(P)
        dt=1e-8 if a.dynamic else 1e-5
        ref_times=[];tensor_times=[]
        # All timing includes exactly the same number of conservative trials.
        for rep in range(a.repeats):
            ref=[s.U.copy() for s in ss];t=time.perf_counter()
            for _ in range(a.steps):
                for i,s in enumerate(ss):ref[i]=s.advance(0.,ref[i],dt,first_order=False)[0]
            ref_times.append(time.perf_counter()-t)
        k=TensorCondensedKernel(ss,a.device);d=k.tensor(np.full(B,dt));first=k.tensor(np.zeros(B,bool),dtype=torch.bool)
        trial=CapturedTrial(k,k.initial,d,first) if a.cuda_graph else k.trial
        for _ in range(3):trial(k.initial,d,first)
        if a.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for rep in range(a.repeats):
            U=k.initial.clone()
            if a.device=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            for _ in range(a.steps):U,ledger,diag,fields,valid=trial(U,d,first)
            if a.device=='cuda':torch.cuda.synchronize()
            tensor_times.append(time.perf_counter()-t)
        if not bool(valid.all()):raise RuntimeError('Benchmark trial invalid')
        final=U.cpu().numpy();reference=np.stack(ref)
        np.testing.assert_allclose(final,reference,rtol=2e-10,atol=2e-6)
        tn,tt=statistics.median(ref_times),statistics.median(tensor_times)
        row=dict(grid=n,batch_size=B,steps=a.steps,dynamic_mechanics=a.dynamic,
            numpy_serial_cases_median_s=tn,tensor_batch_median_s=tt,
            kernel_only_numpy_serial_over_tensor_speed_ratio=tn/tt,
            max_absolute_state_difference=float(np.max(abs(final-reference))),
            max_scaled_state_difference=float(np.max(abs(final-reference)/np.maximum(1.,abs(reference)))),
            peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
            numpy_repeat_s=ref_times,tensor_repeat_s=tensor_times)
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(schema='ecsp.reactive-arithmetic-benchmark/v1',scope='synthetic kernel-only excluding I/O scheduler and NP/BV',
        device=a.device,cuda_graph=a.cuda_graph,cpu_count=os.cpu_count(),python=platform.python_version(),
        numpy=np.__version__,torch=str(torch.__version__),torch_cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name() if a.device=='cuda' else None,
        A100_hardware_tested=a.device=='cuda' and 'A100' in torch.cuda.get_device_name(),
        material_coefficients_calibrated=False,full_pipeline_speedup_measured=False,rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
