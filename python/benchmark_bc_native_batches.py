#!/usr/bin/env python3
"""Benchmark identical grammar-generated candidates at different CUDA batch sizes.
Measures synchronized wall time and numerical validity, not GPU utilization alone.
A short horizon is ONLY a screening benchmark; repeat near onset/full horizon
before choosing a production batch. Does not change the supplied input config.
"""
from __future__ import annotations
import argparse,copy,json,math,time
from pathlib import Path
import numpy as np
import torch,yaml
from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
from ecsp_nsga2.evaluator import _strict_json_value

def benchmark_items(workflow, directory):
    pop,rasters=workflow._initial_population()
    return [(rasters[i.geometry_id].anode_mask,rasters[i.geometry_id].cathode_mask,
             workflow._metadata(i,rasters[i.geometry_id],0),directory/i.geometry_id) for i in pop]

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=root/'config/nsga2_bc_global_native_a100_batch32.yaml')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batches',nargs='+',type=int,default=[4,8,16,32,64])
    p.add_argument('--cases',type=int,default=64);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--horizon',type=float,default=.05)
    p.add_argument('--include-vmin',action='store_true')
    p.add_argument('--memory-fraction',type=float,default=.85);args=p.parse_args()
    if not torch.cuda.is_available():p.error('Actual CUDA hardware required; no CPU timing is reported as A100 timing')
    if args.cases<1 or args.repeats<1 or min(args.batches)<1 or args.horizon<=0:p.error('Counts and horizon must be positive')
    if not 0<args.memory_fraction<1:p.error('memory-fraction must be between 0 and 1')
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(args.config.read_text());cfg['project']['device']='cuda'
    cfg['evaluator'].update(backend='bc_global_native',device='cuda')
    cfg['evaluator']['base_overrides']['numerics']['physicsDevice']='cuda'
    cfg['bc_global']['endTime_s']=cfg['bc_global']['evaluationTime_s']=args.horizon
    cfg['physics']['end_time_s']=cfg['physics']['metric_evaluation_time_s']=args.horizon
    cfg['evaluator']['end_time_s']=args.horizon
    cfg['minimum_ignition_voltage_search']['enabled']=args.include_vmin
    opt=cfg['optimization'];opt['population_size']=args.cases
    nt=min(8,args.cases)
    while args.cases%nt:nt-=1
    opt['initial_topologies']=nt;opt['variants_per_topology']=args.cases//nt
    w=NSGA2ElectricalSolidWorkflow(root,cfg,out/'generated');e=w.evaluator
    items=benchmark_items(w,out/'candidates')
    # Compile + representative warmup are deliberately outside the timed region.
    e._safe_run(items[:min(2,len(items))],[e.voltage]*min(2,len(items)),role='benchmark_warmup')
    torch.cuda.synchronize();device=torch.cuda.get_device_properties(e.device);rows=[];reference=None
    for batch in args.batches:
        e.internal_batch_size=batch;e.actual_batch_limit=batch
        times=[];validities=[];mem=[];parity=True
        for repeat in range(args.repeats):
            torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            t=time.perf_counter()
            if args.include_vmin:results=e.evaluate_batch(items)
            else:results,_=e._safe_run(items,[e.voltage]*len(items),role=f'benchmark_B{batch}_R{repeat}')
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            valid=[e._trial_valid(x)[0] and (not args.include_vmin or x.get('minimumIgnitionVoltageSearchValid',False)) for x in results]
            times.append(elapsed);validities.append(sum(valid));mem.append(torch.cuda.max_memory_reserved())
            summary=[(v,x.get('ignitionSucceeded'),x.get('condensedPhaseIgnitionDelay_s'),x.get('remainingReactiveMassFractionAt2s'),x.get('peakCurrentCongestion')) for v,x in zip(valid,results)]
            if reference is None:reference=summary
            else:
                for a,b in zip(reference,summary):
                    if a[:2]!=b[:2]:parity=False;continue
                    for x,y in zip(a[2:],b[2:]):
                        if x is None or y is None:
                            if x is not y:parity=False
                        elif not np.isclose(x,y,rtol=1e-3,atol=1e-4,equal_nan=True):parity=False
        total=sum(times);peak=max(mem)
        rows.append({'requested_batch':batch,'effective_batch_after_fallback':e.actual_batch_limit,
          'wall_s':times,'attempted_candidates_per_s':len(items)*args.repeats/total,
          'numerically_valid_candidates_per_s':sum(validities)/total,'valid_counts':validities,
          'peak_reserved_bytes':peak,'peak_reserved_fraction':peak/device.total_memory,
          'batch_result_parity':parity,'eligible':all(v==len(items) for v in validities) and parity and peak<args.memory_fraction*device.total_memory and e.actual_batch_limit==batch})
        print(json.dumps(rows[-1]),flush=True)
    eligible=[x for x in rows if x['eligible']]
    best=max(eligible,key=lambda x:x['numerically_valid_candidates_per_s'])['requested_batch'] if eligible else None
    report={'hardware':device.name,'total_memory_bytes':device.total_memory,'grid':e.grid_size,
      'horizon_s':args.horizon,'include_vmin':args.include_vmin,'candidates':len(items),'results':rows,
      'recommended_batch_for_this_test_only':best,'warning':'Short-horizon results do not establish near-onset/full-production throughput.'}
    (out/'batch_benchmark.json').write_text(json.dumps(_strict_json_value(report),indent=2,allow_nan=False))
    return 0 if best else 1
if __name__=='__main__':raise SystemExit(main())
