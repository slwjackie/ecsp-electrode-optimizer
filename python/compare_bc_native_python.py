#!/usr/bin/env python3
"""Numerical port parity report; not a physical calibration or production benchmark."""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch,yaml
from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
from ecsp_v6.physics.bc_global import run_bc_global_batch
from ecsp_native import run_native_batch

def main():
    root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--grids',type=int,nargs='+',default=[17,33])
    p.add_argument('--steps',type=int,default=10);p.add_argument('--strict-common-tolerances',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    rows=[];scratch=args.output.parent/'parity_cases';scratch.mkdir(parents=True,exist_ok=True)
    for n in args.grids:
        cfg=yaml.safe_load((root/'config/nsga2_bc_global_native_debug.yaml').read_text())
        cfg['evaluator']['grid_size']=n;dt=float(cfg['bc_global']['timeStep_s'])
        horizon=args.steps*dt
        cfg['bc_global']['endTime_s']=cfg['bc_global']['evaluationTime_s']=horizon
        cfg['physics']['end_time_s']=cfg['physics']['metric_evaluation_time_s']=horizon
        cfg['evaluator']['end_time_s']=cfg['evaluator']['metric_evaluation_time_s']=horizon
        cfg['condensed_ignition']['reference_time_s']=horizon
        cfg['bc_global']['electricalUpdateInterval_s']=5*dt
        if args.strict_common_tolerances:
            override=cfg['evaluator']['base_overrides']
            override['numerics']['potentialSolver'].update(relativeToleranceCoupled=1e-12,maximumIterationsCoupled=4000)
            override['interface'].setdefault('nonlinearRobin',{}).update(maximumIterations=300,
                potentialTolerance_V=1e-6,relativeReactionCurrentTolerance=1e-6,currentBalanceTolerance=1e-6,
                localRobinResidualTolerance_A_per_m2=1e-6,localRobinRelativeResidualTolerance=1e-8,gaugeMaximumIterations=60)
        e=NSGA2ElectricalSolidWorkflow(root,cfg,scratch/f'grid_{n}').evaluator
        a=np.zeros((32,32),bool);b=a.copy();a[5:8,5:26]=True;b[24:27,5:26]=True
        masks=[(a,b),(np.rot90(a).copy(),np.rot90(b).copy())]
        for voltage in (20.,260.):
            items=[(x,y,{'geometry_id':f'orientation_{i}'},scratch/f'g{n}_v{voltage}_{i}') for i,(x,y) in enumerate(masks)]
            g=e._build_geometry_batch(items)
            t=time.perf_counter();reference=run_bc_global_batch(g,e.config,e.composition,voltage,torch.float64,save_fields=True,save_handoff=True);tpy=time.perf_counter()-t
            t=time.perf_counter();native=run_native_batch(g,e.config,e.composition,[voltage]*2,save_fields=True,save_handoff=True);tn=time.perf_counter()-t
            comparisons=[]
            for group in ('finalFields','histories','handoffFields'):
                for key,x in reference[group].items():
                    if not isinstance(x,torch.Tensor):continue
                    y=native[group][key]
                    if x.dtype==torch.bool:
                        comparisons.append({'group':group,'field':key,'passed':bool(torch.equal(x,y))});continue
                    atol=1e-3 if key=='potential_V' else 2e-4 if key=='currentCongestion' else 1e-5
                    rtol=5e-4 if group=='histories' or key.startswith('q') else 2e-6
                    ok=torch.allclose(x,y,rtol=rtol,atol=atol,equal_nan=True)
                    finite=torch.isfinite(x)&torch.isfinite(y)
                    max_abs=float((x[finite]-y[finite]).abs().max()) if finite.any() else 0.
                    comparisons.append({'group':group,'field':key,'passed':bool(ok),'max_abs_error':max_abs,'rtol':rtol,'atol':atol})
            for key in ('ignitionDelay_s','remainingReactiveMassFractionAt2s','peakCurrentCongestion','inputElectricalEnergyAt2s_J'):
                x,y=reference[key],native[key];ok=torch.allclose(x,y,rtol=5e-4,atol=1e-9,equal_nan=True)
                comparisons.append({'group':'metrics','field':key,'passed':bool(ok)})
            rows.append({'grid':n,'steps':args.steps,'voltage_V':voltage,'candidates':2,
                         'python_wall_s':tpy,'native_wall_s_including_first_load':tn,
                         'all_passed':all(z['passed'] for z in comparisons),'comparisons':comparisons})
    report={'status':'passed' if all(x['all_passed'] for x in rows) else 'failed','device':'cpu',
            'dtype':'float64','torch':torch.__version__,'scope':'short evolving-state numerical port comparison',
            'not_experimental_validation':True,'strict_common_tolerances':args.strict_common_tolerances,'cases':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2));return int(report['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
