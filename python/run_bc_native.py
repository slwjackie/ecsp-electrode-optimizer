#!/usr/bin/env python3
"""Explicit v8.2 corrected B/C native launcher; legacy launchers remain available."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
import yaml

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['debug','cpu','cuda','hybrid'],default='hybrid')
    p.add_argument('--batch',type=int,choices=[32,64],default=32)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--population',type=int)
    p.add_argument('--generations',type=int)
    p.add_argument('--cpu-workers',type=int,default=None)
    p.add_argument('--allow-no-feasible',action='store_true')
    args=p.parse_args()
    if args.cpu_workers is not None and args.cpu_workers<0:p.error('cpu-workers must be nonnegative')
    name={'debug':'nsga2_bc_global_native_debug.yaml','cpu':'nsga2_bc_global_native_cpu8.yaml'}.get(args.mode,f'nsga2_bc_global_native_a100_batch{args.batch}.yaml')
    cfg=yaml.safe_load((root/'config'/name).read_text())
    if args.mode=='cuda':cfg['evaluator']['backend']='bc_global_native';cfg['evaluator']['native']['cpu_workers']=0
    if args.cpu_workers is not None:cfg['evaluator']['native']['cpu_workers']=args.cpu_workers
    if args.mode in ('cuda','hybrid'):
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        if not torch.cuda.is_available():p.error('CUDA device unavailable; choose --mode cpu or debug explicitly')
        if CUDA_HOME is None:p.error('CUDA toolkit/nvcc required for native CUDA compilation; CPU-only PyTorch is insufficient')
    from ecsp_nsga2.bc_native import available_cpu_budget
    native=cfg['evaluator']['native']
    budget=available_cpu_budget(native.get('cpu_budget',8))
    available=max(1,budget-int(native.get('host_reserve',2)))
    cfg['propagation_refinement']['execution']['parallel_cases']=min(
        int(cfg['propagation_refinement']['execution'].get('parallel_cases',1)),available)
    if args.mode=='cuda':cfg['optimization']['physics_batch_size']=args.batch
    work=args.workdir.resolve()
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        p.error(f'Use a new empty workdir (no files are deleted): {work}')
    work.mkdir(parents=True,exist_ok=True)
    resolved=work/'launcher_config.yaml';resolved.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    command=[sys.executable,str(root/'python/run_nsga2_electrical_solid_loop.py'),'--package-root',str(root),
             '--config',str(resolved),'--workdir',str(work)]
    if args.population is not None:command+=['--population-size',str(args.population)]
    if args.generations is not None:command+=['--generations',str(args.generations)]
    if args.mode=='debug' or args.allow_no_feasible:command+=['--allow-no-feasible']
    env=os.environ.copy();env['PYTHONPATH']=str(root/'python')+os.pathsep+env.get('PYTHONPATH','')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MAX_JOBS'):env.setdefault(key,'1')
    env.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
    print(json.dumps({'mode':args.mode,'requested_gpu_batch':args.batch if args.mode in ('cuda','hybrid') else None,
                      'configuration':str(resolved)},indent=2),flush=True)
    return subprocess.run(command,cwd=root,env=env).returncode
if __name__=='__main__':raise SystemExit(main())
