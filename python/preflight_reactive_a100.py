#!/usr/bin/env python3
"""Fresh A100 proof for the native BC + tensor Reactive + spawned CPU pipeline.

Always fails closed; no skipped CUDA test is accepted and no CPU substitution.
The report is functional evidence, not a production speed or physical validation.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
NODES=(
 'python/tests/test_reactive_tensor_backend.py::test_cuda_fp64_stationary_dynamic_and_batch_parity',
 'python/tests/test_reactive_tensor_backend.py::test_cuda_graph_matches_eager_with_changing_timestep',
 'python/tests/test_reactive_tensor_backend.py::test_tensor_NSGA_workflow_really_dispatches_batch[cuda]',
)

def source_fingerprint(root=ROOT):
    entries={}
    for directory in ('python','cpp','config','tools'):
        for p in (root/directory).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.hpp','.h','.cpp','.cc','.cu','.cuh','.yaml','.sh','.txt'} and '__pycache__' not in p.parts:
                entries[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    entries['VERSION.json']=hashlib.sha256((root/'VERSION.json').read_bytes()).hexdigest()
    digest=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return digest,entries


def require_gpu_tests(path):
    root=ET.parse(path).getroot();cases=list(root.iter('testcase'))
    expected={n.split('::')[-1] for n in NODES}
    observed={c.attrib.get('name') for c in cases}
    bad=[c.attrib.get('name') for c in cases if any(c.find(k) is not None for k in ('failure','error','skipped'))]
    if len(cases)!=len(NODES) or observed!=expected or bad:
        raise RuntimeError(f'Required CUDA tests did not all actually pass: observed={observed}; bad={bad}')
    return sorted(observed)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists() or out.is_symlink():raise SystemExit('Preflight output already exists; choose a fresh path')
    out.mkdir(parents=True)
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'python')+os.pathsep+env.get('PYTHONPATH','')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','MAX_JOBS'):env[key]='1'
    env['TORCH_CUDA_ARCH_LIST']='8.0'
    from a100_device_gate import collect_device_report
    report,code=collect_device_report(allow_non_a100=False)
    (out/'device.json').write_text(json.dumps(report,indent=2)+'\n')
    if code:raise SystemExit('A100 preflight refused: '+str(report.get('failure_reason')))
    initial_digest,entries=source_fingerprint()
    def run(command,name):
        with (out/(name+'.log')).open('w') as f:
            subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    run(['bash',str(ROOT/'tools/run_bc_native_a100_preflight.sh'),str(out/'native_bc')],'native_bc')
    run([sys.executable,'-m','pytest','-q',*NODES,'--junitxml='+str(out/'reactive_cuda.xml')],'reactive_cuda')
    tests=require_gpu_tests(out/'reactive_cuda.xml')
    # Native BC and Reactive share the same GPU, at different pipeline phases.
    import yaml
    cfg=yaml.safe_load((ROOT/'config/nsga2_bc_reactive_a100_cpu8_debug.yaml').read_text())
    native=yaml.safe_load((ROOT/'config/nsga2_bc_reactive_native_debug.yaml').read_text())['evaluator']
    cfg['evaluator']=native
    cfg['evaluator'].update(backend='bc_global_native_hybrid',device='cuda',internal_batch_size=2)
    cfg['project']['device']='cuda';cfg['project']['name']='ECSP_v8_4_1_nativeBC_tensorCUDA_CPUbaseline_preflight'
    cfg['evaluator']['base_overrides']['numerics']['physicsDevice']='cuda'
    cfg['evaluator']['native'].update(cpu_budget=8,host_reserve=2,cpu_workers=2,cpu_threads_per_worker=1,cpu_runtime='standalone')
    cfgpath=out/'native_reactive_smoke.yaml';cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False))
    work=out/'native_reactive_smoke'
    run([sys.executable,str(ROOT/'python/run_nsga2_electrical_solid_loop.py'),'--package-root',str(ROOT),'--config',str(cfgpath),'--workdir',str(work),'--allow-no-feasible'],'native_reactive_smoke')
    if not (work/'RUN_COMPLETE.json').is_file():raise RuntimeError('Missing full-pipeline completion')
    paths=list((work/'final/propagation_candidates').glob('*/reactive_euler/propagation_metrics.json'))
    metrics=[json.loads(p.read_text()) for p in paths]
    if not metrics or any(m.get('computeBackend')!='torch_cuda_batch' or m.get('status')!='complete' for m in metrics):
        raise RuntimeError('Full pipeline did not produce completed CUDA Reactive results')
    pairs=[json.loads(p.read_text()) for p in (work/'final/propagation_candidates').glob('*/backend_comparison.json')]
    if not pairs or not all(p.get('bothBackendsCompleted') for p in pairs):
        raise RuntimeError('Full pipeline did not complete paired CPU baseline comparisons')
    final_digest,_=source_fingerprint()
    if final_digest!=initial_digest:raise RuntimeError('Source changed during preflight')
    summary=dict(schema='ecsp.reactive-a100-preflight/v1',status='passed',
        created_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=final_digest,
        bound_source_files=entries,device=report,required_cuda_tests=tests,
        native_BC_preflight=True,real_BC_tensor_CUDA_CPU_baseline_smoke_cases=len(metrics),
        physical_material_validation=False,production_throughput_benchmarked=False,
        graph_replay_tested=True)
    (out/'PREFLIGHT_COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'bound_source_files','device'}},indent=2))

if __name__=='__main__':main()
