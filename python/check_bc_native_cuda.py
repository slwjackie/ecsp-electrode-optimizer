#!/usr/bin/env python3
"""Report CUDA source/build readiness, WITHOUT pretending a static audit is a build."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import CUDA_HOME
from ecsp_native.loader import SOURCES,load_native

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-compile',action='store_true');args=p.parse_args()
    cu=(SOURCES/'bc_ops_cuda.cu').read_text();scalar=(SOURCES/'bc_scalar.h').read_text()
    checks={
      'shared_fp64_scalar_arithmetic':'BC_HD' in scalar and 'double' in scalar,
      'all_five_cuda_entrypoints':all(('Tensor '+x+'_cuda' in cu or 'vector<Tensor> '+x+'_cuda' in cu) for x in ('props','advance','contacts','current','stencil')),
      'device_guard':cu.count('CUDAGuard')>=5,
      'current_stream':cu.count('getCurrentCUDAStream')>=5,
      'launch_checks':cu.count('C10_CUDA_KERNEL_LAUNCH_CHECK')>=5,
      'kernel_parameter_size_assert':'sizeof(Params)<4096' in scalar,
      'no_fast_math':'--use_fast_math' not in (SOURCES.parent.parent/'python/ecsp_native/loader.py').read_text(),
    }
    nvcc=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin/nvcc') if CUDA_HOME and (Path(CUDA_HOME)/'bin/nvcc').is_file() else None)
    report={'source_static_checks':checks,'all_source_static_checks_passed':all(checks.values()),
            'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,'cuda_home':CUDA_HOME,
            'nvcc':nvcc,'cuda_device_available':torch.cuda.is_available(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES.glob('*')) if p.is_file()},
            'target_architecture':os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0'),
            'compile_status':'not_attempted_missing_cuda_toolchain_or_cuda_pytorch',
            'runtime_status':'not_tested_by_this_script','static_checks_are_not_cuda_compilation':True}
    code=0 if all(checks.values()) else 1
    if nvcc and torch.version.cuda and CUDA_HOME:
        try:
            report['nvcc_version']=subprocess.check_output([nvcc,'--version'],text=True)
            ext=load_native(True,True);report['compile_status']='compiled_and_loaded'
            report['extension_has_cuda']=bool(ext.has_cuda)
            if not report['extension_has_cuda']:
                raise RuntimeError(
                    'Native extension compiled and loaded without CUDA entrypoints'
                )
        except Exception as exc:
            report['compile_status']='failed';report['compile_error']=str(exc);code=1
    elif args.require_compile:code=2
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return code
if __name__=='__main__':raise SystemExit(main())
