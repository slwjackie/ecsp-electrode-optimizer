from __future__ import annotations
import hashlib
import os
from pathlib import Path
import platform
import re
import sys
from functools import lru_cache
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / 'cpp' / 'preflame_cpp_cuda'

def build_root() -> Path:
    """Resolve the C++/CUDA build cache, accepting the old environment override."""
    return Path(os.environ.get(
        'ECSP_CPP_CUDA_BUILD_ROOT',
        os.environ.get('ECSP_NATIVE_BUILD_ROOT', str(ROOT / '.cpp_cuda_build')),
    ))

def required_cpp_standard(torch_version: str | None = None) -> str:
    """Return the translation-unit standard required by the PyTorch headers.

    The solver sources retain a C++17 language floor.  PyTorch 2.14 and newer
    require consumers to compile as C++20, so the build standard follows the
    linked LibTorch rather than silently pinning an incompatible flag.
    """
    version=str(torch.__version__ if torch_version is None else torch_version)
    numbers=re.match(r'\s*(\d+)\.(\d+)',version)
    if numbers is None:
        raise RuntimeError(f'Cannot determine C++ standard for PyTorch version {version!r}')
    major,minor=(int(numbers.group(1)),int(numbers.group(2)))
    return 'c++20' if (major,minor)>=(2,14) else 'c++17'

def build_info(with_cuda: bool = False) -> dict:
    paths = sorted(SOURCES.glob('*'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    standard=required_cpp_standard()
    context = (torch.__version__, torch.version.cuda, platform.machine(), sys.implementation.cache_tag,
               standard, with_cuda, os.environ.get('TORCH_CUDA_ARCH_LIST', '8.0'), os.environ.get('CXX', 'c++'))
    digest.update(repr(context).encode())
    name = 'preflame_cpp_cuda_' + digest.hexdigest()[:16]
    return {'name': name, 'source_sha256': digest.hexdigest(), 'torch': torch.__version__,
            'torch_cuda': torch.version.cuda, 'architecture': platform.machine(), 'with_cuda': with_cuda,
            'compiler_language_standard':standard,'core_source_language_floor':'c++17',
            'build_directory': str(build_root() / name), 'source_directory': str(SOURCES)}

@lru_cache(maxsize=2)
def load_cpp_cuda(with_cuda: bool = False, verbose: bool = False):
    """Build once on the target architecture. Never silently substitute Python/FP32."""
    from torch.utils.cpp_extension import load, CUDA_HOME
    if with_cuda and (CUDA_HOME is None or torch.version.cuda is None):
        raise RuntimeError('The compiled CUDA backend needs CUDA-enabled PyTorch AND a CUDA toolkit (nvcc). '
                           'Use the compiled CPU profile or the preflame_torch backend explicitly.')
    info = build_info(with_cuda); folder = Path(info['build_directory']); folder.mkdir(parents=True, exist_ok=True)
    sources = [SOURCES / n for n in ('bindings.cpp','preflame_engine.cpp','preflame_ops_cpu.cpp')]
    # PyTorch >=2.14 headers require C++20.  Earlier supported releases build
    # the same C++17-compatible core as C++17.  CPU and CUDA host code always
    # use the same selected standard so cached artifacts cannot mix ABIs.
    standard=required_cpp_standard();flags = ['-O3',f'-std={standard}','-ffp-contract=off']
    if with_cuda:
        sources.append(SOURCES/'preflame_ops_cuda.cu'); flags += ['-DWITH_CUDA']
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '1')
    return load(name=info['name'], sources=[str(p) for p in sources], extra_cflags=flags,
                extra_cuda_cflags=['-O3',f'-std={standard}','-DWITH_CUDA','--fmad=false'],
                with_cuda=with_cuda, build_directory=str(folder), verbose=verbose)
