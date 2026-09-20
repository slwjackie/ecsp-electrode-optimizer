"""C++17-compatible FP64 worker, built to the linked LibTorch requirement."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import tempfile
import threading
import numpy as np
import torch
from filelock import FileLock
from .loader import ROOT, SOURCES, required_cpp_standard

def standalone_build_info() -> dict:
    digest=hashlib.sha256()
    for p in sorted(SOURCES.glob('*')):
        digest.update(p.name.encode());digest.update(p.read_bytes())
    cxx=os.environ.get('CXX','c++')
    standard=required_cpp_standard()
    digest.update(repr((torch.__version__,platform.machine(),platform.system(),cxx,
                       torch._C._GLIBCXX_USE_CXX11_ABI,standard,'standalone-protocol-v1')).encode())
    folder=Path(os.environ.get('ECSP_NATIVE_BUILD_ROOT',str(ROOT/'.native_build')))/('standalone_'+digest.hexdigest()[:16])
    return {'source_sha256':digest.hexdigest(),'build_directory':str(folder),
            'executable':str(folder/'ecsp_bc_native_cpu'),'compiler':cxx,
            'torch':torch.__version__,'compiler_language_standard':standard,
            'core_source_language_floor':'c++17','python_runtime_required_by_executable':False}

def build_standalone(verbose:bool=False) -> Path:
    from torch.utils.cpp_extension import include_paths,library_paths
    # Some sandboxed macOS/Linux workers deny OpenMP's shared-memory control
    # segment before main() is entered.  LibTorch does not require that segment
    # for this process-isolated solver; disabling it also makes direct `--version`
    # and malformed-input diagnostics reliable in restricted runtimes.
    os.environ.setdefault('KMP_USE_SHM','0')
    info=standalone_build_info();folder=Path(info['build_directory']);folder.mkdir(parents=True,exist_ok=True)
    binary=Path(info['executable'])
    with FileLock(str(folder/'compile.lock'),timeout=900):
        if binary.is_file():return binary
        compiler=shlex.split(info['compiler'])
        standard=required_cpp_standard()
        if not compiler or not shutil.which(compiler[0]):raise RuntimeError(f'{standard} compiler missing for native CPU solver')
        if platform.system()=='Windows':raise RuntimeError('Standalone launcher supports Linux/macOS; use extension CPU runtime on Windows')
        flags=['-O3',f'-std={standard}','-ffp-contract=off',f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}']
        flags += ['-I'+p for p in include_paths()]
        commands=[]
        for name in ('standalone.cpp','bc_engine.cpp','bc_ops_cpu.cpp'):
            obj=folder/(Path(name).stem+'.o');cmd=compiler+flags+['-c',str(SOURCES/name),'-o',str(obj)];commands.append(cmd)
        lib=library_paths();cmd=compiler+[str(folder/(Path(n).stem+'.o')) for n in ('standalone.cpp','bc_engine.cpp','bc_ops_cpu.cpp')]
        cmd += ['-L'+p for p in lib]+['-Wl,-rpath,'+p for p in lib]+['-ltorch','-ltorch_cpu','-lc10','-o',str(binary)]
        # Link to a temporary path; an interrupted build must not leave a cache hit.
        temporary=binary.with_suffix('.tmp');cmd[-1]=str(temporary)
        commands.append(cmd)
        with (folder/'build.log').open('w') as log:
            for command in commands:
                log.write(shlex.join(command)+'\n');log.flush()
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if proc.returncode:
                    binary.unlink(missing_ok=True)
                    raise RuntimeError(f'Native standalone build failed; see {folder/"build.log"}')
        os.replace(temporary,binary)
        if verbose:print(f'Built {binary}',flush=True)
    return binary

def _write_tensor(stream,tensor):
    t=tensor.detach().cpu().contiguous()
    types={torch.float64:(1,'<f8'),torch.int64:(2,'<i8'),torch.bool:(3,'u1')}
    if t.dtype not in types:raise TypeError(f'Unsupported native dtype: {t.dtype}')
    code,dtype=types[t.dtype];stream.write(struct.pack('<BB',code,t.ndim))
    for n in t.shape:stream.write(struct.pack('<q',n))
    stream.write(t.numpy().astype(dtype,copy=False).tobytes())

def _read_exact(stream,n):
    b=stream.read(n)
    if len(b)!=n:raise RuntimeError('Truncated standalone solver output')
    return b

def _read_tensor(stream):
    code,rank=struct.unpack('<BB',_read_exact(stream,2))
    if code not in (1,2,3) or rank>8:raise RuntimeError('Invalid standalone tensor metadata')
    dims=tuple(struct.unpack('<q',_read_exact(stream,8))[0] for _ in range(rank))
    n=1
    for d in dims:
        if d<0 or d>10**9:raise RuntimeError('Invalid standalone tensor shape')
        n*=d
    if n>10**9:raise RuntimeError('Oversized standalone response')
    dtype={1:np.dtype('<f8'),2:np.dtype('<i8'),3:np.dtype('?')}[code]
    data=np.frombuffer(_read_exact(stream,n*dtype.itemsize),dtype=dtype).reshape(dims).copy()
    return torch.from_numpy(data)

def run_standalone(anode,cathode,voltage,table,settings,*,threads=1,timeout_s=0.,temp_directory=None):
    """Fresh standalone process per trial; no simulator state crosses voltages."""
    binary=build_standalone()
    if any(t.device.type!='cpu' for t in (anode,cathode,voltage,table)):
        raise ValueError('Standalone CPU path cannot accept CUDA tensors')
    with tempfile.TemporaryDirectory(prefix='ecsp_bc_',dir=temp_directory) as directory:
        request=Path(directory)/'request.bin';response=Path(directory)/'response.bin'
        with request.open('wb') as out:
            out.write(b'BCNATV1I');out.write(struct.pack('<I',len(settings)))
            for key,value in sorted(settings.items()):
                name=key.encode('utf8');out.write(struct.pack('<I',len(name)));out.write(name);out.write(struct.pack('<d',float(value)))
            for t in (anode,cathode,voltage,table):_write_tensor(out,t)
        child_env=os.environ.copy()
        child_env['KMP_USE_SHM']='0'
        child_env['OMP_NUM_THREADS']=str(max(1,int(threads)))
        child_env['MKL_NUM_THREADS']=str(max(1,int(threads)))
        proc=subprocess.run([str(binary),str(request),str(response),str(max(1,int(threads)))],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                            env=child_env,
                            # Select posix_spawn on supported POSIX Python builds.
                            # Forking a process after PyTorch initialized OpenMP can
                            # leave the child unable to register its SHM runtime.
                            close_fds=False,
                            timeout=None if timeout_s<=0 else timeout_s)
        if proc.returncode:raise RuntimeError(f'Native CPU process failed ({proc.returncode}): {proc.stderr.strip()}')
        with response.open('rb') as stream:
            if _read_exact(stream,8)!=b'BCNATV1O':raise RuntimeError('Invalid standalone response magic/version')
            n,=struct.unpack('<I',_read_exact(stream,4))
            if n>64:raise RuntimeError('Unexpected standalone output count')
            result={}
            for _ in range(n):
                length,=struct.unpack('<I',_read_exact(stream,4))
                if length>4096:raise RuntimeError('Invalid standalone output key')
                name=_read_exact(stream,length).decode('utf8')
                if name in result:raise RuntimeError('Duplicate standalone result key')
                result[name]=_read_tensor(stream)
            if stream.read(1):raise RuntimeError('Trailing standalone output bytes')
        return result
