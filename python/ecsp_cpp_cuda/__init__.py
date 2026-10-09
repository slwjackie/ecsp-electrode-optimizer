"""Compiled electrochemical-thermal-decomposition FP64 backend (CPU and optional CUDA); legacy solvers untouched."""
from .loader import load_cpp_cuda, build_info
from .adapter import run_cpp_cuda_batch
__all__ = ["load_cpp_cuda", "build_info", "run_cpp_cuda_batch"]
