"""Compiled B/C FP64 backend (CPU and optional CUDA); legacy solvers untouched."""
from .loader import load_native, build_info
from .adapter import run_native_batch
__all__ = ["load_native", "build_info", "run_native_batch"]
