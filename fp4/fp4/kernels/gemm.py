"""GEMM backend dispatcher.

Tries to import a compiled CUTLASS Blackwell NVFP4 GEMM extension; otherwise
falls back to torch.matmul (which works on CPU/Hopper/Ada too).
"""
from __future__ import annotations

import torch

try:
    # Will exist once kernels/gemm_kernel.cu is built into a torch extension.
    from . import _nvfp4_gemm_ext  # type: ignore
    _HAS_CUTLASS = True
except Exception:
    _nvfp4_gemm_ext = None
    _HAS_CUTLASS = False


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute a @ b. On Blackwell with the compiled extension this dispatches
    to the CUTLASS NVFP4xNVFP4 kernel; otherwise it uses torch.matmul on the
    already-dequantized emulation tensors.
    """
    if _HAS_CUTLASS and a.is_cuda and b.is_cuda:
        try:
            return _nvfp4_gemm_ext.matmul(a, b)
        except Exception:
            pass
    return torch.matmul(a, b)
