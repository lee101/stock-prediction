from __future__ import annotations

import contextlib
import warnings

import torch


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@contextlib.contextmanager
def enable_fast_kernels():
    """
    Context manager that enables useful CUDA fast paths (TF32 + Flash attention) when available.
    """
    # TF32 on Ampere/Hopper improves throughput without hurting accuracy much.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception as exc:
        warnings.warn(f"Unable to enable TF32 fast matmul: {exc}")

    if not torch.cuda.is_available():
        yield
        return

    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True,
        ):
            yield
    except Exception as exc:
        warnings.warn(f"Falling back to default SDP kernels: {exc}")
        yield
