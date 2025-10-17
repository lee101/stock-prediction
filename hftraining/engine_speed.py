"""
Utilities for enabling the fastest Torch execution settings used across the
Hugging Face training scripts in this project.

The helpers here wrap the lower-level knobs (TF32 matmuls, Flash/SDPA kernels,
torch.compile with CUDA Graphs) in safe, opt-in primitives that degrade
gracefully when running on unsupported hardware or PyTorch versions.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from typing import Iterator, Optional

import torch

from traininglib.runtime_flags import enable_fast_kernels  # Reuse existing guardrails.


def enable_tf32() -> None:
    """
    Allow TF32 matmul/cudnn paths on Ampere/Hopper GPUs.
    """
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        # Non-fatal; TF32 simply remains disabled on this platform.
        pass


def prefer_flash_sdp(math_fallback: bool = False) -> None:
    """
    Toggle PyTorch's scaled-dot product attention backend preferences so that
    FlashAttention/efficient kernels are used whenever possible.
    """
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(math_fallback)
    except Exception:
        # Older PyTorch builds expose a context manager instead; nothing to do.
        pass


def compile_model(
    model: torch.nn.Module,
    *,
    mode: str = "max-autotune",
    dynamic: bool = True,
) -> torch.nn.Module:
    """
    Attempt to wrap ``torch.compile`` around a module. Falls back to the input
    module unchanged if compilation is unavailable or fails at runtime.
    """
    if not hasattr(torch, "compile"):
        return model

    try:
        return torch.compile(model, mode=mode, dynamic=dynamic)
    except Exception:
        return model


@contextmanager
def fast_context() -> Iterator[None]:
    """
    Composite context manager that flips all relevant speed knobs.

    Usage::

        with fast_context():
            trainer.train()
    """
    with ExitStack() as stack:
        enable_tf32()
        prefer_flash_sdp()
        stack.enter_context(enable_fast_kernels())
        yield
