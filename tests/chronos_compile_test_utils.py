"""Shared helpers for Chronos torch.compile test stability."""

from __future__ import annotations

import torch


def reset_torch_compile_state() -> None:
    """Clear process-global torch.compile state between Chronos wrappers."""

    reset = getattr(getattr(torch, "compiler", None), "reset", None)
    if callable(reset):
        try:
            reset()
            return
        except Exception:
            pass
    legacy_reset = getattr(getattr(torch, "_dynamo", None), "reset", None)
    if callable(legacy_reset):
        try:
            legacy_reset()
        except Exception:
            pass


def clear_cuda_memory_if_available() -> None:
    """Release CUDA cache best-effort when the process has a GPU."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_transient_nonfinite_forecast_error(exc: BaseException) -> bool:
    """Return whether a Chronos runtime error only reflects transient non-finite output."""

    return isinstance(exc, RuntimeError) and "non-finite forecasts" in str(exc)
