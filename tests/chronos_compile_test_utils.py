"""Shared test utilities for Chronos2 compile fuzzing and accuracy tests."""

from __future__ import annotations


try:
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]


def clear_cuda_memory_if_available() -> None:
    """Free CUDA cache if torch + CUDA are available; no-op otherwise."""
    if _torch is None:
        return
    try:
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def is_transient_nonfinite_forecast_error(exc: BaseException) -> bool:
    """Return *True* when *exc* looks like a retryable non-finite forecast error."""
    if not isinstance(exc, (RuntimeError, ValueError)):
        return False
    msg = str(exc).lower()
    return "non-finite" in msg or "nonfinite" in msg or ("nan" in msg and "forecast" in msg)


def reset_torch_compile_state() -> None:
    """Best-effort reset of ``torch.compile`` / ``torch._dynamo`` caches."""
    if _torch is None:
        return

    # Prefer the modern ``torch.compiler.reset()`` API.
    reset = getattr(getattr(_torch, "compiler", None), "reset", None)
    if callable(reset):
        try:
            reset()
            return
        except Exception:  # pragma: no cover
            pass

    # Fall back to the legacy ``torch._dynamo.reset()``.
    legacy_reset = getattr(getattr(_torch, "_dynamo", None), "reset", None)
    if callable(legacy_reset):
        try:
            legacy_reset()
        except Exception:  # pragma: no cover
            pass


__all__ = [
    "clear_cuda_memory_if_available",
    "is_transient_nonfinite_forecast_error",
    "reset_torch_compile_state",
]
