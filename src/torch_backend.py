"""Helpers for configuring torch backend defaults across fal apps."""

from __future__ import annotations

from typing import Any, Dict, Optional


def configure_tf32_backends(torch_module: Any, *, logger: Optional[Any] = None) -> Dict[str, bool]:
    """Enable TF32 execution in a way that stays compatible with torch.compile.

    We prefer the legacy ``allow_tf32`` toggles when they exist. On PyTorch 2.9,
    setting the new ``fp32_precision`` knobs and later reading
    ``torch.backends.cuda.matmul.allow_tf32`` raises a RuntimeError, and
    torch.compile / Inductor still reads the legacy flag internally. As a
    result, using the "new" API unconditionally can break compilation.

    Returns a dict with flags describing which API surface was exercised so
    callers can log or branch if necessary. Falls back to the modern
    ``fp32_precision`` knobs only when legacy toggles are unavailable.
    """

    state = {"new_api": False, "legacy_api": False}

    def _debug(msg: str) -> None:
        if logger is not None:
            logger.debug(msg)

    cuda_backend = getattr(torch_module.backends, "cuda", None)
    cudnn_backend = getattr(torch_module.backends, "cudnn", None)

    cuda_available = True
    try:
        cuda_module = getattr(torch_module, "cuda", None)
        is_available = getattr(cuda_module, "is_available", None)
        if callable(is_available):
            cuda_available = bool(is_available())
    except Exception:
        cuda_available = True

    if cuda_available:
        legacy_configured = False

        # Prefer legacy toggles when available (see docstring).
        try:
            matmul = getattr(cuda_backend, "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                matmul.allow_tf32 = True
                state["legacy_api"] = True
                legacy_configured = True
                _debug("Configured torch.backends.cuda.matmul.allow_tf32 = True")
        except Exception:
            _debug("Failed to configure torch.backends.cuda.matmul.allow_tf32")

        try:
            cudnn = cudnn_backend
            if cudnn is not None and hasattr(cudnn, "allow_tf32"):
                cudnn.allow_tf32 = True
                state["legacy_api"] = True
                legacy_configured = True
                _debug("Configured torch.backends.cudnn.allow_tf32 = True")
        except Exception:
            _debug("Failed to configure torch.backends.cudnn.allow_tf32")

        if legacy_configured:
            return state

        # Fallback to the newer precision controls only when the legacy toggles
        # are absent on this torch build.
        try:
            matmul = getattr(cuda_backend, "matmul", None)
            if matmul is not None and hasattr(matmul, "fp32_precision"):
                matmul.fp32_precision = "tf32"
                state["new_api"] = True
                _debug("Configured torch.backends.cuda.matmul.fp32_precision = 'tf32'")
        except Exception:
            _debug("Failed to configure torch.backends.cuda.matmul.fp32_precision")

        try:
            cudnn_conv = getattr(getattr(cuda_backend, "cudnn", None), "conv", None)
        except Exception:
            cudnn_conv = None
        if cudnn_conv is None and cudnn_backend is not None:
            cudnn_conv = getattr(cudnn_backend, "conv", None)
        try:
            if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
                cudnn_conv.fp32_precision = "tf32"
                state["new_api"] = True
                _debug("Configured torch.backends.cudnn.conv.fp32_precision = 'tf32'")
        except Exception:
            _debug("Failed to configure torch.backends.cudnn.conv.fp32_precision")

    return state


def maybe_set_float32_precision(torch_module: Any, mode: str = "high") -> None:
    """Invoke ``torch.set_float32_matmul_precision`` only when legacy knobs are required.

    PyTorch 2.9 introduces the ``fp32_precision`` interface on backend objects and
    simultaneously emits deprecation warnings when the older global setter is
    used. To remain quiet on newer builds we only call the legacy setter when the
    modern surface is unavailable.
    """

    try:
        cuda_backend = getattr(torch_module.backends, "cuda", None)
        matmul = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
        if matmul is not None and hasattr(matmul, "fp32_precision"):
            return
        is_available = getattr(torch_module.cuda, "is_available", None)
        if callable(is_available) and not is_available():
            return
    except Exception:
        return

    set_precision = getattr(torch_module, "set_float32_matmul_precision", None)
    if not callable(set_precision):  # pragma: no cover - legacy guard
        return
    try:
        set_precision(mode)
    except Exception:
        pass
