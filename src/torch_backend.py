"""Helpers for configuring torch backend defaults across fal apps."""

from __future__ import annotations

from typing import Any, Dict, Optional


def configure_tf32_backends(torch_module: Any, *, logger: Optional[Any] = None) -> Dict[str, bool]:
    """Enable TF32 execution using the modern precision knobs when available.

    Returns a dict with flags describing which API surface was exercised so
    callers can log or branch if necessary. Falls back to the legacy
    ``allow_tf32`` toggles when running against older torch releases.
    """

    state = {"new_api": False, "legacy_api": False}

    def _debug(msg: str) -> None:
        if logger is not None:
            logger.debug(msg)

    # Prefer the PyTorch 2.9+ precision controls.
    try:
        matmul = getattr(getattr(torch_module.backends, "cuda", None), "matmul", None)
        if matmul is not None and hasattr(matmul, "fp32_precision"):
            matmul.fp32_precision = "tf32"
            state["new_api"] = True
            _debug("Configured torch.backends.cuda.matmul.fp32_precision = 'tf32'")
    except Exception:
        _debug("Failed to configure torch.backends.cuda.matmul.fp32_precision")

    try:
        cudnn_conv = getattr(getattr(torch_module.backends, "cudnn", None), "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = "tf32"
            state["new_api"] = True
            _debug("Configured torch.backends.cudnn.conv.fp32_precision = 'tf32'")
    except Exception:
        _debug("Failed to configure torch.backends.cudnn.conv.fp32_precision")

    if state["new_api"]:
        return state

    # Fallback for torch builds that still rely on the legacy switches.
    try:
        matmul = getattr(getattr(torch_module.backends, "cuda", None), "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = True
            state["legacy_api"] = True
            _debug("Configured torch.backends.cuda.matmul.allow_tf32 = True")
    except Exception:
        _debug("Failed to configure torch.backends.cuda.matmul.allow_tf32")

    try:
        cudnn = getattr(torch_module.backends, "cudnn", None)
        if cudnn is not None and hasattr(cudnn, "allow_tf32"):
            cudnn.allow_tf32 = True
            state["legacy_api"] = True
            _debug("Configured torch.backends.cudnn.allow_tf32 = True")
    except Exception:
        _debug("Failed to configure torch.backends.cudnn.allow_tf32")

    return state
