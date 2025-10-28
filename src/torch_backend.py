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

    def _sync_legacy_toggles(enabled: bool) -> None:
        """
        Mirror TF32 state to the legacy allow_tf32 toggles so downstream code
        that still probes them (e.g. torch.compile/inductor checks) does not
        raise runtime errors. Prefer the internal setters to avoid triggering
        getter warnings during assignment.
        """
        try:
            set_cublas = getattr(getattr(torch_module, "_C", None), "_set_cublas_allow_tf32", None)
            set_cudnn = getattr(getattr(torch_module, "_C", None), "_set_cudnn_allow_tf32", None)
            if callable(set_cublas):
                set_cublas(enabled)
            if callable(set_cudnn):
                set_cudnn(enabled)
            state["legacy_api"] = state["legacy_api"] or callable(set_cublas) or callable(set_cudnn)
        except Exception:
            _debug("Internal TF32 legacy setters unavailable; falling back to public attributes")
            matmul = getattr(getattr(torch_module.backends, "cuda", None), "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                try:
                    matmul.allow_tf32 = enabled
                    state["legacy_api"] = True
                except Exception:
                    _debug("Failed to mirror torch.backends.cuda.matmul.allow_tf32")
            cudnn_backend = getattr(torch_module.backends, "cudnn", None)
            if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
                try:
                    cudnn_backend.allow_tf32 = enabled
                    state["legacy_api"] = True
                except Exception:
                    _debug("Failed to mirror torch.backends.cudnn.allow_tf32")

    # Prefer the PyTorch 2.9+ precision controls.
    try:
        matmul = getattr(getattr(torch_module.backends, "cuda", None), "matmul", None)
        if matmul is not None and hasattr(matmul, "fp32_precision"):
            matmul.fp32_precision = "tf32"
            state["new_api"] = True
            _debug("Configured torch.backends.cuda.matmul.fp32_precision = 'tf32'")
            _sync_legacy_toggles(True)
    except Exception:
        _debug("Failed to configure torch.backends.cuda.matmul.fp32_precision")

    try:
        cudnn_conv = getattr(getattr(torch_module.backends, "cudnn", None), "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = "tf32"
            state["new_api"] = True
            _debug("Configured torch.backends.cudnn.conv.fp32_precision = 'tf32'")
            _sync_legacy_toggles(True)
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
