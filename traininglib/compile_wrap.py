from __future__ import annotations

import logging
import torch


def maybe_compile(module: torch.nn.Module, do_compile: bool = True, mode: str = "max-autotune"):
    """
    Wrap torch.compile with graceful fallback when unsupported.
    """
    if not do_compile:
        return module

    if not hasattr(torch, "compile"):
        logging.warning("torch.compile not available in this PyTorch build.")
        return module

    try:
        return torch.compile(module, mode=mode)
    except Exception as exc:  # pragma: no cover - safety net
        logging.warning("torch.compile disabled due to: %s", exc)
        return module
