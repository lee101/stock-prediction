from __future__ import annotations

import logging
import warnings

import torch
from src.torch_device_utils import resolve_runtime_device, should_auto_fallback_to_cpu

logger = logging.getLogger(__name__)


def resolve_device(preferred: str | None = None) -> torch.device:
    return resolve_runtime_device(preferred)


def should_fallback_to_cpu(
    preferred: str | None,
    device: torch.device,
    exc: BaseException,
) -> bool:
    return should_auto_fallback_to_cpu(preferred, device, exc)


def warn_cuda_fallback(context: str, exc: BaseException) -> None:
    message = f"{context}: auto-selected CUDA unavailable at runtime, falling back to CPU: {exc}"
    warnings.warn(message, RuntimeWarning)
    logger.warning(message)


__all__ = ["resolve_device", "should_fallback_to_cpu", "warn_cuda_fallback"]
