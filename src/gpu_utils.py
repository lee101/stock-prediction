"""Utility helpers for GPU memory aware configuration."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple


try:  # torch is optional for some CPU-bound utilities.
    import torch
except ImportError:  # pragma: no cover - torch not installed in some contexts
    torch = None  # type: ignore

try:  # Prefer pynvml if available for multi-GPU insights.
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore


Gigabytes = float


def detect_total_vram_bytes(device: Optional[str] = None) -> Optional[int]:
    """Return total VRAM (in bytes) for the current or requested CUDA device.

    Falls back to NVML if torch is unavailable or no CUDA context is active.
    Returns ``None`` when no GPU information can be gathered.
    """

    device = device or os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if torch is not None and torch.cuda.is_available():
        try:
            if device:
                cuda_device = torch.device(device)
            else:
                cuda_device = torch.device("cuda")
            props = torch.cuda.get_device_properties(cuda_device)
            return int(props.total_memory)
        except Exception:
            pass

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.total)
        except Exception:
            return None
        finally:  # pragma: no branch - NVML always needs shutdown
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    return None


def recommend_batch_size(
    total_vram_bytes: Optional[int],
    default_batch_size: int,
    thresholds: Sequence[Tuple[Gigabytes, int]],
    *,
    allow_increase: bool = True,
) -> int:
    """Pick a batch size based on available VRAM thresholds.

    Args:
        total_vram_bytes: Detected VRAM in bytes, or ``None`` if unknown.
        default_batch_size: Caller provided batch size.
        thresholds: Pairs of ``(vram_gb, batch_size)`` sorted ascending.
        allow_increase: When ``False`` the result will never exceed the
            provided ``default_batch_size``.

    Returns:
        An integer batch size that respects the threshold mapping.
    """

    if total_vram_bytes is None:
        return default_batch_size

    total_vram_gb = total_vram_bytes / (1024 ** 3)
    chosen = thresholds[0][1] if thresholds else default_batch_size
    for vram_gb, batch_size in thresholds:
        if total_vram_gb >= vram_gb:
            chosen = batch_size
        else:
            break

    if not allow_increase and chosen > default_batch_size:
        return default_batch_size
    return chosen


def cli_flag_was_provided(flag_name: str, argv: Optional[Iterable[str]] = None) -> bool:
    """Return True if the given CLI flag appears in argv.

    Simple helper used to distinguish between defaults and user overrides.
    Supports ``--flag=value`` forms. ``argv`` defaults to ``sys.argv[1:]``.
    """

    import sys

    search_space = list(argv) if argv is not None else sys.argv[1:]
    flag_prefix = f"{flag_name}="
    for item in search_space:
        if item == flag_name or item.startswith(flag_prefix):
            return True
    return False
