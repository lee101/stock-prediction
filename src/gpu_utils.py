"""Utility helpers for GPU memory aware configuration."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple, Union


try:  # torch is optional for some CPU-bound utilities.
    import torch
except ImportError:  # pragma: no cover - torch not installed in some contexts
    torch = None  # type: ignore

try:  # Prefer pynvml if available for multi-GPU insights.
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore


Gigabytes = float


def _split_visible_devices(env_value: str) -> Sequence[str]:
    """Return sanitized tokens from CUDA_VISIBLE_DEVICES."""

    return [token.strip() for token in env_value.split(",") if token.strip()]


def _token_is_int(token: str) -> bool:
    """Return True when the token represents a non-negative integer index."""

    return token.isdigit()


def _normalize_for_torch(
    device_override: Optional[str],
    visible_tokens: Sequence[str],
) -> Optional[str]:
    """Convert a device specification into something torch.device accepts."""

    spec = (device_override or "").strip()

    if not spec:
        if visible_tokens:
            return "cuda:0"
        return "cuda"

    lowered = spec.lower()

    if lowered == "cpu":
        return None

    if lowered == "cuda":
        return "cuda"

    if lowered.startswith("cuda:"):
        index_part = lowered.split(":", 1)[1]
        if _token_is_int(index_part) and visible_tokens:
            visible_index = int(index_part)
            if visible_index < len(visible_tokens):
                return f"cuda:{visible_index}"
        return spec

    if "," in spec:
        return _normalize_for_torch(spec.split(",", 1)[0], visible_tokens)

    if _token_is_int(spec):
        if visible_tokens:
            try:
                visible_index = visible_tokens.index(spec)
            except ValueError:
                return f"cuda:{spec}"
            else:
                return f"cuda:{visible_index}"
        return f"cuda:{spec}"

    if lowered.startswith("gpu"):
        suffix = spec[3:]
        if _token_is_int(suffix):
            return _normalize_for_torch(suffix, visible_tokens)

    return spec


def _select_nvml_target(
    device_override: Optional[str],
    visible_tokens: Sequence[str],
) -> Optional[Union[int, str]]:
    """Select the NVML target (index or PCI bus id) honoring CUDA visibility."""

    def pick_from_token(token: str) -> Optional[Union[int, str]]:
        token = token.strip()
        if not token:
            return None
        if _token_is_int(token):
            return int(token)
        return token

    spec = (device_override or "").strip()

    if spec:
        lowered = spec.lower()

        if lowered == "cpu":
            return None

        if lowered.startswith("cuda:"):
            index_part = lowered.split(":", 1)[1]
            if _token_is_int(index_part):
                visible_index = int(index_part)
                if visible_tokens and 0 <= visible_index < len(visible_tokens):
                    return pick_from_token(visible_tokens[visible_index])
                return int(index_part)
            return None

        if "," in spec:
            return _select_nvml_target(spec.split(",", 1)[0], visible_tokens)

        if _token_is_int(spec):
            if visible_tokens and spec in visible_tokens:
                return pick_from_token(spec)
            return int(spec)

        if lowered.startswith("gpu"):
            suffix = spec[3:]
            return _select_nvml_target(suffix, visible_tokens)

        return pick_from_token(spec)

    if visible_tokens:
        return pick_from_token(visible_tokens[0])

    return 0


def _nvml_get_handle(target: Union[int, str]) -> "pynvml.c_nvmlDevice_t":
    """Obtain an NVML handle for the desired device target."""

    if isinstance(target, str):
        if _token_is_int(target):
            return pynvml.nvmlDeviceGetHandleByIndex(int(target))
        return pynvml.nvmlDeviceGetHandleByPciBusId(target)
    return pynvml.nvmlDeviceGetHandleByIndex(int(target))


def detect_total_vram_bytes(device: Optional[str] = None) -> Optional[int]:
    """Return total VRAM (in bytes) for the current or requested CUDA device.

    Falls back to NVML if torch is unavailable or no CUDA context is active.
    Returns ``None`` when no GPU information can be gathered.
    """

    visible_tokens = _split_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    torch_device_spec = _normalize_for_torch(device, visible_tokens)
    nvml_target = _select_nvml_target(device, visible_tokens)

    if torch is not None and torch.cuda.is_available():
        try:
            if torch_device_spec is None:
                return None
            cuda_device = torch.device(torch_device_spec)
            props = torch.cuda.get_device_properties(cuda_device)
            return int(props.total_memory)
        except Exception:
            pass

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            if nvml_target is None:
                return None
            handle = _nvml_get_handle(nvml_target)
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
