"""Lightweight wrappers around torch.profiler with graceful CPU fallback."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Iterable, Optional

try:
    import torch
    from torch.profiler import (
        ProfilerActivity,
        profile,
        schedule,
        tensorboard_trace_handler,
    )
except Exception:  # pragma: no cover - torch profiler may be unavailable on CPU-only builds
    profile = None  # type: ignore[assignment]


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def maybe_profile(
    enabled: bool,
    logdir: str | Path = "runs/prof",
    *,
    wait: int = 2,
    warmup: int = 2,
    active: int = 6,
) -> ContextManager[None]:
    """
    Optionally wrap a block with ``torch.profiler.profile``.

    Parameters
    ----------
    enabled:
        If ``False`` or profiler support is unavailable, returns a ``nullcontext``.
    logdir:
        Directory where TensorBoard traces should be written.
    wait, warmup, active:
        Scheduling knobs forwarded to ``torch.profiler.schedule``.
    """

    if not enabled or profile is None:
        return nullcontext()

    activities: Iterable[ProfilerActivity]
    if torch.cuda.is_available():
        activities = (ProfilerActivity.CPU, ProfilerActivity.CUDA)
    else:
        activities = (ProfilerActivity.CPU,)

    log_path = _ensure_dir(logdir)
    return profile(  # type: ignore[return-value]
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=tensorboard_trace_handler(str(log_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


__all__ = ["maybe_profile"]
