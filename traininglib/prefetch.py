"""Utilities to overlap host->device copies with compute."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Iterable

import torch


def _to_device(batch: Any, device: torch.device | str, *, non_blocking: bool) -> Any:
    """Recursively move supported containers to ``device``."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {k: _to_device(v, device, non_blocking=non_blocking) for k, v in batch.items()}
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        if hasattr(batch, "_fields"):  # NamedTuple (e.g., MaskedTimeseries)
            return type(batch)._make(_to_device(v, device, non_blocking=non_blocking) for v in batch)
        return type(batch)(_to_device(v, device, non_blocking=non_blocking) for v in batch)
    return batch


class CudaPrefetcher(Iterator):
    """
    Wrap a ``DataLoader`` to prefetch batches to GPU using a dedicated CUDA stream.
    Falls back to a no-op wrapper if CUDA is unavailable.
    """

    def __init__(self, loader: Iterable, device: torch.device | str = "cuda"):
        self.loader = loader
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested
        self.stream = torch.cuda.Stream() if (torch.cuda.is_available() and self.device.type == "cuda") else None
        self.next_batch: Any | None = None

    def __iter__(self) -> "CudaPrefetcher":
        if self.stream is None:
            self._it = iter(self.loader)
            return self

        self._it = iter(self.loader)
        self._preload()
        return self

    def __next__(self) -> Any:
        if self.stream is None:
            batch = next(self._it)
            return _to_device(batch, self.device, non_blocking=False)

        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self._preload()
        return batch

    def _preload(self) -> None:
        if self.stream is None:
            return

        try:
            next_batch = next(self._it)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = _to_device(next_batch, self.device, non_blocking=True)
