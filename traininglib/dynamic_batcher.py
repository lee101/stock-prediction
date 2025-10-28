from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, List, Sequence, Tuple, TypeVar


@dataclass(frozen=True)
class WindowSpec:
    """
    Lightweight identifier for a single sliding window within a timeseries.

    The ``series_id`` references whichever internal structure a dataset uses to
    store individual sequences, while ``left`` marks the starting timestep of
    the context slice for that window.
    """

    series_id: int
    left: int


SampleT = TypeVar("SampleT")
BatchT = TypeVar("BatchT")

CollateFn = Callable[[Sequence[SampleT], int, int], BatchT]


class SupportsDynamicWindows(Generic[SampleT]):
    """
    Minimal protocol describing the dataset surface needed by :class:`WindowBatcher`.
    """

    def enumerate_window_specs(self, context: int, horizon: int, stride: int) -> Iterable[WindowSpec]:
        raise NotImplementedError

    def load_window(self, spec: WindowSpec, context: int, horizon: int) -> SampleT:
        raise NotImplementedError

    def collate_windows(self, samples: Sequence[SampleT], context: int, horizon: int) -> BatchT:
        raise NotImplementedError


@dataclass
class WindowBatch(Generic[BatchT]):
    """
    Container emitted by :class:`WindowBatcher` describing a mini-batch.
    """

    context: int
    horizon: int
    batch: BatchT
    size: int

    @property
    def batch_size(self) -> int:
        return self.size


class WindowBatcher(Generic[SampleT, BatchT]):
    """
    Generate near-constant token-count batches from variable length windows.

    The batcher groups windows by ``(context, horizon)`` buckets to keep tensor
    shapes static, computes a per-bucket micro-batch that respects the provided
    ``max_tokens_per_batch`` budget, and yields collated batches ready for GPU
    transfer.
    """

    def __init__(
        self,
        dataset: SupportsDynamicWindows[SampleT],
        *,
        max_tokens_per_batch: int,
        context_buckets: Sequence[int],
        horizon_buckets: Sequence[int],
        stride: int,
        collate_fn: CollateFn | None = None,
        shuffle: bool = True,
        pack_windows: bool = True,
    ) -> None:
        if max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be a positive integer.")
        if not context_buckets or not horizon_buckets:
            raise ValueError("context_buckets and horizon_buckets must be non-empty.")
        self.dataset = dataset
        self.max_tokens = max_tokens_per_batch
        self.context_buckets = tuple(sorted({int(c) for c in context_buckets if c > 0}))
        self.horizon_buckets = tuple(sorted({int(h) for h in horizon_buckets if h > 0}))
        if not self.context_buckets or not self.horizon_buckets:
            raise ValueError("Buckets must include at least one positive integer for context and horizon.")
        self.stride = max(1, int(stride))
        self.shuffle = shuffle
        self.pack_windows = pack_windows
        self._collate: CollateFn = collate_fn or getattr(dataset, "collate_windows")
        self._bins: Dict[Tuple[int, int], List[WindowSpec]] = {}
        for context in self.context_buckets:
            for horizon in self.horizon_buckets:
                specs = list(dataset.enumerate_window_specs(context, horizon, self.stride))
                if specs:
                    self._bins[(context, horizon)] = specs
        if not self._bins:
            raise ValueError("WindowBatcher initialisation produced no windows; check dataset and bucket sizes.")
        self._total_samples = sum(len(specs) for specs in self._bins.values())

    def __len__(self) -> int:
        return self._total_samples

    def __iter__(self) -> Iterable[WindowBatch[BatchT]]:
        keys = list(self._bins.keys())
        if self.shuffle:
            random.shuffle(keys)

        for key in keys:
            context, horizon = key
            specs = self._bins[key]
            if self.shuffle:
                random.shuffle(specs)
            tokens_per_sample = context + horizon
            micro_batch = max(1, self.max_tokens // max(tokens_per_sample, 1))
            idx = 0
            length = len(specs)
            while idx < length:
                end = min(idx + micro_batch, length)
                chunk = specs[idx:end]
                idx = end
                samples = [self.dataset.load_window(spec, context, horizon) for spec in chunk]
                batch_payload = self._collate(samples, context, horizon)
                yield WindowBatch(context=context, horizon=horizon, batch=batch_payload, size=len(chunk))
