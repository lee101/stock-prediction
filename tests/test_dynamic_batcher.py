import torch
import pytest

from traininglib.dynamic_batcher import WindowBatcher, WindowSpec


class DummyDataset:
    def __init__(self, length: int = 32):
        self._series = torch.arange(length, dtype=torch.float32)

    @property
    def series_ids(self):
        return (0,)

    def enumerate_window_specs(self, context: int, horizon: int, stride: int):
        if context <= 0 or horizon <= 0:
            return []
        upper = len(self._series) - (context + horizon) + 1
        if upper <= 0:
            return []
        return [WindowSpec(0, left) for left in range(0, upper, stride)]

    def load_window(self, spec: WindowSpec, context: int, horizon: int):
        start = spec.left
        ctx = self._series[start : start + context]
        tgt = self._series[start + context : start + context + horizon]
        return ctx, tgt

    def collate_windows(self, samples, context: int, horizon: int):
        contexts, targets = zip(*samples)
        return torch.stack(contexts), torch.stack(targets)


def test_window_batcher_respects_token_budget():
    dataset = DummyDataset(length=20)
    batcher = WindowBatcher(
        dataset,
        max_tokens_per_batch=12,
        context_buckets=[3],
        horizon_buckets=[1],
        stride=2,
    )
    batches = list(batcher)
    assert batches, "Expected at least one batch"
    for batch in batches:
        ctx, tgt = batch.batch
        total_tokens = (ctx.shape[1] + tgt.shape[1]) * ctx.shape[0]
        assert total_tokens <= 12
        assert ctx.shape[1] == 3
        assert tgt.shape[1] == 1


def test_window_batcher_multiple_buckets():
    dataset = DummyDataset(length=30)
    batcher = WindowBatcher(
        dataset,
        max_tokens_per_batch=16,
        context_buckets=[2, 4],
        horizon_buckets=[1, 2],
        stride=1,
    )
    seen_shapes = set()
    for batch in batcher:
        ctx, tgt = batch.batch
        seen_shapes.add((ctx.shape[1], tgt.shape[1]))
        assert ctx.shape[0] > 0
    assert seen_shapes == {(2, 1), (2, 2), (4, 1), (4, 2)}


def test_window_batcher_no_windows_raises():
    dataset = DummyDataset(length=3)
    with pytest.raises(ValueError):
        WindowBatcher(dataset, max_tokens_per_batch=8, context_buckets=[5], horizon_buckets=[2], stride=1)


def test_oversized_buckets_are_skipped():
    dataset = DummyDataset(length=40)
    # Budget too small for (10+10) but fine for (3+1)
    batcher = WindowBatcher(
        dataset,
        max_tokens_per_batch=12,
        context_buckets=[3, 10],
        horizon_buckets=[1, 10],
        stride=2,
        shuffle=False,
    )
    shapes = {(b.batch[0].shape[1], b.batch[1].shape[1]) for b in batcher}
    assert (3, 1) in shapes
    assert (10, 10) not in shapes


def test_all_buckets_oversized_raises():
    dataset = DummyDataset(length=200)
    with pytest.raises(ValueError):
        # Every (context+horizon) exceeds budget
        WindowBatcher(
            dataset,
            max_tokens_per_batch=8,
            context_buckets=[12, 16],
            horizon_buckets=[4, 8],
            stride=1,
        )
