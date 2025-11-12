"""Unit tests for the DIRECT-specific helpers in benchmark_chronos2."""

from types import SimpleNamespace

from benchmark_chronos2 import (
    MIN_CONTEXT,
    VAL_WINDOW,
    _build_direct_search_space,
    _resolve_context_lengths,
)


def _base_args(**overrides):
    defaults = dict(
        auto_context_lengths=False,
        auto_context_min=MIN_CONTEXT,
        auto_context_max=4096,
        auto_context_step=128,
        auto_context_guard=VAL_WINDOW * 2,
        context_lengths=[MIN_CONTEXT, 512, 2048],
        batch_sizes=[64, 128],
        aggregations=["median"],
        sample_counts=[0, 256],
        scalers=["none", "meanstd"],
        direct_sample_counts=None,
        direct_batch_sizes=None,
        direct_aggregations=None,
        direct_scalers=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_context_lengths_auto_caps_by_dataset():
    args = _base_args(auto_context_lengths=True, auto_context_min=256, auto_context_step=256, auto_context_guard=64)
    lengths = _resolve_context_lengths(series_length=1000, args=args)
    assert lengths[0] == 256
    assert lengths[-1] == 936  # 1000 - guard
    assert all(length >= MIN_CONTEXT for length in lengths)


def test_build_direct_search_space_filters_invalid_entries():
    args = _base_args(context_lengths=[128, 256, 9000], batch_sizes=[0, 64, 64], sample_counts=[-1, 0, 128])
    space = _build_direct_search_space(series_length=600, args=args)
    assert space.context_lengths == (128, 256)
    assert space.batch_sizes == (64,)
    assert space.sample_counts == (0, 128)
    assert space.scalers == ("none", "meanstd")
