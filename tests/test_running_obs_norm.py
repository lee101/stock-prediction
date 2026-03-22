"""Unit tests for RunningObsNorm Welford online normalizer.

RunningObsNorm uses Welford's parallel algorithm to maintain running mean/variance
across batches of observations.  These tests verify correctness of the statistics,
edge-case robustness (zero variance, clip, dtype), and shape invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from pufferlib_market.train import RunningObsNorm


def _make_norm(shape: int = 4, clip: float = 10.0, eps: float = 1e-8) -> RunningObsNorm:
    return RunningObsNorm(shape=shape, clip=clip, eps=eps)


# ---------------------------------------------------------------------------

def test_single_update_mean_close_to_sample():
    """After one single-row update, mean should be very close to that sample.

    The initial pseudo-count is 1e-4, so the true mean is:
        new_mean = 0 + sample * 1 / (1e-4 + 1) ≈ sample * 0.9999
    Relative tolerance of 1e-3 covers this.
    """
    norm = _make_norm(shape=3)
    sample = np.array([[5.0, -2.0, 0.5]], dtype=np.float32)
    norm.update(sample)

    np.testing.assert_allclose(
        norm.mean,
        sample[0].astype(np.float64),
        rtol=1e-3,
    )


def test_multiple_updates_mean():
    """Running mean after N updates must match np.mean(data, axis=0)."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((500, 6)).astype(np.float32)
    norm = _make_norm(shape=6)
    norm.update(data)

    np.testing.assert_allclose(
        norm.mean,
        np.mean(data, axis=0).astype(np.float64),
        atol=1e-5,
    )


def test_multiple_updates_variance():
    """Running variance after N updates must match np.var(data, axis=0) (population).

    The implementation divides by `total` (not total-1), so it tracks population
    variance.  The small initial count (1e-4) causes negligible bias for large N.
    """
    rng = np.random.default_rng(1)
    data = rng.standard_normal((500, 6)).astype(np.float32)
    norm = _make_norm(shape=6)
    norm.update(data)

    np.testing.assert_allclose(
        norm.var,
        np.var(data, axis=0).astype(np.float64),
        atol=1e-5,
    )


def test_incremental_batches_match_single_batch():
    """Feeding data in two batches must give the same mean/var as one big batch.

    float32 batch moments introduce ~1e-7 rounding; float64 accumulation keeps
    the error small but not machine-epsilon.
    """
    rng = np.random.default_rng(2)
    data = rng.standard_normal((200, 4)).astype(np.float32)

    norm_single = _make_norm(shape=4)
    norm_single.update(data)

    norm_batched = _make_norm(shape=4)
    norm_batched.update(data[:100])
    norm_batched.update(data[100:])

    np.testing.assert_allclose(norm_batched.mean, norm_single.mean, atol=1e-6)
    np.testing.assert_allclose(norm_batched.var, norm_single.var, atol=1e-6)


def test_zero_variance_no_div_by_zero():
    """Constant inputs: variance stays near 0; normalize must not raise."""
    norm = _make_norm(shape=2)
    data = np.ones((100, 2), dtype=np.float32) * 3.0
    norm.update(data)

    assert norm.var[0] < 1e-4
    result = norm.normalize(data[0])
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("clip", [5.0, 10.0])
def test_clip_bounds_respected(clip):
    """After normalization, all values must lie in [-clip, clip]."""
    norm = RunningObsNorm(shape=4, clip=clip)
    rng = np.random.default_rng(3)
    data = rng.standard_normal((200, 4)).astype(np.float32)
    norm.update(data)

    extreme_obs = np.array([1e6, -1e6, 0.0, 1e6], dtype=np.float32)
    result = norm.normalize(extreme_obs)

    assert np.all(result >= -clip), f"min result {result.min()} below -{clip}"
    assert np.all(result <= clip), f"max result {result.max()} above +{clip}"


def test_normalize_dtype_is_float32():
    """normalize() must always return a float32 array regardless of input dtype."""
    norm = _make_norm(shape=4)
    rng = np.random.default_rng(5)
    norm.update(rng.standard_normal((100, 4)).astype(np.float32))

    assert norm.normalize(np.zeros(4, dtype=np.float32)).dtype == np.float32
    assert norm.normalize(np.zeros(4, dtype=np.float64)).dtype == np.float32


def test_normalize_shape_preserved():
    """Output shape of normalize() must match input shape."""
    shape = 7
    norm = _make_norm(shape=shape)
    rng = np.random.default_rng(6)
    norm.update(rng.standard_normal((50, shape)).astype(np.float32))

    obs = rng.standard_normal(shape).astype(np.float32)
    assert norm.normalize(obs).shape == (shape,)


def test_initial_state_zeros_normalize_to_zeros():
    """Fresh RunningObsNorm has mean=0, var=1, so normalize(zeros) returns zeros."""
    norm = _make_norm(shape=5)
    np.testing.assert_allclose(
        norm.normalize(np.zeros(5, dtype=np.float32)),
        np.zeros(5, dtype=np.float32),
        atol=1e-6,
    )


def test_normalization_centers_and_scales():
    """After sufficient data, the mean of normalized outputs should be near 0."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((1000, 4)).astype(np.float32)
    norm = _make_norm(shape=4)
    norm.update(data)

    normalized = np.array([norm.normalize(row) for row in data])
    np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(4), atol=0.1)
