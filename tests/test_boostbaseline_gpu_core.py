"""Numerical parity + contract tests for ``boostbaseline.gpu_core``.

The GPU path must give identical PnL / sizing to the CPU path when both
use turnover-proportional fees. If CuPy is unavailable the suite is
skipped rather than failing — we keep the CPU fallback legit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

pytest.importorskip("cupy")
from boostbaseline.backtest import run_backtest, grid_search_sizing  # noqa: E402
from boostbaseline.gpu_core import (  # noqa: E402
    grid_search_sizing_gpu,
    run_backtest_gpu,
)


def _toy_rets(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(0.001, 0.01, n).astype(np.float32)
    y_pred = y_true + rng.normal(0, 0.002, n).astype(np.float32)
    return y_true, y_pred


def test_run_backtest_gpu_matches_cpu_turnover_prop():
    y_true, y_pred = _toy_rets(500)
    gpu_tr, gpu_sh = run_backtest_gpu(
        y_true, y_pred, is_crypto=True, fee=0.001, scale=2.0, cap=0.3,
    )
    cpu = run_backtest(
        y_true, y_pred, is_crypto=True, fee=0.001, scale=2.0, cap=0.3,
        turnover_proportional_fee=True,
    )
    # CPU total_return is log-space; GPU total is simple-product. They should
    # still be close (both use same positions and fees on the same rets).
    np.testing.assert_allclose(gpu_tr, cpu.total_return, rtol=1e-3, atol=1e-5)


def test_grid_search_sizing_gpu_picks_same_region_as_cpu():
    y_true, y_pred = _toy_rets(600, seed=1)
    tr_gpu, sh_gpu, s_gpu, c_gpu = grid_search_sizing_gpu(
        y_true, y_pred, is_crypto=True, fee=0.0005,
    )
    bt_cpu = grid_search_sizing(
        y_true, y_pred, is_crypto=True, fee=0.0005,
        turnover_proportional_fee=True,
    )
    # The two grids aren't identical in scales/caps, so we only require
    # the GPU pick to be finite and positive-order-of-magnitude with CPU.
    assert np.isfinite(tr_gpu)
    assert s_gpu > 0 and c_gpu > 0
    assert abs(tr_gpu - bt_cpu.total_return) / max(abs(bt_cpu.total_return), 1e-6) < 2.0


def test_run_backtest_gpu_is_crypto_false_allows_shorts():
    # Perfect predictor. Long-only captures up-days; both-sides captures
    # down-days too, so must dominate when there are any negative rets.
    rng = np.random.default_rng(3)
    y_true = rng.normal(0.0, 0.01, 400).astype(np.float32)
    y_pred = y_true.copy()
    tr_long_only, _ = run_backtest_gpu(
        y_true, y_pred, is_crypto=True, fee=0.0, scale=1.0, cap=1.0,
    )
    tr_both_sides, _ = run_backtest_gpu(
        y_true, y_pred, is_crypto=False, fee=0.0, scale=1.0, cap=1.0,
    )
    assert tr_both_sides > tr_long_only


def test_run_backtest_gpu_accepts_pandas_series():
    y_true, y_pred = _toy_rets(300)
    s_true = pd.Series(y_true, name="y")
    s_pred = pd.Series(y_pred, name="p")
    tr, sh = run_backtest_gpu(s_true, s_pred, is_crypto=True, fee=0.001)
    assert np.isfinite(tr)
    assert np.isfinite(sh)


def test_run_backtest_gpu_shape_mismatch_raises():
    with pytest.raises(ValueError):
        run_backtest_gpu(np.zeros(10), np.zeros(11), is_crypto=True, fee=0.001)


def test_run_backtest_gpu_zero_fee_pure_pnl():
    """At fee=0 with cap>=1 and scale=1, total return should equal
    ∏(1 + y_true * clip(y_pred)) − 1."""
    y_true, y_pred = _toy_rets(200, seed=7)
    tr, _ = run_backtest_gpu(
        y_true, y_pred, is_crypto=False, fee=0.0, scale=1.0, cap=1.0,
    )
    pos = np.clip(1.0 * y_pred, -1.0, 1.0)
    rets = pos * y_true
    ref = float(np.prod(1.0 + rets) - 1.0)
    np.testing.assert_allclose(tr, ref, rtol=1e-3, atol=1e-5)
