"""Tests for trend-following position sizing overlay."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rebalance_trend_overlay import TrendOverlay, TrendOverlayConfig


class TestTrendOverlay:
    def test_bull_market_goes_long(self):
        closes = np.linspace(100, 200, 1000)
        overlay = TrendOverlay()
        alloc = overlay.compute_allocation(closes)
        assert alloc[-1] > 0, "Should be long in uptrend"

    def test_bear_market_goes_short(self):
        closes = np.linspace(200, 100, 1000)
        overlay = TrendOverlay()
        alloc = overlay.compute_allocation(closes)
        assert alloc[-1] < 0, "Should be short in downtrend"

    def test_flat_market_near_zero(self):
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(2000) * 0.01)
        overlay = TrendOverlay()
        alloc = overlay.compute_allocation(closes)
        assert abs(alloc[-1]) < 5.0

    def test_output_shape(self):
        closes = np.linspace(100, 150, 500)
        overlay = TrendOverlay()
        alloc = overlay.compute_allocation(closes)
        assert alloc.shape == closes.shape

    def test_max_alloc_respected(self):
        closes = np.linspace(100, 200, 1000)
        cfg = TrendOverlayConfig(target_vol=10.0, max_alloc=1.5)
        overlay = TrendOverlay(cfg)
        alloc = overlay.compute_allocation(closes)
        assert np.all(np.abs(alloc) <= 1.5 + 1e-6)

    def test_drawdown_cut(self):
        closes = np.concatenate([np.linspace(100, 200, 500), np.linspace(200, 160, 300)])
        cfg = TrendOverlayConfig(dd_cut=0.10, dd_reduce_factor=0.2)
        overlay_nocut = TrendOverlay(TrendOverlayConfig())
        overlay_cut = TrendOverlay(cfg)
        alloc_nocut = overlay_nocut.compute_allocation(closes)
        alloc_cut = overlay_cut.compute_allocation(closes)
        # During drawdown phase, cut should have smaller alloc
        assert np.abs(alloc_cut[-1]) < np.abs(alloc_nocut[-1])

    def test_single_bar_matches_last(self):
        closes = np.linspace(100, 150, 500)
        overlay = TrendOverlay()
        full_alloc = overlay.compute_allocation(closes)
        single = overlay.compute_allocation_single(closes)
        assert abs(full_alloc[-1] - single) < 1e-6

    def test_vol_targeting_reduces_in_high_vol(self):
        np.random.seed(42)
        low_vol = 100 + np.cumsum(np.random.randn(1000) * 0.1)
        high_vol = 100 + np.cumsum(np.random.randn(1000) * 2.0)
        cfg = TrendOverlayConfig(target_vol=0.2)
        overlay = TrendOverlay(cfg)
        alloc_low = overlay.compute_allocation(low_vol)
        alloc_high = overlay.compute_allocation(high_vol)
        # High vol should have smaller allocations on average
        assert np.mean(np.abs(alloc_high[200:])) < np.mean(np.abs(alloc_low[200:]))
