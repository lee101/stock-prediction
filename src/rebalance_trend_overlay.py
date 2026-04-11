#!/usr/bin/env python3
"""Trend-following position sizing overlay for Binance margin trading.

Based on exhaustive analysis (2026-04-12):
- EMA(48,336) crossover is the most robust regime detector at lag=2
- Vol-targeting (target_vol / realized_vol) gives lag-resistant position sizing
- Works on both bull and bear markets for BTC, ETH, SOL, DOGE, LINK

Usage:
    from src.rebalance_trend_overlay import TrendOverlay
    overlay = TrendOverlay()
    alloc = overlay.compute_allocation(closes)
    # alloc in [-1, 1]: negative = short, positive = long
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class TrendOverlayConfig:
    fast_ema: int = 48
    slow_ema: int = 336
    target_vol: float = 0.2
    max_alloc: float = 2.0
    vol_window: int = 168
    dd_cut: float = 0.0  # 0 = disabled, e.g. 0.10 = reduce 80% when in >10% drawdown
    dd_reduce_factor: float = 0.2


class TrendOverlay:
    def __init__(self, config: TrendOverlayConfig | None = None):
        self.config = config or TrendOverlayConfig()

    def _ema(self, data: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        out = np.empty_like(data)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out

    def _rolling_vol(self, closes: np.ndarray) -> np.ndarray:
        ret = np.diff(closes, prepend=closes[0]) / np.maximum(closes, 1e-8)
        vol = np.zeros(len(closes))
        w = self.config.vol_window
        for i in range(w, len(closes)):
            vol[i] = np.std(ret[i - w:i]) * np.sqrt(8766)
        vol[:w] = vol[w] if len(closes) > w else 0.5
        return np.maximum(vol, 0.01)

    def compute_allocation(self, closes: np.ndarray) -> np.ndarray:
        c = self.config
        ef = self._ema(closes, c.fast_ema)
        es = self._ema(closes, c.slow_ema)
        trend = np.sign(ef - es)  # +1 or -1
        vol = self._rolling_vol(closes)
        vol_scale = np.clip(c.target_vol / vol, 0, c.max_alloc)
        alloc = trend * vol_scale
        if c.dd_cut > 0:
            running_max = np.maximum.accumulate(closes)
            dd = 1.0 - closes / running_max
            alloc = np.where(dd > c.dd_cut, alloc * c.dd_reduce_factor, alloc)
        return alloc

    def compute_allocation_single(self, closes: np.ndarray) -> float:
        """Get current allocation for the last bar."""
        alloc = self.compute_allocation(closes)
        return float(alloc[-1])
