"""Tests for build_cross_asset_context and cross-asset prompt integration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rl_trading_agent_binance"))
sys.path.insert(0, str(REPO))

from rl_trading_agent_binance_prompt import (
    build_cross_asset_context,
    build_live_prompt,
    build_live_prompt_freeform,
)


def _make_bars(base_price: float, ret_24h_pct: float, n: int = 72) -> list[dict]:
    """Generate synthetic hourly bars with a given 24h return."""
    prices = []
    start = base_price / (1 + ret_24h_pct / 100)
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 1.0
        p = start + (base_price - start) * frac
        prices.append(p)
    rows = []
    for i, p in enumerate(prices):
        rows.append({
            "timestamp": f"2026-03-19T{i % 24:02d}:00:00+00:00",
            "open": p * 0.999,
            "high": p * 1.002,
            "low": p * 0.998,
            "close": p,
            "volume": 1000.0,
        })
    return rows


def _make_df_bars(base_price: float, ret_24h_pct: float, n: int = 72) -> pd.DataFrame:
    rows = _make_bars(base_price, ret_24h_pct, n)
    return pd.DataFrame(rows)


class TestBuildCrossAssetContext:
    def test_empty_dict_returns_empty(self):
        assert build_cross_asset_context({}) == ""

    def test_single_symbol_returns_empty(self):
        bars = {"BTCUSD": _make_bars(85000, 2.0)}
        assert build_cross_asset_context(bars) == ""

    def test_risk_on_regime(self):
        bars = {
            "BTCUSD": _make_bars(85000, 3.0),
            "ETHUSD": _make_bars(2000, 4.0),
            "SOLUSD": _make_bars(140, 5.0),
        }
        ctx = build_cross_asset_context(bars)
        assert "risk-on" in ctx
        assert "=== Market Regime ===" in ctx
        assert "BTC-ETH correlation" in ctx
        assert "Top movers" in ctx

    def test_risk_off_regime(self):
        bars = {
            "BTCUSD": _make_bars(85000, -3.0),
            "ETHUSD": _make_bars(2000, -4.0),
            "SOLUSD": _make_bars(140, -2.5),
        }
        ctx = build_cross_asset_context(bars)
        assert "risk-off" in ctx

    def test_rotation_regime(self):
        bars = {
            "BTCUSD": _make_bars(85000, 5.0),
            "ETHUSD": _make_bars(2000, -1.0),
            "SOLUSD": _make_bars(140, -2.0),
        }
        ctx = build_cross_asset_context(bars)
        assert "rotation" in ctx

    def test_neutral_regime(self):
        bars = {
            "BTCUSD": _make_bars(85000, 0.1),
            "ETHUSD": _make_bars(2000, 0.2),
            "SOLUSD": _make_bars(140, -0.1),
        }
        ctx = build_cross_asset_context(bars)
        assert "neutral" in ctx

    def test_dataframe_input(self):
        bars = {
            "BTCUSD": _make_df_bars(85000, 2.0),
            "ETHUSD": _make_df_bars(2000, 3.0),
        }
        ctx = build_cross_asset_context(bars)
        assert "=== Market Regime ===" in ctx

    def test_mixed_list_and_df_input(self):
        bars = {
            "BTCUSD": _make_bars(85000, 2.0),
            "ETHUSD": _make_df_bars(2000, 3.0),
        }
        ctx = build_cross_asset_context(bars)
        assert "=== Market Regime ===" in ctx

    def test_missing_symbols_graceful(self):
        bars = {
            "BTCUSD": _make_bars(85000, 2.0),
            "ETHUSD": None,
            "SOLUSD": _make_bars(140, 1.5),
        }
        ctx = build_cross_asset_context(bars)
        assert "=== Market Regime ===" in ctx

    def test_empty_df_graceful(self):
        bars = {
            "BTCUSD": _make_bars(85000, 2.0),
            "ETHUSD": pd.DataFrame(),
            "SOLUSD": _make_bars(140, 1.0),
        }
        ctx = build_cross_asset_context(bars)
        assert ctx != ""

    def test_too_few_bars_graceful(self):
        bars = {
            "BTCUSD": _make_bars(85000, 2.0, n=5),
            "ETHUSD": _make_bars(2000, 1.0, n=5),
        }
        ctx = build_cross_asset_context(bars)
        assert ctx == ""

    def test_btc_dominance_positive(self):
        bars = {
            "BTCUSD": _make_bars(85000, 5.0),
            "ETHUSD": _make_bars(2000, 1.0),
            "SOLUSD": _make_bars(140, 0.5),
        }
        ctx = build_cross_asset_context(bars)
        assert "BTC" in ctx
        assert "+" in ctx

    def test_top_movers_sorted_by_magnitude(self):
        bars = {
            "BTCUSD": _make_bars(85000, 1.0),
            "ETHUSD": _make_bars(2000, -5.0),
            "SOLUSD": _make_bars(140, 8.0),
            "DOGEUSD": _make_bars(0.15, 0.1),
        }
        ctx = build_cross_asset_context(bars)
        lines = ctx.split("\n")
        movers_line = [l for l in lines if "Top movers" in l][0]
        assert "SOL" in movers_line.split(",")[0]

    def test_correlation_computed(self):
        bars = {
            "BTCUSD": _make_bars(85000, 3.0),
            "ETHUSD": _make_bars(2000, 3.0),
        }
        ctx = build_cross_asset_context(bars)
        corr_line = [l for l in ctx.split("\n") if "correlation" in l][0]
        assert "N/A" not in corr_line

    def test_no_btc_still_works(self):
        bars = {
            "ETHUSD": _make_bars(2000, 2.0),
            "SOLUSD": _make_bars(140, -1.0),
        }
        ctx = build_cross_asset_context(bars)
        assert ctx != ""
        assert "=== Market Regime ===" in ctx


class TestPromptIntegration:
    def _minimal_history(self):
        return _make_bars(85000, 1.0, n=72)

    def test_build_live_prompt_without_cross_asset(self):
        prompt = build_live_prompt("BTCUSD", self._minimal_history(), 85000.0)
        assert "Market Regime" not in prompt
        assert "BTCUSD" in prompt

    def test_build_live_prompt_with_cross_asset(self):
        bars = {
            "BTCUSD": _make_bars(85000, 3.0),
            "ETHUSD": _make_bars(2000, 2.0),
            "SOLUSD": _make_bars(140, 4.0),
        }
        ctx = build_cross_asset_context(bars)
        prompt = build_live_prompt(
            "BTCUSD", self._minimal_history(), 85000.0,
            cross_asset_context=ctx,
        )
        assert "=== Market Regime ===" in prompt
        assert "risk-on" in prompt

    def test_build_live_prompt_freeform_with_cross_asset(self):
        bars = {
            "BTCUSD": _make_bars(85000, -3.0),
            "ETHUSD": _make_bars(2000, -2.0),
            "SOLUSD": _make_bars(140, -4.0),
        }
        ctx = build_cross_asset_context(bars)
        prompt = build_live_prompt_freeform(
            "BTCUSD", self._minimal_history(), 85000.0,
            cross_asset_context=ctx,
        )
        assert "=== Market Regime ===" in prompt
        assert "risk-off" in prompt

    def test_prompt_length_reasonable(self):
        bars = {
            "BTCUSD": _make_bars(85000, 3.0),
            "ETHUSD": _make_bars(2000, 2.0),
            "SOLUSD": _make_bars(140, 4.0),
            "DOGEUSD": _make_bars(0.15, 1.0),
            "AAVEUSD": _make_bars(300, -1.0),
        }
        ctx = build_cross_asset_context(bars)
        assert len(ctx) < 500
        prompt = build_live_prompt(
            "BTCUSD", self._minimal_history(), 85000.0,
            cross_asset_context=ctx,
        )
        assert len(prompt) < 5000

    def test_backwards_compat_no_cross_asset(self):
        prompt_old = build_live_prompt("BTCUSD", self._minimal_history(), 85000.0)
        prompt_new = build_live_prompt(
            "BTCUSD", self._minimal_history(), 85000.0,
            cross_asset_context="",
        )
        assert prompt_old == prompt_new

    def test_freeform_backwards_compat(self):
        prompt_old = build_live_prompt_freeform("BTCUSD", self._minimal_history(), 85000.0)
        prompt_new = build_live_prompt_freeform(
            "BTCUSD", self._minimal_history(), 85000.0,
            cross_asset_context="",
        )
        assert prompt_old == prompt_new
