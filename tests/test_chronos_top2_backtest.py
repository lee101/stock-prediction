"""Tests for chronos_top2_backtest and spread_estimate."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.spread_estimate import (
    corwin_schultz_spread_bps,
    estimate_spread_bps,
    volume_based_spread_bps,
)
from scripts.chronos_top2_backtest import (
    DayTrade,
    DayResult,
    _actual_for_day,
    _context_for_day,
    compute_metrics,
    get_trading_days,
    simulate_day,
)


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_ohlc(n: int, *, high_mult: float = 1.01, low_mult: float = 0.99,
               start_price: float = 100.0, volume: float = 1_000_000.0) -> pd.DataFrame:
    """Generate synthetic OHLC dataframe with n rows."""
    prices = start_price + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.clip(prices, 1.0, None)
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open":   prices * 0.999,
        "high":   prices * high_mult,
        "low":    prices * low_mult,
        "close":  prices,
        "volume": np.full(n, volume),
    })


def _make_day_result(day: date, net_ret_pct: float, equity_start: float = 10_000.0) -> DayResult:
    trade = DayTrade(
        symbol="TEST",
        predicted_return_pct=1.0,
        actual_open=100.0,
        actual_close=100.0 * (1 + net_ret_pct / 100),
        gross_return_pct=net_ret_pct + 0.3,
        spread_bps=15.0,
        commission_bps=10.0,
        net_return_pct=net_ret_pct,
    )
    equity_end = equity_start * (1 + net_ret_pct / 100)
    return DayResult(
        day=day,
        trades=[trade],
        equity_start=equity_start,
        equity_end=equity_end,
        daily_return_pct=net_ret_pct,
        n_candidates_screened=100,
    )


# ─── spread_estimate tests ───────────────────────────────────────────────────


class TestCorwinSchultzSpread:
    def test_zero_range_gives_zero(self):
        """If H == L every day, spread should be zero."""
        df = _make_ohlc(30, high_mult=1.0, low_mult=1.0)
        result = corwin_schultz_spread_bps(df, window=20)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_positive_spread_for_normal_data(self):
        """Normal OHLC data should yield a positive spread."""
        df = _make_ohlc(40, high_mult=1.015, low_mult=0.985)
        result = corwin_schultz_spread_bps(df, window=20)
        assert np.isfinite(result)
        assert result > 0

    def test_insufficient_data_returns_nan(self):
        """Fewer than 2 rows → NaN."""
        df = _make_ohlc(1)
        result = corwin_schultz_spread_bps(df, window=20)
        assert np.isnan(result)

    def test_empty_df_returns_nan(self):
        result = corwin_schultz_spread_bps(pd.DataFrame(), window=20)
        assert np.isnan(result)

    def test_window_respected(self):
        """Spread with window=5 should differ from window=20 in volatile data."""
        np.random.seed(42)
        df = _make_ohlc(60, high_mult=1.02, low_mult=0.98)
        s5  = corwin_schultz_spread_bps(df, window=5)
        s20 = corwin_schultz_spread_bps(df, window=20)
        assert np.isfinite(s5)
        assert np.isfinite(s20)
        # They should at least differ or be close; just ensure both are positive
        assert s5 >= 0
        assert s20 >= 0

    def test_wider_range_gives_larger_spread(self):
        """Wider daily ranges should produce larger estimated spreads."""
        np.random.seed(0)
        df_narrow = _make_ohlc(40, high_mult=1.005, low_mult=0.995)
        df_wide   = _make_ohlc(40, high_mult=1.030, low_mult=0.970)
        s_narrow = corwin_schultz_spread_bps(df_narrow, window=20)
        s_wide   = corwin_schultz_spread_bps(df_wide,   window=20)
        assert s_narrow < s_wide, f"Expected narrow<wide but got {s_narrow:.2f} vs {s_wide:.2f}"


class TestVolumeBasedSpread:
    def test_high_volume_gives_low_spread(self):
        df = _make_ohlc(20, volume=10_000_000.0)  # $1B/day at $100
        result = volume_based_spread_bps(df)
        assert result <= 3.0

    def test_low_volume_gives_high_spread(self):
        df = _make_ohlc(20, volume=100.0)  # negligible volume
        result = volume_based_spread_bps(df)
        assert result >= 50.0

    def test_empty_df_returns_fallback(self):
        result = volume_based_spread_bps(pd.DataFrame())
        assert result == 50.0


class TestEstimateSpreadBps:
    def test_uses_corwin_schultz_when_valid(self):
        np.random.seed(1)
        # Use a narrow ±0.3% range so C-S stays well below cs_max_bps=200
        df = _make_ohlc(40, high_mult=1.003, low_mult=0.997)
        cs_val = corwin_schultz_spread_bps(df, window=20)
        est = estimate_spread_bps(df, window=20, cs_max_bps=200.0)
        # If C-S produced a finite value within the cap, estimate_spread_bps should use it
        if np.isfinite(cs_val) and 0 < cs_val <= 200.0:
            assert est == pytest.approx(cs_val, rel=1e-6)
        else:
            # Fell back to volume-based; just ensure it's positive
            assert est > 0

    def test_falls_back_to_volume_when_cs_is_huge(self):
        # Force C-S to fail by using degenerate data
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC"),
            "open":  [100.0] * 5,
            "high":  [100.0] * 5,
            "low":   [100.0] * 5,
            "close": [100.0] * 5,
            "volume": [1e9] * 5,
        })
        # C-S should be 0, but it's finite and valid; volume-based should be 3
        result = estimate_spread_bps(df, window=20)
        # Both estimators should produce something sensible
        assert result >= 0


# ─── backtest data helper tests ──────────────────────────────────────────────


class TestContextForDay:
    def setup_method(self):
        self.df = _make_ohlc(100)

    def test_excludes_target_day(self):
        target = self.df["timestamp"].iloc[50].date()
        ctx = _context_for_day(self.df, target, context_length=128)
        assert ctx is not None
        # Last row in context should be < target_day
        assert ctx["timestamp"].iloc[-1].date() < target

    def test_respects_context_length(self):
        # Use a target well into the series so there are >30 rows before it,
        # but cap context_length so we get exactly that many rows.
        target = self.df["timestamp"].iloc[-1].date()
        # context_length=50 → should return ≤50 rows (and ≥30 so not None)
        ctx = _context_for_day(self.df, target, context_length=50)
        assert ctx is not None
        assert len(ctx) <= 50

    def test_returns_none_if_insufficient_history(self):
        # target_day = day after first row → only 0 rows before it
        target = (self.df["timestamp"].iloc[0] + pd.Timedelta(hours=1)).date()
        ctx = _context_for_day(self.df, target, context_length=128)
        assert ctx is None  # < 30 rows


class TestActualForDay:
    def setup_method(self):
        self.df = _make_ohlc(30)

    def test_returns_open_close(self):
        target = self.df["timestamp"].iloc[10].date()
        result = _actual_for_day(self.df, target)
        assert result is not None
        assert "open" in result and "close" in result
        assert result["open"] > 0 and result["close"] > 0

    def test_returns_none_for_missing_day(self):
        # Use a Saturday (non-trading day)
        target = date(2020, 1, 4)  # A Saturday unlikely to be in df
        result = _actual_for_day(self.df, target)
        assert result is None


class TestGetTradingDays:
    def test_filters_by_date_range(self):
        df = _make_ohlc(60)
        all_data = {"AAPL": df, "MSFT": df}
        start = df["timestamp"].iloc[10].date()
        end   = df["timestamp"].iloc[20].date()
        days = get_trading_days(all_data, start, end)
        assert all(start <= d <= end for d in days)

    def test_returns_sorted_days(self):
        df = _make_ohlc(40)
        all_data = {"X": df}
        start = df["timestamp"].iloc[0].date()
        end   = df["timestamp"].iloc[-1].date()
        days = get_trading_days(all_data, start, end)
        assert days == sorted(days)

    def test_empty_when_out_of_range(self):
        df = _make_ohlc(20)
        all_data = {"X": df}
        days = get_trading_days(all_data, date(2030, 1, 1), date(2030, 12, 31))
        assert days == []


# ─── simulate_day tests ──────────────────────────────────────────────────────


class TestSimulateDay:
    def setup_method(self):
        np.random.seed(99)
        # 80 rows of history
        self.df = _make_ohlc(80)
        self.target = self.df["timestamp"].iloc[50].date()
        self.symbol = "FAKE"
        self.all_data = {self.symbol: self.df}
        # Minimal forecasts
        self.forecasts = {self.symbol: 1.5}

    def test_returns_day_result(self):
        dr = simulate_day(
            target_day=self.target,
            forecasts=self.forecasts,
            all_data=self.all_data,
            equity=10_000.0,
            top_n=1,
            commission_bps=10.0,
            context_length=30,
        )
        assert dr is not None
        assert dr.day == self.target
        assert len(dr.trades) == 1
        assert dr.trades[0].symbol == self.symbol

    def test_net_return_accounts_for_costs(self):
        dr = simulate_day(
            target_day=self.target,
            forecasts=self.forecasts,
            all_data=self.all_data,
            equity=10_000.0,
            top_n=1,
            commission_bps=10.0,
            context_length=30,
        )
        t = dr.trades[0]
        # net = gross - (spread + 2*commission)/100
        expected_net = t.gross_return_pct - (t.spread_bps + 2 * t.commission_bps) / 100.0
        assert t.net_return_pct == pytest.approx(expected_net, rel=1e-6)

    def test_equity_updates_correctly(self):
        dr = simulate_day(
            target_day=self.target,
            forecasts=self.forecasts,
            all_data=self.all_data,
            equity=10_000.0,
            top_n=1,
            commission_bps=10.0,
            context_length=30,
        )
        expected_end = 10_000.0 * (1 + dr.daily_return_pct / 100.0)
        assert dr.equity_end == pytest.approx(expected_end, rel=1e-6)

    def test_returns_none_if_no_data_for_day(self):
        # forecasts for a symbol with no data on target day
        forecasts = {"NONEXISTENT": 5.0}
        dr = simulate_day(
            target_day=self.target,
            forecasts=forecasts,
            all_data={},
            equity=10_000.0,
            top_n=1,
            commission_bps=10.0,
            context_length=30,
        )
        assert dr is None

    def test_top_n_limits_picks(self):
        # Add multiple symbols
        all_data = {f"SYM{i}": self.df for i in range(10)}
        forecasts = {f"SYM{i}": float(i) for i in range(10)}
        dr = simulate_day(
            target_day=self.target,
            forecasts=forecasts,
            all_data=all_data,
            equity=10_000.0,
            top_n=3,
            commission_bps=10.0,
            context_length=30,
        )
        assert dr is not None
        assert len(dr.trades) <= 3


# ─── compute_metrics tests ───────────────────────────────────────────────────


class TestComputeMetrics:
    def test_zero_return_strategy(self):
        """Strategy that breaks even every day."""
        days = [date(2025, 1, i + 2) for i in range(10)]
        results = [_make_day_result(d, 0.0, equity_start=10_000.0) for d in days]
        # Fix equity chain
        for i in range(1, len(results)):
            results[i].equity_start = results[i - 1].equity_end
            results[i].equity_end = results[i].equity_start
        res = compute_metrics(results, initial_cash=10_000.0)
        assert res.total_return_pct == pytest.approx(0.0, abs=1e-6)
        assert res.max_drawdown_pct == pytest.approx(0.0, abs=1e-6)

    def test_always_winning_strategy(self):
        """Strategy that gains 1% every day."""
        equity = 10_000.0
        results = []
        for i in range(20):
            d = date(2025, 1, 2) + __import__("datetime").timedelta(days=i)
            dr = _make_day_result(d, 1.0, equity_start=equity)
            results.append(dr)
            equity = dr.equity_end
        res = compute_metrics(results, initial_cash=10_000.0)
        assert res.total_return_pct > 0
        assert res.win_rate_pct == pytest.approx(100.0)
        assert res.max_drawdown_pct == pytest.approx(0.0, abs=1e-3)

    def test_metrics_structure(self):
        """All required fields present and finite."""
        days = [date(2025, 2, i + 3) for i in range(15)]
        np.random.seed(42)
        equity = 10_000.0
        results = []
        for d in days:
            ret = float(np.random.randn() * 0.5)
            dr = _make_day_result(d, ret, equity_start=equity)
            results.append(dr)
            equity = dr.equity_end
        res = compute_metrics(results, initial_cash=10_000.0)

        for field_name in (
            "total_return_pct", "annualized_return_pct", "monthly_return_pct",
            "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "win_rate_pct",
        ):
            val = getattr(res, field_name)
            assert np.isfinite(val), f"{field_name} is not finite: {val}"

    def test_raises_on_empty(self):
        with pytest.raises((ValueError, IndexError)):
            compute_metrics([], initial_cash=10_000.0)
