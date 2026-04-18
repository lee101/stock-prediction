"""Tests for xgbnew feature engineering, dataset builder, and model."""

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

from xgbnew.features import (
    CHRONOS_FEATURE_COLS,
    DAILY_FEATURE_COLS,
    build_features_for_symbol,
    build_features_for_symbol_hourly,
    _rsi_series,
    _cs_spread_series,
)
from xgbnew.backtest import BacktestConfig, simulate, DayResult, DayTrade


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_ohlcv(
    n: int = 300,
    start: str = "2022-01-03",
    high_mult: float = 1.01,
    low_mult: float = 0.99,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.clip(prices, 1.0, None)
    ts = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open":   prices * 0.999,
        "high":   prices * high_mult,
        "low":    prices * low_mult,
        "close":  prices,
        "volume": np.random.uniform(1e5, 1e7, n),
    })


# ─── _rsi_series ────────────────────────────────────────────────────────────


class TestRSISeries:
    def test_range_0_to_100(self):
        close = pd.Series(np.random.randn(100).cumsum() + 50)
        rsi = _rsi_series(close, 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_all_up_gives_high_rsi(self):
        close = pd.Series(np.arange(1.0, 101.0))
        rsi = _rsi_series(close, 14)
        assert rsi.iloc[-1] > 90

    def test_all_down_gives_low_rsi(self):
        close = pd.Series(np.arange(100.0, 0.0, -1.0))
        rsi = _rsi_series(close, 14)
        assert rsi.iloc[-1] < 10


# ─── _cs_spread_series ───────────────────────────────────────────────────────


class TestCSSpread:
    def test_positive_for_normal_data(self):
        df = _make_ohlcv(50)
        s = _cs_spread_series(df["high"], df["low"])
        valid = s.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_zero_range_gives_zero(self):
        df = _make_ohlcv(40, high_mult=1.0, low_mult=1.0)
        s = _cs_spread_series(df["high"], df["low"])
        assert s.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-6)


# ─── build_features_for_symbol ───────────────────────────────────────────────


class TestBuildFeaturesForSymbol:
    def setup_method(self):
        self.df = _make_ohlcv(350)

    def test_returns_dataframe(self):
        feat = build_features_for_symbol(self.df, symbol="TEST")
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) == len(self.df)

    def test_all_feature_cols_present(self):
        feat = build_features_for_symbol(self.df)
        for col in DAILY_FEATURE_COLS:
            assert col in feat.columns, f"Missing: {col}"

    def test_target_columns_present(self):
        feat = build_features_for_symbol(self.df)
        assert "target_oc" in feat.columns
        assert "target_oc_up" in feat.columns

    def test_no_lookahead_for_returns(self):
        """features[D]['ret_1d'] must not use close[D]."""
        feat = build_features_for_symbol(self.df, symbol="X")
        # ret_1d[D] should be (close[D-1] / close[D-2] - 1)
        # Not (close[D] / close[D-1] - 1)
        close = self.df["close"].values
        for i in range(5, 15):
            expected = close[i - 1] / close[i - 2] - 1.0
            actual = feat["ret_1d"].iloc[i]
            if np.isfinite(actual):
                assert actual == pytest.approx(expected, rel=1e-4)

    def test_target_oc_correct(self):
        """target_oc[D] = (close[D] - open[D]) / open[D]."""
        feat = build_features_for_symbol(self.df)
        for i in range(5, 15):
            o = self.df["open"].iloc[i]
            c = self.df["close"].iloc[i]
            expected = (c - o) / o
            actual = feat["target_oc"].iloc[i]
            assert actual == pytest.approx(np.clip(expected, -0.5, 0.5), rel=1e-5)

    def test_target_oc_up_binary(self):
        feat = build_features_for_symbol(self.df)
        vals = feat["target_oc_up"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_no_future_leakage_rsi(self):
        """RSI at row D uses only data up to D-1 (prev_close = close.shift(1)).

        Since rsi_14[D] is computed on prev_close = close.shift(1), modifying
        close[D] should NOT change rsi_14[D] — the feature at D only sees
        close[0..D-1].  We verify:
          1. Changing close[-1] leaves rsi_14[-1] unchanged (prev_close[-1] =
             close[-2], so close[-1] is invisible).
          2. Changing close[-2] DOES change rsi_14[-1] (prev_close[-1] =
             close[-2]), confirming the signal actually propagates.
        """
        df_base = self.df.copy()

        # --- test 1: modifying today's close must NOT affect today's RSI ---
        df_modified_last = df_base.copy()
        df_modified_last.loc[df_modified_last.index[-1], "close"] = \
            df_modified_last["close"].iloc[-1] * 2.0
        feat_base = build_features_for_symbol(df_base)
        feat_mod_last = build_features_for_symbol(df_modified_last)
        assert feat_base["rsi_14"].iloc[-1] == pytest.approx(
            feat_mod_last["rsi_14"].iloc[-1], rel=1e-4
        ), "rsi_14[-1] must not change when only close[-1] is modified (no-lookahead)"

        # --- test 2: modifying yesterday's close SHOULD change today's RSI ---
        df_modified_prev = df_base.copy()
        df_modified_prev.loc[df_modified_prev.index[-2], "close"] = \
            df_modified_prev["close"].iloc[-2] * 2.0
        feat_mod_prev = build_features_for_symbol(df_modified_prev)
        assert feat_base["rsi_14"].iloc[-1] != pytest.approx(
            feat_mod_prev["rsi_14"].iloc[-1], rel=1e-4
        ), "rsi_14[-1] must change when close[-2] changes (prev_close[-1] = close[-2])"

    def test_symbol_column(self):
        feat = build_features_for_symbol(self.df, symbol="AAPL")
        assert "symbol" in feat.columns
        assert (feat["symbol"] == "AAPL").all()

    def test_day_of_week_range(self):
        feat = build_features_for_symbol(self.df)
        vals = feat["day_of_week"].dropna()
        assert ((vals >= 0) & (vals <= 4)).all()


# ─── build_features_for_symbol_hourly ───────────────────────────────────────


class TestBuildFeaturesHourly:
    def setup_method(self):
        n = 200
        np.random.seed(1)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        ts = pd.date_range("2024-01-02 09:30", periods=n, freq="h", tz="UTC")
        self.df = pd.DataFrame({
            "timestamp": ts,
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.uniform(1e4, 1e6, n),
        })

    def test_returns_dataframe(self):
        feat = build_features_for_symbol_hourly(self.df, symbol="X")
        assert isinstance(feat, pd.DataFrame)

    def test_has_hour_feature(self):
        feat = build_features_for_symbol_hourly(self.df)
        assert "hour_of_day" in feat.columns

    def test_target_columns_present(self):
        feat = build_features_for_symbol_hourly(self.df)
        assert "target_oc" in feat.columns
        assert "target_oc_up" in feat.columns


# ─── BacktestConfig and simulate ────────────────────────────────────────────


class _DummyModel:
    """Minimal model that always returns 0.6 (predicts up) for any input."""

    feature_cols = DAILY_FEATURE_COLS
    _col_medians = np.zeros(len(DAILY_FEATURE_COLS))
    _fitted = True

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.6, index=df.index, name="xgb_score")


class TestBacktestSimulate:
    def _make_test_df(self, n_days: int = 20, n_syms: int = 5) -> pd.DataFrame:
        np.random.seed(99)
        days = pd.date_range("2026-01-05", periods=n_days, freq="B").date
        rows = []
        for d in days:
            for s_idx in range(n_syms):
                sym = f"SYM{s_idx}"
                o = 100.0 + s_idx
                c = o * (1 + np.random.randn() * 0.01)
                rows.append({
                    **{col: 0.0 for col in DAILY_FEATURE_COLS},
                    **{col: 0.0 for col in CHRONOS_FEATURE_COLS},
                    "date": d, "symbol": sym,
                    "actual_open": o, "actual_close": c,
                    "spread_bps": 10.0,
                    "dolvol_20d_log": np.log1p(1e7),
                    "target_oc_up": int(c > o),
                })
        return pd.DataFrame(rows)

    def test_returns_backtest_result(self):
        df = self._make_test_df()
        cfg = BacktestConfig(top_n=2, leverage=1.0, initial_cash=10_000.0)
        result = simulate(df, _DummyModel(), cfg)
        assert result is not None
        assert result.total_trades > 0

    def test_top_n_respected(self):
        df = self._make_test_df(n_syms=10)
        cfg = BacktestConfig(top_n=3, leverage=1.0, initial_cash=10_000.0)
        result = simulate(df, _DummyModel(), cfg)
        for dr in result.day_results:
            assert len(dr.trades) <= 3

    def test_leverage_doubles_net(self):
        """2x leverage should give ~2x net return per trade (with zero commission).
        gross_return_pct is the unleveraged asset return; net_return_pct applies leverage.
        With commission=0, net ≈ leverage * gross_oc.
        """
        df = self._make_test_df(n_days=20, n_syms=5)
        cfg1 = BacktestConfig(top_n=1, leverage=1.0, commission_bps=0.0,
                               initial_cash=10_000.0)
        cfg2 = BacktestConfig(top_n=1, leverage=2.0, commission_bps=0.0,
                               initial_cash=10_000.0)
        r1 = simulate(df, _DummyModel(), cfg1)
        r2 = simulate(df, _DummyModel(), cfg2)
        n1 = sum(t.net_return_pct for dr in r1.day_results for t in dr.trades)
        n2 = sum(t.net_return_pct for dr in r2.day_results for t in dr.trades)
        assert n1 != 0, "Need non-zero returns to test leverage"
        assert abs(n2 / n1 - 2.0) < 0.2, f"Expected ~2x leverage, got {n2/n1:.2f}x"

    def test_net_less_than_gross(self):
        """Net return must always be ≤ gross return (costs subtracted)."""
        df = self._make_test_df(n_syms=5)
        cfg = BacktestConfig(top_n=2, leverage=1.0, commission_bps=10.0,
                              initial_cash=10_000.0)
        result = simulate(df, _DummyModel(), cfg)
        for dr in result.day_results:
            for t in dr.trades:
                assert t.net_return_pct <= t.gross_return_pct + 1e-6

    def test_equity_chain_consistent(self):
        """equity_start[D+1] should equal equity_end[D]."""
        df = self._make_test_df()
        cfg = BacktestConfig(top_n=2, leverage=1.0, initial_cash=10_000.0)
        result = simulate(df, _DummyModel(), cfg)
        days = result.day_results
        for i in range(1, len(days)):
            assert days[i].equity_start == pytest.approx(days[i - 1].equity_end, rel=1e-6)

    def test_empty_test_df_returns_empty_result(self):
        df = pd.DataFrame()
        cfg = BacktestConfig(top_n=2, initial_cash=10_000.0)
        # Should not raise; returns empty result
        # We need to supply required columns
        for col in DAILY_FEATURE_COLS + CHRONOS_FEATURE_COLS + \
                   ["date", "symbol", "actual_open", "actual_close", "spread_bps",
                    "dolvol_20d_log", "target_oc_up"]:
            df[col] = pd.Series([], dtype=float)
        result = simulate(df, _DummyModel(), cfg)
        assert result.total_trades == 0
        assert result.final_equity == cfg.initial_cash

    def test_fill_buffer_and_fee_reduce_realized_trade_return(self):
        df = pd.DataFrame(
            [
                {
                    **{col: 0.0 for col in DAILY_FEATURE_COLS},
                    **{col: 0.0 for col in CHRONOS_FEATURE_COLS},
                    "date": date(2026, 1, 5),
                    "symbol": "AAPL",
                    "actual_open": 100.0,
                    "actual_close": 101.0,
                    "spread_bps": 3.0,
                    "dolvol_20d_log": np.log1p(1e8),
                    "target_oc_up": 1,
                }
            ]
        )
        cfg = BacktestConfig(
            top_n=1,
            leverage=1.0,
            fee_rate=0.001,
            fill_buffer_bps=5.0,
            commission_bps=0.0,
        )
        result = simulate(df, _DummyModel(), cfg)
        trade = result.day_results[0].trades[0]

        assert trade.entry_fill_price == pytest.approx(100.05)
        assert trade.exit_fill_price == pytest.approx(100.9495)
        assert trade.gross_return_pct < 1.0
        assert trade.net_return_pct < trade.gross_return_pct
        assert result.avg_fee_bps == pytest.approx(10.0)

    def test_tie_break_prefers_higher_chronos_signal(self):
        rows = []
        for symbol, chronos_signal in [("SLOW", 1.0), ("FAST", 3.0)]:
            rows.append(
                {
                    **{col: 0.0 for col in DAILY_FEATURE_COLS},
                    **{col: 0.0 for col in CHRONOS_FEATURE_COLS},
                    "date": date(2026, 1, 5),
                    "symbol": symbol,
                    "actual_open": 100.0,
                    "actual_close": 101.0,
                    "spread_bps": 3.0,
                    "dolvol_20d_log": np.log1p(1e8),
                    "target_oc_up": 1,
                    "chronos_oc_return": chronos_signal,
                }
            )
        df = pd.DataFrame(rows)
        cfg = BacktestConfig(top_n=1, leverage=1.0, fee_rate=0.0, fill_buffer_bps=0.0, xgb_weight=1.0)
        result = simulate(df, _DummyModel(), cfg)

        assert result.day_results[0].trades[0].symbol == "FAST"

    def test_precomputed_scores_override_model_scoring(self):
        rows = []
        for symbol, score in [("LOW", 0.2), ("HIGH", 0.9)]:
            rows.append(
                {
                    **{col: 0.0 for col in DAILY_FEATURE_COLS},
                    **{col: 0.0 for col in CHRONOS_FEATURE_COLS},
                    "date": date(2026, 1, 5),
                    "symbol": symbol,
                    "actual_open": 100.0,
                    "actual_close": 101.0,
                    "spread_bps": 3.0,
                    "dolvol_20d_log": np.log1p(1e8),
                    "target_oc_up": 1,
                    "_custom_score": score,
                }
            )
        df = pd.DataFrame(rows)
        cfg = BacktestConfig(top_n=1, leverage=1.0, fee_rate=0.0, fill_buffer_bps=0.0)
        scores = pd.Series(df["_custom_score"].to_numpy(dtype=float), index=df.index)

        result = simulate(df, _DummyModel(), cfg, precomputed_scores=scores)

        assert result.day_results[0].trades[0].symbol == "HIGH"
