"""Tests for Chronos2 forecast integration in work-stealing candidate scoring."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from binance_worksteal.forecast_integration import (
    get_forecast_multiplier,
    load_daily_forecasts,
)
from binance_worksteal.strategy import WorkStealConfig, run_worksteal_backtest


def _make_bars(symbol: str, prices: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="1D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": [1e6] * len(prices),
        "symbol": symbol,
    })


def _make_forecast_df(symbol: str, dates: list, predicted_closes: list[float]) -> pd.DataFrame:
    ts_list = []
    for d in dates:
        t = pd.Timestamp(d)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        ts_list.append(t)
    return pd.DataFrame({
        "issued_at": ts_list,
        "predicted_close_p50": predicted_closes,
        "predicted_close_p10": [p * 0.95 for p in predicted_closes],
        "predicted_close_p90": [p * 1.05 for p in predicted_closes],
        "predicted_high_p50": [p * 1.03 for p in predicted_closes],
        "predicted_low_p50": [p * 0.97 for p in predicted_closes],
    })


def _dip_then_recover(symbol: str, n_flat: int = 25, dip_pct: float = 0.22,
                       recover_pct: float = 0.18) -> pd.DataFrame:
    base = 100.0
    prices = [base] * n_flat
    prices.append(base * (1 - dip_pct))
    for i in range(1, 15):
        prices.append(base * (1 - dip_pct) * (1 + recover_pct * i / 14))
    return _make_bars(symbol, prices)


class TestForecastMultiplier:
    def test_bullish(self):
        fc = {"BTCUSD": _make_forecast_df("BTCUSD", ["2025-01-25"], [110.0])}
        m = get_forecast_multiplier("BTCUSD", pd.Timestamp("2025-01-25", tz="UTC"), fc, 100.0)
        assert abs(m - 0.10) < 1e-6

    def test_bearish(self):
        fc = {"BTCUSD": _make_forecast_df("BTCUSD", ["2025-01-25"], [90.0])}
        m = get_forecast_multiplier("BTCUSD", pd.Timestamp("2025-01-25", tz="UTC"), fc, 100.0)
        assert abs(m - (-0.10)) < 1e-6

    def test_neutral(self):
        fc = {"BTCUSD": _make_forecast_df("BTCUSD", ["2025-01-25"], [100.0])}
        m = get_forecast_multiplier("BTCUSD", pd.Timestamp("2025-01-25", tz="UTC"), fc, 100.0)
        assert abs(m) < 1e-6

    def test_missing_symbol(self):
        fc = {"ETHUSD": _make_forecast_df("ETHUSD", ["2025-01-25"], [3000.0])}
        m = get_forecast_multiplier("BTCUSD", pd.Timestamp("2025-01-25", tz="UTC"), fc, 100.0)
        assert m == 0.0

    def test_empty_forecasts(self):
        m = get_forecast_multiplier("BTCUSD", pd.Timestamp("2025-01-25", tz="UTC"), {}, 100.0)
        assert m == 0.0

    def test_zero_close(self):
        fc = {"BTCUSD": _make_forecast_df("BTCUSD", ["2025-01-25"], [110.0])}
        m = get_forecast_multiplier("BTCUSD", pd.Timestamp("2025-01-25", tz="UTC"), fc, 0.0)
        assert m == 0.0

    def test_date_before_forecast(self):
        fc = {"BTCUSD": _make_forecast_df("BTCUSD", ["2025-02-01"], [110.0])}
        m = get_forecast_multiplier("BTCUSD", pd.Timestamp("2025-01-25", tz="UTC"), fc, 100.0)
        assert m == 0.0

    def test_uses_latest_eligible(self):
        fc = {"BTCUSD": _make_forecast_df(
            "BTCUSD",
            ["2025-01-20", "2025-01-25", "2025-01-30"],
            [105.0, 110.0, 120.0],
        )}
        m = get_forecast_multiplier("BTCUSD", pd.Timestamp("2025-01-26", tz="UTC"), fc, 100.0)
        assert abs(m - 0.10) < 1e-6


class TestCandidateReorderingWithForecast:
    def _make_two_symbol_scenario(self):
        bars_a = _dip_then_recover("ASYMUSD")
        bars_b = _dip_then_recover("BSYMUSD")
        return {"ASYMUSD": bars_a, "BSYMUSD": bars_b}

    def test_zero_weight_identical_ordering(self):
        all_bars = self._make_two_symbol_scenario()
        config = WorkStealConfig(
            dip_pct=0.20, profit_target_pct=0.15, stop_loss_pct=0.10,
            sma_filter_period=0, trailing_stop_pct=0.0,
            max_positions=5, max_hold_days=14, lookback_days=20,
            initial_cash=10000.0, max_leverage=1.0,
            max_drawdown_exit=0.0, forecast_bias_weight=0.0,
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
        )
        start = str(all_bars["ASYMUSD"]["timestamp"].min().date())
        end = str(all_bars["ASYMUSD"]["timestamp"].max().date())

        fc = {
            "ASYMUSD": _make_forecast_df(
                "ASYMUSD",
                pd.date_range("2025-01-01", periods=40, freq="1D", tz="UTC").tolist(),
                [200.0] * 40,
            ),
            "BSYMUSD": _make_forecast_df(
                "BSYMUSD",
                pd.date_range("2025-01-01", periods=40, freq="1D", tz="UTC").tolist(),
                [50.0] * 40,
            ),
        }

        eq_no_fc, trades_no_fc, m_no_fc = run_worksteal_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=start, end_date=end,
        )
        eq_fc, trades_fc, m_fc = run_worksteal_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=start, end_date=end, forecast_data=fc,
        )
        assert len(trades_no_fc) == len(trades_fc)
        for t1, t2 in zip(trades_no_fc, trades_fc):
            assert t1.symbol == t2.symbol
            assert t1.side == t2.side
            assert abs(t1.price - t2.price) < 1e-6

    def test_forecast_bias_reorders(self):
        base = 100.0
        prices_a = [base] * 25
        prices_a.append(base * 0.78)  # 22% dip
        for i in range(1, 15):
            prices_a.append(base * 0.78 * (1 + 0.18 * i / 14))

        prices_b = [base] * 25
        prices_b.append(base * 0.78)  # identical dip
        for i in range(1, 15):
            prices_b.append(base * 0.78 * (1 + 0.18 * i / 14))

        all_bars = {
            "ASYMUSD": _make_bars("ASYMUSD", prices_a),
            "BSYMUSD": _make_bars("BSYMUSD", prices_b),
        }

        fc_dates = pd.date_range("2025-01-01", periods=40, freq="1D", tz="UTC").tolist()

        fc_bullish_a = {
            "ASYMUSD": _make_forecast_df("ASYMUSD", fc_dates, [120.0] * 40),
            "BSYMUSD": _make_forecast_df("BSYMUSD", fc_dates, [60.0] * 40),
        }

        config = WorkStealConfig(
            dip_pct=0.20, profit_target_pct=0.15, stop_loss_pct=0.10,
            sma_filter_period=0, trailing_stop_pct=0.0,
            max_positions=1, max_hold_days=14, lookback_days=20,
            initial_cash=10000.0, max_leverage=1.0,
            max_drawdown_exit=0.0, forecast_bias_weight=2.0,
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
        )
        start = str(all_bars["ASYMUSD"]["timestamp"].min().date())
        end = str(all_bars["ASYMUSD"]["timestamp"].max().date())

        _, trades, _ = run_worksteal_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=start, end_date=end, forecast_data=fc_bullish_a,
        )
        buys = [t for t in trades if t.side == "buy"]
        if buys:
            assert buys[0].symbol == "ASYMUSD"

    def test_no_forecast_data_unchanged(self):
        all_bars = self._make_two_symbol_scenario()
        config = WorkStealConfig(
            dip_pct=0.20, profit_target_pct=0.15, stop_loss_pct=0.10,
            sma_filter_period=0, trailing_stop_pct=0.0,
            max_positions=5, max_hold_days=14, lookback_days=20,
            initial_cash=10000.0, max_leverage=1.0,
            max_drawdown_exit=0.0, forecast_bias_weight=5.0,
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
        )
        start = str(all_bars["ASYMUSD"]["timestamp"].min().date())
        end = str(all_bars["ASYMUSD"]["timestamp"].max().date())

        eq1, trades1, m1 = run_worksteal_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=start, end_date=end, forecast_data=None,
        )
        eq2, trades2, m2 = run_worksteal_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=start, end_date=end,
        )
        assert len(trades1) == len(trades2)
        for t1, t2 in zip(trades1, trades2):
            assert t1.symbol == t2.symbol


class TestLoadDailyForecasts:
    def test_nonexistent_dir(self):
        result = load_daily_forecasts(["BTCUSD"], cache_dir="/tmp/nonexistent_fc_dir_xyz")
        assert result == {}

    def test_load_from_real_cache(self):
        cache_dir = "/home/lee/code/stock/binanceneural/forecast_cache/h24/"
        p = Path(cache_dir)
        if not p.exists():
            return
        fc = load_daily_forecasts(["BTCUSD", "ETHUSD"], cache_dir=cache_dir)
        for sym in fc:
            assert "predicted_close_p50" in fc[sym].columns
            assert len(fc[sym]) > 0
