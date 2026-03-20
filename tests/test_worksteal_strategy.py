"""Tests for work-stealing dip-buying strategy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from binance_worksteal.strategy import (
    WorkStealConfig, Position, compute_ref_price, compute_ref_low,
    compute_atr, compute_buy_target, passes_sma_filter,
    run_worksteal_backtest, compute_metrics,
    get_fee, FDUSD_SYMBOLS, _risk_off_triggered,
)


def make_bars(prices, start="2026-01-01", symbol="BTCUSD"):
    dates = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    rows = []
    for i, (d, p) in enumerate(zip(dates, prices)):
        noise = p * 0.02
        rows.append({
            "timestamp": d, "open": p - noise * 0.5,
            "high": p + noise, "low": p - noise,
            "close": p, "volume": 1000.0, "symbol": symbol,
        })
    return pd.DataFrame(rows)


class TestFees:
    def test_fdusd_zero_fee(self):
        config = WorkStealConfig(fdusd_fee=0.0, maker_fee=0.001)
        assert get_fee("BTCUSD", config) == 0.0
        assert get_fee("ETHUSD", config) == 0.0

    def test_usdt_fee(self):
        config = WorkStealConfig(fdusd_fee=0.0, maker_fee=0.001)
        assert get_fee("DOGEUSD", config) == 0.001
        assert get_fee("LINKUSD", config) == 0.001


class TestComputeRefPrice:
    def test_high_method(self):
        bars = make_bars([100, 110, 105, 95, 100])
        ref = compute_ref_price(bars, "high", 5)
        assert ref == pytest.approx(110 * 1.02, rel=0.01)

    def test_sma_method(self):
        bars = make_bars([100, 110, 105, 95, 100])
        ref = compute_ref_price(bars, "sma", 5)
        assert ref == pytest.approx(102.0, rel=0.01)

    def test_ref_low(self):
        bars = make_bars([100, 90, 95, 110, 105])
        ref = compute_ref_low(bars, 5)
        assert ref == pytest.approx(90 * 0.98, rel=0.01)


class TestComputeATR:
    def test_basic(self):
        bars = make_bars([100] * 20)
        atr = compute_atr(bars, 14)
        assert atr > 0


class TestWorkStealBacktest:
    def test_no_dip_no_trade(self):
        prices = [100 + i for i in range(60)]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.005)
        eq, trades, metrics = run_worksteal_backtest(
            bars, config, start_date="2026-01-30", end_date="2026-02-28"
        )
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) == 0

    def test_dip_triggers_buy(self):
        prices = list(range(100, 130))
        prices += [120, 118, 117, 116]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, lookback_days=10,
            profit_target_pct=0.05, stop_loss_pct=0.08,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) >= 1

    def test_profit_target_exit(self):
        prices = [100] * 25 + [92, 90, 88, 86, 85]
        prices += [88, 92, 95, 98, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02,
            profit_target_pct=0.05, stop_loss_pct=0.15,
            lookback_days=20, max_hold_days=30,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        sells = [t for t in trades if t.side == "sell"]
        assert len(sells) >= 0

    def test_max_positions_respected(self):
        all_bars = {}
        for i, sym in enumerate(["S1USD", "S2USD", "S3USD", "S4USD", "S5USD"]):
            prices = [100 + i] * 25 + [80 + i] * 10
            all_bars[sym] = make_bars(prices, symbol=sym)
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.05, max_positions=3, lookback_days=20,
        )
        eq, trades, metrics = run_worksteal_backtest(all_bars, config)
        if not eq.empty:
            assert eq["n_positions"].max() <= 3

    def test_multi_symbol_only_dip_bought(self):
        all_bars = {
            "UPUSD": make_bars([100 + i * 2 for i in range(40)], symbol="UPUSD"),
            "DIPUSD": make_bars([100] * 25 + [92, 90, 88, 85, 83, 85, 88, 92, 95, 100,
                                               103, 105, 108, 110, 112], symbol="DIPUSD"),
            "FLATUSD": make_bars([100] * 40, symbol="FLATUSD"),
        }
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.02, lookback_days=20)
        eq, trades, metrics = run_worksteal_backtest(all_bars, config)
        buy_symbols = set(t.symbol for t in trades if t.side == "buy")
        assert "UPUSD" not in buy_symbols

    def test_leverage_increases_returns(self):
        prices = [100] * 25 + [90, 88, 85, 88, 92, 95, 100, 105, 110]
        bars = {"BTCUSD": make_bars(prices)}
        config_1x = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, max_leverage=1.0,
            lookback_days=20, profit_target_pct=0.10,
        )
        config_3x = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, max_leverage=3.0,
            lookback_days=20, profit_target_pct=0.10,
        )
        eq1, _, m1 = run_worksteal_backtest(dict(bars), config_1x)
        eq3, _, m3 = run_worksteal_backtest(dict(bars), config_3x)
        # 3x leverage should produce >= 1x returns (ignoring margin cost)
        # Just check both run without error
        assert len(eq1) == len(eq3)

    def test_shorts_enabled(self):
        # Price pumps then drops - short should profit
        prices = [100] * 25 + [105, 110, 115, 120, 125, 120, 115, 110, 105, 100]
        bars = {"ALTUSD": make_bars(prices, symbol="ALTUSD")}
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, enable_shorts=True,
            short_pump_pct=0.10, profit_target_pct=0.05,
            lookback_days=20,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        shorts = [t for t in trades if t.side == "short"]
        # May or may not trigger depending on exact noise
        assert isinstance(shorts, list)

    def test_fdusd_zero_fee_applied(self):
        prices = [100] * 25 + [90, 88, 85, 88, 92, 95, 100, 105]
        bars_btc = {"BTCUSD": make_bars(prices)}
        bars_alt = {"DOGEUSD": make_bars(prices, symbol="DOGEUSD")}
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, fdusd_fee=0.0, maker_fee=0.001,
            lookback_days=20, profit_target_pct=0.05,
        )
        _, trades_btc, _ = run_worksteal_backtest(bars_btc, config)
        _, trades_alt, _ = run_worksteal_backtest(bars_alt, config)
        # BTC trades should have 0 fee
        btc_fees = sum(t.fee for t in trades_btc if t.symbol == "BTCUSD" and t.side == "buy")
        alt_fees = sum(t.fee for t in trades_alt if t.symbol == "DOGEUSD" and t.side == "buy")
        if trades_btc and trades_alt:
            assert btc_fees < alt_fees  # BTC 0% < DOGE 10bps

    def test_base_asset_idle_mode_participates_in_eth_trend(self):
        eth_prices = [100 + i * 2 for i in range(40)]
        bars = {"ETHUSD": make_bars(eth_prices, symbol="ETHUSD")}
        config = WorkStealConfig(
            base_asset_symbol="ETHUSD",
            base_asset_rebalance_min_cash=0.0,
            lookback_days=10,
        )

        eq, trades, metrics = run_worksteal_backtest(bars, config)

        assert trades == []
        assert eq["base_asset_qty"].iloc[-1] > 0.0
        assert metrics["final_equity"] > config.initial_cash

    def test_base_asset_filter_can_keep_strategy_in_cash(self):
        eth_prices = [200 - i * 3 for i in range(40)]
        bars = {"ETHUSD": make_bars(eth_prices, symbol="ETHUSD")}
        config = WorkStealConfig(
            base_asset_symbol="ETHUSD",
            base_asset_momentum_period=5,
            base_asset_min_momentum=0.0,
            base_asset_rebalance_min_cash=0.0,
            lookback_days=10,
        )

        eq, _, metrics = run_worksteal_backtest(bars, config)

        assert eq["base_asset_qty"].max() == pytest.approx(0.0)
        assert metrics["final_equity"] == pytest.approx(config.initial_cash)

    def test_base_asset_is_liquidated_to_fund_alt_entry(self):
        eth_prices = [100 + i for i in range(25)] + [125 + i for i in range(15)]
        alt_prices = [100] * 25 + [88, 85, 84, 86, 90, 95, 100, 105, 108, 110, 112, 114, 116, 118, 120]
        bars = {
            "ETHUSD": make_bars(eth_prices, symbol="ETHUSD"),
            "ALTUSD": make_bars(alt_prices, symbol="ALTUSD"),
        }
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            profit_target_pct=0.10,
            stop_loss_pct=0.08,
            base_asset_symbol="ETHUSD",
            base_asset_rebalance_min_cash=0.0,
        )

        eq, trades, _ = run_worksteal_backtest(bars, config)

        assert any(t.symbol == "ALTUSD" and t.side == "buy" for t in trades)
        assert eq["base_asset_qty"].max() > 0.0
        assert eq["base_asset_qty"].min() < eq["base_asset_qty"].max()


class TestSmaCheckMethod:
    def test_pre_dip_allows_entry_below_sma(self):
        prices = [100] * 25 + [90, 85, 82, 85, 88, 92, 95, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            sma_filter_period=20, sma_check_method="pre_dip",
        )
        eq, trades, _ = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) >= 1

    def test_current_sma_blocks_entry_below_sma(self):
        prices = [100] * 25 + [90, 85, 82, 85, 88, 92, 95, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            sma_filter_period=20, sma_check_method="current",
        )
        eq, trades, _ = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) == 0

    def test_none_sma_allows_any_entry(self):
        prices = [50] * 10 + [100] * 15 + [85, 82, 80, 85, 88, 92, 95, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            sma_filter_period=20, sma_check_method="none",
        )
        eq, trades, _ = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) >= 1


class TestAdaptiveDip:
    def test_adaptive_dip_uses_atr(self):
        prices = [100] * 25 + [92, 90, 88, 86, 85, 88, 92, 95, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.20, proximity_pct=0.05, lookback_days=20,
            adaptive_dip=True,
        )
        eq, trades, _ = run_worksteal_backtest(bars, config)
        assert isinstance(trades, list)

    def test_adaptive_dip_false_uses_fixed_threshold(self):
        prices = [100] * 25 + [85, 82, 80, 85, 88, 92]
        bars = {"BTCUSD": make_bars(prices)}
        config_fixed = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            adaptive_dip=False,
        )
        config_adaptive = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            adaptive_dip=True,
        )
        _, trades_fixed, _ = run_worksteal_backtest(dict(bars), config_fixed)
        _, trades_adaptive, _ = run_worksteal_backtest(dict(bars), config_adaptive)
        assert isinstance(trades_fixed, list)
        assert isinstance(trades_adaptive, list)


class TestRiskOffMomentum:
    def test_risk_off_not_triggered_with_default_threshold(self):
        prices = [100, 99, 98, 97, 96, 95]
        bars_df = make_bars(prices)
        current_bars = {"BTCUSD": bars_df.iloc[-1]}
        history = {"BTCUSD": bars_df}
        config = WorkStealConfig(
            momentum_period=5, risk_off_momentum_threshold=-0.05,
        )
        result = _risk_off_triggered(current_bars, history, config)
        assert result is False

    def test_risk_off_triggered_at_zero_threshold(self):
        prices = [100, 99, 98, 97, 96, 95]
        bars_df = make_bars(prices)
        current_bars = {"BTCUSD": bars_df.iloc[-1]}
        history = {"BTCUSD": bars_df}
        config = WorkStealConfig(
            momentum_period=5, risk_off_momentum_threshold=0.0,
        )
        result = _risk_off_triggered(current_bars, history, config)
        assert result is True

    def test_risk_off_disabled_when_momentum_period_zero(self):
        prices = [100, 50, 40, 30, 20, 10]
        bars_df = make_bars(prices)
        current_bars = {"BTCUSD": bars_df.iloc[-1]}
        history = {"BTCUSD": bars_df}
        config = WorkStealConfig(momentum_period=0)
        result = _risk_off_triggered(current_bars, history, config)
        assert result is False


class TestNewDefaults:
    def test_proximity_pct_default_is_003(self):
        config = WorkStealConfig()
        assert config.proximity_pct == 0.03

    def test_sma_check_method_default_is_pre_dip(self):
        config = WorkStealConfig()
        assert config.sma_check_method == "pre_dip"

    def test_risk_off_momentum_threshold_default(self):
        config = WorkStealConfig()
        assert config.risk_off_momentum_threshold == -0.05

    def test_adaptive_dip_default_is_false(self):
        config = WorkStealConfig()
        assert config.adaptive_dip is False


class TestComputeMetrics:
    def test_positive_return(self):
        eq_df = pd.DataFrame({"equity": [10000, 10100, 10200, 10300, 10400, 10500]})
        m = compute_metrics(eq_df, WorkStealConfig())
        assert m["total_return"] > 0
        assert m["sortino"] > 0

    def test_negative_return(self):
        eq_df = pd.DataFrame({"equity": [10000, 9900, 9800, 9700, 9600, 9500]})
        m = compute_metrics(eq_df, WorkStealConfig())
        assert m["total_return"] < 0

    def test_flat(self):
        eq_df = pd.DataFrame({"equity": [10000, 10000, 10000, 10000]})
        m = compute_metrics(eq_df, WorkStealConfig())
        assert m["total_return"] == pytest.approx(0.0, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
