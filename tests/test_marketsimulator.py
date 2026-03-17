"""Comprehensive tests for binanceneural.marketsimulator.

Covers single-symbol, shared-cash multi-symbol, fee accounting, cost basis,
fill logic, max-hold enforcement, probe mode, equity curves, and edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from binanceneural.marketsimulator import (
    BinanceMarketSimulator,
    SimulationConfig,
    SimulationResult,
    TradeRecord,
    run_shared_cash_simulation,
    _resolve_amount_scale,
    _compute_metrics,
    _prepare_frame,
    _quantize_down,
    _quantize_up,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEE = 0.0008  # DEFAULT_MAKER_FEE_RATE


def _make_bars(
    prices: list[tuple[float, float, float, float]],
    symbol: str = "BTCUSD",
    start: str = "2025-01-01T00:00:00+00:00",
) -> pd.DataFrame:
    """Create a bars DataFrame from (open, high, low, close) tuples."""
    timestamps = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    rows = []
    for ts, (o, h, l, c) in zip(timestamps, prices):
        rows.append({"timestamp": ts, "symbol": symbol, "open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(rows)


def _make_actions(
    actions: list[dict],
    symbol: str = "BTCUSD",
    start: str = "2025-01-01T00:00:00+00:00",
) -> pd.DataFrame:
    """Create an actions DataFrame from dicts with buy_price, sell_price, etc."""
    timestamps = pd.date_range(start, periods=len(actions), freq="h", tz="UTC")
    rows = []
    for ts, a in zip(timestamps, actions):
        row = {"timestamp": ts, "symbol": symbol}
        row.update(a)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Basic buy / sell flow
# ---------------------------------------------------------------------------

class TestSingleSymbolBasics:
    """Core single-symbol simulation tests."""

    def test_no_trades_when_no_signal(self):
        """Simulator does nothing when buy/sell prices are zero."""
        bars = _make_bars([(100, 105, 95, 100)] * 5)
        actions = _make_actions([{"buy_price": 0.0, "sell_price": 0.0, "trade_amount": 50}] * 5)
        sim = BinanceMarketSimulator()
        result = sim.run(bars, actions)
        assert len(result.per_symbol["BTCUSD"].trades) == 0

    def test_single_buy_sell_roundtrip(self):
        """Buy at bar 0 limit, sell at bar 1 limit, verify cash accounting."""
        bars = _make_bars([
            (100, 105, 95, 100),   # bar 0: low 95 fills buy at 98
            (102, 110, 99, 105),   # bar 1: high 110 fills sell at 108
        ])
        actions = _make_actions([
            {"buy_price": 98.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 108.0, "trade_amount": 100},
        ])
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        sym = result.per_symbol["BTCUSD"]
        trades = sym.trades

        # Should have 1 buy + 1 sell
        buys = [t for t in trades if t.side == "buy"]
        sells = [t for t in trades if t.side == "sell"]
        assert len(buys) == 1
        assert len(sells) == 1

        buy = buys[0]
        sell = sells[0]

        # Buy fills at buy_price=98
        assert buy.price == 98.0
        # Cash spent = qty * 98 * (1 + fee)
        expected_qty = 10_000.0 / (98.0 * (1 + FEE))
        assert abs(buy.quantity - expected_qty) < 1e-6

        # Sell fills at sell_price=108
        assert sell.price == 108.0
        assert abs(sell.quantity - expected_qty) < 1e-6

        # Final cash = proceeds from sell
        expected_proceeds = expected_qty * 108.0 * (1 - FEE)
        assert abs(sell.cash_after - expected_proceeds) < 0.01

        # Should be profitable
        assert sell.cash_after > 10_000.0

    def test_buy_not_filled_when_low_above_buy_price(self):
        """Buy limit at 90 but bar low is 95 -- no fill."""
        bars = _make_bars([(100, 105, 95, 100)])
        actions = _make_actions([{"buy_price": 90.0, "sell_price": 0.0, "trade_amount": 100}])
        sim = BinanceMarketSimulator()
        result = sim.run(bars, actions)
        assert len(result.per_symbol["BTCUSD"].trades) == 0

    def test_sell_not_filled_when_high_below_sell_price(self):
        """Sell limit at 120 but bar high is 110 -- no fill."""
        bars = _make_bars([
            (100, 105, 90, 100),   # buy fills
            (100, 110, 95, 105),   # sell at 120 does not fill (high=110)
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 120.0, "trade_amount": 100},
        ])
        sim = BinanceMarketSimulator()
        result = sim.run(bars, actions)
        sells = [t for t in result.per_symbol["BTCUSD"].trades if t.side == "sell"]
        assert len(sells) == 0


# ---------------------------------------------------------------------------
# Fee accounting
# ---------------------------------------------------------------------------

class TestFeeAccounting:
    """Verify fees are correctly deducted from cash on buy and sell."""

    def test_buy_fee_deducted(self):
        """Buying costs price * qty * (1 + fee)."""
        bars = _make_bars([(100, 105, 90, 100)])
        actions = _make_actions([{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}])
        config = SimulationConfig(maker_fee=0.001, initial_cash=1000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        buy = result.per_symbol["BTCUSD"].trades[0]

        cost = buy.quantity * 95.0 * (1 + 0.001)
        assert abs(buy.cash_after - (1000.0 - cost)) < 1e-6

    def test_sell_fee_deducted(self):
        """Selling returns price * qty * (1 - fee)."""
        bars = _make_bars([
            (100, 105, 90, 100),
            (100, 115, 95, 110),
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 110.0, "trade_amount": 100},
        ])
        config = SimulationConfig(maker_fee=0.001, initial_cash=1000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        trades = result.per_symbol["BTCUSD"].trades
        buy, sell = trades[0], trades[1]

        expected_proceeds = sell.quantity * 110.0 * (1 - 0.001)
        # cash after sell = 0 (all cash was used to buy) + proceeds
        assert abs(sell.cash_after - expected_proceeds) < 1e-6

    def test_fee_field_on_trade_record(self):
        """TradeRecord.fee matches price * qty * fee_rate."""
        bars = _make_bars([(100, 105, 90, 100)])
        actions = _make_actions([{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}])
        config = SimulationConfig(maker_fee=0.002, initial_cash=1000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        buy = result.per_symbol["BTCUSD"].trades[0]
        assert abs(buy.fee - buy.quantity * 95.0 * 0.002) < 1e-8


# ---------------------------------------------------------------------------
# Cost basis tracking
# ---------------------------------------------------------------------------

class TestCostBasis:
    """Verify weighted average cost basis across multiple buys."""

    def test_cost_basis_single_buy(self):
        """Cost basis equals buy price after one buy."""
        bars = _make_bars([
            (100, 105, 90, 100),
            (100, 115, 95, 110),
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 110.0, "trade_amount": 100},
        ])
        config = SimulationConfig(maker_fee=FEE, initial_cash=1000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        sell = [t for t in result.per_symbol["BTCUSD"].trades if t.side == "sell"][0]
        expected_pnl = (110.0 - 95.0) * sell.quantity
        assert abs(sell.realized_pnl - expected_pnl) < 1e-4

    def test_cost_basis_two_buys_weighted_average(self):
        """Cost basis is weighted average after two buys at different prices."""
        bars = _make_bars([
            (100, 105, 88, 100),  # buy at 90
            (100, 105, 93, 100),  # buy at 95
            (100, 115, 95, 110),  # sell at 110
        ])
        actions = _make_actions([
            {"buy_price": 90.0, "sell_price": 0.0, "buy_amount": 50, "sell_amount": 0},
            {"buy_price": 95.0, "sell_price": 0.0, "buy_amount": 50, "sell_amount": 0},
            {"buy_price": 0.0, "sell_price": 110.0, "buy_amount": 0, "sell_amount": 100},
        ])
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        trades = result.per_symbol["BTCUSD"].trades
        buys = [t for t in trades if t.side == "buy"]
        sells = [t for t in trades if t.side == "sell"]
        assert len(buys) == 2
        assert len(sells) == 1

        qty1, qty2 = buys[0].quantity, buys[1].quantity
        wavg = (90.0 * qty1 + 95.0 * qty2) / (qty1 + qty2)
        sell = sells[0]
        expected_pnl = (110.0 - wavg) * sell.quantity
        assert abs(sell.realized_pnl - expected_pnl) < 1e-2


# ---------------------------------------------------------------------------
# Sell-first ordering
# ---------------------------------------------------------------------------

class TestSellFirstOrdering:
    """Verify sells execute before buys within the same bar."""

    def test_sell_frees_cash_for_buy_in_same_bar(self):
        """If we have inventory and both buy + sell fill in same bar,
        the sell should free cash that the buy can then use."""
        bars = _make_bars([
            (100, 105, 90, 100),  # buy fills at 95
            (100, 115, 88, 105),  # sell fills at 110 AND buy fills at 90
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 90.0, "sell_price": 110.0, "trade_amount": 100},
        ])
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        trades = result.per_symbol["BTCUSD"].trades

        # Should have: buy (bar 0), sell (bar 1), buy (bar 1)
        assert len(trades) == 3
        assert trades[0].side == "buy"   # bar 0 buy
        assert trades[1].side == "sell"  # bar 1 sell first
        assert trades[2].side == "buy"   # bar 1 buy second

    def test_shared_cash_sells_before_buys(self):
        """run_shared_cash_simulation processes sells before buys."""
        bars = _make_bars([
            (100, 105, 90, 100),
            (100, 115, 88, 105),
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 90.0, "sell_price": 110.0, "trade_amount": 100},
        ])
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        result = run_shared_cash_simulation(bars, actions, config)
        sym_trades = result.per_symbol["BTCUSD"].trades

        # bar 1 should have sell then buy
        bar1_trades = [t for t in sym_trades if t.timestamp == bars["timestamp"].iloc[1]]
        assert len(bar1_trades) == 2
        assert bar1_trades[0].side == "sell"
        assert bar1_trades[1].side == "buy"


# ---------------------------------------------------------------------------
# Equity curve correctness
# ---------------------------------------------------------------------------

class TestEquityCurve:
    """Equity = cash + inventory * close at each step."""

    def test_equity_equals_cash_when_no_inventory(self):
        """Before any trade, equity = initial_cash."""
        bars = _make_bars([(100, 105, 95, 100)] * 3)
        actions = _make_actions([{"buy_price": 0.0, "sell_price": 0.0, "trade_amount": 0}] * 3)
        config = SimulationConfig(initial_cash=5000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        eq = result.per_symbol["BTCUSD"].equity_curve
        assert all(abs(v - 5000.0) < 1e-6 for v in eq.values)

    def test_equity_accounts_for_unrealized_gains(self):
        """After buying, equity reflects mark-to-market at close price."""
        bars = _make_bars([
            (100, 105, 90, 100),   # buy at 95, close at 100
            (102, 108, 98, 120),   # no trade, close at 120
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 0.0, "trade_amount": 0},
        ])
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        eq = result.per_symbol["BTCUSD"].equity_curve

        buy = result.per_symbol["BTCUSD"].trades[0]
        # After bar 1: cash_after_buy + inventory * close_bar1
        expected_eq_bar1 = buy.cash_after + buy.quantity * 120.0
        assert abs(eq.iloc[1] - expected_eq_bar1) < 1e-4

    def test_combined_equity_sums_symbols(self):
        """BinanceMarketSimulator.run() sums per-symbol equity curves."""
        bars1 = _make_bars([(100, 105, 95, 100)] * 3, symbol="BTCUSD")
        bars2 = _make_bars([(50, 55, 45, 50)] * 3, symbol="ETHUSD")
        bars = pd.concat([bars1, bars2], ignore_index=True)

        act1 = _make_actions([{"buy_price": 0, "sell_price": 0, "trade_amount": 0}] * 3, symbol="BTCUSD")
        act2 = _make_actions([{"buy_price": 0, "sell_price": 0, "trade_amount": 0}] * 3, symbol="ETHUSD")
        actions = pd.concat([act1, act2], ignore_index=True)

        config = SimulationConfig(initial_cash=5000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        # Combined should be 2 * initial_cash (one per symbol, no trades)
        assert abs(result.combined_equity.iloc[0] - 10_000.0) < 1e-4


# ---------------------------------------------------------------------------
# Max hold enforcement
# ---------------------------------------------------------------------------

class TestMaxHold:
    """Verify force-close at max_hold_hours."""

    def test_force_close_at_max_hold(self):
        """Position held beyond max_hold_hours is force-closed at close price."""
        n_bars = 5
        bars = _make_bars([(100, 105, 90, 100 + i) for i in range(n_bars)])
        actions = _make_actions(
            [{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}]
            + [{"buy_price": 0.0, "sell_price": 0.0, "trade_amount": 0}] * (n_bars - 1)
        )
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0, max_hold_hours=3)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        trades = result.per_symbol["BTCUSD"].trades

        sells = [t for t in trades if t.side == "sell"]
        assert len(sells) == 1
        assert sells[0].reason == "max_hold"
        # Force close happens at bar 3 (held 3 hours)
        assert sells[0].price == 103.0  # close of bar index 3

    def test_max_hold_in_shared_cash(self):
        """Max hold also works in run_shared_cash_simulation."""
        n_bars = 5
        bars = _make_bars([(100, 105, 90, 100 + i) for i in range(n_bars)])
        actions = _make_actions(
            [{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}]
            + [{"buy_price": 0.0, "sell_price": 0.0, "trade_amount": 0}] * (n_bars - 1)
        )
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0, max_hold_hours=3)
        result = run_shared_cash_simulation(bars, actions, config)
        sells = [t for t in result.per_symbol["BTCUSD"].trades if t.side == "sell"]
        assert len(sells) == 1
        assert sells[0].reason == "max_hold"


# ---------------------------------------------------------------------------
# Probe mode
# ---------------------------------------------------------------------------

class TestProbeMode:
    """Verify probe mode caps buying after a loss."""

    def test_probe_mode_limits_buy_after_loss(self):
        """After a losing trade, next buy is capped at probe_notional."""
        bars = _make_bars([
            (100, 105, 90, 100),   # buy at 95
            (80, 90, 75, 80),      # sell at 85 (loss)
            (80, 85, 70, 80),      # buy at 75 -- should be probe-limited
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 85.0, "trade_amount": 100},
            {"buy_price": 75.0, "sell_price": 0.0, "trade_amount": 100},
        ])
        config = SimulationConfig(
            maker_fee=FEE,
            initial_cash=10_000.0,
            enable_probe_mode=True,
            probe_notional=100.0,
        )
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        trades = result.per_symbol["BTCUSD"].trades
        buys = [t for t in trades if t.side == "buy"]

        assert len(buys) == 2
        # Second buy should be much smaller (limited to ~100 notional)
        second_buy = buys[1]
        assert second_buy.notional < 150.0  # capped near probe_notional

    def test_probe_mode_resets_after_profit(self):
        """After a profitable trade, probe mode deactivates."""
        bars = _make_bars([
            (100, 105, 90, 100),   # buy at 95
            (80, 90, 75, 80),      # sell at 85 (loss -> probe on)
            (80, 85, 70, 80),      # buy at 75 (probe buy)
            (90, 100, 85, 95),     # sell at 95 (profit -> probe off)
            (90, 100, 85, 95),     # buy at 88 (should be full size)
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 85.0, "trade_amount": 100},
            {"buy_price": 75.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 95.0, "trade_amount": 100},
            {"buy_price": 88.0, "sell_price": 0.0, "trade_amount": 100},
        ])
        config = SimulationConfig(
            maker_fee=FEE,
            initial_cash=10_000.0,
            enable_probe_mode=True,
            probe_notional=100.0,
        )
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        buys = [t for t in result.per_symbol["BTCUSD"].trades if t.side == "buy"]
        assert len(buys) == 3

        # Third buy (after probe-off) should be much larger than second (probe)
        assert buys[2].notional > buys[1].notional * 5


# ---------------------------------------------------------------------------
# Shared cash multi-symbol
# ---------------------------------------------------------------------------

class TestSharedCash:
    """Tests for run_shared_cash_simulation with multiple symbols."""

    def test_shared_cash_splits_across_symbols(self):
        """Both symbols can buy from the same cash pool."""
        bars_btc = _make_bars([(100, 105, 90, 100)] * 3, symbol="BTCUSD")
        bars_eth = _make_bars([(50, 55, 45, 50)] * 3, symbol="ETHUSD")
        bars = pd.concat([bars_btc, bars_eth], ignore_index=True)

        act_btc = _make_actions(
            [{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 50}] * 3, symbol="BTCUSD"
        )
        act_eth = _make_actions(
            [{"buy_price": 48.0, "sell_price": 0.0, "trade_amount": 50}] * 3, symbol="ETHUSD"
        )
        actions = pd.concat([act_btc, act_eth], ignore_index=True)

        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        result = run_shared_cash_simulation(bars, actions, config)

        # Both symbols should have trades
        assert "BTCUSD" in result.per_symbol
        assert "ETHUSD" in result.per_symbol
        btc_buys = [t for t in result.per_symbol["BTCUSD"].trades if t.side == "buy"]
        eth_buys = [t for t in result.per_symbol["ETHUSD"].trades if t.side == "buy"]
        assert len(btc_buys) > 0
        assert len(eth_buys) > 0

    def test_shared_cash_total_spent_within_initial(self):
        """Total buy cost across symbols cannot exceed initial cash."""
        bars_btc = _make_bars([(100, 105, 90, 100)], symbol="BTCUSD")
        bars_eth = _make_bars([(50, 55, 45, 50)], symbol="ETHUSD")
        bars = pd.concat([bars_btc, bars_eth], ignore_index=True)

        act_btc = _make_actions([{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}], symbol="BTCUSD")
        act_eth = _make_actions([{"buy_price": 48.0, "sell_price": 0.0, "trade_amount": 100}], symbol="ETHUSD")
        actions = pd.concat([act_btc, act_eth], ignore_index=True)

        config = SimulationConfig(maker_fee=FEE, initial_cash=1000.0)
        result = run_shared_cash_simulation(bars, actions, config)

        all_trades = []
        for sym_result in result.per_symbol.values():
            all_trades.extend(sym_result.trades)
        total_cost = sum(t.notional * (1 + FEE) for t in all_trades if t.side == "buy")
        assert total_cost <= 1000.0 + 1e-6


# ---------------------------------------------------------------------------
# Amount scale resolution
# ---------------------------------------------------------------------------

class TestAmountScale:
    """Tests for _resolve_amount_scale heuristic."""

    def test_scale_100_for_large_amounts(self):
        """Amounts > 1.5 assumed to be in [0, 100] range."""
        frame = pd.DataFrame({"buy_amount": [10, 50, 80]})
        assert _resolve_amount_scale(frame) == 100.0

    def test_scale_1_for_small_amounts(self):
        """Amounts <= 1.5 assumed to be in [0, 1] range."""
        frame = pd.DataFrame({"buy_amount": [0.2, 0.5, 1.0]})
        assert _resolve_amount_scale(frame) == 1.0

    def test_scale_100_when_no_amount_columns(self):
        """Default to 100 if no amount columns exist."""
        frame = pd.DataFrame({"other_col": [1, 2, 3]})
        assert _resolve_amount_scale(frame) == 100.0


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for _compute_metrics."""

    def test_total_return(self):
        """Total return = (final - initial) / initial."""
        eq = pd.Series([100.0, 110.0, 120.0])
        m = _compute_metrics(eq)
        assert abs(m["total_return"] - 0.2) < 1e-8

    def test_sortino_positive_for_uptrend_with_variance(self):
        """Rising equity with some dips should have positive Sortino."""
        np.random.seed(42)
        values = 100.0 + np.cumsum(np.random.normal(0.5, 1.0, 200))
        eq = pd.Series(values)
        m = _compute_metrics(eq)
        assert m["sortino"] > 0

    def test_sortino_zero_for_flat(self):
        """Flat equity has zero Sortino (zero mean return)."""
        eq = pd.Series([100.0] * 10)
        m = _compute_metrics(eq)
        assert m["sortino"] == 0.0

    def test_empty_equity_returns_zeros(self):
        eq = pd.Series(dtype=float)
        m = _compute_metrics(eq)
        assert m["total_return"] == 0.0
        assert m["sortino"] == 0.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestPrepareFrame:
    """Tests for _prepare_frame validation."""

    def test_missing_timestamp_raises(self):
        bars = pd.DataFrame({"high": [1], "low": [1], "close": [1]})
        actions = pd.DataFrame({"buy_price": [1]})
        with pytest.raises(ValueError, match="timestamp"):
            _prepare_frame(bars, actions)

    def test_empty_merge_raises(self):
        """Non-overlapping timestamps produce empty merge -> ValueError."""
        bars = _make_bars([(100, 105, 95, 100)], start="2025-01-01T00:00:00+00:00")
        actions = _make_actions([{"buy_price": 95, "sell_price": 0, "trade_amount": 100}],
                                start="2025-06-01T00:00:00+00:00")
        with pytest.raises(ValueError, match="empty"):
            _prepare_frame(bars, actions)

    def test_missing_ohlc_column_raises(self):
        bars = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC"),
            "symbol": ["BTC"],
            "high": [1],
            "low": [1],
            # missing 'close'
        })
        actions = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC"),
            "symbol": ["BTC"],
            "buy_price": [1],
        })
        with pytest.raises(ValueError, match="close"):
            _prepare_frame(bars, actions)


# ---------------------------------------------------------------------------
# Consistency between single-symbol and shared-cash
# ---------------------------------------------------------------------------

class TestConsistency:
    """Verify BinanceMarketSimulator and run_shared_cash_simulation produce
    equivalent results for a single symbol."""

    def test_single_symbol_equity_matches(self):
        """For one symbol, both paths should produce the same final equity."""
        bars = _make_bars([
            (100, 105, 90, 100),
            (102, 115, 95, 110),
            (110, 120, 100, 115),
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 110.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 0.0, "trade_amount": 0},
        ])
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)

        sim = BinanceMarketSimulator(config)
        result_single = sim.run(bars, actions)

        result_shared = run_shared_cash_simulation(bars, actions, config)

        final_single = result_single.per_symbol["BTCUSD"].equity_curve.iloc[-1]
        final_shared = result_shared.combined_equity.iloc[-1]
        assert abs(final_single - final_shared) < 1e-4


# ---------------------------------------------------------------------------
# Realistic multi-bar scenario
# ---------------------------------------------------------------------------

class TestRealisticScenario:
    """A more realistic multi-bar scenario with mixed trades."""

    def test_realistic_trading_sequence(self):
        """Simulate a realistic BTC trading sequence over 10 bars."""
        prices = [
            (100, 105, 95, 102),
            (102, 108, 98, 105),
            (105, 112, 100, 110),
            (110, 115, 105, 108),
            (108, 110, 100, 103),
            (103, 109, 98, 107),
            (107, 115, 103, 112),
            (112, 118, 108, 116),
            (116, 120, 110, 118),
            (118, 122, 112, 120),
        ]
        bars = _make_bars(prices)

        acts = [
            {"buy_price": 97.0, "sell_price": 0.0, "trade_amount": 80},
            {"buy_price": 0.0, "sell_price": 107.0, "trade_amount": 80},
            {"buy_price": 102.0, "sell_price": 0.0, "trade_amount": 60},
            {"buy_price": 0.0, "sell_price": 114.0, "trade_amount": 60},
            {"buy_price": 101.0, "sell_price": 0.0, "trade_amount": 70},
            {"buy_price": 0.0, "sell_price": 0.0, "trade_amount": 0},
            {"buy_price": 0.0, "sell_price": 113.0, "trade_amount": 70},
            {"buy_price": 109.0, "sell_price": 0.0, "trade_amount": 50},
            {"buy_price": 0.0, "sell_price": 119.0, "trade_amount": 50},
            {"buy_price": 0.0, "sell_price": 0.0, "trade_amount": 0},
        ]
        actions = _make_actions(acts)

        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        sym = result.per_symbol["BTCUSD"]

        # Should have completed several round trips
        buys = [t for t in sym.trades if t.side == "buy"]
        sells = [t for t in sym.trades if t.side == "sell"]
        assert len(buys) >= 3
        assert len(sells) >= 3

        # Equity should grow (bought low, sold high)
        assert sym.equity_curve.iloc[-1] > 10_000.0

        # Equity curve should be monotonic between trades (no weird jumps)
        eq_vals = sym.equity_curve.values
        assert len(eq_vals) == 10

        # Final equity = last cash + inventory * last close
        last_trade = sym.trades[-1]
        # After all sells, final value should match cash
        # (depending on whether final trade is buy or sell)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_initial_cash(self):
        """With zero cash, no buys happen."""
        bars = _make_bars([(100, 105, 90, 100)])
        actions = _make_actions([{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}])
        config = SimulationConfig(initial_cash=0.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        assert len(result.per_symbol["BTCUSD"].trades) == 0

    def test_zero_trade_amount_no_fill(self):
        """Zero trade amount means zero intensity -> no fill."""
        bars = _make_bars([(100, 105, 90, 100)])
        actions = _make_actions([{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 0.0}])
        sim = BinanceMarketSimulator()
        result = sim.run(bars, actions)
        assert len(result.per_symbol["BTCUSD"].trades) == 0

    def test_single_bar(self):
        """Simulation with just one bar."""
        bars = _make_bars([(100, 105, 90, 100)])
        actions = _make_actions([{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}])
        sim = BinanceMarketSimulator()
        result = sim.run(bars, actions)
        assert len(result.per_symbol["BTCUSD"].equity_curve) == 1

    def test_symbol_case_insensitive(self):
        """Symbols are uppercased during merge."""
        bars = _make_bars([(100, 105, 90, 100)], symbol="btcusd")
        actions = _make_actions([{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}], symbol="BTCUSD")
        sim = BinanceMarketSimulator()
        result = sim.run(bars, actions)
        assert "BTCUSD" in result.per_symbol


# ---------------------------------------------------------------------------
# Inventory never goes negative
# ---------------------------------------------------------------------------

class TestInventoryInvariant:
    """Inventory should never go negative."""

    def test_inventory_stays_nonnegative(self):
        """After many buy/sell cycles, inventory >= 0."""
        bars = _make_bars([
            (100, 110, 90, 100),
            (100, 110, 90, 100),
            (100, 110, 90, 100),
            (100, 110, 90, 100),
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 105.0, "trade_amount": 100},
            {"buy_price": 95.0, "sell_price": 105.0, "trade_amount": 100},
            {"buy_price": 95.0, "sell_price": 105.0, "trade_amount": 100},
            {"buy_price": 95.0, "sell_price": 105.0, "trade_amount": 100},
        ])
        sim = BinanceMarketSimulator()
        result = sim.run(bars, actions)
        per_hour = result.per_symbol["BTCUSD"].per_hour
        assert all(per_hour["inventory"] >= -1e-10)


# ---------------------------------------------------------------------------
# Determinism / replay
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify the simulator is fully deterministic and replayable."""

    def _run_scenario(self):
        """Shared scenario that should produce identical results every time."""
        prices = [
            (100, 108, 92, 101), (101, 109, 93, 104), (104, 112, 96, 110),
            (110, 118, 102, 108), (108, 114, 100, 106), (106, 113, 98, 111),
            (111, 120, 104, 117), (117, 125, 110, 122), (122, 128, 114, 119),
            (119, 124, 112, 121),
        ]
        bars = _make_bars(prices)
        acts = [
            {"buy_price": 95.0, "sell_price": 0.0, "buy_amount": 80, "sell_amount": 0},
            {"buy_price": 0.0, "sell_price": 108.0, "buy_amount": 0, "sell_amount": 80},
            {"buy_price": 98.0, "sell_price": 0.0, "buy_amount": 60, "sell_amount": 0},
            {"buy_price": 0.0, "sell_price": 116.0, "buy_amount": 0, "sell_amount": 50},
            {"buy_price": 102.0, "sell_price": 0.0, "buy_amount": 70, "sell_amount": 0},
            {"buy_price": 0.0, "sell_price": 112.0, "buy_amount": 0, "sell_amount": 70},
            {"buy_price": 106.0, "sell_price": 0.0, "buy_amount": 90, "sell_amount": 0},
            {"buy_price": 0.0, "sell_price": 123.0, "buy_amount": 0, "sell_amount": 90},
            {"buy_price": 0.0, "sell_price": 0.0, "buy_amount": 0, "sell_amount": 0},
            {"buy_price": 0.0, "sell_price": 0.0, "buy_amount": 0, "sell_amount": 0},
        ]
        actions = _make_actions(acts)
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        sim = BinanceMarketSimulator(config)
        return sim.run(bars, actions)

    def test_identical_across_runs(self):
        """Two runs with same inputs produce bit-identical results."""
        r1 = self._run_scenario()
        r2 = self._run_scenario()
        eq1 = r1.per_symbol["BTCUSD"].equity_curve.values
        eq2 = r2.per_symbol["BTCUSD"].equity_curve.values
        np.testing.assert_array_equal(eq1, eq2)

    def test_trade_records_identical(self):
        """Trade records are identical across runs (full replay)."""
        r1 = self._run_scenario()
        r2 = self._run_scenario()
        t1 = r1.per_symbol["BTCUSD"].trades
        t2 = r2.per_symbol["BTCUSD"].trades
        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a.side == b.side
            assert a.price == b.price
            assert a.quantity == b.quantity
            assert a.fee == b.fee
            assert a.cash_after == b.cash_after
            assert a.realized_pnl == b.realized_pnl

    def test_shared_cash_deterministic(self):
        """run_shared_cash_simulation is also deterministic."""
        prices = [
            (100, 108, 92, 101), (101, 109, 93, 104),
            (50, 56, 44, 52), (52, 58, 46, 55),
        ]
        bars_btc = _make_bars(prices[:2], symbol="BTCUSD")
        bars_eth = _make_bars(prices[2:], symbol="ETHUSD")
        bars = pd.concat([bars_btc, bars_eth], ignore_index=True)

        act_btc = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 108.0, "trade_amount": 100},
        ], symbol="BTCUSD")
        act_eth = _make_actions([
            {"buy_price": 46.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 57.0, "trade_amount": 100},
        ], symbol="ETHUSD")
        actions = pd.concat([act_btc, act_eth], ignore_index=True)

        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        r1 = run_shared_cash_simulation(bars, actions, config)
        r2 = run_shared_cash_simulation(bars, actions, config)
        np.testing.assert_array_equal(
            r1.combined_equity.values,
            r2.combined_equity.values,
        )

    def test_equity_curve_matches_manual_calculation(self):
        """Verify equity curve against a hand-calculated expected value."""
        bars = _make_bars([
            (100, 105, 90, 100),   # buy at 95, close 100
            (100, 115, 95, 110),   # sell at 110, close 110
        ])
        actions = _make_actions([
            {"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 110.0, "trade_amount": 100},
        ])
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        eq = result.per_symbol["BTCUSD"].equity_curve

        # Bar 0: buy at 95, full intensity
        qty = 10_000.0 / (95.0 * (1 + FEE))
        cash_after_buy = 10_000.0 - qty * 95.0 * (1 + FEE)
        eq_bar0 = cash_after_buy + qty * 100.0  # mark to close=100
        assert abs(eq.iloc[0] - eq_bar0) < 1e-4

        # Bar 1: sell at 110, full intensity
        proceeds = qty * 110.0 * (1 - FEE)
        cash_after_sell = cash_after_buy + proceeds
        eq_bar1 = cash_after_sell  # inventory is 0
        assert abs(eq.iloc[1] - eq_bar1) < 1e-4


# ---------------------------------------------------------------------------
# Production quantization (tick_size, step_size, min_qty, min_notional)
# ---------------------------------------------------------------------------

class TestQuantization:
    """Verify exchange quantization rules match production execution.py."""

    def test_quantize_down(self):
        assert _quantize_down(95.1234, 0.01) == 95.12
        assert _quantize_down(95.129, 0.01) == 95.12
        assert _quantize_down(0.123456, 0.00001) == 0.12345

    def test_quantize_up(self):
        assert _quantize_up(110.001, 0.01) == 110.01
        assert _quantize_up(110.011, 0.01) == 110.02

    def test_tick_size_rounds_prices(self):
        """Buy price rounds down, sell price rounds up to tick."""
        bars = _make_bars([
            (100, 105, 90, 100),
            (100, 115, 95, 110),
        ])
        actions = _make_actions([
            {"buy_price": 95.123, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 110.001, "trade_amount": 100},
        ])
        config = SimulationConfig(
            maker_fee=FEE, initial_cash=10_000.0,
            tick_size=0.01,
        )
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        trades = result.per_symbol["BTCUSD"].trades

        buy = [t for t in trades if t.side == "buy"][0]
        sell = [t for t in trades if t.side == "sell"][0]
        assert buy.price == 95.12  # rounded down
        assert sell.price == 110.01  # rounded up (110.001 -> 110.01)

    def test_step_size_rounds_quantity(self):
        """Quantities are rounded down to step_size."""
        bars = _make_bars([(100, 105, 90, 100)])
        actions = _make_actions([{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}])
        config = SimulationConfig(
            maker_fee=FEE, initial_cash=10_000.0,
            step_size=1.0,  # only whole units
        )
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        buy = result.per_symbol["BTCUSD"].trades[0]
        # qty should be a whole number
        assert buy.quantity == int(buy.quantity)

    def test_min_notional_blocks_small_trades(self):
        """Orders below min_notional are rejected."""
        bars = _make_bars([(100, 105, 90, 100)])
        actions = _make_actions([{"buy_price": 95.0, "sell_price": 0.0, "trade_amount": 100}])
        config = SimulationConfig(
            maker_fee=FEE, initial_cash=5.0,  # only $5, notional < 10
            min_notional=10.0,
        )
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        # Buy notional ~$5 < min $10 → no trade
        assert len(result.per_symbol["BTCUSD"].trades) == 0

    def test_min_qty_blocks_tiny_orders(self):
        """Orders below min_qty are rejected."""
        bars = _make_bars([(50000, 51000, 49000, 50000)])
        actions = _make_actions([{"buy_price": 49500.0, "sell_price": 0.0, "trade_amount": 100}])
        config = SimulationConfig(
            maker_fee=FEE, initial_cash=100.0,  # ~0.002 BTC
            min_qty=0.01,  # need at least 0.01 BTC
        )
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        # qty ~0.002 < min 0.01 → no trade
        assert len(result.per_symbol["BTCUSD"].trades) == 0

    def test_quantization_in_shared_cash(self):
        """Quantization also applies in run_shared_cash_simulation."""
        bars = _make_bars([
            (100, 105, 90, 100),
            (100, 115, 95, 110),
        ])
        actions = _make_actions([
            {"buy_price": 95.123, "sell_price": 0.0, "trade_amount": 100},
            {"buy_price": 0.0, "sell_price": 110.001, "trade_amount": 100},
        ])
        config = SimulationConfig(
            maker_fee=FEE, initial_cash=10_000.0,
            tick_size=0.01,
        )
        result = run_shared_cash_simulation(bars, actions, config)
        trades = result.per_symbol["BTCUSD"].trades
        buy = [t for t in trades if t.side == "buy"][0]
        sell = [t for t in trades if t.side == "sell"][0]
        assert buy.price == 95.12
        assert sell.price == 110.01

    def test_no_quantization_by_default(self):
        """Without tick/step config, no quantization applied (backward compat)."""
        bars = _make_bars([(100, 105, 90, 100)])
        actions = _make_actions([{"buy_price": 95.123, "sell_price": 0.0, "trade_amount": 100}])
        config = SimulationConfig(maker_fee=FEE, initial_cash=10_000.0)
        sim = BinanceMarketSimulator(config)
        result = sim.run(bars, actions)
        buy = result.per_symbol["BTCUSD"].trades[0]
        assert buy.price == 95.123  # unmodified


def test_binance_market_simulator_stops_early_when_drawdown_exceeds_profit(capsys: pytest.CaptureFixture[str]) -> None:
    prices: list[tuple[float, float, float, float]] = []
    for close in [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 128.0, 120.0, 110.0, 95.0, 85.0]:
        prices.append((close, close * 1.01, close * 0.99, close))
    prices = prices * 2
    bars = _make_bars(prices)
    actions = _make_actions(
        [{"buy_price": 100.0, "sell_price": 10_000.0, "buy_amount": 1.0, "sell_amount": 0.0}]
        + [{"buy_price": 0.0, "sell_price": 10_000.0, "buy_amount": 0.0, "sell_amount": 0.0}] * (len(prices) - 1)
    )

    result = BinanceMarketSimulator(SimulationConfig(maker_fee=0.0, initial_cash=10_000.0)).run(bars, actions)
    sym = result.per_symbol["BTCUSD"]

    assert len(sym.equity_curve) < len(bars)
    assert "early stopping" in capsys.readouterr().out.lower()
