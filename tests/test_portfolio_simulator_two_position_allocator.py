from __future__ import annotations

import pandas as pd

from unified_hourly_experiment.marketsimulator.portfolio_simulator import (
    PortfolioConfig,
    run_portfolio_simulation,
)


def _base_cfg(**overrides) -> PortfolioConfig:
    payload = {
        "initial_cash": 1_000.0,
        "max_positions": 2,
        "max_leverage": 1.0,
        "fee_by_symbol": {"AAA": 0.0, "BBB": 0.0},
        "decision_lag_bars": 0,
        "enforce_market_hours": False,
        "close_at_eod": False,
        "max_hold_hours": 0,
        "bar_margin": 0.0,
        "int_qty": False,
        "sim_backend": "python",
    }
    payload.update(overrides)
    return PortfolioConfig(**payload)


def test_min_active_positions_blocks_lone_flat_entry_candidate() -> None:
    ts = pd.Timestamp("2026-04-01T00:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "AAA", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAA",
                "buy_price": 100.0,
                "sell_price": 130.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
        ]
    )

    result = run_portfolio_simulation(
        bars,
        actions,
        _base_cfg(entry_allocator_min_active_positions=2),
        horizon=1,
    )

    assert result.trades == []


def test_min_second_position_fraction_prevents_token_second_allocation() -> None:
    ts = pd.Timestamp("2026-04-01T00:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "AAA", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"timestamp": ts, "symbol": "BBB", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAA",
                "buy_price": 100.0,
                "sell_price": 130.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
            {
                "timestamp": ts,
                "symbol": "BBB",
                "buy_price": 100.0,
                "sell_price": 101.0,
                "buy_amount": 1.0,
                "sell_amount": 0.0,
                "trade_amount": 1.0,
            },
        ]
    )

    result = run_portfolio_simulation(
        bars,
        actions,
        _base_cfg(
            entry_allocator_mode="concentrated",
            entry_allocator_max_single_position_fraction=1.0,
            entry_allocator_reserve_fraction=0.0,
            entry_allocator_min_active_positions=2,
            entry_allocator_min_second_position_fraction=0.2,
        ),
        horizon=1,
    )

    entry_notional = {
        trade.symbol: trade.price * trade.quantity
        for trade in result.trades
        if trade.reason == "entry"
    }
    total_entry_notional = sum(entry_notional.values())

    assert set(entry_notional) == {"AAA", "BBB"}
    assert entry_notional["BBB"] / total_entry_notional >= 0.2
    assert entry_notional["AAA"] / total_entry_notional <= 0.8


def test_concentrated_allocator_uses_configured_leverage_budget() -> None:
    ts = pd.Timestamp("2026-04-01T00:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "AAAUSDT", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"timestamp": ts, "symbol": "BBBUSDT", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAAUSDT",
                "buy_price": 100.0,
                "sell_price": 130.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
            {
                "timestamp": ts,
                "symbol": "BBBUSDT",
                "buy_price": 100.0,
                "sell_price": 125.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
        ]
    )
    common_cfg = dict(
        fee_by_symbol={"AAAUSDT": 0.0, "BBBUSDT": 0.0},
        max_positions=2,
        entry_allocator_mode="concentrated",
        entry_allocator_max_single_position_fraction=0.5,
        entry_allocator_reserve_fraction=0.0,
        apply_leverage_to_crypto=True,
    )

    one_x = run_portfolio_simulation(bars, actions, _base_cfg(**common_cfg, max_leverage=1.0), horizon=1)
    two_x = run_portfolio_simulation(bars, actions, _base_cfg(**common_cfg, max_leverage=2.0), horizon=1)

    one_x_notional = sum(trade.price * trade.quantity for trade in one_x.trades if trade.reason == "entry")
    two_x_notional = sum(trade.price * trade.quantity for trade in two_x.trades if trade.reason == "entry")

    assert two_x_notional > one_x_notional * 1.9


def test_entry_drawdown_pause_blocks_new_entries_after_loss() -> None:
    ts = pd.date_range("2026-04-01T00:00:00Z", periods=3, freq="h")
    bars = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
            for timestamp, aaa_price, bbb_price in zip(ts, [100.0, 80.0, 80.0], [100.0, 100.0, 100.0])
            for symbol, price in (("AAAUSDT", aaa_price), ("BBBUSDT", bbb_price))
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts[0],
                "symbol": "AAAUSDT",
                "buy_price": 100.0,
                "sell_price": 200.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
            {
                "timestamp": ts[1],
                "symbol": "BBBUSDT",
                "buy_price": 100.0,
                "sell_price": 120.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
        ]
    )

    baseline = run_portfolio_simulation(
        bars,
        actions,
        _base_cfg(
            fee_by_symbol={"AAAUSDT": 0.0, "BBBUSDT": 0.0},
            max_hold_hours=1,
            force_close_slippage=0.0,
            drawdown_profit_early_exit=False,
        ),
        horizon=1,
    )
    paused = run_portfolio_simulation(
        bars,
        actions,
        _base_cfg(
            fee_by_symbol={"AAAUSDT": 0.0, "BBBUSDT": 0.0},
            max_hold_hours=1,
            force_close_slippage=0.0,
            drawdown_profit_early_exit=False,
            entry_drawdown_pause_threshold=0.05,
            entry_drawdown_cooldown_hours=24,
        ),
        horizon=1,
    )

    baseline_entries = [(trade.timestamp, trade.symbol) for trade in baseline.trades if trade.reason == "entry"]
    paused_entries = [(trade.timestamp, trade.symbol) for trade in paused.trades if trade.reason == "entry"]

    assert baseline_entries == [(ts[0], "AAAUSDT"), (ts[1], "BBBUSDT")]
    assert paused_entries == [(ts[0], "AAAUSDT")]


def test_entry_drawdown_indefinite_pause_accepts_naive_timestamps() -> None:
    ts = pd.date_range("2026-04-01 00:00:00", periods=3, freq="h")
    bars = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
            for timestamp, aaa_price, bbb_price in zip(ts, [100.0, 80.0, 80.0], [100.0, 100.0, 100.0])
            for symbol, price in (("AAAUSDT", aaa_price), ("BBBUSDT", bbb_price))
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts[0],
                "symbol": "AAAUSDT",
                "buy_price": 100.0,
                "sell_price": 200.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
            {
                "timestamp": ts[1],
                "symbol": "BBBUSDT",
                "buy_price": 100.0,
                "sell_price": 120.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
        ]
    )

    result = run_portfolio_simulation(
        bars,
        actions,
        _base_cfg(
            fee_by_symbol={"AAAUSDT": 0.0, "BBBUSDT": 0.0},
            max_hold_hours=1,
            force_close_slippage=0.0,
            drawdown_profit_early_exit=False,
            entry_drawdown_pause_threshold=0.05,
            entry_drawdown_cooldown_hours=0,
        ),
        horizon=1,
    )

    entries = [(trade.timestamp, trade.symbol) for trade in result.trades if trade.reason == "entry"]
    assert entries == [(ts[0], "AAAUSDT")]


def test_entry_drawdown_scale_reduces_new_position_size_after_loss() -> None:
    ts = pd.date_range("2026-04-01T00:00:00Z", periods=3, freq="h")
    bars = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
            for timestamp, aaa_price, bbb_price in zip(ts, [100.0, 80.0, 80.0], [100.0, 100.0, 100.0])
            for symbol, price in (("AAAUSDT", aaa_price), ("BBBUSDT", bbb_price))
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts[0],
                "symbol": "AAAUSDT",
                "buy_price": 100.0,
                "sell_price": 200.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
            {
                "timestamp": ts[1],
                "symbol": "BBBUSDT",
                "buy_price": 100.0,
                "sell_price": 120.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
        ]
    )
    common_cfg = dict(
        fee_by_symbol={"AAAUSDT": 0.0, "BBBUSDT": 0.0},
        max_hold_hours=1,
        force_close_slippage=0.0,
        drawdown_profit_early_exit=False,
    )

    baseline = run_portfolio_simulation(bars, actions, _base_cfg(**common_cfg), horizon=1)
    scaled = run_portfolio_simulation(
        bars,
        actions,
        _base_cfg(
            **common_cfg,
            entry_drawdown_scale_start=0.02,
            entry_drawdown_scale_full=0.10,
            entry_drawdown_scale_floor=0.25,
        ),
        horizon=1,
    )

    baseline_bbb_qty = next(trade.quantity for trade in baseline.trades if trade.reason == "entry" and trade.symbol == "BBBUSDT")
    scaled_bbb_qty = next(trade.quantity for trade in scaled.trades if trade.reason == "entry" and trade.symbol == "BBBUSDT")

    assert scaled_bbb_qty < baseline_bbb_qty * 0.4


def test_crypto_short_direction_can_come_from_action_amounts() -> None:
    ts = pd.Timestamp("2026-04-01T00:00:00Z")
    bars = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "BTCUSDT",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            },
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "BTCUSDT",
                "buy_price": 95.0,
                "sell_price": 100.0,
                "buy_amount": 0.0,
                "sell_amount": 100.0,
                "trade_amount": 100.0,
            },
        ]
    )

    result = run_portfolio_simulation(
        bars,
        actions,
        _base_cfg(
            max_positions=1,
            fee_by_symbol={"BTCUSDT": 0.0},
            allow_crypto_shorts=True,
        ),
        horizon=1,
    )

    assert [(trade.side, trade.symbol, trade.reason) for trade in result.trades] == [
        ("short_sell", "BTCUSDT", "entry")
    ]


def test_diversified_selection_prefers_distinct_buckets() -> None:
    ts = pd.Timestamp("2026-04-01T00:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": symbol, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}
            for symbol in ("AAA", "BBB", "CCC")
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAA",
                "buy_price": 100.0,
                "sell_price": 130.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
                "entry_diversify_bucket": 0,
            },
            {
                "timestamp": ts,
                "symbol": "BBB",
                "buy_price": 100.0,
                "sell_price": 125.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
                "entry_diversify_bucket": 0,
            },
            {
                "timestamp": ts,
                "symbol": "CCC",
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
                "entry_diversify_bucket": 1,
            },
        ]
    )

    result = run_portfolio_simulation(
        bars,
        actions,
        _base_cfg(
            max_positions=2,
            entry_selection_mode="edge_rank_diversified",
            entry_diversify_bucket_column="entry_diversify_bucket",
            entry_allocator_min_active_positions=2,
        ),
        horizon=1,
    )

    entry_symbols = [trade.symbol for trade in result.trades if trade.reason == "entry"]
    assert entry_symbols == ["AAA", "CCC"]


def test_correlation_aware_selection_skips_redundant_same_side_candidate() -> None:
    ts = pd.date_range("2026-04-01T00:00:00Z", periods=5, freq="h")
    price_paths = {
        "AAAUSDT": [100.0, 101.0, 102.0, 103.0, 103.0],
        "BBBUSDT": [100.0, 101.0, 102.0, 103.0, 103.0],
        "CCCUSDT": [100.0, 102.0, 99.0, 101.0, 101.0],
    }
    bars = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "open": close,
                "high": max(close, 105.0),
                "low": min(close, 99.0),
                "close": close,
            }
            for symbol, closes in price_paths.items()
            for timestamp, close in zip(ts, closes)
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts[-1],
                "symbol": symbol,
                "buy_price": 100.0,
                "sell_price": sell_price,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            }
            for symbol, sell_price in (("AAAUSDT", 130.0), ("BBBUSDT", 125.0), ("CCCUSDT", 110.0))
        ]
    )

    result = run_portfolio_simulation(
        bars,
        actions,
        _base_cfg(
            max_positions=2,
            fee_by_symbol={"AAAUSDT": 0.0, "BBBUSDT": 0.0, "CCCUSDT": 0.0},
            entry_selection_mode="edge_rank",
            entry_corr_window_bars=4,
            entry_corr_min_periods=2,
            entry_corr_max_signed=0.5,
        ),
        horizon=1,
    )

    entry_symbols = [trade.symbol for trade in result.trades if trade.reason == "entry"]
    assert entry_symbols == ["AAAUSDT", "CCCUSDT"]
