from __future__ import annotations

import pandas as pd
import pytest

from src.current_price_locked_execution import CurrentPriceLockedConfig, simulate_current_price_locked


def _bars(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([ts for ts, _ in rows], utc=True),
            "close": [close for _, close in rows],
        }
    )


def _signal(*, buy_price: float, sell_price: float, buy_amount: float, sell_amount: float) -> dict[str, float]:
    return {
        "buy_price": buy_price,
        "sell_price": sell_price,
        "buy_amount": buy_amount,
        "sell_amount": sell_amount,
    }


def test_long_exit_is_blocked_until_lock_expires_when_profit_lock_not_met() -> None:
    config = CurrentPriceLockedConfig(
        name="locked_long",
        fee=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        min_expected_edge_bps=0.0,
        min_profit_exit_bps=20.0,
        lock_minutes=60,
        cooldown_minutes_after_exit=0,
        max_hold_hours=12.0,
        allow_short=False,
        long_max_leverage=1.0,
        short_max_leverage=0.0,
        min_notional=1.0,
        step_size=0.001,
    )
    signals = {
        pd.Timestamp("2026-03-01 00:00:00+00:00"): _signal(
            buy_price=100.0,
            sell_price=100.05,
            buy_amount=100.0,
            sell_amount=100.0,
        ),
        pd.Timestamp("2026-03-01 01:00:00+00:00"): _signal(
            buy_price=100.0,
            sell_price=100.05,
            buy_amount=100.0,
            sell_amount=100.0,
        ),
    }

    trades, _final_eq, _cash, qty, _trace, stats = simulate_current_price_locked(
        config,
        signals,
        _bars(
            [
                ("2026-03-01 01:00:00+00:00", 100.0),
                ("2026-03-01 01:05:00+00:00", 100.10),
                ("2026-03-01 02:05:00+00:00", 100.10),
            ]
        ),
        start_ts="2026-03-01 01:00:00+00:00",
        initial_cash=100.0,
    )

    assert [trade["side"] for trade in trades] == ["buy", "sell"]
    assert trades[0]["ts"] == pd.Timestamp("2026-03-01 01:00:00+00:00")
    assert trades[1]["ts"] == pd.Timestamp("2026-03-01 02:05:00+00:00")
    assert stats["blocked_loss_exit_count"] == 1
    assert qty == pytest.approx(0.0)


def test_short_exit_is_blocked_until_lock_expires_when_profit_lock_not_met() -> None:
    config = CurrentPriceLockedConfig(
        name="locked_short",
        fee=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        min_expected_edge_bps=0.0,
        min_profit_exit_bps=20.0,
        lock_minutes=60,
        cooldown_minutes_after_exit=0,
        max_hold_hours=12.0,
        allow_short=True,
        long_max_leverage=1.0,
        short_max_leverage=1.0,
        min_notional=1.0,
        step_size=0.001,
    )
    signals = {
        pd.Timestamp("2026-03-01 00:00:00+00:00"): _signal(
            buy_price=99.95,
            sell_price=100.0,
            buy_amount=100.0,
            sell_amount=100.0,
        ),
        pd.Timestamp("2026-03-01 01:00:00+00:00"): _signal(
            buy_price=99.95,
            sell_price=100.0,
            buy_amount=100.0,
            sell_amount=100.0,
        ),
    }

    trades, _final_eq, _cash, qty, _trace, stats = simulate_current_price_locked(
        config,
        signals,
        _bars(
            [
                ("2026-03-01 01:00:00+00:00", 100.0),
                ("2026-03-01 01:05:00+00:00", 99.90),
                ("2026-03-01 02:05:00+00:00", 99.90),
            ]
        ),
        start_ts="2026-03-01 01:00:00+00:00",
        initial_cash=100.0,
    )

    assert [trade["side"] for trade in trades] == ["sell", "buy"]
    assert trades[0]["ts"] == pd.Timestamp("2026-03-01 01:00:00+00:00")
    assert trades[1]["ts"] == pd.Timestamp("2026-03-01 02:05:00+00:00")
    assert stats["blocked_loss_exit_count"] == 1
    assert qty == pytest.approx(0.0)


def test_cooldown_blocks_same_hour_reentry_after_profitable_exit() -> None:
    config = CurrentPriceLockedConfig(
        name="cooldown",
        fee=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        min_expected_edge_bps=0.0,
        min_profit_exit_bps=0.0,
        lock_minutes=0,
        cooldown_minutes_after_exit=60,
        max_hold_hours=12.0,
        allow_short=False,
        long_max_leverage=1.0,
        short_max_leverage=0.0,
        min_notional=1.0,
        step_size=0.001,
    )
    signals = {
        pd.Timestamp("2026-03-01 00:00:00+00:00"): _signal(
            buy_price=100.0,
            sell_price=102.0,
            buy_amount=100.0,
            sell_amount=100.0,
        ),
    }

    trades, _final_eq, _cash, qty, _trace, stats = simulate_current_price_locked(
        config,
        signals,
        _bars(
            [
                ("2026-03-01 01:00:00+00:00", 100.0),
                ("2026-03-01 01:05:00+00:00", 102.0),
                ("2026-03-01 01:10:00+00:00", 100.0),
            ]
        ),
        start_ts="2026-03-01 01:00:00+00:00",
        initial_cash=100.0,
    )

    assert [trade["side"] for trade in trades] == ["buy", "sell"]
    assert stats["blocked_reentry_count"] == 1
    assert qty == pytest.approx(0.0)


def test_min_expected_edge_threshold_blocks_weak_entry() -> None:
    config = CurrentPriceLockedConfig(
        name="threshold",
        fee=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        min_expected_edge_bps=25.0,
        min_profit_exit_bps=0.0,
        lock_minutes=0,
        cooldown_minutes_after_exit=0,
        max_hold_hours=12.0,
        allow_short=False,
        long_max_leverage=1.0,
        short_max_leverage=0.0,
        min_notional=1.0,
        step_size=0.001,
    )
    signals = {
        pd.Timestamp("2026-03-01 00:00:00+00:00"): _signal(
            buy_price=100.0,
            sell_price=100.10,
            buy_amount=100.0,
            sell_amount=100.0,
        ),
    }

    trades, _final_eq, cash, qty, _trace, stats = simulate_current_price_locked(
        config,
        signals,
        _bars([("2026-03-01 01:00:00+00:00", 100.0)]),
        start_ts="2026-03-01 01:00:00+00:00",
        initial_cash=100.0,
    )

    assert trades == []
    assert cash == pytest.approx(100.0)
    assert qty == pytest.approx(0.0)
    assert stats == {"blocked_loss_exit_count": 0, "blocked_reentry_count": 0}


def test_signal_schedule_uses_latest_live_refresh_snapshot() -> None:
    config = CurrentPriceLockedConfig(
        name="schedule",
        fee=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        min_expected_edge_bps=0.0,
        min_profit_exit_bps=0.0,
        lock_minutes=0,
        cooldown_minutes_after_exit=0,
        max_hold_hours=12.0,
        allow_short=False,
        long_max_leverage=1.0,
        short_max_leverage=0.0,
        min_notional=1.0,
        step_size=0.001,
    )

    trades, _final_eq, _cash, qty, _trace, _stats = simulate_current_price_locked(
        config,
        {},
        _bars(
            [
                ("2026-03-01 01:00:00+00:00", 101.0),
                ("2026-03-01 01:05:00+00:00", 100.0),
            ]
        ),
        start_ts="2026-03-01 01:00:00+00:00",
        initial_cash=100.0,
        signal_schedule=[
            {
                "effective_ts": "2026-03-01 01:02:00+00:00",
                "buy_price": 100.0,
                "sell_price": 102.0,
                "buy_amount": 100.0,
                "sell_amount": 100.0,
            }
        ],
    )

    assert [trade["side"] for trade in trades] == ["buy"]
    assert trades[0]["ts"] == pd.Timestamp("2026-03-01 01:05:00+00:00")
    assert qty > 0.0
