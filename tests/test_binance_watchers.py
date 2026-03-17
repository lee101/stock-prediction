from __future__ import annotations

from datetime import datetime, timedelta, timezone

from binanceneural.binance_watchers import WatcherPlan, _matches_plan


def test_matches_plan_accepts_legacy_binance_symbol_for_same_exchange_pair(monkeypatch) -> None:
    monkeypatch.setattr("binanceneural.binance_watchers._is_pid_alive", lambda pid: True)
    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    metadata = {
        "active": True,
        "pid": 12345,
        "symbol": "BTCUSD",
        "binance_symbol": "BTCFDUSD",
        "side": "buy",
        "mode": "daily_entry",
        "limit_price": 70000.0,
        "target_qty": 0.01,
        "expiry_at": expiry,
    }
    plan = WatcherPlan(
        symbol="BTCUSD",
        side="buy",
        mode="daily_entry",
        limit_price=70000.0,
        target_qty=0.01,
        expiry_minutes=60,
        poll_seconds=30,
        exchange_symbol="BTCFDUSD",
    )

    assert _matches_plan(metadata, plan) is True


def test_matches_plan_rejects_different_exchange_pair(monkeypatch) -> None:
    monkeypatch.setattr("binanceneural.binance_watchers._is_pid_alive", lambda pid: True)
    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    metadata = {
        "active": True,
        "pid": 12345,
        "symbol": "ETHUSD",
        "exchange_symbol": "ETHUSDT",
        "side": "sell",
        "mode": "daily_exit",
        "limit_price": 2400.0,
        "target_qty": 1.0,
        "expiry_at": expiry,
    }
    plan = WatcherPlan(
        symbol="ETHUSD",
        side="sell",
        mode="daily_exit",
        limit_price=2400.0,
        target_qty=1.0,
        expiry_minutes=60,
        poll_seconds=30,
        exchange_symbol="ETHFDUSD",
    )

    assert _matches_plan(metadata, plan) is False
