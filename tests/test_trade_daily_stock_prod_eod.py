"""Tests for the EOD overnight deleverage tick in trade_daily_stock_prod.

Mirrors ``tests/test_xgbnew_live_trader_helpers.py``::
``test_eod_deleverage_tick_*`` since the RL daemon's tick is the same
contract — bring account gross exposure to <= equity * target_leverage
before market close, with a progressive limit-price ramp and a final
force-marketable window.

Test plan covers:
    1. enabled=False -> action "disabled"
    2. market closed (minutes_to_close=None) -> "closed"
    3. mtc > window -> "outside_window"
    4. exposure already <= target -> "already_ok"
    5. exposure > target -> submits SELL limit, computes correct qty,
       skips crypto, ramps aggressiveness with progress
    6. mtc < force_market_minutes -> force_window True, max-aggressive bps
    7. progressive limit price correct mid-window
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import trade_daily_stock_prod as mod


def test_eod_tick_disabled_is_noop():
    out = mod.eod_deleverage_tick(
        MagicMock(),
        enabled=False,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out == {"action": "disabled"}


def test_eod_tick_no_client_is_disabled():
    out = mod.eod_deleverage_tick(
        None,
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out == {"action": "disabled"}


def test_eod_tick_market_closed(monkeypatch):
    monkeypatch.setattr(mod, "_eod_minutes_to_market_close", lambda _c: None)
    out = mod.eod_deleverage_tick(
        MagicMock(),
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out == {"action": "closed"}


def test_eod_tick_outside_window(monkeypatch):
    monkeypatch.setattr(mod, "_eod_minutes_to_market_close", lambda _c: 90.0)
    out = mod.eod_deleverage_tick(
        MagicMock(),
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out["action"] == "outside_window"
    assert out["minutes_to_close"] == pytest.approx(90.0)


def test_eod_tick_already_under_target(monkeypatch):
    monkeypatch.setattr(mod, "_eod_minutes_to_market_close", lambda _c: 30.0)
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="1000")
    client.get_all_positions.return_value = [
        SimpleNamespace(
            symbol="AAPL", qty="10", market_value="1500", current_price="150",
            side="long",
        ),
        # Crypto excluded (matches the prefix list). Even if it added 9000,
        # the equity exposure is still 1500 < equity*2 = 2000.
        SimpleNamespace(
            symbol="BTC/USD", qty="0.1", market_value="9000", current_price="90000",
            side="long",
        ),
    ]
    out = mod.eod_deleverage_tick(
        client,
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out["action"] == "already_ok"
    assert out["exposure"] == pytest.approx(1500.0)
    assert out["leverage"] == pytest.approx(1.5)
    assert out["target_leverage"] == pytest.approx(2.0)


def test_eod_tick_submits_excess_equity_limit_progressive(monkeypatch):
    """Mid-window: aggressiveness ramps from 5 bps -> 25 bps linearly."""
    monkeypatch.setattr(mod, "_eod_minutes_to_market_close", lambda _c: 30.0)
    submitted: list[dict] = []

    def _submit_limit(client, *, symbol, qty, side, limit_price):
        submitted.append({
            "symbol": symbol, "qty": qty, "side": side, "limit_price": limit_price,
        })
        return SimpleNamespace(id="order-1")

    monkeypatch.setattr(mod, "submit_limit_order", _submit_limit)
    monkeypatch.setattr(mod, "_eod_latest_stock_bid_ask", lambda _sym: (149.0, 151.0))

    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="1000")
    client.get_all_positions.return_value = [
        SimpleNamespace(
            symbol="AAPL", qty="20", market_value="3000", current_price="150",
            side="long",
        ),
        # Crypto skipped
        SimpleNamespace(
            symbol="ETH/USD", qty="1", market_value="3000", current_price="3000",
            side="long",
        ),
    ]
    out = mod.eod_deleverage_tick(
        client,
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out["action"] == "submitted"
    assert out["submitted"] == 1
    # Excess = 3000 - 2000 = 1000. Mid-window progress = (60-30)/(60-5).
    progress = (60.0 - 30.0) / (60.0 - 5.0)
    aggressiveness_bps = 5.0 + 20.0 * progress
    assert submitted[0]["symbol"] == "AAPL"
    assert submitted[0]["side"] == "sell"
    assert submitted[0]["qty"] > 0
    assert submitted[0]["limit_price"] == pytest.approx(
        149.0 * (1.0 - aggressiveness_bps / 10_000.0)
    )


def test_eod_tick_force_market_window_max_aggressive(monkeypatch):
    """mtc < force_market_minutes -> aggressiveness fixed at 25 bps."""
    monkeypatch.setattr(mod, "_eod_minutes_to_market_close", lambda _c: 3.0)
    submitted: list[dict] = []

    def _submit_limit(client, *, symbol, qty, side, limit_price):
        submitted.append({
            "symbol": symbol, "qty": qty, "side": side, "limit_price": limit_price,
        })
        return SimpleNamespace(id="order-1")

    monkeypatch.setattr(mod, "submit_limit_order", _submit_limit)
    monkeypatch.setattr(mod, "_eod_latest_stock_bid_ask", lambda _sym: (149.0, 151.0))

    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="1000")
    client.get_all_positions.return_value = [
        SimpleNamespace(
            symbol="AAPL", qty="20", market_value="3000", current_price="150",
            side="long",
        ),
    ]
    out = mod.eod_deleverage_tick(
        client,
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out["action"] == "submitted"
    assert out["force_window"] is True
    # Force-window aggressiveness is fixed at 25 bps regardless of progress.
    expected_limit = 149.0 * (1.0 - 0.0025)
    assert submitted[0]["side"] == "sell"
    assert submitted[0]["qty"] == pytest.approx(6.6666, abs=1e-3)
    assert submitted[0]["limit_price"] == pytest.approx(expected_limit, rel=1e-9)


def test_eod_tick_skips_zero_qty_positions(monkeypatch):
    """Zero-qty / zero-value positions must not be selected for selling."""
    monkeypatch.setattr(mod, "_eod_minutes_to_market_close", lambda _c: 30.0)
    submitted: list[dict] = []

    def _submit_limit(client, **_kw):
        submitted.append(_kw)
        return SimpleNamespace(id="order-x")

    monkeypatch.setattr(mod, "submit_limit_order", _submit_limit)
    monkeypatch.setattr(mod, "_eod_latest_stock_bid_ask", lambda _sym: (99.0, 101.0))

    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="500")
    client.get_all_positions.return_value = [
        SimpleNamespace(
            symbol="ZZZ", qty="0", market_value="0", current_price="100",
            side="long",
        ),
        SimpleNamespace(
            symbol="QQQ", qty="20", market_value="2000", current_price="100",
            side="long",
        ),
    ]
    out = mod.eod_deleverage_tick(
        client,
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    # Excess = 2000 - 1000 = 1000 -> sell QQQ, skip ZZZ.
    assert out["action"] == "submitted"
    assert len(submitted) == 1
    assert submitted[0]["symbol"] == "QQQ"


def test_eod_tick_short_position_buys_to_cover(monkeypatch):
    """Short-side positions must be reduced via BUY (cover)."""
    monkeypatch.setattr(mod, "_eod_minutes_to_market_close", lambda _c: 30.0)
    submitted: list[dict] = []

    def _submit_limit(client, **_kw):
        submitted.append(_kw)
        return SimpleNamespace(id="order-cover")

    monkeypatch.setattr(mod, "submit_limit_order", _submit_limit)
    monkeypatch.setattr(mod, "_eod_latest_stock_bid_ask", lambda _sym: (99.0, 101.0))

    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="1000")
    client.get_all_positions.return_value = [
        SimpleNamespace(
            symbol="SHORTY", qty="-20", market_value="-3000", current_price="150",
            side="short",
        ),
    ]
    out = mod.eod_deleverage_tick(
        client,
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out["action"] == "submitted"
    assert submitted[0]["side"] == "buy"  # cover


def test_eod_tick_account_query_failure_returned(monkeypatch):
    monkeypatch.setattr(mod, "_eod_minutes_to_market_close", lambda _c: 10.0)
    client = MagicMock()
    client.get_account.side_effect = RuntimeError("boom")
    out = mod.eod_deleverage_tick(
        client,
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out["action"] == "account_error"
    assert "boom" in out["error"]


def test_eod_tick_bad_equity_returned(monkeypatch):
    monkeypatch.setattr(mod, "_eod_minutes_to_market_close", lambda _c: 10.0)
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="0")
    out = mod.eod_deleverage_tick(
        client,
        enabled=True,
        target_leverage=2.0,
        window_minutes=60.0,
        force_market_minutes=5.0,
    )
    assert out["action"] == "bad_equity"
    assert out["equity"] == pytest.approx(0.0)


# ── CLI args wiring ────────────────────────────────────────────────────────


def test_cli_args_default_to_disabled():
    args = mod.parse_args([
        "--symbols", "AAPL",
        "--data-dir", "/tmp",
        "--checkpoint", "fake.pt",
        "--once",
    ])
    assert args.eod_deleverage is False
    assert args.eod_max_gross_leverage == pytest.approx(2.0)
    assert args.eod_deleverage_window_minutes == pytest.approx(60.0)
    assert args.eod_force_market_minutes == pytest.approx(5.0)


def test_cli_args_enable_eod_deleverage():
    args = mod.parse_args([
        "--symbols", "AAPL",
        "--data-dir", "/tmp",
        "--checkpoint", "fake.pt",
        "--daemon",
        "--eod-deleverage",
        "--eod-max-gross-leverage", "1.8",
        "--eod-deleverage-window-minutes", "45",
        "--eod-force-market-minutes", "3",
    ])
    assert args.eod_deleverage is True
    assert args.eod_max_gross_leverage == pytest.approx(1.8)
    assert args.eod_deleverage_window_minutes == pytest.approx(45.0)
    assert args.eod_force_market_minutes == pytest.approx(3.0)
