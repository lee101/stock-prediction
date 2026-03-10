from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from newnanoalpacahourlyexp.trade_alpaca_hourly import (
    _allocation_usd,
    _parse_checkpoint_map,
    _parse_horizon_map,
    _parse_symbols,
    _reconcile_live_symbol_orders,
)
from src.hourly_trader_utils import (
    OrderIntent,
    TradingPlan,
    build_order_intents,
    directional_entry_amount,
    entry_intensity_fraction,
    ensure_valid_levels,
)


def test_parse_symbols_default():
    assert _parse_symbols(None) == ["SOLUSD", "LINKUSD", "UNIUSD"]


def test_parse_symbols_strip():
    parsed = _parse_symbols(" solusd, linkusd ,, UNIUSD ")
    assert parsed == ["SOLUSD", "LINKUSD", "UNIUSD"]


def test_parse_checkpoint_map(tmp_path: Path):
    model_path = tmp_path / "model.pt"
    model_path.write_text("x")
    mapping = _parse_checkpoint_map(f"SOLUSD={model_path}")
    assert mapping["SOLUSD"] == model_path.resolve()


def test_parse_checkpoint_map_invalid():
    with pytest.raises(ValueError):
        _parse_checkpoint_map("SOLUSD")


def test_parse_horizon_map():
    mapping = _parse_horizon_map("SOLUSD=1,24;UNIUSD=1")
    assert mapping["SOLUSD"] == (1, 24)
    assert mapping["UNIUSD"] == (1,)


def test_ensure_valid_levels_rejects_nonpositive():
    assert ensure_valid_levels(-1.0, 2.0, min_gap_pct=0.01) is None


def test_ensure_valid_levels_enforces_gap():
    buy, sell = ensure_valid_levels(100.0, 99.0, min_gap_pct=0.01)
    assert sell > buy


def test_directional_entry_amount_uses_sell_amount_for_short():
    action = {
        "buy_amount": 0.01,
        "sell_amount": 42.0,
        "trade_amount": 42.0,
    }
    assert directional_entry_amount(action, is_short=True) == pytest.approx(42.0)


def test_directional_entry_amount_uses_buy_amount_for_long():
    action = {
        "buy_amount": 37.5,
        "sell_amount": 2.0,
        "trade_amount": 37.5,
    }
    assert directional_entry_amount(action, is_short=False) == pytest.approx(37.5)


def test_directional_entry_amount_respects_explicit_zero_sell_amount_for_short():
    action = {
        "buy_amount": 25.0,
        "sell_amount": 0.0,
        "trade_amount": 25.0,
    }
    assert directional_entry_amount(action, is_short=True) == pytest.approx(0.0)


def test_directional_entry_amount_respects_explicit_zero_buy_amount_for_long():
    action = {
        "buy_amount": 0.0,
        "sell_amount": 25.0,
        "trade_amount": 25.0,
    }
    assert directional_entry_amount(action, is_short=False) == pytest.approx(0.0)


def test_entry_intensity_fraction_namedtuple_source():
    class Row:
        buy_amount = 0.1
        sell_amount = 25.0
        trade_amount = 25.0

    amount, intensity = entry_intensity_fraction(Row(), is_short=True, trade_amount_scale=100.0)
    assert amount == pytest.approx(25.0)
    assert intensity == pytest.approx(0.25)


def test_entry_intensity_fraction_power_boosts_small_signal():
    action = {"buy_amount": 25.0, "sell_amount": 0.0, "trade_amount": 25.0}
    amount, intensity = entry_intensity_fraction(
        action,
        is_short=False,
        trade_amount_scale=100.0,
        intensity_power=0.5,
    )
    assert amount == pytest.approx(25.0)
    assert intensity == pytest.approx(0.5)


def test_entry_intensity_fraction_applies_floor_and_multiplier():
    action = {"buy_amount": 2.0, "sell_amount": 0.0, "trade_amount": 2.0}
    amount, intensity = entry_intensity_fraction(
        action,
        is_short=False,
        trade_amount_scale=100.0,
        min_intensity_fraction=0.10,
        side_multiplier=2.0,
    )
    assert amount == pytest.approx(2.0)
    assert intensity == pytest.approx(0.10)


def test_allocation_usd_prefers_fixed():
    class Account:
        buying_power = 1000.0
        equity = 500.0

    assert _allocation_usd(Account(), allocation_usd=123.0, allocation_pct=0.5) == 123.0


def test_allocation_usd_pct_uses_buying_power():
    class Account:
        buying_power = 1000.0
        equity = 500.0

    assert _allocation_usd(Account(), allocation_usd=None, allocation_pct=0.1) == 100.0


def test_allocation_usd_pct_falls_back_to_equity():
    class Account:
        buying_power = 0.0
        equity = 400.0

    assert _allocation_usd(Account(), allocation_usd=None, allocation_pct=0.25) == 100.0


def test_build_order_intents_flat_prefers_buy_when_equal_notional():
    plan = TradingPlan(
        symbol="NVDA",
        buy_price=10.0,
        sell_price=11.0,
        buy_amount=50.0,
        sell_amount=50.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents = build_order_intents(
        plan,
        position_qty=0.0,
        allocation_usd=1000.0,
        buy_price=10.0,
        sell_price=11.0,
        can_long=True,
        can_short=True,
        allow_short=False,
        exit_only=False,
    )
    assert [(i.kind, i.side, round(i.qty, 6)) for i in intents] == [("entry", "buy", 50.0)]


def test_build_order_intents_flat_short_only_allows_short_entry_when_enabled():
    plan = TradingPlan(
        symbol="EBAY",
        buy_price=10.0,
        sell_price=11.0,
        buy_amount=80.0,
        sell_amount=20.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents = build_order_intents(
        plan,
        position_qty=0.0,
        allocation_usd=1000.0,
        buy_price=10.0,
        sell_price=11.0,
        can_long=False,
        can_short=True,
        allow_short=True,
        exit_only=False,
    )
    assert len(intents) == 1
    assert intents[0].kind == "entry"
    assert intents[0].side == "sell"
    assert intents[0].qty == pytest.approx(1000.0 * 0.20 / 11.0, rel=1e-9)


def test_build_order_intents_long_position_can_exit_and_add():
    plan = TradingPlan(
        symbol="NVDA",
        buy_price=10.0,
        sell_price=11.0,
        buy_amount=50.0,
        sell_amount=30.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents = build_order_intents(
        plan,
        position_qty=10.0,
        allocation_usd=1000.0,
        buy_price=10.0,
        sell_price=11.0,
        can_long=True,
        can_short=False,
        allow_short=False,
        exit_only=False,
    )
    assert [(i.kind, i.side, round(i.qty, 6)) for i in intents] == [
        ("exit", "sell", 3.0),
        ("entry", "buy", 50.0),
    ]


def test_build_order_intents_long_position_default_live_mode_full_exit_no_add():
    plan = TradingPlan(
        symbol="ETHUSD",
        buy_price=1900.0,
        sell_price=1970.0,
        buy_amount=80.0,
        sell_amount=10.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents = build_order_intents(
        plan,
        position_qty=5.5,
        allocation_usd=10_000.0,
        buy_price=1900.0,
        sell_price=1970.0,
        can_long=True,
        can_short=False,
        allow_short=False,
        exit_only=False,
        allow_position_adds=False,
        always_full_exit=True,
    )
    assert [(i.kind, i.side, round(i.qty, 6)) for i in intents] == [("exit", "sell", 5.5)]


def test_build_order_intents_long_position_no_add_blocks_same_side_rebuy_under_target():
    plan = TradingPlan(
        symbol="ETHUSD",
        buy_price=100.0,
        sell_price=105.0,
        buy_amount=80.0,
        sell_amount=10.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents = build_order_intents(
        plan,
        position_qty=2.0,
        allocation_usd=1_000.0,
        buy_price=100.0,
        sell_price=105.0,
        can_long=True,
        can_short=False,
        allow_short=False,
        exit_only=False,
        allow_position_adds=False,
        always_full_exit=True,
    )
    assert [(i.kind, i.side, round(i.qty, 6)) for i in intents] == [("exit", "sell", 2.0)]


def test_build_order_intents_short_position_default_live_mode_full_exit_no_add():
    plan = TradingPlan(
        symbol="EBAY",
        buy_price=10.0,
        sell_price=11.0,
        buy_amount=5.0,
        sell_amount=90.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents = build_order_intents(
        plan,
        position_qty=-8.0,
        allocation_usd=1000.0,
        buy_price=10.0,
        sell_price=11.0,
        can_long=False,
        can_short=True,
        allow_short=True,
        exit_only=False,
        allow_position_adds=False,
        always_full_exit=True,
    )
    assert [(i.kind, i.side, round(i.qty, 6)) for i in intents] == [("exit", "buy", 8.0)]


def test_build_order_intents_short_position_can_cover_and_add_short_when_enabled():
    plan = TradingPlan(
        symbol="EBAY",
        buy_price=10.0,
        sell_price=11.0,
        buy_amount=25.0,
        sell_amount=50.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents = build_order_intents(
        plan,
        position_qty=-8.0,
        allocation_usd=1000.0,
        buy_price=10.0,
        sell_price=11.0,
        can_long=False,
        can_short=True,
        allow_short=True,
        exit_only=False,
    )
    assert len(intents) == 2
    assert intents[0].kind == "exit"
    assert intents[0].side == "buy"
    assert intents[0].qty == pytest.approx(2.0, rel=1e-12)
    assert intents[1].kind == "entry"
    assert intents[1].side == "sell"
    assert intents[1].qty == pytest.approx(1000.0 * 0.50 / 11.0, rel=1e-9)


def test_build_order_intents_short_position_blocks_new_short_when_disabled():
    plan = TradingPlan(
        symbol="EBAY",
        buy_price=10.0,
        sell_price=11.0,
        buy_amount=25.0,
        sell_amount=50.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents = build_order_intents(
        plan,
        position_qty=-8.0,
        allocation_usd=1000.0,
        buy_price=10.0,
        sell_price=11.0,
        can_long=False,
        can_short=True,
        allow_short=False,
        exit_only=False,
    )
    assert [(i.kind, i.side, round(i.qty, 6)) for i in intents] == [("exit", "buy", 2.0)]


def test_build_order_intents_exit_only_flattens_long_and_short():
    plan = TradingPlan(
        symbol="NVDA",
        buy_price=10.0,
        sell_price=11.0,
        buy_amount=99.0,
        sell_amount=99.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents_long = build_order_intents(
        plan,
        position_qty=10.0,
        allocation_usd=1000.0,
        buy_price=10.0,
        sell_price=11.0,
        can_long=True,
        can_short=False,
        allow_short=False,
        exit_only=True,
    )
    assert [(i.kind, i.side, round(i.qty, 6)) for i in intents_long] == [("exit", "sell", 10.0)]

    intents_short = build_order_intents(
        plan,
        position_qty=-8.0,
        allocation_usd=1000.0,
        buy_price=10.0,
        sell_price=11.0,
        can_long=False,
        can_short=True,
        allow_short=True,
        exit_only=True,
    )
    assert [(i.kind, i.side, round(i.qty, 6)) for i in intents_short] == [("exit", "buy", 8.0)]


def _mock_live_order(
    *,
    order_id: str,
    symbol: str,
    side: str,
    qty: float,
    limit_price: float,
    created_at: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        limit_price=limit_price,
        created_at=created_at,
    )


def test_reconcile_live_symbol_orders_cancels_stale_opposite_side_order(monkeypatch: pytest.MonkeyPatch) -> None:
    stale_entry = _mock_live_order(
        order_id="buy-1",
        symbol="ETH/USD",
        side="buy",
        qty=1.0,
        limit_price=1900.0,
        created_at="2026-03-10T00:00:00Z",
    )
    matching_exit = _mock_live_order(
        order_id="sell-1",
        symbol="ETH/USD",
        side="sell",
        qty=2.0,
        limit_price=2000.0,
        created_at="2026-03-10T00:05:00Z",
    )
    cancelled: list[str] = []

    monkeypatch.setattr(
        "newnanoalpacahourlyexp.trade_alpaca_hourly.alpaca_wrapper.cancel_order",
        lambda order: cancelled.append(str(order.id)),
    )

    remaining = _reconcile_live_symbol_orders(
        "ETHUSD",
        position_qty=2.0,
        intents=[OrderIntent(side="sell", qty=2.0, limit_price=2000.0, kind="exit")],
        open_orders=[stale_entry, matching_exit],
        dry_run=False,
    )

    assert cancelled == ["buy-1"]
    assert [str(order.id) for order in remaining] == ["sell-1"]


def test_reconcile_live_symbol_orders_keeps_matching_entry_and_exit_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    matching_entry = _mock_live_order(
        order_id="buy-1",
        symbol="ETHUSD",
        side="buy",
        qty=3.0,
        limit_price=1900.0,
        created_at="2026-03-10T00:00:00Z",
    )
    matching_exit = _mock_live_order(
        order_id="sell-1",
        symbol="ETHUSD",
        side="sell",
        qty=2.0,
        limit_price=2000.0,
        created_at="2026-03-10T00:05:00Z",
    )
    cancelled: list[str] = []

    monkeypatch.setattr(
        "newnanoalpacahourlyexp.trade_alpaca_hourly.alpaca_wrapper.cancel_order",
        lambda order: cancelled.append(str(order.id)),
    )

    remaining = _reconcile_live_symbol_orders(
        "ETHUSD",
        position_qty=2.0,
        intents=[
            OrderIntent(side="sell", qty=2.0, limit_price=2000.0, kind="exit"),
            OrderIntent(side="buy", qty=3.0, limit_price=1900.0, kind="entry"),
        ],
        open_orders=[matching_entry, matching_exit],
        dry_run=False,
    )

    assert cancelled == []
    assert [str(order.id) for order in remaining] == ["buy-1", "sell-1"]
