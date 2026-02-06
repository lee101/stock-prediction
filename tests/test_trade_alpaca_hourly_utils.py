from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from newnanoalpacahourlyexp.trade_alpaca_hourly import (
    TradingPlan,
    _build_order_intents,
    _allocation_usd,
    _ensure_valid_levels,
    _parse_checkpoint_map,
    _parse_horizon_map,
    _parse_symbols,
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
    assert _ensure_valid_levels(-1.0, 2.0, min_gap_pct=0.01) is None


def test_ensure_valid_levels_enforces_gap():
    buy, sell = _ensure_valid_levels(100.0, 99.0, min_gap_pct=0.01)
    assert sell > buy


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
    intents = _build_order_intents(
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
    intents = _build_order_intents(
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
    intents = _build_order_intents(
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


def test_build_order_intents_short_position_can_cover_and_add_short_when_enabled():
    plan = TradingPlan(
        symbol="EBAY",
        buy_price=10.0,
        sell_price=11.0,
        buy_amount=25.0,
        sell_amount=50.0,
        timestamp=datetime.now(timezone.utc),
    )
    intents = _build_order_intents(
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
    intents = _build_order_intents(
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
    intents_long = _build_order_intents(
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

    intents_short = _build_order_intents(
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
