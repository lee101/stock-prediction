from __future__ import annotations

import pytest

from src.margin_position_utils import (
    choose_flat_entry_side,
    directional_signal,
    position_side_from_qty,
    remaining_entry_notional,
)


def test_position_side_from_qty_respects_step_size() -> None:
    assert position_side_from_qty(0.4, step_size=1.0) == ""
    assert position_side_from_qty(1.1, step_size=1.0) == "long"
    assert position_side_from_qty(-1.1, step_size=1.0) == "short"


def test_directional_signal_maps_long_and_short_prices() -> None:
    signal = {
        "buy_price": 100.0,
        "sell_price": 110.0,
        "buy_amount": 25.0,
        "sell_amount": 40.0,
    }
    long_signal = directional_signal(signal, side="long")
    short_signal = directional_signal(signal, side="short")

    assert long_signal.entry_price == pytest.approx(100.0)
    assert long_signal.exit_price == pytest.approx(110.0)
    assert long_signal.entry_amount == pytest.approx(25.0)
    assert long_signal.exit_amount == pytest.approx(40.0)

    assert short_signal.entry_price == pytest.approx(110.0)
    assert short_signal.exit_price == pytest.approx(100.0)
    assert short_signal.entry_amount == pytest.approx(40.0)
    assert short_signal.exit_amount == pytest.approx(25.0)


def test_choose_flat_entry_side_prefers_stronger_short_when_enabled() -> None:
    signal = {
        "buy_price": 100.0,
        "sell_price": 110.0,
        "buy_amount": 30.0,
        "sell_amount": 55.0,
    }
    assert choose_flat_entry_side(signal, allow_short=False) == "long"
    assert choose_flat_entry_side(signal, allow_short=True) == "short"


def test_remaining_entry_notional_uses_side_specific_leverage() -> None:
    assert remaining_entry_notional(
        side="long",
        equity=1000.0,
        current_qty=3.0,
        market_price=100.0,
        long_max_leverage=2.0,
        short_max_leverage=0.5,
    ) == pytest.approx(1700.0)
    assert remaining_entry_notional(
        side="short",
        equity=1000.0,
        current_qty=-2.0,
        market_price=100.0,
        long_max_leverage=2.0,
        short_max_leverage=0.5,
    ) == pytest.approx(300.0)


def test_remaining_entry_notional_sizes_through_opposite_side_inventory() -> None:
    assert remaining_entry_notional(
        side="long",
        equity=1000.0,
        current_qty=-2.0,
        market_price=100.0,
        long_max_leverage=2.0,
        short_max_leverage=0.5,
    ) == pytest.approx(2200.0)
    assert remaining_entry_notional(
        side="short",
        equity=1000.0,
        current_qty=3.0,
        market_price=100.0,
        long_max_leverage=2.0,
        short_max_leverage=0.5,
    ) == pytest.approx(800.0)
