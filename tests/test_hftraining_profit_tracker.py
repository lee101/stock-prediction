from __future__ import annotations

import pytest

from hftraining.profit_tracker import ProfitTracker


def test_short_trade_return_uses_entry_notional_and_both_fees() -> None:
    tracker = ProfitTracker(
        commission=0.001,
        slippage=0.0,
        max_position_size=1.0,
        stop_loss=1.0,
        take_profit=1.0,
    )

    gain = tracker._simulate_trade(
        action=2,
        predicted_price=0.0,
        actual_price=90.0,
        current_price=100.0,
    )
    loss = tracker._simulate_trade(
        action=2,
        predicted_price=0.0,
        actual_price=110.0,
        current_price=100.0,
    )

    assert gain == pytest.approx(0.0981)
    assert loss == pytest.approx(-0.1021)
