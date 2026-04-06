from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.autoresearch_stock.prepare import TaskConfig, build_action_frame, simulate_actions


def test_build_action_frame_scales_stock_strength_to_2x() -> None:
    action_rows = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-03T14:30:00Z"),
                "symbol": "AAPL",
                "close": 100.0,
                "spread_bps": 0.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-03T14:30:00Z"),
                "symbol": "BTCUSD",
                "close": 100.0,
                "spread_bps": 0.0,
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            [0.20, -0.01, 0.20],
            [0.20, -0.01, 0.20],
        ]
    ).to_numpy(dtype="float32")
    config = TaskConfig(
        frequency="daily",
        data_root=Path("trainingdata"),
        recent_data_root=None,
        symbols=("AAPL", "BTCUSD"),
        sequence_length=8,
        hold_bars=5,
        eval_windows=(20,),
    )

    actions = build_action_frame(action_rows, predictions, config)

    assert actions.loc[actions["symbol"] == "AAPL", "buy_amount"].iloc[0] == pytest.approx(200.0)
    assert actions.loc[actions["symbol"] == "BTCUSD", "buy_amount"].iloc[0] == pytest.approx(100.0)


def test_simulate_actions_caps_stock_trade_at_2x_and_charges_financing() -> None:
    bars = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-03T14:30:00Z"),
                "symbol": "AAPL",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 10_000_000.0,
                "is_session_last": False,
            },
            {
                "timestamp": pd.Timestamp("2026-03-04T14:30:00Z"),
                "symbol": "AAPL",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 10_000_000.0,
                "is_session_last": False,
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-03T14:30:00Z"),
                "symbol": "AAPL",
                "side": "long",
                "strength": 2.0,
                "target_price": 110.0,
                "spread_bps": 0.0,
                "expected_edge": 0.10,
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 300.0,
                "sell_amount": 0.0,
                "trade_amount": 300.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-04T14:30:00Z"),
                "symbol": "AAPL",
                "side": "flat",
                "strength": 0.0,
                "target_price": 100.0,
                "spread_bps": 0.0,
                "expected_edge": 0.0,
                "buy_price": 0.0,
                "sell_price": 0.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
                "trade_amount": 0.0,
            }
        ]
    )
    base_config = TaskConfig(
        frequency="daily",
        data_root=Path("trainingdata"),
        recent_data_root=None,
        symbols=("AAPL",),
        sequence_length=8,
        hold_bars=5,
        eval_windows=(20,),
        max_positions=1,
        max_volume_fraction=1.0,
        entry_slippage_bps=0.0,
        exit_slippage_bps=0.0,
        close_at_session_end=False,
        annual_leverage_rate=0.0,
        max_gross_leverage=2.0,
    )

    no_financing = simulate_actions(bars, actions, base_config)
    with_financing = simulate_actions(
        bars,
        actions,
        TaskConfig(**{**base_config.__dict__, "annual_leverage_rate": 0.0625}),
    )

    assert no_financing["trades"][0].quantity == pytest.approx(200.0)
    assert with_financing["return_pct"] < no_financing["return_pct"]
