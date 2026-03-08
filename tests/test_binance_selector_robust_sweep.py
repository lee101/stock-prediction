from __future__ import annotations

import pandas as pd
import pytest

from binanceexp1.sweep_multiasset_selector_robust import (
    build_start_state_kwargs,
    compute_selection_score,
)


def test_build_start_state_kwargs_for_flat_start() -> None:
    merged = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
            "symbol": ["BTCUSD"],
            "close": [100.0],
        }
    )

    kwargs = build_start_state_kwargs(merged, initial_cash=10_000.0, start_symbol=None)

    assert kwargs["initial_cash"] == pytest.approx(10_000.0)
    assert kwargs["initial_inventory"] == pytest.approx(0.0)
    assert kwargs["initial_symbol"] is None


def test_build_start_state_kwargs_seeds_full_notional_position() -> None:
    merged = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T00:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["BTCUSD", "BTCUSD", "ETHUSD"],
            "close": [200.0, 210.0, 50.0],
        }
    )

    kwargs = build_start_state_kwargs(
        merged,
        initial_cash=10_000.0,
        start_symbol="BTCUSD",
        position_fraction=1.0,
    )

    assert kwargs["initial_cash"] == pytest.approx(0.0)
    assert kwargs["initial_symbol"] == "BTCUSD"
    assert kwargs["initial_open_price"] == pytest.approx(200.0)
    assert kwargs["initial_inventory"] == pytest.approx(50.0)


def test_compute_selection_score_penalizes_negative_worst_case_and_low_activity() -> None:
    summary = {
        "robust_score": 12.0,
        "trade_count_mean": 3.0,
        "return_worst_pct": -0.5,
    }

    score = compute_selection_score(summary, min_trade_count_mean=6.0, require_all_positive=True)

    assert score == pytest.approx(12.0 - (0.75 * 3.0) - 100.0 - 5.0)
