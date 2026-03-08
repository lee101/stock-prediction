from __future__ import annotations

import math

import pandas as pd

from unified_hourly_experiment.meta_live_runtime import choose_latest_winner, compute_symbol_edge


def test_choose_latest_winner_uses_latest_day() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    returns = {
        "A": pd.Series([0.01, -0.02, 0.01], index=idx),
        "B": pd.Series([0.02, 0.03, 0.00], index=idx),
    }
    winner = choose_latest_winner(
        returns,
        lookback_days=1,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
    )
    assert winner == "B"


def test_choose_latest_winner_can_sit_out() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    returns = {
        "A": pd.Series([-0.02, -0.01, 0.00], index=idx),
        "B": pd.Series([-0.03, -0.02, 0.00], index=idx),
    }
    winner = choose_latest_winner(
        returns,
        lookback_days=1,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
        sit_out_threshold=0.0,
    )
    assert winner is None


def test_choose_latest_winner_supports_mode_and_gap() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    returns = {
        "A": pd.Series([0.0100, 0.0100, 0.0000], index=idx),
        "B": pd.Series([0.0098, 0.0100, 0.0000], index=idx),
    }
    winner = choose_latest_winner(
        returns,
        lookback_days=1,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
        selection_mode="winner_cash",
        min_score_gap=0.001,
    )
    assert winner is None


def test_choose_latest_winner_supports_recency_halflife() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    returns = {
        "A": pd.Series([-0.30, 0.20, 0.00], index=idx),
        "B": pd.Series([0.01, 0.01, 0.00], index=idx),
    }
    winner = choose_latest_winner(
        returns,
        lookback_days=2,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
        recency_halflife_days=0.5,
    )
    assert winner == "A"


def test_compute_symbol_edge_long_and_short() -> None:
    long_edge = compute_symbol_edge(
        symbol="NVDA",
        action={"buy_price": 100.0, "predicted_high": 102.0, "sell_price": 101.0, "predicted_low": 98.0},
        fee_rate=0.001,
        short_only_symbols=["TRIP"],
    )
    assert math.isclose(long_edge, 0.019, rel_tol=1e-9)

    short_edge = compute_symbol_edge(
        symbol="TRIP",
        action={"buy_price": 99.0, "predicted_high": 100.0, "sell_price": 101.0, "predicted_low": 99.0},
        fee_rate=0.001,
        short_only_symbols=["TRIP"],
    )
    expected = (101.0 - 99.0) / 101.0 - 0.001
    assert math.isclose(short_edge, expected, rel_tol=1e-9)
