from __future__ import annotations

import pandas as pd
import pytest

from unified_hourly_experiment.meta_selector import (
    combine_actions_by_winners,
    score_trailing_returns,
    select_daily_winners,
)


def test_select_daily_winners_uses_previous_day_performance() -> None:
    idx = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
            "2026-01-04T00:00:00Z",
        ],
        utc=True,
    )
    returns_by_strategy = {
        "A": pd.Series([0.10, -0.03, -0.02, 0.01], index=idx),
        "B": pd.Series([0.02, 0.05, 0.01, -0.01], index=idx),
    }

    winners = select_daily_winners(
        returns_by_strategy,
        lookback_days=1,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
    )

    assert winners.iloc[0] == "A"  # fallback
    assert winners.iloc[1] == "A"  # day1: A=+10% > B=+2%
    assert winners.iloc[2] == "B"  # day2: B=+5% > A=-3%
    assert winners.iloc[3] == "B"  # day3: B=+1% > A=-2%


def test_select_daily_winners_tie_is_deterministic_with_priority() -> None:
    idx = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ],
        utc=True,
    )
    # Identical trailing returns -> tie should follow tie_break_order.
    returns_by_strategy = {
        "A": pd.Series([0.01, 0.02, 0.00], index=idx),
        "B": pd.Series([0.01, 0.02, 0.00], index=idx),
    }
    winners = select_daily_winners(
        returns_by_strategy,
        lookback_days=2,
        metric="return",
        fallback_strategy="B",
        tie_break_order=["B", "A"],
    )

    assert winners.iloc[0] == "B"
    assert winners.iloc[1] == "B"
    assert winners.iloc[2] == "B"


def test_select_daily_winners_fallback_when_window_insufficient() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    returns_by_strategy = {
        "A": pd.Series([0.02, 0.01, -0.01], index=idx),
        "B": pd.Series([0.01, 0.02, 0.00], index=idx),
    }

    winners = select_daily_winners(
        returns_by_strategy,
        lookback_days=3,
        metric="calmar",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
        require_full_window=True,
    )

    # No day has a full 3-day trailing window before it.
    assert winners.tolist() == ["A", "A", "A"]


def test_select_daily_winners_can_sit_out_when_all_scores_negative() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    returns_by_strategy = {
        "A": pd.Series([-0.02, -0.01, 0.00], index=idx),
        "B": pd.Series([-0.03, -0.02, 0.00], index=idx),
    }

    winners = select_daily_winners(
        returns_by_strategy,
        lookback_days=1,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
        sit_out_threshold=0.0,
    )

    assert winners.iloc[0] == "A"  # fallback day
    assert winners.iloc[1] is None
    assert winners.iloc[2] is None


def test_select_daily_winners_winner_cash_uses_min_score_gap() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    returns_by_strategy = {
        "A": pd.Series([0.0100, 0.0100, 0.0000], index=idx),
        "B": pd.Series([0.0095, 0.0100, 0.0000], index=idx),
    }

    winners = select_daily_winners(
        returns_by_strategy,
        lookback_days=1,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
        selection_mode="winner_cash",
        min_score_gap=0.001,
    )

    assert winners.tolist() == ["A", None, None]


def test_select_daily_winners_sticky_respects_switch_margin() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], utc=True)
    returns_by_strategy = {
        "A": pd.Series([0.040, 0.000, 0.000, 0.000], index=idx),
        "B": pd.Series([0.030, 0.041, 0.000, 0.000], index=idx),
    }

    winners = select_daily_winners(
        returns_by_strategy,
        lookback_days=1,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
        selection_mode="sticky",
        switch_margin=0.05,
    )

    assert winners.tolist() == ["A", "A", "A", "A"]


def test_combine_actions_by_winners_picks_expected_strategy_rows() -> None:
    ts = pd.to_datetime(
        [
            "2026-01-01T14:00:00Z",
            "2026-01-01T15:00:00Z",
            "2026-01-02T14:00:00Z",
            "2026-01-02T15:00:00Z",
        ],
        utc=True,
    )
    actions_a = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["NVDA", "NVDA", "NVDA", "NVDA"],
            "buy_price": [100, 101, 102, 103],
            "sell_price": [105, 106, 107, 108],
            "buy_amount": [0.2, 0.2, 0.2, 0.2],
            "sell_amount": [0.0, 0.0, 0.0, 0.0],
        }
    )
    actions_b = actions_a.copy()
    actions_b["buy_amount"] = 0.8

    winners_by_symbol = {
        "NVDA": pd.Series(
            ["A", "B"],
            index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        )
    }

    combined = combine_actions_by_winners({"A": actions_a, "B": actions_b}, winners_by_symbol)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    assert combined["symbol"].tolist() == ["NVDA", "NVDA", "NVDA", "NVDA"]
    assert combined.loc[0, "buy_amount"] == 0.2
    assert combined.loc[1, "buy_amount"] == 0.2
    assert combined.loc[2, "buy_amount"] == 0.8
    assert combined.loc[3, "buy_amount"] == 0.8


def test_combine_actions_by_winners_sets_zero_amounts_on_cash_days() -> None:
    ts = pd.to_datetime(
        [
            "2026-01-01T14:00:00Z",
            "2026-01-01T15:00:00Z",
            "2026-01-02T14:00:00Z",
            "2026-01-02T15:00:00Z",
        ],
        utc=True,
    )
    actions_a = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["NVDA", "NVDA", "NVDA", "NVDA"],
            "buy_price": [100, 101, 102, 103],
            "sell_price": [105, 106, 107, 108],
            "buy_amount": [0.5, 0.5, 0.5, 0.5],
            "sell_amount": [0.0, 0.0, 0.0, 0.0],
            "trade_amount": [0.5, 0.5, 0.5, 0.5],
        }
    )
    actions_b = actions_a.copy()
    actions_b["buy_amount"] = 0.7
    actions_b["trade_amount"] = 0.7

    winners_by_symbol = {
        "NVDA": pd.Series(
            ["A", None],
            index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        )
    }
    combined = combine_actions_by_winners({"A": actions_a, "B": actions_b}, winners_by_symbol)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    assert combined.loc[0, "buy_amount"] == 0.5
    assert combined.loc[1, "buy_amount"] == 0.5
    assert combined.loc[2, "buy_amount"] == 0.0
    assert combined.loc[3, "buy_amount"] == 0.0
    assert combined.loc[2, "trade_amount"] == 0.0
    assert combined.loc[3, "trade_amount"] == 0.0


def test_score_trailing_returns_supports_all_metrics() -> None:
    window = [0.02, -0.01, 0.03, 0.01]
    assert score_trailing_returns(window, "return") != 0.0
    assert score_trailing_returns(window, "sortino") != 0.0
    assert score_trailing_returns(window, "sharpe") != 0.0
    assert score_trailing_returns(window, "calmar") != 0.0
    assert score_trailing_returns(window, "omega") != 0.0
    assert score_trailing_returns(window, "gain_pain") != 0.0
    assert score_trailing_returns(window, "p10") != 0.0
    assert score_trailing_returns(window, "median") != 0.0


def test_select_daily_winners_recency_halflife_changes_winner() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    returns_by_strategy = {
        "A": pd.Series([-0.30, 0.20, 0.00], index=idx),
        "B": pd.Series([0.01, 0.01, 0.00], index=idx),
    }

    winners_unweighted = select_daily_winners(
        returns_by_strategy,
        lookback_days=2,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
        require_full_window=True,
    )
    winners_weighted = select_daily_winners(
        returns_by_strategy,
        lookback_days=2,
        metric="return",
        fallback_strategy="A",
        tie_break_order=["A", "B"],
        require_full_window=True,
        recency_halflife_days=0.5,
    )

    assert winners_unweighted.iloc[2] == "B"
    assert winners_weighted.iloc[2] == "A"


def test_select_daily_winners_recency_halflife_must_be_positive_when_provided() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    returns_by_strategy = {
        "A": pd.Series([0.01, 0.01], index=idx),
        "B": pd.Series([0.00, 0.00], index=idx),
    }
    with pytest.raises(ValueError, match="recency_halflife_days must be > 0"):
        select_daily_winners(
            returns_by_strategy,
            lookback_days=1,
            metric="return",
            fallback_strategy="A",
            recency_halflife_days=0.0,
        )
