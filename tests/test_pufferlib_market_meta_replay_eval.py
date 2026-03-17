from __future__ import annotations

import numpy as np
import pandas as pd

from pufferlib_market.meta_replay_eval import (
    align_daily_returns_by_intersection,
    combine_actions_by_winners,
    summarize_winner_series,
)


def test_align_daily_returns_by_intersection_uses_common_days() -> None:
    a_idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    b_idx = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"], utc=True)
    aligned = align_daily_returns_by_intersection(
        {
            "A": pd.Series([0.1, 0.2, 0.3], index=a_idx),
            "B": pd.Series([0.4, 0.5, 0.6], index=b_idx),
        }
    )

    expected_idx = pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
    assert list(aligned.keys()) == ["A", "B"]
    assert aligned["A"].index.equals(expected_idx)
    assert aligned["B"].index.equals(expected_idx)
    assert aligned["A"].tolist() == [0.2, 0.3]
    assert aligned["B"].tolist() == [0.4, 0.5]


def test_combine_actions_by_winners_flattens_cash_days() -> None:
    decision_days = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    winners = pd.Series(["A", None, "B"], index=decision_days)

    combined = combine_actions_by_winners(
        {
            "A": np.asarray([1, 1, 1], dtype=np.int32),
            "B": np.asarray([2, 2, 2], dtype=np.int32),
        },
        winners,
        decision_days=decision_days,
    )

    assert combined.tolist() == [1, 0, 2]


def test_summarize_winner_series_counts_switches_and_cash() -> None:
    winners = pd.Series(
        ["A", "A", None, "B", "B"],
        index=pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            utc=True,
        ),
    )

    summary = summarize_winner_series(winners)

    assert summary["switch_count"] == 2
    assert summary["winner_counts"] == {"A": 2, "B": 2, "cash": 1}
    assert summary["latest_winner"] == "B"
