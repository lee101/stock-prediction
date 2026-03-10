from __future__ import annotations

import pandas as pd
import pytest

from fastalgorithms.per_stock.meta_selector import MetaSelectorConfig, run_meta_simulation


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-01", periods=len(values), freq="h", tz="UTC"),
            "equity": values,
            "in_position": [True] * len(values),
        }
    )


def test_run_meta_simulation_avoids_lookahead_bias() -> None:
    strategy_frames = {
        "A": _frame([10_000.0, 9_000.0, 9_000.0, 9_000.0]),
        "B": _frame([10_000.0, 11_000.0, 11_000.0, 11_000.0]),
    }
    result = run_meta_simulation(
        strategy_frames,
        MetaSelectorConfig(
            lookback_hours=2,
            min_window_for_sortino=1,
            sit_out_if_all_negative=False,
            reeval_every_n_hours=1,
            periodic_reeval_for_active=True,
            initial_cash=10_000.0,
            selection_method="winner",
        ),
    )

    assert result.equity_curve[-1] == pytest.approx(9_000.0)
    assert result.total_return == pytest.approx(-0.1)


def test_run_meta_simulation_softmax_blends_recent_winners() -> None:
    strategy_frames = {
        "A": _frame([10_000.0, 10_200.0, 10_400.0, 10_600.0, 10_800.0]),
        "B": _frame([10_000.0, 9_800.0, 9_600.0, 9_400.0, 9_200.0]),
    }
    result = run_meta_simulation(
        strategy_frames,
        MetaSelectorConfig(
            lookback_hours=3,
            min_window_for_sortino=1,
            sit_out_if_all_negative=False,
            reeval_every_n_hours=1,
            periodic_reeval_for_active=True,
            initial_cash=10_000.0,
            selection_method="softmax",
            softmax_temperature=0.5,
            top_k=2,
            min_score=-1.0,
        ),
    )

    assert result.equity_curve[-1] > 10_000.0
    assert result.equity_curve[-1] < 10_800.0
    assert result.num_switches >= 1
