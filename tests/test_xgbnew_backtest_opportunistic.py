from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from xgbnew.backtest import BacktestConfig, simulate


def _panel() -> pd.DataFrame:
    rows = [
        {
            "date": date(2025, 1, 2),
            "symbol": "AAA",
            "actual_open": 100.0,
            "actual_high": 104.0,
            "actual_low": 99.60,
            "actual_close": 101.0,
            "spread_bps": 1.0,
            "dolvol_20d_log": np.log1p(1e9),
        },
        {
            "date": date(2025, 1, 2),
            "symbol": "BBB",
            "actual_open": 100.0,
            "actual_high": 104.0,
            "actual_low": 99.76,
            "actual_close": 103.0,
            "spread_bps": 1.0,
            "dolvol_20d_log": np.log1p(1e9),
        },
    ]
    return pd.DataFrame(rows)


def test_opportunistic_limit_requires_penetration_by_fill_buffer() -> None:
    df = _panel()
    scores = pd.Series([0.90, 0.95], index=df.index)
    cfg = BacktestConfig(
        top_n=1,
        opportunistic_watch_n=2,
        opportunistic_entry_discount_bps=20.0,
        min_score=0.0,
        leverage=1.0,
        fee_rate=0.0,
        fill_buffer_bps=5.0,
        commission_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
    )

    result = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(result.day_results) == 1
    trade = result.day_results[0].trades[0]
    # BBB has the better score but its low only touches near the limit; AAA is
    # lower-scored but penetrates 99.80 by the extra 5 bps buffer.
    assert trade.symbol == "AAA"
    assert trade.entry_fill_price == pytest.approx(99.80)
    assert trade.exit_fill_price == pytest.approx(100.9495)


def test_opportunistic_limit_skips_day_when_no_watchlist_order_fills() -> None:
    df = _panel()
    df["actual_low"] = 99.76
    scores = pd.Series([0.95, 0.90], index=df.index)
    cfg = BacktestConfig(
        top_n=1,
        opportunistic_watch_n=2,
        opportunistic_entry_discount_bps=20.0,
        min_score=0.0,
        leverage=1.0,
        fee_rate=0.0,
        fill_buffer_bps=5.0,
        commission_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
    )

    result = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert result.day_results == []
    assert result.total_trades == 0
