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


def test_short_candidates_use_bottom_scores_and_conservative_fills() -> None:
    df = _panel()
    df.loc[df["symbol"] == "BBB", "actual_close"] = 98.0
    scores = pd.Series([0.90, 0.10], index=df.index)
    cfg = BacktestConfig(
        top_n=1,
        short_n=1,
        max_short_score=0.45,
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
    trades = {trade.symbol: trade for trade in result.day_results[0].trades}
    assert trades["AAA"].side == 1
    assert trades["BBB"].side == -1
    assert trades["BBB"].entry_fill_price == pytest.approx(99.95)
    assert trades["BBB"].exit_fill_price == pytest.approx(98.049)
    assert trades["BBB"].net_return_pct > 0


def test_short_zero_score_is_eligible_and_fees_use_entry_notional() -> None:
    df = pd.DataFrame([
        {
            "date": date(2025, 1, 2),
            "symbol": "AAA",
            "actual_open": 100.0,
            "actual_high": 101.0,
            "actual_low": 89.0,
            "actual_close": 90.0,
            "spread_bps": 1.0,
            "dolvol_20d_log": np.log1p(1e9),
        },
    ])
    scores = pd.Series([0.0], index=df.index)
    cfg = BacktestConfig(
        top_n=0,
        short_n=1,
        max_short_score=0.45,
        leverage=1.0,
        fee_rate=0.001,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
    )

    result = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(result.day_results) == 1
    trade = result.day_results[0].trades[0]
    assert trade.symbol == "AAA"
    assert trade.side == -1
    assert trade.score == 0.0
    assert trade.gross_return_pct == pytest.approx(10.0)
    # Short sale proceeds pay fee at entry and cover cost pays fee at exit,
    # both measured against entry notional: (99.9 - 90.09) / 100 = 9.81%.
    assert trade.net_return_pct == pytest.approx(9.81)


def test_hold_through_carries_side_aware_pick_when_next_day_has_no_pick() -> None:
    df = pd.DataFrame([
        {
            "date": date(2025, 1, 2),
            "symbol": "AAA",
            "actual_open": 100.0,
            "actual_high": 102.0,
            "actual_low": 99.0,
            "actual_close": 101.0,
            "spread_bps": 1.0,
            "dolvol_20d_log": np.log1p(1e9),
        },
        {
            "date": date(2025, 1, 3),
            "symbol": "AAA",
            "actual_open": 101.0,
            "actual_high": 104.0,
            "actual_low": 100.0,
            "actual_close": 103.0,
            "spread_bps": 1.0,
            "dolvol_20d_log": np.log1p(1e9),
        },
    ])
    scores = pd.Series([0.90, 0.10], index=df.index)
    cfg = BacktestConfig(
        top_n=1,
        min_score=0.85,
        hold_through=True,
        leverage=1.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
    )

    result = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(result.day_results) == 2
    carry = result.day_results[1].trades[0]
    assert carry.symbol == "AAA"
    assert carry.side == 1
    assert carry.entry_fill_price == pytest.approx(101.0)
    assert carry.exit_fill_price == pytest.approx(103.0)
    assert carry.net_return_pct == pytest.approx((103.0 - 101.0) / 101.0 * 100.0)


def test_opportunistic_short_limit_requires_high_penetration() -> None:
    df = _panel()
    df.loc[df["symbol"] == "AAA", ["actual_high", "actual_close"]] = [100.24, 98.0]
    df.loc[df["symbol"] == "BBB", ["actual_high", "actual_close"]] = [100.26, 97.0]
    scores = pd.Series([0.10, 0.20], index=df.index)
    cfg = BacktestConfig(
        top_n=0,
        short_n=1,
        max_short_score=0.45,
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
    # AAA is the lower score, but its high only touches near the short limit.
    # BBB penetrates 100.20 by the extra 5 bps fill buffer.
    assert trade.symbol == "BBB"
    assert trade.side == -1
    assert trade.entry_fill_price == pytest.approx(100.20)
    assert trade.exit_fill_price == pytest.approx(97.0485)


def test_worksteal_allocation_front_loads_first_filled_trade() -> None:
    df = pd.DataFrame([
        {
            "date": date(2025, 1, 2),
            "symbol": "AAA",
            "actual_open": 100.0,
            "actual_high": 111.0,
            "actual_low": 99.0,
            "actual_close": 110.0,
            "spread_bps": 1.0,
            "dolvol_20d_log": np.log1p(1e9),
        },
        {
            "date": date(2025, 1, 2),
            "symbol": "BBB",
            "actual_open": 100.0,
            "actual_high": 101.0,
            "actual_low": 89.0,
            "actual_close": 90.0,
            "spread_bps": 1.0,
            "dolvol_20d_log": np.log1p(1e9),
        },
    ])
    scores = pd.Series([0.90, 0.80], index=df.index)
    cfg = BacktestConfig(
        top_n=2,
        allocation_mode="worksteal",
        leverage=2.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
    )

    result = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(result.day_results) == 1
    day = result.day_results[0]
    assert [trade.symbol for trade in day.trades] == ["AAA", "BBB"]
    assert day.trades[0].net_return_pct > 19.9
    assert day.trades[1].net_return_pct < -19.9
    # First fill receives 75% of the gross book and the second receives 25%.
    # At 2x leverage this is equivalent to 150% + 50% exposure.
    assert day.daily_return_pct == pytest.approx(
        0.75 * day.trades[0].net_return_pct + 0.25 * day.trades[1].net_return_pct
    )
