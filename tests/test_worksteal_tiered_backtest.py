from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.strategy import (
    WorkStealConfig,
    build_tiered_entry_candidates,
    run_worksteal_backtest,
)


def _make_daily_bars(rows: list[dict], symbol: str) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    data = []
    for idx, row in enumerate(rows):
        close = float(row["close"])
        data.append(
            {
                "timestamp": start + pd.Timedelta(days=idx),
                "open": float(row.get("open", close)),
                "high": float(row.get("high", close)),
                "low": float(row.get("low", close)),
                "close": close,
                "volume": float(row.get("volume", 1_000.0)),
                "symbol": symbol,
            }
        )
    return pd.DataFrame(data)


def test_build_tiered_entry_candidates_uses_shallower_fallback_tier():
    symbol = "TIERUSD"
    history = {
        symbol: _make_daily_bars(
            [{"close": 100.0}] * 20 + [{"open": 100.0, "high": 100.0, "low": 89.0, "close": 89.0}],
            symbol,
        )
    }
    current_bars = {symbol: history[symbol].iloc[-1]}
    config = WorkStealConfig(
        dip_pct=0.18,
        dip_pct_fallback=(0.18, 0.15, 0.12),
        proximity_pct=0.02,
        lookback_days=20,
        sma_filter_period=0,
    )

    candidates, tier_map = build_tiered_entry_candidates(
        date=pd.Timestamp(current_bars[symbol]["timestamp"]),
        current_bars=current_bars,
        history=history,
        positions={},
        last_exit={},
        config=config,
    )

    assert [candidate[0] for candidate in candidates] == [symbol]
    assert tier_map == {symbol: 2}


def test_run_worksteal_backtest_respects_dip_pct_fallback_for_entries():
    symbol = "TIERUSD"
    bars = {
        symbol: _make_daily_bars(
            [{"close": 100.0}] * 20
            + [{"open": 100.0, "high": 100.0, "low": 89.0, "close": 89.0}]
            + [{"open": 89.0, "high": 110.0, "low": 88.0, "close": 109.0}],
            symbol,
        )
    }

    base_kwargs = dict(
        dip_pct=0.18,
        proximity_pct=0.02,
        profit_target_pct=0.20,
        stop_loss_pct=0.15,
        lookback_days=20,
        sma_filter_period=0,
        max_positions=1,
        max_hold_days=14,
        trailing_stop_pct=0.0,
        initial_cash=10_000.0,
        maker_fee=0.0,
        fdusd_fee=0.0,
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=0,
    )
    no_fallback = WorkStealConfig(**base_kwargs)
    with_fallback = WorkStealConfig(**base_kwargs, dip_pct_fallback=(0.18, 0.15, 0.12))

    _eq_base, trades_base, metrics_base = run_worksteal_backtest({k: v.copy() for k, v in bars.items()}, no_fallback)
    _eq_tiered, trades_tiered, metrics_tiered = run_worksteal_backtest({k: v.copy() for k, v in bars.items()}, with_fallback)

    assert [trade.side for trade in trades_base] == []
    assert metrics_base["n_trades"] == 0

    assert [trade.side for trade in trades_tiered] == ["buy", "sell"]
    assert metrics_tiered["n_trades"] == 1
    assert metrics_tiered["total_return_pct"] > 0.0
