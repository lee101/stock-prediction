from __future__ import annotations

import json

import pandas as pd
import pytest

try:
    from unified_hourly_experiment.marketsimulator import (
        PortfolioConfig,
        run_portfolio_simulation,
        write_portfolio_simulation_artifacts,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("Required module unified_hourly_experiment.marketsimulator (or write_portfolio_simulation_artifacts) not available", allow_module_level=True)


def test_write_portfolio_simulation_artifacts_writes_overlay_and_csvs(tmp_path) -> None:
    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 100.0, "high": 111.0, "low": 99.5, "close": 110.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 100.0, "sell_price": 110.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "buy_price": 100.0, "sell_price": 110.0, "buy_amount": 0.0, "sell_amount": 100.0, "trade_amount": 0.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        max_leverage=1.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"NVDA": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1000,
        bar_margin=0.0,
        int_qty=True,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    paths = write_portfolio_simulation_artifacts(
        bars=bars,
        result=result,
        output_dir=tmp_path,
        file_stem="nvda_demo",
    )

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    trades_df = pd.read_csv(paths["trades_csv"])
    assert trades_df["side"].tolist() == ["buy", "sell"]

    bars_df = pd.read_csv(paths["bars_csv"])
    assert bars_df["close"].tolist() == [100.0, 110.0]

    with paths["metrics_json"].open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert "final_equity" in metrics
