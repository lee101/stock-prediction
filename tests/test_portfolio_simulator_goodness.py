from __future__ import annotations

import pandas as pd

from unified_hourly_experiment.marketsimulator.portfolio_simulator import PortfolioConfig, run_portfolio_simulation


def test_portfolio_simulation_emits_goodness_metrics() -> None:
    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 100.0, "high": 110.0, "low": 100.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": t0,
                "symbol": "NVDA",
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            },
            {
                "timestamp": t1,
                "symbol": "NVDA",
                "buy_price": 0.0,
                "sell_price": 0.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
                "trade_amount": 0.0,
            },
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=1_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"NVDA": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=0,
        bar_margin=0.0,
        int_qty=True,
        sim_backend="python",
    )

    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    assert result.metrics["final_equity"] == 1_200.0
    assert result.metrics["annualized_return"] == 0.0
    assert result.metrics["max_drawdown"] == 0.0
    assert result.metrics["pnl_smoothness"] == 0.0
    assert result.metrics["pnl_smoothness_score"] == 1.0
    assert result.metrics["ulcer_index"] == 0.0
    assert result.metrics["trade_rate"] == 1.0 / 24.0
    assert result.metrics["goodness_score"] > 0.0
