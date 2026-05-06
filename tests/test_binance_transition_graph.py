from __future__ import annotations

import pandas as pd

from experiments.binance_transition_graph.graph_policy import GraphPolicyConfig, simulate_graph_policy


def test_one_x_short_position_pays_borrow_on_flat_prices() -> None:
    idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
        },
        index=idx,
    )
    frames = {"BTCUSDT": frame}
    common = dict(
        fast_lookback_hours=1,
        slow_lookback_hours=2,
        regime_lookback_hours=4,
        rebalance_hours=1,
        target_leverage=1.0,
        max_leverage=1.0,
        allow_short=True,
        momentum_threshold=999.0,
        dip_buy_threshold=-999.0,
        rotation_spread_threshold=999.0,
        short_threshold=1.0,
        market_regime_cash_threshold=-999.0,
        market_regime_short_threshold=1.0,
        volatility_cash_threshold=999.0,
        drawdown_cash_threshold=-999.0,
        stop_loss=-999.0,
        take_profit=999.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        decision_lag_hours=0,
    )

    no_borrow = simulate_graph_policy(
        frames,
        GraphPolicyConfig(name="no_borrow", borrow_apr=0.0, **common),
        warmup_rows=0,
    )
    with_borrow = simulate_graph_policy(
        frames,
        GraphPolicyConfig(name="with_borrow", borrow_apr=0.50, **common),
        warmup_rows=0,
    )

    assert no_borrow.metrics.final_equity == 10000.0
    assert with_borrow.metrics.final_equity < no_borrow.metrics.final_equity
    assert "SHORT:BTCUSDT" in with_borrow.state_counts
