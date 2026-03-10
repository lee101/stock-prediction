from __future__ import annotations

import pandas as pd
import pytest

from newnanoalpacahourlystoploss.run_stoploss_sweep import _attach_baseline_deltas, _max_drawdown


def test_max_drawdown_returns_most_negative_peak_to_trough_move() -> None:
    equity = pd.Series([100.0, 110.0, 90.0, 95.0, 120.0])
    assert _max_drawdown(equity) == (90.0 / 110.0) - 1.0


def test_attach_baseline_deltas_uses_matching_non_stop_configuration() -> None:
    summary = pd.DataFrame(
        [
            {
                "symbols": "ETHUSD,BTCUSD",
                "fill_buffer_bps": 5.0,
                "intensity_scale": 1.0,
                "price_offset_pct": 0.0,
                "stop_loss_pct": 0.0,
                "stop_loss_slippage_pct": 0.0,
                "stop_loss_cooldown_bars": 0,
                "total_return": 0.10,
                "sortino": 1.2,
                "max_drawdown": -0.08,
                "fills_total": 12,
            },
            {
                "symbols": "ETHUSD,BTCUSD",
                "fill_buffer_bps": 5.0,
                "intensity_scale": 1.0,
                "price_offset_pct": 0.0,
                "stop_loss_pct": 0.02,
                "stop_loss_slippage_pct": 0.0005,
                "stop_loss_cooldown_bars": 1,
                "total_return": 0.13,
                "sortino": 1.5,
                "max_drawdown": -0.05,
                "fills_total": 10,
            },
        ]
    )

    enriched = _attach_baseline_deltas(summary)
    candidate = enriched.loc[enriched["stop_loss_pct"] > 0.0].iloc[0]

    assert candidate["baseline_total_return"] == 0.10
    assert candidate["baseline_sortino"] == 1.2
    assert candidate["delta_total_return"] == pytest.approx(0.03)
    assert candidate["delta_sortino"] == pytest.approx(0.3)
    assert candidate["delta_max_drawdown"] == pytest.approx(0.03)
    assert candidate["delta_fills_total"] == -2
