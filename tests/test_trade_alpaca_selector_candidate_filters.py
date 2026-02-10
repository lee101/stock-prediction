from __future__ import annotations

from datetime import datetime, timezone

from newnanoalpacahourlyexp.trade_alpaca_selector import _build_candidate


def test_build_candidate_require_min_edge_controls_filtering():
    symbol = "BTCUSD"
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    action = {
        "buy_price": 100.0,
        "sell_price": 101.0,
        "buy_amount": 10.0,
        "sell_amount": 10.0,
    }
    bar_row = {
        "timestamp": ts,
        "close": 100.0,
        "low": 99.5,
        "high": 100.5,
        "predicted_high_p50_h1": 100.2,
        "predicted_low_p50_h1": 99.8,
        "predicted_close_p50_h1": 100.0,
    }

    # Edge is small; require_min_edge should drop it, but exit-path should still produce a candidate.
    candidate_filtered = _build_candidate(
        symbol=symbol,
        action=action,
        bar_row=bar_row,
        horizon=1,
        intensity_scale=1.0,
        price_offset_pct=0.0,
        min_gap_pct=0.001,
        dip_threshold_pct=0.0,
        fee_rate=0.0,
        risk_weight=0.3,
        edge_mode="high_low",
        min_edge=0.01,
        require_min_edge=True,
    )
    assert candidate_filtered is None

    candidate_unfiltered = _build_candidate(
        symbol=symbol,
        action=action,
        bar_row=bar_row,
        horizon=1,
        intensity_scale=1.0,
        price_offset_pct=0.0,
        min_gap_pct=0.001,
        dip_threshold_pct=0.0,
        fee_rate=0.0,
        risk_weight=0.3,
        edge_mode="high_low",
        min_edge=0.01,
        require_min_edge=False,
    )
    assert candidate_unfiltered is not None
    assert candidate_unfiltered.symbol == symbol

