from __future__ import annotations

import pandas as pd
import pytest

try:
    from llm_hourly_trader.gemini_wrapper import TradePlan
    from src.daily_mixed_hybrid import (
        CandidatePlan,
        DailyPosition,
        build_candidate_plan,
        build_daily_hybrid_prompt,
        current_allocation_from_position,
        previous_forecast_error_pct,
        select_best_candidate,
    )
    from unified_orchestrator.rl_gemini_bridge import RLSignal
except (ImportError, ModuleNotFoundError):
    pytest.skip("Required modules for daily_mixed_hybrid not available", allow_module_level=True)


def test_build_daily_hybrid_prompt_includes_previous_state_and_forecasts():
    prompt = build_daily_hybrid_prompt(
        symbol="NVDA",
        asof=pd.Timestamp("2026-03-15T00:00:00Z"),
        asset_class="stock",
        current_price=105.0,
        rl_signal=RLSignal(
            symbol_idx=0,
            symbol_name="NVDA",
            direction="long",
            confidence=0.82,
            logit_gap=2.4,
            allocation_pct=1.0,
        ),
        other_rl_signals=[
            RLSignal(
                symbol_idx=1,
                symbol_name="MSFT",
                direction="short",
                confidence=0.61,
                logit_gap=1.1,
                allocation_pct=0.5,
            )
        ],
        history_rows=[
            {
                "timestamp": "2026-03-13T00:00:00Z",
                "open": 100.0,
                "high": 106.0,
                "low": 99.0,
                "close": 104.0,
                "volume": 1_000_000.0,
            },
            {
                "timestamp": "2026-03-14T00:00:00Z",
                "open": 104.0,
                "high": 107.0,
                "low": 103.0,
                "close": 105.0,
                "volume": 1_200_000.0,
            },
        ],
        forecast_1h={
            "predicted_close_p50": 105.8,
            "predicted_close_p10": 104.9,
            "predicted_close_p90": 106.7,
            "predicted_high_p50": 106.4,
            "predicted_low_p50": 104.8,
        },
        forecast_24h={
            "predicted_close_p50": 108.2,
            "predicted_close_p10": 104.5,
            "predicted_close_p90": 110.4,
            "predicted_high_p50": 109.3,
            "predicted_low_p50": 103.7,
        },
        current_position=DailyPosition(
            symbol="NVDA",
            direction="long",
            qty=5.0,
            entry_price=100.0,
            entry_timestamp="2026-03-10T00:00:00+00:00",
            current_price=105.0,
            unrealized_pnl_pct=5.0,
            hold_days=5,
        ),
        portfolio_positions=[
            DailyPosition(
                symbol="NVDA",
                direction="long",
                qty=5.0,
                entry_price=100.0,
                entry_timestamp="2026-03-10T00:00:00+00:00",
                current_price=105.0,
                unrealized_pnl_pct=5.0,
                hold_days=5,
            ),
            DailyPosition(
                symbol="MSFT",
                direction="short",
                qty=2.0,
                entry_price=410.0,
                entry_timestamp="2026-03-12T00:00:00+00:00",
                current_price=402.0,
                unrealized_pnl_pct=1.95,
                hold_days=3,
            ),
        ],
        previous_plan={
            "timestamp": "2026-03-14T00:00:00+00:00",
            "direction": "long",
            "buy_price": 103.25,
            "sell_price": 108.90,
            "confidence": 0.76,
            "reasoning": "Previous plan context",
        },
        recent_trades=[
            {
                "timestamp": "2026-03-11T00:00:00+00:00",
                "symbol": "NVDA",
                "side": "open_long",
                "price": 100.0,
                "qty": 5.0,
                "pnl_pct": 0.0,
                "reason": "Opened prior position",
            }
        ],
        previous_forecast_error=1.25,
        cash=4_250.0,
        equity=4_775.0,
        tracked_symbols=23,
        max_positions=3,
        allowed_directions=["long", "short", "hold"],
        current_allocation=1.10,
        refined_allocation=2.75,
        overnight_allocation=2.0,
        allocation_reason="RL + Chronos2 alignment supports adding risk but keep overnight cap.",
    )

    assert "ANALYSIS TIME: 2026-03-15T00:00:00+00:00" in prompt
    assert "Previous plan time=2026-03-14T00:00:00+00:00" in prompt
    assert "CURRENT POSITION: LONG NVDA" in prompt
    assert "Open positions=2 / max 3" in prompt
    assert "- SHORT MSFT" in prompt
    assert "Current signed allocation: +1.10x" in prompt
    assert "Refined target allocation: +2.75x" in prompt
    assert "Overnight capped allocation: +2.00x" in prompt
    assert "Previous Chronos2 forecast error vs realized move: +1.25%" in prompt
    assert "MSFT: SHORT" in prompt
    assert "Opened prior position" in prompt
    assert "24h: close=108.20" in prompt


def test_previous_forecast_error_pct_uses_reference_and_predicted_close():
    state = {
        "forecast_refs": {
            "BTCUSD": {
                "reference_price": 100.0,
                "predicted_close_p50": 110.0,
                "timestamp": "2026-03-14T00:00:00+00:00",
            }
        }
    }

    error_pct = previous_forecast_error_pct(state, "BTCUSD", current_price=105.0)

    assert error_pct == pytest.approx(-5.0)


def test_select_best_candidate_prefers_positive_score():
    long_signal = RLSignal(
        symbol_idx=0,
        symbol_name="AAPL",
        direction="long",
        confidence=0.72,
        logit_gap=1.8,
        allocation_pct=1.0,
    )
    short_signal = RLSignal(
        symbol_idx=1,
        symbol_name="BTCUSD",
        direction="short",
        confidence=0.81,
        logit_gap=2.1,
        allocation_pct=1.0,
    )
    best = CandidatePlan(
        symbol="AAPL",
        asset_class="stock",
        current_price=200.0,
        rl_signal=long_signal,
        plan=TradePlan(direction="long", buy_price=198.0, sell_price=206.0, confidence=0.8, reasoning="long"),
        score=0.031,
        expected_edge_pct=0.04,
    )
    blocked = CandidatePlan(
        symbol="BTCUSD",
        asset_class="crypto",
        current_price=90_000.0,
        rl_signal=short_signal,
        plan=TradePlan(direction="hold", buy_price=0.0, sell_price=0.0, confidence=0.3, reasoning="blocked"),
        score=-1.0,
        expected_edge_pct=0.0,
    )

    selected = select_best_candidate([blocked, best])

    assert selected == best


def test_current_allocation_from_position_computes_signed_leverage() -> None:
    position = DailyPosition(
        symbol="NVDA",
        direction="short",
        qty=3.0,
        entry_price=120.0,
        entry_timestamp="2026-03-10T00:00:00+00:00",
        current_price=125.0,
        unrealized_pnl_pct=-4.0,
        hold_days=2,
    )

    allocation = current_allocation_from_position(position, equity=500.0)

    assert allocation == pytest.approx(-0.75)


def test_build_candidate_plan_attaches_refined_allocation_context() -> None:
    signal = RLSignal(
        symbol_idx=0,
        symbol_name="AAPL",
        direction="long",
        confidence=0.84,
        logit_gap=2.2,
        allocation_pct=1.0,
    )
    position = DailyPosition(
        symbol="AAPL",
        direction="long",
        qty=2.0,
        entry_price=100.0,
        entry_timestamp="2026-03-10T00:00:00+00:00",
        current_price=110.0,
        unrealized_pnl_pct=10.0,
        hold_days=3,
    )
    plan = TradePlan(
        direction="long",
        buy_price=109.0,
        sell_price=116.0,
        confidence=0.8,
        reasoning="aligned",
    )

    candidate = build_candidate_plan(
        symbol="AAPL",
        asset_class="stock",
        current_price=110.0,
        rl_signal=signal,
        plan=plan,
        current_position=position,
        equity=1_000.0,
        forecast_1h={
            "predicted_close_p50": 111.5,
            "predicted_close_p10": 110.2,
            "predicted_close_p90": 112.3,
        },
        forecast_24h={
            "predicted_close_p50": 116.0,
            "predicted_close_p10": 113.0,
            "predicted_close_p90": 118.5,
        },
        previous_forecast_error=0.3,
    )

    assert candidate.current_allocation == pytest.approx(0.22)
    assert candidate.target_allocation > 0.22
    assert candidate.overnight_allocation <= 2.0
    assert "rl=long" in candidate.allocation_reason
