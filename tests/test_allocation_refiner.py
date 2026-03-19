from __future__ import annotations

import pytest

try:
    from src.allocation_refiner import (
        leverage_limits_for_asset,
        refine_allocation,
        scale_allocations_to_gross_limit,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("Required module src.allocation_refiner not available", allow_module_level=True)


def test_refine_allocation_boosts_aligned_stock_long_and_caps_overnight() -> None:
    result = refine_allocation(
        asset_class="stock",
        rl_direction="long",
        rl_allocation_pct=1.0,
        rl_confidence=0.92,
        rl_logit_gap=2.1,
        current_allocation=1.2,
        current_price=100.0,
        forecast_1h={
            "predicted_close_p50": 101.8,
            "predicted_close_p10": 100.8,
            "predicted_close_p90": 102.7,
        },
        forecast_24h={
            "predicted_close_p50": 106.0,
            "predicted_close_p10": 103.5,
            "predicted_close_p90": 108.0,
        },
        previous_forecast_error=0.4,
    )

    assert result.target_allocation > 2.0
    assert result.target_allocation <= leverage_limits_for_asset("stock").long_max_leverage
    assert result.overnight_allocation <= leverage_limits_for_asset("stock").overnight_max_gross
    assert result.forecast_alignment > 0.0


def test_refine_allocation_reduces_on_forecast_disagreement_and_large_error() -> None:
    result = refine_allocation(
        asset_class="stock",
        rl_direction="long",
        rl_allocation_pct=1.0,
        rl_confidence=0.85,
        rl_logit_gap=1.4,
        current_allocation=1.0,
        current_price=100.0,
        forecast_1h={
            "predicted_close_p50": 99.1,
            "predicted_close_p10": 98.0,
            "predicted_close_p90": 100.2,
        },
        forecast_24h={
            "predicted_close_p50": 97.5,
            "predicted_close_p10": 95.0,
            "predicted_close_p90": 100.0,
        },
        previous_forecast_error=5.5,
    )

    assert result.target_allocation < 1.0
    assert result.forecast_alignment < 0.0
    assert "prev_err" in result.reason


def test_refine_allocation_blocks_crypto_short() -> None:
    result = refine_allocation(
        asset_class="crypto",
        rl_direction="short",
        rl_allocation_pct=0.8,
        rl_confidence=0.9,
        rl_logit_gap=2.0,
        current_price=100_000.0,
    )

    assert result.target_allocation == 0.0
    assert result.overnight_allocation == 0.0
    assert result.reason == "shorts_disabled_for_asset"


def test_scale_allocations_to_gross_limit_preserves_relative_signs() -> None:
    scaled = scale_allocations_to_gross_limit(
        {"AAPL": 2.4, "MSFT": -0.8, "NVDA": 1.2},
        max_gross=2.0,
    )

    assert sum(abs(value) for value in scaled.values()) == pytest.approx(2.0)
    assert scaled["AAPL"] > 0.0
    assert scaled["MSFT"] < 0.0
    assert scaled["NVDA"] > 0.0
