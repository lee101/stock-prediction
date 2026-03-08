from __future__ import annotations

import math

from src.binan.binance_conversion import (
    build_stable_quote_conversion_plan,
    coerce_amount,
    compute_spendable_quote,
)


def test_coerce_amount_handles_none_and_non_numeric() -> None:
    assert coerce_amount(None) == 0.0
    assert coerce_amount("not-a-number") == 0.0


def test_coerce_amount_handles_nan_and_inf() -> None:
    assert coerce_amount(float("nan")) == 0.0
    assert coerce_amount(float("inf")) == 0.0
    assert coerce_amount(float("-inf")) == 0.0

    assert coerce_amount("nan") == 0.0
    assert coerce_amount("inf") == 0.0


def test_coerce_amount_keeps_finite_values() -> None:
    assert coerce_amount(1) == 1.0
    assert coerce_amount(1.25) == 1.25
    assert coerce_amount("-2.5") == -2.5
    assert math.isfinite(coerce_amount("-2.5"))


def test_compute_spendable_quote_applies_leave_buffer() -> None:
    assert compute_spendable_quote(free_quote=100.0, leave_quote=10.0, max_spend=None) == 90.0
    assert compute_spendable_quote(free_quote=100.0, leave_quote=110.0, max_spend=None) == 0.0


def test_compute_spendable_quote_clamps_negative_inputs() -> None:
    # Negative leave buffers should not increase spendable beyond free_quote.
    assert compute_spendable_quote(free_quote=100.0, leave_quote=-5.0, max_spend=None) == 100.0
    # Negative max_spend effectively caps spendable at 0.
    assert compute_spendable_quote(free_quote=100.0, leave_quote=10.0, max_spend=-1.0) == 0.0


def test_compute_spendable_quote_applies_max_cap() -> None:
    assert compute_spendable_quote(free_quote=100.0, leave_quote=10.0, max_spend=50.0) == 50.0


def test_build_stable_quote_conversion_plan_prefers_buying_target_symbol() -> None:
    plan = build_stable_quote_conversion_plan(
        from_asset="USDT",
        to_asset="FDUSD",
        amount=50.0,
        available_pairs=["FDUSDUSDT"],
    )
    assert plan is not None
    assert plan.symbol == "FDUSDUSDT"
    assert plan.side == "BUY"
    assert plan.quote_order_qty == 50.0
    assert plan.quantity is None


def test_build_stable_quote_conversion_plan_falls_back_to_selling_source_symbol() -> None:
    plan = build_stable_quote_conversion_plan(
        from_asset="FDUSD",
        to_asset="USDT",
        amount=75.0,
        available_pairs=["FDUSDUSDT"],
    )
    assert plan is not None
    assert plan.symbol == "FDUSDUSDT"
    assert plan.side == "SELL"
    assert plan.quantity == 75.0
    assert plan.quote_order_qty is None


def test_build_stable_quote_conversion_plan_returns_none_without_direct_pair() -> None:
    plan = build_stable_quote_conversion_plan(
        from_asset="USDT",
        to_asset="FDUSD",
        amount=10.0,
        available_pairs=["BTCUSDT"],
    )
    assert plan is None
