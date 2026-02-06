from __future__ import annotations

import math

from src.binan.binance_conversion import coerce_amount, compute_spendable_quote


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

