from __future__ import annotations

import math


def coerce_amount(value: object) -> float:
    """Convert arbitrary inputs into a finite float amount.

    Returns 0.0 for None, non-numeric, NaN, or infinite values.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def compute_spendable_quote(*, free_quote: float, leave_quote: float, max_spend: float | None) -> float:
    """Compute the spendable quote amount given a "leave buffer" and optional max cap."""
    free_quote = coerce_amount(free_quote)
    leave_quote = max(0.0, coerce_amount(leave_quote))
    spendable = max(0.0, free_quote - leave_quote)
    if max_spend is not None:
        cap = max(0.0, coerce_amount(max_spend))
        spendable = min(spendable, cap)
    return spendable


__all__ = ["coerce_amount", "compute_spendable_quote"]

