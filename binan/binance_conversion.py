from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

from binanceneural.execution import quantize_qty, resolve_symbol_rules
from src.binan import binance_wrapper


@dataclass(frozen=True)
class StableQuoteConversionPlan:
    symbol: str
    side: str
    from_asset: str
    to_asset: str
    amount: float
    quantity: Optional[float] = None
    quote_order_qty: Optional[float] = None


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


def build_stable_quote_conversion_plan(
    *,
    from_asset: str,
    to_asset: str,
    amount: float,
    available_pairs: Sequence[str],
) -> StableQuoteConversionPlan | None:
    """Build a direct spot-market stablecoin conversion plan.

    If `TO/FROM` exists, we buy `TO` using `FROM` via `quoteOrderQty`.
    If only `FROM/TO` exists, we sell `FROM` directly by quantity.
    """

    source = str(from_asset or "").strip().upper()
    target = str(to_asset or "").strip().upper()
    spend = coerce_amount(amount)
    if not source or not target or source == target or spend <= 0.0:
        return None

    normalized_pairs = {
        str(pair).replace("/", "").replace("-", "").replace("_", "").strip().upper()
        for pair in available_pairs
        if str(pair).strip()
    }
    buy_symbol = f"{target}{source}"
    if buy_symbol in normalized_pairs:
        return StableQuoteConversionPlan(
            symbol=buy_symbol,
            side="BUY",
            from_asset=source,
            to_asset=target,
            amount=spend,
            quote_order_qty=spend,
        )

    sell_symbol = f"{source}{target}"
    if sell_symbol in normalized_pairs:
        return StableQuoteConversionPlan(
            symbol=sell_symbol,
            side="SELL",
            from_asset=source,
            to_asset=target,
            amount=spend,
            quantity=spend,
        )
    return None


def execute_stable_quote_conversion(
    plan: StableQuoteConversionPlan,
    *,
    dry_run: bool = False,
    client=None,
) -> dict:
    """Execute a direct spot-market stablecoin conversion plan."""

    if plan.side == "BUY":
        if not plan.quote_order_qty or plan.quote_order_qty <= 0.0:
            raise ValueError(f"BUY conversion plan requires quote_order_qty, got {plan.quote_order_qty}.")
        return binance_wrapper.create_market_buy_quote(
            plan.symbol,
            quote_amount=float(plan.quote_order_qty),
            client=client,
            dry_run=dry_run,
        )

    if plan.side != "SELL":
        raise ValueError(f"Unsupported conversion side {plan.side!r}.")
    normalized_plan = _normalize_sell_conversion_plan(plan)
    if not normalized_plan.quantity or normalized_plan.quantity <= 0.0:
        raise ValueError(f"SELL conversion plan requires quantity, got {plan.quantity}.")

    resolved_client = client or binance_wrapper.get_client()
    if resolved_client is None:
        raise RuntimeError("Binance client unavailable; cannot execute quote conversion.")
    payload = {
        "symbol": str(normalized_plan.symbol).upper(),
        "side": "SELL",
        "type": "MARKET",
        "quantity": float(normalized_plan.quantity),
    }
    if dry_run and hasattr(resolved_client, "create_test_order"):
        return resolved_client.create_test_order(**payload)
    return resolved_client.create_order(**payload)


def _normalize_sell_conversion_plan(plan: StableQuoteConversionPlan) -> StableQuoteConversionPlan:
    if plan.side != "SELL":
        return plan
    raw_quantity = coerce_amount(plan.quantity)
    if raw_quantity <= 0.0:
        return plan
    rules = resolve_symbol_rules(plan.symbol)
    quantity = quantize_qty(raw_quantity, step_size=rules.step_size)
    if rules.min_qty is not None and quantity < rules.min_qty:
        raise ValueError(
            f"Stable quote conversion quantity {quantity} is below minQty {rules.min_qty} for {plan.symbol}."
        )
    if quantity <= 0.0:
        raise ValueError(f"Stable quote conversion quantity rounded to zero for {plan.symbol}.")
    return replace(plan, quantity=quantity)


__all__ = [
    "StableQuoteConversionPlan",
    "_normalize_sell_conversion_plan",
    "build_stable_quote_conversion_plan",
    "coerce_amount",
    "compute_spendable_quote",
    "execute_stable_quote_conversion",
]
