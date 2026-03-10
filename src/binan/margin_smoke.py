from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math


def _coerce_nonnegative(value: float | int | str | None) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric) or numeric <= 0.0:
        return 0.0
    return numeric


def _quantize_up(value: float, step_size: float) -> float:
    numeric = _coerce_nonnegative(value)
    step = _coerce_nonnegative(step_size)
    if numeric <= 0.0:
        return 0.0
    if step <= 0.0:
        return numeric
    ratio = (Decimal(str(numeric)) / Decimal(str(step))).to_integral_value(rounding=ROUND_CEILING)
    return float(ratio * Decimal(str(step)))


def _quantize_down(value: float, step_size: float) -> float:
    numeric = _coerce_nonnegative(value)
    step = _coerce_nonnegative(step_size)
    if numeric <= 0.0:
        return 0.0
    if step <= 0.0:
        return numeric
    ratio = (Decimal(str(numeric)) / Decimal(str(step))).to_integral_value(rounding=ROUND_FLOOR)
    return float(ratio * Decimal(str(step)))


@dataclass(frozen=True)
class MarginAssetSnapshot:
    free: float
    borrowed: float
    interest: float
    net_asset: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "free", _coerce_nonnegative(self.free))
        object.__setattr__(self, "borrowed", _coerce_nonnegative(self.borrowed))
        object.__setattr__(self, "interest", _coerce_nonnegative(self.interest))
        try:
            net_asset = float(self.net_asset)
        except (TypeError, ValueError):
            net_asset = 0.0
        if not math.isfinite(net_asset):
            net_asset = 0.0
        object.__setattr__(self, "net_asset", net_asset)

    @property
    def liability(self) -> float:
        return self.borrowed + self.interest

    @property
    def deficit(self) -> float:
        return max(0.0, self.liability - self.free)

    @property
    def repayable(self) -> float:
        return max(0.0, min(self.free, self.liability))

    def as_dict(self) -> dict[str, float]:
        return {
            "free": self.free,
            "borrowed": self.borrowed,
            "interest": self.interest,
            "net_asset": self.net_asset,
            "liability": self.liability,
            "deficit": self.deficit,
            "repayable": self.repayable,
        }


def build_market_qty_from_notional(
    *,
    target_notional: float,
    market_price: float,
    step_size: float,
    min_qty: float = 0.0,
    min_notional: float = 0.0,
) -> float:
    price = _coerce_nonnegative(market_price)
    if price <= 0.0:
        return 0.0
    effective_notional = max(_coerce_nonnegative(target_notional), _coerce_nonnegative(min_notional))
    if effective_notional <= 0.0:
        return 0.0
    qty = _quantize_up(effective_notional / price, step_size)
    min_qty_value = _coerce_nonnegative(min_qty)
    if min_qty_value > 0.0 and qty < min_qty_value:
        qty = _quantize_up(min_qty_value, step_size)
    if qty <= 0.0:
        return 0.0
    if _coerce_nonnegative(min_notional) > 0.0 and qty * price < _coerce_nonnegative(min_notional):
        qty = _quantize_up(_coerce_nonnegative(min_notional) / price, step_size)
    return qty


def build_liability_cleanup_qty(
    *,
    snapshot: MarginAssetSnapshot,
    market_price: float,
    step_size: float,
    min_qty: float = 0.0,
    min_notional: float = 0.0,
) -> float:
    if snapshot.deficit <= 0.0:
        return 0.0
    qty = _quantize_up(snapshot.deficit, step_size)
    min_qty_value = _coerce_nonnegative(min_qty)
    if min_qty_value > 0.0 and qty < min_qty_value:
        qty = _quantize_up(min_qty_value, step_size)
    price = _coerce_nonnegative(market_price)
    min_notional_value = _coerce_nonnegative(min_notional)
    if price > 0.0 and min_notional_value > 0.0 and qty * price < min_notional_value:
        qty = max(
            qty,
            build_market_qty_from_notional(
                target_notional=min_notional_value,
                market_price=price,
                step_size=step_size,
                min_qty=min_qty,
                min_notional=min_notional,
            ),
        )
    return qty


def build_excess_flatten_qty(
    *,
    snapshot: MarginAssetSnapshot,
    market_price: float,
    step_size: float,
    min_qty: float = 0.0,
    min_notional: float = 0.0,
) -> float:
    if snapshot.liability > 0.000001 or snapshot.net_asset <= 0.0:
        return 0.0
    qty = _quantize_down(snapshot.net_asset, step_size)
    min_qty_value = _coerce_nonnegative(min_qty)
    if min_qty_value > 0.0 and qty < min_qty_value:
        return 0.0
    price = _coerce_nonnegative(market_price)
    min_notional_value = _coerce_nonnegative(min_notional)
    if price > 0.0 and min_notional_value > 0.0 and qty * price < min_notional_value:
        return 0.0
    return qty


__all__ = [
    "MarginAssetSnapshot",
    "build_excess_flatten_qty",
    "build_liability_cleanup_qty",
    "build_market_qty_from_notional",
]
