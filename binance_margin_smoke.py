from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _step_decimals(step: float) -> int:
    text = f"{step:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def quantize_qty_up(quantity: float, *, step_size: float | None) -> float:
    if step_size is None or step_size <= 0:
        return max(0.0, float(quantity))
    if quantity <= 0:
        return 0.0
    decimals = _step_decimals(step_size)
    steps = math.ceil(quantity / step_size)
    return round(steps * step_size, decimals)


@dataclass(frozen=True)
class AssetBalanceSnapshot:
    asset: str
    free: float
    borrowed: float
    interest: float
    net_asset: float

    @classmethod
    def from_margin_entry(cls, asset: str, entry: dict[str, Any] | None) -> "AssetBalanceSnapshot":
        payload = entry or {}
        return cls(
            asset=str(asset).upper(),
            free=_safe_float(payload.get("free")),
            borrowed=_safe_float(payload.get("borrowed")),
            interest=_safe_float(payload.get("interest")),
            net_asset=_safe_float(payload.get("netAsset")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShortSmokePlan:
    symbol: str
    base_asset: str
    reference_price: float
    target_notional: float
    effective_notional: float
    qty: float
    projected_notional: float
    min_notional: float | None
    min_qty: float | None
    step_size: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_short_smoke_plan(
    *,
    symbol: str,
    base_asset: str,
    reference_price: float,
    target_notional: float,
    min_notional: float | None,
    min_qty: float | None,
    step_size: float | None,
) -> ShortSmokePlan:
    price = _safe_float(reference_price)
    if price <= 0:
        raise ValueError(f"reference_price must be positive, got {reference_price!r}")

    requested_notional = _safe_float(target_notional)
    if requested_notional <= 0:
        raise ValueError(f"target_notional must be positive, got {target_notional!r}")

    effective_notional = max(requested_notional, _safe_float(min_notional))
    qty = quantize_qty_up(effective_notional / price, step_size=step_size)
    if min_qty and qty < min_qty:
        qty = quantize_qty_up(min_qty, step_size=step_size)

    projected_notional = qty * price
    if min_notional and projected_notional < min_notional:
        qty = quantize_qty_up(min_notional / price, step_size=step_size)
        projected_notional = qty * price

    if qty <= 0 or projected_notional <= 0:
        raise ValueError(f"resolved smoke quantity must be positive, got qty={qty} price={price}")

    return ShortSmokePlan(
        symbol=str(symbol).upper(),
        base_asset=str(base_asset).upper(),
        reference_price=price,
        target_notional=requested_notional,
        effective_notional=effective_notional,
        qty=qty,
        projected_notional=projected_notional,
        min_notional=min_notional,
        min_qty=min_qty,
        step_size=step_size,
    )


def residual_repay_qty(
    snapshot: AssetBalanceSnapshot,
    *,
    step_size: float | None,
    min_qty: float | None,
    safety_steps: int = 1,
) -> float:
    if safety_steps < 0:
        raise ValueError(f"safety_steps must be >= 0, got {safety_steps}")
    base_target = max(snapshot.borrowed + snapshot.interest, -snapshot.net_asset, 0.0)
    if base_target <= 0:
        return 0.0
    bump = (step_size or 0.0) * safety_steps
    qty = quantize_qty_up(base_target + bump, step_size=step_size)
    if min_qty and qty < min_qty:
        qty = quantize_qty_up(min_qty, step_size=step_size)
    return qty


def balance_is_flat(
    snapshot: AssetBalanceSnapshot,
    *,
    borrowed_tolerance: float = 1e-9,
    net_tolerance: float | None = None,
) -> bool:
    net_tol = borrowed_tolerance if net_tolerance is None else max(0.0, float(net_tolerance))
    return snapshot.borrowed <= max(0.0, float(borrowed_tolerance)) and abs(snapshot.net_asset) <= net_tol
