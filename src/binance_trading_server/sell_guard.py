from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class SellGuardConfig:
    cooldown_seconds: int = 1800
    min_markup_pct: float = 0.001
    mode: str = "block"


@dataclass
class SellGuardResult:
    allowed: bool
    reason: str
    sell_floor: float
    limit_price: float
    entry_price: float
    seconds_since_buy: float


def check_sell_guard(
    *,
    entry_price: float,
    limit_price: float,
    last_buy_at: datetime | None,
    config: SellGuardConfig,
    now: datetime | None = None,
) -> SellGuardResult:
    now = now or datetime.now(timezone.utc)
    if entry_price <= 0:
        return SellGuardResult(True, "no entry price", 0.0, limit_price, entry_price, 0.0)

    seconds_since_buy = 0.0
    if last_buy_at is not None:
        seconds_since_buy = max((now - last_buy_at).total_seconds(), 0.0)

    within_cooldown = last_buy_at is not None and seconds_since_buy < config.cooldown_seconds
    if within_cooldown:
        sell_floor = entry_price * (1.0 + config.min_markup_pct)
    else:
        sell_floor = entry_price

    if limit_price + 1e-9 >= sell_floor:
        return SellGuardResult(True, "ok", sell_floor, limit_price, entry_price, seconds_since_buy)

    reason = (
        f"sell at {limit_price:.6f} below floor {sell_floor:.6f} "
        f"(entry={entry_price:.6f}, {seconds_since_buy:.0f}s since buy, "
        f"cooldown={'active' if within_cooldown else 'expired'})"
    )

    if config.mode == "alert":
        return SellGuardResult(True, f"ALERT: {reason}", sell_floor, limit_price, entry_price, seconds_since_buy)

    return SellGuardResult(False, reason, sell_floor, limit_price, entry_price, seconds_since_buy)


def sell_guard_event(result: SellGuardResult, *, symbol: str, account: str) -> dict[str, Any]:
    return {
        "account": account,
        "symbol": symbol,
        "allowed": result.allowed,
        "reason": result.reason,
        "sell_floor": result.sell_floor,
        "limit_price": result.limit_price,
        "entry_price": result.entry_price,
        "seconds_since_buy": result.seconds_since_buy,
    }
