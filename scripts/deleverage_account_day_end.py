"""
Progressively deleverage the live portfolio as the session approaches the close.

The script monitors gross exposure against equity and, whenever the account
is above the maximum intraday leverage (default 1.94x), submits partial exit
orders weighted by position size. Orders start as near-market limits and
become increasingly aggressive as the close approaches. During the final
five minutes, any remaining excess leverage is flattened with market orders
to respect the PDT hard cap.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence

from loguru import logger

# Ensure repository root is importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import alpaca_wrapper
from src.fixtures import crypto_symbols
from src.trading_obj_utils import filter_to_realistic_positions

# --- Configuration -----------------------------------------------------------------

MAX_GROSS_LEVERAGE = float(os.getenv("EOD_MAX_GROSS_LEVERAGE", "1.94"))
RAMP_WINDOW_MINUTES = int(os.getenv("EOD_DELEVERAGE_WINDOW_MINUTES", "60"))
FORCE_MARKET_MINUTES = int(os.getenv("EOD_FORCE_MARKET_MINUTES", "5"))
ACTIVE_SLEEP_SECONDS = int(os.getenv("EOD_ACTIVE_SLEEP_SECONDS", "60"))
IDLE_SLEEP_SECONDS = int(os.getenv("EOD_IDLE_SLEEP_SECONDS", "300"))
MIN_NOTIONAL_PER_SLICE = float(os.getenv("EOD_MIN_NOTIONAL_PER_SLICE", "100"))
MIN_ORDER_QTY = float(os.getenv("EOD_MIN_ORDER_QTY", "0.0001"))
PROGRESS_POWER = float(os.getenv("EOD_PROGRESS_POWER", "1.0"))
LIMIT_OFFSET_START = float(os.getenv("EOD_LIMIT_OFFSET_START", "0.003"))  # 0.30%
LIMIT_OFFSET_END = float(os.getenv("EOD_LIMIT_OFFSET_END", "0.02"))       # 2.00%


# --- Helpers -----------------------------------------------------------------------


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _minutes_to_close() -> float | None:
    """
    Return minutes until the next market close. Falls back to ``None`` when the
    clock cannot be retrieved.
    """
    try:
        # Bypass the cached wrapper for fresher reads inside the closing window.
        clock = alpaca_wrapper.get_clock_internal()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Unable to fetch Alpaca clock: {exc}")
        return None

    close_dt = getattr(clock, "next_close", None)
    if not close_dt:
        return None

    now = _now_utc()
    if getattr(close_dt, "tzinfo", None) is None:
        close_dt = close_dt.replace(tzinfo=timezone.utc)
    else:
        close_dt = close_dt.astimezone(timezone.utc)

    delta_seconds = (close_dt - now).total_seconds()
    return max(delta_seconds / 60.0, 0.0)


def _normalize_qty(symbol: str, qty: float) -> float:
    """
    Round quantities to valid increments for the given instrument.
    """
    if qty <= 0:
        return 0.0

    if symbol in crypto_symbols or symbol.endswith("USD"):
        # Allow fractional crypto sizes with a sensible precision.
        return round(qty, 6)

    # Equity legs must be whole shares.
    return float(math.floor(qty))


@dataclass
class ReductionOrder:
    symbol: str
    side: str
    qty: float
    notional: float
    use_market: bool
    limit_offset: float

    def to_position_slice(self) -> SimpleNamespace:
        return SimpleNamespace(symbol=self.symbol, side=self.side, qty=str(self.qty))


def _is_crypto_symbol(symbol: str) -> bool:
    upper = symbol.upper()
    return upper in crypto_symbols or upper.endswith("USD")


def _filter_equity_positions(positions: Sequence) -> List:
    """Return only equity positions (skip crypto exposure)."""
    equities: List = []
    for position in positions:
        symbol = getattr(position, "symbol", "")
        if not symbol:
            continue
        if _is_crypto_symbol(symbol):
            continue
        equities.append(position)
    return equities


def _gross_exposure(positions: Sequence) -> float:
    exposure = 0.0
    for pos in positions:
        exposure += abs(_safe_float(getattr(pos, "market_value", 0.0)))
    return exposure


def _per_unit_value(position) -> float:
    qty = abs(_safe_float(getattr(position, "qty", 0.0)))
    if qty <= 0:
        return 0.0
    value = abs(_safe_float(getattr(position, "market_value", 0.0)))
    if value <= 0:
        return 0.0
    return value / qty


def _compute_limit_offset(progress: float, side: str) -> float:
    """
    Interpolate between a gentle and aggressive offset as we approach the close.
    Positive offsets move prices away from the market, negative offsets cross it.
    """
    bounded_progress = min(max(progress, 0.0), 1.0)
    start = LIMIT_OFFSET_START if side == "long" else -LIMIT_OFFSET_START
    end = -LIMIT_OFFSET_END if side == "long" else LIMIT_OFFSET_END
    return start + (end - start) * bounded_progress


def _build_reduction_plan(
    positions: Sequence,
    target_notional: float,
    use_market: bool,
    progress: float,
) -> List[ReductionOrder]:
    total_exposure = _gross_exposure(positions)
    if total_exposure <= 0 or target_notional >= total_exposure:
        return []

    target_notional = max(0.0, min(total_exposure, target_notional))
    scale = target_notional / total_exposure if total_exposure > 0 else 0.0
    remaining_notional = total_exposure - target_notional

    orders: List[ReductionOrder] = []
    equity_positions = _filter_equity_positions(positions)
    sorted_positions = sorted(
        equity_positions,
        key=lambda p: abs(_safe_float(getattr(p, "market_value", 0.0))),
        reverse=True,
    )

    if not sorted_positions:
        return []

    for position in sorted_positions:
        symbol = getattr(position, "symbol", "").upper()
        side = getattr(position, "side", "").lower()
        qty_total = abs(_safe_float(getattr(position, "qty", 0.0)))
        value_total = abs(_safe_float(getattr(position, "market_value", 0.0)))
        if qty_total <= 0 or value_total <= 0:
            continue

        desired_value = value_total * scale
        reduce_value = max(0.0, value_total - desired_value)
        if reduce_value <= 0:
            continue
        reduce_value = min(reduce_value, remaining_notional)

        unit_value = value_total / qty_total
        reduce_qty = reduce_value / unit_value if unit_value > 0 else 0.0
        reduce_qty = _normalize_qty(symbol, reduce_qty)
        if reduce_qty <= 0:
            continue

        actual_notional = reduce_qty * unit_value
        if actual_notional < MIN_NOTIONAL_PER_SLICE and not use_market:
            continue

        if reduce_qty > qty_total:
            reduce_qty = _normalize_qty(symbol, qty_total)
            actual_notional = reduce_qty * unit_value
            if reduce_qty <= 0:
                continue
            if actual_notional < MIN_NOTIONAL_PER_SLICE and not use_market:
                continue

        limit_offset = _compute_limit_offset(progress, side)
        if reduce_qty < MIN_ORDER_QTY and not use_market:
            continue

        orders.append(
            ReductionOrder(
                symbol=symbol,
                side=side,
                qty=reduce_qty,
                notional=actual_notional,
                use_market=use_market,
                limit_offset=limit_offset,
            )
        )
        remaining_notional -= actual_notional

        if remaining_notional <= MIN_NOTIONAL_PER_SLICE:
            break

    return orders


def _apply_orders(orders: Iterable[ReductionOrder]) -> None:
    for order in orders:
        position_slice = order.to_position_slice()
        try:
            if order.use_market:
                logger.info(
                    "Submitting market deleverage order: %s %s qty=%.6f ($%.2f)",
                    order.symbol,
                    order.side,
                    order.qty,
                    order.notional,
                )
                alpaca_wrapper.close_position_violently(position_slice)
            else:
                logger.info(
                    "Submitting near-market deleverage order: %s %s qty=%.6f offset=%.3f ($%.2f)",
                    order.symbol,
                    order.side,
                    order.qty,
                    order.limit_offset,
                    order.notional,
                )
                alpaca_wrapper.close_position_near_market(
                    position_slice, pct_above_market=order.limit_offset
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to place deleverage order for {order.symbol}: {exc}")


def _log_state(equity: float, exposure: float, leverage: float, minutes_to_close: float | None) -> None:
    mtc_repr = "unknown" if minutes_to_close is None else f"{minutes_to_close:.1f}m"
    logger.info(
        "Equity=$%.2f Exposure=$%.2f Leverage=%.3fx MinutesToClose=%s",
        equity,
        exposure,
        leverage,
        mtc_repr,
    )


def _idle_sleep(seconds: int) -> None:
    try:
        time.sleep(max(seconds, 1))
    except Exception:
        # Sleep interruptions should not crash the process.
        pass


def run() -> None:
    """
    Main loop. Continues running until interrupted.
    """
    logger.info(
        "Starting day-end deleverage monitor (max leverage %.2fx, ramp window %d min, market sweep %d min).",
        MAX_GROSS_LEVERAGE,
        RAMP_WINDOW_MINUTES,
        FORCE_MARKET_MINUTES,
    )

    while True:
        try:
            account = alpaca_wrapper.get_account()
            equity = _safe_float(getattr(account, "equity", 0.0))
        except Exception as exc:
            logger.error(f"Failed to load account: {exc}")
            _idle_sleep(IDLE_SLEEP_SECONDS)
            continue

        if equity <= 0:
            logger.warning("Equity is non-positive (%.2f). Targeting flat book.", equity)

        try:
            positions_raw = alpaca_wrapper.get_all_positions()
        except Exception as exc:
            logger.error(f"Failed to load positions: {exc}")
            _idle_sleep(IDLE_SLEEP_SECONDS)
            continue

        positions = _filter_equity_positions(filter_to_realistic_positions(positions_raw))
        if not positions:
            logger.info("No positions to manage; sleeping.")
            _idle_sleep(IDLE_SLEEP_SECONDS)
            continue

        exposure = _gross_exposure(positions)
        leverage = float("inf") if equity <= 0 else exposure / max(equity, 1e-6)
        minutes_to_close = _minutes_to_close()
        _log_state(equity, exposure, leverage, minutes_to_close)

        target_notional = MAX_GROSS_LEVERAGE * equity if equity > 0 else 0.0

        if exposure <= target_notional:
            _idle_sleep(IDLE_SLEEP_SECONDS if (minutes_to_close or float("inf")) > RAMP_WINDOW_MINUTES else ACTIVE_SLEEP_SECONDS)
            continue

        if minutes_to_close is None:
            logger.warning("Cannot determine minutes to close; holding current reductions.")
            _idle_sleep(ACTIVE_SLEEP_SECONDS)
            continue

        if minutes_to_close > RAMP_WINDOW_MINUTES:
            logger.info("Outside deleverage window (%.1f min to close); sleeping.", minutes_to_close)
            _idle_sleep(IDLE_SLEEP_SECONDS)
            continue

        if minutes_to_close <= FORCE_MARKET_MINUTES:
            use_market = True
            progress = 1.0
        else:
            use_market = False
            span = max(RAMP_WINDOW_MINUTES - FORCE_MARKET_MINUTES, 1)
            elapsed = RAMP_WINDOW_MINUTES - minutes_to_close
            raw_progress = min(max(elapsed / span, 0.0), 1.0)
            progress = raw_progress ** PROGRESS_POWER

        target_progressive = max(
            target_notional,
            target_notional + (exposure - target_notional) * (1.0 - progress),
        )

        orders = _build_reduction_plan(
            positions=positions,
            target_notional=target_progressive,
            use_market=use_market,
            progress=progress,
        )

        if not orders:
            logger.info("No actionable deleverage orders generated; sleeping.")
            _idle_sleep(ACTIVE_SLEEP_SECONDS)
            continue

        _apply_orders(orders)
        _idle_sleep(ACTIVE_SLEEP_SECONDS)


if __name__ == "__main__":
    run()
