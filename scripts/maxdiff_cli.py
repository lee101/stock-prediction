import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import typer

import alpaca_wrapper
from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from scripts.alpaca_cli import get_strategy_for_symbol, set_strategy_for_symbol
from src.logging_utils import setup_logging
from src.stock_utils import pairs_equal
from src.trading_obj_utils import filter_to_realistic_positions

logger = setup_logging("maxdiff_cli.log")

try:
    from alpaca.data import StockHistoricalDataClient
except Exception as exc:  # pragma: no cover - fallback in simulator environments
    StockHistoricalDataClient = None  # type: ignore[assignment]
    logger.warning(f"StockHistoricalDataClient unavailable: {exc}")


app = typer.Typer(help="Maxdiff strategy helpers for staged entry/exit automation.")
STATUS_VERSION = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_side(side: str) -> str:
    return "buy" if str(side).lower().startswith("b") else "sell"


def _normalize_config_path(config_path: Optional[Path]) -> Optional[Path]:
    if config_path is None:
        return None
    return config_path.expanduser().resolve()


def _load_status(config_path: Optional[Path]) -> dict:
    if config_path is None:
        return {}
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to read watcher status %s: %s", config_path, exc)
        return {}


def _write_status_file(config_path: Optional[Path], status: dict) -> None:
    if config_path is None:
        return
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = config_path.with_suffix(config_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)
        temp_path.replace(config_path)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to persist watcher status %s: %s", config_path, exc)


def _prepare_status(config_path: Optional[Path], defaults: dict) -> dict:
    status = _load_status(config_path)
    status.update(defaults)
    status.setdefault("config_version", STATUS_VERSION)
    status["pid"] = os.getpid()
    if config_path is not None:
        status["config_path"] = str(config_path)
    return status


def _update_status(config_path: Optional[Path], status: dict, **changes) -> dict:
    status.update(changes)
    status["last_update"] = _now().isoformat()
    _write_status_file(config_path, status)
    return status


def _entry_requires_cash(side: str, price: float, qty: float) -> bool:
    if qty <= 0 or price <= 0:
        return False
    notional = abs(price * qty)
    cash = float(getattr(alpaca_wrapper, "cash", 0.0) or 0.0)
    if side == "buy":
        if notional > cash:
            logger.info(
                "Skipping order to avoid leverage: notional=%.2f, cash=%.2f", notional, cash
            )
            return False
        return True
    # For shorts, require enough cash buffer to avoid immediate leverage swings.
    total_bp = float(getattr(alpaca_wrapper, "total_buying_power", 0.0) or 0.0)
    equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)
    if total_bp > equity * 1.05:
        logger.info(
            "Skipping short to avoid leverage: total_bp=%.2f exceeds equity=%.2f",
            total_bp,
            equity,
        )
        return False
    return True


def _position_for_symbol(symbol: str, entry_side: str) -> Optional[object]:
    positions = filter_to_realistic_positions(alpaca_wrapper.get_all_positions())
    for pos in positions:
        if not hasattr(pos, "symbol") or not pairs_equal(pos.symbol, symbol):
            continue
        side = getattr(pos, "side", "").lower()
        if entry_side == "buy" and side == "long":
            return pos
        if entry_side == "sell" and side == "short":
            return pos
    return None


def _orders_for_symbol(symbol: str, side: Optional[str] = None) -> Iterable[object]:
    desired_side = None if side is None else side.lower()
    orders = alpaca_wrapper.get_open_orders()
    for order in orders:
        if not hasattr(order, "symbol") or not pairs_equal(order.symbol, symbol):
            continue
        if desired_side and getattr(order, "side", "").lower() != desired_side:
            continue
        yield order


def _cancel_orders(symbol: str, side: Optional[str] = None) -> None:
    for order in list(_orders_for_symbol(symbol, side=side)):
        try:
            alpaca_wrapper.cancel_order(order)
            time.sleep(0.25)
        except Exception as exc:
            logger.warning("Failed cancelling %s order for %s: %s", side or "any", symbol, exc)


def _latest_reference_price(symbol: str, side: str, fallback_client=None) -> Optional[float]:
    try:
        quote = alpaca_wrapper.latest_data(symbol)
        if side == "buy":
            price = getattr(quote, "ask_price", None)
            if price in (None, 0):
                price = getattr(quote, "bid_price", None)
        else:
            price = getattr(quote, "bid_price", None)
            if price in (None, 0):
                price = getattr(quote, "ask_price", None)
        if price and price > 0:
            return float(price)
    except Exception as exc:
        logger.debug("latest_data unavailable for %s: %s", symbol, exc)

    if fallback_client is None:
        return None
    try:
        download_exchange_latest_data(fallback_client, symbol)
        if side == "buy":
            price = get_ask(symbol)
        else:
            price = get_bid(symbol)
        return float(price) if price else None
    except Exception as exc:
        logger.warning("Fallback price fetch failed for %s: %s", symbol, exc)
        return None


def _within_tolerance(reference_price: float, limit_price: float, tolerance_pct: float) -> bool:
    if reference_price <= 0 or limit_price <= 0:
        return False
    diff = abs(reference_price - limit_price) / limit_price
    return diff <= tolerance_pct


def _has_takeprofit_order(symbol: str, exit_side: str, target_price: float, tolerance: float) -> bool:
    for order in _orders_for_symbol(symbol, side=exit_side):
        limit_price = getattr(order, "limit_price", None)
        if limit_price is None:
            continue
        try:
            limit_value = float(limit_price)
        except (TypeError, ValueError):
            continue
        if math.isclose(limit_value, target_price, rel_tol=tolerance, abs_tol=target_price * tolerance):
            return True
    return False


def _ensure_strategy_tag(symbol: str) -> None:
    existing = get_strategy_for_symbol(symbol)
    if existing != "highlow":
        set_strategy_for_symbol(symbol, "highlow")


@app.command("open-position")
def open_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str = typer.Option(..., "--side", help="Entry side for the strategy (buy/sell)."),
    limit_price: float = typer.Option(..., "--limit-price", help="Limit price to stage entry."),
    target_qty: float = typer.Option(..., "--target-qty", help="Quantity to accumulate."),
    tolerance_pct: float = typer.Option(
        0.0066,
        "--tolerance-pct",
        help="Relative tolerance (e.g. 0.0066 == 0.66%%) before staging the entry order.",
    ),
    expiry_minutes: int = typer.Option(
        24 * 60,
        "--expiry-minutes",
        help="Maximum time to keep watching/maintaining the entry order.",
    ),
    asset_class: str = typer.Option(
        "equity",
        "--asset-class",
        help="Asset class hint (affects clean-up semantics).",
    ),
    poll_seconds: int = typer.Option(
        45,
        "--poll-seconds",
        help="Polling cadence while monitoring price thresholds.",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config-path",
        help="Path to persist watcher status updates.",
    ),
    force_immediate: bool = typer.Option(
        False,
        "--force-immediate",
        help="Submit the staged order without waiting for price tolerance triggers.",
    ),
    priority_rank: Optional[int] = typer.Option(
        None,
        "--priority-rank",
        help="Optional priority ranking used for coordination across watchers.",
    ),
) -> None:
    side = _normalize_side(side)
    if limit_price <= 0 or target_qty <= 0:
        logger.error(
            "Invalid maxdiff open parameters for %s: limit_price=%.4f target_qty=%.4f",
            symbol,
            limit_price,
            target_qty,
        )
        return

    _ensure_strategy_tag(symbol)
    config_path = _normalize_config_path(config_path)
    expiry_minutes = max(int(expiry_minutes), 1)
    now = _now()
    expiry = now + timedelta(minutes=expiry_minutes)

    if priority_rank is not None:
        logger.info(
            "Starting maxdiff entry watcher for %s side=%s limit=%.4f qty=%.4f tolerance=%.4f "
            "expiry=%s force_immediate=%s priority_rank=%s",
            symbol,
            side,
            limit_price,
            target_qty,
            tolerance_pct,
            expiry.isoformat(),
            force_immediate,
            priority_rank,
        )
    else:
        logger.info(
            "Starting maxdiff entry watcher for %s side=%s limit=%.4f qty=%.4f tolerance=%.4f expiry=%s "
            "force_immediate=%s",
            symbol,
            side,
            limit_price,
            target_qty,
            tolerance_pct,
            expiry.isoformat(),
            force_immediate,
        )

    status = _prepare_status(
        config_path,
        {
            "mode": "entry",
            "symbol": symbol,
            "side": side,
            "limit_price": float(limit_price),
            "target_qty": float(target_qty),
            "tolerance_pct": float(tolerance_pct),
            "expiry_minutes": expiry_minutes,
            "expiry_at": expiry.isoformat(),
            "started_at": now.isoformat(),
            "asset_class": asset_class,
            "active": True,
            "force_immediate": bool(force_immediate),
        },
    )
    if priority_rank is not None:
        try:
            status["priority_rank"] = int(priority_rank)
        except (TypeError, ValueError):
            status["priority_rank"] = priority_rank
    status = _update_status(config_path, status, state="initializing")

    fallback_client = None
    if StockHistoricalDataClient is not None:
        try:
            fallback_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
        except Exception as exc:
            logger.debug("StockHistoricalDataClient init failed: %s", exc)

    try:
        while True:
            now = _now()
            if now >= expiry:
                logger.info("Entry watcher expired for %s; cancelling staged orders.", symbol)
                _cancel_orders(symbol, side=side)
                if asset_class.lower() == "crypto":
                    _cancel_orders(symbol)
                status = _update_status(
                    config_path,
                    status,
                    state="expired",
                    active=False,
                    expired_at=now.isoformat(),
                )
                break

            position = _position_for_symbol(symbol, side)
            active_orders = list(_orders_for_symbol(symbol, side=side))
            status = _update_status(
                config_path,
                status,
                active=True,
                position_qty=float(getattr(position, "qty", 0.0) or 0.0) if position else 0.0,
                open_order_count=len(active_orders),
            )

            if position is not None:
                status = _update_status(config_path, status, state="position_open")
                time.sleep(poll_seconds)
                continue

            if active_orders:
                status = _update_status(config_path, status, state="awaiting_fill")
                time.sleep(poll_seconds)
                continue

            reference_price = _latest_reference_price(symbol, side, fallback_client=fallback_client)
            status = _update_status(
                config_path,
                status,
                last_reference_price=reference_price,
            )

            if reference_price is None:
                status = _update_status(config_path, status, state="awaiting_price")
                time.sleep(poll_seconds)
                continue

            skip_tolerance = bool(status.get("force_immediate"))
            if not skip_tolerance and not _within_tolerance(reference_price, limit_price, tolerance_pct):
                status = _update_status(config_path, status, state="waiting_for_trigger")
                time.sleep(poll_seconds)
                continue
            if skip_tolerance and status.get("state") != "trigger_override":
                status = _update_status(config_path, status, state="trigger_override")

            if not _entry_requires_cash(side, limit_price, target_qty):
                status = _update_status(config_path, status, state="blocked_no_cash")
                time.sleep(poll_seconds)
                continue

            status = _update_status(config_path, status, state="submitting_order")
            try:
                result = alpaca_wrapper.open_order_at_price_or_all(symbol, target_qty, side, limit_price)
            except Exception as exc:
                logger.error("Failed to submit staged entry order for %s: %s", symbol, exc)
                status = _update_status(
                    config_path,
                    status,
                    state="order_error",
                    error=str(exc),
                )
            else:
                outcome = "accepted" if result is not None else "queued"
                status = _update_status(
                    config_path,
                    status,
                    state="order_submitted",
                    order_submission=outcome,
                )
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("Entry watcher interrupted for %s; marking as cancelled.", symbol)
        status = _update_status(
            config_path,
            status,
            state="cancelled",
            active=False,
            cancelled_at=_now().isoformat(),
        )
        raise
    except Exception as exc:
        status = _update_status(
            config_path,
            status,
            state="error",
            active=False,
            error=str(exc),
        )
        raise
    finally:
        if status.get("active", False):
            _update_status(config_path, status, active=False)


@app.command("close-position")
def close_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str = typer.Option(..., "--side", help="Entry side originally used (buy/sell)."),
    takeprofit_price: float = typer.Option(
        ..., "--takeprofit-price", help="Target price to unwind the position."
    ),
    expiry_minutes: int = typer.Option(
        24 * 60,
        "--expiry-minutes",
        help="Maximum monitoring window to re-arm take-profit orders.",
    ),
    asset_class: str = typer.Option(
        "equity",
        "--asset-class",
        help="Asset class hint (affects clean-up semantics).",
    ),
    poll_seconds: int = typer.Option(
        45,
        "--poll-seconds",
        help="Polling cadence while monitoring take-profit placement.",
    ),
    price_tolerance: float = typer.Option(
        0.001,
        "--price-tolerance",
        help="Relative tolerance when checking for existing take-profit orders.",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config-path",
        help="Path to persist watcher status updates.",
    ),
) -> None:
    side = _normalize_side(side)
    exit_side = "sell" if side == "buy" else "buy"
    if takeprofit_price <= 0:
        logger.error("Invalid takeprofit price %.4f for %s", takeprofit_price, symbol)
        return

    _ensure_strategy_tag(symbol)
    config_path = _normalize_config_path(config_path)
    expiry_minutes = max(int(expiry_minutes), 1)
    now = _now()
    expiry = now + timedelta(minutes=expiry_minutes)

    logger.info(
        "Starting maxdiff takeprofit watcher for %s entry_side=%s takeprofit=%.4f expiry=%s",
        symbol,
        side,
        takeprofit_price,
        expiry.isoformat(),
    )

    status = _prepare_status(
        config_path,
        {
            "mode": "exit",
            "symbol": symbol,
            "side": side,
            "exit_side": exit_side,
            "takeprofit_price": float(takeprofit_price),
            "price_tolerance": float(price_tolerance),
            "expiry_minutes": expiry_minutes,
            "expiry_at": expiry.isoformat(),
            "started_at": now.isoformat(),
            "asset_class": asset_class,
            "active": True,
        },
    )
    status = _update_status(config_path, status, state="initializing")

    try:
        while True:
            now = _now()
            if now >= expiry:
                logger.info("Takeprofit watcher expired for %s; cancelling exit orders.", symbol)
                _cancel_orders(symbol, side=exit_side)
                if asset_class.lower() == "crypto":
                    _cancel_orders(symbol)
                status = _update_status(
                    config_path,
                    status,
                    state="expired",
                    active=False,
                    expired_at=now.isoformat(),
                )
                break

            position = _position_for_symbol(symbol, side)
            qty = abs(float(getattr(position, "qty", 0.0) or 0.0)) if position else 0.0
            status = _update_status(
                config_path,
                status,
                active=True,
                position_qty=qty,
            )

            if position is None or qty <= 0:
                status = _update_status(config_path, status, state="awaiting_position")
                _cancel_orders(symbol, side=exit_side)
                time.sleep(poll_seconds)
                continue

            if _has_takeprofit_order(symbol, exit_side, takeprofit_price, tolerance=price_tolerance):
                status = _update_status(config_path, status, state="watching_orders")
                time.sleep(poll_seconds)
                continue

            _cancel_orders(symbol, side=exit_side)
            status = _update_status(config_path, status, state="submitting_exit")
            try:
                alpaca_wrapper.open_order_at_price(symbol, qty, exit_side, takeprofit_price)
            except Exception as exc:
                logger.error("Failed to submit takeprofit order for %s: %s", symbol, exc)
                status = _update_status(
                    config_path,
                    status,
                    state="exit_error",
                    error=str(exc),
                )
            else:
                open_orders = list(_orders_for_symbol(symbol, side=exit_side))
                status = _update_status(
                    config_path,
                    status,
                    state="exit_submitted",
                    open_order_count=len(open_orders),
                )
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("Takeprofit watcher interrupted for %s; marking as cancelled.", symbol)
        status = _update_status(
            config_path,
            status,
            state="cancelled",
            active=False,
            cancelled_at=_now().isoformat(),
        )
        raise
    except Exception as exc:
        status = _update_status(
            config_path,
            status,
            state="error",
            active=False,
            error=str(exc),
        )
        raise
    finally:
        if status.get("active", False):
            _update_status(config_path, status, active=False)


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    app()
