from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from src.binan import binance_wrapper
from src.binan.binance_margin import (
    create_margin_limit_buy,
    create_margin_limit_sell,
    cancel_margin_order,
)
from src.stock_utils import binance_remap_symbols

from .execution import resolve_symbol_rules, quantize_price, quantize_qty
from .pnl_state import record_fill

app = typer.Typer(help="Binance watcher CLI for limit order automation.")


@dataclass
class WatchStatus:
    config_version: int
    symbol: str
    side: str
    mode: str
    limit_price: float
    target_qty: float
    expiry_at: str
    started_at: str
    state: str
    active: bool
    order_id: Optional[int] = None
    order_status: Optional[str] = None
    last_update: Optional[str] = None
    binance_symbol: Optional[str] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_status(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read watcher status %s: %s", path, exc)
        return {}


def _write_status(path: Optional[Path], status: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)
        temp_path.replace(path)
    except Exception as exc:
        logger.warning("Failed to write watcher status %s: %s", path, exc)


def _update_status(path: Optional[Path], status: dict, **changes) -> dict:
    status.update(changes)
    status["last_update"] = _now().isoformat()
    _write_status(path, status)
    return status


def _coerce_order_id(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _extract_fill_details(order: dict) -> Optional[dict]:
    """Extract executed quantity and average fill price from a Binance order payload.

    Binance can mark an order as CANCELED/EXPIRED with a non-zero executedQty (partial fill).
    We treat any executedQty > 0 as a fill event for PnL/accounting purposes.
    """

    if not isinstance(order, dict):
        return None
    qty = _coerce_float(order.get("executedQty"))
    if qty <= 0:
        return None
    quote_qty = _coerce_float(order.get("cummulativeQuoteQty"))

    # `price` is the limit price, not necessarily the average fill price. Prefer quote/qty when available.
    limit_price = _coerce_float(order.get("price"))
    fill_price = quote_qty / qty if quote_qty > 0 else limit_price
    if fill_price <= 0:
        return None
    return {"fill_qty": qty, "fill_price": fill_price, "fill_quote": quote_qty}


def _get_order_status(symbol: str, order_id: int):
    client = binance_wrapper.get_client()
    try:
        order = client.get_order(symbol=symbol, orderId=order_id)
    except Exception as exc:
        logger.warning("Failed to fetch order %s for %s: %s", order_id, symbol, exc)
        return None
    return order


def _place_limit_order(symbol: str, side: str, quantity: float, price: float):
    return binance_wrapper.create_order(symbol, side.upper(), quantity, price)


def _place_margin_limit_order(symbol: str, side: str, quantity: float, price: float, side_effect_type: str = "NO_SIDE_EFFECT"):
    if side.upper() == "BUY":
        return create_margin_limit_buy(symbol, quantity, price, side_effect_type=side_effect_type)
    return create_margin_limit_sell(symbol, quantity, price, side_effect_type=side_effect_type)


def _get_margin_order_status(symbol: str, order_id: int):
    client = binance_wrapper.get_client()
    try:
        return client.get_margin_order(symbol=symbol, orderId=order_id, isIsolated="FALSE")
    except Exception as exc:
        logger.warning("Failed to fetch margin order %s for %s: %s", order_id, symbol, exc)
        return None


@app.command("watch")
def watch(
    symbol: str = typer.Argument(..., help="Symbol, e.g. SOLUSD"),
    side: str = typer.Option(..., "--side", help="buy or sell"),
    limit_price: float = typer.Option(..., "--limit-price"),
    target_qty: float = typer.Option(..., "--target-qty"),
    mode: str = typer.Option("entry", "--mode"),
    expiry_minutes: int = typer.Option(60, "--expiry-minutes"),
    poll_seconds: int = typer.Option(30, "--poll-seconds"),
    price_tolerance: float = typer.Option(0.0, "--price-tolerance"),
    config_path: Optional[Path] = typer.Option(None, "--config-path"),
    exchange_symbol: Optional[str] = typer.Option(None, "--exchange-symbol"),
    dry_run: bool = typer.Option(False, "--dry-run/--live"),
    margin: bool = typer.Option(False, "--margin/--no-margin"),
    side_effect_type: str = typer.Option("NO_SIDE_EFFECT", "--side-effect-type"),
) -> None:
    config_path = config_path.expanduser().resolve() if config_path else None
    status = _load_status(config_path)
    started_at = _now()
    expiry_at = started_at + timedelta(minutes=max(1, int(expiry_minutes)))
    binance_symbol = str(exchange_symbol or binance_remap_symbols(symbol)).upper()

    status_defaults = {
        "config_version": status.get("config_version", 1),
        "symbol": symbol,
        "side": side,
        "mode": mode,
        "limit_price": float(limit_price),
        "target_qty": float(target_qty),
        "expiry_at": expiry_at.isoformat(),
        "started_at": started_at.isoformat(),
        "state": "running",
        "active": True,
        "exchange_symbol": binance_symbol,
        "binance_symbol": binance_symbol,
    }
    status.update({k: v for k, v in status_defaults.items() if k not in status})
    _write_status(config_path, status)

    rules = resolve_symbol_rules(exchange_symbol or symbol)
    price = quantize_price(float(limit_price), tick_size=rules.tick_size, side=side)
    qty = quantize_qty(float(target_qty), step_size=rules.step_size)

    if rules.min_price is not None and price < rules.min_price:
        _update_status(config_path, status, state="below_min_price", active=False)
        return

    if qty <= 0 or price <= 0:
        _update_status(config_path, status, state="invalid_order", active=False)
        return

    min_notional = rules.min_notional or 0.0
    if min_notional and qty * price < min_notional:
        _update_status(config_path, status, state="below_min_notional", active=False)
        return

    if dry_run:
        logger.info(
            "[DRY RUN] Would place %s %s qty=%.8f @ %.6f",
            binance_symbol,
            side,
            qty,
            price,
        )
        _update_status(config_path, status, state="dry_run_complete", active=False)
        return

    order_id: Optional[int] = None
    last_status: Optional[str] = None

    while _now() < expiry_at:
        current_price = binance_wrapper.get_symbol_price(binance_symbol)
        if current_price is None:
            time.sleep(poll_seconds)
            continue

        if order_id is None:
            try:
                if margin:
                    order = _place_margin_limit_order(binance_symbol, side, qty, price, side_effect_type)
                else:
                    order = _place_limit_order(binance_symbol, side, qty, price)
                order_id = _coerce_order_id(order.get("orderId")) if isinstance(order, dict) else None
                last_status = order.get("status") if isinstance(order, dict) else None
                logger.info("Placed %s order %s for %s", "margin" if margin else "spot", order_id, binance_symbol)
                status = _update_status(
                    config_path,
                    status,
                    order_id=order_id,
                    order_status=last_status,
                    state="order_placed",
                )
            except Exception as exc:
                status = _update_status(config_path, status, state="order_failed", active=False, error=str(exc))
                return

        if order_id is not None:
            order = _get_margin_order_status(binance_symbol, order_id) if margin else _get_order_status(binance_symbol, order_id)
            if order:
                last_status = order.get("status")
                status = _update_status(config_path, status, order_status=last_status)
                if last_status in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}:
                    fill_details = _extract_fill_details(order) if isinstance(order, dict) else None
                    if fill_details:
                        status = _update_status(config_path, status, **fill_details)
                        record_fill(symbol, side, float(fill_details["fill_price"]), float(fill_details["fill_qty"]))
                    status = _update_status(config_path, status, state=last_status.lower(), active=False)
                    return

        time.sleep(poll_seconds)

    if order_id is not None:
        try:
            if margin:
                cancel_margin_order(binance_symbol, order_id=order_id)
            else:
                client = binance_wrapper.get_client()
                client.cancel_order(symbol=binance_symbol, orderId=order_id)
        except Exception as exc:
            logger.warning("Failed to cancel expired order %s for %s: %s", order_id, binance_symbol, exc)
        else:
            order = _get_margin_order_status(binance_symbol, order_id) if margin else _get_order_status(binance_symbol, order_id)
            if isinstance(order, dict):
                fill_details = _extract_fill_details(order)
                if fill_details:
                    status = _update_status(config_path, status, **fill_details)
                    record_fill(symbol, side, float(fill_details["fill_price"]), float(fill_details["fill_qty"]))
    _update_status(config_path, status, state="expired", active=False)


if __name__ == "__main__":
    app()
