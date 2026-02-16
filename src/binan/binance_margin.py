from __future__ import annotations

import math
from typing import Any, Dict, List

from binance import Client
from loguru import logger

from src.binan.binance_wrapper import (
    _coerce_balance_value,
    _format_price,
    _normalize_symbol,
    _resolve_client,
)


def get_margin_account(client: Client | None = None) -> Dict[str, Any]:
    client = _resolve_client(client)
    try:
        result = client.get_margin_account()
    except Exception as exc:
        logger.error(f"Failed to fetch margin account: {exc}")
        return {}
    if not isinstance(result, dict):
        logger.error(f"Unexpected margin account payload: {result}")
        return {}
    return result


def get_margin_balances(client: Client | None = None) -> List[Dict[str, Any]]:
    account = get_margin_account(client=client)
    raw = account.get("userAssets", [])
    if not isinstance(raw, list):
        logger.error(f"Unexpected userAssets payload: {raw}")
        return []
    return [e for e in raw if isinstance(e, dict)]


def get_margin_asset_balance(asset: str, client: Client | None = None) -> Dict[str, Any] | None:
    normalized = _normalize_symbol(asset)
    for entry in get_margin_balances(client=client):
        if entry.get("asset") == normalized:
            return entry
    return None


def get_margin_free_balance(asset: str, client: Client | None = None) -> float:
    entry = get_margin_asset_balance(asset, client=client)
    if entry is None:
        return 0.0
    return _coerce_balance_value(entry.get("free"))


def get_margin_borrowed_balance(asset: str, client: Client | None = None) -> float:
    entry = get_margin_asset_balance(asset, client=client)
    if entry is None:
        return 0.0
    return _coerce_balance_value(entry.get("borrowed"))


def get_margin_net_balance(asset: str, client: Client | None = None) -> float:
    entry = get_margin_asset_balance(asset, client=client)
    if entry is None:
        return 0.0
    return _coerce_balance_value(entry.get("netAsset"))


def get_max_borrowable(asset: str, client: Client | None = None) -> float:
    client = _resolve_client(client)
    normalized = _normalize_symbol(asset)
    try:
        result = client.margin_max_borrowable(asset=normalized)
    except Exception as exc:
        logger.error(f"Failed to fetch max borrowable for {normalized}: {exc}")
        return 0.0
    if not isinstance(result, dict):
        return 0.0
    return _coerce_balance_value(result.get("amount"))


def get_margin_interest_rate(
    assets: str | List[str], client: Client | None = None
) -> Dict[str, float]:
    client = _resolve_client(client)
    if isinstance(assets, list):
        asset_str = ",".join(a.strip().upper() for a in assets)
    else:
        asset_str = assets.strip().upper()
    try:
        result = client.margin_next_hourly_interest_rate(assets=asset_str, isIsolated="FALSE")
    except Exception as exc:
        logger.error(f"Failed to fetch margin interest rates: {exc}")
        return {}
    if not isinstance(result, list):
        logger.error(f"Unexpected interest rate payload: {result}")
        return {}
    rates: Dict[str, float] = {}
    for entry in result:
        if isinstance(entry, dict):
            a = entry.get("asset", "")
            try:
                rates[a] = float(entry.get("nextHourlyInterestRate", 0))
            except (TypeError, ValueError):
                pass
    return rates


# --- Transfers ---


def _validate_amount(amount: float) -> float:
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"Amount must be positive and finite, got {amount}")
    return amount


def transfer_spot_to_margin(
    asset: str, amount: float, client: Client | None = None
) -> Dict[str, Any]:
    client = _resolve_client(client)
    _validate_amount(amount)
    normalized = _normalize_symbol(asset)
    try:
        result = client.transfer_spot_to_margin(asset=normalized, amount=_format_price(amount))
    except Exception as exc:
        logger.error(f"Failed to transfer {amount} {normalized} spot->margin: {exc}")
        raise
    return result


def transfer_margin_to_spot(
    asset: str, amount: float, client: Client | None = None
) -> Dict[str, Any]:
    client = _resolve_client(client)
    _validate_amount(amount)
    normalized = _normalize_symbol(asset)
    try:
        result = client.transfer_margin_to_spot(asset=normalized, amount=_format_price(amount))
    except Exception as exc:
        logger.error(f"Failed to transfer {amount} {normalized} margin->spot: {exc}")
        raise
    return result


# --- Borrow / Repay ---


def margin_borrow(
    asset: str, amount: float, client: Client | None = None
) -> Dict[str, Any]:
    client = _resolve_client(client)
    _validate_amount(amount)
    normalized = _normalize_symbol(asset)
    try:
        result = client.margin_borrow_repay(
            asset=normalized,
            amount=_format_price(amount),
            type="BORROW",
            isIsolated="FALSE",
        )
    except Exception as exc:
        logger.error(f"Failed to borrow {amount} {normalized}: {exc}")
        raise
    logger.info(f"Borrowed {amount} {normalized}: {result}")
    return result


def margin_repay(
    asset: str, amount: float, client: Client | None = None
) -> Dict[str, Any]:
    client = _resolve_client(client)
    _validate_amount(amount)
    normalized = _normalize_symbol(asset)
    try:
        result = client.margin_borrow_repay(
            asset=normalized,
            amount=_format_price(amount),
            type="REPAY",
            isIsolated="FALSE",
        )
    except Exception as exc:
        logger.error(f"Failed to repay {amount} {normalized}: {exc}")
        raise
    logger.info(f"Repaid {amount} {normalized}: {result}")
    return result


def margin_repay_all(
    asset: str, client: Client | None = None
) -> Dict[str, Any] | None:
    borrowed = get_margin_borrowed_balance(asset, client=client)
    if borrowed <= 0:
        return None
    return margin_repay(asset, borrowed, client=client)


def get_borrow_repay_records(
    record_type: str,
    *,
    asset: str | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int | None = None,
    client: Client | None = None,
) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    rt = record_type.upper()
    if rt not in ("BORROW", "REPAY"):
        raise ValueError(f"record_type must be BORROW or REPAY, got {record_type!r}")
    payload: Dict[str, Any] = {"type": rt}
    if asset is not None:
        payload["asset"] = _normalize_symbol(asset)
    if start_time is not None:
        payload["startTime"] = int(start_time)
    if end_time is not None:
        payload["endTime"] = int(end_time)
    if limit is not None:
        payload["size"] = min(int(limit), 100)
    try:
        result = client.margin_get_borrow_repay_records(**payload)
    except Exception as exc:
        logger.error(f"Failed to fetch borrow/repay records: {exc}")
        return []
    if not isinstance(result, dict):
        return []
    rows = result.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [e for e in rows if isinstance(e, dict)]


# --- Orders ---


def create_margin_order(
    symbol: str,
    side: str,
    order_type: str,
    quantity: float | None = None,
    *,
    price: float | None = None,
    quote_order_qty: float | None = None,
    side_effect_type: str | None = None,
    time_in_force: str | None = None,
    client: Client | None = None,
) -> Dict[str, Any]:
    client = _resolve_client(client)
    normalized = _normalize_symbol(symbol)
    payload: Dict[str, Any] = {
        "symbol": normalized,
        "side": side.upper(),
        "type": order_type.upper(),
        "isIsolated": "FALSE",
    }
    if quantity is not None:
        payload["quantity"] = _format_price(quantity)
    if quote_order_qty is not None:
        payload["quoteOrderQty"] = _format_price(quote_order_qty)
    if price is not None:
        payload["price"] = _format_price(price)
    if side_effect_type is not None:
        payload["sideEffectType"] = side_effect_type
    if time_in_force is not None:
        payload["timeInForce"] = time_in_force
    try:
        result = client.create_margin_order(**payload)
    except Exception as exc:
        logger.error(f"Failed to create margin order: {exc}")
        logger.error(f"Payload: {payload}")
        raise
    return result


def create_margin_market_buy(
    symbol: str,
    quantity: float,
    *,
    side_effect_type: str = "NO_SIDE_EFFECT",
    client: Client | None = None,
) -> Dict[str, Any]:
    return create_margin_order(
        symbol, "BUY", "MARKET", quantity,
        side_effect_type=side_effect_type, client=client,
    )


def create_margin_market_sell(
    symbol: str,
    quantity: float,
    *,
    side_effect_type: str = "NO_SIDE_EFFECT",
    client: Client | None = None,
) -> Dict[str, Any]:
    return create_margin_order(
        symbol, "SELL", "MARKET", quantity,
        side_effect_type=side_effect_type, client=client,
    )


def create_margin_limit_buy(
    symbol: str,
    quantity: float,
    price: float,
    *,
    side_effect_type: str = "NO_SIDE_EFFECT",
    time_in_force: str = "GTC",
    client: Client | None = None,
) -> Dict[str, Any]:
    return create_margin_order(
        symbol, "BUY", "LIMIT", quantity,
        price=price, side_effect_type=side_effect_type,
        time_in_force=time_in_force, client=client,
    )


def create_margin_limit_sell(
    symbol: str,
    quantity: float,
    price: float,
    *,
    side_effect_type: str = "NO_SIDE_EFFECT",
    time_in_force: str = "GTC",
    client: Client | None = None,
) -> Dict[str, Any]:
    return create_margin_order(
        symbol, "SELL", "LIMIT", quantity,
        price=price, side_effect_type=side_effect_type,
        time_in_force=time_in_force, client=client,
    )


def cancel_margin_order(
    symbol: str,
    order_id: int | None = None,
    orig_client_order_id: str | None = None,
    client: Client | None = None,
) -> Dict[str, Any]:
    if order_id is None and orig_client_order_id is None:
        raise ValueError("Either order_id or orig_client_order_id is required")
    client = _resolve_client(client)
    normalized = _normalize_symbol(symbol)
    payload: Dict[str, Any] = {"symbol": normalized, "isIsolated": "FALSE"}
    if order_id is not None:
        payload["orderId"] = order_id
    if orig_client_order_id is not None:
        payload["origClientOrderId"] = orig_client_order_id
    try:
        result = client.cancel_margin_order(**payload)
    except Exception as exc:
        logger.error(f"Failed to cancel margin order: {exc}")
        raise
    return result


def cancel_all_margin_orders(
    symbol: str, client: Client | None = None
) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    normalized = _normalize_symbol(symbol)
    try:
        result = client.cancel_all_open_margin_orders(symbol=normalized, isIsolated="FALSE")
    except Exception as exc:
        logger.error(f"Failed to cancel all margin orders for {normalized}: {exc}")
        return []
    if not isinstance(result, list):
        return []
    return [e for e in result if isinstance(e, dict)]


def get_open_margin_orders(
    symbol: str | None = None, client: Client | None = None
) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    payload: Dict[str, Any] = {"isIsolated": "FALSE"}
    if symbol is not None:
        payload["symbol"] = _normalize_symbol(symbol)
    try:
        result = client.get_open_margin_orders(**payload)
    except Exception as exc:
        logger.error(f"Failed to fetch open margin orders: {exc}")
        return []
    if not isinstance(result, list):
        return []
    return [e for e in result if isinstance(e, dict)]


def get_all_margin_orders(
    symbol: str,
    *,
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int | None = None,
    client: Client | None = None,
) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    payload: Dict[str, Any] = {
        "symbol": _normalize_symbol(symbol),
        "isIsolated": "FALSE",
    }
    if start_time is not None:
        payload["startTime"] = int(start_time)
    if end_time is not None:
        payload["endTime"] = int(end_time)
    if limit is not None:
        payload["limit"] = int(limit)
    try:
        result = client.get_all_margin_orders(**payload)
    except Exception as exc:
        logger.error(f"Failed to fetch all margin orders: {exc}")
        return []
    if not isinstance(result, list):
        return []
    return [e for e in result if isinstance(e, dict)]


def get_margin_trades(
    symbol: str,
    *,
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int | None = None,
    client: Client | None = None,
) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    payload: Dict[str, Any] = {
        "symbol": _normalize_symbol(symbol),
        "isIsolated": "FALSE",
    }
    if start_time is not None:
        payload["startTime"] = int(start_time)
    if end_time is not None:
        payload["endTime"] = int(end_time)
    if limit is not None:
        payload["limit"] = int(limit)
    try:
        result = client.get_margin_trades(**payload)
    except Exception as exc:
        logger.error(f"Failed to fetch margin trades: {exc}")
        return []
    if not isinstance(result, list):
        return []
    return [e for e in result if isinstance(e, dict)]
