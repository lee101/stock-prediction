from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, cast

from binance import Client
from loguru import logger

from env_real import BINANCE_API_KEY, BINANCE_SECRET
from src.stock_utils import binance_remap_symbols

_client: Client | None


def _init_client() -> Client | None:
    kwargs: Dict[str, Any] = {}
    tld = os.getenv("BINANCE_TLD")
    base_endpoint = os.getenv("BINANCE_BASE_ENDPOINT")
    testnet = os.getenv("BINANCE_TESTNET", "").lower() in {"1", "true", "yes", "on"}
    demo = os.getenv("BINANCE_DEMO", "").lower() in {"1", "true", "yes", "on"}
    ping_setting = os.getenv("BINANCE_PING")

    if tld:
        kwargs["tld"] = tld
    if base_endpoint:
        kwargs["base_endpoint"] = base_endpoint
    if testnet:
        kwargs["testnet"] = True
    if demo:
        kwargs["demo"] = True
    if ping_setting is not None:
        kwargs["ping"] = ping_setting.lower() in {"1", "true", "yes", "on"}

    try:
        return Client(BINANCE_API_KEY, BINANCE_SECRET, **kwargs)
    except Exception as exc:  # pragma: no cover - connectivity / credential issues
        logger.error(f"Failed to initialise Binance client: {exc}")
        logger.info(
            "Maybe you are offline or using a restricted region. "
            "Set BINANCE_TLD (e.g., 'us') or BINANCE_BASE_ENDPOINT for alternate domains."
        )
        return None


_client = _init_client()


def _require_client() -> Client:
    if _client is None:
        raise RuntimeError(
            "Binance client is not initialised; check credentials, network connectivity, "
            "or configure BINANCE_TLD/BINANCE_BASE_ENDPOINT for your region."
        )
    return _client


def _resolve_client(client: Client | None = None) -> Client:
    if client is not None:
        return client
    global _client
    if _client is None:
        _client = _init_client()
    return _require_client()


def get_client(client: Client | None = None) -> Client:
    """Return a live Binance client, reinitializing if needed."""
    return _resolve_client(client)


def _coerce_balance_value(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise TypeError(f"Symbol must be a string, received {type(symbol).__name__}.")
    normalized = symbol.replace("/", "").strip().upper()
    normalized = binance_remap_symbols(normalized)
    return normalized


def _extract_filter_value(filter_entry: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in filter_entry:
            return _coerce_balance_value(filter_entry.get(key))
    return None


def _coerce_price(value: float | str | None) -> float:
    if value is None:
        raise ValueError("A price is required for Binance limit orders.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid price {value!r} supplied to Binance order helper.") from exc


def _format_price(value: float) -> str:
    # Binance expects a string; avoid scientific notation.
    return f"{value:.8f}".rstrip("0").rstrip(".") or "0"


def get_symbol_filters(symbol: str, client: Client | None = None) -> Dict[str, Dict[str, Any]]:
    client = _resolve_client(client)
    normalized = _normalize_symbol(symbol)
    try:
        info = client.get_symbol_info(normalized)
    except Exception as exc:
        logger.error(f"Failed to fetch Binance symbol info for {normalized}: {exc}")
        return {}
    if not isinstance(info, dict):
        logger.error(f"Unexpected Binance symbol info payload for {normalized}: {info}")
        return {}
    raw_filters = info.get("filters", [])
    if not isinstance(raw_filters, list):
        logger.error(f"Unexpected Binance symbol filters payload for {normalized}: {raw_filters}")
        return {}
    filters: Dict[str, Dict[str, Any]] = {}
    for entry in raw_filters:
        if not isinstance(entry, dict):
            continue
        filter_type = entry.get("filterType")
        if isinstance(filter_type, str) and filter_type:
            filters[filter_type] = entry
    return filters


def get_min_notional(symbol: str, client: Client | None = None) -> float | None:
    filters = get_symbol_filters(symbol, client=client)
    for filter_type in ("MIN_NOTIONAL", "NOTIONAL"):
        entry = filters.get(filter_type)
        if not entry:
            continue
        value = _extract_filter_value(entry, ("minNotional", "notional"))
        if value and value > 0:
            return value
    return None


def get_symbol_price(symbol: str, client: Client | None = None) -> float | None:
    client = _resolve_client(client)
    normalized = _normalize_symbol(symbol)
    data: Dict[str, Any] | None = None
    try:
        if hasattr(client, "get_symbol_ticker"):
            data = cast(Dict[str, Any], client.get_symbol_ticker(symbol=normalized))
        elif hasattr(client, "get_avg_price"):
            data = cast(Dict[str, Any], client.get_avg_price(symbol=normalized))
        else:
            logger.error("Binance client does not expose price fetch helpers.")
            return None
    except Exception as exc:
        logger.error(f"Failed to fetch Binance price for {normalized}: {exc}")
        return None
    if not isinstance(data, dict):
        logger.error(f"Unexpected Binance price payload for {normalized}: {data}")
        return None
    try:
        return float(data.get("price"))
    except (TypeError, ValueError):
        logger.error(f"Unexpected Binance price value for {normalized}: {data}")
        return None


crypto_symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "LTCUSDT",
    "UNIUSDT",
]


def create_order(symbol: str, side: str, quantity: float, price: float | str | None = None) -> Dict[str, Any]:
    client = _require_client()
    payload: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "type": Client.ORDER_TYPE_LIMIT,
        "timeInForce": Client.TIME_IN_FORCE_GTC,
        "quantity": quantity,
    }
    if price is not None:
        payload["price"] = _format_price(_coerce_price(price))

    order: Dict[str, Any]
    try:
        order = client.create_order(**payload)
    except Exception as exc:
        logger.error(f"Failed to create Binance order: {exc}")
        logger.error(f"Payload: {payload}")
        raise
    return order


def create_market_buy_quote(
    symbol: str,
    quote_amount: float,
    client: Client | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    client = _resolve_client(client)
    normalized = _normalize_symbol(symbol)
    if not math.isfinite(quote_amount) or quote_amount <= 0:
        raise ValueError(f"Quote amount must be positive, received {quote_amount}.")
    payload: Dict[str, Any] = {
        "symbol": normalized,
        "side": "BUY",
        "type": Client.ORDER_TYPE_MARKET,
        "quoteOrderQty": _format_price(quote_amount),
    }
    try:
        if dry_run and hasattr(client, "create_test_order"):
            return cast(Dict[str, Any], client.create_test_order(**payload))
        return cast(Dict[str, Any], client.create_order(**payload))
    except Exception as exc:
        logger.error(f"Failed to create Binance market buy: {exc}")
        logger.error(f"Payload: {payload}")
        raise


def create_all_in_order(symbol: str, side: str, price: float | str | None = None) -> Dict[str, Any]:
    balance_sell: float | None = None
    balance_buy: float | None = None
    balances = get_account_balances()
    for balance in balances:
        asset = balance.get("asset")
        free = balance.get("free")
        if free is None:
            continue
        try:
            free_amount = float(free)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring balance with unparsable free amount: {balance}")
            continue
        if asset == symbol[:3]:
            balance_sell = free_amount
        if asset == symbol[3:]:
            balance_buy = free_amount

    if balance_sell is None or balance_buy is None:
        raise RuntimeError(f"Cannot determine balances for symbol {symbol}, received: {balances}")

    side_upper = side.upper()
    limit_price = _coerce_price(price) if price is not None else None
    if side_upper == "SELL":
        quantity = balance_sell
    elif side_upper == "BUY":
        if limit_price is None:
            raise ValueError("Price is required for BUY orders.")
        quantity = balance_buy / limit_price
    else:
        raise ValueError(f"Invalid side '{side}'. Expected 'BUY' or 'SELL'.")

    quantity = math.floor(quantity * 1000) / 1000
    if quantity <= 0:
        raise RuntimeError(f"Calculated Binance order quantity {quantity} is not positive for symbol {symbol}.")

    order = create_order(symbol, side_upper, quantity, limit_price)
    logger.info(f"Created order on Binance: {order}")
    return order


def open_take_profit_position(position, row, price, qty):
    # entry_price = float(position.avg_entry_price)
    # current_price = row['close_last_price_minute']
    # current_symbol = row['symbol']
    try:
        mapped_symbol = binance_remap_symbols(position.symbol)
        if position.side == "long":
            create_all_in_order(mapped_symbol, "SELL", float(math.ceil(float(price))))
        else:
            create_all_in_order(mapped_symbol, "BUY", float(math.floor(float(price))))
    except Exception as e:
        logger.error(e)  # can be because theres a sell order already which is still relevant
        # close all positions? perhaps not
        return None
    return True


def close_position_at_current_price(position, row):
    if not row["close_last_price_minute"]:
        logger.info(f"nan price - for {position.symbol} market likely closed")
        return False
    try:
        if position.side == "long":
            create_all_in_order(binance_remap_symbols(position.symbol), "SELL", row["close_last_price_minute"])

        else:
            create_all_in_order(binance_remap_symbols(position.symbol), "BUY",
                                float(row["close_last_price_minute"]))
    except Exception as e:
        logger.error(e)  # cant convert nan to integer because market is closed for stocks
        # Out of range float values are not JSON compliant
        # could be because theres no minute data /trying to close at when market isn't open (might as well err/do nothing)
        # close all positions? perhaps not
        return None


def cancel_all_orders(client: Client | None = None):
    client = _resolve_client(client)
    for symbol in crypto_symbols:
        orders = get_all_orders(symbol, client=client)
        for order in orders:
            if order["status"] == "CANCELED" or order["status"] == "FILLED":
                continue
            try:
                client.cancel_order(symbol=order["symbol"], orderId=order["orderId"])
            except Exception as e:
                print(e)
                logger.error(e)


def get_all_orders(symbol: str, client: Client | None = None) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    try:
        raw_orders = client.get_all_orders(symbol=symbol)
    except Exception as e:
        logger.error(e)
        return []
    if not isinstance(raw_orders, list):
        logger.error(f"Unexpected orders payload from Binance: {raw_orders}")
        return []
    orders: List[Dict[str, Any]] = []
    for entry in raw_orders:
        if isinstance(entry, dict):
            orders.append(entry)
        else:
            logger.debug(f"Discarding non-dict order entry: {entry}")
    return orders


def get_open_orders(symbol: str | None = None, client: Client | None = None) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    payload: Dict[str, Any] = {}
    if symbol:
        payload["symbol"] = _normalize_symbol(symbol)
    try:
        raw_orders = client.get_open_orders(**payload)
    except Exception as exc:
        logger.error(f"Failed to fetch Binance open orders: {exc}")
        return []
    if not isinstance(raw_orders, list):
        logger.error(f"Unexpected open orders payload from Binance: {raw_orders}")
        return []
    orders: List[Dict[str, Any]] = []
    for entry in raw_orders:
        if isinstance(entry, dict):
            orders.append(entry)
        else:
            logger.debug(f"Discarding non-dict open order entry: {entry}")
    return orders


def get_my_trades(
    symbol: str,
    *,
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int | None = None,
    client: Client | None = None,
) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    payload: Dict[str, Any] = {"symbol": _normalize_symbol(symbol)}
    if start_time is not None:
        payload["startTime"] = int(start_time)
    if end_time is not None:
        payload["endTime"] = int(end_time)
    if limit is not None:
        payload["limit"] = int(limit)
    try:
        raw_trades = client.get_my_trades(**payload)
    except Exception as exc:
        logger.error(f"Failed to fetch Binance trades for {payload['symbol']}: {exc}")
        return []
    if not isinstance(raw_trades, list):
        logger.error(f"Unexpected trades payload from Binance: {raw_trades}")
        return []
    trades: List[Dict[str, Any]] = []
    for entry in raw_trades:
        if isinstance(entry, dict):
            trades.append(entry)
        else:
            logger.debug(f"Discarding non-dict trade entry: {entry}")
    return trades
def get_account_balances(client: Client | None = None) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    try:
        account = cast(Dict[str, Any], client.get_account())
        balances_obj = cast(Iterable[Dict[str, Any]] | None, account.get("balances", []))
    except Exception as e:
        logger.error(e)
        return []

    if balances_obj is None:
        logger.error(f"Binance account payload missing 'balances' key: {account}")
        return []

    filtered: List[Dict[str, Any]] = []
    for entry in balances_obj:
        if isinstance(entry, dict):
            filtered.append(entry)
        else:
            logger.debug(f"Discarding non-dict balance entry: {entry}")
    return filtered


def get_asset_balance(asset: str, balances: Iterable[Dict[str, Any]] | None = None) -> Dict[str, Any] | None:
    normalized = _normalize_symbol(asset)
    balance_list = list(balances) if balances is not None else get_account_balances()
    for entry in balance_list:
        if entry.get("asset") == normalized:
            return entry
    return None


def get_asset_free_balance(asset: str, client: Client | None = None) -> float:
    balance = get_asset_balance(asset, balances=get_account_balances(client=client))
    if balance is None:
        return 0.0
    return _coerce_balance_value(balance.get("free"))


def get_asset_total_balance(asset: str, client: Client | None = None) -> float:
    balance = get_asset_balance(asset, balances=get_account_balances(client=client))
    if balance is None:
        return 0.0
    free = _coerce_balance_value(balance.get("free"))
    locked = _coerce_balance_value(balance.get("locked"))
    return free + locked


_STABLECOIN_ASSETS = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "U"}


def get_account_value_usdt(
    include_locked: bool = True, client: Client | None = None
) -> Dict[str, Any]:
    balances = get_account_balances(client=client)
    total_value = 0.0
    assets: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for entry in balances:
        asset = entry.get("asset")
        if not isinstance(asset, str) or not asset:
            continue
        free = _coerce_balance_value(entry.get("free"))
        locked = _coerce_balance_value(entry.get("locked"))
        total_amount = free + locked if include_locked else free
        if total_amount <= 0:
            continue
        asset_upper = asset.upper()
        if asset_upper in _STABLECOIN_ASSETS:
            price = 1.0
            value = total_amount
        else:
            symbol = f"{asset_upper}USDT"
            price = get_symbol_price(symbol, client=client)
            if price is None:
                skipped.append(
                    {"asset": asset_upper, "amount": total_amount, "reason": "missing_usdt_price"}
                )
                continue
            value = total_amount * price
        assets.append(
            {
                "asset": asset_upper,
                "free": free,
                "locked": locked,
                "amount": total_amount,
                "price_usdt": price,
                "value_usdt": value,
            }
        )
        total_value += value

    return {
        "total_usdt": total_value,
        "assets": assets,
        "skipped": skipped,
    }


def get_account_snapshots_spot(limit: int = 5, client: Client | None = None) -> List[Dict[str, Any]]:
    client = _resolve_client(client)
    safe_limit = max(1, min(int(limit), 30))
    try:
        payload = cast(Dict[str, Any], client.get_account_snapshot(type="SPOT", limit=safe_limit))
    except Exception as exc:
        logger.error(f"Failed to fetch Binance account snapshots: {exc}")
        return []
    if not isinstance(payload, dict):
        logger.error(f"Unexpected snapshot payload from Binance: {payload}")
        return []
    snapshots = payload.get("snapshotVos", [])
    if not isinstance(snapshots, list):
        logger.error(f"Unexpected snapshot list from Binance: {snapshots}")
        return []
    cleaned: List[Dict[str, Any]] = []
    for entry in snapshots:
        if isinstance(entry, dict):
            cleaned.append(entry)
        else:
            logger.debug(f"Discarding non-dict snapshot entry: {entry}")
    cleaned.sort(key=lambda item: item.get("updateTime", 0))
    return cleaned


def get_prev_day_pnl_usdt(client: Client | None = None) -> Dict[str, Any]:
    snapshots = get_account_snapshots_spot(limit=2, client=client)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough account snapshots to compute previous-day PnL.")

    prev_snap = snapshots[-2]
    latest_snap = snapshots[-1]

    def _total_btc(entry: Dict[str, Any]) -> float:
        data = entry.get("data", {})
        if not isinstance(data, dict):
            return 0.0
        try:
            return float(data.get("totalAssetOfBtc", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _format_time(entry: Dict[str, Any]) -> str | None:
        ts = entry.get("updateTime")
        if not isinstance(ts, (int, float)):
            return None
        return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat()

    prev_btc = _total_btc(prev_snap)
    latest_btc = _total_btc(latest_snap)
    delta_btc = latest_btc - prev_btc

    btc_price = get_symbol_price("BTCUSDT", client=client)
    if btc_price is None:
        raise RuntimeError("Unable to fetch BTCUSDT price to value account snapshots.")
    delta_usdt = delta_btc * btc_price

    return {
        "prev_total_btc": prev_btc,
        "latest_total_btc": latest_btc,
        "delta_btc": delta_btc,
        "btc_price_usdt": btc_price,
        "delta_usdt": delta_usdt,
        "prev_update_time": _format_time(prev_snap),
        "latest_update_time": _format_time(latest_snap),
    }


def buy_usdt_to_btc(
    quote_amount: float,
    client: Client | None = None,
    min_notional_override: float | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    client = _resolve_client(client)
    if not math.isfinite(quote_amount) or quote_amount <= 0:
        raise ValueError(f"Quote amount must be positive, received {quote_amount}.")
    symbol = "BTCUSDT"
    min_notional = min_notional_override or get_min_notional(symbol, client=client)
    trade_notional = quote_amount
    if min_notional and trade_notional < min_notional:
        logger.warning(
            f"Requested {quote_amount:.2f} USDT below Binance minimum {min_notional:.2f}; "
            "using minimum notional."
        )
        trade_notional = min_notional

    free_usdt = get_asset_free_balance("USDT", client=client)
    if free_usdt < trade_notional:
        raise RuntimeError(
            f"Insufficient USDT balance for BTC purchase. Available={free_usdt:.4f}, "
            f"required={trade_notional:.4f}."
        )

    order = create_market_buy_quote(
        symbol,
        trade_notional,
        client=client,
        dry_run=dry_run,
    )
    logger.info(f"Placed Binance BTC buy order: {order}")
    return order
