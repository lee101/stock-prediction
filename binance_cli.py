from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import typer

from src.binan import binance_wrapper
from src.fixtures import crypto_symbols as default_crypto_symbols
from src.binan.holdings_snapshot import DEFAULT_DB_PATH, load_latest_snapshot, record_snapshot

app = typer.Typer(help="Binance spot trading CLI utilities.")


def _coerce_balance_value(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _format_amount(value: float, precision: int = 8) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    formatted = f"{numeric:.{precision}f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def _format_usdt(value: float) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"${numeric:,.2f}"


def _format_side(value) -> str:
    if not isinstance(value, str):
        return "n/a"
    return value.upper()

def _format_delta(value: float) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    sign = "+" if numeric >= 0 else ""
    return f"{sign}{numeric:,.2f}"


_KNOWN_QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "USD")
_KNOWN_QUOTE_ASSETS = tuple(sorted(_KNOWN_QUOTE_ASSETS, key=len, reverse=True))
_STABLECOIN_ASSETS = set(_KNOWN_QUOTE_ASSETS) - {"USD"}


def _split_symbol_pair(symbol: str) -> Tuple[str, str]:
    normalized = symbol.replace("/", "").strip().upper()
    for quote in _KNOWN_QUOTE_ASSETS:
        if normalized.endswith(quote):
            base = normalized[: -len(quote)] or normalized
            return base, quote
    if len(normalized) >= 3:
        return normalized[:-3], normalized[-3:]
    return normalized, ""


def _coerce_trade_float(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _coerce_trade_time_ms(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def _fetch_trades_window(
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> List[Dict[str, object]]:
    trades: List[Dict[str, object]] = []
    seen_ids: set[int] = set()
    cursor = start_ms
    safe_limit = max(1, min(int(limit), 1000))

    while True:
        batch = binance_wrapper.get_my_trades(
            symbol,
            start_time=cursor,
            end_time=end_ms,
            limit=safe_limit,
        )
        if not batch:
            break
        max_time = cursor
        for trade in batch:
            if not isinstance(trade, dict):
                continue
            trade_id = _coerce_trade_time_ms(trade.get("id"))
            if trade_id and trade_id in seen_ids:
                continue
            if trade_id:
                seen_ids.add(trade_id)
            trade_time = _coerce_trade_time_ms(trade.get("time"))
            if trade_time < start_ms or trade_time > end_ms:
                continue
            trades.append(trade)
            if trade_time > max_time:
                max_time = trade_time
        if len(batch) < safe_limit:
            break
        if max_time <= cursor:
            break
        cursor = max_time + 1

    trades.sort(key=lambda item: _coerce_trade_time_ms(item.get("time")))
    return trades


def _estimate_fee_usdt(
    trade: Dict[str, object],
    *,
    base_asset: str,
    quote_asset: str,
    price_cache: Dict[str, Optional[float]],
) -> Tuple[float, Optional[str]]:
    commission = _coerce_trade_float(trade.get("commission"))
    if commission <= 0:
        return 0.0, None
    asset = trade.get("commissionAsset")
    if not isinstance(asset, str) or not asset:
        return 0.0, None
    asset = asset.upper()

    if asset == quote_asset or asset in _STABLECOIN_ASSETS:
        return commission, None
    if asset == base_asset:
        price = _coerce_trade_float(trade.get("price"))
        if price > 0:
            return commission * price, None
        return 0.0, asset

    price_key = f"{asset}USDT"
    if price_key not in price_cache:
        price_cache[price_key] = binance_wrapper.get_symbol_price(price_key)
    price = price_cache.get(price_key)
    if price is None:
        return 0.0, asset
    return commission * float(price), None
def _normalize_assets_filter(assets: Optional[List[str]]) -> Optional[set[str]]:
    if not assets:
        return None
    normalized: set[str] = set()
    for entry in assets:
        if not entry:
            continue
        for token in entry.replace(",", " ").split():
            token = token.strip().upper()
            if token:
                normalized.add(token)
    return normalized or None


def _normalize_symbols_filter(symbols: Optional[List[str]]) -> Optional[List[str]]:
    if not symbols:
        return None
    normalized: List[str] = []
    seen: set[str] = set()
    for entry in symbols:
        if not entry:
            continue
        for token in entry.replace(",", " ").split():
            token = token.strip().upper().replace("/", "")
            if token and token not in seen:
                normalized.append(token)
                seen.add(token)
    return normalized or None


def _format_time_ms(value) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return "n/a"
    if numeric <= 0:
        return "n/a"
    dt = datetime.fromtimestamp(numeric / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def _handle_cli_error(exc: Exception) -> None:
    typer.echo(f"Error: {exc}")
    raise typer.Exit(code=1)


@app.command("balances")
def balances(
    assets: Optional[List[str]] = typer.Option(
        None, "--asset", "-a", help="Filter to asset(s); repeatable or comma-separated."
    ),
    min_total: float = typer.Option(0.0, "--min-total", help="Hide balances below this total."),
    include_locked: bool = typer.Option(True, "--include-locked/--free-only"),
    show_zero: bool = typer.Option(False, "--show-zero/--hide-zero"),
) -> None:
    """List account balances from Binance spot."""
    normalized_assets = _normalize_assets_filter(assets)
    try:
        balances_list = binance_wrapper.get_account_balances()
    except RuntimeError as exc:
        _handle_cli_error(exc)

    lines: List[str] = []
    for entry in balances_list:
        asset = entry.get("asset")
        if not isinstance(asset, str) or not asset:
            continue
        asset = asset.upper()
        if normalized_assets and asset not in normalized_assets:
            continue
        free = _coerce_balance_value(entry.get("free"))
        locked = _coerce_balance_value(entry.get("locked"))
        total = free + locked if include_locked else free
        if not show_zero and total <= 0:
            continue
        if total < min_total:
            continue
        line = (
            f"- {asset} free={_format_amount(free)} locked={_format_amount(locked)} "
            f"total={_format_amount(total)}"
        )
        lines.append(line)

    if not lines:
        typer.echo("No matching balances.")
        raise typer.Exit(code=0)

    typer.echo("Balances:")
    for line in sorted(lines):
        typer.echo(line)


@app.command("balance")
def balance(
    asset: str = typer.Argument(..., help="Asset symbol, e.g. BTC or USDT."),
    include_locked: bool = typer.Option(True, "--include-locked/--free-only"),
) -> None:
    """Show a single asset balance."""
    try:
        entry = binance_wrapper.get_asset_balance(asset)
    except RuntimeError as exc:
        _handle_cli_error(exc)
    if entry is None:
        typer.echo(f"Asset {asset} not found in account balances.")
        raise typer.Exit(code=1)
    free = _coerce_balance_value(entry.get("free"))
    locked = _coerce_balance_value(entry.get("locked"))
    total = free + locked if include_locked else free
    typer.echo(
        f"{asset.upper()} free={_format_amount(free)} locked={_format_amount(locked)} "
        f"total={_format_amount(total)}"
    )


@app.command("account-value")
def account_value(
    include_locked: bool = typer.Option(True, "--include-locked/--free-only"),
    show_assets: bool = typer.Option(True, "--show-assets/--hide-assets"),
) -> None:
    """Estimate total account value in USDT using spot prices."""
    try:
        result = binance_wrapper.get_account_value_usdt(include_locked=include_locked)
    except RuntimeError as exc:
        _handle_cli_error(exc)
    total = result.get("total_usdt", 0.0)
    typer.echo(f"Total Account Value (USDT) = {_format_usdt(total)}")

    assets = result.get("assets", [])
    if show_assets and isinstance(assets, list):
        typer.echo("Assets:")
        for entry in sorted(assets, key=lambda item: item.get("value_usdt", 0.0), reverse=True):
            asset = entry.get("asset", "n/a")
            amount = entry.get("amount", 0.0)
            price = entry.get("price_usdt", 0.0)
            value = entry.get("value_usdt", 0.0)
            typer.echo(
                f"- {asset} amount={_format_amount(amount)} price={_format_amount(price, 4)} "
                f"value={_format_usdt(value)}"
            )

    skipped = result.get("skipped", [])
    if skipped:
        typer.echo("Skipped assets (missing USDT price):")
        for entry in skipped:
            asset = entry.get("asset", "n/a")
            amount = entry.get("amount", 0.0)
            typer.echo(f"- {asset} amount={_format_amount(amount)}")


@app.command("orders")
def orders(
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Symbol(s), repeatable or comma-separated (e.g., BTCUSDT)."
    ),
    open_only: bool = typer.Option(True, "--open-only/--all"),
) -> None:
    """Show Binance orders for the provided symbols."""
    normalized_symbols = _normalize_assets_filter(symbols)
    if not normalized_symbols:
        normalized_symbols = {symbol.upper() for symbol in binance_wrapper.crypto_symbols}
    lines: List[str] = []
    for symbol in sorted(normalized_symbols):
        try:
            if open_only:
                orders_list = binance_wrapper.get_open_orders(symbol)
            else:
                orders_list = binance_wrapper.get_all_orders(symbol)
        except RuntimeError as exc:
            _handle_cli_error(exc)
        for order in orders_list:
            status = order.get("status", "n/a")
            side = _format_side(order.get("side"))
            qty = _format_amount(order.get("origQty", order.get("executedQty", 0)))
            price = _format_amount(order.get("price", 0))
            order_id = order.get("orderId", "n/a")
            lines.append(
                f"- {symbol} order_id={order_id} side={side} status={status} qty={qty} price={price}"
            )
    if not lines:
        typer.echo("No orders found.")
        raise typer.Exit(code=0)
    typer.echo("Orders:")
    for line in lines:
        typer.echo(line)


@app.command("trades")
def trades(
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Symbol(s), repeatable or comma-separated (e.g., BTCUSDT)."
    ),
    limit: int = typer.Option(50, "--limit", help="Max trades to show per symbol."),
) -> None:
    """Show recent Binance trades per symbol."""
    normalized_symbols = _normalize_assets_filter(symbols)
    if not normalized_symbols:
        normalized_symbols = {symbol.upper() for symbol in binance_wrapper.crypto_symbols}
    lines: List[str] = []
    total_count = 0
    for symbol in sorted(normalized_symbols):
        try:
            trades_list = binance_wrapper.get_my_trades(symbol)
        except RuntimeError as exc:
            _handle_cli_error(exc)
        trades_list = trades_list[: max(limit, 0)]
        total_count += len(trades_list)
        for trade in trades_list:
            side = "BUY" if trade.get("isBuyer") else "SELL"
            qty = _format_amount(trade.get("qty", 0))
            price = _format_amount(trade.get("price", 0))
            trade_id = trade.get("id", "n/a")
            lines.append(f"- {symbol} trade_id={trade_id} side={side} qty={qty} price={price}")
    if not lines:
        typer.echo("No trades found.")
        raise typer.Exit(code=0)
    typer.echo(f"Trades (showing {len(lines)} total, across {total_count} fetched):")
    for line in lines:
        typer.echo(line)


@app.command("summary")
def summary() -> None:
    """Show a quick summary of balances, open orders, and trade counts."""
    typer.echo("Balances:")
    try:
        balances_list = binance_wrapper.get_account_balances()
    except RuntimeError as exc:
        _handle_cli_error(exc)
    for entry in balances_list:
        asset = entry.get("asset")
        if not isinstance(asset, str) or not asset:
            continue
        free = _coerce_balance_value(entry.get("free"))
        locked = _coerce_balance_value(entry.get("locked"))
        total = free + locked
        if total <= 0:
            continue
        typer.echo(
            f"- {asset.upper()} free={_format_amount(free)} locked={_format_amount(locked)} "
            f"total={_format_amount(total)}"
        )

    typer.echo("Open Orders:")
    order_count = 0
    for symbol in sorted(binance_wrapper.crypto_symbols):
        try:
            orders_list = binance_wrapper.get_open_orders(symbol)
        except RuntimeError as exc:
            _handle_cli_error(exc)
        for order in orders_list:
            order_count += 1
            status = order.get("status", "n/a")
            side = _format_side(order.get("side"))
            qty = _format_amount(order.get("origQty", order.get("executedQty", 0)))
            price = _format_amount(order.get("price", 0))
            order_id = order.get("orderId", "n/a")
            typer.echo(
                f"- {symbol} order_id={order_id} side={side} status={status} qty={qty} price={price}"
            )
    typer.echo(f"Open Orders Count: {order_count}")

    typer.echo("Trade Counts:")
    total_trades = 0
    for symbol in sorted(binance_wrapper.crypto_symbols):
        try:
            trades_list = binance_wrapper.get_my_trades(symbol)
        except RuntimeError as exc:
            _handle_cli_error(exc)
        count = len(trades_list)
        total_trades += count
        typer.echo(f"- {symbol} trades={count}")
    typer.echo(f"Total Trades: {total_trades}")
@app.command("open-orders")
def open_orders(
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Filter to symbol(s), e.g. BTCUSDT or BTCUSD."
    ),
) -> None:
    """List open spot orders on Binance."""
    normalized = _normalize_symbols_filter(symbols)
    try:
        if not normalized:
            orders = binance_wrapper.get_open_orders()
        else:
            orders = []
            for symbol in normalized:
                orders.extend(binance_wrapper.get_open_orders(symbol))
    except RuntimeError as exc:
        _handle_cli_error(exc)

    if not orders:
        typer.echo("No open orders.")
        raise typer.Exit(code=0)

    typer.echo("Open Orders:")
    for order in sorted(orders, key=lambda item: (item.get("symbol", ""), item.get("time", 0))):
        symbol = order.get("symbol", "n/a")
        side = order.get("side", "n/a")
        order_type = order.get("type", "n/a")
        status = order.get("status", "n/a")
        price = order.get("price")
        qty = order.get("origQty")
        filled = order.get("executedQty")
        typer.echo(
            f"- {symbol} {side} {order_type} status={status} "
            f"price={_format_amount(price, 8)} qty={_format_amount(qty, 8)} filled={_format_amount(filled, 8)}"
        )


@app.command("daily-pnl")
def daily_pnl() -> None:
    """Estimate previous-day PnL using Binance spot account snapshots."""
    try:
        result = binance_wrapper.get_prev_day_pnl_usdt()
    except RuntimeError as exc:
        _handle_cli_error(exc)

    typer.echo("Previous-day PnL (spot snapshot):")
    typer.echo(f"- prev_total_btc: {_format_amount(result.get('prev_total_btc', 0.0), 8)}")
    typer.echo(f"- latest_total_btc: {_format_amount(result.get('latest_total_btc', 0.0), 8)}")
    typer.echo(f"- delta_btc: {_format_amount(result.get('delta_btc', 0.0), 8)}")
    typer.echo(f"- btc_price_usdt: {_format_usdt(result.get('btc_price_usdt', 0.0))}")
    typer.echo(f"- delta_usdt: {_format_usdt(result.get('delta_usdt', 0.0))}")
    typer.echo(f"- prev_update_time: {result.get('prev_update_time')}")
    typer.echo(f"- latest_update_time: {result.get('latest_update_time')}")


@app.command("recent-trades")
def recent_trades(
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Filter to symbol(s), e.g. BTCUSDT or BTCUSD."
    ),
    days: float = typer.Option(1.0, "--days", help="Lookback window in days."),
    limit: int = typer.Option(1000, "--limit", help="Max trades per symbol (default 1000)."),
) -> None:
    """Show executed trades for the past N days."""
    if not math.isfinite(days) or days <= 0:
        _handle_cli_error(ValueError(f"Days must be positive, received {days}."))
    normalized = _normalize_symbols_filter(symbols)
    if not normalized:
        normalized = ["BTCUSD", "ETHUSD", "LINKUSD"]

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)

    all_trades = []
    for symbol in normalized:
        trades = binance_wrapper.get_my_trades(
            symbol,
            start_time=start_ms,
            limit=limit,
        )
        for trade in trades:
            trade_time = trade.get("time")
            try:
                trade_ms = int(trade_time)
            except (TypeError, ValueError):
                trade_ms = 0
            if trade_ms >= start_ms:
                all_trades.append(trade)

    if not all_trades:
        typer.echo("No executed trades in the requested window.")
        raise typer.Exit(code=0)

    all_trades.sort(key=lambda item: (item.get("time", 0), item.get("symbol", "")))

    typer.echo("Executed trades:")
    for trade in all_trades:
        symbol = trade.get("symbol", "n/a")
        side = "BUY" if trade.get("isBuyer") else "SELL"
        price = trade.get("price")
        qty = trade.get("qty")
        quote_qty = trade.get("quoteQty")
        commission = trade.get("commission")
        commission_asset = trade.get("commissionAsset", "")
        trade_id = trade.get("id", "n/a")
        order_id = trade.get("orderId", "n/a")
        time_str = _format_time_ms(trade.get("time"))
        typer.echo(
            f"- {time_str} {symbol} {side} price={_format_amount(price, 8)} "
            f"qty={_format_amount(qty, 8)} quote={_format_amount(quote_qty, 8)} "
            f"order_id={order_id} trade_id={trade_id} "
            f"commission={_format_amount(commission, 8)} {commission_asset}"
        )


@app.command("trade-pnl")
def trade_pnl(
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Filter to symbol(s), e.g. SOLUSD or BTCUSDT."
    ),
    days: float = typer.Option(1.0, "--days", help="Lookback window in days."),
    limit: int = typer.Option(1000, "--limit", help="Max trades per API call (default 1000)."),
    include_fees: bool = typer.Option(True, "--include-fees/--exclude-fees"),
) -> None:
    """Estimate realized PnL from executed Binance trades over the past N days."""
    if not math.isfinite(days) or days <= 0:
        _handle_cli_error(ValueError(f"Days must be positive, received {days}."))

    normalized = _normalize_symbols_filter(symbols)
    if not normalized:
        normalized = [symbol.upper() for symbol in default_crypto_symbols]

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    typer.echo(f"Trade PnL Window: {start.isoformat()} -> {now.isoformat()}")

    total_realized = 0.0
    total_fees = 0.0
    total_matched_qty = 0.0
    total_unmatched_sells = 0.0
    total_trades = 0
    total_buy_qty = 0.0
    total_sell_qty = 0.0

    price_cache: Dict[str, Optional[float]] = {}

    for symbol in normalized:
        trades = _fetch_trades_window(symbol, start_ms=start_ms, end_ms=end_ms, limit=limit)
        if not trades:
            typer.echo(f"{symbol}: No executed trades in window.")
            continue

        base_asset, quote_asset = _split_symbol_pair(symbol)
        buy_lots: List[Tuple[float, float]] = []
        buy_qty = 0.0
        sell_qty = 0.0
        buy_notional = 0.0
        sell_notional = 0.0
        realized = 0.0
        matched_qty = 0.0
        unmatched_sells = 0.0
        fees_usdt = 0.0
        unknown_fee_assets: Dict[str, float] = {}

        for trade in trades:
            price = _coerce_trade_float(trade.get("price"))
            qty = _coerce_trade_float(trade.get("qty"))
            if price <= 0 or qty <= 0:
                continue
            side_is_buy = _boolish(trade.get("isBuyer"))

            fee_usdt, unknown_asset = _estimate_fee_usdt(
                trade,
                base_asset=base_asset,
                quote_asset=quote_asset,
                price_cache=price_cache,
            )
            fees_usdt += fee_usdt
            if unknown_asset:
                unknown_fee_assets[unknown_asset] = unknown_fee_assets.get(unknown_asset, 0.0) + _coerce_trade_float(
                    trade.get("commission")
                )

            if side_is_buy:
                buy_lots.append((qty, price))
                buy_qty += qty
                buy_notional += qty * price
            else:
                sell_qty += qty
                sell_notional += qty * price
                remaining = qty
                while remaining > 0 and buy_lots:
                    lot_qty, lot_price = buy_lots[0]
                    match_qty = min(lot_qty, remaining)
                    realized += match_qty * (price - lot_price)
                    matched_qty += match_qty
                    lot_qty -= match_qty
                    remaining -= match_qty
                    if lot_qty <= 0:
                        buy_lots.pop(0)
                    else:
                        buy_lots[0] = (lot_qty, lot_price)
                if remaining > 0:
                    unmatched_sells += remaining

        trade_count = len(trades)
        avg_buy = buy_notional / buy_qty if buy_qty > 0 else 0.0
        avg_sell = sell_notional / sell_qty if sell_qty > 0 else 0.0
        net_realized = realized - fees_usdt if include_fees else realized

        typer.echo(f"\n{symbol} ({trade_count} trades)")
        typer.echo(
            f"Buys: qty={_format_amount(buy_qty, 8)} avg={_format_amount(avg_buy, 6)} "
            f"notional={_format_usdt(buy_notional)}"
        )
        typer.echo(
            f"Sells: qty={_format_amount(sell_qty, 8)} avg={_format_amount(avg_sell, 6)} "
            f"notional={_format_usdt(sell_notional)}"
        )
        typer.echo(
            f"Matched: qty={_format_amount(matched_qty, 8)} realized={_format_usdt(realized)}"
        )
        typer.echo(f"Unmatched sells: qty={_format_amount(unmatched_sells, 8)}")
        if include_fees:
            typer.echo(f"Fees (USDT est): {_format_usdt(fees_usdt)}")
        if unknown_fee_assets:
            unknown_str = ", ".join(
                f"{asset}={_format_amount(amount, 8)}" for asset, amount in sorted(unknown_fee_assets.items())
            )
            typer.echo(f"Unknown fee assets: {unknown_str}")
        if include_fees:
            typer.echo(f"Net realized (after fees): {_format_usdt(net_realized)}")

        total_realized += realized
        total_fees += fees_usdt
        total_matched_qty += matched_qty
        total_unmatched_sells += unmatched_sells
        total_trades += trade_count
        total_buy_qty += buy_qty
        total_sell_qty += sell_qty

    if total_trades == 0:
        raise typer.Exit(code=0)

    if include_fees:
        total_net = total_realized - total_fees
        typer.echo(
            f"\nTOTAL trades={total_trades} matched_qty={_format_amount(total_matched_qty, 8)} "
            f"realized={_format_usdt(total_realized)} fees={_format_usdt(total_fees)} "
            f"net={_format_usdt(total_net)} unmatched_sells={_format_amount(total_unmatched_sells, 8)}"
        )
    else:
        typer.echo(
            f"\nTOTAL trades={total_trades} matched_qty={_format_amount(total_matched_qty, 8)} "
            f"realized={_format_usdt(total_realized)} unmatched_sells={_format_amount(total_unmatched_sells, 8)}"
        )


@app.command("holdings-summary")
def holdings_summary(
    include_locked: bool = typer.Option(True, "--include-locked/--free-only"),
    top: int = typer.Option(10, "--top", help="Show top N holdings by value (0 shows all)."),
    db_path: Optional[Path] = typer.Option(
        None, "--db-path", help="Override snapshot DB path."
    ),
) -> None:
    """Summarize current holdings and compare against the previous day snapshot."""
    try:
        account = binance_wrapper.get_account_value_usdt(include_locked=include_locked)
    except RuntimeError as exc:
        _handle_cli_error(exc)

    total_usdt = float(account.get("total_usdt", 0.0))
    assets = account.get("assets", [])
    if not isinstance(assets, list):
        assets = []

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    db_path = db_path or DEFAULT_DB_PATH

    prev_snapshot = load_latest_snapshot(
        db_path=db_path,
        before_ts_ms=int(today_start.timestamp() * 1000),
    )
    comparison_label = "previous day snapshot"
    if prev_snapshot is None:
        prev_snapshot = load_latest_snapshot(db_path=db_path)
        comparison_label = "previous snapshot (same-day)" if prev_snapshot else "no previous snapshot"

    current_snapshot = record_snapshot(
        total_usdt=total_usdt,
        assets=assets,
        db_path=db_path,
    )

    typer.echo(f"Holdings Snapshot: {current_snapshot.ts_iso}")
    typer.echo(f"- total_usdt: {_format_usdt(total_usdt)}")
    typer.echo(f"- db_path: {db_path}")

    if prev_snapshot is None:
        typer.echo("No previous snapshot available to compare.")
    else:
        delta_usdt = total_usdt - prev_snapshot.total_usdt
        pct = (delta_usdt / prev_snapshot.total_usdt * 100.0) if prev_snapshot.total_usdt else 0.0
        typer.echo(f"Comparison: {comparison_label} ({prev_snapshot.ts_iso})")
        typer.echo(f"- prev_total_usdt: {_format_usdt(prev_snapshot.total_usdt)}")
        typer.echo(f"- delta_usdt: {_format_delta(delta_usdt)} ({pct:+.2f}%)")

        prev_assets = {asset.asset: asset for asset in prev_snapshot.assets}
        current_assets = {asset.get("asset", "").upper(): asset for asset in assets if isinstance(asset, dict)}
        combined_keys = sorted(set(prev_assets) | set(current_assets))

        deltas = []
        for key in combined_keys:
            prev = prev_assets.get(key)
            curr = current_assets.get(key, {})
            curr_amount = float(curr.get("amount", 0.0)) if curr else 0.0
            curr_value = float(curr.get("value_usdt", 0.0)) if curr else 0.0
            prev_amount = prev.amount if prev else 0.0
            prev_value = prev.value_usdt if prev else 0.0
            delta_amount = curr_amount - prev_amount
            delta_value = curr_value - prev_value
            deltas.append((key, delta_value, delta_amount, curr_value, curr_amount))

        deltas.sort(key=lambda item: abs(item[1]), reverse=True)
        if deltas:
            typer.echo("Asset deltas (sorted by abs value change):")
            for asset, delta_value, delta_amount, curr_value, curr_amount in deltas:
                typer.echo(
                    f"- {asset} delta_value={_format_delta(delta_value)} "
                    f"delta_amount={_format_amount(delta_amount, 8)} "
                    f"current_value={_format_usdt(curr_value)} "
                    f"current_amount={_format_amount(curr_amount, 8)}"
                )

    holdings = sorted(
        [
            (
                entry.get("asset", "n/a"),
                float(entry.get("amount", 0.0)),
                float(entry.get("price_usdt", 0.0)),
                float(entry.get("value_usdt", 0.0)),
            )
            for entry in assets
            if isinstance(entry, dict)
        ],
        key=lambda item: item[3],
        reverse=True,
    )
    if holdings:
        if top > 0:
            holdings = holdings[: int(top)]
        typer.echo("Holdings (by value):")
        for asset, amount, price, value in holdings:
            typer.echo(
                f"- {asset} amount={_format_amount(amount, 8)} "
                f"price={_format_amount(price, 6)} value={_format_usdt(value)}"
            )

@app.command("buy-btc")
def buy_btc(
    usdt_amount: float = typer.Argument(..., help="USDT amount to spend buying BTC."),
    min_notional: Optional[float] = typer.Option(
        None, "--min-notional", help="Override minimum notional check (USDT)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run/--live", help="Use Binance test order."),
) -> None:
    """Market buy BTC with a USDT amount."""
    try:
        order = binance_wrapper.buy_usdt_to_btc(
            usdt_amount,
            min_notional_override=min_notional,
            dry_run=dry_run,
        )
    except (RuntimeError, ValueError) as exc:
        _handle_cli_error(exc)
    typer.echo("Order response:")
    typer.echo(str(order))


if __name__ == "__main__":
    app()
