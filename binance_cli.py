from __future__ import annotations

import math
from typing import Iterable, List, Optional

import typer

from src.binan import binance_wrapper

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
