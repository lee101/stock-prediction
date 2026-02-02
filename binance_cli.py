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
