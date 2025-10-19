from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import alpaca_wrapper
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytz
import typer

from src.portfolio_risk import (
    PortfolioSnapshotRecord,
    fetch_latest_snapshot,
    fetch_snapshots,
    get_global_risk_threshold,
    get_configured_max_risk_threshold,
)
from src.leverage_settings import get_leverage_settings
from src.trading_obj_utils import filter_to_realistic_positions
from stock.state_utils import StateLoadError, collect_probe_statuses, render_ascii_line

MAX_RISK_AXIS_LIMIT = 1.6

app = typer.Typer(help="Portfolio analytics CLI utilities.")


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_timestamp(ts: datetime, timezone_name: str) -> str:
    try:
        tz = pytz.timezone(timezone_name)
    except pytz.UnknownTimeZoneError:
        tz = pytz.UTC
    return ts.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def _format_optional_timestamp(ts: Optional[datetime], timezone_name: str) -> str:
    if ts is None:
        return "n/a"
    return _format_timestamp(ts, timezone_name)


def _summarize_positions(positions: Sequence, timezone_name: str) -> Sequence[str]:
    lines = []
    for position in positions:
        symbol = getattr(position, "symbol", "UNKNOWN")
        side = getattr(position, "side", "n/a")
        qty = getattr(position, "qty", "0")
        market_value = _safe_float(getattr(position, "market_value", 0.0))
        unrealized = _safe_float(getattr(position, "unrealized_pl", 0.0))
        current_price = _safe_float(getattr(position, "current_price", 0.0))
        last_trade_at = getattr(position, "last_trade_at", None)
        ts_repr = "n/a"
        if isinstance(last_trade_at, datetime):
            ts_repr = _format_timestamp(last_trade_at, timezone_name)
        lines.append(
            f"  - {symbol} [{side}] qty={qty} price={current_price:.2f} "
            f"value={_format_currency(market_value)} pnl={_format_currency(unrealized)} "
            f"last_trade={ts_repr}"
        )
    return lines


def _summarize_orders(orders: Sequence, timezone_name: str) -> Sequence[str]:
    lines = []
    for order in orders:
        symbol = getattr(order, "symbol", "UNKNOWN")
        side = getattr(order, "side", "n/a")
        qty = getattr(order, "qty", getattr(order, "quantity", "0"))
        limit_price = getattr(order, "limit_price", None)
        status = getattr(order, "status", "n/a")
        order_type = getattr(order, "type", getattr(order, "order_type", "n/a"))
        submitted_at = getattr(order, "submitted_at", None)
        ts_repr = "n/a"
        if isinstance(submitted_at, datetime):
            ts_repr = _format_timestamp(submitted_at, timezone_name)
        price_repr = f"@{limit_price}" if limit_price else ""
        lines.append(
            f"  - {symbol} {side} {qty} {order_type}{price_repr} status={status} submitted={ts_repr}"
        )
    return lines


@app.command()
def status(
    timezone_name: str = typer.Option("US/Eastern", "--tz", help="Timezone for timestamp display."),
    max_orders: int = typer.Option(20, help="Maximum number of open orders to display."),
):
    """Show live account, position, and risk metadata."""
    typer.echo("== Portfolio Status ==")

    leverage_settings = get_leverage_settings()

    # Global risk snapshot
    try:
        risk_threshold = get_global_risk_threshold()
    except Exception as exc:
        typer.secho(f"Failed to obtain global risk threshold: {exc}", err=True, fg=typer.colors.RED)
        risk_threshold = None

    try:
        latest_snapshot: Optional[PortfolioSnapshotRecord] = fetch_latest_snapshot()
    except Exception as exc:
        typer.secho(f"Failed to load portfolio snapshots: {exc}", err=True, fg=typer.colors.RED)
        latest_snapshot = None

    typer.echo(":: Global Risk")
    if risk_threshold is not None:
        configured_cap = get_configured_max_risk_threshold()
        typer.echo(f"  Threshold: {risk_threshold:.2f}x (cap {configured_cap:.2f}x)")
    else:
        typer.echo("  Threshold: n/a")
    if latest_snapshot:
        typer.echo(
            f"  Last Snapshot: {_format_timestamp(latest_snapshot.observed_at, timezone_name)} "
            f"({ _format_currency(latest_snapshot.portfolio_value) })"
        )
    else:
        typer.echo("  Last Snapshot: n/a")

    # Account summary
    typer.echo("\n:: Account")
    try:
        account = alpaca_wrapper.get_account()
    except Exception as exc:
        typer.secho(f"  Account fetch failed: {exc}", err=True, fg=typer.colors.RED)
        account = None

    if account is not None:
        equity = _safe_float(getattr(account, "equity", 0.0))
        cash = _safe_float(getattr(account, "cash", 0.0))
        buying_power = _safe_float(getattr(account, "buying_power", getattr(account, "buying_power", 0.0)))
        multiplier = _safe_float(getattr(account, "multiplier", 1.0), 1.0)
        last_equity = _safe_float(getattr(account, "last_equity", equity))
        day_pl = equity - last_equity
        status = getattr(account, "status", "n/a")
        typer.echo(f"  Status: {status}")
        typer.echo(f"  Equity: {_format_currency(equity)} (Δ day {_format_currency(day_pl)})")
        typer.echo(f"  Cash: {_format_currency(cash)}")
        typer.echo(f"  Buying Power: {_format_currency(buying_power)} (multiplier {multiplier:.2f}x)")
    else:
        typer.echo("  Account unavailable.")

    # Positions
    typer.echo("\n:: Positions")
    try:
        positions = alpaca_wrapper.get_all_positions()
        positions = filter_to_realistic_positions(positions)
    except Exception as exc:
        typer.secho(f"  Failed to load positions: {exc}", err=True, fg=typer.colors.RED)
        positions = []

    if positions:
        total_value = sum(_safe_float(getattr(pos, "market_value", 0.0)) for pos in positions)
        typer.echo(f"  Count: {len(positions)} | Total Market Value: {_format_currency(total_value)}")
        for line in _summarize_positions(positions, timezone_name):
            typer.echo(line)
    else:
        typer.echo("  No active positions.")

    # Orders
    typer.echo("\n:: Open Orders")
    try:
        orders = alpaca_wrapper.get_orders()
    except Exception as exc:
        typer.secho(f"  Failed to fetch open orders: {exc}", err=True, fg=typer.colors.RED)
        orders = []

    if orders:
        orders_to_show = list(orders)[:max_orders]
        typer.echo(f"  Count: {len(orders)} (showing {len(orders_to_show)})")
        for line in _summarize_orders(orders_to_show, timezone_name):
            typer.echo(line)
    else:
        typer.echo("  No open orders.")

    # Settings overview
    typer.echo("\n:: Settings")
    state_suffix = os.getenv("TRADE_STATE_SUFFIX", "").strip() or "<unset>"
    typer.echo(f"  TRADE_STATE_SUFFIX={state_suffix}")
    if risk_threshold is not None:
        typer.echo(f"  Global Risk Threshold={risk_threshold:.2f}x")
    if latest_snapshot:
        typer.echo(
            f"  Portfolio Value={_format_currency(latest_snapshot.portfolio_value)} "
            f"as of {_format_timestamp(latest_snapshot.observed_at, timezone_name)}"
        )


@app.command("plot-risk")
def plot_risk(
    output: Path = typer.Option(
        Path("portfolio_risk.png"), "--output", "-o", help="Destination for the chart image."
    ),
    limit: Optional[int] = typer.Option(None, help="Limit the number of snapshot points included."),
    timezone_name: str = typer.Option("US/Eastern", "--tz", help="Timezone for chart timestamps."),
):
    """Render a chart of portfolio value and global risk threshold over time."""
    snapshots = fetch_snapshots(limit=limit)
    if not snapshots:
        typer.echo("No portfolio snapshots available.")
        raise typer.Exit(code=1)

    try:
        tz = pytz.timezone(timezone_name)
    except pytz.UnknownTimeZoneError as exc:
        typer.echo(f"Unknown timezone '{timezone_name}': {exc}")
        raise typer.Exit(code=2) from exc

    times = [record.observed_at.astimezone(tz) for record in snapshots]
    portfolio_values = [record.portfolio_value for record in snapshots]
    risk_thresholds = [record.risk_threshold for record in snapshots]

    fig, ax_value = plt.subplots(figsize=(10, 5))
    ax_value.plot(times, portfolio_values, label="Portfolio Value", color="tab:blue")
    ax_value.set_ylabel("Portfolio Value ($)", color="tab:blue")
    ax_value.tick_params(axis="y", labelcolor="tab:blue")

    ax_risk = ax_value.twinx()
    ax_risk.plot(times, risk_thresholds, label="Risk Threshold", color="tab:red")
    ax_risk.set_ylabel("Global Risk Threshold (x)", color="tab:red")
    ax_risk.tick_params(axis="y", labelcolor="tab:red")
    ax_risk.set_ylim(0, MAX_RISK_AXIS_LIMIT)

    locator = mdates.AutoDateLocator()
    ax_value.xaxis.set_major_locator(locator)
    ax_value.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax_value.set_xlabel(f"Timestamp ({timezone_name})")

    fig.tight_layout()
    output_path = output.expanduser().resolve()
    fig.savefig(output_path)
    plt.close(fig)

    typer.echo(f"Saved portfolio risk chart to {output_path}")


@app.command("risk-text")
def risk_text(
    limit: Optional[int] = typer.Option(
        90,
        help="Number of portfolio snapshots to include (default 90).",
    ),
    width: int = typer.Option(60, help="Width of the ASCII graph."),
):
    """Render recent portfolio value history as an ASCII graph."""
    snapshots = fetch_snapshots(limit=limit)
    if not snapshots:
        typer.echo("No portfolio snapshots available.")
        raise typer.Exit(code=1)

    values = [record.portfolio_value for record in snapshots]
    ascii_lines = render_ascii_line(values, width=width)
    typer.echo("== Portfolio Value (ASCII) ==")
    for line in ascii_lines:
        typer.echo(line)

    min_value = min(values)
    max_value = max(values)
    latest = snapshots[-1]
    typer.echo(
        f"Min={_format_currency(min_value)}  Max={_format_currency(max_value)}  "
        f"Latest={_format_currency(latest.portfolio_value)} at {_format_timestamp(latest.observed_at, 'US/Eastern')}"
    )


@app.command("probe-status")
def probe_status(
    timezone_name: str = typer.Option(
        "US/Eastern",
        "--tz",
        help="Timezone for probe timestamps.",
    ),
    suffix: Optional[str] = typer.Option(
        None,
        help="Override the trade state suffix to inspect.",
    ),
):
    """Display the current probe and learning states tracked by the trading bot."""
    typer.echo("== Probe Status ==")
    try:
        statuses = collect_probe_statuses(suffix)
    except StateLoadError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if not statuses:
        typer.echo("No recorded probe state found.")
        raise typer.Exit()

    for status in statuses:
        last_closed = _format_optional_timestamp(status.last_closed_at, timezone_name)
        active_opened = _format_optional_timestamp(status.active_opened_at, timezone_name)
        learning_updated = _format_optional_timestamp(status.learning_updated_at, timezone_name)
        pnl_repr = "n/a" if status.last_pnl is None else _format_currency(status.last_pnl)
        qty_repr = f"{status.active_qty:.4f}" if status.active_qty is not None else "n/a"

        typer.echo(
            f"- {status.symbol} [{status.side}] "
            f"pending={status.pending_probe} active={status.probe_active} "
            f"last_pnl={pnl_repr} reason={status.last_reason or 'n/a'}"
        )
        typer.echo(f"    last_closed={last_closed} active_mode={status.active_mode or 'n/a'}")
        typer.echo(f"    active_qty={qty_repr} opened={active_opened}")
        typer.echo(f"    learning_updated={learning_updated}")


if __name__ == "__main__":
    app()
