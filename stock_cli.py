from __future__ import annotations

import math
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    record_portfolio_snapshot,
)
from src.leverage_settings import get_leverage_settings
from src.symbol_utils import is_crypto_symbol
from src.trading_obj_utils import filter_to_realistic_positions
from stock.state import get_state_dir, get_state_file, resolve_state_suffix, get_paper_suffix
from stock.state_utils import StateLoadError, collect_probe_statuses, render_ascii_line

MAX_RISK_AXIS_LIMIT = 1.6
STATE_SUFFIX = resolve_state_suffix()
PAPER_SUFFIX = get_paper_suffix()
ACTIVE_TRADES_PATH = get_state_file("active_trades", STATE_SUFFIX)
MAXDIFF_WATCHERS_DIR = get_state_dir() / f"maxdiff_watchers{PAPER_SUFFIX}{STATE_SUFFIX or ''}"

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


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


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


def _estimate_live_portfolio_value(account, positions: Sequence) -> Optional[float]:
    equity = _optional_float(getattr(account, "equity", None)) if account is not None else None
    if equity and equity > 0:
        return equity

    total_market_value = 0.0
    for position in positions:
        total_market_value += _safe_float(getattr(position, "market_value", 0.0))

    cash = _optional_float(getattr(account, "cash", None)) if account is not None else None
    if cash is not None:
        estimated_value = total_market_value + cash
    else:
        estimated_value = total_market_value

    if estimated_value != 0.0:
        return estimated_value

    return None


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_price(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    precision = 4 if abs(numeric) < 1 else 2
    return f"{numeric:.{precision}f}"


def _format_quantity(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    formatted = f"{numeric:.6f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def _normalize_symbols_filter(symbols: Optional[List[str]]) -> Optional[set[str]]:
    """Normalize CLI symbols option (comma or space separated) to uppercase set."""
    if not symbols:
        return None
    normalized: set[str] = set()
    for entry in symbols:
        if not entry:
            continue
        for token in entry.replace(",", " ").split():
            token = token.strip().upper()
            if token:
                normalized.add(token)
    return normalized or None


def _coerce_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


STRATEGY_PROFIT_FIELDS = (
    ("entry", "entry_takeprofit_profit"),
    ("maxdiff", "maxdiffprofit_profit"),
    ("takeprofit", "takeprofit_profit"),
)

ENTRY_STRATEGY_PROFIT_LOOKUP = {
    "maxdiff": "maxdiffprofit_profit",
    "highlow": "maxdiffprofit_profit",
    "entry": "entry_takeprofit_profit",
    "entry_takeprofit": "entry_takeprofit_profit",
    "simple": "entry_takeprofit_profit",
    "all_signals": "entry_takeprofit_profit",
    "takeprofit": "takeprofit_profit",
}


def _format_strategy_profit_summary(entry_strategy: Optional[str], forecast: Dict[str, object]) -> Optional[str]:
    if not forecast:
        return None
    normalized_strategy = (entry_strategy or "").strip().lower()
    selected_key = ENTRY_STRATEGY_PROFIT_LOOKUP.get(normalized_strategy)
    entries = []
    for label, key in STRATEGY_PROFIT_FIELDS:
        value = _coerce_optional_float(forecast.get(key))
        if value is None:
            continue
        formatted = f"{value:.4f}"
        if key == selected_key:
            formatted = f"{formatted}*"
        entries.append(f"{label}={formatted}")
    if not entries:
        return None
    return f"profits {' '.join(entries)}"


def _format_timedelta(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    if total_seconds < 60:
        return f"{total_seconds}s"
    if total_seconds < 3600:
        minutes, seconds = divmod(total_seconds, 60)
        if seconds and minutes < 10:
            return f"{minutes}m{seconds}s"
        return f"{minutes}m"
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    if minutes == 0:
        return f"{hours}h"
    return f"{hours}h{minutes}m"


def _format_since(timestamp: Optional[str]) -> str:
    parsed = _parse_iso_timestamp(timestamp)
    if parsed is None:
        return "n/a"
    delta = datetime.now(timezone.utc) - parsed
    return f"{_format_timedelta(delta)} ago"


def _is_pid_alive(pid: Optional[int]) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def _load_json_data(path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except Exception as exc:
        typer.secho(f"  Failed to read {path}: {exc}", err=True, fg=typer.colors.YELLOW)
        return None


def _load_active_trading_plan() -> List[Dict]:
    data = _load_json_data(ACTIVE_TRADES_PATH)
    if not data:
        return []
    entries: List[Dict] = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        symbol, side = (key.split("|", 1) + ["n/a"])[:2]
        entry = dict(value)
        entry["symbol"] = symbol
        entry["side"] = side
        entries.append(entry)
    entries.sort(key=lambda item: (item.get("symbol", ""), item.get("side", "")))
    return entries


def _load_maxdiff_watchers() -> List[Dict]:
    if not MAXDIFF_WATCHERS_DIR.exists():
        return []
    watchers: List[Dict] = []
    for path in sorted(MAXDIFF_WATCHERS_DIR.glob("*.json")):
        data = _load_json_data(path)
        if not isinstance(data, dict):
            continue
        data["config_path"] = str(path)
        pid = data.get("pid")
        data["process_alive"] = _is_pid_alive(pid)
        watchers.append(data)
    return watchers


def _select_watchers(watchers: List[Dict], symbol: str, side: str, mode: str) -> List[Dict]:
    return [
        watcher
        for watcher in watchers
        if watcher.get("symbol") == symbol and watcher.get("side") == side and watcher.get("mode") == mode
    ]


def _is_watcher_expired(watcher: Dict) -> bool:
    """Check if a watcher has expired."""
    expiry_at = watcher.get("expiry_at")
    expiry_ts = _parse_iso_timestamp(expiry_at)
    if not expiry_ts:
        return False
    remaining = expiry_ts - datetime.now(timezone.utc)
    return remaining.total_seconds() <= 0


def _is_watcher_inactive(watcher: Dict) -> bool:
    """Check if a watcher is inactive (has pid but process not running)."""
    pid = watcher.get("pid")
    if not pid:
        return False
    return not watcher.get("process_alive", False)


def _format_watcher_summary(watcher: Dict) -> str:
    mode = watcher.get("mode", "watcher")
    side = watcher.get("side", "?")
    entry_strategy = watcher.get("entry_strategy")
    if entry_strategy:
        parts = [f"{mode} watcher [{side}] strategy={entry_strategy}"]
    else:
        parts = [f"{mode} watcher [{side}]"]
    state = watcher.get("state")
    if state:
        parts.append(f"state={state}")
    if watcher.get("process_alive"):
        parts.append(f"pid={watcher.get('pid')}")
    elif watcher.get("pid"):
        parts.append("inactive")
    limit_price = watcher.get("limit_price")
    if limit_price is not None:
        parts.append(f"limit={_format_price(limit_price)}")
    takeprofit_price = watcher.get("takeprofit_price")
    if takeprofit_price is not None:
        parts.append(f"tp={_format_price(takeprofit_price)}")
    tolerance_pct = watcher.get("tolerance_pct")
    if tolerance_pct is not None:
        try:
            parts.append(f"tol={float(tolerance_pct) * 100:.2f}%")
        except (TypeError, ValueError):
            pass
    price_tolerance = watcher.get("price_tolerance")
    if price_tolerance is not None and tolerance_pct is None:
        try:
            parts.append(f"tol={float(price_tolerance) * 100:.2f}%")
        except (TypeError, ValueError):
            pass
    qty = watcher.get("target_qty")
    if qty is not None:
        parts.append(f"qty={_format_quantity(qty)}")
    open_orders = watcher.get("open_order_count")
    if open_orders is not None:
        parts.append(f"orders={open_orders}")
    last_reference = watcher.get("last_reference_price")
    if last_reference is not None:
        parts.append(f"ref={_format_price(last_reference)}")
    last_update = watcher.get("last_update")
    if last_update:
        parts.append(f"updated {_format_since(last_update)}")
    expiry_at = watcher.get("expiry_at")
    expiry_ts = _parse_iso_timestamp(expiry_at)
    if expiry_ts:
        remaining = expiry_ts - datetime.now(timezone.utc)
        if remaining.total_seconds() > 0:
            parts.append(f"expires in {_format_timedelta(remaining)}")
        else:
            parts.append("expired")
    return " | ".join(parts)


def _fetch_forecast_snapshot() -> tuple[Dict[str, Dict], Optional[str]]:
    try:
        from trade_stock_e2e import _load_latest_forecast_snapshot  # type: ignore

        return _load_latest_forecast_snapshot(), None
    except Exception as exc:
        return {}, str(exc)


@app.command()
def status(
    timezone_name: str = typer.Option("US/Eastern", "--tz", help="Timezone for timestamp display."),
    symbols: Optional[List[str]] = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Only show data for specified symbols (comma- or space-separated, can repeat).",
    ),
    maxdiff_only: bool = typer.Option(
        False,
        "--maxdiff",
        help="Show only trades/watchers using MaxDiff-based entry strategies.",
    ),
    positions_only: bool = typer.Option(
        False,
        "--positions",
        help="Show only positions (skip orders and trading plan/watchers).",
    ),
    max_orders: int = typer.Option(20, help="Maximum number of open orders to display."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all details including expired watchers."),
):
    """Show live account, position, and risk metadata."""
    typer.echo("== Portfolio Status ==")

    symbol_filter = _normalize_symbols_filter(symbols)
    if symbol_filter:
        typer.echo(f"  Filtering to symbols: {', '.join(sorted(symbol_filter))}")

    leverage_settings = get_leverage_settings()

    # Global risk snapshot
    live_portfolio_value: Optional[float] = None
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

    positions_all = positions
    if symbol_filter:
        positions = [pos for pos in positions if getattr(pos, "symbol", "").upper() in symbol_filter]

    if positions:
        total_value = sum(_safe_float(getattr(pos, "market_value", 0.0)) for pos in positions)
        typer.echo(f"  Count: {len(positions)} | Total Market Value: {_format_currency(total_value)}")
        for line in _summarize_positions(positions, timezone_name):
            typer.echo(line)
    else:
        typer.echo("  No active positions.")

    live_portfolio_value = _estimate_live_portfolio_value(account, positions_all)

    if not positions_only:
        # Orders
        typer.echo("\n:: Open Orders")
        try:
            orders = alpaca_wrapper.get_orders()
        except Exception as exc:
            typer.secho(f"  Failed to fetch open orders: {exc}", err=True, fg=typer.colors.RED)
            orders = []

        if symbol_filter:
            orders = [order for order in orders if getattr(order, "symbol", "").upper() in symbol_filter]

        if orders:
            orders_to_show = list(orders)[:max_orders]
            typer.echo(f"  Count: {len(orders)} (showing {len(orders_to_show)})")
            for line in _summarize_orders(orders_to_show, timezone_name):
                typer.echo(line)
        else:
            typer.echo("  No open orders.")

        # Trading plan overview
        typer.echo("\n:: Trading Plan")
        trading_plan = _load_active_trading_plan()
        forecast_snapshot, forecast_error = _fetch_forecast_snapshot()
        watchers = _load_maxdiff_watchers()
        used_watcher_keys = set()
        hidden_watcher_count = 0

        if forecast_error:
            typer.secho(f"  Forecast snapshot unavailable: {forecast_error}", fg=typer.colors.YELLOW)

        # Build set of symbols with open positions
        position_symbols = set()
        if positions:
            for pos in positions:
                position_symbols.add(getattr(pos, "symbol", ""))

        if trading_plan and symbol_filter:
            trading_plan = [
                entry
                for entry in trading_plan
                if (entry.get("symbol") or "").upper() in symbol_filter
            ]

        if trading_plan and maxdiff_only:
            trading_plan = [
                entry
                for entry in trading_plan
                if (entry.get("entry_strategy") or "").lower().startswith("maxdiff")
            ]

        if trading_plan:
            for entry in trading_plan:
                symbol = entry.get("symbol", "UNKNOWN")
                side = entry.get("side", "n/a")
                strategy = entry.get("entry_strategy", "n/a")
                mode = entry.get("mode", "n/a")
                qty_repr = _format_quantity(entry.get("qty"))
                opened_repr = _format_optional_timestamp(
                    _parse_iso_timestamp(entry.get("opened_at")),
                    timezone_name,
                )
                line = (
                    f"  - {symbol} [{side}] strategy={strategy} "
                    f"mode={mode} qty={qty_repr} opened={opened_repr}"
                )
                forecast = forecast_snapshot.get(symbol, {})
                high_price = forecast.get("maxdiffprofit_high_price")
                low_price = forecast.get("maxdiffprofit_low_price")
                if high_price is not None or low_price is not None:
                    line += (
                        f" | maxdiff_high={_format_price(high_price)} "
                        f"low={_format_price(low_price)}"
                    )
                profit_summary = _format_strategy_profit_summary(strategy, forecast)
                if profit_summary:
                    line += f" | {profit_summary}"

                # Color code: gold for crypto, green for symbols with positions
                if is_crypto_symbol(symbol):
                    typer.secho(line, fg=typer.colors.YELLOW)
                elif symbol in position_symbols:
                    typer.secho(line, fg=typer.colors.GREEN)
                else:
                    typer.echo(line)

                entry_watchers = _select_watchers(watchers, symbol, side, "entry")
                exit_watchers = _select_watchers(watchers, symbol, side, "exit")
                for watcher in entry_watchers + exit_watchers:
                    key = watcher.get("config_path") or f"{symbol}|{side}|{watcher.get('mode')}"
                    used_watcher_keys.add(key)
                    if not verbose and (_is_watcher_expired(watcher) or _is_watcher_inactive(watcher)):
                        hidden_watcher_count += 1
                        continue
                    typer.echo(f"    {_format_watcher_summary(watcher)}")
        else:
            typer.echo("  No recorded active trades.")

        remaining_watchers = [
            watcher
            for watcher in watchers
            if (watcher.get("config_path") or f"{watcher.get('symbol')}|{watcher.get('side')}|{watcher.get('mode')}") not in used_watcher_keys
        ]
        if symbol_filter:
            remaining_watchers = [w for w in remaining_watchers if (w.get("symbol") or "").upper() in symbol_filter]
        if remaining_watchers:
            # Filter expired and inactive watchers if not in verbose mode
            if not verbose:
                active_remaining = [w for w in remaining_watchers if not (_is_watcher_expired(w) or _is_watcher_inactive(w))]
                hidden_watcher_count += len(remaining_watchers) - len(active_remaining)
                remaining_watchers = active_remaining

            if remaining_watchers:
                typer.echo("\n:: MaxDiff Watchers")
                for watcher in remaining_watchers:
                    symbol = watcher.get("symbol", "UNKNOWN")
                    typer.echo(f"  - {symbol} {_format_watcher_summary(watcher)}")

        # Show hidden count if any were hidden
        if not verbose and hidden_watcher_count > 0:
            typer.echo(f"\n  ({hidden_watcher_count} inactive/expired watcher{'s' if hidden_watcher_count > 1 else ''} hidden, use --verbose to show)")

    # Settings overview
    typer.echo("\n:: Settings")
    state_suffix = os.getenv("TRADE_STATE_SUFFIX", "").strip() or "<unset>"
    typer.echo(f"  TRADE_STATE_SUFFIX={state_suffix}")
    if state_suffix == "<unset>":
        typer.echo("    Using default strategy state files.")
    if risk_threshold is not None:
        typer.echo(f"  Global Risk Threshold={risk_threshold:.2f}x")
    if latest_snapshot:
        typer.echo(
            f"  Last Recorded Portfolio Value={_format_currency(latest_snapshot.portfolio_value)} "
            f"as of {_format_timestamp(latest_snapshot.observed_at, timezone_name)}"
        )
    else:
        typer.echo("  Last Recorded Portfolio Value=n/a")
    if live_portfolio_value is not None:
        typer.echo(f"  Live Portfolio Value={_format_currency(live_portfolio_value)} (account equity estimate)")


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


@app.command("set-risk")
def set_risk(
    day_pl: Optional[float] = typer.Option(
        None,
        help="Day P&L value (currently ignored - risk always set to 2.0x).",
    ),
):
    """Manually record a portfolio snapshot and update the risk threshold.

    Risk threshold is currently hardcoded to 2.0x (dynamic adjustment disabled).

    Example:
        PAPER=0 python stock_cli.py set-risk
    """
    typer.echo("== Setting Risk Threshold ==")

    # Get current account to determine portfolio value
    try:
        account = alpaca_wrapper.get_account()
    except Exception as exc:
        typer.secho(f"Failed to fetch account: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if account is None:
        typer.secho("Account unavailable.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    equity = _safe_float(getattr(account, "equity", 0.0))
    if equity <= 0:
        typer.secho("Invalid equity value.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Get current leverage settings
    leverage_settings = get_leverage_settings()
    typer.echo(f"Current max gross leverage setting: {leverage_settings.max_gross_leverage:.2f}x")
    typer.echo(f"Current portfolio equity: {_format_currency(equity)}")

    # Record snapshot
    try:
        if day_pl is not None:
            typer.echo(f"Recording snapshot with day P&L: {_format_currency(day_pl)}")
            snapshot = record_portfolio_snapshot(equity, day_pl=day_pl)
        else:
            typer.echo("Recording snapshot (no day P&L specified)")
            snapshot = record_portfolio_snapshot(equity)
    except Exception as exc:
        typer.secho(f"Failed to record snapshot: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"\n✓ Risk threshold updated to: {snapshot.risk_threshold:.2f}x")
    typer.echo(f"  Portfolio value: {_format_currency(snapshot.portfolio_value)}")
    typer.echo(f"  Recorded at: {_format_timestamp(snapshot.observed_at, 'US/Eastern')}")


if __name__ == "__main__":
    app()
