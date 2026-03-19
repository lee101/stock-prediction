"""Summarize recent realized PnL from Alpaca fills.

Usage (paper account by default):

    PAPER=1 PYTHONPATH=. python scripts/trade_pnls.py

The script fetches the most recent day with any FILL activities, groups fills
into round trips, prints a concise summary, and writes a bar plot to the
`results/` directory (ignored by git).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import typer

import alpaca_wrapper
from src.trade_pnl_analyzer import (
    FillEvent,
    TradePnL,
    compute_round_trips,
    normalize_fill_activities,
)


app = typer.Typer(add_completion=False)


def _as_list(response) -> List[dict]:
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        if "activities" in response and isinstance(response["activities"], list):
            return response["activities"]
        if "data" in response and isinstance(response["data"], list):
            return response["data"]
    return []


def fetch_fills_window(max_days_back: int = 7) -> Tuple[List[FillEvent], datetime | None]:
    """Fetch fills using both account activities and filled orders."""

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_days_back)
    future_allowance = timedelta(days=1)

    all_fills: List[FillEvent] = []
    latest_day_with_fills: datetime | None = None

    # Try account activities (may return only the most recent page)
    resp = alpaca_wrapper.get_account_activities(
        alpaca_wrapper.alpaca_api,
        activity_types="FILL",
        direction="desc",
        page_size=100,
        page_token=None,
    )
    activity_fills = normalize_fill_activities(_as_list(resp))
    all_fills.extend(
        f for f in activity_fills if f.transacted_at >= cutoff - future_allowance
    )

    # Also pull closed orders (covers cases where activities are truncated)
    try:
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(
            status="all",
            direction="asc",
            limit=500,
            after=cutoff,
        )
        orders = alpaca_wrapper.alpaca_api.get_orders(filter=req)
    except Exception:
        orders = []

    for order in orders or []:
        try:
            filled_at = getattr(order, "filled_at", None)
            if not filled_at:
                continue
            ts = datetime.fromisoformat(str(filled_at).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts = ts.astimezone(timezone.utc)
            if ts < cutoff - future_allowance:
                continue
            qty = float(getattr(order, "filled_qty", 0.0) or 0.0)
            price = float(getattr(order, "filled_avg_price", 0.0) or 0.0)
            fee = float(getattr(order, "fee", 0.0) or 0.0)
            raw_side = getattr(order, "side", "")
            side = getattr(raw_side, "value", str(raw_side)).lower()
            symbol = str(getattr(order, "symbol", "")).upper()
            if qty > 0 and price > 0 and side in {"buy", "sell"} and symbol:
                all_fills.append(
                    FillEvent(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        price=price,
                        fee=fee,
                        transacted_at=ts,
                    )
                )
        except Exception:
            continue

    if all_fills:
        latest_day_with_fills = datetime.combine(
            max(f.transacted_at for f in all_fills).date(), datetime.min.time(), tzinfo=timezone.utc
        )

    all_fills.sort(key=lambda f: f.transacted_at)
    return all_fills, latest_day_with_fills


def _format_money(value: float) -> str:
    sign = "+" if value > 0 else ""  # negative already has '-'
    return f"{sign}${value:,.2f}"


def _print_summary(trades: Iterable[TradePnL]) -> None:
    trades_list = list(trades)
    trades_list.sort(key=lambda t: t.closed_at)

    symbol_totals = defaultdict(float)
    total = 0.0
    for trade in trades_list:
        symbol_totals[trade.symbol] += trade.net
        total += trade.net

    typer.echo("\nClosed trades:")
    for trade in trades_list:
        duration_min = trade.duration_seconds / 60 if trade.duration_seconds else 0.0
        typer.echo(
            f"- {trade.closed_at.strftime('%Y-%m-%d %H:%M')} {trade.symbol} "
            f"{trade.direction.upper()} qty={trade.quantity:.4g} "
            f"entry={trade.entry_price:.4f} exit={trade.exit_price:.4f} "
            f"fees={_format_money(trade.fees)} net={_format_money(trade.net)} "
            f"hold={duration_min:.1f}m"
        )

    typer.echo("\nBy symbol:")
    for symbol, net in sorted(symbol_totals.items(), key=lambda kv: kv[0]):
        typer.echo(f"- {symbol}: {_format_money(net)}")

    typer.echo(f"\nTotal: {_format_money(total)}\n")


def _print_open_positions() -> None:
    try:
        positions = alpaca_wrapper.alpaca_api.get_all_positions()
    except Exception as exc:
        typer.echo(f"\nOpen positions: unavailable ({exc})")
        return

    if not positions:
        typer.echo("\nOpen positions: none")
        return

    typer.echo("\nOpen positions:")
    for pos in positions:
        try:
            symbol = pos.symbol
            qty = float(getattr(pos, "qty", 0.0) or 0.0)
            avg = float(getattr(pos, "avg_entry_price", 0.0) or 0.0)
            market = float(getattr(pos, "current_price", 0.0) or 0.0)
            side = "LONG" if qty > 0 else "SHORT"
            unreal = (market - avg) * qty if qty else 0.0
            typer.echo(
                f"- {symbol} {side} qty={qty:.4g} avg={avg:.4f} last={market:.4f} "
                f"unrealized={_format_money(unreal)}"
            )
        except Exception:
            continue


def _plot_trades(trades: List[TradePnL], plot_path: Path) -> None:
    if not trades:
        return
    # Use Agg backend for headless environments
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trades_sorted = sorted(trades, key=lambda t: t.closed_at)
    labels = [f"{t.symbol}\n{t.closed_at.strftime('%H:%M')}" for t in trades_sorted]
    values = [t.net for t in trades_sorted]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(trades_sorted) * 0.9), 4))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.axhline(0, color="#444", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Net PnL ($)")
    ax.set_title("Closed trade PnL (latest day)")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.02 * max(values + [1])) if value >= 0 else value - (0.02 * max(values + [1])),
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def _print_account_change() -> None:
    try:
        account = alpaca_wrapper.get_account(use_cache=False)
    except Exception as exc:
        typer.echo(f"Account change: unavailable ({exc})")
        return
    try:
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        last_equity = float(getattr(account, "last_equity", equity) or equity)
        change = equity - last_equity
        pct = (change / last_equity * 100) if last_equity else 0.0
        typer.echo(
            f"Account day change (equity vs last close): {_format_money(change)} "
            f"({pct:+.2f}%)  equity={_format_money(equity)} last={_format_money(last_equity)}"
        )
    except Exception as exc:
        typer.echo(f"Account change: unavailable ({exc})")


@app.command()
def main(max_days: int = typer.Option(7, help="Look back this many days for fills")) -> None:
    fills, latest_day = fetch_fills_window(max_days_back=max_days)
    if not fills or latest_day is None:
        typer.echo(f"No filled orders found in the last {max_days} day(s).")
        raise typer.Exit(code=1)

    trades_all = compute_round_trips(fills)
    target_date = latest_day.date()
    trades = [t for t in trades_all if t.closed_at.date() == target_date]

    if not trades:
        typer.echo(
            f"No round-trip trades closed on {target_date}. Showing zeroed summary; "
            "increase --max-days if openings precede this window."
        )

    env_desc = "PAPER" if getattr(alpaca_wrapper, "_IS_PAPER", False) else "LIVE"
    day_label = target_date.strftime("%Y-%m-%d")
    typer.echo(f"Analyzing fills for {env_desc} account on {day_label} (trades={len(trades)})")

    _print_account_change()
    _print_summary(trades)
    _print_open_positions()

    plot_name = f"trade_pnls_{day_label}.png"
    plot_path = Path("results") / plot_name
    _plot_trades(trades, plot_path)
    typer.echo(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    app()
