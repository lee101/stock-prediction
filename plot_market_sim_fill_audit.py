#!/usr/bin/env python3
"""Plot market-simulator fills on OHLC bars for visual execution audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_hourly_experiment.marketsimulator.portfolio_simulator import (
    PortfolioConfig,
    run_portfolio_simulation,
)


def _load_entry_actions_from_trade_log(log_path: Path, symbol: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") != "entry" or rec.get("symbol") != symbol:
            continue
        ts = pd.to_datetime(rec.get("logged_at"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        ts = ts.floor("h")
        side = str(rec.get("side") or "").lower()
        entry_price = float(rec.get("price") or 0.0)
        exit_price = float(rec.get("exit_price") or 0.0)
        signal_amount = float(rec.get("signal_amount") or 0.0)
        if signal_amount <= 0:
            signal_amount = 10.0

        if side == "short":
            buy_price = exit_price if exit_price > 0 else entry_price * 0.99
            sell_price = entry_price
            buy_amount = 0.1
            sell_amount = signal_amount
        else:
            buy_price = entry_price
            sell_price = exit_price if exit_price > 0 else entry_price * 1.01
            buy_amount = signal_amount
            sell_amount = 0.1

        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "buy_amount": buy_amount,
                "sell_amount": sell_amount,
                "trade_amount": signal_amount,
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("timestamp")
        .drop_duplicates(["timestamp", "symbol"], keep="last")
        .reset_index(drop=True)
    )


def _plot_candles_with_fills(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    symbol: str,
    bar_margin: float,
    output_png: Path,
) -> None:
    plot_df = bars.tail(120).copy().set_index("timestamp")
    x = np.arange(len(plot_df))
    opens = plot_df["open"].to_numpy(dtype=float)
    closes = plot_df["close"].to_numpy(dtype=float)
    highs = plot_df["high"].to_numpy(dtype=float)
    lows = plot_df["low"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(len(plot_df)):
        color = "#16a34a" if closes[i] >= opens[i] else "#dc2626"
        ax.vlines(i, lows[i], highs[i], color=color, linewidth=1.0, alpha=0.9)
        body_low = min(opens[i], closes[i])
        body_h = max(0.001, abs(closes[i] - opens[i]))
        ax.add_patch(
            plt.Rectangle(
                (i - 0.3, body_low),
                0.6,
                body_h,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
        )

    if not trades.empty:
        idx_map = {ts: i for i, ts in enumerate(plot_df.index)}
        for _, row in trades.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            if ts not in idx_map:
                continue
            i = idx_map[ts]
            y = float(row["price"])
            side = str(row["side"])
            if side in {"buy", "buy_cover"}:
                marker, color = "^", "#2563eb"
            else:
                marker, color = "v", "#7c3aed"
            ax.scatter(i, y, marker=marker, color=color, s=70, zorder=5)

    ax.set_title(f"{symbol} fills on OHLC bars (bar_margin={bar_margin*10000:.1f} bps)")
    ax.set_ylabel("Price")
    ax.set_xlabel("Bars (recent 120)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fill audit chart for market simulator")
    parser.add_argument("--symbol", default="MTCH")
    parser.add_argument("--bars-csv", default=None, help="Optional explicit OHLC csv path")
    parser.add_argument("--trade-log", default="strategy_state/stock_trade_log.jsonl")
    parser.add_argument("--output-dir", default="experiments/sim_execution_charts")
    parser.add_argument("--bar-margin-bps", type=float, default=5.0)
    parser.add_argument("--symbol-bar-margin-bps", type=float, default=None)
    parser.add_argument("--window-padding-hours", type=int, default=24)
    args = parser.parse_args()

    symbol = str(args.symbol).upper()
    bar_margin = float(args.bar_margin_bps) / 10000.0
    symbol_bar_margin_bps = None
    if args.symbol_bar_margin_bps is not None:
        symbol_bar_margin_bps = {symbol: float(args.symbol_bar_margin_bps)}

    bars_path = Path(args.bars_csv) if args.bars_csv else Path("trainingdatahourly/stocks") / f"{symbol}.csv"
    bars = pd.read_csv(bars_path)
    bars.columns = [c.lower() for c in bars.columns]
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["symbol"] = symbol

    actions = _load_entry_actions_from_trade_log(Path(args.trade_log), symbol)
    if actions.empty:
        raise RuntimeError(f"No entry rows found for {symbol} in {args.trade_log}")

    pad = pd.Timedelta(hours=int(args.window_padding_hours))
    start_ts = actions["timestamp"].min() - pad
    end_ts = actions["timestamp"].max() + pad
    bars = bars[(bars["timestamp"] >= start_ts) & (bars["timestamp"] <= end_ts)].copy()

    cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        min_edge=0.0,
        max_hold_hours=8,
        enforce_market_hours=False,
        close_at_eod=False,
        trade_amount_scale=100.0,
        min_buy_amount=0.0,
        decision_lag_bars=0,
        market_order_entry=False,
        bar_margin=bar_margin,
        symbol_bar_margin_bps=symbol_bar_margin_bps,
        int_qty=True,
        fee_by_symbol={symbol: 0.001},
    )
    sim = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    bars_idx = bars.set_index("timestamp")
    audits: list[dict[str, Any]] = []
    for t in sim.trades:
        ts = pd.Timestamp(t.timestamp)
        if ts not in bars_idx.index:
            continue
        row = bars_idx.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        side = str(t.side)
        px = float(t.price)
        effective_margin = (
            float(args.symbol_bar_margin_bps) / 10000.0
            if args.symbol_bar_margin_bps is not None
            else bar_margin
        )
        if side in {"buy", "buy_cover"}:
            trigger = px * (1 - effective_margin)
            touched = float(row["low"]) <= trigger
        elif side in {"sell", "short_sell"}:
            trigger = px * (1 + effective_margin)
            touched = float(row["high"]) >= trigger
        else:
            trigger = np.nan
            touched = False
        audits.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "side": side,
                "price": px,
                "bar_low": float(row["low"]),
                "bar_high": float(row["high"]),
                "trigger": float(trigger),
                "touched_under_rule": bool(touched),
                "effective_margin_bps": effective_margin * 10000.0,
            }
        )

    out_dir = Path(args.output_dir)
    suffix = f"_{symbol.lower()}_margin{int(round(bar_margin*10000))}bp"
    if args.symbol_bar_margin_bps is not None:
        suffix += f"_sym{int(round(float(args.symbol_bar_margin_bps)))}bp"
    output_png = out_dir / f"sim_fill_audit{suffix}.png"
    output_csv = out_dir / f"sim_fill_audit{suffix}.csv"
    trades_df = pd.DataFrame(audits)
    trades_df.to_csv(output_csv, index=False)
    _plot_candles_with_fills(bars, trades_df, symbol=symbol, bar_margin=bar_margin, output_png=output_png)

    touched_count = int(trades_df["touched_under_rule"].sum()) if not trades_df.empty else 0
    print(f"chart={output_png}")
    print(f"audit={output_csv}")
    print(f"trades={len(trades_df)} touched={touched_count}")


if __name__ == "__main__":
    main()
