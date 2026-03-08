#!/usr/bin/env python3
"""Replay logged live stock entries against the portfolio simulator.

This evaluates how well simulator entry timing/counts match live trade-log
entries when actions are sparse (only hours where live attempted entries).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newnanoalpacahourlyexp.marketsimulator import HourlyTraderMarketSimulator, HourlyTraderSimulationConfig
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in parse_csv_list(value)]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in parse_csv_list(value)]


def parse_bool_list(value: str) -> list[bool]:
    out: list[bool] = []
    for token in parse_csv_list(value):
        raw = token.strip().lower()
        if raw in {"1", "true", "t", "yes", "y", "on"}:
            out.append(True)
            continue
        if raw in {"0", "false", "f", "no", "n", "off"}:
            out.append(False)
            continue
        raise ValueError(f"Invalid boolean token: {token!r}")
    return out


def _as_utc(ts_value: Any) -> pd.Timestamp:
    return pd.to_datetime(ts_value, utc=True)


def load_live_entries(
    *,
    trade_log: Path,
    symbols: set[str] | None,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with trade_log.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "entry":
                continue
            symbol = str(payload.get("symbol", "")).upper()
            if not symbol:
                continue
            if symbols and symbol not in symbols:
                continue

            ts = _as_utc(payload.get("logged_at"))
            if pd.isna(ts):
                continue
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue

            side = str(payload.get("side", "")).lower().strip()
            if side not in {"long", "short"}:
                continue
            try:
                entry_price = float(payload.get("price", 0.0))
                exit_price = float(payload.get("exit_price", 0.0))
            except (TypeError, ValueError):
                continue
            if entry_price <= 0 or exit_price <= 0:
                continue

            intensity = payload.get("intensity", 1.0)
            try:
                intensity = float(intensity)
            except (TypeError, ValueError):
                intensity = 1.0
            amount = min(max(intensity * 100.0, 0.0), 100.0)

            if side == "long":
                buy_price, sell_price = entry_price, exit_price
                buy_amount, sell_amount = amount, 0.0
            else:
                # Short entries in live use sell-to-open at `price` and buy-to-cover at `exit_price`.
                buy_price, sell_price = exit_price, entry_price
                buy_amount, sell_amount = 0.0, amount

            qty_raw = payload.get("qty", 0.0)
            try:
                qty = float(qty_raw)
            except (TypeError, ValueError):
                qty = 0.0

            rows.append(
                {
                    "timestamp": ts.floor("h"),
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "qty": qty,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "buy_amount": buy_amount,
                    "sell_amount": sell_amount,
                    "trade_amount": amount,
                    "logged_at": ts,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "side",
                "entry_price",
                "qty",
                "buy_price",
                "sell_price",
                "buy_amount",
                "sell_amount",
                "trade_amount",
                "logged_at",
            ]
        )

    raw = pd.DataFrame(rows).sort_values("logged_at")
    # Simulator action frame is one row per (timestamp, symbol). Keep latest live action in that hour.
    actions = raw.drop_duplicates(subset=["timestamp", "symbol"], keep="last").copy()
    actions = actions.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return actions


def load_live_entry_fills(
    *,
    event_log: Path,
    symbols: set[str] | None,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    if not event_log.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "entry_price", "logged_at", "order_id"])

    entry_orders: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    with event_log.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = str(payload.get("event_type", "")).strip().lower()
            if event_type == "entry_order_submit_succeeded":
                order_id = str(payload.get("order_id", "")).strip()
                symbol = str(payload.get("symbol", "")).upper()
                side = str(payload.get("side", "")).strip().lower()
                if not order_id or not symbol or side not in {"long", "short"}:
                    continue
                entry_orders[order_id] = {"symbol": symbol, "side": side}
                continue

            if event_type != "broker_closed_order":
                continue

            order = payload.get("order") or {}
            if not isinstance(order, dict):
                continue
            order_id = str(order.get("id", "")).strip()
            if not order_id or order_id not in entry_orders:
                continue

            symbol = str(order.get("symbol", entry_orders[order_id]["symbol"])).upper()
            if not symbol:
                continue
            if symbols and symbol not in symbols:
                continue

            event_ts = _as_utc(
                payload.get("event_ts")
                or order.get("filled_at")
                or order.get("updated_at")
                or payload.get("logged_at")
            )
            if pd.isna(event_ts):
                continue
            if start is not None and event_ts < start:
                continue
            if end is not None and event_ts > end:
                continue

            try:
                qty = float(order.get("filled_qty", 0.0) or 0.0)
                entry_price = float(order.get("filled_avg_price", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if qty <= 0 or entry_price <= 0:
                continue

            rows.append(
                {
                    "timestamp": event_ts.floor("h"),
                    "symbol": symbol,
                    "side": entry_orders[order_id]["side"],
                    "qty": qty,
                    "entry_price": entry_price,
                    "logged_at": event_ts,
                    "order_id": order_id,
                    "order_status": str(order.get("status", "")).lower(),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "entry_price", "logged_at", "order_id"])

    out = pd.DataFrame(rows).sort_values("logged_at")
    out = out.drop_duplicates(subset=["order_id"], keep="last").reset_index(drop=True)
    return out


def load_bars_for_symbols(
    *,
    data_root: Path,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol in symbols:
        csv_path = data_root / f"{symbol}.csv"
        if not csv_path.exists():
            logger.warning("Missing bar file for {}", symbol)
            continue
        frame = pd.read_csv(csv_path)
        if "timestamp" not in frame.columns:
            logger.warning("Skipping {}: missing timestamp column", csv_path)
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = symbol
        frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].copy()
        if frame.empty:
            continue
        parts.append(frame)
    if not parts:
        raise RuntimeError("No bar data loaded for selected symbols/window")
    bars = pd.concat(parts, ignore_index=True)
    bars = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return bars


def _entry_counts(df: pd.DataFrame, *, side_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["hour", "symbol", "side", "count"])
    out = df.copy()
    out["hour"] = pd.to_datetime(out["timestamp"], utc=True).dt.floor("h")
    out["side"] = out[side_col].astype(str).str.lower()
    grouped = (
        out.groupby(["hour", "symbol", "side"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["hour", "symbol", "side"])
        .reset_index(drop=True)
    )
    return grouped


def _entry_summary(
    df: pd.DataFrame,
    *,
    side_col: str,
    price_col: str | None = None,
    qty_col: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["hour", "symbol", "side", "count", "qty_total", "avg_price"])

    out = df.copy()
    out["hour"] = pd.to_datetime(out["timestamp"], utc=True).dt.floor("h")
    out["side"] = out[side_col].astype(str).str.lower()
    if qty_col and qty_col in out.columns:
        out["qty_value"] = pd.to_numeric(out[qty_col], errors="coerce").fillna(0.0)
    else:
        out["qty_value"] = 0.0
    if price_col and price_col in out.columns:
        out["price_value"] = pd.to_numeric(out[price_col], errors="coerce").fillna(0.0)
    else:
        out["price_value"] = 0.0
    out["notional_value"] = out["qty_value"] * out["price_value"]

    grouped = (
        out.groupby(["hour", "symbol", "side"], as_index=False)
        .agg(
            count=("side", "size"),
            qty_total=("qty_value", "sum"),
            notional_total=("notional_value", "sum"),
        )
        .sort_values(["hour", "symbol", "side"])
        .reset_index(drop=True)
    )
    grouped["avg_price"] = grouped["notional_total"] / grouped["qty_total"].where(grouped["qty_total"] > 0, 1.0)
    grouped.loc[grouped["qty_total"] <= 0, "avg_price"] = 0.0
    return grouped[["hour", "symbol", "side", "count", "qty_total", "avg_price"]]


def run_replay(
    *,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    symbols: list[str],
    initial_cash: float,
    max_positions: int,
    max_hold_hours: int,
    min_edge: float,
    fee_rate: float,
    leverage: float,
    decision_lag_bars: int,
    bar_margin: float,
    entry_order_ttl_hours: int,
    market_order_entry: bool,
    sim_backend: str,
    cancel_ack_delay_bars: int = 1,
    partial_fill_on_touch: bool = False,
) -> dict[str, Any]:
    if str(sim_backend).strip().lower() == "hourly_trader":
        short_symbols = set()
        if "side" in actions.columns:
            short_symbols = {
                str(symbol).upper()
                for symbol in actions.loc[actions["side"].astype(str).str.lower() == "short", "symbol"].tolist()
            }
        long_symbols = [s for s in symbols if s not in short_symbols]
        hourly_cfg = HourlyTraderSimulationConfig(
            initial_cash=float(initial_cash),
            allocation_usd=None,
            allocation_pct=(float(leverage) / float(max_positions)) if int(max_positions) > 0 else None,
            allocation_mode="per_symbol",
            max_leverage=float(leverage),
            intensity_scale=1.0,
            fill_buffer_bps=float(bar_margin) * 10_000.0,
            decision_lag_bars=int(decision_lag_bars),
            cancel_ack_delay_bars=int(cancel_ack_delay_bars),
            partial_fill_on_touch=bool(partial_fill_on_touch),
            enforce_market_hours=True,
            fee_by_symbol={s: float(fee_rate) for s in symbols},
            allow_short=bool(short_symbols),
            long_only_symbols=long_symbols,
            short_only_symbols=sorted(short_symbols),
            allow_position_adds=False,
            always_full_exit=True,
            symbols=symbols,
        )
        sim = HourlyTraderMarketSimulator(hourly_cfg).run(
            bars.copy(),
            actions.drop(columns=["logged_at"], errors="ignore").copy(),
        )
        sim_entries_df = pd.DataFrame(
            [
                {
                    "timestamp": fill.timestamp,
                    "symbol": fill.symbol,
                    "side": ("long" if fill.side == "buy" else "short"),
                    "qty": float(fill.quantity),
                    "entry_price": float(fill.price),
                }
                for fill in sim.fills
                if fill.kind == "entry"
            ]
        )
        sim_counts = _entry_counts(sim_entries_df, side_col="side")
        return {"sim": sim, "sim_counts": sim_counts, "sim_entries": sim_entries_df}

    cfg = PortfolioConfig(
        initial_cash=float(initial_cash),
        max_positions=int(max_positions),
        min_edge=float(min_edge),
        max_hold_hours=int(max_hold_hours),
        enforce_market_hours=True,
        close_at_eod=True,
        symbols=symbols,
        trade_amount_scale=100.0,
        decision_lag_bars=int(decision_lag_bars),
        market_order_entry=bool(market_order_entry),
        bar_margin=float(bar_margin),
        entry_order_ttl_hours=int(entry_order_ttl_hours),
        max_leverage=float(leverage),
        force_close_slippage=0.003,
        int_qty=True,
        fee_by_symbol={s: float(fee_rate) for s in symbols},
        sim_backend=sim_backend,
    )
    sim = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    sim_entries = [
        {
            "timestamp": t.timestamp,
            "symbol": t.symbol,
            "side": ("long" if t.side == "buy" else "short"),
            "qty": float(t.quantity),
            "entry_price": float(t.price),
        }
        for t in sim.trades
        if t.side in {"buy", "short_sell"}
    ]
    sim_entries_df = pd.DataFrame(sim_entries)
    sim_counts = _entry_counts(sim_entries_df, side_col="side")
    return {"sim": sim, "sim_counts": sim_counts, "sim_entries": sim_entries_df}


def compare_counts(live_counts: pd.DataFrame, sim_counts: pd.DataFrame) -> dict[str, Any]:
    merged = live_counts.merge(
        sim_counts,
        on=["hour", "symbol", "side"],
        how="outer",
        suffixes=("_live", "_sim"),
    )
    merged["count_live"] = merged["count_live"].fillna(0.0)
    merged["count_sim"] = merged["count_sim"].fillna(0.0)
    merged["abs_delta"] = (merged["count_live"] - merged["count_sim"]).abs()
    merged["exact"] = (merged["count_live"] == merged["count_sim"]).astype(float)

    per_symbol = (
        merged.groupby("symbol", as_index=False)[["count_live", "count_sim", "abs_delta"]]
        .sum()
        .sort_values("abs_delta", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "hourly_abs_count_delta_total": float(merged["abs_delta"].sum()),
        "exact_row_ratio": float(merged["exact"].mean()) if len(merged) else 1.0,
        "live_entries": int(live_counts["count"].sum()) if len(live_counts) else 0,
        "sim_entries": int(sim_counts["count"].sum()) if len(sim_counts) else 0,
        "rows_compared": int(len(merged)),
        "per_symbol": per_symbol.to_dict(orient="records"),
    }


def compare_entries(live_entries: pd.DataFrame, sim_entries: pd.DataFrame) -> dict[str, Any]:
    live_summary = _entry_summary(live_entries, side_col="side", price_col="entry_price", qty_col="qty")
    sim_summary = _entry_summary(sim_entries, side_col="side", price_col="entry_price", qty_col="qty")

    merged = live_summary.merge(
        sim_summary,
        on=["hour", "symbol", "side"],
        how="outer",
        suffixes=("_live", "_sim"),
    )
    for col in ("count_live", "count_sim", "qty_total_live", "qty_total_sim", "avg_price_live", "avg_price_sim"):
        merged[col] = merged[col].fillna(0.0)
    merged["abs_delta"] = (merged["count_live"] - merged["count_sim"]).abs()
    merged["qty_abs_delta"] = (merged["qty_total_live"] - merged["qty_total_sim"]).abs()
    merged["exact"] = (merged["count_live"] == merged["count_sim"]).astype(float)
    price_mask = (merged["qty_total_live"] > 0) & (merged["qty_total_sim"] > 0)
    merged["price_abs_delta"] = 0.0
    merged.loc[price_mask, "price_abs_delta"] = (
        merged.loc[price_mask, "avg_price_live"] - merged.loc[price_mask, "avg_price_sim"]
    ).abs()

    per_symbol = (
        merged.groupby("symbol", as_index=False)[["count_live", "count_sim", "abs_delta", "qty_abs_delta"]]
        .sum()
        .sort_values(["abs_delta", "qty_abs_delta"], ascending=False)
        .reset_index(drop=True)
    )

    return {
        "hourly_abs_count_delta_total": float(merged["abs_delta"].sum()),
        "hourly_abs_qty_delta_total": float(merged["qty_abs_delta"].sum()),
        "matched_price_mae": (
            float(merged.loc[price_mask, "price_abs_delta"].mean()) if bool(price_mask.any()) else 0.0
        ),
        "exact_row_ratio": float(merged["exact"].mean()) if len(merged) else 1.0,
        "live_entries": int(live_summary["count"].sum()) if len(live_summary) else 0,
        "sim_entries": int(sim_summary["count"].sum()) if len(sim_summary) else 0,
        "rows_compared": int(len(merged)),
        "per_symbol": per_symbol.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay sparse live stock entry log in simulator.")
    parser.add_argument("--trade-log", type=Path, default=Path("strategy_state/stock_trade_log.jsonl"))
    parser.add_argument("--event-log", type=Path, default=Path("strategy_state/stock_event_log.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--initial-cash", type=float, default=50_000.0)
    parser.add_argument("--max-positions", type=int, default=7)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--min-edge", type=float, default=-1.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--decision-lag-bars", type=int, default=0)
    parser.add_argument("--bar-margins", default="0.0005,0.001,0.002")
    parser.add_argument("--entry-order-ttls", default="0,1,2")
    parser.add_argument(
        "--market-order-entries",
        default="0,1",
        help="Comma-separated bool flags (0/1,true/false) for market-order entry fill mode.",
    )
    parser.add_argument(
        "--sim-backend",
        choices=["python", "native", "auto", "hourly_trader"],
        default="python",
        help="Use python by default because pending-entry TTL is python-only.",
    )
    parser.add_argument("--cancel-ack-delays", default="1")
    parser.add_argument(
        "--partial-fill-on-touch",
        default="0,1",
        help="Comma-separated bool flags for the live-like hourly trader backend.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    start = _as_utc(args.start) if str(args.start).strip() else None
    end = _as_utc(args.end) if str(args.end).strip() else None
    symbols_filter = {s.upper() for s in parse_csv_list(args.symbols)} if args.symbols.strip() else None

    actions = load_live_entries(
        trade_log=args.trade_log,
        symbols=symbols_filter,
        start=start,
        end=end,
    )
    if actions.empty:
        raise RuntimeError("No live entry actions found in selected log/window")

    symbols = sorted(actions["symbol"].astype(str).str.upper().unique().tolist())
    if start is None:
        start = actions["timestamp"].min() - pd.Timedelta(hours=2)
    if end is None:
        end = actions["timestamp"].max() + pd.Timedelta(days=1)

    bars = load_bars_for_symbols(data_root=args.data_root, symbols=symbols, start=start, end=end)
    compare_live_entries = load_live_entry_fills(
        event_log=args.event_log,
        symbols=symbols_filter,
        start=start,
        end=end,
    )
    compare_source = "broker_closed_orders"
    if compare_live_entries.empty:
        compare_live_entries = actions[["timestamp", "symbol", "side", "qty", "entry_price"]].copy()
        compare_source = "trade_log"
    live_counts = _entry_counts(compare_live_entries, side_col="side")

    is_hourly_trader_backend = str(args.sim_backend).strip().lower() == "hourly_trader"
    market_order_values = [False] if is_hourly_trader_backend else parse_bool_list(args.market_order_entries)
    ttl_values = [0] if is_hourly_trader_backend else parse_int_list(args.entry_order_ttls)
    cancel_ack_values = parse_int_list(args.cancel_ack_delays) if is_hourly_trader_backend else [0]
    partial_fill_values = parse_bool_list(args.partial_fill_on_touch) if is_hourly_trader_backend else [False]

    rows: list[dict[str, Any]] = []
    for market_order_entry in market_order_values:
        for ttl in ttl_values:
            for bar_margin in parse_float_list(args.bar_margins):
                for cancel_ack_delay in cancel_ack_values:
                    for partial_fill_on_touch in partial_fill_values:
                        replay = run_replay(
                            bars=bars,
                            actions=actions,
                            symbols=symbols,
                            initial_cash=args.initial_cash,
                            max_positions=args.max_positions,
                            max_hold_hours=args.max_hold_hours,
                            min_edge=args.min_edge,
                            fee_rate=args.fee_rate,
                            leverage=args.leverage,
                            decision_lag_bars=args.decision_lag_bars,
                            bar_margin=bar_margin,
                            entry_order_ttl_hours=ttl,
                            market_order_entry=market_order_entry,
                            sim_backend=args.sim_backend,
                            cancel_ack_delay_bars=cancel_ack_delay,
                            partial_fill_on_touch=partial_fill_on_touch,
                        )
                        compare = compare_entries(compare_live_entries, replay["sim_entries"])
                        row = {
                            "market_order_entry": bool(market_order_entry),
                            "entry_order_ttl_hours": int(ttl),
                            "bar_margin": float(bar_margin),
                            "decision_lag_bars": int(args.decision_lag_bars),
                            "cancel_ack_delay_bars": int(cancel_ack_delay),
                            "partial_fill_on_touch": bool(partial_fill_on_touch),
                            **compare,
                            "sim_metrics": replay["sim"].metrics,
                        }
                        rows.append(row)
                        logger.info(
                            "backend={} mkt={} ttl={} margin={:.4f} cancel_ack={} partial={} -> abs_delta={} qty_delta={:.2f} price_mae={:.4f}",
                            args.sim_backend,
                            int(bool(market_order_entry)),
                            ttl,
                            bar_margin,
                            cancel_ack_delay,
                            int(bool(partial_fill_on_touch)),
                            row["hourly_abs_count_delta_total"],
                            row["hourly_abs_qty_delta_total"],
                            row["matched_price_mae"],
                        )

    rows.sort(
        key=lambda r: (
            r["hourly_abs_count_delta_total"],
            r["hourly_abs_qty_delta_total"],
            r["matched_price_mae"],
            -r["exact_row_ratio"],
        )
    )
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "trade_log": str(args.trade_log),
        "event_log": str(args.event_log),
        "window_start_utc": str(start),
        "window_end_utc": str(end),
        "symbols": symbols,
        "compare_source": compare_source,
        "live_entry_count": int(live_counts["count"].sum()) if len(live_counts) else 0,
        "top": rows[:10],
        "all": rows,
    }

    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_path = Path("experiments") / f"stock_trade_log_sim_replay_{stamp}.json"
    else:
        output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Saved replay report to {}", output_path)


if __name__ == "__main__":
    main()
