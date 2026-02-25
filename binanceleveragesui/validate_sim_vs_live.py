#!/usr/bin/env python3
"""Validate market simulator against actual Binance production fills.

Architecture matching the live bot:
- Hourly signal generation (neural model on hourly bars)
- 5-minute execution granularity (check fills on 5m bars within each hour)
- Sell-first, no same-5m-bar roundtrips
- Lag=1 hour (signal from hour T-1 applied during hour T)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta

from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.inference import generate_latest_action
from src.binan.binance_margin import get_margin_trades, get_all_margin_orders

REPO = Path(__file__).resolve().parents[1]


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pull_prod_fills(symbol: str, start_ms: int, end_ms: int):
    MS_24H = 24 * 3600 * 1000
    raw_trades, raw_orders = [], []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + MS_24H, end_ms)
        raw_trades.extend(get_margin_trades(symbol, start_time=cursor, end_time=chunk_end, limit=1000))
        raw_orders.extend(get_all_margin_orders(symbol, start_time=cursor, end_time=chunk_end, limit=500))
        cursor = chunk_end

    trades = []
    for t in raw_trades:
        trades.append({
            "timestamp": pd.Timestamp(int(t["time"]), unit="ms", tz="UTC"),
            "side": "buy" if t.get("isBuyer") else "sell",
            "price": float(t["price"]),
            "qty": float(t["qty"]),
            "quote_qty": float(t.get("quoteQty", 0)),
            "commission": float(t.get("commission", 0)),
            "commission_asset": t.get("commissionAsset", ""),
            "order_id": t.get("orderId"),
        })
    if not trades:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    tdf = pd.DataFrame(trades)
    agg = tdf.groupby("order_id").agg(
        timestamp=("timestamp", "first"),
        side=("side", "first"),
        avg_price=("price", lambda x: np.average(x, weights=tdf.loc[x.index, "qty"])),
        total_qty=("qty", "sum"),
        total_quote=("quote_qty", "sum"),
        total_commission=("commission", "sum"),
        commission_asset=("commission_asset", "first"),
        n_fills=("qty", "count"),
    ).reset_index().sort_values("timestamp").reset_index(drop=True)

    orders = []
    for o in raw_orders:
        orders.append({
            "order_id": o.get("orderId"),
            "timestamp": pd.Timestamp(int(o["time"]), unit="ms", tz="UTC"),
            "side": o.get("side", "").lower(),
            "status": o.get("status", ""),
            "limit_price": float(o.get("price", 0)),
            "orig_qty": float(o.get("origQty", 0)),
            "executed_qty": float(o.get("executedQty", 0)),
            "cumm_quote": float(o.get("cummulativeQuoteQty", 0)),
        })
    odf = pd.DataFrame(orders) if orders else pd.DataFrame()
    return agg, tdf, odf


def load_5m_bars(symbol: str, start_ts, end_ts):
    path = REPO / "trainingdata5min" / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No 5m data: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    return df[mask].sort_values("timestamp").reset_index(drop=True)


def generate_hourly_signals(args, frame, model, normalizer, feature_columns, meta):
    """Generate signals for each hourly bar."""
    seq_len = meta.get("sequence_length", args.sequence_length)
    start_ts = pd.Timestamp(args.start, tz="UTC") - pd.Timedelta(hours=2)
    start_idx_arr = frame.index[frame["timestamp"] >= start_ts]
    if len(start_idx_arr) == 0:
        return {}
    start_idx = start_idx_arr[0]

    signals = {}
    for bar_idx in range(start_idx, len(frame)):
        ts = frame.iloc[bar_idx]["timestamp"]
        sub_frame = frame.iloc[:bar_idx + 1].copy()
        set_seeds(42)
        action = generate_latest_action(
            model=model, frame=sub_frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=seq_len, horizon=args.horizon,
        )

        bp = float(action.get("buy_price", 0))
        sp = float(action.get("sell_price", 0))
        ba = max(0.0, min(100.0, float(action.get("buy_amount", 0)) * args.intensity_scale))
        sa = max(0.0, min(100.0, float(action.get("sell_amount", 0)) * args.intensity_scale))

        min_spread = 0.002
        if bp > 0 and sp > 0 and sp <= bp * (1 + min_spread):
            mid = (bp + sp) / 2
            bp = mid * (1 - min_spread / 2)
            sp = mid * (1 + min_spread / 2)

        signals[ts] = {"buy_price": bp, "sell_price": sp,
                        "buy_amount": ba, "sell_amount": sa}
    return signals


def simulate_5m(args, hourly_signals, bars_5m):
    """Execute on 5m bars using hourly signals with lag=1.

    Models the live bot behavior:
    - Signal from hour T-1 is active during all 5m bars of hour T
    - ONE buy order per signal (fills once, done until sell or new signal)
    - ONE sell order per signal (fills once, done until buy or new signal)
    - After sell fills, can re-buy on later 5m bar (oscillation)
    - No same-5m-bar roundtrips
    """
    fee = args.fee
    fill_buf = args.fill_buffer_pct
    cash = args.initial_cash
    inv = 0.0
    sim_trades = []
    entry_ts = None

    # state tracking per hourly signal window
    current_sig_hour = None
    bought_this_signal = False
    sold_this_signal = False

    def get_active_signal(ts_5m):
        bar_hour = ts_5m.floor("h")
        prev_hour = bar_hour - pd.Timedelta(hours=1)
        return hourly_signals.get(prev_hour), prev_hour

    for _, bar in bars_5m.iterrows():
        ts = bar["timestamp"]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        sig, sig_hour = get_active_signal(ts)
        if sig is None:
            continue

        # reset fill tracking when hourly signal changes
        if sig_hour != current_sig_hour:
            current_sig_hour = sig_hour
            bought_this_signal = False
            sold_this_signal = False

        bp = sig["buy_price"]
        sp = sig["sell_price"]
        ba = sig["buy_amount"] / 100.0
        sa = sig["sell_amount"] / 100.0

        # max hold check
        if args.max_hold_hours > 0 and inv > 0 and entry_ts is not None:
            hours_held = (ts - entry_ts).total_seconds() / 3600.0
            if hours_held >= args.max_hold_hours:
                fp = close * 0.999
                cash += inv * fp * (1 - fee)
                sim_trades.append({"ts": ts, "side": "force_sell", "price": fp, "qty": inv})
                inv = 0.0
                entry_ts = None
                bought_this_signal = False
                sold_this_signal = False
                continue

        acted_this_bar = False

        # SELL FIRST (one fill per signal, unless we re-bought after a sell)
        if (inv > 0 and sa > 0 and sp > 0 and not sold_this_signal
                and high >= sp * (1 + fill_buf)):
            sq = min(sa * inv, inv)
            if sq > 0:
                cash += sq * sp * (1 - fee)
                inv -= sq
                sim_trades.append({"ts": ts, "side": "sell", "price": sp, "qty": sq})
                acted_this_bar = True
                sold_this_signal = True
                bought_this_signal = False  # allow re-buy after sell (oscillation)
                if inv <= 0:
                    entry_ts = None

        # BUY (one fill per signal, no same-bar roundtrip)
        if (not acted_this_bar and not bought_this_signal
                and ba > 0 and bp > 0 and low <= bp * (1 - fill_buf)):
            equity = cash + inv * close
            max_bv = args.max_leverage * max(equity, 0) - inv * bp
            if max_bv > 0:
                qty = ba * max_bv / (bp * (1 + fee))
                if qty > 0:
                    cash -= qty * bp * (1 + fee)
                    inv += qty
                    sim_trades.append({"ts": ts, "side": "buy", "price": bp, "qty": qty})
                    bought_this_signal = True
                    sold_this_signal = False  # allow sell after buy
                    if entry_ts is None:
                        entry_ts = ts

    last_close = float(bars_5m.iloc[-1]["close"]) if len(bars_5m) > 0 else 0
    final_eq = cash + inv * last_close
    return sim_trades, final_eq, cash, inv


def match_trades(prod_fills, sim_trades):
    if prod_fills.empty or not sim_trades:
        return [], pd.DataFrame()

    sim_df = pd.DataFrame(sim_trades)
    sim_df["_matched"] = False

    matches = []
    for _, prod in prod_fills.iterrows():
        pt = prod["timestamp"]
        # match within 30 min window, same direction
        cands = sim_df[
            (sim_df["side"] == prod["side"]) &
            (~sim_df["_matched"]) &
            (abs((sim_df["ts"] - pt).dt.total_seconds()) <= 1800)
        ]
        if len(cands) > 0:
            # pick closest in time
            diffs = abs((cands["ts"] - pt).dt.total_seconds())
            best_idx = diffs.idxmin()
            best = sim_df.loc[best_idx]
            sim_df.at[best_idx, "_matched"] = True
            diff_bps = (best["price"] - prod["avg_price"]) / prod["avg_price"] * 10000
            matches.append({
                "prod_ts": prod["timestamp"], "sim_ts": best["ts"],
                "side": prod["side"],
                "prod_price": prod["avg_price"], "sim_price": best["price"],
                "diff_bps": diff_bps,
                "prod_qty": prod["total_qty"], "sim_qty": best["qty"],
                "matched": True,
            })
        else:
            matches.append({
                "prod_ts": prod["timestamp"], "side": prod["side"],
                "prod_price": prod["avg_price"], "prod_qty": prod["total_qty"],
                "matched": False,
            })

    unmatched_sim = sim_df[~sim_df["_matched"]]
    return matches, unmatched_sim


def slippage_analysis(prod_fills, orders_df, fill_buf):
    if prod_fills.empty or orders_df.empty:
        return
    merged = prod_fills.merge(orders_df[["order_id", "limit_price"]], on="order_id", how="left")
    merged["slippage_bps"] = (merged["avg_price"] - merged["limit_price"]) / merged["limit_price"] * 10000
    print("\n" + "=" * 70)
    print("SLIPPAGE ANALYSIS (fill_price vs limit_price)")
    print("=" * 70)
    for _, r in merged.iterrows():
        d = "+" if r["slippage_bps"] > 0 else ""
        print(f"  {str(r['timestamp'])[:16]} {r['side']:>4s} limit={r['limit_price']:.5f} "
              f"fill={r['avg_price']:.5f} slip={d}{r['slippage_bps']:.1f}bps")
    buys = merged[merged["side"] == "buy"]["slippage_bps"]
    sells = merged[merged["side"] == "sell"]["slippage_bps"]
    if len(buys) > 0:
        print(f"\n  Buy slippage:  mean={buys.mean():.1f}bps std={buys.std():.1f}bps")
    if len(sells) > 0:
        print(f"  Sell slippage: mean={sells.mean():.1f}bps std={sells.std():.1f}bps")
    all_slip = merged["slippage_bps"].abs()
    print(f"  Overall |slip|: mean={all_slip.mean():.1f}bps  (fill_buffer={fill_buf*10000:.0f}bps)")


def print_report(prod_fills, sim_trades, matches, unmatched_sim, prod_orders,
                 final_eq, initial_cash, fill_buf):
    print("=" * 70)
    print("SIM vs LIVE VALIDATION REPORT (5m execution)")
    print("=" * 70)

    print(f"\nPRODUCTION FILLS ({len(prod_fills)}):")
    if not prod_fills.empty:
        for _, r in prod_fills.iterrows():
            comm_str = f"comm={r['total_commission']:.4f} {r['commission_asset']}"
            print(f"  {str(r['timestamp'])[:16]} {r['side']:>4s} {r['total_qty']:>10.0f} "
                  f"@ {r['avg_price']:.5f}  ({comm_str})")

    print(f"\nSIMULATED TRADES ({len(sim_trades)}):")
    for t in sim_trades:
        print(f"  {str(t['ts'])[:16]} {t['side']:>4s} {t['qty']:>10.0f} @ {t['price']:.5f}")

    print(f"\nTRADE-BY-TRADE COMPARISON:")
    n_matched = sum(1 for m in matches if m["matched"])
    for m in matches:
        if m["matched"]:
            d = "+" if m["diff_bps"] > 0 else ""
            tdiff = abs((m["sim_ts"] - m["prod_ts"]).total_seconds() / 60)
            print(f"  {str(m['prod_ts'])[:16]} {m['side']:>4s} "
                  f"prod={m['prod_price']:.5f} sim={m['sim_price']:.5f} "
                  f"({d}{m['diff_bps']:.1f}bps, {tdiff:.0f}min apart)")
        else:
            print(f"  {str(m['prod_ts'])[:16]} {m['side']:>4s} "
                  f"prod={m['prod_price']:.5f} -> NO SIM MATCH")

    if len(unmatched_sim) > 0:
        print(f"\n  Sim-only trades ({len(unmatched_sim)}):")
        for _, s in unmatched_sim.iterrows():
            print(f"    {str(s['ts'])[:16]} {s['side']:>4s} {s['qty']:.0f} @ {s['price']:.5f}")

    print(f"\nACCURACY METRICS:")
    n_prod = len(prod_fills)
    match_rate = n_matched / max(n_prod, 1) * 100
    price_diffs = [m["diff_bps"] for m in matches if m["matched"]]
    avg_diff = np.mean(price_diffs) if price_diffs else 0
    print(f"  Fill direction match: {n_matched}/{n_prod} ({match_rate:.0f}%)")
    print(f"  Avg price diff:       {avg_diff:+.1f} bps")
    print(f"  Sim final equity:     ${final_eq:.2f} ({(final_eq/initial_cash - 1)*100:+.2f}%)")

    slippage_analysis(prod_fills, prod_orders, fill_buf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="DOGEUSDT")
    p.add_argument("--data-symbol", default="DOGEUSD")
    p.add_argument("--checkpoint", default=str(REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_ep4_full.pt"))
    p.add_argument("--start", default=(datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d"))
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    p.add_argument("--intensity-scale", type=float, default=5.0)
    p.add_argument("--max-hold-hours", type=int, default=6)
    p.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--initial-cash", type=float, default=3754.0)
    p.add_argument("--sequence-length", type=int, default=72)
    p.add_argument("--horizon", type=int, default=1)
    args = p.parse_args()

    torch.use_deterministic_algorithms(True)
    set_seeds(42)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, normalizer, feature_columns, meta = load_policy_checkpoint(args.checkpoint, device="cuda")
    seq_len = meta.get("sequence_length", args.sequence_length)

    print(f"Loading hourly data for {args.data_symbol}...")
    dm = ChronosSolDataModule(
        symbol=args.data_symbol,
        data_root=REPO / "trainingdatahourlybinance",
        forecast_cache_root=REPO / "binanceneural/forecast_cache",
        forecast_horizons=(args.horizon,), context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32, model_id="amazon/chronos-t5-small",
        sequence_length=seq_len,
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=True, max_history_days=365,
    )
    frame = dm.full_frame.copy()
    print(f"Hourly frame: {len(frame)} bars, {frame['timestamp'].min()} to {frame['timestamp'].max()}")

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    print(f"\nLoading 5m bars for {args.symbol}...")
    bars_5m = load_5m_bars(args.symbol, start_ts - pd.Timedelta(hours=1), end_ts)
    print(f"5m bars: {len(bars_5m)} ({bars_5m['timestamp'].min()} to {bars_5m['timestamp'].max()})")

    print(f"\nPulling production fills: {args.symbol} {start_ts} to {end_ts}")
    prod_fills, raw_trades, prod_orders = pull_prod_fills(args.symbol, start_ms, end_ms)
    print(f"Got {len(prod_fills)} aggregated fills from {len(raw_trades)} raw trades")

    print(f"\nGenerating hourly signals (deterministic)...")
    hourly_signals = generate_hourly_signals(args, frame, model, normalizer, feature_columns, meta)
    print(f"Generated {len(hourly_signals)} hourly signals")

    print(f"\nSimulating on 5m bars (hourly signals, 5m execution, lag=1)...")
    sim_trades, final_eq, cash, inv = simulate_5m(args, hourly_signals, bars_5m)
    print(f"Sim: {len(sim_trades)} trades, equity=${final_eq:.2f}, position={inv:.0f}")

    matches, unmatched_sim = match_trades(prod_fills, sim_trades)
    print_report(prod_fills, sim_trades, matches, unmatched_sim, prod_orders,
                 final_eq, args.initial_cash, args.fill_buffer_pct)


if __name__ == "__main__":
    main()
