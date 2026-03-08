#!/usr/bin/env python3
"""Compare realistic sim vs production fills for DOGE margin bot."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta

from binanceleveragesui.validate_sim_vs_live import (
    set_seeds, pull_prod_fills, load_5m_bars, reconstruct_initial_state,
    generate_hourly_signals, simulate_5m, match_trades,
)
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint

REPO = Path(__file__).resolve().parents[1]
DEPLOYED_CKPT = "binanceleveragesui/checkpoints/DOGEUSD_r4_R4_h384_cosine/binanceneural_20260301_065549/epoch_001.pt"


def run_sim(args, mode="realistic"):
    args.realistic = (mode == "realistic")
    torch.use_deterministic_algorithms(True)
    set_seeds(42)

    model, normalizer, feature_columns, meta = load_policy_checkpoint(args.checkpoint, device="cuda")
    seq_len = meta.get("sequence_length", args.sequence_length)

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

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")

    bars_5m = load_5m_bars(args.symbol, start_ts - pd.Timedelta(hours=1), end_ts)

    start_ms = int(start_ts.timestamp() * 1000)
    initial_inv, initial_entry_ts = reconstruct_initial_state(args.symbol, start_ms, lookback_hours=48)

    hourly_signals = generate_hourly_signals(args, frame, model, normalizer, feature_columns, meta)

    if initial_inv > 0 and len(bars_5m) > 0:
        first_close = float(bars_5m[bars_5m["timestamp"] >= start_ts].iloc[0]["close"])
        args.initial_cash = args.initial_cash - initial_inv * first_close

    sim_trades, final_eq, cash, inv = simulate_5m(
        args, hourly_signals, bars_5m,
        initial_inv=initial_inv, initial_entry_ts=initial_entry_ts,
    )
    return sim_trades, final_eq, hourly_signals, bars_5m


def print_comparison(prod_fills, sim_opt, sim_real, hourly_signals, start_ts, end_ts):
    print("=" * 70)
    print("SIM vs PROD COMPARISON")
    print(f"Window: {start_ts} -> {end_ts}")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Prod':>10} {'Sim(opt)':>10} {'Sim(real)':>10}")
    print("-" * 62)
    print(f"{'Trades':<30} {len(prod_fills):>10} {len(sim_opt):>10} {len(sim_real):>10}")

    prod_buys = len(prod_fills[prod_fills["side"] == "buy"]) if not prod_fills.empty else 0
    prod_sells = len(prod_fills[prod_fills["side"] == "sell"]) if not prod_fills.empty else 0
    opt_buys = sum(1 for t in sim_opt if t["side"] == "buy")
    opt_sells = sum(1 for t in sim_opt if t["side"] in ("sell", "force_sell"))
    real_buys = sum(1 for t in sim_real if t["side"] == "buy")
    real_sells = sum(1 for t in sim_real if t["side"] in ("sell", "force_sell"))
    print(f"{'  Buys':<30} {prod_buys:>10} {opt_buys:>10} {real_buys:>10}")
    print(f"{'  Sells':<30} {prod_sells:>10} {opt_sells:>10} {real_sells:>10}")

    n_signals = len(hourly_signals)
    active = sum(1 for s in hourly_signals.values() if s["buy_price"] > 0 or s["sell_price"] > 0)
    print(f"\n{'Hourly signals':<30} {n_signals:>10}")
    print(f"{'  With nonzero prices':<30} {active:>10}")

    # show signal details for the window
    print(f"\nHourly signal -> price vs market:")
    sorted_ts = sorted(hourly_signals.keys())
    for ts in sorted_ts:
        if ts < start_ts - pd.Timedelta(hours=2) or ts > end_ts:
            continue
        s = hourly_signals[ts]
        if s["buy_price"] <= 0:
            continue
        applied_hour = ts + pd.Timedelta(hours=1)
        print(f"  {str(ts)[:16]} -> applied {str(applied_hour)[:13]}: "
              f"buy={s['buy_price']:.5f} sell={s['sell_price']:.5f} "
              f"ba={s['buy_amount']:.0f}% sa={s['sell_amount']:.0f}%")


def main():
    p = argparse.ArgumentParser(description="Compare sim vs prod DOGE fills")
    p.add_argument("--symbol", default="DOGEUSDT")
    p.add_argument("--data-symbol", default="DOGEUSD")
    p.add_argument("--checkpoint", default=str(REPO / DEPLOYED_CKPT))
    p.add_argument("--start", default=(datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"))
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    p.add_argument("--intensity-scale", type=float, default=5.0)
    p.add_argument("--max-hold-hours", type=int, default=6)
    p.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--initial-cash", type=float, default=3320.0)
    p.add_argument("--sequence-length", type=int, default=72)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--expiry-minutes", type=int, default=90)
    p.add_argument("--max-fill-fraction", type=float, default=0.01)
    p.add_argument("--min-notional", type=float, default=5.0)
    p.add_argument("--tick-size", type=float, default=0.00001)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument(
        "--margin-hourly-rate",
        type=float,
        default=0.0000025457,
        help="Hourly margin interest rate applied to borrowed exposure in simulator.",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    print(f"pulling prod fills {args.symbol} {start_ts} -> {end_ts}...")
    prod_fills, _, _ = pull_prod_fills(args.symbol, start_ms, end_ms)
    print(f"prod: {len(prod_fills)} fills")

    print("\n--- optimistic sim (bar touch = fill) ---")
    sim_opt, eq_opt, signals, bars_5m = run_sim(args, mode="optimistic")
    print(f"optimistic: {len(sim_opt)} trades, eq=${eq_opt:.2f}")

    print("\n--- realistic sim (expiry + volume + validation) ---")
    sim_real, eq_real, _, _ = run_sim(args, mode="realistic")
    print(f"realistic: {len(sim_real)} trades, eq=${eq_real:.2f}")

    print_comparison(prod_fills, sim_opt, sim_real, signals, start_ts, end_ts)

    # trade match analysis
    if not prod_fills.empty:
        print("\n--- optimistic match ---")
        matches_opt, unmatched_opt = match_trades(prod_fills, sim_opt)
        n_match_opt = sum(1 for m in matches_opt if m["matched"])
        print(f"matched {n_match_opt}/{len(prod_fills)} prod fills")

        print("\n--- realistic match ---")
        matches_real, unmatched_real = match_trades(prod_fills, sim_real)
        n_match_real = sum(1 for m in matches_real if m["matched"])
        print(f"matched {n_match_real}/{len(prod_fills)} prod fills")


if __name__ == "__main__":
    main()
