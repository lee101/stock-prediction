#!/usr/bin/env python3
"""Compare actual prod trades vs what simulator would predict for same period."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.inference import generate_actions_from_frame

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_ep4_full.pt"
FILL_BUF = 0.0013
MAKER_FEE = 0.001
INITIAL_CASH = 3896.67

# Load model and generate actions on latest data
dm = ChronosSolDataModule(
    symbol="DOGEUSD",
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binanceneural/forecast_cache",
    forecast_horizons=(1,), context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32, model_id="amazon/chronos-t5-small", sequence_length=72,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True, max_history_days=365,
)

model, normalizer, feature_columns, meta = load_policy_checkpoint(CKPT, device="cuda")
seq_len = meta.get("sequence_length", 72)

# Use full frame (not just test) to include recent data
frame = dm.full_frame.copy()
print(f"Full frame: {len(frame)} bars, {frame['timestamp'].min()} to {frame['timestamp'].max()}")

actions = generate_actions_from_frame(
    model=model, frame=frame, feature_columns=feature_columns,
    normalizer=normalizer, sequence_length=seq_len, horizon=1,
)

bars = frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner")

# Apply lag
for col in ['buy_price', 'sell_price', 'buy_amount', 'sell_amount']:
    merged[col] = merged[col].shift(1)
merged = merged.dropna(subset=['buy_price']).reset_index(drop=True)

# Filter to the period when bot was funded: 02-23 16:57 UTC onwards (approx)
# Bot got funded ~16:57, first ENTER signal ~17:00, first fill at 14:19
# Let's look at 02-23 00:00 UTC to now for full context
start_ts = pd.Timestamp("2026-02-23 00:00:00", tz="UTC")
mask = merged['timestamp'] >= start_ts
recent = merged[mask].copy().reset_index(drop=True)

print(f"\nBars since {start_ts}: {len(recent)}")
print(f"Period: {recent['timestamp'].iloc[0]} to {recent['timestamp'].iloc[-1]}")

# ============================================================
# PROD TRADE TIMELINE
# ============================================================
print("\n" + "=" * 70)
print("ACTUAL PROD TRADES")
print("=" * 70)
print("02-23 14:19  BUY  40,601 DOGE @ $0.09654  ($3,919.62)")
print("02-23 20:20  SELL 40,561 DOGE @ $0.09285  ($3,768.87)")
print("02-23 20:20  BUY  40,549 DOGE @ $0.09285  ($3,764.97)")
print("              [still holding, sell limit @ $0.09426]")
print()
print(f"Trade 1 PnL: $3,768.87 - $3,919.62 = -$150.75 (-3.85%)")
print(f"Current pos: 40,549 DOGE @ $0.09285 entry")

# ============================================================
# SIMULATOR ON SAME PERIOD (sell-first, live bot order)
# ============================================================
print("\n" + "=" * 70)
print("SIMULATOR PREDICTION (sell-first, last 24h bars)")
print("=" * 70)

cash = INITIAL_CASH
inv = 0.0
trades = []

for i, (_, row) in enumerate(recent.iterrows()):
    ts = row['timestamp']
    close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
    bp = float(row.get("buy_price", 0) or 0)
    sp = float(row.get("sell_price", 0) or 0)
    ba = float(row.get("buy_amount", 0) or 0) / 100.0
    sa = float(row.get("sell_amount", 0) or 0) / 100.0

    # Sell first (live bot order)
    if inv > 0 and sa > 0 and sp > 0 and high >= sp * (1 + FILL_BUF):
        sq = min(sa * inv, inv)
        if sq > 0:
            proceeds = sq * sp * (1 - MAKER_FEE)
            cash += proceeds
            inv -= sq
            trades.append((str(ts)[:16], "SELL", sq, sp, cash + inv * close))

    # Buy
    if ba > 0 and bp > 0 and low <= bp * (1 - FILL_BUF):
        equity = cash + inv * close
        max_bv = 1.0 * max(equity, 0) - inv * bp
        if max_bv > 0:
            qty = ba * max_bv / (bp * (1 + MAKER_FEE))
            if qty > 0:
                cost = qty * bp * (1 + MAKER_FEE)
                cash -= cost
                inv += qty
                trades.append((str(ts)[:16], "BUY", qty, bp, cash + inv * close))

    # Print bar detail
    equity = cash + inv * close
    action = ""
    if trades and trades[-1][0] == str(ts)[:16]:
        last_trades = [t for t in trades if t[0] == str(ts)[:16]]
        action = " + ".join([f"{t[1]} {t[2]:.0f}@{t[3]:.5f}" for t in last_trades])

    print(f"{str(ts)[:16]} O={row['open']:.5f} H={high:.5f} L={low:.5f} C={close:.5f} | "
          f"bp={bp:.5f} sp={sp:.5f} ba={ba*100:.0f}% sa={sa*100:.0f}% | "
          f"inv={inv:.0f} eq=${equity:.2f}"
          + (f" << {action}" if action else ""))

# Mark-to-market
last_close = float(recent.iloc[-1]["close"])
final_eq = cash + inv * last_close
print(f"\nSim final equity: ${final_eq:.2f} (return: {(final_eq/INITIAL_CASH - 1)*100:.2f}%)")
print(f"Sim trades: {len(trades)}")
for t in trades:
    print(f"  {t[0]} {t[1]:4s} {t[2]:>10.1f} @ {t[3]:.5f} (eq=${t[4]:.2f})")

# ============================================================
# COMPARE
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
prod_equity = 3764.97 + 40549 * (last_close - 0.09285)  # approx current prod equity
prod_return = (prod_equity / INITIAL_CASH - 1) * 100
sim_return = (final_eq / INITIAL_CASH - 1) * 100

print(f"Prod equity (approx): ${prod_equity:.2f} ({prod_return:.2f}%)")
print(f"Sim equity:           ${final_eq:.2f} ({sim_return:.2f}%)")
print(f"Current DOGE price:   ${last_close:.5f}")
print(f"Prod in position:     40,549 DOGE (entry $0.09285)")
print(f"Sim in position:      {inv:.0f} DOGE")
