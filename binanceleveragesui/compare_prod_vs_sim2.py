#!/usr/bin/env python3
"""Compare prod vs sim using EXACT same inference path as live bot.

Key differences from previous comparison:
1. Uses generate_latest_action (same as live bot) instead of batch generate_actions_from_frame
2. Sets torch seed before each inference for determinism
3. Reloads frame for each bar (simulating live bot's frame reload per signal)
4. Applies same intensity_scale=1.0, min_spread=20bp logic as live bot
5. Sell-first order matching live bot's _fast_check
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.inference import generate_latest_action, generate_actions_from_frame

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_ep4_full.pt"
FILL_BUF = 0.0013
MAKER_FEE = 0.001
INITIAL_CASH = 3896.67
MIN_SPREAD_BP = 0.002
INTENSITY_SCALE = 1.0

model, normalizer, feature_columns, meta = load_policy_checkpoint(CKPT, device="cuda")
seq_len = meta.get("sequence_length", 72)

dm = ChronosSolDataModule(
    symbol="DOGEUSD",
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binanceneural/forecast_cache",
    forecast_horizons=(1,), context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32, model_id="amazon/chronos-t5-small", sequence_length=seq_len,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True, max_history_days=365,
)

frame = dm.full_frame.copy()
print(f"Frame: {len(frame)} bars, {frame['timestamp'].min()} to {frame['timestamp'].max()}")

# === Step 1: verify batch vs single inference produce same signals ===
print("\n" + "=" * 70)
print("VERIFYING: batch vs single-bar inference consistency")
print("=" * 70)

# Batch inference
torch.manual_seed(42)
torch.cuda.manual_seed(42)
batch_actions = generate_actions_from_frame(
    model=model, frame=frame, feature_columns=feature_columns,
    normalizer=normalizer, sequence_length=seq_len, horizon=1,
)

# Single-bar inference on last 5 bars
last_5_bars = []
for offset in range(5, 0, -1):
    idx = len(frame) - offset
    sub_frame = frame.iloc[:idx + 1].copy()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    action = generate_latest_action(
        model=model, frame=sub_frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=seq_len, horizon=1,
    )
    last_5_bars.append(action)

print(f"{'BAR':>4} {'BATCH_BP':>10} {'SINGLE_BP':>10} {'BATCH_SP':>10} {'SINGLE_SP':>10} {'BATCH_BA':>10} {'SINGLE_BA':>10} {'MATCH':>6}")
for i, single in enumerate(last_5_bars):
    batch_idx = len(batch_actions) - 5 + i
    b = batch_actions.iloc[batch_idx]
    match = (abs(b['buy_price'] - single['buy_price']) < 1e-6 and
             abs(b['sell_price'] - single['sell_price']) < 1e-6)
    print(f"{batch_idx:>4} {b['buy_price']:>10.5f} {single['buy_price']:>10.5f} "
          f"{b['sell_price']:>10.5f} {single['sell_price']:>10.5f} "
          f"{b['buy_amount']:>10.1f} {single['buy_amount']:>10.1f} "
          f"{'OK' if match else 'DIFF':>6}")

# === Step 2: generate signals bar-by-bar matching live bot ===
print("\n" + "=" * 70)
print("LIVE-BOT-STYLE SIMULATION (bar-by-bar, sell-first)")
print("=" * 70)

start_ts = pd.Timestamp("2026-02-23 00:00:00", tz="UTC")
# Find index of start bar in frame
start_idx = frame.index[frame['timestamp'] >= start_ts][0]

bars_data = frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()

cash = INITIAL_CASH
inv = 0.0
trades = []
prev_signal = None

for bar_idx in range(start_idx, len(frame)):
    row = frame.iloc[bar_idx]
    ts = row['timestamp']
    close, high, low = float(row['close']), float(row['high']), float(row['low'])

    # Generate signal for THIS bar (will be applied to NEXT bar due to lag)
    sub_frame = frame.iloc[:bar_idx + 1].copy()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    action = generate_latest_action(
        model=model, frame=sub_frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=seq_len, horizon=1,
    )

    # Apply live bot's intensity scale + min spread logic
    bp_raw = float(action.get("buy_price", 0))
    sp_raw = float(action.get("sell_price", 0))
    ba_raw = max(0.0, min(100.0, float(action.get("buy_amount", 0)) * INTENSITY_SCALE))
    sa_raw = max(0.0, min(100.0, float(action.get("sell_amount", 0)) * INTENSITY_SCALE))

    if bp_raw > 0 and sp_raw > 0 and sp_raw <= bp_raw * (1 + MIN_SPREAD_BP):
        mid = (bp_raw + sp_raw) / 2
        bp_raw = mid * (1 - MIN_SPREAD_BP / 2)
        sp_raw = mid * (1 + MIN_SPREAD_BP / 2)

    current_signal = {"buy_price": bp_raw, "sell_price": sp_raw,
                      "buy_amount": ba_raw, "sell_amount": sa_raw}

    # Use PREVIOUS bar's signal for this bar's execution (lag=1)
    if prev_signal is not None:
        bp = prev_signal["buy_price"]
        sp = prev_signal["sell_price"]
        ba = prev_signal["buy_amount"] / 100.0
        sa = prev_signal["sell_amount"] / 100.0

        actions_taken = []

        # SELL FIRST (matching live bot _fast_check: handle_exit before handle_entry/add)
        if inv > 0 and sa > 0 and sp > 0 and high >= sp * (1 + FILL_BUF):
            sq = min(sa * inv, inv)
            if sq > 0:
                proceeds = sq * sp * (1 - MAKER_FEE)
                cash += proceeds
                inv -= sq
                trades.append((str(ts)[:16], "SELL", sq, sp))
                actions_taken.append(f"SELL {sq:.0f}@{sp:.5f}")

        # BUY (entry when flat, or add when in position)
        if ba > 0 and bp > 0 and low <= bp * (1 - FILL_BUF):
            equity = cash + inv * close
            max_bv = 1.0 * max(equity, 0) - inv * bp
            if max_bv > 0:
                qty = ba * max_bv / (bp * (1 + MAKER_FEE))
                if qty > 0:
                    cost = qty * bp * (1 + MAKER_FEE)
                    cash -= cost
                    inv += qty
                    trades.append((str(ts)[:16], "BUY", qty, bp))
                    actions_taken.append(f"BUY {qty:.0f}@{bp:.5f}")

        equity = cash + inv * close
        action_str = " | ".join(actions_taken) if actions_taken else ""
        print(f"{str(ts)[:16]} H={high:.5f} L={low:.5f} C={close:.5f} | "
              f"sig bp={bp:.5f} sp={sp:.5f} ba={ba*100:.0f}% sa={sa*100:.0f}% | "
              f"inv={inv:.0f} eq=${equity:.2f}"
              + (f" << {action_str}" if action_str else ""))
    else:
        equity = cash + inv * close
        print(f"{str(ts)[:16]} H={high:.5f} L={low:.5f} C={close:.5f} | "
              f"(first bar, generating signal) | inv={inv:.0f} eq=${equity:.2f}")

    prev_signal = current_signal

# Mark-to-market
last_close = float(frame.iloc[-1]["close"])
final_eq = cash + inv * last_close

print(f"\n{'='*70}")
print(f"SIMULATOR RESULT")
print(f"{'='*70}")
print(f"Final equity:  ${final_eq:.2f} ({(final_eq/INITIAL_CASH - 1)*100:.2f}%)")
print(f"Position:      {inv:.0f} DOGE")
print(f"Trades:")
for t in trades:
    print(f"  {t[0]} {t[1]:4s} {t[2]:>10.1f} @ {t[3]:.5f}")

print(f"\n{'='*70}")
print(f"ACTUAL PROD TRADES")
print(f"{'='*70}")
print(f"02-23 14:19  BUY  40,601 @ $0.09654")
print(f"02-23 20:20  SELL 40,561 @ $0.09285")
print(f"02-23 20:20  BUY  40,549 @ $0.09285")
print(f"Currently holding 40,549 DOGE, eq ~${0.09 + 40549 * last_close:.2f}")

prod_eq = 0.09 + 40549 * last_close
print(f"\n{'='*70}")
print(f"COMPARISON")
print(f"{'='*70}")
print(f"Prod equity:  ~${prod_eq:.2f} ({(prod_eq/INITIAL_CASH - 1)*100:.2f}%)")
print(f"Sim equity:   ${final_eq:.2f} ({(final_eq/INITIAL_CASH - 1)*100:.2f}%)")
print(f"Difference:   ${final_eq - prod_eq:.2f}")
