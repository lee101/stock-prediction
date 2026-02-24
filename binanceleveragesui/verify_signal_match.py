#!/usr/bin/env python3
"""Check why prod signals differ from sim signals.

Prod log shows: signal buy=0.0921(2.7%) sell=0.0944(100.0%)
Sim generates:  bp=0.09286 sp=0.09425 for same bar

The difference must be in the data/frame used.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.inference import generate_latest_action

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_ep4_full.pt"

model, normalizer, feature_columns, meta = load_policy_checkpoint(CKPT, device="cuda")
seq_len = meta.get("sequence_length", 72)

# Sim uses test_frame with split (val_days=30, test_days=30)
dm_split = ChronosSolDataModule(
    symbol="DOGEUSD",
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binanceneural/forecast_cache",
    forecast_horizons=(1,), context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32, model_id="amazon/chronos-t5-small", sequence_length=seq_len,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True, max_history_days=365,
)

# Live bot uses full_frame with minimal split (val_days=1, test_days=1)
dm_live = ChronosSolDataModule(
    symbol="DOGEUSD",
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binanceneural/forecast_cache",
    forecast_horizons=(1,), context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32, model_id="amazon/chronos-t5-small", sequence_length=seq_len,
    split_config=SplitConfig(val_days=1, test_days=1),
    cache_only=True,
)

split_frame = dm_split.full_frame
live_frame = dm_live.full_frame

print(f"Split frame: {len(split_frame)} bars, {split_frame['timestamp'].min()} to {split_frame['timestamp'].max()}")
print(f"Live frame:  {len(live_frame)} bars, {live_frame['timestamp'].min()} to {live_frame['timestamp'].max()}")

# Check feature differences for the last few bars
target_ts = pd.Timestamp("2026-02-23 19:00:00", tz="UTC")

split_row = split_frame[split_frame['timestamp'] == target_ts]
live_row = live_frame[live_frame['timestamp'] == target_ts]

if len(split_row) > 0 and len(live_row) > 0:
    print(f"\nBar at {target_ts}:")
    print(f"  Split close: {split_row['close'].values[0]:.5f}")
    print(f"  Live close:  {live_row['close'].values[0]:.5f}")

    # Check reference_close and chronos columns
    for col in ['reference_close', 'predicted_high_p50_h1', 'predicted_low_p50_h1']:
        if col in split_frame.columns and col in live_frame.columns:
            sv = split_row[col].values[0]
            lv = live_row[col].values[0]
            print(f"  {col}: split={sv:.5f} live={lv:.5f} {'MATCH' if abs(sv-lv) < 1e-6 else 'DIFF'}")

    # Check some feature columns
    diffs = 0
    for col in list(feature_columns)[:5]:
        if col in split_frame.columns and col in live_frame.columns:
            sv = split_row[col].values[0]
            lv = live_row[col].values[0]
            if abs(sv - lv) > 1e-6:
                diffs += 1
                print(f"  DIFF {col}: split={sv:.6f} live={lv:.6f}")
    print(f"  Feature diffs in first 5 cols: {diffs}")

# Generate signal from both frames at the same timestamp
for name, frame in [("split", split_frame), ("live", live_frame)]:
    # Trim to target_ts
    trimmed = frame[frame['timestamp'] <= target_ts].copy()
    if len(trimmed) < seq_len:
        print(f"\n{name}: not enough data")
        continue
    action = generate_latest_action(
        model=model, frame=trimmed, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=seq_len, horizon=1,
    )
    print(f"\n{name} signal at {target_ts}:")
    print(f"  buy_price={action['buy_price']:.5f} sell_price={action['sell_price']:.5f}")
    print(f"  buy_amount={action['buy_amount']:.1f} sell_amount={action['sell_amount']:.1f}")

# Also check what prod reported at that time
print(f"\nProd log signals around {target_ts}:")
print(f"  signal buy=0.0921(2.7%) sell=0.0944(100.0%)")
print(f"  signal buy=0.0929(0.3%) sell=0.0943(100.0%)")
