#!/usr/bin/env python3
"""Deeper DOGE trade analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import pandas as pd
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import _build_policy

REPO = Path(__file__).resolve().parents[1]

dm = ChronosSolDataModule(
    symbol="DOGEUSD",
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binanceneural" / "forecast_cache",
    forecast_horizons=(1,),
    context_hours=512,
    quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32,
    model_id="amazon/chronos-t5-small",
    sequence_length=72,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True,
    max_history_days=365,
)

ckpt = REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_wd03/binanceneural_20260222_063258/epoch_004.pt"
payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
sd = payload.get("state_dict", payload)
cfg = payload.get("config", {})
model = _build_policy(sd, cfg, len(dm.feature_columns))

test_frame = dm.test_frame.copy()
actions = generate_actions_from_frame(
    model=model, frame=test_frame, feature_columns=list(dm.feature_columns),
    normalizer=dm.normalizer, sequence_length=72, horizon=1,
)
test_start = dm.test_window_start
a = actions[actions["timestamp"] >= test_start].copy()

# Analyze direction bias
long_count = (a["buy_amount"] > a["sell_amount"]).sum()
short_count = (a["sell_amount"] > a["buy_amount"]).sum()
flat_count = ((a["buy_amount"] == 0) & (a["sell_amount"] == 0)).sum()
net_amounts = a["buy_amount"] - a["sell_amount"]

print(f"Total actions: {len(a)}")
print(f"Long signals: {long_count} ({100*long_count/len(a):.1f}%)")
print(f"Short signals: {short_count} ({100*short_count/len(a):.1f}%)")
print(f"Flat signals: {flat_count} ({100*flat_count/len(a):.1f}%)")
print(f"Net amount mean: {net_amounts.mean():.4f}")
print(f"Net amount std: {net_amounts.std():.4f}")
print()

# First/last 10 days
bars = test_frame[test_frame["timestamp"] >= test_start].copy()
mid = len(bars) // 2
bars_first = bars.iloc[:mid]
bars_last = bars.iloc[mid:]
print(f"First half price: {bars_first['close'].iloc[0]:.4f} -> {bars_first['close'].iloc[-1]:.4f} ({(bars_first['close'].iloc[-1]/bars_first['close'].iloc[0]-1)*100:+.1f}%)")
print(f"Second half price: {bars_last['close'].iloc[0]:.4f} -> {bars_last['close'].iloc[-1]:.4f} ({(bars_last['close'].iloc[-1]/bars_last['close'].iloc[0]-1)*100:+.1f}%)")

a_first = a.iloc[:mid]
a_last = a.iloc[mid:]
net_first = (a_first["buy_amount"] - a_first["sell_amount"]).mean()
net_last = (a_last["buy_amount"] - a_last["sell_amount"]).mean()
print(f"First half net bias: {net_first:.4f}")
print(f"Second half net bias: {net_last:.4f}")
print()

# Trade amount distribution
trade_amt = a["trade_amount"]
print(f"Trade amount: mean={trade_amt.mean():.4f} median={trade_amt.median():.4f} max={trade_amt.max():.4f}")
print(f"Non-zero trade amounts: {(trade_amt > 0).sum()}")

# Show buy/sell price spread
bp = a["buy_price"]
sp = a["sell_price"]
spread = (bp - sp) / ((bp + sp) / 2)
print(f"\nBid-ask spread: mean={spread.mean()*10000:.1f}bp median={spread.median()*10000:.1f}bp")

# Sample actions
print("\nSample actions (first 20):")
for _, row in a.head(20).iterrows():
    net = row["buy_amount"] - row["sell_amount"]
    direction = "LONG" if net > 0 else "SHORT" if net < 0 else "FLAT"
    print(f"  {row['timestamp']} buy={row['buy_amount']:.3f} sell={row['sell_amount']:.3f} -> {direction} net={net:+.3f}")
