#!/usr/bin/env python3
"""Analyze intra-bar fill assumptions and model behavior patterns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.inference import generate_actions_from_frame

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_ep4_full.pt"
FILL_BUF = 0.0005
MAKER_FEE = 0.001

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
test_frame = dm.test_frame.copy()
actions = generate_actions_from_frame(
    model=model, frame=test_frame, feature_columns=feature_columns,
    normalizer=normalizer, sequence_length=meta.get("sequence_length", 72), horizon=1,
)

bars = test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner")
merged['buy_price'] = merged['buy_price'].shift(1)
merged['sell_price'] = merged['sell_price'].shift(1)
merged['buy_amount'] = merged['buy_amount'].shift(1)
merged['sell_amount'] = merged['sell_amount'].shift(1)
merged = merged.dropna(subset=['buy_price']).reset_index(drop=True)

# Analyze model signal patterns
print("=" * 60)
print("MODEL SIGNAL ANALYSIS")
print("=" * 60)

ba = merged['buy_amount'].values / 100.0
sa = merged['sell_amount'].values / 100.0
bp = merged['buy_price'].values
sp = merged['sell_price'].values
highs = merged['high'].values
lows = merged['low'].values
closes = merged['close'].values

buy_fills = (ba > 0) & (bp > 0) & (lows <= bp * (1 - FILL_BUF))
sell_fills = (sa > 0) & (sp > 0) & (highs >= sp * (1 + FILL_BUF))
both_fill = buy_fills & sell_fills

print(f"Total bars:       {len(merged)}")
print(f"Buy signals:      {(ba > 0).sum()} ({(ba > 0).mean()*100:.0f}%)")
print(f"Sell signals:     {(sa > 0).sum()} ({(sa > 0).mean()*100:.0f}%)")
print(f"Buy fills:        {buy_fills.sum()} ({buy_fills.mean()*100:.0f}%)")
print(f"Sell fills:       {sell_fills.sum()} ({sell_fills.mean()*100:.0f}%)")
print(f"Both fill:        {both_fill.sum()} ({both_fill.mean()*100:.0f}%)")

print(f"\nbuy_amount dist:  mean={ba.mean()*100:.1f}% median={np.median(ba)*100:.1f}% max={ba.max()*100:.1f}%")
print(f"sell_amount dist: mean={sa.mean()*100:.1f}% median={np.median(sa)*100:.1f}% max={sa.max()*100:.1f}%")

# Spread analysis
spreads = (sp - bp) / bp
print(f"\nModel spread:     mean={spreads.mean()*100:.2f}% median={np.median(spreads)*100:.2f}%")

# Bar range analysis
bar_range = (highs - lows) / closes
print(f"Bar range:        mean={bar_range.mean()*100:.2f}% median={np.median(bar_range)*100:.2f}%")

# For bars where both fill: what's the typical range needed?
if both_fill.sum() > 0:
    both_spreads = spreads[both_fill]
    both_ranges = bar_range[both_fill]
    print(f"\nBoth-fill bars ({both_fill.sum()}):")
    print(f"  Spread needed:  mean={both_spreads.mean()*100:.2f}%")
    print(f"  Bar range:      mean={both_ranges.mean()*100:.2f}%")
    # Can range support both fills?
    needed_range = (sp[both_fill] * (1 + FILL_BUF) - bp[both_fill] * (1 - FILL_BUF)) / closes[both_fill]
    actual_range = (highs[both_fill] - lows[both_fill]) / closes[both_fill]
    print(f"  Range needed:   mean={needed_range.mean()*100:.2f}%")
    print(f"  Range actual:   mean={actual_range.mean()*100:.2f}%")
    slack = actual_range - needed_range
    print(f"  Range slack:    mean={slack.mean()*100:.2f}% (>0 means bar is wide enough)")

# KEY: how much does buy_price sit below reference_close, and sell_price above?
ref = merged['close'].shift(1).values  # reference_close is prev bar close
valid = ~np.isnan(ref) & (ref > 0) & (bp > 0) & (sp > 0)
buy_offset = (ref[valid] - bp[valid]) / ref[valid]
sell_offset = (sp[valid] - ref[valid]) / ref[valid]
print(f"\nBuy below ref:    mean={buy_offset.mean()*100:.2f}% median={np.median(buy_offset)*100:.2f}%")
print(f"Sell above ref:   mean={sell_offset.mean()*100:.2f}% median={np.median(sell_offset)*100:.2f}%")

# Trade profitability per round-trip
print("\n" + "=" * 60)
print("ROUND-TRIP ANALYSIS (sell-first order)")
print("=" * 60)

cash = 10000.0
inv = 0.0
entry_prices = []
exit_prices = []
rts = []
current_avg_entry = 0.0

for _, row in merged.iterrows():
    close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
    bprice = float(row.get("buy_price", 0) or 0)
    sprice = float(row.get("sell_price", 0) or 0)
    bamt = float(row.get("buy_amount", 0) or 0) / 100.0
    samt = float(row.get("sell_amount", 0) or 0) / 100.0

    # Sell first
    if inv > 0 and samt > 0 and sprice > 0 and high >= sprice * (1 + FILL_BUF):
        sq = min(samt * inv, inv)
        if sq > 0:
            proceeds = sq * sprice * (1 - MAKER_FEE)
            pnl_pct = (sprice * (1 - MAKER_FEE) / current_avg_entry) - 1
            rts.append(pnl_pct)
            cash += proceeds
            inv -= sq

    # Buy
    if bamt > 0 and bprice > 0 and low <= bprice * (1 - FILL_BUF):
        equity = cash + inv * close
        max_bv = max(equity, 0) - inv * bprice
        if max_bv > 0:
            qty = bamt * max_bv / (bprice * (1 + MAKER_FEE))
            if qty > 0:
                # Update weighted avg entry
                total_qty = inv + qty
                if total_qty > 0:
                    current_avg_entry = (inv * current_avg_entry + qty * bprice * (1 + MAKER_FEE)) / total_qty
                cash -= qty * bprice * (1 + MAKER_FEE)
                inv += qty

rts = np.array(rts)
print(f"Total sells (round-trips): {len(rts)}")
print(f"Win rate:        {(rts > 0).mean()*100:.1f}%")
print(f"Avg win:         {rts[rts > 0].mean()*100:.2f}%")
print(f"Avg loss:        {rts[rts < 0].mean()*100:.2f}%")
print(f"Profit factor:   {rts[rts > 0].sum() / abs(rts[rts < 0].sum()):.2f}" if (rts < 0).sum() > 0 else "inf")
print(f"Avg per trade:   {rts.mean()*100:.3f}%")

# Distribution
print(f"\nPnL distribution:")
for pct in [-2, -1, -0.5, 0, 0.5, 1, 2, 5]:
    count = (rts > pct/100).sum() if pct < 0 else (rts < pct/100).sum()
    print(f"  {'>' if pct < 0 else '<'}{pct}%: {count} ({count/len(rts)*100:.0f}%)")

# Holding time analysis
print("\n" + "=" * 60)
print("HOLDING TIME ANALYSIS")
print("=" * 60)

inv_state = []
in_pos = False
entry_bar = 0
hold_times = []

for i, (_, row) in enumerate(merged.iterrows()):
    close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
    bprice = float(row.get("buy_price", 0) or 0)
    sprice = float(row.get("sell_price", 0) or 0)
    bamt = float(row.get("buy_amount", 0) or 0) / 100.0
    samt = float(row.get("sell_amount", 0) or 0) / 100.0

    # Track transitions
    was_in = in_pos

    if not in_pos:
        if bamt > 0 and bprice > 0 and low <= bprice * (1 - FILL_BUF):
            in_pos = True
            entry_bar = i

    if in_pos and samt > 0 and sprice > 0 and high >= sprice * (1 + FILL_BUF):
        hold_times.append(i - entry_bar)
        # Don't set in_pos=False because partial sells are common
        # Check if fully sold next round

hold_times = np.array(hold_times) if hold_times else np.array([0])
print(f"Sell events:     {len(hold_times)}")
print(f"Avg bars held:   {hold_times.mean():.1f}h")
print(f"Median held:     {np.median(hold_times):.0f}h")
print(f"1-bar holds:     {(hold_times <= 1).sum()} ({(hold_times <= 1).mean()*100:.0f}%)")
print(f"<= 3h holds:     {(hold_times <= 3).sum()} ({(hold_times <= 3).mean()*100:.0f}%)")
