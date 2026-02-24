#!/usr/bin/env python3
"""Analyze how often the sim's lag-1 signal misses fills that
intra-hour signal updates would catch (like the live bot does).

The live bot gets a NEW signal every 5min. So during bar T, it first
uses signal from bar T-1 (same as sim), then partway through bar T
it gets the signal from bar T applied within bar T itself (lag=0 behavior).

We compare:
1. lag=1 sim (current): signal from T-1 applied to bar T
2. lag=0 sim: signal from T applied to bar T (too optimistic, sees OHLC)
3. "live bot" sim: signal from T-1 for first half, signal from T for second half
   Approximation: a fill counts if EITHER lag=1 OR lag=0 would fill.
"""
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
FILL_BUF = 0.0013
MAKER_FEE = 0.001
INITIAL_CASH = 10000.0

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

# Create lag-0 and lag-1 versions
lag0 = merged.copy()
lag1 = merged.copy()
for col in ['buy_price', 'sell_price', 'buy_amount', 'sell_amount']:
    lag1[col] = lag1[col].shift(1)
lag1 = lag1.dropna(subset=['buy_price']).reset_index(drop=True)

# Align lag0 to same rows as lag1
lag0 = lag0.iloc[1:].reset_index(drop=True)

assert len(lag0) == len(lag1)

# Analyze fill differences
print("=" * 70)
print("FILL ANALYSIS: lag=0 vs lag=1 vs live-bot-approximation")
print("=" * 70)

buy_fills_lag1 = 0
buy_fills_lag0 = 0
buy_fills_either = 0
sell_fills_lag1 = 0
sell_fills_lag0 = 0
sell_fills_either = 0
sell_only_lag0 = 0
buy_only_lag0 = 0

for i in range(len(lag1)):
    high = float(lag1.iloc[i]['high'])
    low = float(lag1.iloc[i]['low'])

    # Lag-1 signals
    bp1 = float(lag1.iloc[i]['buy_price'] or 0)
    sp1 = float(lag1.iloc[i]['sell_price'] or 0)
    ba1 = float(lag1.iloc[i]['buy_amount'] or 0)
    sa1 = float(lag1.iloc[i]['sell_amount'] or 0)

    # Lag-0 signals (same bar)
    bp0 = float(lag0.iloc[i]['buy_price'] or 0)
    sp0 = float(lag0.iloc[i]['sell_price'] or 0)
    ba0 = float(lag0.iloc[i]['buy_amount'] or 0)
    sa0 = float(lag0.iloc[i]['sell_amount'] or 0)

    # Buy fills
    bf1 = ba1 > 0 and bp1 > 0 and low <= bp1 * (1 - FILL_BUF)
    bf0 = ba0 > 0 and bp0 > 0 and low <= bp0 * (1 - FILL_BUF)
    if bf1: buy_fills_lag1 += 1
    if bf0: buy_fills_lag0 += 1
    if bf1 or bf0: buy_fills_either += 1
    if bf0 and not bf1: buy_only_lag0 += 1

    # Sell fills
    sf1 = sa1 > 0 and sp1 > 0 and high >= sp1 * (1 + FILL_BUF)
    sf0 = sa0 > 0 and sp0 > 0 and high >= sp0 * (1 + FILL_BUF)
    if sf1: sell_fills_lag1 += 1
    if sf0: sell_fills_lag0 += 1
    if sf1 or sf0: sell_fills_either += 1
    if sf0 and not sf1: sell_only_lag0 += 1

total_bars = len(lag1)
print(f"Total bars: {total_bars}")
print(f"\n{'':>20} {'lag=1':>8} {'lag=0':>8} {'either':>8} {'only-lag0':>10}")
print(f"{'Buy fills':>20} {buy_fills_lag1:>8} {buy_fills_lag0:>8} {buy_fills_either:>8} {buy_only_lag0:>10}")
print(f"{'Sell fills':>20} {sell_fills_lag1:>8} {sell_fills_lag0:>8} {sell_fills_either:>8} {sell_only_lag0:>10}")

print(f"\nSells only-in-lag0: {sell_only_lag0} ({sell_only_lag0/total_bars*100:.1f}% of bars)")
print(f"Buys only-in-lag0:  {buy_only_lag0} ({buy_only_lag0/total_bars*100:.1f}% of bars)")
print(f"\nThese 'only-lag0' fills represent trades the LIVE BOT might catch")
print(f"(via intra-hour signal refresh) that the lag=1 sim misses.")

# Now run three simulators and compare PnL
def run_sim(name, use_lag1_signal, use_lag0_fallback=False):
    cash = INITIAL_CASH
    inv = 0.0
    eq = [INITIAL_CASH]
    buys, sells = 0, 0

    for i in range(len(lag1)):
        high = float(lag1.iloc[i]['high'])
        low = float(lag1.iloc[i]['low'])
        close = float(lag1.iloc[i]['close'])

        bp1 = float(lag1.iloc[i]['buy_price'] or 0)
        sp1 = float(lag1.iloc[i]['sell_price'] or 0)
        ba1 = float(lag1.iloc[i]['buy_amount'] or 0) / 100.0
        sa1 = float(lag1.iloc[i]['sell_amount'] or 0) / 100.0

        bp0 = float(lag0.iloc[i]['buy_price'] or 0)
        sp0 = float(lag0.iloc[i]['sell_price'] or 0)
        ba0 = float(lag0.iloc[i]['buy_amount'] or 0) / 100.0
        sa0 = float(lag0.iloc[i]['sell_amount'] or 0) / 100.0

        # Pick signal: lag1 primary, lag0 fallback
        if use_lag1_signal:
            bp, sp, ba, sa = bp1, sp1, ba1, sa1
            # Fallback to lag0 if lag1 wouldn't fill but lag0 would
            if use_lag0_fallback:
                sell_would_fill_1 = sa1 > 0 and sp1 > 0 and high >= sp1 * (1 + FILL_BUF)
                sell_would_fill_0 = sa0 > 0 and sp0 > 0 and high >= sp0 * (1 + FILL_BUF)
                buy_would_fill_1 = ba1 > 0 and bp1 > 0 and low <= bp1 * (1 - FILL_BUF)
                buy_would_fill_0 = ba0 > 0 and bp0 > 0 and low <= bp0 * (1 + FILL_BUF)
                if not sell_would_fill_1 and sell_would_fill_0:
                    sp, sa = sp0, sa0
                if not buy_would_fill_1 and buy_would_fill_0:
                    bp, ba = bp0, ba0
        else:
            bp, sp, ba, sa = bp0, sp0, ba0, sa0

        # Sell first
        if inv > 0 and sa > 0 and sp > 0 and high >= sp * (1 + FILL_BUF):
            sq = min(sa * inv, inv)
            if sq > 0:
                cash += sq * sp * (1 - MAKER_FEE)
                inv -= sq
                sells += 1

        # Buy
        if ba > 0 and bp > 0 and low <= bp * (1 - FILL_BUF):
            equity = cash + inv * close
            max_bv = max(equity, 0) - inv * bp
            if max_bv > 0:
                qty = ba * max_bv / (bp * (1 + MAKER_FEE))
                if qty > 0:
                    cash -= qty * bp * (1 + MAKER_FEE)
                    inv += qty
                    buys += 1

        eq.append(cash + inv * close)

    if inv > 0:
        cash += inv * float(lag1.iloc[-1]['close']) * (1 - MAKER_FEE)
        inv = 0
    eq[-1] = cash

    eq = np.array(eq)
    ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    neg = ret[ret < 0]
    sortino = (np.mean(ret) / (np.std(neg) + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0

    print(f"\n  {name}:")
    print(f"    Buys={buys} Sells={sells} Return={(cash/INITIAL_CASH-1)*100:.1f}% Sortino={sortino:.2f}")
    return sortino, (cash/INITIAL_CASH - 1) * 100

print("\n" + "=" * 70)
print("PnL COMPARISON: lag=1 vs lag=0 vs live-bot-approx")
print("=" * 70)

run_sim("lag=1 (current sim)", use_lag1_signal=True, use_lag0_fallback=False)
run_sim("lag=0 (sees current bar - too optimistic)", use_lag1_signal=False)
run_sim("live-bot-approx (lag=1 + lag=0 fallback)", use_lag1_signal=True, use_lag0_fallback=True)
