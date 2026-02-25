#!/usr/bin/env python3
"""Compare simulator execution orders: buy-first vs sell-first (live bot)."""
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
SYMBOL = "DOGEUSD"
FILL_BUF = 0.0005
MAKER_FEE = 0.001
INITIAL_CASH = 10000.0
DECISION_LAG = 1

dm = ChronosSolDataModule(
    symbol=SYMBOL,
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
if DECISION_LAG > 0:
    for col in ['buy_price', 'sell_price', 'buy_amount', 'sell_amount']:
        merged[col] = merged[col].shift(DECISION_LAG)
    merged = merged.dropna(subset=['buy_price']).reset_index(drop=True)


def run_sim(name, buy_first=True, allow_same_bar_sell=True, max_leverage=1.0):
    cash = INITIAL_CASH
    inv = 0.0
    eq = [INITIAL_CASH]
    buys, sells, same_bar_rt = 0, 0, 0

    for i, (_, row) in enumerate(merged.iterrows()):
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
        bp = float(row.get("buy_price", 0) or 0)
        sp = float(row.get("sell_price", 0) or 0)
        ba = float(row.get("buy_amount", 0) or 0) / 100.0
        sa = float(row.get("sell_amount", 0) or 0) / 100.0

        inv_before = inv
        bought_this = False

        def do_buy():
            nonlocal cash, inv, buys, bought_this
            if ba > 0 and bp > 0 and low <= bp * (1 - FILL_BUF):
                equity = cash + inv * close
                max_bv = max_leverage * max(equity, 0) - inv * bp
                if max_bv > 0:
                    qty = ba * max_bv / (bp * (1 + MAKER_FEE))
                    if qty > 0:
                        cash -= qty * bp * (1 + MAKER_FEE)
                        inv += qty
                        buys += 1
                        bought_this = True

        def do_sell():
            nonlocal cash, inv, sells, same_bar_rt
            if not allow_same_bar_sell and bought_this:
                return
            if sa > 0 and sp > 0 and high >= sp * (1 + FILL_BUF):
                if inv > 0:
                    qty = min(sa * inv, inv)
                    if qty > 0:
                        cash += qty * sp * (1 - MAKER_FEE)
                        inv -= qty
                        sells += 1
                        if bought_this:
                            same_bar_rt += 1

        if buy_first:
            do_buy()
            do_sell()
        else:
            do_sell()
            do_buy()

        eq.append(cash + inv * close)

    if inv > 0:
        cash += inv * float(merged.iloc[-1]["close"]) * (1 - MAKER_FEE)
        inv = 0
    eq[-1] = cash

    eq = np.array(eq)
    ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    neg = ret[ret < 0]
    sortino = (np.mean(ret) / (np.std(neg) + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0
    dd = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - dd) / (dd + 1e-10)))

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Buys:         {buys}")
    print(f"  Sells:        {sells}")
    print(f"  Same-bar RT:  {same_bar_rt}")
    print(f"  Return:       {(cash/INITIAL_CASH - 1)*100:.1f}%")
    print(f"  Sortino:      {sortino:.2f}")
    print(f"  Max DD:       {max_dd*100:.1f}%")
    print(f"  Final equity: ${cash:.2f}")
    return sortino, (cash/INITIAL_CASH - 1)*100


# 1. Original simulator: buy first, same-bar sells allowed
run_sim("ORIGINAL SIMULATOR (buy first, same-bar sells OK)", buy_first=True, allow_same_bar_sell=True)

# 2. Buy first, no same-bar sells
run_sim("BUY FIRST, NO SAME-BAR SELLS", buy_first=True, allow_same_bar_sell=False)

# 3. Sell first (live bot order), same-bar buys after sell OK
run_sim("SELL FIRST (live bot order), adds OK", buy_first=False, allow_same_bar_sell=True)

# 4. Sell first, no same-bar operations (strictest)
run_sim("SELL FIRST, NO SAME-BAR RT", buy_first=False, allow_same_bar_sell=False)

# Now also test with edge filter like sweep used
print("\n\n" + "#"*60)
print("# WITH MIN_EDGE=0.10 (10%) FILTER")
print("#"*60)

# Redo with edge filter applied
for edge_val in [0.0, 0.05, 0.10]:
    cash = INITIAL_CASH
    inv = 0.0
    eq = [INITIAL_CASH]
    buys, sells = 0, 0

    for i, (_, row) in enumerate(merged.iterrows()):
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
        bp = float(row.get("buy_price", 0) or 0)
        sp = float(row.get("sell_price", 0) or 0)
        ba = float(row.get("buy_amount", 0) or 0) / 100.0
        sa = float(row.get("sell_amount", 0) or 0) / 100.0

        edge = (sp - bp) / bp if bp > 0 and sp > 0 else 0.0
        if edge_val > 0 and edge < edge_val:
            eq.append(cash + inv * close)
            continue

        # SELL FIRST (live bot order)
        if sa > 0 and sp > 0 and high >= sp * (1 + FILL_BUF):
            if inv > 0:
                qty = min(sa * inv, inv)
                if qty > 0:
                    cash += qty * sp * (1 - MAKER_FEE)
                    inv -= qty
                    sells += 1

        if ba > 0 and bp > 0 and low <= bp * (1 - FILL_BUF):
            equity = cash + inv * close
            max_bv = 1.0 * max(equity, 0) - inv * bp
            if max_bv > 0:
                qty = ba * max_bv / (bp * (1 + MAKER_FEE))
                if qty > 0:
                    cash -= qty * bp * (1 + MAKER_FEE)
                    inv += qty
                    buys += 1

        eq.append(cash + inv * close)

    if inv > 0:
        cash += inv * float(merged.iloc[-1]["close"]) * (1 - MAKER_FEE)
        inv = 0
    eq[-1] = cash

    eq = np.array(eq)
    ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    neg = ret[ret < 0]
    sortino = (np.mean(ret) / (np.std(neg) + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0

    print(f"\n  SELL-FIRST, edge>={edge_val:.0%}: buys={buys} sells={sells} "
          f"return={(cash/INITIAL_CASH - 1)*100:.1f}% sortino={sortino:.2f}")
