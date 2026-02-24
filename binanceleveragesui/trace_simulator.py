#!/usr/bin/env python3
"""Trace through DOGE backtest bar-by-bar to audit simulator accuracy."""
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
FILL_BUF = 0.0013
MAKER_FEE = 0.001
INITIAL_CASH = 10000.0
DECISION_LAG = 1
MIN_EDGE = 0.0  # edge0 for full picture

dm = ChronosSolDataModule(
    symbol=SYMBOL,
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binanceneural/forecast_cache",
    forecast_horizons=(1,),
    context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32, model_id="amazon/chronos-t5-small",
    sequence_length=72,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True, max_history_days=365,
)

model, normalizer, feature_columns, meta = load_policy_checkpoint(CKPT, device="cuda")

test_frame = dm.test_frame.copy()
actions = generate_actions_from_frame(
    model=model, frame=test_frame, feature_columns=feature_columns,
    normalizer=normalizer, sequence_length=meta.get("sequence_length", 72),
    horizon=1,
)

bars = test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner")

if DECISION_LAG > 0:
    for col in ['buy_price', 'sell_price', 'buy_amount', 'sell_amount']:
        merged[col] = merged[col].shift(DECISION_LAG)
    merged = merged.dropna(subset=['buy_price']).reset_index(drop=True)

cash = INITIAL_CASH
inventory = 0.0
same_bar_roundtrips = 0
total_buys = 0
total_sells = 0
buys_when_flat = 0
sells_same_bar = 0
profit_from_same_bar = 0.0
profit_from_normal = 0.0

print(f"{'BAR':>4} {'TIMESTAMP':>20} {'OPEN':>8} {'HIGH':>8} {'LOW':>8} {'CLOSE':>8} | "
      f"{'BUY_P':>8} {'SELL_P':>8} {'B_AMT':>5} {'S_AMT':>5} | "
      f"{'ACTION':>10} {'QTY':>8} {'INV':>10} {'CASH':>10} {'EQUITY':>10}")
print("-" * 170)

# Only trace last 48 bars for readability, but count stats on all
for i, (_, row) in enumerate(merged.iterrows()):
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    buy_price = float(row.get("buy_price", 0) or 0)
    sell_price = float(row.get("sell_price", 0) or 0)
    buy_amount = float(row.get("buy_amount", 0) or 0) / 100.0
    sell_amount = float(row.get("sell_amount", 0) or 0) / 100.0

    inv_before = inventory
    action_str = ""
    bar_profit = 0.0

    edge = (sell_price - buy_price) / buy_price if buy_price > 0 and sell_price > 0 else 0.0
    if MIN_EDGE > 0 and edge < MIN_EDGE:
        action_str = "SKIP_EDGE"
    else:
        bought_this_bar = False
        buy_qty = 0
        # BUY
        if buy_amount > 0 and buy_price > 0 and low <= buy_price * (1 - FILL_BUF):
            equity = cash + inventory * close
            max_buy_value = 1.0 * max(equity, 0) - inventory * buy_price
            if max_buy_value > 0:
                buy_qty = buy_amount * max_buy_value / (buy_price * (1 + MAKER_FEE))
                if buy_qty > 0:
                    cost = buy_qty * buy_price * (1 + MAKER_FEE)
                    cash -= cost
                    inventory += buy_qty
                    total_buys += 1
                    if inv_before == 0:
                        buys_when_flat += 1
                    bought_this_bar = True
                    action_str = "BUY"

        # SELL
        if sell_amount > 0 and sell_price > 0 and high >= sell_price * (1 + FILL_BUF):
            if inventory > 0:
                sell_qty = min(sell_amount * inventory, inventory)
                if sell_qty > 0:
                    proceeds = sell_qty * sell_price * (1 - MAKER_FEE)
                    cash += proceeds
                    inventory -= sell_qty
                    total_sells += 1

                    if bought_this_bar:
                        same_bar_roundtrips += 1
                        sells_same_bar += 1
                        bar_profit = sell_qty * sell_price * (1 - MAKER_FEE) - sell_qty * buy_price * (1 + MAKER_FEE)
                        profit_from_same_bar += bar_profit
                        action_str = "BUY+SELL"
                    else:
                        bar_profit = sell_qty * (sell_price * (1 - MAKER_FEE) - (cash + inventory * close) / max(inventory + sell_qty, 1e-10))
                        profit_from_normal += bar_profit
                        action_str = "SELL"

    equity = cash + inventory * close

    # Print last 48 bars in detail
    if i >= len(merged) - 48:
        ts_str = str(row["timestamp"])[:19]
        print(f"{i:>4} {ts_str:>20} {row['open']:8.5f} {row['high']:8.5f} {row['low']:8.5f} {row['close']:8.5f} | "
              f"{buy_price:8.5f} {sell_price:8.5f} {buy_amount*100:5.1f} {sell_amount*100:5.1f} | "
              f"{action_str:>10} {'':>8} {inventory:10.2f} {cash:10.2f} {equity:10.2f}")

# Close remaining position
if inventory != 0:
    last_close = float(merged.iloc[-1]["close"])
    if inventory > 0:
        cash += inventory * last_close * (1 - MAKER_FEE)
    inventory = 0

final_equity = cash
total_return = (final_equity / INITIAL_CASH - 1) * 100

print("\n" + "=" * 80)
print("SIMULATOR AUDIT SUMMARY")
print("=" * 80)
print(f"Total bars:              {len(merged)}")
print(f"Total buys:              {total_buys}")
print(f"Total sells:             {total_sells}")
print(f"Buys when flat:          {buys_when_flat}")
print(f"Same-bar roundtrips:     {same_bar_roundtrips} ({same_bar_roundtrips/max(total_buys,1)*100:.1f}% of buys)")
print(f"Final equity:            ${final_equity:.2f}")
print(f"Total return:            {total_return:.1f}%")
print(f"Profit from same-bar RT: ${profit_from_same_bar:.2f}")
print()

# Now simulate with NO same-bar sells (more realistic - live bot needs 5min+ between entry and exit)
print("=" * 80)
print("REALISTIC SIM: No same-bar sells (buy bar N, earliest sell bar N+1)")
print("=" * 80)
cash2 = INITIAL_CASH
inv2 = 0.0
bought_bar2 = -99
buys2, sells2, same_blocked = 0, 0, 0
eq2 = [INITIAL_CASH]

for i, (_, row) in enumerate(merged.iterrows()):
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    buy_price = float(row.get("buy_price", 0) or 0)
    sell_price = float(row.get("sell_price", 0) or 0)
    buy_amount = float(row.get("buy_amount", 0) or 0) / 100.0
    sell_amount = float(row.get("sell_amount", 0) or 0) / 100.0

    edge = (sell_price - buy_price) / buy_price if buy_price > 0 and sell_price > 0 else 0.0

    # BUY (only when flat, like live bot)
    if inv2 == 0 and buy_amount > 0 and buy_price > 0 and low <= buy_price * (1 - FILL_BUF):
        equity = cash2
        max_buy_value = 1.0 * max(equity, 0)
        if max_buy_value > 0:
            buy_qty = buy_amount * max_buy_value / (buy_price * (1 + MAKER_FEE))
            if buy_qty > 0:
                cost = buy_qty * buy_price * (1 + MAKER_FEE)
                cash2 -= cost
                inv2 += buy_qty
                buys2 += 1
                bought_bar2 = i

    # SELL (only when in position AND not same bar as buy)
    if inv2 > 0 and i > bought_bar2 and sell_amount > 0 and sell_price > 0 and high >= sell_price * (1 + FILL_BUF):
        sell_qty = min(sell_amount * inv2, inv2)
        if sell_qty > 0:
            proceeds = sell_qty * sell_price * (1 - MAKER_FEE)
            cash2 += proceeds
            inv2 -= sell_qty
            sells2 += 1
    elif inv2 > 0 and i == bought_bar2 and sell_amount > 0 and sell_price > 0 and high >= sell_price * (1 + FILL_BUF):
        same_blocked += 1

    eq2.append(cash2 + inv2 * close)

if inv2 > 0:
    cash2 += inv2 * float(merged.iloc[-1]["close"]) * (1 - MAKER_FEE)
    inv2 = 0
eq2[-1] = cash2

eq2 = np.array(eq2)
ret2 = np.diff(eq2) / (np.abs(eq2[:-1]) + 1e-10)
neg2 = ret2[ret2 < 0]
sort2 = (np.mean(ret2) / (np.std(neg2) + 1e-10)) * np.sqrt(8760) if len(neg2) > 0 else 0

print(f"Buys:                    {buys2}")
print(f"Sells:                   {sells2}")
print(f"Same-bar sells blocked:  {same_blocked}")
print(f"Final equity:            ${cash2:.2f}")
print(f"Total return:            {(cash2/INITIAL_CASH - 1)*100:.1f}%")
print(f"Sortino:                 {sort2:.2f}")

# Also do strict live-bot style: entry when flat, exit when in position, one action per bar
print()
print("=" * 80)
print("STRICT LIVE BOT SIM: entry-only when flat, exit-only when in position")
print("=" * 80)
cash3 = INITIAL_CASH
inv3 = 0.0
buys3, sells3 = 0, 0
eq3 = [INITIAL_CASH]

for i, (_, row) in enumerate(merged.iterrows()):
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    buy_price = float(row.get("buy_price", 0) or 0)
    sell_price = float(row.get("sell_price", 0) or 0)
    buy_amount = float(row.get("buy_amount", 0) or 0) / 100.0
    sell_amount = float(row.get("sell_amount", 0) or 0) / 100.0

    # Flat -> try buy only
    if inv3 == 0:
        if buy_amount > 0 and buy_price > 0 and low <= buy_price * (1 - FILL_BUF):
            equity = cash3
            max_buy_value = 1.0 * max(equity, 0)
            if max_buy_value > 0:
                buy_qty = buy_amount * max_buy_value / (buy_price * (1 + MAKER_FEE))
                if buy_qty > 0:
                    cost = buy_qty * buy_price * (1 + MAKER_FEE)
                    cash3 -= cost
                    inv3 += buy_qty
                    buys3 += 1
    # In position -> try sell only
    elif inv3 > 0:
        if sell_amount > 0 and sell_price > 0 and high >= sell_price * (1 + FILL_BUF):
            sell_qty = min(sell_amount * inv3, inv3)
            if sell_qty > 0:
                proceeds = sell_qty * sell_price * (1 - MAKER_FEE)
                cash3 += proceeds
                inv3 -= sell_qty
                sells3 += 1

    eq3.append(cash3 + inv3 * close)

if inv3 > 0:
    cash3 += inv3 * float(merged.iloc[-1]["close"]) * (1 - MAKER_FEE)
    inv3 = 0
eq3[-1] = cash3

eq3 = np.array(eq3)
ret3 = np.diff(eq3) / (np.abs(eq3[:-1]) + 1e-10)
neg3 = ret3[ret3 < 0]
sort3 = (np.mean(ret3) / (np.std(neg3) + 1e-10)) * np.sqrt(8760) if len(neg3) > 0 else 0

print(f"Buys:                    {buys3}")
print(f"Sells:                   {sells3}")
print(f"Final equity:            ${cash3:.2f}")
print(f"Total return:            {(cash3/INITIAL_CASH - 1)*100:.1f}%")
print(f"Sortino:                 {sort3:.2f}")
