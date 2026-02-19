#!/usr/bin/env python3
"""Eval latest 3d with probe-mode shutdown strategy."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost

CHECKPOINT = "binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt"
PROBE_AMOUNT_USD = 2.0


def simulate_with_probe_mode(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: LeverageConfig,
    probe_usd: float = PROBE_AMOUNT_USD,
) -> dict:
    """Sim with probe-mode: after unprofitable trade, shrink to $probe_usd orders
    until a profitable probe trade occurs, then re-enable full trading."""
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    actions = actions.sort_values("timestamp").reset_index(drop=True)
    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))

    cash = config.initial_cash
    inventory = 0.0
    equity_curve = [cash]
    trades = []
    margin_cost_total = 0.0

    probe_mode = False
    entry_equity = cash
    last_trade_was_buy = False
    probe_switches = 0
    full_trades = 0
    probe_trades = 0
    probe_wins = 0

    for _, row in merged.iterrows():
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        buy_price = float(row.get("buy_price", 0) or 0)
        sell_price = float(row.get("sell_price", 0) or 0)
        buy_amount = float(row.get("buy_amount", 0) or 0) / 100.0
        sell_amount = float(row.get("sell_amount", 0) or 0) / 100.0

        equity = cash + inventory * close

        if cash < 0:
            interest = abs(cash) * config.margin_hourly_rate
            cash -= interest
            margin_cost_total += interest
        if inventory < 0:
            borrowed_value = abs(inventory) * close
            interest = borrowed_value * config.margin_hourly_rate
            cash -= interest
            margin_cost_total += interest

        if probe_mode:
            # In probe mode: only place tiny orders
            if buy_amount > 0 and buy_price > 0 and low <= buy_price and inventory == 0:
                buy_qty = probe_usd / (buy_price * (1 + config.maker_fee))
                if buy_qty > 0:
                    cost = buy_qty * buy_price * (1 + config.maker_fee)
                    cash -= cost
                    inventory += buy_qty
                    entry_equity = cash + inventory * close
                    last_trade_was_buy = True
                    trades.append(("probe_buy", 0, buy_price, buy_qty))
                    probe_trades += 1

            if sell_amount > 0 and sell_price > 0 and high >= sell_price and inventory > 0:
                sell_qty = inventory
                proceeds = sell_qty * sell_price * (1 - config.maker_fee)
                cash += proceeds
                inventory -= sell_qty
                trades.append(("probe_sell", 0, sell_price, sell_qty))
                exit_equity = cash + inventory * close
                if exit_equity > entry_equity:
                    probe_mode = False
                    probe_wins += 1
        else:
            # Full trading mode
            if buy_amount > 0 and buy_price > 0 and low <= buy_price:
                max_buy_value = config.max_leverage * max(equity, 0) - inventory * buy_price
                if max_buy_value > 0:
                    buy_qty = buy_amount * max_buy_value / (buy_price * (1 + config.maker_fee))
                    if buy_qty > 0:
                        cost = buy_qty * buy_price * (1 + config.maker_fee)
                        if inventory == 0:
                            entry_equity = equity
                        cash -= cost
                        inventory += buy_qty
                        last_trade_was_buy = True
                        trades.append(("buy", 0, buy_price, buy_qty))
                        full_trades += 1

            if sell_amount > 0 and sell_price > 0 and high >= sell_price:
                if inventory > 0:
                    sell_qty = min(sell_amount * inventory, inventory)
                elif config.can_short:
                    max_short_value = config.max_leverage * max(equity, 0)
                    sell_qty = min(
                        sell_amount * max_short_value / (sell_price * (1 + config.maker_fee)),
                        max_short_value / (sell_price * (1 + config.maker_fee)),
                    )
                else:
                    sell_qty = 0
                sell_qty = max(sell_qty, 0)
                if sell_qty > 0:
                    proceeds = sell_qty * sell_price * (1 - config.maker_fee)
                    cash += proceeds
                    inventory -= sell_qty
                    trades.append(("sell", 0, sell_price, sell_qty))
                    full_trades += 1

                    if abs(inventory) < 1e-10:
                        exit_equity = cash
                        if exit_equity < entry_equity:
                            probe_mode = True
                            probe_switches += 1

        equity_curve.append(cash + inventory * close)

    if len(merged) > 0 and inventory != 0:
        last_close = float(merged.iloc[-1]["close"])
        if inventory > 0:
            cash += inventory * last_close * (1 - config.maker_fee)
        else:
            cash -= abs(inventory) * last_close * (1 + config.maker_fee)
        inventory = 0
        equity_curve[-1] = cash

    eq = np.array(equity_curve)
    ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    neg = ret[ret < 0]
    sortino = (np.mean(ret) / (np.std(neg) + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0
    running_max = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - running_max) / (running_max + 1e-10)))

    return {
        "total_return": (eq[-1] / eq[0]) - 1 if eq[0] > 0 else 0,
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "final_equity": float(eq[-1]),
        "num_trades": len(trades),
        "full_trades": full_trades,
        "probe_trades": probe_trades,
        "probe_switches": probe_switches,
        "probe_wins": probe_wins,
        "margin_cost_total": margin_cost_total,
        "margin_cost_pct": margin_cost_total / config.initial_cash * 100,
    }


def main():
    model, normalizer, feature_columns, _ = load_policy_checkpoint(CHECKPOINT)

    windows = [
        (15, 3, "3d"),
        (15, 7, "7d"),
        (15, 10, "10d"),
        (15, 14, "14d"),
        (15, 30, "30d"),
    ]

    all_results = {}
    for val_days, test_days, label in windows:
        try:
            dm = ChronosSolDataModule(
                symbol="SUIUSDT",
                data_root=Path("trainingdatahourlybinance"),
                forecast_cache_root=Path("binancechronossolexperiment/forecast_cache_sui_10bp"),
                forecast_horizons=(1, 4, 24),
                context_hours=512,
                quantile_levels=(0.1, 0.5, 0.9),
                batch_size=32,
                model_id="amazon/chronos-t5-small",
                sequence_length=72,
                split_config=SplitConfig(val_days=val_days, test_days=test_days),
                cache_only=True,
            )
        except Exception as e:
            print(f"\n=== {label}: SKIP ({e}) ===")
            continue

        test_frame = dm.test_frame.copy()
        actions = generate_actions_from_frame(
            model=model, frame=test_frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=72, horizon=1,
        )
        test_start = dm.test_window_start
        bars = test_frame[test_frame["timestamp"] >= test_start].copy()
        actions = actions[actions["timestamp"] >= test_start].copy()
        print(f"\n=== {label} ({len(bars)} bars, start={test_start}) ===")

        window_results = {}
        for lev in [1.0, 2.0, 3.0, 4.0, 5.0]:
            cfg = LeverageConfig(max_leverage=lev, initial_cash=5000.0)

            base = simulate_with_margin_cost(bars, actions, cfg)
            probe = simulate_with_probe_mode(bars, actions, cfg)

            bm = base['final_equity'] / 5000
            pm = probe['final_equity'] / 5000
            print(f"  {lev:.0f}x BASE: {bm:.2f}x sort={base['sortino']:.0f} dd={base['max_drawdown']:.4f} trades={base['num_trades']}")
            print(f"  {lev:.0f}x PROBE: {pm:.2f}x sort={probe['sortino']:.0f} dd={probe['max_drawdown']:.4f} trades={probe['num_trades']} probes={probe['probe_switches']} p_wins={probe['probe_wins']}")
            dd_improvement = (probe['max_drawdown'] - base['max_drawdown']) / abs(base['max_drawdown'] + 1e-10) * 100
            print(f"       DD change: {dd_improvement:+.1f}%  Return change: {(pm/bm - 1)*100:+.1f}%")

            window_results[f"{lev:.0f}x"] = {"base": base, "probe": probe}

        all_results[label] = window_results

    out = Path("binanceleveragesui/eval_3d_probe_results.json")
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
