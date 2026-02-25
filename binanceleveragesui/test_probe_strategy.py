#!/usr/bin/env python3
"""Test probe trading strategy: after a losing trade, switch to tiny $5 probe
trades until a probe is profitable, then resume full-size trading.

This reduces consecutive loss streaks and should improve Sortino.
"""
from __future__ import annotations
import json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from binanceleveragesui.run_leverage_sweep import LeverageConfig, SUI_HOURLY_MARGIN_RATE, MAKER_FEE_10BP
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.inference import generate_actions_from_frame


def simulate_with_probe(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: LeverageConfig,
    probe_amount: float = 5.0,
) -> dict:
    """Backtest with probe trading strategy.

    State machine:
    - Start in PROBE mode
    - PROBE: trade with probe_amount instead of full equity
    - If probe trade closes positive -> switch to FULL mode
    - FULL: trade with full equity (normal)
    - If any trade closes negative -> switch to PROBE mode
    """
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    actions = actions.sort_values("timestamp").reset_index(drop=True)
    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))

    if config.decision_lag_bars > 0:
        for col in ['buy_price', 'sell_price', 'buy_amount', 'sell_amount']:
            if col in merged.columns:
                merged[col] = merged[col].shift(config.decision_lag_bars)
        merged = merged.dropna(subset=['buy_price']).reset_index(drop=True)

    cash = config.initial_cash
    inventory = 0.0
    equity_curve = [cash]
    margin_cost_total = 0.0

    # Probe state
    probe_mode = True  # start in probe mode
    entry_equity = cash  # equity when position was opened
    has_position = False
    trade_results = []  # (pnl_pct, was_probe)
    mode_history = []  # track mode per bar

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

        fill_buf = config.fill_buffer_pct

        # Determine effective trading capital
        if probe_mode:
            effective_equity = probe_amount
        else:
            effective_equity = max(equity, 0)

        # Buy execution
        if buy_amount > 0 and buy_price > 0 and low <= buy_price * (1 - fill_buf):
            max_buy_value = config.max_leverage * effective_equity - inventory * buy_price
            if max_buy_value > 0:
                buy_qty = buy_amount * max_buy_value / (buy_price * (1 + config.maker_fee))
                if buy_qty > 0:
                    cost = buy_qty * buy_price * (1 + config.maker_fee)
                    cash -= cost
                    inventory += buy_qty
                    if not has_position:
                        entry_equity = equity
                        has_position = True

        # Sell execution
        if sell_amount > 0 and sell_price > 0 and high >= sell_price * (1 + fill_buf):
            if inventory > 0:
                sell_qty = min(sell_amount * inventory, inventory)
            elif config.can_short:
                max_short_value = config.max_leverage * effective_equity
                sell_qty = min(sell_amount * max_short_value / (sell_price * (1 + config.maker_fee)),
                               max_short_value / (sell_price * (1 + config.maker_fee)))
            else:
                sell_qty = 0
            sell_qty = max(sell_qty, 0)
            if sell_qty > 0:
                proceeds = sell_qty * sell_price * (1 - config.maker_fee)
                cash += proceeds
                inventory -= sell_qty

        # Check if position just closed
        new_equity = cash + inventory * close
        if has_position and abs(inventory) < 1e-10:
            trade_pnl = new_equity - entry_equity
            trade_pnl_pct = trade_pnl / (entry_equity + 1e-10)
            was_probe = probe_mode
            trade_results.append((trade_pnl_pct, was_probe, probe_mode))

            if trade_pnl >= 0 and probe_mode:
                probe_mode = False  # positive probe -> go full
            elif trade_pnl < 0:
                probe_mode = True  # any negative -> back to probe

            has_position = False

        mode_history.append(probe_mode)
        equity_curve.append(new_equity)

    # Close remaining
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

    n_probe = sum(1 for _, wp, _ in trade_results if wp)
    n_full = sum(1 for _, wp, _ in trade_results if not wp)
    probe_wins = sum(1 for pnl, wp, _ in trade_results if wp and pnl >= 0)
    full_wins = sum(1 for pnl, wp, _ in trade_results if not wp and pnl >= 0)
    probe_losses = sum(1 for pnl, wp, _ in trade_results if wp and pnl < 0)
    full_losses = sum(1 for pnl, wp, _ in trade_results if not wp and pnl < 0)

    return {
        "total_return": (eq[-1] / eq[0]) - 1 if eq[0] > 0 else 0,
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "final_equity": float(eq[-1]),
        "num_trades": len(trade_results),
        "probe_trades": n_probe,
        "full_trades": n_full,
        "probe_wins": probe_wins,
        "probe_losses": probe_losses,
        "full_wins": full_wins,
        "full_losses": full_losses,
        "margin_cost_total": margin_cost_total,
    }


def main():
    ckpt = "binanceleveragesui/checkpoints/sweep_rw016/policy_checkpoint.pt"
    model, normalizer, feature_columns, _ = load_policy_checkpoint(ckpt)

    dm = ChronosSolDataModule(
        symbol="SUIUSDT",
        data_root=Path("trainingdatahourlybinance"),
        forecast_cache_root=Path("binancechronossolexperiment/forecast_cache_sui_10bp"),
        forecast_horizons=(1, 4, 24),
        context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32, model_id="amazon/chronos-t5-small",
        sequence_length=72,
        split_config=SplitConfig(val_days=20, test_days=10),
        cache_only=True,
    )

    test_frame = dm.test_frame.copy()
    actions = generate_actions_from_frame(
        model=model, frame=test_frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=72, horizon=1,
    )
    test_start = dm.test_window_start
    bars = test_frame[test_frame["timestamp"] >= test_start].copy()
    actions = actions[actions["timestamp"] >= test_start].copy()

    from binanceleveragesui.run_leverage_sweep import simulate_with_margin_cost

    print(f"\n{'='*80}")
    print(f"PROBE TRADING STRATEGY TEST - checkpoint: {ckpt}")
    print(f"{'='*80}")

    for lev in [1.0]:
        for lag in [0, 1]:
            # Baseline (no probe)
            cfg = LeverageConfig(max_leverage=lev, initial_cash=5000.0, decision_lag_bars=lag)
            baseline = simulate_with_margin_cost(bars, actions, cfg)

            # Probe strategy with different probe amounts
            for probe_amt in [5.0, 10.0, 25.0, 50.0]:
                probe = simulate_with_probe(bars, actions, cfg, probe_amount=probe_amt)

                print(f"\n--- {lev:.0f}x lag={lag} probe=${probe_amt:.0f} ---")
                print(f"  Baseline: ret={baseline['total_return']:+.3f} sort={baseline['sortino']:.1f} dd={baseline['max_drawdown']:.4f} trades={baseline['num_trades']}")
                print(f"  Probe:    ret={probe['total_return']:+.3f} sort={probe['sortino']:.1f} dd={probe['max_drawdown']:.4f}")
                print(f"  Trades: {probe['num_trades']} total ({probe['probe_trades']} probe, {probe['full_trades']} full)")
                print(f"  Probe: {probe['probe_wins']}W/{probe['probe_losses']}L  Full: {probe['full_wins']}W/{probe['full_losses']}L")
                sort_delta = probe['sortino'] - baseline['sortino']
                dd_delta = probe['max_drawdown'] - baseline['max_drawdown']
                print(f"  Delta: sortino {sort_delta:+.1f}, dd {dd_delta:+.4f}")


if __name__ == "__main__":
    main()
