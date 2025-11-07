#!/usr/bin/env python3
"""
Experiment 2: Disable Crypto Fighting
Goal: Test if fighting detection hurts with only 3 crypto pairs
"""

import sys

sys.path.insert(0, "..")

from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from simulator import SimConfig, WorkStealingSimulator


class NoFightSimulator(WorkStealingSimulator):
    """Simulator that disables fighting for crypto-crypto steals."""

    def would_cause_fight(self, new_symbol: str, steal_from_symbol: str, timestamp) -> bool:
        # Never fight for crypto-crypto steals
        if new_symbol.endswith("USD") and steal_from_symbol.endswith("USD"):
            return False
        # Normal fighting logic for stocks
        return super().would_cause_fight(new_symbol, steal_from_symbol, timestamp)


def run_simulation_no_fight(config, hourly_data_dir, strategy_pnl_df) -> Dict[str, float]:
    sim = NoFightSimulator(hourly_data_dir, config)

    crypto_symbols = list(sim.crypto_data.keys())
    if not crypto_symbols:
        return {"total_pnl": 0, "sharpe": 0, "win_rate": 0, "trades": 0, "steals": 0, "blocks": 0}

    all_timestamps = sorted(set(ts for df in sim.crypto_data.values() for ts in df["timestamp"]))
    pnls = []
    last_fee_day = None

    for i, ts in enumerate(all_timestamps[::24]):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(all_timestamps) // 24}", end="\r", flush=True)

        if last_fee_day is None or (ts - last_fee_day).days >= 1:
            sim.apply_leverage_fees(ts)
            last_fee_day = ts

        available_cryptos = []
        for symbol in crypto_symbols:
            df = sim.crypto_data[symbol]
            row = df[df["timestamp"] == ts]
            if row.empty:
                continue

            current_price = row.iloc[0]["close"]
            perf = strategy_pnl_df[(strategy_pnl_df["symbol"] == symbol) & (strategy_pnl_df["is_crypto"] == True)]
            forecasted_pnl = perf["avg_pnl"].mean() if not perf.empty else 0
            available_cryptos.append((symbol, forecasted_pnl, current_price))

        available_cryptos.sort(key=lambda x: x[1], reverse=True)

        for rank, (symbol, forecasted_pnl, current_price) in enumerate(available_cryptos, 1):
            df = sim.crypto_data[symbol]
            row = df[df["timestamp"] == ts].iloc[0]

            limit_price = row["low"] * 1.001
            qty = 3500 / limit_price

            from simulator import SimOrder

            order = SimOrder(
                symbol=symbol,
                side="buy",
                limit_price=limit_price,
                current_price=current_price,
                qty=qty,
                forecasted_pnl=forecasted_pnl,
                timestamp=ts,
                crypto_rank=rank,
            )

            if sim.attempt_entry(order, ts):
                exit_idx = min(i + 7, len(all_timestamps[::24]) - 1)
                exit_ts = all_timestamps[::24][exit_idx] if exit_idx < len(all_timestamps[::24]) else all_timestamps[-1]
                exit_row = df[df["timestamp"] >= exit_ts]
                if not exit_row.empty:
                    exit_price = exit_row.iloc[0]["high"] * 0.999
                    pnl = sim.simulate_exit(order, exit_price)
                    pnls.append(pnl)

    if len(pnls) == 0:
        return {"total_pnl": 0, "sharpe": 0, "win_rate": 0, "trades": 0, "steals": 0, "blocks": 0}

    total_pnl = sum(pnls)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9) * np.sqrt(252)
    win_rate = len([p for p in pnls if p > 0]) / len(pnls)

    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trades": sim.trades_executed,
        "steals": sim.steals_performed,
        "blocks": sim.orders_blocked,
        "leverage_fees": sim.leverage_fees_paid,
        "net_pnl": total_pnl,
    }


def run_experiment():
    strategy_df = pd.read_parquet("full_strategy_dataset_20251101_211202_strategy_performance.parquet")

    def objective(params):
        config = SimConfig(
            crypto_ooh_force_count=int(params[0]),
            crypto_ooh_tolerance_pct=params[1],
            crypto_normal_tolerance_pct=params[2],
            entry_tolerance_pct=params[3],
            protection_pct=params[4],
            cooldown_seconds=int(params[5]),
            fight_threshold=int(params[6]),  # Still used for stocks
            fight_cooldown_seconds=int(params[7]),
        )

        results = run_simulation_no_fight(config, "../trainingdatahourly", strategy_df)

        score = (
            results["total_pnl"] * 0.5 + results["sharpe"] * 1000 + results["win_rate"] * 2000 - results["blocks"] * 10
        )

        print(f"\nConfig: {params}", flush=True)
        print(f"Results: {results}", flush=True)
        print(f"Score: {score}", flush=True)

        return -score

    bounds = [
        (2, 3),
        (0.02, 0.05),
        (0.008, 0.013),
        (0.005, 0.013),
        (0.0012, 0.0022),
        (120, 200),
        (5, 10),  # Fight threshold (only for stocks)
        (400, 1000),
    ]

    print("=" * 60)
    print("EXPERIMENT 2: NO CRYPTO FIGHTING")
    print("=" * 60)
    print("Fighting disabled for crypto-crypto steals")
    print("Baseline to beat: $19.1M PnL, 12 steals, 0 blocks")
    print("=" * 60)

    result = differential_evolution(
        objective, bounds, maxiter=12, popsize=4, seed=44, workers=1, updating="deferred", polish=False
    )

    optimal = SimConfig(
        crypto_ooh_force_count=int(result.x[0]),
        crypto_ooh_tolerance_pct=result.x[1],
        crypto_normal_tolerance_pct=result.x[2],
        entry_tolerance_pct=result.x[3],
        protection_pct=result.x[4],
        cooldown_seconds=int(result.x[5]),
        fight_threshold=int(result.x[6]),
        fight_cooldown_seconds=int(result.x[7]),
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 - OPTIMAL CONFIGURATION:")
    print("=" * 60)
    for field, value in optimal.__dict__.items():
        print(f"{field}: {value}")

    final_results = run_simulation_no_fight(optimal, "../trainingdatahourly", strategy_df)
    print("\nFINAL PERFORMANCE:")
    for metric, value in final_results.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    run_experiment()
