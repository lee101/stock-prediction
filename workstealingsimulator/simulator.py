#!/usr/bin/env python3
"""
Work stealing parameter optimizer using hourly crypto data and strategy performance.
Uses scipy.optimize to find optimal config values across different market patterns.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
from scipy.optimize import differential_evolution

EST = pytz.timezone("US/Eastern")


def is_nyse_open(dt: datetime) -> bool:
    dt_est = dt.astimezone(EST) if dt.tzinfo else EST.localize(dt)
    if dt_est.weekday() >= 5:
        return False
    hour = dt_est.hour
    minute = dt_est.minute
    if hour < 9 or hour >= 16:
        return False
    if hour == 9 and minute < 30:
        return False
    return True


@dataclass
class SimOrder:
    symbol: str
    side: str
    limit_price: float
    current_price: float
    qty: float
    forecasted_pnl: float
    timestamp: datetime
    crypto_rank: int

    @property
    def distance_pct(self) -> float:
        if self.side == "buy":
            return max(0, (self.current_price - self.limit_price) / self.limit_price)
        else:
            return max(0, (self.limit_price - self.current_price) / self.limit_price)

    @property
    def notional(self) -> float:
        return self.qty * self.limit_price


@dataclass
class SimConfig:
    crypto_ooh_force_count: int = 2
    crypto_ooh_tolerance_pct: float = 0.020
    crypto_normal_tolerance_pct: float = 0.0066
    entry_tolerance_pct: float = 0.0066
    protection_pct: float = 0.004
    cooldown_seconds: int = 120
    fight_threshold: int = 5
    fight_cooldown_seconds: int = 600
    max_leverage: float = 2.0
    capital: float = 10000.0


class WorkStealingSimulator:
    def __init__(self, hourly_data_dir: str, config: SimConfig):
        self.hourly_data_dir = Path(hourly_data_dir)
        self.config = config
        self.crypto_data: Dict[str, pd.DataFrame] = {}
        self.load_hourly_data()

        self.active_orders: List[SimOrder] = []
        self.capital_used = 0.0
        self.total_pnl = 0.0
        self.trades_executed = 0
        self.steals_performed = 0
        self.orders_blocked = 0
        self.steal_history: Dict[str, List[datetime]] = {}

    def load_hourly_data(self):
        for csv_file in self.hourly_data_dir.glob("*USD.csv"):
            symbol = csv_file.stem
            df = pd.read_csv(csv_file, parse_dates=["timestamp"])
            df = df.sort_values("timestamp")
            self.crypto_data[symbol] = df

    def get_entry_tolerance(self, symbol: str, crypto_rank: int, timestamp: datetime) -> float:
        is_ooh = not is_nyse_open(timestamp)
        if is_ooh and crypto_rank <= self.config.crypto_ooh_force_count:
            return float("inf")  # Force immediate
        if is_ooh:
            return self.config.crypto_ooh_tolerance_pct
        return self.config.crypto_normal_tolerance_pct

    def check_capacity(self) -> bool:
        max_capital = self.config.capital * self.config.max_leverage
        return self.capital_used < max_capital

    def is_protected(self, order: SimOrder, timestamp: datetime) -> bool:
        return order.distance_pct <= self.config.protection_pct

    def would_cause_fight(self, new_symbol: str, steal_from_symbol: str, timestamp: datetime) -> bool:
        key = f"{new_symbol}->{steal_from_symbol}"
        if key not in self.steal_history:
            return False
        recent_steals = [
            t for t in self.steal_history[key] if (timestamp - t).total_seconds() < self.config.fight_cooldown_seconds
        ]
        return len(recent_steals) >= self.config.fight_threshold

    def attempt_steal(self, new_order: SimOrder, timestamp: datetime) -> Optional[str]:
        candidates = sorted(self.active_orders, key=lambda o: (-o.distance_pct, o.forecasted_pnl))

        for candidate in candidates:
            if self.is_protected(candidate, timestamp):
                continue

            if self.would_cause_fight(new_order.symbol, candidate.symbol, timestamp):
                if new_order.forecasted_pnl <= candidate.forecasted_pnl:
                    continue

            self.active_orders.remove(candidate)
            self.capital_used -= candidate.notional
            self.steals_performed += 1

            key = f"{new_order.symbol}->{candidate.symbol}"
            if key not in self.steal_history:
                self.steal_history[key] = []
            self.steal_history[key].append(timestamp)

            return candidate.symbol

        return None

    def attempt_entry(self, order: SimOrder, timestamp: datetime) -> bool:
        tolerance = self.get_entry_tolerance(order.symbol, order.crypto_rank, timestamp)

        if order.distance_pct > tolerance:
            return False

        if self.check_capacity():
            self.active_orders.append(order)
            self.capital_used += order.notional
            return True

        stolen_from = self.attempt_steal(order, timestamp)
        if stolen_from:
            self.active_orders.append(order)
            self.capital_used += order.notional
            return True

        self.orders_blocked += 1
        return False

    def simulate_exit(self, order: SimOrder, exit_price: float) -> float:
        if order.side == "buy":
            pnl = (exit_price - order.limit_price) * order.qty
        else:
            pnl = (order.limit_price - exit_price) * order.qty

        self.active_orders.remove(order)
        self.capital_used -= order.notional
        self.total_pnl += pnl
        self.trades_executed += 1
        return pnl


def run_simulation(config: SimConfig, hourly_data_dir: str, strategy_pnl_df: pd.DataFrame) -> Dict[str, float]:
    sim = WorkStealingSimulator(hourly_data_dir, config)

    crypto_symbols = list(sim.crypto_data.keys())
    if not crypto_symbols:
        return {"total_pnl": 0, "sharpe": 0, "win_rate": 0, "trades": 0}

    all_timestamps = sorted(set(ts for df in sim.crypto_data.values() for ts in df["timestamp"]))

    pnls = []

    for i, ts in enumerate(all_timestamps[::168]):  # Every week to speed up
        if i % 10 == 0:
            print(f"Progress: {i}/{len(all_timestamps) // 168}", end="\r", flush=True)

        available_cryptos = []
        for symbol in crypto_symbols:
            df = sim.crypto_data[symbol]
            row = df[df["timestamp"] == ts]
            if row.empty:
                continue

            current_price = row.iloc[0]["close"]

            perf = strategy_pnl_df[(strategy_pnl_df["symbol"] == symbol) & (strategy_pnl_df["is_crypto"] == True)]
            if perf.empty:
                forecasted_pnl = 0
            else:
                forecasted_pnl = perf["avg_pnl"].mean()

            available_cryptos.append((symbol, forecasted_pnl, current_price))

        available_cryptos.sort(key=lambda x: x[1], reverse=True)

        for rank, (symbol, forecasted_pnl, current_price) in enumerate(available_cryptos[:5], 1):
            df = sim.crypto_data[symbol]
            row = df[df["timestamp"] == ts].iloc[0]

            limit_price = row["low"] * 1.001  # Simulate maxdiff entry
            qty = 1000 / limit_price  # Fixed $1000 position for more capacity pressure

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
                exit_idx = min(i + 1, len(all_timestamps[::168]) - 1)
                exit_ts = (
                    all_timestamps[::168][exit_idx] if exit_idx < len(all_timestamps[::168]) else all_timestamps[-1]
                )
                exit_row = df[df["timestamp"] >= exit_ts]
                if not exit_row.empty:
                    exit_price = exit_row.iloc[0]["high"] * 0.999
                    pnl = sim.simulate_exit(order, exit_price)
                    pnls.append(pnl)

    if len(pnls) == 0:
        return {"total_pnl": 0, "sharpe": 0, "win_rate": 0, "trades": 0}

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
    }


def optimize_params(hourly_data_dir: str, strategy_pnl_path: str):
    strategy_df = pd.read_parquet(strategy_pnl_path)

    def objective(params):
        config = SimConfig(
            crypto_ooh_force_count=int(params[0]),
            crypto_ooh_tolerance_pct=params[1],
            crypto_normal_tolerance_pct=params[2],
            entry_tolerance_pct=params[3],
            protection_pct=params[4],
            cooldown_seconds=int(params[5]),
            fight_threshold=int(params[6]),
            fight_cooldown_seconds=int(params[7]),
        )

        results = run_simulation(config, hourly_data_dir, strategy_df)

        score = (
            results["total_pnl"] * 0.5 + results["sharpe"] * 1000 + results["win_rate"] * 2000 - results["blocks"] * 10
        )

        print(f"\nConfig: {params}", flush=True)
        print(f"Results: {results}", flush=True)
        print(f"Score: {score}", flush=True)

        return -score  # Minimize negative score

    bounds = [
        (1, 3),  # crypto_ooh_force_count
        (0.01, 0.05),  # crypto_ooh_tolerance_pct
        (0.003, 0.015),  # crypto_normal_tolerance_pct
        (0.003, 0.015),  # entry_tolerance_pct
        (0.001, 0.01),  # protection_pct
        (60, 300),  # cooldown_seconds
        (3, 10),  # fight_threshold
        (300, 1800),  # fight_cooldown_seconds
    ]

    result = differential_evolution(
        objective, bounds, maxiter=10, popsize=4, seed=42, workers=1, updating="deferred", polish=False
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
    print("OPTIMAL CONFIGURATION:")
    print("=" * 60)
    for field, value in optimal.__dict__.items():
        print(f"{field}: {value}")

    final_results = run_simulation(optimal, hourly_data_dir, strategy_df)
    print("\nFINAL PERFORMANCE:")
    for metric, value in final_results.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    optimize_params(
        hourly_data_dir="../trainingdatahourly",
        strategy_pnl_path="full_strategy_dataset_20251101_211202_strategy_performance.parquet",
    )
