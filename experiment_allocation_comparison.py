#!/usr/bin/env python3
"""Compare learned allocation vs fixed allocation strategies.

Tests whether the model's learned position sizing outperforms fixed sizing.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from neuralhourlytradingv5.config import SimulationConfigV5
from neuralhourlytradingv5.data import FeatureNormalizer, HOURLY_FEATURES_V5
from neuralhourlytradingv5.model import HourlyCryptoPolicyV5


@dataclass
class AllocationStrategy:
    """Allocation strategy configuration."""
    name: str
    position_size: Optional[float]  # None = use model's learned allocation
    description: str


STRATEGIES = [
    AllocationStrategy("learned", None, "Model's learned allocation (0-100%)"),
    AllocationStrategy("fixed_100", 1.0, "Always 100% position"),
    AllocationStrategy("fixed_75", 0.75, "Always 75% position"),
    AllocationStrategy("fixed_50", 0.5, "Always 50% position"),
    AllocationStrategy("fixed_25", 0.25, "Always 25% position"),
]


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_config = checkpoint["config"]["policy"]
    model = HourlyCryptoPolicyV5(policy_config)

    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    feature_columns = checkpoint["feature_columns"]

    return model, normalizer, feature_columns


def run_backtest_with_strategy(
    model: HourlyCryptoPolicyV5,
    bars: pd.DataFrame,
    features: np.ndarray,
    strategy: AllocationStrategy,
    config: SimulationConfigV5,
    sequence_length: int = 168,
) -> Dict:
    """Run backtest with a specific allocation strategy."""
    device = next(model.parameters()).device

    cash = config.initial_cash
    inventory = 0.0
    equity_values = []
    timestamps = []
    trades = []
    current_position = None

    position_sizes_used = []

    for i in range(sequence_length, len(bars) - 24):
        current_bar = bars.iloc[i]
        current_ts = pd.to_datetime(current_bar["timestamp"])
        current_close = float(current_bar["close"])
        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])

        # Get model predictions
        feature_seq = features[i - sequence_length : i].copy()
        feature_tensor = (
            torch.from_numpy(feature_seq).unsqueeze(0).float().contiguous().to(device)
        )
        ref_close_tensor = torch.tensor([current_close], device=device)

        with torch.no_grad():
            outputs = model(feature_tensor)
            actions = model.get_hard_actions(outputs, ref_close_tensor)

        position_length = int(actions["position_length"].item())

        # Override position size if using fixed strategy
        if strategy.position_size is not None:
            position_size = strategy.position_size
        else:
            position_size = float(actions["position_size"].item())

        buy_price = float(actions["buy_price"].item())
        sell_price = float(actions["sell_price"].item())

        # Process existing position
        if current_position is not None:
            hours_held = current_position["hours_held"]
            target_hours = current_position["target_hours"]

            if current_high >= current_position["sell_price"]:
                exit_price = current_position["sell_price"]
                exit_value = inventory * exit_price * (1 - config.maker_fee)
                pnl = exit_value - current_position["entry_value"]

                trades.append({
                    "exit_type": "tp",
                    "pnl": pnl,
                    "return_pct": pnl / current_position["entry_value"],
                    "position_size": current_position["position_size"],
                })

                cash += exit_value
                inventory = 0.0
                current_position = None

            elif hours_held >= target_hours:
                exit_price = current_close * (1 - config.forced_exit_slippage)
                exit_value = inventory * exit_price * (1 - config.maker_fee)
                pnl = exit_value - current_position["entry_value"]

                trades.append({
                    "exit_type": "forced",
                    "pnl": pnl,
                    "return_pct": pnl / current_position["entry_value"],
                    "position_size": current_position["position_size"],
                })

                cash += exit_value
                inventory = 0.0
                current_position = None
            else:
                current_position["hours_held"] += 1

        # Check for new entry
        if current_position is None and position_length > 0:
            if current_low <= buy_price:
                max_spend = cash * position_size
                entry_value = max_spend / (1 + config.maker_fee)
                entry_cost = max_spend

                if entry_cost <= cash and entry_cost > 0:
                    inventory = entry_value / buy_price
                    cash -= entry_cost
                    position_sizes_used.append(position_size)

                    current_position = {
                        "entry_ts": current_ts,
                        "entry_price": buy_price,
                        "entry_value": entry_value,
                        "sell_price": sell_price,
                        "position_size": position_size,
                        "target_hours": position_length,
                        "hours_held": 0,
                    }

        portfolio_value = cash + inventory * current_close
        equity_values.append(portfolio_value)
        timestamps.append(current_ts)

    # Close remaining position
    if current_position is not None and inventory > 0:
        final_close = float(bars.iloc[-1]["close"])
        exit_value = inventory * final_close * (1 - config.maker_fee)
        pnl = exit_value - current_position["entry_value"]

        trades.append({
            "exit_type": "forced",
            "pnl": pnl,
            "return_pct": pnl / current_position["entry_value"],
            "position_size": current_position["position_size"],
        })

        cash += exit_value
        equity_values[-1] = cash

    # Compute metrics
    if equity_values:
        initial = config.initial_cash
        final = equity_values[-1]
        total_return = (final - initial) / initial

        returns = np.diff(equity_values) / np.clip(equity_values[:-1], 1e-8, None)
        mean_ret = returns.mean() if len(returns) else 0.0
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) else 1e-8
        sortino = mean_ret / downside_std * np.sqrt(24 * 365) if downside_std > 0 else 0.0
    else:
        total_return = 0.0
        sortino = 0.0
        final = config.initial_cash

    winning = [t for t in trades if t["pnl"] > 0]
    tp_trades = [t for t in trades if t["exit_type"] == "tp"]

    return {
        "strategy": strategy.name,
        "description": strategy.description,
        "total_return_pct": total_return * 100,
        "final_equity": final,
        "sortino": sortino,
        "num_trades": len(trades),
        "win_rate": len(winning) / len(trades) * 100 if trades else 0,
        "tp_rate": len(tp_trades) / len(trades) * 100 if trades else 0,
        "avg_position_size": np.mean(position_sizes_used) if position_sizes_used else 0,
        "pnl_dollars": final - config.initial_cash,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare allocation strategies")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--symbol", type=str, default="LINKUSD")
    parser.add_argument("--data-root", type=str, default="trainingdatahourly/crypto")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"ALLOCATION STRATEGY COMPARISON")
    print(f"Checkpoint: {Path(args.checkpoint).name}")
    print(f"Symbol: {args.symbol}")
    print(f"{'='*70}\n")

    # Load model
    model, normalizer, feature_columns = load_model(args.checkpoint, args.device)

    # Load data
    data_path = Path(args.data_root) / f"{args.symbol}.csv"
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add missing features
    for feat in HOURLY_FEATURES_V5:
        if feat not in df.columns:
            df[feat] = 0.0

    features = df[list(feature_columns)].values
    features_normalized = normalizer.transform(features)

    # Get validation period
    sequence_length = 168
    validation_hours = 240
    start_idx = max(0, len(df) - validation_hours - sequence_length - 24)

    val_bars = df.iloc[start_idx:].reset_index(drop=True)
    val_features = features_normalized[start_idx:]

    print(f"Validation: {val_bars['timestamp'].iloc[0]} to {val_bars['timestamp'].iloc[-1]}")
    print(f"Hours: {len(val_bars)}\n")

    config = SimulationConfigV5(
        initial_cash=args.initial_cash,
        maker_fee=0.0008,
        forced_exit_slippage=0.001,
    )

    # Run all strategies
    results = []
    for strategy in STRATEGIES:
        print(f"Testing: {strategy.name} - {strategy.description}")
        result = run_backtest_with_strategy(
            model, val_bars, val_features, strategy, config, sequence_length
        )
        results.append(result)

    # Print comparison table
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':<15} {'Return':>10} {'PnL':>12} {'Sortino':>10} {'Trades':>8} {'Win%':>8} {'TP%':>8} {'AvgPos':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r['strategy']:<15} {r['total_return_pct']:>+9.2f}% ${r['pnl_dollars']:>+10.2f} {r['sortino']:>10.2f} {r['num_trades']:>8} {r['win_rate']:>7.1f}% {r['tp_rate']:>7.1f}% {r['avg_position_size']:>7.1%}")

    print(f"{'='*70}\n")

    # Determine best strategy
    best = max(results, key=lambda x: x["total_return_pct"])
    print(f"Best strategy: {best['strategy']} with {best['total_return_pct']:+.2f}% return\n")

    # Compare learned vs best fixed
    learned = next(r for r in results if r["strategy"] == "learned")
    best_fixed = max([r for r in results if r["strategy"] != "learned"], key=lambda x: x["total_return_pct"])

    diff = learned["total_return_pct"] - best_fixed["total_return_pct"]
    print(f"Learned vs Best Fixed ({best_fixed['strategy']}):")
    print(f"  Learned: {learned['total_return_pct']:+.2f}%")
    print(f"  {best_fixed['strategy']}: {best_fixed['total_return_pct']:+.2f}%")
    print(f"  Difference: {diff:+.2f}%")

    return results


if __name__ == "__main__":
    main()
