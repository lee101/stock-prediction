#!/usr/bin/env python3
"""Sweep trading parameters to optimize Sortino/Sharpe jointly."""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from binanceexp1.config import DatasetConfig, ExperimentConfig
from binanceexp1.data import BinanceExp1DataModule
from binanceexp1.sweep import apply_action_overrides
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig
from binanceneural.model import build_policy, policy_config_from_payload, align_state_dict_input_dim
from src.torch_load_utils import torch_load_compat

SYMBOL = "SOLUSD"
CHECKPOINT = "binanceneural/checkpoints/solusd_h1only_ft30_20260208/epoch_026.pt"

# Sweep parameters
INTENSITY_SCALES = [2.0, 3.0, 4.0, 5.0, 6.0]
PRICE_OFFSETS = [0.0, 0.0002, 0.0005, 0.001]
MAX_HOLD_HOURS = [4, 6, 8, 12, None]
MIN_EDGES = [0.0, 0.002, 0.004, 0.006]


def compute_sharpe(returns: np.ndarray, periods_per_year: float = 8760) -> float:
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret < 1e-10:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(periods_per_year))


def compute_sortino(returns: np.ndarray, periods_per_year: float = 8760) -> float:
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    downside = returns[returns < 0]
    if len(downside) < 1:
        return float("inf") if mean_ret > 0 else 0.0
    downside_std = np.std(downside)
    if downside_std < 1e-10:
        return float("inf") if mean_ret > 0 else 0.0
    return float(mean_ret / downside_std * np.sqrt(periods_per_year))


def run_simulation(
    actions: pd.DataFrame,
    frame: pd.DataFrame,
    intensity_scale: float,
    price_offset_pct: float,
    max_hold_hours: int | None,
    min_edge: float,
) -> dict:
    # Apply overrides
    adjusted = apply_action_overrides(
        actions,
        intensity_scale=intensity_scale,
        price_offset_pct=price_offset_pct,
    )

    # Filter by min edge if applicable
    if min_edge > 0 and "edge" in adjusted.columns:
        adjusted.loc[adjusted["edge"] < min_edge, ["buy_amount", "sell_amount", "trade_amount"]] = 0.0

    sim = BinanceMarketSimulator(
        SimulationConfig(
            initial_cash=10000.0,
            max_hold_hours=max_hold_hours,
        )
    )
    result = sim.run(frame, adjusted)
    metrics = result.metrics

    # Compute additional metrics
    equity_curve = result.combined_equity.values
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-8)
    else:
        returns = np.array([])

    sharpe = compute_sharpe(returns)
    sortino = float(metrics.get("sortino", 0.0))

    # Combined score: weight sortino more but include sharpe
    combined = 0.6 * sortino + 0.4 * sharpe

    return {
        "total_return": float(metrics["total_return"]),
        "sortino": sortino,
        "sharpe": sharpe,
        "combined": combined,
        "intensity_scale": intensity_scale,
        "price_offset_pct": price_offset_pct,
        "max_hold_hours": max_hold_hours,
        "min_edge": min_edge,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="strategy_state/sortino_sharpe_sweep.json")
    args = parser.parse_args()

    # Load data
    data_cfg = DatasetConfig(
        symbol=args.symbol,
        data_root=Path("trainingdatahourly/crypto"),
        sequence_length=96,
        validation_days=70,
        forecast_horizons=(1, 24),  # use cached only
        cache_only=True,
    )
    data = BinanceExp1DataModule(data_cfg)

    # Load model
    payload = torch_load_compat(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    input_dim = len(data.feature_columns)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)

    policy_cfg = policy_config_from_payload(
        payload.get("config", {}),
        input_dim=input_dim,
        state_dict=state_dict,
    )
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Generate base actions
    val_frame = data.val_dataset.frame
    print(f"Generating actions for {len(val_frame)} rows...")
    base_actions = generate_actions_from_frame(
        model=model,
        frame=val_frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=96,
        horizon=1,
    )

    # Sweep
    results = []
    total = len(INTENSITY_SCALES) * len(PRICE_OFFSETS) * len(MAX_HOLD_HOURS) * len(MIN_EDGES)
    print(f"Running {total} combinations...")

    for i, (intensity, offset, hold, edge) in enumerate(
        itertools.product(INTENSITY_SCALES, PRICE_OFFSETS, MAX_HOLD_HOURS, MIN_EDGES)
    ):
        try:
            result = run_simulation(
                base_actions.copy(),
                val_frame,
                intensity,
                offset,
                hold,
                edge,
            )
            results.append(result)
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{total}")
        except Exception as e:
            print(f"Error: {e}")

    # Sort by combined score
    results.sort(key=lambda x: x["combined"], reverse=True)

    print(f"\nTop {args.top_k} configurations:")
    for r in results[:args.top_k]:
        print(
            f"  combined={r['combined']:.2f} sortino={r['sortino']:.2f} sharpe={r['sharpe']:.2f} "
            f"return={r['total_return']:.4f} intensity={r['intensity_scale']} offset={r['price_offset_pct']} "
            f"hold={r['max_hold_hours']} edge={r['min_edge']}"
        )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "sweep_time": datetime.now().isoformat(),
            "symbol": args.symbol,
            "checkpoint": args.checkpoint,
            "top_results": results[:args.top_k],
            "all_results": results,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
