#!/usr/bin/env python3
"""Evaluate existing checkpoint at different leverage levels with detailed metrics."""
from __future__ import annotations
import json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost

CHECKPOINT = "binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt"
model, normalizer, feature_columns, _ = load_policy_checkpoint(CHECKPOINT)

all_results = {}
windows = [
    (20, 3, "3d_test"),
    (20, 7, "7d_test"),
    (20, 10, "10d_test"),
    (20, 14, "14d_test"),
    (20, 30, "30d_test"),
    (20, 70, "70d_test"),
]

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
        m = simulate_with_margin_cost(bars, actions, cfg)
        mult = m['final_equity'] / 5000
        print(f"  {lev:.0f}x: {mult:.1f}x sortino={m['sortino']:.1f} dd={m['max_drawdown']:.3f} margin={m['margin_cost_pct']:.2f}% trades={m['num_trades']}")
        window_results[f"{lev:.0f}x"] = m

    # Also check 4x with 0-fee (like FDUSD)
    cfg_0fee = LeverageConfig(max_leverage=4.0, initial_cash=5000.0, maker_fee=0.0)
    m0 = simulate_with_margin_cost(bars, actions, cfg_0fee)
    print(f"  4x 0fee: {m0['final_equity']/5000:.1f}x sortino={m0['sortino']:.1f} dd={m0['max_drawdown']:.3f}")
    window_results["4x_0fee"] = m0

    all_results[label] = window_results

# Verify sim sanity: check equity never goes negative
print("\n=== SANITY CHECKS ===")
for label, wr in all_results.items():
    for lev_key, m in wr.items():
        if m['final_equity'] < 0:
            print(f"  WARNING: {label} {lev_key} final_equity={m['final_equity']:.2f} < 0")
        if m['max_drawdown'] < -0.5:
            print(f"  WARNING: {label} {lev_key} max_dd={m['max_drawdown']:.3f} > 50%")

with open("binanceleveragesui/eval_results_comprehensive.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved: binanceleveragesui/eval_results_comprehensive.json")
