#!/usr/bin/env python3
"""Re-evaluate all sweep checkpoints with decision_lag_bars=0 and =1."""
from __future__ import annotations
import json, sys, time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from binanceleveragesui.run_leverage_sweep import (
    LeverageConfig, simulate_with_margin_cost, DEFAULT_MODEL_ID,
)
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.inference import generate_actions_from_frame

CHECKPOINTS = sorted(Path("binanceleveragesui/checkpoints").glob("sweep_*/policy_checkpoint.pt"))


def evaluate_checkpoint(ckpt_path: Path, lag: int, leverages=(1.0, 3.0, 5.0)) -> dict:
    name = ckpt_path.parent.name
    model, normalizer, feature_columns, _ = load_policy_checkpoint(str(ckpt_path))

    horizons = (1, 4, 24)
    dm = ChronosSolDataModule(
        symbol="SUIUSDT",
        data_root=Path("trainingdatahourlybinance"),
        forecast_cache_root=Path("binancechronossolexperiment/forecast_cache_sui_10bp"),
        forecast_horizons=horizons,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
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

    result = {"name": name, "lag": lag}
    for lev in leverages:
        cfg = LeverageConfig(max_leverage=lev, decision_lag_bars=lag)
        metrics = simulate_with_margin_cost(bars, actions, cfg)
        result[f"lev_{lev:.0f}x"] = metrics
    return result


def main():
    all_results = []
    for ckpt in CHECKPOINTS:
        name = ckpt.parent.name
        for lag in [0, 1]:
            logger.info("Evaluating {} lag={}", name, lag)
            r = evaluate_checkpoint(ckpt, lag)
            all_results.append(r)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"binanceleveragesui/reeval_lag_{ts}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'Checkpoint':<30} {'Lag':>3} {'1x Sort':>8} {'1x Ret':>8} {'1x DD':>8} {'5x Sort':>8} {'5x Ret':>8} {'5x DD':>8}")
    print("-" * 105)
    for r in all_results:
        l1 = r.get("lev_1x", {})
        l5 = r.get("lev_5x", {})
        print(f"{r['name']:<30} {r['lag']:>3} "
              f"{l1.get('sortino',0):>8.1f} {l1.get('total_return',0):>8.3f} {l1.get('max_drawdown',0):>8.4f} "
              f"{l5.get('sortino',0):>8.1f} {l5.get('total_return',0):>8.3f} {l5.get('max_drawdown',0):>8.4f}")

    logger.info("Saved: {}", out)


if __name__ == "__main__":
    main()
