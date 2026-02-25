#!/usr/bin/env python3
"""Train SUI policy with fill_buffer_pct=0.0012 (12 ticks).
Finetunes from existing rw016 checkpoint for faster convergence.
"""
from __future__ import annotations

import json, sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from loguru import logger
from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import (
    LeverageConfig, SUI_HOURLY_MARGIN_RATE, MAKER_FEE_10BP,
    simulate_with_margin_cost,
)

SYMBOL = "SUIUSDT"
DATA_ROOT = REPO / "trainingdatahourlybinance"
FORECAST_CACHE = REPO / "binancechronossolexperiment" / "forecast_cache_sui_10bp"
CHECKPOINT_ROOT = REPO / "binanceleveragesui" / "checkpoints"
FILL_BUFFER = 0.0012
RW_VALUES = [0.012, 0.014, 0.016, 0.018, 0.020, 0.025]
EPOCHS = 25
HORIZONS = (1, 4, 24)

PRELOAD_CKPT = REPO / "binanceleveragesui" / "checkpoints" / "sweep_rw016" / "policy_checkpoint.pt"


def train_and_eval(rw: float, preload: Path | None = None):
    tag = f"buf12_rw{str(rw).replace('.','')}"
    logger.info(f"\n=== Training {tag} (rw={rw}, buffer={FILL_BUFFER}) ===")

    dm = ChronosSolDataModule(
        symbol=SYMBOL,
        data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE,
        forecast_horizons=HORIZONS,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=72,
        split_config=SplitConfig(val_days=20, test_days=10),
        cache_only=True,
    )

    ckpt_root = CHECKPOINT_ROOT / f"sweep_{tag}"
    tc = TrainingConfig(
        epochs=EPOCHS,
        batch_size=16,
        sequence_length=72,
        learning_rate=1e-4,
        weight_decay=1e-4,
        return_weight=rw,
        seed=1337,
        transformer_dim=256,
        transformer_layers=4,
        transformer_heads=8,
        maker_fee=MAKER_FEE_10BP,
        fill_buffer_pct=FILL_BUFFER,
        checkpoint_root=ckpt_root,
        log_dir=Path("tensorboard_logs/binanceleveragesui"),
        use_compile=False,
        preload_checkpoint_path=preload,
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()

    ckpt_root.mkdir(parents=True, exist_ok=True)
    sd = artifacts.state_dict
    if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
        ckpt = torch.load(artifacts.best_checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
    ckpt_path = ckpt_root / "policy_checkpoint.pt"
    torch.save({
        "state_dict": sd,
        "config": asdict(tc),
        "feature_columns": list(artifacts.feature_columns),
        "normalizer": artifacts.normalizer.to_dict(),
    }, ckpt_path)

    model, normalizer, feature_columns, _ = load_policy_checkpoint(str(ckpt_path))
    test_frame = dm.test_frame.copy()
    actions = generate_actions_from_frame(
        model=model, frame=test_frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=72, horizon=HORIZONS[0],
    )
    test_start = dm.test_window_start
    bars = test_frame[test_frame["timestamp"] >= test_start].copy()
    actions_test = actions[actions["timestamp"] >= test_start].copy()

    row_results = {}
    for lev in [1.0, 3.0, 5.0]:
        for lag in [0, 1, 2]:
            for buf in [0.0, FILL_BUFFER]:
                lcfg = LeverageConfig(
                    max_leverage=lev, initial_cash=5000.0,
                    decision_lag_bars=lag,
                    fill_buffer_pct=buf,
                    margin_hourly_rate=SUI_HOURLY_MARGIN_RATE,
                    maker_fee=MAKER_FEE_10BP,
                )
                r = simulate_with_margin_cost(bars, actions_test, lcfg)
                key = f"{lev:.0f}x_lag{lag}_buf{int(buf*10000)}"
                row_results[key] = {
                    "return": r["total_return"] + 1,
                    "sortino": r["sortino"],
                    "max_dd": r["max_drawdown"],
                }
                logger.info(f"  {key}: {r['total_return']+1:.3f}x sort={r['sortino']:.1f} dd={r['max_drawdown']:.4f}")

    result = {"rw": rw, "buffer": FILL_BUFFER, "checkpoint": str(ckpt_path), "evals": row_results}
    out_path = REPO / "binanceleveragesui" / f"result_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    results = {}
    for i, rw in enumerate(RW_VALUES):
        preload = PRELOAD_CKPT if PRELOAD_CKPT.exists() else None
        r = train_and_eval(rw, preload=preload)
        results[f"buf12_rw{str(rw).replace('.','')}" ] = r

    summary_path = REPO / "binanceleveragesui" / "buffer_sweep_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll results saved to {summary_path}")

    best_tag = max(results, key=lambda t: results[t]["evals"].get("1.0x_lag1_buf12", {}).get("sortino", -999))
    best = results[best_tag]
    logger.info(f"\nBest at 1x lag=1 buf=12: {best_tag} -> sort={best['evals']['1.0x_lag1_buf12']['sortino']:.1f}")


if __name__ == "__main__":
    main()
