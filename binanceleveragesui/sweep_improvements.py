#!/usr/bin/env python3
"""Sweep training improvements for SUI margin policy near current best."""
from __future__ import annotations
import json, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost

FORECAST_CACHE = Path("binancechronossolexperiment/forecast_cache_sui_10bp")
DATA_ROOT = Path("trainingdatahourlybinance")
HORIZONS = (1, 4, 24)

EXPERIMENTS = [
    # Baseline (current best params)
    {"name": "baseline_rw012", "return_weight": 0.012, "lr_schedule": "none", "smoothness_penalty": 0.0},
    # Return weight sweep
    {"name": "rw008", "return_weight": 0.008},
    {"name": "rw010", "return_weight": 0.010},
    {"name": "rw014", "return_weight": 0.014},
    {"name": "rw016", "return_weight": 0.016},
    {"name": "rw020", "return_weight": 0.020},
    # Cosine LR schedule
    {"name": "cosine_rw012", "return_weight": 0.012, "lr_schedule": "cosine", "lr_min_ratio": 0.1},
    {"name": "cosine_rw012_min01", "return_weight": 0.012, "lr_schedule": "cosine", "lr_min_ratio": 0.01},
    # Linear warmdown
    {"name": "warmdown_rw012", "return_weight": 0.012, "lr_schedule": "linear_warmdown", "lr_warmdown_ratio": 0.3},
    # Smoothness penalty
    {"name": "smooth001_rw012", "return_weight": 0.012, "smoothness_penalty": 0.001},
    {"name": "smooth005_rw012", "return_weight": 0.012, "smoothness_penalty": 0.005},
    # Combined: cosine + smoothness
    {"name": "cosine_smooth001", "return_weight": 0.012, "lr_schedule": "cosine", "lr_min_ratio": 0.1, "smoothness_penalty": 0.001},
    # More epochs
    {"name": "ep35_rw012", "return_weight": 0.012, "epochs": 35},
    {"name": "ep40_rw012", "return_weight": 0.012, "epochs": 40},
    # Nano arch
    {"name": "nano_rw012", "return_weight": 0.012, "model_arch": "nano"},
    # Weight decay schedule
    {"name": "wd_linear_rw012", "return_weight": 0.012, "weight_decay_schedule": "linear", "weight_decay_end": 0.0},
]


def run_experiment(exp: dict, seed: int = 1337, val_days: int = 20, test_days: int = 10) -> dict:
    name = exp["name"]
    logger.info("=== {} ===", name)

    dm = ChronosSolDataModule(
        symbol="SUIUSDT", data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE, forecast_horizons=HORIZONS,
        context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32, model_id="amazon/chronos-t5-small",
        sequence_length=72, split_config=SplitConfig(val_days=val_days, test_days=test_days),
        cache_only=True,
    )

    checkpoint_root = Path(f"binanceleveragesui/checkpoints/sweep_{name}")
    tc = TrainingConfig(
        epochs=exp.get("epochs", 25),
        batch_size=16, sequence_length=72,
        learning_rate=exp.get("learning_rate", 1e-4),
        weight_decay=1e-4,
        return_weight=exp.get("return_weight", 0.012),
        smoothness_penalty=exp.get("smoothness_penalty", 0.0),
        seed=seed,
        transformer_dim=256, transformer_layers=4, transformer_heads=8,
        maker_fee=0.001,
        checkpoint_root=checkpoint_root,
        log_dir=Path("tensorboard_logs/binanceleveragesui_sweep"),
        use_compile=False,
        lr_schedule=exp.get("lr_schedule", "none"),
        lr_min_ratio=exp.get("lr_min_ratio", 0.0),
        lr_warmdown_ratio=exp.get("lr_warmdown_ratio", 0.5),
        model_arch=exp.get("model_arch", "classic"),
        weight_decay_schedule=exp.get("weight_decay_schedule", "none"),
        weight_decay_end=exp.get("weight_decay_end", 0.0),
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    sd = artifacts.state_dict
    if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
        ckpt = torch.load(artifacts.best_checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
    ckpt_path = checkpoint_root / "policy_checkpoint.pt"
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
        normalizer=normalizer, sequence_length=72, horizon=1,
    )
    test_start = dm.test_window_start
    bars = test_frame[test_frame["timestamp"] >= test_start].copy()
    actions = actions[actions["timestamp"] >= test_start].copy()

    result = {"name": name, "config": exp, "seed": seed}
    for lev in [1.0, 3.0, 5.0]:
        cfg = LeverageConfig(max_leverage=lev, initial_cash=5000.0)
        m = simulate_with_margin_cost(bars, actions, cfg)
        result[f"lev_{lev:.0f}x"] = m
        mult = m["final_equity"] / 5000
        logger.info("  {}x: {:.1f}x sortino={:.1f} dd={:.3f}", lev, mult, m["sortino"], m["max_drawdown"])

    result["best_epoch"] = artifacts.best_epoch if hasattr(artifacts, "best_epoch") else -1
    result["val_sortino"] = artifacts.best_val_sortino if hasattr(artifacts, "best_val_sortino") else 0.0
    return result


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"binanceleveragesui/sweep_results_{timestamp}.json")
    all_results = []

    for exp in EXPERIMENTS:
        try:
            result = run_experiment(exp)
            all_results.append(result)
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            logger.error("{} failed: {}", exp["name"], e)
            import traceback; traceback.print_exc()
            all_results.append({"name": exp["name"], "error": str(e)})

    logger.info("\n=== SUMMARY ===")
    for r in all_results:
        if "error" in r:
            logger.info("{}: FAILED - {}", r["name"], r["error"])
            continue
        m5 = r.get("lev_5x", {})
        m1 = r.get("lev_1x", {})
        logger.info("{}: 1x={:.1f}x 5x={:.1f}x sortino_5x={:.1f} dd_5x={:.3f}",
                     r["name"],
                     m1.get("final_equity", 0) / 5000,
                     m5.get("final_equity", 0) / 5000,
                     m5.get("sortino", 0),
                     m5.get("max_drawdown", 0))

    logger.info("Saved: {}", output_path)


if __name__ == "__main__":
    main()
