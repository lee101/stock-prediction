#!/usr/bin/env python3
"""Train a single lag=1 experiment, evaluate, print results."""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rw", type=float, required=True)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--lag-range", type=str, default="")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-schedule", default="none")
    p.add_argument("--lr-min-ratio", type=float, default=0.0)
    p.add_argument("--smoothness-penalty", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--name", type=str, default=None)
    args = p.parse_args()

    name = args.name or f"lag{args.lag}_rw{str(args.rw).replace('.','')}"
    logger.info("=== Training {} (lag={}, rw={}) ===", name, args.lag, args.rw)

    dm = ChronosSolDataModule(
        symbol="SUIUSDT", data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE, forecast_horizons=HORIZONS,
        context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32, model_id="amazon/chronos-t5-small",
        sequence_length=72, split_config=SplitConfig(val_days=20, test_days=10),
        cache_only=True,
    )

    checkpoint_root = Path(f"binanceleveragesui/checkpoints/sweep_{name}")
    tc = TrainingConfig(
        epochs=args.epochs, batch_size=16, sequence_length=72,
        learning_rate=args.lr, weight_decay=1e-4,
        return_weight=args.rw, smoothness_penalty=args.smoothness_penalty,
        decision_lag_bars=args.lag,
        decision_lag_range=args.lag_range,
        seed=args.seed,
        transformer_dim=256, transformer_layers=4, transformer_heads=8,
        maker_fee=0.001, checkpoint_root=checkpoint_root,
        log_dir=Path("tensorboard_logs/binanceleveragesui_lag1"),
        use_compile=False,
        lr_schedule=args.lr_schedule, lr_min_ratio=args.lr_min_ratio,
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
        "state_dict": sd, "config": asdict(tc),
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

    result = {"name": name, "rw": args.rw, "lag": args.lag, "lag_range": args.lag_range}
    logger.info("\n=== RESULTS: {} ===", name)
    for lev in [1.0, 3.0, 5.0]:
        for lag in [0, 1, 2]:
            cfg = LeverageConfig(max_leverage=lev, initial_cash=5000.0, decision_lag_bars=lag)
            m = simulate_with_margin_cost(bars, actions, cfg)
            key = f"lev_{lev:.0f}x_lag{lag}"
            result[key] = m
            mult = m["final_equity"] / 5000
            logger.info("  {}x lag={}: {:.3f}x sort={:.1f} dd={:.3f}", lev, lag, mult, m["sortino"], m["max_drawdown"])

    out = Path(f"binanceleveragesui/result_{name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Checkpoint: {}", ckpt_path)
    logger.info("Saved: {}", out)


if __name__ == "__main__":
    main()
