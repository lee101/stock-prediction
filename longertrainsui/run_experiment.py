#!/usr/bin/env python3
"""Longer training experiment for SUI - test if more epochs improve 7d PnL."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig

SYMBOL = "SUIUSDT"
MAKER_FEE = 0.001
EVAL_DAYS = 7


def run_backtest(checkpoint_path, test_frame, fee=MAKER_FEE):
    from binancechronossolexperiment.inference import load_policy_checkpoint

    model, normalizer, feature_columns, _ = load_policy_checkpoint(str(checkpoint_path))
    actions = generate_actions_from_frame(
        model=model,
        frame=test_frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=72,
        horizon=1,
    )

    config = SimulationConfig(maker_fee=fee, initial_cash=10000.0)
    sim = BinanceMarketSimulator(config)

    bars = test_frame.copy()
    if "timestamp" not in bars.columns and bars.index.name == "timestamp":
        bars = bars.reset_index()

    result = sim.run(bars, actions)
    eq = result.combined_equity

    ret = eq.pct_change().dropna()
    neg = ret[ret < 0]
    sortino = (ret.mean() / (neg.std() + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1

    return {
        "total_return": total_return,
        "sortino": sortino,
        "final_equity": eq.iloc[-1],
    }


def train_long(epochs: int, run_name: str):
    from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

    data_root = Path("trainingdatahourlybinance")
    forecast_cache = Path("binancechronossolexperiment/forecast_cache_sui_10bp")
    checkpoint_root = Path("longertrainsui/checkpoints") / run_name

    dm = ChronosSolDataModule(
        symbol=SYMBOL,
        data_root=data_root,
        forecast_cache_root=forecast_cache,
        forecast_horizons=(1, 4, 24),
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=72,
        split_config=SplitConfig(val_days=EVAL_DAYS, test_days=EVAL_DAYS),
        max_history_days=365,  # More history
        cache_only=True,
    )

    config = TrainingConfig(
        epochs=epochs,
        batch_size=64,
        sequence_length=72,
        learning_rate=3e-4,
        weight_decay=1e-4,
        optimizer_name="muon_mix",
        model_arch="nano",
        transformer_dim=512,  # Larger model
        transformer_layers=8,  # More layers
        transformer_heads=8,
        maker_fee=MAKER_FEE,
        checkpoint_root=checkpoint_root.parent,
        run_name=run_name,
        use_compile=False,
    )

    trainer = BinanceHourlyTrainer(config, dm)
    artifacts = trainer.train()

    ckpt = artifacts.best_checkpoint or sorted(checkpoint_root.glob("*.pt"))[-1]

    # Save packaged checkpoint
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": artifacts.state_dict,
        "config": asdict(config),
        "feature_columns": list(artifacts.feature_columns),
        "normalizer": artifacts.normalizer.to_dict(),
    }
    output_path = checkpoint_root / "policy_checkpoint.pt"
    torch.save(payload, output_path)

    return output_path, dm.test_frame, artifacts.history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"sui_long_{args.epochs}ep_{run_id}"

    logger.info(f"Training {args.epochs} epochs...")
    ckpt, test_frame, history = train_long(args.epochs, run_name)

    logger.info("Running backtest with 10bp fee...")
    result = run_backtest(ckpt, test_frame, fee=MAKER_FEE)

    print(f"\n{'='*60}")
    print(f"Longer Training Result ({args.epochs} epochs, 10bp fee)")
    print(f"{'='*60}")
    print(f"7d Return: {result['total_return']:.4f} ({result['total_return']*100:.2f}%)")
    print(f"Sortino: {result['sortino']:.2f}")
    print(f"Final Equity: ${result['final_equity']:.2f}")

    # Save results
    output = Path(f"longertrainsui/results_{run_name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "epochs": args.epochs,
        "run_name": run_name,
        "result": result,
        "history": [{"epoch": h.epoch, "val_sortino": h.val_sortino, "val_return": h.val_return}
                   for h in history],
    }, indent=2))
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    main()
