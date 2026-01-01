#!/usr/bin/env python3
"""Experiment with different model sizes to find optimal architecture.

Training budget: ~1 day
Inference: no constraint (can be compute-heavy)

Tests various model configurations and compares validation performance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv3timed.config import DailyTrainingConfigV3, DailyDatasetConfigV3
from neuraldailyv3timed.data import DailyDataModuleV3
from neuraldailyv3timed.trainer import NeuralDailyTrainerV3


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    dim: int
    layers: int
    heads: int
    kv_heads: int
    epochs: int
    batch_size: int
    lr: float = 0.0003

    @property
    def estimated_params(self) -> int:
        """Rough parameter count estimate."""
        # Transformer params: ~12 * dim^2 * layers (rough)
        # Plus embeddings and head
        return int(12 * self.dim * self.dim * self.layers + self.dim * 20 * 2 + self.dim * 5)


# Model configurations to test
MODEL_CONFIGS = [
    # Tiny - fast baseline
    ModelConfig("tiny", dim=128, layers=2, heads=4, kv_heads=2, epochs=100, batch_size=64),

    # Small - current default
    ModelConfig("small", dim=256, layers=4, heads=8, kv_heads=4, epochs=150, batch_size=32),

    # Medium - more capacity
    ModelConfig("medium", dim=384, layers=6, heads=8, kv_heads=4, epochs=150, batch_size=32),

    # Large - significant capacity
    ModelConfig("large", dim=512, layers=8, heads=8, kv_heads=4, epochs=200, batch_size=16),

    # XL - maximum capacity (compute-heavy inference OK)
    ModelConfig("xl", dim=768, layers=12, heads=12, kv_heads=6, epochs=200, batch_size=8),

    # Deep narrow - more layers, smaller dim
    ModelConfig("deep_narrow", dim=256, layers=12, heads=8, kv_heads=4, epochs=150, batch_size=32),

    # Wide shallow - fewer layers, larger dim
    ModelConfig("wide_shallow", dim=512, layers=4, heads=8, kv_heads=4, epochs=150, batch_size=16),

    # Attention heavy - more heads
    ModelConfig("attn_heavy", dim=384, layers=6, heads=16, kv_heads=8, epochs=150, batch_size=32),
]


def run_experiment(
    config: ModelConfig,
    dataset_config: DailyDatasetConfigV3,
    output_dir: Path,
    dry_run_steps: Optional[int] = None,
) -> Dict:
    """Run a single experiment with given model config."""

    run_name = f"exp_{config.name}_{datetime.now().strftime('%H%M%S')}"
    logger.info(f"=" * 60)
    logger.info(f"Running experiment: {config.name}")
    logger.info(f"  dim={config.dim}, layers={config.layers}, heads={config.heads}")
    logger.info(f"  estimated params: {config.estimated_params:,}")
    logger.info(f"  epochs={config.epochs}, batch_size={config.batch_size}")
    logger.info(f"=" * 60)

    start_time = time.time()

    # Create training config
    training_config = DailyTrainingConfigV3(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.lr,
        sequence_length=256,
        lookahead_days=20,

        # Model architecture
        transformer_dim=config.dim,
        transformer_layers=config.layers,
        transformer_heads=config.heads,
        transformer_kv_heads=config.kv_heads,

        # V3 settings
        max_hold_days=20,
        min_hold_days=1,
        forced_exit_slippage=0.001,
        min_exit_days=1.0,
        max_exit_days=20.0,

        # Loss weights
        return_weight=0.08,
        forced_exit_penalty=0.15,  # Slightly higher penalty
        risk_penalty=0.05,
        hold_time_penalty=0.02,

        # Simulation
        maker_fee=0.0008,
        equity_max_leverage=2.0,
        crypto_max_leverage=1.0,

        # Temperature
        initial_temperature=0.01,
        final_temperature=0.0001,
        temp_warmup_epochs=10,
        temp_anneal_epochs=min(150, config.epochs - 10),

        # Optimizer
        optimizer_name="dual",
        matrix_lr=0.02,
        embed_lr=0.2,
        head_lr=0.004,
        grad_clip=1.0,

        # Scheduler
        warmup_steps=100,
        use_cosine_schedule=True,

        # Run settings
        run_name=run_name,
        checkpoint_root=str(output_dir / "checkpoints"),
        use_compile=False,  # Skip compile for faster iteration
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,

        # Dry run
        dry_train_steps=dry_run_steps,

        # Dataset
        dataset=dataset_config,
    )

    # Create data module and trainer
    data_module = DailyDataModuleV3(dataset_config)
    trainer = NeuralDailyTrainerV3(training_config, data_module)

    # Get actual param count
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    logger.info(f"Actual params: {total_params:,} total, {trainable_params:,} trainable")

    # Train
    try:
        history = trainer.train()
        training_time = time.time() - start_time

        # Get best metrics
        if history["val_sharpe"]:
            best_idx = max(range(len(history["val_sharpe"])), key=lambda i: history["val_sharpe"][i])
            best_sharpe = history["val_sharpe"][best_idx]
            best_tp_rate = history["val_tp_rate"][best_idx]
            best_hold = history["val_avg_hold"][best_idx]
            best_epoch = best_idx + 1
        else:
            best_sharpe = float("-inf")
            best_tp_rate = 0.0
            best_hold = 0.0
            best_epoch = 0

        result = {
            "config_name": config.name,
            "dim": config.dim,
            "layers": config.layers,
            "heads": config.heads,
            "kv_heads": config.kv_heads,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "training_time_sec": training_time,
            "best_epoch": best_epoch,
            "best_val_sharpe": best_sharpe,
            "best_val_tp_rate": best_tp_rate,
            "best_val_avg_hold": best_hold,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "checkpoint_dir": str(trainer.checkpoint_dir),
            "success": True,
            "error": None,
        }

        logger.info(f"Experiment {config.name} complete!")
        logger.info(f"  Best Sharpe: {best_sharpe:.4f} (epoch {best_epoch})")
        logger.info(f"  Best TP Rate: {best_tp_rate:.2%}")
        logger.info(f"  Training time: {training_time/60:.1f} min")

    except Exception as e:
        logger.exception(f"Experiment {config.name} failed: {e}")
        result = {
            "config_name": config.name,
            "dim": config.dim,
            "layers": config.layers,
            "heads": config.heads,
            "success": False,
            "error": str(e),
        }

    return result


def print_results_table(results: List[Dict]):
    """Print results in a formatted table."""
    logger.info("\n" + "=" * 100)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 100)

    # Sort by best_val_sharpe
    successful = [r for r in results if r.get("success")]
    successful.sort(key=lambda x: x.get("best_val_sharpe", float("-inf")), reverse=True)

    header = f"{'Name':<15} {'Params':>10} {'Epochs':>6} {'Time':>8} {'Sharpe':>8} {'TP Rate':>8} {'Hold':>6}"
    logger.info(header)
    logger.info("-" * 100)

    for r in successful:
        name = r["config_name"]
        params = f"{r['total_params']/1e6:.1f}M"
        epochs = f"{r['best_epoch']}/{r['epochs']}"
        time_min = f"{r['training_time_sec']/60:.1f}m"
        sharpe = f"{r['best_val_sharpe']:.4f}"
        tp_rate = f"{r['best_val_tp_rate']:.1%}"
        hold = f"{r['best_val_avg_hold']:.1f}d"

        logger.info(f"{name:<15} {params:>10} {epochs:>6} {time_min:>8} {sharpe:>8} {tp_rate:>8} {hold:>6}")

    # Failed experiments
    failed = [r for r in results if not r.get("success")]
    if failed:
        logger.info("\nFailed experiments:")
        for r in failed:
            logger.info(f"  {r['config_name']}: {r.get('error', 'Unknown error')}")

    # Best model
    if successful:
        best = successful[0]
        logger.info(f"\nBEST MODEL: {best['config_name']}")
        logger.info(f"  Sharpe: {best['best_val_sharpe']:.4f}")
        logger.info(f"  Params: {best['total_params']:,}")
        logger.info(f"  Checkpoint: {best['checkpoint_dir']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Model size experiments")
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        help="Specific configs to run (default: all)")
    parser.add_argument("--dry-run", type=int, default=None,
                        help="Stop each experiment after N steps")
    parser.add_argument("--output-dir", type=str, default="experiments/model_sizes",
                        help="Output directory for results")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Symbols to train on (default: standard set)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which configs to run
    if args.configs:
        configs = [c for c in MODEL_CONFIGS if c.name in args.configs]
        if not configs:
            logger.error(f"No matching configs found. Available: {[c.name for c in MODEL_CONFIGS]}")
            sys.exit(1)
    else:
        configs = MODEL_CONFIGS

    logger.info(f"Running {len(configs)} experiments: {[c.name for c in configs]}")

    # Dataset config
    symbols = tuple(args.symbols) if args.symbols else (
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        "BTCUSD", "ETHUSD",
    )

    dataset_config = DailyDatasetConfigV3(
        symbols=symbols,
        sequence_length=256,
        lookahead_days=20,
        validation_days=40,
        min_history_days=300,
        include_weekly_features=True,
    )

    logger.info(f"Training on {len(symbols)} symbols: {symbols}")

    # Run experiments
    results = []
    for config in configs:
        result = run_experiment(
            config,
            dataset_config,
            output_dir,
            dry_run_steps=args.dry_run,
        )
        results.append(result)

        # Save intermediate results
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary
    print_results_table(results)

    # Save final results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
