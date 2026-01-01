#!/usr/bin/env python3
"""Train neuraldailyv5 model with NEPA-style portfolio latent prediction.

Key features:
- NEPA: Sequence-of-latents prediction with cosine similarity loss
- Portfolio weights: Target allocations per asset instead of trade decisions
- Ramp-into-position: Gradual rebalancing simulation
- Sortino optimization: Focus on downside risk
- Atrous convolution: Long-range dependencies without full attention
- Multi-resolution: Aggregate across time scales

Usage:
    python train_v5.py
    python train_v5.py --epochs 200 --batch-size 32
    python train_v5.py --dry-run 100  # Quick test
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv5.config import DailyTrainingConfigV5, DailyDatasetConfigV5
from neuraldailyv5.data import DailyDataModuleV5
from neuraldailyv5.trainer import NeuralDailyTrainerV5


def parse_args():
    parser = argparse.ArgumentParser(description="Train V5 NEPA-style portfolio model")

    # Training params
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Base learning rate")
    parser.add_argument("--dry-run", type=int, default=None, help="Stop after N steps")

    # V5 specific params
    parser.add_argument("--latent-dim", type=int, default=64, help="Portfolio latent dimension")
    parser.add_argument("--use-nepa-loss", action="store_true", default=True, help="Use NEPA loss")
    parser.add_argument("--nepa-weight", type=float, default=0.1, help="NEPA loss weight")
    parser.add_argument("--use-atrous", action="store_true", default=True, help="Use atrous convolution")
    parser.add_argument("--use-multi-res", action="store_true", default=True, help="Use multi-resolution")

    # Model params (wide & shallow from experiments)
    parser.add_argument("--dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--patch-size", type=int, default=5, help="Patch size (days)")

    # Data params
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--lookahead", type=int, default=20, help="Lookahead days")
    parser.add_argument("--crypto-only", action="store_true", help="Train only on crypto")
    parser.add_argument("--equity-only", action="store_true", help="Train only on equities")

    # Loss weights
    parser.add_argument("--sortino-weight", type=float, default=0.2, help="Sortino loss weight")
    parser.add_argument("--turnover-penalty", type=float, default=0.05, help="Turnover penalty")

    # Run params
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N epochs")

    return parser.parse_args()


def main():
    args = parse_args()

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"v5_nepa_{timestamp}"

    logger.info(f"Starting V5 NEPA training run: {args.run_name}")
    logger.info(f"Latent dim: {args.latent_dim}, NEPA loss: {args.use_nepa_loss}, Atrous: {args.use_atrous}")

    # Select symbols
    if args.crypto_only:
        symbols = ("BTCUSD", "ETHUSD")
        logger.info("Training on crypto only")
    elif args.equity_only:
        symbols = (
            "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        )
        logger.info("Training on equities only")
    else:
        symbols = (
            "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
            "BTCUSD", "ETHUSD",
        )

    logger.info(f"Training on {len(symbols)} symbols: {symbols}")

    # Dataset config
    dataset_config = DailyDatasetConfigV5(
        symbols=symbols,
        sequence_length=args.seq_len,
        lookahead_days=args.lookahead,
        validation_days=40,
        min_history_days=300,
        include_weekly_features=True,
        crypto_only=args.crypto_only,
    )

    # Training config
    training_config = DailyTrainingConfigV5(
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.seq_len,
        lookahead_days=args.lookahead,

        # V5 specific
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        use_nepa_loss=args.use_nepa_loss,
        nepa_loss_weight=args.nepa_weight,
        use_atrous_conv=args.use_atrous,
        use_multi_resolution=args.use_multi_res,

        # Model (wide & shallow)
        transformer_dim=args.dim,
        transformer_layers=args.layers,
        transformer_heads=args.heads,
        transformer_kv_heads=args.heads // 2,

        # Simulation
        equity_max_leverage=2.0,
        crypto_max_leverage=1.0,
        max_single_asset_weight=0.5,
        rebalance_threshold=0.02,

        # Loss weights
        return_loss_weight=1.0,
        sortino_loss_weight=args.sortino_weight,
        turnover_penalty=args.turnover_penalty,
        concentration_penalty=0.05,
        volatility_calibration_weight=0.05,

        # Temperature
        initial_temperature=0.01,
        final_temperature=0.0001,
        temp_warmup_epochs=10,
        temp_anneal_epochs=min(150, args.epochs - 10),

        # Optimizer (dual from nanochat)
        optimizer_name="dual",
        matrix_lr=0.02,
        embed_lr=0.2,
        head_lr=0.004,
        grad_clip=1.0,

        # Run settings
        run_name=args.run_name,
        checkpoint_root="neuraldailyv5/checkpoints",
        device=args.device,
        use_compile=args.compile,
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,

        # Dry run
        dry_train_steps=args.dry_run,

        # Resume
        preload_checkpoint_path=args.resume,

        # Dataset
        dataset=dataset_config,
    )

    # Create data module
    logger.info("Preparing data module...")
    data_module = DailyDataModuleV5(dataset_config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = NeuralDailyTrainerV5(training_config, data_module)

    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
    logger.info(f"Number of assets: {data_module.num_assets}")

    # Training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    try:
        history = trainer.train(log_every=args.log_every)

        # Print final summary
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)

        if history["val_sortino"]:
            best_idx = max(range(len(history["val_sortino"])), key=lambda i: history["val_sortino"][i])
            logger.info(f"Best validation Sortino: {history['val_sortino'][best_idx]:.4f} (epoch {best_idx + 1})")
            logger.info(f"Best Sharpe: {history['val_sharpe'][best_idx]:.4f}")
            logger.info(f"Best Return: {history['val_return'][best_idx]:.4%}")
            logger.info(f"Turnover: {history['val_turnover'][best_idx]:.2%}")

        logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
