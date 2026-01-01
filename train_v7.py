#!/usr/bin/env python3
"""V7: Optimized Model Based on Backtest Findings.

Key improvements from V4-V6 backtest analysis:
1. Exclude bad symbols: NFLX, META, MSFT, AMZN (all hurt performance)
2. Shorter lookahead (8 days vs 20) - matches optimal max_exit_days=2
3. Higher position_regularization (0.15 vs 0.01) - prevents 100% position saturation
4. Add quantile_ordering_weight (0.1) - enforce q10 < q50 < q90
5. Add exit_days_penalty_weight (0.05) - train for shorter holds

Backtest results showed:
- max_exit_days=2: 21.91% return, Sharpe 0.139 (vs 7.73%, 0.076 baseline)
- Excluding NFLX: +102% return improvement (NFLX was -102% from 4 trades)
- Excluding bad symbols: 54.84% total return, 0.612 Sharpe

Usage:
    python train_v7.py --epochs 200 --batch-size 64
    python train_v7.py --epochs 200 --run-name v7_optimized
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv4.config import DailyTrainingConfigV4, DailyDatasetConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.trainer import NeuralDailyTrainerV4


# V7 Optimized symbol list: excludes consistently poor performers
# Removed: NFLX (-102% in backtest), META, MSFT, AMZN (all negative in backtest)
V7_SYMBOLS = (
    # Major indices & ETFs (solid performance)
    "SPY", "QQQ",
    # Tech that works well with model
    "AAPL", "GOOGL", "NVDA", "TSLA",
    # Semiconductors (AMD was +28% in 90-day backtest)
    "AMD", "AVGO", "QCOM", "MU", "MRVL", "AMAT", "LRCX",
    # Cloud & SaaS
    "CRM", "ORCL", "SNOW", "PLTR", "NOW", "DDOG",
    # Fintech
    "V", "MA", "PYPL", "SQ",
    # Consumer
    "WMT", "HD", "NKE", "COST", "TGT",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV", "MRK",
    # Note: Crypto excluded by default (trade_crypto=False in runtime)
)


def main():
    parser = argparse.ArgumentParser(description="V7 Optimized Training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10, help="Log every N epochs")
    parser.add_argument("--include-bad-symbols", action="store_true",
                        help="Include NFLX, META, MSFT, AMZN (not recommended)")
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"v7_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Select symbols
    if args.include_bad_symbols:
        # Include all traditional symbols (not recommended)
        symbols = V7_SYMBOLS + ("NFLX", "META", "MSFT", "AMZN")
        logger.warning("Including bad symbols - expect lower performance!")
    else:
        symbols = V7_SYMBOLS

    logger.info("=" * 70)
    logger.info("V7 OPTIMIZED TRAINING")
    logger.info("=" * 70)
    logger.info(f"Training on {len(symbols)} symbols (excluding NFLX, META, MSFT, AMZN)")
    logger.info("")
    logger.info("Key V7 improvements:")
    logger.info("  1. Bad symbols excluded (NFLX was -102%, META/MSFT/AMZN negative)")
    logger.info("  2. Shorter lookahead (8 days) - matches optimal max_exit_days=2")
    logger.info("  3. Higher position_regularization (0.15) - prevents saturation")
    logger.info("  4. Quantile ordering loss - enforce q10 < q50 < q90")
    logger.info("  5. Exit days penalty - train for faster exits")
    logger.info("")

    # Dataset config with shorter lookahead
    dataset_config = DailyDatasetConfigV4(
        symbols=symbols,
        sequence_length=256,
        lookahead_days=8,  # Shorter lookahead (was 20)
        validation_days=40,
        min_history_days=300,
        include_weekly_features=True,
        grouping_strategy="correlation",
        correlation_min_corr=0.6,
        correlation_max_group_size=12,
    )

    # Training config with V7 improvements
    training_config = DailyTrainingConfigV4(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=256,
        lookahead_days=8,  # Matches dataset
        patch_size=5,
        num_windows=4,
        window_size=2,  # 2 days per window (4 windows * 2 days = 8 days max)
        num_quantiles=3,
        transformer_dim=512,
        transformer_layers=4,
        transformer_heads=8,
        transformer_kv_heads=4,
        max_hold_days=8,  # Shorter max hold (was 20)
        min_hold_days=1,
        maker_fee=0.0008,

        # V7 Loss weights (based on backtest findings)
        return_loss_weight=1.0,
        sharpe_loss_weight=0.2,  # Slightly higher for risk-adjusted focus
        forced_exit_penalty=0.2,  # Higher penalty for deadline exits
        quantile_calibration_weight=0.05,
        position_regularization=0.15,  # Much higher (was 0.01) - prevents saturation
        quantile_ordering_weight=0.1,  # NEW: enforce q10 < q50 < q90
        exit_days_penalty_weight=0.05,  # NEW: prefer shorter holds

        # Temperature annealing
        initial_temperature=0.01,
        final_temperature=0.0001,
        temp_warmup_epochs=10,
        temp_anneal_epochs=150,

        # Optimizer
        optimizer_name="dual",  # Muon + AdamW
        matrix_lr=0.02,
        embed_lr=0.2,
        head_lr=0.004,

        # Training
        run_name=args.run_name,
        checkpoint_root="neuraldailyv4/checkpoints",
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,
        use_cross_attention=True,
        dataset=dataset_config,
    )

    # Create data module
    logger.info("Loading data module...")
    data_module = DailyDataModuleV4(dataset_config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = NeuralDailyTrainerV4(training_config, data_module)

    # Log model info
    params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"Model parameters: {params:,}")
    logger.info(f"Feature columns: {len(data_module.feature_columns)}")
    logger.info("")

    # Train
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(log_every=args.log_every)

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best validation Sharpe: {trainer.best_val_sharpe:.4f}")
    logger.info(f"Best epoch: {trainer.best_epoch}")
    logger.info(f"Checkpoint dir: {trainer.checkpoint_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Run backtest: python backtest_v4.py --checkpoint {trainer.checkpoint_dir}/epoch_{trainer.best_epoch:04d}.pt")
    logger.info(f"  2. If profitable, deploy: update trade_v4_e2e.py checkpoint path")


if __name__ == "__main__":
    main()
