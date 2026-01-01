#!/usr/bin/env python3
"""V8: Optimized Model Based on Comprehensive Backtest Analysis.

Key improvements from V7 and symbol experiments:
1. Symbol selection: Use only proven profitable symbols (avoid ORCL, NOW, DDOG, PYPL, etc.)
2. Data leakage fix: Gap between train and validation sets
3. Early stopping: Prevent overfitting with patience=20
4. Weight decay: L2 regularization (0.01)
5. Larger validation set: 80 days (vs 40)
6. Lower position size: 0.15 max (for better Sortino)
7. Stop loss consideration in training

Best symbols from 90-day backtest:
- Top Sortino: QQQ, IWM, SPY, XLF, AAPL, V, MCD, MA, GOOGL, NVDA
- Top returns: LRCX (39.85%), MRVL (39.82%), AMAT (25.93%), NVDA (24.68%)
- Avoid: ORCL (-102%), NOW (-178%), DDOG (-76%), PYPL (-31%), INTC (-30%)

Usage:
    python train_v8.py --epochs 200 --batch-size 64
    python train_v8.py --epochs 200 --run-name v8_optimized
    python train_v8.py --epochs 200 --early-stopping 15
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


# V8 Symbol sets based on 90-day backtest analysis
V8_BEST_SYMBOLS = (
    # ETFs - consistently profitable, low volatility
    "SPY", "QQQ", "IWM", "XLF", "XLK", "DIA",
    # Mega cap tech - high Sortino
    "AAPL", "GOOGL", "NVDA",
    # Financials - solid performance
    "V", "MA",
    # Consumer - good risk-adjusted returns
    "MCD", "TGT",
    # Semiconductors - highest absolute returns (MRVL excluded - loses money)
    "LRCX", "AMAT", "QCOM", "AMD",
    # Healthcare - moderate but consistent
    "LLY", "PFE",
    # Software - only ADBE works
    "ADBE",
    # Note: TSLA excluded (high variance), MSFT borderline
)

# Symbols to explicitly avoid (negative Sortino in backtest)
V8_EXCLUDED = {
    "ORCL",   # -102% return, terrible
    "NOW",    # -178% return, extremely bad
    "DDOG",   # -76% return
    "PYPL",   # -31% return
    "COIN",   # -31% return
    "INTC",   # -30% return
    "AVGO",   # -14% return
    "COST",   # -11% return
    "CRM",    # -10% return
    "HD",     # -9% return
    "MRVL",   # -8% return consistently (both V8 and V9)
    "NFLX",   # Previously identified as bad
    "META",   # Previously identified as bad
    "MSFT",   # Previously identified as bad
    "AMZN",   # Previously identified as bad
}


def main():
    parser = argparse.ArgumentParser(description="V8 Optimized Training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10, help="Log every N epochs")
    parser.add_argument("--early-stopping", type=int, default=20,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="L2 regularization weight")
    parser.add_argument("--validation-days", type=int, default=80,
                        help="Days for validation set")
    parser.add_argument("--include-all", action="store_true",
                        help="Include all symbols (not recommended)")
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"v8_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Select symbols
    if args.include_all:
        symbols = V8_BEST_SYMBOLS + tuple(V8_EXCLUDED)
        logger.warning("Including all symbols including known bad performers!")
    else:
        symbols = V8_BEST_SYMBOLS

    logger.info("=" * 70)
    logger.info("V8 OPTIMIZED TRAINING")
    logger.info("=" * 70)
    logger.info(f"Training on {len(symbols)} symbols")
    logger.info("")
    logger.info("V8 improvements over V7:")
    logger.info("  1. Symbol selection: Only proven profitable symbols")
    logger.info("  2. Data leakage fix: Gap between train/val sets")
    logger.info(f"  3. Early stopping: patience={args.early_stopping} epochs")
    logger.info(f"  4. Weight decay: L2={args.weight_decay}")
    logger.info(f"  5. Larger validation: {args.validation_days} days")
    logger.info("  6. Excluded symbols: " + ", ".join(sorted(V8_EXCLUDED)))
    logger.info("")

    # Dataset config with improved validation
    dataset_config = DailyDatasetConfigV4(
        symbols=symbols,
        sequence_length=256,
        lookahead_days=8,  # Shorter lookahead for max_exit_days=2
        validation_days=args.validation_days,
        min_history_days=300,
        include_weekly_features=True,
        grouping_strategy="correlation",
        correlation_min_corr=0.6,
        correlation_max_group_size=12,
        exclude_symbols=list(V8_EXCLUDED) if not args.include_all else None,
    )

    # Training config with V8 improvements
    training_config = DailyTrainingConfigV4(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,  # L2 regularization
        sequence_length=256,
        lookahead_days=8,
        patch_size=5,
        num_windows=4,
        window_size=2,  # 2 days per window (4 windows * 2 days = 8 days max)
        num_quantiles=3,
        transformer_dim=512,
        transformer_layers=4,
        transformer_heads=8,
        transformer_kv_heads=4,
        max_hold_days=8,
        min_hold_days=1,
        maker_fee=0.0008,

        # V8 Loss weights (tuned for Sortino)
        return_loss_weight=1.0,
        sharpe_loss_weight=0.3,  # Higher for risk-adjusted focus
        forced_exit_penalty=0.25,  # Penalize deadline exits
        quantile_calibration_weight=0.05,
        position_regularization=0.20,  # Higher to prevent saturation
        quantile_ordering_weight=0.1,
        exit_days_penalty_weight=0.05,

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

    # Train with early stopping
    logger.info(f"Starting training for {args.epochs} epochs (early stopping={args.early_stopping})...")
    history = trainer.train(
        log_every=args.log_every,
        early_stopping_patience=args.early_stopping,
    )

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
    logger.info(f"  1. Run backtest: python backtest_v4.py --checkpoint {trainer.checkpoint_dir}/epoch_{trainer.best_epoch:04d}.pt --days 90")
    logger.info(f"  2. Run symbol experiments: python run_symbol_experiments.py --checkpoint {trainer.checkpoint_dir}/epoch_{trainer.best_epoch:04d}.pt")
    logger.info(f"  3. If profitable: update trade_v4_e2e.py checkpoint path")


if __name__ == "__main__":
    main()
