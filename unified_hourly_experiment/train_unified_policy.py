#!/usr/bin/env python3
"""Train unified neural policy on stocks + crypto forecasts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.data import MultiSymbolDataModule
from binanceneural.config import DatasetConfig, TrainingConfig
from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-symbols", default="NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP")
    parser.add_argument("--crypto-symbols", default="SOLUSD,AVAXUSD,ETHUSD,UNIUSD")
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--crypto-data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--stock-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--crypto-cache-root", type=Path, default=Path("binanceneural/forecast_cache"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))
    parser.add_argument("--preload", type=Path, default=None, help="Preload checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None, help="Override stock symbols")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--forecast-horizons", type=str, default="1,24", help="Comma-sep horizons")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine", "linear_warmdown"])
    parser.add_argument("--lr-min-ratio", type=float, default=0.0)
    parser.add_argument("--return-weight", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fill-temperature", type=float, default=5e-4)
    parser.add_argument("--logits-softcap", type=float, default=12.0)
    parser.add_argument("--smoothness-penalty", type=float, default=0.0)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--use-residual-scalars", action="store_true")
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0)
    parser.add_argument("--decision-lag-bars", type=int, default=0)
    parser.add_argument("--decision-lag-range", type=str, default="")
    parser.add_argument("--market-order-entry", action="store_true")
    args = parser.parse_args()

    if args.symbols:
        stocks = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        stocks = [s.strip().upper() for s in args.stock_symbols.split(",") if s.strip()]
    cryptos = [s.strip().upper() for s in args.crypto_symbols.split(",") if s.strip()]

    horizons = [int(h.strip()) for h in args.forecast_horizons.split(",")]
    logger.info("Training unified policy on {} stocks + {} crypto, horizons={}", len(stocks), len(cryptos), horizons)

    stock_config = DatasetConfig(
        symbol=stocks[0],
        data_root=str(args.stock_data_root),
        forecast_cache_root=str(args.stock_cache_root),
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        val_fraction=0.15,
        min_history_hours=100,
        validation_days=30,
        cache_only=True,
    )

    crypto_config = DatasetConfig(
        symbol=cryptos[0] if cryptos else "SOLUSD",
        data_root=str(args.crypto_data_root),
        forecast_cache_root=str(args.crypto_cache_root),
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        val_fraction=0.1,
        cache_only=True,
    )

    # Load stock data module
    logger.info("Loading stock data for {} symbols", len(stocks))
    stock_data = MultiSymbolDataModule(stocks, stock_config)

    # Load crypto data module if symbols provided
    crypto_data = None
    if cryptos:
        logger.info("Loading crypto data for {} symbols", len(cryptos))
        try:
            crypto_data = MultiSymbolDataModule(cryptos, crypto_config)
        except Exception as e:
            logger.warning("Failed to load crypto data: {}", e)

    # Use stock data as primary (most symbols)
    data_module = stock_data

    logger.info("Training on {} features", len(data_module.feature_columns))
    logger.info("Train samples: {}", len(data_module.train_dataset))

    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.sequence_length,
        checkpoint_root=args.checkpoint_root,
        run_name=args.checkpoint_name or args.run_name,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        transformer_dim=args.hidden_dim,
        transformer_heads=args.num_heads,
        transformer_layers=args.num_layers,
        model_arch="gemma",
        transformer_dropout=args.dropout,
        lr_schedule=args.lr_schedule,
        lr_min_ratio=args.lr_min_ratio,
        return_weight=args.return_weight,
        grad_clip=args.grad_clip,
        fill_temperature=args.fill_temperature,
        logits_softcap=args.logits_softcap,
        smoothness_penalty=args.smoothness_penalty,
        feature_noise_std=args.feature_noise_std,
        use_residual_scalars=args.use_residual_scalars,
        max_leverage=args.max_leverage,
        margin_annual_rate=args.margin_annual_rate,
        decision_lag_bars=args.decision_lag_bars,
        decision_lag_range=args.decision_lag_range,
        market_order_entry=args.market_order_entry,
        preload_checkpoint_path=str(args.preload) if args.preload else None,
        use_compile=not args.no_compile,
        seed=args.seed,
    )

    trainer = BinanceHourlyTrainer(train_config, data_module)
    artifacts = trainer.train()

    logger.success("Training complete!")
    logger.info("Best checkpoint: {}", artifacts.best_checkpoint)
    logger.info("Final history: epochs={}", len(artifacts.history))

    if artifacts.history:
        last = artifacts.history[-1]
        logger.info("Last val sortino: {:.4f}, return: {:.4f}",
                   last.val_sortino or 0, last.val_return or 0)

    # Save config
    config_path = trainer.checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "stock_symbols": stocks,
            "crypto_symbols": cryptos,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sequence_length": args.sequence_length,
            "feature_columns": data_module.feature_columns,
            "transformer_dim": args.hidden_dim,
            "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers,
        }, f, indent=2)

    history_rows = []
    for row in artifacts.history:
        history_rows.append({
            "epoch": row.epoch,
            "train_loss": row.train_loss,
            "train_score": row.train_score,
            "train_sortino": row.train_sortino,
            "train_return": row.train_return,
            "val_loss": row.val_loss,
            "val_score": row.val_score,
            "val_sortino": row.val_sortino,
            "val_return": row.val_return,
        })
    best = max(artifacts.history, key=lambda h: h.val_sortino or float("-inf")) if artifacts.history else None
    meta_path = trainer.checkpoint_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "run_name": args.checkpoint_name or args.run_name,
            "symbols": stocks,
            "epochs": args.epochs,
            "sequence_length": args.sequence_length,
            "feature_columns": data_module.feature_columns,
            "decision_lag_bars": args.decision_lag_bars,
            "decision_lag_range": args.decision_lag_range,
            "smoothness_penalty": args.smoothness_penalty,
            "return_weight": args.return_weight,
            "fill_temperature": args.fill_temperature,
            "history": history_rows,
            "best_epoch": best.epoch if best else None,
            "best_val_sortino": best.val_sortino if best else None,
            "best_val_return": best.val_return if best else None,
            "best_checkpoint": str(artifacts.best_checkpoint) if artifacts.best_checkpoint else None,
        }, f, indent=2)

    logger.info("Config saved to {}", config_path)
    logger.info("Training metadata saved to {}", meta_path)


if __name__ == "__main__":
    main()
