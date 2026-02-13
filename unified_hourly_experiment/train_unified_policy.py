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
    args = parser.parse_args()

    stocks = [s.strip().upper() for s in args.stock_symbols.split(",") if s.strip()]
    cryptos = [s.strip().upper() for s in args.crypto_symbols.split(",") if s.strip()]

    logger.info("Training unified policy on {} stocks + {} crypto", len(stocks), len(cryptos))

    stock_config = DatasetConfig(
        symbol=stocks[0],
        data_root=str(args.stock_data_root),
        forecast_cache_root=str(args.stock_cache_root),
        forecast_horizons=[1, 24],
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
        forecast_horizons=[1, 24],
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
        run_name=args.run_name,
        warmup_steps=100,
        weight_decay=0.01,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=3,
        model_arch="gemma",
        preload_checkpoint_path=str(args.preload) if args.preload else None,
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
        }, f, indent=2)

    logger.info("Config saved to {}", config_path)


if __name__ == "__main__":
    main()
