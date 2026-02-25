#!/usr/bin/env python3
"""Train crypto hourly policy (BTC or ETH) for off-hours trading."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig, TrainingConfig
from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--cache-root", type=Path, default=Path("cryptoalpacaexperiment/forecast_cache"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("cryptoalpacaexperiment/checkpoints"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--return-weight", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--forecast-horizons", type=str, default="1,6")
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--fill-temperature", type=float, default=5e-4)
    parser.add_argument("--logits-softcap", type=float, default=12.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--loss-type", type=str, default="sortino")
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--preload", type=Path, default=None)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--smoothness-penalty", type=float, default=0.0)
    parser.add_argument("--dd-penalty", type=float, default=1.0)
    parser.add_argument("--validation-use-binary-fills", action="store_true")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    horizons = tuple(int(h.strip()) for h in args.forecast_horizons.split(","))
    run_name = args.run_name or f"{symbol}_lag{args.decision_lag_bars}_{args.num_layers}L_lr{args.lr}_wd{str(args.weight_decay).replace('.','')}_rw{str(args.return_weight).replace('.','')}_seq{args.sequence_length}"

    logger.info("Training {} policy, horizons={}, seq={}", symbol, horizons, args.sequence_length)

    data_config = DatasetConfig(
        symbol=symbol,
        data_root=str(args.data_root),
        forecast_cache_root=str(args.cache_root),
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        val_fraction=0.15,
        min_history_hours=100,
        validation_days=args.validation_days,
        cache_only=True,
    )
    data_module = BinanceHourlyDataModule(data_config)

    logger.info("{} features, {} train samples", len(data_module.feature_columns), len(data_module.train_dataset))

    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.sequence_length,
        checkpoint_root=args.checkpoint_root,
        run_name=run_name,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        transformer_dim=args.hidden_dim,
        transformer_heads=args.num_heads,
        transformer_layers=args.num_layers,
        model_arch="gemma",
        transformer_dropout=args.dropout,
        return_weight=args.return_weight,
        grad_clip=args.grad_clip,
        fill_temperature=args.fill_temperature,
        logits_softcap=args.logits_softcap,
        loss_type=args.loss_type,
        dd_penalty=args.dd_penalty,
        smoothness_penalty=args.smoothness_penalty,
        feature_noise_std=args.feature_noise_std,
        max_leverage=1.0,
        decision_lag_bars=args.decision_lag_bars,
        fill_buffer_pct=args.fill_buffer_pct,
        maker_fee=args.maker_fee,
        validation_use_binary_fills=args.validation_use_binary_fills,
        preload_checkpoint_path=str(args.preload) if args.preload else None,
        use_compile=not args.no_compile,
        seed=args.seed,
    )

    trainer = BinanceHourlyTrainer(train_config, data_module)
    artifacts = trainer.train()

    logger.success("Training complete: {}", run_name)
    logger.info("Best checkpoint: {}", artifacts.best_checkpoint)

    if artifacts.history:
        last = artifacts.history[-1]
        logger.info("Last val sortino={:.4f} return={:.4f}", last.val_sortino or 0, last.val_return or 0)

    config_path = trainer.checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "symbol": symbol,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sequence_length": args.sequence_length,
            "feature_columns": list(data_module.feature_columns),
            "transformer_dim": args.hidden_dim,
            "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers,
            "normalizer": data_module.normalizer.to_dict(),
            "decision_lag_bars": args.decision_lag_bars,
            "forecast_horizons": list(horizons),
            "fill_buffer_pct": args.fill_buffer_pct,
            "maker_fee": args.maker_fee,
        }, f, indent=2)

    history_rows = [
        {"epoch": h.epoch, "train_loss": h.train_loss, "train_sortino": h.train_sortino,
         "train_return": h.train_return, "val_loss": h.val_loss, "val_sortino": h.val_sortino,
         "val_return": h.val_return}
        for h in artifacts.history
    ]
    best = max(artifacts.history, key=lambda h: h.val_sortino or float("-inf")) if artifacts.history else None
    meta_path = trainer.checkpoint_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "run_name": run_name, "symbol": symbol, "epochs": args.epochs,
            "sequence_length": args.sequence_length, "decision_lag_bars": args.decision_lag_bars,
            "return_weight": args.return_weight, "weight_decay": args.weight_decay,
            "history": history_rows,
            "best_epoch": best.epoch if best else None,
            "best_val_sortino": best.val_sortino if best else None,
            "best_val_return": best.val_return if best else None,
            "best_checkpoint": str(artifacts.best_checkpoint) if artifacts.best_checkpoint else None,
        }, f, indent=2)

    logger.info("Config: {}, Meta: {}", config_path, meta_path)


if __name__ == "__main__":
    main()
