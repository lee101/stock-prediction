#!/usr/bin/env python3
"""Train a single-stock model with the existing BinanceHourlyTrainer.

Usage:
    python -u -m fastalgorithms.per_stock.train_per_stock --symbol NVDA --return-weight 0.15 --weight-decay 0.06
    python -u -m fastalgorithms.per_stock.train_per_stock --symbol TRIP --return-weight 0.10 --weight-decay 0.04

Reuses the same trainer/model/data infrastructure as train_bf16_efficient.py
but operates on a single symbol at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from loguru import logger

from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.data import MultiSymbolDataModule
from binanceneural.config import DatasetConfig, TrainingConfig
from src.trade_directions import resolve_trade_directions


def setup_bf16():
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU: {} ({:.1f} GB)", gpu_name, gpu_mem)


def main():
    parser = argparse.ArgumentParser(description="Per-stock model training")

    # Symbol
    parser.add_argument("--symbol", type=str, required=True, help="Single stock symbol")

    # Data paths
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.06)
    parser.add_argument("--return-weight", type=float, default=0.15)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)

    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Trading simulation
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0625)
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--fill-temperature", type=float, default=5e-4)
    parser.add_argument("--logits-softcap", type=float, default=12.0)

    # Loss function
    parser.add_argument("--loss-type", type=str, default="sortino",
                        choices=["sortino", "sharpe", "calmar", "log_wealth",
                                 "sortino_dd", "multiwindow", "multiwindow_dd"])

    # BF16
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")

    # Output
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("fastalgorithms/per_stock/checkpoints"))
    parser.add_argument("--forecast-horizons", type=str, default="1")

    args = parser.parse_args()
    setup_bf16()

    symbol = args.symbol.strip().upper()
    horizons = [int(h.strip()) for h in args.forecast_horizons.split(",")]

    # Resolve trade direction
    td = resolve_trade_directions(symbol, allow_short=True)
    dir_constraints = {symbol: (float(td.can_long), float(td.can_short))}
    direction_label = "SHORT" if td.can_short and not td.can_long else "LONG" if td.can_long and not td.can_short else "BOTH"

    logger.info("=== Per-Stock Training: {} ({}) ===", symbol, direction_label)
    logger.info("Architecture: h{} {}L {}H", args.hidden_dim, args.num_layers, args.num_heads)
    logger.info("Training: epochs={} bs={} lr={} wd={} rw={} seq={}",
                args.epochs, args.batch_size, args.lr, args.weight_decay,
                args.return_weight, args.sequence_length)

    # Load data for single symbol
    data_config = DatasetConfig(
        symbol=symbol,
        data_root=str(args.data_root),
        forecast_cache_root=str(args.cache_root),
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        val_fraction=0.15,
        min_history_hours=100,
        validation_days=30,
        cache_only=True,
    )

    t0 = time.perf_counter()
    data_module = MultiSymbolDataModule([symbol], data_config,
                                         directional_constraints=dir_constraints)
    logger.info("Data loaded in {:.1f}s ({} train samples, {} features)",
                time.perf_counter() - t0,
                len(data_module.train_dataset),
                len(data_module.feature_columns))

    # Checkpoint naming: SYMBOL_rwXX_wdXX_sXX
    rw_str = str(args.return_weight).replace(".", "")
    wd_str = str(args.weight_decay).replace(".", "")
    run_name = f"{symbol}_rw{rw_str}_wd{wd_str}_s{args.seed}"

    # Training config
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
        transformer_dropout=args.dropout,
        return_weight=args.return_weight,
        grad_clip=args.grad_clip,
        fill_temperature=args.fill_temperature,
        logits_softcap=args.logits_softcap,
        maker_fee=args.maker_fee,
        max_leverage=args.max_leverage,
        margin_annual_rate=args.margin_annual_rate,
        decision_lag_bars=args.decision_lag_bars,
        fill_buffer_pct=args.fill_buffer_pct,
        validation_use_binary_fills=True,
        use_amp=not args.no_bf16,
        amp_dtype="bfloat16",
        use_tf32=True,
        use_flash_attention=True,
        use_compile=not args.no_compile,
        seed=args.seed,
        loss_type=args.loss_type,
    )

    # Train
    t_train = time.perf_counter()
    trainer = BinanceHourlyTrainer(train_config, data_module)
    artifacts = trainer.train()
    train_time = time.perf_counter() - t_train

    logger.success("Training complete in {:.1f}s ({:.1f}s/epoch)",
                   train_time, train_time / args.epochs)

    if artifacts.history:
        last = artifacts.history[-1]
        logger.info("Last val: sortino={:.4f} return={:.4f}",
                     last.val_sortino or 0, last.val_return or 0)

    # Save config.json (same format as train_bf16_efficient.py)
    config_path = trainer.checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "stock_symbols": [symbol],
            "crypto_symbols": [],
            "symbol": symbol,
            "direction": direction_label,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sequence_length": args.sequence_length,
            "feature_columns": data_module.feature_columns,
            "transformer_dim": args.hidden_dim,
            "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers,
            "model_arch": "classic",
            "normalizer": data_module.normalizer.to_dict(),
            "return_weight": args.return_weight,
            "weight_decay": args.weight_decay,
            "decision_lag_bars": args.decision_lag_bars,
            "fill_temperature": args.fill_temperature,
            "logits_softcap": args.logits_softcap,
            "maker_fee": args.maker_fee,
            "max_leverage": args.max_leverage,
            "margin_annual_rate": args.margin_annual_rate,
            "fill_buffer_pct": args.fill_buffer_pct,
            "bf16_training": not args.no_bf16,
            "train_time_seconds": train_time,
        }, f, indent=2)

    # Save training meta
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
            "run_name": run_name,
            "symbol": symbol,
            "direction": direction_label,
            "epochs": args.epochs,
            "sequence_length": args.sequence_length,
            "transformer_dim": args.hidden_dim,
            "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers,
            "return_weight": args.return_weight,
            "weight_decay": args.weight_decay,
            "history": history_rows,
            "best_epoch": best.epoch if best else None,
            "best_val_sortino": best.val_sortino if best else None,
            "best_val_return": best.val_return if best else None,
            "best_checkpoint": str(artifacts.best_checkpoint) if artifacts.best_checkpoint else None,
            "train_time_seconds": train_time,
        }, f, indent=2)

    logger.info("Checkpoint: {}", trainer.checkpoint_dir)
    logger.info("Config: {}", config_path)
    logger.info("Meta: {}", meta_path)

    return trainer.checkpoint_dir


if __name__ == "__main__":
    main()
