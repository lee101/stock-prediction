#!/usr/bin/env python3
"""Efficient training: BF16 split AMP + vectorized sim + torch.compile.

Same interface as train_unified_policy.py with efficiency optimizations enabled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.data import MultiSymbolDataModule
from binanceneural.config import DatasetConfig, TrainingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT")
    parser.add_argument("--crypto-symbols", type=str, default="")
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--crypto-data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--stock-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--crypto-cache-root", type=Path, default=Path("binanceneural/forecast_cache"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--return-weight", type=float, default=0.15)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--model-arch", type=str, default="classic")
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0625)
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    parser.add_argument("--fill-temperature", type=float, default=5e-4)
    parser.add_argument("--logits-softcap", type=float, default=12.0)
    parser.add_argument("--loss-type", type=str, default="sortino")
    parser.add_argument("--forecast-horizons", type=str, default="1,24")
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-vectorized-sim", action="store_true")
    parser.add_argument("--no-split-amp", action="store_true")
    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU: {} ({:.1f}GB) BF16={}", gpu, mem, torch.cuda.is_bf16_supported())

    stocks = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(h.strip()) for h in args.forecast_horizons.split(",")]

    use_bf16 = not args.no_bf16
    use_split = not args.no_split_amp and use_bf16
    use_vsim = not args.no_vectorized_sim
    use_compile = not args.no_compile

    logger.info("Efficient training: bf16={} split_amp={} vectorized_sim={} compile={}",
                use_bf16, use_split, use_vsim, use_compile)
    logger.info("Symbols: {} | arch: h{} {}L {}H seq={}",
                ",".join(stocks), args.hidden_dim, args.num_layers, args.num_heads, args.sequence_length)

    stock_config = DatasetConfig(
        symbol=stocks[0], data_root=str(args.stock_data_root),
        forecast_cache_root=str(args.stock_cache_root),
        forecast_horizons=horizons, sequence_length=args.sequence_length,
        val_fraction=0.15, min_history_hours=100, validation_days=30, cache_only=True,
    )
    data = MultiSymbolDataModule(stocks, stock_config)
    logger.info("Data: {} train samples, {} features", len(data.train_dataset), len(data.feature_columns))

    if args.checkpoint_name is None:
        args.checkpoint_name = f"efficient_wd{str(args.weight_decay).replace('.','')}_rw{str(args.return_weight).replace('.','')}_s{args.seed}"

    train_config = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        sequence_length=args.sequence_length, checkpoint_root=args.checkpoint_root,
        run_name=args.checkpoint_name, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, transformer_dim=args.hidden_dim,
        transformer_heads=args.num_heads, transformer_layers=args.num_layers,
        model_arch=args.model_arch, return_weight=args.return_weight,
        grad_clip=args.grad_clip, fill_temperature=args.fill_temperature,
        logits_softcap=args.logits_softcap, maker_fee=args.maker_fee,
        max_leverage=args.max_leverage, margin_annual_rate=args.margin_annual_rate,
        fill_buffer_pct=args.fill_buffer_pct,
        validation_use_binary_fills=True,
        loss_type=args.loss_type, seed=args.seed,
        use_amp=use_bf16, amp_dtype="bfloat16",
        split_amp=use_split,
        use_vectorized_sim=use_vsim,
        use_tf32=True, use_flash_attention=True,
        use_compile=use_compile,
    )

    t0 = time.perf_counter()
    trainer = BinanceHourlyTrainer(train_config, data)
    artifacts = trainer.train()
    train_time = time.perf_counter() - t0

    logger.success("Training complete in {:.1f}s ({:.1f}s/epoch)", train_time, train_time / args.epochs)
    logger.info("Best checkpoint: {}", artifacts.best_checkpoint)

    if artifacts.history:
        last = artifacts.history[-1]
        logger.info("Last val: sortino={:.4f} return={:.4f}", last.val_sortino or 0, last.val_return or 0)

    config_path = trainer.checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "stock_symbols": stocks,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "sequence_length": args.sequence_length,
            "feature_columns": data.feature_columns,
            "transformer_dim": args.hidden_dim, "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers, "model_arch": args.model_arch,
            "normalizer": data.normalizer.to_dict(),
            "return_weight": args.return_weight, "weight_decay": args.weight_decay,
            "fill_temperature": args.fill_temperature,
            "bf16": use_bf16, "split_amp": use_split, "vectorized_sim": use_vsim,
            "compile": use_compile, "train_time_seconds": train_time,
        }, f, indent=2)

    meta_path = trainer.checkpoint_dir / "training_meta.json"
    history_rows = [{"epoch": r.epoch, "train_loss": r.train_loss, "train_score": r.train_score,
                     "train_sortino": r.train_sortino, "train_return": r.train_return,
                     "val_loss": r.val_loss, "val_score": r.val_score,
                     "val_sortino": r.val_sortino, "val_return": r.val_return}
                    for r in artifacts.history]
    best = max(artifacts.history, key=lambda h: h.val_sortino or float("-inf")) if artifacts.history else None
    with open(meta_path, "w") as f:
        json.dump({
            "run_name": args.checkpoint_name, "symbols": stocks,
            "transformer_dim": args.hidden_dim, "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers, "model_arch": args.model_arch,
            "feature_columns": data.feature_columns,
            "history": history_rows,
            "best_epoch": best.epoch if best else None,
            "best_val_sortino": best.val_sortino if best else None,
            "train_time_seconds": train_time,
        }, f, indent=2)

    logger.info("Saved config={} meta={}", config_path, meta_path)


if __name__ == "__main__":
    main()
