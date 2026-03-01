#!/usr/bin/env python3
"""BF16-optimized training pipeline inspired by modded-nanogpt.

Key optimizations over standard train_unified_policy.py:
- BF16 autocast with no grad scaler (BF16 doesn't need it)
- torch.compile with max-autotune for fused kernels
- TF32 matmul precision for bf16-incompatible ops
- Expandable CUDA memory segments
- Pin memory + persistent workers for data loading
- Fused AdamW optimizer
- Multi-period evaluation built-in (1d, 7d, 30d, 60d, 120d, 150d)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Expandable segments for better GPU memory management
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.data import MultiSymbolDataModule
from binanceneural.config import DatasetConfig, TrainingConfig
from src.trade_directions import resolve_trade_directions


def setup_bf16_optimizations():
    """Apply nanoGPT-style BF16 optimizations."""
    # TF32 for any fp32 fallback ops
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Enable flash SDP
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU: {} ({:.1f} GB)", gpu_name, gpu_mem)
        logger.info("BF16 support: {}", torch.cuda.is_bf16_supported())


def main():
    parser = argparse.ArgumentParser(description="BF16-optimized Alpaca stock trader training")

    # Symbol configuration
    parser.add_argument("--symbols", type=str, default="NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT",
                        help="Comma-separated stock symbols")
    parser.add_argument("--crypto-symbols", type=str, default="",
                        help="Comma-separated crypto symbols")

    # Data paths
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--crypto-data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--stock-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--crypto-cache-root", type=Path, default=Path("binanceneural/forecast_cache"))

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--return-weight", type=float, default=0.15)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)

    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Realistic trading simulation
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
                                 "sortino_dd", "multiwindow", "multiwindow_dd"],
                        help="Loss function type")
    parser.add_argument("--multiwindow-fractions", type=str, default="0.33,0.5,0.75,1.0",
                        help="Comma-separated window fractions for multiwindow loss")
    parser.add_argument("--multiwindow-aggregation", type=str, default="minimax",
                        choices=["minimax", "mean", "softmin"],
                        help="Aggregation for multiwindow loss")
    parser.add_argument("--dd-penalty", type=float, default=1.0,
                        help="Drawdown penalty for sortino_dd/multiwindow_dd")

    # BF16 optimization controls
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-bf16", action="store_true", help="Disable BF16 (use FP32)")
    parser.add_argument("--compile-mode", type=str, default="max-autotune",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")

    # Eval configuration
    parser.add_argument("--eval-periods", type=str, default="1,7,30,60,120,150",
                        help="Comma-separated holdout day periods for evaluation")
    parser.add_argument("--eval-after-epoch", type=int, default=5,
                        help="Start multi-period eval after this epoch")
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--eval-fee-rate", type=float, default=0.001)
    parser.add_argument("--eval-margin-rate", type=float, default=0.0625)

    # Trade direction
    parser.add_argument("--allow-short", action="store_true", default=True,
                        help="Enable short-selling for short-only symbols")
    parser.add_argument("--no-short", action="store_true",
                        help="Disable all shorting (long-only)")

    # Output
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("unified_hourly_experiment/checkpoints"))
    parser.add_argument("--forecast-horizons", type=str, default="1")

    args = parser.parse_args()

    # Setup BF16 optimizations
    setup_bf16_optimizations()

    stocks = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cryptos = [s.strip().upper() for s in args.crypto_symbols.split(",") if s.strip()]
    horizons = [int(h.strip()) for h in args.forecast_horizons.split(",")]

    logger.info("=== BF16 Efficient Training ===")
    logger.info("Symbols: {} stocks + {} crypto", len(stocks), len(cryptos))
    logger.info("Architecture: h{} {}L {}H", args.hidden_dim, args.num_layers, args.num_heads)
    logger.info("Training: epochs={} bs={} lr={} wd={} rw={} seq={}",
                args.epochs, args.batch_size, args.lr, args.weight_decay,
                args.return_weight, args.sequence_length)
    allow_short = args.allow_short and not args.no_short
    logger.info("BF16: {} | Compile: {} ({}) | Shorts: {}",
                not args.no_bf16, not args.no_compile,
                args.compile_mode if not args.no_compile else "disabled",
                allow_short)

    # Build directional constraints matching portfolio simulator defaults
    dir_constraints = {}
    for sym in stocks:
        td = resolve_trade_directions(sym, allow_short=allow_short)
        dir_constraints[sym] = (float(td.can_long), float(td.can_short))
        if td.can_short and not td.can_long:
            logger.info("  {} → SHORT-ONLY", sym)
        elif td.can_long and not td.can_short:
            logger.info("  {} → LONG-ONLY", sym)
        else:
            logger.info("  {} → LONG+SHORT", sym)

    # Load data
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

    logger.info("Loading stock data for {} symbols...", len(stocks))
    t0 = time.perf_counter()
    stock_data = MultiSymbolDataModule(stocks, stock_config,
                                       directional_constraints=dir_constraints)
    logger.info("Data loaded in {:.1f}s ({} train samples, {} features)",
                time.perf_counter() - t0,
                len(stock_data.train_dataset),
                len(stock_data.feature_columns))

    # Build checkpoint name
    if args.checkpoint_name is None:
        args.checkpoint_name = (
            f"bf16_rw{str(args.return_weight).replace('.','')}"
            f"_wd{str(args.weight_decay).replace('.','')}"
            f"_seq{args.sequence_length}"
            f"_s{args.seed}"
        )

    # Training config
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.sequence_length,
        checkpoint_root=args.checkpoint_root,
        run_name=args.checkpoint_name,
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
        # BF16 optimizations
        use_amp=not args.no_bf16,
        amp_dtype="bfloat16",
        use_tf32=True,
        use_flash_attention=True,
        use_compile=not args.no_compile,
        seed=args.seed,
        # Loss function
        loss_type=args.loss_type,
        multiwindow_fractions=args.multiwindow_fractions,
        multiwindow_aggregation=args.multiwindow_aggregation,
        dd_penalty=args.dd_penalty,
    )

    # Train
    logger.info("Starting training...")
    t_train_start = time.perf_counter()

    trainer = BinanceHourlyTrainer(train_config, stock_data)
    artifacts = trainer.train()

    train_time = time.perf_counter() - t_train_start
    logger.success("Training complete in {:.1f}s ({:.1f}s/epoch)",
                   train_time, train_time / args.epochs)
    logger.info("Best checkpoint: {}", artifacts.best_checkpoint)

    if artifacts.history:
        last = artifacts.history[-1]
        logger.info("Last val: sortino={:.4f} return={:.4f}",
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
            "feature_columns": stock_data.feature_columns,
            "transformer_dim": args.hidden_dim,
            "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers,
            "model_arch": "classic",
            "normalizer": stock_data.normalizer.to_dict(),
            "return_weight": args.return_weight,
            "weight_decay": args.weight_decay,
            "decision_lag_bars": args.decision_lag_bars,
            "fill_temperature": args.fill_temperature,
            "bf16_training": not args.no_bf16,
            "compile_mode": args.compile_mode if not args.no_compile else None,
            "train_time_seconds": train_time,
        }, f, indent=2)

    # Save training metadata
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
            "run_name": args.checkpoint_name,
            "symbols": stocks,
            "epochs": args.epochs,
            "sequence_length": args.sequence_length,
            "transformer_dim": args.hidden_dim,
            "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers,
            "model_arch": "classic",
            "feature_columns": stock_data.feature_columns,
            "decision_lag_bars": args.decision_lag_bars,
            "return_weight": args.return_weight,
            "fill_temperature": args.fill_temperature,
            "history": history_rows,
            "best_epoch": best.epoch if best else None,
            "best_val_sortino": best.val_sortino if best else None,
            "best_val_return": best.val_return if best else None,
            "best_checkpoint": str(artifacts.best_checkpoint) if artifacts.best_checkpoint else None,
            "train_time_seconds": train_time,
        }, f, indent=2)

    logger.info("Config saved to {}", config_path)
    logger.info("Training metadata saved to {}", meta_path)

    # Run multi-period evaluation
    eval_periods = [int(p) for p in args.eval_periods.split(",")]
    logger.info("\n=== Multi-Period Evaluation ({}d) ===", ",".join(str(p) for p in eval_periods))

    run_multi_period_eval(
        checkpoint_dir=trainer.checkpoint_dir,
        symbols=stocks,
        eval_periods=eval_periods,
        eval_after_epoch=args.eval_after_epoch,
        min_edge=args.min_edge,
        max_positions=args.max_positions,
        max_hold_hours=args.max_hold_hours,
        fee_rate=args.eval_fee_rate,
        margin_rate=args.eval_margin_rate,
    )


def run_multi_period_eval(
    checkpoint_dir: Path,
    symbols: list[str],
    eval_periods: list[int],
    eval_after_epoch: int = 5,
    min_edge: float = 0.0,
    max_positions: int = 5,
    max_hold_hours: int = 6,
    fee_rate: float = 0.001,
    margin_rate: float = 0.0625,
):
    """Run multi-period OOS evaluation on all epochs >= eval_after_epoch."""
    import subprocess

    periods_str = ",".join(str(p) for p in eval_periods)
    symbols_str = ",".join(symbols)

    cmd = [
        sys.executable, "-u",
        "unified_hourly_experiment/sweep_epoch_portfolio.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--symbols", symbols_str,
        "--fee-rate", str(fee_rate),
        "--margin-rate", str(margin_rate),
        "--no-close-at-eod",
        "--max-positions", str(max_positions),
        "--max-hold-hours", str(max_hold_hours),
        "--min-edge", str(min_edge),
        "--holdout-days", periods_str,
    ]

    logger.info("Running: {}", " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    eval_time = time.perf_counter() - t0
    logger.info("Evaluation completed in {:.1f}s (exit code: {})", eval_time, result.returncode)


if __name__ == "__main__":
    main()
