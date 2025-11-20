#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

from neuraldailytraining import (
    DailyDataModule,
    DailyDatasetConfig,
    DailyTradingRuntime,
    DailyTrainingConfig,
    NeuralDailyTrainer,
    TradingPlan,
)


def _positive(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural daily stock/crypto training and inference script.")
    parser.add_argument("--mode", choices=("train", "plan"), default="train")
    parser.add_argument("--symbols", nargs="*", help="Symbols to include (default uses dataset defaults).")
    parser.add_argument("--data-root", default="trainingdata/train", help="Directory containing daily OHLC CSVs.")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache", help="Chronos forecast cache directory.")
    parser.add_argument("--sequence-length", type=_positive, default=256, help="Transformer context length.")
    parser.add_argument("--batch-size", type=_positive, default=32)
    parser.add_argument("--epochs", type=_positive, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-root", default="neuraldailytraining/checkpoints")
    parser.add_argument("--checkpoint", help="Checkpoint path for --mode plan.")
    parser.add_argument("--run-name", help="Optional run name override.")
    parser.add_argument("--dry-train-steps", type=int, help="Optional early stop after N optimizer steps.")
    parser.add_argument("--device", help="Torch device override (cpu, cuda, cuda:0, ...).")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction for training splits.")
    parser.add_argument("--validation-days", type=int, default=90, help="Minimum validation tail length.")
    parser.add_argument("--grouping-strategy", choices=("static", "correlation"), default="correlation")
    parser.add_argument("--symbol-dropout-rate", type=float, default=0.1)
    parser.add_argument("--exclude-symbols-file", help="Optional newline-delimited list of symbols to drop from training.")
    parser.add_argument("--exclude-symbol", nargs="*", help="Inline symbols to drop from training.")
    parser.add_argument("--corr-min", type=float, default=0.6, help="Correlation threshold for grouping.")
    parser.add_argument("--corr-max-group", type=int, default=12, help="Max symbols per correlation cluster.")
    parser.add_argument("--corr-window-days", type=int, default=252)
    parser.add_argument("--corr-min-overlap", type=int, default=60)
    parser.add_argument("--require-forecasts", action=argparse.BooleanOptionalAction, default=False, help="Fail if forecast cache missing rows.")
    parser.add_argument("--forecast-fill-strategy", choices=("persistence", "fail"), default="persistence")
    parser.add_argument("--forecast-cache-writeback", action=argparse.BooleanOptionalAction, default=True, help="Persist filled forecasts to cache.")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"), default="max-autotune")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers for train/val loaders.")
    parser.add_argument("--output-json", help="Optional output path for plan mode results.")
    parser.add_argument(
        "--risk-threshold",
        type=float,
        help="Optional cap for neural trade amounts (defaults to the value baked into the checkpoint).",
    )
    return parser.parse_args()


def _build_dataset_config(args: argparse.Namespace) -> DailyDatasetConfig:
    symbols = tuple(args.symbols) if args.symbols else DailyDatasetConfig().symbols
    return DailyDatasetConfig(
        symbols=symbols,
        data_root=Path(args.data_root),
        forecast_cache_dir=Path(args.forecast_cache),
        sequence_length=args.sequence_length,
        val_fraction=args.val_fraction,
        validation_days=args.validation_days,
        require_forecasts=args.require_forecasts,
        forecast_fill_strategy=args.forecast_fill_strategy,
        forecast_cache_writeback=args.forecast_cache_writeback,
        grouping_strategy=args.grouping_strategy,
        symbol_dropout_rate=args.symbol_dropout_rate,
        exclude_symbols=args.exclude_symbol,
        exclude_symbols_file=Path(args.exclude_symbols_file) if args.exclude_symbols_file else None,
        correlation_min_corr=args.corr_min,
        correlation_max_group_size=args.corr_max_group,
        correlation_window_days=args.corr_window_days,
        correlation_min_overlap=args.corr_min_overlap,
    )


def run_training(args: argparse.Namespace) -> None:
    dataset_cfg = _build_dataset_config(args)
    train_cfg = DailyTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        checkpoint_root=Path(args.checkpoint_root),
        run_name=args.run_name,
        device=args.device,
        dry_train_steps=args.dry_train_steps,
        use_amp=args.use_amp,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        num_workers=args.num_workers,
        dataset=dataset_cfg,
    )
    module = DailyDataModule(dataset_cfg)
    trainer = NeuralDailyTrainer(train_cfg, module)
    artifacts = trainer.train()
    best_path = artifacts.best_checkpoint or (trainer.checkpoint_dir / "best.pt")
    if not best_path.exists():
        best_path.write_text("best checkpoint tracked via manifest; see manifest.json")
    print(f"[train] best checkpoint: {best_path}")


def _print_plan(plan: TradingPlan) -> None:
    print(
        f"{plan.symbol:>8} @ {plan.timestamp} | "
        f"buy={plan.buy_price:.2f} sell={plan.sell_price:.2f} "
        f"qty={plan.trade_amount:.3f} ref={plan.reference_close:.2f}"
    )


def run_plan(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for plan mode")
    dataset_cfg = _build_dataset_config(args)
    runtime = DailyTradingRuntime(
        args.checkpoint,
        dataset_config=dataset_cfg,
        device=args.device,
        risk_threshold=args.risk_threshold,
    )
    symbols = args.symbols or list(dataset_cfg.symbols)
    plans = runtime.generate_plans(symbols)
    if not plans:
        print("[plan] no viable symbols produced trading plans.")
        return
    for plan in plans:
        _print_plan(plan)
    if args.output_json:
        payload = [plan.__dict__ for plan in plans]
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"[plan] wrote plan payload to {args.output_json}")


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    else:
        run_plan(args)


if __name__ == "__main__":
    main()
