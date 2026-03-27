#!/usr/bin/env python3
"""Train the unified stock policy with the current Hugging Face Trainer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).parent.parent))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import MultiSymbolDataModule
from binanceneural.hf_trainer_bridge import (
    _EpochMetricCallback,
    UnifiedPolicyHFModel,
    UnifiedPolicyHFTrainer,
    compute_unified_policy_eval_metrics,
    make_training_arguments,
    write_run_metadata,
)
from src.trade_directions import DEFAULT_ALPACA_LIVE8_STOCKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_ALPACA_LIVE8_STOCKS))
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--forecast-horizons", type=str, default="1,24")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("tensorboard_logs") / "binanceneural")
    parser.add_argument("--top-k-checkpoints", type=int, default=5)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="robust_score",
        choices=["val_score", "val_sortino", "val_return", "robust_score", "robust_sortino"],
    )
    parser.add_argument("--checkpoint-gap-penalty", type=float, default=0.25)
    parser.add_argument("--preload", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--return-weight", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fill-temperature", type=float, default=5e-4)
    parser.add_argument("--logits-softcap", type=float, default=12.0)
    parser.add_argument("--smoothness-penalty", type=float, default=0.0)
    parser.add_argument("--loss-type", type=str, default="sortino", choices=["sortino", "sharpe", "calmar", "log_wealth", "sortino_dd"])
    parser.add_argument("--dd-penalty", type=float, default=1.0)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0)
    parser.add_argument("--decision-lag-bars", type=int, default=0)
    parser.add_argument("--decision-lag-range", type=str, default="")
    parser.add_argument("--market-order-entry", action="store_true")
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0)
    parser.add_argument("--spread-penalty", type=float, default=0.0)
    parser.add_argument("--spread-target", type=float, default=0.0013)
    parser.add_argument(
        "--validation-use-binary-fills",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="auto")
    parser.add_argument("--wandb-log-metrics", action="store_true")
    args = parser.parse_args()

    symbols = [token.strip().upper() for token in args.symbols.split(",") if token.strip()]
    horizons = tuple(int(token.strip()) for token in args.forecast_horizons.split(",") if token.strip())
    run_name = args.run_name or f"alpaca_hf_trainer_{args.seed}"
    checkpoint_dir = args.checkpoint_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    dataset_config = DatasetConfig(
        symbol=symbols[0],
        data_root=args.data_root,
        forecast_cache_root=args.cache_root,
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        validation_days=args.validation_days,
        cache_only=True,
        min_history_hours=max(args.sequence_length + 48, 100),
    )
    data_module = MultiSymbolDataModule(symbols, dataset_config)

    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        maker_fee=args.maker_fee,
        max_leverage=args.max_leverage,
        margin_annual_rate=args.margin_annual_rate,
        return_weight=args.return_weight,
        smoothness_penalty=args.smoothness_penalty,
        fill_temperature=args.fill_temperature,
        decision_lag_bars=args.decision_lag_bars,
        decision_lag_range=args.decision_lag_range,
        market_order_entry=args.market_order_entry,
        fill_buffer_pct=args.fill_buffer_pct,
        spread_penalty=args.spread_penalty,
        spread_target=args.spread_target,
        loss_type=args.loss_type,
        dd_penalty=args.dd_penalty,
        transformer_dim=args.hidden_dim,
        transformer_layers=args.num_layers,
        transformer_heads=args.num_heads,
        transformer_dropout=args.dropout,
        logits_softcap=args.logits_softcap,
        validation_use_binary_fills=args.validation_use_binary_fills,
        feature_noise_std=args.feature_noise_std,
        warmup_steps=args.warmup_steps,
        top_k_checkpoints=args.top_k_checkpoints,
        checkpoint_metric=args.checkpoint_metric,
        checkpoint_gap_penalty=args.checkpoint_gap_penalty,
        preload_checkpoint_path=args.preload,
        run_name=run_name,
        checkpoint_root=args.checkpoint_root,
        log_dir=args.log_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        use_compile=args.torch_compile,
        use_amp=not args.no_amp,
        accumulation_steps=args.accumulation_steps,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        wandb_log_metrics=args.wandb_log_metrics,
    )

    report_to = ["none"]
    bf16 = bool((not args.no_amp) and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16 = bool((not args.no_amp) and torch.cuda.is_available() and not bf16)
    optim_name = args.optim
    if (not torch.cuda.is_available()) and optim_name == "adamw_torch_fused":
        optim_name = "adamw_torch"

    hf_args = make_training_arguments(
        output_dir=checkpoint_dir,
        run_name=run_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        bf16=bf16,
        fp16=fp16,
        tf32=torch.cuda.is_available(),
        torch_compile=args.torch_compile,
        num_workers=args.num_workers,
        logging_steps=args.logging_steps,
        optim_name=optim_name,
        report_to=report_to,
    )

    model = UnifiedPolicyHFModel(train_config, input_dim=len(data_module.feature_columns))
    trainer = UnifiedPolicyHFTrainer(
        model=model,
        args=hf_args,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        compute_metrics=compute_unified_policy_eval_metrics,
        callbacks=[],
        train_config=train_config,
        data_module=data_module,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.add_callback(_EpochMetricCallback(trainer))

    write_run_metadata(
        checkpoint_dir=checkpoint_dir,
        train_config=train_config,
        data_module=data_module,
        symbols=symbols,
    )

    train_result = trainer.train()
    metrics = trainer.evaluate()

    summary = {
        "run_name": run_name,
        "symbols": symbols,
        "train_result": train_result.metrics,
        "eval_metrics": metrics,
        "best_checkpoint": str(trainer.best_checkpoint) if trainer.best_checkpoint else None,
    }
    (checkpoint_dir / "hf_trainer_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
