#!/usr/bin/env python3
"""Train the production-path classic hourly stock policy in JAX."""
from __future__ import annotations

import argparse
from pathlib import Path

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import MultiSymbolDataModule
from binanceneural.jax_trainer import JaxClassicTrainer
from src.trade_directions import DEFAULT_LONG_ONLY_STOCKS, DEFAULT_SHORT_ONLY_STOCKS


LONG_ONLY = set(DEFAULT_LONG_ONLY_STOCKS)
SHORT_ONLY = set(DEFAULT_SHORT_ONLY_STOCKS)


def build_directional_constraints(symbols: list[str]) -> dict[str, tuple[float, float]]:
    constraints: dict[str, tuple[float, float]] = {}
    for symbol in symbols:
        if symbol in LONG_ONLY:
            constraints[symbol] = (1.0, 0.0)
        elif symbol in SHORT_ONLY:
            constraints[symbol] = (0.0, 1.0)
        else:
            constraints[symbol] = (1.0, 1.0)
    return constraints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        default="NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV",
    )
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("tensorboard_logs") / "binanceneural")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-outputs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.06)
    parser.add_argument("--return-weight", type=float, default=0.15)
    parser.add_argument("--smoothness-penalty", type=float, default=0.0)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--fill-temperature", type=float, default=5e-4)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--forecast-horizons", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hold-hours", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0625)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--market-order-entry", action="store_true")
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    parser.add_argument("--preload", type=Path, default=None)
    parser.add_argument("--dry-train-steps", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="auto")
    parser.add_argument("--wandb-log-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    horizons = tuple(int(token) for token in args.forecast_horizons.split(",") if token.strip())
    constraints = build_directional_constraints(symbols)

    dataset_cfg = DatasetConfig(
        symbol=symbols[0],
        data_root=args.data_root,
        forecast_cache_root=args.cache_root,
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        validation_days=args.validation_days,
        cache_only=True,
        min_history_hours=args.sequence_length + args.validation_days * 24 + 48,
    )
    data_module = MultiSymbolDataModule(symbols=symbols, config=dataset_cfg, directional_constraints=constraints)

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        transformer_dim=args.hidden_dim,
        transformer_layers=args.num_layers,
        transformer_heads=args.num_heads,
        model_arch="classic",
        num_outputs=args.num_outputs,
        return_weight=args.return_weight,
        smoothness_penalty=args.smoothness_penalty,
        maker_fee=args.maker_fee,
        fill_temperature=args.fill_temperature,
        max_hold_hours=args.max_hold_hours,
        max_leverage=args.max_leverage,
        margin_annual_rate=args.margin_annual_rate,
        decision_lag_bars=args.decision_lag_bars,
        market_order_entry=args.market_order_entry,
        fill_buffer_pct=args.fill_buffer_pct,
        checkpoint_root=args.checkpoint_root,
        log_dir=args.log_dir,
        run_name=args.run_name,
        preload_checkpoint_path=args.preload,
        seed=args.seed,
        dry_train_steps=args.dry_train_steps,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        wandb_log_metrics=args.wandb_log_metrics,
    )
    trainer = JaxClassicTrainer(train_cfg, data_module)
    artifacts = trainer.train()
    print(f"Best checkpoint: {artifacts.best_checkpoint}")
    if artifacts.history:
        best = max(artifacts.history, key=lambda item: item.val_score or float('-inf'))
        print(
            f"Best epoch {best.epoch}: "
            f"val_score={best.val_score:.4f} "
            f"val_sortino={best.val_sortino:.4f} "
            f"val_return={best.val_return:.4f}"
        )


if __name__ == "__main__":
    main()
