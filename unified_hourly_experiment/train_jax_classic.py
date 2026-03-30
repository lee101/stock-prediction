#!/usr/bin/env python3
"""Train the production-path classic hourly stock policy in JAX."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from binanceneural.data import MultiSymbolDataModule
from binanceneural.trainer_factory import build_trainer
from unified_hourly_experiment.classic_training_common import (
    ArgsFileParser,
    build_classic_data_module,
    build_classic_dataset_config,
    build_classic_training_config,
    build_classic_training_kwargs,
    parse_horizons,
    render_classic_run_plan_summary,
    write_effective_args_artifacts,
)
from unified_hourly_experiment.directional_constraints import build_directional_constraints
from unified_hourly_experiment.jax_classic_defaults import (
    DEFAULT_JAX_CLASSIC_SYMBOLS_CSV,
    JAX_CLASSIC_DEFAULT_BATCH_SIZE,
    JAX_CLASSIC_DEFAULT_DECISION_LAG_BARS,
    JAX_CLASSIC_DEFAULT_FILL_BUFFER_PCT,
    JAX_CLASSIC_DEFAULT_FILL_TEMPERATURE,
    JAX_CLASSIC_DEFAULT_GRAD_CLIP,
    JAX_CLASSIC_DEFAULT_HIDDEN_DIM,
    JAX_CLASSIC_DEFAULT_LEARNING_RATE,
    JAX_CLASSIC_DEFAULT_MAKER_FEE,
    JAX_CLASSIC_DEFAULT_MARGIN_ANNUAL_RATE,
    JAX_CLASSIC_DEFAULT_MAX_HOLD_HOURS,
    JAX_CLASSIC_DEFAULT_MAX_LEVERAGE,
    JAX_CLASSIC_DEFAULT_NUM_HEADS,
    JAX_CLASSIC_DEFAULT_NUM_LAYERS,
    JAX_CLASSIC_DEFAULT_NUM_OUTPUTS,
    JAX_CLASSIC_DEFAULT_RETURN_WEIGHT,
    JAX_CLASSIC_DEFAULT_SEED,
    JAX_CLASSIC_DEFAULT_SEQUENCE_LENGTH,
    JAX_CLASSIC_DEFAULT_SMOOTHNESS_PENALTY,
    JAX_CLASSIC_DEFAULT_TRAIN_EPOCHS,
    JAX_CLASSIC_DEFAULT_VALIDATION_DAYS,
    JAX_CLASSIC_DEFAULT_WEIGHT_DECAY,
)
from unified_hourly_experiment.symbol_validation import parse_symbols


def build_arg_parser() -> argparse.ArgumentParser:
    parser = ArgsFileParser(
        description="Train the production-path classic hourly stock policy in JAX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--symbols",
        default=DEFAULT_JAX_CLASSIC_SYMBOLS_CSV,
    )
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("tensorboard_logs") / "binanceneural")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=JAX_CLASSIC_DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=JAX_CLASSIC_DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=JAX_CLASSIC_DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--hidden-dim", type=int, default=JAX_CLASSIC_DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=JAX_CLASSIC_DEFAULT_NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=JAX_CLASSIC_DEFAULT_NUM_HEADS)
    parser.add_argument("--num-outputs", type=int, default=JAX_CLASSIC_DEFAULT_NUM_OUTPUTS)
    parser.add_argument("--learning-rate", type=float, default=JAX_CLASSIC_DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=JAX_CLASSIC_DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=JAX_CLASSIC_DEFAULT_GRAD_CLIP)
    parser.add_argument("--return-weight", type=float, default=JAX_CLASSIC_DEFAULT_RETURN_WEIGHT)
    parser.add_argument("--smoothness-penalty", type=float, default=JAX_CLASSIC_DEFAULT_SMOOTHNESS_PENALTY)
    parser.add_argument("--maker-fee", type=float, default=JAX_CLASSIC_DEFAULT_MAKER_FEE)
    parser.add_argument("--fill-temperature", type=float, default=JAX_CLASSIC_DEFAULT_FILL_TEMPERATURE)
    parser.add_argument("--validation-days", type=int, default=JAX_CLASSIC_DEFAULT_VALIDATION_DAYS)
    parser.add_argument("--forecast-horizons", default="1")
    parser.add_argument("--cache-only", dest="cache_only", action="store_true", default=True)
    parser.add_argument(
        "--allow-forecast-refresh",
        dest="cache_only",
        action="store_false",
        help="Allow forecast generation on demand instead of requiring cache-only inputs.",
    )
    parser.add_argument("--seed", type=int, default=JAX_CLASSIC_DEFAULT_SEED)
    parser.add_argument("--max-hold-hours", type=float, default=JAX_CLASSIC_DEFAULT_MAX_HOLD_HOURS)
    parser.add_argument("--max-leverage", type=float, default=JAX_CLASSIC_DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--margin-annual-rate", type=float, default=JAX_CLASSIC_DEFAULT_MARGIN_ANNUAL_RATE)
    parser.add_argument("--decision-lag-bars", type=int, default=JAX_CLASSIC_DEFAULT_DECISION_LAG_BARS)
    parser.add_argument("--market-order-entry", action="store_true")
    parser.add_argument("--fill-buffer-pct", type=float, default=JAX_CLASSIC_DEFAULT_FILL_BUFFER_PCT)
    parser.add_argument("--preload", type=Path, default=None)
    parser.add_argument("--dry-train-steps", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="auto")
    parser.add_argument("--wandb-log-metrics", action="store_true")
    parser.add_argument(
        "--describe-run",
        action="store_true",
        help="Print the resolved training plan and exit before preparing data or training.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def build_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols)
    horizons = parse_horizons(args.forecast_horizons)
    dataset_cfg = build_classic_dataset_config(args, symbols=symbols, horizons=horizons)
    training_kwargs = build_classic_training_kwargs(args)
    return {
        "symbols": symbols,
        "symbol_count": len(symbols),
        "forecast_horizons": list(horizons),
        "cache_only": bool(args.cache_only),
        "data_root": str(args.data_root),
        "cache_root": str(args.cache_root),
        "checkpoint_root": str(args.checkpoint_root),
        "log_dir": str(args.log_dir),
        "preload": str(args.preload) if args.preload else None,
        "run_name": args.run_name,
        "seed": int(args.seed),
        "describe_run": bool(args.describe_run),
        "training": {
            "epochs": int(training_kwargs["epochs"]),
            "batch_size": int(training_kwargs["batch_size"]),
            "sequence_length": int(training_kwargs["sequence_length"]),
            "validation_days": int(args.validation_days),
            "min_history_hours": int(dataset_cfg.min_history_hours),
            "hidden_dim": int(training_kwargs["transformer_dim"]),
            "num_layers": int(training_kwargs["transformer_layers"]),
            "num_heads": int(training_kwargs["transformer_heads"]),
            "num_outputs": int(args.num_outputs),
            "learning_rate": float(training_kwargs["learning_rate"]),
            "weight_decay": float(training_kwargs["weight_decay"]),
            "grad_clip": float(training_kwargs["grad_clip"]),
            "return_weight": float(training_kwargs["return_weight"]),
            "smoothness_penalty": float(training_kwargs["smoothness_penalty"]),
            "maker_fee": float(training_kwargs["maker_fee"]),
            "fill_temperature": float(training_kwargs["fill_temperature"]),
            "max_hold_hours": float(training_kwargs["max_hold_hours"]),
            "max_leverage": float(training_kwargs["max_leverage"]),
            "margin_annual_rate": float(training_kwargs["margin_annual_rate"]),
            "decision_lag_bars": int(training_kwargs["decision_lag_bars"]),
            "market_order_entry": bool(training_kwargs["market_order_entry"]),
            "fill_buffer_pct": float(training_kwargs["fill_buffer_pct"]),
            "dry_train_steps": training_kwargs["dry_train_steps"],
            "wandb_mode": str(args.wandb_mode),
            "wandb_log_metrics": bool(args.wandb_log_metrics),
        },
    }


def main() -> None:
    args = parse_args()
    try:
        plan = build_run_plan(args)
    except Exception as exc:
        raise SystemExit(f"Plan error: {exc}") from exc
    if args.describe_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(render_classic_run_plan_summary(plan, title="JAX Classic Training Plan"))
    symbols = list(plan["symbols"])
    horizons = tuple(int(token) for token in plan["forecast_horizons"])
    constraints = build_directional_constraints(symbols)
    _, data_module = build_classic_data_module(
        args,
        symbols=symbols,
        horizons=horizons,
        data_module_cls=MultiSymbolDataModule,
        directional_constraints=constraints,
    )

    train_cfg = build_classic_training_config(
        args,
        backend="jax_classic",
        checkpoint_root=args.checkpoint_root,
        log_dir=args.log_dir,
        run_name=args.run_name,
        extra_kwargs={
            "num_outputs": int(args.num_outputs),
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "wandb_tags": args.wandb_tags,
            "wandb_notes": args.wandb_notes,
            "wandb_mode": args.wandb_mode,
            "wandb_log_metrics": bool(args.wandb_log_metrics),
        },
    )
    trainer = build_trainer(train_cfg, data_module)
    resolved_run_name = getattr(getattr(trainer, "config", None), "run_name", None) or train_cfg.run_name
    if resolved_run_name:
        run_dir = Path(train_cfg.checkpoint_root) / str(resolved_run_name)
        parser = build_arg_parser()
        try:
            effective_args_path, effective_args_cli_path = write_effective_args_artifacts(
                parser,
                args,
                run_dir,
                module_name="unified_hourly_experiment.train_jax_classic",
            )
        except Exception as exc:
            print(
                f"[train_jax_classic] Failed to write effective args artifacts in {run_dir}: {exc}",
                file=sys.stderr,
            )
        else:
            print(f"Wrote {effective_args_path}")
            print(f"Wrote {effective_args_cli_path}")
            print(
                "Rerun with: "
                "python -m unified_hourly_experiment.train_jax_classic "
                f"@{effective_args_cli_path}"
            )
    artifacts = trainer.train()
    print(f"Best checkpoint: {artifacts.best_checkpoint}")
    if artifacts.stop_reason:
        print(artifacts.stop_reason)
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
