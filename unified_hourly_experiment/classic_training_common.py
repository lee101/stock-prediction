from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Mapping, TypedDict

from binanceneural.config import DatasetConfig, TrainerBackend, TrainingConfig
from binanceneural.data import MultiSymbolDataModule
from unified_hourly_experiment.directional_constraints import build_directional_constraints
from unified_hourly_experiment.jax_classic_defaults import compute_jax_classic_min_history_hours


class ClassicTrainingKwargs(TypedDict):
    epochs: int
    batch_size: int
    sequence_length: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    transformer_dim: int
    transformer_layers: int
    transformer_heads: int
    return_weight: float
    smoothness_penalty: float
    maker_fee: float
    fill_temperature: float
    max_hold_hours: float
    max_leverage: float
    margin_annual_rate: float
    decision_lag_bars: int
    market_order_entry: bool
    fill_buffer_pct: float
    preload_checkpoint_path: Path | None
    seed: int
    dry_train_steps: int | None


class ArgsFileParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        return shlex.split(arg_line, comments=True, posix=True)


EFFECTIVE_ARGS_JSON_FILENAME = "effective_args.json"
EFFECTIVE_ARGS_TXT_FILENAME = "effective_args.txt"


def parse_horizons(raw: str) -> tuple[int, ...]:
    horizons = tuple(int(token.strip()) for token in str(raw).split(",") if token.strip())
    if not horizons:
        raise ValueError("At least one forecast horizon is required.")
    return horizons


def build_classic_dataset_config(
    args: argparse.Namespace,
    *,
    symbols: list[str],
    horizons: tuple[int, ...],
) -> DatasetConfig:
    return DatasetConfig(
        symbol=symbols[0],
        data_root=args.data_root,
        forecast_cache_root=args.cache_root,
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        validation_days=args.validation_days,
        cache_only=bool(args.cache_only),
        min_history_hours=compute_jax_classic_min_history_hours(args.sequence_length, args.validation_days),
    )


def build_classic_training_kwargs(args: argparse.Namespace) -> ClassicTrainingKwargs:
    return {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "sequence_length": int(args.sequence_length),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "transformer_dim": int(args.hidden_dim),
        "transformer_layers": int(args.num_layers),
        "transformer_heads": int(args.num_heads),
        "return_weight": float(args.return_weight),
        "smoothness_penalty": float(args.smoothness_penalty),
        "maker_fee": float(args.maker_fee),
        "fill_temperature": float(args.fill_temperature),
        "max_hold_hours": float(args.max_hold_hours),
        "max_leverage": float(args.max_leverage),
        "margin_annual_rate": float(args.margin_annual_rate),
        "decision_lag_bars": int(args.decision_lag_bars),
        "market_order_entry": bool(args.market_order_entry),
        "fill_buffer_pct": float(args.fill_buffer_pct),
        "preload_checkpoint_path": args.preload,
        "seed": int(args.seed),
        "dry_train_steps": int(args.dry_train_steps) if args.dry_train_steps is not None else None,
    }


def build_classic_training_config(
    args: argparse.Namespace,
    *,
    backend: TrainerBackend,
    checkpoint_root: Path,
    log_dir: Path,
    run_name: str | None,
    extra_kwargs: Mapping[str, object] | None = None,
) -> TrainingConfig:
    training_kwargs = build_classic_training_kwargs(args)
    payload: dict[str, object] = {
        **training_kwargs,
        "trainer_backend": backend,
        "model_arch": "classic",
        "checkpoint_root": checkpoint_root,
        "log_dir": log_dir,
        "run_name": run_name,
    }
    if extra_kwargs:
        payload.update(extra_kwargs)
    return TrainingConfig(**payload)


def build_classic_data_module(
    args: argparse.Namespace,
    *,
    symbols: list[str],
    horizons: tuple[int, ...],
    data_module_cls: type[MultiSymbolDataModule] = MultiSymbolDataModule,
    directional_constraints: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[DatasetConfig, MultiSymbolDataModule]:
    dataset_cfg = build_classic_dataset_config(args, symbols=symbols, horizons=horizons)
    constraints = dict(directional_constraints) if directional_constraints is not None else build_directional_constraints(symbols)
    try:
        data_module = data_module_cls(
            symbols=symbols,
            config=dataset_cfg,
            directional_constraints=constraints,
        )
    except Exception as exc:
        if bool(args.cache_only):
            raise RuntimeError(f"{exc} Use --allow-forecast-refresh to build missing forecasts on demand.") from exc
        raise
    return dataset_cfg, data_module


def render_classic_run_plan_summary(
    plan: Mapping[str, Any],
    *,
    title: str = "Run Plan",
) -> str:
    symbols = [str(item) for item in plan.get("symbols", [])]
    horizons = [str(item) for item in plan.get("forecast_horizons", [])]
    lines = [title]
    if symbols:
        lines.append(f"Symbols: {','.join(symbols)}")
        lines.append(f"Symbol count: {len(symbols)}")
    if "backends" in plan:
        backends = [str(item) for item in plan.get("backends", [])]
        if backends:
            lines.append(f"Backends: {','.join(backends)}")
    if "seeds" in plan:
        seeds = [str(item) for item in plan.get("seeds", [])]
        if seeds:
            lines.append(f"Seeds: {','.join(seeds)}")
    if horizons:
        lines.append(f"Forecast horizons: {','.join(horizons)}")
    if "cache_only" in plan:
        lines.append(f"Cache only: {bool(plan.get('cache_only'))}")
    if plan.get("output_dir"):
        lines.append(f"Output dir: {plan['output_dir']}")
    if plan.get("checkpoint_root"):
        lines.append(f"Checkpoint root: {plan['checkpoint_root']}")
    if plan.get("log_dir"):
        lines.append(f"Log dir: {plan['log_dir']}")
    if plan.get("run_name"):
        lines.append(f"Run name: {plan['run_name']}")
    if plan.get("preload"):
        lines.append(f"Preload: {plan['preload']}")

    training = plan.get("training")
    if isinstance(training, Mapping):
        fields: list[str] = []
        for key in ("epochs", "dry_train_steps", "batch_size", "sequence_length", "validation_days"):
            value = training.get(key)
            if value is not None:
                fields.append(f"{key}={value}")
        if fields:
            lines.append(f"Training: {', '.join(fields)}")

    plan_error = plan.get("plan_error")
    if plan_error:
        lines.append(f"Plan error: {plan_error}")
    return "\n".join(lines)


def _preferred_option_string(action: argparse.Action) -> str | None:
    long_options = [option for option in action.option_strings if option.startswith("--")]
    if long_options:
        return max(long_options, key=len)
    if action.option_strings:
        return action.option_strings[-1]
    return None


def render_effective_args_file(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    module_name: str,
) -> str:
    actions_by_dest: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        actions_by_dest.setdefault(action.dest, []).append(action)

    lines = [
        f"# Re-run with: python -m {module_name} @{EFFECTIVE_ARGS_TXT_FILENAME}",
    ]
    for dest, actions in actions_by_dest.items():
        if not hasattr(args, dest):
            continue
        value = getattr(args, dest)
        if value is None:
            continue

        bool_actions = [
            action
            for action in actions
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
        ]
        if bool_actions:
            chosen = next((action for action in bool_actions if getattr(action, "const", None) == value), None)
            if chosen is None:
                continue
            option = _preferred_option_string(chosen)
            if option is not None:
                lines.append(option)
            continue

        action = actions[-1]
        option = _preferred_option_string(action)
        if option is None:
            continue
        if isinstance(value, Path):
            rendered_value = str(value)
        elif isinstance(value, (list, tuple)):
            rendered_value = ",".join(str(item) for item in value)
        else:
            rendered_value = str(value)
        lines.append(f"{option} {shlex.quote(rendered_value)}")
    return "\n".join(lines) + "\n"


def write_effective_args_artifacts(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    module_name: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    json_path = output_dir / EFFECTIVE_ARGS_JSON_FILENAME
    txt_path = output_dir / EFFECTIVE_ARGS_TXT_FILENAME
    json_path.write_text(json.dumps(effective_args, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(
        render_effective_args_file(parser, args, module_name=module_name),
        encoding="utf-8",
    )
    return json_path, txt_path
