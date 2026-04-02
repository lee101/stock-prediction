from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.torch_device_utils import move_module_to_runtime_device, resolve_runtime_device

from RLgpt.config import (
    DailyPlanDataConfig,
    PlannerConfig,
    SimulatorConfig,
    TrainingConfig,
    DEFAULT_RLGPT_BATCH_SIZE,
    DEFAULT_RLGPT_DATA_ROOT,
    DEFAULT_RLGPT_DEPTH,
    DEFAULT_RLGPT_DROPOUT,
    DEFAULT_RLGPT_EPOCHS,
    DEFAULT_RLGPT_FILL_BUFFER_BPS,
    DEFAULT_RLGPT_FILL_TEMPERATURE_BPS,
    DEFAULT_RLGPT_FORECAST_CACHE_ROOT,
    DEFAULT_RLGPT_HEADS,
    DEFAULT_RLGPT_HIDDEN_DIM,
    DEFAULT_RLGPT_INITIAL_CASH,
    DEFAULT_RLGPT_LEARNING_RATE,
    DEFAULT_RLGPT_MAKER_FEE_BPS,
    DEFAULT_RLGPT_MAX_FEATURE_LOOKBACK_HOURS,
    DEFAULT_RLGPT_MAX_UNITS_PER_ASSET,
    DEFAULT_RLGPT_MIN_BARS_PER_DAY,
    DEFAULT_RLGPT_MIN_HISTORY_HOURS,
    DEFAULT_RLGPT_NUM_WORKERS,
    DEFAULT_RLGPT_OUTPUT_ROOT,
    DEFAULT_RLGPT_SEED,
    DEFAULT_RLGPT_SEQUENCE_LENGTH,
    DEFAULT_RLGPT_SHARED_UNIT_BUDGET,
    DEFAULT_RLGPT_SLIPPAGE_BPS,
    DEFAULT_RLGPT_VALIDATION_DAYS,
    DEFAULT_RLGPT_WEIGHT_DECAY,
    default_forecast_horizons_csv,
    normalize_symbol_list,
    parse_horizon_list,
    validate_training_config,
)
from RLgpt.data import DailyPlanTensorDataset, TensorNormalizer, prepare_daily_plan_tensors
from RLgpt.model import CrossAssetDailyPlanner
from RLgpt.simulator import compute_trading_objective, simulate_daily_plans


def run_training(config: TrainingConfig) -> dict[str, Any]:
    config_errors = validate_training_config(config)
    if config_errors:
        raise ValueError(f"Invalid RLgpt training config: {'; '.join(config_errors)}")
    _set_seed(config.seed)
    device = resolve_runtime_device(config.device)
    bundle = prepare_daily_plan_tensors(config.data)
    train_bundle, val_bundle = bundle.train_val_split(config.data.validation_days)

    if config.max_train_days is not None:
        train_bundle = train_bundle.slice(max(0, len(train_bundle) - int(config.max_train_days)), None)
    if config.max_val_days is not None:
        val_bundle = val_bundle.slice(max(0, len(val_bundle) - int(config.max_val_days)), None)

    normalizer = TensorNormalizer.fit(train_bundle.features)
    train_bundle = train_bundle.with_features(normalizer.transform(train_bundle.features))
    val_bundle = val_bundle.with_features(normalizer.transform(val_bundle.features))

    train_loader = DataLoader(
        DailyPlanTensorDataset(train_bundle),
        batch_size=min(int(config.batch_size), len(train_bundle)),
        shuffle=True,
        num_workers=int(config.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        DailyPlanTensorDataset(val_bundle),
        batch_size=min(int(config.batch_size), len(val_bundle)),
        shuffle=False,
        num_workers=int(config.num_workers),
        drop_last=False,
    )

    model, device = move_module_to_runtime_device(
        CrossAssetDailyPlanner(input_dim=train_bundle.feature_dim, config=config.planner),
        config.device,
        device,
        context="RLgpt",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    run_name = config.run_name or _default_run_name(config.data.symbols)
    out_dir = Path(config.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", _json_ready(asdict(config)))
    _write_json(out_dir / "data_manifest.json", {
        "symbols": list(bundle.symbols),
        "feature_names": list(bundle.feature_names),
        "train_days": [ts.isoformat() for ts in train_bundle.days],
        "val_days": [ts.isoformat() for ts in val_bundle.days],
    })
    _write_json(out_dir / "normalizer.json", normalizer.to_dict())

    best_val_score = float("-inf")
    best_checkpoint = out_dir / "best.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            simulator_config=config.simulator,
            optimizer=optimizer,
            grad_clip=float(config.grad_clip),
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            simulator_config=config.simulator,
            optimizer=None,
            grad_clip=float(config.grad_clip),
        )
        epoch_row = {
            "epoch": float(epoch),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        history.append(epoch_row)
        print(
            f"epoch={epoch:03d} "
            f"train_score={epoch_row['train_score']:.4f} "
            f"val_score={epoch_row['val_score']:.4f} "
            f"train_return_pct={epoch_row['train_return_pct']:.2f} "
            f"val_return_pct={epoch_row['val_return_pct']:.2f}"
        )

        if epoch_row["val_score"] > best_val_score:
            best_val_score = epoch_row["val_score"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "planner_config": _json_ready(asdict(config.planner)),
                    "simulator_config": _json_ready(asdict(config.simulator)),
                    "symbols": list(bundle.symbols),
                    "feature_names": list(bundle.feature_names),
                    "normalizer": normalizer.to_dict(),
                    "epoch": epoch,
                    "val_score": best_val_score,
                },
                best_checkpoint,
            )

    result = {
        "run_name": run_name,
        "output_dir": str(out_dir),
        "best_checkpoint": str(best_checkpoint),
        "best_val_score": best_val_score,
        "history": history,
    }
    _write_json(out_dir / "metrics.json", result)
    return result


def _run_epoch(
    *,
    model: CrossAssetDailyPlanner,
    loader: DataLoader,
    device: torch.device,
    simulator_config: SimulatorConfig,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    steps = 0

    for batch in loader:
        batch = {key: value.to(device=device, dtype=torch.float32) for key, value in batch.items()}
        plans = model(batch["features"])
        sim_out = simulate_daily_plans(
            hourly_open=batch["hourly_open"],
            hourly_high=batch["hourly_high"],
            hourly_low=batch["hourly_low"],
            hourly_close=batch["hourly_close"],
            hourly_mask=batch["hourly_mask"],
            daily_anchor=batch["daily_anchor"],
            plans=plans,
            config=simulator_config,
        )
        loss, metrics = compute_trading_objective(sim_out, simulator_config)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
        steps += 1

    if steps == 0:
        raise ValueError("DataLoader produced no batches.")
    return {key: value / steps for key, value in totals.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RLgpt daily-plan differentiable trader.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument("--data-root", default=str(DEFAULT_RLGPT_DATA_ROOT))
    parser.add_argument("--forecast-cache-root", default=str(DEFAULT_RLGPT_FORECAST_CACHE_ROOT))
    parser.add_argument("--forecast-horizons", default=default_forecast_horizons_csv())
    parser.add_argument("--validation-days", type=int, default=DEFAULT_RLGPT_VALIDATION_DAYS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_RLGPT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RLGPT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_RLGPT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_RLGPT_WEIGHT_DECAY)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_RLGPT_HIDDEN_DIM)
    parser.add_argument("--depth", type=int, default=DEFAULT_RLGPT_DEPTH)
    parser.add_argument("--heads", type=int, default=DEFAULT_RLGPT_HEADS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_RLGPT_DROPOUT)
    parser.add_argument("--shared-unit-budget", type=float, default=DEFAULT_RLGPT_SHARED_UNIT_BUDGET)
    parser.add_argument("--max-units-per-asset", type=float, default=DEFAULT_RLGPT_MAX_UNITS_PER_ASSET)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_RLGPT_INITIAL_CASH)
    parser.add_argument("--maker-fee-bps", type=float, default=DEFAULT_RLGPT_MAKER_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_RLGPT_SLIPPAGE_BPS)
    parser.add_argument("--fill-buffer-bps", type=float, default=DEFAULT_RLGPT_FILL_BUFFER_BPS)
    parser.add_argument("--fill-temperature-bps", type=float, default=DEFAULT_RLGPT_FILL_TEMPERATURE_BPS)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", default=str(DEFAULT_RLGPT_OUTPUT_ROOT))
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=DEFAULT_RLGPT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_RLGPT_NUM_WORKERS)
    parser.add_argument("--max-train-days", type=int)
    parser.add_argument("--max-val-days", type=int)
    parser.add_argument("--min-history-hours", type=int, default=DEFAULT_RLGPT_MIN_HISTORY_HOURS)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_RLGPT_SEQUENCE_LENGTH)
    parser.add_argument("--max-feature-lookback-hours", type=int, default=DEFAULT_RLGPT_MAX_FEATURE_LOOKBACK_HOURS)
    parser.add_argument("--min-bars-per-day", type=int, default=DEFAULT_RLGPT_MIN_BARS_PER_DAY)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--carry-inventory", action="store_true")
    parser.add_argument("--check-config", action="store_true", help="Print a setup readiness report and exit.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved training config and exit.")
    return parser.parse_args(argv)


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    symbols = normalize_symbol_list(args.symbols.split(","))
    horizons = parse_horizon_list(args.forecast_horizons)
    data_config = DailyPlanDataConfig(
        symbols=symbols,
        data_root=Path(args.data_root),
        forecast_cache_root=Path(args.forecast_cache_root),
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        min_history_hours=args.min_history_hours,
        max_feature_lookback_hours=args.max_feature_lookback_hours,
        min_bars_per_day=args.min_bars_per_day,
        validation_days=args.validation_days,
        cache_only=bool(args.cache_only),
    )
    planner_config = PlannerConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
    )
    simulator_config = SimulatorConfig(
        initial_cash=args.initial_cash,
        shared_unit_budget=args.shared_unit_budget,
        max_units_per_asset=args.max_units_per_asset,
        maker_fee_bps=args.maker_fee_bps,
        slippage_bps=args.slippage_bps,
        fill_buffer_bps=args.fill_buffer_bps,
        fill_temperature_bps=args.fill_temperature_bps,
        carry_inventory=bool(args.carry_inventory),
    )
    return TrainingConfig(
        data=data_config,
        planner=planner_config,
        simulator=simulator_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        max_train_days=args.max_train_days,
        max_val_days=args.max_val_days,
        run_name=args.run_name,
        output_root=Path(args.output_root),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_training_config(args)
    if args.check_config:
        payload = _training_preflight_payload(config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not payload["ready"]:
            raise SystemExit(1)
        return
    if args.print_config:
        print(json.dumps(_training_config_payload(config), indent=2, sort_keys=True))
        return
    result = run_training(config)
    print(json.dumps(result, indent=2))

def _default_run_name(symbols: tuple[str, ...]) -> str:
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    head = "-".join(symbols[:3]).lower()
    return f"{head}_{suffix}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _training_config_payload(config: TrainingConfig) -> dict[str, Any]:
    payload = _json_ready(asdict(config))
    payload["symbol_count"] = len(config.data.symbols)
    payload["forecast_horizon_count"] = len(config.data.forecast_horizons)
    payload["resolved_device"] = str(config.device or "auto")
    return payload


def _training_preflight_payload(config: TrainingConfig) -> dict[str, Any]:
    errors = list(validate_training_config(config))
    warnings_list: list[str] = []
    data_root = Path(config.data.data_root)
    forecast_cache_root = Path(config.data.forecast_cache_root)

    data_root_exists = data_root.exists()
    forecast_cache_root_exists = forecast_cache_root.exists()
    missing_price_files: list[str] = []

    if not data_root_exists:
        errors.append(f"Hourly data root does not exist: {data_root}")
    else:
        missing_price_files = [
            str(data_root / f"{symbol}.csv")
            for symbol in config.data.symbols
            if not (data_root / f"{symbol}.csv").exists()
        ]
        if missing_price_files:
            errors.append(
                "Missing hourly price CSVs for symbols: "
                + ", ".join(Path(path).name for path in missing_price_files)
            )

    if config.data.cache_only and not forecast_cache_root_exists:
        errors.append(
            f"Forecast cache root does not exist in cache-only mode: {forecast_cache_root}"
        )
    elif config.data.cache_only:
        warnings_list.append(
            "Cache-only preflight checks the forecast cache root but not per-symbol forecast parquet coverage."
        )

    payload = _training_config_payload(config)
    payload.update(
        {
            "ready": not errors,
            "errors": errors,
            "warnings": warnings_list,
            "data_root_exists": data_root_exists,
            "forecast_cache_root_exists": forecast_cache_root_exists,
            "missing_price_files": missing_price_files,
        }
    )
    return payload


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
