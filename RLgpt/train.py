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

from RLgpt.config import DailyPlanDataConfig, PlannerConfig, SimulatorConfig, TrainingConfig
from RLgpt.data import DailyPlanTensorDataset, TensorNormalizer, prepare_daily_plan_tensors
from RLgpt.model import CrossAssetDailyPlanner
from RLgpt.simulator import compute_trading_objective, simulate_daily_plans


def run_training(config: TrainingConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
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

    model = CrossAssetDailyPlanner(input_dim=train_bundle.feature_dim, config=config.planner).to(device)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RLgpt daily-plan differentiable trader.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument("--data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--forecast-cache-root", default="binanceneural/forecast_cache")
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shared-unit-budget", type=float, default=20.0)
    parser.add_argument("--max-units-per-asset", type=float, default=10.0)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--maker-fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--fill-temperature-bps", type=float, default=8.0)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", default="experiments/RLgpt")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-days", type=int)
    parser.add_argument("--max-val-days", type=int)
    parser.add_argument("--min-history-hours", type=int, default=24 * 45)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--max-feature-lookback-hours", type=int, default=24 * 7)
    parser.add_argument("--min-bars-per-day", type=int, default=1)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--carry-inventory", action="store_true")
    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    symbols = tuple(token.strip().upper() for token in args.symbols.split(",") if token.strip())
    horizons = tuple(int(token.strip()) for token in args.forecast_horizons.split(",") if token.strip())
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


def main() -> None:
    args = parse_args()
    result = run_training(build_training_config(args))
    print(json.dumps(result, indent=2))


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
