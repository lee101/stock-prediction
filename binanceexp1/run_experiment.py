from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from binanceneural.config import TrainingConfig
from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig
from binanceneural.model import (
    BinancePolicyBase,
    align_state_dict_input_dim,
    build_policy,
    policy_config_from_payload,
)
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame

from .config import DatasetConfig, ExperimentConfig
from .data import BinanceExp1DataModule
from .inference import blend_actions, generate_actions_multi_context
from .sweep import apply_action_overrides


@dataclass
class ExperimentResult:
    checkpoint: Optional[Path]
    total_return: float
    sortino: float


def _load_model(checkpoint_path: Path, input_dim: int, default_cfg: TrainingConfig) -> BinancePolicyBase:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        payload = {"state_dict": payload}
    state_dict = payload.get("state_dict", payload)
    if isinstance(state_dict, dict):
        state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", default_cfg)
    payload_cfg = cfg if isinstance(cfg, dict) else getattr(cfg, "__dict__", {})
    policy_cfg = policy_config_from_payload(
        payload_cfg,
        input_dim=input_dim,
        state_dict=state_dict if isinstance(state_dict, dict) else None,
    )
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def train_model(data: BinanceExp1DataModule, args: argparse.Namespace) -> Path:
    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        run_name=args.run_name,
        dry_train_steps=args.dry_train_steps,
        use_compile=args.use_compile,
    )
    trainer = BinanceHourlyTrainer(cfg, data)
    artifacts = trainer.train()
    if artifacts.best_checkpoint is None:
        raise RuntimeError("No checkpoint saved during training")
    return artifacts.best_checkpoint


def evaluate_model(
    *,
    model: BinancePolicyBase,
    data: BinanceExp1DataModule,
    horizon: int,
    aggregate: bool,
    experiment_cfg: ExperimentConfig,
    blend_horizons: Optional[Sequence[int]] = None,
    blend_weights: Optional[Sequence[float]] = None,
    enable_probe_mode: bool = False,
    probe_notional: float = 1.0,
    max_hold_hours: Optional[int] = None,
    intensity_scale: Optional[float] = None,
    price_offset_pct: Optional[float] = None,
) -> ExperimentResult:
    val_frame = data.val_dataset.frame

    def _actions_for_horizon(target_horizon: int):
        if aggregate:
            agg = generate_actions_multi_context(
                model=model,
                frame=val_frame,
                feature_columns=data.feature_columns,
                normalizer=data.normalizer,
                base_sequence_length=data.config.sequence_length,
                horizon=target_horizon,
                experiment=experiment_cfg,
            )
            return agg.aggregated
        return generate_actions_from_frame(
            model=model,
            frame=val_frame,
            feature_columns=data.feature_columns,
            normalizer=data.normalizer,
            sequence_length=data.config.sequence_length,
            horizon=target_horizon,
        )

    if blend_horizons:
        actions_list = [_actions_for_horizon(h) for h in blend_horizons]
        actions = blend_actions(actions_list, weights=blend_weights)
    else:
        actions = _actions_for_horizon(horizon)

    if intensity_scale is not None or price_offset_pct is not None:
        actions = apply_action_overrides(
            actions,
            intensity_scale=float(intensity_scale or 1.0),
            price_offset_pct=float(price_offset_pct or 0.0),
        )

    sim = BinanceMarketSimulator(
        SimulationConfig(
            initial_cash=10_000.0,
            enable_probe_mode=enable_probe_mode,
            probe_notional=probe_notional,
            max_hold_hours=max_hold_hours,
        )
    )
    result = sim.run(val_frame, actions)
    metrics = result.metrics
    return ExperimentResult(
        checkpoint=None,
        total_return=float(metrics.get("total_return", 0.0)),
        sortino=float(metrics.get("sortino", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run binanceexp1 training + evaluation.")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--dry-train-steps", type=int, default=300)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--run-name", default="binanceexp1_run")
    parser.add_argument("--checkpoint", help="Skip training and evaluate this checkpoint")
    parser.add_argument("--aggregate", action="store_true", help="Use multi-context aggregation")
    parser.add_argument("--context-lengths", default="64,96,192")
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--blend-horizons", help="Comma-separated horizons to blend (e.g., 1,24)")
    parser.add_argument("--blend-weights", help="Comma-separated weights for blended horizons")
    parser.add_argument(
        "--forecast-horizons",
        help="Comma-separated Chronos horizons to load (e.g., 1,24 or 24,1)",
    )
    parser.add_argument("--probe-after-loss", action="store_true", help="Enable probe trades after a losing sell")
    parser.add_argument("--probe-notional", type=float, default=1.0, help="Notional size for probe trades")
    parser.add_argument("--max-hold-hours", type=int, default=None, help="Force close positions after N hours")
    parser.add_argument("--intensity-scale", type=float, default=None, help="Scale trade intensity during evaluation")
    parser.add_argument("--price-offset-pct", type=float, default=None, help="Offset buy/sell prices during evaluation")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--validation-days",
        type=float,
        default=None,
        help="Override validation window length in days (e.g., 10 for a 10-day sim).",
    )
    parser.add_argument(
        "--data-root",
        default=str(DatasetConfig().data_root),
        help="Root directory for hourly data (e.g., trainingdatahourly/crypto or trainingdatahourly/stocks).",
    )
    args = parser.parse_args()

    ctx_lengths = tuple(int(x) for x in args.context_lengths.split(",") if x)
    experiment_cfg = ExperimentConfig(context_lengths=ctx_lengths, trim_ratio=args.trim_ratio)

    if args.forecast_horizons:
        forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x)
    else:
        forecast_horizons = DatasetConfig().forecast_horizons
    data_cfg_kwargs = dict(
        symbol=args.symbol,
        data_root=Path(args.data_root),
        sequence_length=args.sequence_length,
        cache_only=args.cache_only,
        forecast_horizons=forecast_horizons,
    )
    if args.validation_days is not None:
        data_cfg_kwargs["validation_days"] = float(args.validation_days)
    data_cfg = DatasetConfig(**data_cfg_kwargs)
    data = BinanceExp1DataModule(data_cfg)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if checkpoint_path is None:
        checkpoint_path = train_model(data, args)

    model = _load_model(checkpoint_path, len(data.feature_columns), TrainingConfig(sequence_length=args.sequence_length))
    blend_horizons = None
    blend_weights = None
    if args.blend_horizons:
        blend_horizons = [int(x) for x in args.blend_horizons.split(",") if x]
        missing = [h for h in blend_horizons if h not in data_cfg.forecast_horizons]
        if missing:
            raise ValueError(f"Blend horizons {missing} not in forecast_horizons {data_cfg.forecast_horizons}.")
    if args.blend_weights:
        blend_weights = [float(x) for x in args.blend_weights.split(",") if x]

    if blend_horizons is None and args.horizon not in data_cfg.forecast_horizons:
        raise ValueError(f"Horizon {args.horizon} not in forecast_horizons {data_cfg.forecast_horizons}.")

    result = evaluate_model(
        model=model,
        data=data,
        horizon=args.horizon,
        aggregate=args.aggregate,
        experiment_cfg=experiment_cfg,
        blend_horizons=blend_horizons,
        blend_weights=blend_weights,
        enable_probe_mode=args.probe_after_loss,
        probe_notional=args.probe_notional,
        max_hold_hours=args.max_hold_hours,
        intensity_scale=args.intensity_scale,
        price_offset_pct=args.price_offset_pct,
    )
    result.checkpoint = checkpoint_path
    print(f"Checkpoint: {checkpoint_path}")
    print(f"total_return: {result.total_return:.4f}")
    print(f"sortino: {result.sortino:.4f}")


if __name__ == "__main__":
    main()
