from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch

from binanceneural.config import TrainingConfig
from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from binanceneural.trainer import BinanceHourlyTrainer, TrainingArtifacts
from binanceexp1.sweep import apply_action_overrides

from .config import DatasetConfig, ExperimentConfig
from .data import AlpacaHourlyDataModule, AlpacaMultiSymbolDataModule
from .inference import blend_actions, generate_actions_multi_context
from .marketsimulator import AlpacaMarketSimulator, SimulationConfig, save_trade_plot
from src.torch_device_utils import require_cuda as require_cuda_device


@dataclass
class ExperimentResult:
    checkpoint: Optional[Path]
    total_return: float
    sortino: float


def _load_model(checkpoint_path: Path, input_dim: int, sequence_length: int) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", {})
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    policy_cfg.max_len = max(policy_cfg.max_len, sequence_length)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _resolve_device(device_arg: Optional[str], *, symbol: str) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type != "cuda":
            raise RuntimeError(f"GPU required for training/inference; received device={device_arg!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for training/inference but CUDA is not available.")
        return device
    return require_cuda_device("alpaca hourly training", symbol=symbol, allow_fallback=False)


def train_model(
    data: AlpacaHourlyDataModule | AlpacaMultiSymbolDataModule,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> TrainingArtifacts:
    maker_fee = float(args.maker_fee) if args.maker_fee is not None else float(data.asset_meta.maker_fee)
    periods_per_year = float(args.periods_per_year) if args.periods_per_year is not None else float(
        data.asset_meta.periods_per_year
    )
    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        run_name=args.run_name,
        dry_train_steps=args.dry_train_steps,
        use_compile=args.use_compile,
        maker_fee=maker_fee,
        periods_per_year=periods_per_year,
        device=str(device),
    )
    trainer = BinanceHourlyTrainer(cfg, data)
    artifacts = trainer.train()
    if artifacts.best_checkpoint is None:
        raise RuntimeError("No checkpoint saved during training")
    return artifacts


def _slice_eval_window(
    actions: pd.DataFrame,
    bars: pd.DataFrame,
    eval_days: Optional[float],
    eval_hours: Optional[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if actions.empty or bars.empty:
        return actions, bars
    hours = 0.0
    if eval_days:
        hours = max(hours, float(eval_days) * 24.0)
    if eval_hours:
        hours = max(hours, float(eval_hours))
    if hours <= 0:
        return actions, bars
    ts_end = pd.to_datetime(bars["timestamp"], utc=True).max()
    if pd.isna(ts_end):
        return actions, bars
    ts_start = ts_end - pd.Timedelta(hours=hours)
    bars_slice = bars[pd.to_datetime(bars["timestamp"], utc=True) >= ts_start]
    actions_slice = actions[pd.to_datetime(actions["timestamp"], utc=True) >= ts_start]
    return actions_slice.reset_index(drop=True), bars_slice.reset_index(drop=True)


def evaluate_model(
    *,
    model: torch.nn.Module,
    data: AlpacaHourlyDataModule | AlpacaMultiSymbolDataModule,
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
    enforce_market_hours: bool = True,
    close_at_eod: bool = True,
    maker_fee: Optional[float] = None,
    output_dir: Optional[Path] = None,
    eval_days: Optional[float] = None,
    eval_hours: Optional[float] = None,
    device: Optional[torch.device] = None,
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
                device=device,
            )
            return agg.aggregated
        return generate_actions_from_frame(
            model=model,
            frame=val_frame,
            feature_columns=data.feature_columns,
            normalizer=data.normalizer,
            sequence_length=data.config.sequence_length,
            horizon=target_horizon,
            device=device,
            require_gpu=True,
        )

    if blend_horizons:
        actions_list = [_actions_for_horizon(h) for h in blend_horizons]
        actions = blend_actions(actions_list, weights=blend_weights)
    else:
        actions = _actions_for_horizon(horizon)

    if eval_days or eval_hours:
        actions, val_frame = _slice_eval_window(actions, val_frame, eval_days, eval_hours)

    if intensity_scale is not None or price_offset_pct is not None:
        actions = apply_action_overrides(
            actions,
            intensity_scale=float(intensity_scale or 1.0),
            price_offset_pct=float(price_offset_pct or 0.0),
        )

    fee_value = float(maker_fee) if maker_fee is not None else float(data.asset_meta.maker_fee)
    sim = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=10_000.0,
            enable_probe_mode=enable_probe_mode,
            probe_notional=probe_notional,
            max_hold_hours=max_hold_hours,
            enforce_market_hours=enforce_market_hours,
            close_at_eod=close_at_eod,
            fee_by_symbol={data.asset_meta.symbol: fee_value},
            periods_per_year_by_symbol={data.asset_meta.symbol: data.asset_meta.periods_per_year},
        )
    )
    result = sim.run(val_frame, actions)
    metrics = result.metrics

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "actions.csv").write_text(actions.to_csv(index=False))
        per_symbol = result.per_symbol[data.asset_meta.symbol]
        (output_dir / "per_hour.csv").write_text(per_symbol.per_hour.to_csv(index=False))
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        save_trade_plot(data.asset_meta.symbol, val_frame, actions, per_symbol, output_dir / "trade_plot.png")

    return ExperimentResult(
        checkpoint=None,
        total_return=float(metrics.get("total_return", 0.0)),
        sortino=float(metrics.get("sortino", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alpaca hourly experiment training + evaluation.")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--dry-train-steps", type=int, default=300)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--device", default=None, help="Override training device (e.g., cuda, cuda:0).")
    parser.add_argument("--run-name", default="alpaca_hourly_run")
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
    parser.add_argument("--maker-fee", type=float, default=None, help="Override maker fee (per-side)")
    parser.add_argument("--periods-per-year", type=float, default=None, help="Override periods-per-year for annualization")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory for hourly data (defaults to crypto/stocks based on symbol).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-enforce-market-hours", action="store_true")
    parser.add_argument("--no-close-at-eod", action="store_true")
    parser.add_argument("--eval-days", type=float, default=None, help="Limit evaluation to last N days")
    parser.add_argument("--eval-hours", type=float, default=None, help="Limit evaluation to last N hours")
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbols for multi-symbol training")
    parser.add_argument("--allow-mixed-asset", action="store_true", help="Allow mixing crypto + stocks for training")
    args = parser.parse_args()

    device = _resolve_device(args.device, symbol=args.symbol)

    ctx_lengths = tuple(int(x) for x in args.context_lengths.split(",") if x)
    experiment_cfg = ExperimentConfig(context_lengths=ctx_lengths, trim_ratio=args.trim_ratio)

    if args.forecast_horizons:
        forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x)
    else:
        forecast_horizons = DatasetConfig().forecast_horizons
    data_cfg = DatasetConfig(
        symbol=args.symbol,
        data_root=Path(args.data_root) if args.data_root else None,
        sequence_length=args.sequence_length,
        cache_only=args.cache_only,
        forecast_horizons=forecast_horizons,
        allow_mixed_asset_class=args.allow_mixed_asset,
    )
    if args.symbols:
        symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
        data = AlpacaMultiSymbolDataModule(symbols, data_cfg)
    else:
        data = AlpacaHourlyDataModule(data_cfg)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    training_artifacts: Optional[TrainingArtifacts] = None
    if checkpoint_path is None:
        training_artifacts = train_model(data, args, device=device)
        checkpoint_path = training_artifacts.best_checkpoint

    model = _load_model(checkpoint_path, len(data.feature_columns), args.sequence_length)
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
        enforce_market_hours=not args.no_enforce_market_hours,
        close_at_eod=not args.no_close_at_eod,
        maker_fee=args.maker_fee,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        eval_days=args.eval_days,
        eval_hours=args.eval_hours,
        device=device,
    )
    if training_artifacts is not None and args.output_dir:
        history_payload = [entry.__dict__ for entry in training_artifacts.history]
        (Path(args.output_dir) / "training_history.json").write_text(json.dumps(history_payload, indent=2))
    result.checkpoint = checkpoint_path
    print(f"Checkpoint: {checkpoint_path}")
    print(f"total_return: {result.total_return:.4f}")
    print(f"sortino: {result.sortino:.4f}")


if __name__ == "__main__":
    main()
