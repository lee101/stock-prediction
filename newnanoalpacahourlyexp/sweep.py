from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from binanceneural.config import TrainingConfig
from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload

from binanceexp1.sweep import apply_action_overrides

from .config import DatasetConfig, ExperimentConfig
from .data import AlpacaHourlyDataModule, FeatureNormalizer
from .inference import blend_actions, generate_actions_multi_context
from .marketsimulator import AlpacaMarketSimulator, SimulationConfig
from src.torch_device_utils import require_cuda as require_cuda_device


@dataclass
class SweepResult:
    intensity_scale: float
    price_offset_pct: float
    total_return: float
    sortino: float


def _actions_for_horizon(
    *,
    model: torch.nn.Module,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    normalizer: FeatureNormalizer,
    sequence_length: int,
    horizon: int,
    aggregate: bool,
    experiment_cfg: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    if aggregate:
        agg = generate_actions_multi_context(
            model=model,
            frame=frame,
            feature_columns=feature_columns,
            normalizer=normalizer,
            base_sequence_length=sequence_length,
            horizon=horizon,
            experiment=experiment_cfg,
            device=device,
        )
        return agg.aggregated
    return generate_actions_from_frame(
        model=model,
        frame=frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=sequence_length,
        horizon=horizon,
        device=device,
        require_gpu=True,
    )


def sweep_action_overrides(
    *,
    model: torch.nn.Module,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    normalizer: FeatureNormalizer,
    sequence_length: int,
    horizon: int,
    aggregate: bool,
    experiment_cfg: ExperimentConfig,
    blend_horizons: Optional[Sequence[int]],
    blend_weights: Optional[Sequence[float]],
    intensity_scales: Sequence[float],
    price_offsets: Sequence[float],
    initial_cash: float = 10_000.0,
    enforce_market_hours: bool = True,
    close_at_eod: bool = True,
    maker_fee: Optional[float] = None,
    periods_per_year: Optional[float] = None,
    symbol: str = "",
    eval_days: Optional[float] = None,
    eval_hours: Optional[float] = None,
    device: torch.device,
) -> List[SweepResult]:
    if blend_horizons:
        actions_list = [
            _actions_for_horizon(
                model=model,
                frame=frame,
                feature_columns=feature_columns,
                normalizer=normalizer,
                sequence_length=sequence_length,
                horizon=h,
                aggregate=aggregate,
                experiment_cfg=experiment_cfg,
                device=device,
            )
            for h in blend_horizons
        ]
        base_actions = blend_actions(actions_list, weights=blend_weights)
    else:
        base_actions = _actions_for_horizon(
            model=model,
            frame=frame,
            feature_columns=feature_columns,
            normalizer=normalizer,
            sequence_length=sequence_length,
            horizon=horizon,
            aggregate=aggregate,
            experiment_cfg=experiment_cfg,
            device=device,
        )

    if eval_days or eval_hours:
        base_actions, frame = _slice_eval_window(base_actions, frame, eval_days, eval_hours)

    results: List[SweepResult] = []
    fee_by_symbol = {symbol: float(maker_fee)} if maker_fee is not None and symbol else None
    periods_by_symbol = {symbol: float(periods_per_year)} if periods_per_year is not None and symbol else None
    simulator = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=initial_cash,
            enforce_market_hours=enforce_market_hours,
            close_at_eod=close_at_eod,
            fee_by_symbol=fee_by_symbol,
            periods_per_year_by_symbol=periods_by_symbol,
        )
    )
    for intensity in intensity_scales:
        for offset in price_offsets:
            adjusted = apply_action_overrides(
                base_actions,
                intensity_scale=float(intensity),
                price_offset_pct=float(offset),
            )
            sim_result = simulator.run(frame, adjusted)
            metrics = sim_result.metrics
            results.append(
                SweepResult(
                    intensity_scale=float(intensity),
                    price_offset_pct=float(offset),
                    total_return=float(metrics["total_return"]),
                    sortino=float(metrics["sortino"]),
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep action scaling for Alpaca hourly models.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--context-lengths", default="64,96,192")
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--blend-horizons")
    parser.add_argument("--blend-weights")
    parser.add_argument("--intensity", nargs="+", type=float, default=[0.6, 0.8, 1.0, 1.2])
    parser.add_argument("--offset", nargs="+", type=float, default=[0.0, 0.0002, 0.0005])
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--maker-fee", type=float, default=None)
    parser.add_argument("--periods-per-year", type=float, default=None)
    parser.add_argument("--no-enforce-market-hours", action="store_true")
    parser.add_argument("--no-close-at-eod", action="store_true")
    parser.add_argument("--eval-days", type=float, default=None, help="Limit evaluation to last N days")
    parser.add_argument("--eval-hours", type=float, default=None, help="Limit evaluation to last N hours")
    parser.add_argument("--device", default=None, help="Override inference device (e.g., cuda, cuda:0).")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
        if device.type != "cuda":
            raise RuntimeError(f"GPU required for inference; received device={args.device!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for inference but CUDA is not available.")
    else:
        device = require_cuda_device("sweep inference", allow_fallback=False)

    forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x)
    data_cfg = DatasetConfig(
        symbol=args.symbol,
        data_root=Path(args.data_root) if args.data_root else None,
        sequence_length=args.sequence_length,
        cache_only=args.cache_only,
        forecast_horizons=forecast_horizons,
    )
    data = AlpacaHourlyDataModule(data_cfg)
    frame = data.val_dataset.frame

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=len(data.feature_columns))
    cfg = payload.get("config", TrainingConfig(sequence_length=args.sequence_length))
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=len(data.feature_columns), state_dict=state_dict)
    policy_cfg.max_len = max(policy_cfg.max_len, args.sequence_length)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    ctx_lengths = tuple(int(x) for x in args.context_lengths.split(",") if x)
    experiment_cfg = ExperimentConfig(context_lengths=ctx_lengths, trim_ratio=args.trim_ratio)

    blend_horizons = [int(x) for x in args.blend_horizons.split(",") if x] if args.blend_horizons else None
    blend_weights = [float(x) for x in args.blend_weights.split(",") if x] if args.blend_weights else None

    results = sweep_action_overrides(
        model=model,
        frame=frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        aggregate=args.aggregate,
        experiment_cfg=experiment_cfg,
        blend_horizons=blend_horizons,
        blend_weights=blend_weights,
        intensity_scales=args.intensity,
        price_offsets=args.offset,
        initial_cash=args.initial_cash,
        enforce_market_hours=not args.no_enforce_market_hours,
        close_at_eod=not args.no_close_at_eod,
        maker_fee=args.maker_fee or data.asset_meta.maker_fee,
        periods_per_year=args.periods_per_year or data.asset_meta.periods_per_year,
        symbol=data.asset_meta.symbol,
        eval_days=args.eval_days,
        eval_hours=args.eval_hours,
        device=device,
    )

    for result in results:
        print(
            f"intensity={result.intensity_scale:.3f} offset={result.price_offset_pct:.5f} "
            f"return={result.total_return:.4f} sortino={result.sortino:.4f}"
        )


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


if __name__ == "__main__":
    main()
