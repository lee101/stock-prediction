from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from binanceneural.config import TrainingConfig
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig
from binanceneural.model import (
    BinancePolicyBase,
    align_state_dict_input_dim,
    build_policy,
    policy_config_from_payload,
)

from .config import DatasetConfig, ExperimentConfig
from .data import BinanceExp1DataModule, FeatureNormalizer
from .inference import blend_actions, generate_actions_multi_context


@dataclass
class SweepResult:
    intensity_scale: float
    price_offset_pct: float
    total_return: float
    sortino: float


def _infer_max_len(state_dict: Mapping[str, torch.Tensor], cfg: TrainingConfig) -> int:
    """Infer the model's maximum supported sequence length from a checkpoint."""

    pe = state_dict.get("pos_encoding.pe")
    if isinstance(pe, torch.Tensor) and pe.ndim >= 1:
        return int(pe.shape[0])
    return int(getattr(cfg, "sequence_length", 0) or 0)


def apply_action_overrides(
    actions: pd.DataFrame,
    *,
    intensity_scale: float,
    price_offset_pct: float,
) -> pd.DataFrame:
    adjusted = actions.copy()
    for col in ("buy_amount", "sell_amount", "trade_amount"):
        if col in adjusted.columns:
            adjusted[col] = np.clip(adjusted[col] * intensity_scale, 0.0, 100.0)
    adjusted["buy_price"] = adjusted["buy_price"] * (1.0 - price_offset_pct)
    adjusted["sell_price"] = adjusted["sell_price"] * (1.0 + price_offset_pct)
    adjusted["sell_price"] = np.maximum(adjusted["sell_price"], adjusted["buy_price"])
    return adjusted


def _actions_for_horizon(
    *,
    model: BinancePolicyBase,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    normalizer: FeatureNormalizer,
    sequence_length: int,
    horizon: int,
    aggregate: bool,
    experiment_cfg: ExperimentConfig,
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
        )
        return agg.aggregated
    return generate_actions_from_frame(
        model=model,
        frame=frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=sequence_length,
        horizon=horizon,
    )


def sweep_action_overrides(
    *,
    model: BinancePolicyBase,
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
        )

    results: List[SweepResult] = []
    simulator = BinanceMarketSimulator(SimulationConfig(initial_cash=initial_cash))
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
    parser = argparse.ArgumentParser(description="Sweep action scaling for binanceexp1 models.")
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
    parser.add_argument(
        "--data-root",
        default=str(DatasetConfig().data_root),
        help="Root directory for hourly data (e.g., trainingdatahourly/crypto or trainingdatahourly/stocks).",
    )
    args = parser.parse_args()

    forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x)
    data_cfg = DatasetConfig(
        symbol=args.symbol,
        data_root=Path(args.data_root),
        sequence_length=args.sequence_length,
        cache_only=args.cache_only,
        forecast_horizons=forecast_horizons,
    )
    data = BinanceExp1DataModule(data_cfg)
    frame = data.val_dataset.frame

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        payload = {"state_dict": payload}
    state_dict = payload.get("state_dict", payload)
    input_dim = len(data.feature_columns)
    if isinstance(state_dict, dict):
        state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", TrainingConfig(sequence_length=args.sequence_length))
    payload_cfg = cfg if isinstance(cfg, dict) else getattr(cfg, "__dict__", {})
    policy_cfg = policy_config_from_payload(
        payload_cfg,
        input_dim=input_dim,
        state_dict=state_dict if isinstance(state_dict, dict) else None,
    )
    policy_cfg.max_len = max(int(policy_cfg.max_len or 0), _infer_max_len(state_dict if isinstance(state_dict, dict) else {}, TrainingConfig(sequence_length=args.sequence_length)))
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    ctx_lengths = tuple(int(x) for x in args.context_lengths.split(",") if x)
    experiment_cfg = ExperimentConfig(context_lengths=ctx_lengths, trim_ratio=args.trim_ratio)

    blend_horizons = None
    blend_weights = None
    if args.blend_horizons:
        blend_horizons = [int(x) for x in args.blend_horizons.split(",") if x]
    if args.blend_weights:
        blend_weights = [float(x) for x in args.blend_weights.split(",") if x]

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
    )

    results.sort(key=lambda r: (r.sortino, r.total_return), reverse=True)
    top = results[:10]
    print("Top sweep results:")
    for item in top:
        print(
            f"intensity={item.intensity_scale:.2f} offset={item.price_offset_pct:.5f} "
            f"return={item.total_return:.4f} sortino={item.sortino:.4f}"
        )


if __name__ == "__main__":
    main()
