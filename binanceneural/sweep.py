from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .inference import generate_actions_from_frame
from .marketsimulator import BinanceMarketSimulator, SimulationConfig
from .model import BinancePolicyBase, build_policy, policy_config_from_payload
from .data import FeatureNormalizer


@dataclass
class SweepResult:
    intensity_scale: float
    price_offset_pct: float
    total_return: float
    sortino: float


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
    # Ensure prices are ordered
    adjusted["sell_price"] = np.maximum(adjusted["sell_price"], adjusted["buy_price"])
    return adjusted


def sweep_action_overrides(
    *,
    model: BinancePolicyBase,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    normalizer: FeatureNormalizer,
    sequence_length: int,
    horizon: int,
    intensity_scales: Sequence[float],
    price_offsets: Sequence[float],
    initial_cash: float = 10_000.0,
) -> List[SweepResult]:
    base_actions = generate_actions_from_frame(
        model=model,
        frame=frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=sequence_length,
        horizon=horizon,
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
    parser = argparse.ArgumentParser(description="Sweep action scaling to optimize validation PnL/Sortino.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--intensity", nargs="+", type=float, default=[0.6, 0.8, 1.0, 1.2])
    parser.add_argument("--offset", nargs="+", type=float, default=[0.0, 0.0002, 0.0005])
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    args = parser.parse_args()

    from .config import DatasetConfig, TrainingConfig
    from .data import BinanceHourlyDataModule
    import torch

    data_cfg = DatasetConfig(symbol=args.symbol, sequence_length=args.sequence_length)
    data = BinanceHourlyDataModule(data_cfg)
    frame = data.val_dataset.frame

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    cfg = payload.get("config", TrainingConfig(sequence_length=args.sequence_length))
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=len(data.feature_columns), state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    results = sweep_action_overrides(
        model=model,
        frame=frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
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
