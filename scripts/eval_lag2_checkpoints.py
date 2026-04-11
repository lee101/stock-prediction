#!/usr/bin/env python3
"""Evaluate binanceneural lag=2 checkpoints with realistic market simulation.

Tests all epochs at decision_lag=0,1,2 with fee=10bps, fill_buffer=5bps.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from src.torch_load_utils import torch_load_compat


def load_checkpoint(ckpt_path: str, input_dim: int, device: str = "cpu"):
    payload = torch_load_compat(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", TrainingConfig())
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, payload


def _max_drawdown(equity: pd.Series) -> float:
    vals = equity.to_numpy(dtype=float)
    if len(vals) < 2:
        return 0.0
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / np.clip(peak, 1e-8, None)
    return float(dd.min())


def eval_checkpoint_on_window(
    model, frame, feature_columns, normalizer, seq_len,
    lag: int, fee_rate: float, fill_buffer_bps: float,
    max_hold_hours: float = 24.0, primary_horizon: int = 1,
    market_order_entry: bool = False,
):
    actions = generate_actions_from_frame(
        model=model,
        frame=frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=seq_len,
        horizon=primary_horizon,
    )
    if actions.empty:
        return None

    sim_cfg = SimulationConfig(
        maker_fee=fee_rate,
        fill_buffer_bps=fill_buffer_bps,
        decision_lag_bars=lag,
        max_hold_hours=int(max_hold_hours),
        initial_cash=10_000.0,
        market_order_entry=market_order_entry,
    )
    sim = BinanceMarketSimulator(sim_cfg)
    result = sim.run(frame, actions)

    metrics = result.metrics
    total_return = metrics.get("total_return", 0.0)
    sortino = metrics.get("sortino", 0.0)
    mdd = _max_drawdown(result.combined_equity)
    num_trades = sum(len(sr.trades) for sr in result.per_symbol.values())

    return {
        "total_return": float(total_return),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "num_trades": int(num_trades),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--lags", default="0,1,2")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--window-days", type=int, default=70)
    parser.add_argument("--max-hold-hours", type=float, default=24.0)
    parser.add_argument("--primary-horizon", type=int, default=1,
                        help="Chronos horizon for price anchoring (1 or 24)")
    parser.add_argument("--market-order-entry", action="store_true",
                        help="Use open price for buys instead of limit")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    lags = [int(x) for x in args.lags.split(",")]

    horizons = (args.primary_horizon,) + tuple(
        h for h in (1, 24) if h != args.primary_horizon
    )
    dataset_cfg = DatasetConfig(
        symbol=args.symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=72,
        validation_days=args.window_days,
        forecast_horizons=horizons,
        cache_only=True,
    )

    data = BinanceHourlyDataModule(dataset_cfg)
    val_frame = data.val_dataset.frame.copy()
    print(f"{args.symbol}: {len(val_frame)} val bars ({len(val_frame)/24:.0f} days)")
    print(f"Features: {len(data.feature_columns)}")

    epochs = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not epochs:
        print("No epoch checkpoints found")
        return

    results = []
    for ckpt_path in epochs:
        ep_num = int(ckpt_path.stem.split("_")[1])
        model, payload = load_checkpoint(str(ckpt_path), input_dim=len(data.feature_columns))

        for lag in lags:
            try:
                entry = eval_checkpoint_on_window(
                    model=model,
                    frame=val_frame,
                    feature_columns=data.feature_columns,
                    normalizer=data.normalizer,
                    seq_len=72,
                    lag=lag,
                    fee_rate=args.fee_rate,
                    fill_buffer_bps=args.fill_buffer_bps,
                    max_hold_hours=args.max_hold_hours,
                    primary_horizon=args.primary_horizon,
                    market_order_entry=args.market_order_entry,
                )
                if entry is None:
                    continue
                entry["epoch"] = ep_num
                entry["lag"] = lag
                results.append(entry)
                ret_pct = entry["total_return"] * 100
                print(f"  ep{ep_num:02d} lag={lag}: ret={ret_pct:+6.2f}% sort={entry['sortino']:+7.2f} "
                      f"dd={entry['max_drawdown']:.2%} trades={entry['num_trades']}")
            except Exception as e:
                print(f"  ep{ep_num:02d} lag={lag}: ERROR {e}")

    out_path = ckpt_dir / "holdout_eval.json"
    with open(out_path, "w") as f:
        json.dump({"symbol": args.symbol, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*70}")
    print(f"{'Epoch':>5} | {'Lag=0':>12} | {'Lag=1':>12} | {'Lag=2':>12}")
    print(f"{'':>5} | {'Ret%':>6} {'Sort':>5} | {'Ret%':>6} {'Sort':>5} | {'Ret%':>6} {'Sort':>5}")
    print(f"{'-'*70}")
    for ep in sorted(set(r["epoch"] for r in results)):
        parts = []
        for lag in lags:
            match = [r for r in results if r["epoch"] == ep and r["lag"] == lag]
            if match:
                r = match[0]
                parts.append(f"{r['total_return']*100:+6.2f} {r['sortino']:+5.1f}")
            else:
                parts.append(f"{'N/A':>12}")
        print(f"  {ep:3d} | {' | '.join(parts)}")


if __name__ == "__main__":
    main()
