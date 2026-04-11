#!/usr/bin/env python3
"""Evaluate rebalance-mode checkpoints with position-target sim.

Tests all epochs at decision_lag=0,1,2 with fee=10bps.
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
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from differentiable_loss_utils import simulate_rebalance
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


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.clip(peak, 1e-8, None)
    return float(dd.min())


def generate_allocations_from_frame(
    model, frame, feature_columns, normalizer, sequence_length, device=None,
):
    """Slide window over frame, extract allocation fraction from model."""
    if device is None:
        device = torch.device("cpu")

    allocations = []
    timestamps = []

    for start in range(0, len(frame) - sequence_length + 1):
        end = start + sequence_length
        window = frame.iloc[start:end]

        feats = normalizer.transform(window[list(feature_columns)].values)
        feats_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.inference_mode():
            outputs = model(feats_t)
            alloc_logits = outputs.get("allocation_logits", outputs.get("buy_amount_logits"))
            alloc = torch.sigmoid(alloc_logits.squeeze(-1))
            allocations.append(float(alloc[0, -1]))

        timestamps.append(window.index[-1] if hasattr(window.index, '__getitem__') else end - 1)

    return pd.DataFrame({
        "allocation": allocations,
    }, index=timestamps[:len(allocations)])


def eval_checkpoint_on_window(
    model, frame, feature_columns, normalizer, seq_len,
    lag: int, fee_rate: float,
):
    alloc_df = generate_allocations_from_frame(
        model=model, frame=frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=seq_len,
    )
    if alloc_df.empty:
        return None

    aligned = frame.iloc[seq_len - 1:seq_len - 1 + len(alloc_df)]
    closes_t = torch.tensor(aligned["close"].values, dtype=torch.float32)
    opens_t = torch.tensor(aligned["open"].values, dtype=torch.float32) if "open" in aligned.columns else None
    alloc_t = torch.tensor(alloc_df["allocation"].values, dtype=torch.float32)

    result = simulate_rebalance(
        closes=closes_t,
        opens=opens_t,
        allocation=alloc_t,
        maker_fee=fee_rate,
        initial_cash=10_000.0,
        decision_lag_bars=lag,
    )

    values = result.portfolio_values.numpy()
    total_return = float((values[-1] - 10000.0) / 10000.0)
    returns_np = result.returns.numpy()
    neg_returns = returns_np[returns_np < 0]
    downside_std = float(np.std(neg_returns)) if len(neg_returns) > 0 else 1e-8
    mean_ret = float(np.mean(returns_np))
    sortino = mean_ret / max(downside_std, 1e-8) * np.sqrt(8760)
    mdd = _max_drawdown(values)
    num_rebalances = int((result.executed_buys > 0).sum() + (result.executed_sells > 0).sum())

    return {
        "total_return": total_return,
        "sortino": sortino,
        "max_drawdown": mdd,
        "num_rebalances": num_rebalances,
        "mean_allocation": float(alloc_df["allocation"].mean()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--lags", default="0,1,2")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--window-days", type=int, default=70)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    lags = [int(x) for x in args.lags.split(",")]

    dataset_cfg = DatasetConfig(
        symbol=args.symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=72,
        validation_days=args.window_days,
        forecast_horizons=(1, 24),
        cache_only=True,
    )

    data = BinanceHourlyDataModule(dataset_cfg)
    val_frame = data.val_dataset.frame.copy()
    print(f"{args.symbol}: {len(val_frame)} val bars ({len(val_frame)/24:.0f} days)")

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
                    model=model, frame=val_frame,
                    feature_columns=data.feature_columns,
                    normalizer=data.normalizer, seq_len=72,
                    lag=lag, fee_rate=args.fee_rate,
                )
                if entry is None:
                    continue
                entry["epoch"] = ep_num
                entry["lag"] = lag
                results.append(entry)
                ret_pct = entry["total_return"] * 100
                print(f"  ep{ep_num:02d} lag={lag}: ret={ret_pct:+6.2f}% sort={entry['sortino']:+7.2f} "
                      f"dd={entry['max_drawdown']:.2%} rebal={entry['num_rebalances']} "
                      f"alloc={entry['mean_allocation']:.2f}")
            except Exception as e:
                import traceback
                print(f"  ep{ep_num:02d} lag={lag}: ERROR {e}")
                traceback.print_exc()

    out_path = ckpt_dir / "rebalance_eval.json"
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
