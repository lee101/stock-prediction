#!/usr/bin/env python3
"""Ensemble evaluation: average actions from multiple residual models."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui_rlcuda.residual_env import (
    ResidualLeverageEnv,
    ResidualLeverageEnvConfig,
    evaluate_baseline_episode,
    HOURLY_PERIODS_PER_YEAR,
)


class _EnsembleModel:
    """Average predictions from multiple PPO models."""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, obs, deterministic=True):
        actions = []
        for m in self.models:
            a, _ = m.predict(obs, deterministic=deterministic)
            actions.append(a)
        avg = np.average(actions, axis=0, weights=self.weights)
        return avg, None


def build_test_data(device="cpu"):
    dm = ChronosSolDataModule(
        symbol="SUIUSDT", data_root=Path("trainingdatahourlybinance"),
        forecast_cache_root=Path("binancechronossolexperiment/forecast_cache_sui_stable_best"),
        forecast_horizons=(1, 6, 12), context_hours=256,
        quantile_levels=(0.1, 0.5, 0.9), batch_size=32,
        model_id="amazon/chronos-t5-small", sequence_length=72,
        split_config=SplitConfig(val_days=15, test_days=7), cache_only=True,
    )
    frame = dm.full_frame.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    baseline_ckpt = "binanceleveragesui/checkpoints/lev5x_rw0.012_s1337/policy_checkpoint.pt"
    model, normalizer, feature_columns, _ = load_policy_checkpoint(baseline_ckpt)
    base_actions = generate_actions_from_frame(
        model=model, frame=frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=72, horizon=1,
        require_gpu=(device == "cuda"),
    )
    base_actions["timestamp"] = pd.to_datetime(base_actions["timestamp"], utc=True)
    merged = frame.merge(base_actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_base"))
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    feature_cols = list(dm.feature_columns)
    features_raw = merged[feature_cols].to_numpy(dtype=np.float32)
    val_start_ts = pd.to_datetime(dm.val_window_start, utc=True)
    test_start_ts = pd.to_datetime(dm.test_window_start, utc=True)
    val_idx = int(merged.index[merged["timestamp"] >= val_start_ts][0])
    test_idx = int(merged.index[merged["timestamp"] >= test_start_ts][0])
    mean = features_raw[:val_idx].mean(axis=0)
    std = features_raw[:val_idx].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    features = ((features_raw - mean) / std).astype(np.float32)

    n = len(merged)
    ws = 32

    arrays = {
        "features": features, "highs": merged["high"].to_numpy(dtype=np.float32),
        "lows": merged["low"].to_numpy(dtype=np.float32),
        "closes": merged["close"].to_numpy(dtype=np.float32),
        "base_buy_prices": merged["buy_price"].to_numpy(dtype=np.float32),
        "base_sell_prices": merged["sell_price"].to_numpy(dtype=np.float32),
        "base_buy_amounts": merged["buy_amount"].to_numpy(dtype=np.float32),
        "base_sell_amounts": merged["sell_amount"].to_numpy(dtype=np.float32),
        "timestamps": merged["timestamp"].astype(str).tolist(),
    }

    results = {}
    for window_days in [3, 7, 10, 14, 30]:
        wh = window_days * 24
        end = n
        start = max(ws, end - wh)
        seg_start = max(0, start - ws + 1)
        seg = {k: (v[seg_start:end] if isinstance(v, np.ndarray) else v[seg_start:end])
               for k, v in arrays.items()}
        seg["start_index"] = start - seg_start
        seg["end_index"] = end - seg_start
        results[f"{window_days}d"] = seg
    return results


MODELS_TO_ENSEMBLE = [
    # Best smoothness configs from sweep
    "binanceleveragesui_rlcuda/artifacts_smoothness_smooth_dd01_ps0_cap05_cs1e3/seed_2024/best_model.zip",
    "binanceleveragesui_rlcuda/artifacts_smoothness_smooth_dd1_ps1e3_cap05/seed_2024/best_model.zip",
    "binanceleveragesui_rlcuda/artifacts_smoothness_smooth_dd05_ps5e4_cap03/seed_2024/best_model.zip",
    "binanceleveragesui_rlcuda/artifacts_smoothness_smooth_dd1_ps1e3/seed_1337/best_model.zip",
]


def eval_episode(model, env):
    obs, _ = env.reset(options={"random_start": False})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
    return env.metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("Loading data...")
    windows = build_test_data(device=args.device)

    print("Loading models...")
    models = []
    for p in MODELS_TO_ENSEMBLE:
        if Path(p).exists():
            models.append(PPO.load(p, device=args.device))
            print(f"  loaded {p}")
        else:
            print(f"  SKIP {p} (not found)")

    if len(models) < 2:
        print("Need at least 2 models for ensemble")
        return

    ensemble = _EnsembleModel(models)

    # Also test subsets
    ensembles = {
        f"ensemble_{len(models)}": _EnsembleModel(models),
        "ensemble_top2": _EnsembleModel(models[:2]),
        "ensemble_top3": _EnsembleModel(models[:3]),
        "single_best": _EnsembleModel([models[0]]),
    }

    results = {}
    for ens_name, ens_model in ensembles.items():
        results[ens_name] = {}
        print(f"\n=== {ens_name} ===")
        for wname, wdata in windows.items():
            for lev in [5.0]:
                cfg = ResidualLeverageEnvConfig(
                    window_size=32, max_leverage=lev, maker_fee=0.001,
                    margin_hourly_rate=0.0000025457, initial_cash=10000.0,
                    cap_floor_ratio=0.5, max_cap_change_per_step=0.1,
                    random_start=False, episode_length=None,
                )
                env_rl = ResidualLeverageEnv(
                    features=wdata["features"], highs=wdata["highs"],
                    lows=wdata["lows"], closes=wdata["closes"],
                    base_buy_prices=wdata["base_buy_prices"],
                    base_sell_prices=wdata["base_sell_prices"],
                    base_buy_amounts=wdata["base_buy_amounts"],
                    base_sell_amounts=wdata["base_sell_amounts"],
                    timestamps=wdata["timestamps"], config=cfg,
                    start_index=int(wdata["start_index"]),
                    end_index=int(wdata["end_index"]),
                )
                env_bl = ResidualLeverageEnv(
                    features=wdata["features"], highs=wdata["highs"],
                    lows=wdata["lows"], closes=wdata["closes"],
                    base_buy_prices=wdata["base_buy_prices"],
                    base_sell_prices=wdata["base_sell_prices"],
                    base_buy_amounts=wdata["base_buy_amounts"],
                    base_sell_amounts=wdata["base_sell_amounts"],
                    timestamps=wdata["timestamps"], config=cfg,
                    start_index=int(wdata["start_index"]),
                    end_index=int(wdata["end_index"]),
                )
                rl_m = eval_episode(ens_model, env_rl)
                bl_m = evaluate_baseline_episode(env_bl)["metrics"]
                results[ens_name][wname] = {
                    "rl_return": rl_m["total_return"],
                    "rl_sortino": rl_m["sortino"],
                    "rl_dd": rl_m["max_drawdown"],
                    "bl_return": bl_m["total_return"],
                    "bl_sortino": bl_m["sortino"],
                    "bl_dd": bl_m["max_drawdown"],
                }
                print(f"  {wname} 5x: RL {rl_m['total_return']:.1f}x sort={rl_m['sortino']:.0f} dd={rl_m['max_drawdown']:.1%} | BL {bl_m['total_return']:.1f}x")

    out = Path("experiments/sui_sortino_max/ensemble_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
