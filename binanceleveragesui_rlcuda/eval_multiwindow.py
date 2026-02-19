#!/usr/bin/env python3
"""Multi-window eval for smoothness-optimized residual models."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui_rlcuda.residual_env import (
    ResidualLeverageEnv,
    ResidualLeverageEnvConfig,
    evaluate_baseline_episode,
    evaluate_deterministic_episode,
)


def _build_full_data(
    symbol="SUIUSDT",
    data_root="trainingdatahourlybinance",
    forecast_cache="binancechronossolexperiment/forecast_cache_sui_stable_best",
    horizons=(1, 6, 12),
    context_hours=256,
    sequence_length=72,
    baseline_ckpt="binanceleveragesui/checkpoints/lev5x_rw0.012_s1337/policy_checkpoint.pt",
    device="cpu",
):
    dm = ChronosSolDataModule(
        symbol=symbol, data_root=Path(data_root),
        forecast_cache_root=Path(forecast_cache),
        forecast_horizons=horizons, context_hours=context_hours,
        quantile_levels=(0.1, 0.5, 0.9), batch_size=32,
        model_id="amazon/chronos-t5-small", sequence_length=sequence_length,
        split_config=SplitConfig(val_days=15, test_days=7), cache_only=True,
    )
    frame = dm.full_frame.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    model, normalizer, feature_columns, _ = load_policy_checkpoint(baseline_ckpt)
    base_actions = generate_actions_from_frame(
        model=model, frame=frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=sequence_length,
        horizon=horizons[0], require_gpu=(device == "cuda"),
    )
    base_actions["timestamp"] = pd.to_datetime(base_actions["timestamp"], utc=True)
    merged = frame.merge(base_actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_base"))
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    feature_cols = list(dm.feature_columns)
    features_raw = merged[feature_cols].to_numpy(dtype=np.float32)
    train_end = len(merged) - 7 * 24 - 15 * 24
    mean = features_raw[:train_end].mean(axis=0)
    std = features_raw[:train_end].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    features = ((features_raw - mean) / std).astype(np.float32)

    return {
        "features": features,
        "highs": merged["high"].to_numpy(dtype=np.float32),
        "lows": merged["low"].to_numpy(dtype=np.float32),
        "closes": merged["close"].to_numpy(dtype=np.float32),
        "base_buy_prices": merged["buy_price"].to_numpy(dtype=np.float32),
        "base_sell_prices": merged["sell_price"].to_numpy(dtype=np.float32),
        "base_buy_amounts": merged["buy_amount"].to_numpy(dtype=np.float32),
        "base_sell_amounts": merged["sell_amount"].to_numpy(dtype=np.float32),
        "timestamps": merged["timestamp"].astype(str).tolist(),
        "total_rows": len(merged),
    }


def eval_window(data, model_path, window_hours, leverage=5.0, window_size=32,
                maker_fee=0.001, margin_rate=0.0000025457, initial_cash=10000.0,
                cap_floor=0.0, max_cap_change=None, device="cpu"):
    n = data["total_rows"]
    end_idx = n
    start_idx = max(window_size, end_idx - window_hours)

    seg_start = max(0, start_idx - window_size + 1)
    rel_start = start_idx - seg_start
    rel_end = end_idx - seg_start

    seg = {}
    for k in ["features", "highs", "lows", "closes", "base_buy_prices",
              "base_sell_prices", "base_buy_amounts", "base_sell_amounts"]:
        seg[k] = data[k][seg_start:end_idx]
    seg["timestamps"] = data["timestamps"][seg_start:end_idx]

    env_cfg = ResidualLeverageEnvConfig(
        window_size=window_size, max_leverage=leverage, maker_fee=maker_fee,
        margin_hourly_rate=margin_rate, initial_cash=initial_cash,
        cap_floor_ratio=cap_floor, max_cap_change_per_step=max_cap_change,
        random_start=False, episode_length=None,
    )

    def make_env():
        return ResidualLeverageEnv(
            features=seg["features"], highs=seg["highs"], lows=seg["lows"],
            closes=seg["closes"], base_buy_prices=seg["base_buy_prices"],
            base_sell_prices=seg["base_sell_prices"],
            base_buy_amounts=seg["base_buy_amounts"],
            base_sell_amounts=seg["base_sell_amounts"],
            timestamps=seg["timestamps"], config=env_cfg,
            start_index=rel_start, end_index=rel_end,
        )

    ppo = PPO.load(model_path, device=device)
    rl_env = make_env()
    bl_env = make_env()
    rl = evaluate_deterministic_episode(ppo, rl_env)
    bl = evaluate_baseline_episode(bl_env)
    return rl["metrics"], bl["metrics"]


MODELS = {
    "dd01_ps0_cap05_cs1e3": {
        "path": "binanceleveragesui_rlcuda/artifacts_smoothness_smooth_dd01_ps0_cap05_cs1e3/seed_2024/best_model.zip",
        "cap_floor": 0.5, "max_cap_change": 0.1,
    },
    "dd1_ps1e3_cap05": {
        "path": "binanceleveragesui_rlcuda/artifacts_smoothness_smooth_dd1_ps1e3_cap05/seed_2024/best_model.zip",
        "cap_floor": 0.5, "max_cap_change": 0.05,
    },
    "dd05_ps5e4_cap03": {
        "path": "binanceleveragesui_rlcuda/artifacts_smoothness_smooth_dd05_ps5e4_cap03/seed_2024/best_model.zip",
        "cap_floor": 0.3, "max_cap_change": 0.1,
    },
    "dd1_ps1e3": {
        "path": "binanceleveragesui_rlcuda/artifacts_smoothness_smooth_dd1_ps1e3/seed_1337/best_model.zip",
        "cap_floor": 0.0, "max_cap_change": None,
    },
}

WINDOWS = [3 * 24, 7 * 24, 10 * 24, 14 * 24, 30 * 24]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", default="all")
    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("Loading data...")
    data = _build_full_data(device=args.device)
    print(f"  {data['total_rows']} rows loaded")

    models = MODELS if args.models == "all" else {k: MODELS[k] for k in args.models.split(",")}
    results = {}

    for name, mcfg in models.items():
        print(f"\n=== {name} ===")
        results[name] = {}
        for wh in WINDOWS:
            wd = wh // 24
            rl_m, bl_m = eval_window(
                data, mcfg["path"], wh, device=args.device,
                cap_floor=mcfg["cap_floor"], max_cap_change=mcfg["max_cap_change"],
            )
            results[name][f"{wd}d"] = {
                "rl_return": rl_m["total_return"],
                "rl_sortino": rl_m["sortino"],
                "rl_dd": rl_m["max_drawdown"],
                "bl_return": bl_m["total_return"],
                "bl_sortino": bl_m["sortino"],
                "bl_dd": bl_m["max_drawdown"],
            }
            print(f"  {wd}d: RL {rl_m['total_return']:.1f}x sort={rl_m['sortino']:.0f} dd={rl_m['max_drawdown']:.1%} | BL {bl_m['total_return']:.1f}x dd={bl_m['max_drawdown']:.1%}")

    out = Path("binanceleveragesui_rlcuda/multiwindow_smoothness_eval.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
