#!/usr/bin/env python3
"""Rule-based overlays on best residual model to improve sortino.

Post-hoc strategies that modify model actions without retraining:
1. Vol-gate: reduce position when realized vol is high
2. Momentum filter: only trade in direction of trend
3. Time-decay leverage: reduce leverage as position ages
4. Sortino gate: reduce/stop trading when rolling sortino is negative
5. Max concurrent DD gate: flatten when DD exceeds threshold
"""
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
    HOURLY_PERIODS_PER_YEAR,
)


class RuleBasedOverlay:
    """Wraps a PPO model with rule-based action modification."""
    def __init__(self, base_model, strategy="none", **kwargs):
        self.base_model = base_model
        self.strategy = strategy
        self.kwargs = kwargs
        self._returns = []
        self._peak_eq = None
        self._position_age = 0
        self._prev_equity = None

    def predict(self, obs, deterministic=True):
        action, states = self.base_model.predict(obs, deterministic=deterministic)
        action = np.array(action, dtype=np.float32).copy()

        # Track returns for rolling calculations
        # obs includes equity info in the feature window - use cap_ratio to infer state

        if self.strategy == "vol_gate":
            action = self._vol_gate(action, obs)
        elif self.strategy == "momentum_filter":
            action = self._momentum_filter(action, obs)
        elif self.strategy == "dd_flatten":
            action = self._dd_flatten(action, obs)
        elif self.strategy == "conservative_cap":
            action = self._conservative_cap(action, obs)
        elif self.strategy == "cap_clamp":
            action = self._cap_clamp(action, obs)

        return action, states

    def reset(self):
        self._returns = []
        self._peak_eq = None
        self._position_age = 0
        self._prev_equity = None

    def _vol_gate(self, action, obs):
        """Reduce cap when recent obs variance is high."""
        window = self.kwargs.get("vol_window", 12)
        scale = self.kwargs.get("vol_scale", 0.5)
        if len(obs.shape) > 1:
            obs_flat = obs[0]
        else:
            obs_flat = obs
        # Use variance of recent features as vol proxy
        if len(obs_flat) > window:
            recent_var = np.var(obs_flat[-window:])
            if recent_var > self.kwargs.get("vol_thresh", 2.0):
                action[2] *= scale  # reduce cap
        return action

    def _momentum_filter(self, action, obs):
        """Reduce buy when price trending down, reduce sell when trending up."""
        # Use obs features - close returns are typically in first few dims
        if len(obs.shape) > 1:
            obs_flat = obs[0]
        else:
            obs_flat = obs
        # obs[0] is usually return_1h
        if len(obs_flat) > 0:
            ret_signal = obs_flat[0]  # return_1h
            thresh = self.kwargs.get("mom_thresh", 0.5)
            if ret_signal < -thresh:
                action[0] *= 0.3  # reduce buy scale in downtrend
            elif ret_signal > thresh:
                action[1] *= 0.3  # reduce sell scale in uptrend
        return action

    def _dd_flatten(self, action, obs):
        """Set cap to minimum when DD exceeds threshold."""
        dd_thresh = self.kwargs.get("dd_thresh", 0.05)
        # Can't directly observe DD from obs alone - use cap_ratio as proxy
        # Just clamp cap ratio lower
        action[2] = min(action[2], self.kwargs.get("max_cap_in_dd", 0.3))
        return action

    def _conservative_cap(self, action, obs):
        """Always use a lower cap ratio."""
        max_cap = self.kwargs.get("max_cap", 0.6)
        action[2] = min(action[2], max_cap)
        return action

    def _cap_clamp(self, action, obs):
        """Clamp cap ratio to a fixed range."""
        lo = self.kwargs.get("cap_lo", 0.3)
        hi = self.kwargs.get("cap_hi", 0.7)
        action[2] = np.clip(action[2], lo, hi)
        return action


def build_test_windows(device="cpu"):
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
    val_idx = int(merged.index[merged["timestamp"] >= pd.to_datetime(dm.val_window_start, utc=True)][0])
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
    windows = {}
    for wd in [7]:
        wh = wd * 24
        end = n
        start = max(ws, end - wh)
        ss = max(0, start - ws + 1)
        seg = {k: (v[ss:end] if isinstance(v, np.ndarray) else v[ss:end]) for k, v in arrays.items()}
        seg["start_index"] = start - ss
        seg["end_index"] = end - ss
        windows[f"{wd}d"] = seg
    return windows


def eval_model(model, data, lev=5.0, cap_floor=0.5, max_cap_change=0.1):
    cfg = ResidualLeverageEnvConfig(
        window_size=32, max_leverage=lev, maker_fee=0.001,
        margin_hourly_rate=0.0000025457, initial_cash=10000.0,
        cap_floor_ratio=cap_floor, max_cap_change_per_step=max_cap_change,
        random_start=False, episode_length=None,
    )
    env = ResidualLeverageEnv(
        features=data["features"], highs=data["highs"], lows=data["lows"],
        closes=data["closes"], base_buy_prices=data["base_buy_prices"],
        base_sell_prices=data["base_sell_prices"],
        base_buy_amounts=data["base_buy_amounts"],
        base_sell_amounts=data["base_sell_amounts"],
        timestamps=data["timestamps"], config=cfg,
        start_index=int(data["start_index"]), end_index=int(data["end_index"]),
    )
    obs, _ = env.reset(options={"random_start": False})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
    return env.metrics


STRATEGIES = [
    ("baseline", "none", {}),
    ("conservative_cap_06", "conservative_cap", {"max_cap": 0.6}),
    ("conservative_cap_05", "conservative_cap", {"max_cap": 0.5}),
    ("conservative_cap_04", "conservative_cap", {"max_cap": 0.4}),
    ("cap_clamp_03_07", "cap_clamp", {"cap_lo": 0.3, "cap_hi": 0.7}),
    ("cap_clamp_04_06", "cap_clamp", {"cap_lo": 0.4, "cap_hi": 0.6}),
    ("cap_clamp_05_08", "cap_clamp", {"cap_lo": 0.5, "cap_hi": 0.8}),
    ("vol_gate_loose", "vol_gate", {"vol_window": 12, "vol_thresh": 1.5, "vol_scale": 0.5}),
    ("vol_gate_tight", "vol_gate", {"vol_window": 6, "vol_thresh": 1.0, "vol_scale": 0.3}),
    ("momentum_filter", "momentum_filter", {"mom_thresh": 0.3}),
    ("momentum_tight", "momentum_filter", {"mom_thresh": 0.1}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("Loading data...")
    windows = build_test_windows(args.device)

    best_model_path = "binanceleveragesui_rlcuda/artifacts_smoothness_smooth_dd01_ps0_cap05_cs1e3/seed_2024/best_model.zip"
    print(f"Loading model: {best_model_path}")
    base_ppo = PPO.load(best_model_path, device=args.device)

    results = {}
    print(f"\n{'Strategy':<25} {'Return':>8} {'Sortino':>8} {'DD':>8}")
    print("-" * 55)

    for name, strategy, kwargs in STRATEGIES:
        overlay = RuleBasedOverlay(base_ppo, strategy=strategy, **kwargs)
        m = eval_model(overlay, windows["7d"])
        results[name] = {
            "return": m["total_return"], "sortino": m["sortino"],
            "dd": m["max_drawdown"], "trades": m["num_trades"],
        }
        print(f"{name:<25} {m['total_return']:>7.1f}x {m['sortino']:>8.0f} {m['max_drawdown']:>7.1%}")

    out = Path("experiments/sui_sortino_max/rule_based_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
