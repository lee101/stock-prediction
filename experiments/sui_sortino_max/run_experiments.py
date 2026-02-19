#!/usr/bin/env python3
"""Creative experiments to maximize Sortino ratio for SUI residual controller.

Approaches:
1. Sortino-weighted reward: reward = sortino_component per step
2. Drawdown threshold: only penalize DD > X% (ignore small DD)
3. Asymmetric penalty: losses penalized 3x vs gains rewarded 1x
4. Volatility-gated: scale residual actions inversely to realized vol
5. Rolling Sortino reward: reward based on rolling window Sortino
6. DD-adaptive leverage: automatically reduce cap when in drawdown
7. Ensemble: average actions from top-K models
"""
from __future__ import annotations
import argparse, json, sys, time, copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui_rlcuda.residual_env import (
    ResidualLeverageEnv,
    ResidualLeverageEnvConfig,
    evaluate_baseline_episode,
    evaluate_deterministic_episode,
    HOURLY_PERIODS_PER_YEAR,
)

import gymnasium as gym
from gymnasium import spaces


class SortinoRewardWrapper(gym.Wrapper):
    """Replace step reward with Sortino-focused reward.
    reward = pnl_step - asymmetry * max(0, -pnl_step) - dd_threshold_pen
    """
    def __init__(self, env, asymmetry=2.0, dd_threshold=0.05, dd_pen=5.0,
                 rolling_window=24, sortino_bonus=0.01):
        super().__init__(env)
        self.asymmetry = asymmetry
        self.dd_threshold = dd_threshold
        self.dd_pen = dd_pen
        self.rolling_window = rolling_window
        self.sortino_bonus = sortino_bonus
        self._returns_buffer = []
        self._peak_equity = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._returns_buffer = []
        self._peak_equity = getattr(self.env, '_equity', self.env.cfg.initial_cash)
        return obs, info

    def step(self, action):
        equity_before = getattr(self.env, '_equity', self.env.cfg.initial_cash)
        obs, reward, terminated, truncated, info = self.env.step(action)
        equity_after = getattr(self.env, '_equity', equity_before)

        step_return = (equity_after - equity_before) / max(equity_before, 1e-9)
        self._returns_buffer.append(step_return)

        # Asymmetric reward: losses hurt more
        if step_return >= 0:
            r = step_return
        else:
            r = step_return * (1.0 + self.asymmetry)

        # DD threshold penalty: only penalize DD exceeding threshold
        if self._peak_equity is not None:
            self._peak_equity = max(self._peak_equity, equity_after)
            dd = (equity_after - self._peak_equity) / max(self._peak_equity, 1e-9)
            if dd < -self.dd_threshold:
                r -= self.dd_pen * (abs(dd) - self.dd_threshold)

        # Rolling sortino bonus
        if len(self._returns_buffer) >= self.rolling_window:
            window = np.array(self._returns_buffer[-self.rolling_window:])
            downside = window[window < 0]
            if len(downside) > 0:
                ds_std = np.std(downside)
                if ds_std > 1e-9:
                    rolling_sortino = np.mean(window) / ds_std
                    r += self.sortino_bonus * max(0, rolling_sortino)

        return obs, float(r), terminated, truncated, info


class DDAwareLeverageWrapper(gym.Wrapper):
    """Automatically reduce leverage cap when in drawdown."""
    def __init__(self, env, dd_scale_start=0.02, dd_scale_full=0.10, min_cap=0.2):
        super().__init__(env)
        self.dd_scale_start = dd_scale_start
        self.dd_scale_full = dd_scale_full
        self.min_cap = min_cap
        self._peak_equity = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._peak_equity = getattr(self.env, '_equity', self.env.cfg.initial_cash)
        return obs, info

    def step(self, action):
        equity = getattr(self.env, '_equity', self.env.cfg.initial_cash)
        if self._peak_equity is not None:
            self._peak_equity = max(self._peak_equity, equity)
            dd = (equity - self._peak_equity) / max(self._peak_equity, 1e-9)
            dd_frac = max(0, (abs(dd) - self.dd_scale_start) / max(self.dd_scale_full - self.dd_scale_start, 1e-9))
            dd_frac = min(1.0, dd_frac)
            cap_scale = 1.0 - dd_frac * (1.0 - self.min_cap)
            # Scale down the cap_ratio action
            action = np.array(action, dtype=np.float32)
            action[2] = action[2] * cap_scale
        return self.env.step(action)


class VolGatedWrapper(gym.Wrapper):
    """Scale actions inversely to realized volatility."""
    def __init__(self, env, vol_window=24, vol_target=0.02, vol_clip=3.0):
        super().__init__(env)
        self.vol_window = vol_window
        self.vol_target = vol_target
        self.vol_clip = vol_clip
        self._returns_buffer = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._returns_buffer = []
        return obs, info

    def step(self, action):
        equity_before = getattr(self.env, '_equity', self.env.cfg.initial_cash)
        obs, reward, terminated, truncated, info = self.env.step(action)
        equity_after = getattr(self.env, '_equity', equity_before)
        step_return = (equity_after - equity_before) / max(equity_before, 1e-9)
        self._returns_buffer.append(step_return)
        return obs, reward, terminated, truncated, info

    def _get_vol_scale(self):
        if len(self._returns_buffer) < self.vol_window:
            return 1.0
        window = np.array(self._returns_buffer[-self.vol_window:])
        vol = np.std(window)
        if vol < 1e-9:
            return 1.0
        scale = self.vol_target / vol
        return float(np.clip(scale, 1.0 / self.vol_clip, self.vol_clip))


def build_dataset(device="cpu"):
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

    arrays = {
        "features": features,
        "highs": merged["high"].to_numpy(dtype=np.float32),
        "lows": merged["low"].to_numpy(dtype=np.float32),
        "closes": merged["close"].to_numpy(dtype=np.float32),
        "base_buy_prices": merged["buy_price"].to_numpy(dtype=np.float32),
        "base_sell_prices": merged["sell_price"].to_numpy(dtype=np.float32),
        "base_buy_amounts": merged["buy_amount"].to_numpy(dtype=np.float32),
        "base_sell_amounts": merged["sell_amount"].to_numpy(dtype=np.float32),
        "timestamps": merged["timestamp"].astype(str).tolist(),
    }

    def slice_seg(start, end, ws=32):
        s = max(0, start - ws + 1)
        return {k: (v[s:end] if isinstance(v, np.ndarray) else v[s:end])
                for k, v in arrays.items()}, start - s, end - s

    train_arr, train_si, train_ei = slice_seg(31, val_idx)
    val_arr, val_si, val_ei = slice_seg(val_idx, test_idx)
    test_arr, test_si, test_ei = slice_seg(test_idx, len(merged))

    return {
        "train": {**train_arr, "start_index": train_si, "end_index": train_ei},
        "val": {**val_arr, "start_index": val_si, "end_index": val_ei},
        "test": {**test_arr, "start_index": test_si, "end_index": test_ei},
    }


def make_env(data, cfg, wrapper_cls=None, wrapper_kwargs=None):
    env = ResidualLeverageEnv(
        features=data["features"], highs=data["highs"], lows=data["lows"],
        closes=data["closes"], base_buy_prices=data["base_buy_prices"],
        base_sell_prices=data["base_sell_prices"],
        base_buy_amounts=data["base_buy_amounts"],
        base_sell_amounts=data["base_sell_amounts"],
        timestamps=data["timestamps"], config=cfg,
        start_index=int(data["start_index"]), end_index=int(data["end_index"]),
    )
    if wrapper_cls is not None:
        env = wrapper_cls(env, **(wrapper_kwargs or {}))
    return env


@dataclass
class ExperimentConfig:
    name: str
    cap_floor: float = 0.5
    max_cap_change: Optional[float] = 0.1
    downside_penalty: float = 0.1
    cs_pen: float = 1e-3
    wrapper_cls: Optional[str] = None
    wrapper_kwargs: dict = field(default_factory=dict)
    total_timesteps: int = 60_000
    learning_rate: float = 2e-4
    gamma: float = 0.995
    ent_coef: float = 0.001
    n_steps: int = 1024
    batch_size: int = 512
    seeds: list = field(default_factory=lambda: [1337, 2024])


EXPERIMENTS = [
    # 1. Sortino reward with asymmetric loss (2x penalty on losses)
    ExperimentConfig(
        name="sortino_asym2",
        wrapper_cls="SortinoRewardWrapper",
        wrapper_kwargs={"asymmetry": 2.0, "dd_threshold": 0.05, "dd_pen": 3.0,
                        "rolling_window": 24, "sortino_bonus": 0.02},
    ),
    # 2. Heavy asymmetry (4x loss penalty) + tight DD threshold
    ExperimentConfig(
        name="sortino_asym4_tight",
        wrapper_cls="SortinoRewardWrapper",
        wrapper_kwargs={"asymmetry": 4.0, "dd_threshold": 0.03, "dd_pen": 10.0,
                        "rolling_window": 12, "sortino_bonus": 0.05},
    ),
    # 3. DD-adaptive leverage (auto-deleverage in drawdown)
    ExperimentConfig(
        name="dd_adaptive_lev",
        wrapper_cls="DDAwareLeverageWrapper",
        wrapper_kwargs={"dd_scale_start": 0.02, "dd_scale_full": 0.08, "min_cap": 0.3},
    ),
    # 4. DD-adaptive + sortino reward combo
    ExperimentConfig(
        name="dd_adaptive_sortino",
        wrapper_cls="SortinoRewardWrapper",
        wrapper_kwargs={"asymmetry": 2.0, "dd_threshold": 0.04, "dd_pen": 5.0,
                        "sortino_bonus": 0.03},
    ),
    # 5. Higher cap floor (0.7) + sortino reward
    ExperimentConfig(
        name="high_cap_sortino",
        cap_floor=0.7,
        wrapper_cls="SortinoRewardWrapper",
        wrapper_kwargs={"asymmetry": 1.5, "dd_threshold": 0.06, "dd_pen": 2.0,
                        "sortino_bonus": 0.01},
    ),
    # 6. Conservative (low cap) + strong sortino
    ExperimentConfig(
        name="conservative_sortino",
        cap_floor=0.3,
        max_cap_change=0.05,
        wrapper_cls="SortinoRewardWrapper",
        wrapper_kwargs={"asymmetry": 3.0, "dd_threshold": 0.03, "dd_pen": 8.0,
                        "sortino_bonus": 0.05},
    ),
    # 7. Higher gamma (more forward-looking, values long-term sortino)
    ExperimentConfig(
        name="high_gamma_sortino",
        gamma=0.999,
        wrapper_cls="SortinoRewardWrapper",
        wrapper_kwargs={"asymmetry": 2.0, "dd_threshold": 0.05, "dd_pen": 3.0,
                        "sortino_bonus": 0.02},
    ),
    # 8. Larger network + more steps
    ExperimentConfig(
        name="big_model_sortino",
        total_timesteps=120_000,
        wrapper_cls="SortinoRewardWrapper",
        wrapper_kwargs={"asymmetry": 2.0, "dd_threshold": 0.05, "dd_pen": 3.0,
                        "sortino_bonus": 0.02},
    ),
    # 9. Very low ent_coef (more deterministic policy)
    ExperimentConfig(
        name="low_entropy_sortino",
        ent_coef=0.0001,
        wrapper_cls="SortinoRewardWrapper",
        wrapper_kwargs={"asymmetry": 2.0, "dd_threshold": 0.04, "dd_pen": 5.0,
                        "sortino_bonus": 0.03},
    ),
    # 10. Pure sortino: no cap_change limit, reward is pure rolling sortino
    ExperimentConfig(
        name="pure_sortino",
        max_cap_change=None,
        cap_floor=0.4,
        wrapper_cls="SortinoRewardWrapper",
        wrapper_kwargs={"asymmetry": 0.0, "dd_threshold": 1.0, "dd_pen": 0.0,
                        "rolling_window": 48, "sortino_bonus": 0.1},
    ),
]

WRAPPER_MAP = {
    "SortinoRewardWrapper": SortinoRewardWrapper,
    "DDAwareLeverageWrapper": DDAwareLeverageWrapper,
    "VolGatedWrapper": VolGatedWrapper,
}


def run_experiment(exp: ExperimentConfig, dataset: dict, device: str,
                   artifacts_root: Path) -> dict:
    results_per_seed = []

    for seed in exp.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        run_dir = artifacts_root / exp.name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        env_cfg_train = ResidualLeverageEnvConfig(
            window_size=32, max_leverage=5.0, maker_fee=0.001,
            margin_hourly_rate=0.0000025457, initial_cash=10000.0,
            cap_floor_ratio=exp.cap_floor,
            max_cap_change_per_step=exp.max_cap_change,
            downside_penalty=exp.downside_penalty,
            leverage_cap_smoothness_penalty=exp.cs_pen,
            random_start=True, episode_length=240,
        )
        env_cfg_eval = ResidualLeverageEnvConfig(
            window_size=32, max_leverage=5.0, maker_fee=0.001,
            margin_hourly_rate=0.0000025457, initial_cash=10000.0,
            cap_floor_ratio=exp.cap_floor,
            max_cap_change_per_step=exp.max_cap_change,
            random_start=False, episode_length=None,
        )

        wcls = WRAPPER_MAP.get(exp.wrapper_cls) if exp.wrapper_cls else None
        wkw = exp.wrapper_kwargs

        train_envs = DummyVecEnv([
            (lambda d=dataset["train"], c=env_cfg_train, w=wcls, wk=wkw:
             Monitor(make_env(d, c, w, wk)))
            for _ in range(8)
        ])
        val_envs = DummyVecEnv([
            (lambda d=dataset["val"], c=env_cfg_eval:
             Monitor(make_env(d, c)))
        ])

        policy_kwargs = {
            "activation_fn": torch.nn.SiLU,
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        }

        model = PPO(
            "MlpPolicy", train_envs,
            learning_rate=exp.learning_rate, n_steps=exp.n_steps,
            batch_size=exp.batch_size, gamma=exp.gamma,
            gae_lambda=0.95, clip_range=0.2, ent_coef=exp.ent_coef,
            vf_coef=0.5, max_grad_norm=0.5, seed=seed,
            device=device, policy_kwargs=policy_kwargs, verbose=0,
        )

        eval_cb = EvalCallback(
            val_envs, best_model_save_path=str(run_dir),
            log_path=str(run_dir), eval_freq=8000,
            n_eval_episodes=1, deterministic=True, verbose=0,
        )

        t0 = time.time()
        model.learn(total_timesteps=exp.total_timesteps, callback=eval_cb, progress_bar=False)
        train_time = time.time() - t0

        best_path = run_dir / "best_model.zip"
        if best_path.exists():
            model = PPO.load(str(best_path), device=device)

        # Eval on test set at multiple leverage levels
        metrics = {}
        for lev in [3.0, 5.0]:
            eval_cfg = ResidualLeverageEnvConfig(
                window_size=32, max_leverage=lev, maker_fee=0.001,
                margin_hourly_rate=0.0000025457, initial_cash=10000.0,
                cap_floor_ratio=exp.cap_floor,
                max_cap_change_per_step=exp.max_cap_change,
                random_start=False, episode_length=None,
            )
            test_env_rl = make_env(dataset["test"], eval_cfg)
            test_env_bl = make_env(dataset["test"], eval_cfg)
            rl_res = evaluate_deterministic_episode(model, test_env_rl)
            bl_res = evaluate_baseline_episode(test_env_bl)
            metrics[f"lev_{lev:.0f}x"] = {
                "rl": rl_res["metrics"], "bl": bl_res["metrics"],
            }

        rl5 = metrics["lev_5x"]["rl"]
        bl5 = metrics["lev_5x"]["bl"]
        score = rl5["sortino"] - 0.5 * abs(rl5["max_drawdown"]) * 1000

        results_per_seed.append({
            "seed": seed, "score": score, "train_time": train_time,
            "metrics": metrics,
        })

    best = max(results_per_seed, key=lambda r: r["score"])
    rl5 = best["metrics"]["lev_5x"]["rl"]
    bl5 = best["metrics"]["lev_5x"]["bl"]

    return {
        "name": exp.name,
        "best_seed": best["seed"],
        "rl_5x_return": rl5["total_return"],
        "rl_5x_sortino": rl5["sortino"],
        "rl_5x_dd": rl5["max_drawdown"],
        "bl_5x_return": bl5["total_return"],
        "bl_5x_dd": bl5["max_drawdown"],
        "rl_3x_return": best["metrics"]["lev_3x"]["rl"]["total_return"],
        "rl_3x_sortino": best["metrics"]["lev_3x"]["rl"]["sortino"],
        "rl_3x_dd": best["metrics"]["lev_3x"]["rl"]["max_drawdown"],
        "all_seeds": results_per_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiments", default="all")
    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("Loading dataset...")
    dataset = build_dataset(device=args.device)
    print("Dataset loaded.")

    artifacts = Path("experiments/sui_sortino_max/artifacts")
    experiments = EXPERIMENTS
    if args.experiments != "all":
        indices = [int(i) for i in args.experiments.split(",")]
        experiments = [EXPERIMENTS[i] for i in indices]

    results = {}
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp.name}")
        print(f"  wrapper={exp.wrapper_cls} cap_floor={exp.cap_floor} gamma={exp.gamma}")
        print(f"{'='*60}")
        try:
            r = run_experiment(exp, dataset, args.device, artifacts)
            results[exp.name] = r
            print(f"  BEST seed={r['best_seed']}: ret={r['rl_5x_return']:.2f}x sort={r['rl_5x_sortino']:.0f} dd={r['rl_5x_dd']:.1%}")
            print(f"  3x: ret={r['rl_3x_return']:.2f}x sort={r['rl_3x_sortino']:.0f} dd={r['rl_3x_dd']:.1%}")
            print(f"  BL: ret={r['bl_5x_return']:.2f}x dd={r['bl_5x_dd']:.1%}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[exp.name] = {"error": str(e)}

    out = Path("experiments/sui_sortino_max/results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    print("\n" + "="*80)
    print("SUMMARY (sorted by sortino)")
    print("="*80)
    valid = [(n, r) for n, r in results.items() if "error" not in r]
    valid.sort(key=lambda x: x[1]["rl_5x_sortino"], reverse=True)
    print(f"{'Name':<30} {'Return':>8} {'Sortino':>8} {'DD':>8} {'3x Ret':>8} {'3x Sort':>8}")
    for name, r in valid:
        print(f"{name:<30} {r['rl_5x_return']:>7.1f}x {r['rl_5x_sortino']:>8.0f} {r['rl_5x_dd']:>7.1%} {r['rl_3x_return']:>7.1f}x {r['rl_3x_sortino']:>8.0f}")
    print(f"\nBaseline 5x: ret={valid[0][1]['bl_5x_return']:.1f}x dd={valid[0][1]['bl_5x_dd']:.1%}" if valid else "")


if __name__ == "__main__":
    main()
