from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.torch_backend import configure_tf32_backends, maybe_set_float32_precision

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
configure_tf32_backends(torch)
if torch.cuda.is_available():
    maybe_set_float32_precision(torch)

from .config import ExperimentConfig
from .envs.dm_env import KronosDMEnv
from .kronos_embedder import KronosEmbedder, KronosFeatureSpec, precompute_feature_table


def make_env(prices: pd.Series, features: pd.DataFrame, env_cfg):
    def _thunk():
        return KronosDMEnv(
            prices=prices,
            features=features,
            returns_window=0,
            transaction_cost_bps=env_cfg.transaction_cost_bps,
            slippage_bps=env_cfg.slippage_bps,
            max_position=env_cfg.max_position,
            hold_penalty=env_cfg.hold_penalty,
            reward=env_cfg.reward,
        )

    return _thunk


class SaveBestCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean")
            if reward is not None and reward > self.best_mean_reward:
                self.best_mean_reward = float(reward)
                path = os.path.join(self.save_path, "best_model.zip")
                self.model.save(path)
                if self.verbose:
                    print(f"[save] New best reward {self.best_mean_reward:.6f} -> {path}")
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", type=str, required=True, help="Path to OHLCV CSV/Parquet")
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument("--save-dir", type=str, default="runs/differentiable_market_kronos")
    parser.add_argument("--use-subproc", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig()

    path = Path(args.ohlcv)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df[cfg.data.timestamp_col] = pd.to_datetime(df[cfg.data.timestamp_col])
    df = df.dropna().sort_values(cfg.data.timestamp_col).reset_index(drop=True)

    embedder = KronosEmbedder(
        model_id=cfg.kronos.model_id,
        tokenizer_id=cfg.kronos.tokenizer_id,
        device=cfg.kronos.device,
        max_context=cfg.kronos.max_context,
        temperature=cfg.kronos.temperature,
        top_p=cfg.kronos.top_p,
        sample_count=cfg.kronos.sample_count,
        bf16=cfg.train.bf16,
        feature_spec=KronosFeatureSpec(horizons=(1, 12, cfg.env.pred_horizon)),
    )

    cols = [cfg.data.open_col, cfg.data.high_col, cfg.data.low_col, cfg.data.price_col]
    if cfg.data.volume_col in df.columns:
        cols.append(cfg.data.volume_col)
    if cfg.data.amount_col in df.columns:
        cols.append(cfg.data.amount_col)
    x_df = df[cols].rename(
        columns={
            cfg.data.open_col: "open",
            cfg.data.high_col: "high",
            cfg.data.low_col: "low",
            cfg.data.price_col: "close",
            cfg.data.volume_col: "volume" if cfg.data.volume_col in df.columns else cfg.data.volume_col,
            cfg.data.amount_col: "amount" if cfg.data.amount_col in df.columns else cfg.data.amount_col,
        }
    )
    ts = df[cfg.data.timestamp_col]

    features_df = precompute_feature_table(
        df=x_df,
        ts=ts,
        lookback=cfg.env.lookback,
        horizon_main=cfg.env.pred_horizon,
        embedder=embedder,
    ).astype("float32")

    price_series = df.set_index(cfg.data.timestamp_col)[cfg.data.price_col].loc[features_df.index]
    split_idx = int(len(features_df) * 0.8)
    tr_features = features_df.iloc[:split_idx]
    tr_price = price_series.iloc[:split_idx]

    env_fns = [make_env(tr_price, tr_features, cfg.env) for _ in range(max(cfg.train.n_envs, 1))]
    VecCls = SubprocVecEnv if (args.use_subproc and cfg.train.n_envs > 1) else DummyVecEnv
    vec_env = VecCls(env_fns)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = configure(folder=args.save_dir, format_strings=["stdout", "csv", "tensorboard"])

    policy_kwargs = dict(net_arch=[256, 256], ortho_init=False)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        batch_size=cfg.train.batch_size,
        n_steps=cfg.train.rollout_steps,
        learning_rate=cfg.train.learning_rate,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
        clip_range=cfg.train.clip_range,
        ent_coef=cfg.train.ent_coef,
        vf_coef=cfg.train.vf_coef,
        max_grad_norm=cfg.train.max_grad_norm,
        policy_kwargs=policy_kwargs,
        device=cfg.kronos.device,
    )
    model.set_logger(logger)

    callback = SaveBestCallback(
        save_freq=max(1, cfg.train.save_freq_steps // max(1, cfg.train.rollout_steps)),
        save_path=args.save_dir,
    )
    model.learn(total_timesteps=cfg.train.total_timesteps, callback=callback)
    model.save(os.path.join(args.save_dir, "final_model.zip"))
    print("[done] training complete")


if __name__ == "__main__":
    main()
