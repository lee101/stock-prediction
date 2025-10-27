from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .config import ExperimentConfig
from .envs.dm_env import KronosDMEnv
from .kronos_embedder import KronosEmbedder, KronosFeatureSpec, precompute_feature_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
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
    env = KronosDMEnv(
        prices=price_series,
        features=features_df,
        transaction_cost_bps=cfg.env.transaction_cost_bps,
        slippage_bps=cfg.env.slippage_bps,
        max_position=cfg.env.max_position,
        hold_penalty=cfg.env.hold_penalty,
        reward=cfg.env.reward,
    )

    model = PPO.load(os.path.join(args.model_path))

    obs, _ = env.reset()
    rewards = []
    nav = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        nav.append(info["nav"])
        done = terminated or truncated

    rewards = np.array(rewards)
    nav = np.array(nav)
    sharpe = rewards.mean() / (rewards.std(ddof=1) + 1e-8)
    returns = nav[-1] - 1.0
    print(f"total_return={returns:.4f} sharpe={sharpe:.4f}")


if __name__ == "__main__":
    main()
