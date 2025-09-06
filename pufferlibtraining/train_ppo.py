#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from typing import List

from pufferlibtraining.envs.stock_env import StockTradingEnv


def load_data(symbol: str, data_dir: str = "data") -> pd.DataFrame:
    data_path = Path(data_dir)
    csvs = sorted(data_path.glob(f"*{symbol}*.csv")) or sorted(data_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    df = pd.read_csv(csvs[0])
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    req = ["open", "high", "low", "close", "volume"]
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]
    if "open" not in df.columns and "adj open" in df.columns:
        df["open"] = df["adj open"]
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            df[c] = df.get("close", df.iloc[:, 0])
    if "volume" not in df.columns:
        df["volume"] = 1_000_000
    df.columns = [c.title() for c in df.columns]
    # simple features
    df["Returns"] = df["Close"].pct_change()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["Rsi"] = 100 - (100 / (1 + rs))
    df["Volume_Ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean())
    df["High_Low_Ratio"] = df["High"] / df["Low"].replace(0, np.nan)
    df["Close_Open_Ratio"] = df["Close"] / df["Open"].replace(0, np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def make_env(df: pd.DataFrame, window_size: int, initial_balance: float, transaction_cost: float) -> StockTradingEnv:
    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Returns",
        "Rsi",
        "Volume_Ratio",
        "High_Low_Ratio",
        "Close_Open_Ratio",
    ]
    features = [f for f in features if f in df.columns]
    return StockTradingEnv(
        df=df,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        features=features,
    )


def main():
    parser = argparse.ArgumentParser(description="PufferLib-based RL training for stocks")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-dir", type=str, default="pufferlibtraining/models")
    parser.add_argument("--log-dir", type=str, default="pufferlibtraining/logs")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    # PPO hyperparameters
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-range", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load_data(args.symbol, args.data_dir)

    # Train/Test split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    # Vectorized envs: use pufferlib if present, otherwise gymnasium.vector
    try:
        import pufferlib  # noqa: F401
        from gymnasium.vector import SyncVectorEnv
        def thunk():
            return make_env(train_df, args.window_size, args.initial_balance, args.transaction_cost)
        env = SyncVectorEnv([thunk for _ in range(args.n_envs)])
    except Exception:
        from gymnasium.vector import SyncVectorEnv
        def thunk():
            return make_env(train_df, args.window_size, args.initial_balance, args.transaction_cost)
        env = SyncVectorEnv([thunk for _ in range(args.n_envs)])

    # Use Stable-Baselines3 PPO with PyTorch (GPU if available)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.torch_layers import FlattenExtractor
        from stable_baselines3.common.vec_env import VecMonitor
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError as e:
        raise SystemExit("stable-baselines3 is required. Please pip install stable-baselines3")

    env = VecMonitor(env)

    policy_kwargs = dict(features_extractor_class=FlattenExtractor)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu",
        tensorboard_log=args.log_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        clip_range=args.clip_range,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation env on test set
    from gymnasium.vector import SyncVectorEnv as _Sync
    eval_env = _Sync([lambda: make_env(test_df, args.window_size, args.initial_balance, args.transaction_cost)])
    eval_cb = EvalCallback(eval_env, best_model_save_path=str(save_dir), log_path=args.log_dir, eval_freq=10_000, deterministic=True)

    model.learn(total_timesteps=args.total_timesteps, callback=eval_cb)
    model.save(str(save_dir / f"ppo_{args.symbol.lower()}"))

    print("Training complete. Saved to:", save_dir)


if __name__ == "__main__":
    main()
