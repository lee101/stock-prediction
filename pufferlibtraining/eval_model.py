#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gymnasium.vector import SyncVectorEnv

from pufferlibtraining.envs.stock_env import StockTradingEnv


def load_data(symbol: str, data_dir: str = "data") -> pd.DataFrame:
    data_path = Path(data_dir)
    csvs = sorted(data_path.glob(f"*{symbol}*.csv")) or sorted(data_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    df = pd.read_csv(csvs[0])
    cols = [c.lower() for c in df.columns]
    df.columns = cols
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
    # engineered features
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
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on held-out test data")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--output-dir", type=str, default="pufferlibtraining/output")
    args = parser.parse_args()

    df = load_data(args.symbol, args.data_dir)
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].reset_index(drop=True)

    env = SyncVectorEnv([
        lambda: make_env(test_df, args.window_size, args.initial_balance, args.transaction_cost)
    ])

    from stable_baselines3 import PPO
    model = PPO.load(args.model_path, device="cpu")

    obs, _ = env.reset()
    dones = np.array([False])
    balances = []
    positions = []
    rews = []
    infos = []
    while not dones[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        dones = np.logical_or(term, trunc)
        rews.append(reward[0])
        balances.append(info[0].get("balance", np.nan))
        positions.append(info[0].get("position", np.nan))
        infos.append(info[0])

    # Metrics
    final_balance = balances[-1] if balances else args.initial_balance
    total_return = (final_balance - args.initial_balance) / args.initial_balance
    returns = np.array([i.get("daily_return", 0.0) for i in infos if "daily_return" in i])
    sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252) if len(returns) else 0.0
    cumulative = np.cumprod(1 + returns) if len(returns) else np.array([1.0])
    running_max = np.maximum.accumulate(cumulative)
    dd = (cumulative - running_max) / np.maximum(running_max, 1e-8)
    max_drawdown = dd.min() if len(dd) else 0.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot equity curve
    if balances:
        import json
        plt.figure(figsize=(12, 6))
        plt.plot(balances)
        plt.title(f"Equity Curve ({args.symbol})")
        plt.xlabel("Steps")
        plt.ylabel("Balance ($)")
        plt.grid(True)
        png_path = out_dir / f"equity_{args.symbol.lower()}.png"
        plt.savefig(png_path)
        plt.close()

        metrics = {
            "symbol": args.symbol,
            "final_balance": float(final_balance),
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "num_steps": len(balances),
            "num_trades": int(infos[-1].get("trades", 0)) if infos else 0,
        }
        with open(out_dir / f"metrics_{args.symbol.lower()}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("Evaluation complete. Metrics:", metrics)
        print("Saved:", png_path)
    else:
        print("Evaluation finished with no steps — check data or window size.")


if __name__ == "__main__":
    main()

