#!/usr/bin/env python3
"""Backtest multi-asset RL policy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig
from unified_hourly_experiment.multiasset_policy import (
    MultiAssetConfig,
    MultiAssetPolicy,
    DifferentiablePortfolioSim,
)


def load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = MultiAssetConfig(
        num_assets=ckpt["config"]["num_assets"],
        feature_dim=ckpt["config"]["feature_dim"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        num_heads=ckpt["config"]["num_heads"],
        num_layers=ckpt["config"]["num_layers"],
        max_len=ckpt["config"]["max_len"],
    )
    model = MultiAssetPolicy(config)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["symbols"], ckpt["feature_columns"]


def load_data(symbols, feature_columns, data_root, cache_root, seq_len=32):
    frames = {}
    for symbol in symbols:
        data_config = DatasetConfig(
            symbol=symbol,
            data_root=str(data_root),
            forecast_cache_root=str(cache_root),
            forecast_horizons=[1, 24],
            sequence_length=seq_len,
            min_history_hours=100,
            validation_days=30,
            cache_only=True,
        )
        try:
            dm = BinanceHourlyDataModule(data_config)
            frames[symbol] = dm.frame
        except Exception as e:
            logger.warning("Failed to load {}: {}", symbol, e)
    return frames, feature_columns


def backtest(model, frames, feature_columns, seq_len=32, horizon=24, tx_cost=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    symbols = list(frames.keys())
    num_assets = len(symbols)

    # Find common timestamps
    timestamp_sets = [set(f.index.tolist()) for f in frames.values()]
    common_timestamps = sorted(set.intersection(*timestamp_sets))

    # Use last 30 days for backtesting (validation period)
    test_len = min(720, len(common_timestamps) - seq_len - horizon)  # 30 days
    test_start = len(common_timestamps) - test_len - seq_len - horizon

    # Build feature and return arrays
    features = {}
    returns = {}
    for symbol, frame in frames.items():
        frame = frame.loc[common_timestamps]
        available_cols = [c for c in feature_columns if c in frame.columns]
        features[symbol] = frame[available_cols].values.astype(np.float32)
        returns[symbol] = frame["return_1h"].values.astype(np.float32) if "return_1h" in frame.columns else np.zeros(len(frame), dtype=np.float32)

    # Simulate
    equity = 1.0
    equity_curve = [equity]
    prev_alloc = np.ones(num_assets) / num_assets
    all_returns = []

    sim = DifferentiablePortfolioSim(num_assets, tx_cost)

    with torch.no_grad():
        for t in range(test_start, len(common_timestamps) - seq_len - 1):
            # Get features for this timestep
            feat_list = []
            for symbol in symbols:
                feat = features[symbol][t:t + seq_len]
                feat_list.append(torch.from_numpy(feat))

            feat_tensor = torch.stack(feat_list).unsqueeze(0).to(device)  # (1, num_assets, seq_len, feat_dim)

            # Get allocation from model
            portfolio_state = torch.tensor(prev_alloc, dtype=torch.float32).unsqueeze(0).to(device)
            alloc, _ = model(feat_tensor, portfolio_state)
            alloc = alloc.cpu().numpy()[0]

            # Get next period returns
            ret_list = []
            for symbol in symbols:
                ret_list.append(returns[symbol][t + seq_len])
            ret = np.array(ret_list)

            # Compute portfolio return
            turnover = np.abs(alloc - prev_alloc).sum()
            port_return = (alloc * ret).sum() - turnover * tx_cost

            equity *= (1 + port_return)
            equity_curve.append(equity)
            all_returns.append(port_return)
            prev_alloc = alloc

    all_returns = np.array(all_returns)

    # Compute metrics
    total_return = (equity - 1) * 100
    downside = np.minimum(all_returns, 0)
    downside_std = np.sqrt((downside ** 2).mean()) + 1e-8
    sortino = all_returns.mean() / downside_std
    sharpe = all_returns.mean() / (all_returns.std() + 1e-8)
    max_dd = np.min(np.array(equity_curve) / np.maximum.accumulate(equity_curve)) - 1

    logger.info("Backtest Results:")
    logger.info("  Total Return: {:.2f}%", total_return)
    logger.info("  Sortino Ratio: {:.4f}", sortino)
    logger.info("  Sharpe Ratio: {:.4f}", sharpe)
    logger.info("  Max Drawdown: {:.2f}%", max_dd * 100)
    logger.info("  Num Steps: {}", len(all_returns))

    return {
        "total_return": total_return,
        "sortino": sortino,
        "sharpe": sharpe,
        "max_drawdown": max_dd * 100,
        "equity_curve": equity_curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args()

    model, symbols, feature_columns = load_model(args.checkpoint)
    logger.info("Loaded model with {} assets, {} features", len(symbols), len(feature_columns))

    frames, feature_columns = load_data(
        symbols, feature_columns, args.data_root, args.cache_root, args.sequence_length
    )

    results = backtest(model, frames, feature_columns, args.sequence_length)
    return results


if __name__ == "__main__":
    main()
