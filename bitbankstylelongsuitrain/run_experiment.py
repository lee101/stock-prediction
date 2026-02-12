#!/usr/bin/env python3
"""Bitbank-style training experiment for SUI."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from differentiable_loss_utils import simulate_hourly_trades, combined_sortino_pnl_loss, compute_hourly_objective
from .model import BitbankHourlyCryptoPolicy

SYMBOL = "SUIUSDT"
MAKER_FEE = 0.001
EVAL_DAYS = 7


@dataclass
class TrainResult:
    checkpoint_path: Path
    val_sortino: float
    val_return: float
    history: list


def prepare_data(data_root: Path, forecast_cache: Path, seq_len: int = 72, val_days: int = 7, test_days: int = 7):
    from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

    dm = ChronosSolDataModule(
        symbol=SYMBOL,
        data_root=data_root,
        forecast_cache_root=forecast_cache,
        forecast_horizons=(1, 4, 24),
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=seq_len,
        split_config=SplitConfig(val_days=val_days, test_days=test_days),
        max_history_days=365,
        cache_only=True,
    )

    return dm


def create_dataloaders(dm, batch_size: int = 64):
    train_loader = dm.train_dataloader(batch_size=batch_size, num_workers=0)
    val_loader = dm.val_dataloader(batch_size=batch_size, num_workers=0)
    return train_loader, val_loader


def train_bitbank_style(
    epochs: int,
    run_name: str,
    hidden_dim: int = 256,
    n_layers: int = 4,
    n_heads: int = 8,
    aggressiveness: float = 0.8,
    lr: float = 3e-4,
):
    data_root = Path("trainingdatahourlybinance")
    forecast_cache = Path("binancechronossolexperiment/forecast_cache_sui_10bp")
    checkpoint_root = Path("bitbankstylelongsuitrain/checkpoints") / run_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    dm = prepare_data(data_root, forecast_cache)
    train_loader, val_loader = create_dataloaders(dm)

    input_dim = len(dm.feature_columns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BitbankHourlyCryptoPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.1,
        seq_len=72,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    best_val_score = float("-inf")
    best_checkpoint = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses, train_sortinos, train_returns = [], [], []

        for batch in train_loader:
            features = batch["features"].to(device)
            highs = batch["high"].to(device)
            lows = batch["low"].to(device)
            closes = batch["close"].to(device)
            ref_close = batch["reference_close"].to(device)
            chronos_high = batch["chronos_high"].to(device)
            chronos_low = batch["chronos_low"].to(device)

            outputs = model(features)
            actions = model.decode_actions(
                outputs,
                reference_close=ref_close,
                chronos_low=chronos_low,
                chronos_high=chronos_high,
                aggressiveness=aggressiveness,
            )

            sim = simulate_hourly_trades(
                highs=highs,
                lows=lows,
                closes=closes,
                buy_prices=actions["buy_price"],
                sell_prices=actions["sell_price"],
                trade_intensity=actions["trade_amount"],
                buy_trade_intensity=actions["buy_amount"],
                sell_trade_intensity=actions["sell_amount"],
                maker_fee=MAKER_FEE,
                initial_cash=1.0,
            )

            loss = combined_sortino_pnl_loss(sim.returns.float())
            score, sortino, annual_return = compute_hourly_objective(sim.returns.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_sortinos.append(sortino.mean().item())
            train_returns.append(annual_return.mean().item())

        # Validate
        model.eval()
        val_losses, val_sortinos, val_returns = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                highs = batch["high"].to(device)
                lows = batch["low"].to(device)
                closes = batch["close"].to(device)
                ref_close = batch["reference_close"].to(device)
                chronos_high = batch["chronos_high"].to(device)
                chronos_low = batch["chronos_low"].to(device)

                outputs = model(features)
                actions = model.decode_actions(
                    outputs,
                    reference_close=ref_close,
                    chronos_low=chronos_low,
                    chronos_high=chronos_high,
                    aggressiveness=aggressiveness,
                )

                sim = simulate_hourly_trades(
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    buy_prices=actions["buy_price"],
                    sell_prices=actions["sell_price"],
                    trade_intensity=actions["trade_amount"],
                    buy_trade_intensity=actions["buy_amount"],
                    sell_trade_intensity=actions["sell_amount"],
                    maker_fee=MAKER_FEE,
                    initial_cash=1.0,
                )

                loss = combined_sortino_pnl_loss(sim.returns.float())
                score, sortino, annual_return = compute_hourly_objective(sim.returns.float())

                val_losses.append(loss.item())
                val_sortinos.append(sortino.mean().item())
                val_returns.append(annual_return.mean().item())

        avg_val_sortino = np.mean(val_sortinos)
        avg_val_return = np.mean(val_returns)

        history.append({
            "epoch": epoch,
            "train_sortino": np.mean(train_sortinos),
            "train_return": np.mean(train_returns),
            "val_sortino": avg_val_sortino,
            "val_return": avg_val_return,
        })

        print(f"Epoch {epoch}/{epochs} | Train Sortino: {np.mean(train_sortinos):.2f} Return: {np.mean(train_returns):.2f} | "
              f"Val Sortino: {avg_val_sortino:.2f} Return: {avg_val_return:.2f}")

        if avg_val_sortino > best_val_score:
            best_val_score = avg_val_sortino
            best_checkpoint = checkpoint_root / f"best_epoch_{epoch:03d}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_sortino": avg_val_sortino,
                "config": {
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "aggressiveness": aggressiveness,
                },
                "feature_columns": list(dm.feature_columns),
                "normalizer": dm.normalizer.to_dict(),
            }, best_checkpoint)

    return TrainResult(
        checkpoint_path=best_checkpoint,
        val_sortino=best_val_score,
        val_return=history[-1]["val_return"] if history else 0,
        history=history,
    )


def run_backtest(checkpoint_path: Path, test_frame: pd.DataFrame, aggressiveness: float = 0.8):
    from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BitbankHourlyCryptoPolicy(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    feature_columns = ckpt["feature_columns"]
    normalizer_data = ckpt["normalizer"]

    from binanceneural.data import FeatureNormalizer
    normalizer = FeatureNormalizer.from_dict(normalizer_data)

    # Generate actions
    from binanceneural.inference import generate_actions_from_frame

    # Use custom inference
    bars = test_frame.copy()
    if "timestamp" not in bars.columns and bars.index.name == "timestamp":
        bars = bars.reset_index()

    # Run simulation
    sim_config = SimulationConfig(maker_fee=MAKER_FEE, initial_cash=10000.0)
    sim = BinanceMarketSimulator(sim_config)

    # For now, use standard inference then adjust
    from binancechronossolexperiment.inference import load_policy_checkpoint
    # This won't work directly, so we need to do manual inference

    # Manual inference
    seq_len = 72
    actions_list = []

    for i in range(seq_len, len(test_frame)):
        window = test_frame.iloc[i-seq_len:i]
        features = normalizer.transform(window[feature_columns].values)
        features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(features_t)

        ref_close = test_frame["close"].iloc[i-1]
        chronos_low = test_frame.get("chronos_low_h1", pd.Series([ref_close * 0.99])).iloc[i-1] if "chronos_low_h1" in test_frame.columns else ref_close * 0.99
        chronos_high = test_frame.get("chronos_high_h1", pd.Series([ref_close * 1.01])).iloc[i-1] if "chronos_high_h1" in test_frame.columns else ref_close * 1.01

        actions = model.decode_actions(
            outputs,
            reference_close=torch.tensor([ref_close], device=device),
            chronos_low=torch.tensor([chronos_low], device=device),
            chronos_high=torch.tensor([chronos_high], device=device),
            aggressiveness=aggressiveness,
        )

        ts = test_frame.index[i] if isinstance(test_frame.index, pd.DatetimeIndex) else test_frame["timestamp"].iloc[i]
        actions_list.append({
            "timestamp": ts,
            "symbol": SYMBOL,
            "buy_price": actions["buy_price"].item(),
            "sell_price": actions["sell_price"].item(),
            "buy_amount": actions["buy_amount"].item(),
            "sell_amount": actions["sell_amount"].item(),
        })

    actions_df = pd.DataFrame(actions_list)
    result = sim.run(bars, actions_df)
    eq = result.combined_equity

    ret = eq.pct_change().dropna()
    neg = ret[ret < 0]
    sortino = (ret.mean() / (neg.std() + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1

    return {
        "total_return": total_return,
        "sortino": sortino,
        "final_equity": eq.iloc[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--aggressiveness", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"bitbank_sui_{args.epochs}ep_h{args.hidden_dim}_l{args.n_layers}_{run_id}"

    logger.info(f"Training bitbank-style model: {args.epochs} epochs, hidden={args.hidden_dim}, layers={args.n_layers}")
    result = train_bitbank_style(
        epochs=args.epochs,
        run_name=run_name,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        aggressiveness=args.aggressiveness,
        lr=args.lr,
    )

    logger.info(f"Best val sortino: {result.val_sortino:.2f}")

    # Run backtest
    from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
    dm = prepare_data(
        Path("trainingdatahourlybinance"),
        Path("binancechronossolexperiment/forecast_cache_sui_10bp"),
    )

    logger.info("Running backtest...")
    backtest = run_backtest(result.checkpoint_path, dm.test_frame, aggressiveness=args.aggressiveness)

    print(f"\n{'='*60}")
    print(f"Bitbank-Style Result ({args.epochs} epochs, 10bp fee)")
    print(f"{'='*60}")
    print(f"7d Return: {backtest['total_return']:.4f} ({backtest['total_return']*100:.2f}%)")
    print(f"Sortino: {backtest['sortino']:.2f}")
    print(f"Final Equity: ${backtest['final_equity']:.2f}")

    # Save results
    output = Path(f"bitbankstylelongsuitrain/results_{run_name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "aggressiveness": args.aggressiveness,
        "run_name": run_name,
        "val_sortino": result.val_sortino,
        "backtest": backtest,
        "history": result.history,
    }, indent=2))
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    main()
