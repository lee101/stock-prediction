#!/usr/bin/env python3
"""Run SUI clamped forecast experiment: MAE comparison + trading bot + simulation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from suiclampedwithin.clamped_forecaster import (
    ClampedForecastConfig,
    ClampedForecaster,
    load_hourly_data,
    run_mae_experiment,
)
from suiclampedwithin.dual_forecast_data import DualForecastConfig, DualForecastDataModule

SYMBOL = "SUIUSDT"
DATA_ROOT = Path("binance_spot_hourly")
OUTPUT_DIR = Path("suiclampedwithin/results")
CACHE_ROOT = Path("suiclampedwithin/cache")


def run_mae_comparison(n_samples: int = 50) -> pd.DataFrame:
    """Run MAE comparison between clamped and unclamped forecasts."""
    logger.info("Running MAE comparison experiment")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for mode in ["scale", "clamp", "affine"]:
        logger.info(f"Testing clamp mode: {mode}")
        comparison = run_mae_experiment(
            symbol=SYMBOL,
            data_root=DATA_ROOT,
            cache_root=CACHE_ROOT,
            n_samples=n_samples,
            clamp_mode=mode,
        )
        summary = comparison.groupby("horizon").agg({
            "unclamped_mae_pct": "mean",
            "clamped_mae_pct": "mean",
        })
        results[mode] = summary
        comparison.to_csv(OUTPUT_DIR / f"mae_comparison_{mode}.csv", index=False)

    # Compile summary
    summary_rows = []
    for mode, df in results.items():
        for horizon in df.index:
            summary_rows.append({
                "mode": mode,
                "horizon": horizon,
                "unclamped_mae": df.loc[horizon, "unclamped_mae_pct"],
                "clamped_mae": df.loc[horizon, "clamped_mae_pct"],
                "improvement_pct": (
                    (df.loc[horizon, "unclamped_mae_pct"] - df.loc[horizon, "clamped_mae_pct"])
                    / df.loc[horizon, "unclamped_mae_pct"] * 100
                ),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "mae_summary.csv", index=False)
    logger.info(f"\nMAE Summary:\n{summary_df.to_string()}")
    return summary_df


def train_dual_forecast_policy(epochs: int = 20, lr: float = 1e-4) -> dict:
    """Train trading policy with dual forecast inputs."""
    logger.info("Training dual-forecast trading policy")

    config = DualForecastConfig(
        symbol=SYMBOL,
        data_root=DATA_ROOT,
        forecast_cache=CACHE_ROOT,
        sequence_length=48,
    )
    dm = DualForecastDataModule(config)
    dm.prepare()
    train_ds, val_ds, test_ds = dm.build_datasets()

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    logger.info(f"Features: {len(dm.feature_cols)}")

    # Simple MLP policy for now
    input_dim = len(dm.feature_cols)
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim * config.sequence_length, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 4),  # buy_price, sell_price, buy_amt, sell_amt
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            B, S, F = features.shape
            features_flat = features.view(B, -1)

            close = batch["close"][:, -1].to(device)
            clamped_high = batch["clamped_high"][:, -1].to(device)
            clamped_low = batch["clamped_low"][:, -1].to(device)

            outputs = model(features_flat)
            buy_price = close * (1 - torch.sigmoid(outputs[:, 0]) * 0.05)
            sell_price = close * (1 + torch.sigmoid(outputs[:, 1]) * 0.05)

            # Loss: encourage buy below clamped_low, sell above clamped_high
            buy_loss = torch.relu(buy_price - clamped_low).mean()
            sell_loss = torch.relu(clamped_high - sell_price).mean()
            spread_loss = torch.relu(buy_price - sell_price + 0.001 * close).mean()
            loss = buy_loss + sell_loss + spread_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device).view(batch["features"].size(0), -1)
                close = batch["close"][:, -1].to(device)
                clamped_high = batch["clamped_high"][:, -1].to(device)
                clamped_low = batch["clamped_low"][:, -1].to(device)

                outputs = model(features)
                buy_price = close * (1 - torch.sigmoid(outputs[:, 0]) * 0.05)
                sell_price = close * (1 + torch.sigmoid(outputs[:, 1]) * 0.05)

                buy_loss = torch.relu(buy_price - clamped_low).mean()
                sell_loss = torch.relu(clamped_high - sell_price).mean()
                val_loss += (buy_loss + sell_loss).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), OUTPUT_DIR / "best_policy.pt")
        logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_train:.4f}, Val: {avg_val:.4f}")

    return {"best_val_loss": best_val_loss, "epochs": epochs}


def run_backtest_simulation(fee_pct: float = 0.001) -> dict:
    """Run backtest simulation with trained policy."""
    logger.info("Running backtest simulation")

    config = DualForecastConfig(
        symbol=SYMBOL,
        data_root=DATA_ROOT,
        forecast_cache=CACHE_ROOT,
    )
    dm = DualForecastDataModule(config)
    dm.prepare()
    _, _, test_ds = dm.build_datasets()

    # Load model
    input_dim = len(dm.feature_cols) * config.sequence_length
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 4),
    )
    ckpt_path = OUTPUT_DIR / "best_policy.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    # Simple simulation
    cash = 10000.0
    position = 0.0
    trades = []

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    for i, batch in enumerate(test_loader):
        features = batch["features"].view(1, -1)
        close = batch["close"][0, -1].item()
        high = batch["high"][0, -1].item()
        low = batch["low"][0, -1].item()

        with torch.no_grad():
            outputs = model(features)
            buy_price = close * (1 - torch.sigmoid(outputs[0, 0]).item() * 0.05)
            sell_price = close * (1 + torch.sigmoid(outputs[0, 1]).item() * 0.05)
            trade_amt = torch.sigmoid(outputs[0, 2]).item() * 0.1

        # Execute if price hits
        if low <= buy_price and cash > 0:
            buy_qty = (cash * trade_amt) / buy_price
            cost = buy_qty * buy_price * (1 + fee_pct)
            if cost <= cash:
                cash -= cost
                position += buy_qty
                trades.append({"type": "buy", "price": buy_price, "qty": buy_qty, "i": i})

        if high >= sell_price and position > 0:
            sell_qty = position * trade_amt
            proceeds = sell_qty * sell_price * (1 - fee_pct)
            cash += proceeds
            position -= sell_qty
            trades.append({"type": "sell", "price": sell_price, "qty": sell_qty, "i": i})

    # Final value
    final_close = test_ds.closes[-1]
    final_value = cash + position * final_close
    pnl_pct = (final_value / 10000.0 - 1) * 100

    result = {
        "final_value": final_value,
        "pnl_pct": pnl_pct,
        "n_trades": len(trades),
        "final_cash": cash,
        "final_position": position,
    }
    logger.info(f"Backtest result: {result}")

    with open(OUTPUT_DIR / "backtest_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def update_progress_doc(mae_summary: pd.DataFrame, backtest_result: dict) -> None:
    """Update binanceprogress.md with experiment results."""
    progress_path = Path("binanceprogress.md")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    section = f"""

## SUI Clamped Forecast Experiment ({timestamp})

### Approach
- Generate 1-step daily forecast as envelope (high/low bounds)
- Generate 24-step hourly forecast
- Warp/clamp hourly forecast to fit within daily bounds
- Train trading policy using clamped forecasts

### MAE Comparison Results

| Mode | Horizon | Unclamped MAE% | Clamped MAE% | Improvement |
|------|---------|----------------|--------------|-------------|
"""
    for _, row in mae_summary.iterrows():
        section += f"| {row['mode']} | {row['horizon']}h | {row['unclamped_mae']:.3f} | {row['clamped_mae']:.3f} | {row['improvement_pct']:.1f}% |\n"

    section += f"""
### Backtest Results
- Final Value: ${backtest_result['final_value']:.2f}
- PnL: {backtest_result['pnl_pct']:.2f}%
- Trades: {backtest_result['n_trades']}
"""

    if progress_path.exists():
        with open(progress_path, "a") as f:
            f.write(section)
    else:
        with open(progress_path, "w") as f:
            f.write("# Binance Progress\n" + section)

    logger.info(f"Updated {progress_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mae-samples", type=int, default=50)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--skip-mae", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mae_summary = None
    if not args.skip_mae:
        mae_summary = run_mae_comparison(args.mae_samples)

    if not args.skip_train:
        train_dual_forecast_policy(epochs=args.train_epochs)

    backtest_result = {"final_value": 0, "pnl_pct": 0, "n_trades": 0}
    if not args.skip_backtest:
        backtest_result = run_backtest_simulation()

    if mae_summary is not None:
        update_progress_doc(mae_summary, backtest_result)


if __name__ == "__main__":
    main()
