#!/usr/bin/env python3
"""Sortino-focused training for hourly stocks (no forecasts needed)."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig

DATA_ROOT = Path("trainingdatahourly/stocks")
CHECKPOINT_ROOT = Path("stocksortino/checkpoints")
RESULTS_ROOT = Path("stocksortino/results")

STOCK_FEE = 0.0001  # $0.01 per share est for liquid stocks
EVAL_DAYS = 5  # 5 trading days = ~1 week

TECH_STOCKS = ["NVDA", "MSFT", "META", "GOOG", "NET", "PLTR"]
SHORTABLE_STOCKS = ["NYT", "YELP", "DBX", "TRIP"]


class StockSortinoDataModule:
    """Data module for stock hourly data (no forecasts)."""

    def __init__(
        self,
        symbol: str,
        sequence_length: int = 48,  # 48 hours = ~6 trading days
        val_days: int = EVAL_DAYS,
        test_days: int = EVAL_DAYS,
        max_history_days: int = 90,  # stocks have less history
    ):
        from torch.utils.data import DataLoader
        from binanceneural.data import BinanceHourlyDataset, FeatureNormalizer

        self.symbol = symbol.upper()
        self.sequence_length = sequence_length

        csv_path = DATA_ROOT / f"{self.symbol}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing data: {csv_path}")

        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Limit history
        if max_history_days:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=max_history_days)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

        df = self._add_features(df)

        # Split (use market hours: ~8 bars per day)
        bars_per_day = 8
        total_rows = len(df)
        test_rows = test_days * bars_per_day
        val_rows = val_days * bars_per_day
        train_end = total_rows - test_rows - val_rows
        val_end = total_rows - test_rows

        self.train_frame = df.iloc[:train_end].copy()
        self.val_frame = df.iloc[train_end - sequence_length:val_end].copy()
        self.test_frame = df.iloc[val_end - sequence_length:].copy()
        self.full_frame = df

        self.feature_columns = self._build_feature_columns()

        train_features = self.train_frame[list(self.feature_columns)].to_numpy(dtype=np.float32)
        self.normalizer = FeatureNormalizer.fit(train_features)

        norm_train = self.normalizer.transform(train_features)
        norm_val = self.normalizer.transform(self.val_frame[list(self.feature_columns)].to_numpy(dtype=np.float32))
        norm_test = self.normalizer.transform(self.test_frame[list(self.feature_columns)].to_numpy(dtype=np.float32))

        self.train_dataset = BinanceHourlyDataset(self.train_frame, norm_train, sequence_length, primary_horizon=1)
        self.val_dataset = BinanceHourlyDataset(self.val_frame, norm_val, sequence_length, primary_horizon=1)
        self.test_dataset = BinanceHourlyDataset(self.test_frame, norm_test, sequence_length, primary_horizon=1)

        logger.info(f"{symbol}: {len(self.train_frame)} train, {len(self.val_frame)} val, {len(self.test_frame)} test")

    def train_dataloader(self, batch_size: int, num_workers: int = 0):
        from torch.utils.data import DataLoader
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self, batch_size: int, num_workers: int = 0):
        from torch.utils.data import DataLoader
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features."""
        df["reference_close"] = df["close"]
        df["symbol"] = self.symbol

        # Add dummy chronos predictions (use close as prediction)
        df["predicted_close_p50_h1"] = df["close"]
        df["predicted_high_p50_h1"] = df["high"]
        df["predicted_low_p50_h1"] = df["low"]

        # Returns at various lookbacks
        df["return_1h"] = df["close"].pct_change(1).fillna(0)
        df["return_4h"] = df["close"].pct_change(4).fillna(0)
        df["return_8h"] = df["close"].pct_change(8).fillna(0)  # ~1 trading day
        df["return_24h"] = df["close"].pct_change(24).fillna(0)  # ~3 trading days

        # Volatility
        df["volatility_8h"] = df["return_1h"].rolling(8).std().fillna(0)
        df["volatility_24h"] = df["return_1h"].rolling(24).std().fillna(0)

        # Price momentum
        df["high_low_ratio"] = (df["high"] / df["low"].clip(lower=1e-10)) - 1
        df["close_open_ratio"] = (df["close"] / df["open"].clip(lower=1e-10)) - 1

        # Volume features
        if "volume" in df.columns:
            df["volume_ma8"] = df["volume"].rolling(8).mean().fillna(df["volume"])
            df["volume_ratio"] = df["volume"] / df["volume_ma8"].clip(lower=1e-10)
        else:
            df["volume_ratio"] = 1.0

        # Moving averages
        df["sma_8"] = df["close"].rolling(8).mean().fillna(df["close"])
        df["sma_24"] = df["close"].rolling(24).mean().fillna(df["close"])
        df["ma_cross"] = (df["sma_8"] / df["sma_24"].clip(lower=1e-10)) - 1

        # RSI approximation (14-period)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().fillna(0)
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().fillna(0)
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = (100 - (100 / (1 + rs))) / 100 - 0.5  # normalize to [-0.5, 0.5]

        return df

    def _build_feature_columns(self) -> tuple:
        """Build feature column list."""
        return (
            "return_1h", "return_4h", "return_8h", "return_24h",
            "volatility_8h", "volatility_24h",
            "high_low_ratio", "close_open_ratio", "volume_ratio",
            "ma_cross", "rsi_14",
        )


def run_backtest(checkpoint_path: Path, test_frame: pd.DataFrame, feature_columns, normalizer, fee: float = STOCK_FEE):
    """Run backtest with trained model."""
    from binancechronossolexperiment.inference import load_policy_checkpoint

    model, norm, feat_cols, cfg = load_policy_checkpoint(str(checkpoint_path))

    actions = generate_actions_from_frame(
        model=model,
        frame=test_frame,
        feature_columns=feat_cols,
        normalizer=norm,
        sequence_length=48,
        horizon=1,
    )

    config = SimulationConfig(maker_fee=fee, initial_cash=10000.0)
    sim = BinanceMarketSimulator(config)

    bars = test_frame.copy()
    if "timestamp" not in bars.columns and bars.index.name == "timestamp":
        bars = bars.reset_index()

    result = sim.run(bars, actions)
    eq = result.combined_equity

    ret = eq.pct_change().dropna()
    neg = ret[ret < 0]
    sortino = (ret.mean() / (neg.std() + 1e-10)) * np.sqrt(252 * 8) if len(neg) > 0 else 0  # annualized for stocks
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1

    return {
        "total_return": total_return,
        "sortino": sortino,
        "final_equity": eq.iloc[-1],
    }


def train_stock_model(
    symbol: str,
    epochs: int = 50,
    hidden_dim: int = 128,
    n_layers: int = 3,
    learning_rate: float = 1e-4,
    run_name: str = None,
):
    """Train Sortino-focused model for a stock."""
    run_name = run_name or f"{symbol.lower()}_sortino_{time.strftime('%Y%m%d_%H%M%S')}"

    dm = StockSortinoDataModule(symbol)

    config = TrainingConfig(
        epochs=epochs,
        batch_size=32,  # smaller batch for stocks (less data)
        sequence_length=48,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        optimizer_name="adamw",
        model_arch="classic",
        transformer_dim=hidden_dim,
        transformer_layers=n_layers,
        transformer_heads=4,
        transformer_dropout=0.1,
        maker_fee=STOCK_FEE,
        checkpoint_root=CHECKPOINT_ROOT,
        run_name=run_name,
        use_compile=False,
        use_amp=True,
        amp_dtype="bfloat16",
    )

    trainer = BinanceHourlyTrainer(config, dm)
    artifacts = trainer.train()

    checkpoint_dir = CHECKPOINT_ROOT / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": artifacts.state_dict,
        "config": asdict(config),
        "feature_columns": list(dm.feature_columns),
        "normalizer": dm.normalizer.to_dict(),
    }
    output_path = checkpoint_dir / "policy_checkpoint.pt"
    torch.save(payload, output_path)

    return output_path, dm.test_frame, dm.feature_columns, dm.normalizer, artifacts.history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Symbol to train (e.g., NVDA)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    logger.info(f"Training {args.symbol}: {args.epochs}ep, h={args.hidden_dim}, l={args.n_layers}")

    ckpt, test_frame, feat_cols, normalizer, history = train_stock_model(
        args.symbol,
        args.epochs,
        args.hidden_dim,
        args.n_layers,
        args.learning_rate,
        args.run_name,
    )

    logger.info("Running backtest...")
    result = run_backtest(ckpt, test_frame, feat_cols, normalizer)

    print(f"\n{'='*60}")
    print(f"{args.symbol} Stock Sortino Training Result ({args.epochs} epochs)")
    print(f"Hidden: {args.hidden_dim}, Layers: {args.n_layers}")
    print(f"{'='*60}")
    print(f"5d Return: {result['total_return']:.4f} ({result['total_return']*100:.2f}%)")
    print(f"Sortino: {result['sortino']:.2f}")
    print(f"Final Equity: ${result['final_equity']:.2f}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.symbol.lower()}_sortino"
    output = RESULTS_ROOT / f"results_{run_name}.json"
    output.write_text(json.dumps({
        "symbol": args.symbol,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "result": result,
        "history": [{"epoch": h.epoch, "val_sortino": h.val_sortino, "val_return": h.val_return}
                   for h in history],
    }, indent=2))
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    main()
