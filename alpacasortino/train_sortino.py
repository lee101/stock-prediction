#!/usr/bin/env python3
"""Sortino-focused training for Alpaca crypto pairs."""
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

DATA_ROOT = Path("trainingdatahourly/crypto")
FORECAST_CACHE = Path("binanceneural/forecast_cache")
CHECKPOINT_ROOT = Path("alpacasortino/checkpoints")
RESULTS_ROOT = Path("alpacasortino/results")

MAKER_FEE = 0.0015  # Alpaca crypto fee
EVAL_DAYS = 7


class AlpacaSortinoDataModule:
    """Data module for Alpaca hourly data with Chronos forecasts."""

    def __init__(
        self,
        symbol: str,
        sequence_length: int = 72,
        val_days: int = EVAL_DAYS,
        test_days: int = EVAL_DAYS,
        max_history_days: int = 365,
    ):
        from torch.utils.data import DataLoader
        from binanceneural.data import BinanceHourlyDataset, FeatureNormalizer

        self.symbol = symbol.upper()
        self.sequence_length = sequence_length

        # Load price data
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

        # Load forecasts
        df = self._add_forecasts(df)

        # Add features
        df = self._add_features(df)

        # Split
        total_hours = len(df)
        test_hours = test_days * 24
        val_hours = val_days * 24
        train_end = total_hours - test_hours - val_hours
        val_end = total_hours - test_hours

        self.train_frame = df.iloc[:train_end].copy()
        self.val_frame = df.iloc[train_end - sequence_length:val_end].copy()
        self.test_frame = df.iloc[val_end - sequence_length:].copy()
        self.full_frame = df

        # Build feature columns
        self.feature_columns = self._build_feature_columns()

        # Fit normalizer on train
        train_features = self.train_frame[list(self.feature_columns)].to_numpy(dtype=np.float32)
        self.normalizer = FeatureNormalizer.fit(train_features)

        # Create datasets
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

    def _add_forecasts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Chronos forecast columns."""
        self.available_horizons = []
        for horizon in [1, 24]:  # Only h1 and h24 available
            cache_path = FORECAST_CACHE / f"h{horizon}" / f"{self.symbol}.parquet"
            if cache_path.exists():
                forecasts = pd.read_parquet(cache_path)
                forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"], utc=True)
                self.available_horizons.append(horizon)

                # Merge forecasts
                for col in ["predicted_close_p50", "predicted_high_p50", "predicted_low_p50"]:
                    if col in forecasts.columns:
                        forecast_col = f"{col}_h{horizon}"
                        merge_df = forecasts[["timestamp", col]].rename(columns={col: forecast_col})
                        df = df.merge(merge_df, on="timestamp", how="left")

        # Fill missing with close
        for col in df.columns:
            if col.startswith("predicted_"):
                df[col] = df[col].fillna(df["close"])

        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features."""
        df["reference_close"] = df["close"]
        df["symbol"] = self.symbol

        # Returns
        df["return_1h"] = df["close"].pct_change(1).fillna(0)
        df["return_4h"] = df["close"].pct_change(4).fillna(0)
        df["return_24h"] = df["close"].pct_change(24).fillna(0)

        # Volatility
        df["volatility_24h"] = df["return_1h"].rolling(24).std().fillna(0)

        # Price ratios
        df["high_low_ratio"] = (df["high"] / df["low"].clip(lower=1e-10)) - 1
        df["close_open_ratio"] = (df["close"] / df["open"].clip(lower=1e-10)) - 1

        # Volume features
        if "volume" in df.columns:
            df["volume_ma24"] = df["volume"].rolling(24).mean().fillna(df["volume"])
            df["volume_ratio"] = df["volume"] / df["volume_ma24"].clip(lower=1e-10)
        else:
            df["volume_ratio"] = 1.0

        # Forecast features (only for available horizons)
        for horizon in self.available_horizons:
            pred_col = f"predicted_close_p50_h{horizon}"
            if pred_col in df.columns:
                df[f"forecast_return_h{horizon}"] = (df[pred_col] / df["close"].clip(lower=1e-10)) - 1

        return df

    def _build_feature_columns(self) -> tuple:
        """Build feature column list."""
        cols = [
            "return_1h", "return_4h", "return_24h",
            "volatility_24h", "high_low_ratio", "close_open_ratio", "volume_ratio",
        ]
        for horizon in self.available_horizons:
            cols.append(f"forecast_return_h{horizon}")
        return tuple(cols)


def run_backtest(checkpoint_path: Path, test_frame: pd.DataFrame, feature_columns, normalizer, fee: float = MAKER_FEE):
    """Run backtest with trained model."""
    from binancechronossolexperiment.inference import load_policy_checkpoint

    model, norm, feat_cols, cfg = load_policy_checkpoint(str(checkpoint_path))

    actions = generate_actions_from_frame(
        model=model,
        frame=test_frame,
        feature_columns=feat_cols,
        normalizer=norm,
        sequence_length=72,
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
    sortino = (ret.mean() / (neg.std() + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1

    return {
        "total_return": total_return,
        "sortino": sortino,
        "final_equity": eq.iloc[-1],
    }


def train_sortino_model(
    symbol: str,
    epochs: int = 30,
    hidden_dim: int = 256,
    n_layers: int = 4,
    learning_rate: float = 3e-4,
    run_name: str = None,
):
    """Train Sortino-focused model for a symbol."""
    run_name = run_name or f"{symbol.lower()}_sortino_{time.strftime('%Y%m%d_%H%M%S')}"

    dm = AlpacaSortinoDataModule(symbol)

    config = TrainingConfig(
        epochs=epochs,
        batch_size=64,
        sequence_length=72,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        optimizer_name="adamw",
        model_arch="classic",
        transformer_dim=hidden_dim,
        transformer_layers=n_layers,
        transformer_heads=8,
        transformer_dropout=0.1,
        maker_fee=MAKER_FEE,
        checkpoint_root=CHECKPOINT_ROOT,
        run_name=run_name,
        use_compile=False,
    )

    trainer = BinanceHourlyTrainer(config, dm)
    artifacts = trainer.train()

    # Save checkpoint
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
    parser.add_argument("--symbol", required=True, help="Symbol to train (e.g., ETHUSD)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    logger.info(f"Training {args.symbol}: {args.epochs}ep, h={args.hidden_dim}, l={args.n_layers}")

    ckpt, test_frame, feat_cols, normalizer, history = train_sortino_model(
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
    print(f"{args.symbol} Sortino Training Result ({args.epochs} epochs)")
    print(f"Hidden: {args.hidden_dim}, Layers: {args.n_layers}")
    print(f"{'='*60}")
    print(f"7d Return: {result['total_return']:.4f} ({result['total_return']*100:.2f}%)")
    print(f"Sortino: {result['sortino']:.2f}")
    print(f"Final Equity: ${result['final_equity']:.2f}")

    # Save results
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
