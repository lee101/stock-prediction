#!/usr/bin/env python3
"""Train multi-asset RL policy with differentiable simulation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig
from unified_hourly_experiment.multiasset_policy import (
    MultiAssetConfig,
    MultiAssetPolicy,
    DifferentiablePortfolioSim,
    build_multiasset_policy,
)


class MultiAssetDataset(Dataset):
    """Dataset for multi-asset training."""

    def __init__(
        self,
        frames: Dict[str, "pd.DataFrame"],
        feature_columns: List[str],
        sequence_length: int = 32,
        horizon: int = 24,
    ):
        import pandas as pd
        import numpy as np

        self.symbols = list(frames.keys())
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.horizon = horizon

        # Find common timestamps across all frames
        timestamp_sets = [set(frame.index.tolist()) for frame in frames.values()]
        common_timestamps = sorted(set.intersection(*timestamp_sets))

        # Build feature arrays and return arrays
        self.features = {}
        self.returns = {}

        for symbol, frame in frames.items():
            frame = frame.loc[common_timestamps]
            available_cols = [c for c in feature_columns if c in frame.columns]
            feat_arr = frame[available_cols].values.astype(np.float32)
            ret_arr = frame["return_1h"].values.astype(np.float32) if "return_1h" in frame.columns else np.zeros(len(frame), dtype=np.float32)
            self.features[symbol] = feat_arr
            self.returns[symbol] = ret_arr

        self.num_samples = len(common_timestamps) - sequence_length - horizon
        self.feature_dim = len(available_cols)

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Features: (num_assets, seq_len, feature_dim)
        features = []
        returns = []

        for symbol in self.symbols:
            feat = self.features[symbol][idx : idx + self.sequence_length]
            ret = self.returns[symbol][idx + self.sequence_length : idx + self.sequence_length + self.horizon]
            features.append(torch.from_numpy(feat))
            returns.append(torch.from_numpy(ret))

        features = torch.stack(features)  # (num_assets, seq_len, feature_dim)
        returns = torch.stack(returns).T  # (horizon, num_assets)

        return features, returns


class MultiAssetTrainer:
    """Trainer for multi-asset policy."""

    def __init__(
        self,
        config: MultiAssetConfig,
        symbols: List[str],
        data_root: Path,
        cache_root: Path,
        checkpoint_root: Path,
        sequence_length: int = 32,
        horizon: int = 24,
        batch_size: int = 32,
        lr: float = 1e-4,
        epochs: int = 100,
    ):
        self.config = config
        self.symbols = symbols
        self.data_root = data_root
        self.cache_root = cache_root
        self.checkpoint_root = checkpoint_root
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_root / f"multiasset_{len(symbols)}s_{config.hidden_dim}h"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[MultiAssetDataset, MultiAssetDataset]:
        """Load data for all symbols."""
        import pandas as pd

        frames = {}
        feature_columns = None

        for symbol in self.symbols:
            data_config = DatasetConfig(
                symbol=symbol,
                data_root=str(self.data_root),
                forecast_cache_root=str(self.cache_root),
                forecast_horizons=[1, 24],
                sequence_length=self.sequence_length,
                min_history_hours=100,
                validation_days=30,
                cache_only=True,
            )
            try:
                dm = BinanceHourlyDataModule(data_config)
                frames[symbol] = dm.frame
                if feature_columns is None:
                    feature_columns = dm.feature_columns
                logger.info("Loaded {} with {} rows", symbol, len(dm.frame))
            except Exception as e:
                logger.warning("Failed to load {}: {}", symbol, e)

        if not frames:
            raise ValueError("No data loaded")

        # Split into train/val by time
        min_len = min(len(f) for f in frames.values())
        split_idx = int(min_len * 0.85)

        train_frames = {s: f.iloc[:split_idx] for s, f in frames.items()}
        val_frames = {s: f.iloc[split_idx:] for s, f in frames.items()}

        train_dataset = MultiAssetDataset(train_frames, feature_columns, self.sequence_length, self.horizon)
        val_dataset = MultiAssetDataset(val_frames, feature_columns, self.sequence_length, self.horizon)

        self.feature_columns = feature_columns
        return train_dataset, val_dataset

    def compute_loss(
        self,
        model: MultiAssetPolicy,
        sim: DifferentiablePortfolioSim,
        features: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch."""
        batch_size, horizon, num_assets = returns.shape

        # Get allocations for each timestep
        allocations = []
        portfolio_state = torch.ones(batch_size, num_assets, device=self.device) / num_assets

        for t in range(horizon):
            alloc, _ = model(features, portfolio_state)
            allocations.append(alloc)
            # Soft update portfolio state based on returns
            portfolio_state = portfolio_state * (1 + returns[:, t])
            portfolio_state = portfolio_state / (portfolio_state.sum(dim=-1, keepdim=True) + 1e-8)

        allocations = torch.stack(allocations, dim=1)  # (batch, horizon, num_assets)

        # Simulate portfolio
        equity_curve, portfolio_returns = sim(allocations, returns)

        # Compute Sortino ratio as reward
        sortino = sim.compute_sortino(portfolio_returns)

        # Entropy regularization - encourage diverse allocations
        entropy = -(allocations * torch.log(allocations + 1e-8)).sum(dim=-1).mean()

        # Loss = -Sortino - entropy_coef * entropy
        loss = -sortino.mean() - 0.01 * entropy

        # Additional metrics
        total_return = (equity_curve[:, -1] - 1).mean()
        turnover = torch.abs(allocations[:, 1:] - allocations[:, :-1]).sum(dim=-1).mean()

        metrics = {
            "loss": loss.item(),
            "sortino": sortino.mean().item(),
            "return": total_return.item() * 100,
            "turnover": turnover.item(),
        }

        return loss, metrics

    def train(self) -> Path:
        """Train the multi-asset policy."""
        train_dataset, val_dataset = self.load_data()

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # Update config with actual feature dim
        self.config.feature_dim = len(self.feature_columns)
        self.config.num_assets = len(self.symbols)

        model = build_multiasset_policy(self.config).to(self.device)
        sim = DifferentiablePortfolioSim(len(self.symbols)).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_sortino = float("-inf")
        best_checkpoint = None
        patience_counter = 0
        patience = 15

        for epoch in range(1, self.epochs + 1):
            # Training
            model.train()
            train_metrics = {"loss": 0, "sortino": 0, "return": 0, "turnover": 0}
            num_batches = 0

            for features, returns in train_loader:
                features = features.to(self.device)
                returns = returns.to(self.device)

                optimizer.zero_grad()
                loss, metrics = self.compute_loss(model, sim, features, returns)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                for k, v in metrics.items():
                    train_metrics[k] += v
                num_batches += 1

            for k in train_metrics:
                train_metrics[k] /= max(1, num_batches)

            # Validation
            model.eval()
            val_metrics = {"loss": 0, "sortino": 0, "return": 0, "turnover": 0}
            num_batches = 0

            with torch.no_grad():
                for features, returns in val_loader:
                    features = features.to(self.device)
                    returns = returns.to(self.device)
                    _, metrics = self.compute_loss(model, sim, features, returns)
                    for k, v in metrics.items():
                        val_metrics[k] += v
                    num_batches += 1

            for k in val_metrics:
                val_metrics[k] /= max(1, num_batches)

            scheduler.step()

            # Save best
            if val_metrics["sortino"] > best_sortino:
                best_sortino = val_metrics["sortino"]
                patience_counter = 0
                best_checkpoint = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": {
                        "num_assets": self.config.num_assets,
                        "feature_dim": self.config.feature_dim,
                        "hidden_dim": self.config.hidden_dim,
                        "num_heads": self.config.num_heads,
                        "num_layers": self.config.num_layers,
                        "max_len": self.config.max_len,
                    },
                    "symbols": self.symbols,
                    "feature_columns": self.feature_columns,
                }, best_checkpoint)
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Sortino: {train_metrics['sortino']:.4f} Return: {train_metrics['return']:.2f}% | "
                f"Val Sortino: {val_metrics['sortino']:.4f} Return: {val_metrics['return']:.2f}%"
            )

            if patience_counter >= patience:
                logger.info("Early stopping at epoch {}", epoch)
                break

        # Save config
        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump({
                "symbols": self.symbols,
                "feature_columns": self.feature_columns,
                "hidden_dim": self.config.hidden_dim,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "sequence_length": self.sequence_length,
                "horizon": self.horizon,
            }, f, indent=2)

        logger.success("Training complete! Best Sortino: {:.4f}", best_sortino)
        return best_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP,KIND,EBAY,MTCH,ANGI,Z,EXPE,BKNG,NWSA")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    logger.info("Training multi-asset policy on {} symbols", len(symbols))

    config = MultiAssetConfig(
        num_assets=len(symbols),
        feature_dim=16,  # Will be updated
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=args.sequence_length,
    )

    trainer = MultiAssetTrainer(
        config=config,
        symbols=symbols,
        data_root=args.data_root,
        cache_root=args.cache_root,
        checkpoint_root=args.checkpoint_root,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
    )

    trainer.train()


if __name__ == "__main__":
    main()
