#!/usr/bin/env python3
"""Training script with enhanced features (v2)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bagsneural.dataset_v2 import (
    load_ohlc_dataframe,
    build_features_and_targets_v2,
    fit_normalizer_v2,
    FeatureNormalizerV2,
)
from bagsneural.model import BagsNeuralModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_model_v2(
    ohlc_path: Path,
    mint: str,
    context_bars: int = 64,
    horizon: int = 3,
    cost_bps: float = 130.0,
    min_return: float = 0.002,
    size_scale: float = 0.02,
    hidden_dims: list = None,
    dropout: float = 0.1,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_split: float = 0.2,
    pos_weight: Optional[float] = None,
    signal_weight: float = 1.0,
    size_weight: float = 1.0,
    device: str = "cuda",
    out_dir: Path = None,
) -> dict:
    """Train model with enhanced features.

    Returns:
        Dict with training info and best validation loss.
    """
    if hidden_dims is None:
        hidden_dims = [128, 64]
    if out_dir is None:
        out_dir = Path("bagsneural/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_ohlc_dataframe(ohlc_path, mint)
    logger.info(f"Loaded {len(df)} OHLC bars for {mint[:8]}...")

    # Build features and targets
    features, signal_targets, size_targets, timestamps = build_features_and_targets_v2(
        df=df,
        context_bars=context_bars,
        horizon=horizon,
        cost_bps=cost_bps,
        min_return=min_return,
        size_scale=size_scale,
    )

    logger.info(f"Built {len(features)} samples with {features.shape[1]} features (v2)")

    # Split
    split_idx = int(len(features) * (1 - val_split))
    train_features = features[:split_idx]
    train_signals = signal_targets[:split_idx]
    train_sizes = size_targets[:split_idx]

    val_features = features[split_idx:]
    val_signals = signal_targets[split_idx:]
    val_sizes = size_targets[split_idx:]

    # Fit normalizer on training data
    normalizer = fit_normalizer_v2(train_features)
    train_features = normalizer.transform(train_features)
    val_features = normalizer.transform(val_features)

    # Compute positive weight for class imbalance
    pos_rate = train_signals.mean()
    if pos_weight is None:
        pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)
    logger.info(f"Signal positive rate: {pos_rate:.4f} | pos_weight={pos_weight:.2f}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_signals, dtype=torch.float32),
        torch.tensor(train_sizes, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_features, dtype=torch.float32),
        torch.tensor(val_signals, dtype=torch.float32),
        torch.tensor(val_sizes, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = BagsNeuralModel(
        input_dim=features.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    # Losses
    signal_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    size_loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_signal, batch_size_t in train_loader:
            batch_x = batch_x.to(device)
            batch_signal = batch_signal.to(device)
            batch_size_t = batch_size_t.to(device)

            optimizer.zero_grad()
            signal_logit, size_logit = model(batch_x)

            loss_signal = signal_loss_fn(signal_logit.squeeze(), batch_signal)
            loss_size = size_loss_fn(torch.sigmoid(size_logit.squeeze()), batch_size_t)
            loss = signal_weight * loss_signal + size_weight * loss_size

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= len(train_dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_signal, batch_size_t in val_loader:
                batch_x = batch_x.to(device)
                batch_signal = batch_signal.to(device)
                batch_size_t = batch_size_t.to(device)

                signal_logit, size_logit = model(batch_x)
                loss_signal = signal_loss_fn(signal_logit.squeeze(), batch_signal)
                loss_size = size_loss_fn(torch.sigmoid(size_logit.squeeze()), batch_size_t)
                loss = signal_weight * loss_signal + size_weight * loss_size
                val_loss += loss.item() * len(batch_x)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "config": {
            "context": context_bars,
            "horizon": horizon,
            "cost_bps": cost_bps,
            "min_return": min_return,
            "size_scale": size_scale,
            "hidden": hidden_dims,
            "dropout": dropout,
            "version": "v2",
        },
    }

    save_path = out_dir / f"bagsneural_v2_{mint}_best.pt"
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")

    return {
        "best_val_loss": best_val_loss,
        "checkpoint_path": save_path,
        "input_dim": features.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Train Bags.fm neural model with enhanced features")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining/ohlc_data.csv"))
    parser.add_argument("--mint", type=str, required=True)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--min-return", type=float, default=0.002)
    parser.add_argument("--size-scale", type=float, default=0.02)
    parser.add_argument("--cost-bps", type=float, default=130.0)
    parser.add_argument("--hidden", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("bagsneural/checkpoints"))

    args = parser.parse_args()
    hidden_dims = [int(x) for x in args.hidden.split(",")]

    result = train_model_v2(
        ohlc_path=args.ohlc,
        mint=args.mint,
        context_bars=args.context,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        min_return=args.min_return,
        size_scale=args.size_scale,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device,
        out_dir=args.out_dir,
    )

    logger.info(f"Best validation loss: {result['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
