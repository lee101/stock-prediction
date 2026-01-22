#!/usr/bin/env python3
"""Train a neural model for Bags.fm trading signals and sizing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bagsfm.config import CostConfig
from bagsneural.dataset import (
    FeatureNormalizer,
    build_features_and_targets,
    fit_normalizer,
    load_ohlc_dataframe,
)
from bagsneural.model import BagsNeuralModel

logger = logging.getLogger("bagsneural_train")


def _split_by_time(
    features: np.ndarray,
    signal_targets: np.ndarray,
    size_targets: np.ndarray,
    timestamps: np.ndarray,
    val_split: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")

    split_idx = int(len(features) * (1 - val_split))
    train = (features[:split_idx], signal_targets[:split_idx], size_targets[:split_idx])
    val = (features[split_idx:], signal_targets[split_idx:], size_targets[split_idx:])
    return train, val


def _make_loaders(
    train: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_features, train_signal, train_size = train
    val_features, val_signal, val_size = val

    train_ds = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_signal, dtype=torch.float32),
        torch.tensor(train_size, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(val_features, dtype=torch.float32),
        torch.tensor(val_signal, dtype=torch.float32),
        torch.tensor(val_size, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bags.fm neural trading model")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining") / "ohlc_data.csv")
    parser.add_argument("--mint", type=str, required=True, help="Token mint to train on")
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--min-return", type=float, default=0.002)
    parser.add_argument("--size-scale", type=float, default=0.02)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--hidden", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--train-split",
        type=float,
        default=1.0,
        help="Fraction of data to use for training (rest is held out)",
    )
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--signal-weight", type=float, default=1.0)
    parser.add_argument("--size-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("bagsneural") / "checkpoints")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.cost_bps is None:
        costs = CostConfig()
        args.cost_bps = costs.estimated_swap_fee_bps + costs.default_slippage_bps

    logger.info(
        "Training with cost_bps=%.2f min_return=%.4f size_scale=%.4f",
        args.cost_bps,
        args.min_return,
        args.size_scale,
    )

    df = load_ohlc_dataframe(args.ohlc, args.mint, dedupe=True)
    if args.train_split < 1.0:
        if not 0.0 < args.train_split < 1.0:
            raise ValueError("train-split must be between 0 and 1")
        train_rows = int(len(df) * args.train_split)
        df = df.iloc[:train_rows].reset_index(drop=True)
    features, signal_targets, size_targets, timestamps = build_features_and_targets(
        df,
        context_bars=args.context,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        min_return=args.min_return,
        size_scale=args.size_scale,
    )

    train, val = _split_by_time(features, signal_targets, size_targets, timestamps, args.val_split)

    normalizer = fit_normalizer(train[0])
    train_features = normalizer.transform(train[0])
    val_features = normalizer.transform(val[0])

    train = (train_features, train[1], train[2])
    val = (val_features, val[1], val[2])

    hidden_dims = [int(x) for x in args.hidden.split(",") if x.strip()]

    model = BagsNeuralModel(
        input_dim=train_features.shape[1],
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader = _make_loaders(train, val, args.batch_size)

    if args.pos_weight is None:
        positives = float(signal_targets.sum())
        negatives = float(len(signal_targets) - positives)
        pos_weight = negatives / max(positives, 1.0)
        args.pos_weight = min(pos_weight, 50.0)
    logger.info(
        "Signal positive rate: %.4f | pos_weight=%.2f",
        signal_targets.mean(),
        args.pos_weight,
    )

    signal_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    )
    size_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x, signal_target, size_target = [b.to(device) for b in batch]
            optimizer.zero_grad()
            signal_logit, size_logit = model(x)
            size_pred = torch.sigmoid(size_logit)

            signal_loss = signal_loss_fn(signal_logit, signal_target)
            size_loss = size_loss_fn(size_pred, size_target)
            loss = args.signal_weight * signal_loss + args.size_weight * size_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, signal_target, size_target = [b.to(device) for b in batch]
                signal_logit, size_logit = model(x)
                size_pred = torch.sigmoid(size_logit)
                signal_loss = signal_loss_fn(signal_logit, signal_target)
                size_loss = size_loss_fn(size_pred, size_target)
                loss = args.signal_weight * signal_loss + args.size_weight * size_loss
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        logger.info(
            "Epoch %d/%d - train_loss=%.6f val_loss=%.6f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "normalizer": normalizer.to_dict(),
                "config": {
                    "context": args.context,
                    "horizon": args.horizon,
                    "min_return": args.min_return,
                    "size_scale": args.size_scale,
                    "cost_bps": args.cost_bps,
                    "train_split": args.train_split,
                    "pos_weight": args.pos_weight,
                    "signal_weight": args.signal_weight,
                    "size_weight": args.size_weight,
                    "hidden": hidden_dims,
                    "dropout": args.dropout,
                },
            }
            out_path = args.out_dir / f"bagsneural_{args.mint}_best.pt"
            torch.save(checkpoint, out_path)
            with open(args.out_dir / f"bagsneural_{args.mint}_best.json", "w") as f:
                json.dump(checkpoint["config"], f, indent=2)

    logger.info("Best validation loss: %.6f", best_val)


if __name__ == "__main__":
    main()
