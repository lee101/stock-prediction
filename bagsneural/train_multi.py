#!/usr/bin/env python3
"""Train a neural model on multiple Bags.fm tokens."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bagsfm.config import CostConfig
from bagsneural.dataset import (
    build_features_and_targets,
    fit_normalizer,
    load_ohlc_dataframe,
)
from bagsneural.model import BagsNeuralModel

logger = logging.getLogger("bagsneural_train_multi")


def build_multi_token_dataset(
    dataframes: Dict[str, np.ndarray],
    context_bars: int,
    horizon: int,
    cost_bps: float,
    min_return: float,
    size_scale: float,
    val_split: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")

    train_parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    val_parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for mint, df in dataframes.items():
        features, signal_targets, size_targets, _ = build_features_and_targets(
            df,
            context_bars=context_bars,
            horizon=horizon,
            cost_bps=cost_bps,
            min_return=min_return,
            size_scale=size_scale,
        )

        split_idx = int(len(features) * (1 - val_split))
        if split_idx <= 0 or split_idx >= len(features):
            logger.warning("Skipping %s: not enough samples", mint)
            continue

        train_parts.append(
            (
                features[:split_idx],
                signal_targets[:split_idx],
                size_targets[:split_idx],
            )
        )
        val_parts.append(
            (
                features[split_idx:],
                signal_targets[split_idx:],
                size_targets[split_idx:],
            )
        )

    if not train_parts or not val_parts:
        raise ValueError("No tokens produced train/val data")

    train_features = np.concatenate([p[0] for p in train_parts], axis=0)
    train_signal = np.concatenate([p[1] for p in train_parts], axis=0)
    train_size = np.concatenate([p[2] for p in train_parts], axis=0)

    val_features = np.concatenate([p[0] for p in val_parts], axis=0)
    val_signal = np.concatenate([p[1] for p in val_parts], axis=0)
    val_size = np.concatenate([p[2] for p in val_parts], axis=0)

    return (train_features, train_signal, train_size), (val_features, val_signal, val_size)


def _make_loader(features: np.ndarray, signal: np.ndarray, size: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(signal, dtype=torch.float32),
        torch.tensor(size, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bags.fm neural model on multiple tokens")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining") / "ohlc_data.csv")
    parser.add_argument("--mints", type=str, required=True, help="Comma-separated token mints")
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

    mint_list = [m.strip() for m in args.mints.split(",") if m.strip()]
    if not mint_list:
        raise ValueError("No mints provided")

    if args.cost_bps is None:
        costs = CostConfig()
        args.cost_bps = costs.estimated_swap_fee_bps + costs.default_slippage_bps

    dataframes: Dict[str, np.ndarray] = {}
    for mint in mint_list:
        try:
            df = load_ohlc_dataframe(args.ohlc, mint, dedupe=True)
        except Exception as exc:
            logger.warning("Skipping %s: %s", mint, exc)
            continue
        if args.train_split < 1.0:
            if not 0.0 < args.train_split < 1.0:
                raise ValueError("train-split must be between 0 and 1")
            train_rows = int(len(df) * args.train_split)
            df = df.iloc[:train_rows].reset_index(drop=True)
        dataframes[mint] = df

    if not dataframes:
        raise ValueError("No token data available for training")

    train, val = build_multi_token_dataset(
        dataframes,
        context_bars=args.context,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        min_return=args.min_return,
        size_scale=args.size_scale,
        val_split=args.val_split,
    )

    normalizer = fit_normalizer(train[0])
    train_features = normalizer.transform(train[0])
    val_features = normalizer.transform(val[0])

    train_loader = _make_loader(train_features, train[1], train[2], args.batch_size)
    val_loader = _make_loader(val_features, val[1], val[2], args.batch_size)

    hidden_dims = [int(x) for x in args.hidden.split(",") if x.strip()]

    model = BagsNeuralModel(
        input_dim=train_features.shape[1],
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.pos_weight is None:
        positives = float(train[1].sum())
        negatives = float(len(train[1]) - positives)
        pos_weight = negatives / max(positives, 1.0)
        args.pos_weight = min(pos_weight, 50.0)

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
            safe_name = "_".join(mint_list)[:64]
            checkpoint = {
                "model_state": model.state_dict(),
                "normalizer": normalizer.to_dict(),
                "config": {
                    "mints": mint_list,
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
            out_path = args.out_dir / f"bagsneural_multi_{safe_name}_best.pt"
            torch.save(checkpoint, out_path)
            with open(args.out_dir / f"bagsneural_multi_{safe_name}_best.json", "w") as f:
                json.dump(checkpoint["config"], f, indent=2)

    logger.info("Best validation loss: %.6f", best_val)


if __name__ == "__main__":
    main()
