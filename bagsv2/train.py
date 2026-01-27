#!/usr/bin/env python3
"""Training script for Bags v2 neural model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bagsv2.dataset import (
    load_ohlc_dataframe,
    build_features_and_targets,
    fit_normalizer,
    FeatureNormalizerV2,
    expanding_window_split,
)
from bagsv2.model import BagsNeuralModelV2, BagsNeuralModelV2Simple, FocalLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    signal_loss_fn: nn.Module,
    size_loss_fn: nn.Module,
    device: torch.device,
    signal_weight: float = 1.0,
    size_weight: float = 0.5,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_x, batch_signal, batch_size in loader:
        batch_x = batch_x.to(device)
        batch_signal = batch_signal.to(device)
        batch_size = batch_size.to(device)

        optimizer.zero_grad()

        signal_logit, size_logit = model(batch_x)

        loss_signal = signal_loss_fn(signal_logit, batch_signal)
        loss_size = size_loss_fn(torch.sigmoid(size_logit), batch_size)
        loss = signal_weight * loss_signal + size_weight * loss_size

        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item() * len(batch_x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    signal_loss_fn: nn.Module,
    size_loss_fn: nn.Module,
    device: torch.device,
    signal_weight: float = 1.0,
    size_weight: float = 0.5,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_x, batch_signal, batch_size in loader:
        batch_x = batch_x.to(device)
        batch_signal = batch_signal.to(device)
        batch_size = batch_size.to(device)

        signal_logit, size_logit = model(batch_x)

        loss_signal = signal_loss_fn(signal_logit, batch_signal)
        loss_size = size_loss_fn(torch.sigmoid(size_logit), batch_size)
        loss = signal_weight * loss_signal + size_weight * loss_size
        total_loss += loss.item() * len(batch_x)

        # Collect predictions for metrics
        probs = torch.sigmoid(signal_logit)
        all_preds.extend(probs.cpu().numpy())
        all_targets.extend(batch_signal.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    avg_loss = total_loss / len(loader.dataset)

    # Accuracy at 0.5 threshold
    accuracy = np.mean((all_preds > 0.5) == all_targets)

    # AUC (simple approximation)
    pos_preds = all_preds[all_targets == 1]
    neg_preds = all_preds[all_targets == 0]
    if len(pos_preds) > 0 and len(neg_preds) > 0:
        auc = np.mean(pos_preds[:, None] > neg_preds[None, :])
    else:
        auc = 0.5

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
        "pos_rate": all_targets.mean(),
    }


def train_model(
    ohlc_path: Path,
    mint: str,
    # Model params
    model_type: str = "lstm",  # "lstm" or "mlp"
    context_bars: int = 32,
    lstm_hidden: int = 64,
    lstm_layers: int = 2,
    fc_hidden: int = 64,
    mlp_hidden: tuple = (128, 64, 32),
    dropout: float = 0.2,
    # Data params
    horizon: int = 3,
    cost_bps: float = 130.0,
    min_return: float = 0.002,
    size_scale: float = 0.02,
    # Training params
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    signal_weight: float = 1.0,
    size_weight: float = 0.5,
    # Other
    device: str = "cuda",
    out_dir: Path = None,
    early_stop_patience: int = 10,
) -> Dict[str, Any]:
    """Train the v2 model.

    Returns:
        Dict with training results and checkpoint path.
    """
    if out_dir is None:
        out_dir = Path("bagsv2/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_ohlc_dataframe(ohlc_path, mint)
    logger.info(f"Loaded {len(df)} OHLC bars for {mint[:8]}...")

    features, signal_targets, size_targets, timestamps = build_features_and_targets(
        df=df,
        context_bars=context_bars,
        horizon=horizon,
        cost_bps=cost_bps,
        min_return=min_return,
        size_scale=size_scale,
    )

    logger.info(f"Built {len(features)} samples with {features.shape[1]} features")
    logger.info(f"Positive rate: {signal_targets.mean():.4f}")

    # Simple train/val split (time-based)
    split_idx = int(len(features) * (1 - val_split))
    train_features = features[:split_idx]
    train_signals = signal_targets[:split_idx]
    train_sizes = size_targets[:split_idx]

    val_features = features[split_idx:]
    val_signals = signal_targets[split_idx:]
    val_sizes = size_targets[split_idx:]

    logger.info(f"Train: {len(train_features)}, Val: {len(val_features)}")

    # Fit normalizer on training data only
    normalizer = fit_normalizer(train_features)
    train_features = normalizer.transform(train_features)
    val_features = normalizer.transform(val_features)

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
    logger.info(f"Using device: {device}")

    features_per_bar = 5  # returns, range_pct, oc_return, upper_wick, lower_wick

    if model_type == "lstm":
        model = BagsNeuralModelV2(
            features_per_bar=features_per_bar,
            context_bars=context_bars,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            fc_hidden=fc_hidden,
            dropout=dropout,
        ).to(device)
    else:
        model = BagsNeuralModelV2Simple(
            input_dim=features.shape[1],
            hidden_dims=mlp_hidden,
            dropout=dropout,
        ).to(device)

    logger.info(f"Model type: {model_type}, params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions
    if use_focal_loss:
        signal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        pos_weight = (1 - train_signals.mean()) / max(train_signals.mean(), 1e-6)
        signal_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    size_loss_fn = nn.MSELoss()

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
    )

    # Training loop
    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = 0.0
        model.train()
        for batch_x, batch_signal, batch_size_t in train_loader:
            batch_x = batch_x.to(device)
            batch_signal = batch_signal.to(device)
            batch_size_t = batch_size_t.to(device)

            optimizer.zero_grad()
            signal_logit, size_logit = model(batch_x)

            loss_signal = signal_loss_fn(signal_logit, batch_signal)
            loss_size = size_loss_fn(torch.sigmoid(size_logit), batch_size_t)
            loss = signal_weight * loss_signal + size_weight * loss_size

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(batch_x)

        train_loss /= len(train_dataset)

        # Validate
        val_metrics = validate(
            model, val_loader, signal_loss_fn, size_loss_fn, device,
            signal_weight, size_weight
        )

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        # Save best model (by AUC, more meaningful than loss)
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_val_loss = val_metrics['loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Save checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "config": {
            "model_type": model_type,
            "context_bars": context_bars,
            "features_per_bar": features_per_bar,
            "horizon": horizon,
            "cost_bps": cost_bps,
            "min_return": min_return,
            "size_scale": size_scale,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "fc_hidden": fc_hidden,
            "mlp_hidden": list(mlp_hidden) if model_type == "mlp" else None,
            "dropout": dropout,
            "version": "v2",
        },
        "metrics": {
            "best_val_loss": best_val_loss,
            "best_val_auc": best_val_auc,
        },
    }

    save_path = out_dir / f"bagsv2_{mint[:8]}_best.pt"
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")
    logger.info(f"Best val loss: {best_val_loss:.4f}, Best val AUC: {best_val_auc:.4f}")

    return {
        "checkpoint_path": save_path,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
        "input_dim": features.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Train Bags v2 neural model")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining/ohlc_data.csv"))
    parser.add_argument("--mint", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="lstm", choices=["lstm", "mlp"])
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--fc-hidden", type=int, default=64)
    parser.add_argument("--mlp-hidden", type=str, default="128,64,32")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--focal-loss", action="store_true", default=True)
    parser.add_argument("--no-focal-loss", action="store_false", dest="focal_loss")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("bagsv2/checkpoints"))

    args = parser.parse_args()
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(","))

    result = train_model(
        ohlc_path=args.ohlc,
        mint=args.mint,
        model_type=args.model_type,
        context_bars=args.context,
        horizon=args.horizon,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        fc_hidden=args.fc_hidden,
        mlp_hidden=mlp_hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        use_focal_loss=args.focal_loss,
        device=args.device,
        out_dir=args.out_dir,
    )

    print(f"\nTraining complete!")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Best val AUC: {result['best_val_auc']:.4f}")


if __name__ == "__main__":
    main()
