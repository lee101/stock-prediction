#!/usr/bin/env python3
"""Pre-training script for BagsV3LLM on stock/crypto data.

Uses masked reconstruction objective to learn price patterns from
diverse stock data before fine-tuning on CODEX.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from bagsv3llm.model import (
    BagsV3Config,
    BagsV3PretrainModel,
    PretrainLoss,
)
from bagsv3llm.dataset import (
    load_pretraining_data,
    PretrainingDataset,
    fit_normalizer,
    create_dataloaders,
    FeatureNormalizerV3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: PretrainLoss,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[Any] = None,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_metrics = {}
    num_batches = 0

    for batch in loader:
        bar_features = batch["bar_features"].to(device)
        chronos_features = batch["chronos_features"].to(device)
        agg_features = batch["agg_features"].to(device)
        signal_target = batch["signal_target"].to(device)
        size_target = batch["size_target"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                reconstructed, signal_logit, size_logit = model(
                    bar_features, chronos_features, agg_features, mask
                )
                loss, metrics = loss_fn(
                    reconstructed=reconstructed,
                    original=bar_features,
                    mask=mask,
                    signal_logit=signal_logit,
                    signal_target=signal_target,
                    size_logit=size_logit,
                    size_target=size_target,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, signal_logit, size_logit = model(
                bar_features, chronos_features, agg_features, mask
            )
            loss, metrics = loss_fn(
                reconstructed=reconstructed,
                original=bar_features,
                mask=mask,
                signal_logit=signal_logit,
                signal_target=signal_target,
                size_logit=size_logit,
                size_target=size_target,
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        for key, value in metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value
        num_batches += 1

    # Average metrics
    return {key: value / num_batches for key, value in total_metrics.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: PretrainLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_metrics = {}
    all_preds = []
    all_targets = []
    num_batches = 0

    for batch in loader:
        bar_features = batch["bar_features"].to(device)
        chronos_features = batch["chronos_features"].to(device)
        agg_features = batch["agg_features"].to(device)
        signal_target = batch["signal_target"].to(device)
        size_target = batch["size_target"].to(device)
        mask = batch["mask"].to(device)

        reconstructed, signal_logit, size_logit = model(
            bar_features, chronos_features, agg_features, mask
        )

        _, metrics = loss_fn(
            reconstructed=reconstructed,
            original=bar_features,
            mask=mask,
            signal_logit=signal_logit,
            signal_target=signal_target,
            size_logit=size_logit,
            size_target=size_target,
        )

        # Collect predictions for AUC
        probs = torch.sigmoid(signal_logit)
        all_preds.extend(probs.cpu().numpy())
        all_targets.extend(signal_target.cpu().numpy())

        for key, value in metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value
        num_batches += 1

    # Average metrics
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    # Compute AUC
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    pos_preds = all_preds[all_targets == 1]
    neg_preds = all_preds[all_targets == 0]
    if len(pos_preds) > 0 and len(neg_preds) > 0:
        auc = np.mean(pos_preds[:, None] > neg_preds[None, :])
    else:
        auc = 0.5

    avg_metrics["auc"] = auc
    avg_metrics["accuracy"] = np.mean((all_preds > 0.5) == all_targets)

    return avg_metrics


def pretrain(
    data_dir: Path = Path("trainingdata"),
    # Model config
    context_length: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
    n_embd: int = 128,
    dropout: float = 0.1,
    # Data config
    min_rows_per_symbol: int = 300,
    max_symbols: Optional[int] = None,
    horizon: int = 3,
    mask_ratio: float = 0.15,
    # Training config
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_grad_norm: float = 1.0,
    val_split: float = 0.2,
    # Loss weights
    recon_weight: float = 1.0,
    signal_weight: float = 0.1,
    size_weight: float = 0.05,
    # Other
    device: str = "cuda",
    out_dir: Path = Path("bagsv3llm/checkpoints"),
    save_every: int = 5,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """Pre-train BagsV3LLM on stock data.

    Returns:
        Dict with training results and checkpoint path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading pre-training data from {data_dir}")
    df = load_pretraining_data(
        data_dir=data_dir,
        min_rows_per_symbol=min_rows_per_symbol,
        max_symbols=max_symbols,
    )

    # Create dataset
    dataset = PretrainingDataset(
        df=df,
        context_length=context_length,
        horizon=horizon,
        mask_ratio=mask_ratio,
    )

    # Fit normalizers
    logger.info("Fitting normalizers...")
    bar_normalizer, chronos_normalizer, agg_normalizer = fit_normalizer(dataset)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        train_split=1.0 - val_split,
        num_workers=4,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    config = BagsV3Config(
        context_length=context_length,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        pretrain_mask_ratio=mask_ratio,
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = BagsV3PretrainModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Loss function
    loss_fn = PretrainLoss(
        recon_weight=recon_weight,
        signal_weight=signal_weight,
        size_weight=size_weight,
    )

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    # Mixed precision
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scaler=scaler, scheduler=scheduler, max_grad_norm=max_grad_norm,
        )

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
            f"train_loss={train_metrics['total_loss']:.4f} "
            f"train_recon={train_metrics['recon_loss']:.4f} | "
            f"val_loss={val_metrics['total_loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        # Save best model
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  -> New best model (val_loss={best_val_loss:.4f})")

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config.__dict__,
                "normalizers": {
                    "bar": bar_normalizer.to_dict(),
                    "chronos": chronos_normalizer.to_dict(),
                    "agg": agg_normalizer.to_dict(),
                },
                "metrics": val_metrics,
            }
            save_path = out_dir / f"pretrain_epoch{epoch+1}.pt"
            torch.save(checkpoint, save_path)
            logger.info(f"  -> Saved checkpoint: {save_path}")

    total_time = time.time() - start_time
    logger.info(f"Pre-training complete in {total_time/60:.1f} minutes")

    # Save final best model
    if best_state is not None:
        model.load_state_dict(best_state)

    final_checkpoint = {
        "model_state": model.state_dict(),
        "transformer_state": model.transformer.state_dict(),  # For fine-tuning
        "config": config.__dict__,
        "normalizers": {
            "bar": bar_normalizer.to_dict(),
            "chronos": chronos_normalizer.to_dict(),
            "agg": agg_normalizer.to_dict(),
        },
        "metrics": {
            "best_val_loss": best_val_loss,
            "final_epoch": epochs,
        },
        "version": "v3_pretrain",
    }

    save_path = out_dir / "pretrain_best.pt"
    torch.save(final_checkpoint, save_path)
    logger.info(f"Saved best pre-trained model: {save_path}")

    return {
        "checkpoint_path": save_path,
        "best_val_loss": best_val_loss,
        "num_params": num_params,
        "total_samples": len(dataset),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-train BagsV3LLM")
    parser.add_argument("--data-dir", type=Path, default=Path("trainingdata"))
    parser.add_argument("--out-dir", type=Path, default=Path("bagsv3llm/checkpoints"))

    # Model config
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data config
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.15)

    # Training config
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--val-split", type=float, default=0.2)

    # Loss weights
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--signal-weight", type=float, default=0.1)
    parser.add_argument("--size-weight", type=float, default=0.05)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)

    args = parser.parse_args()

    result = pretrain(
        data_dir=args.data_dir,
        context_length=args.context_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        min_rows_per_symbol=args.min_rows,
        max_symbols=args.max_symbols,
        horizon=args.horizon,
        mask_ratio=args.mask_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        val_split=args.val_split,
        recon_weight=args.recon_weight,
        signal_weight=args.signal_weight,
        size_weight=args.size_weight,
        device=args.device,
        out_dir=args.out_dir,
        save_every=args.save_every,
        use_amp=not args.no_amp,
    )

    print(f"\nPre-training complete!")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Best val loss: {result['best_val_loss']:.4f}")
    print(f"Model params: {result['num_params']:,}")
    print(f"Total samples: {result['total_samples']:,}")


if __name__ == "__main__":
    main()
