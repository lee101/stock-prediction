#!/usr/bin/env python3
"""Fine-tuning script for BagsV3LLM on CODEX data.

Fine-tunes a pre-trained model for CODEX trading signals,
optionally integrating Chronos2 forecasts.
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
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from bagsv3llm.model import BagsV3Config, BagsV3Transformer, FocalLoss
from bagsv3llm.dataset import (
    load_ohlc_dataframe,
    FinetuningDataset,
    fit_normalizer,
    FeatureNormalizerV3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _make_split_indices(
    num_samples: int,
    val_split: float,
    random_split: bool,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create train/val index splits with optional randomization."""
    if num_samples < 2:
        raise ValueError(f"Need at least 2 samples for split, got {num_samples}")
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")

    val_size = int(round(num_samples * val_split))
    if val_size <= 0:
        val_size = 1
    if val_size >= num_samples:
        val_size = num_samples - 1

    if random_split:
        rng = np.random.default_rng(seed)
        indices = np.arange(num_samples)
        rng.shuffle(indices)
        val_indices = indices[:val_size].tolist()
        train_indices = indices[val_size:].tolist()
    else:
        split_idx = num_samples - val_size
        train_indices = list(range(split_idx))
        val_indices = list(range(split_idx, num_samples))

    return train_indices, val_indices


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    signal_loss_fn: nn.Module,
    size_loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[Any] = None,
    signal_weight: float = 1.0,
    size_weight: float = 0.5,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        bar_features = batch["bar_features"].to(device)
        chronos_features = batch["chronos_features"].to(device)
        agg_features = batch["agg_features"].to(device)
        signal_target = batch["signal_target"].to(device)
        size_target = batch["size_target"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                signal_logit, size_logit = model(
                    bar_features, chronos_features, agg_features
                )
                loss_signal = signal_loss_fn(signal_logit, signal_target)
                loss_size = size_loss_fn(torch.sigmoid(size_logit), size_target)
                loss = signal_weight * loss_signal + size_weight * loss_size

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            signal_logit, size_logit = model(
                bar_features, chronos_features, agg_features
            )
            loss_signal = signal_loss_fn(signal_logit, signal_target)
            loss_size = size_loss_fn(torch.sigmoid(size_logit), size_target)
            loss = signal_weight * loss_signal + size_weight * loss_size

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(bar_features)

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

    for batch in loader:
        bar_features = batch["bar_features"].to(device)
        chronos_features = batch["chronos_features"].to(device)
        agg_features = batch["agg_features"].to(device)
        signal_target = batch["signal_target"].to(device)
        size_target = batch["size_target"].to(device)

        signal_logit, size_logit = model(
            bar_features, chronos_features, agg_features
        )

        loss_signal = signal_loss_fn(signal_logit, signal_target)
        loss_size = size_loss_fn(torch.sigmoid(size_logit), size_target)
        loss = signal_weight * loss_signal + size_weight * loss_size
        total_loss += loss.item() * len(bar_features)

        probs = torch.sigmoid(signal_logit)
        all_preds.extend(probs.cpu().numpy())
        all_targets.extend(signal_target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.mean((all_preds > 0.5) == all_targets)

    # AUC
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


def finetune(
    ohlc_path: Path,
    mint: str,
    pretrained_path: Optional[Path] = None,
    # Model config (used if no pretrained model)
    context_length: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
    n_embd: int = 128,
    dropout: float = 0.1,
    # Data config
    horizon: int = 3,
    cost_bps: float = 130.0,
    min_return: float = 0.002,
    size_scale: float = 0.02,
    # Training config
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,  # Lower LR for fine-tuning
    weight_decay: float = 0.01,
    warmup_epochs: int = 5,
    val_split: float = 0.2,
    random_split: bool = False,
    split_seed: int = 42,
    signal_weight: float = 1.0,
    size_weight: float = 0.5,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    # Regularization
    freeze_layers: int = 0,  # Number of transformer layers to freeze
    # Other
    device: str = "cuda",
    out_dir: Path = Path("bagsv3llm/checkpoints"),
    early_stop_patience: int = 15,
    use_amp: bool = True,
    use_chronos: bool = False,  # Whether to compute Chronos features
) -> Dict[str, Any]:
    """Fine-tune BagsV3LLM on CODEX data.

    Returns:
        Dict with training results and checkpoint path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CODEX data
    logger.info(f"Loading CODEX data from {ohlc_path}")
    df = load_ohlc_dataframe(ohlc_path, mint)
    logger.info(f"Loaded {len(df)} OHLC bars for {mint[:8]}...")

    # Initialize Chronos wrapper if requested
    chronos_wrapper = None
    if use_chronos:
        try:
            from src.models.chronos2_wrapper import Chronos2OHLCWrapper
            chronos_wrapper = Chronos2OHLCWrapper.from_pretrained(
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                default_context_length=512,
            )
            logger.info("Initialized Chronos2 wrapper for forecast features")
        except Exception as e:
            logger.warning(f"Failed to initialize Chronos2: {e}")
            chronos_wrapper = None

    # Create dataset
    dataset = FinetuningDataset(
        df=df,
        context_length=context_length,
        horizon=horizon,
        cost_bps=cost_bps,
        min_return=min_return,
        size_scale=size_scale,
        chronos_wrapper=chronos_wrapper,
    )

    logger.info(f"Built {len(dataset)} fine-tuning samples")

    # Split data (time-based by default; random split optional)
    train_indices, val_indices = _make_split_indices(
        num_samples=len(dataset),
        val_split=val_split,
        random_split=random_split,
        seed=split_seed,
    )

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SequentialSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0
    )

    split_mode = "random" if random_split else "time-based"
    logger.info(
        f"Split mode: {split_mode} | Train: {len(train_indices)}, Val: {len(val_indices)}"
    )

    # Load or create model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if pretrained_path and pretrained_path.exists():
        logger.info(f"Loading pre-trained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        # Use config from checkpoint
        if "config" in checkpoint:
            saved_config = checkpoint["config"]
            config = BagsV3Config(
                context_length=saved_config.get("context_length", context_length),
                n_layer=saved_config.get("n_layer", n_layer),
                n_head=saved_config.get("n_head", n_head),
                n_embd=saved_config.get("n_embd", n_embd),
                dropout=dropout,  # Use new dropout
            )
        else:
            config = BagsV3Config(
                context_length=context_length,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                dropout=dropout,
            )

        model = BagsV3Transformer(config).to(device)

        # Load transformer weights from pre-trained model
        if "transformer_state" in checkpoint:
            model.load_state_dict(checkpoint["transformer_state"], strict=False)
            logger.info("Loaded transformer weights from pre-trained checkpoint")
        elif "model_state" in checkpoint:
            # Try to load from full model state
            state = checkpoint["model_state"]
            # Filter for transformer keys
            transformer_state = {
                k.replace("transformer.", ""): v
                for k, v in state.items()
                if k.startswith("transformer.")
            }
            if transformer_state:
                model.load_state_dict(transformer_state, strict=False)
                logger.info("Loaded transformer weights from full model state")

        # Load normalizers if available
        normalizers = checkpoint.get("normalizers", {})
    else:
        logger.info("Training from scratch (no pre-trained model)")
        config = BagsV3Config(
            context_length=context_length,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
        )
        model = BagsV3Transformer(config).to(device)
        normalizers = {}

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Optionally freeze early layers for transfer learning
    if freeze_layers > 0:
        for i, layer in enumerate(model.layers[:freeze_layers]):
            for param in layer.parameters():
                param.requires_grad = False
        logger.info(f"Froze first {freeze_layers} transformer layers")

    # Loss functions
    if use_focal_loss:
        signal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        # Get positive class weight from training data
        pos_rate = dataset[0]["signal_target"].item()  # Approximate
        pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6) if pos_rate > 0 else 1.0
        signal_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )

    size_loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
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
    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, signal_loss_fn, size_loss_fn, device,
            scaler=scaler, scheduler=scheduler,
            signal_weight=signal_weight, size_weight=size_weight,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, signal_loss_fn, size_loss_fn, device,
            signal_weight=signal_weight, size_weight=size_weight,
        )

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        # Save best model (by AUC)
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info(f"  -> New best model (val_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    total_time = time.time() - start_time
    logger.info(f"Fine-tuning complete in {total_time/60:.1f} minutes")

    # Save final checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "normalizers": normalizers,
        "training_config": {
            "mint": mint,
            "horizon": horizon,
            "cost_bps": cost_bps,
            "min_return": min_return,
            "size_scale": size_scale,
        },
        "metrics": {
            "best_val_loss": best_val_loss,
            "best_val_auc": best_val_auc,
        },
        "version": "v3_finetune",
    }

    save_path = out_dir / f"bagsv3_{mint[:8]}_best.pt"
    torch.save(checkpoint, save_path)
    logger.info(f"Saved best model: {save_path}")
    logger.info(f"Best val loss: {best_val_loss:.4f}, Best val AUC: {best_val_auc:.4f}")

    # Cleanup Chronos
    if chronos_wrapper is not None:
        chronos_wrapper.unload()

    return {
        "checkpoint_path": save_path,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
        "num_params": num_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BagsV3LLM on CODEX")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining/ohlc_data.csv"))
    parser.add_argument("--mint", type=str, required=True)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("bagsv3llm/checkpoints"))

    # Model config
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data config
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=130.0)

    # Training config
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--random-split", action="store_true", help="Randomize train/val split")
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomized split")
    parser.add_argument("--early-stop", type=int, default=15)

    # Loss config
    parser.add_argument("--focal-loss", action="store_true", default=True)
    parser.add_argument("--no-focal-loss", action="store_false", dest="focal_loss")
    parser.add_argument("--signal-weight", type=float, default=1.0)
    parser.add_argument("--size-weight", type=float, default=0.5)

    # Transfer learning
    parser.add_argument("--freeze-layers", type=int, default=0)
    parser.add_argument("--use-chronos", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    result = finetune(
        ohlc_path=args.ohlc,
        mint=args.mint,
        pretrained_path=args.pretrained,
        context_length=args.context_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        val_split=args.val_split,
        random_split=args.random_split,
        split_seed=args.seed,
        signal_weight=args.signal_weight,
        size_weight=args.size_weight,
        use_focal_loss=args.focal_loss,
        freeze_layers=args.freeze_layers,
        use_chronos=args.use_chronos,
        device=args.device,
        out_dir=args.out_dir,
        early_stop_patience=args.early_stop,
        use_amp=not args.no_amp,
    )

    print(f"\nFine-tuning complete!")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Best val AUC: {result['best_val_auc']:.4f}")
    print(f"Model params: {result['num_params']:,}")


if __name__ == "__main__":
    main()
