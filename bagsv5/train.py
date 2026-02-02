"""Training script for BagsV5 with Muon optimizer.

Training pipeline:
1. Pre-train on multiple BAGS tokens (ELIZA, GAS, VIBE, GROK, RO, ISONYC)
2. Fine-tune on CODEX
3. Validate on held-out CODEX data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

from bagsv5.model import BagsV5Config, BagsV5Transformer, FocalLoss
from bagsv5.dataset import (
    MultiTokenDataset,
    SingleTokenDataset,
    FeatureNormalizer,
    load_multi_token_data,
    load_ohlc_dataframe,
    build_bar_features,
    build_aggregate_features,
)

logger = logging.getLogger(__name__)


def create_optimizers(
    model: BagsV5Transformer,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    muon_momentum: float = 0.95,
) -> Tuple[Muon, AdamW]:
    """Create Muon optimizer for weights and AdamW for other params."""

    weight_params = model.get_weight_params()
    other_params = model.get_other_params()

    logger.info(f"Muon params: {sum(p.numel() for p in weight_params):,}")
    logger.info(f"AdamW params: {sum(p.numel() for p in other_params):,}")

    # Muon for 2D weight matrices (hidden layers)
    muon_optimizer = Muon(
        weight_params,
        lr=lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
    )

    # AdamW for embeddings, biases, heads
    adamw_optimizer = AdamW(
        other_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    return muon_optimizer, adamw_optimizer


def train_epoch(
    model: BagsV5Transformer,
    dataloader: DataLoader,
    muon_opt: Muon,
    adamw_opt: AdamW,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        bar_features = batch['bar_features'].to(device)
        agg_features = batch['agg_features'].to(device)
        signal_target = batch['signal_target'].to(device)

        # Check for NaN in input
        if torch.isnan(bar_features).any() or torch.isnan(agg_features).any():
            continue

        muon_opt.zero_grad()
        adamw_opt.zero_grad()

        with autocast(device_type='cuda', enabled=scaler is not None):
            signal_logit, _ = model(bar_features, agg_features)
            loss = criterion(signal_logit, signal_target)

        # Skip NaN losses
        if torch.isnan(loss):
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(muon_opt)
            scaler.unscale_(adamw_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(muon_opt)
            scaler.step(adamw_opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            muon_opt.step()
            adamw_opt.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: BagsV5Transformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        bar_features = batch['bar_features'].to(device)
        agg_features = batch['agg_features'].to(device)
        signal_target = batch['signal_target'].to(device)

        signal_logit, _ = model(bar_features, agg_features)
        loss = criterion(signal_logit, signal_target)

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(signal_logit).cpu().numpy())
        all_targets.extend(signal_target.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    acc = ((preds > 0.5) == targets).mean()

    try:
        auc = roc_auc_score(targets, preds)
    except:
        auc = 0.5

    return avg_loss, acc, auc


def pretrain_multi_token(
    ohlc_path: Path,
    exclude_mint: str,  # CODEX mint to exclude
    config: BagsV5Config,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    min_rows_per_token: int = 50,
    val_split: float = 0.2,
    device: str = "cuda",
    out_dir: Path = Path("bagsv5/checkpoints"),
) -> Dict[str, Any]:
    """Pre-train on multiple BAGS tokens (excluding CODEX)."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load multi-token data
    token_dfs = load_multi_token_data(
        ohlc_path,
        exclude_mints=[exclude_mint],
        min_rows=min_rows_per_token,
    )
    logger.info(f"Loaded {len(token_dfs)} tokens for pre-training: {list(token_dfs.keys())}")

    if not token_dfs:
        logger.warning("No tokens found for pre-training!")
        return {'pretrain_skipped': True}

    # Create dataset
    dataset = MultiTokenDataset(
        token_dfs,
        context_length=config.context_length,
    )

    if len(dataset) < 50:
        logger.warning(f"Too few samples ({len(dataset)}) for pre-training")
        return {'pretrain_skipped': True}

    # Split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Pre-train: {train_size}, Val: {val_size}")

    # Model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = BagsV5Transformer(config).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizers
    muon_opt, adamw_opt = create_optimizers(model, lr=lr)

    # Scheduler with warmup
    warmup_epochs = min(3, epochs // 5)
    warmup_scheduler = LinearLR(muon_opt, start_factor=0.1, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(muon_opt, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(muon_opt, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scaler = GradScaler() if device.type == "cuda" else None

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, muon_opt, adamw_opt, criterion, device, scaler)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Pretrain Epoch {epoch+1}/{epochs} - train_loss={train_loss:.4f} | "
                   f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            logger.info(f"  -> New best pretrain model (auc={val_auc:.4f})")

    # Save
    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = out_dir / "pretrain_best.pt"
        torch.save({
            'model_state': best_state,
            'config': config.__dict__,
            'best_auc': best_auc,
        }, ckpt_path)
        logger.info(f"Saved pretrain model: {ckpt_path}")

    return {
        'best_auc': best_auc,
        'model': model,
        'config': config,
    }


def finetune_codex(
    ohlc_path: Path,
    mint: str,
    model: Optional[BagsV5Transformer] = None,
    config: Optional[BagsV5Config] = None,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 5e-4,  # Lower LR for fine-tuning
    val_split: float = 0.2,
    early_stop: int = 10,
    device: str = "cuda",
    out_dir: Path = Path("bagsv5/checkpoints"),
) -> Dict[str, Any]:
    """Fine-tune on CODEX data."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CODEX data
    df = load_ohlc_dataframe(ohlc_path, mint)
    logger.info(f"Loaded {len(df)} CODEX bars")

    # Use last portion for validation (temporal split)
    val_start = int(len(df) * (1 - val_split))

    train_dataset = SingleTokenDataset(
        df,
        context_length=config.context_length,
        end_idx=val_start,
    )
    val_dataset = SingleTokenDataset(
        df,
        context_length=config.context_length,
        start_idx=val_start,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Fine-tune train: {len(train_dataset)}, val: {len(val_dataset)}")

    # Model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if model is None:
        model = BagsV5Transformer(config).to(device)
    else:
        model = model.to(device)

    # Optimizers (lower LR for fine-tuning)
    muon_opt, adamw_opt = create_optimizers(model, lr=lr)

    # Cosine scheduler
    scheduler = CosineAnnealingLR(muon_opt, T_max=epochs)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scaler = GradScaler() if device.type == "cuda" else None

    best_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, muon_opt, adamw_opt, criterion, device, scaler)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Finetune Epoch {epoch+1}/{epochs} - train_loss={train_loss:.4f} | "
                   f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            no_improve = 0
            logger.info(f"  -> New best model (auc={val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Build normalizers
    normalizers = {}

    # Fit bar normalizer
    bar_norm = FeatureNormalizer()
    all_bar_features = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        all_bar_features.append(sample['bar_features'].numpy())
    bar_norm.fit(np.concatenate(all_bar_features).reshape(-1, 5))
    normalizers['bar'] = bar_norm

    # Fit agg normalizer
    agg_norm = FeatureNormalizer()
    all_agg_features = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        all_agg_features.append(sample['agg_features'].numpy())
    agg_norm.fit(np.stack(all_agg_features))
    normalizers['agg'] = agg_norm

    # Save
    model.load_state_dict(best_state)
    ckpt_path = out_dir / f"bagsv5_{mint[:8]}_best.pt"
    torch.save({
        'model_state': best_state,
        'config': config.__dict__,
        'normalizers': {k: v.to_dict() for k, v in normalizers.items()},
        'best_auc': best_auc,
    }, ckpt_path)
    logger.info(f"Saved fine-tuned model: {ckpt_path}")

    return {
        'best_auc': best_auc,
        'checkpoint_path': str(ckpt_path),
        'model': model,
        'config': config,
        'normalizers': normalizers,
    }


def train_v5(
    ohlc_path: Path = Path("bagstraining/ohlc_data.csv"),
    codex_mint: str = "HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS",
    context_length: int = 96,
    n_layer: int = 4,
    n_head: int = 8,
    n_embd: int = 64,
    pretrain_epochs: int = 30,
    finetune_epochs: int = 50,
    batch_size: int = 16,
    pretrain_lr: float = 1e-3,
    finetune_lr: float = 5e-4,
    pretrain_min_rows: int = 50,
    device: str = "cuda",
    out_dir: Path = Path("bagsv5/checkpoints"),
) -> Dict[str, Any]:
    """Full V5 training pipeline."""

    config = BagsV5Config(
        context_length=context_length,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )

    # Step 1: Pre-train on other BAGS tokens
    logger.info("=" * 60)
    logger.info("Step 1: Pre-training on BAGS tokens (excluding CODEX)")
    logger.info("=" * 60)

    pretrain_result = pretrain_multi_token(
        ohlc_path=ohlc_path,
        exclude_mint=codex_mint,
        config=config,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        lr=pretrain_lr,
        min_rows_per_token=pretrain_min_rows,
        device=device,
        out_dir=out_dir,
    )

    model = pretrain_result.get('model')

    # Step 2: Fine-tune on CODEX
    logger.info("=" * 60)
    logger.info("Step 2: Fine-tuning on CODEX")
    logger.info("=" * 60)

    finetune_result = finetune_codex(
        ohlc_path=ohlc_path,
        mint=codex_mint,
        model=model,
        config=config,
        epochs=finetune_epochs,
        batch_size=batch_size,
        lr=finetune_lr,
        device=device,
        out_dir=out_dir,
    )

    return {
        'pretrain_auc': pretrain_result.get('best_auc', 0),
        'finetune_auc': finetune_result['best_auc'],
        'checkpoint_path': finetune_result['checkpoint_path'],
    }


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train BagsV5")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining/ohlc_data.csv"))
    parser.add_argument("--mint", type=str, default="HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrain-min-rows", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("bagsv5/checkpoints"))

    args = parser.parse_args()

    result = train_v5(
        ohlc_path=args.ohlc,
        codex_mint=args.mint,
        context_length=args.context_length,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        pretrain_min_rows=args.pretrain_min_rows,
        device=args.device,
        out_dir=args.out_dir,
    )

    print(f"\nTraining complete!")
    print(f"Pre-train AUC: {result['pretrain_auc']:.4f}")
    print(f"Fine-tune AUC: {result['finetune_auc']:.4f}")
    print(f"Checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
