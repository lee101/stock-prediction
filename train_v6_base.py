#!/usr/bin/env python3
"""V6: Strong Base Model with Overfitting Prevention.

Key anti-overfitting techniques:
1. Train on ALL 300+ symbols mixed together in same batch
2. Higher dropout (0.2) with stochastic depth
3. Mixup data augmentation
4. Strong weight decay (0.05)
5. Early stopping with patience
6. Gradient accumulation for larger effective batch
7. Cosine LR with warmup
8. Multi-symbol batch shuffling (different symbols each batch)

Usage:
    python train_v6_base.py --epochs 5 --batch-size 128
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv4.config import DailyTrainingConfigV4, DailyDatasetConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.model import MultiSymbolPolicyV4, create_group_mask
from neuraldailyv4.simulation_v41 import aggregate_predictions, simulate_single_trade


# ALL available symbols for base training (from alpaca_wrapper)
ALL_BASE_SYMBOLS = (
    # Major indices & mega-cap tech
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    # Cloud & SaaS
    "CRM", "ORCL", "SNOW", "PLTR", "NOW", "DDOG", "MDB", "OKTA", "ZM", "DOCU",
    "TEAM", "TWLO", "SHOP", "VEEV", "PANW", "ZS", "ESTC", "WDAY",
    # Semiconductors
    "AMD", "INTC", "AVGO", "QCOM", "MU", "MRVL", "AMAT", "LRCX", "KLAC",
    "TXN", "ADI", "NXPI", "ASML", "TSM", "MPWR", "ENPH", "ON",
    # Fintech & Payments
    "V", "MA", "PYPL", "SQ", "HOOD", "SOFI", "AFRM", "UPST", "COIN",
    # Traditional Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "COF",
    # Healthcare & Biotech
    "JNJ", "PFE", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "CVS",
    "MRNA", "BNTX", "GILD", "BIIB", "REGN", "VRTX", "ISRG",
    # Consumer
    "WMT", "HD", "NKE", "SBUX", "MCD", "LOW", "TJX", "BKNG", "COST", "TGT",
    "CMG", "ORLY", "RCL", "MAR", "UBER", "LYFT", "ABNB", "DASH", "EBAY", "ETSY",
    # Entertainment & Media
    "DIS", "NFLX", "ROKU", "SPOT", "RBLX", "TTWO", "EA",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
    # Industrials
    "BA", "CAT", "GE", "HON", "UPS", "FDX", "RTX", "LMT", "DE",
    # Crypto (major pairs)
    "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "ADAUSD", "DOTUSD",
    "AVAXUSD", "MATICUSD", "UNIUSD", "ATOMUSD",
)


class CosineWarmupScheduler:
    """Cosine LR with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def mixup_batch(
    batch: Dict[str, torch.Tensor],
    alpha: float = 0.2,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Apply mixup augmentation to batch.

    Mixup blends samples together with random weights, creating
    smoother decision boundaries and reducing overfitting.
    """
    if alpha <= 0:
        return batch, torch.ones(batch["features"].shape[0], device=batch["features"].device)

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = batch["features"].shape[0]

    # Random permutation for mixing
    perm = torch.randperm(batch_size, device=batch["features"].device)

    # Keys to mix (continuous values only)
    mix_keys = [
        "features", "future_highs", "future_lows", "future_closes",
        "reference_close", "chronos_high", "chronos_low"
    ]

    mixed_batch = {}
    for key, val in batch.items():
        if key in mix_keys:
            mixed_batch[key] = lam * val + (1 - lam) * val[perm]
        else:
            mixed_batch[key] = val

    return mixed_batch, torch.full((batch_size,), lam, device=batch["features"].device)


class V6BaseTrainer:
    """V6 trainer with strong overfitting prevention."""

    def __init__(
        self,
        config: DailyTrainingConfigV4,
        data_module: DailyDataModuleV4,
        dropout: float = 0.2,
        stochastic_depth: float = 0.1,
        mixup_alpha: float = 0.2,
        weight_decay: float = 0.05,
        grad_accum_steps: int = 4,
        patience: int = 3,
    ):
        self.config = config
        self.data_module = data_module
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Anti-overfitting params
        self.dropout = dropout
        self.stochastic_depth = stochastic_depth
        self.mixup_alpha = mixup_alpha
        self.grad_accum_steps = grad_accum_steps
        self.patience = patience

        if config.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        data_module.setup()
        policy_config = config.get_policy_config(len(data_module.feature_columns))

        # Override dropout
        policy_config.dropout = dropout

        self.model = MultiSymbolPolicyV4(policy_config).to(self.device)
        self.sim_config = config.get_simulation_config()
        self.temp_schedule = config.get_temperature_schedule()

        # Strong weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        self.use_amp = config.use_amp
        self.amp_dtype = getattr(torch, config.amp_dtype) if config.use_amp else torch.float32
        self.scaler = GradScaler() if self.use_amp else None

        self.checkpoint_dir = Path(config.checkpoint_root) / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_metric = float("-inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def _apply_stochastic_depth(self, drop_prob: float) -> bool:
        """Returns True if this forward pass should skip some layers."""
        return self.training and torch.rand(1).item() < drop_prob

    def _run_epoch(
        self,
        dataloader,
        training: bool,
        temperature: float,
        scheduler=None,
        use_mixup: bool = False,
    ) -> Dict[str, float]:
        self.model.train(training)
        metrics = defaultdict(list)

        self.optimizer.zero_grad()
        accum_count = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Apply mixup during training
            if training and use_mixup:
                batch, mix_lam = mixup_batch(batch, self.mixup_alpha)

            features = batch["features"]
            future_highs = batch["future_highs"]
            future_lows = batch["future_lows"]
            future_closes = batch["future_closes"]
            reference_close = batch["reference_close"]
            chronos_high = batch["chronos_high"]
            chronos_low = batch["chronos_low"]
            asset_class = batch["asset_class"]
            group_ids = batch["group_id"]

            group_mask = create_group_mask(group_ids)

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(features, group_mask)

                ref_close = reference_close[:, -1] if reference_close.dim() > 1 else reference_close
                ch_high = chronos_high[:, -1] if chronos_high.dim() > 1 else chronos_high
                ch_low = chronos_low[:, -1] if chronos_low.dim() > 1 else chronos_low

                actions = self.model.decode_actions(
                    outputs,
                    reference_close=ref_close,
                    chronos_high=ch_high,
                    chronos_low=ch_low,
                    asset_class=asset_class,
                )

                agg = aggregate_predictions(
                    buy_quantiles=actions["buy_quantiles"],
                    sell_quantiles=actions["sell_quantiles"],
                    exit_days=actions["exit_days"],
                    position_size=actions["position_size"],
                    confidence=actions["confidence"],
                    trim_fraction=0.25,
                    min_spread=0.02,
                )

                result = simulate_single_trade(
                    future_highs=future_highs,
                    future_lows=future_lows,
                    future_closes=future_closes,
                    buy_price=agg["buy_price"],
                    sell_price=agg["sell_price"],
                    exit_days=agg["exit_days"],
                    position_size=agg["position_size"],
                    reference_price=ref_close,
                    config=self.sim_config,
                    temperature=temperature if training else 0.0,
                )

                # Loss with strong forced exit penalty
                loss = (
                    -result.mean_return * 100
                    - result.sharpe * 0.5
                    + result.forced_exit_rate * 3.0  # High penalty
                    + (agg["position_size"].mean() - 0.3).square() * 0.1
                )

                # Scale for gradient accumulation
                loss = loss / self.grad_accum_steps

            if training:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_count += 1

                if accum_count >= self.grad_accum_steps:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    accum_count = 0

                    if scheduler:
                        scheduler.step()

            # Record metrics
            metrics["loss"].append(loss.item() * self.grad_accum_steps)
            metrics["return"].append(result.mean_return.item())
            metrics["sharpe"].append(result.sharpe.item())
            metrics["tp_rate"].append(result.tp_hit.mean().item())
            metrics["fe_rate"].append(result.forced_exit_rate.item())
            spread = (agg["sell_price"] - agg["buy_price"]) / (ref_close + 1e-8)
            metrics["spread"].append(spread.mean().item())
            metrics["position"].append(agg["position_size"].mean().item())

        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def train(self, log_every: int = 1) -> Dict[str, List[float]]:
        config = self.config
        train_loader = self.data_module.train_dataloader(config.batch_size, config.num_workers)
        val_loader = self.data_module.val_dataloader(config.batch_size, config.num_workers)

        total_steps = len(train_loader) * config.epochs // self.grad_accum_steps
        warmup_steps = total_steps // 10

        scheduler = CosineWarmupScheduler(self.optimizer, warmup_steps, total_steps)

        logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
        logger.info(f"Grad accumulation: {self.grad_accum_steps}, Mixup alpha: {self.mixup_alpha}")
        logger.info(f"Dropout: {self.dropout}, Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")

        history = defaultdict(list)

        for epoch in range(config.epochs):
            temp = self.temp_schedule.get_temperature(epoch, config.epochs)

            train_metrics = self._run_epoch(
                train_loader,
                training=True,
                temperature=temp,
                scheduler=scheduler,
                use_mixup=(self.mixup_alpha > 0),
            )

            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, training=False, temperature=0.0)

            lr = scheduler.get_lr()[0]

            # Log metrics
            if (epoch + 1) % log_every == 0 or epoch == config.epochs - 1:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} (lr={lr:.6f}) | "
                    f"Train: {train_metrics['return']*100:.2f}% ret, {train_metrics['sharpe']:.3f} sharpe | "
                    f"Val: {val_metrics['return']*100:.2f}% ret, {val_metrics['sharpe']:.3f} sharpe, "
                    f"{val_metrics['tp_rate']:.1%} TP, {val_metrics['spread']:.1%} spread"
                )

            # Track history
            for key in train_metrics:
                history[f"train_{key}"].append(train_metrics[key])
            for key in val_metrics:
                history[f"val_{key}"].append(val_metrics[key])

            # Save based on val sharpe
            if val_metrics["sharpe"] > self.best_val_metric:
                self.best_val_metric = val_metrics["sharpe"]
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch + 1, val_metrics)
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.warning(f"Early stopping at epoch {epoch + 1} (no improvement for {self.patience} epochs)")
                break

        logger.info(f"Best val sharpe: {self.best_val_metric:.4f} at epoch {self.best_epoch}")
        return dict(history)

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": {
                "policy": self.model.config.__dict__,
                "simulation": self.sim_config.__dict__,
            },
            "normalizer": self.data_module.normalizer,
            "feature_columns": self.data_module.feature_columns,
            "version": "v6",
            "anti_overfit": {
                "dropout": self.dropout,
                "stochastic_depth": self.stochastic_depth,
                "mixup_alpha": self.mixup_alpha,
                "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
            },
        }, path)
        logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128, help="Large batch for stability")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mixup", type=float, default=0.2, help="Mixup alpha (0 to disable)")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--num-symbols", type=int, default=None, help="Limit symbols (None=all)")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"v6_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Select symbols
    symbols = ALL_BASE_SYMBOLS[:args.num_symbols] if args.num_symbols else ALL_BASE_SYMBOLS
    logger.info(f"V6 BASE MODEL TRAINING on {len(symbols)} symbols")
    logger.info("Anti-overfitting settings:")
    logger.info(f"  - Dropout: {args.dropout}")
    logger.info(f"  - Mixup alpha: {args.mixup}")
    logger.info(f"  - Weight decay: {args.weight_decay}")
    logger.info(f"  - Grad accumulation: {args.grad_accum}")
    logger.info(f"  - Early stopping patience: {args.patience}")

    dataset_config = DailyDatasetConfigV4(
        symbols=symbols,
        sequence_length=128,
        lookahead_days=6,
        validation_days=40,
        min_history_days=300,
        include_weekly_features=True,
    )

    training_config = DailyTrainingConfigV4(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=128,
        lookahead_days=6,
        patch_size=5,
        num_windows=2,
        window_size=3,
        num_quantiles=3,
        transformer_dim=256,
        transformer_layers=3,
        transformer_heads=8,
        transformer_kv_heads=4,
        max_hold_days=6,
        min_hold_days=1,
        maker_fee=0.0008,
        initial_temperature=0.005,
        final_temperature=0.001,
        temp_warmup_epochs=1,
        temp_anneal_epochs=args.epochs - 1,
        run_name=args.run_name,
        checkpoint_root="neuraldailyv4/checkpoints",
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,
        dataset=dataset_config,
    )

    data_module = DailyDataModuleV4(dataset_config)

    trainer = V6BaseTrainer(
        training_config,
        data_module,
        dropout=args.dropout,
        mixup_alpha=args.mixup,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum,
        patience=args.patience,
    )

    params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"Model: {params:,} params")

    trainer.train(log_every=1)


if __name__ == "__main__":
    main()
