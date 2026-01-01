#!/usr/bin/env python3
"""V5: Pretrain + Finetune approach.

Stage 1 (Pretrain): Train on ALL symbols (24) to learn general market patterns
Stage 2 (Finetune): Specialize on target symbols (7 equities) with lower LR

Key improvements:
- More data (24 symbols, 4 years each)
- Gradient accumulation for larger effective batch size
- Cosine LR with warmup
- Pretrain with higher dropout, finetune with lower

Usage:
    python train_v5_pretrain.py --stage pretrain --epochs 10
    python train_v5_pretrain.py --stage finetune --checkpoint ... --epochs 2
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv4.config import DailyTrainingConfigV4, DailyDatasetConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.model import MultiSymbolPolicyV4, create_group_mask
from neuraldailyv4.simulation_v41 import aggregate_predictions, simulate_single_trade


# All available symbols for pretraining
ALL_SYMBOLS = (
    # Major indices & tech
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    # Semiconductors
    "AMD", "INTC",
    # Finance
    "JPM", "BAC", "GS", "V", "MA",
    # Consumer
    "DIS", "NFLX", "COST", "HD",
    # Crypto
    "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD",
)

# Target symbols for finetuning (best performers)
TARGET_SYMBOLS = ("SPY", "QQQ", "AAPL", "GOOGL", "AMZN", "NVDA", "TSLA")


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
            # Linear warmup
            scale = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class V5Trainer:
    """V5 trainer with pretrain/finetune support."""

    def __init__(
        self,
        config: DailyTrainingConfigV4,
        data_module: DailyDataModuleV4,
        pretrained_path: Optional[Path] = None,
        stage: str = "pretrain",
    ):
        self.config = config
        self.data_module = data_module
        self.stage = stage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        data_module.setup()
        policy_config = config.get_policy_config(len(data_module.feature_columns))

        # Adjust dropout based on stage
        if stage == "pretrain":
            policy_config.dropout = 0.15  # Higher dropout for pretrain
        else:
            policy_config.dropout = 0.05  # Lower for finetune

        self.model = MultiSymbolPolicyV4(policy_config).to(self.device)

        # Load pretrained weights if finetuning
        if pretrained_path and pretrained_path.exists():
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.sim_config = config.get_simulation_config()
        self.temp_schedule = config.get_temperature_schedule()

        # Different LR for pretrain vs finetune
        lr = config.learning_rate if stage == "pretrain" else config.learning_rate * 0.1
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01 if stage == "pretrain" else 0.001,
            betas=(0.9, 0.95),
        )

        self.use_amp = config.use_amp
        self.amp_dtype = getattr(torch, config.amp_dtype) if config.use_amp else torch.float32
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.grad_accum_steps = 4 if stage == "pretrain" else 1

        self.checkpoint_dir = Path(config.checkpoint_root) / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_metric = float("-inf")
        self.best_epoch = 0

    def _run_epoch(self, dataloader, training: bool, temperature: float, scheduler=None) -> Dict[str, float]:
        self.model.train(training)
        metrics = defaultdict(list)

        self.optimizer.zero_grad()
        accum_count = 0

        for batch_idx, batch in enumerate(dataloader):
            features = batch["features"].to(self.device)
            future_highs = batch["future_highs"].to(self.device)
            future_lows = batch["future_lows"].to(self.device)
            future_closes = batch["future_closes"].to(self.device)
            reference_close = batch["reference_close"].to(self.device)
            chronos_high = batch["chronos_high"].to(self.device)
            chronos_low = batch["chronos_low"].to(self.device)
            asset_class = batch["asset_class"].to(self.device)
            group_ids = batch["group_id"].to(self.device)

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

                # Loss: maximize returns, penalize forced exits
                loss = (
                    -result.mean_return * 100  # Scale for gradients
                    - result.sharpe * 0.5
                    + result.forced_exit_rate * 2.0
                    + (agg["position_size"].mean() - 0.3).square() * 0.1  # Position reg
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

            metrics["loss"].append(loss.item() * self.grad_accum_steps)
            metrics["return"].append(result.mean_return.item())
            metrics["sharpe"].append(result.sharpe.item())
            metrics["tp_rate"].append(result.tp_hit.mean().item())
            metrics["fe_rate"].append(result.forced_exit_rate.item())
            spread = (agg["sell_price"] - agg["buy_price"]) / (ref_close + 1e-8)
            metrics["spread"].append(spread.mean().item())
            metrics["position"].append(agg["position_size"].mean().item())

        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def train(self, log_every: int = 1):
        config = self.config
        train_loader = self.data_module.train_dataloader(config.batch_size, config.num_workers)
        val_loader = self.data_module.val_dataloader(config.batch_size, config.num_workers)

        total_steps = len(train_loader) * config.epochs // self.grad_accum_steps
        warmup_steps = total_steps // 10  # 10% warmup

        scheduler = CosineWarmupScheduler(self.optimizer, warmup_steps, total_steps)

        logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
        logger.info(f"Grad accumulation: {self.grad_accum_steps}")

        for epoch in range(config.epochs):
            temp = self.temp_schedule.get_temperature(epoch, config.epochs)

            train_metrics = self._run_epoch(train_loader, training=True, temperature=temp, scheduler=scheduler)
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, training=False, temperature=0.0)

            lr = scheduler.get_lr()[0]

            if (epoch + 1) % log_every == 0 or epoch == config.epochs - 1:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} (lr={lr:.6f}) | "
                    f"Train: {train_metrics['return']*100:.2f}% ret, {train_metrics['sharpe']:.3f} sharpe | "
                    f"Val: {val_metrics['return']*100:.2f}% ret, {val_metrics['sharpe']:.3f} sharpe, "
                    f"{val_metrics['tp_rate']:.1%} TP, {val_metrics['spread']:.1%} spread"
                )

            # Save based on val sharpe
            if val_metrics["sharpe"] > self.best_val_metric:
                self.best_val_metric = val_metrics["sharpe"]
                self.best_epoch = epoch + 1
                self._save_checkpoint(epoch + 1, val_metrics)

        logger.info(f"Best val sharpe: {self.best_val_metric:.4f} at epoch {self.best_epoch}")
        return self.best_val_metric

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
            "version": "v5",
            "stage": self.stage,
        }, path)
        logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained checkpoint for finetune")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"v5_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Select symbols based on stage
    if args.stage == "pretrain":
        symbols = ALL_SYMBOLS
        logger.info(f"PRETRAIN on {len(symbols)} symbols: {symbols}")
    else:
        symbols = TARGET_SYMBOLS
        logger.info(f"FINETUNE on {len(symbols)} target symbols: {symbols}")

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
        temp_warmup_epochs=2,
        temp_anneal_epochs=args.epochs - 2,
        run_name=args.run_name,
        checkpoint_root="neuraldailyv4/checkpoints",
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,
        dataset=dataset_config,
    )

    data_module = DailyDataModuleV4(dataset_config)

    pretrained_path = Path(args.checkpoint) if args.checkpoint else None
    trainer = V5Trainer(training_config, data_module, pretrained_path, stage=args.stage)

    params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"Model: {params:,} params")
    logger.info(f"Stage: {args.stage.upper()}")

    trainer.train(log_every=1)


if __name__ == "__main__":
    main()
