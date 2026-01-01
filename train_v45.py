#!/usr/bin/env python3
"""V4.5: Focus on improving trade selection with volatility awareness.

Key changes:
- Add volatility-scaled position sizing
- Lower min spread (1.5%) but stronger forced exit penalty
- Directly optimize for pct return

Usage:
    python train_v45.py --equity-only --epochs 3
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv4.config import DailyTrainingConfigV4, DailyDatasetConfigV4, PolicyConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.model import MultiSymbolPolicyV4, create_group_mask
from neuraldailyv4.simulation_v41 import aggregate_predictions, simulate_single_trade, TradeResult


class V45Trainer:
    """V4.5 trainer with pct-return focus and volatility awareness."""

    def __init__(self, config: DailyTrainingConfigV4, data_module: DailyDataModuleV4):
        self.config = config
        self.data_module = data_module
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        data_module.setup()
        policy_config = config.get_policy_config(len(data_module.feature_columns))
        self.model = MultiSymbolPolicyV4(policy_config).to(self.device)
        self.sim_config = config.get_simulation_config()
        self.temp_schedule = config.get_temperature_schedule()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.02,  # Stronger regularization
        )

        self.use_amp = config.use_amp
        self.amp_dtype = getattr(torch, config.amp_dtype) if config.use_amp else torch.float32
        self.scaler = GradScaler() if self.use_amp else None

        self.checkpoint_dir = Path(config.checkpoint_root) / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_metric = float("-inf")
        self.best_epoch = 0

    def _compute_pct_return_loss(
        self,
        result: TradeResult,
        position_size: torch.Tensor,
        reference_price: torch.Tensor,
    ) -> torch.Tensor:
        """Loss focused on percentage returns."""
        # Main objective: maximize pct returns
        pct_return = result.returns / (position_size + 1e-8)  # Per-unit return
        mean_pct = pct_return.mean()

        # Sharpe of pct returns
        std_pct = pct_return.std() + 1e-8
        sharpe = mean_pct / std_pct

        # Forced exit penalty (key!)
        fe_penalty = result.forced_exit_rate * 3.0  # Very high penalty

        # Position variance penalty (encourage consistent sizing)
        pos_std = position_size.std()

        loss = (
            -mean_pct * 100  # Scale up pct returns
            - sharpe * 0.5
            + fe_penalty
            + pos_std * 0.1
        )
        return loss, mean_pct, sharpe

    def _run_epoch(self, dataloader, training: bool, temperature: float) -> Dict[str, float]:
        self.model.train(training)
        metrics = defaultdict(list)

        for batch in dataloader:
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

            if training:
                self.optimizer.zero_grad()

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

                # Aggregate with lower min spread
                agg = aggregate_predictions(
                    buy_quantiles=actions["buy_quantiles"],
                    sell_quantiles=actions["sell_quantiles"],
                    exit_days=actions["exit_days"],
                    position_size=actions["position_size"],
                    confidence=actions["confidence"],
                    trim_fraction=0.25,
                    min_spread=0.015,  # 1.5% spread
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

                loss, mean_pct, sharpe = self._compute_pct_return_loss(
                    result, agg["position_size"], ref_close
                )

            if training:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

            metrics["loss"].append(loss.item())
            metrics["pct_return"].append(mean_pct.item() * 100)
            metrics["sharpe"].append(sharpe.item())
            metrics["tp_rate"].append(result.tp_hit.mean().item())
            metrics["position_size"].append(agg["position_size"].mean().item())
            spread = (agg["sell_price"] - agg["buy_price"]) / (ref_close + 1e-8)
            metrics["spread"].append(spread.mean().item())

        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def train(self, log_every: int = 1):
        config = self.config
        train_loader = self.data_module.train_dataloader(config.batch_size, config.num_workers)
        val_loader = self.data_module.val_dataloader(config.batch_size, config.num_workers)

        logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

        for epoch in range(config.epochs):
            temp = self.temp_schedule.get_temperature(epoch, config.epochs)

            train_metrics = self._run_epoch(train_loader, training=True, temperature=temp)
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, training=False, temperature=0.0)

            # Use pct_return as primary metric
            val_pct = val_metrics["pct_return"]

            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Train: {train_metrics['pct_return']:.2f}% ret, {train_metrics['sharpe']:.3f} sharpe | "
                f"Val: {val_pct:.2f}% ret, {val_metrics['sharpe']:.3f} sharpe, "
                f"{val_metrics['tp_rate']:.1%} TP, {val_metrics['spread']:.1%} spread"
            )

            if val_pct > self.best_val_metric:
                self.best_val_metric = val_pct
                self.best_epoch = epoch + 1
                self._save_checkpoint(epoch + 1, val_metrics)

        logger.info(f"Best val pct return: {self.best_val_metric:.2f}% at epoch {self.best_epoch}")

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "config": {
                "policy": self.model.config.__dict__,
                "simulation": self.sim_config.__dict__,
            },
            "normalizer": self.data_module.normalizer,
            "feature_columns": self.data_module.feature_columns,
            "version": "v4.5",
        }, path)
        logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--equity-only", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"v45_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"V4.5 Training: {args.run_name}")
    logger.info("  - 1.5% min spread (vs 2%)")
    logger.info("  - PCT return focused loss")
    logger.info("  - Very high FE penalty (3x)")

    symbols = ("SPY", "QQQ", "AAPL", "GOOGL", "AMZN", "NVDA", "TSLA")

    dataset_config = DailyDatasetConfigV4(
        symbols=symbols,
        sequence_length=128,
        lookahead_days=6,
        validation_days=30,
        min_history_days=500,
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
        initial_temperature=0.003,
        final_temperature=0.001,
        temp_warmup_epochs=0,
        temp_anneal_epochs=args.epochs,
        run_name=args.run_name,
        checkpoint_root="neuraldailyv4/checkpoints",
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,
        dataset=dataset_config,
    )

    data_module = DailyDataModuleV4(dataset_config)
    trainer = V45Trainer(training_config, data_module)

    params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"Model: {params:,} params")
    trainer.train(log_every=1)


if __name__ == "__main__":
    main()
