from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_

from differentiable_loss_utils import (
    compute_hourly_objective,
    get_periods_per_year,
    simulate_hourly_trades,
    simulate_hourly_trades_binary,
)
from hourlycryptotraining.optimizers import Muon
from wandboard import WandBoardLogger

from .checkpoints import CheckpointRecord, save_checkpoint, write_manifest
from .config import DailyTrainingConfig
from .data import DailyDataModule, FeatureNormalizer
from .model import DailyMultiAssetPolicy, DailyPolicyConfig, MultiSymbolDailyPolicy


def apply_symbol_dropout(
    features: torch.Tensor,
    group_mask: Optional[torch.Tensor],
    dropout_rate: float,
    *,
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Randomly drop symbols to make cross-attention robust to missing members."""

    if dropout_rate <= 0 or not training:
        zero_mask = torch.zeros(features.shape[0], device=features.device, dtype=torch.bool)
        return features, group_mask, zero_mask

    drop_mask = torch.rand(features.shape[0], device=features.device) < dropout_rate
    if drop_mask.any():
        keep_mask = (~drop_mask).view(-1, 1, 1)
        features = features * keep_mask
        if group_mask is not None:
            keep_vec = (~drop_mask).view(-1, 1)
            group_mask = group_mask & keep_vec & keep_vec.transpose(0, 1)
    return features, group_mask, drop_mask


@dataclass
class TrainingHistoryEntry:
    epoch: int
    train_loss: float
    train_score: float
    train_sortino: float
    train_return: float
    val_loss: Optional[float] = None
    val_score: Optional[float] = None
    val_sortino: Optional[float] = None
    val_return: Optional[float] = None


@dataclass
class TrainingArtifacts:
    state_dict: Dict[str, torch.Tensor]
    normalizer: FeatureNormalizer
    history: List[TrainingHistoryEntry] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    config: Optional[DailyTrainingConfig] = None
    checkpoint_paths: List[Path] = field(default_factory=list)
    best_checkpoint: Optional[Path] = None


class NeuralDailyTrainer:
    """Train the daily multi-asset transformer policy."""

    def __init__(self, config: DailyTrainingConfig, data_module: DailyDataModule) -> None:
        self.config = config
        self.data = data_module
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        run_name = self.config.run_name or time.strftime("neuraldaily_%Y%m%d_%H%M%S")
        self.config.run_name = run_name
        self.checkpoint_dir = self.config.checkpoint_root / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_records: List[CheckpointRecord] = []
        self.best_checkpoint_path: Optional[Path] = None

        # Determine periods per year based on first symbol (252 for stocks, 365 for crypto)
        first_symbol = self.data.symbols[0] if self.data.symbols else ""
        self.periods_per_year = get_periods_per_year(frequency="daily", symbol=first_symbol)

    def train(self) -> TrainingArtifacts:
        torch.manual_seed(self.config.seed)
        if self.config.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        policy_cfg = DailyPolicyConfig(
            input_dim=len(self.data.feature_columns),
            hidden_dim=self.config.transformer_dim,
            dropout=self.config.transformer_dropout,
            price_offset_pct=self.config.price_offset_pct,
            max_trade_qty=self.config.max_trade_qty,
            min_price_gap_pct=self.config.min_price_gap_pct,
            num_heads=self.config.transformer_heads,
            num_layers=self.config.transformer_layers,
            use_cross_attention=self.config.use_cross_attention,
        )
        if self.config.use_cross_attention:
            policy = MultiSymbolDailyPolicy(policy_cfg).to(self.device)
        else:
            policy = DailyMultiAssetPolicy(policy_cfg).to(self.device)
        if self.config.use_compile:
            compile_mode = getattr(self.config, "compile_mode", "max-autotune")
            policy = torch.compile(policy, mode=compile_mode)  # type: ignore[attr-defined]
        optimizer = self._build_optimizer(policy)
        train_loader = self.data.train_dataloader(self.config.batch_size, self.config.num_workers)
        val_loader = self.data.val_dataloader(self.config.batch_size, self.config.num_workers)
        scheduler = self._build_scheduler(optimizer, train_loader)
        scaler = torch.cuda.amp.GradScaler() if (self.config.use_amp and torch.cuda.is_available()) else None
        ema_state: Optional[Dict[str, torch.Tensor]] = None
        if self.config.ema_decay and self.config.ema_decay > 0:
            ema_state = {name: param.detach().clone() for name, param in policy.named_parameters() if param.requires_grad}
        history: List[TrainingHistoryEntry] = []
        best_state = None
        best_score = float("-inf")
        wandb_kwargs = {
            "run_name": self.config.run_name,
            "project": self.config.wandb_project,
            "entity": self.config.wandb_entity,
            "log_dir": self.config.log_dir,
            "tensorboard_subdir": self.config.run_name or "neuraldaily",
            "log_metrics": True,
        }
        with WandBoardLogger(**wandb_kwargs) as tracker:
            tracker.watch(policy)
            global_step = 0
            best_symbol_stats: Dict[int, Dict[str, float]] = {}
            for epoch in range(1, self.config.epochs + 1):
                train_metrics, global_step, train_symbol_stats = self._run_epoch(
                    policy,
                    train_loader,
                    optimizer,
                    scheduler,
                    train=True,
                    global_step=global_step,
                    ema_state=ema_state,
                    scaler=scaler,
                )
                val_metrics, _, val_symbol_stats = self._run_epoch(
                    policy,
                    val_loader,
                    optimizer=None,
                    scheduler=None,
                    train=False,
                    global_step=global_step,
                    scaler=None,
                )
                entry = TrainingHistoryEntry(
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    train_score=train_metrics["score"],
                    train_sortino=train_metrics["sortino"],
                    train_return=train_metrics["return"],
                    val_loss=val_metrics["loss"],
                    val_score=val_metrics["score"],
                    val_sortino=val_metrics["sortino"],
                    val_return=val_metrics["return"],
                )
                history.append(entry)
                self._maybe_save_checkpoint(policy, val_metrics, epoch)

                binary_info = ""
                if "binary_return" in val_metrics:
                    binary_info = (
                        f" | Binary: Sortino {val_metrics['binary_sortino']:.4f} "
                        f"Return {val_metrics['binary_return']:.4f}"
                    )

                summary = (
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"Train Loss {train_metrics['loss']:.4f} Score {train_metrics['score']:.4f} "
                    f"Sortino {train_metrics['sortino']:.4f} Return {train_metrics['return']:.4f} | "
                    f"Val Loss {val_metrics['loss']:.4f} Score {val_metrics['score']:.4f} "
                    f"Sortino {val_metrics['sortino']:.4f} Return {val_metrics['return']:.4f}"
                    f"{binary_info}"
                )
                print(summary)
                log_dict = {
                    "loss/train": train_metrics["loss"],
                    "loss/val": val_metrics["loss"],
                    "score/train": train_metrics["score"],
                    "score/val": val_metrics["score"],
                    "sortino/train": train_metrics["sortino"],
                    "sortino/val": val_metrics["sortino"],
                    "return/train": train_metrics["return"],
                    "return/val": val_metrics["return"],
                    "fill/buy_train": train_metrics["buy_fill"],
                    "fill/sell_train": train_metrics["sell_fill"],
                    "fill/buy_val": val_metrics["buy_fill"],
                    "fill/sell_val": val_metrics["sell_fill"],
                    "exposure/train": train_metrics.get("avg_trade_amount", 0.0),
                    "exposure/val": val_metrics.get("avg_trade_amount", 0.0),
                    "leverage/excess_train": train_metrics.get("avg_over_leverage", 0.0),
                    "leverage/excess_val": val_metrics.get("avg_over_leverage", 0.0),
                }

                # Add binary metrics if available
                if "binary_return" in val_metrics:
                    log_dict["binary/sortino"] = val_metrics["binary_sortino"]
                    log_dict["binary/return"] = val_metrics["binary_return"]
                    log_dict["binary/buy_fill"] = val_metrics["binary_buy_fill"]
                    log_dict["binary/sell_fill"] = val_metrics["binary_sell_fill"]

                # Per-symbol logging for quick triage
                def _symbol_logs(stats: Dict[int, Dict[str, float]], prefix: str) -> Dict[str, float]:
                    logs: Dict[str, float] = {}
                    for sid, payload in stats.items():
                        symbol = self.data.id_to_symbol.get(int(sid), f"id_{sid}")
                        for key, value in payload.items():
                            logs[f"{prefix}/{symbol}/{key}"] = float(value)
                    return logs

                log_dict.update(_symbol_logs(train_symbol_stats, "symbol/train"))
                log_dict.update(_symbol_logs(val_symbol_stats, "symbol/val"))

                tracker.log(log_dict, step=epoch)
                if val_metrics["score"] > best_score:
                    best_score = val_metrics["score"]
                    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                    best_symbol_stats = val_symbol_stats
                if self.config.dry_train_steps and global_step >= self.config.dry_train_steps:
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        if self.data.normalizer is None:
            raise RuntimeError("DailyDataModule did not expose a fitted normalizer.")
        artifacts = TrainingArtifacts(
            state_dict=best_state,
            normalizer=self.data.normalizer,
            history=history,
            feature_columns=list(self.data.feature_columns),
            config=self.config,
            checkpoint_paths=[record.path for record in self._checkpoint_records],
            best_checkpoint=self.best_checkpoint_path,
        )
        self._write_non_tradable(best_symbol_stats)
        return artifacts

    # ------------------------------------------------------------------
    def _run_epoch(
        self,
        model: DailyMultiAssetPolicy,
        loader,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
        *,
        train: bool,
        global_step: int,
        ema_state: Optional[Dict[str, torch.Tensor]] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> (Dict[str, float], int, Dict[int, Dict[str, float]]):
        if train:
            model.train()
        else:
            model.eval()
        totals = {key: 0.0 for key in ("loss", "score", "sortino", "return", "buy_fill", "sell_fill")}
        totals["avg_trade_amount"] = 0.0
        totals["avg_over_leverage"] = 0.0
        b_totals = {key: 0.0 for key in ("score", "return", "buy_fill", "sell_fill")}
        batches = 0
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16
        symbol_totals: Dict[int, Dict[str, float]] = {}

        def _accumulate_symbol(metric: torch.Tensor, name: str, symbol_ids: torch.Tensor) -> None:
            flat_ids = symbol_ids.detach().cpu().view(-1).tolist()
            flat_vals = metric.detach().cpu().view(-1).tolist()
            for sid, val in zip(flat_ids, flat_vals):
                slot = symbol_totals.setdefault(int(sid), {"count": 0})
                slot[name] = slot.get(name, 0.0) + float(val)
                slot["count"] += 1

        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            use_amp = scaler is not None
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                features = batch["features"]
                # Create group_mask for cross-symbol attention if group_ids available
                group_mask = None
                if "group_id" in batch:
                    group_ids = batch["group_id"]
                    # Items with same group_id can attend to each other
                    group_mask = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)

                # Drop random symbols to make model robust to missing members
                features, group_mask, drop_mask = apply_symbol_dropout(
                    features,
                    group_mask,
                    getattr(self.config.dataset, "symbol_dropout_rate", 0.0),
                    training=train,
                )

                # Call model with group_mask if it supports it
                if hasattr(model, 'blocks') and group_mask is not None:
                    # MultiSymbolDailyPolicy with cross-attention
                    outputs = model(features, group_mask=group_mask)
                else:
                    # Legacy DailyMultiAssetPolicy
                    outputs = model(features)

                asset_class_flag = batch["asset_class"].view(-1)
                leverage_limits = torch.where(
                    asset_class_flag > 0.5,
                    torch.as_tensor(self.config.crypto_max_leverage, device=batch["high"].device, dtype=batch["high"].dtype),
                    torch.as_tensor(self.config.equity_max_leverage, device=batch["high"].device, dtype=batch["high"].dtype),
                )
                actions = model.decode_actions(
                    outputs,
                    reference_close=batch["reference_close"],
                    chronos_high=batch["chronos_high"],
                    chronos_low=batch["chronos_low"],
                    asset_class=asset_class_flag,
                )
                if drop_mask.any():
                    keep = (~drop_mask).view(-1, 1)
                    actions["trade_amount"] = actions["trade_amount"] * keep
                sim = simulate_hourly_trades(
                    highs=batch["high"],
                    lows=batch["low"],
                    closes=batch["close"],
                    buy_prices=actions["buy_price"],
                    sell_prices=actions["sell_price"],
                    trade_intensity=actions["trade_amount"],
                    max_leverage=leverage_limits.unsqueeze(-1),
                    maker_fee=self.config.maker_fee,
                    initial_cash=self.config.initial_cash,
                )
                score, sortino, ann_return = compute_hourly_objective(
                    sim.returns,
                    return_weight=self.config.return_weight,
                    periods_per_year=self.periods_per_year,
                )
                symbol_ids = batch.get("symbol_id")
                if symbol_ids is not None:
                    _accumulate_symbol(score, "score", symbol_ids)
                    _accumulate_symbol(ann_return, "return", symbol_ids)

                loss = -score.mean()
                if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(score).any():
                    raise RuntimeError("Non-finite loss/score detected; aborting training to prevent corrupt checkpoints.")
                inventory_path = sim.inventory_path

                # Calculate leverage cost: charge for any position > 1x (stocks only)
                # Crypto is already capped at 1x so won't have leverage
                leveraged_amount = torch.relu(inventory_path - 1.0)

                # Only stocks pay leverage fee (asset_class_flag <= 0.5 means stock)
                stock_mask = (asset_class_flag <= 0.5).float().unsqueeze(-1)
                stock_leverage = leveraged_amount * stock_mask

                fee_per_step = self.config.leverage_fee_rate / max(1, self.config.steps_per_year)
                if fee_per_step > 0:
                    loss = loss + fee_per_step * stock_leverage.mean()

                # Also track excess above hard limits for monitoring
                limits = torch.where(
                    asset_class_flag > 0.5,
                    torch.as_tensor(self.config.crypto_max_leverage, device=inventory_path.device, dtype=inventory_path.dtype),
                    torch.as_tensor(self.config.equity_max_leverage, device=inventory_path.device, dtype=inventory_path.dtype),
                )
                excess = torch.relu(inventory_path - limits.unsqueeze(-1))
                over_leverage_value = excess.mean().detach()
                if self.config.exposure_penalty > 0:
                    avg_amount = actions["trade_amount"].mean()
                    risk_cap = torch.as_tensor(self.config.risk_threshold, dtype=avg_amount.dtype, device=avg_amount.device)
                    over_exposure = torch.relu(avg_amount - risk_cap)
                    loss = loss + self.config.exposure_penalty * over_exposure.pow(2)

            if train and optimizer is not None:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if ema_state is not None and self.config.ema_decay > 0:
                    decay = self.config.ema_decay
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                ema_state[name].mul_(decay).add_(param.detach(), alpha=1 - decay)

            batch_size = sim.pnl.shape[0]
            batches += batch_size
            totals["loss"] += float(loss.item()) * batch_size
            totals["score"] += float(score.mean().item()) * batch_size
            totals["sortino"] += float(sortino.mean().item()) * batch_size
            totals["return"] += float(ann_return.mean().item()) * batch_size
            totals["buy_fill"] += float(sim.buy_fill_probability.mean().item()) * batch_size
            totals["sell_fill"] += float(sim.sell_fill_probability.mean().item()) * batch_size
            totals["avg_trade_amount"] += float(actions["trade_amount"].mean().item()) * batch_size
            totals["avg_over_leverage"] += float(over_leverage_value.item()) * batch_size

            if not train:
                    with torch.no_grad():
                        binary = simulate_hourly_trades_binary(
                            highs=batch["high"],
                            lows=batch["low"],
                            closes=batch["close"],
                            buy_prices=actions["buy_price"],
                            sell_prices=actions["sell_price"],
                            trade_intensity=actions["trade_amount"],
                            max_leverage=leverage_limits.unsqueeze(-1),
                            maker_fee=self.config.maker_fee,
                            initial_cash=self.config.initial_cash,
                        )
                    _, b_sortino, b_return = compute_hourly_objective(
                        binary.returns,
                        return_weight=self.config.return_weight,
                        periods_per_year=self.periods_per_year,
                        )
                    b_totals["score"] += float(b_sortino.mean().item()) * batch_size
                    b_totals["return"] += float(b_return.mean().item()) * batch_size
                    b_totals["buy_fill"] += float(binary.buy_fill_probability.mean().item()) * batch_size
                    b_totals["sell_fill"] += float(binary.sell_fill_probability.mean().item()) * batch_size

        metrics = {
            "loss": totals["loss"] / max(1, batches),
            "score": totals["score"] / max(1, batches),
            "sortino": totals["sortino"] / max(1, batches),
            "return": totals["return"] / max(1, batches),
            "buy_fill": totals["buy_fill"] / max(1, batches),
            "sell_fill": totals["sell_fill"] / max(1, batches),
            "avg_trade_amount": totals["avg_trade_amount"] / max(1, batches),
            "avg_over_leverage": totals["avg_over_leverage"] / max(1, batches),
        }
        if not train:
            metrics["binary_sortino"] = b_totals["score"] / max(1, batches)
            metrics["binary_return"] = b_totals["return"] / max(1, batches)
            metrics["binary_buy_fill"] = b_totals["buy_fill"] / max(1, batches)
            metrics["binary_sell_fill"] = b_totals["sell_fill"] / max(1, batches)
        averaged_symbol_stats: Dict[int, Dict[str, float]] = {}
        for sid, payload in symbol_totals.items():
            count = max(1, payload.pop("count", 1))
            averaged_symbol_stats[sid] = {k: v / count for k, v in payload.items()}
        return metrics, global_step, averaged_symbol_stats

    def _build_optimizer(self, model: DailyMultiAssetPolicy) -> torch.optim.Optimizer:
        name = (self.config.optimizer_name or "adamw").lower()
        if name == "muon":
            return Muon(model.parameters(), lr=self.config.learning_rate, momentum=0.95, weight_decay=self.config.weight_decay)
        return torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, loader) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
        if self.config.warmup_steps <= 0:
            return None
        total_steps = max(1, len(loader) * max(1, self.config.epochs))
        warmup = min(self.config.warmup_steps, total_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _maybe_save_checkpoint(self, model: DailyMultiAssetPolicy, val_metrics: Dict[str, float], epoch: int) -> None:
        val_loss = val_metrics["loss"]
        ckpt_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        save_checkpoint(
            ckpt_path,
            state_dict=model.state_dict(),
            normalizer=self.data.normalizer,
            feature_columns=list(self.data.feature_columns),
            metrics=val_metrics,
            config=self.config,
            symbol_to_group_id=self.data.symbol_to_group_id,
        )
        record = CheckpointRecord(path=ckpt_path, val_loss=val_loss, epoch=epoch, timestamp=time.time())
        self._checkpoint_records.append(record)
        # Sort by val_loss ASCENDING: more negative losses come first and are better (loss = -score)
        self._checkpoint_records.sort(key=lambda item: item.val_loss)
        if len(self._checkpoint_records) > self.config.top_k_checkpoints:
            dropped = self._checkpoint_records.pop()
            try:
                dropped.path.unlink(missing_ok=True)
            except Exception:
                pass
        write_manifest(self.checkpoint_dir, self._checkpoint_records, self.config, list(self.data.feature_columns))
        self.best_checkpoint_path = self._checkpoint_records[0].path if self._checkpoint_records else ckpt_path

    def _write_non_tradable(self, val_symbol_stats: Dict[int, Dict[str, float]], *, return_threshold: float = 0.0) -> Optional[Path]:
        if not val_symbol_stats:
            return None
        non_tradable: List[Dict[str, float | str]] = []
        for sid, metrics in val_symbol_stats.items():
            symbol = self.data.id_to_symbol.get(int(sid), f"id_{sid}")
            ann_return = float(metrics.get("return", 0.0))
            score = float(metrics.get("score", 0.0))
            if ann_return <= return_threshold:
                non_tradable.append({"symbol": symbol, "return": ann_return, "score": score})
        if not non_tradable:
            return None
        payload = {
            "return_threshold": return_threshold,
            "non_tradable": sorted(non_tradable, key=lambda row: row["return"]),
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
        }
        path = self.checkpoint_dir / "non_tradable.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


__all__ = ["NeuralDailyTrainer", "TrainingArtifacts", "TrainingHistoryEntry"]
