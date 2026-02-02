from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_  # type: ignore

from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    combined_sortino_pnl_loss,
    compute_hourly_objective,
    simulate_hourly_trades,
)

from .config import TrainingConfig
from .data import BinanceHourlyDataModule, FeatureNormalizer, MultiSymbolDataModule
from .model import BinancePolicyBase, PolicyConfig, build_policy
from traininglib.optim_factory import MultiOptim

try:  # pragma: no cover - optional dependency
    from nanochat.nanochat.muon import Muon
    MUON_AVAILABLE = True
except Exception:  # pragma: no cover
    Muon = None  # type: ignore
    MUON_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    config: Optional[TrainingConfig] = None
    checkpoint_paths: List[Path] = field(default_factory=list)
    best_checkpoint: Optional[Path] = None


class BinanceHourlyTrainer:
    def __init__(self, config: TrainingConfig, data_module: BinanceHourlyDataModule | MultiSymbolDataModule) -> None:
        self.config = config
        self.data = data_module
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        run_name = self.config.run_name or time.strftime("binanceneural_%Y%m%d_%H%M%S")
        self.config.run_name = run_name
        self.checkpoint_dir = self.config.checkpoint_root / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._warmup_base_lrs: list[float] = []

    def train(self) -> TrainingArtifacts:
        torch.manual_seed(self.config.seed)
        if self.config.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        policy_cfg = PolicyConfig(
            input_dim=len(self.data.feature_columns),
            hidden_dim=self.config.transformer_dim,
            dropout=self.config.transformer_dropout,
            price_offset_pct=self.config.price_offset_pct,
            min_price_gap_pct=self.config.min_price_gap_pct,
            trade_amount_scale=self.config.trade_amount_scale,
            num_heads=self.config.transformer_heads,
            num_layers=self.config.transformer_layers,
            max_len=max(self.config.sequence_length, 32),
            model_arch=self.config.model_arch,
            num_kv_heads=self.config.num_kv_heads,
            mlp_ratio=self.config.mlp_ratio,
            logits_softcap=self.config.logits_softcap,
            rope_base=self.config.rope_base,
            use_qk_norm=self.config.use_qk_norm,
            use_causal_attention=self.config.use_causal_attention,
            rms_norm_eps=self.config.rms_norm_eps,
            use_midpoint_offsets=True,
        )
        model = build_policy(policy_cfg).to(self.device)

        if self.config.preload_checkpoint_path:
            preload_path = Path(self.config.preload_checkpoint_path)
            if preload_path.exists():
                logger.info("Preloading weights from %s", preload_path)
                checkpoint = torch.load(preload_path, map_location="cpu")
                state_dict = checkpoint.get("state_dict", checkpoint)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning("Missing keys during preload: %s", missing)
                if unexpected:
                    logger.warning("Unexpected keys during preload: %s", unexpected)
            else:
                logger.warning("Preload checkpoint not found: %s", preload_path)

        if self.config.use_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode="max-autotune")

        optimizer = self._build_optimizer(model)
        self._warmup_base_lrs = [group.get("lr", self.config.learning_rate) for group in optimizer.param_groups]

        train_loader = self.data.train_dataloader(self.config.batch_size, self.config.num_workers)
        val_loader = self.data.val_dataloader(self.config.batch_size, self.config.num_workers)

        history: List[TrainingHistoryEntry] = []
        best_score = float("-inf")
        best_checkpoint: Optional[Path] = None

        global_step = 0
        for epoch in range(1, self.config.epochs + 1):
            train_metrics, global_step = self._run_epoch(
                model,
                train_loader,
                optimizer,
                train=True,
                global_step=global_step,
            )
            val_metrics, _ = self._run_epoch(
                model,
                val_loader,
                optimizer=None,
                train=False,
                global_step=global_step,
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

            if val_metrics["score"] > best_score:
                best_score = val_metrics["score"]
                best_checkpoint = self._save_checkpoint(model, epoch, val_metrics)

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Score: {train_metrics['score']:.4f} "
                f"Sortino: {train_metrics['sortino']:.4f} Return: {train_metrics['return']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Score: {val_metrics['score']:.4f} "
                f"Sortino: {val_metrics['sortino']:.4f} Return: {val_metrics['return']:.4f}"
            )

        return TrainingArtifacts(
            state_dict=model.state_dict(),
            normalizer=self.data.normalizer,
            history=history,
            feature_columns=list(self.data.feature_columns),
            config=self.config,
            checkpoint_paths=list(self.checkpoint_dir.glob("*.pt")),
            best_checkpoint=best_checkpoint,
        )

    def _run_epoch(
        self,
        model: BinancePolicyBase,
        loader: torch.utils.data.DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        *,
        train: bool,
        global_step: int,
    ) -> Tuple[Dict[str, float], int]:
        model.train(train)
        total_loss = 0.0
        total_score = 0.0
        total_sortino = 0.0
        total_return = 0.0
        steps = 0

        for batch in loader:
            features = batch["features"].to(self.device)
            highs = batch["high"].to(self.device)
            lows = batch["low"].to(self.device)
            closes = batch["close"].to(self.device)
            reference_close = batch["reference_close"].to(self.device)
            chronos_high = batch["chronos_high"].to(self.device)
            chronos_low = batch["chronos_low"].to(self.device)

            outputs = model(features)
            actions = model.decode_actions(
                outputs,
                reference_close=reference_close,
                chronos_high=chronos_high,
                chronos_low=chronos_low,
            )

            scale = float(self.config.trade_amount_scale)
            trade_intensity = actions["trade_amount"] / scale
            buy_intensity = actions["buy_amount"] / scale
            sell_intensity = actions["sell_amount"] / scale

            sim = simulate_hourly_trades(
                highs=highs,
                lows=lows,
                closes=closes,
                buy_prices=actions["buy_price"],
                sell_prices=actions["sell_price"],
                trade_intensity=trade_intensity,
                buy_trade_intensity=buy_intensity,
                sell_trade_intensity=sell_intensity,
                maker_fee=self.config.maker_fee,
                initial_cash=self.config.initial_cash,
            )
            returns = sim.returns
            score, sortino, annual_return = compute_hourly_objective(
                returns,
                periods_per_year=HOURLY_PERIODS_PER_YEAR,
                return_weight=self.config.return_weight,
            )
            loss = combined_sortino_pnl_loss(
                returns,
                target_sign=self.config.sortino_target_sign,
                periods_per_year=HOURLY_PERIODS_PER_YEAR,
                return_weight=self.config.return_weight,
            )

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.grad_clip:
                    clip_grad_norm_(model.parameters(), self.config.grad_clip)
                if self.config.warmup_steps and global_step < self.config.warmup_steps:
                    warmup_frac = float(global_step + 1) / float(self.config.warmup_steps)
                    for group, base_lr in zip(optimizer.param_groups, self._warmup_base_lrs):
                        group["lr"] = float(base_lr) * warmup_frac
                optimizer.step()
                global_step += 1

            total_loss += float(loss.detach().mean().item())
            total_score += float(score.detach().mean().item())
            total_sortino += float(sortino.detach().mean().item())
            total_return += float(annual_return.detach().mean().item())
            steps += 1

            if self.config.dry_train_steps and steps >= self.config.dry_train_steps:
                break

        if steps == 0:
            raise RuntimeError("No batches available for training/validation")
        metrics = {
            "loss": total_loss / steps,
            "score": total_score / steps,
            "sortino": total_sortino / steps,
            "return": total_return / steps,
        }
        return metrics, global_step

    def _save_checkpoint(self, model: BinancePolicyBase, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        payload = {
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "epoch": epoch,
            "config": self.config,
        }
        torch.save(payload, path)
        return path

    def _build_optimizer(self, model: BinancePolicyBase) -> torch.optim.Optimizer:
        name = (self.config.optimizer_name or "adamw").lower()
        if name in {"muon", "muon_mix", "dual"}:
            if not MUON_AVAILABLE:
                logger.warning("Muon optimizer requested but not available; falling back to AdamW.")
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            muon_params, adam_params = self._split_muon_params(model)
            optimizers: list[torch.optim.Optimizer] = []
            if adam_params:
                optimizers.append(
                    torch.optim.AdamW(
                        adam_params,
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay,
                    )
                )
            if muon_params:
                optimizers.append(
                    Muon(  # type: ignore[call-arg]
                        muon_params,
                        lr=self.config.muon_lr,
                        momentum=self.config.muon_momentum,
                        nesterov=self.config.muon_nesterov,
                        ns_steps=self.config.muon_ns_steps,
                    )
                )
            if not optimizers:
                logger.warning("No trainable parameters found; defaulting to AdamW.")
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            if len(optimizers) == 1:
                return optimizers[0]
            return MultiOptim(optimizers)

        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    @staticmethod
    def _split_muon_params(model: BinancePolicyBase) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        muon_params: list[torch.Tensor] = []
        adam_params: list[torch.Tensor] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 2 and "embed" not in name and "head" not in name:
                muon_params.append(param)
            else:
                adam_params.append(param)
        return muon_params, adam_params


__all__ = ["BinanceHourlyTrainer", "TrainingArtifacts", "TrainingHistoryEntry"]
