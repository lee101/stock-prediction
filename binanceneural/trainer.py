from __future__ import annotations

import logging
import time
from contextlib import nullcontext
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
from src.serialization_utils import serialize_for_checkpoint
from src.torch_load_utils import torch_load_compat

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
        self._amp_context = nullcontext()
        self._scaler = None
        self._total_train_steps = 0
        self._weight_decay_groups: list[tuple[dict, float]] = []

    def train(self) -> TrainingArtifacts:
        torch.manual_seed(self.config.seed)
        if self.config.use_tf32:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
            if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            if not hasattr(torch.backends.cuda.matmul, "fp32_precision") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if not hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
        if self.config.use_flash_attention and self.device.type == "cuda":
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)

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
            attention_window=self.config.attention_window,
            use_residual_scalars=self.config.use_residual_scalars,
            residual_scale_init=self.config.residual_scale_init,
            skip_scale_init=self.config.skip_scale_init,
            use_value_embedding=self.config.use_value_embedding,
            value_embedding_every=self.config.value_embedding_every,
            value_embedding_scale=self.config.value_embedding_scale,
            use_midpoint_offsets=True,
        )
        model = build_policy(policy_cfg).to(self.device)

        if self.config.preload_checkpoint_path:
            preload_path = Path(self.config.preload_checkpoint_path)
            if preload_path.exists():
                logger.info("Preloading weights from %s", preload_path)
                checkpoint = torch_load_compat(preload_path, map_location="cpu", weights_only=False)
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
        self._weight_decay_groups = [
            (group, float(group.get("weight_decay", 0.0))) for group in self._iter_param_groups(optimizer)
        ]
        self._amp_context, self._scaler = self._build_amp()

        train_loader = self.data.train_dataloader(self.config.batch_size, self.config.num_workers)
        val_loader = self.data.val_dataloader(self.config.batch_size, self.config.num_workers)
        self._total_train_steps = max(1, len(train_loader) * max(1, self.config.epochs))
        if self.config.dry_train_steps:
            self._total_train_steps = min(
                self._total_train_steps,
                int(self.config.dry_train_steps) * max(1, self.config.epochs),
            )

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

        # Torch compile may wrap forward in CUDAGraphs; mark the start of each step to
        # avoid "overwritten output" errors when internal tensors are reused across runs.
        mark_step_begin = None
        if bool(self.config.use_compile):
            compiler = getattr(torch, "compiler", None)
            mark_step_begin = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler is not None else None

        for batch in loader:
            if mark_step_begin is not None:
                mark_step_begin()
            features = batch["features"].to(self.device)
            highs = batch["high"].to(self.device)
            lows = batch["low"].to(self.device)
            closes = batch["close"].to(self.device)
            reference_close = batch["reference_close"].to(self.device)
            chronos_high = batch["chronos_high"].to(self.device)
            chronos_low = batch["chronos_low"].to(self.device)

            with self._amp_context:
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
                    maker_fee=batch.get("maker_fee", self.config.maker_fee),
                    initial_cash=self.config.initial_cash,
                    can_short=batch.get("can_short", False),
                    can_long=batch.get("can_long", True),
                )

            returns = sim.returns.float()
            periods_per_year = batch.get("periods_per_year", None)
            if periods_per_year is None:
                periods_per_year = float(self.config.periods_per_year or HOURLY_PERIODS_PER_YEAR)
            score, sortino, annual_return = compute_hourly_objective(
                returns,
                periods_per_year=periods_per_year,
                return_weight=self.config.return_weight,
            )
            loss = combined_sortino_pnl_loss(
                returns,
                target_sign=self.config.sortino_target_sign,
                periods_per_year=periods_per_year,
                return_weight=self.config.return_weight,
            )

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    if self.config.grad_clip:
                        self._scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    self._apply_schedules(optimizer, global_step)
                    if self.config.warmup_steps and global_step < self.config.warmup_steps:
                        warmup_frac = float(global_step + 1) / float(self.config.warmup_steps)
                        for group, base_lr in zip(optimizer.param_groups, self._warmup_base_lrs):
                            group["lr"] = float(base_lr) * warmup_frac
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    if self.config.grad_clip:
                        clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    self._apply_schedules(optimizer, global_step)
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

    def _build_amp(self):
        if not self.config.use_amp or self.device.type != "cuda":
            return nullcontext(), None
        dtype_name = str(self.config.amp_dtype or "bfloat16").lower()
        if dtype_name in {"float16", "fp16"}:
            dtype = torch.float16
            use_scaler = True
        else:
            dtype = torch.bfloat16
            use_scaler = False

        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:  # pragma: no cover - legacy fallback
            amp_ctx = torch.cuda.amp.autocast(dtype=dtype)

        scaler = None
        if use_scaler:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                try:
                    scaler = torch.amp.GradScaler(device_type="cuda")
                except TypeError:  # pragma: no cover - older signature
                    scaler = torch.amp.GradScaler()
            else:  # pragma: no cover - legacy fallback
                scaler = torch.cuda.amp.GradScaler()
        return amp_ctx, scaler

    def _iter_param_groups(self, optimizer: torch.optim.Optimizer) -> List[dict]:
        if isinstance(optimizer, MultiOptim):
            groups: List[dict] = []
            for opt in optimizer.optimizers:
                groups.extend(opt.param_groups)
            return groups
        return list(optimizer.param_groups)

    def _apply_schedules(self, optimizer: torch.optim.Optimizer, global_step: int) -> None:
        if self._total_train_steps <= 0:
            return
        progress = min(max(global_step / float(self._total_train_steps), 0.0), 1.0)
        schedule = (self.config.weight_decay_schedule or "none").lower()
        if schedule in {"linear", "linear_to_zero", "linear_decay"}:
            wd_end = float(self.config.weight_decay_end)
            for group, base_wd in self._weight_decay_groups:
                if base_wd <= 0:
                    continue
                group["weight_decay"] = float(base_wd + (wd_end - base_wd) * progress)

        if self.config.muon_momentum_start is not None and self.config.muon_momentum_warmup_steps > 0:
            start = float(self.config.muon_momentum_start)
            end = float(self.config.muon_momentum)
            frac = min(global_step / float(self.config.muon_momentum_warmup_steps), 1.0)
            momentum = start + (end - start) * frac
            for group in self._iter_param_groups(optimizer):
                if "momentum" in group:
                    group["momentum"] = momentum

    def _save_checkpoint(self, model: BinancePolicyBase, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        payload = {
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "epoch": epoch,
            # Keep checkpoints portable across Python versions by avoiding Path objects.
            "config": serialize_for_checkpoint(self.config),
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
