from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:  # PyTorch >= 2.1
    from torch.amp import GradScaler as TorchGradScaler  # type: ignore[attr-defined]
    from torch.amp import autocast as torch_autocast  # type: ignore[attr-defined]

    _AMP_SUPPORTS_DEVICE = True
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore
    from torch.cuda.amp import autocast as torch_autocast  # type: ignore

    _AMP_SUPPORTS_DEVICE = False

from wandboard import WandBoardLogger

from .config import FastForecasterConfig
from .data import DataBundle, build_data_bundle
from .kernels import weighted_mae_loss
from .model import FastForecasterModel


@dataclass
class EpochResult:
    loss: float
    mae: float
    rmse: float
    mape: float
    direction_accuracy: float
    objective_price_mae: float
    objective_return_mae: float
    objective_direction: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FastForecasterTrainer:
    """Train/evaluate FastForecaster with MAE-centered objectives."""

    def __init__(
        self,
        config: FastForecasterConfig,
        *,
        metrics_logger: Optional[WandBoardLogger] = None,
    ) -> None:
        self.config = config
        self.config.ensure_output_dirs()
        _set_seed(config.seed)

        self.device = torch.device(config.resolved_device())
        self._metrics_logger = metrics_logger
        self._optimizer_step = 0

        self._configure_cuda()
        self._amp_dtype = self._resolve_amp_dtype(config.precision)
        self._amp_enabled = self._amp_dtype is not None and self.device.type == "cuda"
        self.scaler = self._build_scaler()

        self.data: DataBundle = build_data_bundle(config)

        self.model = FastForecasterModel(
            input_dim=self.data.feature_dim,
            horizon=config.horizon,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_multiplier=config.ff_multiplier,
            dropout=config.dropout,
            max_symbols=max(1, len(self.data.symbols)),
            qk_norm=config.qk_norm,
            qk_norm_eps=config.qk_norm_eps,
        )
        self._move_model_to_device()

        if config.torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode=config.compile_mode, fullgraph=False)
                print(f"[fastforecaster] torch.compile enabled ({config.compile_mode}).")
            except Exception as exc:
                print(f"[fastforecaster] torch.compile failed ({exc}); using eager mode.")

        self.optimizer = self._build_optimizer()
        self._horizon_weights = self._build_horizon_weights()
        self._ema_params: Dict[str, torch.Tensor] | None = None
        if self.config.use_ema_eval:
            self._ema_params = {
                name: param.detach().clone().float()
                for name, param in self.model.named_parameters()
            }

        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / config.grad_accum_steps))
        self.total_optim_steps = max(1, steps_per_epoch * config.epochs)

        if self._metrics_logger is not None:
            try:
                self._metrics_logger.log_hparams({"fastforecaster": config.as_dict()}, {})
            except Exception as exc:  # pragma: no cover
                print(f"[fastforecaster] WandBoard hparam logging failed: {exc}")
                self._metrics_logger = None

        print(
            "[fastforecaster] Ready: "
            f"device={self.device} symbols={len(self.data.symbols)} "
            f"train={len(self.data.train_dataset)} val={len(self.data.val_dataset)} test={len(self.data.test_dataset)}"
        )

    def _should_fallback_to_cpu(self, exc: BaseException) -> bool:
        if self.config.device is not None or self.device.type != "cuda":
            return False
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        accelerator_error = getattr(torch, "AcceleratorError", None)
        return accelerator_error is not None and isinstance(exc, accelerator_error)

    def _move_model_to_device(self) -> None:
        try:
            self.model = self.model.to(self.device)
        except Exception as exc:
            if self._should_fallback_to_cpu(exc):
                print(f"[fastforecaster] CUDA unavailable at runtime, falling back to CPU: {exc}")
                self.device = torch.device("cpu")
                self._amp_enabled = False
                self.scaler = self._build_scaler()
                self.model = self.model.to(self.device)
            else:
                raise

    def _resolve_amp_dtype(self, precision: str) -> torch.dtype | None:
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        return None

    def _build_scaler(self):
        enabled = self._amp_enabled and self.config.precision == "fp16"
        try:
            return TorchGradScaler(self.device.type, enabled=enabled)
        except TypeError:  # pragma: no cover
            return TorchGradScaler(enabled=enabled)

    def _configure_cuda(self) -> None:
        if self.device.type != "cuda":
            return
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        elif not hasattr(torch, "set_float32_matmul_precision"):  # pragma: no cover - compatibility path
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        elif not hasattr(torch, "set_float32_matmul_precision"):  # pragma: no cover - compatibility path
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    def _build_horizon_weights(self) -> torch.Tensor:
        # Emphasize shorter-horizon errors slightly while keeping mean weight at 1.
        horizon = self.config.horizon
        if self.config.horizon_weight_power <= 0:
            return torch.ones(horizon, dtype=torch.float32)
        idx = torch.arange(1, horizon + 1, dtype=torch.float32)
        weights = idx.pow(-self.config.horizon_weight_power)
        weights = weights / torch.clamp(weights.mean(), min=1e-8)
        return weights

    def _training_objective(
        self,
        pred_ret: torch.Tensor,
        pred_price: torch.Tensor,
        target_ret: torch.Tensor,
        target_close: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self._horizon_weights.to(device=pred_price.device, dtype=pred_price.dtype)
        weights_view = weights.view(1, -1)
        price_mae = weighted_mae_loss(
            pred_price,
            target_close,
            weights,
            use_cpp=self.config.use_cpp_kernels,
            build_extension=self.config.build_cpp_extension,
        )
        return_mae = weighted_mae_loss(
            pred_ret,
            target_ret,
            weights,
            use_cpp=self.config.use_cpp_kernels,
            build_extension=self.config.build_cpp_extension,
        )

        signed_target = torch.where(target_ret >= 0, torch.ones_like(target_ret), -torch.ones_like(target_ret))
        signed_margin = pred_ret * signed_target * self.config.direction_margin_scale
        direction_loss = torch.mean(F.softplus(-signed_margin) * weights_view)

        total = price_mae
        if self.config.return_loss_weight > 0:
            total = total + (self.config.return_loss_weight * return_mae)
        if self.config.direction_loss_weight > 0:
            total = total + (self.config.direction_loss_weight * direction_loss)
        return total, price_mae, return_mae, direction_loss

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self._ema_params is None:
            return
        decay = self.config.ema_decay
        one_minus = 1.0 - decay
        for name, param in self.model.named_parameters():
            ema = self._ema_params.get(name)
            if ema is None:
                continue
            ema.mul_(decay).add_(param.detach().float(), alpha=one_minus)

    @torch.no_grad()
    def _swap_in_ema_weights(self) -> Dict[str, torch.Tensor] | None:
        if self._ema_params is None:
            return None
        backup: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            backup[name] = param.detach().clone()
            ema = self._ema_params.get(name)
            if ema is None:
                continue
            param.data.copy_(ema.to(device=param.device, dtype=param.dtype))
        return backup

    @torch.no_grad()
    def _restore_weights(self, backup: Dict[str, torch.Tensor] | None) -> None:
        if not backup:
            return
        for name, param in self.model.named_parameters():
            source = backup.get(name)
            if source is None:
                continue
            param.data.copy_(source.to(device=param.device, dtype=param.dtype))

    def _autocast(self):
        if not self._amp_enabled or self._amp_dtype is None:
            return nullcontext()
        if _AMP_SUPPORTS_DEVICE:
            return torch_autocast(self.device.type, dtype=self._amp_dtype, enabled=True)
        return torch_autocast(dtype=self._amp_dtype, enabled=True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and "norm" not in name and not name.endswith("bias"):
                decay.append(param)
            else:
                no_decay.append(param)

        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = self.config.use_fused_optimizer and self.device.type == "cuda"

        try:
            optimizer = torch.optim.AdamW(groups, lr=self.config.learning_rate, betas=(0.9, 0.95), fused=fused)
            if fused:
                print("[fastforecaster] Using fused AdamW.")
            return optimizer
        except TypeError:
            return torch.optim.AdamW(groups, lr=self.config.learning_rate, betas=(0.9, 0.95))

    def _build_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        base_args = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory and self.device.type == "cuda",
            "drop_last": False,
        }
        if self.config.num_workers > 0:
            base_args["persistent_workers"] = True
            base_args["prefetch_factor"] = 2

        train_loader = DataLoader(self.data.train_dataset, shuffle=True, **base_args)
        val_loader = DataLoader(self.data.val_dataset, shuffle=False, **base_args)
        test_loader = DataLoader(self.data.test_dataset, shuffle=False, **base_args)
        return train_loader, val_loader, test_loader

    def _lr_at_step(self, step: int) -> float:
        warmup = self.config.warmup_steps
        max_lr = self.config.learning_rate
        min_lr = self.config.min_learning_rate

        if step < warmup:
            return max_lr * float(step + 1) / float(max(1, warmup))

        progress = float(step - warmup) / float(max(1, self.total_optim_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cosine

    def _set_optimizer_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _move_batch(self, batch):
        x, target_ret, target_close, base_close, symbol_idx = batch
        x = x.to(self.device, non_blocking=True)
        target_ret = target_ret.to(self.device, non_blocking=True)
        target_close = target_close.to(self.device, non_blocking=True)
        base_close = base_close.to(self.device, non_blocking=True)
        symbol_idx = symbol_idx.to(self.device, non_blocking=True)
        return x, target_ret, target_close, base_close, symbol_idx

    def _compute_predictions(
        self,
        x: torch.Tensor,
        base_close: torch.Tensor,
        symbol_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_return = self.model(x, symbol_idx)
        pred_price = base_close.unsqueeze(1) * (1.0 + pred_return)
        return pred_return, pred_price

    def _run_epoch(self, epoch: int) -> EpochResult:
        self.model.train()
        running_loss = 0.0
        mae_sum = 0.0
        mse_sum = 0.0
        mape_sum = 0.0
        dir_hits = 0.0
        obj_price_sum = 0.0
        obj_return_sum = 0.0
        obj_direction_sum = 0.0
        element_count = 0
        sample_count = 0

        grad_accum = self.config.grad_accum_steps
        accum_counter = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            x, target_ret, target_close, base_close, symbol_idx = self._move_batch(batch)

            with self._autocast():
                pred_ret, pred_price = self._compute_predictions(x, base_close, symbol_idx)
                loss, price_obj, return_obj, direction_obj = self._training_objective(
                    pred_ret,
                    pred_price,
                    target_ret,
                    target_close,
                )

            batch_loss = float(loss.detach().item())
            scaled_loss = loss / grad_accum

            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_counter += 1
            if accum_counter == grad_accum:
                lr = self._lr_at_step(self._optimizer_step)
                self._set_optimizer_lr(lr)

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self._update_ema()
                self.optimizer.zero_grad(set_to_none=True)

                self._optimizer_step += 1
                accum_counter = 0

            with torch.no_grad():
                error = pred_price - target_close
                abs_error = torch.abs(error)
                mae_sum += float(abs_error.sum().item())
                mse_sum += float(torch.square(error).sum().item())
                mape_sum += float((abs_error / torch.clamp(torch.abs(target_close), min=1e-5)).sum().item())
                dir_hits += float((torch.sign(pred_ret[:, 0]) == torch.sign(target_ret[:, 0])).float().sum().item())
                obj_price_sum += float(price_obj.detach().item()) * x.shape[0]
                obj_return_sum += float(return_obj.detach().item()) * x.shape[0]
                obj_direction_sum += float(direction_obj.detach().item()) * x.shape[0]

                running_loss += batch_loss * x.shape[0]
                element_count += int(pred_price.numel())
                sample_count += int(x.shape[0])

            if batch_idx % max(1, self.config.log_interval) == 0:
                avg_mae = mae_sum / max(1, element_count)
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[fastforecaster] epoch={epoch} step={batch_idx}/{len(self.train_loader)} "
                    f"loss={batch_loss:.6f} train_mae={avg_mae:.6f} lr={lr:.3e}"
                )

        if accum_counter > 0:
            lr = self._lr_at_step(self._optimizer_step)
            self._set_optimizer_lr(lr)
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self._update_ema()
            self.optimizer.zero_grad(set_to_none=True)
            self._optimizer_step += 1

        avg_loss = running_loss / max(1, sample_count)
        mae = mae_sum / max(1, element_count)
        rmse = math.sqrt(mse_sum / max(1, element_count))
        mape = (mape_sum / max(1, element_count)) * 100.0
        direction = dir_hits / max(1, sample_count)
        return EpochResult(
            loss=avg_loss,
            mae=mae,
            rmse=rmse,
            mape=mape,
            direction_accuracy=direction,
            objective_price_mae=(obj_price_sum / max(1, sample_count)),
            objective_return_mae=(obj_return_sum / max(1, sample_count)),
            objective_direction=(obj_direction_sum / max(1, sample_count)),
        )

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> EpochResult:
        self.model.eval()
        loss_sum = 0.0
        mae_sum = 0.0
        mse_sum = 0.0
        mape_sum = 0.0
        dir_hits = 0.0
        obj_price_sum = 0.0
        obj_return_sum = 0.0
        obj_direction_sum = 0.0
        element_count = 0
        sample_count = 0

        for batch in loader:
            x, target_ret, target_close, base_close, symbol_idx = self._move_batch(batch)
            with self._autocast():
                pred_ret, pred_price = self._compute_predictions(x, base_close, symbol_idx)
                loss, price_obj, return_obj, direction_obj = self._training_objective(
                    pred_ret,
                    pred_price,
                    target_ret,
                    target_close,
                )

            error = pred_price - target_close
            abs_error = torch.abs(error)

            loss_sum += float(loss.item()) * x.shape[0]
            mae_sum += float(abs_error.sum().item())
            mse_sum += float(torch.square(error).sum().item())
            mape_sum += float((abs_error / torch.clamp(torch.abs(target_close), min=1e-5)).sum().item())
            dir_hits += float((torch.sign(pred_ret[:, 0]) == torch.sign(target_ret[:, 0])).float().sum().item())
            obj_price_sum += float(price_obj.detach().item()) * x.shape[0]
            obj_return_sum += float(return_obj.detach().item()) * x.shape[0]
            obj_direction_sum += float(direction_obj.detach().item()) * x.shape[0]
            element_count += int(pred_price.numel())
            sample_count += int(x.shape[0])

        avg_loss = loss_sum / max(1, sample_count)
        mae = mae_sum / max(1, element_count)
        rmse = math.sqrt(mse_sum / max(1, element_count))
        mape = (mape_sum / max(1, element_count)) * 100.0
        direction = dir_hits / max(1, sample_count)
        return EpochResult(
            loss=avg_loss,
            mae=mae,
            rmse=rmse,
            mape=mape,
            direction_accuracy=direction,
            objective_price_mae=(obj_price_sum / max(1, sample_count)),
            objective_return_mae=(obj_return_sum / max(1, sample_count)),
            objective_direction=(obj_direction_sum / max(1, sample_count)),
        )

    @torch.no_grad()
    def _evaluate_per_symbol(self, loader: DataLoader) -> list[dict[str, float]]:
        self.model.eval()
        stats: dict[str, dict[str, float]] = {}

        for batch in loader:
            x, target_ret, target_close, base_close, symbol_idx = self._move_batch(batch)
            with self._autocast():
                pred_ret, pred_price = self._compute_predictions(x, base_close, symbol_idx)

            error = pred_price - target_close
            abs_error = torch.abs(error)
            sq_error = torch.square(error)
            rel_error = abs_error / torch.clamp(torch.abs(target_close), min=1e-5)
            dir_hit = (torch.sign(pred_ret[:, 0]) == torch.sign(target_ret[:, 0])).float()

            for row in range(x.shape[0]):
                sym = self.data.symbols[int(symbol_idx[row].item())]
                state = stats.setdefault(
                    sym,
                    {
                        "sum_abs": 0.0,
                        "sum_sq": 0.0,
                        "sum_rel": 0.0,
                        "count": 0.0,
                        "dir_hits": 0.0,
                        "samples": 0.0,
                    },
                )
                state["sum_abs"] += float(abs_error[row].sum().item())
                state["sum_sq"] += float(sq_error[row].sum().item())
                state["sum_rel"] += float(rel_error[row].sum().item())
                state["count"] += float(abs_error[row].numel())
                state["dir_hits"] += float(dir_hit[row].item())
                state["samples"] += 1.0

        per_symbol: list[dict[str, float]] = []
        for symbol, state in sorted(stats.items()):
            count = max(1.0, state["count"])
            samples = max(1.0, state["samples"])
            per_symbol.append(
                {
                    "symbol": symbol,
                    "mae": state["sum_abs"] / count,
                    "rmse": math.sqrt(state["sum_sq"] / count),
                    "mape": (state["sum_rel"] / count) * 100.0,
                    "direction_acc": state["dir_hits"] / samples,
                    "samples": samples,
                }
            )
        per_symbol.sort(key=lambda item: item["mae"])
        return per_symbol

    def _save_checkpoint(self, path: Path, *, epoch: int, val_mae: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "val_mae": float(val_mae),
            "model_state": self.model.state_dict(),
            "config": self.config.as_dict(),
            "feature_names": list(self.data.feature_names),
            "symbols": list(self.data.symbols),
            "feature_mean": self.data.feature_mean.tolist(),
            "feature_std": self.data.feature_std.tolist(),
        }
        torch.save(payload, path)

    def _log_metrics(self, metrics: Dict[str, float], *, step: int | None = None) -> None:
        if self._metrics_logger is None:
            return
        try:
            self._metrics_logger.log(metrics, step=step)
        except Exception as exc:  # pragma: no cover
            print(f"[fastforecaster] WandBoard metric logging failed: {exc}")

    def train(self) -> Dict[str, float]:
        best_val_mae = float("inf")
        best_epoch = 0
        stale_epochs = 0
        train_start = time.time()
        epoch_history: list[Dict[str, float]] = []

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(epoch)
            if self.config.use_ema_eval:
                backup = self._swap_in_ema_weights()
                val_metrics = self._evaluate(self.val_loader)
            else:
                backup = None
                val_metrics = self._evaluate(self.val_loader)

            print(
                f"[fastforecaster] Epoch {epoch}/{self.config.epochs} | "
                f"train_mae={train_metrics.mae:.6f} val_mae={val_metrics.mae:.6f} "
                f"val_rmse={val_metrics.rmse:.6f} val_mape={val_metrics.mape:.3f}%"
            )

            self._log_metrics(
                {
                    "epoch": float(epoch),
                    "train/loss": float(train_metrics.loss),
                    "train/mae": float(train_metrics.mae),
                    "train/rmse": float(train_metrics.rmse),
                    "train/mape": float(train_metrics.mape),
                    "train/direction_acc": float(train_metrics.direction_accuracy),
                    "train/objective_price_mae": float(train_metrics.objective_price_mae),
                    "train/objective_return_mae": float(train_metrics.objective_return_mae),
                    "train/objective_direction": float(train_metrics.objective_direction),
                    "val/loss": float(val_metrics.loss),
                    "val/mae": float(val_metrics.mae),
                    "val/rmse": float(val_metrics.rmse),
                    "val/mape": float(val_metrics.mape),
                    "val/direction_acc": float(val_metrics.direction_accuracy),
                    "val/objective_price_mae": float(val_metrics.objective_price_mae),
                    "val/objective_return_mae": float(val_metrics.objective_return_mae),
                    "val/objective_direction": float(val_metrics.objective_direction),
                    "optim/step": float(self._optimizer_step),
                    "optim/lr": float(self.optimizer.param_groups[0]["lr"]),
                },
                step=epoch,
            )

            epoch_history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_metrics.loss),
                    "train_mae": float(train_metrics.mae),
                    "train_rmse": float(train_metrics.rmse),
                    "train_mape": float(train_metrics.mape),
                    "train_direction_acc": float(train_metrics.direction_accuracy),
                    "train_objective_price_mae": float(train_metrics.objective_price_mae),
                    "train_objective_return_mae": float(train_metrics.objective_return_mae),
                    "train_objective_direction": float(train_metrics.objective_direction),
                    "val_loss": float(val_metrics.loss),
                    "val_mae": float(val_metrics.mae),
                    "val_rmse": float(val_metrics.rmse),
                    "val_mape": float(val_metrics.mape),
                    "val_direction_acc": float(val_metrics.direction_accuracy),
                    "val_objective_price_mae": float(val_metrics.objective_price_mae),
                    "val_objective_return_mae": float(val_metrics.objective_return_mae),
                    "val_objective_direction": float(val_metrics.objective_direction),
                    "optimizer_step": float(self._optimizer_step),
                    "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                }
            )

            if val_metrics.mae + 1e-12 < best_val_mae:
                best_val_mae = val_metrics.mae
                best_epoch = epoch
                stale_epochs = 0
                self._save_checkpoint(self.config.best_checkpoint_path, epoch=epoch, val_mae=best_val_mae)
                print(f"[fastforecaster] Saved new best checkpoint: {self.config.best_checkpoint_path}")
            else:
                stale_epochs += 1

            if backup is not None:
                self._restore_weights(backup)

            self._save_checkpoint(self.config.last_checkpoint_path, epoch=epoch, val_mae=val_metrics.mae)

            if stale_epochs >= self.config.early_stopping_patience:
                print(
                    f"[fastforecaster] Early stopping at epoch {epoch} "
                    f"(no val MAE improvement for {stale_epochs} epochs)."
                )
                break

        # Reload best checkpoint before final test evaluation.
        if self.config.best_checkpoint_path.exists():
            ckpt = torch.load(self.config.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])

        test_metrics = self._evaluate(self.test_loader)
        per_symbol_metrics = self._evaluate_per_symbol(self.test_loader)
        duration_min = (time.time() - train_start) / 60.0

        summary = {
            "best_epoch": float(best_epoch),
            "best_val_mae": float(best_val_mae),
            "test_mae": float(test_metrics.mae),
            "test_rmse": float(test_metrics.rmse),
            "test_mape": float(test_metrics.mape),
            "test_direction_acc": float(test_metrics.direction_accuracy),
            "training_minutes": float(duration_min),
            "optimizer_steps": float(self._optimizer_step),
            "symbols": float(len(self.data.symbols)),
            "train_windows": float(len(self.data.train_dataset)),
            "val_windows": float(len(self.data.val_dataset)),
            "test_windows": float(len(self.data.test_dataset)),
            "objective_return_loss_weight": float(self.config.return_loss_weight),
            "objective_direction_loss_weight": float(self.config.direction_loss_weight),
            "ema_eval": float(self.config.use_ema_eval),
            "epoch_history": epoch_history,
        }

        with open(self.config.metrics_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        with open(self.config.metrics_dir / "epoch_metrics.json", "w", encoding="utf-8") as fp:
            json.dump(epoch_history, fp, indent=2)
        with open(self.config.metrics_dir / "test_per_symbol.json", "w", encoding="utf-8") as fp:
            json.dump(per_symbol_metrics, fp, indent=2)

        self._log_metrics({f"summary/{k}": v for k, v in summary.items()}, step=self.config.epochs + 1)

        print("[fastforecaster] Final summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return summary
