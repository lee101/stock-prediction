from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from binanceneural.marketsimulator import SimulationConfig as MarketSimulationConfig
from binanceneural.marketsimulator import run_shared_cash_simulation
from differentiable_loss_utils import HOURLY_PERIODS_PER_YEAR
from src.robust_trading_metrics import (
    compute_market_sim_goodness_score,
    compute_pnl_smoothness,
    compute_pnl_smoothness_score,
    compute_trade_rate,
)

try:  # PyTorch >= 2.1
    from torch.amp import GradScaler as TorchGradScaler  # type: ignore[attr-defined]
    from torch.amp import autocast as torch_autocast  # type: ignore[attr-defined]

    _AMP_SUPPORTS_DEVICE = True
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore
    from torch.cuda.amp import autocast as torch_autocast  # type: ignore

    _AMP_SUPPORTS_DEVICE = False

from wandboard import WandBoardLogger

from .config import FastForecaster2Config
from .data import DataBundle, build_data_bundle
from .kernels import weighted_mae_loss
from .model import FastForecaster2Model


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


class FastForecaster2Trainer:
    """Train/evaluate FastForecaster2 with MAE-centered objectives."""

    def __init__(
        self,
        config: FastForecaster2Config,
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

        self.model = FastForecaster2Model(
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
        ).to(self.device)
        self._chronos_embeddings_loaded = self._apply_chronos_embedding_bootstrap()

        if config.torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode=config.compile_mode, fullgraph=False)
                print(f"[fastforecaster2] torch.compile enabled ({config.compile_mode}).")
            except Exception as exc:
                print(f"[fastforecaster2] torch.compile failed ({exc}); using eager mode.")

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
                self._metrics_logger.log_hparams({"fastforecaster2": config.as_dict()}, {})
            except Exception as exc:  # pragma: no cover
                print(f"[fastforecaster2] WandBoard hparam logging failed: {exc}")
                self._metrics_logger = None

        print(
            "[fastforecaster2] Ready: "
            f"device={self.device} symbols={len(self.data.symbols)} "
            f"train={len(self.data.train_dataset)} val={len(self.data.val_dataset)} test={len(self.data.test_dataset)} "
            f"chronos_embed_bootstrap={self._chronos_embeddings_loaded}"
        )

    def _load_chronos_embedding_map(self) -> dict[str, np.ndarray]:
        path = self.config.chronos_embeddings_path
        if path is None:
            return {}
        if not path.exists():
            print(f"[fastforecaster2] Chronos embedding file not found, skipping bootstrap: {path}")
            return {}

        payload: Any
        try:
            suffix = path.suffix.lower()
            if suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            elif suffix == ".npz":
                raw = np.load(path, allow_pickle=True)
                payload = {str(key): raw[key] for key in raw.files}
            elif suffix == ".npy":
                payload = np.load(path, allow_pickle=True).item()
            elif suffix in {".pt", ".pth"}:
                payload = torch.load(path, map_location="cpu")
            else:
                print(f"[fastforecaster2] Unsupported embedding format '{suffix}', skipping bootstrap.")
                return {}
        except Exception as exc:
            print(f"[fastforecaster2] Failed to load Chronos embedding file {path}: {exc}")
            return {}

        if not isinstance(payload, dict):
            print(f"[fastforecaster2] Expected mapping payload for Chronos embeddings, got {type(payload)!r}.")
            return {}

        mapping: dict[str, np.ndarray] = {}
        for key, value in payload.items():
            symbol = str(key).strip().upper()
            if not symbol:
                continue
            try:
                vector = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if vector.size == 0:
                continue
            mapping[symbol] = vector
        return mapping

    def _fit_embedding_dim(self, vector: np.ndarray, dim: int) -> np.ndarray:
        clipped = np.asarray(vector, dtype=np.float32).reshape(-1)
        if clipped.size > dim:
            clipped = clipped[:dim]
        elif clipped.size < dim:
            pad = np.zeros(dim - clipped.size, dtype=np.float32)
            clipped = np.concatenate([clipped, pad], axis=0)
        mean = float(clipped.mean())
        std = float(clipped.std())
        if std > 1e-8:
            clipped = (clipped - mean) / std
        return clipped.astype(np.float32, copy=False)

    def _apply_chronos_embedding_bootstrap(self) -> int:
        if self.config.chronos_embeddings_blend <= 0:
            return 0
        mapping = self._load_chronos_embedding_map()
        if not mapping:
            return 0

        emb = self.model.symbol_embedding.weight
        dim = int(emb.shape[1])
        blend = float(self.config.chronos_embeddings_blend)
        loaded = 0

        with torch.no_grad():
            for idx, symbol in enumerate(self.data.symbols):
                vector = mapping.get(symbol.upper())
                if vector is None:
                    continue
                fitted = self._fit_embedding_dim(vector, dim)
                fitted_tensor = torch.from_numpy(fitted).to(device=emb.device, dtype=emb.dtype)
                emb[idx].mul_(1.0 - blend).add_(fitted_tensor, alpha=blend)
                loaded += 1

        print(
            "[fastforecaster2] Chronos embedding bootstrap: "
            f"loaded={loaded}/{len(self.data.symbols)} blend={blend:.3f}"
        )
        return loaded

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
                print("[fastforecaster2] Using fused AdamW.")
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
                    f"[fastforecaster2] epoch={epoch} step={batch_idx}/{len(self.train_loader)} "
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
            print(f"[fastforecaster2] WandBoard metric logging failed: {exc}")

    @staticmethod
    def _as_utc_timestamp(raw_value: Any) -> pd.Timestamp:
        ts = pd.Timestamp(raw_value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _window_realized_vol(self, close: np.ndarray, target_idx: int) -> float:
        lookback = int(self.config.market_sim_vol_lookback)
        left = max(1, target_idx - lookback)
        if target_idx <= left:
            return 0.0
        prices = close[left - 1 : target_idx + 1].astype(np.float64, copy=False)
        if prices.size < 3:
            return 0.0
        returns = np.diff(prices) / np.clip(prices[:-1], a_min=1e-8, a_max=None)
        return float(np.std(returns))

    @staticmethod
    def _aggregate_predicted_return(predicted_path: np.ndarray) -> float:
        horizon = int(predicted_path.shape[0])
        use_steps = min(4, horizon)
        if use_steps <= 0:
            return 0.0
        weights = np.arange(use_steps, 0, -1, dtype=np.float64)
        weights = weights / np.clip(weights.sum(), a_min=1e-8, a_max=None)
        return float(np.dot(predicted_path[:use_steps].astype(np.float64, copy=False), weights))

    @staticmethod
    def _signal_intensity(
        predicted_return: float,
        realized_vol: float,
        threshold: float,
        *,
        max_trade_intensity: float,
        min_trade_intensity: float,
        vol_target: float,
        slot_count: int,
    ) -> float:
        if threshold <= 0:
            confidence = min(1.0, abs(predicted_return) * 100.0)
        else:
            confidence = min(1.0, abs(predicted_return) / threshold)
        vol_scale = min(1.0, vol_target / max(realized_vol, 1e-8))
        scaled = (max_trade_intensity / max(1, slot_count)) * confidence * vol_scale
        if scaled > 0:
            scaled = max(min_trade_intensity, scaled)
        return float(np.clip(scaled, 0.0, 100.0))

    @staticmethod
    def _plan_market_sim_actions_from_scores(
        frame: pd.DataFrame,
        *,
        buy_threshold: float,
        sell_threshold: float,
        entry_score_threshold: float,
        top_k: int,
        max_trade_intensity: float,
        min_trade_intensity: float,
        vol_target: float,
        switch_score_gap: float,
        entry_buffer_bps: float,
        exit_buffer_bps: float,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        active_symbols: set[str] = set()
        planned_chunks: list[pd.DataFrame] = []

        for _, chunk in frame.groupby("timestamp", sort=True):
            chunk = chunk.sort_values(
                ["smoothed_score", "predicted_return", "symbol"],
                ascending=[False, False, True],
            ).reset_index(drop=True)

            held_candidates = (
                chunk[chunk["symbol"].isin(active_symbols) & (chunk["smoothed_return"] >= sell_threshold)]
                .sort_values(["smoothed_score", "predicted_return", "symbol"], ascending=[False, False, True])
                .reset_index(drop=True)
            )
            new_candidates = (
                chunk[
                    ~chunk["symbol"].isin(active_symbols)
                    & (chunk["smoothed_return"] >= buy_threshold)
                    & (chunk["smoothed_score"] >= entry_score_threshold)
                ]
                .sort_values(["smoothed_score", "predicted_return", "symbol"], ascending=[False, False, True])
                .reset_index(drop=True)
            )

            desired_symbols = held_candidates["symbol"].head(top_k).tolist()
            desired_set = set(desired_symbols)

            for row in new_candidates.itertuples(index=False):
                if len(desired_symbols) >= top_k:
                    break
                symbol = str(row.symbol)
                if symbol in desired_set:
                    continue
                desired_symbols.append(symbol)
                desired_set.add(symbol)

            if switch_score_gap > 0.0 and desired_symbols:
                score_by_symbol = {
                    str(row.symbol): float(row.smoothed_score)
                    for row in chunk.itertuples(index=False)
                }
                new_queue = [
                    str(row.symbol)
                    for row in new_candidates.itertuples(index=False)
                    if str(row.symbol) not in desired_set
                ]

                while new_queue:
                    held_symbols = [symbol for symbol in desired_symbols if symbol in active_symbols]
                    if not held_symbols:
                        break
                    weakest_held = min(
                        held_symbols,
                        key=lambda symbol: (score_by_symbol.get(symbol, float("-inf")), symbol),
                    )
                    strongest_new = new_queue[0]
                    held_score = score_by_symbol.get(weakest_held, float("-inf"))
                    new_score = score_by_symbol.get(strongest_new, float("-inf"))
                    if new_score < (held_score + switch_score_gap):
                        break
                    desired_symbols = [symbol for symbol in desired_symbols if symbol != weakest_held]
                    desired_symbols.append(strongest_new)
                    desired_set = set(desired_symbols)
                    new_queue = [symbol for symbol in new_queue[1:] if symbol not in desired_set]

            slot_count = max(1, len(desired_set))
            buy_amounts: list[float] = []
            sell_amounts: list[float] = []
            buy_prices: list[float] = []
            sell_prices: list[float] = []

            for row in chunk.itertuples(index=False):
                close = float(row.close)
                predicted_return = float(row.predicted_return)
                realized_vol = float(row.realized_vol)
                symbol = str(row.symbol)
                enter_position = symbol in desired_set and symbol not in active_symbols
                exit_position = symbol in active_symbols and symbol not in desired_set

                if enter_position:
                    intensity = FastForecaster2Trainer._signal_intensity(
                        predicted_return,
                        realized_vol,
                        buy_threshold,
                        max_trade_intensity=max_trade_intensity,
                        min_trade_intensity=min_trade_intensity,
                        vol_target=vol_target,
                        slot_count=slot_count,
                    )
                    buy_amounts.append(intensity)
                    sell_amounts.append(0.0)
                    buy_prices.append(close * (1.0 + entry_buffer_bps / 10_000.0))
                    sell_prices.append(0.0)
                elif exit_position:
                    buy_amounts.append(0.0)
                    sell_amounts.append(100.0)
                    buy_prices.append(0.0)
                    sell_prices.append(max(1e-8, close * (1.0 - exit_buffer_bps / 10_000.0)))
                else:
                    buy_amounts.append(0.0)
                    sell_amounts.append(0.0)
                    buy_prices.append(0.0)
                    sell_prices.append(0.0)

            updated = chunk.copy()
            updated["buy_amount"] = buy_amounts
            updated["sell_amount"] = sell_amounts
            updated["buy_price"] = buy_prices
            updated["sell_price"] = sell_prices
            updated["desired_long"] = updated["symbol"].isin(desired_set).astype(float)
            updated["active_before"] = updated["symbol"].isin(active_symbols).astype(float)
            planned_chunks.append(updated)
            active_symbols = desired_set

        return pd.concat(planned_chunks, ignore_index=True)

    def _build_market_bar_frame(self) -> pd.DataFrame:
        rows: list[dict[str, float | str | pd.Timestamp]] = []
        for series in self.data.test_dataset.series:
            total_rows = len(series.close)
            if total_rows == 0:
                continue
            for idx in range(total_rows):
                rows.append(
                    {
                        "timestamp": self._as_utc_timestamp(series.timestamps[idx]),
                        "symbol": str(series.symbol).upper(),
                        "open": float(series.open_[idx]),
                        "high": float(series.high[idx]),
                        "low": float(series.low[idx]),
                        "close": float(series.close[idx]),
                        "realized_vol": self._window_realized_vol(series.close, idx),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _densify_market_sim_signal_frame(frame: pd.DataFrame, bars_frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or bars_frame.empty:
            return frame.copy()

        timestamps = pd.Index(sorted(frame["timestamp"].unique()), name="timestamp")
        dense_frames: list[pd.DataFrame] = []

        for symbol, bar_chunk in bars_frame.groupby("symbol", sort=False):
            signal_chunk = frame[frame["symbol"] == symbol]
            if signal_chunk.empty:
                continue

            bar_chunk = (
                bar_chunk.sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="last")
                .set_index("timestamp")
            )
            signal_chunk = (
                signal_chunk.sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="last")
                .set_index("timestamp")
            )

            common_index = timestamps.intersection(bar_chunk.index)
            if common_index.empty:
                continue

            expanded = bar_chunk.reindex(common_index)
            signal_cols = ["predicted_return", "signal_strength", "raw_symbol_idx"]
            expanded = expanded.join(signal_chunk[signal_cols], how="left")
            expanded[signal_cols] = expanded[signal_cols].ffill()
            expanded = expanded[expanded["predicted_return"].notna()]
            if expanded.empty:
                continue

            expanded["symbol"] = symbol
            dense_frames.append(expanded.reset_index())

        if not dense_frames:
            return frame.copy()
        return pd.concat(dense_frames, ignore_index=True)

    @torch.no_grad()
    def _build_market_sim_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_dataset = self.data.test_dataset
        if len(test_dataset) == 0:
            empty = pd.DataFrame()
            return empty, empty, empty

        sim_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        refs = list(test_dataset.windows)
        row_cursor = 0
        rows: list[dict[str, float | str | pd.Timestamp]] = []

        self.model.eval()
        for batch in sim_loader:
            x, _, _, base_close, symbol_idx = self._move_batch(batch)
            with self._autocast():
                pred_ret, _ = self._compute_predictions(x, base_close, symbol_idx)

            pred_path = pred_ret.detach().float().cpu().numpy()
            symbol_idx_np = symbol_idx.detach().cpu().numpy()
            batch_size = pred_path.shape[0]

            for row in range(batch_size):
                ref = refs[row_cursor + row]
                series = test_dataset.series[ref.symbol_idx]
                target_idx = int(ref.start + test_dataset.lookback)
                if target_idx < 0 or target_idx >= len(series.close):
                    continue

                predicted_return = self._aggregate_predicted_return(pred_path[row])
                realized_vol = self._window_realized_vol(series.close, target_idx)

                rows.append(
                    {
                        "timestamp": self._as_utc_timestamp(series.timestamps[target_idx]),
                        "symbol": str(series.symbol).upper(),
                        "open": float(series.open_[target_idx]),
                        "high": float(series.high[target_idx]),
                        "low": float(series.low[target_idx]),
                        "close": float(series.close[target_idx]),
                        "predicted_return": predicted_return,
                        "realized_vol": realized_vol,
                        "signal_strength": abs(predicted_return),
                        "raw_symbol_idx": float(symbol_idx_np[row]),
                    }
                )
            row_cursor += batch_size

        if not rows:
            empty = pd.DataFrame()
            return empty, empty, empty

        frame = pd.DataFrame(rows).sort_values(["timestamp", "symbol", "signal_strength"], ascending=[True, True, False])
        frame = frame.groupby(["timestamp", "symbol"], as_index=False).first()
        bars_frame = self._build_market_bar_frame()
        frame = self._densify_market_sim_signal_frame(frame, bars_frame)
        frame["signal_strength"] = np.abs(frame["predicted_return"])
        frame["raw_score"] = frame["predicted_return"] / np.maximum(frame["realized_vol"], self.config.market_sim_vol_target)
        frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        frame["smoothed_return"] = frame.groupby("symbol", sort=False)["predicted_return"].transform(
            lambda series: series.ewm(alpha=self.config.market_sim_signal_ema_alpha, adjust=False).mean()
        )
        frame["smoothed_score"] = frame.groupby("symbol", sort=False)["raw_score"].transform(
            lambda series: series.ewm(alpha=self.config.market_sim_signal_ema_alpha, adjust=False).mean()
        )
        frame = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        signal_frame = self._plan_market_sim_actions_from_scores(
            frame,
            buy_threshold=self.config.market_sim_buy_threshold,
            sell_threshold=self.config.market_sim_sell_threshold,
            entry_score_threshold=self.config.market_sim_entry_score_threshold,
            top_k=self.config.market_sim_top_k,
            max_trade_intensity=self.config.market_sim_max_trade_intensity,
            min_trade_intensity=self.config.market_sim_min_trade_intensity,
            vol_target=self.config.market_sim_vol_target,
            switch_score_gap=self.config.market_sim_switch_score_gap,
            entry_buffer_bps=self.config.market_sim_entry_buffer_bps,
            exit_buffer_bps=self.config.market_sim_exit_buffer_bps,
        )
        bars = signal_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
        actions = signal_frame[
            ["timestamp", "symbol", "buy_price", "sell_price", "buy_amount", "sell_amount"]
        ].copy()
        return bars, actions, signal_frame

    @staticmethod
    def _compute_equity_risk_metrics(equity_curve: pd.Series, *, initial_cash: float) -> dict[str, float]:
        if equity_curve.empty or len(equity_curve) < 2:
            return {
                "sim_sortino": 0.0,
                "sim_mean_hourly_return": 0.0,
                "sim_max_drawdown": 0.0,
                "sim_annualized_volatility": 0.0,
                "sim_pnl_smoothness": 0.0,
                "sim_smoothness": 0.0,
                "sim_win_rate": 0.0,
            }
        values = equity_curve.to_numpy(dtype=np.float64)
        denominator_floor = max(1.0, float(initial_cash) * 0.05)
        step_returns = np.diff(values) / np.clip(np.abs(values[:-1]), a_min=denominator_floor, a_max=None)
        step_returns = np.nan_to_num(step_returns, nan=0.0, posinf=0.0, neginf=0.0)
        step_returns = np.clip(step_returns, -1.0, 1.0)
        if step_returns.size == 0:
            return {
                "sim_sortino": 0.0,
                "sim_mean_hourly_return": 0.0,
                "sim_max_drawdown": 0.0,
                "sim_annualized_volatility": 0.0,
                "sim_pnl_smoothness": 0.0,
                "sim_smoothness": 0.0,
                "sim_win_rate": 0.0,
            }

        running_max = np.maximum.accumulate(values)
        drawdown = (running_max - values) / np.clip(running_max, a_min=1e-8, a_max=None)
        drawdown = np.clip(drawdown, 0.0, 1.0)
        mean_hourly_return = float(np.mean(step_returns))
        downside = step_returns[step_returns < 0]
        downside_std = float(np.std(downside)) if downside.size else 0.0
        annualized_vol = float(np.std(step_returns) * math.sqrt(HOURLY_PERIODS_PER_YEAR))
        sortino = (
            mean_hourly_return / downside_std * math.sqrt(HOURLY_PERIODS_PER_YEAR) if downside_std > 0 else 0.0
        )
        pnl_smoothness = float(compute_pnl_smoothness(step_returns))
        smoothness = float(compute_pnl_smoothness_score(pnl_smoothness))
        win_rate = float(np.mean(step_returns > 0))
        return {
            "sim_sortino": float(sortino),
            "sim_mean_hourly_return": mean_hourly_return,
            "sim_max_drawdown": float(np.max(drawdown)),
            "sim_annualized_volatility": annualized_vol,
            "sim_pnl_smoothness": pnl_smoothness,
            "sim_smoothness": smoothness,
            "sim_win_rate": win_rate,
        }

    @torch.no_grad()
    def _run_market_sim_eval(self) -> dict[str, float]:
        if not self.config.use_market_sim_eval:
            return {}

        bars, actions, signal_frame = self._build_market_sim_frames()
        if signal_frame.empty:
            return {"sim_trades": 0.0, "sim_signal_rows": 0.0}

        signal_frame.to_csv(self.config.simulator_actions_file, index=False)
        actionable = signal_frame[(signal_frame["buy_amount"] > 0) | (signal_frame["sell_amount"] > 0)]
        period_count = int(pd.Index(signal_frame["timestamp"]).nunique()) if "timestamp" in signal_frame.columns else 0
        if actionable.empty:
            summary = {
                "sim_total_return": 0.0,
                "sim_trades": 0.0,
                "sim_signal_rows": float(len(signal_frame)),
                "sim_active_signal_rows": 0.0,
                "sim_periods": float(period_count),
                "sim_trade_rate": 0.0,
                "sim_final_equity": float(self.config.market_sim_initial_cash),
                "sim_pnl": 0.0,
                "sim_sortino": 0.0,
                "sim_mean_hourly_return": 0.0,
                "sim_max_drawdown": 0.0,
                "sim_annualized_volatility": 0.0,
                "sim_pnl_smoothness": 0.0,
                "sim_smoothness": 1.0,
                "sim_win_rate": 0.0,
                "sim_goodness_score": compute_market_sim_goodness_score(
                    total_return=0.0,
                    sortino=0.0,
                    max_drawdown=0.0,
                    pnl_smoothness=0.0,
                    trade_rate=0.0,
                    period_count=period_count,
                ),
            }
            with open(self.config.simulator_metrics_file, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            return summary

        sim_result = run_shared_cash_simulation(
            bars,
            actions,
            config=MarketSimulationConfig(
                maker_fee=self.config.market_sim_fee,
                initial_cash=self.config.market_sim_initial_cash,
                max_hold_hours=self.config.market_sim_max_hold_hours,
            ),
        )

        equity_curve = sim_result.combined_equity.sort_index()
        if not equity_curve.empty:
            equity_df = equity_curve.rename("equity").to_frame().reset_index()
            equity_df = equity_df.rename(columns={"index": "timestamp"})
            equity_df.to_csv(self.config.simulator_equity_file, index=False)

        trade_count = float(sum(len(result.trades) for result in sim_result.per_symbol.values()))
        end_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else self.config.market_sim_initial_cash
        start_equity = (
            float(equity_curve.iloc[0]) if not equity_curve.empty else float(self.config.market_sim_initial_cash)
        )
        total_return = ((end_equity - start_equity) / start_equity) if start_equity > 0 else 0.0

        summary: dict[str, float] = {
            "sim_total_return": float(total_return),
            "sim_trades": trade_count,
            "sim_signal_rows": float(len(signal_frame)),
            "sim_active_signal_rows": float(len(actionable)),
            "sim_periods": float(period_count),
            "sim_final_equity": end_equity,
            "sim_pnl": end_equity - float(self.config.market_sim_initial_cash),
        }
        summary.update(
            self._compute_equity_risk_metrics(
                equity_curve,
                initial_cash=float(self.config.market_sim_initial_cash),
            )
        )
        summary["sim_trade_rate"] = compute_trade_rate(trade_count, period_count)
        summary["sim_goodness_score"] = compute_market_sim_goodness_score(
            total_return=float(summary.get("sim_total_return", 0.0)),
            sortino=float(summary.get("sim_sortino", 0.0)),
            max_drawdown=float(summary.get("sim_max_drawdown", 0.0)),
            pnl_smoothness=float(summary.get("sim_pnl_smoothness", 0.0)),
            trade_rate=float(summary.get("sim_trade_rate", 0.0)),
            period_count=period_count,
        )

        with open(self.config.simulator_metrics_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        return summary

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
                f"[fastforecaster2] Epoch {epoch}/{self.config.epochs} | "
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
                print(f"[fastforecaster2] Saved new best checkpoint: {self.config.best_checkpoint_path}")
            else:
                stale_epochs += 1

            if backup is not None:
                self._restore_weights(backup)

            self._save_checkpoint(self.config.last_checkpoint_path, epoch=epoch, val_mae=val_metrics.mae)

            if stale_epochs >= self.config.early_stopping_patience:
                print(
                    f"[fastforecaster2] Early stopping at epoch {epoch} "
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
            "chronos_embeddings_loaded": float(self._chronos_embeddings_loaded),
            "epoch_history": epoch_history,
        }
        sim_summary = self._run_market_sim_eval()
        summary.update(sim_summary)

        with open(self.config.metrics_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        with open(self.config.metrics_dir / "epoch_metrics.json", "w", encoding="utf-8") as fp:
            json.dump(epoch_history, fp, indent=2)
        with open(self.config.metrics_dir / "test_per_symbol.json", "w", encoding="utf-8") as fp:
            json.dump(per_symbol_metrics, fp, indent=2)

        self._log_metrics({f"summary/{k}": v for k, v in summary.items()}, step=self.config.epochs + 1)

        print("[fastforecaster2] Final summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return summary
