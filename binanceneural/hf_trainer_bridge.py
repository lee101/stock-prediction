from __future__ import annotations

import json
import math
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments

from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    compute_loss_by_type,
    simulate_hourly_trades,
    simulate_hourly_trades_binary,
)
try:
    from trainingefficiency.fast_differentiable_sim import simulate_hourly_trades_fast
except ImportError:
    simulate_hourly_trades_fast = None
try:
    from trainingefficiency.triton_sim_kernel import simulate_hourly_trades_triton
    HAS_TRITON_SIM = True
except ImportError:
    simulate_hourly_trades_triton = None
    HAS_TRITON_SIM = False

from src.checkpoint_manager import TopKCheckpointManager
from src.serialization_utils import serialize_for_checkpoint
from src.torch_load_utils import torch_load_compat
from wandboard import WandBoardLogger

from .config import TrainingConfig
from .data import MultiSymbolDataModule
from .model import PolicyConfig, build_policy

logger = logging.getLogger(__name__)


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


@dataclass
class TrainEpochMetrics:
    loss: float
    score: float
    sortino: float
    annual_return: float

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.loss),
            "score": float(self.score),
            "sortino": float(self.sortino),
            "return": float(self.annual_return),
        }


class UnifiedPolicyHFModel(nn.Module):
    def __init__(self, config: TrainingConfig, input_dim: int) -> None:
        super().__init__()
        self.training_config = config
        self.policy = build_policy(
            PolicyConfig(
                input_dim=input_dim,
                hidden_dim=config.transformer_dim,
                dropout=config.transformer_dropout,
                price_offset_pct=config.price_offset_pct,
                min_price_gap_pct=config.min_price_gap_pct,
                trade_amount_scale=config.trade_amount_scale,
                num_heads=config.transformer_heads,
                num_layers=config.transformer_layers,
                max_len=max(config.sequence_length, 32),
                model_arch=config.model_arch,
                num_kv_heads=config.num_kv_heads,
                mlp_ratio=config.mlp_ratio,
                logits_softcap=config.logits_softcap,
                rope_base=config.rope_base,
                use_qk_norm=config.use_qk_norm,
                use_causal_attention=config.use_causal_attention,
                rms_norm_eps=config.rms_norm_eps,
                attention_window=config.attention_window,
                attention_backend=config.attention_backend,
                flex_block_size=config.flex_block_size,
                use_residual_scalars=config.use_residual_scalars,
                residual_scale_init=config.residual_scale_init,
                skip_scale_init=config.skip_scale_init,
                use_value_embedding=config.use_value_embedding,
                value_embedding_every=config.value_embedding_every,
                value_embedding_scale=config.value_embedding_scale,
                use_midpoint_offsets=True,
                num_outputs=config.num_outputs,
                max_hold_hours=config.max_hold_hours,
                num_memory_tokens=config.num_memory_tokens,
                dilated_strides=config.dilated_strides,
                use_flex_attention=config.use_flex_attention,
            )
        )
        self._preload_checkpoint()

    def _preload_checkpoint(self) -> None:
        preload_path = self.training_config.preload_checkpoint_path
        if not preload_path:
            return
        path = Path(preload_path)
        if not path.exists():
            raise FileNotFoundError(f"Preload checkpoint not found: {path}")
        checkpoint = torch_load_compat(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        model_state = self.policy.state_dict()
        filtered = {}
        skipped: list[str] = []
        for key, value in state_dict.items():
            target = model_state.get(key)
            if target is None or tuple(target.shape) != tuple(value.shape):
                skipped.append(key)
                continue
            filtered[key] = value
        missing, unexpected = self.policy.load_state_dict(filtered, strict=False)
        logger.info(
            "HF preload loaded %s/%s tensors from %s",
            len(filtered),
            len(state_dict),
            path,
        )
        if skipped:
            logger.warning("HF preload skipped %s mismatched tensors", len(skipped))
        if missing:
            logger.warning("HF preload missing keys after partial load: %s", list(missing))
        if unexpected:
            logger.warning("HF preload unexpected keys after partial load: %s", list(unexpected))

    def _simulator(self):
        if HAS_TRITON_SIM and next(self.policy.parameters()).device.type == "cuda":
            return simulate_hourly_trades_triton
        if bool(self.training_config.use_vectorized_sim) and simulate_hourly_trades_fast is not None:
            return simulate_hourly_trades_fast
        return simulate_hourly_trades

    def forward(
        self,
        features: torch.Tensor,
        open: torch.Tensor | None = None,
        high: torch.Tensor | None = None,
        low: torch.Tensor | None = None,
        close: torch.Tensor | None = None,
        reference_close: torch.Tensor | None = None,
        chronos_high: torch.Tensor | None = None,
        chronos_low: torch.Tensor | None = None,
        can_long: torch.Tensor | float | None = None,
        can_short: torch.Tensor | float | None = None,
        periods_per_year: torch.Tensor | float | None = None,
        maker_fee: torch.Tensor | float | None = None,
        return_loss: bool = True,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        del return_loss
        cfg = self.training_config
        if high is None or low is None or close is None or reference_close is None:
            raise ValueError("Expected high/low/close/reference_close tensors in the batch")
        if chronos_high is None or chronos_low is None:
            raise ValueError("Expected chronos_high/chronos_low tensors in the batch")

        if self.training and cfg.feature_noise_std > 0:
            features = features + cfg.feature_noise_std * torch.randn_like(features)

        outputs = self.policy(features)
        actions = self.policy.decode_actions(
            outputs,
            reference_close=reference_close,
            chronos_high=chronos_high,
            chronos_low=chronos_low,
        )

        sim_kwargs = {
            "highs": high,
            "lows": low,
            "closes": close,
            "opens": open,
            "buy_prices": actions["buy_price"],
            "sell_prices": actions["sell_price"],
            "trade_intensity": actions["trade_amount"] / cfg.trade_amount_scale,
            "buy_trade_intensity": actions["buy_amount"] / cfg.trade_amount_scale,
            "sell_trade_intensity": actions["sell_amount"] / cfg.trade_amount_scale,
            "maker_fee": maker_fee if maker_fee is not None else cfg.maker_fee,
            "initial_cash": cfg.initial_cash,
            "can_short": can_short if can_short is not None else False,
            "can_long": can_long if can_long is not None else True,
            "max_leverage": cfg.max_leverage,
            "market_order_entry": cfg.market_order_entry,
            "fill_buffer_pct": cfg.fill_buffer_pct,
            "margin_annual_rate": float(cfg.margin_annual_rate),
        }
        sim_fn = self._simulator()
        lag_values = [int(x) for x in cfg.decision_lag_range.split(",") if x.strip()] if cfg.decision_lag_range.strip() else [int(cfg.decision_lag_bars)]

        if self.training and len(lag_values) > 1:
            losses: list[torch.Tensor] = []
            scores: list[torch.Tensor] = []
            sortinos: list[torch.Tensor] = []
            returns_list: list[torch.Tensor] = []
            for lag in lag_values:
                sim = sim_fn(
                    **sim_kwargs,
                    temperature=float(cfg.fill_temperature),
                    decision_lag_bars=lag,
                )
                result = compute_loss_by_type(
                    sim.returns.float(),
                    cfg.loss_type,
                    target_sign=cfg.sortino_target_sign,
                    periods_per_year=periods_per_year if periods_per_year is not None else float(cfg.periods_per_year or HOURLY_PERIODS_PER_YEAR),
                    return_weight=cfg.return_weight,
                    smoothness_penalty=cfg.smoothness_penalty,
                    dd_penalty=cfg.dd_penalty,
                    multiwindow_fractions=cfg.multiwindow_fractions,
                    multiwindow_aggregation=cfg.multiwindow_aggregation,
                )
                losses.append(result[0])
                scores.append(result[1])
                sortinos.append(result[2])
                returns_list.append(result[3])
            loss = sum(losses) / len(losses)
            score = sum(scores) / len(scores)
            sortino = sum(sortinos) / len(sortinos)
            annual_return = sum(returns_list) / len(returns_list)
        else:
            lag = max(lag_values) if not self.training else int(cfg.decision_lag_bars)
            if (not self.training) and bool(cfg.validation_use_binary_fills):
                sim = simulate_hourly_trades_binary(**sim_kwargs, decision_lag_bars=lag)
            else:
                sim = sim_fn(
                    **sim_kwargs,
                    temperature=float(cfg.fill_temperature),
                    decision_lag_bars=lag,
                )
            loss, score, sortino, annual_return = compute_loss_by_type(
                sim.returns.float(),
                cfg.loss_type,
                target_sign=cfg.sortino_target_sign,
                periods_per_year=periods_per_year if periods_per_year is not None else float(cfg.periods_per_year or HOURLY_PERIODS_PER_YEAR),
                return_weight=cfg.return_weight,
                smoothness_penalty=cfg.smoothness_penalty,
                dd_penalty=cfg.dd_penalty,
                multiwindow_fractions=cfg.multiwindow_fractions,
                multiwindow_aggregation=cfg.multiwindow_aggregation,
            )

        if self.training and cfg.spread_penalty > 0:
            buy_gap = (actions["buy_price"] - low[..., : actions["buy_price"].shape[-1]]) / close[..., : actions["buy_price"].shape[-1]].clamp(min=1e-4)
            sell_gap = (high[..., : actions["sell_price"].shape[-1]] - actions["sell_price"]) / close[..., : actions["sell_price"].shape[-1]].clamp(min=1e-4)
            buy_pen = torch.relu(cfg.spread_target - buy_gap).mean()
            sell_pen = torch.relu(cfg.spread_target - sell_gap).mean()
            loss = loss + cfg.spread_penalty * (buy_pen + sell_pen)

        batch_size = int(features.shape[0])
        metric_row = torch.stack(
            [
                score.detach().mean().reshape(()),
                sortino.detach().mean().reshape(()),
                annual_return.detach().mean().reshape(()),
            ]
        ).unsqueeze(0)
        metrics = metric_row.expand(batch_size, -1).contiguous()
        return {"loss": loss, "metrics": metrics}


class _EpochMetricCallback(TrainerCallback):
    def __init__(self, trainer: "UnifiedPolicyHFTrainer") -> None:
        self._trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        self._trainer.reset_train_epoch_metrics()

    def on_epoch_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        self._trainer.finalize_train_epoch_metrics(state.epoch)


def compute_unified_policy_eval_metrics(eval_pred) -> dict[str, float]:
    values = np.asarray(eval_pred.predictions)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return {
        "score": float(np.mean(values[:, 0])),
        "sortino": float(np.mean(values[:, 1])),
        "return": float(np.mean(values[:, 2])),
    }


class UnifiedPolicyHFTrainer(Trainer):
    def __init__(
        self,
        *,
        train_config: TrainingConfig,
        data_module: MultiSymbolDataModule,
        checkpoint_dir: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.training_config = train_config
        self.data_module = data_module
        self.checkpoint_dir = checkpoint_dir
        self.ckpt_mgr = TopKCheckpointManager(
            checkpoint_dir,
            max_keep=max(1, int(train_config.top_k_checkpoints)),
            mode="max",
        )
        self.best_metric = float("-inf")
        self.best_checkpoint: Path | None = None
        self.last_train_epoch_metrics: TrainEpochMetrics | None = None
        self._train_metric_sums = torch.zeros(4, dtype=torch.float64)
        self._train_metric_steps = 0
        self.model_accepts_loss_kwargs = False
        self.metrics_logger: WandBoardLogger | None = None

    def _wandb_enabled(self) -> bool:
        return bool(self.training_config.wandb_project or os.getenv("WANDB_PROJECT"))

    def _wandb_tags(self) -> tuple[str, ...]:
        raw = str(getattr(self.training_config, "wandb_tags", "") or "")
        return tuple(token.strip() for token in raw.split(",") if token.strip())

    def reset_train_epoch_metrics(self) -> None:
        self._train_metric_sums.zero_()
        self._train_metric_steps = 0

    def finalize_train_epoch_metrics(self, epoch: float | None) -> None:
        del epoch
        if self._train_metric_steps <= 0:
            self.last_train_epoch_metrics = None
            return
        means = (self._train_metric_sums / float(self._train_metric_steps)).tolist()
        self.last_train_epoch_metrics = TrainEpochMetrics(
            loss=float(means[0]),
            score=float(means[1]),
            sortino=float(means[2]),
            annual_return=float(means[3]),
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        del num_items_in_batch
        outputs = model(**inputs)
        loss = outputs["loss"]
        if model.training:
            metrics = outputs["metrics"][0]
            self._train_metric_sums += torch.tensor(
                [
                    float(loss.detach().mean().cpu()),
                    float(metrics[0].detach().cpu()),
                    float(metrics[1].detach().cpu()),
                    float(metrics[2].detach().cpu()),
                ],
                dtype=torch.float64,
            )
            self._train_metric_steps += 1
        return (loss, outputs) if return_outputs else loss

    def _checkpoint_metric(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> tuple[float, str, float]:
        metric_name = str(self.training_config.checkpoint_metric or "robust_score").strip().lower()
        gap_penalty = float(self.training_config.checkpoint_gap_penalty)
        score_gap = max(float(train_metrics["score"]) - float(val_metrics["score"]), 0.0)
        sortino_gap = max(float(train_metrics["sortino"]) - float(val_metrics["sortino"]), 0.0)
        if metric_name == "val_score":
            return float(val_metrics["score"]), metric_name, score_gap
        if metric_name == "val_sortino":
            return float(val_metrics["sortino"]), metric_name, sortino_gap
        if metric_name == "val_return":
            return float(val_metrics["return"]), metric_name, score_gap
        if metric_name == "robust_sortino":
            return float(val_metrics["sortino"]) - gap_penalty * sortino_gap, metric_name, sortino_gap
        return float(val_metrics["score"]) - gap_penalty * score_gap, "robust_score", score_gap

    def _refresh_best_checkpoint_alias(self, checkpoint_path: Path) -> None:
        alias_path = self.checkpoint_dir / "best.pt"
        try:
            if alias_path.exists() or alias_path.is_symlink():
                alias_path.unlink()
            alias_path.symlink_to(checkpoint_path.name)
        except OSError:
            shutil.copy2(checkpoint_path, alias_path)

    def _save_portable_checkpoint(self, epoch: int, metrics: dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, UnifiedPolicyHFModel):
            raise TypeError(f"Expected UnifiedPolicyHFModel, got {type(model)!r}")
        payload = {
            "state_dict": model.policy.state_dict(),
            "metrics": metrics,
            "epoch": int(epoch),
            "config": serialize_for_checkpoint(self.training_config),
            "feature_columns": list(self.data_module.feature_columns),
            "normalizer": self.data_module.normalizer.to_dict(),
            "trainer_backend": "transformers_trainer",
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        torch.save(payload, path)
        return path

    def _write_progress(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        checkpoint_metric: float,
        checkpoint_metric_name: str,
        generalization_gap: float,
    ) -> None:
        payload = {
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_name": self.training_config.run_name,
            "epoch": int(epoch),
            "epochs": int(self.training_config.epochs),
            "checkpoint_metric_name": checkpoint_metric_name,
            "checkpoint_metric": float(checkpoint_metric),
            "checkpoint_gap_penalty": float(self.training_config.checkpoint_gap_penalty),
            "generalization_gap": float(generalization_gap),
            "train_metrics": {key: float(value) for key, value in train_metrics.items()},
            "val_metrics": {key: float(value) for key, value in val_metrics.items()},
            "best_metric": float(self.best_metric),
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
        }
        (self.checkpoint_dir / "training_progress.json").write_text(json.dumps(payload, indent=2))

    def evaluate(self, *args: Any, **kwargs: Any):
        metrics = super().evaluate(*args, **kwargs)
        epoch = max(1, int(math.ceil(float(self.state.epoch or 0.0))))
        if epoch <= 0:
            return metrics
        train_metrics = (
            self.last_train_epoch_metrics.as_dict()
            if self.last_train_epoch_metrics is not None
            else {"loss": float("nan"), "score": float(metrics.get("eval_score", 0.0)), "sortino": float(metrics.get("eval_sortino", 0.0)), "return": float(metrics.get("eval_return", 0.0))}
        )
        val_metrics = {
            "loss": float(metrics.get("eval_loss", 0.0)),
            "score": float(metrics.get("eval_score", 0.0)),
            "sortino": float(metrics.get("eval_sortino", 0.0)),
            "return": float(metrics.get("eval_return", 0.0)),
        }
        checkpoint_metric, checkpoint_metric_name, generalization_gap = self._checkpoint_metric(train_metrics, val_metrics)
        ckpt_path = self._save_portable_checkpoint(epoch, val_metrics)
        self.ckpt_mgr.register(ckpt_path, checkpoint_metric, epoch=epoch)
        if checkpoint_metric > self.best_metric:
            self.best_metric = checkpoint_metric
            self.best_checkpoint = ckpt_path
            self._refresh_best_checkpoint_alias(ckpt_path)
        self._write_progress(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            checkpoint_metric=checkpoint_metric,
            checkpoint_metric_name=checkpoint_metric_name,
            generalization_gap=generalization_gap,
        )
        if self.metrics_logger is not None:
            self.metrics_logger.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/score": train_metrics["score"],
                    "train/sortino": train_metrics["sortino"],
                    "train/return": train_metrics["return"],
                    "val/loss": val_metrics["loss"],
                    "val/score": val_metrics["score"],
                    "val/sortino": val_metrics["sortino"],
                    "val/return": val_metrics["return"],
                    "val/generalization_gap": generalization_gap,
                    f"val/{checkpoint_metric_name}": checkpoint_metric,
                },
                step=epoch,
            )
        return metrics

    def train(self, *args: Any, **kwargs: Any):
        with WandBoardLogger(
            run_name=self.training_config.run_name,
            project=self.training_config.wandb_project,
            entity=self.training_config.wandb_entity,
            group=self.training_config.wandb_group,
            tags=self._wandb_tags(),
            notes=self.training_config.wandb_notes,
            mode=self.training_config.wandb_mode,
            enable_wandb=self._wandb_enabled(),
            log_dir=self.training_config.log_dir,
            tensorboard_subdir=self.training_config.run_name,
            config=serialize_for_checkpoint(self.training_config),
            log_metrics=self.training_config.wandb_log_metrics,
        ) as metrics_logger:
            self.metrics_logger = metrics_logger
            metrics_logger.log_text(
                "train/feature_columns",
                json.dumps(list(self.data_module.feature_columns)),
                step=0,
            )
            result = super().train(*args, **kwargs)
            if self.best_checkpoint is not None:
                best_progress = json.loads((self.checkpoint_dir / "training_progress.json").read_text())
                best_payload = torch_load_compat(self.best_checkpoint, map_location="cpu", weights_only=False)
                best_metrics = best_payload.get("metrics", {})
                metrics_logger.log_hparams(
                    {
                        "run_name": self.training_config.run_name,
                        "epochs": int(self.training_config.epochs),
                        "batch_size": int(self.training_config.batch_size),
                        "sequence_length": int(self.training_config.sequence_length),
                        "transformer_dim": int(self.training_config.transformer_dim),
                        "transformer_heads": int(self.training_config.transformer_heads),
                        "transformer_layers": int(self.training_config.transformer_layers),
                        "learning_rate": float(self.training_config.learning_rate),
                        "weight_decay": float(self.training_config.weight_decay),
                        "fill_temperature": float(self.training_config.fill_temperature),
                        "return_weight": float(self.training_config.return_weight),
                        "checkpoint_metric": str(self.training_config.checkpoint_metric),
                    },
                    {
                        "best/metric": float(best_progress["best_metric"]),
                        "best/val_score": float(best_metrics.get("score", best_progress["val_metrics"]["score"])),
                        "best/val_sortino": float(best_metrics.get("sortino", best_progress["val_metrics"]["sortino"])),
                        "best/val_return": float(best_metrics.get("return", best_progress["val_metrics"]["return"])),
                    },
                    step=int(best_progress["epoch"]),
                    table_name="hf_train_summary",
                )
            self.metrics_logger = None
            return result


def make_training_arguments(
    *,
    output_dir: Path,
    run_name: str,
    batch_size: int,
    epochs: int,
    max_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    grad_clip: float,
    accumulation_steps: int,
    bf16: bool,
    fp16: bool,
    tf32: bool,
    torch_compile: bool,
    num_workers: int,
    logging_steps: int,
    optim_name: str,
    report_to: list[str] | None = None,
    use_cpu: bool | None = None,
) -> TrainingArguments:
    def _build_kwargs(*, resolved_use_cpu: bool, resolved_bf16: bool, resolved_fp16: bool, resolved_tf32: bool) -> dict[str, Any]:
        return {
            "output_dir": str(output_dir),
            "run_name": run_name,
            "per_device_train_batch_size": int(batch_size),
            "per_device_eval_batch_size": int(batch_size),
            "num_train_epochs": float(epochs),
            "max_steps": int(max_steps),
            "learning_rate": float(learning_rate),
            "warmup_steps": int(warmup_steps),
            "weight_decay": float(weight_decay),
            "gradient_accumulation_steps": int(accumulation_steps),
            "max_grad_norm": float(grad_clip),
            "bf16": bool(resolved_bf16),
            "fp16": bool(resolved_fp16),
            "tf32": bool(resolved_tf32),
            "torch_compile": bool(torch_compile),
            "optim": optim_name,
            "eval_strategy": "epoch",
            "save_strategy": "no",
            "logging_strategy": "steps",
            "logging_steps": max(1, int(logging_steps)),
            "report_to": report_to or ["none"],
            "remove_unused_columns": False,
            "dataloader_num_workers": int(num_workers),
            "dataloader_pin_memory": torch.cuda.is_available() and not resolved_use_cpu,
            "dataloader_persistent_workers": bool(num_workers > 0),
            "do_train": True,
            "do_eval": True,
            "use_cpu": bool(resolved_use_cpu),
        }

    requested_use_cpu = bool(use_cpu) if use_cpu is not None else False
    base_kwargs = _build_kwargs(
        resolved_use_cpu=requested_use_cpu,
        resolved_bf16=bf16,
        resolved_fp16=fp16,
        resolved_tf32=tf32,
    )
    try:
        return TrainingArguments(**base_kwargs)
    except Exception as exc:
        if use_cpu is None and _is_cuda_resource_pressure_error(exc):
            logger.warning(
                "Falling back to CPU for HF trainer arguments after CUDA resource error: %s",
                exc,
            )
            fallback_kwargs = _build_kwargs(
                resolved_use_cpu=True,
                resolved_bf16=False,
                resolved_fp16=False,
                resolved_tf32=False,
            )
            return TrainingArguments(**fallback_kwargs)
        raise


def write_run_metadata(
    *,
    checkpoint_dir: Path,
    train_config: TrainingConfig,
    data_module: MultiSymbolDataModule,
    symbols: list[str],
) -> None:
    config_payload = {
        "symbols": symbols,
        "sequence_length": train_config.sequence_length,
        "feature_columns": list(data_module.feature_columns),
        "transformer_dim": train_config.transformer_dim,
        "transformer_heads": train_config.transformer_heads,
        "transformer_layers": train_config.transformer_layers,
        "model_arch": train_config.model_arch,
        "trainer_backend": "transformers_trainer",
        "checkpoint_metric": train_config.checkpoint_metric,
        "checkpoint_gap_penalty": train_config.checkpoint_gap_penalty,
        "normalizer": data_module.normalizer.to_dict(),
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    meta_payload = {
        **config_payload,
        "run_name": train_config.run_name,
        "epochs": train_config.epochs,
        "batch_size": train_config.batch_size,
        "learning_rate": train_config.learning_rate,
        "weight_decay": train_config.weight_decay,
        "seed": train_config.seed,
    }
    (checkpoint_dir / "training_meta.json").write_text(json.dumps(meta_payload, indent=2))
