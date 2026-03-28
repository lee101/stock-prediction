from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from differentiable_loss_utils import HOURLY_PERIODS_PER_YEAR
from src.serialization_utils import serialize_for_checkpoint
from wandboard import WandBoardLogger

from .config import PolicyConfig, TrainingConfig
from .data import BinanceHourlyDataModule, FeatureNormalizer, MultiSymbolDataModule
from .jax_losses import (
    combined_sortino_pnl_loss,
    compute_hourly_objective,
    simulate_hourly_trades,
    simulate_hourly_trades_binary,
)
from .jax_policy import (
    JaxClassicPolicy,
    build_classic_policy_config,
    convert_torch_classic_state_dict,
    decode_actions_jax,
    init_classic_params,
)


@dataclass
class JaxTrainingHistoryEntry:
    epoch: int
    train_loss: float
    train_score: float
    train_sortino: float
    train_return: float
    val_loss: float | None = None
    val_score: float | None = None
    val_sortino: float | None = None
    val_return: float | None = None


@dataclass
class JaxTrainingArtifacts:
    params: dict[str, Any]
    normalizer: FeatureNormalizer
    history: list[JaxTrainingHistoryEntry] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    config: TrainingConfig | None = None
    checkpoint_paths: list[Path] = field(default_factory=list)
    best_checkpoint: Path | None = None


def _tensor_to_jax(value: Any) -> jax.Array:
    if isinstance(value, torch.Tensor):
        return jnp.asarray(value.detach().cpu().numpy(), dtype=jnp.float32)
    return jnp.asarray(value, dtype=jnp.float32)


def _batch_to_jax(batch: dict[str, Any]) -> dict[str, jax.Array]:
    return {key: _tensor_to_jax(value) for key, value in batch.items()}


class JaxClassicTrainer:
    def __init__(self, config: TrainingConfig, data_module: BinanceHourlyDataModule | MultiSymbolDataModule) -> None:
        if (config.model_arch or "classic").lower() != "classic":
            raise ValueError("JaxClassicTrainer supports only model_arch=classic")
        self.config = config
        self.data = data_module
        run_name = config.run_name or time.strftime("jax_classic_%Y%m%d_%H%M%S")
        self.config.run_name = run_name
        self.checkpoint_dir = self.config.checkpoint_root / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state_dict = None
        payload: dict[str, Any] = {}
        if self.config.preload_checkpoint_path:
            preload_path = Path(self.config.preload_checkpoint_path)
            if preload_path.exists() and preload_path.suffix == ".pt":
                checkpoint = torch.load(preload_path, map_location="cpu", weights_only=False)
                payload = checkpoint.get("config", {}) or {}
                state_dict = checkpoint.get("state_dict", checkpoint)

        self.policy_config = build_classic_policy_config(
            payload or serialize_for_checkpoint(self.config),
            input_dim=len(self.data.feature_columns),
            state_dict=state_dict,
        )
        self.policy_config.hidden_dim = int(self.config.transformer_dim)
        self.policy_config.num_heads = int(self.config.transformer_heads)
        self.policy_config.num_layers = int(self.config.transformer_layers)
        self.policy_config.dropout = float(self.config.transformer_dropout)
        self.policy_config.max_len = max(int(self.config.sequence_length), int(self.policy_config.max_len))
        self.policy_config.trade_amount_scale = float(self.config.trade_amount_scale)
        self.policy_config.max_hold_hours = float(self.config.max_hold_hours)

        self.model = JaxClassicPolicy(self.policy_config)
        self.rng = jax.random.PRNGKey(int(self.config.seed))
        if state_dict is not None:
            self.params = convert_torch_classic_state_dict(state_dict, config=self.policy_config)
        else:
            self.params = init_classic_params(
                config=self.policy_config,
                rng=self.rng,
                sequence_length=int(self.config.sequence_length),
            )

        self.optimizer = optax.adamw(
            learning_rate=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.opt_state = self.optimizer.init(self.params)
        self._train_step = self._build_train_step()
        self._eval_step = self._build_eval_step()

    def _wandb_enabled(self) -> bool:
        return bool(self.config.wandb_project or os.getenv("WANDB_PROJECT"))

    def _wandb_tags(self) -> tuple[str, ...]:
        raw = str(getattr(self.config, "wandb_tags", "") or "")
        return tuple(token.strip() for token in raw.split(",") if token.strip())

    def _loss_and_metrics(
        self,
        params: dict[str, Any],
        batch: dict[str, jax.Array],
        *,
        train: bool,
        dropout_rng: jax.Array | None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        rngs = {"dropout": dropout_rng} if (train and dropout_rng is not None) else None
        outputs = self.model.apply({"params": params}, batch["features"], deterministic=not train, rngs=rngs)
        actions = decode_actions_jax(
            outputs,
            reference_close=batch["reference_close"],
            chronos_high=batch["chronos_high"],
            chronos_low=batch["chronos_low"],
            price_offset_pct=float(self.policy_config.price_offset_pct),
            min_price_gap_pct=float(self.policy_config.min_price_gap_pct),
            trade_amount_scale=float(self.policy_config.trade_amount_scale),
            use_midpoint_offsets=bool(self.policy_config.use_midpoint_offsets),
            max_hold_hours=float(self.policy_config.max_hold_hours),
        )

        scale = float(self.config.trade_amount_scale)
        sim_kwargs = {
            "highs": batch["high"],
            "lows": batch["low"],
            "closes": batch["close"],
            "opens": batch.get("open"),
            "buy_prices": actions["buy_price"],
            "sell_prices": actions["sell_price"],
            "trade_intensity": actions["trade_amount"] / scale,
            "buy_trade_intensity": actions["buy_amount"] / scale,
            "sell_trade_intensity": actions["sell_amount"] / scale,
            "maker_fee": float(self.config.maker_fee),
            "initial_cash": float(self.config.initial_cash),
            "can_short": batch.get("can_short", jnp.zeros(batch["close"].shape[:-1], dtype=jnp.float32)),
            "can_long": batch.get("can_long", jnp.ones(batch["close"].shape[:-1], dtype=jnp.float32)),
            "max_leverage": float(self.config.max_leverage),
            "decision_lag_bars": int(self.config.decision_lag_bars),
            "market_order_entry": bool(self.config.market_order_entry),
            "fill_buffer_pct": float(self.config.fill_buffer_pct),
            "margin_annual_rate": float(self.config.margin_annual_rate),
        }
        if train:
            sim = simulate_hourly_trades(
                **sim_kwargs,
                temperature=float(self.config.fill_temperature),
            )
        else:
            sim = simulate_hourly_trades_binary(**sim_kwargs)

        periods_per_year = batch.get("periods_per_year")
        if periods_per_year is None:
            periods_per_year = float(self.config.periods_per_year or HOURLY_PERIODS_PER_YEAR)
        score, sortino, annual_return = compute_hourly_objective(
            sim.returns,
            periods_per_year=periods_per_year,
            return_weight=float(self.config.return_weight),
            smoothness_penalty=float(self.config.smoothness_penalty),
        )
        loss = combined_sortino_pnl_loss(
            sim.returns,
            target_sign=float(self.config.sortino_target_sign),
            periods_per_year=periods_per_year,
            return_weight=float(self.config.return_weight),
            smoothness_penalty=float(self.config.smoothness_penalty),
        )
        metrics = {
            "loss": loss,
            "score": score.mean(),
            "sortino": sortino.mean(),
            "return": annual_return.mean(),
        }
        return loss, metrics

    def _build_train_step(self):
        @jax.jit
        def _step(
            params: dict[str, Any],
            opt_state: optax.OptState,
            batch: dict[str, jax.Array],
            rng: jax.Array,
        ) -> tuple[dict[str, Any], optax.OptState, dict[str, jax.Array]]:
            dropout_rng, next_rng = jax.random.split(rng)

            def _loss_fn(p):
                return self._loss_and_metrics(p, batch, train=True, dropout_rng=dropout_rng)

            (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            metrics = dict(metrics)
            metrics["loss"] = loss
            metrics["next_rng"] = next_rng
            return params, opt_state, metrics

        return _step

    def _build_eval_step(self):
        @jax.jit
        def _step(params: dict[str, Any], batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
            _, metrics = self._loss_and_metrics(params, batch, train=False, dropout_rng=None)
            return metrics

        return _step

    def _run_epoch(self, *, train: bool) -> dict[str, float]:
        loader = (
            self.data.train_dataloader(self.config.batch_size, self.config.num_workers)
            if train
            else self.data.val_dataloader(self.config.batch_size, self.config.num_workers)
        )
        totals = {"loss": 0.0, "score": 0.0, "sortino": 0.0, "return": 0.0}
        steps = 0
        for raw_batch in loader:
            batch = _batch_to_jax(raw_batch)
            if train:
                self.params, self.opt_state, metrics = self._train_step(self.params, self.opt_state, batch, self.rng)
                self.rng = metrics.pop("next_rng")
            else:
                metrics = self._eval_step(self.params, batch)
            for key in totals:
                totals[key] += float(metrics[key])
            steps += 1
            if self.config.dry_train_steps and steps >= self.config.dry_train_steps:
                break
        if steps == 0:
            raise RuntimeError("No batches available for training/validation")
        return {key: value / steps for key, value in totals.items()}

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.flax"
        payload = {
            "params": self.params,
            "metrics": metrics,
            "epoch": epoch,
            "config": serialize_for_checkpoint(self.config),
        }
        path.write_bytes(serialization.to_bytes(payload))
        return path

    def train(self) -> JaxTrainingArtifacts:
        history: list[JaxTrainingHistoryEntry] = []
        best_score = float("-inf")
        best_checkpoint: Path | None = None
        checkpoint_paths: list[Path] = []
        with WandBoardLogger(
            run_name=self.config.run_name,
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            group=self.config.wandb_group,
            tags=self._wandb_tags(),
            notes=self.config.wandb_notes,
            mode=self.config.wandb_mode,
            enable_wandb=self._wandb_enabled(),
            log_dir=self.config.log_dir,
            tensorboard_subdir=self.config.run_name,
            config=serialize_for_checkpoint(self.config),
            log_metrics=self.config.wandb_log_metrics,
        ) as metrics_logger:
            metrics_logger.log_text(
                "jax/feature_columns",
                json.dumps(list(self.data.feature_columns)),
                step=0,
            )
            for epoch in range(1, int(self.config.epochs) + 1):
                train_metrics = self._run_epoch(train=True)
                val_metrics = self._run_epoch(train=False)
                history.append(
                    JaxTrainingHistoryEntry(
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
                )
                metrics_logger.log(
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
                    },
                    step=epoch,
                )
                ckpt_path = self._save_checkpoint(epoch, val_metrics)
                checkpoint_paths.append(ckpt_path)
                if val_metrics["score"] > best_score:
                    best_score = val_metrics["score"]
                    best_checkpoint = ckpt_path

            if history:
                best_entry = max(history, key=lambda item: item.val_score or float("-inf"))
                metrics_logger.log_hparams(
                    {
                        "run_name": self.config.run_name,
                        "epochs": int(self.config.epochs),
                        "sequence_length": int(self.config.sequence_length),
                        "transformer_dim": int(self.config.transformer_dim),
                        "transformer_heads": int(self.config.transformer_heads),
                        "transformer_layers": int(self.config.transformer_layers),
                        "decision_lag_bars": int(self.config.decision_lag_bars),
                        "fill_temperature": float(self.config.fill_temperature),
                        "return_weight": float(self.config.return_weight),
                    },
                    {
                        "best/val_score": best_entry.val_score or float("-inf"),
                        "best/val_sortino": best_entry.val_sortino or float("-inf"),
                        "best/val_return": best_entry.val_return or float("-inf"),
                    },
                    step=best_entry.epoch,
                    table_name="jax_train_summary",
                )

        config_path = self.checkpoint_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "feature_columns": list(self.data.feature_columns),
                    "normalizer": self.data.normalizer.to_dict(),
                    "sequence_length": int(self.config.sequence_length),
                    "transformer_dim": int(self.config.transformer_dim),
                    "transformer_heads": int(self.config.transformer_heads),
                    "transformer_layers": int(self.config.transformer_layers),
                    "model_arch": "classic",
                    "num_outputs": int(self.policy_config.num_outputs),
                },
                indent=2,
            )
        )
        meta_path = self.checkpoint_dir / "training_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "run_name": self.config.run_name,
                    "epochs": int(self.config.epochs),
                    "sequence_length": int(self.config.sequence_length),
                    "transformer_dim": int(self.config.transformer_dim),
                    "transformer_heads": int(self.config.transformer_heads),
                    "transformer_layers": int(self.config.transformer_layers),
                    "model_arch": "classic",
                    "feature_columns": list(self.data.feature_columns),
                    "decision_lag_bars": int(self.config.decision_lag_bars),
                    "return_weight": float(self.config.return_weight),
                    "fill_temperature": float(self.config.fill_temperature),
                    "history": [entry.__dict__ for entry in history],
                    "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
                },
                indent=2,
            )
        )

        return JaxTrainingArtifacts(
            params=self.params,
            normalizer=self.data.normalizer,
            history=history,
            feature_columns=list(self.data.feature_columns),
            config=self.config,
            checkpoint_paths=checkpoint_paths,
            best_checkpoint=best_checkpoint,
        )
