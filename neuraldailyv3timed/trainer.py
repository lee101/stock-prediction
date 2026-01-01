"""V3 Timed Trainer with trade episode simulation.

Key V3 features:
- Trade episode simulation instead of continuous inventory
- Model outputs exit_days (1-10) for maximum hold duration
- Temperature annealing still used for differentiable fills
- Dual optimizer support (Muon + AdamW)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from neuraldailyv3timed.config import (
    DailyDatasetConfigV3,
    DailyTrainingConfigV3,
    PolicyConfigV3,
    SimulationConfig,
    TemperatureSchedule,
)
from neuraldailyv3timed.data import DailyDataModuleV3
from neuraldailyv3timed.model import MultiSymbolPolicyV3, create_group_mask
from neuraldailyv3timed.simulation import compute_episode_loss, simulate_trade_episode

# Try to import Muon optimizer
try:
    from nanochat.nanochat.muon import Muon
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    logger.warning("Muon optimizer not available, falling back to AdamW")


class NeuralDailyTrainerV3:
    """V3 trainer with trade episode simulation and explicit exit timing."""

    def __init__(
        self,
        config: DailyTrainingConfigV3,
        data_module: DailyDataModuleV3,
    ):
        self.config = config
        self.data = data_module
        self.temp_schedule = config.get_temperature_schedule()
        self.sim_config = config.get_simulation_config()

        # Set device
        if config.device:
            self.device = torch.device(config.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Enable TF32 for faster training on Ampere+
        if config.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Prepare data
        self.data.prepare()

        # Build model
        policy_config = config.get_policy_config(self.data.input_dim)
        self.model = MultiSymbolPolicyV3(policy_config).to(self.device)

        # Optionally compile model
        if config.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode=config.compile_mode)
            logger.info(f"Model compiled with mode={config.compile_mode}")

        # Build optimizer(s)
        self.optimizers = self._build_optimizers()

        # Build scheduler
        self.scheduler = self._build_scheduler()

        # AMP setup
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.amp_dtype = getattr(torch, config.amp_dtype, torch.bfloat16)
        self.scaler = GradScaler() if self.use_amp and self.amp_dtype == torch.float16 else None

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_root) / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = float("-inf")

        logger.info(f"V3 Timed Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Temperature schedule: {self.temp_schedule.initial_temp} -> {self.temp_schedule.final_temp}")
        logger.info(f"Max hold days: {self.sim_config.max_hold_days}")

    def _build_optimizers(self) -> List[torch.optim.Optimizer]:
        """Build optimizer(s) - dual setup if Muon available."""
        config = self.config

        if config.optimizer_name == "dual" and MUON_AVAILABLE:
            return self._build_dual_optimizers()
        elif config.optimizer_name == "muon" and MUON_AVAILABLE:
            return [Muon(self.model.parameters(), lr=config.matrix_lr, momentum=config.muon_momentum)]
        else:
            # Fall back to AdamW
            return [AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.adamw_betas,
                eps=config.adamw_eps,
            )]

    def _build_dual_optimizers(self) -> List[torch.optim.Optimizer]:
        """Build Muon + AdamW dual optimizer setup from nanochat."""
        config = self.config

        # Separate parameters by type
        matrix_params = []
        embed_params = []
        head_params = list(self.model.head.parameters())

        for name, param in self.model.named_parameters():
            if "embed" in name:
                embed_params.append(param)
            elif param.ndim == 2 and "head" not in name:
                matrix_params.append(param)
            elif "head" not in name:
                # 1D params (biases, norms) go to AdamW
                embed_params.append(param)

        # LR scaling by model dimension (from nanochat)
        dmodel_scale = (config.transformer_dim / 768) ** -0.5

        # AdamW for embeddings and head
        adamw_groups = [
            {"params": head_params, "lr": config.head_lr * dmodel_scale},
            {"params": embed_params, "lr": config.embed_lr * dmodel_scale},
        ]
        adamw = AdamW(
            adamw_groups,
            betas=config.adamw_betas,
            eps=config.adamw_eps,
            weight_decay=0.0,
        )

        # Muon for matrix layers
        muon = Muon(matrix_params, lr=config.matrix_lr, momentum=config.muon_momentum)

        logger.info(f"Dual optimizer: Muon ({len(matrix_params)} params), AdamW ({len(embed_params) + len(head_params)} params)")
        return [adamw, muon]

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler with warmup."""
        if not self.config.use_cosine_schedule:
            return None

        # Estimate total steps
        train_dataset = self.data.get_train_dataset()
        steps_per_epoch = len(train_dataset) // self.config.batch_size
        total_steps = steps_per_epoch * self.config.epochs

        # Create warmup + cosine schedule for first optimizer
        warmup = LinearLR(
            self.optimizers[0],
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizers[0],
            T_max=total_steps - self.config.warmup_steps,
            eta_min=1e-6,
        )
        return SequentialLR(
            self.optimizers[0],
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_steps],
        )

    def train(self) -> Dict[str, Any]:
        """Run full training loop."""
        config = self.config

        # Load checkpoint if specified
        if config.preload_checkpoint_path and not config.force_retrain:
            self._load_checkpoint(Path(config.preload_checkpoint_path))

        # Get data loaders
        train_loader = self.data.get_train_loader(
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        val_loader = self.data.get_val_loader(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # Get symbol group mapping for cross-attention
        symbol_group_ids = self.data.get_symbol_group_ids()

        logger.info(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)}")
        logger.info(f"Symbols: {len(self.data.symbols)}")

        history = {
            "train_loss": [], "val_loss": [],
            "val_sharpe": [], "val_tp_rate": [], "val_avg_hold": [],
            "temperature": []
        }

        for epoch in range(self.current_epoch, config.epochs):
            self.current_epoch = epoch
            temperature = self.temp_schedule.get_temperature(epoch, config.epochs)

            # Training
            train_metrics = self._run_epoch(
                train_loader,
                train=True,
                temperature=temperature,
                symbol_group_ids=symbol_group_ids,
            )

            # Validation (with binary fills like inference)
            val_metrics = self._run_epoch(
                val_loader,
                train=False,
                temperature=0.0,  # Binary fills for validation
                symbol_group_ids=symbol_group_ids,
            )

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Train Loss {train_metrics['loss']:.4f} | "
                f"Val Loss {val_metrics['loss']:.4f} | "
                f"Val Sharpe {val_metrics['sharpe']:.4f} | "
                f"TP Rate {val_metrics['tp_rate']:.2%} | "
                f"Avg Hold {val_metrics['avg_hold_days']:.1f}d | "
                f"Temp {temperature:.6f}"
            )

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_sharpe"].append(val_metrics["sharpe"])
            history["val_tp_rate"].append(val_metrics["tp_rate"])
            history["val_avg_hold"].append(val_metrics["avg_hold_days"])
            history["temperature"].append(temperature)

            # Save checkpoint if improved (use sharpe as validation score)
            val_score = val_metrics["sharpe"]
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self._save_checkpoint(epoch, val_metrics)

            # Early stopping check
            if config.dry_train_steps and self.global_step >= config.dry_train_steps:
                logger.info(f"Dry run complete at step {self.global_step}")
                break

        # Save final manifest
        self._save_manifest()

        return history

    def _run_epoch(
        self,
        loader,
        *,
        train: bool,
        temperature: float,
        symbol_group_ids: Dict[str, int],
    ) -> Dict[str, float]:
        """Run one epoch of training or validation."""
        config = self.config

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_sharpe = 0.0
        total_tp_rate = 0.0
        total_forced_exit_rate = 0.0
        total_avg_hold = 0.0
        total_avg_return = 0.0
        num_batches = 0

        context = torch.no_grad() if not train else torch.enable_grad()

        with context:
            for batch in loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Get features and targets
                features = batch["features"]
                reference_close = batch["reference_close"]
                chronos_high = batch["chronos_high"]
                chronos_low = batch["chronos_low"]
                asset_class = batch["asset_class"]

                # V3: Get future OHLC for trade episode simulation
                future_highs = batch["future_highs"]
                future_lows = batch["future_lows"]
                future_closes = batch["future_closes"]

                # Create group mask for cross-attention
                group_ids = batch.get("group_id")
                group_mask = None
                if group_ids is not None and config.use_cross_attention:
                    group_mask = create_group_mask(group_ids)

                # Forward pass with AMP
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    # Get model outputs
                    outputs = self.model(features, group_mask=group_mask)

                    # Decode actions
                    actions = self.model.decode_actions(
                        outputs,
                        reference_close=reference_close,
                        chronos_high=chronos_high,
                        chronos_low=chronos_low,
                        asset_class=asset_class,
                    )

                    # V3: Run trade episode simulation
                    # Use predictions from last timestep (the decision point)
                    episode_result = simulate_trade_episode(
                        future_highs=future_highs,
                        future_lows=future_lows,
                        future_closes=future_closes,
                        buy_price=actions["buy_price"][:, -1],
                        sell_price=actions["sell_price"][:, -1],
                        exit_days=actions["exit_days"][:, -1],
                        trade_amount=actions["trade_amount"][:, -1],
                        config=self.sim_config,
                        temperature=temperature,
                    )

                    # V3: Compute trade episode loss
                    loss, metrics = compute_episode_loss(
                        episode_result,
                        exit_days=actions["exit_days"][:, -1],
                        asset_class=asset_class,
                        config=self.sim_config,
                        return_weight=config.return_weight,
                        forced_exit_penalty=config.forced_exit_penalty,
                        risk_penalty=config.risk_penalty,
                        hold_time_penalty=config.hold_time_penalty,
                    )

                # Backward pass (training only)
                if train:
                    for opt in self.optimizers:
                        opt.zero_grad()

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizers[0])
                        nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                        for opt in self.optimizers:
                            self.scaler.step(opt)
                        self.scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                        for opt in self.optimizers:
                            opt.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.global_step += 1

                # Accumulate metrics
                total_loss += metrics["loss"]
                total_sharpe += metrics["sharpe"]
                total_tp_rate += metrics["tp_rate"]
                total_forced_exit_rate += metrics["forced_exit_rate"]
                total_avg_hold += metrics["avg_hold_days"]
                total_avg_return += metrics["avg_return"]
                num_batches += 1

        return {
            "loss": total_loss / max(1, num_batches),
            "sharpe": total_sharpe / max(1, num_batches),
            "tp_rate": total_tp_rate / max(1, num_batches),
            "forced_exit_rate": total_forced_exit_rate / max(1, num_batches),
            "avg_hold_days": total_avg_hold / max(1, num_batches),
            "avg_return": total_avg_return / max(1, num_batches),
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> Path:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dicts": [opt.state_dict() for opt in self.optimizers],
            "best_val_score": self.best_val_score,
            "global_step": self.global_step,
            "config": asdict(self.config),
            "policy_config": asdict(self.config.get_policy_config(self.data.input_dim)),
            "feature_columns": self.data.feature_columns,
            "normalizer": {
                "mean": self.data.normalizer.mean.tolist(),
                "std": self.data.normalizer.std.tolist(),
            },
            "metrics": metrics,
            "version": "v3",  # V3 version marker
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path} (sharpe={metrics['sharpe']:.4f}, tp_rate={metrics['tp_rate']:.2%})")
        return checkpoint_path

    def _load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        for opt, state_dict in zip(self.optimizers, checkpoint["optimizer_state_dicts"]):
            opt.load_state_dict(state_dict)

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_score = checkpoint.get("best_val_score", float("-inf"))

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def _save_manifest(self) -> None:
        """Save training manifest with checkpoint info."""
        # Convert config to JSON-serializable format
        config_dict = asdict(self.config)

        def _make_serializable(obj):
            """Recursively convert Path objects to strings."""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_serializable(x) for x in obj]
            return obj

        manifest = {
            "config": _make_serializable(config_dict),
            "feature_columns": self.data.feature_columns,
            "checkpoints": [],
            "version": "v3",
        }

        # List all checkpoints
        for ckpt_path in sorted(self.checkpoint_dir.glob("epoch_*.pt")):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                manifest["checkpoints"].append({
                    "path": ckpt_path.name,
                    "sharpe": ckpt["metrics"]["sharpe"],
                    "tp_rate": ckpt["metrics"]["tp_rate"],
                    "epoch": ckpt["epoch"],
                    "timestamp": ckpt_path.stat().st_mtime,
                })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {ckpt_path}: {e}")

        manifest_path = self.checkpoint_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(f"Manifest saved: {manifest_path}")
