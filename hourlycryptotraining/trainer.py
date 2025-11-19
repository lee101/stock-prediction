from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.nn.utils import clip_grad_norm_

from differentiable_loss_utils import compute_hourly_objective, simulate_hourly_trades, simulate_hourly_trades_binary
from wandboard import WandBoardLogger

from .config import TrainingConfig
from .data import FeatureNormalizer, HourlyCryptoDataModule, MultiSymbolDataModule
from .model import HourlyCryptoPolicy, PolicyHeadConfig
from .optimizers import Muon
from .checkpoints import CheckpointRecord, save_checkpoint, write_manifest


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


class HourlyCryptoTrainer:
    def __init__(self, config: TrainingConfig, data_module: HourlyCryptoDataModule) -> None:
        self.config = config
        self.data = data_module
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        run_name = self.config.run_name or time.strftime("hourlycrypto_%Y%m%d_%H%M%S")
        self.config.run_name = run_name
        self.checkpoint_dir = self.config.checkpoint_root / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_records: List[CheckpointRecord] = []
        self.best_checkpoint_path: Optional[Path] = None

    def train(self) -> TrainingArtifacts:
        torch.manual_seed(self.config.seed)
        # Enable TF32 for faster matmul on Ampere+ GPUs
        if self.config.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model = HourlyCryptoPolicy(
            PolicyHeadConfig(
                input_dim=len(self.data.feature_columns),
                hidden_dim=self.config.transformer_dim,
                dropout=self.config.transformer_dropout,
                price_offset_pct=self.config.price_offset_pct,
                max_trade_qty=self.config.max_trade_qty,
                min_price_gap_pct=self.config.min_price_gap_pct,
                num_heads=self.config.transformer_heads,
                num_layers=self.config.transformer_layers,
            )
        ).to(self.device)
        # Compile model for 2x speedup (disabled for long sequences due to CUDA RNG overflow)
        if self.config.use_compile:
            model = torch.compile(model, mode="max-autotune")
        optimizer = self._build_optimizer(model)
        use_ema = self.config.ema_decay and self.config.ema_decay > 0
        ema_state: Optional[Dict[str, torch.Tensor]] = None
        if use_ema:
            ema_state = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        train_loader = self.data.train_dataloader(self.config.batch_size, self.config.num_workers)
        val_loader = self.data.val_dataloader(self.config.batch_size, self.config.num_workers)
        scheduler = self._build_scheduler(optimizer, train_loader)
        # Setup AMP scaler if using mixed precision
        scaler = None
        if self.config.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        history: List[TrainingHistoryEntry] = []
        best_state = None
        best_score = float("-inf")
        wandb_kwargs = {
            "run_name": self.config.run_name,
            "project": self.config.wandb_project,
            "entity": self.config.wandb_entity,
            "log_dir": self.config.log_dir,
            "tensorboard_subdir": self.config.run_name or "hourlycrypto",
            "log_metrics": True,
        }
        with WandBoardLogger(**wandb_kwargs) as tracker:
            tracker.watch(model)
            global_step = 0
            stop_requested = False
            for epoch in range(1, self.config.epochs + 1):
                train_metrics, global_step = self._run_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    train=True,
                    global_step=global_step,
                    ema_state=ema_state,
                    scaler=scaler,
                )
                val_metrics, _ = self._run_epoch(
                    model,
                    val_loader,
                    optimizer=None,
                    scheduler=None,
                    train=False,
                    global_step=global_step,
                    scaler=None,
                )

                # Per-symbol validation for multi-symbol training
                per_symbol_metrics = {}
                if isinstance(self.data, MultiSymbolDataModule):
                    for symbol in self.data.symbols:
                        symbol_loader = self.data.modules[symbol].val_dataloader(
                            self.config.batch_size, self.config.num_workers
                        )
                        symbol_metrics, _ = self._run_epoch(
                            model,
                            symbol_loader,
                            optimizer=None,
                            scheduler=None,
                            train=False,
                            global_step=global_step,
                            scaler=None,
                        )
                        per_symbol_metrics[symbol] = symbol_metrics

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
                self._maybe_save_checkpoint(model, val_metrics, epoch)

                # Print metrics
                binary_info = ""
                if "binary_return" in val_metrics:
                    binary_info = (
                        f" | Binary: Sortino: {val_metrics['binary_sortino']:.4f} "
                        f"Return: {val_metrics['binary_return']:.4f}"
                    )
                print(f"Epoch {epoch}/{self.config.epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} Score: {train_metrics['score']:.4f} "
                      f"Sortino: {train_metrics['sortino']:.4f} Return: {train_metrics['return']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} Score: {val_metrics['score']:.4f} "
                      f"Sortino: {val_metrics['sortino']:.4f} Return: {val_metrics['return']:.4f}"
                      f"{binary_info}")

                # Print per-symbol metrics if available
                if per_symbol_metrics:
                    for symbol, metrics in per_symbol_metrics.items():
                        print(f"  {symbol}: Sortino: {metrics['sortino']:.4f} "
                              f"Return: {metrics['return']:.4f} "
                              f"Binary Sortino: {metrics['binary_sortino']:.4f} "
                              f"Binary Return: {metrics['binary_return']:.4f}")

                # Log metrics to wandb
                log_dict = {
                    "loss/train": train_metrics["loss"],
                    "loss/val": val_metrics["loss"],
                    "score/train": train_metrics["score"],
                    "score/val": val_metrics["score"],
                    "sortino/train": train_metrics["sortino"],
                    "sortino/val": val_metrics["sortino"],
                    "return/train": train_metrics["return"],
                    "return/val": val_metrics["return"],
                    "fill_ratio/buy_train": train_metrics["buy_fill"],
                    "fill_ratio/sell_train": train_metrics["sell_fill"],
                    "fill_ratio/buy_val": val_metrics["buy_fill"],
                    "fill_ratio/sell_val": val_metrics["sell_fill"],
                }

                # Add binary metrics if available
                if "binary_return" in val_metrics:
                    log_dict["binary/sortino"] = val_metrics["binary_sortino"]
                    log_dict["binary/return"] = val_metrics["binary_return"]
                    log_dict["binary/buy_fill"] = val_metrics["binary_buy_fill"]
                    log_dict["binary/sell_fill"] = val_metrics["binary_sell_fill"]

                # Add per-symbol metrics if available
                if per_symbol_metrics:
                    for symbol, metrics in per_symbol_metrics.items():
                        log_dict[f"{symbol}/sortino"] = metrics["sortino"]
                        log_dict[f"{symbol}/return"] = metrics["return"]
                        if "binary_return" in metrics:
                            log_dict[f"{symbol}/binary_sortino"] = metrics["binary_sortino"]
                            log_dict[f"{symbol}/binary_return"] = metrics["binary_return"]

                tracker.log(log_dict, step=epoch)

                # Check for early stopping
                if self.config.dry_train_steps and global_step >= self.config.dry_train_steps:
                    stop_requested = True
                if stop_requested:
                    break
                if val_metrics["score"] > best_score:
                    best_score = val_metrics["score"]
                    if ema_state is not None and self.config.ema_decay > 0:
                        best_state = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
                    else:
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return TrainingArtifacts(
            state_dict=best_state,
            normalizer=self.data.normalizer,
            history=history,
            feature_columns=list(self.data.feature_columns),
            config=self.config,
            checkpoint_paths=[record.path for record in self._checkpoint_records],
            best_checkpoint=self.best_checkpoint_path,
        )

    def _run_epoch(
        self,
        model: HourlyCryptoPolicy,
        loader,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
        *,
        train: bool,
        global_step: int,
        ema_state: Optional[Dict[str, torch.Tensor]] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> (Dict[str, float], int):
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        total_score = 0.0
        total_sortino = 0.0
        total_return = 0.0
        buy_fill = 0.0
        sell_fill = 0.0
        # Binary simulation metrics (validation only)
        total_binary_sortino = 0.0
        total_binary_return = 0.0
        binary_buy_fill = 0.0
        binary_sell_fill = 0.0
        batches = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Determine AMP dtype and whether to use AMP
            use_amp = scaler is not None
            amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16

            # Forward pass with optional AMP
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(batch["features"])
                actions = model.decode_actions(
                    outputs,
                    reference_close=batch["reference_close"],
                    chronos_high=batch["chronos_high"],
                    chronos_low=batch["chronos_low"],
                )
                sim = simulate_hourly_trades(
                    highs=batch["high"],
                    lows=batch["low"],
                    closes=batch["close"],
                    buy_prices=actions["buy_price"],
                    sell_prices=actions["sell_price"],
                    trade_intensity=actions["trade_amount"],
                    maker_fee=self.config.maker_fee,
                    initial_cash=self.config.initial_cash,
                )
                score, sortino, ann_return = compute_hourly_objective(
                    sim.returns,
                    return_weight=self.config.return_weight,
                )
                loss = -score.mean()

                # Compute binary simulation metrics for validation
                if not train:
                    with torch.no_grad():
                        binary_sim = simulate_hourly_trades_binary(
                            highs=batch["high"],
                            lows=batch["low"],
                            closes=batch["close"],
                            buy_prices=actions["buy_price"],
                            sell_prices=actions["sell_price"],
                            trade_intensity=actions["trade_amount"],
                            maker_fee=self.config.maker_fee,
                            initial_cash=self.config.initial_cash,
                        )
                        _, binary_sortino, binary_ann_return = compute_hourly_objective(
                            binary_sim.returns,
                            return_weight=self.config.return_weight,
                        )

            # Backward pass with optional AMP
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
            total_loss += float(loss.item()) * batch_size
            total_score += float(score.mean().item()) * batch_size
            total_sortino += float(sortino.mean().item()) * batch_size
            total_return += float(ann_return.mean().item()) * batch_size
            buy_fill += float(sim.buy_fill_probability.mean().item()) * batch_size
            sell_fill += float(sim.sell_fill_probability.mean().item()) * batch_size

            # Accumulate binary metrics (validation only)
            if not train:
                total_binary_sortino += float(binary_sortino.mean().item()) * batch_size
                total_binary_return += float(binary_ann_return.mean().item()) * batch_size
                binary_buy_fill += float(binary_sim.buy_fill_probability.mean().item()) * batch_size
                binary_sell_fill += float(binary_sim.sell_fill_probability.mean().item()) * batch_size

            batches += batch_size
        metrics = {
            "loss": total_loss / max(1, batches),
            "score": total_score / max(1, batches),
            "sortino": total_sortino / max(1, batches),
            "return": total_return / max(1, batches),
            "buy_fill": buy_fill / max(1, batches),
            "sell_fill": sell_fill / max(1, batches),
        }

        # Add binary simulation metrics for validation
        if not train:
            metrics["binary_sortino"] = total_binary_sortino / max(1, batches)
            metrics["binary_return"] = total_binary_return / max(1, batches)
            metrics["binary_buy_fill"] = binary_buy_fill / max(1, batches)
            metrics["binary_sell_fill"] = binary_sell_fill / max(1, batches)

        return metrics, global_step

    def _build_optimizer(self, model: HourlyCryptoPolicy) -> torch.optim.Optimizer:
        name = (self.config.optimizer_name or "adamw").lower()
        if name == "muon":
            return Muon(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.95,
                weight_decay=self.config.weight_decay,
            )
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

    def _maybe_save_checkpoint(self, model: HourlyCryptoPolicy, val_metrics: Dict[str, float], epoch: int) -> None:
        val_loss = float(val_metrics.get("loss", float("inf")))
        checkpoint_name = f"epoch{epoch:04d}_valloss{val_loss:.6f}.pt"
        path = self.checkpoint_dir / checkpoint_name
        save_checkpoint(
            path,
            state_dict=model.state_dict(),
            normalizer=self.data.normalizer,
            feature_columns=list(self.data.feature_columns),
            metrics={k: float(v) for k, v in val_metrics.items()},
            config=self.config,
        )
        record = CheckpointRecord(path=path, val_loss=val_loss, epoch=epoch, timestamp=time.time())
        self._checkpoint_records.append(record)
        # Sort by val_loss ASCENDING: more negative losses come first and are better (loss = -score)
        self._checkpoint_records.sort(key=lambda rec: rec.val_loss)
        while len(self._checkpoint_records) > max(1, self.config.top_k_checkpoints):
            removed = self._checkpoint_records.pop()
            try:
                removed.path.unlink()
            except FileNotFoundError:
                pass
        write_manifest(self.checkpoint_dir, self._checkpoint_records, self.config, list(self.data.feature_columns))
        if self._checkpoint_records:
            self.best_checkpoint_path = self._checkpoint_records[0].path
