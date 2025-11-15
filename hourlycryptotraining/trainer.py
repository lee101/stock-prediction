from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.nn.utils import clip_grad_norm_

from differentiable_loss_utils import compute_hourly_objective, simulate_hourly_trades
from wandboard import WandBoardLogger

from .config import TrainingConfig
from .data import FeatureNormalizer, HourlyCryptoDataModule
from .model import HourlyCryptoPolicy, PolicyHeadConfig
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
        model = HourlyCryptoPolicy(
            PolicyHeadConfig(
                input_dim=len(self.data.feature_columns),
                hidden_dim=256,
                depth=3,
                dropout=0.1,
                price_offset_pct=self.config.price_offset_pct,
                max_trade_qty=self.config.max_trade_qty,
            )
        ).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        train_loader = self.data.train_dataloader(self.config.batch_size, self.config.num_workers)
        val_loader = self.data.val_dataloader(self.config.batch_size, self.config.num_workers)
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
            for epoch in range(1, self.config.epochs + 1):
                train_metrics = self._run_epoch(model, train_loader, optimizer, train=True)
                val_metrics = self._run_epoch(model, val_loader, optimizer=None, train=False)
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
                tracker.log(
                    {
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
                    },
                    step=epoch,
                )
                if val_metrics["score"] > best_score:
                    best_score = val_metrics["score"]
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
        *,
        train: bool,
    ) -> Dict[str, float]:
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
        batches = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
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
                buy_amounts=actions["buy_amount"],
                sell_amounts=actions["sell_amount"],
                maker_fee=self.config.maker_fee,
                initial_cash=self.config.initial_cash,
            )
            score, sortino, ann_return = compute_hourly_objective(
                sim.returns,
                return_weight=self.config.return_weight,
            )
            loss = -score.mean()
            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()
            batch_size = sim.pnl.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_score += float(score.mean().item()) * batch_size
            total_sortino += float(sortino.mean().item()) * batch_size
            total_return += float(ann_return.mean().item()) * batch_size
            buy_fill += float(sim.buy_fill_probability.mean().item()) * batch_size
            sell_fill += float(sim.sell_fill_probability.mean().item()) * batch_size
            batches += batch_size
        metrics = {
            "loss": total_loss / max(1, batches),
            "score": total_score / max(1, batches),
            "sortino": total_sortino / max(1, batches),
            "return": total_return / max(1, batches),
            "buy_fill": buy_fill / max(1, batches),
            "sell_fill": sell_fill / max(1, batches),
        }
        return metrics

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
