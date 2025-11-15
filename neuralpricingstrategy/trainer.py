from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import PricingDataset, TARGET_HIGH_COLUMN, TARGET_LOW_COLUMN, TARGET_PNL_GAIN_COLUMN
from .models import PricingAdjustmentModel, PricingModelConfig


class MetricLogger(Protocol):
    def log(self, metrics: Mapping[str, float], *, step: Optional[int] = None) -> None: ...


@dataclass
class PricingTrainingConfig:
    epochs: int = 200
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    pnl_weight: float = 0.2
    max_grad_norm: Optional[float] = 5.0
    device: Optional[str] = None


@dataclass
class PricingTrainingHistoryEntry:
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_mae: Optional[float] = None
    val_mae: Optional[float] = None


@dataclass
class PricingTrainingResult:
    model: PricingAdjustmentModel
    history: Sequence[PricingTrainingHistoryEntry] = field(default_factory=list)
    final_metrics: Dict[str, float] = field(default_factory=dict)


def _resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sortino_ratio(returns: Iterable[float]) -> float:
    arr = np.asarray(list(returns), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    downside = np.sqrt(np.mean(np.square(np.minimum(arr, 0.0))) + 1e-12)
    if downside == 0.0:
        return 0.0
    return float(arr.mean() / downside)


def _dataset_sortino_gain(dataset: PricingDataset) -> float:
    frame = dataset.frame
    base = frame.get("maxdiffalwayson_return")
    target = frame.get("maxdiff_return")
    if base is None or target is None:
        return 0.0
    return _sortino_ratio(target) - _sortino_ratio(base)


def _evaluate_model(
    model: PricingAdjustmentModel, dataset: PricingDataset, device: torch.device
) -> Tuple[float, float]:
    if dataset.features.shape[0] == 0:
        return 0.0, 0.0
    model.eval()
    with torch.no_grad():
        preds = model(dataset.features.to(device))
        targets = dataset.targets.to(device)
        price_loss = F.smooth_l1_loss(preds[:, :2], targets[:, :2])
        pnl_loss = F.smooth_l1_loss(preds[:, 2], targets[:, 2])
        loss = price_loss + pnl_loss
        mae = torch.mean(torch.abs(preds - targets))
    return float(loss.detach().cpu()), float(mae.detach().cpu())


def train_pricing_model(
    train_dataset: PricingDataset,
    *,
    validation_dataset: Optional[PricingDataset] = None,
    config: PricingTrainingConfig = PricingTrainingConfig(),
    logger: Optional[MetricLogger] = None,
) -> PricingTrainingResult:
    device = _resolve_device(config.device)
    model_config = PricingModelConfig(
        input_dim=train_dataset.features.shape[1],
        max_delta_pct=train_dataset.clamp_pct,
    )
    model = PricingAdjustmentModel(model_config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dataset = TensorDataset(train_dataset.features, train_dataset.targets)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    history: List[PricingTrainingHistoryEntry] = []
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            preds = model(batch_features)
            price_loss = F.smooth_l1_loss(preds[:, :2], batch_targets[:, :2])
            pnl_loss = F.smooth_l1_loss(preds[:, 2], batch_targets[:, 2])
            loss = price_loss + config.pnl_weight * pnl_loss
            optimiser.zero_grad()
            loss.backward()
            if config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimiser.step()
            epoch_loss += float(loss.detach().cpu()) * batch_features.shape[0]
        avg_train_loss = epoch_loss / len(dataset)
        train_loss, train_mae = _evaluate_model(model, train_dataset, device)

        entry = PricingTrainingHistoryEntry(epoch=epoch, train_loss=train_loss, train_mae=train_mae)

        if validation_dataset is not None and validation_dataset.features.shape[0] > 0:
            val_loss, val_mae = _evaluate_model(model, validation_dataset, device)
            entry.val_loss = val_loss
            entry.val_mae = val_mae
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(entry)

        if logger:
            payload = {
                "train/loss": train_loss,
                "train/mae": train_mae,
            }
            if validation_dataset is not None:
                payload["val/loss"] = entry.val_loss or 0.0
                payload["val/mae"] = entry.val_mae or 0.0
            logger.log(payload, step=epoch)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = {
        "train/loss": history[-1].train_loss if history else 0.0,
        "train/mae": history[-1].train_mae if history else 0.0,
        "train/sortino_gain": _dataset_sortino_gain(train_dataset),
        "train/pnl_gain_mean": float(
            (train_dataset.frame[TARGET_PNL_GAIN_COLUMN]).mean()
        ),
    }
    if validation_dataset is not None and validation_dataset.features.shape[0] > 0:
        val_loss, val_mae = _evaluate_model(model, validation_dataset, device)
        final_metrics["val/loss"] = val_loss
        final_metrics["val/mae"] = val_mae
        final_metrics["val/sortino_gain"] = _dataset_sortino_gain(validation_dataset)
        final_metrics["val/pnl_gain_mean"] = float(
            (validation_dataset.frame[TARGET_PNL_GAIN_COLUMN]).mean()
        )

    return PricingTrainingResult(model=model, history=history, final_metrics=final_metrics)


__all__ = [
    "PricingTrainingConfig",
    "PricingTrainingHistoryEntry",
    "PricingTrainingResult",
    "train_pricing_model",
]
