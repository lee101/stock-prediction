from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import xgboost as xgb

from .data import DailyStrategyDataset
from .metrics import (
    ObjectiveResult,
    aggregate_daily_pnl,
    combine_sortino_and_return,
    l2_penalty,
)
from .models import PolicyConfig, PortfolioPolicy


class MetricLogger(Protocol):
    def log(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
        commit: Optional[bool] = None,
    ) -> None: ...


def _maybe_log(logger: Optional[MetricLogger], metrics: Mapping[str, float], *, step: Optional[int] = None) -> None:
    if logger is None or not metrics:
        return
    try:
        logger.log(metrics, step=step)
    except Exception:
        pass


def _maybe_log_hparams(
    logger: Optional[MetricLogger],
    hparams: Mapping[str, float],
    metrics: Mapping[str, float],
    *,
    step: Optional[int] = None,
) -> None:
    if logger is None:
        return
    log_hparams = getattr(logger, "log_hparams", None)
    if callable(log_hparams):
        try:
            log_hparams(hparams, metrics, step=step)
        except Exception:
            pass


def _maybe_watch(logger: Optional[MetricLogger], model: torch.nn.Module) -> None:
    if logger is None:
        return
    watch = getattr(logger, "watch", None)
    if callable(watch):
        try:
            watch(model)
        except Exception:
            pass


@dataclass
class TrainingHistoryEntry:
    epoch: int
    train_sortino: float
    train_return: float
    train_score: float
    val_sortino: Optional[float] = None
    val_return: Optional[float] = None
    val_score: Optional[float] = None


@dataclass
class TrainingResult:
    model: PortfolioPolicy
    history: List[TrainingHistoryEntry] = field(default_factory=list)
    final_metrics: Dict[str, float] = field(default_factory=dict)


def _device_from_string(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_policy(
    model: PortfolioPolicy,
    dataset: DailyStrategyDataset,
    *,
    return_weight: float = 0.05,
    trading_days: int = 252,
    device: Optional[str] = None,
) -> ObjectiveResult:
    model.eval()
    with torch.no_grad():
        data = dataset.to_device(_device_from_string(device))
        weights = model(data.features)
        daily = aggregate_daily_pnl(weights, data.daily_returns, data.day_index)
        objective = combine_sortino_and_return(
            daily, trading_days=trading_days, return_weight=return_weight
        )
    return objective


def train_sortino_policy(
    train_dataset: DailyStrategyDataset,
    *,
    validation_dataset: Optional[DailyStrategyDataset] = None,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    return_weight: float = 0.05,
    l2_strength: float = 1e-4,
    trading_days: int = 252,
    device: Optional[str] = None,
    max_grad_norm: Optional[float] = 5.0,
    allow_short: bool = False,
    max_weight: float = 1.0,
    logger: Optional[MetricLogger] = None,
    initial_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
) -> TrainingResult:
    device_obj = _device_from_string(device)
    data = train_dataset.to_device(device_obj)
    config = PolicyConfig(
        input_dim=data.features.shape[1],
        allow_short=allow_short,
        max_weight=max_weight,
    )
    model = PortfolioPolicy(config).to(device_obj)
    if initial_state_dict:
        try:
            model.load_state_dict(initial_state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Failed to load the provided neural checkpoint; ensure the resume run uses the same architecture and feature layout."
            ) from exc
    def _snapshot_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    _maybe_watch(logger, model)
    _maybe_log_hparams(
        logger,
        {
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "epochs": epochs,
            "return_weight": return_weight,
            "l2_strength": l2_strength,
            "allow_short": int(allow_short),
            "max_weight": max_weight,
        },
        {
            "dataset/train_rows": float(len(train_dataset.frame)),
            "dataset/val_rows": float(len(validation_dataset.frame)) if validation_dataset else 0.0,
        },
        step=0,
    )

    history: List[TrainingHistoryEntry] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_score = float("-inf")
    for epoch in range(1, epochs + 1):
        model.train()
        weights = model(data.features)
        daily = aggregate_daily_pnl(weights, data.daily_returns, data.day_index)
        train_obj = combine_sortino_and_return(
            daily, trading_days=trading_days, return_weight=return_weight
        )
        penalty = l2_penalty(weights, l2_strength)
        loss = -train_obj.score + penalty
        optimiser.zero_grad()
        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimiser.step()

        entry = TrainingHistoryEntry(
            epoch=epoch,
            train_sortino=float(train_obj.sortino.detach().cpu()),
            train_return=float(train_obj.annual_return.detach().cpu()),
            train_score=float(train_obj.score.detach().cpu()),
        )

        if validation_dataset is not None:
            val_obj = evaluate_policy(
                model,
                validation_dataset,
                return_weight=return_weight,
                trading_days=trading_days,
                device=device,
            )
            entry.val_sortino = float(val_obj.sortino.detach().cpu())
            entry.val_return = float(val_obj.annual_return.detach().cpu())
            entry.val_score = float(val_obj.score.detach().cpu())
            if entry.val_score > best_val_score:
                best_val_score = entry.val_score
                best_epoch = epoch
                best_state = _snapshot_state(model)

        history.append(entry)
        metrics_payload: Dict[str, float] = {
            "neural/train/sortino": entry.train_sortino,
            "neural/train/annual_return": entry.train_return,
            "neural/train/score": entry.train_score,
        }
        if entry.val_sortino is not None:
            metrics_payload["neural/val/sortino"] = entry.val_sortino
        if entry.val_return is not None:
            metrics_payload["neural/val/annual_return"] = entry.val_return
        if entry.val_score is not None:
            metrics_payload["neural/val/score"] = entry.val_score
        _maybe_log(logger, metrics_payload, step=epoch)

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    final_obj = evaluate_policy(
        model,
        validation_dataset or train_dataset,
        return_weight=return_weight,
        trading_days=trading_days,
        device=device,
    )
    metrics = {
        "sortino": float(final_obj.sortino.detach().cpu()),
        "annual_return": float(final_obj.annual_return.detach().cpu()),
        "score": float(final_obj.score.detach().cpu()),
        "best_epoch": best_epoch if best_epoch else epochs,
    }
    if best_state is not None:
        metrics["best_val_score"] = float(best_val_score)
    _maybe_log(
        logger,
        {
            "neural/final/sortino": metrics["sortino"],
            "neural/final/annual_return": metrics["annual_return"],
            "neural/final/score": metrics["score"],
        },
        step=epochs + 1,
    )
    return TrainingResult(model=model, history=history, final_metrics=metrics)


@dataclass
class XGBoostResult:
    booster: xgb.Booster
    temperature: float
    score: float
    sortino: float
    annual_return: float


def _apply_temperature(weights: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(weights / max(temperature, 1e-6), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def evaluate_weights_numpy(
    weight_array: np.ndarray,
    dataset: DailyStrategyDataset,
    *,
    return_weight: float = 0.05,
    trading_days: int = 252,
) -> ObjectiveResult:
    device = _device_from_string("cpu")
    weights_tensor = torch.from_numpy(weight_array.astype(np.float32)).to(device)
    data = dataset.to_device(device)
    daily = aggregate_daily_pnl(weights_tensor, data.daily_returns, data.day_index)
    return combine_sortino_and_return(
        daily, trading_days=trading_days, return_weight=return_weight
    )


def train_xgboost_policy(
    dataset: DailyStrategyDataset,
    *,
    evaluation_dataset: Optional[DailyStrategyDataset] = None,
    num_rounds: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample: float = 0.8,
    temperature_grid: Sequence[float] = (0.15, 0.25, 0.5, 1.0, 2.0),
    return_weight: float = 0.05,
    trading_days: int = 252,
    logger: Optional[MetricLogger] = None,
) -> XGBoostResult:
    target = dataset.daily_returns.cpu().numpy()
    dtrain = xgb.DMatrix(dataset.features.cpu().numpy(), label=target)
    tree_method = "gpu_hist" if torch.cuda.is_available() else "hist"
    params = {
        "objective": "reg:squarederror",
        "eta": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "tree_method": tree_method,
        "eval_metric": "rmse",
    }
    _maybe_log_hparams(
        logger,
        {
            "num_rounds": num_rounds,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "return_weight": return_weight,
        },
        {
            "dataset/train_rows": float(len(dataset.frame)),
            "dataset/eval_rows": float(len(evaluation_dataset.frame)) if evaluation_dataset else 0.0,
        },
        step=0,
    )
    try:
        booster = xgb.train(params, dtrain, num_boost_round=num_rounds)
    except xgb.core.XGBoostError as err:  # pragma: no cover - rare fallback
        if "Invalid Input: 'gpu_hist'" in str(err) and tree_method == "gpu_hist":
            params["tree_method"] = "hist"
            booster = xgb.train(params, dtrain, num_boost_round=num_rounds)
        else:
            raise
    eval_data = evaluation_dataset or dataset
    deval = xgb.DMatrix(eval_data.features.cpu().numpy())
    raw_pred = booster.predict(deval)

    best: Optional[XGBoostResult] = None
    for idx, temp in enumerate(temperature_grid, start=1):
        weights = _apply_temperature(raw_pred, temp)
        obj = evaluate_weights_numpy(
            weights,
            eval_data,
            return_weight=return_weight,
            trading_days=trading_days,
        )
        score = float(obj.score.detach().cpu())
        result = XGBoostResult(
            booster=booster,
            temperature=temp,
            score=score,
            sortino=float(obj.sortino.detach().cpu()),
            annual_return=float(obj.annual_return.detach().cpu()),
        )
        _maybe_log(
            logger,
            {
                "xgboost/temperature": float(temp),
                "xgboost/score": result.score,
                "xgboost/sortino": result.sortino,
                "xgboost/annual_return": result.annual_return,
            },
            step=num_rounds + idx,
        )
        if best is None or result.score > best.score:
            best = result
    if best is None:
        raise RuntimeError("Failed to evaluate any temperature for XGBoost policy.")
    _maybe_log(
        logger,
        {
            "xgboost/final/temperature": best.temperature,
            "xgboost/final/score": best.score,
            "xgboost/final/sortino": best.sortino,
            "xgboost/final/annual_return": best.annual_return,
        },
        step=num_rounds + len(temperature_grid) + 1,
    )
    return best
