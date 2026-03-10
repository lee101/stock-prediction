from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.fees import get_fee_for_symbol
from src.robust_trading_metrics import summarize_scenario_results

from .experiments.ambiguity_plan_gate import (
    derive_plan_class_weights,
    derive_plan_labels,
)
from .experiments.budget_consensus_dispersion import apply_budget_consensus_dispersion_gate
from .experiments.budget_entropy_confidence import apply_budget_entropy_confidence_gate
from .experiments.budget_guided_keep_count import apply_budget_guided_rank_gate
from .experiments.continuous_budget_thresholds import apply_continuous_budget_gate
from .experiments.cost_margin_gate import (
    apply_cost_margin_gate,
    compute_edge_quantiles,
    derive_margin_targets,
)
from .experiments.cross_sectional_rank_gate import (
    apply_dynamic_cross_sectional_gate,
    apply_topk_cross_sectional_gate,
    compute_cross_sectional_rank_signal,
)
from .experiments.dynamic_score_floor import apply_dynamic_score_floor_gate
from .experiments.soft_rank_sizing import apply_soft_rank_sizing
from .experiments.timestamp_budget_head import (
    apply_budget_aware_soft_rank_sizing,
    derive_budget_class_weights,
    derive_budget_labels,
    rebuild_split_sample_rows,
)
from .prepare import (
    TIME_BUDGET,
    _lag_action_frame,
    build_action_frame,
    parse_csv_list,
    parse_int_list,
    prepare_task,
    resolve_task_config,
    simulate_actions,
)


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        symbol_ids: np.ndarray,
        weights: np.ndarray | None = None,
        plan_labels: np.ndarray | None = None,
        margin_targets: np.ndarray | None = None,
        budget_labels: np.ndarray | None = None,
    ) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.symbol_ids = symbol_ids.astype(np.int64, copy=False)
        if weights is None:
            self.weights = np.ones((len(self.features),), dtype=np.float32)
        else:
            self.weights = weights.astype(np.float32, copy=False)
        if plan_labels is None:
            self.plan_labels = np.zeros((len(self.features),), dtype=np.int64)
        else:
            self.plan_labels = plan_labels.astype(np.int64, copy=False)
        if margin_targets is None:
            self.margin_targets = np.zeros((len(self.features),), dtype=np.float32)
        else:
            self.margin_targets = margin_targets.astype(np.float32, copy=False)
        if budget_labels is None:
            self.budget_labels = np.zeros((len(self.features),), dtype=np.int64)
        else:
            self.budget_labels = budget_labels.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(len(self.features))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[index]),
            "targets": torch.from_numpy(self.targets[index]),
            "symbol_ids": torch.tensor(int(self.symbol_ids[index]), dtype=torch.long),
            "weights": torch.tensor(float(self.weights[index]), dtype=torch.float32),
            "plan_labels": torch.tensor(int(self.plan_labels[index]), dtype=torch.long),
            "margin_targets": torch.tensor(float(self.margin_targets[index]), dtype=torch.float32),
            "budget_labels": torch.tensor(int(self.budget_labels[index]), dtype=torch.long),
        }


@dataclass
class PlannerConfig:
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.10
    symbol_embedding_dim: int = 16
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    eval_batch_size: int = 512
    ema_decay: float = 0.995
    ambiguity_quantile: float = 0.25
    strong_gap_multiplier: float = 2.0
    weak_action_scale: float = 0.45
    plan_loss_weight: float = 0.20
    margin_loss_weight: float = 0.15
    margin_floor_quantile: float = 0.25
    margin_ceiling_quantile: float = 0.75
    spread_margin_scale: float = 4.0
    volatility_margin_scale: float = 0.5
    rank_top_k: int = 3
    rank_min_score: float = 0.20
    rank_floor_min_strength: float = 0.06
    rank_floor_quantile: float = 0.25
    rank_floor_gap_scale: float = 1.60
    use_dynamic_score_floor: bool = False
    use_soft_rank_sizing: bool = False
    use_timestamp_budget_head: bool = False
    use_budget_guided_keep_count: bool = False
    use_continuous_budget_thresholds: bool = False
    use_budget_entropy_confidence: bool = False
    use_budget_consensus_dispersion: bool = False
    budget_loss_weight: float = 0.08
    budget_broad_count_threshold: int = 2
    budget_skip_scale: float = 0.55
    budget_selective_scale: float = 0.82
    budget_selective_top_k: int = 3
    budget_selective_max_keep: int = 2
    budget_broad_top_k: int = 5
    budget_broad_max_keep: int = 4
    budget_selective_gap_scale: float = 0.50
    budget_broad_gap_scale: float = 0.75
    budget_skip_gap_scale: float = 0.35
    budget_skip_min_score_scale: float = 1.10
    budget_selective_min_score_scale: float = 1.00
    budget_broad_min_score_scale: float = 0.80
    budget_confidence_power: float = 1.5
    budget_selective_prior_skip_weight: float = 0.20
    budget_uncertainty_gap_floor: float = 0.60
    budget_uncertainty_fractional_floor: float = 0.50
    budget_broad_consensus_power: float = 1.6
    budget_consensus_selective_reallocation: float = 0.85
    budget_consensus_gap_floor: float = 0.80
    budget_consensus_fractional_floor: float = 0.75
    rank_sizing_reference_quantile: float = 0.25
    rank_sizing_regime_floor_scale: float = 0.75
    rank_sizing_name_floor_scale: float = 0.55
    rank_sizing_regime_power: float = 0.75
    rank_sizing_rank_power: float = 1.5
    rank_min_keep: int = 1
    rank_max_keep: int = 4
    rank_reference_quantile: float = 0.25
    rank_gap_scale: float = 0.50
    seed: int = 20260310


class PlannerNet(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        symbol_count: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        symbol_embedding_dim: int,
        weak_action_scale: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.symbol_embedding = nn.Embedding(int(symbol_count), int(symbol_embedding_dim))
        plan_hidden_size = max(int(hidden_size) // 2, 8)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size + symbol_embedding_dim),
            nn.Linear(hidden_size + symbol_embedding_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 3),
        )
        self.plan_head = nn.Sequential(
            nn.LayerNorm(hidden_size + symbol_embedding_dim),
            nn.Linear(hidden_size + symbol_embedding_dim, plan_hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(plan_hidden_size, 5),
        )
        self.margin_head = nn.Sequential(
            nn.LayerNorm(hidden_size + symbol_embedding_dim),
            nn.Linear(hidden_size + symbol_embedding_dim, plan_hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(plan_hidden_size, 1),
        )
        self.budget_head = nn.Sequential(
            nn.LayerNorm(hidden_size + symbol_embedding_dim),
            nn.Linear(hidden_size + symbol_embedding_dim, plan_hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(plan_hidden_size, 3),
        )
        self.weak_action_scale = float(weak_action_scale)

    def _encode_context(self, features: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(features)
        _, hidden = self.gru(encoded)
        summary = hidden[-1]
        symbol_context = self.symbol_embedding(symbol_ids)
        return torch.cat([summary, symbol_context], dim=-1)

    def predict_with_plan(
        self,
        features: torch.Tensor,
        symbol_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        combined = self._encode_context(features, symbol_ids)
        raw_predictions = self.regression_head(combined)
        plan_logits = self.plan_head(combined)
        margin_logits = self.margin_head(combined)
        budget_logits = self.budget_head(combined)
        gated_predictions = apply_cost_margin_gate(
            raw_predictions,
            plan_logits,
            margin_logits,
            weak_action_scale=self.weak_action_scale,
        )
        return gated_predictions, plan_logits, margin_logits, budget_logits

    def forward(self, features: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        gated_predictions, _, _, _ = self.predict_with_plan(features, symbol_ids)
        return gated_predictions


def _base_loss_terms(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base = F.smooth_l1_loss(predictions, targets, reduction="none").mean(dim=1)
    pred_high = predictions[:, 0]
    pred_low = predictions[:, 1]
    pred_close = predictions[:, 2]
    order_penalty = F.relu(pred_close - pred_high) + F.relu(pred_low - pred_close)
    return base, order_penalty


def planner_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    base, order_penalty = _base_loss_terms(predictions, targets)
    weighted = (base + 0.25 * order_penalty) * weights
    return weighted.mean()


def _trade_terms(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    high = values[:, 0]
    low = values[:, 1]
    close = values[:, 2]
    zeros = torch.zeros_like(close)
    upside = torch.maximum(torch.maximum(high, close), zeros)
    downside = torch.maximum(torch.maximum(-low, -close), zeros)
    gap = upside - downside
    opportunity = torch.maximum(upside, downside)
    return upside, downside, gap, opportunity


def compute_ambiguity_floor(targets: np.ndarray, *, quantile: float) -> float:
    if len(targets) == 0:
        return 1e-4
    high = np.maximum.reduce([targets[:, 0], targets[:, 2], np.zeros(len(targets), dtype=np.float32)])
    downside = np.maximum.reduce([-targets[:, 1], -targets[:, 2], np.zeros(len(targets), dtype=np.float32)])
    gap = np.abs(high - downside)
    floor = float(np.quantile(gap.astype(np.float64, copy=False), np.clip(float(quantile), 0.0, 1.0)))
    return float(max(floor, 1e-4))


def decision_aware_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    ambiguity_floor: float,
) -> torch.Tensor:
    base, order_penalty = _base_loss_terms(predictions, targets)
    pred_upside, pred_downside, pred_gap, pred_opportunity = _trade_terms(predictions)
    target_upside, target_downside, target_gap, _ = _trade_terms(targets)
    target_asymmetry = torch.abs(target_gap)
    target_selectivity = F.relu(target_asymmetry - float(ambiguity_floor))
    gap_loss = F.smooth_l1_loss(pred_gap, target_gap, reduction="none")
    selectivity_loss = F.smooth_l1_loss(pred_opportunity, target_selectivity, reduction="none")
    ambiguous_weight = torch.sigmoid((float(ambiguity_floor) - target_asymmetry) * 80.0)
    overtrade_penalty = ambiguous_weight * pred_opportunity
    weighted = (
        base
        + 0.25 * order_penalty
        + 0.75 * gap_loss
        + 0.35 * selectivity_loss
        + 0.02 * overtrade_penalty
    ) * weights
    return weighted.mean()


def plan_loss(
    plan_logits: torch.Tensor,
    plan_labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(plan_logits, plan_labels, weight=class_weights, reduction="none")
    return (losses * weights).mean()


def margin_loss(margin_logits: torch.Tensor, margin_targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    margin_values = torch.sigmoid(margin_logits.squeeze(-1))
    losses = F.smooth_l1_loss(margin_values, margin_targets, reduction="none")
    return (losses * weights).mean()


def budget_loss(
    budget_logits: torch.Tensor,
    budget_labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(budget_logits, budget_labels, weight=class_weights, reduction="none")
    return (losses * weights).mean()


def _predict_ranked_batches(
    model: PlannerNet,
    *,
    features: np.ndarray,
    symbol_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(features) == 0:
        empty_predictions = np.zeros((0, 3), dtype=np.float32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        empty_budget_logits = np.zeros((0, 3), dtype=np.float32)
        return empty_predictions, empty_scores, empty_budget_logits

    model.eval()
    prediction_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    budget_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            stop = min(len(features), start + int(batch_size))
            feature_batch = torch.from_numpy(features[start:stop]).to(device=device, dtype=torch.float32)
            symbol_batch = torch.from_numpy(symbol_ids[start:stop]).to(device=device, dtype=torch.long)
            predictions, plan_logits, margin_logits, budget_logits = model.predict_with_plan(feature_batch, symbol_batch)
            rank_signal = compute_cross_sectional_rank_signal(
                plan_logits,
                margin_logits,
                weak_action_scale=model.weak_action_scale,
            )
            prediction_parts.append(predictions.detach().cpu().numpy().astype(np.float32, copy=False))
            score_parts.append(rank_signal.detach().cpu().numpy().astype(np.float32, copy=False))
            budget_parts.append(budget_logits.detach().cpu().numpy().astype(np.float32, copy=False))
    return (
        np.concatenate(prediction_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(score_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(budget_parts, axis=0).astype(np.float32, copy=False),
    )


def evaluate_ranked_model(
    model: PlannerNet,
    task,
    *,
    device: torch.device,
    batch_size: int,
    rank_top_k: int,
    rank_min_score: float,
    use_dynamic_score_floor: bool,
    rank_floor_min_strength: float,
    rank_floor_quantile: float,
    rank_floor_gap_scale: float,
    use_soft_rank_sizing: bool,
    use_timestamp_budget_head: bool,
    use_budget_guided_keep_count: bool,
    use_continuous_budget_thresholds: bool,
    use_budget_entropy_confidence: bool,
    use_budget_consensus_dispersion: bool,
    budget_skip_scale: float,
    budget_selective_scale: float,
    budget_selective_top_k: int,
    budget_selective_max_keep: int,
    budget_broad_top_k: int,
    budget_broad_max_keep: int,
    budget_selective_gap_scale: float,
    budget_broad_gap_scale: float,
    budget_skip_gap_scale: float,
    budget_skip_min_score_scale: float,
    budget_selective_min_score_scale: float,
    budget_broad_min_score_scale: float,
    budget_confidence_power: float,
    budget_selective_prior_skip_weight: float,
    budget_uncertainty_gap_floor: float,
    budget_uncertainty_fractional_floor: float,
    budget_broad_consensus_power: float,
    budget_consensus_selective_reallocation: float,
    budget_consensus_gap_floor: float,
    budget_consensus_fractional_floor: float,
    rank_sizing_reference_quantile: float,
    rank_sizing_regime_floor_scale: float,
    rank_sizing_name_floor_scale: float,
    rank_sizing_regime_power: float,
    rank_sizing_rank_power: float,
    rank_min_keep: int,
    rank_max_keep: int,
    rank_reference_quantile: float,
    rank_gap_scale: float,
) -> dict[str, Any]:
    scenario_rows: list[dict[str, float]] = []
    scenario_outputs: list[dict[str, Any]] = []
    total_trade_count = 0

    for scenario in task.scenarios:
        predictions, rank_signal, budget_logits = _predict_ranked_batches(
            model,
            features=scenario.features,
            symbol_ids=scenario.symbol_ids,
            device=device,
            batch_size=batch_size,
        )
        if bool(use_dynamic_score_floor):
            ranked_input = apply_dynamic_score_floor_gate(
                predictions,
                scenario.action_rows,
                rank_signal,
                min_score=float(rank_min_score),
                min_strength=float(rank_floor_min_strength),
                floor_quantile=float(rank_floor_quantile),
                floor_gap_scale=float(rank_floor_gap_scale),
            )
        else:
            ranked_input = predictions
        if bool(use_budget_consensus_dispersion):
            ranked_predictions = apply_budget_consensus_dispersion_gate(
                ranked_input,
                scenario.action_rows,
                rank_signal,
                budget_logits,
                min_score=float(rank_min_score),
                base_min_keep=int(rank_min_keep),
                selective_top_k=int(budget_selective_top_k),
                selective_max_keep=int(budget_selective_max_keep),
                broad_top_k=int(budget_broad_top_k),
                broad_max_keep=int(budget_broad_max_keep),
                reference_quantile=float(rank_reference_quantile),
                selective_gap_scale=float(budget_selective_gap_scale),
                broad_gap_scale=float(budget_broad_gap_scale),
                skip_gap_scale=float(budget_skip_gap_scale),
                skip_min_score_scale=float(budget_skip_min_score_scale),
                selective_min_score_scale=float(budget_selective_min_score_scale),
                broad_min_score_scale=float(budget_broad_min_score_scale),
                broad_consensus_power=float(budget_broad_consensus_power),
                selective_reallocation=float(budget_consensus_selective_reallocation),
                consensus_gap_floor=float(budget_consensus_gap_floor),
                consensus_fractional_floor=float(budget_consensus_fractional_floor),
            )
        elif bool(use_budget_entropy_confidence):
            ranked_predictions = apply_budget_entropy_confidence_gate(
                ranked_input,
                scenario.action_rows,
                rank_signal,
                budget_logits,
                min_score=float(rank_min_score),
                base_min_keep=int(rank_min_keep),
                selective_top_k=int(budget_selective_top_k),
                selective_max_keep=int(budget_selective_max_keep),
                broad_top_k=int(budget_broad_top_k),
                broad_max_keep=int(budget_broad_max_keep),
                reference_quantile=float(rank_reference_quantile),
                selective_gap_scale=float(budget_selective_gap_scale),
                broad_gap_scale=float(budget_broad_gap_scale),
                skip_gap_scale=float(budget_skip_gap_scale),
                skip_min_score_scale=float(budget_skip_min_score_scale),
                selective_min_score_scale=float(budget_selective_min_score_scale),
                broad_min_score_scale=float(budget_broad_min_score_scale),
                confidence_power=float(budget_confidence_power),
                selective_prior_skip_weight=float(budget_selective_prior_skip_weight),
                uncertainty_gap_floor=float(budget_uncertainty_gap_floor),
                uncertainty_fractional_floor=float(budget_uncertainty_fractional_floor),
            )
        elif bool(use_continuous_budget_thresholds):
            ranked_predictions = apply_continuous_budget_gate(
                ranked_input,
                scenario.action_rows,
                rank_signal,
                budget_logits,
                min_score=float(rank_min_score),
                base_min_keep=int(rank_min_keep),
                selective_top_k=int(budget_selective_top_k),
                selective_max_keep=int(budget_selective_max_keep),
                broad_top_k=int(budget_broad_top_k),
                broad_max_keep=int(budget_broad_max_keep),
                reference_quantile=float(rank_reference_quantile),
                selective_gap_scale=float(budget_selective_gap_scale),
                broad_gap_scale=float(budget_broad_gap_scale),
                skip_gap_scale=float(budget_skip_gap_scale),
                skip_min_score_scale=float(budget_skip_min_score_scale),
                selective_min_score_scale=float(budget_selective_min_score_scale),
                broad_min_score_scale=float(budget_broad_min_score_scale),
            )
        elif bool(use_budget_guided_keep_count):
            ranked_predictions = apply_budget_guided_rank_gate(
                ranked_input,
                scenario.action_rows,
                rank_signal,
                budget_logits,
                min_score=float(rank_min_score),
                base_min_keep=int(rank_min_keep),
                selective_top_k=int(budget_selective_top_k),
                selective_max_keep=int(budget_selective_max_keep),
                broad_top_k=int(budget_broad_top_k),
                broad_max_keep=int(budget_broad_max_keep),
                reference_quantile=float(rank_reference_quantile),
                selective_gap_scale=float(budget_selective_gap_scale),
                broad_gap_scale=float(budget_broad_gap_scale),
                skip_gap_scale=float(budget_skip_gap_scale),
            )
        else:
            topk_predictions = apply_topk_cross_sectional_gate(
                ranked_input,
                scenario.action_rows,
                rank_signal,
                top_k=int(rank_top_k),
                min_score=float(rank_min_score),
            )
            ranked_predictions = apply_dynamic_cross_sectional_gate(
                topk_predictions,
                scenario.action_rows,
                rank_signal,
                min_score=float(rank_min_score),
                min_keep=int(rank_min_keep),
                max_keep=int(rank_max_keep),
                reference_quantile=float(rank_reference_quantile),
                gap_scale=float(rank_gap_scale),
            )
        if bool(use_timestamp_budget_head):
            ranked_predictions = apply_budget_aware_soft_rank_sizing(
                ranked_predictions,
                scenario.action_rows,
                rank_signal,
                budget_logits,
                min_score=float(rank_min_score),
                reference_quantile=float(rank_sizing_reference_quantile),
                regime_floor_scale=float(rank_sizing_regime_floor_scale),
                name_floor_scale=float(rank_sizing_name_floor_scale),
                regime_power=float(rank_sizing_regime_power),
                rank_power=float(rank_sizing_rank_power),
                budget_skip_scale=float(budget_skip_scale),
                budget_selective_scale=float(budget_selective_scale),
            )
        elif bool(use_soft_rank_sizing):
            ranked_predictions = apply_soft_rank_sizing(
                ranked_predictions,
                scenario.action_rows,
                rank_signal,
                min_score=float(rank_min_score),
                reference_quantile=float(rank_sizing_reference_quantile),
                regime_floor_scale=float(rank_sizing_regime_floor_scale),
                name_floor_scale=float(rank_sizing_name_floor_scale),
                regime_power=float(rank_sizing_regime_power),
                rank_power=float(rank_sizing_rank_power),
            )
        actions = build_action_frame(scenario.action_rows, ranked_predictions, task.config)
        actions = _lag_action_frame(actions, scenario.bars, int(task.config.decision_lag_bars))
        result = simulate_actions(scenario.bars, actions, task.config)
        total_trade_count += int(result["trade_count"])
        scenario_rows.append(
            {
                "sortino": float(result["sortino"]),
                "return_pct": float(result["return_pct"]),
                "annualized_return_pct": float(result["annualized_return_pct"]),
                "max_drawdown_pct": float(result["max_drawdown_pct"]),
                "pnl_smoothness": float(result["pnl_smoothness"]),
                "trade_count": float(result["trade_count"]),
            }
        )
        scenario_outputs.append(
            {
                "name": scenario.name,
                "metrics": {key: value for key, value in result.items() if key not in {"equity_curve", "trades"}},
                "actions": actions,
                "equity_curve": result["equity_curve"],
                "trades": result["trades"],
            }
        )

    robust_summary = summarize_scenario_results(scenario_rows)
    robust_summary["total_trade_count"] = float(total_trade_count)
    return {
        "summary": robust_summary,
        "scenarios": scenario_outputs,
    }


def select_device(raw: str | None) -> torch.device:
    token = (raw or "auto").strip().lower()
    if token == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(token)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def evaluate_validation(
    model: PlannerNet,
    loader: DataLoader,
    *,
    device: torch.device,
    ambiguity_floor: float,
    plan_loss_weight: float,
    margin_loss_weight: float,
    budget_loss_weight: float,
    plan_class_weights: torch.Tensor,
    budget_class_weights: torch.Tensor,
) -> tuple[float, float]:
    if len(loader.dataset) == 0:
        return 0.0, 0.0
    model.eval()
    planner_losses: list[float] = []
    selection_losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            targets = batch["targets"].to(device=device, dtype=torch.float32)
            symbol_ids = batch["symbol_ids"].to(device=device, dtype=torch.long)
            weights = batch["weights"].to(device=device, dtype=torch.float32)
            plan_labels = batch["plan_labels"].to(device=device, dtype=torch.long)
            margin_targets = batch["margin_targets"].to(device=device, dtype=torch.float32)
            budget_labels = batch["budget_labels"].to(device=device, dtype=torch.long)
            predictions, plan_logits, margin_logits, budget_logits = model.predict_with_plan(features, symbol_ids)
            planner_value = planner_loss(predictions, targets, weights)
            selection_value = decision_aware_loss(
                predictions,
                targets,
                weights,
                ambiguity_floor=ambiguity_floor,
            ) + float(plan_loss_weight) * plan_loss(
                plan_logits,
                plan_labels,
                weights,
                class_weights=plan_class_weights,
            ) + float(margin_loss_weight) * margin_loss(
                margin_logits,
                margin_targets,
                weights,
            ) + float(budget_loss_weight) * budget_loss(
                budget_logits,
                budget_labels,
                weights,
                class_weights=budget_class_weights,
            )
            planner_losses.append(float(planner_value.detach().cpu()))
            selection_losses.append(float(selection_value.detach().cpu()))
    planner_mean = float(np.mean(planner_losses)) if planner_losses else 0.0
    selection_mean = float(np.mean(selection_losses)) if selection_losses else 0.0
    return planner_mean, selection_mean


def build_dataloaders(task, cfg: PlannerConfig, *, ambiguity_floor: float) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    spread_feature_index = task.feature_names.index("spread_bps_norm")
    volatility_feature_index = task.feature_names.index("volatility")
    symbol_fee_rates = np.zeros((len(task.symbol_to_id),), dtype=np.float32)
    for symbol, symbol_id in task.symbol_to_id.items():
        symbol_fee_rates[int(symbol_id)] = float(get_fee_for_symbol(symbol))

    train_plan_labels = derive_plan_labels(
        task.train_features,
        task.train_targets,
        task.train_symbol_ids,
        spread_feature_index=spread_feature_index,
        symbol_fee_rates=symbol_fee_rates,
        min_edge_bps=task.config.min_edge_bps,
        entry_slippage_bps=task.config.entry_slippage_bps,
        exit_slippage_bps=task.config.exit_slippage_bps,
        ambiguity_floor=ambiguity_floor,
        strong_gap_multiplier=cfg.strong_gap_multiplier,
    )
    val_plan_labels = derive_plan_labels(
        task.val_features,
        task.val_targets,
        task.val_symbol_ids,
        spread_feature_index=spread_feature_index,
        symbol_fee_rates=symbol_fee_rates,
        min_edge_bps=task.config.min_edge_bps,
        entry_slippage_bps=task.config.entry_slippage_bps,
        exit_slippage_bps=task.config.exit_slippage_bps,
        ambiguity_floor=ambiguity_floor,
        strong_gap_multiplier=cfg.strong_gap_multiplier,
    )
    edge_floor, edge_ceiling = compute_edge_quantiles(
        task.train_features,
        task.train_targets,
        task.train_symbol_ids,
        spread_feature_index=spread_feature_index,
        symbol_fee_rates=symbol_fee_rates,
        min_edge_bps=task.config.min_edge_bps,
        entry_slippage_bps=task.config.entry_slippage_bps,
        exit_slippage_bps=task.config.exit_slippage_bps,
        floor_quantile=cfg.margin_floor_quantile,
        ceiling_quantile=cfg.margin_ceiling_quantile,
    )
    train_margin_targets = derive_margin_targets(
        task.train_features,
        task.train_targets,
        task.train_symbol_ids,
        train_plan_labels,
        spread_feature_index=spread_feature_index,
        volatility_feature_index=volatility_feature_index,
        symbol_fee_rates=symbol_fee_rates,
        entry_slippage_bps=task.config.entry_slippage_bps,
        exit_slippage_bps=task.config.exit_slippage_bps,
        edge_floor=edge_floor,
        edge_ceiling=edge_ceiling,
        spread_margin_scale=cfg.spread_margin_scale,
        volatility_margin_scale=cfg.volatility_margin_scale,
    )
    val_margin_targets = derive_margin_targets(
        task.val_features,
        task.val_targets,
        task.val_symbol_ids,
        val_plan_labels,
        spread_feature_index=spread_feature_index,
        volatility_feature_index=volatility_feature_index,
        symbol_fee_rates=symbol_fee_rates,
        entry_slippage_bps=task.config.entry_slippage_bps,
        exit_slippage_bps=task.config.exit_slippage_bps,
        edge_floor=edge_floor,
        edge_ceiling=edge_ceiling,
        spread_margin_scale=cfg.spread_margin_scale,
        volatility_margin_scale=cfg.volatility_margin_scale,
    )
    train_budget_labels = np.zeros((len(task.train_features),), dtype=np.int64)
    val_budget_labels = np.zeros((len(task.val_features),), dtype=np.int64)
    budget_class_weights = np.ones((3,), dtype=np.float32)
    if bool(cfg.use_timestamp_budget_head):
        train_rows, val_rows = rebuild_split_sample_rows(
            task.config,
            spread_profile_bps=task.spread_profile_bps,
        )
        if len(train_rows) != len(task.train_features):
            raise ValueError("timestamp budget head train rows are not aligned with train features")
        if len(val_rows) != len(task.val_features):
            raise ValueError("timestamp budget head val rows are not aligned with val features")
        train_budget_labels = derive_budget_labels(
            train_rows,
            train_plan_labels,
            broad_count_threshold=cfg.budget_broad_count_threshold,
        )
        val_budget_labels = derive_budget_labels(
            val_rows,
            val_plan_labels,
            broad_count_threshold=cfg.budget_broad_count_threshold,
        )
        budget_class_weights = derive_budget_class_weights(train_budget_labels)
    train_dataset = SequenceDataset(
        task.train_features,
        task.train_targets,
        task.train_symbol_ids,
        task.train_weights,
        train_plan_labels,
        train_margin_targets,
        train_budget_labels,
    )
    val_weights = np.ones((len(task.val_features),), dtype=np.float32)
    val_dataset = SequenceDataset(
        task.val_features,
        task.val_targets,
        task.val_symbol_ids,
        val_weights,
        val_plan_labels,
        val_margin_targets,
        val_budget_labels,
    )
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        drop_last=bool(len(train_dataset) >= int(cfg.batch_size)),
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
    )
    return train_loader, val_loader, derive_plan_class_weights(train_plan_labels), budget_class_weights


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def update_ema_state(
    ema_state: dict[str, torch.Tensor],
    model: nn.Module,
    *,
    step_count: int,
    decay: float,
) -> None:
    adaptive_decay = min(float(decay), (1.0 + float(step_count)) / (10.0 + float(step_count)))
    for name, value in model.state_dict().items():
        current = value.detach()
        ema_value = ema_state.get(name)
        if ema_value is None:
            ema_state[name] = current.clone()
            continue
        if torch.is_floating_point(current):
            ema_value.lerp_(current, weight=1.0 - adaptive_decay)
        else:
            ema_value.copy_(current)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the autoresearch-style stock planner for a fixed time budget.")
    parser.add_argument("--frequency", choices=("hourly", "daily"), default="hourly")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--hold-bars", type=int, default=None)
    parser.add_argument("--eval-windows", default="")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-volume-fraction", type=float, default=None)
    parser.add_argument("--min-edge-bps", type=float, default=4.0)
    parser.add_argument("--entry-slippage-bps", type=float, default=1.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=1.0)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--disable-short", action="store_true")
    parser.add_argument("--dashboard-db", default="dashboards/metrics.db")
    parser.add_argument("--spread-lookback-days", type=int, default=14)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--symbol-embedding-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--dynamic-score-floor", action="store_true")
    parser.add_argument("--soft-rank-sizing", action="store_true")
    parser.add_argument("--timestamp-budget-head", action="store_true")
    parser.add_argument("--budget-guided-keep-count", action="store_true")
    parser.add_argument("--continuous-budget-thresholds", action="store_true")
    parser.add_argument("--budget-entropy-confidence", action="store_true")
    parser.add_argument("--budget-consensus-dispersion", action="store_true")
    args = parser.parse_args(argv)

    total_start = time.perf_counter()
    device = select_device(args.device)
    seed_everything(args.seed)

    task_config = resolve_task_config(
        frequency=args.frequency,
        symbols=parse_csv_list(args.symbols) or None,
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        hold_bars=args.hold_bars,
        eval_windows=parse_int_list(args.eval_windows) or None,
        max_positions=args.max_positions,
        max_volume_fraction=args.max_volume_fraction,
        min_edge_bps=args.min_edge_bps,
        entry_slippage_bps=args.entry_slippage_bps,
        exit_slippage_bps=args.exit_slippage_bps,
        decision_lag_bars=args.decision_lag_bars,
        allow_short=not bool(args.disable_short),
        dashboard_db_path=args.dashboard_db,
        spread_lookback_days=args.spread_lookback_days,
    )
    task = prepare_task(task_config)
    planner_cfg = PlannerConfig(
        hidden_size=int(args.hidden_size),
        num_layers=int(args.layers),
        dropout=float(args.dropout),
        symbol_embedding_dim=int(args.symbol_embedding_dim),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        eval_batch_size=int(args.eval_batch_size),
        use_dynamic_score_floor=bool(args.dynamic_score_floor),
        use_soft_rank_sizing=bool(args.soft_rank_sizing),
        use_timestamp_budget_head=bool(
            args.timestamp_budget_head
            or args.budget_guided_keep_count
            or args.continuous_budget_thresholds
            or args.budget_entropy_confidence
            or args.budget_consensus_dispersion
        ),
        use_budget_guided_keep_count=bool(args.budget_guided_keep_count),
        use_continuous_budget_thresholds=bool(args.continuous_budget_thresholds),
        use_budget_entropy_confidence=bool(args.budget_entropy_confidence),
        use_budget_consensus_dispersion=bool(args.budget_consensus_dispersion),
        seed=int(args.seed),
    )
    ambiguity_floor = compute_ambiguity_floor(
        task.train_targets,
        quantile=planner_cfg.ambiguity_quantile,
    )
    budget_loss_weight = planner_cfg.budget_loss_weight if planner_cfg.use_timestamp_budget_head else 0.0

    train_loader, val_loader, plan_class_weights_np, budget_class_weights_np = build_dataloaders(
        task,
        planner_cfg,
        ambiguity_floor=ambiguity_floor,
    )
    model = PlannerNet(
        feature_dim=int(task.train_features.shape[-1]),
        symbol_count=max(len(task.symbol_to_id), 1),
        hidden_size=planner_cfg.hidden_size,
        num_layers=planner_cfg.num_layers,
        dropout=planner_cfg.dropout,
        symbol_embedding_dim=planner_cfg.symbol_embedding_dim,
        weak_action_scale=planner_cfg.weak_action_scale,
    ).to(device)
    ema_state = clone_model_state(model)
    plan_class_weights = torch.from_numpy(plan_class_weights_np).to(device=device, dtype=torch.float32)
    budget_class_weights = torch.from_numpy(budget_class_weights_np).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(planner_cfg.learning_rate),
        weight_decay=float(planner_cfg.weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter()
    step_count = 0
    data_iter = iter(train_loader)
    while (time.perf_counter() - train_start) < TIME_BUDGET:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        features = batch["features"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        symbol_ids = batch["symbol_ids"].to(device=device, dtype=torch.long)
        weights = batch["weights"].to(device=device, dtype=torch.float32)
        plan_labels = batch["plan_labels"].to(device=device, dtype=torch.long)
        margin_targets = batch["margin_targets"].to(device=device, dtype=torch.float32)
        budget_labels = batch["budget_labels"].to(device=device, dtype=torch.long)

        autocast_enabled = device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            predictions, plan_logits, margin_logits, budget_logits = model.predict_with_plan(features, symbol_ids)
            loss = decision_aware_loss(
                predictions,
                targets,
                weights,
                ambiguity_floor=ambiguity_floor,
            ) + planner_cfg.plan_loss_weight * plan_loss(
                plan_logits,
                plan_labels,
                weights,
                class_weights=plan_class_weights,
            ) + planner_cfg.margin_loss_weight * margin_loss(
                margin_logits,
                margin_targets,
                weights,
            ) + budget_loss_weight * budget_loss(
                budget_logits,
                budget_labels,
                weights,
                class_weights=budget_class_weights,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        step_count += 1
        update_ema_state(ema_state, model, step_count=step_count, decay=planner_cfg.ema_decay)

    training_seconds = time.perf_counter() - train_start
    raw_state = clone_model_state(model)
    raw_val_loss, raw_selection_loss = evaluate_validation(
        model,
        val_loader,
        device=device,
        ambiguity_floor=ambiguity_floor,
        plan_loss_weight=planner_cfg.plan_loss_weight,
        margin_loss_weight=planner_cfg.margin_loss_weight,
        budget_loss_weight=budget_loss_weight,
        plan_class_weights=plan_class_weights,
        budget_class_weights=budget_class_weights,
    )
    model.load_state_dict(ema_state, strict=True)
    ema_val_loss, ema_selection_loss = evaluate_validation(
        model,
        val_loader,
        device=device,
        ambiguity_floor=ambiguity_floor,
        plan_loss_weight=planner_cfg.plan_loss_weight,
        margin_loss_weight=planner_cfg.margin_loss_weight,
        budget_loss_weight=budget_loss_weight,
        plan_class_weights=plan_class_weights,
        budget_class_weights=budget_class_weights,
    )
    raw_is_better = raw_selection_loss < ema_selection_loss
    if math.isclose(raw_selection_loss, ema_selection_loss, rel_tol=1e-4, abs_tol=1e-6):
        raw_is_better = raw_val_loss <= ema_val_loss
    if raw_is_better:
        model.load_state_dict(raw_state, strict=True)
        val_loss = raw_val_loss
    else:
        val_loss = ema_val_loss
    eval_result = evaluate_ranked_model(
        model,
        task,
        device=device,
        batch_size=planner_cfg.eval_batch_size,
        rank_top_k=planner_cfg.rank_top_k,
        rank_min_score=planner_cfg.rank_min_score,
        use_dynamic_score_floor=planner_cfg.use_dynamic_score_floor,
        rank_floor_min_strength=planner_cfg.rank_floor_min_strength,
        rank_floor_quantile=planner_cfg.rank_floor_quantile,
        rank_floor_gap_scale=planner_cfg.rank_floor_gap_scale,
        use_soft_rank_sizing=planner_cfg.use_soft_rank_sizing,
        use_timestamp_budget_head=planner_cfg.use_timestamp_budget_head,
        use_budget_guided_keep_count=planner_cfg.use_budget_guided_keep_count,
        use_continuous_budget_thresholds=planner_cfg.use_continuous_budget_thresholds,
        use_budget_entropy_confidence=planner_cfg.use_budget_entropy_confidence,
        use_budget_consensus_dispersion=planner_cfg.use_budget_consensus_dispersion,
        budget_skip_scale=planner_cfg.budget_skip_scale,
        budget_selective_scale=planner_cfg.budget_selective_scale,
        budget_selective_top_k=planner_cfg.budget_selective_top_k,
        budget_selective_max_keep=planner_cfg.budget_selective_max_keep,
        budget_broad_top_k=planner_cfg.budget_broad_top_k,
        budget_broad_max_keep=planner_cfg.budget_broad_max_keep,
        budget_selective_gap_scale=planner_cfg.budget_selective_gap_scale,
        budget_broad_gap_scale=planner_cfg.budget_broad_gap_scale,
        budget_skip_gap_scale=planner_cfg.budget_skip_gap_scale,
        budget_skip_min_score_scale=planner_cfg.budget_skip_min_score_scale,
        budget_selective_min_score_scale=planner_cfg.budget_selective_min_score_scale,
        budget_broad_min_score_scale=planner_cfg.budget_broad_min_score_scale,
        budget_confidence_power=planner_cfg.budget_confidence_power,
        budget_selective_prior_skip_weight=planner_cfg.budget_selective_prior_skip_weight,
        budget_uncertainty_gap_floor=planner_cfg.budget_uncertainty_gap_floor,
        budget_uncertainty_fractional_floor=planner_cfg.budget_uncertainty_fractional_floor,
        budget_broad_consensus_power=planner_cfg.budget_broad_consensus_power,
        budget_consensus_selective_reallocation=planner_cfg.budget_consensus_selective_reallocation,
        budget_consensus_gap_floor=planner_cfg.budget_consensus_gap_floor,
        budget_consensus_fractional_floor=planner_cfg.budget_consensus_fractional_floor,
        rank_sizing_reference_quantile=planner_cfg.rank_sizing_reference_quantile,
        rank_sizing_regime_floor_scale=planner_cfg.rank_sizing_regime_floor_scale,
        rank_sizing_name_floor_scale=planner_cfg.rank_sizing_name_floor_scale,
        rank_sizing_regime_power=planner_cfg.rank_sizing_regime_power,
        rank_sizing_rank_power=planner_cfg.rank_sizing_rank_power,
        rank_min_keep=planner_cfg.rank_min_keep,
        rank_max_keep=planner_cfg.rank_max_keep,
        rank_reference_quantile=planner_cfg.rank_reference_quantile,
        rank_gap_scale=planner_cfg.rank_gap_scale,
    )
    summary = eval_result["summary"]
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    total_seconds = time.perf_counter() - total_start
    print("---")
    print(f"robust_score:      {float(summary['robust_score']):.6f}")
    print(f"val_loss:          {float(val_loss):.6f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"scenario_count:    {int(summary['scenario_count'])}")
    print(f"total_trade_count: {int(summary['total_trade_count'])}")
    print(f"train_samples:     {len(task.train_features)}")
    print(f"num_steps:         {int(step_count)}")
    print(f"frequency:         {task.config.frequency}")
    print(f"hold_bars:         {int(task.config.hold_bars)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
