from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, Protocol, Sequence, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.checkpoint_manager import TopKCheckpointManager
from src.torch_device_utils import move_module_to_runtime_device, resolve_runtime_device, should_auto_fallback_to_cpu

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
    TASK_INPUT_CHECK_WORKERS_ENV_VAR,
    TIME_BUDGET,
    apply_execution_modifiers,
    build_action_frame,
    data_root_help_text,
    lag_action_frame,
    parse_csv_list,
    parse_int_list,
    prepare_task,
    resolve_task_config,
    run_task_input_check,
    simulate_actions,
    symbols_help_text,
)

DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_AUTO_LR_CANDIDATES: tuple[float, ...] = (1e-4, 2e-4, 3e-4, 5e-4, 7.5e-4, 1e-3)
DEFAULT_TOP_K_CHECKPOINTS = 5
RuntimeValueT = TypeVar("RuntimeValueT")


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
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.10
    symbol_embedding_dim: int = 32
    context_blocks: int = 2
    batch_size: int = 384
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    eval_batch_size: int = 768
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
    dataloader_workers: int = 0
    deterministic: bool = False
    seed: int = 20260310


@dataclass(frozen=True)
class ExecutionModifierSet:
    buy_price_modifier_bps: float = 0.0
    sell_price_modifier_bps: float = 0.0
    amount_modifier_pct: float = 0.0


class RuntimeExecutionOperation(Protocol[RuntimeValueT]):
    def __call__(self, device: torch.device, fallback_used: bool, /) -> RuntimeValueT: ...


@dataclass(frozen=True)
class RuntimeExecutionResult(Generic[RuntimeValueT]):
    value: RuntimeValueT
    final_device: torch.device
    fallback_used: bool


class ResidualContextBlock(nn.Module):
    def __init__(self, width: int, *, dropout: float) -> None:
        super().__init__()
        inner = max(int(width) * 2, 64)
        self.norm = nn.LayerNorm(int(width))
        self.ff = nn.Sequential(
            nn.Linear(int(width), inner),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner, int(width)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.ff(self.norm(inputs))


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
        context_blocks: int,
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
        context_width = int(hidden_size) + int(symbol_embedding_dim)
        plan_hidden_size = max(context_width // 2, 16)
        self.context_blocks = nn.ModuleList(
            ResidualContextBlock(context_width, dropout=float(dropout))
            for _ in range(max(int(context_blocks), 0))
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(context_width),
            nn.Linear(context_width, hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 3),
        )
        self.plan_head = nn.Sequential(
            nn.LayerNorm(context_width),
            nn.Linear(context_width, plan_hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(plan_hidden_size, 5),
        )
        self.margin_head = nn.Sequential(
            nn.LayerNorm(context_width),
            nn.Linear(context_width, plan_hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(plan_hidden_size, 1),
        )
        self.budget_head = nn.Sequential(
            nn.LayerNorm(context_width),
            nn.Linear(context_width, plan_hidden_size),
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
        combined = torch.cat([summary, symbol_context], dim=-1)
        for block in self.context_blocks:
            combined = block(combined)
        return combined

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
    with torch.inference_mode():
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
    buy_price_modifier_bps: float = 0.0,
    sell_price_modifier_bps: float = 0.0,
    amount_modifier_pct: float = 0.0,
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
        actions = apply_execution_modifiers(
            actions,
            buy_price_modifier_bps=float(buy_price_modifier_bps),
            sell_price_modifier_bps=float(sell_price_modifier_bps),
            amount_modifier_pct=float(amount_modifier_pct),
        )
        actions = lag_action_frame(actions, scenario.bars, int(task.config.decision_lag_bars))
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


def tune_execution_modifiers(
    evaluate_fn,
    *,
    buy_grid_bps: Sequence[float],
    sell_grid_bps: Sequence[float],
    amount_grid_pct: Sequence[float],
) -> tuple[ExecutionModifierSet, dict[str, Any]]:
    grids = {
        "buy_price_modifier_bps": tuple(float(value) for value in buy_grid_bps) or (0.0,),
        "sell_price_modifier_bps": tuple(float(value) for value in sell_grid_bps) or (0.0,),
        "amount_modifier_pct": tuple(float(value) for value in amount_grid_pct) or (0.0,),
    }
    best = ExecutionModifierSet()
    best_result = evaluate_fn(best)
    best_score = float(best_result["summary"]["robust_score"])

    for field_name, candidates in grids.items():
        for candidate in candidates:
            payload = dict(best.__dict__)
            payload[field_name] = float(candidate)
            proposal = ExecutionModifierSet(**payload)
            result = evaluate_fn(proposal)
            score = float(result["summary"]["robust_score"])
            if score > best_score:
                best = proposal
                best_result = result
                best_score = score

    return best, best_result

def seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not bool(deterministic)
        torch.backends.cudnn.deterministic = bool(deterministic)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = not bool(deterministic)
        except Exception:
            pass


def resolve_dataloader_workers(*, device: torch.device, requested: int) -> int:
    if int(requested) > 0:
        return int(requested)
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count // 2))


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
    non_blocking = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            targets = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            symbol_ids = batch["symbol_ids"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
            weights = batch["weights"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            plan_labels = batch["plan_labels"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
            margin_targets = batch["margin_targets"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            budget_labels = batch["budget_labels"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
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


def build_dataloaders(
    task,
    cfg: PlannerConfig,
    *,
    ambiguity_floor: float,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
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
    dataloader_kwargs: dict[str, Any] = {
        "num_workers": int(cfg.dataloader_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(cfg.dataloader_workers) > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        drop_last=bool(len(train_dataset) >= int(cfg.batch_size)),
        generator=generator,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs,
    )
    return train_loader, val_loader, derive_plan_class_weights(train_plan_labels), budget_class_weights


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def load_planner_model(
    *,
    model_factory,
    requested_device: str | None,
    device: torch.device,
    context: str,
) -> tuple[PlannerNet, torch.device]:
    return move_module_to_runtime_device(
        model_factory(),
        requested_device=requested_device,
        device=device,
        context=context,
    )


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


def _cuda_total_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.get_device_properties(device).total_memory) / (1024.0 ** 3)


def resolve_planner_sizing(
    *,
    frequency: str,
    device: torch.device,
    hidden_size: int | None,
    num_layers: int | None,
    symbol_embedding_dim: int | None,
    batch_size: int | None,
    eval_batch_size: int | None,
) -> tuple[int, int, int, int, int]:
    if device.type != "cuda":
        default_hidden = 160 if frequency == "hourly" else 128
        default_layers = 2
        default_embedding = 16
        default_batch = 128
        default_eval_batch = 256
    else:
        vram_gb = _cuda_total_memory_gb(device)
        if vram_gb >= 20.0:
            default_hidden = 384 if frequency == "hourly" else 320
            default_layers = 3 if frequency == "hourly" else 2
            default_embedding = 32
            default_batch = 512 if frequency == "hourly" else 384
            default_eval_batch = 1024
        elif vram_gb >= 10.0:
            default_hidden = 256 if frequency == "hourly" else 224
            default_layers = 2
            default_embedding = 24
            default_batch = 384 if frequency == "hourly" else 256
            default_eval_batch = 768
        else:
            default_hidden = 192 if frequency == "hourly" else 160
            default_layers = 2
            default_embedding = 16
            default_batch = 192 if frequency == "hourly" else 160
            default_eval_batch = 512

    return (
        int(hidden_size if hidden_size is not None else default_hidden),
        int(num_layers if num_layers is not None else default_layers),
        int(symbol_embedding_dim if symbol_embedding_dim is not None else default_embedding),
        int(batch_size if batch_size is not None else default_batch),
        int(eval_batch_size if eval_batch_size is not None else default_eval_batch),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def parse_float_list(raw: str | None) -> tuple[float, ...]:
    if raw is None:
        return ()
    values: list[float] = []
    for token in parse_csv_list(raw):
        values.append(float(token))
    return tuple(values)


def _cuda_runtime_diagnostics(
    requested: str | None,
    *,
    resolved_device: torch.device | None = None,
) -> str:
    requested_token = (requested or "auto").strip().lower()
    diagnostics = [f"requested_device={requested_token!r}"]
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        diagnostics.append(f"cuda_available_error={type(exc).__name__}: {exc}")
    else:
        diagnostics.append(f"cuda_available={cuda_available}")
    try:
        diagnostics.append(f"cuda_device_count={int(torch.cuda.device_count())}")
    except Exception as exc:
        diagnostics.append(f"cuda_device_count_error={type(exc).__name__}: {exc}")
    if resolved_device is not None:
        diagnostics.append(f"resolved_device={resolved_device}")
    return ", ".join(diagnostics)


def resolve_autoresearch_training_device(requested: str | None) -> torch.device:
    token = (requested or "auto").strip().lower()
    if token == "cpu":
        raise RuntimeError(
            "Autoresearch stock trainer requires CUDA and cannot run on CPU "
            f"({_cuda_runtime_diagnostics(requested)})."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Autoresearch stock trainer requires CUDA but no CUDA device is available "
            f"({_cuda_runtime_diagnostics(requested)})."
        )
    device = resolve_runtime_device(requested)
    if device.type != "cuda":
        raise RuntimeError(
            "Autoresearch stock trainer requires CUDA, "
            f"got resolved device {device} ({_cuda_runtime_diagnostics(requested, resolved_device=device)})."
        )
    return device


def resolve_execution_modifier_tuning_enabled(*, requested: bool, disabled: bool) -> bool:
    return bool(requested) and not bool(disabled)


def run_with_auto_cpu_fallback(
    *,
    requested_device: str | None,
    device: torch.device,
    context: str,
    operation: RuntimeExecutionOperation[RuntimeValueT],
) -> RuntimeExecutionResult[RuntimeValueT]:
    current_device = device
    fallback_used = False
    while True:
        try:
            return RuntimeExecutionResult(
                value=operation(current_device, fallback_used),
                final_device=current_device,
                fallback_used=fallback_used,
            )
        except Exception as exc:
            if fallback_used or not should_auto_fallback_to_cpu(requested_device, current_device, exc):
                raise
            warnings.warn(
                f"{context}: auto-selected CUDA unavailable during execution, falling back to CPU: {exc}",
                RuntimeWarning,
            )
            if current_device.type == "cuda":
                torch.cuda.empty_cache()
            current_device = torch.device("cpu")
            fallback_used = True


def auto_lr_cache_path(cache_path: str | None = None) -> Path:
    raw_path = cache_path or os.getenv(
        "AUTORESEARCH_STOCK_AUTO_LR_CACHE_PATH",
        ".cache/autoresearch_stock/auto_lr_cache.json",
    )
    return Path(raw_path)


def build_auto_lr_cache_key(
    *,
    frequency: str,
    device: torch.device,
    feature_dim: int,
    symbol_count: int,
    hidden_size: int,
    num_layers: int,
    symbol_embedding_dim: int,
    context_blocks: int,
    batch_size: int,
) -> str:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        total_memory_gb = round(_cuda_total_memory_gb(device), 2)
    else:
        device_name = str(device)
        total_memory_gb = 0.0
    return json.dumps(
        {
            "batch_size": int(batch_size),
            "context_blocks": int(context_blocks),
            "device_name": str(device_name),
            "feature_dim": int(feature_dim),
            "frequency": str(frequency),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "symbol_count": int(symbol_count),
            "symbol_embedding_dim": int(symbol_embedding_dim),
            "total_memory_gb": float(total_memory_gb),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def load_auto_lr_cache(path: Path) -> dict[str, float]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    cache: dict[str, float] = {}
    for key, value in raw.items():
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float) and value_float > 0.0:
            cache[str(key)] = value_float
    return cache


def write_auto_lr_cache(path: Path, key: str, learning_rate: float) -> None:
    cache = load_auto_lr_cache(path)
    cache[str(key)] = float(learning_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def resolve_learning_rate(
    *,
    requested_lr: float | None,
    disable_auto_lr_find: bool,
    cache_path: str | None,
    cache_key: str | None,
    tuner,
) -> tuple[float, str]:
    if requested_lr is not None:
        return float(requested_lr), "cli"
    if bool(disable_auto_lr_find) or cache_key is None:
        return float(DEFAULT_LEARNING_RATE), "default"

    path = auto_lr_cache_path(cache_path)
    cached_lr = load_auto_lr_cache(path).get(cache_key)
    if cached_lr is not None:
        return float(cached_lr), "cache"

    resolved_lr = float(tuner())
    if math.isfinite(resolved_lr) and resolved_lr > 0.0:
        write_auto_lr_cache(path, cache_key, resolved_lr)
        return resolved_lr, "auto"
    return float(DEFAULT_LEARNING_RATE), "default"


def _preview_training_batches(loader: DataLoader, *, max_batches: int) -> list[dict[str, torch.Tensor]]:
    preview_batches: list[dict[str, torch.Tensor]] = []
    iterator = iter(loader)
    for _ in range(max(int(max_batches), 0)):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        preview_batches.append(batch)
    return preview_batches


def _planner_objective(
    model: PlannerNet,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    ambiguity_floor: float,
    plan_loss_weight: float,
    margin_loss_weight: float,
    budget_loss_weight: float,
    plan_class_weights: torch.Tensor,
    budget_class_weights: torch.Tensor,
) -> torch.Tensor:
    non_blocking = device.type == "cuda"
    features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    targets = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    symbol_ids = batch["symbol_ids"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
    weights = batch["weights"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    plan_labels = batch["plan_labels"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
    margin_targets = batch["margin_targets"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    budget_labels = batch["budget_labels"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
    plan_class_weights = plan_class_weights.to(device=device, dtype=torch.float32)
    budget_class_weights = budget_class_weights.to(device=device, dtype=torch.float32)

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        predictions, plan_logits, margin_logits, budget_logits = model.predict_with_plan(features, symbol_ids)
        return decision_aware_loss(
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


def auto_find_learning_rate(
    *,
    model_factory,
    base_state: dict[str, torch.Tensor],
    train_loader: DataLoader,
    requested_device: str | None,
    device: torch.device,
    ambiguity_floor: float,
    plan_loss_weight: float,
    margin_loss_weight: float,
    budget_loss_weight: float,
    plan_class_weights: torch.Tensor,
    budget_class_weights: torch.Tensor,
    weight_decay: float,
    candidate_lrs: Sequence[float] = DEFAULT_AUTO_LR_CANDIDATES,
    max_batches: int = 3,
) -> float:
    preview_batches = _preview_training_batches(train_loader, max_batches=max_batches)
    if not preview_batches:
        return float(DEFAULT_LEARNING_RATE)

    best_lr = float(DEFAULT_LEARNING_RATE)
    best_score = float("inf")
    for learning_rate in tuple(float(value) for value in candidate_lrs):
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            continue
        model, tuner_device = load_planner_model(
            model_factory=model_factory,
            requested_device=requested_device,
            device=device,
            context="Autoresearch stock auto-lr probe",
        )
        model.load_state_dict(base_state, strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(tuner_device.type == "cuda"))
        losses: list[float] = []
        unstable = False
        retried_on_cpu = False
        batch_index = 0

        while batch_index < len(preview_batches):
            batch = preview_batches[batch_index]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            try:
                loss = _planner_objective(
                    model,
                    batch,
                    device=tuner_device,
                    ambiguity_floor=ambiguity_floor,
                    plan_loss_weight=plan_loss_weight,
                    margin_loss_weight=margin_loss_weight,
                    budget_loss_weight=budget_loss_weight,
                    plan_class_weights=plan_class_weights,
                    budget_class_weights=budget_class_weights,
                )
                loss_value = float(loss.detach().item())
                if not math.isfinite(loss_value):
                    unstable = True
                    break
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss_value)
                batch_index += 1
            except Exception as exc:
                if retried_on_cpu or not should_auto_fallback_to_cpu(requested_device, tuner_device, exc):
                    raise
                retried_on_cpu = True
                del model
                del optimizer
                del scaler
                if tuner_device.type == "cuda":
                    torch.cuda.empty_cache()
                tuner_device = torch.device("cpu")
                model = model_factory().to(tuner_device)
                model.load_state_dict(base_state, strict=True)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(learning_rate),
                    weight_decay=float(weight_decay),
                )
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                losses = []
                unstable = False
                batch_index = 0

        if not unstable and losses:
            score = float(sum(losses[-2:]) / min(len(losses), 2))
            if score < best_score:
                best_score = score
                best_lr = float(learning_rate)
        del model
        del optimizer
        del scaler
        if tuner_device.type == "cuda":
            torch.cuda.empty_cache()

    return float(best_lr)


def default_autoresearch_checkpoint_root() -> Path:
    env_path = os.getenv("AUTORESEARCH_STOCK_CHECKPOINT_ROOT")
    if env_path:
        return Path(env_path)
    preferred = Path("/sdb-disk/code/stock-prediction/checkpoints/autoresearch_stock")
    if preferred.parent.parent.exists():
        return preferred
    return Path("checkpoints/autoresearch_stock")


def resolve_autoresearch_checkpoint_dir(*, frequency: str, checkpoint_dir: str | None) -> Path:
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    return default_autoresearch_checkpoint_root() / str(frequency).strip().lower()


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _task_config_payload(task_config) -> dict[str, Any]:
    return {
        "frequency": str(task_config.frequency),
        "data_root": str(task_config.data_root),
        "recent_data_root": None if task_config.recent_data_root is None else str(task_config.recent_data_root),
        "symbols": list(task_config.symbols),
        "sequence_length": int(task_config.sequence_length),
        "hold_bars": int(task_config.hold_bars),
        "eval_windows": [int(value) for value in task_config.eval_windows],
        "recent_overlay_bars": int(task_config.recent_overlay_bars),
        "initial_cash": float(task_config.initial_cash),
        "max_positions": int(task_config.max_positions),
        "max_volume_fraction": float(task_config.max_volume_fraction),
        "min_edge_bps": float(task_config.min_edge_bps),
        "entry_slippage_bps": float(task_config.entry_slippage_bps),
        "exit_slippage_bps": float(task_config.exit_slippage_bps),
        "decision_lag_bars": int(task_config.decision_lag_bars),
        "allow_short": bool(task_config.allow_short),
        "close_at_session_end": bool(task_config.close_at_session_end),
        "spread_lookback_days": int(task_config.spread_lookback_days),
        "periods_per_year": float(task_config.periods_per_year),
        "annual_leverage_rate": float(getattr(task_config, "annual_leverage_rate", 0.0625)),
        "max_gross_leverage": float(getattr(task_config, "max_gross_leverage", 2.0)),
        "dashboard_db_path": str(task_config.dashboard_db_path),
    }


def _current_topk_paths(manifest_path: Path) -> list[Path]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    paths: list[Path] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        raw_path = str(row.get("path") or "").strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists():
            paths.append(path)
    return paths


def _refresh_best_checkpoint_alias(checkpoint_dir: Path, *, manifest_path: Path) -> Path | None:
    topk_paths = _current_topk_paths(manifest_path)
    if not topk_paths:
        return None
    best_path = topk_paths[0]
    best_alias = checkpoint_dir / "best.pt"
    if best_path.resolve() != best_alias.resolve():
        shutil.copy2(best_path, best_alias)
    return best_path


def save_autoresearch_checkpoint(
    *,
    checkpoint_dir: Path,
    frequency: str,
    model: nn.Module,
    planner_cfg: PlannerConfig,
    task,
    summary: dict[str, Any],
    val_loss: float,
    model_parameters: int,
    step_count: int,
    training_seconds: float,
    total_seconds: float,
    peak_vram_mb: float,
    runtime_device: str,
    auto_cpu_fallback_used: bool,
    best_modifiers: ExecutionModifierSet,
    learning_rate_source: str,
    execution_modifier_tuning_enabled: bool,
    top_k_checkpoints: int,
) -> tuple[Path, Path | None]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    robust_score = float(summary["robust_score"])
    score_tag = f"{robust_score:+010.4f}".replace("+", "p").replace("-", "m")
    checkpoint_path = checkpoint_dir / f"{str(frequency).lower()}_{created_at}_{score_tag}.pt"

    payload = {
        "model": _clone_state_dict_to_cpu(model),
        "metadata": {
            "created_at": created_at,
            "frequency": str(frequency),
            "planner_config": asdict(planner_cfg),
            "task_config": _task_config_payload(task.config),
            "summary": {key: float(value) for key, value in summary.items()},
            "val_loss": float(val_loss),
            "model_parameters": int(model_parameters),
            "num_steps": int(step_count),
            "training_seconds": float(training_seconds),
            "total_seconds": float(total_seconds),
            "peak_vram_mb": float(peak_vram_mb),
            "runtime_device": str(runtime_device),
            "auto_cpu_fallback_used": bool(auto_cpu_fallback_used),
            "learning_rate_source": str(learning_rate_source),
            "execution_modifier_tuning_enabled": bool(execution_modifier_tuning_enabled),
            "best_modifiers": asdict(best_modifiers),
            "train_samples": int(len(task.train_features)),
            "symbol_count": int(max(len(task.symbol_to_id), 1)),
            "feature_dim": int(task.train_features.shape[-1]),
            "top_k_checkpoints": int(top_k_checkpoints),
        },
    }
    torch.save(payload, checkpoint_path)

    latest_path = checkpoint_dir / "latest.pt"
    latest_summary_path = checkpoint_dir / "latest_summary.json"
    shutil.copy2(checkpoint_path, latest_path)
    latest_summary_path.write_text(
        json.dumps(payload["metadata"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manager = TopKCheckpointManager(checkpoint_dir, max_keep=max(int(top_k_checkpoints), 1), mode="max")
    manager.register(checkpoint_path, robust_score)
    best_path = _refresh_best_checkpoint_alias(checkpoint_dir, manifest_path=(checkpoint_dir / ".topk_manifest.json"))
    return checkpoint_path, best_path


def _task_input_check_cli_args(argv: Sequence[str] | None) -> list[str]:
    return list(sys.argv[1:] if argv is None else argv)


def _without_task_input_check_flags(args: Sequence[str]) -> list[str]:
    return [arg for arg in args if arg not in {"--check-inputs", "--check-inputs-text"}]


def _task_input_follow_up_command(
    args: Sequence[str],
    *,
    add_check_inputs: bool,
) -> str:
    command = [sys.executable, "-m", "autoresearch_stock.train", *_without_task_input_check_flags(args)]
    if add_check_inputs:
        command.append("--check-inputs")
    return shlex.join(command)


def _task_input_check_failure_message(exc: Exception) -> str:
    message = f"Autoresearch stock input check failed: {exc}"
    if TASK_INPUT_CHECK_WORKERS_ENV_VAR in str(exc):
        return (
            f"{message}\n"
            f"Unset or set {TASK_INPUT_CHECK_WORKERS_ENV_VAR} to a positive integer before rerunning."
        )
    return message


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the autoresearch-style stock planner for a fixed time budget.")
    parser.add_argument("--frequency", choices=("hourly", "daily"), default="hourly")
    parser.add_argument("--symbols", default="", help=symbols_help_text())
    parser.add_argument("--data-root", default=None, help=data_root_help_text())
    parser.add_argument(
        "--recent-data-root",
        default=None,
        help="Optional overlay directory with recent per-symbol CSVs named like SYMBOL.csv.",
    )
    check_inputs_group = parser.add_mutually_exclusive_group()
    check_inputs_group.add_argument("--check-inputs", action="store_true")
    check_inputs_group.add_argument("--check-inputs-text", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--hold-bars", type=int, default=None)
    parser.add_argument("--eval-windows", default="")
    parser.add_argument("--recent-overlay-bars", type=int, default=0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-volume-fraction", type=float, default=None)
    parser.add_argument("--min-edge-bps", type=float, default=4.0)
    parser.add_argument("--entry-slippage-bps", type=float, default=1.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=1.0)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--disable-short", action="store_true")
    parser.add_argument("--dashboard-db", default="dashboards/metrics.db")
    parser.add_argument("--spread-lookback-days", type=int, default=14)
    parser.add_argument("--annual-leverage-rate", type=float, default=0.0625)
    parser.add_argument("--max-gross-leverage", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--symbol-embedding-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--disable-auto-lr-find", action="store_true")
    parser.add_argument("--auto-lr-cache", default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--context-blocks", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--top-k-checkpoints", type=int, default=DEFAULT_TOP_K_CHECKPOINTS)
    parser.add_argument("--execution-modifier-tuning", action="store_true")
    parser.add_argument("--disable-execution-modifier-tuning", action="store_true")
    parser.add_argument("--buy-price-modifier-grid-bps", default="-25,-15,-10,-5,0,5,10,15,25")
    parser.add_argument("--sell-price-modifier-grid-bps", default="-25,-15,-10,-5,0,5,10,15,25")
    parser.add_argument("--amount-modifier-grid-pct", default="-30,-20,-10,0,10,20,30")
    parser.add_argument("--dynamic-score-floor", action="store_true")
    parser.add_argument("--soft-rank-sizing", action="store_true")
    parser.add_argument("--timestamp-budget-head", action="store_true")
    parser.add_argument("--budget-guided-keep-count", action="store_true")
    parser.add_argument("--continuous-budget-thresholds", action="store_true")
    parser.add_argument("--budget-entropy-confidence", action="store_true")
    parser.add_argument("--budget-consensus-dispersion", action="store_true")
    args = parser.parse_args(argv)

    task_config = resolve_task_config(
        frequency=args.frequency,
        symbols=parse_csv_list(args.symbols) or None,
        data_root=args.data_root,
        recent_data_root=args.recent_data_root,
        sequence_length=args.sequence_length,
        hold_bars=args.hold_bars,
        eval_windows=parse_int_list(args.eval_windows) or None,
        recent_overlay_bars=args.recent_overlay_bars,
        max_positions=args.max_positions,
        max_volume_fraction=args.max_volume_fraction,
        min_edge_bps=args.min_edge_bps,
        entry_slippage_bps=args.entry_slippage_bps,
        exit_slippage_bps=args.exit_slippage_bps,
        decision_lag_bars=args.decision_lag_bars,
        allow_short=not bool(args.disable_short),
        dashboard_db_path=args.dashboard_db,
        spread_lookback_days=args.spread_lookback_days,
        annual_leverage_rate=args.annual_leverage_rate,
        max_gross_leverage=args.max_gross_leverage,
    )
    cli_args = _task_input_check_cli_args(argv)
    if args.check_inputs or args.check_inputs_text:
        try:
            check_result = run_task_input_check(task_config, text_output=bool(args.check_inputs_text))
        except (OSError, RuntimeError, ValueError) as exc:
            print(_task_input_check_failure_message(exc), file=sys.stderr)
            return 2
        if args.check_inputs_text:
            print(check_result.rendered_output)
            if check_result.payload["all_symbols_ready"]:
                print("Suggested command:")
                print(f"  {_task_input_follow_up_command(cli_args, add_check_inputs=False)}")
            else:
                print("Suggested JSON command:")
                print(f"  {_task_input_follow_up_command(cli_args, add_check_inputs=True)}")
        else:
            print(check_result.rendered_output)
        return check_result.exit_code

    total_start = time.perf_counter()
    device = resolve_autoresearch_training_device(args.device)
    seed_everything(args.seed, deterministic=bool(args.deterministic))
    resolved_hidden_size, resolved_layers, resolved_symbol_embedding_dim, resolved_batch_size, resolved_eval_batch_size = (
        resolve_planner_sizing(
            frequency=args.frequency,
            device=device,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            symbol_embedding_dim=args.symbol_embedding_dim,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
        )
    )
    task = prepare_task(task_config)
    planner_cfg = PlannerConfig(
        hidden_size=int(resolved_hidden_size),
        num_layers=int(resolved_layers),
        dropout=float(args.dropout),
        symbol_embedding_dim=int(resolved_symbol_embedding_dim),
        context_blocks=int(args.context_blocks),
        batch_size=int(resolved_batch_size),
        learning_rate=float(DEFAULT_LEARNING_RATE if args.lr is None else args.lr),
        weight_decay=float(args.weight_decay),
        eval_batch_size=int(resolved_eval_batch_size),
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
        dataloader_workers=resolve_dataloader_workers(
            device=device,
            requested=(-1 if args.num_workers is None else int(args.num_workers)),
        ),
        deterministic=bool(args.deterministic),
        seed=int(args.seed),
    )
    ambiguity_floor = compute_ambiguity_floor(
        task.train_targets,
        quantile=planner_cfg.ambiguity_quantile,
    )
    budget_loss_weight = planner_cfg.budget_loss_weight if planner_cfg.use_timestamp_budget_head else 0.0

    model_factory = lambda: PlannerNet(
        feature_dim=int(task.train_features.shape[-1]),
        symbol_count=max(len(task.symbol_to_id), 1),
        hidden_size=planner_cfg.hidden_size,
        num_layers=planner_cfg.num_layers,
        dropout=planner_cfg.dropout,
        symbol_embedding_dim=planner_cfg.symbol_embedding_dim,
        context_blocks=planner_cfg.context_blocks,
        weak_action_scale=planner_cfg.weak_action_scale,
    )
    def _run_training_for_device(runtime_device: torch.device, auto_cpu_fallback_used: bool) -> int:
        planner_cfg.dataloader_workers = resolve_dataloader_workers(
            device=runtime_device,
            requested=(-1 if args.num_workers is None else int(args.num_workers)),
        )
        model, runtime_device = load_planner_model(
            model_factory=model_factory,
            requested_device=args.device,
            device=runtime_device,
            context="Autoresearch stock planner",
        )
        train_loader, val_loader, plan_class_weights_np, budget_class_weights_np = build_dataloaders(
            task,
            planner_cfg,
            ambiguity_floor=ambiguity_floor,
            pin_memory=(runtime_device.type == "cuda"),
        )
        model_parameters = count_trainable_parameters(model)
        ema_state = clone_model_state(model)
        plan_class_weights = torch.from_numpy(plan_class_weights_np).to(device=runtime_device, dtype=torch.float32)
        budget_class_weights = torch.from_numpy(budget_class_weights_np).to(device=runtime_device, dtype=torch.float32)
        learning_rate, learning_rate_source = resolve_learning_rate(
            requested_lr=args.lr,
            disable_auto_lr_find=bool(args.disable_auto_lr_find),
            cache_path=args.auto_lr_cache,
            cache_key=build_auto_lr_cache_key(
                frequency=task.config.frequency,
                device=runtime_device,
                feature_dim=int(task.train_features.shape[-1]),
                symbol_count=max(len(task.symbol_to_id), 1),
                hidden_size=planner_cfg.hidden_size,
                num_layers=planner_cfg.num_layers,
                symbol_embedding_dim=planner_cfg.symbol_embedding_dim,
                context_blocks=planner_cfg.context_blocks,
                batch_size=planner_cfg.batch_size,
            ),
            tuner=lambda: auto_find_learning_rate(
                model_factory=model_factory,
                base_state=ema_state,
                train_loader=train_loader,
                requested_device=args.device,
                device=runtime_device,
                ambiguity_floor=ambiguity_floor,
                plan_loss_weight=planner_cfg.plan_loss_weight,
                margin_loss_weight=planner_cfg.margin_loss_weight,
                budget_loss_weight=budget_loss_weight,
                plan_class_weights=plan_class_weights,
                budget_class_weights=budget_class_weights,
                weight_decay=planner_cfg.weight_decay,
            ),
        )
        planner_cfg.learning_rate = float(learning_rate)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(planner_cfg.learning_rate),
            weight_decay=float(planner_cfg.weight_decay),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(runtime_device.type == "cuda"))

        if runtime_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime_device)

        train_start = time.perf_counter()
        step_count = 0
        data_iter = iter(train_loader)
        non_blocking = runtime_device.type == "cuda"
        while (time.perf_counter() - train_start) < TIME_BUDGET:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device=runtime_device, dtype=torch.float32, non_blocking=non_blocking)
            targets = batch["targets"].to(device=runtime_device, dtype=torch.float32, non_blocking=non_blocking)
            symbol_ids = batch["symbol_ids"].to(device=runtime_device, dtype=torch.long, non_blocking=non_blocking)
            weights = batch["weights"].to(device=runtime_device, dtype=torch.float32, non_blocking=non_blocking)
            plan_labels = batch["plan_labels"].to(device=runtime_device, dtype=torch.long, non_blocking=non_blocking)
            margin_targets = batch["margin_targets"].to(device=runtime_device, dtype=torch.float32, non_blocking=non_blocking)
            budget_labels = batch["budget_labels"].to(device=runtime_device, dtype=torch.long, non_blocking=non_blocking)

            autocast_enabled = runtime_device.type == "cuda"
            with torch.amp.autocast(device_type=runtime_device.type, enabled=autocast_enabled):
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
            device=runtime_device,
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
            device=runtime_device,
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
        eval_kwargs = {
            "device": runtime_device,
            "batch_size": planner_cfg.eval_batch_size,
            "rank_top_k": planner_cfg.rank_top_k,
            "rank_min_score": planner_cfg.rank_min_score,
            "use_dynamic_score_floor": planner_cfg.use_dynamic_score_floor,
            "rank_floor_min_strength": planner_cfg.rank_floor_min_strength,
            "rank_floor_quantile": planner_cfg.rank_floor_quantile,
            "rank_floor_gap_scale": planner_cfg.rank_floor_gap_scale,
            "use_soft_rank_sizing": planner_cfg.use_soft_rank_sizing,
            "use_timestamp_budget_head": planner_cfg.use_timestamp_budget_head,
            "use_budget_guided_keep_count": planner_cfg.use_budget_guided_keep_count,
            "use_continuous_budget_thresholds": planner_cfg.use_continuous_budget_thresholds,
            "use_budget_entropy_confidence": planner_cfg.use_budget_entropy_confidence,
            "use_budget_consensus_dispersion": planner_cfg.use_budget_consensus_dispersion,
            "budget_skip_scale": planner_cfg.budget_skip_scale,
            "budget_selective_scale": planner_cfg.budget_selective_scale,
            "budget_selective_top_k": planner_cfg.budget_selective_top_k,
            "budget_selective_max_keep": planner_cfg.budget_selective_max_keep,
            "budget_broad_top_k": planner_cfg.budget_broad_top_k,
            "budget_broad_max_keep": planner_cfg.budget_broad_max_keep,
            "budget_selective_gap_scale": planner_cfg.budget_selective_gap_scale,
            "budget_broad_gap_scale": planner_cfg.budget_broad_gap_scale,
            "budget_skip_gap_scale": planner_cfg.budget_skip_gap_scale,
            "budget_skip_min_score_scale": planner_cfg.budget_skip_min_score_scale,
            "budget_selective_min_score_scale": planner_cfg.budget_selective_min_score_scale,
            "budget_broad_min_score_scale": planner_cfg.budget_broad_min_score_scale,
            "budget_confidence_power": planner_cfg.budget_confidence_power,
            "budget_selective_prior_skip_weight": planner_cfg.budget_selective_prior_skip_weight,
            "budget_uncertainty_gap_floor": planner_cfg.budget_uncertainty_gap_floor,
            "budget_uncertainty_fractional_floor": planner_cfg.budget_uncertainty_fractional_floor,
            "budget_broad_consensus_power": planner_cfg.budget_broad_consensus_power,
            "budget_consensus_selective_reallocation": planner_cfg.budget_consensus_selective_reallocation,
            "budget_consensus_gap_floor": planner_cfg.budget_consensus_gap_floor,
            "budget_consensus_fractional_floor": planner_cfg.budget_consensus_fractional_floor,
            "rank_sizing_reference_quantile": planner_cfg.rank_sizing_reference_quantile,
            "rank_sizing_regime_floor_scale": planner_cfg.rank_sizing_regime_floor_scale,
            "rank_sizing_name_floor_scale": planner_cfg.rank_sizing_name_floor_scale,
            "rank_sizing_regime_power": planner_cfg.rank_sizing_regime_power,
            "rank_sizing_rank_power": planner_cfg.rank_sizing_rank_power,
            "rank_min_keep": planner_cfg.rank_min_keep,
            "rank_max_keep": planner_cfg.rank_max_keep,
            "rank_reference_quantile": planner_cfg.rank_reference_quantile,
            "rank_gap_scale": planner_cfg.rank_gap_scale,
        }

        def _evaluate_with_modifiers(modifiers: ExecutionModifierSet) -> dict[str, Any]:
            return evaluate_ranked_model(
                model,
                task,
                buy_price_modifier_bps=modifiers.buy_price_modifier_bps,
                sell_price_modifier_bps=modifiers.sell_price_modifier_bps,
                amount_modifier_pct=modifiers.amount_modifier_pct,
                **eval_kwargs,
            )

        best_modifiers = ExecutionModifierSet()
        execution_modifier_tuning_enabled = resolve_execution_modifier_tuning_enabled(
            requested=bool(args.execution_modifier_tuning),
            disabled=bool(args.disable_execution_modifier_tuning),
        )
        eval_result = _evaluate_with_modifiers(best_modifiers)
        if execution_modifier_tuning_enabled:
            best_modifiers, eval_result = tune_execution_modifiers(
                _evaluate_with_modifiers,
                buy_grid_bps=parse_float_list(args.buy_price_modifier_grid_bps),
                sell_grid_bps=parse_float_list(args.sell_price_modifier_grid_bps),
                amount_grid_pct=parse_float_list(args.amount_modifier_grid_pct),
            )
        summary = eval_result["summary"]
        peak_vram_mb = 0.0
        if runtime_device.type == "cuda":
            peak_vram_mb = float(torch.cuda.max_memory_allocated(runtime_device) / (1024.0 * 1024.0))

        total_seconds = time.perf_counter() - total_start
        checkpoint_dir = resolve_autoresearch_checkpoint_dir(
            frequency=task.config.frequency,
            checkpoint_dir=args.checkpoint_dir,
        )
        saved_checkpoint_path, best_checkpoint_path = save_autoresearch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            frequency=task.config.frequency,
            model=model,
            planner_cfg=planner_cfg,
            task=task,
            summary=summary,
            val_loss=float(val_loss),
            model_parameters=model_parameters,
            step_count=step_count,
            training_seconds=training_seconds,
            total_seconds=total_seconds,
            peak_vram_mb=peak_vram_mb,
            runtime_device=str(runtime_device),
            auto_cpu_fallback_used=auto_cpu_fallback_used,
            best_modifiers=best_modifiers,
            learning_rate_source=learning_rate_source,
            execution_modifier_tuning_enabled=execution_modifier_tuning_enabled,
            top_k_checkpoints=int(args.top_k_checkpoints),
        )
        print("---")
        print(f"robust_score:      {float(summary['robust_score']):.6f}")
        print(f"val_loss:          {float(val_loss):.6f}")
        print(f"training_seconds:  {training_seconds:.1f}")
        print(f"total_seconds:     {total_seconds:.1f}")
        print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
        print(f"runtime_device:    {runtime_device}")
        print(f"auto_cpu_fallback: {'yes' if auto_cpu_fallback_used else 'no'}")
        print(f"scenario_count:    {int(summary['scenario_count'])}")
        print(f"total_trade_count: {int(summary['total_trade_count'])}")
        print(f"train_samples:     {len(task.train_features)}")
        print(f"num_steps:         {int(step_count)}")
        print(f"model_parameters:  {model_parameters}")
        print(f"hidden_size:       {planner_cfg.hidden_size}")
        print(f"layers:            {planner_cfg.num_layers}")
        print(f"context_blocks:    {planner_cfg.context_blocks}")
        print(f"learning_rate:     {planner_cfg.learning_rate:.6g}")
        print(f"lr_source:         {learning_rate_source}")
        print(f"batch_size:        {planner_cfg.batch_size}")
        print(f"eval_batch_size:   {planner_cfg.eval_batch_size}")
        print(f"num_workers:       {planner_cfg.dataloader_workers}")
        print(f"deterministic:     {planner_cfg.deterministic}")
        print(f"exec_mod_tuning:   {execution_modifier_tuning_enabled}")
        print(f"buy_mod_bps:       {best_modifiers.buy_price_modifier_bps:.1f}")
        print(f"sell_mod_bps:      {best_modifiers.sell_price_modifier_bps:.1f}")
        print(f"amount_mod_pct:    {best_modifiers.amount_modifier_pct:.1f}")
        print(f"frequency:         {task.config.frequency}")
        print(f"hold_bars:         {int(task.config.hold_bars)}")
        print(f"checkpoint_dir:    {checkpoint_dir}")
        print(f"saved_checkpoint:  {saved_checkpoint_path}")
        print(f"best_checkpoint:   {best_checkpoint_path or ''}")
        return 0

    execution_result = run_with_auto_cpu_fallback(
        requested_device=args.device,
        device=device,
        context="Autoresearch stock trainer",
        operation=_run_training_for_device,
    )
    return int(execution_result.value)


if __name__ == "__main__":
    raise SystemExit(main())
