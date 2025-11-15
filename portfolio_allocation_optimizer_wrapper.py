"""Wrapper for neural portfolio allocation policies used during live trading.

This module wraps ``NeuralStrategyEvaluator`` with a thin helper that handles
device placement, reference-strategy selection, caching, and scale computation
so that production callers (trade_stock_e2e, marketsimulator, etc.) can request
pre-trained neural weights without duplicating boilerplate.  The wrapper keeps
the most recent scores, exposes the disabled reason (if loading fails), and
uses ``torch.inference_mode`` during inference for maximal throughput.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from strategytrainingneural.runtime import NeuralStrategyEvaluator, StrategyScore

logger = logging.getLogger(__name__)


@dataclass
class PortfolioOptimizerConfig:
    """Runtime configuration for the neural allocation optimizer."""

    run_dir: str
    metrics_csv: str
    reference_strategies: Sequence[str]
    min_scale: float = 0.35
    max_scale: float = 1.0
    asset_class: str = "all"
    device: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PortfolioAllocationOptimizer:
    """Caches and serves neural sizing weights for production allocators."""

    def __init__(self, config: PortfolioOptimizerConfig) -> None:
        self.config = config
        self._evaluator: Optional[NeuralStrategyEvaluator] = None
        self._last_scores: List[StrategyScore] = []
        self._last_scale: Optional[float] = None
        self._strategy_weights: Dict[str, float] = {}
        self._disabled_reason: Optional[str] = None

    @property
    def last_scale(self) -> Optional[float]:
        return self._last_scale

    @property
    def last_scores(self) -> List[StrategyScore]:
        return list(self._last_scores)

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def _ensure_evaluator(self) -> Optional[NeuralStrategyEvaluator]:
        if self._disabled_reason:
            return None
        if self._evaluator is not None:
            return self._evaluator
        if not self.config.reference_strategies:
            self._disabled_reason = "No reference strategies configured"
            return None
        try:
            self._evaluator = NeuralStrategyEvaluator(
                run_dir=self.config.run_dir,
                metrics_csv=self.config.metrics_csv,
                strategy_filter=list(self.config.reference_strategies),
                device=self.config.device,
                asset_class=self.config.asset_class,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
        except Exception as exc:  # pragma: no cover - guarded by integration tests
            self._disabled_reason = str(exc)
            logger.warning("Neural allocator disabled: %s", exc)
            return None
        logger.info(
            "Neural optimizer ready (run_dir=%s strategies=%s device=%s)",
            self.config.run_dir,
            ",".join(self.config.reference_strategies),
            self.config.device or "auto",
        )
        return self._evaluator

    def _scale_from_weight(self, weight: float) -> float:
        min_scale = max(0.0, float(self.config.min_scale))
        max_scale = max(min_scale, float(self.config.max_scale))
        span = max_scale - min_scale
        clamped = max(0.0, min(float(weight), 1.0))
        return min_scale + span * clamped

    def compute_scale(self, *, force: bool = False, cutoff_date: Optional[str] = None) -> Optional[float]:
        evaluator = self._ensure_evaluator()
        if evaluator is None:
            return None
        if self._last_scale is not None and not force:
            return self._last_scale
        scores = evaluator.score_strategies(self.config.reference_strategies, cutoff_date=cutoff_date)
        if not scores:
            logger.warning(
                "Neural optimizer received no scores for strategies=%s",
                ",".join(self.config.reference_strategies),
            )
            return None
        avg_weight = sum(score.weight for score in scores) / len(scores)
        scale = self._scale_from_weight(avg_weight)
        self._last_scores = scores
        self._last_scale = scale
        logger.info(
            "Neural sizing weights: %s | avg=%.3f → scale=%.3f",
            ", ".join(f"{score.name}={score.weight:.3f}@{score.date}" for score in scores),
            avg_weight,
            scale,
        )
        return scale

    def refresh_strategy_weights(self, *, force: bool = False, cutoff_date: Optional[str] = None) -> Dict[str, float]:
        evaluator = self._ensure_evaluator()
        if evaluator is None:
            return {}
        if self._strategy_weights and not force:
            return dict(self._strategy_weights)
        raw = evaluator.latest_strategy_weights(cutoff_date=cutoff_date)
        scaled = {name: self._scale_from_weight(weight) for name, weight in raw.items()}
        self._strategy_weights = scaled
        return dict(scaled)

    def get_strategy_weight(self, strategy_name: str) -> Optional[float]:
        return self._strategy_weights.get(strategy_name)


__all__ = [
    "PortfolioAllocationOptimizer",
    "PortfolioOptimizerConfig",
]
