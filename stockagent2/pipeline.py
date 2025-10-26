from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .black_litterman import BlackLittermanFuser, BlackLittermanResult
from .config import OptimizationConfig, PipelineConfig
from .forecasting import ForecastReturnSet, combine_forecast_sets
from .optimizer import CostAwareOptimizer, OptimizerResult
from .views_schema import LLMViews


@dataclass(frozen=True)
class AllocationResult:
    universe: Tuple[str, ...]
    weights: np.ndarray
    optimizer: OptimizerResult
    black_litterman: BlackLittermanResult
    mu_prior: np.ndarray
    sigma_prior: np.ndarray
    diagnostics: Dict[str, float]


class AllocationPipeline:
    """
    End-to-end pipeline that merges probabilistic forecasts, LLM views,
    and robust optimisation into production-ready weights.
    """

    def __init__(
        self,
        *,
        optimisation_config: OptimizationConfig,
        pipeline_config: PipelineConfig | None = None,
        fuser: Optional[BlackLittermanFuser] = None,
        optimizer: Optional[CostAwareOptimizer] = None,
    ) -> None:
        self.optimisation_config = optimisation_config
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.fuser = fuser or BlackLittermanFuser(
            tau=self.pipeline_config.tau,
            market_prior_weight=self.pipeline_config.market_prior_weight,
        )
        self.optimizer = optimizer or CostAwareOptimizer(optimisation_config)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        chronos: Optional[ForecastReturnSet] = None,
        timesfm: Optional[ForecastReturnSet] = None,
        additional_models: Sequence[Tuple[ForecastReturnSet, float]] = (),
        llm_views: Optional[LLMViews] = None,
        previous_weights: Optional[np.ndarray] = None,
        sector_map: Optional[Mapping[str, str]] = None,
        market_caps: Optional[Mapping[str, float]] = None,
    ) -> AllocationResult:
        forecast_sets, weights = self._collect_forecasts(
            chronos=chronos,
            timesfm=timesfm,
            additional_models=additional_models,
        )
        universe = forecast_sets[0].universe
        mu_prior, sigma_prior = combine_forecast_sets(
            forecast_sets,
            weights=weights,
            shrinkage=self.pipeline_config.shrinkage,
        )

        market_weights = self._resolve_market_weights(universe, market_caps)
        filtered_views = self._prepare_views(llm_views, universe)

        bl_result = self.fuser.fuse(
            mu_prior,
            sigma_prior,
            market_weights=market_weights,
            risk_aversion=self.pipeline_config.risk_aversion,
            views=filtered_views,
            universe=universe,
        )

        mu_for_optimizer = bl_result.mu_posterior
        sigma_for_optimizer = bl_result.sigma_posterior

        opt_result = self.optimizer.solve(
            mu_for_optimizer,
            sigma_for_optimizer,
            previous_weights=previous_weights,
            universe=universe,
            sector_map=self._normalise_sector_map(sector_map),
        )

        diagnostics = self._build_diagnostics(
            mu_prior,
            bl_result,
            opt_result,
            llm_views=filtered_views,
            universe=universe,
        )

        return AllocationResult(
            universe=universe,
            weights=opt_result.weights,
            optimizer=opt_result,
            black_litterman=bl_result,
            mu_prior=mu_prior,
            sigma_prior=sigma_prior,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _collect_forecasts(
        self,
        *,
        chronos: Optional[ForecastReturnSet],
        timesfm: Optional[ForecastReturnSet],
        additional_models: Sequence[Tuple[ForecastReturnSet, float]],
    ) -> Tuple[Sequence[ForecastReturnSet], np.ndarray]:
        models: list[ForecastReturnSet] = []
        weights: list[float] = []

        if chronos is not None:
            models.append(chronos)
            weights.append(self.pipeline_config.chronos_weight)
        if timesfm is not None:
            models.append(timesfm)
            weights.append(self.pipeline_config.timesfm_weight)

        for model, weight in additional_models:
            models.append(model)
            weights.append(float(weight))

        if not models:
            raise ValueError("At least one forecast distribution must be provided.")

        # If any weights are zero or negative, default to equal weighting.
        weight_array = np.asarray(weights, dtype=float)
        if np.any(weight_array <= 0):
            weight_array = np.ones_like(weight_array) / len(weight_array)
        return models, weight_array

    def _prepare_views(
        self,
        llm_views: Optional[LLMViews],
        universe: Sequence[str],
    ) -> Optional[LLMViews]:
        if llm_views is None:
            return None
        return llm_views.filter_for_universe(universe)

    def _normalise_sector_map(
        self,
        sector_map: Optional[Mapping[str, str]],
    ) -> Optional[Dict[str, str]]:
        if sector_map is None:
            return None
        return {symbol.upper(): sector for symbol, sector in sector_map.items()}

    def _resolve_market_weights(
        self,
        universe: Sequence[str],
        market_caps: Optional[Mapping[str, float]],
    ) -> Optional[np.ndarray]:
        source = market_caps or self.pipeline_config.default_market_caps
        if not source:
            return None
        values = np.array([float(source.get(symbol, 0.0)) for symbol in universe], dtype=float)
        total = values.sum()
        if total <= 0:
            return None
        return values / total

    def _build_diagnostics(
        self,
        mu_prior: np.ndarray,
        bl_result: BlackLittermanResult,
        opt_result: OptimizerResult,
        *,
        llm_views: Optional[LLMViews],
        universe: Sequence[str],
    ) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {
            "expected_return_prior": float(mu_prior.mean()),
            "expected_return_posterior": float(bl_result.mu_posterior.mean()),
            "risk_prior": float(np.trace(bl_result.sigma_prior)),
            "risk_posterior": float(np.trace(bl_result.sigma_posterior)),
            "turnover": float(opt_result.turnover),
        }
        if llm_views is not None:
            diagnostics["llm_view_count"] = float(len(llm_views.views))
            view_vec = llm_views.expected_return_vector(
                universe,
                apply_confidence=self.pipeline_config.apply_confidence_to_mu,
                min_confidence=self.pipeline_config.min_confidence,
            )
            diagnostics["llm_view_mean"] = float(view_vec.mean())
        diagnostics["bl_market_weight"] = bl_result.market_weight
        return diagnostics
