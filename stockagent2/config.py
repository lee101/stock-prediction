from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Tunable parameters controlling the risk-aware optimiser.

    All limits are expressed in fraction of net portfolio capital (1.0 = 100%).
    """

    net_exposure_target: float = 1.0
    gross_exposure_limit: float = 1.2
    long_cap: float = 0.12
    short_cap: float = 0.05
    transaction_cost_bps: float = 5.0
    turnover_penalty_bps: float = 2.5
    risk_aversion: float = 5.0
    min_weight: float = -0.25
    max_weight: float = 0.25
    sector_exposure_limits: Mapping[str, float] = field(default_factory=dict)

    def sector_limits(self) -> Dict[str, float]:
        """Return a mutable copy of the configured sector limits."""
        return dict(self.sector_exposure_limits)


@dataclass(frozen=True)
class PipelineConfig:
    """
    Aggregate configuration for `AllocationPipeline`.

    Attributes
    ----------
    tau:
        Scaling factor for the prior covariance within the Black–Litterman model.
    shrinkage:
        Linear shrinkage coefficient applied to the covariance estimated from
        Monte Carlo samples.
    """

    tau: float = 0.05
    shrinkage: float = 0.1
    min_confidence: float = 1e-3
    annualisation_periods: int = 252
    chronos_weight: float = 0.7
    timesfm_weight: float = 0.3
    risk_aversion: float = 3.0
    apply_confidence_to_mu: bool = True
    default_market_caps: Optional[Mapping[str, float]] = None
    market_prior_weight: float = 0.5
