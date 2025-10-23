"""
Configuration dataclasses for the GymRL experiment.

These classes centralise knobs shared across the feature pipeline,
offline dataset builder, and the Gymnasium environment so that the
experiment can be tuned from scripts without rewriting logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from src.alpaca_utils import (
    ANNUAL_MARGIN_RATE,
    BASE_GROSS_EXPOSURE,
    INTRADAY_GROSS_EXPOSURE,
    MAX_GROSS_EXPOSURE,
    TRADING_DAYS_PER_YEAR,
)

@dataclass
class PortfolioEnvConfig:
    """
    Parameters controlling the trading environment dynamics.

    Attributes:
        costs_bps: Baseline proportional trading cost expressed in basis points.
        per_asset_costs_bps: Optional per-asset overrides; length must match asset universe.
        turnover_penalty: Scalar multiplied with portfolio turnover in the reward.
        drawdown_penalty: Scalar applied to running max drawdown to discourage deep losses.
        cvar_penalty: Scalar applied to predicted CVaR (if supplied by the feature cube).
        uncertainty_penalty: Scalar applied to forecast uncertainty term (e.g., q90 - q10).
        weight_cap: Maximum per-asset allocation for long-only configurations. If None,
            no per-asset cap is enforced beyond feasibility constraints.
        allow_short: Enable long/short allocations with symmetric leverage bounds.
        leverage_cap: Gross leverage cap when allow_short=True. For long-only this is ignored.
        include_cash: If True, a synthetic cash asset is appended with deterministic return.
        cash_return: Deterministic return for the synthetic cash asset per step.
        forecast_cvar_alpha: Alpha level assumed when interpreting CVaR forecast inputs.
        leverage_head: When True, appends a leverage control head to the action vector.
        base_gross_exposure: Gross exposure that is free of financing costs (typically 1×).
        max_gross_leverage: End-of-day gross leverage cap.
        intraday_leverage_cap: Intraday gross leverage cap; enforced immediately after order execution.
        closing_leverage_cap: Optional explicit end-of-day cap. Defaults to ``max_gross_leverage``.
        daily_leverage_rate: Optional explicit daily financing rate applied to excess leverage while intraday.
        leverage_interest_rate: Annualised interest rate applied to positions carried overnight above ``base_gross_exposure``.
        trading_days_per_year: Trading days used when converting annualised rates to daily equivalents.
        leverage_penalty_annual_rate: Annualised leverage cost used when ``daily_leverage_rate`` is None.
        leverage_penalty_trading_days: Trading days per year used for rate conversion when ``daily_leverage_rate`` is None.
        enforce_end_of_day_cap: If True, positions are automatically scaled back to ``max_gross_leverage`` after each step.
        loss_shutdown_enabled: Enable cooldown gating when recent trades in an asset/direction
            were unprofitable.
        loss_shutdown_cooldown: Number of steps an asset/direction remains in cooldown after a loss.
        loss_shutdown_probe_weight: Maximum weight magnitude allowed while an asset/direction is
            in cooldown. Acts as the "probe trade" size.
        loss_shutdown_penalty: Additional reward penalty applied to allocations that remain in
            cooldown, proportional to absolute weight.
        loss_shutdown_min_position: Minimum absolute weight treated as an active position for the
            shutdown logic to avoid noise from tiny allocations.
        loss_shutdown_return_tolerance: Absolute return threshold below which outcomes are treated
            as neutral (neither profit nor loss) for cooldown updates.
        intraday_leverage_cap: Optional gross exposure cap applied immediately after the action
            projection. If None, defaults to leverage_cap (long/short) or 1.0 (long-only).
        closing_leverage_cap: Optional gross exposure cap enforced at the end of every step before
            carrying positions overnight. If None, defaults to intraday_leverage_cap.
        leverage_interest_rate: Annualised interest rate applied to borrowed exposure above 1x when
            held overnight (after enforcing closing_leverage_cap).
        trading_days_per_year: Number of trading days used to annualise leverage interest.
        regime_filters_enabled: If True, activate the regime guard heuristics that downshift leverage
            and increase turnover penalties in unfavourable windows.
        regime_drawdown_threshold: Drawdown level triggering leverage reduction (None disables).
        regime_leverage_scale: Multiplicative scale applied to allocations when the drawdown guard fires.
        regime_negative_return_window: Trailing window (in steps) used to evaluate cumulative returns.
        regime_negative_return_threshold: Cumulative return threshold; values at or below trigger guards.
        regime_negative_return_turnover_penalty: Turnover penalty applied when the negative-return guard fires.
        regime_turnover_threshold: Turnover level triggering stricter loss-shutdown probe sizing.
        regime_turnover_probe_weight: Loss-shutdown probe weight enforced when the turnover guard fires.
    """

    costs_bps: float = 3.0
    per_asset_costs_bps: Optional[Sequence[float]] = None
    turnover_penalty: float = 5e-4
    drawdown_penalty: float = 0.0
    cvar_penalty: float = 0.0
    uncertainty_penalty: float = 0.0
    weight_cap: Optional[float] = 0.3
    allow_short: bool = False
    loss_shutdown_enabled: bool = False
    loss_shutdown_cooldown: int = 3
    loss_shutdown_probe_weight: float = 0.05
    loss_shutdown_penalty: float = 0.0
    loss_shutdown_min_position: float = 1e-4
    loss_shutdown_return_tolerance: float = 1e-5
    leverage_cap: float = 1.0
    intraday_leverage_cap: Optional[float] = None
    closing_leverage_cap: Optional[float] = None
    leverage_interest_rate: float = ANNUAL_MARGIN_RATE
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    include_cash: bool = True
    cash_return: float = 0.0
    forecast_cvar_alpha: float = 0.05
    leverage_head: bool = True
    base_gross_exposure: float = BASE_GROSS_EXPOSURE
    max_gross_leverage: float = MAX_GROSS_EXPOSURE
    daily_leverage_rate: Optional[float] = None
    leverage_penalty_annual_rate: float = ANNUAL_MARGIN_RATE
    leverage_penalty_trading_days: int = TRADING_DAYS_PER_YEAR
    enforce_end_of_day_cap: bool = True
    regime_filters_enabled: bool = False
    regime_drawdown_threshold: Optional[float] = None
    regime_leverage_scale: float = 0.5
    regime_negative_return_window: int = 42
    regime_negative_return_threshold: float = 0.0
    regime_negative_return_turnover_penalty: Optional[float] = None
    regime_turnover_threshold: Optional[float] = None
    regime_turnover_probe_weight: Optional[float] = None


@dataclass
class FeatureBuilderConfig:
    """
    Parameters driving feature cube construction from historical data.

    Attributes:
        context_window: Number of trailing observations to provide to the forecaster.
        prediction_length: Forecast horizon in steps (1 = next period).
        realized_horizon: Horizon over which realized returns are computed.
        resample_rule: Optional pandas offset alias to resample source data.
        forecast_backend: Identifier for forecasting backend ("auto", "toto", "kronos", "chronos", or "bootstrap").
        num_samples: Number of Monte Carlo samples to draw from the forecasting backend.
        min_history: Minimum observations required before emitting the first feature row.
        lookahead_buffer: Extra steps dropped from the tail to avoid lookahead bias.
        realized_feature_windows: Rolling window lengths (in steps) for realized stats.
        store_intermediate: Persist intermediate artefacts (e.g., per-symbol feature frames).
        intermediate_dir: Directory where intermediate files are written if enabled.
        enforce_common_index: Require all symbols to share timestamps (inner join) if True.
        fill_method: Method name passed to pandas.DataFrame.fillna for optional imputation.
        bootstrap_block_size: Block size for bootstrap fallback if no forecaster is available.
    """

    context_window: int = 192
    prediction_length: int = 1
    realized_horizon: int = 1
    resample_rule: Optional[str] = None
    forecast_backend: str = "auto"
    num_samples: int = 512
    min_history: int = 256
    lookahead_buffer: int = 0
    realized_feature_windows: Sequence[int] = field(default_factory=lambda: (5, 20, 60))
    store_intermediate: bool = False
    intermediate_dir: Optional[str] = None
    enforce_common_index: bool = True
    fill_method: Optional[str] = None
    bootstrap_block_size: int = 8


@dataclass
class OfflineDatasetConfig:
    """
    Options for serialising offline RL datasets derived from the feature cube.

    Attributes:
        output_path: Target path (npz/parquet) for the dataset.
        create_behavior_policy: If True, derive behaviour policy weights for IQL/CQL.
        behaviour_policy_name: Label describing how the behaviour policy was generated.
        normalize_rewards: Apply mean/std normalisation to rewards before saving.
        shard_limit: Optional limit on number of shards saved; useful for debugging.
        compress: Whether to compress numpy archives (using np.savez_compressed).
        metadata_only: Emit metadata JSON without heavy arrays for quick inspection.
    """

    output_path: Optional[str] = None
    create_behavior_policy: bool = True
    behaviour_policy_name: str = "top2_equal_weight"
    normalize_rewards: bool = False
    shard_limit: Optional[int] = None
    compress: bool = True
    metadata_only: bool = False
