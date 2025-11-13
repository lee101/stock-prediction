"""
Position sizing strategy implementations for testing different approaches.

Each strategy returns a position size as a fraction of total equity (or leverage-adjusted).
Strategies account for:
- 2x max leverage (stocks/ETFs only)
- 6.5% annual interest on leverage (daily calculation)
- Crypto: no leverage, no shorting
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import correlation utils
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MarketContext:
    """Market information needed for sizing decisions."""
    symbol: str
    predicted_return: float  # Expected return (e.g., 0.05 = 5%)
    predicted_volatility: float  # Expected volatility (e.g., 0.02 = 2%)
    current_price: float
    equity: float
    is_crypto: bool
    existing_position_value: float = 0.0  # Current position value in this symbol

    @property
    def max_leverage(self) -> float:
        """Max allowed leverage for this asset."""
        return 1.0 if self.is_crypto else 2.0

    @property
    def can_short(self) -> bool:
        """Whether shorting is allowed."""
        return not self.is_crypto


@dataclass
class SizingResult:
    """Result of a sizing calculation."""
    position_fraction: float  # Fraction of equity to allocate (-1 to +2 for stocks, 0 to 1 for crypto)
    position_value: float  # Dollar value of position
    quantity: float  # Number of shares/units
    leverage_used: float  # Total leverage (1.0 = no leverage)
    rationale: str  # Human-readable reasoning

    @property
    def is_long(self) -> bool:
        return self.position_fraction > 0

    @property
    def is_short(self) -> bool:
        return self.position_fraction < 0


class SizingStrategy(ABC):
    """Base class for position sizing strategies."""

    @abstractmethod
    def calculate_size(self, ctx: MarketContext) -> SizingResult:
        """Calculate position size given market context."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        pass


class FixedFractionStrategy(SizingStrategy):
    """Fixed fraction of equity per position."""

    def __init__(self, fraction: float = 0.5):
        self.fraction = fraction

    @property
    def name(self) -> str:
        return f"Fixed_{int(self.fraction*100)}pct"

    def calculate_size(self, ctx: MarketContext) -> SizingResult:
        # Respect direction from prediction
        sign = 1 if ctx.predicted_return > 0 else -1
        if not ctx.can_short and sign < 0:
            sign = 0

        frac = sign * self.fraction
        value = frac * ctx.equity
        qty = value / ctx.current_price if ctx.current_price > 0 else 0
        leverage = abs(frac)

        # Cap at max leverage
        if leverage > ctx.max_leverage:
            frac = sign * ctx.max_leverage
            value = frac * ctx.equity
            qty = value / ctx.current_price if ctx.current_price > 0 else 0
            leverage = ctx.max_leverage

        return SizingResult(
            position_fraction=frac,
            position_value=value,
            quantity=qty,
            leverage_used=leverage,
            rationale=f"Fixed {self.fraction:.1%} allocation"
        )


class KellyStrategy(SizingStrategy):
    """Kelly criterion with fractional scaling."""

    def __init__(self, fraction: float = 0.25, cap: float = 1.0):
        """
        Args:
            fraction: Fractional Kelly (0.25 = quarter Kelly)
            cap: Max position size as fraction of equity
        """
        self.fraction = fraction
        self.cap = cap

    @property
    def name(self) -> str:
        return f"Kelly_{int(self.fraction*100)}pct"

    def calculate_size(self, ctx: MarketContext) -> SizingResult:
        # Kelly: f = edge / variance = expected_return / variance
        if ctx.predicted_volatility <= 0:
            return SizingResult(0, 0, 0, 1.0, "Zero volatility, no position")

        variance = ctx.predicted_volatility ** 2
        kelly_full = ctx.predicted_return / variance
        kelly_frac = kelly_full * self.fraction

        # Apply cap
        kelly_frac = np.clip(kelly_frac, -self.cap, self.cap)

        # Respect shorting constraints
        if not ctx.can_short and kelly_frac < 0:
            kelly_frac = 0

        # Respect leverage constraints
        if abs(kelly_frac) > ctx.max_leverage:
            kelly_frac = np.sign(kelly_frac) * ctx.max_leverage

        value = kelly_frac * ctx.equity
        qty = value / ctx.current_price if ctx.current_price > 0 else 0
        leverage = abs(kelly_frac)

        return SizingResult(
            position_fraction=kelly_frac,
            position_value=value,
            quantity=qty,
            leverage_used=leverage,
            rationale=f"Kelly (full={kelly_full:.2f}, frac={self.fraction}, final={kelly_frac:.2f})"
        )


class VolatilityTargetStrategy(SizingStrategy):
    """Size positions to target a specific portfolio volatility."""

    def __init__(self, target_vol: float = 0.10, max_position: float = 1.0):
        """
        Args:
            target_vol: Target portfolio volatility (e.g., 0.10 = 10% annual)
            max_position: Max position size as fraction of equity
        """
        self.target_vol = target_vol
        self.max_position = max_position

    @property
    def name(self) -> str:
        return f"VolTarget_{int(self.target_vol*100)}pct"

    def calculate_size(self, ctx: MarketContext) -> SizingResult:
        if ctx.predicted_volatility <= 0:
            return SizingResult(0, 0, 0, 1.0, "Zero volatility, no position")

        # Size = target_vol / position_vol
        size = self.target_vol / ctx.predicted_volatility

        # Respect direction
        sign = 1 if ctx.predicted_return > 0 else -1
        if not ctx.can_short and sign < 0:
            sign = 0

        size = sign * min(abs(size), self.max_position)

        # Respect leverage constraints
        if abs(size) > ctx.max_leverage:
            size = np.sign(size) * ctx.max_leverage

        value = size * ctx.equity
        qty = value / ctx.current_price if ctx.current_price > 0 else 0
        leverage = abs(size)

        return SizingResult(
            position_fraction=size,
            position_value=value,
            quantity=qty,
            leverage_used=leverage,
            rationale=f"Vol target {self.target_vol:.1%} / position vol {ctx.predicted_volatility:.1%}"
        )


class RiskParityStrategy(SizingStrategy):
    """Equal risk contribution across positions."""

    def __init__(self, target_risk_per_position: float = 0.05, num_positions: int = 4):
        """
        Args:
            target_risk_per_position: Target risk contribution per position (e.g., 0.05 = 5%)
            num_positions: Expected number of positions for risk allocation
        """
        self.target_risk = target_risk_per_position
        self.num_positions = num_positions

    @property
    def name(self) -> str:
        return f"RiskParity_{int(self.target_risk*100)}pct"

    def calculate_size(self, ctx: MarketContext) -> SizingResult:
        if ctx.predicted_volatility <= 0:
            return SizingResult(0, 0, 0, 1.0, "Zero volatility, no position")

        # Size such that position_size * vol = target_risk
        size = self.target_risk / ctx.predicted_volatility

        # Respect direction
        sign = 1 if ctx.predicted_return > 0 else -1
        if not ctx.can_short and sign < 0:
            sign = 0

        size = sign * size

        # Respect leverage constraints
        if abs(size) > ctx.max_leverage:
            size = np.sign(size) * ctx.max_leverage

        value = size * ctx.equity
        qty = value / ctx.current_price if ctx.current_price > 0 else 0
        leverage = abs(size)

        return SizingResult(
            position_fraction=size,
            position_value=value,
            quantity=qty,
            leverage_used=leverage,
            rationale=f"Risk parity: {self.target_risk:.1%} risk / {ctx.predicted_volatility:.1%} vol"
        )


class OptimalFStrategy(SizingStrategy):
    """Maximize expected log growth accounting for leverage costs."""

    def __init__(self,
                 leverage_cost_annual: float = 0.065,
                 trading_days_per_year: int = 252,
                 max_position: float = 1.5):
        """
        Args:
            leverage_cost_annual: Annual interest rate on leverage (e.g., 0.065 = 6.5%)
            trading_days_per_year: Trading days for daily rate calculation
            max_position: Max position size cap
        """
        self.leverage_cost_annual = leverage_cost_annual
        self.trading_days_per_year = trading_days_per_year
        self.max_position = max_position
        self.daily_leverage_cost = leverage_cost_annual / trading_days_per_year

    @property
    def name(self) -> str:
        return f"OptimalF_cost{int(self.leverage_cost_annual*10000)}bps"

    def calculate_size(self, ctx: MarketContext) -> SizingResult:
        """
        Maximize E[log(1 + f*R - cost*leverage)] where cost applies to leveraged portion.

        For small returns: E[log(1+x)] ≈ E[x] - 0.5*Var[x]
        = f*mu - f*cost*I(f>1) - 0.5*f^2*sigma^2

        Taking derivative and solving: f ≈ (mu - cost*I(f>1)) / sigma^2
        """
        if ctx.predicted_volatility <= 0:
            return SizingResult(0, 0, 0, 1.0, "Zero volatility, no position")

        variance = ctx.predicted_volatility ** 2

        # Adjust expected return for leverage cost if we expect to use leverage
        adjusted_return = ctx.predicted_return

        # Try both scenarios: with and without leverage cost
        f_no_cost = ctx.predicted_return / variance
        f_with_cost = (ctx.predicted_return - self.daily_leverage_cost) / variance

        # Choose the appropriate f based on whether it exceeds 1.0
        if abs(f_no_cost) <= 1.0:
            f_opt = f_no_cost
            cost_applied = False
        else:
            f_opt = f_with_cost
            cost_applied = True

        # Cap at max position
        f_opt = np.clip(f_opt, -self.max_position, self.max_position)

        # Respect shorting constraints
        if not ctx.can_short and f_opt < 0:
            f_opt = 0

        # Respect leverage constraints
        if abs(f_opt) > ctx.max_leverage:
            f_opt = np.sign(f_opt) * ctx.max_leverage

        value = f_opt * ctx.equity
        qty = value / ctx.current_price if ctx.current_price > 0 else 0
        leverage = abs(f_opt)

        cost_note = f" (cost adjusted)" if cost_applied else " (no cost)"

        return SizingResult(
            position_fraction=f_opt,
            position_value=value,
            quantity=qty,
            leverage_used=leverage,
            rationale=f"Optimal f={f_opt:.2f}{cost_note}, mu={ctx.predicted_return:.3f}, vol={ctx.predicted_volatility:.3f}"
        )


class CorrelationAwareStrategy(SizingStrategy):
    """
    Size positions accounting for correlation and covariance matrix.

    Implements robust Kelly with uncertainty penalty:
    w = (Σ + τ*V_μ)^{-1} * μ

    Where:
    - Σ is the covariance matrix
    - V_μ is the forecast uncertainty
    - τ is the uncertainty penalty parameter
    """

    def __init__(
        self,
        corr_data: Optional[Dict] = None,
        uncertainty_penalty: float = 1.0,
        fractional_kelly: float = 0.5,
        max_position: float = 1.0,
    ):
        """
        Args:
            corr_data: Loaded correlation matrix data (from load_correlation_matrix())
            uncertainty_penalty: τ parameter (higher = more conservative)
            fractional_kelly: Fractional Kelly scaling
            max_position: Max position size per symbol
        """
        self.corr_data = corr_data
        self.uncertainty_penalty = uncertainty_penalty
        self.fractional_kelly = fractional_kelly
        self.max_position = max_position

        # Load correlation data if not provided
        if self.corr_data is None:
            try:
                from trainingdata.load_correlation_utils import load_correlation_matrix
                self.corr_data = load_correlation_matrix()
            except Exception:
                self.corr_data = None

        # Extract correlation matrix and volatility metrics
        if self.corr_data:
            self.symbols = self.corr_data.get('symbols', [])
            corr_matrix = np.array(self.corr_data.get('correlation_matrix', []))

            # Build covariance matrix: Σ = D * ρ * D where D = diag(volatilities)
            vols = []
            for sym in self.symbols:
                try:
                    vol_metrics = self.corr_data['volatility_metrics'].get(sym, {})
                    vol = vol_metrics.get('annualized_volatility', 0.2)  # Default 20%
                    vols.append(vol)
                except Exception:
                    vols.append(0.2)

            D = np.diag(vols)
            self.covariance_matrix = D @ corr_matrix @ D

            # Replace NaN correlations with 0 (assume uncorrelated)
            self.covariance_matrix = np.nan_to_num(self.covariance_matrix, nan=0.0)

            # Ensure positive semi-definite by setting diagonal to variances
            for i in range(len(vols)):
                self.covariance_matrix[i, i] = vols[i] ** 2

            # Forecast uncertainty: assume proportional to volatility
            # V_μ = diag(σ^2 / sqrt(n)) where n is effective sample size
            n_effective = 60  # ~3 months of data
            self.forecast_uncertainty = np.diag([v**2 / np.sqrt(n_effective) for v in vols])
        else:
            self.symbols = []
            self.covariance_matrix = None
            self.forecast_uncertainty = None

    @property
    def name(self) -> str:
        return f"CorrAware_tau{self.uncertainty_penalty:.1f}_frac{int(self.fractional_kelly*100)}"

    def calculate_size(
        self,
        ctx: MarketContext,
        portfolio_context: Optional[Dict[str, MarketContext]] = None
    ) -> SizingResult:
        """
        Calculate size accounting for portfolio correlations.

        Args:
            ctx: Market context for this symbol
            portfolio_context: Dict of symbol -> MarketContext for all positions
        """
        # Fallback to simple Kelly if no correlation data
        if self.covariance_matrix is None or ctx.symbol not in self.symbols:
            return self._fallback_kelly(ctx)

        # If no portfolio context, use single-asset sizing
        if portfolio_context is None:
            portfolio_context = {ctx.symbol: ctx}

        # Build expected return vector and filter to available symbols
        active_symbols = [s for s in portfolio_context.keys() if s in self.symbols]
        if not active_symbols or ctx.symbol not in active_symbols:
            return self._fallback_kelly(ctx)

        # Get indices in correlation matrix
        indices = [self.symbols.index(s) for s in active_symbols]

        # Extract relevant submatrices
        Sigma = self.covariance_matrix[np.ix_(indices, indices)]
        V_mu = self.forecast_uncertainty[np.ix_(indices, indices)]

        # Build expected return vector
        mu = np.array([portfolio_context[s].predicted_return for s in active_symbols])

        # Check for NaN in inputs
        if not np.all(np.isfinite(Sigma)):
            return self._fallback_kelly(ctx)
        if not np.all(np.isfinite(V_mu)):
            return self._fallback_kelly(ctx)
        if not np.all(np.isfinite(mu)):
            return self._fallback_kelly(ctx)

        # Robust Kelly: w = (Σ + τ*V_μ)^{-1} * μ
        try:
            regularized_cov = Sigma + self.uncertainty_penalty * V_mu
            # Add small ridge for numerical stability
            regularized_cov += np.eye(len(regularized_cov)) * 1e-6
            weights = np.linalg.solve(regularized_cov, mu)

            # Check for NaN/inf
            if not np.all(np.isfinite(weights)):
                return self._fallback_kelly(ctx)

        except (np.linalg.LinAlgError, ValueError):
            return self._fallback_kelly(ctx)

        # Apply fractional Kelly
        weights = weights * self.fractional_kelly

        # Extract weight for current symbol
        symbol_idx = active_symbols.index(ctx.symbol)
        raw_weight = weights[symbol_idx]

        # Apply caps and constraints
        weight = np.clip(raw_weight, -self.max_position, self.max_position)

        if not ctx.can_short and weight < 0:
            weight = 0

        if abs(weight) > ctx.max_leverage:
            weight = np.sign(weight) * ctx.max_leverage

        value = weight * ctx.equity
        qty = value / ctx.current_price if ctx.current_price > 0 else 0
        leverage = abs(weight)

        return SizingResult(
            position_fraction=weight,
            position_value=value,
            quantity=qty,
            leverage_used=leverage,
            rationale=f"Corr-aware: raw={raw_weight:.2f}, τ={self.uncertainty_penalty}, final={weight:.2f}"
        )

    def _fallback_kelly(self, ctx: MarketContext) -> SizingResult:
        """Fallback to simple fractional Kelly if correlation data unavailable."""
        if ctx.predicted_volatility <= 0:
            return SizingResult(0, 0, 0, 1.0, "Zero volatility (fallback)")

        variance = ctx.predicted_volatility ** 2
        kelly = (ctx.predicted_return / variance) * self.fractional_kelly
        kelly = np.clip(kelly, -self.max_position, self.max_position)

        if not ctx.can_short and kelly < 0:
            kelly = 0
        if abs(kelly) > ctx.max_leverage:
            kelly = np.sign(kelly) * ctx.max_leverage

        value = kelly * ctx.equity
        qty = value / ctx.current_price if ctx.current_price > 0 else 0

        return SizingResult(
            position_fraction=kelly,
            position_value=value,
            quantity=qty,
            leverage_used=abs(kelly),
            rationale=f"Fallback Kelly: {kelly:.2f}"
        )


class VolatilityAdjustedStrategy(SizingStrategy):
    """
    Adjust position sizes by realized volatility from correlation matrix.
    Simpler alternative to full correlation-aware sizing.
    """

    def __init__(
        self,
        corr_data: Optional[Dict] = None,
        target_vol_contribution: float = 0.10,
        max_position: float = 1.0,
    ):
        """
        Args:
            corr_data: Loaded correlation matrix data
            target_vol_contribution: Target volatility contribution per position
            max_position: Max position size
        """
        self.target_vol = target_vol_contribution
        self.max_position = max_position
        self.corr_data = corr_data

        # Load if not provided
        if self.corr_data is None:
            try:
                from trainingdata.load_correlation_utils import load_correlation_matrix
                self.corr_data = load_correlation_matrix()
            except Exception:
                self.corr_data = None

    @property
    def name(self) -> str:
        return f"VolAdj_target{int(self.target_vol*100)}pct"

    def calculate_size(self, ctx: MarketContext) -> SizingResult:
        # Try to get realized volatility from correlation data
        realized_vol = ctx.predicted_volatility

        if self.corr_data and ctx.symbol in self.corr_data.get('symbols', []):
            try:
                vol_metrics = self.corr_data['volatility_metrics'][ctx.symbol]
                # Use 30-day rolling vol if available, else annualized
                realized_vol = vol_metrics.get('rolling_vol_30d') or vol_metrics.get('annualized_volatility')
                if realized_vol is None:
                    realized_vol = ctx.predicted_volatility
            except Exception:
                pass

        if realized_vol <= 0:
            return SizingResult(0, 0, 0, 1.0, "Zero volatility")

        # Size = target_vol / realized_vol
        size = self.target_vol / realized_vol

        # Respect direction
        sign = 1 if ctx.predicted_return > 0 else -1
        if not ctx.can_short and sign < 0:
            sign = 0

        size = sign * min(abs(size), self.max_position)

        if abs(size) > ctx.max_leverage:
            size = np.sign(size) * ctx.max_leverage

        value = size * ctx.equity
        qty = value / ctx.current_price if ctx.current_price > 0 else 0

        return SizingResult(
            position_fraction=size,
            position_value=value,
            quantity=qty,
            leverage_used=abs(size),
            rationale=f"Vol-adjusted: target={self.target_vol:.1%} / realized={realized_vol:.1%}"
        )


# Preset strategy configurations for testing
SIZING_STRATEGIES = {
    "fixed_25": FixedFractionStrategy(0.25),
    "fixed_50": FixedFractionStrategy(0.50),
    "fixed_75": FixedFractionStrategy(0.75),
    "fixed_100": FixedFractionStrategy(1.0),
    "kelly_10": KellyStrategy(fraction=0.10, cap=1.5),
    "kelly_25": KellyStrategy(fraction=0.25, cap=1.5),
    "kelly_50": KellyStrategy(fraction=0.50, cap=1.5),
    "voltarget_10": VolatilityTargetStrategy(target_vol=0.10, max_position=1.5),
    "voltarget_15": VolatilityTargetStrategy(target_vol=0.15, max_position=1.5),
    "riskparity_5": RiskParityStrategy(target_risk_per_position=0.05),
    "riskparity_10": RiskParityStrategy(target_risk_per_position=0.10),
    "optimal_f": OptimalFStrategy(),
    # New correlation-aware strategies
    "corr_aware_conservative": CorrelationAwareStrategy(uncertainty_penalty=2.0, fractional_kelly=0.25),
    "corr_aware_moderate": CorrelationAwareStrategy(uncertainty_penalty=1.0, fractional_kelly=0.5),
    "corr_aware_aggressive": CorrelationAwareStrategy(uncertainty_penalty=0.5, fractional_kelly=0.75),
    "vol_adjusted_10": VolatilityAdjustedStrategy(target_vol_contribution=0.10),
    "vol_adjusted_15": VolatilityAdjustedStrategy(target_vol_contribution=0.15),
}
