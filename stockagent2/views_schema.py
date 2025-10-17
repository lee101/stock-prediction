from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class TickerView(BaseModel):
    """
    Canonical representation of an LLM generated view that can be fused with
    quantitative forecasts.

    The schema deliberately keeps confidence and half-life separate so that the
    downstream pipeline can reason about structural conviction (confidence) and
    temporal decay (half-life) independently.
    """

    ticker: str = Field(..., description="Ticker symbol in canonical uppercase form.")
    horizon_days: int = Field(
        default=5,
        ge=1,
        le=63,
        description="Forecast horizon, constrained to a practical range (≈ one quarter).",
    )
    mu_bps: float = Field(
        ...,
        description="Expected excess return over cash expressed in basis points for the full horizon.",
    )
    stdev_bps: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional standard deviation estimate (basis points over the full horizon).",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the view: 0 disables the view, 1 is full conviction.",
    )
    half_life_days: int = Field(
        default=10,
        ge=1,
        le=126,
        description="Half-life (in trading days) used to decay the view back to the market prior.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Free-form rationale retained for audit logs, ignored by optimisers.",
    )

    @field_validator("ticker")
    @classmethod
    def _ticker_uppercase(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Ticker symbol cannot be empty.")
        return cleaned

    @field_validator("mu_bps")
    @classmethod
    def _mu_not_nan(cls, value: float) -> float:
        if math.isnan(value):
            raise ValueError("mu_bps must be a finite number.")
        return float(value)

    @field_validator("stdev_bps")
    @classmethod
    def _stdev_not_nan(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if math.isnan(value):
            raise ValueError("stdev_bps must be a finite number when provided.")
        return float(value)


class LLMViews(BaseModel):
    """
    Container for a batch of structured LLM views.

    The model enforces that the view universe is coherent with the provided
    `universe` attribute and that the as-of timestamp adheres to ISO formatting.
    """

    asof: str = Field(..., description="ISO 8601 date (YYYY-MM-DD) for the view snapshot.")
    universe: List[str] = Field(..., description="Universe in which the agent operates.")
    views: List[TickerView] = Field(default_factory=list)

    @field_validator("asof")
    @classmethod
    def _validate_asof(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.strip()).date()
        except Exception as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Invalid asof date: {value!r}") from exc
        return value.strip()

    @field_validator("universe", mode="before")
    @classmethod
    def _coerce_universe(cls, value: Iterable[str]) -> List[str]:
        cleaned = [str(item).strip().upper() for item in value]
        if any(not symbol for symbol in cleaned):
            raise ValueError("Universe symbols must be non-empty strings.")
        return cleaned

    @model_validator(mode="after")
    def _ensure_view_universe(self) -> "LLMViews":
        universe = set(self.universe)
        for view in self.views:
            if view.ticker not in universe:
                raise ValueError(f"View ticker {view.ticker!r} not present in universe.")
        return self

    # ------------------------------------------------------------------ #
    # Helper utilities for downstream allocators
    # ------------------------------------------------------------------ #
    def _decay_weight(self, view: TickerView) -> float:
        if view.half_life_days <= 0:
            return 1.0
        # Exponential decay to dampen longer-dated views
        decay = math.exp(-math.log(2) * max(view.horizon_days - 1, 0) / view.half_life_days)
        return float(decay)

    def expected_return_vector(
        self,
        universe: Sequence[str],
        *,
        apply_confidence: bool = True,
        min_confidence: float = 1e-3,
    ) -> np.ndarray:
        """
        Convert the LLM views into a vector of expected daily excess returns ordered
        by `universe`.

        Parameters
        ----------
        universe:
            Sequence of tickers defining the ordering of the result vector.
        apply_confidence:
            If True (default) multiplies each view's contribution by its confidence.
        min_confidence:
            Lower bound to avoid division by zero when normalising weights.
        """
        size = len(universe)
        idx_map = {symbol.upper(): i for i, symbol in enumerate(universe)}
        totals = np.zeros(size, dtype=float)
        weights = np.zeros(size, dtype=float)

        for view in self.views:
            idx = idx_map.get(view.ticker)
            if idx is None:
                continue  # silently ignore views outside the requested ordering
            horizon = max(float(view.horizon_days), 1.0)
            daily_return = (view.mu_bps / 1e4) / horizon
            confidence = max(min(view.confidence, 1.0), 0.0) if apply_confidence else 1.0
            effective_weight = max(confidence * self._decay_weight(view), min_confidence)
            totals[idx] += daily_return * effective_weight
            weights[idx] += effective_weight

        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.divide(
                totals,
                weights,
                out=np.zeros_like(totals),
                where=weights > 0.0,
            )
        return result

    def black_litterman_inputs(
        self,
        universe: Sequence[str],
        *,
        min_confidence: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce the (P, Q, omega, confidences) tuple used by the Black–Litterman
        fusion step.

        Returns
        -------
        P : np.ndarray
            Pick matrix of shape (k, n) where each row selects a ticker.
        Q : np.ndarray
            Vector of view returns in daily decimal form.
        omega : np.ndarray
            Diagonal covariance matrix that scales with the inverse of confidence.
        confidences : np.ndarray
            Handy copy of the effective confidences for downstream logging.
        """
        n = len(universe)
        idx_map = {symbol.upper(): i for i, symbol in enumerate(universe)}
        rows: List[np.ndarray] = []
        q_vals: List[float] = []
        omega_vals: List[float] = []
        confidences: List[float] = []

        for view in self.views:
            idx = idx_map.get(view.ticker)
            if idx is None:
                continue
            horizon = max(float(view.horizon_days), 1.0)
            mean = (view.mu_bps / 1e4) / horizon
            decay_weight = self._decay_weight(view)
            base_confidence = max(min(view.confidence, 1.0), 0.0)
            effective_confidence = max(base_confidence * decay_weight, min_confidence)
            stdev = (
                (view.stdev_bps or max(abs(view.mu_bps), 1.0)) / 1e4
            ) / math.sqrt(horizon)
            variance = float(stdev**2) / max(effective_confidence, min_confidence)

            row = np.zeros(n, dtype=float)
            row[idx] = 1.0

            rows.append(row)
            q_vals.append(mean)
            omega_vals.append(variance)
            confidences.append(effective_confidence)

        if not rows:
            return (
                np.zeros((0, n), dtype=float),
                np.zeros(0, dtype=float),
                np.zeros((0, 0), dtype=float),
                np.zeros(0, dtype=float),
            )

        P = np.vstack(rows)
        Q = np.asarray(q_vals, dtype=float)
        omega = np.diag(np.asarray(omega_vals, dtype=float))
        conf = np.asarray(confidences, dtype=float)
        return P, Q, omega, conf

    def tickers(self) -> Tuple[str, ...]:
        """Return the tickers referenced by at least one view in declaration order."""
        return tuple(view.ticker for view in self.views)

    def filter_for_universe(self, universe: Iterable[str]) -> "LLMViews":
        """
        Return a copy that contains only the views present in `universe`.

        The original object is not mutated.
        """
        ordered = [symbol.strip().upper() for symbol in universe]
        allowed = set(ordered)
        filtered = [view for view in self.views if view.ticker in allowed]
        new_universe = [symbol for symbol in ordered if symbol in allowed]
        return LLMViews(asof=self.asof, universe=new_universe, views=filtered)
