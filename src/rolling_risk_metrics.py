from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_PERIODS_PER_YEAR = 252.0
SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0


@dataclass(frozen=True)
class SymbolVolatilityStats:
    """Container for realised and annualised volatility per symbol."""

    symbol: str
    realized_volatility: float
    annualized_volatility: float
    sample_count: int
    last_timestamp: datetime


@dataclass(frozen=True)
class RiskMetricsSnapshot:
    """Snapshot of the current rolling volatility/correlation state."""

    observed_at: datetime
    window_duration: Optional[timedelta]
    symbol_stats: Dict[str, SymbolVolatilityStats]
    correlation_matrix: pd.DataFrame

    def to_metadata(self) -> Dict[str, object]:
        """Return a JSON-serialisable view of the snapshot."""

        return {
            "observed_at": self.observed_at.isoformat(),
            "window_seconds": None if self.window_duration is None else self.window_duration.total_seconds(),
            "symbols": {
                symbol: {
                    "realized_volatility": stats.realized_volatility,
                    "annualized_volatility": stats.annualized_volatility,
                    "sample_count": stats.sample_count,
                    "last_timestamp": stats.last_timestamp.isoformat(),
                }
                for symbol, stats in self.symbol_stats.items()
            },
            "correlation_columns": list(self.correlation_matrix.columns),
            "correlation_index": list(self.correlation_matrix.index),
            "correlation_values": self.correlation_matrix.values.tolist(),
        }


class RollingRiskMetrics:
    """Maintain rolling per-symbol volatility and cross-symbol correlation."""

    def __init__(
        self,
        *,
        window_duration: Optional[timedelta] = timedelta(hours=6),
        max_samples: Optional[int] = 720,
        min_history: int = 5,
        annualization_override: Optional[float] = None,
    ) -> None:
        if window_duration is not None and window_duration <= timedelta(0):
            raise ValueError("window_duration must be positive or None")
        if max_samples is not None and max_samples <= 1:
            raise ValueError("max_samples must be greater than 1 when provided")
        if min_history < 2:
            raise ValueError("min_history must be >= 2")

        self.window_duration = window_duration
        self.max_samples = max_samples
        self.min_history = int(min_history)
        self._annualization_override = float(annualization_override) if annualization_override else None
        self._returns: Dict[str, pd.Series] = {}
        self._last_price: Dict[str, float] = {}
        self._latest_timestamp: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_from_price_frame(
        self,
        symbol: str,
        price_frame: pd.DataFrame | Sequence[Mapping[str, object]],
        *,
        price_column: str = "close",
        timestamp_column: str = "timestamp",
    ) -> int:
        """Ingest new price data for a symbol.

        Args:
            symbol: Trading symbol identifier.
            price_frame: Price observations containing timestamp + close columns.
            price_column: Column containing price data (default: ``close``).
            timestamp_column: Timestamp column name (default: ``timestamp``).

        Returns:
            Number of new return observations added to the rolling window.
        """

        normalized = self._normalize_price_frame(price_frame, price_column, timestamp_column)
        if normalized.empty:
            return 0

        returns = normalized.pct_change()
        last_price = self._last_price.get(symbol)
        if last_price is not None:
            first_price = float(normalized.iloc[0])
            if math.isfinite(first_price) and last_price > 0:
                first_return = (first_price / last_price) - 1.0
                returns.iloc[0] = first_return
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        self._last_price[symbol] = float(normalized.iloc[-1])
        if returns.empty:
            return 0

        series = pd.Series(returns.values, index=returns.index, dtype="float64")
        series = series[~series.index.duplicated(keep="last")]

        existing = self._returns.get(symbol)
        if existing is not None:
            series = pd.concat([existing, series]).sort_index()
            series = series[~series.index.duplicated(keep="last")]
        series = self._trim_series(series)
        self._returns[symbol] = series

        if not series.empty:
            latest = series.index[-1].to_pydatetime()
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=timezone.utc)
            else:
                latest = latest.astimezone(timezone.utc)
            if self._latest_timestamp is None or latest > self._latest_timestamp:
                self._latest_timestamp = latest

        return len(returns)

    def get_symbol_returns(self, symbol: str) -> pd.Series:
        """Return a copy of the stored return series for ``symbol``."""

        series = self._returns.get(symbol)
        if series is None:
            return pd.Series(dtype="float64")
        return series.copy()

    def get_symbol_volatility_stats(self, symbol: str) -> Optional[SymbolVolatilityStats]:
        series = self._returns.get(symbol)
        if series is None or len(series) < self.min_history:
            return None

        realized = float(series.std(ddof=1))
        if not math.isfinite(realized):
            return None
        periods = self._annualization_override or self._infer_periods_per_year(series.index)
        annualized = realized * math.sqrt(max(periods, 1.0))
        last_timestamp = series.index[-1].to_pydatetime()
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
        else:
            last_timestamp = last_timestamp.astimezone(timezone.utc)
        return SymbolVolatilityStats(
            symbol=symbol,
            realized_volatility=realized,
            annualized_volatility=annualized,
            sample_count=len(series),
            last_timestamp=last_timestamp,
        )

    def get_all_volatility_stats(self) -> Dict[str, SymbolVolatilityStats]:
        stats: Dict[str, SymbolVolatilityStats] = {}
        for symbol in sorted(self._returns):
            stat = self.get_symbol_volatility_stats(symbol)
            if stat is not None:
                stats[symbol] = stat
        return stats

    def get_correlation_matrix(self, *, min_overlap: int = 3) -> pd.DataFrame:
        frame = self._build_return_frame()
        if frame.empty:
            return pd.DataFrame()

        if min_overlap > 1:
            frame = frame.dropna(axis=1, thresh=min_overlap)
        frame = frame.dropna(how="all")
        if frame.shape[1] == 0:
            return pd.DataFrame()
        if frame.shape[1] == 1:
            column = frame.columns[0]
            return pd.DataFrame([[1.0]], index=[column], columns=[column], dtype="float64")
        correlation = frame.corr()
        return correlation.fillna(0.0)

    def build_snapshot(self, *, min_overlap: int = 3) -> RiskMetricsSnapshot:
        stats = self.get_all_volatility_stats()
        matrix = self.get_correlation_matrix(min_overlap=min_overlap)
        observed_at = self._latest_timestamp or datetime.now(timezone.utc)
        return RiskMetricsSnapshot(
            observed_at=observed_at,
            window_duration=self.window_duration,
            symbol_stats=stats,
            correlation_matrix=matrix,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_price_frame(
        self,
        price_frame: pd.DataFrame | Sequence[Mapping[str, object]],
        price_column: str,
        timestamp_column: str,
    ) -> pd.Series:
        if isinstance(price_frame, pd.DataFrame):
            frame = price_frame.copy()
        else:
            frame = pd.DataFrame(list(price_frame))
        if frame.empty:
            return pd.Series(dtype="float64")
        missing = {price_column, timestamp_column} - set(frame.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        frame = frame[[timestamp_column, price_column]].copy()
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
        frame = frame.dropna(subset=[timestamp_column, price_column])
        frame = frame.sort_values(timestamp_column)
        frame = frame.drop_duplicates(subset=timestamp_column, keep="last")
        series = frame.set_index(timestamp_column)[price_column].astype("float64")
        return series

    def _trim_series(self, series: pd.Series) -> pd.Series:
        if series.empty:
            return series
        result = series
        if self.window_duration is not None:
            cutoff = series.index.max() - self.window_duration
            result = result[result.index >= cutoff]
        if self.max_samples is not None and len(result) > self.max_samples:
            result = result.iloc[-self.max_samples :]
        return result

    def _build_return_frame(self) -> pd.DataFrame:
        if not self._returns:
            return pd.DataFrame()
        aligned: MutableMapping[str, pd.Series] = {}
        for symbol, series in self._returns.items():
            aligned[symbol] = series
        frame = pd.DataFrame(aligned)
        if frame.empty:
            return frame
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        return frame

    def _infer_periods_per_year(self, index: pd.DatetimeIndex) -> float:
        if self._annualization_override is not None:
            return float(self._annualization_override)
        if len(index) < 2:
            return DEFAULT_PERIODS_PER_YEAR
        ordinals = index.view("int64")
        if ordinals.size < 2:
            return DEFAULT_PERIODS_PER_YEAR
        diffs = np.diff(ordinals)  # Nanoseconds
        if diffs.size == 0:
            return DEFAULT_PERIODS_PER_YEAR
        seconds = diffs.astype(float) / 1e9
        seconds = seconds[np.isfinite(seconds) & (seconds > 0)]
        if seconds.size == 0:
            return DEFAULT_PERIODS_PER_YEAR
        median_seconds = float(np.median(seconds))
        if not math.isfinite(median_seconds) or median_seconds <= 0:
            return DEFAULT_PERIODS_PER_YEAR
        periods = SECONDS_PER_YEAR / median_seconds
        return float(max(1.0, min(periods, 1e6)))


__all__ = [
    "RollingRiskMetrics",
    "RiskMetricsSnapshot",
    "SymbolVolatilityStats",
]
