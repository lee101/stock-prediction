"""High-level orchestration for ensemble forecasting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .aggregator import ClippedMeanAggregator, EnsembleForecast
from .backends import BackendResult, EnsembleBackend, EnsembleRequest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EnsembleForecastOutput:
    """Wrapper returned by :class:`EnsembleForecastPipeline`."""

    request: EnsembleRequest
    forecast: EnsembleForecast
    backend_results: List[BackendResult]


class EnsembleForecastPipeline:
    """Coordinate multiple backends + shared aggregation policy."""

    def __init__(
        self,
        backends: Sequence[EnsembleBackend],
        *,
        aggregator: Optional[ClippedMeanAggregator] = None,
    ) -> None:
        if not backends:
            raise ValueError("At least one backend must be provided")
        self.backends = list(backends)
        self.aggregator = aggregator or ClippedMeanAggregator()

    def predict(
        self,
        data: pd.DataFrame,
        *,
        timestamp_col: str = "timestamp",
        columns: Sequence[str] = ("close",),
        prediction_length: int = 1,
        lookback: Optional[int] = None,
    ) -> EnsembleForecastOutput:
        if data.empty:
            raise ValueError("Input dataframe is empty")
        request = EnsembleRequest(
            data=data,
            timestamp_col=timestamp_col,
            columns=columns,
            prediction_length=prediction_length,
            lookback=lookback,
        )
        backend_results = self._collect_backend_results(request)
        forecast = self.aggregator.combine(backend_results, columns)
        return EnsembleForecastOutput(request=request, forecast=forecast, backend_results=backend_results)

    def _collect_backend_results(self, request: EnsembleRequest) -> List[BackendResult]:
        results: List[BackendResult] = []
        for backend in self.backends:
            if not backend.enabled:
                continue
            try:
                result = backend.run(request)
            except Exception as exc:
                logger.warning("Backend %s failed: %s", backend.name, exc)
                continue
            if not result.samples:
                logger.warning("Backend %s returned no samples", backend.name)
                continue
            results.append(result)
        if not results:
            raise RuntimeError("All ensemble backends failed; cannot aggregate forecasts")
        return results

