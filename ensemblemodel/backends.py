"""Backend interfaces and default implementations for ensemblemodel."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:  # Optional import; Chronos is not always installed
    from src.forecasting_bolt_wrapper import ForecastingBoltWrapper  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ForecastingBoltWrapper = None  # type: ignore

try:  # Optional import; Chronos2 stack may be absent on some machines
    from src.models.chronos2_wrapper import Chronos2OHLCWrapper, DEFAULT_QUANTILE_LEVELS
except ModuleNotFoundError:  # pragma: no cover - surfaced when chronos2 deps are missing
    Chronos2OHLCWrapper = None  # type: ignore
    DEFAULT_QUANTILE_LEVELS = (0.1, 0.5, 0.9)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.models.kronos_wrapper import KronosForecastingWrapper
    from src.models.toto_wrapper import TotoPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnsembleRequest:
    """Input payload shared across backends."""

    data: pd.DataFrame
    timestamp_col: str
    columns: Sequence[str]
    prediction_length: int
    lookback: Optional[int] = None


@dataclass
class BackendResult:
    """Normalized backend response."""

    name: str
    weight: float
    latency_s: float
    samples: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnsembleBackend:
    """Base class for ensemble backends."""

    def __init__(self, name: str, weight: float = 1.0, enabled: bool = True) -> None:
        self.name = name
        self.weight = float(weight)
        self.enabled = bool(enabled)

    def run(self, request: EnsembleRequest) -> BackendResult:
        raise NotImplementedError


class KronosBackend(EnsembleBackend):
    """Adapter over :class:`KronosForecastingWrapper`."""

    def __init__(
        self,
        *,
        model_name: str = "NeoQuasar/Kronos-base",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cuda:0",
        max_context: int = 256,
        clip: float = 2.0,
        temperature: float = 0.18,
        top_p: float = 0.82,
        top_k: int = 24,
        sample_count: int = 128,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="kronos", weight=weight, enabled=enabled)
        self._wrapper: "Optional[KronosForecastingWrapper]" = None
        self._init_kwargs = {
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "device": device,
            "max_context": max_context,
            "clip": clip,
        }
        self._default_temperature = temperature
        self._default_top_p = top_p
        self._default_top_k = top_k
        self._default_sample_count = sample_count

    def _ensure_wrapper(self) -> KronosForecastingWrapper:
        if self._wrapper is None:
            from src.models.kronos_wrapper import KronosForecastingWrapper

            self._wrapper = KronosForecastingWrapper(**self._init_kwargs)
        return self._wrapper

    def run(self, request: EnsembleRequest) -> BackendResult:
        wrapper = self._ensure_wrapper()
        start = time.perf_counter()
        result = wrapper.predict_series(
            data=request.data,
            timestamp_col=request.timestamp_col,
            columns=request.columns,
            pred_len=request.prediction_length,
            lookback=request.lookback or self._init_kwargs["max_context"],
            temperature=self._default_temperature,
            top_p=self._default_top_p,
            top_k=self._default_top_k,
            sample_count=self._default_sample_count,
        )
        latency = time.perf_counter() - start

        samples: Dict[str, np.ndarray] = {}
        for column in request.columns:
            forecast = result.get(column)
            if forecast is None:
                continue
            values = np.asarray(forecast.absolute[: request.prediction_length], dtype=np.float64)
            samples[column] = values.reshape(1, -1)

        return BackendResult(
            name=self.name,
            weight=self.weight,
            latency_s=latency,
            samples=samples,
            metadata={
                "temperature": self._default_temperature,
                "top_p": self._default_top_p,
                "top_k": self._default_top_k,
                "sample_count": self._default_sample_count,
            },
        )


class TotoBackend(EnsembleBackend):
    """Adapter over :class:`TotoPipeline`."""

    def __init__(
        self,
        *,
        model_id: str = "Datadog/Toto-Open-Base-1.0",
        device_map: str = "cuda",
        torch_dtype: str = "float16",
        num_samples: int = 512,
        samples_per_batch: int = 128,
        aggregate: str = "trimmed_mean_10",
        weight: float = 1.0,
        enabled: bool = True,
        pipeline_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(name="toto", weight=weight, enabled=enabled)
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.num_samples = int(num_samples)
        self.samples_per_batch = int(samples_per_batch)
        self.aggregate = aggregate
        self.pipeline_kwargs = dict(pipeline_kwargs or {})
        self._pipeline: "Optional[TotoPipeline]" = None

    def _ensure_pipeline(self) -> TotoPipeline:
        if self._pipeline is None:
            from src.models.toto_wrapper import TotoPipeline

            self._pipeline = TotoPipeline.from_pretrained(
                model_id=self.model_id,
                device_map=self.device_map,
                torch_dtype=self.pipeline_kwargs.get("torch_dtype_override", self.torch_dtype),
                **{k: v for k, v in self.pipeline_kwargs.items() if k != "torch_dtype_override"},
            )
        return self._pipeline

    def run(self, request: EnsembleRequest) -> BackendResult:
        pipeline = self._ensure_pipeline()
        start = time.perf_counter()
        samples: Dict[str, np.ndarray] = {}
        for column in request.columns:
            series = request.data[column].to_numpy(dtype=np.float32)
            forecasts = pipeline.predict(
                context=series,
                prediction_length=request.prediction_length,
                num_samples=self.num_samples,
                samples_per_batch=self.samples_per_batch,
            )
            if not forecasts:
                continue
            samples[column] = np.asarray(forecasts[0].numpy(), dtype=np.float64)
        latency = time.perf_counter() - start
        return BackendResult(
            name=self.name,
            weight=self.weight,
            latency_s=latency,
            samples=samples,
            metadata={
                "num_samples": self.num_samples,
                "samples_per_batch": self.samples_per_batch,
                "aggregate": self.aggregate,
            },
        )


class Chronos2Backend(EnsembleBackend):
    """Adapter over :class:`Chronos2OHLCWrapper` for quantile-driven ensembles."""

    def __init__(
        self,
        *,
        model_id: str = "amazon/chronos-2",
        device_map: str | Mapping[str, int] = "cuda",
        id_column: str = "symbol",
        timestamp_column: str = "timestamp",
        target_columns: Sequence[str] = ("open", "high", "low", "close"),
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
        context_length: int = 2048,
        batch_size: int = 256,
        known_future_covariates: Optional[Sequence[str]] = None,
        predict_kwargs: Optional[Mapping[str, Any]] = None,
        torch_compile: Optional[bool] = None,
        compile_mode: Optional[str] = None,
        compile_backend: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="chronos2", weight=weight, enabled=enabled)
        self.model_id = model_id
        self.device_map = device_map
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.target_columns: Sequence[str] = tuple(target_columns)
        self.quantile_levels: Sequence[float] = tuple(quantile_levels)
        self.context_length = max(32, int(context_length))
        self.batch_size = max(1, int(batch_size))
        self.known_future_covariates: Sequence[str] = tuple(known_future_covariates or ())
        self.predict_kwargs = dict(predict_kwargs or {})
        self.torch_compile = torch_compile
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend
        self.torch_dtype = torch_dtype
        self._wrapper: Optional[Chronos2OHLCWrapper] = None

    def _ensure_wrapper(self) -> Chronos2OHLCWrapper:
        if Chronos2OHLCWrapper is None:  # pragma: no cover - surfaced when Chronos2 deps missing
            raise RuntimeError(
                "Chronos2OHLCWrapper unavailable. Install chronos-forecasting>=2.0 to enable this backend."
            )
        if self._wrapper is None:
            self._wrapper = Chronos2OHLCWrapper.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                id_column=self.id_column,
                timestamp_column=self.timestamp_column,
                target_columns=self.target_columns,
                default_context_length=self.context_length,
                default_batch_size=self.batch_size,
                quantile_levels=self.quantile_levels,
                torch_compile=self.torch_compile,
                compile_mode=self.compile_mode,
                compile_backend=self.compile_backend,
                torch_dtype=self.torch_dtype,
            )
        return self._wrapper

    def run(self, request: EnsembleRequest) -> BackendResult:
        wrapper = self._ensure_wrapper()
        context_len = self._determine_context_length(len(request.data), request.lookback)
        context_df = request.data.tail(context_len).copy()
        symbol = self._resolve_symbol(context_df)

        start = time.perf_counter()
        batch = wrapper.predict_ohlc(
            context_df,
            symbol=symbol,
            prediction_length=request.prediction_length,
            context_length=context_len,
            quantile_levels=self.quantile_levels,
            known_future_covariates=self.known_future_covariates or None,
            batch_size=self.batch_size,
            predict_kwargs=self.predict_kwargs,
        )
        latency = time.perf_counter() - start

        samples: Dict[str, np.ndarray] = {}
        for column in request.columns:
            trajectories = []
            for level in self.quantile_levels:
                frame = batch.quantile_frames.get(level)
                if frame is None or column not in frame:
                    continue
                values = frame[column].to_numpy(dtype=np.float64)
                trajectories.append(values.reshape(1, -1))
            if trajectories:
                samples[column] = np.vstack(trajectories)

        if not samples:
            raise RuntimeError("Chronos2 backend produced no samples for requested columns")

        metadata = {
            "model_id": self.model_id,
            "quantile_levels": list(self.quantile_levels),
            "context_length": int(context_len),
            "batch_size": self.batch_size,
            "predict_kwargs": dict(self.predict_kwargs),
        }

        return BackendResult(
            name=self.name,
            weight=self.weight,
            latency_s=latency,
            samples=samples,
            metadata=metadata,
        )

    def _determine_context_length(self, available_rows: int, lookback: Optional[int]) -> int:
        window = min(self.context_length, available_rows)
        if lookback is not None:
            window = min(window, int(lookback))
        return max(8, window)

    def _resolve_symbol(self, df: pd.DataFrame) -> Optional[str]:
        if self.id_column in df:
            series = df[self.id_column].dropna()
            if not series.empty:
                return str(series.iloc[-1])
        return None


class ChronosBoltBackend(EnsembleBackend):
    """Adapter over the Chronos Bolt pipeline."""

    def __init__(
        self,
        *,
        model_name: str = "amazon/chronos-bolt-base",
        device: str = "cuda",
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="chronos_bolt", weight=weight, enabled=enabled)
        self.model_name = model_name
        self.device = device
        self._wrapper: Optional[ForecastingBoltWrapper] = None

    def _ensure_wrapper(self) -> ForecastingBoltWrapper:
        if ForecastingBoltWrapper is None:  # pragma: no cover - defensive
            raise RuntimeError("Chronos dependencies unavailable; install chronos to enable this backend")
        if self._wrapper is None:
            self._wrapper = ForecastingBoltWrapper(model_name=self.model_name, device=self.device)
        return self._wrapper

    def run(self, request: EnsembleRequest) -> BackendResult:
        wrapper = self._ensure_wrapper()
        start = time.perf_counter()
        samples: Dict[str, np.ndarray] = {}
        for column in request.columns:
            context = request.data[column].to_numpy(dtype=np.float32)
            predictions = wrapper.predict_sequence(context, prediction_length=request.prediction_length)
            arr = np.asarray(predictions, dtype=np.float64).reshape(1, -1)
            samples[column] = arr
        latency = time.perf_counter() - start
        return BackendResult(
            name=self.name,
            weight=self.weight,
            latency_s=latency,
            samples=samples,
            metadata={"model_name": self.model_name},
        )
