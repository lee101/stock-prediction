from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hyperparamstore.store import HyperparamRecord, HyperparamStore
from src.models.toto_aggregation import aggregate_with_spec

try:  # pragma: no cover - exercised in integration environments
    from src.models.toto_wrapper import TotoPipeline
except Exception as exc:  # pragma: no cover - lazily surfaced when Toto is needed
    TotoPipeline = None  # type: ignore
    _TOTO_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover - only hit when Toto import succeeds
    _TOTO_IMPORT_ERROR = None

try:  # pragma: no cover - exercised in integration environments
    from src.models.kronos_wrapper import KronosForecastResult, KronosForecastingWrapper
except Exception as exc:  # pragma: no cover - lazily surfaced when Kronos is needed
    KronosForecastResult = None  # type: ignore
    KronosForecastingWrapper = None  # type: ignore
    _KRONOS_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover - only hit when Kronos import succeeds
    _KRONOS_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - import is optional at runtime
    import torch


@dataclass(frozen=True)
class ErrorBreakdown:
    """Container for model error statistics."""

    price_mae: float
    pct_return_mae: float
    latency_s: float


@dataclass(frozen=True)
class ModelForecast:
    """Per-model forecast enriched with hyperparameter metadata."""

    symbol: str
    model: str
    config_name: str
    config: Mapping[str, Any]
    validation: ErrorBreakdown
    test: ErrorBreakdown
    average_price_mae: float
    average_pct_return_mae: float
    forecasts: Mapping[str, float]


@dataclass(frozen=True)
class CombinedForecast:
    """Aggregated forecast that blends available model forecasts."""

    symbol: str
    model_forecasts: Mapping[str, ModelForecast]
    combined: Mapping[str, float]
    weights: Mapping[str, float]
    best_model: Optional[str]
    selection_source: Optional[str]


class CombinedForecastGenerator:
    """
    Generate blended OHLC forecasts by combining Kronos and Toto hyperparameter winners.

    The generator loads the persisted hyperparameter evaluations produced by
    ``test_hyperparamtraining_kronos_toto.py`` and rehydrates the corresponding
    forecasting wrappers to produce the next-step forecasts for Open/High/Low/Close.
    """

    def __init__(
        self,
        *,
        data_root: Path | str = Path("trainingdata"),
        hyperparam_root: Path | str = Path("hyperparams"),
        prediction_columns: Optional[Sequence[str]] = None,
        timestamp_column: str = "timestamp",
        hyperparam_store: Optional[HyperparamStore] = None,
        toto_factory: Optional[Callable[[Mapping[str, Any]], Any]] = None,
        kronos_factory: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    ) -> None:
        if "FAST_TESTING" not in os.environ:
            os.environ["FAST_TESTING"] = "1"
        self.fast_testing = os.getenv("FAST_TESTING", "0").strip().lower() in {"1", "true", "yes", "on"}

        self.data_root = Path(data_root)
        self.timestamp_column = timestamp_column
        self.columns = tuple(prediction_columns or ("open", "high", "low", "close"))
        self.store = hyperparam_store or HyperparamStore(hyperparam_root)

        self._toto_factory = toto_factory
        self._kronos_factory = kronos_factory
        self._toto_pipeline: Optional[Any] = None
        self._kronos_cache: MutableMapping[str, Any] = {}

    # --------------------------------------------------------------------- #
    # Public orchestration
    # --------------------------------------------------------------------- #
    def generate(
        self,
        symbols: Iterable[str],
        *,
        prediction_length: int = 1,
        historical_data: Optional[Mapping[str, pd.DataFrame]] = None,
    ) -> Dict[str, CombinedForecast]:
        """Generate combined forecasts for a collection of symbols."""
        results: Dict[str, CombinedForecast] = {}
        for symbol in symbols:
            frame_override = None
            if historical_data is not None:
                frame_override = historical_data.get(symbol)
            results[symbol] = self.generate_for_symbol(
                symbol,
                prediction_length=prediction_length,
                historical_frame=frame_override,
            )
        return results

    def generate_for_symbol(
        self,
        symbol: str,
        *,
        prediction_length: int = 1,
        historical_frame: Optional[pd.DataFrame] = None,
    ) -> CombinedForecast:
        """Generate a combined forecast for a single symbol."""
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive.")

        if historical_frame is not None:
            df = self._prepare_history_frame(historical_frame)
        else:
            df = self._load_symbol_history(symbol)

        if len(df) < prediction_length:
            raise ValueError(
                f"Not enough history ({len(df)}) to forecast {prediction_length} steps for {symbol}."
            )
        selection_payload = self.store.load_selection(symbol)

        model_forecasts: Dict[str, ModelForecast] = {}

        for model_name in ("toto", "kronos"):
            record = self.store.load(model_name, symbol)
            if record is None:
                continue
            forecasts = self._forecast_with_model(
                model_name=model_name,
                record=record,
                df=df,
                prediction_length=prediction_length,
            )
            model_forecasts[model_name] = self._build_model_forecast(
                symbol=symbol,
                model_name=model_name,
                record=record,
                forecasts=forecasts,
            )

        if not model_forecasts:
            raise FileNotFoundError(
                f"No hyperparameter records found for symbol '{symbol}'. "
                f"Expected files under {self.store.root}."
            )

        combined, weights = self._combine_model_forecasts(model_forecasts)

        best_model: Optional[str] = None
        selection_source: Optional[str] = None
        if selection_payload and selection_payload.get("model") in model_forecasts:
            best_model = selection_payload["model"]
            selection_source = "hyperparams/best"
        else:
            # Fall back to the model with the lowest average price MAE.
            best_model = min(
                model_forecasts.keys(),
                key=lambda name: (
                    model_forecasts[name].average_price_mae
                    if not math.isnan(model_forecasts[name].average_price_mae)
                    else float("inf")
                ),
            )
            selection_source = "computed_average_mae"

        return CombinedForecast(
            symbol=symbol,
            model_forecasts=model_forecasts,
            combined=combined,
            weights=weights,
            best_model=best_model,
            selection_source=selection_source,
        )

    # --------------------------------------------------------------------- #
    # Forecast execution helpers
    # --------------------------------------------------------------------- #
    def _forecast_with_model(
        self,
        *,
        model_name: str,
        record: HyperparamRecord,
        df: pd.DataFrame,
        prediction_length: int,
    ) -> Dict[str, float]:
        if model_name == "toto":
            return self._forecast_with_toto(record, df, prediction_length)
        if model_name == "kronos":
            return self._forecast_with_kronos(record, df, prediction_length)
        raise ValueError(f"Unsupported model '{model_name}'.")

    def _forecast_with_toto(
        self,
        record: HyperparamRecord,
        df: pd.DataFrame,
        prediction_length: int,
    ) -> Dict[str, float]:
        pipeline = self._get_toto_pipeline(record.config)

        config = record.config
        num_samples = int(config.get("num_samples", 256))
        samples_per_batch = int(config.get("samples_per_batch", min(num_samples, 512)))
        aggregate_spec = str(config.get("aggregate", "mean"))

        if self.fast_testing:
            fast_cap = int(config.get("fast_num_samples", 256))
            num_samples = max(1, min(num_samples, fast_cap))
            samples_per_batch = max(1, min(samples_per_batch, 128))

        inference_ctx = None
        torch_mod = None
        try:
            import torch  # type: ignore
        except Exception:  # pragma: no cover - tests may omit torch
            torch_mod = None
        else:
            torch_mod = torch  # type: ignore
            inference_ctx = getattr(torch_mod, "inference_mode", None)

        forecasts: Dict[str, float] = {}
        for column in self.columns:
            series = pd.Series(df[column], dtype=np.float64)
            series = series.replace([np.inf, -np.inf], np.nan).ffill().dropna()
            if len(series) < max(2, prediction_length):
                raise ValueError(
                    f"Not enough history ({len(series)} rows) to forecast '{column}' with Toto."
                )
            context = series.to_numpy(dtype=np.float32, copy=False)
            if inference_ctx is not None:
                with inference_ctx():
                    outputs = pipeline.predict(
                        context=context,
                        prediction_length=prediction_length,
                        num_samples=num_samples,
                        samples_per_batch=samples_per_batch,
                    )
            elif torch_mod is not None:
                with torch_mod.no_grad():
                    outputs = pipeline.predict(
                        context=context,
                        prediction_length=prediction_length,
                        num_samples=num_samples,
                        samples_per_batch=samples_per_batch,
                    )
            else:
                outputs = pipeline.predict(
                    context=context,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                    samples_per_batch=samples_per_batch,
                )
            if not outputs:
                raise RuntimeError("Toto pipeline returned no forecasts.")
            aggregated = aggregate_with_spec(outputs[0].samples, aggregate_spec)
            forecasts[column] = float(np.asarray(aggregated, dtype=np.float64).ravel()[0])
        return forecasts

    def _forecast_with_kronos(
        self,
        record: HyperparamRecord,
        df: pd.DataFrame,
        prediction_length: int,
    ) -> Dict[str, float]:
        wrapper = self._get_kronos_wrapper(record.config)
        hydrated_df = self._append_future_rows(df, steps=prediction_length)
        results = wrapper.predict_series(
            data=hydrated_df,
            timestamp_col=self.timestamp_column,
            columns=self.columns,
            pred_len=prediction_length,
            lookback=int(record.config.get("max_context", wrapper.max_context)),
            temperature=float(record.config.get("temperature", wrapper.temperature)),
            top_p=float(record.config.get("top_p", wrapper.top_p)),
            top_k=int(record.config.get("top_k", wrapper.top_k)),
            sample_count=int(record.config.get("sample_count", wrapper.sample_count)),
        )

        forecasts: Dict[str, float] = {}
        for column in self.columns:
            result: KronosForecastResult = results.get(column)
            if result is None:
                raise RuntimeError(f"Kronos wrapper returned no forecast for column '{column}'.")
            if result.absolute.size < prediction_length:
                raise RuntimeError(
                    f"Kronos forecast for '{column}' contains {result.absolute.size} "
                    f"values but prediction_length={prediction_length}."
                )
            forecasts[column] = float(result.absolute[0])
        return forecasts

    # --------------------------------------------------------------------- #
    # Assembly helpers
    # --------------------------------------------------------------------- #
    def _build_model_forecast(
        self,
        *,
        symbol: str,
        model_name: str,
        record: HyperparamRecord,
        forecasts: Mapping[str, float],
    ) -> ModelForecast:
        validation = self._build_error_breakdown(record.validation)
        test = self._build_error_breakdown(record.test)

        avg_price_mae = float(
            np.nanmean([validation.price_mae, test.price_mae])
        )
        avg_pct_return_mae = float(
            np.nanmean([validation.pct_return_mae, test.pct_return_mae])
        )

        config_name = str(record.config.get("name", model_name))

        return ModelForecast(
            symbol=symbol,
            model=model_name,
            config_name=config_name,
            config=record.config,
            validation=validation,
            test=test,
            average_price_mae=avg_price_mae,
            average_pct_return_mae=avg_pct_return_mae,
            forecasts=dict(forecasts),
        )

    def _combine_model_forecasts(
        self,
        model_forecasts: Mapping[str, ModelForecast],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        weights: Dict[str, float] = {}
        for name, forecast in model_forecasts.items():
            mae = forecast.average_price_mae
            if math.isnan(mae) or mae <= 0.0:
                weights[name] = 1.0
            else:
                weights[name] = 1.0 / mae

        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            equal_weight = 1.0 / len(model_forecasts)
            normalized_weights = {name: equal_weight for name in model_forecasts}
        else:
            normalized_weights = {name: weight / weight_sum for name, weight in weights.items()}

        combined: Dict[str, float] = {}
        for column in self.columns:
            total = 0.0
            for name, forecast in model_forecasts.items():
                column_value = forecast.forecasts[column]
                total += normalized_weights[name] * column_value
            combined[column] = total

        return combined, normalized_weights

    # --------------------------------------------------------------------- #
    # Loading helpers
    # --------------------------------------------------------------------- #
    def _prepare_history_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.timestamp_column not in frame.columns:
            if frame.index.name == self.timestamp_column:
                frame = frame.reset_index()
            elif self.timestamp_column in frame.index.names:
                frame = frame.reset_index()
            else:
                raise ValueError(f"Historical frame missing '{self.timestamp_column}' column.")

        result = frame.copy()
        result = result.dropna(subset=[self.timestamp_column])
        result[self.timestamp_column] = pd.to_datetime(
            result[self.timestamp_column],
            utc=True,
            errors="coerce",
        )
        result = result.dropna(subset=[self.timestamp_column])
        result = result.sort_values(self.timestamp_column).reset_index(drop=True)

        missing = [column for column in self.columns if column not in result.columns]
        if missing:
            raise ValueError(f"Historical frame missing required columns: {missing}")
        return result

    def _load_symbol_history(self, symbol: str) -> pd.DataFrame:
        path = self.data_root / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Training data for symbol '{symbol}' not found at {path}.")
        df = pd.read_csv(path)
        if self.timestamp_column not in df.columns:
            raise ValueError(f"Column '{self.timestamp_column}' is missing from {path}.")
        df = df.sort_values(self.timestamp_column).reset_index(drop=True)
        return df

    def _append_future_rows(self, df: pd.DataFrame, *, steps: int) -> pd.DataFrame:
        timestamps_series = pd.Series(
            pd.to_datetime(
                df[self.timestamp_column],
                utc=True,
                errors="coerce",
            ),
            copy=False,
        )
        if timestamps_series.isna().any():
            raise ValueError("Encountered invalid timestamps while preparing Kronos inputs.")
        if len(timestamps_series) < 2:
            raise ValueError("At least two timestamps are required to infer forecast spacing.")

        # Use the most recent non-zero delta; fall back to one day if needed.
        deltas = timestamps_series.diff().dropna()
        deltas = deltas[deltas != pd.Timedelta(0)]
        delta = deltas.iloc[-1] if not deltas.empty else pd.Timedelta(days=1)
        if delta <= pd.Timedelta(0):
            delta = pd.Timedelta(days=1)

        future_rows = []
        last_timestamp = timestamps_series.iloc[-1]
        for step in range(1, steps + 1):
            next_timestamp = last_timestamp + step * delta
            row = {col: np.nan for col in df.columns}
            row[self.timestamp_column] = next_timestamp
            future_rows.append(row)

        future_df = pd.concat([df, pd.DataFrame(future_rows)], ignore_index=True)
        future_df[self.timestamp_column] = pd.to_datetime(future_df[self.timestamp_column], utc=True)
        return future_df

    def _build_error_breakdown(self, payload: Mapping[str, Any]) -> ErrorBreakdown:
        def _extract(key: str) -> float:
            value = payload.get(key, float("nan"))
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        return ErrorBreakdown(
            price_mae=_extract("price_mae"),
            pct_return_mae=_extract("pct_return_mae"),
            latency_s=_extract("latency_s"),
        )

    # --------------------------------------------------------------------- #
    # Wrapper loaders with caching
    # --------------------------------------------------------------------- #
    def _get_toto_pipeline(self, config: Mapping[str, Any]) -> Any:
        if self._toto_pipeline is not None:
            return self._toto_pipeline
        if self._toto_factory is not None:
            self._toto_pipeline = self._toto_factory(config)
            return self._toto_pipeline
        if TotoPipeline is None:  # pragma: no cover - surfaced only when Toto import fails
            assert _TOTO_IMPORT_ERROR is not None
            raise RuntimeError(
                "TotoPipeline is unavailable. Ensure Toto dependencies are installed."
            ) from _TOTO_IMPORT_ERROR

        device_override = os.getenv("STOCKAGENT_TOTO_DEVICE_MAP")
        device_map = str(
            config.get(
                "device_map",
                device_override if device_override else ("cuda" if self._cuda_available() else "cpu"),
            )
        )
        toto_kwargs = self._build_toto_kwargs(config)
        self._apply_default_toto_dtypes(toto_kwargs)
        self._toto_pipeline = TotoPipeline.from_pretrained(
            model_id=config.get("model_id", "Datadog/Toto-Open-Base-1.0"),
            device_map=device_map,
            **toto_kwargs,
        )
        return self._toto_pipeline

    def _get_kronos_wrapper(self, config: Mapping[str, Any]) -> Any:
        name = str(config.get("name", "default"))
        cached = self._kronos_cache.get(name)
        if cached is not None:
            return cached
        if self._kronos_factory is not None:
            wrapper = self._kronos_factory(config)
            self._kronos_cache[name] = wrapper
            return wrapper
        if KronosForecastingWrapper is None:  # pragma: no cover - surfaced only when import fails
            assert _KRONOS_IMPORT_ERROR is not None
            raise RuntimeError(
                "KronosForecastingWrapper is unavailable. Ensure Kronos dependencies are installed."
            ) from _KRONOS_IMPORT_ERROR

        device = config.get("device", "cuda:0")
        wrapper = KronosForecastingWrapper(
            model_name=config.get("model_name", "NeoQuasar/Kronos-base"),
            tokenizer_name=config.get("tokenizer_name", "NeoQuasar/Kronos-Tokenizer-base"),
            device=device,
            max_context=int(config.get("max_context", 512)),
            clip=float(config.get("clip", 5.0)),
            temperature=float(config.get("temperature", 0.75)),
            top_p=float(config.get("top_p", 0.9)),
            top_k=int(config.get("top_k", 0)),
            sample_count=int(config.get("sample_count", 8)),
        )
        self._kronos_cache[name] = wrapper
        return wrapper

    def _build_toto_kwargs(
        self,
        config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if "torch_dtype" in config:
            dtype = self._parse_torch_dtype(config["torch_dtype"])
            if dtype is not None:
                kwargs["torch_dtype"] = dtype
        if "amp_dtype" in config:
            amp_dtype = self._parse_torch_dtype(config["amp_dtype"])
            if amp_dtype is not None:
                kwargs["amp_dtype"] = amp_dtype
        for key in ("compile_model", "compile_mode", "torch_compile", "compile_backend"):
            if key in config:
                kwargs[key] = config[key]
        for key in ("max_oom_retries", "min_samples_per_batch", "min_num_samples"):
            if key in config:
                kwargs[key] = config[key]
        return kwargs

    def _apply_default_toto_dtypes(self, kwargs: Dict[str, Any]) -> None:
        try:
            import torch  # type: ignore
        except Exception:  # pragma: no cover - torch may be missing in stubbed tests
            return

        if not self._cuda_available():
            return

        kwargs.setdefault("torch_dtype", torch.bfloat16)  # type: ignore[attr-defined]
        kwargs.setdefault("amp_dtype", torch.bfloat16)  # type: ignore[attr-defined]

    @staticmethod
    def _parse_torch_dtype(value: Any) -> Optional["torch.dtype"]:
        try:
            import torch
        except Exception:  # pragma: no cover - torch may be missing in stubbed tests
            return None
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "half": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            return mapping.get(normalized)
        return None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
        except Exception:  # pragma: no cover - torch may be missing in tests
            return False
        return torch.cuda.is_available()
