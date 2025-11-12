"""
Chronos2 forecasting helper that standardises OHLC panel preparation and prediction.

The goal is to mirror the ergonomics of ``src/models/toto_wrapper.py`` while leveraging
the pandas-first API that ships with Chronos 2. The wrapper focuses on:

* Preparing multi-target OHLC panels with long (8k+) context windows
* Generating quantile forecasts via ``Chronos2Pipeline.predict_df``
* Pivoting the prediction DataFrame into per-target matrices that downstream
  evaluators can consume without additional munging
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from src.gpu_utils import should_offload_to_cpu as gpu_should_offload_to_cpu
from src.preaug import PreAugmentationChoice, PreAugmentationSelector

from preaug_sweeps.augmentations import BaseAugmentation

try:  # pragma: no cover - exercised when dependencies are available.
    from chronos import Chronos2Pipeline as _Chronos2Pipeline
except Exception as exc:  # pragma: no cover - graceful degradation on missing deps.
    _Chronos2Pipeline = None
    _CHRONOS2_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when chronos is available.
    _CHRONOS2_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

DEFAULT_TARGET_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close")
DEFAULT_QUANTILE_LEVELS: Tuple[float, ...] = (0.1, 0.5, 0.9)
_BOOL_TRUE = {"1", "true", "yes", "on"}


def _require_chronos2_pipeline() -> type:
    """Return the Chronos2Pipeline class or raise a descriptive error."""

    if _Chronos2Pipeline is None:
        raise RuntimeError(
            "chronos>=2.0 is unavailable; install `chronos-forecasting>=2.0` to enable Chronos2Pipeline."
        ) from _CHRONOS2_IMPORT_ERROR
    return _Chronos2Pipeline


def _quantile_column(level: float) -> str:
    """Return the column name used by Chronos for a quantile level."""

    return format(level, "g")


def _normalize_symbol(symbol: Optional[str], df: pd.DataFrame, id_column: str) -> str:
    """Return the symbol/id that should label the prepared panel."""

    if symbol:
        return str(symbol)
    if id_column in df and not df[id_column].isna().all():
        last_value = df[id_column].dropna().iloc[-1]
        return str(last_value)
    return "timeseries_0"


def _parse_torch_dtype(value: Any) -> Optional["torch.dtype"]:
    if torch is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(normalized)
    return None


@dataclass
class Chronos2PreparedPanel:
    """Container describing a single symbol's context/forecast split."""

    symbol: str
    context_df: pd.DataFrame
    future_df: pd.DataFrame | None
    actual_df: pd.DataFrame
    context_length: int
    prediction_length: int
    id_column: str
    timestamp_column: str
    target_columns: Tuple[str, ...]


QuantileFrameMap = Dict[float, pd.DataFrame]


@dataclass
class Chronos2PredictionBatch:
    """Chronos2 prediction plus convenience accessors for quantile pivots."""

    panel: Chronos2PreparedPanel
    raw_dataframe: pd.DataFrame
    quantile_frames: QuantileFrameMap
    applied_augmentation: Optional[str] = None

    def quantile(self, level: float) -> pd.DataFrame:
        """Return the pivoted frame for the requested quantile level."""

        if level not in self.quantile_frames:
            raise KeyError(f"Quantile {level} unavailable; computed levels={list(self.quantile_frames)}")
        return self.quantile_frames[level]

    @property
    def median(self) -> pd.DataFrame:
        """Convenience accessor for the 0.5 quantile."""

        return self.quantile(0.5)


@dataclass
class AppliedAugmentation:
    choice: PreAugmentationChoice
    augmentation: BaseAugmentation
    columns: Tuple[str, ...]
    context_reference: pd.DataFrame


class Chronos2OHLCWrapper:
    """High-level helper around Chronos2Pipeline for OHLC multi-target forecasting."""

    def __init__(
        self,
        pipeline: object,
        *,
        device_hint: Optional[str] = "cuda",
        id_column: str = "symbol",
        timestamp_column: str = "timestamp",
        target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
        default_context_length: int = 8192,
        default_batch_size: int = 256,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
        torch_compile: Optional[bool] = None,
        compile_mode: Optional[str] = None,
        compile_backend: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        preaugmentation_dirs: Optional[Sequence[str | Path]] = None,
    ) -> None:
        if pipeline is None:
            raise RuntimeError("Chronos2Pipeline instance is required.")

        compile_env = os.getenv("CHRONOS_COMPILE")
        if torch_compile is None and compile_env is not None:
            torch_compile = compile_env.strip().lower() in _BOOL_TRUE
        dtype_env = os.getenv("CHRONOS_DTYPE")
        if torch_dtype is None and dtype_env:
            torch_dtype = dtype_env
        if compile_mode is None:
            compile_mode = os.getenv("CHRONOS_COMPILE_MODE")
        if compile_backend is None:
            compile_backend = os.getenv("CHRONOS_COMPILE_BACKEND")

        self.pipeline = pipeline
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.target_columns: Tuple[str, ...] = tuple(target_columns)
        self.default_context_length = int(default_context_length)
        self.default_batch_size = max(1, int(default_batch_size))
        self.quantile_levels: Tuple[float, ...] = tuple(quantile_levels)
        self._device_hint = device_hint or "cuda"
        self._torch_compile_enabled = bool(torch_compile and torch is not None and hasattr(torch, "compile"))
        self._torch_compile_success = False
        self._compile_mode = compile_mode
        self._compile_backend = compile_backend

        default_preaug_dirs: Sequence[str | Path]
        if preaugmentation_dirs is None:
            default_preaug_dirs = (Path("preaugstrategies") / "chronos2", Path("preaugstrategies") / "best")
        else:
            default_preaug_dirs = preaugmentation_dirs
        dirs_list = [Path(d) for d in default_preaug_dirs if d]
        self._preaug_selector: Optional[PreAugmentationSelector] = (
            PreAugmentationSelector(dirs_list) if dirs_list else None
        )

        dtype_obj = _parse_torch_dtype(torch_dtype)
        if dtype_obj is not None and torch is not None:
            try:
                self.pipeline.model = self.pipeline.model.to(dtype=dtype_obj)  # type: ignore[attr-defined]
                logger.info("Chronos2 model moved to dtype=%s", dtype_obj)
            except Exception as exc:  # pragma: no cover - dtype tuning best effort
                logger.warning("Chronos2 dtype cast failed: %s", exc)

        if self._torch_compile_enabled:
            compile_kwargs: Dict[str, Any] = {}
            if compile_mode:
                compile_kwargs["mode"] = compile_mode
            if compile_backend:
                compile_kwargs["backend"] = compile_backend
            try:
                compiled = torch.compile(self.pipeline.model, **compile_kwargs)  # type: ignore[arg-type]
                self.pipeline.model = compiled  # type: ignore[attr-defined]
                self._torch_compile_success = True
                cache_dir = os.getenv("TORCHINDUCTOR_CACHE_DIR")
                if not cache_dir:
                    cache_dir = os.path.join(os.getcwd(), "compiled_models", "chronos2_torch_inductor")
                    os.makedirs(cache_dir, exist_ok=True)
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
                logger.info(
                    "Chronos2 torch.compile enabled (mode=%s backend=%s)",
                    compile_mode or "reduce-overhead",
                    compile_backend or "inductor",
                )
            except Exception as exc:  # pragma: no cover
                self._torch_compile_enabled = False
                logger.warning("Chronos2 torch.compile failed: %s", exc)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "amazon/chronos-2",
        *,
        device_map: str | Mapping[str, str] | None = "cuda",
        id_column: str = "symbol",
        timestamp_column: str = "timestamp",
        target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
        default_context_length: int = 8192,
        default_batch_size: int = 256,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
        torch_compile: Optional[bool] = None,
        compile_mode: Optional[str] = None,
        compile_backend: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        preaugmentation_dirs: Optional[Sequence[str | Path]] = None,
        **kwargs,
    ) -> "Chronos2OHLCWrapper":
        """
        Instantiate Chronos2 via Hugging Face and wrap it with OHLC helpers.
        """

        pipeline_cls = _require_chronos2_pipeline()
        pipeline = pipeline_cls.from_pretrained(model_id, device_map=device_map, **kwargs)
        device_hint = device_map if isinstance(device_map, str) else "cuda"
        return cls(
            pipeline,
            device_hint=device_hint,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            default_context_length=default_context_length,
            default_batch_size=default_batch_size,
            quantile_levels=quantile_levels,
            torch_compile=torch_compile,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
            torch_dtype=torch_dtype,
            preaugmentation_dirs=preaugmentation_dirs,
        )

    def unload(self) -> None:
        """Release GPU memory by offloading the Chronos2 model to CPU if needed."""

        pipeline = getattr(self, "pipeline", None)
        if pipeline is None:
            return

        should_offload = gpu_should_offload_to_cpu(str(self._device_hint))
        if should_offload:
            model = getattr(pipeline, "model", None)
            move_to = getattr(model, "to", None)
            if callable(move_to):
                try:
                    move_to("cpu")
                except Exception as exc:  # pragma: no cover - best-effort cleanup
                    logger.debug("Chronos2 model offload failed: %s", exc)

        self.pipeline = None

    def _maybe_apply_preaugmentation(
        self,
        symbol: str,
        context_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[AppliedAugmentation]]:
        if self._preaug_selector is None:
            return context_df, None

        choice = self._preaug_selector.get_choice(symbol)
        if choice is None or choice.strategy == "baseline":
            return context_df, None

        try:
            augmentation = choice.instantiate()
        except Exception as exc:
            logger.warning(
                "Failed to instantiate pre-augmentation '%s' for %s: %s",
                choice.strategy,
                symbol,
                exc,
            )
            return context_df, None

        target_cols = [col for col in self.target_columns if col in context_df.columns]
        if not target_cols:
            return context_df, None

        reference = context_df[target_cols].copy()
        try:
            transformed = augmentation.transform_dataframe(reference.copy())
        except Exception as exc:
            logger.warning(
                "Pre-augmentation '%s' failed for %s: %s",
                choice.strategy,
                symbol,
                exc,
            )
            return context_df, None

        augmented_context = context_df.copy()
        for column in transformed.columns:
            if column not in augmented_context.columns:
                continue
            series = transformed[column]
            target_dtype = augmented_context[column].dtype
            if hasattr(series, "to_numpy"):
                values = series.to_numpy(dtype=target_dtype, copy=False)
            else:
                values = np.asarray(series, dtype=target_dtype)
            augmented_context[column] = values

        applied = AppliedAugmentation(
            choice=choice,
            augmentation=augmentation,
            columns=tuple(target_cols),
            context_reference=reference,
        )
        return augmented_context, applied

    def _apply_inverse_augmentation(
        self,
        panel: Chronos2PreparedPanel,
        quantile_frames: QuantileFrameMap,
        raw_predictions: pd.DataFrame,
        quantiles: Sequence[float],
        quantile_columns: Sequence[str],
        applied: AppliedAugmentation,
    ) -> Tuple[QuantileFrameMap, pd.DataFrame]:
        columns = list(applied.columns)

        # Restore the context dataframe to original scale for interpretability.
        try:
            restored_context = applied.augmentation.inverse_transform_predictions(
                panel.context_df[columns].to_numpy(),
                context=applied.context_reference,
                columns=columns,
            )
            restored_df = pd.DataFrame(restored_context, index=panel.context_df.index, columns=columns)
            for column in columns:
                target_dtype = panel.context_df[column].dtype
                panel.context_df.loc[:, column] = restored_df[column].astype(target_dtype, copy=False)
        except Exception as exc:
            logger.warning(
                "Failed to inverse-transform context for %s/%s: %s",
                panel.symbol,
                applied.choice.strategy,
                exc,
            )
            return quantile_frames, raw_predictions

        raw_index = raw_predictions.set_index([self.timestamp_column, "target_name"])

        for level, column_name in zip(quantiles, quantile_columns):
            pivot = quantile_frames.get(level)
            if pivot is None:
                continue
            if any(col not in pivot.columns for col in columns):
                continue

            try:
                restored_preds = applied.augmentation.inverse_transform_predictions(
                    pivot[columns].to_numpy(),
                    context=applied.context_reference,
                    columns=columns,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to inverse-transform predictions for %s/%s (quantile %.3f): %s",
                    panel.symbol,
                    applied.choice.strategy,
                    level,
                    exc,
                )
                continue

            restored_df = pd.DataFrame(restored_preds, index=pivot.index, columns=columns)
            for column in columns:
                target_dtype = pivot[column].dtype
                pivot[column] = restored_df[column].astype(target_dtype, copy=False).values
            quantile_frames[level] = pivot

            melted = (
                restored_df.reset_index()
                .melt(id_vars=self.timestamp_column, var_name="target_name", value_name="__value")
                .set_index([self.timestamp_column, "target_name"])
            )
            values = melted["__value"].astype("float32")
            overlapping = raw_index.index.intersection(values.index)
            if len(overlapping) > 0:
                raw_index.loc[overlapping, column_name] = values.loc[overlapping]

        updated_raw = raw_index.reset_index()
        return quantile_frames, updated_raw

    @staticmethod
    def _empty_actual_frame(timestamp_column: str, target_columns: Sequence[str]) -> pd.DataFrame:
        empty_index = pd.DatetimeIndex([], name=timestamp_column)
        return pd.DataFrame(columns=target_columns, index=empty_index)

    @staticmethod
    def build_panel(
        context_df: pd.DataFrame,
        *,
        holdout_df: Optional[pd.DataFrame],
        future_covariates: Optional[pd.DataFrame],
        symbol: Optional[str],
        id_column: str,
        timestamp_column: str,
        target_columns: Sequence[str],
        prediction_length: int,
        context_length: int,
        known_future_covariates: Optional[Sequence[str]] = None,
        dropna: bool = True,
    ) -> Chronos2PreparedPanel:
        """
        Split historical context (and optional holdout targets) into Chronos-friendly payloads.
        """

        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive.")

        if context_length <= 0:
            raise ValueError("context_length must be positive.")

        history = context_df.copy()
        if timestamp_column not in history.columns:
            raise ValueError(f"Expected column '{timestamp_column}' in context dataframe.")

        history[timestamp_column] = pd.to_datetime(history[timestamp_column], utc=True, errors="coerce")

        if dropna:
            required_cols = [timestamp_column, *target_columns]
            history = history.dropna(subset=required_cols)

        history = history.sort_values(timestamp_column).reset_index(drop=True)
        timestamps = history[timestamp_column]
        freq = pd.infer_freq(timestamps)
        if freq is None and len(timestamps) > 1:
            deltas = timestamps.diff().dropna()
            try:
                freq_delta = deltas.value_counts().idxmax()
            except Exception:
                freq_delta = deltas.median()
            if pd.isna(freq_delta) or freq_delta <= pd.Timedelta(0):
                freq_delta = pd.Timedelta(days=1)
            full_index = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq=freq_delta, tz="UTC")
            history = (
                history.set_index(timestamp_column)
                .reindex(full_index)
                .ffill()
                .reset_index()
                .rename(columns={"index": timestamp_column})
            )
            timestamps = history[timestamp_column]

        if history.empty:
            raise ValueError("Context dataframe contains no usable rows after preprocessing.")

        symbol_value = _normalize_symbol(symbol, history, id_column)
        history[id_column] = symbol_value

        effective_context = min(context_length, len(history))
        if effective_context < context_length:
            logger.debug(
                "Requested context_length=%d but trimmed to %d rows based on history availability.",
                context_length,
                effective_context,
            )

        trimmed_history = history.iloc[-effective_context:].copy()

        covariates = [
            col
            for col in (known_future_covariates or [])
            if col not in target_columns and col in trimmed_history.columns
        ]

        context_columns = [id_column, timestamp_column, *target_columns, *covariates]
        context_payload = trimmed_history[context_columns].reset_index(drop=True)
        for column in target_columns:
            context_payload[column] = context_payload[column].astype("float32")

        future_payload: Optional[pd.DataFrame] = None
        if future_covariates is not None:
            future_working = future_covariates.copy()
            if timestamp_column not in future_working.columns:
                raise ValueError(f"Expected column '{timestamp_column}' in future covariates.")
            future_working[timestamp_column] = pd.to_datetime(
                future_working[timestamp_column], utc=True, errors="coerce"
            )
            future_working = future_working.dropna(subset=[timestamp_column])
            future_working[id_column] = symbol_value
            keep_cols = [id_column, timestamp_column, *covariates]
            future_payload = future_working[keep_cols].reset_index(drop=True)

        if holdout_df is not None:
            actual = holdout_df.copy()
            if timestamp_column not in actual.columns:
                raise ValueError(f"Expected column '{timestamp_column}' in holdout dataframe.")
            actual[timestamp_column] = pd.to_datetime(actual[timestamp_column], utc=True, errors="coerce")
            actual = actual.dropna(subset=[timestamp_column])
            actual = actual.sort_values(timestamp_column).reset_index(drop=True)
            available_targets = [col for col in target_columns if col in actual.columns]
            missing = set(target_columns) - set(available_targets)
            if missing:
                raise ValueError(f"Holdout dataframe missing target columns: {sorted(missing)}")
            actual_payload = actual[[timestamp_column, *target_columns]].reset_index(drop=True)
            actual_payload = actual_payload.set_index(timestamp_column)
            for column in target_columns:
                actual_payload[column] = actual_payload[column].astype("float32")
        else:
            actual_payload = Chronos2OHLCWrapper._empty_actual_frame(timestamp_column, target_columns)

        return Chronos2PreparedPanel(
            symbol=symbol_value,
            context_df=context_payload,
            future_df=future_payload,
            actual_df=actual_payload,
            context_length=effective_context,
            prediction_length=prediction_length,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_columns=tuple(target_columns),
        )

    def predict_ohlc(
        self,
        context_df: pd.DataFrame,
        *,
        symbol: Optional[str] = None,
        prediction_length: int,
        context_length: Optional[int] = None,
        quantile_levels: Optional[Sequence[float]] = None,
        known_future_covariates: Optional[Sequence[str]] = None,
        evaluation_df: Optional[pd.DataFrame] = None,
        future_covariates: Optional[pd.DataFrame] = None,
        batch_size: Optional[int] = None,
        predict_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Chronos2PredictionBatch:
        """
        Generate Chronos2 forecasts for the requested dataframe & symbol.
        """

        if self.pipeline is None:
            raise RuntimeError("Chronos2 pipeline has been unloaded.")

        resolved_symbol = _normalize_symbol(symbol, context_df, self.id_column)
        context_for_model = context_df
        applied_aug: Optional[AppliedAugmentation] = None
        if self._preaug_selector is not None:
            context_for_model, applied_aug = self._maybe_apply_preaugmentation(resolved_symbol, context_df)

        actual_context_length = context_length or self.default_context_length
        panel = self.build_panel(
            context_for_model,
            holdout_df=evaluation_df,
            future_covariates=future_covariates,
            symbol=symbol,
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
            target_columns=self.target_columns,
            prediction_length=prediction_length,
            context_length=actual_context_length,
            known_future_covariates=known_future_covariates,
        )

        quantiles = tuple(quantile_levels or self.quantile_levels)
        if 0.5 not in quantiles:
            quantiles = tuple(sorted((*quantiles, 0.5)))
        quantile_columns = [_quantile_column(level) for level in quantiles]

        predict_options: Dict[str, Any] = dict(predict_kwargs or {})
        effective_batch_size = int(batch_size or self.default_batch_size)
        raw_predictions = self.pipeline.predict_df(
            panel.context_df,
            future_df=panel.future_df,
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
            target=list(self.target_columns),
            prediction_length=panel.prediction_length,
            quantile_levels=list(quantiles),
            batch_size=effective_batch_size,
            **predict_options,
        )

        if "target_name" not in raw_predictions.columns:
            raise RuntimeError("Chronos2 predict_df output is missing the 'target_name' column.")

        quantile_frames: QuantileFrameMap = {}
        for level, column_name in zip(quantiles, quantile_columns):
            if column_name not in raw_predictions.columns:
                raise RuntimeError(f"Chronos2 output missing '{column_name}' column for quantile={level}.")
            pivot = (
                raw_predictions.pivot(
                    index=self.timestamp_column,
                    columns="target_name",
                    values=column_name,
                )
                .sort_index()
            )
            quantile_frames[level] = pivot.astype("float32")

        if applied_aug is not None:
            quantile_frames, raw_predictions = self._apply_inverse_augmentation(
                panel,
                quantile_frames,
                raw_predictions,
                quantiles,
                quantile_columns,
                applied_aug,
            )

        return Chronos2PredictionBatch(
            panel=panel,
            raw_dataframe=raw_predictions,
            quantile_frames=quantile_frames,
            applied_augmentation=applied_aug.choice.strategy if applied_aug else None,
        )


__all__ = [
    "Chronos2OHLCWrapper",
    "Chronos2PreparedPanel",
    "Chronos2PredictionBatch",
    "AppliedAugmentation",
    "DEFAULT_TARGET_COLUMNS",
    "DEFAULT_QUANTILE_LEVELS",
]
