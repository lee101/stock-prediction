"""
Chronos2 forecasting helper that standardises OHLC panel preparation and prediction.

The goal is to mirror the ergonomics of ``src/models/toto_wrapper.py`` while leveraging
the pandas-first API that ships with Chronos 2. The wrapper focuses on:

* Preparing multi-target OHLC panels with long (8k+) context windows
* Generating quantile forecasts via ``Chronos2Pipeline.predict_df``
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - surface chronos version info when available
    import chronos as _chronos_pkg
except Exception:  # pragma: no cover
    _chronos_pkg = None  # type: ignore

from chronos import Chronos2Pipeline as _Chronos2Pipeline
from preaug_sweeps.augmentations import BaseAugmentation
try:  # pragma: no cover - backward compatibility with pre-helper snapshots
    from src.cache_utils import find_hf_snapshot_dir
except ImportError:  # pragma: no cover
    def find_hf_snapshot_dir(*_args, **_kwargs):
        return None
from src.gpu_utils import should_offload_to_cpu as gpu_should_offload_to_cpu
from src.preaug import PreAugmentationChoice, PreAugmentationSelector
from .model_cache import ModelCacheError, ModelCacheManager, dtype_to_token

logger = logging.getLogger(__name__)

DEFAULT_TARGET_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close")
DEFAULT_QUANTILE_LEVELS: Tuple[float, ...] = (0.1, 0.5, 0.9)
_BOOL_TRUE = {"1", "true", "yes", "on"}
_DEFAULT_MODEL_ID = "amazon/chronos-2"
_DEFAULT_PREDICTION_CACHE_SIZE = 64
_DEFAULT_PREDICTION_CACHE_DECIMALS = 8


def _safe_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off", "never"}:
        return False
    if normalized in {"1", "true", "yes", "on", "prefer"}:
        return True
    return default


def _round_float_columns(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    if decimals < 0 or df.empty:
        return df
    float_cols = df.select_dtypes(include=["float16", "float32", "float64"])
    if float_cols.empty:
        return df
    rounded = df.copy()
    rounded[float_cols.columns] = float_cols.round(decimals)
    return rounded


def _hash_dataframe_for_cache(df: Optional[pd.DataFrame], decimals: int) -> str:
    if df is None:
        return "none"
    if df.empty:
        return f"empty:{df.shape}"
    rounded = _round_float_columns(df, decimals)
    hashed = hash_pandas_object(rounded, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _serialize_for_cache(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_for_cache(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_for_cache(val) for key, val in sorted(value.items())}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _stable_options_payload(options: Mapping[str, Any]) -> str:
    normalized = {str(key): _serialize_for_cache(val) for key, val in sorted(options.items())}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _chronos_version() -> str:
    if _chronos_pkg is not None:  # pragma: no branch - lightweight accessor
        return getattr(_chronos_pkg, "__version__", "unknown")
    return "unknown"


def _path_contains_config(path: Path) -> bool:
    try:
        return path.is_dir() and (path / "config.json").exists()
    except OSError:
        return False


def _normalize_aliases(raw_aliases: Optional[str]) -> set[str]:
    if not raw_aliases:
        return {"chronos2"}
    aliases: set[str] = set()
    for alias in raw_aliases.split(","):
        normalized = alias.strip().lower()
        if normalized:
            aliases.add(normalized)
    if not aliases:
        aliases.add("chronos2")
    return aliases


def _resolve_model_source(model_id: Optional[str]) -> str:
    requested = (os.getenv("CHRONOS2_MODEL_ID_OVERRIDE") or (model_id or _DEFAULT_MODEL_ID)).strip()
    if not requested:
        requested = _DEFAULT_MODEL_ID

    candidate_path = Path(requested)
    if _path_contains_config(candidate_path):
        return str(candidate_path)

    aliases = _normalize_aliases(os.getenv("CHRONOS2_MODEL_ALIASES"))
    if requested.lower() in aliases:
        local_override = os.getenv("CHRONOS2_LOCAL_MODEL_DIR")
        if local_override:
            local_path = Path(local_override).expanduser()
            if _path_contains_config(local_path):
                return str(local_path)

        snapshot_dir = find_hf_snapshot_dir(_DEFAULT_MODEL_ID, logger=logger)
        if snapshot_dir is not None:
            return str(snapshot_dir)

        logger.warning(
            "Chronos2 alias '%s' requested but no cached snapshot of %s was found; "
            "falling back to the canonical repo identifier.",
            requested,
            _DEFAULT_MODEL_ID,
        )
        return _DEFAULT_MODEL_ID

    return requested


def _normalize_frequency(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"daily", "hourly"}:
        return normalized
    return None


def _default_preaug_dirs(frequency: Optional[str]) -> Tuple[Path, ...]:
    base_dirs = (
        Path("preaugstrategies") / "chronos2",
        Path("preaugstrategies") / "best",
    )
    if not frequency:
        return base_dirs
    freq_dirs = (
        Path("preaugstrategies") / "chronos2" / frequency,
        Path("preaugstrategies") / "best" / frequency,
    )
    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in (*freq_dirs, *base_dirs):
        if candidate in seen:
            continue
        ordered.append(candidate)
        seen.add(candidate)
    return tuple(ordered)


def _require_chronos2_pipeline() -> type:
    """Return the Chronos2Pipeline class or raise a descriptive error."""

    if _Chronos2Pipeline is None:
        raise RuntimeError(
            "chronos>=2.0 is unavailable; install `chronos-forecasting>=2.0` to enable Chronos2Pipeline."
        )
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
    applied_choice: Optional[PreAugmentationChoice] = None


@dataclass
class _CachedPrediction:
    raw_dataframe: pd.DataFrame
    quantile_frames: QuantileFrameMap
    applied_augmentation: Optional[str]
    applied_choice: Optional[PreAugmentationChoice]
    panel: Chronos2PreparedPanel

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
        prediction_cache_enabled: Optional[bool] = None,
        prediction_cache_size: Optional[int] = None,
        prediction_cache_decimals: Optional[int] = None,
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
        self._eager_model = getattr(pipeline, "model", None)
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

        if preaugmentation_dirs is None:
            freq = _normalize_frequency(os.getenv("CHRONOS2_FREQUENCY"))
            default_preaug_dirs = _default_preaug_dirs(freq)
        else:
            default_preaug_dirs = tuple(Path(d) for d in preaugmentation_dirs if d)
        dirs_list = [Path(d) for d in default_preaug_dirs if d]
        self._preaug_selector: Optional[PreAugmentationSelector] = (
            PreAugmentationSelector(dirs_list) if dirs_list else None
        )

        if prediction_cache_enabled is None:
            self._prediction_cache_enabled = _safe_bool(os.getenv("CHRONOS2_PREDICTION_CACHE"), default=True)
        else:
            self._prediction_cache_enabled = bool(prediction_cache_enabled)
        default_cache_size = int(
            os.getenv("CHRONOS2_PREDICTION_CACHE_SIZE", str(_DEFAULT_PREDICTION_CACHE_SIZE))
        )
        default_decimals = int(
            os.getenv("CHRONOS2_CACHE_DECIMALS", str(_DEFAULT_PREDICTION_CACHE_DECIMALS))
        )
        if prediction_cache_size is None:
            self._prediction_cache_capacity = max(0, default_cache_size)
        else:
            self._prediction_cache_capacity = max(0, int(prediction_cache_size))
        if prediction_cache_decimals is None:
            self._prediction_cache_decimals = max(0, default_decimals)
        else:
            self._prediction_cache_decimals = max(0, int(prediction_cache_decimals))
        self._prediction_cache: "OrderedDict[str, _CachedPrediction]" = OrderedDict()
        self._prediction_cache_hits = 0
        self._prediction_cache_misses = 0

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
        prediction_cache_enabled: Optional[bool] = None,
        prediction_cache_size: Optional[int] = None,
        prediction_cache_decimals: Optional[int] = None,
        cache_policy: str = "prefer",
        force_refresh: bool = False,
        cache_manager: Optional[ModelCacheManager] = None,
        **kwargs,
    ) -> "Chronos2OHLCWrapper":
        """
        Instantiate Chronos2 via Hugging Face and wrap it with OHLC helpers.
        """

        policy = (cache_policy or "prefer").strip().lower()
        if policy not in {"prefer", "never", "only"}:
            raise ValueError("Chronos2 cache_policy must be 'prefer', 'never', or 'only'.")

        pipeline_cls = _require_chronos2_pipeline()
        resolved_model_id = _resolve_model_source(model_id)
        if resolved_model_id != model_id:
            logger.info("Resolved Chronos2 model_id '%s' to '%s'", model_id, resolved_model_id)

        manager = cache_manager or ModelCacheManager("chronos2")
        dtype_token = dtype_to_token(torch_dtype if torch_dtype is not None else None)
        torch_version = getattr(torch, "__version__", "unknown") if torch is not None else "unknown"
        metadata_requirements = {
            "model_id": resolved_model_id,
            "dtype": dtype_token,
            "compile_mode": (compile_mode or "none"),
            "compile_backend": (compile_backend or "none"),
            "torch_version": torch_version,
            "chronos_version": _chronos_version(),
        }
        use_cache = policy != "never"
        loaded_from_cache = False

        with manager.compilation_env(resolved_model_id, dtype_token):
            metadata = manager.load_metadata(resolved_model_id, dtype_token) if use_cache else None
            pipeline: _Chronos2Pipeline
            if (
                use_cache
                and not force_refresh
                and metadata
                and manager.metadata_matches(metadata, metadata_requirements)
            ):
                cache_path = manager.load_pretrained_path(resolved_model_id, dtype_token)
                if cache_path is not None:
                    try:
                        pipeline = pipeline_cls.from_pretrained(str(cache_path), device_map=device_map, **kwargs)
                        loaded_from_cache = True
                        logger.info(
                            "Loaded Chronos2 model '%s' (%s) from compiled cache.",
                            resolved_model_id,
                            dtype_token,
                        )
                    except Exception as exc:  # pragma: no cover - unexpected load failures
                        loaded_from_cache = False
                        logger.warning("Failed to load Chronos2 cache from %s: %s", cache_path, exc)
            if policy == "only" and not loaded_from_cache:
                raise RuntimeError(
                    f"Compiled Chronos2 cache unavailable for model '{resolved_model_id}' "
                    f"and dtype '{dtype_token}'. Run cache warmup utilities first."
                )

            if not loaded_from_cache:
                pipeline = pipeline_cls.from_pretrained(resolved_model_id, device_map=device_map, **kwargs)
                logger.info(
                    "Loaded Chronos2 model '%s' from source (cache_policy=%s).",
                    resolved_model_id,
                    policy,
                )

            device_hint = device_map if isinstance(device_map, str) else "cuda"
            wrapper = cls(
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
            prediction_cache_enabled=prediction_cache_enabled,
            prediction_cache_size=prediction_cache_size,
            prediction_cache_decimals=prediction_cache_decimals,
        )

            if use_cache and (force_refresh or not loaded_from_cache):
                model_obj = getattr(wrapper, "pipeline", None)
                if model_obj is not None:
                    model_attr = getattr(model_obj, "model", None)
                else:
                    model_attr = None
                if model_attr is not None:
                    metadata_payload = {
                        **metadata_requirements,
                        "device_map": str(device_map),
                        "cache_policy": policy,
                    }
                    try:
                        manager.persist_model_state(
                            model_id=resolved_model_id,
                            dtype_token=dtype_token,
                            model=model_attr,
                            metadata=metadata_payload,
                            force=force_refresh,
                        )
                    except ModelCacheError as exc:
                        logger.warning(
                            "Failed to persist Chronos2 cache for model '%s': %s",
                            resolved_model_id,
                            exc,
                        )
                else:
                    logger.debug("Chronos2 pipeline missing model attribute; skipping cache persistence.")

        return wrapper

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

    def _disable_torch_compile(self, reason: str, error: Optional[BaseException] = None) -> None:
        if not self._torch_compile_success:
            return
        self._torch_compile_success = False
        self._torch_compile_enabled = False
        if self.pipeline is not None and self._eager_model is not None:
            try:
                self.pipeline.model = self._eager_model
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to restore eager Chronos2 model: %s", exc)
        logger.warning("Chronos2 torch.compile disabled (%s): %s", reason, error)

    def _call_with_compile_fallback(self, func: Callable[[], _T], context: str) -> _T:
        try:
            return func()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if not self._torch_compile_success:
                raise
            self._disable_torch_compile(f"runtime failure during {context}", exc)
            return func()

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
        values_adjusted = False
        for column in transformed.columns:
            if column not in augmented_context.columns:
                continue
            series = transformed[column]
            target_dtype = augmented_context[column].dtype
            if hasattr(series, "to_numpy"):
                values = series.to_numpy(dtype=target_dtype, copy=False)
            else:
                values = np.asarray(series, dtype=target_dtype)

            # SAFETY: Prevent very small values that cause PyTorch compilation errors
            # PyTorch's sympy evaluation in torch._inductor can fail with very small
            # values (both positive and negative) during symbolic math operations.
            # Error example: AssertionError: -834735604272579/1000000000000000 ≈ -0.0008
            #
            # This happens during memory coalescing analysis in torch.compile() when
            # sympy's is_constant() check evaluates expressions with values close to zero.
            # Clamping these values to exactly 0.0 avoids numerical instability.
            #
            # Threshold: 0.001 (1e-3) - chosen to catch the error value (-0.0008) while
            # preserving meaningful signal in augmented data.
            epsilon = 1e-3  # 0.001
            very_small_mask = np.abs(values) < epsilon
            if very_small_mask.any():
                n_adjusted = very_small_mask.sum()
                if not values_adjusted:  # Log only once per symbol
                    logger.debug(
                        "Clamping %d very small values (abs < %.3f) in column '%s' for %s "
                        "to prevent PyTorch compilation issues",
                        n_adjusted,
                        epsilon,
                        column,
                        symbol,
                    )
                    values_adjusted = True
                values = values.copy()
                values[very_small_mask] = 0.0

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
        cache_key = None
        cached_batch: Optional[Chronos2PredictionBatch] = None
        if self._prediction_cache_enabled and self._prediction_cache_capacity > 0:
            cache_key = self._build_prediction_cache_key(
                context_df=panel.context_df,
                future_df=panel.future_df,
                symbol=resolved_symbol,
                quantiles=quantiles,
                prediction_length=panel.prediction_length,
                predict_options=predict_options,
            )
            cached_batch = self._prediction_cache_lookup(cache_key)
            if cached_batch is not None:
                return cached_batch

        def _predict_call() -> pd.DataFrame:
            return self.pipeline.predict_df(
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

        raw_predictions = self._call_with_compile_fallback(_predict_call, "predict_df")

        if "target_name" not in raw_predictions.columns:
            raise RuntimeError("Chronos2 predict_df output is missing the 'target_name' column.")

        quantile_frames: QuantileFrameMap = {}
        for level, column_name in zip(quantiles, quantile_columns):
            if column_name not in raw_predictions.columns:
                raise RuntimeError(f"Chronos2 output missing '{column_name}' column for quantile={level}.")
            pivot = raw_predictions.pivot(
                index=self.timestamp_column,
                columns="target_name",
                values=column_name,
            ).sort_index()
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

        batch_result = Chronos2PredictionBatch(
            panel=panel,
            raw_dataframe=raw_predictions,
            quantile_frames=quantile_frames,
            applied_augmentation=applied_aug.choice.strategy if applied_aug else None,
            applied_choice=applied_aug.choice if applied_aug else None,
        )
        if cache_key is not None:
            self._prediction_cache_store(cache_key, batch_result)
        return batch_result

    # ------------------------------------------------------------------ #
    # Prediction cache helpers
    # ------------------------------------------------------------------ #
    def _build_prediction_cache_key(
        self,
        *,
        context_df: pd.DataFrame,
        future_df: Optional[pd.DataFrame],
        symbol: Optional[str],
        quantiles: Tuple[float, ...],
        prediction_length: int,
        predict_options: Mapping[str, Any],
    ) -> Optional[str]:
        if not self._prediction_cache_enabled or self._prediction_cache_capacity <= 0:
            return None
        payload = {
            "context": _hash_dataframe_for_cache(context_df, self._prediction_cache_decimals),
            "future": _hash_dataframe_for_cache(future_df, self._prediction_cache_decimals),
            "symbol": symbol or "",
            "quantiles": ",".join(format(level, ".6f") for level in quantiles),
            "prediction_length": int(prediction_length),
            "targets": "|".join(self.target_columns),
            "options": _stable_options_payload(predict_options),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _prediction_cache_lookup(self, cache_key: Optional[str]) -> Optional[Chronos2PredictionBatch]:
        if cache_key is None:
            return None
        cached = self._prediction_cache.get(cache_key)
        if cached is None:
            self._prediction_cache_misses += 1
            return None
        self._prediction_cache_hits += 1
        self._prediction_cache.move_to_end(cache_key)
        return self._clone_cached_prediction(cached)

    def _prediction_cache_store(self, cache_key: str, batch: Chronos2PredictionBatch) -> None:
        if not self._prediction_cache_enabled or self._prediction_cache_capacity <= 0:
            return
        snapshot = self._snapshot_prediction(batch)
        self._prediction_cache[cache_key] = snapshot
        self._prediction_cache.move_to_end(cache_key)
        while len(self._prediction_cache) > self._prediction_cache_capacity:
            self._prediction_cache.popitem(last=False)

    def _snapshot_prediction(self, batch: Chronos2PredictionBatch) -> _CachedPrediction:
        quantiles_copy: QuantileFrameMap = {
            level: frame.copy(deep=True) for level, frame in batch.quantile_frames.items()
        }
        return _CachedPrediction(
            raw_dataframe=batch.raw_dataframe.copy(deep=True),
            quantile_frames=quantiles_copy,
            applied_augmentation=batch.applied_augmentation,
            applied_choice=batch.applied_choice,
            panel=batch.panel,
        )

    def _clone_cached_prediction(self, cached: _CachedPrediction) -> Chronos2PredictionBatch:
        quantiles_copy: QuantileFrameMap = {
            level: frame.copy(deep=True) for level, frame in cached.quantile_frames.items()
        }
        return Chronos2PredictionBatch(
            panel=cached.panel,
            raw_dataframe=cached.raw_dataframe.copy(deep=True),
            quantile_frames=quantiles_copy,
            applied_augmentation=cached.applied_augmentation,
            applied_choice=cached.applied_choice,
        )

    def prediction_cache_stats(self) -> Dict[str, Any]:
        total = self._prediction_cache_hits + self._prediction_cache_misses
        hit_rate = (self._prediction_cache_hits / total * 100.0) if total else 0.0
        return {
            "enabled": bool(self._prediction_cache_enabled and self._prediction_cache_capacity > 0),
            "entries": len(self._prediction_cache),
            "capacity": self._prediction_cache_capacity,
            "hits": self._prediction_cache_hits,
            "misses": self._prediction_cache_misses,
            "hit_rate_percent": hit_rate,
            "decimals": self._prediction_cache_decimals,
        }


__all__ = [
    "Chronos2OHLCWrapper",
    "Chronos2PreparedPanel",
    "Chronos2PredictionBatch",
    "AppliedAugmentation",
    "DEFAULT_TARGET_COLUMNS",
    "DEFAULT_QUANTILE_LEVELS",
]
