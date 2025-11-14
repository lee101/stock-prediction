from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence

from src.gpu_utils import should_offload_to_cpu as gpu_should_offload_to_cpu

from .model_cache import ModelCacheError, ModelCacheManager, dtype_to_token

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KRONOS_CANDIDATES = [
    _REPO_ROOT / "external" / "kronos",
    _REPO_ROOT / "external" / "kronos" / "model",
]
for _path in _KRONOS_CANDIDATES:
    if _path.exists():
        path_str = str(_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

logger = logging.getLogger(__name__)


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if torch is None:
        return False
    cuda_mod = getattr(torch, "cuda", None)
    oom_error = getattr(cuda_mod, "OutOfMemoryError", None)
    if oom_error is not None and isinstance(exc, oom_error):
        return True
    return "out of memory" in str(exc).lower()


def _optional_import(module_name: str) -> ModuleType | None:
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        return None


torch: ModuleType | None = _optional_import("torch")
np: ModuleType | None = _optional_import("numpy")
pd: ModuleType | None = _optional_import("pandas")


def setup_kronos_wrapper_imports(
    *,
    torch_module: ModuleType | None = None,
    numpy_module: ModuleType | None = None,
    pandas_module: ModuleType | None = None,
    **_: Any,
) -> None:
    global torch, np, pd
    if torch_module is not None:
        torch = torch_module
    if numpy_module is not None:
        np = numpy_module
    if pandas_module is not None:
        pd = pandas_module


def _require_torch() -> ModuleType:
    global torch
    if torch is not None:
        return torch
    try:
        module = import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Torch is unavailable. Call setup_kronos_wrapper_imports before use.") from exc
    torch = module
    return module


def _require_numpy() -> ModuleType:
    global np
    if np is not None:
        return np
    try:
        module = import_module("numpy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is unavailable. Call setup_kronos_wrapper_imports before use.") from exc
    np = module
    return module


def _require_pandas() -> ModuleType:
    global pd
    if pd is not None:
        return pd
    try:
        module = import_module("pandas")
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is unavailable. Call setup_kronos_wrapper_imports before use.") from exc
    pd = module
    return module


@dataclass(frozen=True)
class KronosForecastResult:
    """Container for Kronos forecasts."""

    absolute: np.ndarray
    percent: np.ndarray
    timestamps: pd.Index


@dataclass(frozen=True)
class _SeriesPayload:
    feature_frame: pd.DataFrame
    history_series: pd.Series
    future_series: pd.Series
    future_index: pd.Index
    last_values: Dict[str, float]


class KronosForecastingWrapper:
    """
    Thin adapter around the external Kronos predictor to match the project API.

    The wrapper lazily initialises the heavyweight Kronos components so callers can
    construct it during module import without incurring GPU/IO cost. Predictions are
    returned as per-column ``KronosForecastResult`` objects containing both absolute
    price levels and step-wise percentage returns.
    """

    def __init__(
        self,
        *,
        model_name: str,
        tokenizer_name: str,
        device: str = "cuda:0",
        max_context: int = 512,
        clip: float = 5.0,
        temperature: float = 0.75,
        top_p: float = 0.9,
        top_k: int = 0,
        sample_count: int = 8,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        prefer_fp32: bool = False,
        compile: bool = False,
        compile_mode: str = "max-autotune",
        compile_backend: Optional[str] = "inductor",
    ) -> None:
        if torch is None or np is None or pd is None:
            raise RuntimeError(
                "Torch, NumPy, and pandas must be configured via setup_kronos_wrapper_imports before instantiating KronosForecastingWrapper."
            )

        device_display = str(device)
        normalized_device = device_display.strip().lower()
        is_cuda_request = normalized_device.startswith("cuda")
        is_cpu_request = normalized_device == "cpu" or normalized_device.startswith("cpu:")
        if not (is_cuda_request or is_cpu_request):
            raise RuntimeError(
                f"KronosForecastingWrapper requires a CUDA or CPU device; received {device_display!r}."
            )
        if is_cuda_request:
            cuda_mod = getattr(torch, "cuda", None)
            is_available = bool(getattr(cuda_mod, "is_available", lambda: False)()) if cuda_mod is not None else False
            if not is_available:
                raise RuntimeError(
                    "CUDA is unavailable. KronosForecastingWrapper requires a CUDA-capable PyTorch installation when using a CUDA device."
                )

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.requested_device = normalized_device
        self._requested_device_display = device_display
        self.max_context = max_context
        self.clip = clip
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.sample_count = sample_count
        self.cache_dir = cache_dir
        self.verbose = verbose
        self._prefer_fp32 = bool(prefer_fp32)
        self.compile = bool(compile)
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend

        self._device = normalized_device
        self._predictor = None
        self._preferred_dtype = self._compute_preferred_dtype(normalized_device, prefer_fp32=self._prefer_fp32)
        self._adaptive_sample_count: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict_series(
        self,
        *,
        data: pd.DataFrame,
        timestamp_col: str,
        columns: Sequence[str],
        pred_len: int,
        lookback: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        sample_count: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> Dict[str, KronosForecastResult]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not columns:
            raise ValueError("columns must contain at least one entry.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")

        payload = self._prepare_series_payloads(
            data_frames=[data],
            timestamp_col=timestamp_col,
            pred_len=pred_len,
            lookback=lookback,
        )[0]

        (
            effective_temperature,
            effective_top_p,
            effective_top_k,
            effective_samples,
            effective_verbose,
        ) = self._resolve_sampling_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            sample_count=sample_count,
            verbose=verbose,
        )

        current_samples = effective_samples
        oom_attempts = 0
        while True:
            predictor = self._ensure_predictor()
            try:
                forecast_df = predictor.predict(
                    payload.feature_frame,
                    x_timestamp=payload.history_series,
                    y_timestamp=payload.future_series,
                    pred_len=int(pred_len),
                    T=effective_temperature,
                    top_k=effective_top_k,
                    top_p=effective_top_p,
                    sample_count=current_samples,
                    verbose=effective_verbose,
                )
                break
            except RuntimeError as exc:
                if not _is_cuda_oom_error(exc) or not self._device.startswith("cuda"):
                    raise
                next_samples = self._next_sample_count_after_oom(current_samples)
                self._handle_cuda_oom()
                if next_samples is None:
                    logger.error(
                        "Kronos GPU inference ran out of memory on %s with sample_count=%d; no smaller retry possible.",
                        self._device,
                        current_samples,
                    )
                    raise RuntimeError(
                        f"Kronos GPU inference ran out of memory on device {self._device}. Reduce sampling requirements or provision a larger GPU."
                    ) from exc
                oom_attempts += 1
                if oom_attempts == 1:
                    logger.warning(
                        "Kronos GPU inference ran out of memory on %s with sample_count=%d; retrying with %d.",
                        self._device,
                        current_samples,
                        next_samples,
                    )
                else:
                    logger.warning(
                        "Kronos GPU inference still OOM on %s; reducing sample_count from %d to %d (attempt %d).",
                        self._device,
                        current_samples,
                        next_samples,
                        oom_attempts,
                    )
                self._register_adaptive_sample_limit(next_samples)
                current_samples = next_samples
                continue

        if not isinstance(forecast_df, pd.DataFrame):
            raise RuntimeError("Kronos predictor returned an unexpected result type.")

        if oom_attempts > 0 and current_samples < effective_samples:
            logger.info(
                "Kronos inference recovered after OOM on %s using sample_count=%d (requested %d).",
                self._device,
                current_samples,
                effective_samples,
            )

        return self._assemble_results(payload, forecast_df, columns)

    def predict_series_batch(
        self,
        *,
        data_frames: Sequence[pd.DataFrame],
        timestamp_col: str,
        columns: Sequence[str],
        pred_len: int,
        lookback: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        sample_count: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> List[Dict[str, KronosForecastResult]]:
        if not data_frames:
            raise ValueError("data_frames must contain at least one dataframe.")
        if not columns:
            raise ValueError("columns must contain at least one entry.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")

        payloads = self._prepare_series_payloads(
            data_frames=data_frames,
            timestamp_col=timestamp_col,
            pred_len=pred_len,
            lookback=lookback,
        )

        (
            effective_temperature,
            effective_top_p,
            effective_top_k,
            effective_samples,
            effective_verbose,
        ) = self._resolve_sampling_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            sample_count=sample_count,
            verbose=verbose,
        )

        current_samples = effective_samples
        oom_attempts = 0
        while True:
            predictor = self._ensure_predictor()
            batch_predict = getattr(predictor, "predict_batch", None)
            if batch_predict is None:
                raise AttributeError("Kronos predictor does not expose 'predict_batch'. Update the Kronos package.")
            try:
                forecast_list = batch_predict(
                    [payload.feature_frame for payload in payloads],
                    [payload.history_series for payload in payloads],
                    [payload.future_series for payload in payloads],
                    pred_len=int(pred_len),
                    T=effective_temperature,
                    top_k=effective_top_k,
                    top_p=effective_top_p,
                    sample_count=current_samples,
                    verbose=effective_verbose,
                )
                break
            except RuntimeError as exc:
                if not _is_cuda_oom_error(exc) or not self._device.startswith("cuda"):
                    raise
                next_samples = self._next_sample_count_after_oom(current_samples)
                self._handle_cuda_oom()
                if next_samples is None:
                    logger.error(
                        "Kronos GPU batch inference ran out of memory on %s with sample_count=%d; no smaller retry possible.",
                        self._device,
                        current_samples,
                    )
                    raise RuntimeError(
                        f"Kronos GPU inference ran out of memory on device {self._device}. Reduce sampling requirements or provision a larger GPU."
                    ) from exc
                oom_attempts += 1
                if oom_attempts == 1:
                    logger.warning(
                        "Kronos GPU batch inference ran out of memory on %s with sample_count=%d; retrying with %d.",
                        self._device,
                        current_samples,
                        next_samples,
                    )
                else:
                    logger.warning(
                        "Kronos GPU batch inference still OOM on %s; reducing sample_count from %d to %d (attempt %d).",
                        self._device,
                        current_samples,
                        next_samples,
                        oom_attempts,
                    )
                self._register_adaptive_sample_limit(next_samples)
                current_samples = next_samples
                continue

        if not isinstance(forecast_list, (list, tuple)):
            raise RuntimeError("Kronos batch predictor returned an unexpected result type.")
        if len(forecast_list) != len(payloads):
            raise RuntimeError("Kronos batch predictor returned a result with mismatched length.")

        if oom_attempts > 0 and current_samples < effective_samples:
            logger.info(
                "Kronos batch inference recovered after OOM on %s using sample_count=%d (requested %d).",
                self._device,
                current_samples,
                effective_samples,
            )

        results: List[Dict[str, KronosForecastResult]] = []
        for payload, forecast_df in zip(payloads, forecast_list):
            if not isinstance(forecast_df, pd.DataFrame):
                raise RuntimeError("Kronos batch predictor returned a non-DataFrame entry.")
            results.append(self._assemble_results(payload, forecast_df, columns))
        return results

    def _resolve_sampling_params(
        self,
        *,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        sample_count: Optional[int],
        verbose: Optional[bool],
    ) -> tuple[float, float, int, int, bool]:
        effective_temperature = float(temperature if temperature is not None else self.temperature)
        effective_top_p = float(top_p if top_p is not None else self.top_p)
        effective_top_k = int(top_k if top_k is not None else self.top_k)
        base_samples = int(sample_count if sample_count is not None else self.sample_count)
        adaptive_limit = self._adaptive_sample_count
        if adaptive_limit is not None and adaptive_limit < base_samples:
            base_samples = adaptive_limit
        effective_samples = max(1, base_samples)
        effective_verbose = bool(verbose if verbose is not None else self.verbose)
        return (
            effective_temperature,
            effective_top_p,
            effective_top_k,
            effective_samples,
            effective_verbose,
        )

    def _prepare_series_payloads(
        self,
        *,
        data_frames: Sequence[pd.DataFrame],
        timestamp_col: str,
        pred_len: int,
        lookback: Optional[int],
    ) -> List[_SeriesPayload]:
        payloads: List[_SeriesPayload] = []
        for idx, frame in enumerate(data_frames):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"data_frames[{idx}] must be a pandas DataFrame.")
            if timestamp_col not in frame.columns:
                raise KeyError(f"{timestamp_col!r} column not present in dataframe index {idx}.")

            working = frame.copy()
            working = working.dropna(subset=[timestamp_col])
            if working.empty:
                raise ValueError(f"dataframe at index {idx} is empty after dropping NaN timestamps.")

            timestamp_series = pd.to_datetime(working[timestamp_col], utc=True, errors="coerce")
            timestamp_series = timestamp_series.dropna()
            if timestamp_series.empty:
                raise ValueError(f"No valid timestamps available for Kronos forecasting (index {idx}).")

            working = working.loc[timestamp_series.index]
            timestamps = pd.DatetimeIndex(timestamp_series)
            if timestamps.tz is None:
                timestamps = timestamps.tz_localize("UTC")
            timestamps = timestamps.tz_convert(None)

            if timestamps.duplicated().any():
                mask = ~timestamps.duplicated(keep="last")
                duplicate_count = int(np.count_nonzero(~mask))
                logger.debug(
                    "Detected %d duplicate timestamps for Kronos payload; keeping last occurrence.",
                    duplicate_count,
                )
                working = working.iloc[mask]
                timestamps = timestamps[mask]

            if lookback:
                span = int(max(1, lookback))
                if len(working) > span:
                    working = working.iloc[-span:]
                    timestamps = timestamps[-span:]

            feature_frame = self._prepare_feature_frame(working)
            if len(feature_frame) < 2:
                raise ValueError("Insufficient history for Kronos forecasting (need at least 2 rows).")

            future_index = self._build_future_index(timestamps, pred_len)
            history_index = pd.DatetimeIndex(timestamps)
            x_timestamp = pd.Series(history_index)
            y_timestamp = pd.Series(future_index)

            last_values: Dict[str, float] = {}
            for column in feature_frame.columns:
                column_key = str(column).lower()
                last_values[column_key] = float(feature_frame[column_key].iloc[-1])

            payloads.append(
                _SeriesPayload(
                    feature_frame=feature_frame,
                    history_series=x_timestamp,
                    future_series=y_timestamp,
                    future_index=future_index,
                    last_values=last_values,
                )
            )

        return payloads

    def _assemble_results(
        self,
        payload: _SeriesPayload,
        forecast_df: pd.DataFrame,
        columns: Sequence[str],
    ) -> Dict[str, KronosForecastResult]:
        results: Dict[str, KronosForecastResult] = {}
        for column in columns:
            key = str(column)
            lower_key = key.lower()
            if lower_key not in forecast_df.columns:
                raise KeyError(f"Kronos forecast missing column '{key}'.")
            absolute = np.asarray(forecast_df[lower_key], dtype=np.float64)
            previous = payload.last_values.get(lower_key)
            if previous is None:
                raise KeyError(f"No historical baseline available for column '{key}'.")
            percent = self._compute_step_returns(previous=previous, absolute=absolute)
            results[key] = KronosForecastResult(
                absolute=absolute,
                percent=percent,
                timestamps=payload.future_index,
            )
        return results

    def _should_offload_to_cpu(self) -> bool:
        """
        Determine if model should be offloaded to CPU based on GPU capabilities.
        Returns False for high-VRAM GPUs like RTX 5090 where we have enough memory.
        """
        return gpu_should_offload_to_cpu(self._device)

    def unload(self) -> None:
        predictor = self._predictor
        if predictor is None:
            return

        should_offload = self._should_offload_to_cpu()

        try:
            if should_offload and hasattr(predictor.model, "to"):
                predictor.model.to("cpu")
            elif not should_offload:
                logger.debug("Skipping CPU offload for model - sufficient GPU VRAM available")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Failed to move Kronos model to CPU during unload: {exc}")
        try:
            if should_offload and hasattr(predictor.tokenizer, "to"):
                predictor.tokenizer.to("cpu")
            elif not should_offload:
                logger.debug("Skipping CPU offload for tokenizer - sufficient GPU VRAM available")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Failed to move Kronos tokenizer to CPU during unload: {exc}")
        self._predictor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_preferred_dtype(device: str, *, prefer_fp32: bool = False) -> Optional[torch.dtype]:
        if prefer_fp32:
            return None
        if not device.startswith("cuda"):
            return None
        if not torch.cuda.is_available():
            return None
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16  # pragma: no cover - depends on hardware
        return None

    def _ensure_predictor(self, *, device_override: Optional[str] = None):
        override_display: Optional[str] = None
        normalized_override: Optional[str] = None
        if device_override is not None:
            override_display = str(device_override)
            normalized_override = override_display.strip().lower()

        predictor = self._predictor
        if predictor is not None:
            if normalized_override is None or self._device == normalized_override:
                return predictor
            self.unload()
            predictor = None

        original_model_module = sys.modules.get("model")
        stub_module: Optional[types.ModuleType] = None
        try:
            # Kronos expects ``model`` to resolve to the vendor package shipped in
            # ``external/kronos``.  If a legacy ``model`` module has already been
            # imported (e.g. the project-level ``model.py``), temporarily install a
            # stub package that points to the Kronos directory so ``model.module`` can
            # be resolved during the import below.  The original module is restored
            # afterwards to avoid leaking changes into the wider application.
            if original_model_module is None or not hasattr(original_model_module, "__path__"):
                stub_module = types.ModuleType("model")
                stub_module.__path__ = [str(_REPO_ROOT / "external" / "kronos" / "model")]  # type: ignore[attr-defined]
                sys.modules["model"] = stub_module
            from external.kronos.model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - import-time guard
            if stub_module is not None:
                sys.modules.pop("model", None)
            if original_model_module is not None:
                sys.modules["model"] = original_model_module
            raise RuntimeError(
                "Failed to import Kronos components. Ensure the external Kronos package is available."
            ) from exc
        finally:
            if stub_module is not None:
                # Remove the temporary stub and reinstate the legacy module if it existed.
                sys.modules.pop("model", None)
                if original_model_module is not None:
                    sys.modules["model"] = original_model_module

        device_display = override_display or self._requested_device_display
        device = normalized_override or self.requested_device
        normalized = device
        is_cuda_request = normalized.startswith("cuda")
        is_cpu_request = normalized == "cpu" or normalized.startswith("cpu:")
        if not (is_cuda_request or is_cpu_request):
            raise RuntimeError(
                f"KronosForecastingWrapper requires a CUDA or CPU device; received {device_display!r}."
            )
        if is_cuda_request and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. KronosForecastingWrapper cannot honour the requested CUDA device.")
        self._device = normalized

        cache_manager = ModelCacheManager("kronos")
        dtype_token = dtype_to_token(self._preferred_dtype or torch.float32)
        with cache_manager.compilation_env(self.model_name, dtype_token):
            tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name, cache_dir=self.cache_dir)
            model = Kronos.from_pretrained(self.model_name, cache_dir=self.cache_dir)

        if self._preferred_dtype is not None:
            try:
                model = model.to(dtype=self._preferred_dtype)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - dtype conversions may fail on older checkpoints
                logger.debug(f"Unable to convert Kronos model to dtype {self._preferred_dtype}: {exc}")

        def _build_predictor(target_device: str):
            return KronosPredictor(
                model=model,
                tokenizer=tokenizer,
                device=target_device,
                max_context=self.max_context,
                clip=self.clip,
            )

        try:
            predictor = _build_predictor(normalized)
        except Exception as exc:
            if normalized.startswith("cuda") and _is_cuda_oom_error(exc):
                raise RuntimeError(
                    f"Kronos predictor initialisation ran out of memory on device {device_display}. CPU fallback is disabled; reduce sampling requirements or provision a larger GPU."
                ) from exc
            raise
        if self._preferred_dtype is not None:
            try:
                predictor.model = predictor.model.to(dtype=self._preferred_dtype)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - predictor may not expose .model
                logger.debug(f"Failed to set Kronos predictor dtype: {exc}")
        predictor.model = predictor.model.eval()

        # Apply torch.compile if requested
        if self.compile and hasattr(torch, "compile"):
            try:
                logger.info(
                    "Applying torch.compile to Kronos decode methods (mode=%s, backend=%s)",
                    self.compile_mode,
                    self.compile_backend or "default",
                )
                compile_kwargs = {"mode": self.compile_mode}
                if self.compile_backend:
                    compile_kwargs["backend"] = self.compile_backend

                # Compile specific decode methods (like kronos_example.py does)
                if hasattr(predictor.model, "decode_s1"):
                    predictor.model.decode_s1 = torch.compile(predictor.model.decode_s1, **compile_kwargs)  # type: ignore[method-assign]
                if hasattr(predictor.model, "decode_s2"):
                    predictor.model.decode_s2 = torch.compile(predictor.model.decode_s2, **compile_kwargs)  # type: ignore[method-assign]

                logger.info("Kronos torch.compile applied successfully")
            except Exception as exc:
                logger.warning(f"Failed to apply torch.compile to Kronos: {exc}; continuing in eager mode")
                self.compile = False

        metadata_requirements = {
            "model_id": self.model_name,
            "tokenizer_id": self.tokenizer_name,
            "dtype": dtype_token,
            "device": self._device,
            "prefer_fp32": self._prefer_fp32,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "compile": self.compile,
        }
        metadata_payload = {
            **metadata_requirements,
            "max_context": int(self.max_context),
            "clip": float(self.clip),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "sample_count": int(self.sample_count),
            "compile_mode": self.compile_mode if self.compile else None,
            "compile_backend": self.compile_backend if self.compile else None,
        }

        should_persist = True
        existing_metadata = cache_manager.load_metadata(self.model_name, dtype_token)
        if existing_metadata is not None and cache_manager.metadata_matches(existing_metadata, metadata_requirements):
            should_persist = False
        weights_dir = cache_manager.weights_dir(self.model_name, dtype_token)
        if not should_persist and not (weights_dir / "model_state.pt").exists():
            should_persist = True

        if should_persist:
            try:
                cache_manager.persist_model_state(
                    model_id=self.model_name,
                    dtype_token=dtype_token,
                    model=model,
                    metadata=metadata_payload,
                    force=True,
                )
                tokenizer_dir = weights_dir / "tokenizer"
                if hasattr(tokenizer, "save_pretrained"):
                    tokenizer_dir.mkdir(parents=True, exist_ok=True)
                    tokenizer.save_pretrained(str(tokenizer_dir))  # type: ignore[arg-type]
            except ModelCacheError as exc:
                logger.warning(
                    "Failed to persist Kronos cache for %s (%s): %s",
                    self.model_name,
                    dtype_token,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - tokenizer persistence best effort
                logger.debug(f"Failed to persist Kronos tokenizer cache: {exc}")

        self._predictor = predictor
        return predictor

    def _handle_cuda_oom(self) -> None:
        if self._device.startswith("cuda") and torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Failed to clear CUDA cache after OOM: {exc}")
        self.unload()

    def _next_sample_count_after_oom(self, current_samples: int) -> Optional[int]:
        if current_samples <= 1:
            return None
        next_samples = max(1, current_samples // 2)
        if next_samples == current_samples and current_samples > 1:
            next_samples = current_samples - 1
        if next_samples < 1:
            return None
        return next_samples

    def _register_adaptive_sample_limit(self, candidate: int) -> None:
        candidate = max(1, int(candidate))
        if self._adaptive_sample_count is None or candidate < self._adaptive_sample_count:
            self._adaptive_sample_count = candidate

    def _prepare_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()

        def _flatten_column_label(label: Any) -> str:
            if isinstance(label, tuple):
                for part in label:
                    if part is None:
                        continue
                    part_str = str(part).strip()
                    if part_str:
                        return part_str
                if label:
                    return str(label[-1])
                return ""
            return str(label)

        if isinstance(working.columns, pd.MultiIndex):
            working.columns = [_flatten_column_label(col) for col in working.columns]
            working = working.loc[:, ~pd.Index(working.columns).duplicated(keep="first")]

        working = working.rename(columns=lambda c: str(c).lower())
        if working.columns.duplicated().any():
            working = working.loc[:, ~working.columns.duplicated(keep="first")]

        price_columns = ["open", "high", "low", "close"]
        if "close" not in working.columns:
            raise KeyError("Input dataframe must contain a 'close' column for Kronos forecasting.")

        for column in price_columns:
            if column not in working.columns:
                working[column] = working["close"]
            series = working[column]
            if isinstance(series, pd.DataFrame):
                if series.shape[1] == 0:
                    series = pd.Series(np.nan, index=working.index, dtype=float)
                else:
                    series = series.iloc[:, 0]
            elif getattr(series, "ndim", 1) != 1:
                series = pd.Series(np.asarray(series).reshape(-1), index=working.index)
            elif not isinstance(series, pd.Series):
                series = pd.Series(series, index=working.index)
            working[column] = pd.to_numeric(series, errors="coerce")
        working[price_columns] = working[price_columns].ffill().bfill()

        if "volume" not in working.columns:
            working["volume"] = 0.0
        volume_series = working["volume"]
        if isinstance(volume_series, pd.DataFrame):
            volume_series = volume_series.iloc[:, 0] if volume_series.shape[1] else pd.Series(
                np.nan, index=working.index, dtype=float
            )
        elif getattr(volume_series, "ndim", 1) != 1:
            volume_series = pd.Series(np.asarray(volume_series).reshape(-1), index=working.index)
        elif not isinstance(volume_series, pd.Series):
            volume_series = pd.Series(volume_series, index=working.index)
        working["volume"] = pd.to_numeric(volume_series, errors="coerce").fillna(0.0)

        if "amount" not in working.columns:
            working["amount"] = working["volume"] * working["close"]
        else:
            amount_series = working["amount"]
            if isinstance(amount_series, pd.DataFrame):
                amount_series = amount_series.iloc[:, 0] if amount_series.shape[1] else pd.Series(
                    np.nan, index=working.index, dtype=float
                )
            elif getattr(amount_series, "ndim", 1) != 1:
                amount_series = pd.Series(np.asarray(amount_series).reshape(-1), index=working.index)
            elif not isinstance(amount_series, pd.Series):
                amount_series = pd.Series(amount_series, index=working.index)
            working["amount"] = pd.to_numeric(amount_series, errors="coerce")
            working["amount"] = working["amount"].fillna(working["volume"] * working["close"])

        feature_cols = ["open", "high", "low", "close", "volume", "amount"]
        feature_frame = working[feature_cols].astype(np.float32)
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
        feature_frame = feature_frame.ffill().bfill()
        return feature_frame

    @staticmethod
    def _build_future_index(timestamps: pd.Series | pd.DatetimeIndex, pred_len: int) -> pd.DatetimeIndex:
        history = pd.DatetimeIndex(timestamps)
        if history.empty:
            raise ValueError("Cannot infer future index from empty timestamps.")
        if len(history) >= 2:
            deltas = history.to_series().diff().dropna()
            step = deltas.median() if not deltas.empty else None
        else:
            step = None
        if step is None or pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta(days=1)
        start = history[-1] + step
        return pd.date_range(start=start, periods=pred_len, freq=step)

    @staticmethod
    def _compute_step_returns(*, previous: float, absolute: np.ndarray) -> np.ndarray:
        returns = np.zeros_like(absolute, dtype=np.float64)
        last_price = previous
        for idx, price in enumerate(absolute):
            if last_price == 0.0:
                returns[idx] = 0.0
            else:
                returns[idx] = (price - last_price) / last_price
            last_price = price
        return returns
