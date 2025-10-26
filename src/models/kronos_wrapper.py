from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence

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
        torch = import_module("torch")  # type: ignore[assignment]
    except ModuleNotFoundError as exc:
        raise RuntimeError("Torch is unavailable. Call setup_kronos_wrapper_imports before use.") from exc
    return torch


def _require_numpy() -> ModuleType:
    global np
    if np is not None:
        return np
    try:
        np = import_module("numpy")  # type: ignore[assignment]
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is unavailable. Call setup_kronos_wrapper_imports before use.") from exc
    return np


def _require_pandas() -> ModuleType:
    global pd
    if pd is not None:
        return pd
    try:
        pd = import_module("pandas")  # type: ignore[assignment]
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is unavailable. Call setup_kronos_wrapper_imports before use.") from exc
    return pd


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
    ) -> None:
        if torch is None or np is None or pd is None:
            raise RuntimeError(
                "Torch, NumPy, and pandas must be configured via setup_kronos_wrapper_imports before instantiating KronosForecastingWrapper."
            )
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.requested_device = device
        self.max_context = max_context
        self.clip = clip
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.sample_count = sample_count
        self.cache_dir = cache_dir
        self.verbose = verbose
        self._prefer_fp32 = bool(prefer_fp32)

        self._device = device
        self._predictor = None
        self._preferred_dtype = self._compute_preferred_dtype(device, prefer_fp32=self._prefer_fp32)

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

        predictor = self._ensure_predictor()
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

        forecast_df = predictor.predict(
            payload.feature_frame,
            x_timestamp=payload.history_series,
            y_timestamp=payload.future_series,
            pred_len=int(pred_len),
            T=effective_temperature,
            top_k=effective_top_k,
            top_p=effective_top_p,
            sample_count=effective_samples,
            verbose=effective_verbose,
        )

        if not isinstance(forecast_df, pd.DataFrame):
            raise RuntimeError("Kronos predictor returned an unexpected result type.")

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

        predictor = self._ensure_predictor()
        batch_predict = getattr(predictor, "predict_batch", None)
        if batch_predict is None:
            raise AttributeError("Kronos predictor does not expose 'predict_batch'. Update the Kronos package.")

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

        forecast_list = batch_predict(
            [payload.feature_frame for payload in payloads],
            [payload.history_series for payload in payloads],
            [payload.future_series for payload in payloads],
            pred_len=int(pred_len),
            T=effective_temperature,
            top_k=effective_top_k,
            top_p=effective_top_p,
            sample_count=effective_samples,
            verbose=effective_verbose,
        )

        if not isinstance(forecast_list, (list, tuple)):
            raise RuntimeError("Kronos batch predictor returned an unexpected result type.")
        if len(forecast_list) != len(payloads):
            raise RuntimeError("Kronos batch predictor returned a result with mismatched length.")

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
        effective_samples = int(sample_count if sample_count is not None else self.sample_count)
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

    def unload(self) -> None:
        predictor = self._predictor
        if predictor is None:
            return
        try:
            if hasattr(predictor.model, "to"):
                predictor.model.to("cpu")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to move Kronos model to CPU during unload: %s", exc)
        try:
            if hasattr(predictor.tokenizer, "to"):
                predictor.tokenizer.to("cpu")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to move Kronos tokenizer to CPU during unload: %s", exc)
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

    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor

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

        device = self.requested_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA device %s requested but unavailable; falling back to CPU.", device)
            device = "cpu"
        self._device = device

        cache_manager = ModelCacheManager("kronos")
        dtype_token = dtype_to_token(self._preferred_dtype or torch.float32)
        with cache_manager.compilation_env(self.model_name, dtype_token):
            tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name, cache_dir=self.cache_dir)
            model = Kronos.from_pretrained(self.model_name, cache_dir=self.cache_dir)

        if self._preferred_dtype is not None:
            try:
                model = model.to(dtype=self._preferred_dtype)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - dtype conversions may fail on older checkpoints
                logger.debug("Unable to convert Kronos model to dtype %s: %s", self._preferred_dtype, exc)

        predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context=self.max_context,
            clip=self.clip,
        )
        if self._preferred_dtype is not None:
            try:
                predictor.model = predictor.model.to(dtype=self._preferred_dtype)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - predictor may not expose .model
                logger.debug("Failed to set Kronos predictor dtype: %s", exc)
        predictor.model = predictor.model.eval()

        metadata_requirements = {
            "model_id": self.model_name,
            "tokenizer_id": self.tokenizer_name,
            "dtype": dtype_token,
            "device": self._device,
            "prefer_fp32": self._prefer_fp32,
            "torch_version": getattr(torch, "__version__", "unknown"),
        }
        metadata_payload = {
            **metadata_requirements,
            "max_context": int(self.max_context),
            "clip": float(self.clip),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "sample_count": int(self.sample_count),
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
                logger.debug("Failed to persist Kronos tokenizer cache: %s", exc)

        self._predictor = predictor
        return predictor

    def _prepare_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.rename(columns=lambda c: str(c).lower()).copy()

        price_columns = ["open", "high", "low", "close"]
        if "close" not in working.columns:
            raise KeyError("Input dataframe must contain a 'close' column for Kronos forecasting.")

        for column in price_columns:
            if column not in working.columns:
                working[column] = working["close"]
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working[price_columns] = working[price_columns].ffill().bfill()

        if "volume" not in working.columns:
            working["volume"] = 0.0
        working["volume"] = pd.to_numeric(working["volume"], errors="coerce").fillna(0.0)

        if "amount" not in working.columns:
            working["amount"] = working["volume"] * working["close"]
        else:
            working["amount"] = pd.to_numeric(working["amount"], errors="coerce")
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
