"""
Feature construction pipeline for the GymRL experiment.

This module converts historical OHLCV data plus Toto/Kronos forecasts into a
feature cube suitable for the ``PortfolioEnv``. It optionally persists
intermediate artefacts for inspection and can fall back to bootstrap sampling
when neither Toto nor Kronos are available in the runtime.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .config import FeatureBuilderConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureCube:
    """
    Container for RL-ready features, returns, and metadata.

    Attributes:
        features: Feature tensor with shape (T, N, F).
        realized_returns: Next-step realized returns aligned with features (T, N).
        feature_names: Names corresponding to the F feature dimensions.
        symbols: Ordered list of asset identifiers (length N).
        timestamps: Ordered list of timestamps for each row (length T).
        forecast_cvar: Optional tensor of predicted CVaR values (T, N).
        forecast_uncertainty: Optional tensor capturing forecast dispersion (T, N).
    """

    features: np.ndarray
    realized_returns: np.ndarray
    feature_names: List[str]
    symbols: List[str]
    timestamps: List[pd.Timestamp]
    forecast_cvar: Optional[np.ndarray] = None
    forecast_uncertainty: Optional[np.ndarray] = None


class FeatureBuilder:
    """
    Builds feature cubes from historical training data.

    The builder orchestrates three steps:
        1. Load OHLCV data per symbol (optionally resampled / filtered).
        2. Produce probabilistic forecasts using Toto, Kronos, Chronos, or bootstrap.
        3. Aggregate forecast statistics and realized market features into a cube.
    """

    def __init__(
        self,
        config: Optional[FeatureBuilderConfig] = None,
        forecast_backend: Optional[str] = None,
        backend_kwargs: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        self.config = config or FeatureBuilderConfig()
        if forecast_backend is not None:
            self.config.forecast_backend = forecast_backend
        self.backend_kwargs = backend_kwargs or {}

        self._backend_call: Optional[
            Callable[[np.ndarray, int, int], np.ndarray]
        ] = None
        self._backend_name: Optional[str] = None
        self._backend_errors: List[str] = []
        self._data_dir: Optional[Path] = None
        self._chronos2_bundles: Dict[str, pd.DataFrame] = {}

    # Public API -------------------------------------------------------------------
    def build_from_directory(
        self,
        data_dir: Path,
        *,
        symbols: Optional[Sequence[str]] = None,
        price_column: str = "close",
    ) -> FeatureCube:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
        self._data_dir = data_dir.resolve()
        self._chronos2_bundles.clear()

        csv_files = list(sorted(data_dir.glob("*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        inferred_symbols = []
        price_frames: Dict[str, pd.DataFrame] = {}
        price_column = price_column.lower()

        for csv in csv_files:
            symbol = csv.stem.split("-")[0]
            inferred_symbols.append(symbol)
            if symbols is not None and symbol not in symbols:
                continue
            df = self._load_csv(csv)
            price_frames[symbol] = df

        if not price_frames:
            raise ValueError("No price data loaded; check symbol filters or file naming.")

        ordered_symbols = sorted(price_frames.keys())
        logger.info(
            "Loaded %d symbols for GymRL feature cube: %s",
            len(ordered_symbols),
            ordered_symbols,
        )

        self._initialise_backend()

        feature_frames: Dict[str, pd.DataFrame] = {}
        for symbol in ordered_symbols:
            df = price_frames[symbol]
            feature_frames[symbol] = self._build_symbol_features(df, price_column=price_column, symbol=symbol)

        aligned_features = self._align_frames(feature_frames)
        return self._to_feature_cube(aligned_features, ordered_symbols)

    # CSV loading ------------------------------------------------------------------
    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [col.strip().lower() for col in df.columns]
        if "timestamp" not in df.columns:
            raise ValueError(f"Expected 'timestamp' column in {path}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        if self.config.resample_rule:
            df = (
                df.resample(self.config.resample_rule)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )

        df = df.dropna()
        if len(df) < self.config.min_history + self.config.realized_horizon + 1:
            raise ValueError(f"Not enough data in {path} after preprocessing.")
        return df

    def _apply_fill(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values according to configuration, including leading gaps."""

        if self.config.fill_method is None:
            return frame
        method = self.config.fill_method.lower()
        if method == "ffill":
            return frame.ffill().bfill()
        if method == "bfill":
            return frame.bfill().ffill()
        return frame.fillna(method=self.config.fill_method)

    # Feature construction ----------------------------------------------------------
    def _build_symbol_features(
        self,
        df: pd.DataFrame,
        *,
        price_column: str,
        symbol: str,
    ) -> pd.DataFrame:
        close = df[price_column].to_numpy(dtype=np.float64)
        highs = df.get("high", pd.Series(index=df.index, dtype=float)).to_numpy(dtype=np.float64)
        lows = df.get("low", pd.Series(index=df.index, dtype=float)).to_numpy(dtype=np.float64)
        volumes = df.get("volume", pd.Series(index=df.index, dtype=float)).to_numpy(dtype=np.float64)
        chronos2_bundle = self._load_chronos2_bundle(symbol) if self._backend_name == "chronos2" else None
        chronos2_horizons = self._parse_chronos2_horizons()

        records: List[Dict[str, float]] = []
        realized_returns: List[float] = []
        forecast_cvar: List[float] = []
        forecast_uncertainty: List[float] = []
        timestamps: List[pd.Timestamp] = []

        min_idx = max(self.config.min_history, self.config.context_window)
        max_idx = len(close) - self.config.realized_horizon - self.config.lookahead_buffer

        horizon = self.config.realized_horizon

        for idx in range(min_idx, max_idx):
            context_start = idx - self.config.context_window
            context_prices = close[context_start:idx]

            if len(context_prices) < self.config.context_window:
                continue

            current_price = close[idx]
            future_price = close[idx + horizon]
            realized_return = future_price / current_price - 1.0

            extra_features: Dict[str, float] = {}
            if chronos2_bundle is not None:
                timestamp = df.index[idx]
                if timestamp not in chronos2_bundle.index:
                    continue
                row = chronos2_bundle.loc[timestamp]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                samples = self._chronos2_samples_from_row(row, current_price=current_price, horizons=chronos2_horizons)
                extra_features = self._chronos2_snapshot_features(
                    row,
                    current_price=current_price,
                    horizons=chronos2_horizons,
                )
            else:
                samples = self._generate_samples(
                    context_prices,
                    symbol=symbol,
                    full_history=df,
                    current_index=idx,
                    price_column=price_column,
                )
            forecast_stats = self._compute_forecast_statistics(
                samples=samples,
                current_price=current_price,
                realized_return=realized_return,
            )
            realized_stats = self._compute_realized_features(
                close=close,
                highs=highs,
                lows=lows,
                volumes=volumes,
                current_index=idx,
            )

            record = {**forecast_stats["features"], **extra_features, **realized_stats}
            records.append(record)

            timestamps.append(df.index[idx])
            realized_returns.append(forecast_stats["realized_return"])
            forecast_cvar.append(forecast_stats["cvar"])
            forecast_uncertainty.append(forecast_stats["uncertainty"])

        feature_frame = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps, name="timestamp"))
        feature_frame["realized_return"] = realized_returns
        feature_frame["forecast_cvar"] = forecast_cvar
        feature_frame["forecast_uncertainty"] = forecast_uncertainty
        feature_frame["symbol"] = symbol

        feature_frame = feature_frame.dropna()
        if feature_frame.empty:
            raise ValueError(f"No feature rows produced for {symbol}; check configuration.")

        return feature_frame

    def _align_frames(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        common_index = None
        for frame in frames.values():
            idx = frame.index
            common_index = idx if common_index is None else common_index.intersection(idx)

        if self.config.enforce_common_index:
            if common_index is None or common_index.empty:
                raise ValueError("No overlapping timestamps across symbols; cannot form feature cube.")
            for symbol, frame in frames.items():
                frames[symbol] = frame.loc[common_index]
        else:
            full_index = sorted({ts for frame in frames.values() for ts in frame.index})
            full_index = pd.DatetimeIndex(full_index)
            for symbol, frame in frames.items():
                reindexed = frame.reindex(full_index)
                reindexed = self._apply_fill(reindexed)
                frames[symbol] = reindexed.dropna()
        return frames

    def _to_feature_cube(self, frames: Dict[str, pd.DataFrame], ordered_symbols: List[str]) -> FeatureCube:
        ordered_frames = [frames[symbol] for symbol in ordered_symbols]

        feature_columns = [
            col
            for col in ordered_frames[0].columns
            if col not in ("realized_return", "forecast_cvar", "forecast_uncertainty", "symbol")
        ]

        features = np.stack(
            [frame[feature_columns].to_numpy(dtype=np.float32) for frame in ordered_frames],
            axis=1,
        )
        realized = np.stack(
            [frame["realized_return"].to_numpy(dtype=np.float32) for frame in ordered_frames],
            axis=1,
        )
        forecast_cvar = np.stack(
            [frame["forecast_cvar"].to_numpy(dtype=np.float32) for frame in ordered_frames],
            axis=1,
        )
        forecast_uncertainty = np.stack(
            [frame["forecast_uncertainty"].to_numpy(dtype=np.float32) for frame in ordered_frames],
            axis=1,
        )

        timestamps = list(ordered_frames[0].index)

        return FeatureCube(
            features=features,
            realized_returns=realized,
            feature_names=feature_columns,
            symbols=ordered_symbols,
            timestamps=timestamps,
            forecast_cvar=forecast_cvar,
            forecast_uncertainty=forecast_uncertainty,
        )

    @property
    def backend_name(self) -> Optional[str]:
        """Return the identifier of the active forecasting backend."""
        return self._backend_name

    @property
    def backend_errors(self) -> List[str]:
        """Return backend initialisation errors encountered during setup."""
        return list(self._backend_errors)

    # Forecasting backends ----------------------------------------------------------
    def _initialise_backend(self) -> None:
        backend_name = self.config.forecast_backend.lower()
        tried_backends: List[str] = []

        if backend_name == "chronos2":
            self._backend_call = self._make_bootstrap_backend()
            self._backend_name = "chronos2"
            return

        if backend_name in ("toto", "auto"):
            backend = self._make_toto_backend()
            if backend:
                self._backend_call = backend
                self._backend_name = "toto"
                return
            tried_backends.append("toto")

        if backend_name in ("kronos", "auto"):
            backend = self._make_kronos_backend()
            if backend:
                self._backend_call = backend
                self._backend_name = "kronos"
                return
            tried_backends.append("kronos")

        if backend_name in ("chronos", "auto"):
            backend = self._make_chronos_backend()
            if backend:
                self._backend_call = backend
                self._backend_name = "chronos"
                return
            tried_backends.append("chronos")

        tried_backends.append("bootstrap")
        self._backend_call = self._make_bootstrap_backend()
        self._backend_name = "bootstrap"
        message = (
            "Falling back to bootstrap forecast backend (tried: %s). "
            "Install Toto or Kronos to enable model-based forecasting."
        )
        if self._backend_errors:
            logger.warning(
                message + " Backend errors: %s",
                ", ".join(tried_backends),
                "; ".join(self._backend_errors),
            )
        else:
            logger.warning(
                message,
                ", ".join(tried_backends),
            )

    def _parse_backend_bool(self, key: str, default: bool = False) -> bool:
        value = self.backend_kwargs.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
        return bool(value)

    def _parse_chronos2_horizons(self) -> list[int]:
        raw = self.backend_kwargs.get("chronos2_horizons")
        if raw is None:
            return [int(self.config.prediction_length)]
        if isinstance(raw, str):
            tokens = [part.strip() for part in raw.split(",")]
        elif isinstance(raw, Sequence):
            tokens = [str(part).strip() for part in raw]
        else:
            tokens = [str(raw).strip()]
        horizons = [int(token) for token in tokens if token]
        if not horizons:
            horizons = [int(self.config.prediction_length)]
        return sorted(set(horizons))

    def _load_chronos2_bundle(self, symbol: str) -> pd.DataFrame:
        cached = self._chronos2_bundles.get(symbol)
        if cached is not None:
            return cached
        if self._data_dir is None:
            raise RuntimeError("Chronos2 forecasts requested before data directory was initialised.")

        from binanceneural.forecasts import build_forecast_bundle

        cache_root_raw = self.backend_kwargs.get("chronos2_cache_root")
        cache_root = Path(cache_root_raw) if cache_root_raw else Path("binanceneural") / "forecast_cache"
        bundle = build_forecast_bundle(
            symbol=symbol,
            data_root=self._data_dir,
            cache_root=cache_root,
            horizons=self._parse_chronos2_horizons(),
            context_hours=int(self.backend_kwargs.get("chronos2_context_hours", self.config.context_window)),
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=max(1, int(self.config.num_samples)),
            cache_only=self._parse_backend_bool("chronos2_cache_only", default=False),
            force_multivariate=self._parse_backend_bool("chronos2_force_multivariate", default=False),
            force_cross_learning=self._parse_backend_bool("chronos2_force_cross_learning", default=False),
        )
        bundle = bundle.copy()
        bundle["timestamp"] = pd.to_datetime(bundle["timestamp"], utc=True, errors="coerce")
        bundle = bundle.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
        indexed = bundle.set_index("timestamp").sort_index()
        self._chronos2_bundles[symbol] = indexed
        return indexed

    def _chronos2_samples_from_row(
        self,
        row: pd.Series,
        *,
        current_price: float,
        horizons: Sequence[int],
    ) -> np.ndarray:
        target_horizon = int(self.config.prediction_length)
        if target_horizon not in horizons:
            target_horizon = int(horizons[0])
        p50 = float(row.get(f"predicted_close_p50_h{target_horizon}", current_price))
        p10 = float(row.get(f"predicted_close_p10_h{target_horizon}", p50))
        p90 = float(row.get(f"predicted_close_p90_h{target_horizon}", p50))
        return np.asarray([[p10], [p50], [p90]], dtype=np.float32)

    def _chronos2_snapshot_features(
        self,
        row: pd.Series,
        *,
        current_price: float,
        horizons: Sequence[int],
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        denom = max(abs(current_price), 1e-8)
        for horizon in horizons:
            p50_key = f"predicted_close_p50_h{horizon}"
            if p50_key not in row or pd.isna(row[p50_key]):
                continue
            p50 = float(row[p50_key])
            p10 = float(row.get(f"predicted_close_p10_h{horizon}", p50))
            p90 = float(row.get(f"predicted_close_p90_h{horizon}", p50))
            features[f"chronos2_return_h{horizon}"] = float(p50 / denom - 1.0)
            features[f"chronos2_spread_h{horizon}"] = float((p90 - p10) / denom)
        return features

    def _make_toto_backend(self) -> Optional[Callable[[np.ndarray, int, int], np.ndarray]]:
        try:
            from src.models.toto_wrapper import TotoPipeline
        except Exception as exc:
            self._backend_errors.append(f"Toto backend unavailable: {exc}")
            logger.debug("Failed to import TotoPipeline: %s", exc)
            return None

        device = self.backend_kwargs.get("device_map")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            pipeline = TotoPipeline.from_pretrained(device_map=device)
        except Exception as exc:
            self._backend_errors.append(f"TotoPipeline initialisation error: {exc}")
            logger.warning("Could not initialise Toto backend: %s", exc)
            return None

        def _backend(
            context: np.ndarray,
            prediction_length: int,
            num_samples: int,
            **_: object,
        ) -> np.ndarray:
            forecasts = pipeline.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
            first = forecasts[0]
            if hasattr(first, "numpy"):
                samples = np.asarray(first.numpy())
            elif torch.is_tensor(first):
                # Ensure CPU before converting to NumPy to avoid CUDA -> NumPy errors
                samples = first.detach().cpu().numpy()
            else:
                samples = np.asarray(first)
            if samples.ndim == 0:
                samples = samples.reshape(1, 1)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            if samples.ndim == 2:
                # Chronos-style: (num_samples, horizon)
                return samples
            if samples.ndim == 3:
                # Toto-style: (prediction_length, num_samples)
                return np.swapaxes(samples, 0, 1)
            raise ValueError(f"Unexpected Toto sample shape: {samples.shape}")

        return _backend

    def _make_kronos_backend(self) -> Optional[Callable[[np.ndarray, int, int], np.ndarray]]:
        try:
            from src.models.kronos_wrapper import KronosForecastingWrapper
        except Exception as exc:
            self._backend_errors.append(f"Kronos backend unavailable: {exc}")
            logger.debug("Failed to import KronosForecastingWrapper: %s", exc)
            return None

        device = self.backend_kwargs.get("kronos_device")
        if device is None:
            device = self.backend_kwargs.get("device_map")
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_str = str(device)

        wrapper_kwargs = {
            "model_name": self.backend_kwargs.get("kronos_model_name", "NeoQuasar/Kronos-base"),
            "tokenizer_name": self.backend_kwargs.get("kronos_tokenizer_name", "NeoQuasar/Kronos-Tokenizer-base"),
            "device": device_str,
            "max_context": int(self.backend_kwargs.get("kronos_max_context", 512)),
            "clip": float(self.backend_kwargs.get("kronos_clip", 5.0)),
            "temperature": float(self.backend_kwargs.get("kronos_temperature", 0.75)),
            "top_p": float(self.backend_kwargs.get("kronos_top_p", 0.9)),
            "top_k": int(self.backend_kwargs.get("kronos_top_k", 0)),
            "sample_count": int(self.backend_kwargs.get("kronos_sample_count", 8)),
        }
        # Kronos wrapper currently manages its own retry logic; discard legacy knobs silently.
        wrapper_kwargs.pop("oom_retries", None)

        try:
            wrapper = KronosForecastingWrapper(**wrapper_kwargs)
        except Exception as exc:
            self._backend_errors.append(f"Kronos initialisation error: {exc}")
            logger.warning("Could not initialise Kronos backend: %s", exc)
            return None

        jitter_std = float(self.backend_kwargs.get("kronos_jitter_std", 0.0))

        def _backend(
            context: np.ndarray,
            prediction_length: int,
            num_samples: int,
            **kwargs: object,
        ) -> np.ndarray:
            full_history = kwargs.get("full_history")
            current_index = kwargs.get("current_index")
            price_column = kwargs.get("price_column", "close")

            if not isinstance(full_history, pd.DataFrame) or not isinstance(current_index, int):
                raise ValueError("Kronos backend requires full_history dataframe and current_index integer.")

            context_len = len(context)
            context_start = max(0, current_index - context_len)
            context_df = full_history.iloc[context_start:current_index]
            target_df = full_history.iloc[current_index : current_index + prediction_length]

            if len(context_df) < context_len:
                raise ValueError("Insufficient context history supplied for Kronos backend.")
            if len(target_df) < prediction_length:
                raise ValueError("Insufficient future rows available for Kronos forecast horizon.")

            combined = pd.concat([context_df, target_df])
            data = combined.reset_index().rename(columns={"index": "timestamp"})

            try:
                results = wrapper.predict_series(
                    data=data,
                    timestamp_col="timestamp",
                    columns=[price_column],
                    pred_len=prediction_length,
                    lookback=context_len,
                )
            except Exception as exc:  # pragma: no cover - surfaced when Kronos fails after init
                raise RuntimeError(f"Kronos forecast failed: {exc}") from exc

            result = results.get(price_column)
            if result is None:
                raise RuntimeError(f"Kronos did not return forecasts for column '{price_column}'.")

            base_path = np.asarray(result.absolute, dtype=np.float32).reshape(1, -1)
            if base_path.shape[1] != prediction_length:
                raise RuntimeError(
                    f"Kronos returned unexpected horizon {base_path.shape[1]} (expected {prediction_length})."
                )

            repeats = max(1, int(np.ceil(num_samples / base_path.shape[0])))
            tiled = np.tile(base_path, (repeats, 1))[:num_samples]

            if jitter_std > 0.0:
                noise = np.random.normal(loc=0.0, scale=jitter_std, size=tiled.shape).astype(np.float32)
                tiled = np.clip(tiled + noise, a_min=1e-6, a_max=None)

            return tiled

        return _backend

    def _make_chronos_backend(self) -> Optional[Callable[[np.ndarray, int, int], np.ndarray]]:
        try:
            from chronos import BaseChronosPipeline
        except Exception as exc:
            self._backend_errors.append(f"Chronos backend unavailable: {exc}")
            logger.debug("Failed to import BaseChronosPipeline: %s", exc)
            return None

        device = self.backend_kwargs.get("device_map")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = self.backend_kwargs.get("model_name", "amazon/chronos-bolt-base")
        try:
            pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map=device)
            pipeline.model = pipeline.model.eval()
        except Exception as exc:
            self._backend_errors.append(f"Chronos initialisation error: {exc}")
            logger.warning("Could not initialise Chronos backend: %s", exc)
            return None

        def _backend(
            context: np.ndarray,
            prediction_length: int,
            num_samples: int,
            **_: object,
        ) -> np.ndarray:
            context_tensor = torch.as_tensor(context, dtype=torch.float32)
            forecast = pipeline.predict(context_tensor, prediction_length=prediction_length)
            samples = forecast.cpu().numpy()

            if samples.ndim == 3 and samples.shape[0] == 1:
                samples = samples.squeeze(0)

            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            elif samples.ndim == 2:
                pass
            else:
                raise ValueError(f"Unexpected Chronos prediction shape: {samples.shape}")

            if samples.shape[0] == 1 and samples.shape[1] == prediction_length:
                base_samples = samples
            else:
                base_samples = samples

            if base_samples.shape[0] < num_samples:
                repeats = int(np.ceil(num_samples / base_samples.shape[0]))
                tiled = np.tile(base_samples, (repeats, 1))
                base_samples = tiled[:num_samples]

            return base_samples.astype(np.float32)

        return _backend

    def _make_bootstrap_backend(self) -> Callable[[np.ndarray, int, int], np.ndarray]:
        block_size = self.config.bootstrap_block_size

        def _backend(
            context: np.ndarray,
            prediction_length: int,
            num_samples: int,
            **_: object,
        ) -> np.ndarray:
            returns = np.diff(np.log(context + 1e-8))
            if len(returns) < block_size:
                block = np.tile(returns, int(math.ceil(block_size / len(returns))))
                returns = block[:block_size]

            samples = np.zeros((num_samples, prediction_length), dtype=np.float32)
            for i in range(num_samples):
                sample_returns = []
                while len(sample_returns) < prediction_length:
                    start = np.random.randint(0, len(returns) - block_size + 1)
                    block = returns[start : start + block_size]
                    sample_returns.extend(block.tolist())
                sample_returns = np.array(sample_returns[:prediction_length])
                prices = context[-1] * np.exp(np.cumsum(sample_returns))
                samples[i] = prices
            return samples

        return _backend

    def _generate_samples(
        self,
        context: np.ndarray,
        *,
        symbol: Optional[str] = None,
        full_history: Optional[pd.DataFrame] = None,
        current_index: Optional[int] = None,
        price_column: str = "close",
    ) -> np.ndarray:
        if self._backend_call is None:
            raise RuntimeError("Forecast backend not initialised.")
        prediction_length = self.config.prediction_length
        num_samples = self.config.num_samples
        return self._backend_call(
            context,
            prediction_length,
            num_samples,
            symbol=symbol,
            full_history=full_history,
            current_index=current_index,
            price_column=price_column,
        )

    # Statistics --------------------------------------------------------------------
    def _compute_forecast_statistics(
        self,
        samples: np.ndarray,
        current_price: float,
        realized_return: float,
    ) -> Dict[str, Any]:
        if samples.ndim != 2:
            raise ValueError(f"Forecast samples must be 2-D (num_samples, horizon); got {samples.shape}")
        terminal_prices = samples[:, -1]
        terminal_prices = np.clip(terminal_prices, a_min=1e-6, a_max=None)

        simple_returns = terminal_prices / current_price - 1.0
        log_returns = np.log1p(simple_returns)

        mu = float(np.mean(log_returns))
        sigma = float(np.std(log_returns))
        prob_up = float(np.mean(simple_returns > 0))
        q10, q50, q90 = np.quantile(simple_returns, [0.1, 0.5, 0.9])
        iqr = float(q90 - q10)
        tail_ratio = float((np.percentile(simple_returns, 95) + 1e-8) / (abs(np.percentile(simple_returns, 5)) + 1e-8))

        bins = min(50, max(10, int(np.sqrt(len(simple_returns)))))
        hist, _ = np.histogram(simple_returns, bins=bins, density=True)
        probs = hist / hist.sum() if hist.sum() > 0 else np.ones_like(hist) / len(hist)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        centered = simple_returns - np.mean(simple_returns)
        skew = float(np.mean(centered**3) / (np.std(simple_returns) ** 3 + 1e-8))
        kurtosis = float(np.mean(centered**4) / (np.std(simple_returns) ** 4 + 1e-8))

        alpha = self.backend_kwargs.get("cvar_alpha", 0.05)
        var_threshold = np.quantile(simple_returns, alpha)
        cvar_mask = simple_returns <= var_threshold
        cvar = float(simple_returns[cvar_mask].mean()) if np.any(cvar_mask) else float(var_threshold)

        mean_simple = float(np.mean(simple_returns))

        features = {
            "forecast_mu": mu,
            "forecast_sigma": sigma,
            "forecast_prob_up": prob_up,
            "forecast_q10": float(q10),
            "forecast_q50": float(q50),
            "forecast_q90": float(q90),
            "forecast_mean_return": mean_simple,
            "forecast_iqr": iqr,
            "forecast_tail_ratio": tail_ratio,
            "forecast_entropy": entropy,
            "forecast_skew": skew,
            "forecast_kurtosis": kurtosis,
        }

        return {
            "features": features,
            # Preserve the ground truth realized return passed by caller
            "realized_return": float(realized_return),
            "cvar": cvar,
            "uncertainty": iqr,
        }

    def _compute_realized_features(
        self,
        *,
        close: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        current_index: int,
    ) -> Dict[str, float]:
        realized_features: Dict[str, float] = {}
        current_price = close[current_index]

        for window in self.config.realized_feature_windows:
            if current_index - window < 0:
                continue
            window_slice = slice(current_index - window, current_index + 1)
            prices = close[window_slice]
            log_returns = np.diff(np.log(prices + 1e-8))

            momentum = prices[-1] / prices[0] - 1.0
            volatility = float(np.std(log_returns))
            realized_features[f"momentum_{window}"] = float(momentum)
            realized_features[f"volatility_{window}"] = volatility

        if current_index > 0:
            true_return = close[current_index] / close[current_index - 1] - 1.0
            range_val = (highs[current_index] - lows[current_index]) / (close[current_index - 1] + 1e-8)
            volume_change = (
                volumes[current_index] - volumes[current_index - 1]
            ) / (volumes[current_index - 1] + 1e-8)
            realized_features["realized_return_prev"] = float(true_return)
            realized_features["intraday_range"] = float(range_val)
            realized_features["volume_change"] = float(volume_change)

        return realized_features


__all__ = ["FeatureBuilder", "FeatureCube"]
