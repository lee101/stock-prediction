from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from hftraining.asset_metadata import get_trading_fee
from hftraining.data_utils import (
    StockDataProcessor,
    load_local_stock_data,
)

from .config import InferenceDataConfig

LOGGER = logging.getLogger(__name__)


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError("Expected 'date' or 'timestamp' column in input dataframe.")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_convert(None)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _trim_by_date(
    df: pd.DataFrame,
    *,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    trimmed = df
    if start is not None:
        trimmed = trimmed[trimmed["date"] >= start]
    if end is not None:
        trimmed = trimmed[trimmed["date"] <= end]
    return trimmed.reset_index(drop=True)


def _resample_frame(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule is None:
        return df
    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    numeric_cols = df.select_dtypes(include=["number"]).columns
    additional_cols = [col for col in numeric_cols if col not in agg_map]
    for col in additional_cols:
        agg_map[col] = "last"
    resampled = (
        df.set_index("date")
        .resample(rule)
        .agg(agg_map)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return resampled


def _fill_missing(df: pd.DataFrame, method: Optional[str]) -> pd.DataFrame:
    if method is None:
        return df
    if method.lower() == "ffill":
        return df.ffill().bfill()
    if method.lower() == "bfill":
        return df.bfill().ffill()
    return df.fillna(method=method)


def load_aligned_price_frames(data_cfg: InferenceDataConfig) -> Dict[str, pd.DataFrame]:
    """
    Load price history for each symbol and align on a shared timestamp index.
    """
    raw_frames = load_local_stock_data(list(data_cfg.normalised_symbols()), str(data_cfg.resolved_data_dir()))
    missing = [sym for sym in data_cfg.normalised_symbols() if sym not in raw_frames]
    if missing:
        raise FileNotFoundError(f"Missing CSV data for symbols: {', '.join(sorted(missing))}")

    start = pd.to_datetime(data_cfg.start_date).tz_localize(None) if data_cfg.start_date else None
    end = pd.to_datetime(data_cfg.end_date).tz_localize(None) if data_cfg.end_date else None

    processed: Dict[str, pd.DataFrame] = {}
    for symbol, frame in raw_frames.items():
        df = _ensure_datetime_index(frame)
        df = _trim_by_date(df, start=start, end=end)
        if data_cfg.max_rows is not None and len(df) > data_cfg.max_rows:
            df = df.tail(data_cfg.max_rows).reset_index(drop=True)
        if data_cfg.resample_rule:
            df = _resample_frame(df, data_cfg.resample_rule)
        df = _fill_missing(df, data_cfg.fill_method)
        processed[symbol] = df

    if not processed:
        raise RuntimeError("No price frames loaded after preprocessing.")

    if not data_cfg.enforce_common_index:
        return processed

    # Intersect timestamps across assets to avoid drift.
    common_index: Optional[pd.DatetimeIndex] = None
    for df in processed.values():
        idx = pd.DatetimeIndex(df["date"])
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or common_index.empty:
        raise RuntimeError("No overlapping timestamps found across provided assets.")

    aligned: Dict[str, pd.DataFrame] = {}
    for symbol, df in processed.items():
        aligned_df = (
            df.set_index("date")
            .reindex(common_index)
            .ffill()
            .bfill()
            .reset_index()
            .rename(columns={"index": "date"})
        )
        aligned[symbol] = aligned_df
    return aligned


@dataclass(slots=True)
class RollingWindowSet:
    """
    Container holding the tensors required to roll a portfolio allocator forward.

    Attributes:
        inputs: Shape (num_samples, sequence_length, input_dim).
        future_returns: Shape (num_samples, num_assets).
        timestamps: Decision timestamps aligned with ``inputs``.
        current_prices: Close price at decision time per asset.
        symbols: Ordered list of tickers.
        per_asset_fees: Trading fees per asset (decimal form).
        feature_dim: Number of features per asset (after scaling).
    """

    inputs: np.ndarray
    future_returns: np.ndarray
    timestamps: np.ndarray
    current_prices: np.ndarray
    symbols: Sequence[str]
    per_asset_fees: np.ndarray
    feature_dim: int

    @property
    def num_assets(self) -> int:
        return len(self.symbols)

    @property
    def sequence_length(self) -> int:
        return self.inputs.shape[1]

    @property
    def input_dim(self) -> int:
        return self.inputs.shape[2]

    @property
    def num_samples(self) -> int:
        return self.inputs.shape[0]


def build_rolling_windows(
    processor: StockDataProcessor,
    price_frames: Mapping[str, pd.DataFrame],
) -> RollingWindowSet:
    """
    Convert aligned price frames into a rolling feature dataset suitable for inference.
    """
    symbols = [sym for sym in price_frames.keys()]
    normalized_features: Dict[str, np.ndarray] = {}
    close_lookup: Dict[str, np.ndarray] = {}

    for symbol in symbols:
        frame = price_frames[symbol]
        features = processor.prepare_features(frame, symbol=symbol)
        transformed = processor.transform(features).astype(np.float32, copy=False)
        normalized_features[symbol] = transformed
        close_lookup[symbol] = frame["close"].to_numpy(dtype=np.float32)

    lengths = {symbol: feats.shape[0] for symbol, feats in normalized_features.items()}
    length_values = set(lengths.values())
    if len(length_values) != 1:
        raise ValueError(f"Aligned features must share identical length; got {lengths!r}")
    total_length = next(iter(length_values))

    seq_len = processor.sequence_length
    horizon = processor.prediction_horizon
    if total_length < seq_len + horizon:
        raise ValueError(
            f"Insufficient history ({total_length}) for sequence_length={seq_len} and horizon={horizon}."
        )

    num_assets = len(symbols)
    feature_dim = normalized_features[symbols[0]].shape[1]
    input_dim = feature_dim * num_assets
    num_samples = total_length - (seq_len + horizon) + 1

    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    future_returns = np.zeros((num_samples, num_assets), dtype=np.float32)
    current_prices = np.zeros((num_samples, num_assets), dtype=np.float32)

    timestamps = price_frames[symbols[0]]["date"].to_numpy(dtype="datetime64[ns]")
    decision_tstamps: List[np.datetime64] = []

    eps = 1e-8
    for sample_idx in range(num_samples):
        window_start = sample_idx
        window_end = sample_idx + seq_len
        decision_idx = window_end - 1
        future_idx = decision_idx + horizon

        decision_tstamps.append(timestamps[decision_idx])
        for asset_idx, symbol in enumerate(symbols):
            feat_matrix = normalized_features[symbol]
            inputs[
                sample_idx, :, asset_idx * feature_dim : (asset_idx + 1) * feature_dim
            ] = feat_matrix[window_start:window_end]

            closes = close_lookup[symbol]
            current_price = float(closes[decision_idx])
            next_price = float(closes[future_idx])
            current_prices[sample_idx, asset_idx] = current_price
            future_returns[sample_idx, asset_idx] = (next_price - current_price) / max(abs(current_price), eps)

    per_asset_fees = np.asarray(
        [float(get_trading_fee(symbol)) for symbol in symbols],
        dtype=np.float32,
    )

    return RollingWindowSet(
        inputs=inputs,
        future_returns=future_returns,
        timestamps=np.asarray(decision_tstamps),
        current_prices=current_prices,
        symbols=symbols,
        per_asset_fees=per_asset_fees,
        feature_dim=feature_dim,
    )


def prepare_inference_windows(
    data_cfg: InferenceDataConfig,
    processor: StockDataProcessor,
) -> RollingWindowSet:
    """
    High-level helper that loads raw price data, aligns it, and emits rolling tensors.
    """
    price_frames = load_aligned_price_frames(data_cfg)
    LOGGER.info(
        "Loaded %d aligned symbols for inference: %s",
        len(price_frames),
        ", ".join(sorted(price_frames)),
    )
    return build_rolling_windows(processor, price_frames)

