from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from presets import build_symbol_metadata, normalize_symbol

FEATURE_NAMES = (
    "ret_1h",
    "ret_6h",
    "ret_24h",
    "trend_72h",
    "vol_24h",
    "vol_72h",
    "range_pct",
    "drawdown_72h",
    "price_vs_ema24",
    "volume_z24",
)


@dataclass(frozen=True)
class HourlyMarketData:
    symbols: list[str]
    timestamps: pd.DatetimeIndex
    features: np.ndarray  # [T, S, F]
    closes: np.ndarray  # [T, S]
    returns: np.ndarray  # [T, S]
    shortable_mask: np.ndarray  # [S]
    trade_fee_bps: np.ndarray  # [S]

    @property
    def num_assets(self) -> int:
        return int(self.features.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def slice(self, start: int, end: int) -> "HourlyMarketData":
        start = max(int(start), 0)
        end = min(int(end), len(self))
        if end <= start:
            raise ValueError(f"Invalid slice bounds: start={start}, end={end}")
        return HourlyMarketData(
            symbols=list(self.symbols),
            timestamps=self.timestamps[start:end],
            features=self.features[start:end].copy(),
            closes=self.closes[start:end].copy(),
            returns=self.returns[start:end].copy(),
            shortable_mask=self.shortable_mask.copy(),
            trade_fee_bps=self.trade_fee_bps.copy(),
        )


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray  # [S, F]
    std: np.ndarray  # [S, F]

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {
            "mean": self.mean.astype(np.float32, copy=False).tolist(),
            "std": self.std.astype(np.float32, copy=False).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "FeatureNormalizer | None":
        if not payload:
            return None
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        if mean.ndim != 2 or std.ndim != 2 or mean.shape != std.shape:
            raise ValueError("Invalid feature normalizer payload shape.")
        return cls(mean=mean, std=std)


_MARKET_CACHE: dict[tuple[str, tuple[str, ...], tuple[str, ...], int, float], HourlyMarketData] = {}


def _read_symbol_frame(data_root: Path, symbol: str) -> pd.DataFrame:
    path = data_root / f"{normalize_symbol(symbol)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Hourly Binance CSV not found for {symbol}: {path}")

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    timestamp_col = "timestamp" if "timestamp" in frame.columns else "date"
    if timestamp_col not in frame.columns:
        raise ValueError(f"{path} must contain a timestamp or date column.")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["timestamp"] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce").dt.floor("h")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    frame["volume"] = frame["volume"].fillna(0.0).clip(lower=0.0)
    if frame.empty:
        raise ValueError(f"{path} contains no valid rows after filtering.")
    return frame[["open", "high", "low", "close", "volume"]]


def _compute_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    ret_1h = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_24h = ret_1h.rolling(24, min_periods=1).std(ddof=0).fillna(0.0)
    vol_72h = ret_1h.rolling(72, min_periods=1).std(ddof=0).fillna(0.0)
    ema_24 = close.ewm(span=24, adjust=False, min_periods=1).mean()
    rolling_peak = close.rolling(72, min_periods=1).max()
    range_pct = ((high - low) / close.clip(lower=1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    log_volume = np.log1p(volume.clip(lower=0.0))
    volume_mean = log_volume.rolling(24, min_periods=1).mean()
    volume_std = log_volume.rolling(24, min_periods=1).std(ddof=0).replace(0.0, 1.0)

    features = pd.DataFrame(
        {
            "ret_1h": ret_1h.clip(-0.5, 0.5),
            "ret_6h": close.pct_change(6).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0),
            "ret_24h": close.pct_change(24).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0),
            "trend_72h": close.pct_change(72).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0),
            "vol_24h": vol_24h.clip(0.0, 1.0),
            "vol_72h": vol_72h.clip(0.0, 1.0),
            "range_pct": range_pct.clip(0.0, 1.0),
            "drawdown_72h": ((close - rolling_peak) / rolling_peak.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0),
            "price_vs_ema24": ((close - ema_24) / ema_24.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 1.0),
            "volume_z24": ((log_volume - volume_mean) / volume_std).fillna(0.0).clip(-8.0, 8.0),
        },
        index=frame.index,
    )
    return features[list(FEATURE_NAMES)].astype(np.float32)


def fit_feature_normalizer(market: HourlyMarketData, *, min_std: float = 1e-4) -> FeatureNormalizer:
    if len(market) < 2:
        raise ValueError("Need at least two timesteps to fit a feature normalizer.")
    mean = market.features.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    std = market.features.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    std = np.maximum(std, float(min_std)).astype(np.float32, copy=False)
    return FeatureNormalizer(mean=mean, std=std)


def apply_feature_normalizer(market: HourlyMarketData, normalizer: FeatureNormalizer | None) -> HourlyMarketData:
    if normalizer is None:
        return market
    expected_shape = (market.num_assets, market.feature_dim)
    if normalizer.mean.shape != expected_shape or normalizer.std.shape != expected_shape:
        raise ValueError(
            "Feature normalizer shape mismatch: "
            f"expected {expected_shape}, got mean={normalizer.mean.shape}, std={normalizer.std.shape}"
        )
    normalized = (market.features - normalizer.mean[None, :, :]) / normalizer.std[None, :, :]
    return HourlyMarketData(
        symbols=list(market.symbols),
        timestamps=market.timestamps.copy(),
        features=normalized.astype(np.float32, copy=False),
        closes=market.closes.copy(),
        returns=market.returns.copy(),
        shortable_mask=market.shortable_mask.copy(),
        trade_fee_bps=market.trade_fee_bps.copy(),
    )


def load_hourly_market_data(
    *,
    data_root: str | Path,
    symbols: Iterable[str],
    shortable_symbols: Iterable[str] | None = None,
    min_history_hours: int = 24 * 90,
    min_coverage: float = 0.95,
    use_cache: bool = True,
) -> HourlyMarketData:
    root = Path(data_root)
    symbol_list = [normalize_symbol(symbol) for symbol in symbols]
    shortable_list = [] if shortable_symbols is None else [normalize_symbol(symbol) for symbol in shortable_symbols]
    cache_key = (
        str(root.resolve()),
        tuple(symbol_list),
        tuple(shortable_list),
        int(min_history_hours),
        float(min_coverage),
    )
    if bool(use_cache) and cache_key in _MARKET_CACHE:
        return _MARKET_CACHE[cache_key]

    metadata = build_symbol_metadata(symbol_list, shortable_symbols=shortable_list)
    raw_frames = {item.symbol: _read_symbol_frame(root, item.symbol) for item in metadata}

    overlap_start = max(frame.index.min() for frame in raw_frames.values()).floor("h")
    overlap_end = min(frame.index.max() for frame in raw_frames.values()).floor("h")
    if overlap_end <= overlap_start:
        raise ValueError("No overlapping hourly history across requested symbols.")

    index = pd.date_range(start=overlap_start, end=overlap_end, freq="h", tz="UTC")
    if len(index) < int(min_history_hours):
        raise ValueError(
            f"Only {len(index)} overlapping hours across symbols; require at least {int(min_history_hours)}."
        )

    feature_blocks: list[np.ndarray] = []
    close_blocks: list[np.ndarray] = []
    return_blocks: list[np.ndarray] = []
    resolved_symbols: list[str] = []
    fee_bps: list[float] = []
    shortable: list[bool] = []

    for item in metadata:
        frame = raw_frames[item.symbol]
        observed = index.isin(frame.index).astype(np.uint8, copy=False)
        coverage = float(observed.mean()) if len(observed) else 0.0
        if coverage < float(min_coverage):
            raise ValueError(
                f"{item.symbol} only has {coverage:.2%} coverage over aligned window; "
                f"require >= {float(min_coverage):.2%}."
            )

        aligned = frame.reindex(index, method="ffill").bfill()
        aligned["volume"] = aligned["volume"].where(observed.astype(bool), 0.0)
        features = _compute_features(aligned)
        close = aligned["close"].astype(np.float32).to_numpy(copy=False)
        returns = pd.Series(close, index=index).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        resolved_symbols.append(item.symbol)
        fee_bps.append(float(item.trade_fee_bps))
        shortable.append(bool(item.shortable))
        feature_blocks.append(features.to_numpy(copy=False))
        close_blocks.append(close)
        return_blocks.append(returns.to_numpy(dtype=np.float32, copy=False))

    feature_arr = np.stack(feature_blocks, axis=1).astype(np.float32, copy=False)
    close_arr = np.stack(close_blocks, axis=1).astype(np.float32, copy=False)
    return_arr = np.stack(return_blocks, axis=1).astype(np.float32, copy=False)
    market = HourlyMarketData(
        symbols=resolved_symbols,
        timestamps=index,
        features=feature_arr,
        closes=close_arr,
        returns=return_arr,
        shortable_mask=np.asarray(shortable, dtype=np.float32),
        trade_fee_bps=np.asarray(fee_bps, dtype=np.float32),
    )
    if bool(use_cache):
        _MARKET_CACHE[cache_key] = market
    return market
