from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from pufferlib_market.export_data_daily import compute_daily_features as compute_daily_feature_history_rsi
from pufferlib_market.inference_daily import compute_daily_features as compute_daily_feature_vector_rsi


DailyStockFeatureSchema = Literal["legacy_prod", "rsi_v5"]
FEATURE_DIMENSION_BY_SCHEMA: dict[DailyStockFeatureSchema, int] = {
    "legacy_prod": 16,
    "rsi_v5": 16,
}

_LEGACY_PROD_MARKERS = (
    "pufferlib_market/prod_ensemble/",
    "/pufferlib_market/prod_ensemble/",
)
_RSI_V5_MARKERS = (
    "stocks12_v5_rsi",
    "v5_rsi",
    "stocks17_sweep",
    "screened32_sweep",
    "screened32_ext",
    "prod_ensemble_screened32",
)


def _normalize_checkpoint_path(path: str | Path) -> str:
    return str(Path(path).expanduser()).replace("\\", "/").lower()


def _schema_for_checkpoint_path(path: str | Path) -> DailyStockFeatureSchema | None:
    normalized = _normalize_checkpoint_path(path)
    if any(marker in normalized for marker in _RSI_V5_MARKERS):
        return "rsi_v5"
    if any(marker in normalized for marker in _LEGACY_PROD_MARKERS):
        return "legacy_prod"
    return None


def resolve_daily_feature_schema(
    checkpoint: str | Path,
    *,
    extra_checkpoints: Iterable[str | Path] | None = None,
) -> DailyStockFeatureSchema:
    paths = [checkpoint, *(extra_checkpoints or ())]
    inferred = {
        schema
        for schema in (_schema_for_checkpoint_path(path) for path in paths)
        if schema is not None
    }
    if len(inferred) > 1:
        path_text = ", ".join(str(path) for path in paths)
        raise ValueError(
            "Daily checkpoint ensemble mixes incompatible feature schemas "
            f"({sorted(inferred)}): {path_text}"
        )
    if inferred:
        return next(iter(inferred))
    return "rsi_v5"


def daily_feature_dimension(schema: DailyStockFeatureSchema) -> int:
    return FEATURE_DIMENSION_BY_SCHEMA[schema]


def compute_legacy_daily_feature_history(price_df: pd.DataFrame) -> pd.DataFrame:
    """Legacy 16-feature daily vector used by the pre-RSI prod ensemble."""
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    volume = price_df["volume"].astype(float)

    feat = pd.DataFrame(index=price_df.index)

    ret_1d = close.pct_change(1).fillna(0.0)
    ret_20d = close.pct_change(20).fillna(0.0).clip(-2.0, 2.0)

    feat["return_1d"] = ret_1d.clip(-0.5, 0.5)
    feat["return_5d"] = close.pct_change(5).fillna(0.0).clip(-1.0, 1.0)
    feat["return_20d"] = ret_20d

    feat["volatility_5d"] = ret_1d.rolling(5, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)
    feat["volatility_20d"] = ret_1d.rolling(20, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)

    ma5 = close.rolling(5, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    feat["ma_delta_5d"] = ((close - ma5) / ma5.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    feat["ma_delta_20d"] = ((close - ma20) / ma20.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    feat["ma_delta_60d"] = ((close - ma60) / ma60.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    feat["atr_pct_14d"] = (atr14 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)
    feat["range_pct_1d"] = ((high - low) / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)

    # Legacy bug preserved intentionally for backwards-compatible inference:
    # trend_20d duplicated return_20d instead of using RSI(14).
    feat["trend_20d"] = ret_20d
    feat["trend_60d"] = close.pct_change(60).fillna(0.0).clip(-3.0, 3.0)

    roll_max_20 = close.rolling(20, min_periods=1).max()
    roll_max_60 = close.rolling(60, min_periods=1).max()
    feat["drawdown_20d"] = ((close - roll_max_20) / roll_max_20.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)
    feat["drawdown_60d"] = ((close - roll_max_60) / roll_max_60.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)

    log_vol = volume.clip(lower=0.0).map(math.log1p)
    log_vol_mean20 = log_vol.rolling(20, min_periods=1).mean()
    log_vol_std20 = log_vol.rolling(20, min_periods=1).std(ddof=0).replace(0.0, 1.0)
    feat["log_volume_z20d"] = ((log_vol - log_vol_mean20) / log_vol_std20).fillna(0.0).clip(-5.0, 5.0)
    feat["log_volume_delta_5d"] = (log_vol - log_vol.rolling(5, min_periods=1).mean()).fillna(0.0).clip(-10.0, 10.0)

    expected_cols = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_5d",
        "volatility_20d",
        "ma_delta_5d",
        "ma_delta_20d",
        "ma_delta_60d",
        "atr_pct_14d",
        "range_pct_1d",
        "trend_20d",
        "trend_60d",
        "drawdown_20d",
        "drawdown_60d",
        "log_volume_z20d",
        "log_volume_delta_5d",
    ]
    return feat[expected_cols].fillna(0.0).astype(np.float32)


def build_daily_feature_history_for_schema(
    price_df: pd.DataFrame,
    *,
    schema: DailyStockFeatureSchema,
) -> pd.DataFrame:
    if schema == "legacy_prod":
        return compute_legacy_daily_feature_history(price_df)
    return compute_daily_feature_history_rsi(price_df)


def compute_daily_feature_vector_for_schema(
    price_df: pd.DataFrame,
    *,
    schema: DailyStockFeatureSchema,
) -> np.ndarray:
    if schema == "legacy_prod":
        return compute_legacy_daily_feature_history(price_df).iloc[-1].to_numpy(dtype=np.float32, copy=False)
    return compute_daily_feature_vector_rsi(price_df)
