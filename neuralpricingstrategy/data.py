from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .feature_builder import FeatureBuilder, FeatureSpec


BACKTEST_DATE_REGEX = re.compile(
    r"\('(?P<symbol>[^']+)',\s*Timestamp\('(?P<timestamp>[^']+)'(?:,\s*tz='[^']+')?\)\)"
)

DEFAULT_NUMERIC_COLUMNS: List[str] = [
    "close",
    "predicted_close",
    "predicted_high",
    "predicted_low",
    "toto_expected_move_pct",
    "kronos_expected_move_pct",
    "realized_volatility_pct",
    "dollar_vol_20d",
    "atr_pct_14",
    "spread_bps_estimate",
    "raw_expected_move_pct",
    "calibrated_expected_move_pct",
    "calibration_slope",
    "calibration_intercept",
    "simple_strategy_return",
    "simple_strategy_sharpe",
    "simple_strategy_finalday",
    "simple_strategy_avg_daily_return",
    "simple_strategy_annual_return",
    "all_signals_strategy_return",
    "all_signals_strategy_sharpe",
    "all_signals_strategy_finalday",
    "all_signals_strategy_avg_daily_return",
    "all_signals_strategy_annual_return",
    "buy_hold_return",
    "buy_hold_sharpe",
    "buy_hold_finalday",
    "buy_hold_avg_daily_return",
    "buy_hold_annual_return",
    "unprofit_shutdown_return",
    "unprofit_shutdown_sharpe",
    "unprofit_shutdown_finalday",
    "unprofit_shutdown_avg_daily_return",
    "unprofit_shutdown_annual_return",
    "entry_takeprofit_return",
    "entry_takeprofit_sharpe",
    "entry_takeprofit_finalday",
    "entry_takeprofit_avg_daily_return",
    "entry_takeprofit_annual_return",
    "highlow_return",
    "highlow_sharpe",
    "highlow_finalday_return",
    "highlow_avg_daily_return",
    "highlow_annual_return",
    "maxdiff_return",
    "maxdiff_sharpe",
    "maxdiff_finalday_return",
    "maxdiff_avg_daily_return",
    "maxdiff_annual_return",
    "maxdiff_turnover",
    "maxdiffalwayson_return",
    "maxdiffalwayson_sharpe",
    "maxdiffalwayson_finalday_return",
    "maxdiffalwayson_avg_daily_return",
    "maxdiffalwayson_annual_return",
    "maxdiffalwayson_turnover",
    "maxdiffalwayson_profit",
    "maxdiffalwayson_high_multiplier",
    "maxdiffalwayson_low_multiplier",
    "maxdiffalwayson_high_price",
    "maxdiffalwayson_low_price",
    "maxdiffalwayson_buy_contribution",
    "maxdiffalwayson_sell_contribution",
    "maxdiffalwayson_filled_buy_trades",
    "maxdiffalwayson_filled_sell_trades",
    "maxdiffalwayson_trades_total",
    "maxdiffalwayson_trade_bias",
    "close_val_loss",
    "high_val_loss",
    "low_val_loss",
    "walk_forward_oos_sharpe",
    "walk_forward_turnover",
    "walk_forward_highlow_sharpe",
    "walk_forward_takeprofit_sharpe",
    "walk_forward_maxdiff_sharpe",
    "walk_forward_maxdiffalwayson_sharpe",
    "simple_forecasted_pnl",
    "all_signals_forecasted_pnl",
    "buy_hold_forecasted_pnl",
    "unprofit_shutdown_forecasted_pnl",
    "entry_takeprofit_forecasted_pnl",
    "highlow_forecasted_pnl",
    "maxdiff_forecasted_pnl",
    "maxdiffalwayson_forecasted_pnl",
]

DEFAULT_CATEGORICAL_COLUMNS: List[str] = ["close_prediction_source", "symbol"]

TARGET_LOW_COLUMN = "neuralpricing_target_low_delta"
TARGET_HIGH_COLUMN = "neuralpricing_target_high_delta"
TARGET_PNL_GAIN_COLUMN = "neuralpricing_target_pnl_gain"


def _expand_paths(paths: Sequence[str]) -> List[Path]:
    expanded: List[Path] = []
    for path in paths:
        candidate = Path(path)
        if any(ch in path for ch in "*?[]"):
            expanded.extend(sorted(Path().glob(path)))
        elif candidate.exists():
            expanded.append(candidate)
    if not expanded:
        raise FileNotFoundError(f"No files matched patterns {paths}")
    return expanded


def _parse_date_value(value) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    if isinstance(value, tuple) and len(value) == 2:
        symbol = str(value[0]).strip() or None
        ts = pd.to_datetime(value[1], utc=True, errors="coerce")
        return symbol, ts if not pd.isna(ts) else None
    if isinstance(value, str):
        match = BACKTEST_DATE_REGEX.search(value)
        if match:
            symbol = match.group("symbol").strip() or None
            raw_ts = match.group("timestamp")
            ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
            return symbol, ts if not pd.isna(ts) else None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        ts = None
    return None, ts if ts is not None and not pd.isna(ts) else None


def load_backtest_frames(
    csv_patterns: Sequence[str],
    *,
    symbol_filter: Optional[Sequence[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load and normalise backtest CSV exports (e.g., backtest_results/*.csv)."""

    symbol_whitelist = {sym.upper() for sym in symbol_filter} if symbol_filter else None
    frames: List[pd.DataFrame] = []
    for path in _expand_paths(csv_patterns):
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"{path} missing 'date' column required for neural pricing dataset.")
        symbols: List[Optional[str]] = []
        timestamps: List[Optional[pd.Timestamp]] = []
        for value in df["date"]:
            symbol, ts = _parse_date_value(value)
            symbols.append(symbol)
            timestamps.append(ts)
        df["symbol"] = symbols
        df["timestamp"] = timestamps
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["timestamp"]).copy()
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined["symbol"] = combined["symbol"].apply(lambda value: (str(value).strip().upper()) if value else "")
    combined = combined[combined["symbol"] != ""]
    if symbol_whitelist:
        combined = combined[combined["symbol"].str.upper().isin(symbol_whitelist)]
    if start_date:
        combined = combined[combined["timestamp"] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        combined = combined[combined["timestamp"] <= pd.to_datetime(end_date, utc=True)]
    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )
    return combined


@dataclass
class PricingDataset:
    features: torch.Tensor
    targets: torch.Tensor
    frame: pd.DataFrame
    feature_spec: FeatureSpec
    clamp_pct: float

    def to_device(self, device: torch.device) -> "PricingDataset":
        return PricingDataset(
            features=self.features.to(device),
            targets=self.targets.to(device),
            frame=self.frame,
            feature_spec=self.feature_spec,
            clamp_pct=self.clamp_pct,
        )


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    new_df = df.copy()
    for column in columns:
        if column not in new_df.columns:
            continue
        new_df[column] = pd.to_numeric(new_df[column], errors="coerce")
    return new_df


def build_pricing_dataset(
    frame: pd.DataFrame,
    *,
    numeric_columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
    clamp_pct: float = 0.08,
    feature_spec: Optional[FeatureSpec] = None,
) -> PricingDataset:
    """Convert a combined backtest frame into tensors for neural pricing training."""

    if clamp_pct <= 0:
        raise ValueError("clamp_pct must be > 0")
    working = frame.copy()
    working = _coerce_numeric_columns(working, numeric_columns)
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working.sort_values(["symbol", "timestamp"], inplace=True)

    base_low = pd.to_numeric(working.get("maxdiffalwayson_low_price"), errors="coerce")
    base_high = pd.to_numeric(working.get("maxdiffalwayson_high_price"), errors="coerce")
    target_low = pd.to_numeric(working.get("maxdiffprofit_low_price"), errors="coerce")
    target_high = pd.to_numeric(working.get("maxdiffprofit_high_price"), errors="coerce")

    mask = (
        base_low.notna()
        & base_high.notna()
        & target_low.notna()
        & target_high.notna()
        & (base_low > 0)
        & (base_high > 0)
    )
    filtered = working[mask].reset_index(drop=True).copy()
    if filtered.empty:
        raise ValueError("No rows with valid pricing targets were found.")

    low_delta = ((target_low.loc[mask] - base_low.loc[mask]) / base_low.loc[mask]).clip(
        -clamp_pct, clamp_pct
    )
    high_delta = ((target_high.loc[mask] - base_high.loc[mask]) / base_high.loc[mask]).clip(
        -clamp_pct, clamp_pct
    )
    pnl_gain = (
        pd.to_numeric(filtered.get("maxdiff_return"), errors="coerce").fillna(0.0)
        - pd.to_numeric(filtered.get("maxdiffalwayson_return"), errors="coerce").fillna(0.0)
    )

    filtered[TARGET_LOW_COLUMN] = low_delta.to_numpy(dtype=np.float32)
    filtered[TARGET_HIGH_COLUMN] = high_delta.to_numpy(dtype=np.float32)
    filtered[TARGET_PNL_GAIN_COLUMN] = pnl_gain.to_numpy(dtype=np.float32)

    builder = FeatureBuilder(
        numeric_columns=list(numeric_columns),
        categorical_columns=list(categorical_columns),
    )
    if feature_spec is not None:
        builder._spec = feature_spec
        features = builder.transform(filtered)
    else:
        features = builder.fit_transform(filtered)

    tensors = PricingDataset(
        features=torch.from_numpy(features.astype(np.float32)),
        targets=torch.from_numpy(
            filtered[[TARGET_LOW_COLUMN, TARGET_HIGH_COLUMN, TARGET_PNL_GAIN_COLUMN]].to_numpy(
                dtype=np.float32
            )
        ),
        frame=filtered.reset_index(drop=True),
        feature_spec=builder.spec,
        clamp_pct=float(clamp_pct),
    )
    return tensors


def split_dataset_by_date(
    dataset: PricingDataset,
    *,
    validation_fraction: float = 0.2,
) -> Tuple[PricingDataset, PricingDataset]:
    """Chronologically split dataset into train/validation sets."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    ordered = dataset.frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    unique_dates = ordered["timestamp"].sort_values().unique()
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct timestamps required to split dataset.")
    split_index = max(1, int(len(unique_dates) * (1.0 - validation_fraction)))
    split_index = min(split_index, len(unique_dates) - 1)
    cutoff = unique_dates[split_index]

    train_mask = ordered["timestamp"] < cutoff
    val_mask = ~train_mask

    def _subset(mask: pd.Series) -> PricingDataset:
        subset = ordered[mask].reset_index(drop=True)
        builder = FeatureBuilder(
            numeric_columns=list(dataset.feature_spec.numeric_stats.keys()),
            categorical_columns=list(dataset.feature_spec.categorical_levels.keys()),
            add_bias="bias" in dataset.feature_spec.feature_names,
        )
        builder._spec = dataset.feature_spec
        features = builder.transform(subset)
        targets = subset[[TARGET_LOW_COLUMN, TARGET_HIGH_COLUMN, TARGET_PNL_GAIN_COLUMN]].to_numpy(
            dtype=np.float32
        )
        return PricingDataset(
            features=torch.from_numpy(features.astype(np.float32)),
            targets=torch.from_numpy(targets),
            frame=subset,
            feature_spec=dataset.feature_spec,
            clamp_pct=dataset.clamp_pct,
        )

    return _subset(train_mask), _subset(val_mask)


__all__ = [
    "DEFAULT_CATEGORICAL_COLUMNS",
    "DEFAULT_NUMERIC_COLUMNS",
    "PricingDataset",
    "build_pricing_dataset",
    "load_backtest_frames",
    "split_dataset_by_date",
]
