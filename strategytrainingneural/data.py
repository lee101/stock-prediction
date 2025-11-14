from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .feature_builder import FeatureBuilder, FeatureSpec

FORECAST_NUMERIC_COLUMNS = [
    "forecast_move_pct",
    "forecast_volatility_pct",
    "predicted_close",
    "predicted_close_p10",
    "predicted_close_p90",
    "predicted_high",
    "predicted_low",
    "context_close",
]

DEFAULT_NUMERIC_COLUMNS = [
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_ann_return",
    "log_capital",
    "lagged_return",
    "gate_active",
    "rolling_volatility_5",
    "rolling_downside_5",
    "cumulative_return",
    "drawdown_pct",
    "symbol_prev_return",
    "symbol_prev2_return",
    "symbol_recent_return_2",
    "symbol_recent_loss_flag",
    "account_prev_return",
    "account_prev2_return",
    "account_recent_return_2",
    "account_recent_loss_flag",
    *FORECAST_NUMERIC_COLUMNS,
]

DEFAULT_CATEGORICAL_COLUMNS = [
    "strategy",
    "symbol",
    "mode",
    "gate_config",
    "day_class",
    "day_of_week",
]


def augment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric / categorical features required by the trainers."""
    df = df.copy()
    df["lagged_return"] = (
        df.sort_values(["strategy", "date"])
        .groupby("strategy")["daily_return"]
        .shift(1)
        .fillna(0.0)
    )
    df["log_capital"] = np.log1p(df["capital"].clip(lower=1.0))
    df["gate_active"] = (df["gate_config"].fillna("-") != "-").astype(float)
    df["day_of_week"] = df["date"].dt.day_name()
    if "symbol" not in df.columns:
        df["symbol"] = df["strategy"]

    ordered = df.sort_values(["strategy", "date"]).copy()
    grouped = ordered.groupby("strategy", group_keys=False)

    rolling_std = (
        grouped["daily_return"]
        .rolling(window=5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    ordered["rolling_volatility_5"] = rolling_std

    ordered["negative_return"] = ordered["daily_return"].clip(upper=0.0)
    ordered["rolling_downside_5"] = (
        grouped["negative_return"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    first_capital = grouped["capital"].transform("first").replace(0.0, np.nan)
    ordered["cumulative_return"] = (ordered["capital"] / first_capital).fillna(0.0) - 1.0
    rolling_max = grouped["capital"].cummax().replace(0.0, np.nan)
    ordered["drawdown_pct"] = (ordered["capital"] / rolling_max).fillna(0.0) - 1.0

    ordered["symbol_prev_return"] = grouped["daily_return"].shift(1).fillna(0.0)
    ordered["symbol_prev2_return"] = grouped["daily_return"].shift(2).fillna(0.0)
    ordered["symbol_recent_return_2"] = ordered["symbol_prev_return"] + ordered["symbol_prev2_return"]
    ordered["symbol_recent_loss_flag"] = (ordered["symbol_recent_return_2"] <= 0).astype(float)

    date_totals = ordered.groupby("date")["daily_return"].sum().sort_index()
    account_prev = date_totals.shift(1)
    account_prev2 = date_totals.shift(2)
    ordered["account_prev_return"] = ordered["date"].map(account_prev).fillna(0.0)
    ordered["account_prev2_return"] = ordered["date"].map(account_prev2).fillna(0.0)
    ordered["account_recent_return_2"] = ordered["account_prev_return"] + ordered["account_prev2_return"]
    ordered["account_recent_loss_flag"] = (ordered["account_recent_return_2"] <= 0).astype(float)

    df.loc[ordered.index, "rolling_volatility_5"] = ordered["rolling_volatility_5"]
    df.loc[ordered.index, "rolling_downside_5"] = ordered["rolling_downside_5"]
    df.loc[ordered.index, "cumulative_return"] = ordered["cumulative_return"]
    ordered.drop(columns=["negative_return"], inplace=True)
    df.loc[ordered.index, "drawdown_pct"] = ordered["drawdown_pct"]
    df.loc[ordered.index, "symbol_prev_return"] = ordered["symbol_prev_return"]
    df.loc[ordered.index, "symbol_prev2_return"] = ordered["symbol_prev2_return"]
    df.loc[ordered.index, "symbol_recent_return_2"] = ordered["symbol_recent_return_2"]
    df.loc[ordered.index, "symbol_recent_loss_flag"] = ordered["symbol_recent_loss_flag"]
    df.loc[ordered.index, "account_prev_return"] = ordered["account_prev_return"]
    df.loc[ordered.index, "account_prev2_return"] = ordered["account_prev2_return"]
    df.loc[ordered.index, "account_recent_return_2"] = ordered["account_recent_return_2"]
    df.loc[ordered.index, "account_recent_loss_flag"] = ordered["account_recent_loss_flag"]
    for column in FORECAST_NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = df[column].fillna(0.0)
    return df


def load_daily_metrics(
    csv_path: str,
    *,
    strategy_filter: Optional[Sequence[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    asset_class_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the sizing_strategy_daily_metrics.csv artifact and apply optional filters.

    The returned DataFrame always contains:
      - date (datetime64),
      - strategy (str),
      - numeric columns required by DEFAULT_NUMERIC_COLUMNS, and
      - additional helper columns (lagged_return, log_capital, gate_active).
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find metrics csv at {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("metrics csv must include a 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "strategy"]).reset_index(drop=True)

    if strategy_filter:
        allowed = set(strategy_filter)
        df = df[df["strategy"].isin(allowed)].reset_index(drop=True)

    if start_date:
        start_ts = pd.Timestamp(start_date)
        df = df[df["date"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date)
        df = df[df["date"] <= end_ts]

    if asset_class_filter and asset_class_filter.lower() != "all":
        target = asset_class_filter.lower()
        df = df[df["day_class"].str.lower() == target]

    if df.empty:
        raise ValueError("No rows left after applying filters; broaden the selection.")

    df = augment_metrics(df)

    return df.reset_index(drop=True)


@dataclass
class DailyStrategyDataset:
    """Container wrapping tensors used by the neural + boosted trainers."""

    features: torch.Tensor
    daily_returns: torch.Tensor
    day_index: torch.Tensor
    strategy_index: torch.Tensor
    frame: pd.DataFrame
    feature_spec: FeatureSpec
    strategy_vocab: Dict[str, int]
    date_values: List[pd.Timestamp]

    def to_device(self, device: torch.device) -> "DailyStrategyDataset":
        return DailyStrategyDataset(
            features=self.features.to(device),
            daily_returns=self.daily_returns.to(device),
            day_index=self.day_index.to(device),
            strategy_index=self.strategy_index.to(device),
            frame=self.frame,
            feature_spec=self.feature_spec,
            strategy_vocab=self.strategy_vocab,
            date_values=self.date_values,
        )


def build_dataset(
    df: pd.DataFrame,
    *,
    numeric_columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
    feature_builder: Optional[FeatureBuilder] = None,
    feature_spec: Optional[FeatureSpec] = None,
) -> DailyStrategyDataset:
    """
    Convert a metrics dataframe into tensors plus metadata for training.
    """

    working = df.copy()
    date_values = sorted(working["date"].unique())
    date_to_index = {date: idx for idx, date in enumerate(date_values)}
    working["day_index"] = working["date"].map(date_to_index).astype(np.int64)

    strategies = sorted(working["strategy"].unique())
    strategy_to_index = {name: idx for idx, name in enumerate(strategies)}
    working["strategy_index"] = working["strategy"].map(strategy_to_index).astype(np.int64)

    builder = feature_builder or FeatureBuilder(numeric_columns, categorical_columns)
    if feature_spec is not None:
        builder._spec = feature_spec
        features = builder.transform(working)
    else:
        features = builder.fit_transform(working)

    tensors = DailyStrategyDataset(
        features=torch.from_numpy(features.astype(np.float32)),
        daily_returns=torch.from_numpy(
            working["daily_return"].to_numpy(dtype=np.float32)
        ),
        day_index=torch.from_numpy(working["day_index"].to_numpy(dtype=np.int64)),
        strategy_index=torch.from_numpy(
            working["strategy_index"].to_numpy(dtype=np.int64)
        ),
        frame=working,
        feature_spec=builder.spec,
        strategy_vocab=strategy_to_index,
        date_values=list(date_values),
    )
    return tensors


def split_dataset_by_date(
    dataset: DailyStrategyDataset,
    *,
    validation_fraction: float = 0.2,
) -> Tuple[DailyStrategyDataset, DailyStrategyDataset]:
    """
    Split dataset chronologically using the provided validation fraction.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    num_days = len(dataset.date_values)
    split_point = int(num_days * (1.0 - validation_fraction))
    split_point = max(1, min(split_point, num_days - 1))
    cutoff_date = dataset.date_values[split_point]

    train_mask = dataset.frame["date"] < cutoff_date
    val_mask = ~train_mask

    def _subset(mask: pd.Series) -> DailyStrategyDataset:
        subset_frame = dataset.frame[mask].reset_index(drop=True)
        builder = FeatureBuilder(
            list(dataset.feature_spec.numeric_stats.keys()),
            list(dataset.feature_spec.categorical_levels.keys()),
            add_bias="bias" in dataset.feature_spec.feature_names,
        )
        builder._spec = dataset.feature_spec  # reuse fitted spec
        features = builder.transform(subset_frame)
        return DailyStrategyDataset(
            features=torch.from_numpy(features.astype(np.float32)),
            daily_returns=torch.from_numpy(
                subset_frame["daily_return"].to_numpy(dtype=np.float32)
            ),
            day_index=torch.from_numpy(
                subset_frame["day_index"].to_numpy(dtype=np.int64)
            ),
            strategy_index=torch.from_numpy(
                subset_frame["strategy_index"].to_numpy(dtype=np.int64)
            ),
            frame=subset_frame,
            feature_spec=dataset.feature_spec,
            strategy_vocab=dataset.strategy_vocab,
            date_values=dataset.date_values,
        )

    return _subset(train_mask), _subset(val_mask)
