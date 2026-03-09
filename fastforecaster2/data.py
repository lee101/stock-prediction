from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import FastForecaster2Config

_REQUIRED_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close")
_OPTIONAL_NUMERIC_COLUMNS: Tuple[str, ...] = ("volume", "vwap")
_SYMBOL_RE = re.compile(r"^[A-Z0-9._-]+$")


@dataclass
class SymbolSeries:
    symbol: str
    timestamps: np.ndarray
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    features: np.ndarray
    train_end: int
    val_end: int


@dataclass(frozen=True)
class WindowRef:
    symbol_idx: int
    start: int


@dataclass
class DataBundle:
    train_dataset: "ForecastWindowDataset"
    val_dataset: "ForecastWindowDataset"
    test_dataset: "ForecastWindowDataset"
    feature_names: Tuple[str, ...]
    symbols: Tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)


class ForecastWindowDataset(Dataset):
    """Sliding-window dataset for multi-symbol forecasting."""

    def __init__(
        self,
        *,
        series: Sequence[SymbolSeries],
        windows: Sequence[WindowRef],
        lookback: int,
        horizon: int,
    ) -> None:
        self.series = tuple(series)
        self.windows = tuple(windows)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        ref = self.windows[idx]
        record = self.series[ref.symbol_idx]
        left = ref.start
        context_right = left + self.lookback
        target_right = context_right + self.horizon

        x = record.features[left:context_right]
        base_close = float(record.close[context_right - 1])
        target_close = record.close[context_right:target_right]

        safe_base = base_close if abs(base_close) > 1e-8 else 1e-8
        target_return = (target_close / safe_base) - 1.0

        return (
            torch.from_numpy(x.astype(np.float32, copy=False)),
            torch.from_numpy(target_return.astype(np.float32, copy=False)),
            torch.from_numpy(target_close.astype(np.float32, copy=False)),
            torch.tensor(base_close, dtype=torch.float32),
            torch.tensor(ref.symbol_idx, dtype=torch.long),
        )


def _resolve_timestamp_column(df: pd.DataFrame, csv_path: Path) -> pd.Series:
    if "timestamp" in df.columns:
        raw = df["timestamp"]
    elif "timestamps" in df.columns:
        raw = df["timestamps"]
    else:
        raise ValueError(f"{csv_path} missing timestamp column ('timestamp' or 'timestamps').")
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if parsed.isna().all():
        raise ValueError(f"{csv_path} has no valid timestamp values.")
    return parsed


def _fill_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(index=df.index)
    for col in columns:
        if col in df.columns:
            series = df[col]
        else:
            series = 0.0
        frame[col] = pd.to_numeric(series, errors="coerce")
    frame = frame.ffill().bfill().fillna(0.0)
    return frame


def _build_features(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    *,
    include_time_features: bool,
) -> tuple[np.ndarray, Tuple[str, ...]]:
    open_ = frame["open"].to_numpy(dtype=np.float64)
    high_ = frame["high"].to_numpy(dtype=np.float64)
    low_ = frame["low"].to_numpy(dtype=np.float64)
    close_ = frame["close"].to_numpy(dtype=np.float64)
    volume_ = frame.get("volume", pd.Series(np.zeros(len(frame), dtype=np.float64))).to_numpy(dtype=np.float64)
    vwap_ = frame.get("vwap", pd.Series(close_)).to_numpy(dtype=np.float64)

    price_stack = np.stack([open_, high_, low_, close_, vwap_], axis=1)
    log_price = np.log(np.clip(price_stack, 1e-8, None))
    price_returns = np.diff(log_price, axis=0, prepend=log_price[:1])

    volume_log = np.log1p(np.clip(volume_, 0.0, None))
    volume_delta = np.diff(volume_log, prepend=volume_log[:1])

    spread = (high_ - low_) / np.clip(np.abs(close_), 1e-8, None)
    body = (close_ - open_) / np.clip(np.abs(open_), 1e-8, None)
    vwap_spread = (vwap_ - close_) / np.clip(np.abs(close_), 1e-8, None)

    feature_columns: List[np.ndarray] = [
        price_returns[:, 0],
        price_returns[:, 1],
        price_returns[:, 2],
        price_returns[:, 3],
        price_returns[:, 4],
        volume_log,
        volume_delta,
        spread,
        body,
        vwap_spread,
    ]
    feature_names: List[str] = [
        "ret_open",
        "ret_high",
        "ret_low",
        "ret_close",
        "ret_vwap",
        "log_volume",
        "delta_log_volume",
        "spread",
        "body",
        "vwap_spread",
    ]

    if include_time_features:
        hours = timestamps.dt.hour.to_numpy(dtype=np.float64)
        weekdays = timestamps.dt.dayofweek.to_numpy(dtype=np.float64)
        months = timestamps.dt.month.to_numpy(dtype=np.float64)

        feature_columns.extend(
            [
                np.sin(2.0 * np.pi * hours / 24.0),
                np.cos(2.0 * np.pi * hours / 24.0),
                np.sin(2.0 * np.pi * weekdays / 7.0),
                np.cos(2.0 * np.pi * weekdays / 7.0),
                np.sin(2.0 * np.pi * (months - 1.0) / 12.0),
                np.cos(2.0 * np.pi * (months - 1.0) / 12.0),
            ]
        )
        feature_names.extend(
            [
                "hour_sin",
                "hour_cos",
                "weekday_sin",
                "weekday_cos",
                "month_sin",
                "month_cos",
            ]
        )

    feature_matrix = np.column_stack(feature_columns).astype(np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_matrix, tuple(feature_names)


def _split_boundaries(total_rows: int, val_fraction: float, test_fraction: float, horizon: int) -> tuple[int, int] | None:
    min_tail = max(8, 4 * horizon)
    test_rows = max(min_tail, int(round(total_rows * test_fraction)))
    val_rows = max(min_tail, int(round(total_rows * val_fraction)))

    total_tail = val_rows + test_rows
    if total_tail >= total_rows - (2 * horizon):
        reserve = max(2 * horizon, total_rows // 5)
        if reserve >= total_rows:
            return None
        total_tail = total_rows - reserve
        val_rows = total_tail // 2
        test_rows = total_tail - val_rows

    train_end = total_rows - (val_rows + test_rows)
    val_end = total_rows - test_rows

    if train_end <= 0 or val_end <= train_end:
        return None
    return train_end, val_end


def _list_symbol_paths(config: FastForecaster2Config) -> list[tuple[str, Path]]:
    if not config.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {config.data_dir}")

    symbol_filter = set(config.symbols or [])
    symbol_paths: list[tuple[str, Path]] = []
    skipped_invalid: list[str] = []
    for csv_path in sorted(config.data_dir.glob("*.csv")):
        if not csv_path.is_file():
            continue
        symbol = csv_path.stem.upper()
        if not _SYMBOL_RE.fullmatch(symbol):
            skipped_invalid.append(csv_path.name)
            continue
        if symbol_filter and symbol not in symbol_filter:
            continue
        symbol_paths.append((symbol, csv_path))

    if skipped_invalid:
        preview = ", ".join(skipped_invalid[:5])
        extra = "" if len(skipped_invalid) <= 5 else f", +{len(skipped_invalid) - 5} more"
        print(
            f"[fastforecaster2:data] Skipped {len(skipped_invalid)} invalid symbol filenames: {preview}{extra}"
        )

    if not symbol_paths:
        raise ValueError(f"No symbol CSV files found in {config.data_dir}")

    if config.max_symbols > 0:
        symbol_paths = symbol_paths[: config.max_symbols]
    return symbol_paths


def _build_windows_for_symbol(
    *,
    symbol_idx: int,
    total_rows: int,
    lookback: int,
    horizon: int,
    train_end: int,
    val_end: int,
    train_stride: int,
    eval_stride: int,
    max_train_windows: int | None,
    max_eval_windows: int | None,
) -> tuple[list[WindowRef], list[WindowRef], list[WindowRef]]:
    last_start = total_rows - lookback - horizon
    if last_start < 0:
        return [], [], []

    train_last = train_end - lookback - horizon
    val_last = val_end - lookback - horizon

    train_refs: list[WindowRef] = []
    val_refs: list[WindowRef] = []
    test_refs: list[WindowRef] = []

    if train_last >= 0:
        for start in range(0, train_last + 1, train_stride):
            train_refs.append(WindowRef(symbol_idx=symbol_idx, start=start))

    val_start = max(0, train_end - lookback - horizon + 1)
    if val_last >= val_start:
        for start in range(val_start, val_last + 1, eval_stride):
            target_end = start + lookback + horizon
            if train_end < target_end <= val_end:
                val_refs.append(WindowRef(symbol_idx=symbol_idx, start=start))

    test_start = max(0, val_end - lookback - horizon + 1)
    if last_start >= test_start:
        for start in range(test_start, last_start + 1, eval_stride):
            target_end = start + lookback + horizon
            if val_end < target_end <= total_rows:
                test_refs.append(WindowRef(symbol_idx=symbol_idx, start=start))

    train_refs = _downsample_refs(train_refs, max_train_windows)
    val_refs = _downsample_refs(val_refs, max_eval_windows)
    test_refs = _downsample_refs(test_refs, max_eval_windows)

    return train_refs, val_refs, test_refs


def _downsample_refs(refs: list[WindowRef], limit: int | None) -> list[WindowRef]:
    if limit is None or limit <= 0 or len(refs) <= limit:
        return refs
    if limit == 1:
        return [refs[len(refs) // 2]]
    picks = np.linspace(0, len(refs) - 1, limit, dtype=np.int64)
    return [refs[int(i)] for i in picks]


def _compute_normalisation_stats(series: Sequence[SymbolSeries]) -> tuple[np.ndarray, np.ndarray]:
    sum_vec: np.ndarray | None = None
    sq_sum_vec: np.ndarray | None = None
    count = 0

    for entry in series:
        train_slice = entry.features[: entry.train_end]
        if train_slice.size == 0:
            continue
        train_slice = train_slice.astype(np.float64, copy=False)
        if sum_vec is None:
            sum_vec = np.zeros(train_slice.shape[1], dtype=np.float64)
            sq_sum_vec = np.zeros(train_slice.shape[1], dtype=np.float64)
        sum_vec += train_slice.sum(axis=0)
        sq_sum_vec += np.square(train_slice).sum(axis=0)
        count += train_slice.shape[0]

    if sum_vec is None or sq_sum_vec is None or count == 0:
        raise RuntimeError("Unable to compute training feature statistics: no valid training rows available.")

    mean = sum_vec / float(count)
    var = np.maximum((sq_sum_vec / float(count)) - np.square(mean), 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def build_data_bundle(config: FastForecaster2Config) -> DataBundle:
    symbol_paths = _list_symbol_paths(config)

    series_records: list[SymbolSeries] = []
    feature_names: tuple[str, ...] | None = None

    for symbol, csv_path in symbol_paths:
        df = pd.read_csv(csv_path)
        timestamps = _resolve_timestamp_column(df, csv_path)
        work_df = df.copy()
        work_df["__timestamp__"] = timestamps
        work_df = work_df.dropna(subset=["__timestamp__"]).sort_values("__timestamp__")
        work_df = work_df.drop_duplicates(subset=["__timestamp__"], keep="last").reset_index(drop=True)

        for col in _REQUIRED_COLUMNS:
            if col not in work_df.columns:
                raise ValueError(f"{csv_path} missing required column '{col}'")
        numeric_cols = list(_REQUIRED_COLUMNS) + list(_OPTIONAL_NUMERIC_COLUMNS)
        numeric_frame = _fill_numeric_frame(work_df, numeric_cols)

        close = numeric_frame[config.target_column].to_numpy(dtype=np.float32)
        total_rows = len(close)
        if total_rows < config.min_rows_per_symbol:
            continue

        split_bounds = _split_boundaries(total_rows, config.val_fraction, config.test_fraction, config.horizon)
        if split_bounds is None:
            continue
        train_end, val_end = split_bounds

        if train_end <= (config.lookback + config.horizon):
            continue

        features, names = _build_features(
            numeric_frame,
            work_df["__timestamp__"],
            include_time_features=config.include_time_features,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Inconsistent feature layout detected between symbols.")

        series_records.append(
            SymbolSeries(
                symbol=symbol,
                timestamps=work_df["__timestamp__"].to_numpy(),
                open_=numeric_frame["open"].to_numpy(dtype=np.float32),
                high=numeric_frame["high"].to_numpy(dtype=np.float32),
                low=numeric_frame["low"].to_numpy(dtype=np.float32),
                close=close,
                features=features,
                train_end=train_end,
                val_end=val_end,
            )
        )

    if not series_records:
        raise RuntimeError("No symbols satisfied FastForecaster2 dataset requirements.")

    feature_mean, feature_std = _compute_normalisation_stats(series_records)

    for entry in series_records:
        normed = (entry.features - feature_mean) / feature_std
        entry.features = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    train_windows: list[WindowRef] = []
    val_windows: list[WindowRef] = []
    test_windows: list[WindowRef] = []

    for symbol_idx, entry in enumerate(series_records):
        train_refs, val_refs, test_refs = _build_windows_for_symbol(
            symbol_idx=symbol_idx,
            total_rows=len(entry.close),
            lookback=config.lookback,
            horizon=config.horizon,
            train_end=entry.train_end,
            val_end=entry.val_end,
            train_stride=config.train_stride,
            eval_stride=config.eval_stride,
            max_train_windows=config.max_train_windows_per_symbol,
            max_eval_windows=config.max_eval_windows_per_symbol,
        )
        train_windows.extend(train_refs)
        val_windows.extend(val_refs)
        test_windows.extend(test_refs)

    if not train_windows:
        raise RuntimeError("No training windows were produced. Reduce lookback/horizon or adjust data filters.")
    if not val_windows:
        raise RuntimeError("No validation windows were produced. Reduce eval_stride or val_fraction.")
    if not test_windows:
        raise RuntimeError("No test windows were produced. Reduce eval_stride or test_fraction.")

    symbols = tuple(entry.symbol for entry in series_records)
    assert feature_names is not None

    return DataBundle(
        train_dataset=ForecastWindowDataset(
            series=series_records,
            windows=train_windows,
            lookback=config.lookback,
            horizon=config.horizon,
        ),
        val_dataset=ForecastWindowDataset(
            series=series_records,
            windows=val_windows,
            lookback=config.lookback,
            horizon=config.horizon,
        ),
        test_dataset=ForecastWindowDataset(
            series=series_records,
            windows=test_windows,
            lookback=config.lookback,
            horizon=config.horizon,
        ),
        feature_names=feature_names,
        symbols=symbols,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
