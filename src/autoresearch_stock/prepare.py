from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from src.date_utils import is_nyse_open_on_date
from src.fees import get_fee_for_symbol
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, summarize_scenario_results
from src.symbol_utils import is_crypto_symbol
from src.trade_directions import DEFAULT_ALPACA_LIVE8_STOCKS, resolve_trade_directions
from src.trade_stock_utils import expected_cost_bps
from src.tradinglib.metrics import pnl_metrics

TIME_BUDGET = int(float(os.getenv("AUTORESEARCH_STOCK_TIME_BUDGET_SECONDS", "300")))
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
NEW_YORK = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
DEFAULT_DAILY_SYMBOLS = ("AAPL", "AMD", "AMZN", "GOOG", "NVDA", "PLTR", "MTCH")
DEFAULT_ANNUAL_LEVERAGE_RATE = 0.0625
DEFAULT_MAX_GROSS_LEVERAGE = 2.0


@dataclass(frozen=True)
class TaskConfig:
    frequency: str
    data_root: Path
    recent_data_root: Path | None
    symbols: tuple[str, ...]
    sequence_length: int
    hold_bars: int
    eval_windows: tuple[int, ...]
    recent_overlay_bars: int = 0
    initial_cash: float = 10_000.0
    max_positions: int = 5
    max_volume_fraction: float = 0.01
    min_edge_bps: float = 4.0
    entry_slippage_bps: float = 1.0
    exit_slippage_bps: float = 1.0
    decision_lag_bars: int = 1
    allow_short: bool = True
    close_at_session_end: bool = True
    spread_lookback_days: int = 14
    periods_per_year: float = 252.0
    annual_leverage_rate: float = DEFAULT_ANNUAL_LEVERAGE_RATE
    max_gross_leverage: float = DEFAULT_MAX_GROSS_LEVERAGE
    dashboard_db_path: Path = Path("dashboards/metrics.db")


@dataclass
class ScenarioData:
    name: str
    bars: pd.DataFrame
    action_rows: pd.DataFrame
    features: np.ndarray
    targets: np.ndarray
    symbol_ids: np.ndarray


@dataclass
class PreparedTask:
    config: TaskConfig
    feature_names: tuple[str, ...]
    symbol_to_id: dict[str, int]
    spread_profile_bps: dict[str, float]
    train_features: np.ndarray
    train_targets: np.ndarray
    train_symbol_ids: np.ndarray
    train_weights: np.ndarray
    val_features: np.ndarray
    val_targets: np.ndarray
    val_symbol_ids: np.ndarray
    scenarios: list[ScenarioData]


@dataclass
class SimTrade:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    price: float
    quantity: float
    reason: str


@dataclass
class _Position:
    symbol: str
    side: str
    qty: float
    entry_price: float
    target_price: float
    entry_ts: pd.Timestamp
    bars_held: int = 0


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def parse_int_list(raw: str | None) -> tuple[int, ...]:
    values = []
    for token in parse_csv_list(raw):
        values.append(int(token))
    return tuple(values)


def resolve_task_config(
    *,
    frequency: str,
    symbols: Sequence[str] | None = None,
    data_root: str | Path | None = None,
    recent_data_root: str | Path | None = None,
    sequence_length: int | None = None,
    hold_bars: int | None = None,
    eval_windows: Sequence[int] | None = None,
    recent_overlay_bars: int = 0,
    initial_cash: float = 10_000.0,
    max_positions: int = 5,
    max_volume_fraction: float | None = None,
    min_edge_bps: float = 4.0,
    entry_slippage_bps: float = 1.0,
    exit_slippage_bps: float = 1.0,
    decision_lag_bars: int = 1,
    allow_short: bool = True,
    close_at_session_end: bool | None = None,
    spread_lookback_days: int = 14,
    annual_leverage_rate: float = DEFAULT_ANNUAL_LEVERAGE_RATE,
    max_gross_leverage: float = DEFAULT_MAX_GROSS_LEVERAGE,
    dashboard_db_path: str | Path = "dashboards/metrics.db",
) -> TaskConfig:
    mode = str(frequency).strip().lower()
    if mode not in {"hourly", "daily"}:
        raise ValueError(f"frequency must be one of hourly/daily, got {frequency!r}")

    if mode == "hourly":
        selected_symbols = symbols or DEFAULT_ALPACA_LIVE8_STOCKS
        root = Path(data_root) if data_root is not None else Path("trainingdatahourly/stocks")
        seq_len = int(sequence_length or 32)
        hold = int(hold_bars or 6)
        windows = tuple(int(v) for v in (eval_windows or (35, 140, 420)))
        max_volume = float(max_volume_fraction if max_volume_fraction is not None else 0.01)
        close_eod = bool(True if close_at_session_end is None else close_at_session_end)
        periods_per_year = 252.0 * 7.0
    else:
        selected_symbols = symbols or DEFAULT_DAILY_SYMBOLS
        root = Path(data_root) if data_root is not None else Path("trainingdata")
        seq_len = int(sequence_length or 32)
        hold = int(hold_bars or 5)
        windows = tuple(int(v) for v in (eval_windows or (20, 60, 120)))
        max_volume = float(max_volume_fraction if max_volume_fraction is not None else 0.02)
        close_eod = bool(False if close_at_session_end is None else close_at_session_end)
        periods_per_year = 252.0

    default_symbols = tuple(str(symbol).upper() for symbol in selected_symbols)
    if not default_symbols:
        raise ValueError("At least one symbol is required")

    if seq_len < 4:
        raise ValueError("sequence_length must be at least 4")
    if hold < 1:
        raise ValueError("hold_bars must be positive")
    if int(decision_lag_bars) < 1:
        raise ValueError("decision_lag_bars must be at least 1 for live-like evaluation")
    if not windows:
        raise ValueError("eval_windows must not be empty")
    if min(windows) < 4:
        raise ValueError("eval_windows must all be at least 4 bars")

    return TaskConfig(
        frequency=mode,
        data_root=root,
        recent_data_root=Path(recent_data_root) if recent_data_root is not None else None,
        symbols=default_symbols,
        sequence_length=seq_len,
        hold_bars=hold,
        eval_windows=tuple(sorted(set(int(v) for v in windows))),
        recent_overlay_bars=max(int(recent_overlay_bars), 0),
        initial_cash=float(initial_cash),
        max_positions=int(max_positions),
        max_volume_fraction=max_volume,
        min_edge_bps=float(min_edge_bps),
        entry_slippage_bps=float(entry_slippage_bps),
        exit_slippage_bps=float(exit_slippage_bps),
        decision_lag_bars=int(decision_lag_bars),
        allow_short=bool(allow_short),
        close_at_session_end=close_eod,
        spread_lookback_days=int(spread_lookback_days),
        periods_per_year=float(periods_per_year),
        annual_leverage_rate=float(max(0.0, annual_leverage_rate)),
        max_gross_leverage=float(max(1.0, max_gross_leverage)),
        dashboard_db_path=Path(dashboard_db_path),
    )


def load_live_spread_profile(
    symbols: Sequence[str],
    *,
    db_path: str | Path = "dashboards/metrics.db",
    lookback_days: int = 14,
    now: datetime | None = None,
) -> dict[str, float]:
    cutoff = (now or datetime.now(tz=timezone.utc)) - pd.Timedelta(days=int(lookback_days))
    cutoff_text = cutoff.strftime(ISO_FORMAT)
    db = Path(db_path)
    grouped: dict[str, list[float]] = {str(symbol).upper(): [] for symbol in symbols}

    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            placeholders = ",".join("?" for _ in grouped)
            query = (
                "SELECT symbol, spread_bps FROM spread_observations "
                f"WHERE recorded_at >= ? AND symbol IN ({placeholders}) "
                "AND spread_bps IS NOT NULL ORDER BY recorded_at DESC"
            )
            rows = conn.execute(query, (cutoff_text, *grouped.keys())).fetchall()
            for symbol, spread_bps in rows:
                value = float(spread_bps)
                if math.isfinite(value) and 0.0 < value < 1_000.0:
                    grouped[str(symbol).upper()].append(value)
        finally:
            conn.close()

    profile: dict[str, float] = {}
    for symbol in grouped:
        values = grouped.get(symbol, [])
        if len(values) >= 8:
            chosen = float(np.percentile(np.asarray(values, dtype=np.float64), 75))
        else:
            chosen = float(expected_cost_bps(symbol))
        profile[symbol] = float(min(max(chosen, 1.0), 250.0))
    return profile


def _read_symbol_bars_from_path(path: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).copy()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    if "symbol" not in frame.columns:
        frame["symbol"] = symbol.upper()
    frame["symbol"] = str(symbol).upper()
    if "vwap" not in frame.columns:
        frame["vwap"] = frame["close"]
    else:
        frame["vwap"] = pd.to_numeric(frame["vwap"], errors="coerce").fillna(frame["close"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap"]]


def _overlay_cutoff_timestamp(frame: pd.DataFrame, config: TaskConfig) -> pd.Timestamp | None:
    if frame.empty or int(config.recent_overlay_bars) <= 0:
        return None
    latest_ts = pd.Timestamp(frame["timestamp"].max())
    if config.frequency == "hourly":
        return latest_ts - pd.Timedelta(hours=int(config.recent_overlay_bars))
    return latest_ts - pd.Timedelta(days=int(config.recent_overlay_bars))


def _try_read_symbol_bars_from_path(path: Path | None, symbol: str) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        return _read_symbol_bars_from_path(path, symbol)
    except FileNotFoundError:
        return None


def _load_symbol_bars(symbol: str, config: TaskConfig) -> pd.DataFrame:
    symbol_name = symbol.upper()
    primary_path = Path(config.data_root) / f"{symbol_name}.csv"
    recent_path = Path(config.recent_data_root) / f"{symbol_name}.csv" if config.recent_data_root is not None else None

    parts: list[pd.DataFrame] = []
    primary = _try_read_symbol_bars_from_path(primary_path, symbol_name)
    if primary is not None:
        parts.append(primary)
    recent = _try_read_symbol_bars_from_path(recent_path, symbol_name)
    if recent is not None:
        cutoff = _overlay_cutoff_timestamp(recent, config)
        if cutoff is not None:
            recent = recent.loc[pd.to_datetime(recent["timestamp"], utc=True) >= cutoff].copy()
        if not recent.empty:
            parts.append(recent)

    if not parts:
        missing = [str(primary_path)]
        if recent_path is not None:
            missing.append(str(recent_path))
        raise FileNotFoundError(f"Missing dataset for {symbol_name}; checked: {', '.join(missing)}")

    combined = pd.concat(parts, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).copy()
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return combined[["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap"]]


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window=window, min_periods=max(5, window // 3)).std()
    std = std.replace(0.0, np.nan)
    return (series - mean) / std


def _future_window_extreme(
    values: np.ndarray,
    horizon: int,
    reducer: str,
    *,
    start_offset: int,
) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for index in range(values.shape[0]):
        start = index + int(start_offset)
        stop = min(values.shape[0], index + int(start_offset) + int(horizon))
        if start >= stop:
            continue
        window = values[start:stop]
        out[index] = float(np.max(window) if reducer == "max" else np.min(window))
    return out


def _is_market_bar(timestamp: pd.Timestamp, symbol: str, frequency: str) -> bool:
    if frequency != "hourly":
        return True
    if symbol.endswith("USD"):
        return True
    ts_ny = timestamp.tz_convert(NEW_YORK) if timestamp.tzinfo else timestamp.tz_localize("UTC").tz_convert(NEW_YORK)
    if not is_nyse_open_on_date(ts_ny):
        return False
    current_time = ts_ny.time()
    return MARKET_OPEN <= current_time < MARKET_CLOSE


def _session_key(timestamp: pd.Timestamp, *, frequency: str) -> str:
    if frequency == "daily":
        return timestamp.date().isoformat()
    ts_ny = timestamp.tz_convert(NEW_YORK) if timestamp.tzinfo else timestamp.tz_localize("UTC").tz_convert(NEW_YORK)
    return ts_ny.date().isoformat()


def _feature_names_for_frequency(frequency: str) -> tuple[str, ...]:
    if frequency == "hourly":
        return (
            "ret_1",
            "ret_3",
            "ret_6",
            "ret_24",
            "range_pct",
            "volatility",
            "volume_z",
            "close_vs_vwap",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "spread_bps_norm",
        )
    return (
        "ret_1",
        "ret_5",
        "ret_20",
        "range_pct",
        "volatility",
        "volume_z",
        "close_vs_vwap",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "spread_bps_norm",
    )


def _build_feature_frame(frame: pd.DataFrame, config: TaskConfig, spread_bps: float) -> pd.DataFrame:
    enriched = frame.copy()
    close = enriched["close"].astype(float)
    volume = enriched["volume"].astype(float).replace(0.0, np.nan)
    log_volume = np.log1p(volume).fillna(0.0)
    lookback_fast = 24 if config.frequency == "hourly" else 20
    lookback_slow = 24 if config.frequency == "hourly" else 20

    enriched["ret_1"] = close.pct_change(1)
    if config.frequency == "hourly":
        enriched["ret_3"] = close.pct_change(3)
        enriched["ret_6"] = close.pct_change(6)
        enriched["ret_24"] = close.pct_change(24)
    else:
        enriched["ret_5"] = close.pct_change(5)
        enriched["ret_20"] = close.pct_change(20)
    enriched["range_pct"] = (enriched["high"] - enriched["low"]) / close.replace(0.0, np.nan)
    enriched["volatility"] = enriched["ret_1"].rolling(window=lookback_fast, min_periods=max(5, lookback_fast // 3)).std()
    enriched["volume_z"] = _rolling_zscore(log_volume, lookback_slow).clip(-5.0, 5.0)
    enriched["close_vs_vwap"] = (close / enriched["vwap"].replace(0.0, np.nan)) - 1.0

    ts = pd.to_datetime(enriched["timestamp"], utc=True)
    if config.frequency == "hourly":
        hours = ts.dt.hour.to_numpy(dtype=np.float64)
        weekdays = ts.dt.dayofweek.to_numpy(dtype=np.float64)
        enriched["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
        enriched["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
        enriched["dow_sin"] = np.sin(2.0 * np.pi * weekdays / 7.0)
        enriched["dow_cos"] = np.cos(2.0 * np.pi * weekdays / 7.0)
    else:
        weekdays = ts.dt.dayofweek.to_numpy(dtype=np.float64)
        months = ts.dt.month.to_numpy(dtype=np.float64)
        enriched["dow_sin"] = np.sin(2.0 * np.pi * weekdays / 7.0)
        enriched["dow_cos"] = np.cos(2.0 * np.pi * weekdays / 7.0)
        enriched["month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
        enriched["month_cos"] = np.cos(2.0 * np.pi * months / 12.0)

    high_values = enriched["high"].to_numpy(dtype=np.float64, copy=False)
    low_values = enriched["low"].to_numpy(dtype=np.float64, copy=False)
    close_values = close.to_numpy(dtype=np.float64, copy=False)
    future_high = _future_window_extreme(
        high_values,
        config.hold_bars,
        "max",
        start_offset=max(int(config.decision_lag_bars), 1),
    )
    future_low = _future_window_extreme(
        low_values,
        config.hold_bars,
        "min",
        start_offset=max(int(config.decision_lag_bars), 1),
    )
    future_close = np.full_like(close_values, np.nan, dtype=np.float64)
    exit_offset = max(int(config.decision_lag_bars), 1) + int(config.hold_bars) - 1
    if exit_offset < close_values.shape[0]:
        future_close[:-exit_offset] = close_values[exit_offset:]

    denominator = np.where(close_values > 0.0, close_values, np.nan)
    enriched["future_high_return"] = (future_high / denominator) - 1.0
    enriched["future_low_return"] = (future_low / denominator) - 1.0
    enriched["future_close_return"] = (future_close / denominator) - 1.0
    enriched["spread_bps"] = float(spread_bps)
    enriched["spread_bps_norm"] = float(spread_bps) / 100.0
    enriched["fee_rate"] = float(get_fee_for_symbol(str(enriched["symbol"].iloc[0])))
    enriched["session_key"] = [
        _session_key(pd.Timestamp(ts_value), frequency=config.frequency) for ts_value in enriched["timestamp"]
    ]
    enriched["is_session_last"] = enriched["session_key"].shift(-1) != enriched["session_key"]

    feature_names = _feature_names_for_frequency(config.frequency)
    enriched = enriched.replace([np.inf, -np.inf], np.nan)
    enriched = enriched.dropna(subset=list(feature_names)).reset_index(drop=True)
    return enriched


def _build_sequence_block(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    sequence_length: int,
    row_mask: np.ndarray,
    include_targets: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_matrix = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float32, copy=False)
    targets = frame.loc[:, ["future_high_return", "future_low_return", "future_close_return"]].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    eligible = np.flatnonzero(row_mask)
    valid_indices = eligible[eligible >= (int(sequence_length) - 1)]
    if include_targets:
        finite_targets = np.isfinite(targets[valid_indices]).all(axis=1)
        valid_indices = valid_indices[finite_targets]

    if valid_indices.size == 0:
        empty_features = np.zeros((0, int(sequence_length), len(feature_names)), dtype=np.float32)
        empty_targets = np.zeros((0, 3), dtype=np.float32)
        return empty_features, empty_targets, frame.iloc[0:0].copy()

    sequences = np.stack(
        [feature_matrix[index - sequence_length + 1 : index + 1] for index in valid_indices],
        axis=0,
    ).astype(np.float32, copy=False)
    target_block = targets[valid_indices].astype(np.float32, copy=False)
    rows = frame.iloc[valid_indices].copy().reset_index(drop=True)
    return sequences, target_block, rows


def prepare_task(config: TaskConfig) -> PreparedTask:
    spread_profile = load_live_spread_profile(
        config.symbols,
        db_path=config.dashboard_db_path,
        lookback_days=config.spread_lookback_days,
    )
    frames_by_symbol: dict[str, pd.DataFrame] = {}
    feature_names = _feature_names_for_frequency(config.frequency)

    for symbol in config.symbols:
        raw = _load_symbol_bars(symbol, config)
        enriched = _build_feature_frame(raw, config, spread_profile[symbol])
        min_rows = config.sequence_length + max(config.eval_windows) + config.hold_bars + config.decision_lag_bars + 16
        if len(enriched) < min_rows:
            raise ValueError(
                f"{symbol} has {len(enriched)} usable rows but {min_rows} are required for "
                f"{config.frequency} setup."
            )
        frames_by_symbol[symbol] = enriched

    all_timestamps = sorted(
        {
            pd.Timestamp(ts)
            for frame in frames_by_symbol.values()
            for ts in frame["timestamp"].tolist()
        }
    )
    reserve = max(config.eval_windows) + config.hold_bars + config.decision_lag_bars
    if len(all_timestamps) <= reserve + config.sequence_length + 8:
        raise ValueError("Not enough shared timestamps to form train/eval splits.")

    train_end_ts = all_timestamps[-reserve - 1]
    train_candidate_ts = [ts for ts in all_timestamps if ts <= train_end_ts]
    if len(train_candidate_ts) <= config.sequence_length + 8:
        raise ValueError("Training split is too small after holding out evaluation windows.")

    val_start_idx = max(config.sequence_length + 1, int(len(train_candidate_ts) * 0.85))
    val_start_ts = train_candidate_ts[val_start_idx]

    train_features_blocks: list[np.ndarray] = []
    train_target_blocks: list[np.ndarray] = []
    train_symbol_blocks: list[np.ndarray] = []
    train_weight_blocks: list[np.ndarray] = []
    val_features_blocks: list[np.ndarray] = []
    val_target_blocks: list[np.ndarray] = []
    val_symbol_blocks: list[np.ndarray] = []
    scenarios: list[ScenarioData] = []
    symbol_to_id = {symbol: index for index, symbol in enumerate(config.symbols)}

    for symbol, frame in frames_by_symbol.items():
        ts = pd.to_datetime(frame["timestamp"], utc=True)
        train_mask = (ts <= train_end_ts).to_numpy(dtype=bool) & (ts < val_start_ts).to_numpy(dtype=bool)
        val_mask = (ts <= train_end_ts).to_numpy(dtype=bool) & (ts >= val_start_ts).to_numpy(dtype=bool)

        train_x, train_y, _ = _build_sequence_block(
            frame,
            feature_names=feature_names,
            sequence_length=config.sequence_length,
            row_mask=train_mask,
            include_targets=True,
        )
        if len(train_x):
            weights = np.clip(
                1.0 + 50.0 * np.abs(train_y[:, 2]) + 25.0 * (train_y[:, 0] - train_y[:, 1]),
                1.0,
                12.0,
            ).astype(np.float32, copy=False)
            train_features_blocks.append(train_x)
            train_target_blocks.append(train_y)
            train_symbol_blocks.append(np.full(len(train_x), symbol_to_id[symbol], dtype=np.int64))
            train_weight_blocks.append(weights)

        val_x, val_y, _ = _build_sequence_block(
            frame,
            feature_names=feature_names,
            sequence_length=config.sequence_length,
            row_mask=val_mask,
            include_targets=True,
        )
        if len(val_x):
            val_features_blocks.append(val_x)
            val_target_blocks.append(val_y)
            val_symbol_blocks.append(np.full(len(val_x), symbol_to_id[symbol], dtype=np.int64))

    for window in config.eval_windows:
        scenario_name = f"{config.frequency}_{int(window)}bar"
        start_ts = all_timestamps[-(int(window) + config.hold_bars + config.decision_lag_bars)]
        bar_parts: list[pd.DataFrame] = []
        action_parts: list[pd.DataFrame] = []
        feature_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        symbol_parts: list[np.ndarray] = []

        for symbol, frame in frames_by_symbol.items():
            ts = pd.to_datetime(frame["timestamp"], utc=True)
            bar_part = frame.loc[ts >= start_ts].copy().reset_index(drop=True)
            bar_parts.append(bar_part)
            action_mask = (ts >= start_ts).to_numpy(dtype=bool)
            scenario_x, scenario_y, scenario_rows = _build_sequence_block(
                frame,
                feature_names=feature_names,
                sequence_length=config.sequence_length,
                row_mask=action_mask,
                include_targets=False,
            )
            if len(scenario_x):
                action_parts.append(scenario_rows.reset_index(drop=True))
                feature_parts.append(scenario_x)
                target_parts.append(scenario_y)
                symbol_parts.append(np.full(len(scenario_x), symbol_to_id[symbol], dtype=np.int64))

        scenario = ScenarioData(
            name=scenario_name,
            bars=pd.concat(bar_parts, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True),
            action_rows=(
                pd.concat(action_parts, ignore_index=True).reset_index(drop=True)
                if action_parts
                else pd.DataFrame()
            ),
            features=(
                np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)
                if feature_parts
                else np.zeros((0, config.sequence_length, len(feature_names)), dtype=np.float32)
            ),
            targets=(
                np.concatenate(target_parts, axis=0).astype(np.float32, copy=False)
                if target_parts
                else np.zeros((0, 3), dtype=np.float32)
            ),
            symbol_ids=(
                np.concatenate(symbol_parts, axis=0).astype(np.int64, copy=False)
                if symbol_parts
                else np.zeros((0,), dtype=np.int64)
            ),
        )
        scenarios.append(scenario)

    if not train_features_blocks:
        raise ValueError("No training samples were generated.")

    val_features = (
        np.concatenate(val_features_blocks, axis=0).astype(np.float32, copy=False)
        if val_features_blocks
        else np.zeros((0, config.sequence_length, len(feature_names)), dtype=np.float32)
    )
    val_targets = (
        np.concatenate(val_target_blocks, axis=0).astype(np.float32, copy=False)
        if val_target_blocks
        else np.zeros((0, 3), dtype=np.float32)
    )
    val_symbol_ids = (
        np.concatenate(val_symbol_blocks, axis=0).astype(np.int64, copy=False)
        if val_symbol_blocks
        else np.zeros((0,), dtype=np.int64)
    )

    return PreparedTask(
        config=config,
        feature_names=feature_names,
        symbol_to_id=symbol_to_id,
        spread_profile_bps=spread_profile,
        train_features=np.concatenate(train_features_blocks, axis=0).astype(np.float32, copy=False),
        train_targets=np.concatenate(train_target_blocks, axis=0).astype(np.float32, copy=False),
        train_symbol_ids=np.concatenate(train_symbol_blocks, axis=0).astype(np.int64, copy=False),
        train_weights=np.concatenate(train_weight_blocks, axis=0).astype(np.float32, copy=False),
        val_features=val_features,
        val_targets=val_targets,
        val_symbol_ids=val_symbol_ids,
        scenarios=scenarios,
    )


def _predict_in_batches(
    model: torch.nn.Module,
    *,
    features: np.ndarray,
    symbol_ids: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    if len(features) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            stop = min(len(features), start + int(batch_size))
            feature_batch = torch.from_numpy(features[start:stop]).to(device=device, dtype=torch.float32)
            symbol_batch = torch.from_numpy(symbol_ids[start:stop]).to(device=device, dtype=torch.long)
            output = model(feature_batch, symbol_batch)
            predictions.append(output.detach().cpu().to(dtype=torch.float32).numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32, copy=False)


def _lag_action_frame(actions: pd.DataFrame, bars: pd.DataFrame, lag: int) -> pd.DataFrame:
    if actions.empty or int(lag) <= 0:
        return actions.copy()

    shifted_parts: list[pd.DataFrame] = []
    for symbol, action_part in actions.groupby("symbol", sort=False):
        action_part = action_part.sort_values("timestamp").copy()
        bar_part = bars[bars["symbol"] == symbol].sort_values("timestamp")
        bar_ts = pd.DatetimeIndex(pd.to_datetime(bar_part["timestamp"], utc=True))
        if bar_ts.empty:
            continue

        action_ts = pd.DatetimeIndex(pd.to_datetime(action_part["timestamp"], utc=True))
        start_pos = bar_ts.searchsorted(action_ts, side="left")
        target_pos = start_pos + int(lag)
        valid = (start_pos < len(bar_ts)) & (target_pos < len(bar_ts))
        if not np.any(valid):
            continue

        shifted = action_part.loc[valid].copy()
        shifted["timestamp"] = bar_ts.take(target_pos[valid]).to_numpy()
        shifted_parts.append(shifted)

    if not shifted_parts:
        return actions.iloc[0:0].copy()
    return pd.concat(shifted_parts, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def build_action_frame(
    action_rows: pd.DataFrame,
    predictions: np.ndarray,
    config: TaskConfig,
) -> pd.DataFrame:
    if action_rows.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "side",
                "strength",
                "target_price",
                "spread_bps",
                "expected_edge",
                "buy_price",
                "sell_price",
                "buy_amount",
                "sell_amount",
                "trade_amount",
                "predicted_high_p50_h1",
                "predicted_low_p50_h1",
                "predicted_close_p50_h1",
            ]
        )
    if len(action_rows) != len(predictions):
        raise ValueError("action_rows and predictions must have the same length")

    direction_cache = {
        symbol: resolve_trade_directions(symbol, allow_short=config.allow_short, use_default_groups=True)
        for symbol in action_rows["symbol"].astype(str).unique()
    }
    min_edge = float(config.min_edge_bps) / 1e4
    rows: list[dict[str, Any]] = []

    for row, prediction in zip(action_rows.itertuples(index=False), predictions, strict=True):
        symbol = str(row.symbol).upper()
        close_price = float(row.close)
        spread_bps = float(getattr(row, "spread_bps", 0.0) or 0.0)
        spread_frac = max(spread_bps, 0.0) / 1e4
        entry_cost = spread_frac * 0.5 + (float(config.entry_slippage_bps) / 1e4)
        exit_cost = spread_frac * 0.5 + (float(config.exit_slippage_bps) / 1e4)
        fee_rate = float(getattr(row, "fee_rate", 0.0) or 0.0)
        round_trip_cost = entry_cost + exit_cost + 2.0 * fee_rate

        pred_high = float(np.clip(prediction[0], -0.25, 0.25))
        pred_low = float(np.clip(prediction[1], -0.25, 0.25))
        pred_close = float(np.clip(prediction[2], -0.25, 0.25))
        pred_high = max(pred_high, pred_close, 0.0)
        pred_low = min(pred_low, pred_close, 0.0)
        max_trade_amount = 100.0 if is_crypto_symbol(symbol) else float(config.max_gross_leverage * 100.0)

        up_move = max(pred_high, pred_close, 0.0)
        down_move = max(-pred_low, -pred_close, 0.0)
        directions = direction_cache[symbol]
        long_edge = up_move - round_trip_cost if directions.can_long else float("-inf")
        short_edge = down_move - round_trip_cost if directions.can_short else float("-inf")

        side = "flat"
        strength = 0.0
        expected_edge = 0.0
        target_price = close_price
        buy_price = close_price
        sell_price = close_price
        buy_amount = 0.0
        sell_amount = 0.0

        if max(long_edge, short_edge) > min_edge:
            if long_edge >= short_edge:
                side = "long"
                expected_edge = long_edge
                target_return = min(max(up_move * 0.75, round_trip_cost + min_edge), 0.20)
                target_price = close_price * (1.0 + target_return)
                sell_price = target_price
            else:
                side = "short"
                expected_edge = short_edge
                target_return = min(max(down_move * 0.75, round_trip_cost + min_edge), 0.20)
                target_price = close_price * (1.0 - target_return)
                buy_price = target_price

            strength = float(np.clip(expected_edge / max(min_edge * 4.0, 0.03), 0.0, 1.0))
            if side == "long":
                buy_amount = max_trade_amount * strength
            else:
                sell_amount = max_trade_amount * strength

        rows.append(
            {
                "timestamp": pd.Timestamp(row.timestamp),
                "symbol": symbol,
                "side": side,
                "strength": strength,
                "target_price": float(target_price),
                "spread_bps": spread_bps,
                "expected_edge": float(expected_edge),
                "buy_price": float(buy_price),
                "sell_price": float(sell_price),
                "buy_amount": float(buy_amount),
                "sell_amount": float(sell_amount),
                "trade_amount": float(max(buy_amount, sell_amount)),
                "predicted_high_p50_h1": close_price * (1.0 + pred_high),
                "predicted_low_p50_h1": close_price * (1.0 + pred_low),
                "predicted_close_p50_h1": close_price * (1.0 + pred_close),
            }
        )

    return pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def apply_execution_modifiers(
    actions: pd.DataFrame,
    *,
    buy_price_modifier_bps: float = 0.0,
    sell_price_modifier_bps: float = 0.0,
    amount_modifier_pct: float = 0.0,
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()

    adjusted = actions.copy()
    buy_scale = max(0.01, 1.0 + float(buy_price_modifier_bps) / 1e4)
    sell_scale = max(0.01, 1.0 + float(sell_price_modifier_bps) / 1e4)
    amount_scale = max(0.0, 1.0 + float(amount_modifier_pct) / 100.0)

    adjusted["buy_price"] = adjusted["buy_price"].astype(float) * buy_scale
    adjusted["sell_price"] = adjusted["sell_price"].astype(float) * sell_scale
    adjusted["buy_amount"] = adjusted["buy_amount"].astype(float) * amount_scale
    adjusted["sell_amount"] = adjusted["sell_amount"].astype(float) * amount_scale
    adjusted["trade_amount"] = np.maximum(
        adjusted["buy_amount"].to_numpy(dtype=np.float64, copy=False),
        adjusted["sell_amount"].to_numpy(dtype=np.float64, copy=False),
    )
    adjusted["strength"] = np.clip(
        adjusted["strength"].astype(float) * amount_scale,
        0.0,
        2.0,
    )

    long_mask = adjusted["side"].astype(str).str.lower() == "long"
    short_mask = adjusted["side"].astype(str).str.lower() == "short"
    adjusted.loc[long_mask, "target_price"] = adjusted.loc[long_mask, "sell_price"]
    adjusted.loc[short_mask, "target_price"] = adjusted.loc[short_mask, "buy_price"]
    return adjusted


def _entry_price(raw_open: float, *, side: str, spread_bps: float, slippage_bps: float) -> float:
    adverse = (max(spread_bps, 0.0) * 0.5 + max(slippage_bps, 0.0)) / 1e4
    if side == "short":
        return raw_open * max(1.0 - adverse, 1e-6)
    return raw_open * (1.0 + adverse)


def _exit_price(raw_price: float, *, side: str, spread_bps: float, slippage_bps: float) -> float:
    adverse = (max(spread_bps, 0.0) * 0.5 + max(slippage_bps, 0.0)) / 1e4
    if side == "short":
        return raw_price * (1.0 + adverse)
    return raw_price * max(1.0 - adverse, 1e-6)


def _mark_to_market(positions: dict[str, _Position], closes: dict[str, float]) -> float:
    value = 0.0
    for position in positions.values():
        close_price = float(closes.get(position.symbol, position.entry_price))
        if position.side == "short":
            value += position.qty * (2.0 * position.entry_price - close_price)
        else:
            value += position.qty * close_price
    return value


def _gross_exposure(positions: dict[str, _Position], closes: dict[str, float]) -> float:
    total = 0.0
    for position in positions.values():
        close_price = float(closes.get(position.symbol, position.entry_price))
        total += abs(float(position.qty) * close_price)
    return total


def _symbol_max_gross_leverage(symbol: str, config: TaskConfig) -> float:
    if is_crypto_symbol(symbol):
        return 1.0
    return float(max(1.0, config.max_gross_leverage))


def simulate_actions(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: TaskConfig,
) -> dict[str, Any]:
    merged = bars.merge(actions, on=["timestamp", "symbol"], how="left", suffixes=("", "_act"))
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if merged.empty:
        raise ValueError("Merged scenario bars/actions are empty.")

    cash = float(config.initial_cash)
    positions: dict[str, _Position] = {}
    last_close: dict[str, float] = {}
    trades: list[SimTrade] = []
    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []

    for timestamp, group in merged.groupby("timestamp", sort=True):
        ts = pd.Timestamp(timestamp)
        group_by_symbol = {str(row.symbol).upper(): row for row in group.itertuples(index=False)}

        for row in group.itertuples(index=False):
            last_close[str(row.symbol).upper()] = float(row.close)

        for symbol, position in list(positions.items()):
            row = group_by_symbol.get(symbol)
            if row is None:
                continue
            position.bars_held += 1
            spread_bps = float(getattr(row, "spread_bps", 0.0) or 0.0)
            fee_rate = float(getattr(row, "fee_rate", 0.0) or 0.0)
            exit_reason: str | None = None
            exit_base_price: float | None = None

            if position.side == "long" and float(row.high) >= position.target_price:
                exit_reason = "target"
                exit_base_price = float(position.target_price)
            elif position.side == "short" and float(row.low) <= position.target_price:
                exit_reason = "target"
                exit_base_price = float(position.target_price)
            elif config.close_at_session_end and bool(getattr(row, "is_session_last", False)):
                exit_reason = "session_end"
                exit_base_price = float(row.close)
            elif position.bars_held >= int(config.hold_bars):
                exit_reason = "timeout"
                exit_base_price = float(row.close)

            if exit_reason is None or exit_base_price is None:
                continue

            exit_price = _exit_price(
                exit_base_price,
                side=position.side,
                spread_bps=spread_bps,
                slippage_bps=float(config.exit_slippage_bps),
            )
            if position.side == "short":
                cash += position.qty * (2.0 * position.entry_price - exit_price * (1.0 + fee_rate))
            else:
                cash += position.qty * exit_price * (1.0 - fee_rate)
            trades.append(
                SimTrade(
                    timestamp=ts,
                    symbol=symbol,
                    side="buy_cover" if position.side == "short" else "sell",
                    price=float(exit_price),
                    quantity=float(position.qty),
                    reason=exit_reason,
                )
            )
            positions.pop(symbol, None)

        open_slots = int(config.max_positions) - len(positions)
        if open_slots > 0:
            equity_now = cash + _mark_to_market(positions, last_close)
            candidates: list[dict[str, Any]] = []
            for row in group.itertuples(index=False):
                symbol = str(row.symbol).upper()
                side = str(getattr(row, "side", "flat") or "flat").lower()
                if side not in {"long", "short"} or symbol in positions:
                    continue
                if config.frequency == "hourly" and not _is_market_bar(pd.Timestamp(row.timestamp), symbol, config.frequency):
                    continue

                directions = resolve_trade_directions(symbol, allow_short=config.allow_short, use_default_groups=True)
                if side == "long" and not directions.can_long:
                    continue
                if side == "short" and not directions.can_short:
                    continue

                strength = float(getattr(row, "strength", 0.0) or 0.0)
                if strength <= 0.0:
                    continue
                spread_bps = float(getattr(row, "spread_bps", 0.0) or 0.0)
                fee_rate = float(getattr(row, "fee_rate", 0.0) or 0.0)
                if side == "long":
                    requested_entry = float(getattr(row, "buy_price", row.open) or row.open)
                    if requested_entry <= 0.0:
                        continue
                    if float(row.open) > requested_entry and float(row.low) > requested_entry:
                        continue
                    entry_base_price = min(float(row.open), requested_entry)
                    target_price = float(getattr(row, "sell_price", getattr(row, "target_price", row.close)) or row.close)
                else:
                    requested_entry = float(getattr(row, "sell_price", row.open) or row.open)
                    if requested_entry <= 0.0:
                        continue
                    if float(row.open) < requested_entry and float(row.high) < requested_entry:
                        continue
                    entry_base_price = max(float(row.open), requested_entry)
                    target_price = float(getattr(row, "buy_price", getattr(row, "target_price", row.close)) or row.close)

                entry_price = _entry_price(
                    entry_base_price,
                    side=side,
                    spread_bps=spread_bps,
                    slippage_bps=float(config.entry_slippage_bps),
                )
                if side == "long" and target_price <= entry_price:
                    continue
                if side == "short" and target_price >= entry_price:
                    continue

                max_gross_leverage = _symbol_max_gross_leverage(symbol, config)
                trade_amount = float(getattr(row, "trade_amount", 0.0) or 0.0)
                if trade_amount > 0.0:
                    allocation_fraction = float(np.clip(trade_amount / 100.0, 0.0, max_gross_leverage))
                else:
                    allocation_fraction = float(np.clip(strength, 0.0, max_gross_leverage))
                desired_alloc = max(0.0, equity_now / max(int(config.max_positions), 1)) * allocation_fraction
                current_gross = _gross_exposure(positions, last_close)
                gross_cap = max(0.0, equity_now) * max_gross_leverage
                remaining_gross = max(0.0, gross_cap - current_gross)
                volume_cap = max(0.0, float(row.volume) * float(row.open) * float(config.max_volume_fraction))
                allocation = min(desired_alloc, volume_cap, remaining_gross)
                qty = allocation / (entry_price * (1.0 + fee_rate)) if entry_price > 0.0 else 0.0
                if qty <= 0.0 or not math.isfinite(qty):
                    continue

                candidates.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "edge": float(getattr(row, "expected_edge", 0.0) or 0.0),
                        "entry_price": float(entry_price),
                        "target_price": float(target_price),
                        "qty": float(qty),
                        "fee_rate": fee_rate,
                    }
                )

            candidates.sort(key=lambda item: item["edge"], reverse=True)
            for candidate in candidates[:open_slots]:
                cost = candidate["qty"] * candidate["entry_price"] * (1.0 + candidate["fee_rate"])
                cash -= cost
                positions[candidate["symbol"]] = _Position(
                    symbol=candidate["symbol"],
                    side=candidate["side"],
                    qty=float(candidate["qty"]),
                    entry_price=float(candidate["entry_price"]),
                    target_price=float(candidate["target_price"]),
                    entry_ts=ts,
                )
                trades.append(
                    SimTrade(
                        timestamp=ts,
                        symbol=candidate["symbol"],
                        side="short_sell" if candidate["side"] == "short" else "buy",
                        price=float(candidate["entry_price"]),
                        quantity=float(candidate["qty"]),
                        reason="entry",
                    )
                )

        equity_now = cash + _mark_to_market(positions, last_close)
        gross_exposure = _gross_exposure(positions, last_close)
        leverage_ratio = 0.0 if equity_now <= 0.0 else gross_exposure / max(equity_now, 1e-9)
        financing_cost = (
            max(0.0, leverage_ratio - 1.0)
            * float(config.annual_leverage_rate)
            / max(float(config.periods_per_year), 1.0)
            * max(equity_now, 0.0)
        )
        cash -= financing_cost
        equity_index.append(ts)
        equity_values.append(float(cash + _mark_to_market(positions, last_close)))

    final_ts = pd.Timestamp(merged["timestamp"].max())
    for symbol, position in list(positions.items()):
        close_price = float(last_close.get(symbol, position.entry_price))
        row = merged.loc[merged["symbol"] == symbol].iloc[-1]
        exit_price = _exit_price(
            close_price,
            side=position.side,
            spread_bps=float(row.get("spread_bps", 0.0) or 0.0),
            slippage_bps=float(config.exit_slippage_bps),
        )
        fee_rate = float(row.get("fee_rate", 0.0) or 0.0)
        if position.side == "short":
            cash += position.qty * (2.0 * position.entry_price - exit_price * (1.0 + fee_rate))
        else:
            cash += position.qty * exit_price * (1.0 - fee_rate)
        trades.append(
            SimTrade(
                timestamp=final_ts,
                symbol=symbol,
                side="buy_cover" if position.side == "short" else "sell",
                price=float(exit_price),
                quantity=float(position.qty),
                reason="final_close",
            )
        )
        positions.pop(symbol, None)

    if equity_values:
        equity_values[-1] = float(cash)

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_index))
    metrics = pnl_metrics(equity_curve=equity_curve.values, periods_per_year=float(config.periods_per_year))
    drawdown_pct = abs(float(metrics.max_drawdown) * 100.0)
    pnl_smoothness = compute_pnl_smoothness_from_equity(equity_curve.values)
    summary = {
        "return_pct": float(metrics.total_return * 100.0),
        "annualized_return_pct": float(metrics.annualized_return * 100.0),
        "sortino": float(metrics.sortino),
        "max_drawdown_pct": drawdown_pct,
        "pnl_smoothness": float(pnl_smoothness),
        "trade_count": float(len([trade for trade in trades if trade.reason == "entry"])),
        "equity_curve": equity_curve,
        "trades": trades,
    }
    return summary


def evaluate_model(
    model: torch.nn.Module,
    task: PreparedTask,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, Any]:
    scenario_rows: list[dict[str, float]] = []
    scenario_outputs: list[dict[str, Any]] = []
    total_trade_count = 0

    for scenario in task.scenarios:
        predictions = _predict_in_batches(
            model,
            features=scenario.features,
            symbol_ids=scenario.symbol_ids,
            device=device,
            batch_size=batch_size,
        )
        actions = build_action_frame(scenario.action_rows, predictions, task.config)
        actions = _lag_action_frame(actions, scenario.bars, int(task.config.decision_lag_bars))
        result = simulate_actions(scenario.bars, actions, task.config)
        total_trade_count += int(result["trade_count"])
        scenario_rows.append(
            {
                "sortino": float(result["sortino"]),
                "return_pct": float(result["return_pct"]),
                "annualized_return_pct": float(result["annualized_return_pct"]),
                "max_drawdown_pct": float(result["max_drawdown_pct"]),
                "pnl_smoothness": float(result["pnl_smoothness"]),
                "trade_count": float(result["trade_count"]),
            }
        )
        scenario_outputs.append(
            {
                "name": scenario.name,
                "metrics": {key: value for key, value in result.items() if key not in {"equity_curve", "trades"}},
                "actions": actions,
                "equity_curve": result["equity_curve"],
                "trades": result["trades"],
            }
        )

    robust_summary = summarize_scenario_results(scenario_rows)
    robust_summary["total_trade_count"] = float(total_trade_count)
    return {
        "summary": robust_summary,
        "scenarios": scenario_outputs,
    }


def task_summary(task: PreparedTask) -> dict[str, Any]:
    recent_data_root = task.config.recent_data_root
    return {
        "config": {
            **asdict(task.config),
            "data_root": str(task.config.data_root),
            "recent_data_root": None if recent_data_root is None else str(recent_data_root),
            "dashboard_db_path": str(task.config.dashboard_db_path),
        },
        "feature_count": len(task.feature_names),
        "train_samples": int(len(task.train_features)),
        "val_samples": int(len(task.val_features)),
        "scenario_names": [scenario.name for scenario in task.scenarios],
        "spread_profile_bps": task.spread_profile_bps,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and inspect the autoresearch-style stock planner task.")
    parser.add_argument("--frequency", choices=("hourly", "daily"), default="hourly")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--recent-data-root", default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--hold-bars", type=int, default=None)
    parser.add_argument("--eval-windows", default="")
    parser.add_argument("--recent-overlay-bars", type=int, default=0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-volume-fraction", type=float, default=None)
    parser.add_argument("--min-edge-bps", type=float, default=4.0)
    parser.add_argument("--entry-slippage-bps", type=float, default=1.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=1.0)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--disable-short", action="store_true")
    parser.add_argument("--spread-lookback-days", type=int, default=14)
    parser.add_argument("--dashboard-db", default="dashboards/metrics.db")
    args = parser.parse_args(argv)

    config = resolve_task_config(
        frequency=args.frequency,
        symbols=parse_csv_list(args.symbols) or None,
        data_root=args.data_root,
        recent_data_root=args.recent_data_root,
        sequence_length=args.sequence_length,
        hold_bars=args.hold_bars,
        eval_windows=parse_int_list(args.eval_windows) or None,
        recent_overlay_bars=args.recent_overlay_bars,
        max_positions=args.max_positions,
        max_volume_fraction=args.max_volume_fraction,
        min_edge_bps=args.min_edge_bps,
        entry_slippage_bps=args.entry_slippage_bps,
        exit_slippage_bps=args.exit_slippage_bps,
        decision_lag_bars=args.decision_lag_bars,
        allow_short=not bool(args.disable_short),
        spread_lookback_days=args.spread_lookback_days,
        dashboard_db_path=args.dashboard_db,
    )
    prepared = prepare_task(config)
    print(json.dumps(task_summary(prepared), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
