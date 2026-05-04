#!/usr/bin/env python3
"""Benchmark a BitBank-style hourly XGB bridge on stock data.

This is a research harness, not a live entry point.  It ports the useful
mechanics from ``../bitbankgo/scripts/train_xgb_hourly_bridge.py``:

* horizon-specific XGBoost return regressors,
* validation-only threshold selection per symbol and horizon,
* sparse-trade and drawdown penalties when choosing thresholds.

Unlike the BitBank signal generator, this script keeps the final benchmark
out-of-sample: thresholds are selected on ``(train_end, val_end]`` and rolling
100-trading-day windows are scored only after ``val_end``.  Execution uses
long/short limit-style fills with ``decision_lag`` and adverse entry/exit
prices.  The portfolio scheduler prefers balanced books with a small long bias
so the search cannot win by becoming an all-long beta proxy.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.artifacts import write_json_atomic  # noqa: E402

from src.daily_stock_defaults import DEFAULT_SYMBOLS  # noqa: E402


STOCK_BARS_PER_DAY = 6.5
STOCK_BARS_PER_MONTH = 21.0 * STOCK_BARS_PER_DAY

FEATURE_COLS = [
    "ret_1h",
    "ret_2h",
    "ret_4h",
    "ret_8h",
    "ret_12h",
    "ret_24h",
    "ret_48h",
    "vol_4h",
    "vol_12h",
    "vol_24h",
    "vol_72h",
    "atr_4h",
    "atr_12h",
    "range_24h",
    "range_72h",
    "dolvol_8h_log",
    "volume_z_24h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    horizon: int
    threshold: float
    mode: str
    val_score: float
    val_monthly_return: float
    val_max_drawdown: float
    val_sortino: float
    val_trades: int
    val_win_rate: float
    val_long_trades: int
    val_short_trades: int


@dataclass(frozen=True)
class CandidateTrade:
    symbol: str
    side: int
    horizon: int
    threshold: float
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    edge: float
    entry_open: float
    exit_close: float


@dataclass(frozen=True)
class WindowResult:
    start: str
    end: str
    total_return: float
    monthly_return: float
    max_drawdown: float
    trades: int
    long_trades: int
    short_trades: int


@dataclass(frozen=True)
class SlippageSummary:
    slippage_bps: float
    median_monthly_return: float
    p10_monthly_return: float
    max_drawdown: float
    negative_windows: int
    n_windows: int
    median_trades: float
    median_short_fraction: float


@dataclass(frozen=True)
class CrossSectionalConfig:
    horizon: int
    threshold: float
    val_score: float
    val_median_monthly_return: float
    val_p10_monthly_return: float
    val_max_drawdown: float
    val_median_trades: float
    val_median_short_fraction: float


def _parse_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def _parse_int_csv(raw: str, name: str) -> tuple[list[int], str | None]:
    values: list[int] = []
    try:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                values.append(int(part))
    except ValueError:
        return [], f"{name} must be a comma-separated integer list"
    if not values:
        return [], f"{name} must not be empty"
    return values, None


def _parse_float_csv(raw: str, name: str) -> tuple[list[float], str | None]:
    values: list[float] = []
    try:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                values.append(float(part))
    except ValueError:
        return [], f"{name} must be a comma-separated finite numeric list"
    if not values:
        return [], f"{name} must not be empty"
    if any(not math.isfinite(value) for value in values):
        return [], f"{name} must contain only finite values"
    return values, None


def _finite_nonnegative(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        return f"{name} must be finite and non-negative"
    return None


def _finite_positive(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        return f"{name} must be finite and positive"
    return None


def validate_args(args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    parsed: dict[str, object] = {}

    eval_symbols = _parse_csv_list(args.symbols)
    if not eval_symbols:
        failures.append("symbols must not be empty")
    parsed["eval_symbols"] = eval_symbols

    horizons, failure = _parse_int_csv(args.horizons, "horizons")
    if failure is not None:
        failures.append(failure)
    elif any(h <= 0 for h in horizons):
        failures.append("horizons must contain only positive integers")
    parsed["horizons"] = horizons

    thresholds, failure = _parse_float_csv(args.thresholds, "thresholds")
    if failure is not None:
        failures.append(failure)
    elif any(t < 0.0 for t in thresholds):
        failures.append("thresholds must contain only non-negative values")
    parsed["thresholds"] = thresholds

    slippages, failure = _parse_float_csv(args.slippage_bps_grid, "slippage_bps_grid")
    if failure is not None:
        failures.append(failure)
    elif any(s < 0.0 for s in slippages):
        failures.append("slippage_bps_grid must contain only non-negative values")
    parsed["slippages"] = slippages

    if args.selection_slippage_bps is None:
        parsed["selection_slippage"] = max(slippages) if slippages else 0.0
    else:
        selection_failure = _finite_nonnegative(args.selection_slippage_bps, "selection_slippage_bps")
        if selection_failure is not None:
            failures.append(selection_failure)
        parsed["selection_slippage"] = float(args.selection_slippage_bps)

    for attr in (
        "fee_rate",
        "fill_buffer_bps",
        "opportunistic_watch_n",
        "opportunistic_entry_discount_bps",
        "eta",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "short_exposure_scale",
        "target_short_fraction",
        "total_short_fraction_cap",
        "balance_penalty",
    ):
        failure = _finite_nonnegative(getattr(args, attr), attr)
        if failure is not None:
            failures.append(failure)
    if float(args.short_exposure_scale) > 1.0:
        failures.append("short_exposure_scale must be <= 1")
    if not 0.0 <= float(args.target_short_fraction) <= 0.5:
        failures.append("target_short_fraction must be between 0 and 0.5")
    if not 0.0 <= float(args.total_short_fraction_cap) <= 1.0:
        failures.append("total_short_fraction_cap must be between 0 and 1")

    for attr in ("leverage",):
        failure = _finite_positive(getattr(args, attr), attr)
        if failure is not None:
            failures.append(failure)

    for attr in (
        "max_train_symbols",
        "min_bars",
        "min_val_trades",
        "window_bars",
        "stride_bars",
        "max_positions",
        "opportunistic_watch_bars",
        "num_boost_round",
        "early_stopping_rounds",
        "max_depth",
    ):
        if int(getattr(args, attr)) <= 0:
            failures.append(f"{attr} must be positive")

    if int(args.decision_lag) < 2:
        failures.append("decision_lag must be >= 2 for production-realism benchmarking")

    try:
        train_start = _parse_timestamp(args.train_start) if args.train_start else None
        train_end = _parse_timestamp(args.train_end)
        val_end = _parse_timestamp(args.val_end)
        test_end = _parse_timestamp(args.test_end)
    except Exception:
        failures.append("train_start/train_end/val_end/test_end must be valid timestamps")
        train_start = None
        train_end = val_end = test_end = pd.Timestamp("1970-01-01", tz="UTC")
    if train_start is not None and train_start > train_end:
        failures.append("train_start must be <= train_end")
    if not (train_end < val_end < test_end):
        failures.append("date splits must satisfy train_end < val_end < test_end")
    parsed["train_start"] = train_start
    parsed["train_end"] = train_end
    parsed["val_end"] = val_end
    parsed["test_end"] = test_end

    return failures, parsed


def _read_ohlcv(path: Path, symbol: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df.columns = [str(c).strip().lower() for c in df.columns]
    ts_col = next((c for c in ("timestamp", "date", "datetime", "time") if c in df.columns), None)
    if ts_col is None or not {"open", "high", "low", "close"}.issubset(df.columns):
        return None
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    if df.empty:
        return None
    df["symbol"] = symbol
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


def _iter_symbol_files(
    data_root: Path,
    *,
    eval_symbols: Sequence[str],
    max_train_symbols: int,
) -> Iterable[tuple[str, Path]]:
    eval_set = {s.upper() for s in eval_symbols}
    yielded: set[str] = set()
    for symbol in sorted(eval_set):
        path = data_root / f"{symbol}.csv"
        if path.exists():
            yielded.add(symbol)
            yield symbol, path
    for path in sorted(data_root.glob("*.csv")):
        symbol = path.stem.upper()
        if "," in symbol:
            continue
        if symbol in yielded:
            continue
        yielded.add(symbol)
        yield symbol, path
        if len(yielded) >= max_train_symbols:
            return


def build_bitbankgo_hourly_features(df: pd.DataFrame, *, max_horizon: int = 6) -> pd.DataFrame:
    """Build leak-free hourly features plus close-to-close horizon targets."""
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float).fillna(0.0)
    prev = close.shift(1)

    out = pd.DataFrame(index=df.index)
    out["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out["symbol"] = df["symbol"].astype(str)
    out["actual_open"] = df["open"].astype(float)
    out["actual_high"] = high
    out["actual_low"] = low
    out["actual_close"] = close

    for h in (1, 2, 4, 8, 12, 24, 48):
        out[f"ret_{h}h"] = prev / close.shift(h + 1) - 1.0

    log_ret = np.log(prev / prev.shift(1))
    for h in (4, 12, 24, 72):
        out[f"vol_{h}h"] = log_ret.rolling(h, min_periods=max(2, h // 2)).std()

    tr = pd.concat(
        [
            high.shift(1) - low.shift(1),
            (high.shift(1) - close.shift(2)).abs(),
            (low.shift(1) - close.shift(2)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_4h"] = tr.rolling(4, min_periods=2).mean() / prev.clip(lower=0.01)
    out["atr_12h"] = tr.rolling(12, min_periods=6).mean() / prev.clip(lower=0.01)

    for h in (24, 72):
        rolling_high = high.shift(1).rolling(h, min_periods=max(4, h // 3)).max()
        rolling_low = low.shift(1).rolling(h, min_periods=max(4, h // 3)).min()
        denom = (rolling_high - rolling_low).clip(lower=1e-12)
        out[f"range_{h}h"] = ((prev - rolling_low) / denom).clip(0.0, 1.0)

    dolvol = (prev * volume.shift(1)).rolling(8, min_periods=2).mean()
    out["dolvol_8h_log"] = np.log1p(dolvol.clip(lower=0.0))
    vol_mean = volume.shift(1).rolling(24, min_periods=8).mean()
    vol_std = volume.shift(1).rolling(24, min_periods=8).std()
    out["volume_z_24h"] = ((volume.shift(1) - vol_mean) / vol_std.replace(0.0, np.nan)).clip(-8.0, 8.0)

    ts = pd.to_datetime(df["timestamp"], utc=True)
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    for h in (1, 2, 4, 6):
        if h <= max_horizon:
            out[f"target_{h}h"] = (close.shift(-h) / close - 1.0).clip(-0.25, 0.25)

    out[FEATURE_COLS] = out[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return out


def _sortino(returns: Sequence[float]) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    mean = float(np.nanmean(arr))
    downside = arr[arr < 0.0]
    if downside.size == 0:
        return 20.0 if mean > 0.0 else 0.0
    semi = float(np.sqrt(np.nanmean(downside * downside)))
    if semi <= 0.0:
        return 0.0
    return float(np.clip(mean / semi * math.sqrt(arr.size), -20.0, 20.0))


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    peak = 1.0
    worst = 0.0
    for eq in equity_curve:
        peak = max(peak, float(eq))
        if peak > 0.0:
            worst = max(worst, (peak - float(eq)) / peak)
    return worst


def _monthly_from_total(total_return: float, bars: int) -> float:
    if bars <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total_return)) * STOCK_BARS_PER_MONTH / float(bars))
    except (ValueError, OverflowError):
        return 0.0


def _round_trip_return(
    entry_open: float,
    exit_close: float,
    *,
    side: int = 1,
    fee_rate: float,
    fill_buffer_bps: float,
    slippage_bps: float,
    leverage: float,
    short_exposure_scale: float = 1.0,
) -> float:
    adverse = (float(fill_buffer_bps) + float(slippage_bps)) / 10_000.0
    side = 1 if int(side) >= 0 else -1
    if side > 0:
        entry = float(entry_open) * (1.0 + adverse)
        exit_ = float(exit_close) * (1.0 - adverse)
    else:
        entry = float(entry_open) * (1.0 - adverse)
        exit_ = float(exit_close) * (1.0 + adverse)
    if entry <= 0.0 or exit_ <= 0.0:
        return 0.0
    if side > 0:
        net = exit_ * (1.0 - fee_rate) / (entry * (1.0 + fee_rate)) - 1.0
        exposure = 1.0
    else:
        net = (entry * (1.0 - fee_rate) - exit_ * (1.0 + fee_rate)) / entry
        exposure = float(short_exposure_scale)
    return float(leverage) * exposure * net


def _simulate_symbol_threshold(
    frame: pd.DataFrame,
    preds: np.ndarray,
    *,
    horizon: int,
    threshold: float,
    decision_lag: int,
    fee_rate: float,
    fill_buffer_bps: float,
    slippage_bps: float,
    leverage: float,
    allow_shorts: bool,
    short_exposure_scale: float,
    target_short_fraction: float,
    balance_penalty: float,
) -> tuple[float, float, float, int, float, int, int]:
    equity = 1.0
    curve = [equity]
    rets: list[float] = []
    trades = 0
    wins = 0
    long_trades = 0
    short_trades = 0
    i = 0
    opens = frame["actual_open"].to_numpy(dtype=np.float64)
    closes = frame["actual_close"].to_numpy(dtype=np.float64)
    while i + decision_lag + horizon < len(frame):
        pred = float(preds[i])
        if not np.isfinite(pred):
            rets.append(0.0)
            i += 1
            continue
        side = 0
        if pred > threshold:
            side = 1
        elif allow_shorts and pred < -threshold:
            side = -1
        if side == 0:
            rets.append(0.0)
            i += 1
            continue
        entry_idx = i + decision_lag
        exit_idx = entry_idx + horizon
        ret = _round_trip_return(
            opens[entry_idx],
            closes[exit_idx],
            side=side,
            fee_rate=fee_rate,
            fill_buffer_bps=fill_buffer_bps,
            slippage_bps=slippage_bps,
            leverage=leverage,
            short_exposure_scale=short_exposure_scale,
        )
        equity *= 1.0 + ret
        curve.append(equity)
        rets.append(ret)
        trades += 1
        wins += int(ret > 0.0)
        long_trades += int(side > 0)
        short_trades += int(side < 0)
        i = exit_idx + 1
    total = equity - 1.0
    monthly = _monthly_from_total(total, max(1, len(frame)))
    if allow_shorts and trades > 0 and balance_penalty > 0.0:
        short_fraction = short_trades / trades
        monthly -= float(balance_penalty) * abs(short_fraction - float(target_short_fraction))
    return (
        monthly,
        _max_drawdown(curve),
        _sortino(rets),
        trades,
        (wins / trades if trades else 0.0),
        long_trades,
        short_trades,
    )


def select_symbol_configs(
    val_df: pd.DataFrame,
    predictions: dict[int, np.ndarray],
    *,
    horizons: Sequence[int],
    thresholds: Sequence[float],
    decision_lag: int,
    fee_rate: float,
    fill_buffer_bps: float,
    slippage_bps: float,
    leverage: float,
    min_val_trades: int,
    allow_shorts: bool,
    short_exposure_scale: float,
    target_short_fraction: float,
    balance_penalty: float,
) -> list[SymbolConfig]:
    configs: list[SymbolConfig] = []
    pred_by_h = {
        h: pd.Series(predictions[h], index=val_df.index, name=f"pred_{h}h")
        for h in horizons
    }
    for symbol, sym_frame in val_df.groupby("symbol", sort=True):
        best: SymbolConfig | None = None
        idx = sym_frame.index
        for h in horizons:
            sym_preds = pred_by_h[h].reindex(idx).to_numpy(dtype=np.float64)
            for threshold in thresholds:
                monthly, dd, sortino, trades, win_rate, long_trades, short_trades = _simulate_symbol_threshold(
                    sym_frame.reset_index(drop=True),
                    sym_preds,
                    horizon=int(h),
                    threshold=float(threshold),
                    decision_lag=int(decision_lag),
                    fee_rate=float(fee_rate),
                    fill_buffer_bps=float(fill_buffer_bps),
                    slippage_bps=float(slippage_bps),
                    leverage=float(leverage),
                    allow_shorts=bool(allow_shorts),
                    short_exposure_scale=float(short_exposure_scale),
                    target_short_fraction=float(target_short_fraction),
                    balance_penalty=float(balance_penalty),
                )
                if trades < min_val_trades:
                    continue
                density = min(1.0, trades / max(1.0, float(min_val_trades) * 2.0))
                short_fraction = short_trades / trades if trades else 0.0
                balance_bonus = 0.025 * density * (1.0 - min(1.0, abs(short_fraction - target_short_fraction) / 0.5))
                score = monthly - 0.75 * dd + 0.015 * sortino * density + balance_bonus
                candidate = SymbolConfig(
                    symbol=str(symbol),
                    horizon=int(h),
                    threshold=float(threshold),
                    mode="long_short" if allow_shorts else "long_only",
                    val_score=float(score),
                    val_monthly_return=float(monthly),
                    val_max_drawdown=float(dd),
                    val_sortino=float(sortino),
                    val_trades=int(trades),
                    val_win_rate=float(win_rate),
                    val_long_trades=int(long_trades),
                    val_short_trades=int(short_trades),
                )
                if best is None or candidate.val_score > best.val_score:
                    best = candidate
        if best is not None and best.val_score > 0.0 and best.val_monthly_return > 0.0:
            configs.append(best)
    return sorted(configs, key=lambda c: c.val_score, reverse=True)


def build_candidate_trades(
    test_df: pd.DataFrame,
    predictions: dict[int, np.ndarray],
    configs: Sequence[SymbolConfig],
    *,
    decision_lag: int,
) -> list[CandidateTrade]:
    cfg_by_symbol = {cfg.symbol: cfg for cfg in configs}
    pred_by_h = {
        h: pd.Series(pred, index=test_df.index, name=f"pred_{h}h")
        for h, pred in predictions.items()
    }
    trades: list[CandidateTrade] = []
    for symbol, sym_frame in test_df.groupby("symbol", sort=True):
        cfg = cfg_by_symbol.get(str(symbol))
        if cfg is None:
            continue
        sym_frame = sym_frame.sort_values("timestamp").reset_index()
        preds = pred_by_h[cfg.horizon].reindex(sym_frame["index"]).to_numpy(dtype=np.float64)
        opens = sym_frame["actual_open"].to_numpy(dtype=np.float64)
        closes = sym_frame["actual_close"].to_numpy(dtype=np.float64)
        ts = pd.to_datetime(sym_frame["timestamp"], utc=True).reset_index(drop=True)
        for i, pred in enumerate(preds):
            entry_idx = i + int(decision_lag)
            exit_idx = entry_idx + int(cfg.horizon)
            if exit_idx >= len(sym_frame):
                break
            if not np.isfinite(pred):
                continue
            side = 0
            if pred > cfg.threshold:
                side = 1
            elif cfg.mode == "long_short" and pred < -cfg.threshold:
                side = -1
            if side == 0:
                continue
            trades.append(
                CandidateTrade(
                    symbol=str(symbol),
                    side=int(side),
                    horizon=int(cfg.horizon),
                    threshold=float(cfg.threshold),
                    signal_ts=ts.iloc[i],
                    entry_ts=ts.iloc[entry_idx],
                    exit_ts=ts.iloc[exit_idx],
                    edge=float(abs(pred) - cfg.threshold),
                    entry_open=float(opens[entry_idx]),
                    exit_close=float(closes[exit_idx]),
                )
            )
    return sorted(trades, key=lambda t: (t.entry_ts, -t.edge, t.side, t.symbol))


def build_cross_sectional_candidate_trades(
    frame: pd.DataFrame,
    predictions: dict[int, np.ndarray],
    *,
    horizon: int,
    threshold: float,
    decision_lag: int,
    allow_shorts: bool,
    opportunistic_watch_n: int = 0,
    opportunistic_entry_discount_bps: float = 0.0,
    opportunistic_watch_bars: int = 1,
    fill_buffer_bps: float = 0.0,
) -> list[CandidateTrade]:
    pred = pd.Series(predictions[int(horizon)], index=frame.index, name="pred")
    work = frame.copy()
    work["pred"] = pred.reindex(work.index).to_numpy(dtype=np.float64)
    work = work[np.isfinite(work["pred"].to_numpy(dtype=np.float64))]
    symbol_frames = {
        str(symbol): sym_frame.sort_values("timestamp")
        for symbol, sym_frame in frame.groupby("symbol", sort=False)
    }
    index_pos = {
        str(symbol): {int(idx): pos for pos, idx in enumerate(sym_frame.index.to_numpy())}
        for symbol, sym_frame in symbol_frames.items()
    }
    trades: list[CandidateTrade] = []
    for _, group in work.groupby("timestamp", sort=True):
        group = group.copy()
        group["rank_score"] = group["pred"] - float(group["pred"].median())
        group = group.sort_values("rank_score", ascending=False)
        if int(opportunistic_watch_n) > 0:
            longs = group[group["rank_score"] > float(threshold)].head(int(opportunistic_watch_n))
        else:
            longs = group[group["rank_score"] > float(threshold)].head(max(1, len(group) // 3))
        if allow_shorts:
            if int(opportunistic_watch_n) > 0:
                shorts = group[group["rank_score"] < -float(threshold)].tail(int(opportunistic_watch_n))
            else:
                shorts = group[group["rank_score"] < -float(threshold)].tail(max(1, len(group) // 4))
        else:
            shorts = group.iloc[0:0]
        for side, selected in ((1, longs), (-1, shorts)):
            for idx, row in selected.iterrows():
                symbol = str(row["symbol"])
                symbol_frame = symbol_frames.get(symbol)
                pos = index_pos.get(symbol, {}).get(int(idx))
                if symbol_frame is None or pos is None:
                    continue
                watch_start = pos + int(decision_lag)
                if watch_start >= len(symbol_frame):
                    continue
                if int(opportunistic_watch_n) <= 0 and float(opportunistic_entry_discount_bps) <= 0.0:
                    entry_pos = watch_start
                    entry = symbol_frame.iloc[entry_pos]
                    entry_price = float(entry["actual_open"])
                else:
                    ref = symbol_frame.iloc[watch_start]
                    ref_open = float(ref["actual_open"])
                    discount = float(opportunistic_entry_discount_bps) / 10_000.0
                    buffer = float(fill_buffer_bps) / 10_000.0
                    target = ref_open * (1.0 - discount if side > 0 else 1.0 + discount)
                    latest_entry = min(len(symbol_frame) - int(horizon) - 1, watch_start + max(0, int(opportunistic_watch_bars) - 1))
                    entry_pos = -1
                    entry_price = target
                    for candidate_pos in range(watch_start, latest_entry + 1):
                        bar = symbol_frame.iloc[candidate_pos]
                        if side > 0:
                            if float(bar.get("actual_low", bar.get("low", np.nan))) <= target * (1.0 - buffer):
                                entry_pos = candidate_pos
                                break
                        elif float(bar.get("actual_high", bar.get("high", np.nan))) >= target * (1.0 + buffer):
                            entry_pos = candidate_pos
                            break
                    if entry_pos < 0:
                        continue
                    entry = symbol_frame.iloc[entry_pos]
                exit_pos = entry_pos + int(horizon)
                if exit_pos >= len(symbol_frame):
                    continue
                exit_ = symbol_frame.iloc[exit_pos]
                trades.append(
                    CandidateTrade(
                        symbol=str(row["symbol"]),
                        side=int(side),
                        horizon=int(horizon),
                        threshold=float(threshold),
                        signal_ts=pd.Timestamp(row["timestamp"]),
                        entry_ts=pd.Timestamp(entry["timestamp"]),
                        exit_ts=pd.Timestamp(exit_["timestamp"]),
                        edge=float(abs(row["rank_score"]) - threshold),
                        entry_open=float(entry_price),
                        exit_close=float(exit_["actual_close"]),
                    )
                )
    return sorted(trades, key=lambda t: (t.entry_ts, -t.edge, t.side, t.symbol))


def select_cross_sectional_config(
    val_df: pd.DataFrame,
    predictions: dict[int, np.ndarray],
    *,
    horizons: Sequence[int],
    thresholds: Sequence[float],
    decision_lag: int,
    fee_rate: float,
    fill_buffer_bps: float,
    slippage_bps: float,
    leverage: float,
    max_positions: int,
    window_bars: int,
    stride_bars: int,
    allow_shorts: bool,
    short_exposure_scale: float,
    target_short_fraction: float,
    total_short_fraction_cap: float = 0.55,
    opportunistic_watch_n: int = 0,
    opportunistic_entry_discount_bps: float = 0.0,
    opportunistic_watch_bars: int = 1,
) -> CrossSectionalConfig | None:
    windows = _build_windows(
        pd.to_datetime(val_df["timestamp"], utc=True),
        window_bars=int(window_bars),
        stride_bars=int(stride_bars),
    )
    if not windows:
        return None
    best: CrossSectionalConfig | None = None
    for horizon in horizons:
        if int(horizon) not in predictions:
            continue
        for threshold in thresholds:
            candidates = build_cross_sectional_candidate_trades(
                val_df,
                predictions,
                horizon=int(horizon),
                threshold=float(threshold),
                decision_lag=int(decision_lag),
                allow_shorts=bool(allow_shorts),
                opportunistic_watch_n=int(opportunistic_watch_n),
                opportunistic_entry_discount_bps=float(opportunistic_entry_discount_bps),
                opportunistic_watch_bars=int(opportunistic_watch_bars),
                fill_buffer_bps=float(fill_buffer_bps),
            )
            if not candidates:
                continue
            results = simulate_candidate_windows(
                candidates,
                windows,
                fee_rate=float(fee_rate),
                fill_buffer_bps=float(fill_buffer_bps),
                slippage_bps=float(slippage_bps),
                leverage=float(leverage),
                window_bars=int(window_bars),
                max_positions=int(max_positions),
                short_exposure_scale=float(short_exposure_scale),
                target_short_fraction=float(target_short_fraction),
                total_short_fraction_cap=float(total_short_fraction_cap),
            )
            summary = summarize_windows(float(slippage_bps), results)
            density = min(1.0, summary.median_trades / max(1.0, float(max_positions) * 6.0))
            balance = 1.0 - min(1.0, abs(summary.median_short_fraction - target_short_fraction) / 0.5)
            score = (
                summary.median_monthly_return
                + 0.35 * summary.p10_monthly_return
                - 0.85 * summary.max_drawdown
                + 0.025 * density
                + 0.025 * balance
            )
            cfg = CrossSectionalConfig(
                horizon=int(horizon),
                threshold=float(threshold),
                val_score=float(score),
                val_median_monthly_return=float(summary.median_monthly_return),
                val_p10_monthly_return=float(summary.p10_monthly_return),
                val_max_drawdown=float(summary.max_drawdown),
                val_median_trades=float(summary.median_trades),
                val_median_short_fraction=float(summary.median_short_fraction),
            )
            if best is None or cfg.val_score > best.val_score:
                best = cfg
    return best


def _build_windows(
    timestamps: Sequence[pd.Timestamp],
    *,
    window_bars: int,
    stride_bars: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    unique = sorted(pd.unique(pd.Series(timestamps)))
    if len(unique) < window_bars:
        return []
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    i = 0
    while i + window_bars <= len(unique):
        span = unique[i : i + window_bars]
        out.append((pd.Timestamp(span[0]), pd.Timestamp(span[-1])))
        i += stride_bars
    return out


def simulate_candidate_windows(
    candidate_trades: Sequence[CandidateTrade],
    windows: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    fee_rate: float,
    fill_buffer_bps: float,
    slippage_bps: float,
    leverage: float,
    window_bars: int,
    max_positions: int = 1,
    short_exposure_scale: float = 1.0,
    target_short_fraction: float = 0.45,
    total_short_fraction_cap: float = 0.55,
) -> list[WindowResult]:
    max_positions = max(1, int(max_positions))
    short_slots = max(1, int(math.floor(max_positions * float(target_short_fraction))))
    long_slots = max(1, max_positions - short_slots)
    out: list[WindowResult] = []
    for start, end in windows:
        selected: list[CandidateTrade] = []
        active: list[tuple[pd.Timestamp, int]] = []
        selected_longs = 0
        selected_shorts = 0
        for trade in candidate_trades:
            if trade.entry_ts < start or trade.exit_ts > end:
                continue
            active = [(ts, side) for ts, side in active if ts > trade.entry_ts]
            if len(active) >= max_positions:
                continue
            active_same_side = sum(1 for _, side in active if side == trade.side)
            side_cap = long_slots if trade.side > 0 else short_slots
            if active_same_side >= side_cap:
                continue
            if trade.side < 0:
                cap = min(max(float(total_short_fraction_cap), 0.0), 1.0)
                allowed_shorts = max(1.0, (selected_longs + 1.0) * cap / max(1.0 - cap, 1e-6))
                if selected_shorts + 1 > allowed_shorts:
                    continue
            if any(existing.symbol == trade.symbol and existing.exit_ts > trade.entry_ts for existing in selected):
                continue
            selected.append(trade)
            active.append((trade.exit_ts, trade.side))
            selected_longs += int(trade.side > 0)
            selected_shorts += int(trade.side < 0)

        equity = 1.0
        curve = [equity]
        long_trades = 0
        short_trades = 0
        for trade in sorted(selected, key=lambda t: (t.exit_ts, t.symbol)):
            ret = _round_trip_return(
                trade.entry_open,
                trade.exit_close,
                side=trade.side,
                fee_rate=fee_rate,
                fill_buffer_bps=fill_buffer_bps,
                slippage_bps=slippage_bps,
                leverage=leverage,
                short_exposure_scale=short_exposure_scale,
            )
            equity *= 1.0 + ret / float(max_positions)
            curve.append(equity)
            long_trades += int(trade.side > 0)
            short_trades += int(trade.side < 0)
        total = equity - 1.0
        out.append(
            WindowResult(
                start=start.isoformat(),
                end=end.isoformat(),
                total_return=float(total),
                monthly_return=float(_monthly_from_total(total, window_bars)),
                max_drawdown=float(_max_drawdown(curve)),
                trades=int(len(selected)),
                long_trades=int(long_trades),
                short_trades=int(short_trades),
            )
        )
    return out


def summarize_windows(slippage_bps: float, windows: Sequence[WindowResult]) -> SlippageSummary:
    monthly = np.asarray([w.monthly_return for w in windows], dtype=np.float64)
    dds = np.asarray([w.max_drawdown for w in windows], dtype=np.float64)
    trades = np.asarray([w.trades for w in windows], dtype=np.float64)
    short_fracs = np.asarray(
        [w.short_trades / w.trades for w in windows if w.trades > 0],
        dtype=np.float64,
    )
    if monthly.size == 0:
        return SlippageSummary(slippage_bps, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)
    return SlippageSummary(
        slippage_bps=float(slippage_bps),
        median_monthly_return=float(np.percentile(monthly, 50)),
        p10_monthly_return=float(np.percentile(monthly, 10)),
        max_drawdown=float(np.max(dds)) if dds.size else 0.0,
        negative_windows=int(np.sum(monthly < 0.0)),
        n_windows=int(monthly.size),
        median_trades=float(np.percentile(trades, 50)) if trades.size else 0.0,
        median_short_fraction=float(np.percentile(short_fracs, 50)) if short_fracs.size else 0.0,
    )


def candidate_trade_to_dict(trade: CandidateTrade) -> dict[str, object]:
    return {
        "symbol": trade.symbol,
        "side": int(trade.side),
        "horizon": int(trade.horizon),
        "threshold": float(trade.threshold),
        "signal_ts": pd.Timestamp(trade.signal_ts).isoformat(),
        "entry_ts": pd.Timestamp(trade.entry_ts).isoformat(),
        "exit_ts": pd.Timestamp(trade.exit_ts).isoformat(),
        "edge": float(trade.edge),
        "entry_open": float(trade.entry_open),
        "exit_close": float(trade.exit_close),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdatahourly/stocks")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated OOS evaluation symbols.")
    p.add_argument("--max-train-symbols", type=int, default=96)
    p.add_argument("--min-bars", type=int, default=500)
    p.add_argument("--train-start", default="")
    p.add_argument("--train-end", default="2025-05-31")
    p.add_argument("--val-end", default="2025-11-30")
    p.add_argument("--test-end", default="2026-04-10")
    p.add_argument("--horizons", default="1,2,4,6")
    p.add_argument("--thresholds", default="0.0000,0.0005,0.0010,0.0015,0.0020,0.0030,0.0040,0.0060")
    p.add_argument(
        "--selection-mode",
        choices=("cross_sectional", "symbol_threshold"),
        default="cross_sectional",
        help="cross_sectional ranks all symbols each bar; symbol_threshold preserves the original per-symbol threshold search.",
    )
    p.add_argument("--min-val-trades", type=int, default=8)
    p.add_argument("--window-bars", type=int, default=650, help="650 ≈ 100 US equity trading days.")
    p.add_argument("--stride-bars", type=int, default=65)
    p.add_argument("--decision-lag", type=int, default=2)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps-grid", default="0,5,10,20")
    p.add_argument(
        "--selection-slippage-bps",
        type=float,
        default=None,
        help="Validation threshold-selection slippage. Defaults to the worst slippage grid cell.",
    )
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--max-positions", type=int, default=8)
    p.add_argument("--long-only", action="store_true", help="Disable short candidates; useful for apples-to-apples legacy runs.")
    p.add_argument("--invert-signals", action="store_true", help="Multiply model predictions by -1 before validation selection and OOS scheduling.")
    p.add_argument("--short-exposure-scale", type=float, default=0.85, help="Scale short returns below long returns to keep the book slightly long-biased.")
    p.add_argument("--target-short-fraction", type=float, default=0.45, help="Desired short trade fraction in validation/window scheduling.")
    p.add_argument("--total-short-fraction-cap", type=float, default=0.55, help="Maximum cumulative short trade fraction allowed inside each evaluation window.")
    p.add_argument("--balance-penalty", type=float, default=0.04, help="Validation monthly-return penalty for drifting away from target short fraction.")
    p.add_argument(
        "--opportunistic-watch-n",
        type=int,
        default=0,
        help="Watch this many long and short ranked candidates per signal bar. 0 keeps classic immediate entries.",
    )
    p.add_argument(
        "--opportunistic-entry-discount-bps",
        type=float,
        default=0.0,
        help="Limit-entry distance from the post-lag reference open. Longs buy below, shorts sell above.",
    )
    p.add_argument(
        "--opportunistic-watch-bars",
        type=int,
        default=1,
        help="Number of hourly bars after decision_lag that a queued candidate can wait for its limit fill.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-boost-round", type=int, default=320)
    p.add_argument("--early-stopping-rounds", type=int, default=35)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.035)
    p.add_argument("--subsample", type=float, default=0.82)
    p.add_argument("--colsample-bytree", type=float, default=0.82)
    p.add_argument("--min-child-weight", type=float, default=50.0)
    p.add_argument("--output", type=Path, default=REPO / "analysis/bitbankgo_stock_bridge/benchmark.json")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validation_failures, parsed_args = validate_args(args)
    if validation_failures:
        for failure in validation_failures:
            print(f"benchmark_bitbankgo_stock_bridge: {failure}", file=sys.stderr)
        return 2

    try:
        import xgboost as xgb
    except Exception as exc:
        print(f"xgboost unavailable: {exc}", file=sys.stderr)
        return 2

    eval_symbols = parsed_args["eval_symbols"]
    horizons = parsed_args["horizons"]
    thresholds = parsed_args["thresholds"]
    slippages = parsed_args["slippages"]
    selection_slippage = parsed_args["selection_slippage"]
    train_start = parsed_args["train_start"]
    train_end = parsed_args["train_end"]
    val_end = parsed_args["val_end"]
    test_end = parsed_args["test_end"]

    frames: list[pd.DataFrame] = []
    for symbol, path in _iter_symbol_files(
        args.data_root,
        eval_symbols=eval_symbols,
        max_train_symbols=args.max_train_symbols,
    ):
        raw = _read_ohlcv(path, symbol)
        if raw is None or len(raw) < args.min_bars:
            continue
        feat = build_bitbankgo_hourly_features(raw, max_horizon=max(horizons))
        target_cols = [f"target_{h}h" for h in horizons]
        feat = feat.dropna(subset=FEATURE_COLS + target_cols)
        if train_start is not None:
            feat = feat[feat["timestamp"] >= train_start]
        if len(feat) >= args.min_bars:
            frames.append(feat)

    if not frames:
        print("no usable hourly stock frames loaded", file=sys.stderr)
        return 2

    panel = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    train = panel[panel["timestamp"] <= train_end].copy()
    val = panel[(panel["timestamp"] > train_end) & (panel["timestamp"] <= val_end)].copy()
    test = panel[(panel["timestamp"] > val_end) & (panel["timestamp"] <= test_end)].copy()
    test = test[test["symbol"].isin(eval_symbols)].copy()
    if train.empty or val.empty or test.empty:
        print(
            f"empty split: train={len(train)} val={len(val)} test={len(test)}",
            file=sys.stderr,
        )
        return 2

    print(
        f"[bitbankgo-stock-bridge] rows train={len(train)} val={len(val)} test={len(test)} "
        f"symbols_train={panel['symbol'].nunique()} symbols_test={test['symbol'].nunique()}",
        flush=True,
    )

    models = {}
    val_predictions: dict[int, np.ndarray] = {}
    test_predictions: dict[int, np.ndarray] = {}
    X_train = train[FEATURE_COLS].to_numpy(np.float32)
    X_val = val[FEATURE_COLS].to_numpy(np.float32)
    X_test = test[FEATURE_COLS].to_numpy(np.float32)
    for h in horizons:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": args.device,
            "max_depth": int(args.max_depth),
            "eta": float(args.eta),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "min_child_weight": float(args.min_child_weight),
            "lambda": 1.5,
            "alpha": 0.05,
            "seed": 8800 + int(h),
        }
        dtrain = xgb.DMatrix(
            X_train,
            label=train[f"target_{h}h"].to_numpy(np.float32),
            feature_names=FEATURE_COLS,
        )
        dval = xgb.DMatrix(
            X_val,
            label=val[f"target_{h}h"].to_numpy(np.float32),
            feature_names=FEATURE_COLS,
        )
        print(f"[bitbankgo-stock-bridge] train horizon={h}h", flush=True)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=int(args.num_boost_round),
            evals=[(dval, "valid")],
            early_stopping_rounds=int(args.early_stopping_rounds),
            verbose_eval=False,
        )
        models[h] = model
        val_predictions[h] = model.predict(xgb.DMatrix(X_val, feature_names=FEATURE_COLS))
        test_predictions[h] = model.predict(xgb.DMatrix(X_test, feature_names=FEATURE_COLS))
    if bool(args.invert_signals):
        val_predictions = {h: -np.asarray(pred, dtype=np.float64) for h, pred in val_predictions.items()}
        test_predictions = {h: -np.asarray(pred, dtype=np.float64) for h, pred in test_predictions.items()}

    val_eval = val[val["symbol"].isin(eval_symbols)].copy()
    if val_eval.empty:
        print("validation split has no rows for requested eval symbols", file=sys.stderr)
        return 2
    val_eval_predictions = {
        h: pd.Series(pred, index=val.index).reindex(val_eval.index).to_numpy(dtype=np.float64)
        for h, pred in val_predictions.items()
    }

    windows = _build_windows(
        pd.to_datetime(test["timestamp"], utc=True),
        window_bars=int(args.window_bars),
        stride_bars=int(args.stride_bars),
    )
    selected_configs: list[dict[str, object]]
    if args.selection_mode == "cross_sectional":
        val_window_bars = min(int(args.window_bars), max(16, val_eval["timestamp"].nunique() // 2))
        val_stride_bars = max(1, min(int(args.stride_bars), max(1, val_window_bars // 4)))
        cross_cfg = select_cross_sectional_config(
            val_eval,
            val_eval_predictions,
            horizons=horizons,
            thresholds=thresholds,
            decision_lag=int(args.decision_lag),
            fee_rate=float(args.fee_rate),
            fill_buffer_bps=float(args.fill_buffer_bps),
            slippage_bps=float(selection_slippage),
            leverage=float(args.leverage),
            max_positions=int(args.max_positions),
            window_bars=val_window_bars,
            stride_bars=val_stride_bars,
            allow_shorts=not bool(args.long_only),
            short_exposure_scale=float(args.short_exposure_scale),
            target_short_fraction=float(args.target_short_fraction),
            total_short_fraction_cap=float(args.total_short_fraction_cap),
            opportunistic_watch_n=int(args.opportunistic_watch_n),
            opportunistic_entry_discount_bps=float(args.opportunistic_entry_discount_bps),
            opportunistic_watch_bars=int(args.opportunistic_watch_bars),
        )
        if cross_cfg is None:
            print("no cross-sectional validation config selected", file=sys.stderr)
            return 3
        print(
            "[bitbankgo-stock-bridge] selected cross-sectional "
            f"horizon={cross_cfg.horizon}h threshold={cross_cfg.threshold:g} "
            f"val_med={cross_cfg.val_median_monthly_return*100:+.2f}% "
            f"val_short_frac={cross_cfg.val_median_short_fraction:.2f}",
            flush=True,
        )
        selected_configs = [asdict(cross_cfg)]
        candidate_trades = build_cross_sectional_candidate_trades(
            test,
            test_predictions,
            horizon=int(cross_cfg.horizon),
            threshold=float(cross_cfg.threshold),
            decision_lag=int(args.decision_lag),
            allow_shorts=not bool(args.long_only),
            opportunistic_watch_n=int(args.opportunistic_watch_n),
            opportunistic_entry_discount_bps=float(args.opportunistic_entry_discount_bps),
            opportunistic_watch_bars=int(args.opportunistic_watch_bars),
            fill_buffer_bps=float(args.fill_buffer_bps),
        )
    else:
        configs = select_symbol_configs(
            val_eval,
            val_eval_predictions,
            horizons=horizons,
            thresholds=thresholds,
            decision_lag=int(args.decision_lag),
            fee_rate=float(args.fee_rate),
            fill_buffer_bps=float(args.fill_buffer_bps),
            slippage_bps=float(selection_slippage),
            leverage=float(args.leverage),
            min_val_trades=int(args.min_val_trades),
            allow_shorts=not bool(args.long_only),
            short_exposure_scale=float(args.short_exposure_scale),
            target_short_fraction=float(args.target_short_fraction),
            balance_penalty=float(args.balance_penalty),
        )
        if not configs:
            print("no positive validation configs selected", file=sys.stderr)
            return 3
        print(f"[bitbankgo-stock-bridge] selected {len(configs)} validation-positive symbol configs", flush=True)
        selected_configs = [asdict(c) for c in configs]
        candidate_trades = build_candidate_trades(
            test,
            test_predictions,
            configs,
            decision_lag=int(args.decision_lag),
        )
    print(
        f"[bitbankgo-stock-bridge] candidate_trades={len(candidate_trades)} windows={len(windows)}",
        flush=True,
    )

    window_results: dict[str, list[dict]] = {}
    summaries: list[SlippageSummary] = []
    for slip in slippages:
        results = simulate_candidate_windows(
            candidate_trades,
            windows,
            fee_rate=float(args.fee_rate),
            fill_buffer_bps=float(args.fill_buffer_bps),
            slippage_bps=float(slip),
            leverage=float(args.leverage),
            window_bars=int(args.window_bars),
            max_positions=int(args.max_positions),
            short_exposure_scale=float(args.short_exposure_scale),
            target_short_fraction=float(args.target_short_fraction),
            total_short_fraction_cap=float(args.total_short_fraction_cap),
        )
        summary = summarize_windows(float(slip), results)
        summaries.append(summary)
        window_results[str(float(slip))] = [asdict(r) for r in results]
        print(
            f"  slip={slip:g} med={summary.median_monthly_return*100:+.2f}% "
            f"p10={summary.p10_monthly_return*100:+.2f}% "
            f"neg={summary.negative_windows}/{summary.n_windows} "
            f"dd={summary.max_drawdown*100:.2f}% trades_med={summary.median_trades:.0f} "
            f"short_frac_med={summary.median_short_fraction:.2f}",
            flush=True,
        )

    worst = min(summaries, key=lambda s: s.median_monthly_return)
    opportunistic_enabled = int(args.opportunistic_watch_n) > 0 or float(args.opportunistic_entry_discount_bps) > 0.0
    payload = {
        "schema_version": 1,
        "strategy": (
            "bitbankgo_hourly_xgb_bridge"
            f"{'_opportunistic' if opportunistic_enabled else ''}"
            f"{'_inverted' if args.invert_signals else ''}"
            f"{'_long_only' if args.long_only else '_long_short'}"
        ),
        "args": {
            "data_root": str(args.data_root),
            "eval_symbols": eval_symbols,
            "max_train_symbols": int(args.max_train_symbols),
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_end": args.val_end,
            "test_end": args.test_end,
            "horizons": horizons,
            "thresholds": thresholds,
            "selection_mode": args.selection_mode,
            "invert_signals": bool(args.invert_signals),
            "decision_lag": int(args.decision_lag),
            "fee_rate": float(args.fee_rate),
            "fill_buffer_bps": float(args.fill_buffer_bps),
            "slippage_bps_grid": slippages,
            "selection_slippage_bps": float(selection_slippage),
            "leverage": float(args.leverage),
            "max_positions": int(args.max_positions),
            "long_only": bool(args.long_only),
            "short_exposure_scale": float(args.short_exposure_scale),
            "target_short_fraction": float(args.target_short_fraction),
            "total_short_fraction_cap": float(args.total_short_fraction_cap),
            "balance_penalty": float(args.balance_penalty),
            "opportunistic_watch_n": int(args.opportunistic_watch_n),
            "opportunistic_entry_discount_bps": float(args.opportunistic_entry_discount_bps),
            "opportunistic_watch_bars": int(args.opportunistic_watch_bars),
            "window_bars": int(args.window_bars),
            "stride_bars": int(args.stride_bars),
        },
        "feature_cols": FEATURE_COLS,
        "split_rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "selected_configs": selected_configs,
        "candidate_trades": [candidate_trade_to_dict(t) for t in candidate_trades],
        "summaries": [asdict(s) for s in summaries],
        "worst_slippage_cell": asdict(worst),
        "passes_27pct_gate": bool(worst.median_monthly_return >= 0.27 and worst.negative_windows == 0),
        "window_results": window_results,
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}", flush=True)
    return 0 if payload["passes_27pct_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
