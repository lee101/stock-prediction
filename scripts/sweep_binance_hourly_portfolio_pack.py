#!/usr/bin/env python3
"""Train hourly Binance XGBoost levels and sweep portfolio packing settings.

This experiment is intentionally grounded in the execution path we already
trust for multi-position hourly simulation:

1. Train XGBoost models to forecast future high/low/close returns per pair.
2. Convert forecasts into opportunistic long limit watchers with buy/sell
   levels, confidence-sized amounts, and a CVaR-style downside penalty.
3. Evaluate those actions with the portfolio simulator using binary OHLC fills,
   fill buffers, decision lag, pending-order TTL, max positions, and leverage.
4. Render a review HTML/MP4 for the best setting so fills, exits, levels, PnL,
   and the text trade log can be inspected visually.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.marketsim_video import render_mp4, trace_from_portfolio_result
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation


FEATURE_COLS = [
    "ret_1h",
    "ret_3h",
    "ret_6h",
    "ret_12h",
    "ret_24h",
    "ret_72h",
    "vol_12h",
    "vol_24h",
    "vol_72h",
    "range_12h",
    "range_24h",
    "body_pct",
    "close_vwap_gap",
    "log_volume",
    "volume_z24",
    "cvar_loss_72h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "symbol_rank",
]

CS_FEATURE_BASE = [
    "ret_1h",
    "ret_6h",
    "ret_24h",
    "vol_24h",
    "vol_72h",
    "range_24h",
    "log_volume",
    "cvar_loss_72h",
]


@dataclass(frozen=True)
class PackConfig:
    risk_penalty: float
    cvar_weight: float
    entry_gap_bps: float
    entry_alpha: float
    exit_alpha: float
    edge_threshold: float
    edge_to_full_size: float
    max_positions: int
    max_pending_entries: int
    entry_ttl_hours: int
    max_hold_hours: int
    max_leverage: float
    entry_selection_mode: str
    entry_allocator_mode: str
    entry_allocator_edge_power: float


def _parse_float_list(value: str) -> list[float]:
    items = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError(f"empty float list: {value!r}")
    return items


def _parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError(f"empty int list: {value!r}")
    return items


def _parse_str_list(value: str) -> list[str]:
    items = [part.strip() for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError(f"empty string list: {value!r}")
    return items


def _to_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if isinstance(ts, pd.DatetimeIndex):
        if len(ts) != 1:
            raise ValueError(f"expected one timestamp, got {value!r}")
        ts = ts[0]
    return pd.Timestamp(ts).tz_convert("UTC")


def _discover_symbols(root: Path, requested: str) -> list[str]:
    if requested.strip():
        return [part.strip().upper() for part in requested.split(",") if part.strip()]
    symbols = []
    for path in sorted(root.glob("*USDT.csv")):
        if path.name == "download_summary.csv":
            continue
        symbols.append(path.stem.upper())
    return symbols


def _load_hourly_frames(
    root: Path,
    *,
    symbols: list[str],
    min_bars: int,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = root / f"{symbol.upper()}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = symbol.upper()
        for col in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "trade_count" not in df.columns:
            df["trade_count"] = 0.0
        if "vwap" not in df.columns:
            df["vwap"] = df["close"]
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[(df["open"] > 0.0) & (df["high"] > 0.0) & (df["low"] > 0.0) & (df["close"] > 0.0)]
        if len(df) >= int(min_bars):
            frames[symbol.upper()] = df.reset_index(drop=True)
    if not frames:
        raise ValueError("no hourly Binance frames loaded")
    return frames


def _choose_end_timestamp(frames: dict[str, pd.DataFrame], min_symbols_per_hour: int) -> pd.Timestamp:
    counts = pd.concat([df["timestamp"] for df in frames.values()], ignore_index=True).value_counts().sort_index()
    eligible = counts[counts >= int(min_symbols_per_hour)]
    if eligible.empty:
        raise ValueError(
            f"no timestamp has at least {min_symbols_per_hour} symbols; max count={int(counts.max())}"
        )
    return pd.Timestamp(eligible.index[-1]).tz_convert("UTC")


def _future_rolling(series: pd.Series, horizon: int, op: str) -> pd.Series:
    shifted = series.shift(-1).iloc[::-1]
    rolled = shifted.rolling(int(horizon), min_periods=int(horizon))
    if op == "max":
        out = rolled.max()
    elif op == "min":
        out = rolled.min()
    else:
        raise ValueError(f"unknown future rolling op: {op}")
    return out.iloc[::-1]


def _add_symbol_features(df: pd.DataFrame, *, horizon: int, symbol_rank: float) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float).clip(lower=0.0)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)

    for period in (1, 3, 6, 12, 24, 72):
        out[f"ret_{period}h"] = close.pct_change(period)
    for period in (12, 24, 72):
        out[f"vol_{period}h"] = returns.rolling(period, min_periods=max(3, period // 3)).std()
        out[f"range_{period}h"] = (
            high.rolling(period, min_periods=max(3, period // 3)).max()
            / low.rolling(period, min_periods=max(3, period // 3)).min()
            - 1.0
        )

    out["body_pct"] = (close - open_) / close.replace(0.0, np.nan)
    out["close_vwap_gap"] = (close - out["vwap"].astype(float)) / close.replace(0.0, np.nan)
    out["log_volume"] = np.log1p(volume)
    vol_mean = out["log_volume"].rolling(24, min_periods=6).mean()
    vol_std = out["log_volume"].rolling(24, min_periods=6).std().replace(0.0, np.nan)
    out["volume_z24"] = (out["log_volume"] - vol_mean) / vol_std
    out["cvar_loss_72h"] = returns.rolling(72, min_periods=24).quantile(0.10).fillna(0.0).clip(upper=0.0).abs()

    hour = out["timestamp"].dt.hour.astype(float)
    dow = out["timestamp"].dt.dayofweek.astype(float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["symbol_rank"] = float(symbol_rank)

    future_high = _future_rolling(high, horizon, "max")
    future_low = _future_rolling(low, horizon, "min")
    future_close = close.shift(-int(horizon))
    ref = close
    out["target_high_ret"] = future_high / ref - 1.0
    out["target_low_ret"] = future_low / ref - 1.0
    out["target_close_ret"] = future_close / ref - 1.0
    out["reference_close"] = ref
    return out


def build_model_frame(
    frames: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    symbols = sorted(frames)
    denom = max(1, len(symbols) - 1)
    for idx, symbol in enumerate(symbols):
        enriched = _add_symbol_features(frames[symbol], horizon=horizon, symbol_rank=(2.0 * idx / denom - 1.0))
        enriched = enriched[(enriched["timestamp"] >= start) & (enriched["timestamp"] <= end)]
        parts.append(enriched)
    combined = pd.concat(parts, ignore_index=True)

    feature_cols = list(FEATURE_COLS)
    group = combined.groupby("timestamp", sort=False)
    for col in CS_FEATURE_BASE:
        mean = group[col].transform("mean")
        std = group[col].transform("std").replace(0.0, np.nan)
        z_col = f"{col}_cs_z"
        rank_col = f"{col}_cs_rank"
        combined[z_col] = ((combined[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        combined[rank_col] = group[col].rank(pct=True).mul(2.0).sub(1.0).fillna(0.0)
        feature_cols.extend([z_col, rank_col])

    keep_cols = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "reference_close",
        "target_high_ret",
        "target_low_ret",
        "target_close_ret",
    ] + feature_cols
    combined = combined[keep_cols].replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=feature_cols + ["target_high_ret", "target_low_ret", "target_close_ret"])
    return combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True), feature_cols


def _train_xgb(x_train: np.ndarray, y_train: np.ndarray, *, rounds: int, device: str, seed: int):
    import xgboost as xgb

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 4,
        "eta": 0.045,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 25.0,
        "lambda": 5.0,
        "tree_method": "hist",
        "device": device,
        "seed": int(seed),
    }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    try:
        return xgb.train(params, dtrain, num_boost_round=int(rounds), verbose_eval=False)
    except Exception:
        if device == "cpu":
            raise
        params["device"] = "cpu"
        return xgb.train(params, dtrain, num_boost_round=int(rounds), verbose_eval=False)


def _predict(model: Any, features: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    return np.asarray(model.predict(xgb.DMatrix(features)), dtype=np.float64)


def fit_forecasters(
    model_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_end: pd.Timestamp,
    rounds: int,
    device: str,
) -> tuple[Any, Any, Any]:
    train = model_frame[model_frame["timestamp"] < train_end]
    if train.empty:
        raise ValueError("empty training split")
    x_train = train[feature_cols].to_numpy(dtype=np.float32, copy=False)
    high_model = _train_xgb(x_train, train["target_high_ret"].to_numpy(dtype=np.float32), rounds=rounds, device=device, seed=1337)
    low_model = _train_xgb(x_train, train["target_low_ret"].to_numpy(dtype=np.float32), rounds=rounds, device=device, seed=1338)
    close_model = _train_xgb(x_train, train["target_close_ret"].to_numpy(dtype=np.float32), rounds=rounds, device=device, seed=1339)
    return high_model, low_model, close_model


def score_eval_rows(
    model_frame: pd.DataFrame,
    feature_cols: list[str],
    models: tuple[Any, Any, Any],
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> pd.DataFrame:
    rows = model_frame[(model_frame["timestamp"] >= eval_start) & (model_frame["timestamp"] <= eval_end)].copy()
    if rows.empty:
        raise ValueError("empty evaluation split")
    x_eval = rows[feature_cols].to_numpy(dtype=np.float32, copy=False)
    pred_high = _predict(models[0], x_eval)
    pred_low = _predict(models[1], x_eval)
    pred_close = _predict(models[2], x_eval)
    rows["pred_high_ret_xgb"] = np.clip(pred_high, -0.05, 0.25)
    rows["pred_low_ret_xgb"] = np.clip(pred_low, -0.25, 0.05)
    rows["pred_close_ret_xgb"] = np.clip(pred_close, -0.20, 0.20)
    return rows.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def build_actions_and_bars(
    scored: pd.DataFrame,
    *,
    cfg: PackConfig,
    label_horizon: int,
    min_take_profit_bps: float,
    max_entry_gap_bps: float,
    max_exit_gap_bps: float,
    fee_rate: float,
    top_candidates_per_hour: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = scored.copy()
    ref = rows["reference_close"].astype(float).to_numpy()
    pred_high = rows["pred_high_ret_xgb"].astype(float).to_numpy()
    pred_low = rows["pred_low_ret_xgb"].astype(float).to_numpy()
    pred_close = rows["pred_close_ret_xgb"].astype(float).to_numpy()
    cvar = rows["cvar_loss_72h"].astype(float).fillna(0.0).to_numpy()

    upside = np.maximum.reduce([pred_high, pred_close, np.zeros_like(pred_high)])
    downside = np.maximum(-pred_low, 0.0)
    base_entry_gap = float(cfg.entry_gap_bps) / 10_000.0
    entry_gap = np.maximum(base_entry_gap, float(cfg.entry_alpha) * downside)
    entry_gap = np.clip(entry_gap, 0.0, float(max_entry_gap_bps) / 10_000.0)
    min_tp = float(min_take_profit_bps) / 10_000.0
    exit_gap = np.maximum(min_tp, float(cfg.exit_alpha) * upside)
    exit_gap = np.clip(exit_gap, min_tp, float(max_exit_gap_bps) / 10_000.0)

    buy_price = ref * (1.0 - entry_gap)
    sell_price = np.maximum(buy_price * (1.0 + min_tp), ref * (1.0 + exit_gap))
    gross_edge = sell_price / np.maximum(buy_price, 1e-12) - 1.0
    risk_charge = float(cfg.risk_penalty) * (downside + float(cfg.cvar_weight) * cvar)
    edge = gross_edge - risk_charge - 2.0 * float(fee_rate)
    amount = 100.0 * np.clip(edge / max(float(cfg.edge_to_full_size), 1e-9), 0.0, 1.0)
    amount = np.where(edge >= float(cfg.edge_threshold), amount, 0.0)

    rows["buy_price"] = buy_price
    rows["sell_price"] = sell_price
    rows["buy_amount"] = amount
    rows["sell_amount"] = 0.0
    rows["trade_amount"] = amount
    rows["xgb_edge"] = edge
    rows["xgb_gross_edge"] = gross_edge
    rows["xgb_risk_charge"] = risk_charge
    rows["watch_entry_gap_bps"] = entry_gap * 10_000.0
    rows["watch_exit_gap_bps"] = exit_gap * 10_000.0
    rows[f"predicted_high_p50_h{label_horizon}"] = ref * (1.0 + pred_high)
    rows[f"predicted_low_p50_h{label_horizon}"] = ref * (1.0 + pred_low)
    rows[f"predicted_close_p50_h{label_horizon}"] = ref * (1.0 + pred_close)

    if int(top_candidates_per_hour) > 0:
        rank = rows.groupby("timestamp")["xgb_edge"].rank(method="first", ascending=False)
        rows.loc[rank > int(top_candidates_per_hour), ["buy_amount", "trade_amount"]] = 0.0

    action_cols = [
        "timestamp",
        "symbol",
        "buy_price",
        "sell_price",
        "buy_amount",
        "sell_amount",
        "trade_amount",
        "xgb_edge",
        "xgb_gross_edge",
        "xgb_risk_charge",
        "watch_entry_gap_bps",
        "watch_exit_gap_bps",
        f"predicted_high_p50_h{label_horizon}",
        f"predicted_low_p50_h{label_horizon}",
        f"predicted_close_p50_h{label_horizon}",
    ]
    bar_cols = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        f"predicted_high_p50_h{label_horizon}",
        f"predicted_low_p50_h{label_horizon}",
        f"predicted_close_p50_h{label_horizon}",
    ]
    actions = rows[action_cols].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    bars = rows[bar_cols].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return bars, actions


def _monthly_return(total_return: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end - start).total_seconds() / 86_400.0, 1.0)
    base = max(1e-9, 1.0 + float(total_return))
    return float(base ** (30.0 / days) - 1.0)


def compute_pack_selection_score(row: dict[str, Any], *, min_result_trades: int = 10) -> float:
    """Rank configs by return/risk while rejecting "smooth because idle" rows."""
    num_sells = int(row.get("num_sells", 0) or 0)
    min_trades = max(0, int(min_result_trades))
    trade_shortfall = max(0, min_trades - num_sells)
    idle_penalty = 100.0 if num_sells <= 0 else 0.0
    return float(
        row["monthly_return_pct"]
        + 8.0 * row["sortino"]
        + 20.0 * row["pnl_smoothness_score"]
        + 5.0 * row["goodness_score"]
        - 1.5 * row["max_drawdown_pct"]
        - idle_penalty
        - 4.0 * trade_shortfall
    )


def evaluate_pack(
    scored: pd.DataFrame,
    *,
    cfg: PackConfig,
    label_horizon: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Any]:
    bars, actions = build_actions_and_bars(
        scored,
        cfg=cfg,
        label_horizon=label_horizon,
        min_take_profit_bps=float(args.min_take_profit_bps),
        max_entry_gap_bps=float(args.max_entry_gap_bps),
        max_exit_gap_bps=float(args.max_exit_gap_bps),
        fee_rate=float(args.fee_rate),
        top_candidates_per_hour=int(args.top_candidates_per_hour),
    )
    symbols = sorted(scored["symbol"].astype(str).str.upper().unique())
    sim_cfg = PortfolioConfig(
        initial_cash=float(args.initial_cash),
        max_positions=int(cfg.max_positions),
        min_edge=float(cfg.edge_threshold),
        max_hold_hours=int(cfg.max_hold_hours),
        enforce_market_hours=False,
        close_at_eod=False,
        symbols=symbols,
        trade_amount_scale=100.0,
        entry_intensity_power=float(args.entry_intensity_power),
        entry_min_intensity_fraction=float(args.entry_min_intensity_fraction),
        fee_by_symbol={symbol: float(args.fee_rate) for symbol in symbols},
        max_leverage=float(cfg.max_leverage),
        decision_lag_bars=int(args.decision_lag),
        market_order_entry=False,
        bar_margin=float(args.fill_buffer_bps) / 10_000.0,
        entry_order_ttl_hours=int(cfg.entry_ttl_hours),
        entry_selection_mode=str(cfg.entry_selection_mode),
        force_close_slippage=float(args.force_close_slippage_bps) / 10_000.0,
        int_qty=False,
        margin_annual_rate=float(args.margin_annual_rate),
        entry_allocator_mode=str(cfg.entry_allocator_mode),
        entry_allocator_edge_power=float(cfg.entry_allocator_edge_power),
        entry_allocator_max_single_position_fraction=float(args.entry_allocator_max_single_position_fraction),
        entry_allocator_reserve_fraction=float(args.entry_allocator_reserve_fraction),
        max_pending_entries=int(cfg.max_pending_entries),
        apply_leverage_to_crypto=True,
        sim_backend="python",
    )
    result = run_portfolio_simulation(bars, actions, sim_cfg, horizon=int(label_horizon))
    start = pd.Timestamp(bars["timestamp"].min())
    end = pd.Timestamp(bars["timestamp"].max())
    total_return = float(result.metrics.get("total_return", 0.0))
    row = {
        **asdict(cfg),
        "symbols": len(symbols),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "eval_days": round((end - start).total_seconds() / 86_400.0, 3),
        "total_return_pct": 100.0 * total_return,
        "monthly_return_pct": 100.0 * _monthly_return(total_return, start, end),
        "sortino": float(result.metrics.get("sortino", 0.0)),
        "max_drawdown_pct": 100.0 * float(result.metrics.get("max_drawdown", 0.0)),
        "pnl_smoothness": float(result.metrics.get("pnl_smoothness", 0.0)),
        "pnl_smoothness_score": float(result.metrics.get("pnl_smoothness_score", 0.0)),
        "ulcer_index": float(result.metrics.get("ulcer_index", 0.0)),
        "goodness_score": float(result.metrics.get("goodness_score", 0.0)),
        "num_buys": int(result.metrics.get("num_buys", 0)),
        "num_sells": int(result.metrics.get("num_sells", 0)),
        "target_exits": int(result.metrics.get("target_exits", 0)),
        "timeout_exits": int(result.metrics.get("timeout_exits", 0)),
    }
    row["min_result_trades"] = int(args.min_result_trades)
    row["selection_score"] = compute_pack_selection_score(
        row,
        min_result_trades=int(args.min_result_trades),
    )
    return row, bars, actions, result


def _json_number(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return round(val, 10)


def _json_numbers(values: Any) -> list[float | None]:
    return [_json_number(v) for v in list(values)]


def _trade_segments(result: Any, timestamps: pd.DatetimeIndex, symbol_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    open_trades: dict[str, tuple[pd.Timestamp, float]] = {}
    segments: list[dict[str, Any]] = []
    last_ts = timestamps[-1] if len(timestamps) else None
    for trade in getattr(result, "trades", []):
        symbol = str(trade.symbol).upper()
        if symbol not in symbol_to_idx:
            continue
        ts = pd.Timestamp(trade.timestamp)
        if trade.side == "buy":
            open_trades[symbol] = (ts, float(trade.price))
        elif trade.side == "sell" and symbol in open_trades:
            entry_ts, entry_price = open_trades.pop(symbol)
            segments.append(
                {
                    "symbol": symbol,
                    "entry_ts": entry_ts.isoformat(),
                    "exit_ts": ts.isoformat(),
                    "entry_price": float(entry_price),
                    "exit_price": float(trade.price),
                    "reason": str(trade.reason or "exit"),
                }
            )
    if last_ts is not None:
        for symbol, (entry_ts, entry_price) in open_trades.items():
            segments.append(
                {
                    "symbol": symbol,
                    "entry_ts": entry_ts.isoformat(),
                    "exit_ts": last_ts.isoformat(),
                    "entry_price": float(entry_price),
                    "exit_price": float("nan"),
                    "reason": "open",
                }
            )
    return segments


def write_review_html(
    *,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    result: Any,
    out_path: Path,
    title: str,
    num_pairs: int,
    label_horizon: int,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = sorted(bars["symbol"].astype(str).str.upper().unique())
    trace = trace_from_portfolio_result(
        bars=bars,
        actions=actions,
        result=result,
        symbols=symbols,
        predicted_close_col=f"predicted_close_p50_h{label_horizon}",
    )
    active = trace.active_symbol_indices(int(num_pairs))
    timestamps = pd.DatetimeIndex(sorted(pd.to_datetime(bars["timestamp"], utc=True).unique()))
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(trace.symbols)}
    idx_to_axis: dict[int, str] = {}
    traces: list[dict[str, Any]] = []
    shapes: list[dict[str, Any]] = []

    dense = (
        bars.assign(timestamp=pd.to_datetime(bars["timestamp"], utc=True), symbol=bars["symbol"].astype(str).str.upper())
        .set_index(["timestamp", "symbol"])
        .sort_index()
    )
    full_index = pd.MultiIndex.from_product([timestamps, trace.symbols], names=["timestamp", "symbol"])
    dense = dense.reindex(full_index)

    def _axis_ref(row_idx: int) -> tuple[str, str, str, str]:
        suffix = "" if row_idx == 0 else str(row_idx + 1)
        return f"x{suffix}", f"y{suffix}", f"xaxis{suffix}", f"yaxis{suffix}"

    trade_rows = []
    for trade in getattr(result, "trades", []):
        trade_rows.append(
            {
                "timestamp": pd.Timestamp(trade.timestamp).isoformat(),
                "symbol": str(trade.symbol).upper(),
                "side": str(trade.side),
                "price": _json_number(trade.price),
                "qty": _json_number(trade.quantity),
                "cash_after": _json_number(trade.cash_after),
                "reason": str(trade.reason or ""),
            }
        )
    segments = _trade_segments(result, timestamps, symbol_to_idx)

    action_sig = actions[actions["buy_amount"].astype(float) > 0.0].copy()
    action_sig["timestamp"] = pd.to_datetime(action_sig["timestamp"], utc=True)
    action_sig["symbol"] = action_sig["symbol"].astype(str).str.upper()

    row_count = len(active) + 1
    axis_layout: dict[str, Any] = {}
    for panel_row, sym_idx in enumerate(active):
        symbol = trace.symbols[sym_idx]
        xref, yref, xlayout, ylayout = _axis_ref(panel_row)
        idx_to_axis[sym_idx] = (xref, yref)  # type: ignore[assignment]
        y_top = 1.0 - panel_row / row_count
        y_bottom = 1.0 - (panel_row + 0.86) / row_count
        axis_layout[xlayout] = {"domain": [0.05, 0.98], "anchor": yref, "rangeslider": {"visible": False}}
        axis_layout[ylayout] = {"domain": [y_bottom, y_top - 0.01], "anchor": xref, "title": symbol}

        sym_slice = dense.xs(symbol, level="symbol", drop_level=False)
        x_vals = [ts.isoformat() for ts in timestamps]
        traces.append(
            {
                "type": "candlestick",
                "x": x_vals,
                "open": _json_numbers(sym_slice["open"]),
                "high": _json_numbers(sym_slice["high"]),
                "low": _json_numbers(sym_slice["low"]),
                "close": _json_numbers(sym_slice["close"]),
                "name": symbol,
                "xaxis": xref,
                "yaxis": yref,
                "increasing": {"line": {"color": "#26a69a"}},
                "decreasing": {"line": {"color": "#ef5350"}},
            }
        )

        sym_actions = action_sig[action_sig["symbol"] == symbol].tail(400)
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [ts.isoformat() for ts in sym_actions["timestamp"]],
                "y": _json_numbers(sym_actions["buy_price"]),
                "text": [
                    f"watch buy {row.buy_price:.6g}<br>sell {row.sell_price:.6g}<br>edge {row.xgb_edge:.4f}<br>amount {row.buy_amount:.1f}"
                    for row in sym_actions.itertuples(index=False)
                ],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {"symbol": "star", "size": 12, "color": "#ffd400", "line": {"color": "#111", "width": 1}},
                "name": f"{symbol} watchers",
                "xaxis": xref,
                "yaxis": yref,
            }
        )

        sym_buys = [t for t in trade_rows if t["symbol"] == symbol and t["side"] == "buy"]
        sym_sells = [t for t in trade_rows if t["symbol"] == symbol and t["side"] == "sell"]
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [t["timestamp"] for t in sym_buys],
                "y": [t["price"] for t in sym_buys],
                "text": [f"BUY qty={t['qty']} cash={t['cash_after']}" for t in sym_buys],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {"symbol": "triangle-up", "size": 14, "color": "#00e676", "line": {"color": "white", "width": 1}},
                "name": f"{symbol} buys",
                "xaxis": xref,
                "yaxis": yref,
            }
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [t["timestamp"] for t in sym_sells],
                "y": [t["price"] for t in sym_sells],
                "text": [f"SELL {t['reason']} qty={t['qty']} cash={t['cash_after']}" for t in sym_sells],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {"symbol": "triangle-down", "size": 14, "color": "#ff5252", "line": {"color": "white", "width": 1}},
                "name": f"{symbol} sells",
                "xaxis": xref,
                "yaxis": yref,
            }
        )

        for segment in segments:
            if segment["symbol"] != symbol:
                continue
            entry_price = _json_number(segment["entry_price"])
            exit_price = _json_number(segment["exit_price"])
            if entry_price is None:
                continue
            shapes.append(
                {
                    "type": "line",
                    "xref": xref,
                    "yref": yref,
                    "x0": segment["entry_ts"],
                    "x1": segment["exit_ts"],
                    "y0": entry_price,
                    "y1": entry_price,
                    "line": {"color": "#00e676", "width": 1.5, "dash": "dot"},
                }
            )
            if exit_price is not None:
                shapes.append(
                    {
                        "type": "line",
                        "xref": xref,
                        "yref": yref,
                        "x0": segment["entry_ts"],
                        "x1": segment["exit_ts"],
                        "y0": exit_price,
                        "y1": exit_price,
                        "line": {"color": "#ff5252", "width": 1.5, "dash": "dash"},
                    }
                )

    equity_row = len(active)
    xref, yref, xlayout, ylayout = _axis_ref(equity_row)
    axis_layout[xlayout] = {"domain": [0.05, 0.98], "anchor": yref}
    axis_layout[ylayout] = {"domain": [0.03, 1.0 - (equity_row + 0.86) / row_count], "anchor": xref, "title": "equity"}
    equity = result.equity_curve.reindex(timestamps).ffill().bfill()
    traces.append(
        {
            "type": "scatter",
            "mode": "lines",
            "x": [ts.isoformat() for ts in timestamps],
            "y": _json_numbers(equity),
            "line": {"color": "#ce93d8", "width": 2.5},
            "name": "equity",
            "xaxis": xref,
            "yaxis": yref,
        }
    )

    top_actions = action_sig.sort_values("xgb_edge", ascending=False).head(120)
    summary = {
        "metrics": {key: _json_number(value) for key, value in result.metrics.items()},
        "symbols_rendered": [trace.symbols[idx] for idx in active],
        "trades": trade_rows[-300:],
        "top_watchers": [
            {
                "timestamp": row.timestamp.isoformat(),
                "symbol": row.symbol,
                "buy_price": _json_number(row.buy_price),
                "sell_price": _json_number(row.sell_price),
                "edge": _json_number(row.xgb_edge),
                "amount": _json_number(row.buy_amount),
                "entry_gap_bps": _json_number(row.watch_entry_gap_bps),
                "exit_gap_bps": _json_number(row.watch_exit_gap_bps),
            }
            for row in top_actions.itertuples(index=False)
        ],
    }
    layout = {
        "title": title,
        "template": "plotly_dark",
        "height": max(900, 310 * row_count),
        "showlegend": True,
        "hovermode": "x unified",
        "shapes": shapes,
        **axis_layout,
    }
    payload = {
        "traces": traces,
        "layout": layout,
        "summary": summary,
    }
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; background: #0e1117; color: #eaeaea; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    #chart {{ width: 100vw; height: 78vh; }}
    #log {{ padding: 16px 22px 28px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #30363d; padding: 4px 6px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: #ffd400; background: #151b23; position: sticky; top: 0; }}
    .metric {{ display: inline-block; margin-right: 18px; }}
  </style>
</head>
<body>
  <div id="chart"></div>
  <div id="log"></div>
  <script>
    const payload = {json.dumps(payload, allow_nan=False)};
    Plotly.newPlot('chart', payload.traces, payload.layout, {{responsive: true}});
    function fmt(v) {{ return (v === null || v === undefined) ? '' : (typeof v === 'number' ? v.toFixed(6) : v); }}
    const log = document.getElementById('log');
    const metrics = payload.summary.metrics;
    log.innerHTML += '<h2>Metrics</h2>' + Object.keys(metrics).map(k => `<span class="metric">${{k}}=${{fmt(metrics[k])}}</span>`).join('');
    function table(title, rows, cols) {{
      let html = `<h2>${{title}}</h2><table><thead><tr>${{cols.map(c => `<th>${{c}}</th>`).join('')}}</tr></thead><tbody>`;
      for (const r of rows) html += `<tr>${{cols.map(c => `<td>${{fmt(r[c])}}</td>`).join('')}}</tr>`;
      html += '</tbody></table>';
      log.innerHTML += html;
    }}
    table('Trade Log', payload.summary.trades, ['timestamp', 'symbol', 'side', 'price', 'qty', 'cash_after', 'reason']);
    table('Top Watcher Signals', payload.summary.top_watchers, ['timestamp', 'symbol', 'buy_price', 'sell_price', 'edge', 'amount', 'entry_gap_bps', 'exit_gap_bps']);
  </script>
</body>
</html>
"""
    out_path.write_text(html)
    return out_path


def render_best_artifacts(
    *,
    best_cfg: PackConfig,
    scored: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    if not args.html_out and not args.mp4_out and not args.trace_json_out:
        return
    latest = pd.Timestamp(scored["timestamp"].max())
    cutoff = latest - pd.Timedelta(days=int(args.render_days))
    scored_vis = scored[scored["timestamp"] >= cutoff].copy()
    row, bars, actions, result = evaluate_pack(scored_vis, cfg=best_cfg, label_horizon=int(args.label_horizon), args=args)
    title = (
        f"Binance hourly XGB portfolio pack "
        f"ret={row['total_return_pct']:+.2f}% monthly={row['monthly_return_pct']:+.2f}% "
        f"dd={row['max_drawdown_pct']:.2f}% trades={row['num_sells']}"
    )
    symbols = sorted(bars["symbol"].astype(str).str.upper().unique())
    trace = trace_from_portfolio_result(
        bars=bars,
        actions=actions,
        result=result,
        symbols=symbols,
        predicted_close_col=f"predicted_close_p50_h{args.label_horizon}",
    )
    if args.trace_json_out:
        trace.to_json(args.trace_json_out)
        print(f"wrote {args.trace_json_out}")
    if args.html_out:
        write_review_html(
            bars=bars,
            actions=actions,
            result=result,
            out_path=Path(args.html_out),
            title=title,
            num_pairs=int(args.num_pairs),
            label_horizon=int(args.label_horizon),
        )
        print(f"wrote {args.html_out}")
    if args.mp4_out:
        try:
            render_mp4(
                trace,
                args.mp4_out,
                num_pairs=int(args.num_pairs),
                fps=int(args.video_fps),
                frames_per_bar=int(args.frames_per_bar),
                title=title,
                fee_rate=float(args.fee_rate),
                periods_per_year=8760.0,
                leverage=float(best_cfg.max_leverage),
                width_px=int(args.video_width),
                height_px=int(args.video_height),
            )
            print(f"wrote {args.mp4_out}")
        except Exception as exc:
            print(f"mp4 render skipped: {exc}")


def iter_pack_configs(args: argparse.Namespace) -> list[PackConfig]:
    configs = []
    for values in itertools.product(
        _parse_float_list(args.risk_penalties),
        _parse_float_list(args.cvar_weights),
        _parse_float_list(args.entry_gap_bps_grid),
        _parse_float_list(args.entry_alpha_grid),
        _parse_float_list(args.exit_alpha_grid),
        _parse_float_list(args.edge_threshold_grid),
        _parse_float_list(args.edge_to_full_size_grid),
        _parse_int_list(args.max_positions_grid),
        _parse_int_list(args.max_pending_entries_grid),
        _parse_int_list(args.entry_ttl_hours_grid),
        _parse_int_list(args.max_hold_hours_grid),
        _parse_float_list(args.max_leverage_grid),
        _parse_str_list(args.entry_selection_modes),
        _parse_str_list(args.entry_allocator_modes),
        _parse_float_list(args.entry_allocator_edge_power_grid),
    ):
        configs.append(PackConfig(*values))
    return configs


def sample_pack_configs(configs: list[PackConfig], *, limit: int, seed: int) -> list[PackConfig]:
    if int(limit) <= 0 or int(limit) >= len(configs):
        return list(configs)
    rng = np.random.default_rng(int(seed))
    selected = sorted(rng.choice(len(configs), size=int(limit), replace=False).tolist())
    return [configs[idx] for idx in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep hourly Binance XGB portfolio packing/watchers.")
    parser.add_argument("--hourly-root", type=Path, default=Path("binance_spot_hourly"))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--min-bars", type=int, default=5000)
    parser.add_argument("--min-symbols-per-hour", type=int, default=20)
    parser.add_argument("--train-days", type=int, default=720)
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--label-horizon", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=220)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("analysis/binance_hourly_portfolio_pack.csv"))

    parser.add_argument("--risk-penalties", default="0.2,0.5")
    parser.add_argument("--cvar-weights", default="0.0,0.5")
    parser.add_argument("--entry-gap-bps-grid", default="25,50,75")
    parser.add_argument("--entry-alpha-grid", default="0.5")
    parser.add_argument("--exit-alpha-grid", default="0.8")
    parser.add_argument("--edge-threshold-grid", default="0.003,0.006")
    parser.add_argument("--edge-to-full-size-grid", default="0.02")
    parser.add_argument("--max-positions-grid", default="5,8")
    parser.add_argument("--max-pending-entries-grid", default="12,24")
    parser.add_argument("--entry-ttl-hours-grid", default="3,6")
    parser.add_argument("--max-hold-hours-grid", default="24")
    parser.add_argument("--max-leverage-grid", default="1.0")
    parser.add_argument("--entry-selection-modes", default="edge_rank,first_trigger")
    parser.add_argument("--entry-allocator-modes", default="legacy,concentrated")
    parser.add_argument("--entry-allocator-edge-power-grid", default="2.0")
    parser.add_argument("--max-configs", type=int, default=96)
    parser.add_argument("--config-sample-seed", type=int, default=1337)

    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--force-close-slippage-bps", type=float, default=10.0)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0625)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--min-take-profit-bps", type=float, default=35.0)
    parser.add_argument("--max-entry-gap-bps", type=float, default=120.0)
    parser.add_argument("--max-exit-gap-bps", type=float, default=250.0)
    parser.add_argument("--top-candidates-per-hour", type=int, default=30)
    parser.add_argument("--entry-intensity-power", type=float, default=1.0)
    parser.add_argument("--entry-min-intensity-fraction", type=float, default=0.0)
    parser.add_argument("--entry-allocator-max-single-position-fraction", type=float, default=0.35)
    parser.add_argument("--entry-allocator-reserve-fraction", type=float, default=0.05)
    parser.add_argument("--min-result-trades", type=int, default=10)

    parser.add_argument("--render-days", type=int, default=14)
    parser.add_argument("--num-pairs", type=int, default=6)
    parser.add_argument("--html-out", default="analysis/binance_hourly_portfolio_pack_best.html")
    parser.add_argument("--trace-json-out", default="analysis/binance_hourly_portfolio_pack_best.json")
    parser.add_argument("--mp4-out", default="analysis/binance_hourly_portfolio_pack_best.mp4")
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--frames-per-bar", type=int, default=1)
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    args = parser.parse_args()

    symbols = _discover_symbols(args.hourly_root, args.symbols)
    frames = _load_hourly_frames(args.hourly_root, symbols=symbols, min_bars=int(args.min_bars))
    min_symbols = min(int(args.min_symbols_per_hour), len(frames))
    end = _to_utc(args.end_date) if args.end_date.strip() else _choose_end_timestamp(frames, min_symbols)
    eval_start = end - pd.Timedelta(days=int(args.eval_days))
    train_start = eval_start - pd.Timedelta(days=int(args.train_days))
    feature_start = train_start - pd.Timedelta(days=10)

    model_frame, feature_cols = build_model_frame(
        frames,
        start=feature_start,
        end=end,
        horizon=int(args.label_horizon),
    )
    print(
        f"loaded {len(frames)} symbols, rows={len(model_frame):,}, "
        f"train=[{train_start}, {eval_start}), eval=[{eval_start}, {end}]"
    )
    models = fit_forecasters(
        model_frame,
        feature_cols,
        train_end=eval_start,
        rounds=int(args.rounds),
        device=str(args.device),
    )
    scored = score_eval_rows(model_frame, feature_cols, models, eval_start=eval_start, eval_end=end)
    print(f"scored eval rows={len(scored):,} symbols={scored['symbol'].nunique()}")

    all_configs = iter_pack_configs(args)
    configs = sample_pack_configs(
        all_configs,
        limit=int(args.max_configs),
        seed=int(args.config_sample_seed),
    )
    if not configs:
        raise ValueError("no pack configs to evaluate")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_cfg: PackConfig | None = None
    fieldnames: list[str] | None = None
    with args.out.open("w", newline="") as fh:
        writer: csv.DictWriter | None = None
        print(f"evaluating {len(configs)} sampled configs from full grid of {len(all_configs)}")
        for idx, cfg in enumerate(configs, start=1):
            row, _bars, _actions, _result = evaluate_pack(
                scored,
                cfg=cfg,
                label_horizon=int(args.label_horizon),
                args=args,
            )
            rows.append(row)
            if best_row is None or float(row["selection_score"]) > float(best_row["selection_score"]):
                best_row = row
                best_cfg = cfg
            if writer is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            print(
                f"{idx}/{len(configs)} monthly={row['monthly_return_pct']:+.2f}% "
                f"ret={row['total_return_pct']:+.2f}% dd={row['max_drawdown_pct']:.2f}% "
                f"score={row['selection_score']:.2f} trades={row['num_sells']} cfg={cfg}",
                flush=True,
            )

    rows.sort(key=lambda item: float(item["selection_score"]), reverse=True)
    print("\n=== Best Binance hourly portfolio packs ===")
    for row in rows[:10]:
        print(
            f"score={row['selection_score']:+8.2f} monthly={row['monthly_return_pct']:+7.2f}% "
            f"ret={row['total_return_pct']:+7.2f}% dd={row['max_drawdown_pct']:6.2f}% "
            f"sortino={row['sortino']:6.2f} trades={row['num_sells']:4d} "
            f"pos={row['max_positions']} pend={row['max_pending_entries']} lev={row['max_leverage']} "
            f"entry={row['entry_gap_bps']} ttl={row['entry_ttl_hours']} sel={row['entry_selection_mode']} "
            f"alloc={row['entry_allocator_mode']}"
        )
    print(f"\nwrote {args.out}")
    if best_cfg is not None:
        render_best_artifacts(best_cfg=best_cfg, scored=scored, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
