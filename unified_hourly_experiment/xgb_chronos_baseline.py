#!/usr/bin/env python3
"""Chronos2 + XGBoost hourly stock baseline for Alpaca-style portfolio simulation.

This baseline reuses the same forecast-merged hourly frames as the legacy
`trade_unified_hourly_meta.py` path, but fits gradient-boosted residual
corrections on top of blended Chronos2 forecasts for:

  - best reachable upside over the next N bars
  - worst reachable downside over the next N bars
  - end-of-horizon close return

The corrected forecasts are converted into simulator action rows
(`buy_price`, `sell_price`, `buy_amount`, `sell_amount`) and evaluated with the
hourly portfolio simulator across multiple holdout windows.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - runtime environment dependent
    raise RuntimeError(
        "xgboost is required for unified_hourly_experiment.xgb_chronos_baseline"
    ) from exc

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule, build_default_feature_columns
from src.trade_directions import resolve_trade_directions
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation


DEFAULT_SYMBOLS = (
    "DBX,TRIP,MTCH,NYT,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV"
)


@dataclass(frozen=True)
class SearchConfig:
    label_horizon_hours: int
    label_basis: str
    residual_scale: float
    risk_penalty: float
    entry_alpha: float
    exit_alpha: float
    edge_threshold: float
    edge_to_full_size: float
    max_positions: int
    max_hold_hours: int
    close_at_eod: bool
    market_order_entry: bool
    entry_selection_mode: str
    entry_allocator_mode: str
    entry_allocator_edge_power: float


@dataclass
class WindowMetrics:
    window_days: int
    total_return_pct: float
    sortino: float
    max_drawdown_pct: float
    pnl_smoothness: float
    pnl_smoothness_score: float
    goodness_score: float
    num_buys: int
    num_sells: int


def parse_csv_list(value: str) -> list[str]:
    return [token.strip().upper() for token in value.split(",") if token.strip()]


def parse_token_list(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def _bool_token(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_bool_list(value: str) -> list[bool]:
    return [_bool_token(token) for token in value.split(",") if token.strip()]


def _is_valid_positive(value: Any) -> bool:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(out) and out > 0.0)


def _forecast_price_col(kind: str, horizon: int) -> str:
    return f"predicted_{kind}_p50_h{int(horizon)}"


def blend_forecast_price(
    row: pd.Series,
    *,
    kind: str,
    target_horizon_hours: int,
    forecast_horizons: tuple[int, ...],
) -> float:
    available: list[tuple[int, float]] = []
    for horizon in sorted({int(token) for token in forecast_horizons}):
        col = _forecast_price_col(kind, horizon)
        value = row.get(col)
        if _is_valid_positive(value):
            available.append((int(horizon), float(value)))

    reference = float(row.get("reference_close", 0.0) or 0.0)
    if not available:
        return reference if _is_valid_positive(reference) else 0.0

    for horizon, value in available:
        if int(horizon) == int(target_horizon_hours):
            return float(value)

    lower = [(horizon, value) for horizon, value in available if horizon < int(target_horizon_hours)]
    upper = [(horizon, value) for horizon, value in available if horizon > int(target_horizon_hours)]
    if lower and upper:
        low_h, low_value = lower[-1]
        high_h, high_value = upper[0]
        x0 = np.log(max(low_h, 1))
        x1 = np.log(max(high_h, 1))
        x = np.log(max(int(target_horizon_hours), 1))
        if x1 > x0:
            weight = float(np.clip((x - x0) / (x1 - x0), 0.0, 1.0))
        else:
            weight = 0.0
        return float((1.0 - weight) * low_value + weight * high_value)

    nearest = min(available, key=lambda item: abs(int(item[0]) - int(target_horizon_hours)))
    return float(nearest[1])


def _repair_forecast_prices(
    *,
    reference_close: float,
    high_price: float,
    low_price: float,
    close_price: float,
) -> tuple[float, float, float]:
    ref = float(reference_close)
    fallback = ref if _is_valid_positive(ref) else 1.0
    high = float(high_price) if _is_valid_positive(high_price) else fallback
    low = float(low_price) if _is_valid_positive(low_price) else fallback
    close = float(close_price) if _is_valid_positive(close_price) else fallback

    high = max(high, close)
    low = min(low, close)
    if low > high:
        midpoint = close
        high = max(high, midpoint)
        low = min(low, midpoint)
    return max(high, 0.01), max(low, 0.01), max(close, 0.01)


def _holdout_cutoff(frame: pd.DataFrame, validation_days: int) -> pd.Timestamp:
    latest = pd.Timestamp(frame["timestamp"].max())
    return latest - pd.Timedelta(days=int(validation_days))


def load_symbol_frame(
    *,
    symbol: str,
    data_root: Path,
    cache_root: Path,
    sequence_length: int,
    forecast_horizons: tuple[int, ...],
) -> pd.DataFrame:
    cfg = DatasetConfig(
        symbol=symbol,
        data_root=str(data_root),
        forecast_cache_root=str(cache_root),
        forecast_horizons=forecast_horizons,
        sequence_length=sequence_length,
        min_history_hours=max(100, sequence_length + 24),
        validation_days=30,
        cache_only=True,
    )
    dm = BinanceHourlyDataModule(cfg)
    frame = dm.frame.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = symbol.upper()

    # Add inverse-spread confidence features; these are useful to the boosted
    # baseline and are causal because the source quantiles are forecast outputs.
    for horizon in forecast_horizons:
        close_col = f"reference_close"
        p90_col = f"predicted_close_p90_h{int(horizon)}"
        p10_col = f"predicted_close_p10_h{int(horizon)}"
        conf_col = f"forecast_confidence_h{int(horizon)}"
        if p90_col in frame.columns and p10_col in frame.columns:
            spread = (frame[p90_col] - frame[p10_col]).abs()
            ref = frame[close_col].replace(0.0, np.nan)
            frame[conf_col] = (1.0 / (1.0 + spread / ref)).replace([np.inf, -np.inf], np.nan).fillna(0.5)
        else:
            frame[conf_col] = 0.5

    ref = frame["reference_close"].replace(0.0, np.nan)
    for horizon in forecast_horizons:
        close_col = _forecast_price_col("close", horizon)
        high_col = _forecast_price_col("high", horizon)
        low_col = _forecast_price_col("low", horizon)
        if close_col not in frame.columns:
            continue
        if high_col in frame.columns and low_col in frame.columns:
            midpoint = (frame[high_col] + frame[low_col]) / 2.0
            frame[f"forecast_range_pct_h{int(horizon)}"] = (
                (frame[high_col] - frame[low_col]).abs() / ref
            )
            frame[f"forecast_mid_delta_h{int(horizon)}"] = (midpoint - ref) / ref
            frame[f"forecast_close_minus_mid_h{int(horizon)}"] = (frame[close_col] - midpoint) / ref
        else:
            frame[f"forecast_range_pct_h{int(horizon)}"] = 0.0
            frame[f"forecast_mid_delta_h{int(horizon)}"] = 0.0
            frame[f"forecast_close_minus_mid_h{int(horizon)}"] = 0.0

    sorted_horizons = sorted({int(token) for token in forecast_horizons})
    for lower, upper in zip(sorted_horizons[:-1], sorted_horizons[1:], strict=False):
        for kind in ("close", "high", "low"):
            upper_col = f"chronos_{kind}_delta_h{upper}"
            lower_col = f"chronos_{kind}_delta_h{lower}"
            gap_col = f"chronos_{kind}_delta_gap_h{lower}_h{upper}"
            if upper_col in frame.columns and lower_col in frame.columns:
                frame[gap_col] = frame[upper_col] - frame[lower_col]
    return frame


def build_labeled_rows(
    frame: pd.DataFrame,
    *,
    label_horizon_hours: int,
    forecast_horizons: tuple[int, ...],
    label_basis: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(frame)
    for idx in range(total):
        row = frame.iloc[idx]
        record = row.to_dict()
        record["symbol"] = str(row["symbol"]).upper()
        record["timestamp"] = pd.Timestamp(row["timestamp"])
        record["label_horizon_hours"] = int(label_horizon_hours)

        next_idx = idx + 1
        end_idx = min(total, idx + 1 + int(label_horizon_hours))
        if next_idx >= total or end_idx <= next_idx:
            record["target_high_ret"] = np.nan
            record["target_low_ret"] = np.nan
            record["target_close_ret"] = np.nan
            record["prior_high_ret"] = np.nan
            record["prior_low_ret"] = np.nan
            record["prior_close_ret"] = np.nan
            record["prior_range_ret"] = np.nan
            record["prior_close_minus_mid_ret"] = np.nan
            rows.append(record)
            continue

        future = frame.iloc[next_idx:end_idx]
        entry_ref = float(future.iloc[0]["open"])
        if label_basis == "reference_close":
            target_anchor = float(row.get("reference_close", 0.0) or 0.0)
        elif label_basis == "next_open":
            target_anchor = entry_ref
        else:
            raise ValueError(f"Unsupported label_basis={label_basis!r}")

        if not np.isfinite(entry_ref) or entry_ref <= 0 or not np.isfinite(target_anchor) or target_anchor <= 0:
            record["target_high_ret"] = np.nan
            record["target_low_ret"] = np.nan
            record["target_close_ret"] = np.nan
            record["prior_high_ret"] = np.nan
            record["prior_low_ret"] = np.nan
            record["prior_close_ret"] = np.nan
            record["prior_range_ret"] = np.nan
            record["prior_close_minus_mid_ret"] = np.nan
            rows.append(record)
            continue

        target_high_price = float(future["high"].max())
        target_low_price = float(future["low"].min())
        target_close_price = float(future.iloc[-1]["close"])
        record["target_high_ret"] = float(target_high_price / target_anchor - 1.0)
        record["target_low_ret"] = float(target_low_price / target_anchor - 1.0)
        record["target_close_ret"] = float(target_close_price / target_anchor - 1.0)

        prior_high_price = blend_forecast_price(
            row,
            kind="high",
            target_horizon_hours=label_horizon_hours,
            forecast_horizons=forecast_horizons,
        )
        prior_low_price = blend_forecast_price(
            row,
            kind="low",
            target_horizon_hours=label_horizon_hours,
            forecast_horizons=forecast_horizons,
        )
        prior_close_price = blend_forecast_price(
            row,
            kind="close",
            target_horizon_hours=label_horizon_hours,
            forecast_horizons=forecast_horizons,
        )
        prior_high_price, prior_low_price, prior_close_price = _repair_forecast_prices(
            reference_close=target_anchor,
            high_price=prior_high_price,
            low_price=prior_low_price,
            close_price=prior_close_price,
        )
        record["prior_high_ret"] = float(prior_high_price / target_anchor - 1.0)
        record["prior_low_ret"] = float(prior_low_price / target_anchor - 1.0)
        record["prior_close_ret"] = float(prior_close_price / target_anchor - 1.0)
        record["prior_range_ret"] = float((prior_high_price - prior_low_price) / target_anchor)
        record["prior_close_minus_mid_ret"] = float(
            (prior_close_price - ((prior_high_price + prior_low_price) / 2.0)) / target_anchor
        )
        rows.append(record)
    return pd.DataFrame(rows)


def prepare_dataset(
    *,
    symbols: list[str],
    data_root: Path,
    cache_root: Path,
    sequence_length: int,
    forecast_horizons: tuple[int, ...],
    label_horizon_hours: int,
    label_basis: str,
    validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = load_symbol_frame(
            symbol=symbol,
            data_root=data_root,
            cache_root=cache_root,
            sequence_length=sequence_length,
            forecast_horizons=forecast_horizons,
        )
        labeled = build_labeled_rows(
            frame,
            label_horizon_hours=label_horizon_hours,
            forecast_horizons=forecast_horizons,
            label_basis=label_basis,
        )
        parts.append(labeled)

    data = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    cutoff = pd.Timestamp(data["timestamp"].max()) - pd.Timedelta(days=int(validation_days))

    train = data[(data["timestamp"] < cutoff)].copy()
    holdout = data[(data["timestamp"] >= cutoff)].copy()
    train = train.dropna(subset=["target_high_ret", "target_low_ret", "target_close_ret"]).reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)
    if train.empty:
        raise RuntimeError("No train rows available for XGBoost baseline")
    if holdout.empty:
        raise RuntimeError("No holdout rows available for XGBoost baseline")
    return train, holdout


def resolve_feature_columns(
    forecast_horizons: tuple[int, ...],
    data: pd.DataFrame,
) -> list[str]:
    feature_columns = list(build_default_feature_columns(forecast_horizons))
    feature_columns.extend(
        [
            "prior_high_ret",
            "prior_low_ret",
            "prior_close_ret",
            "prior_range_ret",
            "prior_close_minus_mid_ret",
        ]
    )
    for horizon in forecast_horizons:
        extra = f"forecast_confidence_h{int(horizon)}"
        if extra in data.columns:
            feature_columns.append(extra)
        for prefix in (
            "forecast_range_pct_h",
            "forecast_mid_delta_h",
            "forecast_close_minus_mid_h",
        ):
            candidate = f"{prefix}{int(horizon)}"
            if candidate in data.columns:
                feature_columns.append(candidate)
    for column in data.columns:
        if column.startswith("chronos_") and "_gap_h" in column:
            feature_columns.append(column)
    return [col for col in feature_columns if col in data.columns]


def encode_features(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base_train = train_df[["symbol", *feature_columns]].copy()
    base_score = score_df[["symbol", *feature_columns]].copy()
    combined = pd.concat([base_train, base_score], axis=0, ignore_index=True)
    encoded = pd.get_dummies(combined, columns=["symbol"], dtype=float)
    encoded = encoded.fillna(0.0)
    train_enc = encoded.iloc[: len(base_train)].reset_index(drop=True)
    score_enc = encoded.iloc[len(base_train) :].reset_index(drop=True)
    feature_names = list(train_enc.columns)
    return train_enc, score_enc, feature_names


def train_regressor(X: pd.DataFrame, y: pd.Series, *, seed: int) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=int(seed),
        n_jobs=4,
    )
    model.fit(X, y)
    return model


def train_models(
    train_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    seed: int,
) -> tuple[dict[str, xgb.XGBRegressor], pd.DataFrame, list[str]]:
    X_train, _, encoded_columns = encode_features(train_df, train_df.iloc[:0], feature_columns=feature_columns)
    models = {
        "high": train_regressor(X_train, train_df["target_high_ret"] - train_df["prior_high_ret"], seed=seed),
        "low": train_regressor(X_train, train_df["target_low_ret"] - train_df["prior_low_ret"], seed=seed + 1),
        "close": train_regressor(
            X_train,
            train_df["target_close_ret"] - train_df["prior_close_ret"],
            seed=seed + 2,
        ),
    }
    return models, X_train, encoded_columns


def score_rows(
    holdout_df: pd.DataFrame,
    train_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    models: dict[str, xgb.XGBRegressor],
    residual_scale: float,
) -> pd.DataFrame:
    _, X_holdout, _ = encode_features(train_df, holdout_df, feature_columns=feature_columns)
    scored = holdout_df.copy()
    high_residual = models["high"].predict(X_holdout)
    low_residual = models["low"].predict(X_holdout)
    close_residual = models["close"].predict(X_holdout)
    scale = float(residual_scale)
    scored["pred_high_ret_xgb"] = scored["prior_high_ret"] + scale * high_residual
    scored["pred_low_ret_xgb"] = scored["prior_low_ret"] + scale * low_residual
    scored["pred_close_ret_xgb"] = scored["prior_close_ret"] + scale * close_residual

    repaired_high: list[float] = []
    repaired_low: list[float] = []
    repaired_close: list[float] = []
    for row in scored.itertuples(index=False):
        ref = float(getattr(row, "reference_close"))
        high_price, low_price, close_price = _repair_forecast_prices(
            reference_close=ref,
            high_price=ref * (1.0 + float(getattr(row, "pred_high_ret_xgb"))),
            low_price=ref * (1.0 + float(getattr(row, "pred_low_ret_xgb"))),
            close_price=ref * (1.0 + float(getattr(row, "pred_close_ret_xgb"))),
        )
        repaired_high.append(float(high_price / ref - 1.0))
        repaired_low.append(float(low_price / ref - 1.0))
        repaired_close.append(float(close_price / ref - 1.0))
    scored["pred_high_ret_xgb"] = repaired_high
    scored["pred_low_ret_xgb"] = repaired_low
    scored["pred_close_ret_xgb"] = repaired_close
    return scored


def _clip_price(value: float, reference: float) -> float:
    price = float(value)
    if not np.isfinite(price) or price <= 0:
        return float(reference)
    return float(max(price, 0.01))


def build_action_row(
    row: pd.Series,
    cfg: SearchConfig,
) -> dict[str, Any]:
    symbol = str(row["symbol"]).upper()
    ref = float(row["reference_close"])
    pred_high = float(row["pred_high_ret_xgb"])
    pred_low = float(row["pred_low_ret_xgb"])
    pred_close = float(row["pred_close_ret_xgb"])

    directions = resolve_trade_directions(symbol, allow_short=True)
    fee_drag = 0.0015
    downside_mag = max(0.0, -pred_low)
    upside_mag = max(0.0, pred_high)
    long_edge = upside_mag - cfg.risk_penalty * downside_mag - fee_drag
    short_edge = downside_mag - cfg.risk_penalty * upside_mag - fee_drag

    choose_long = directions.can_long and long_edge >= max(short_edge, cfg.edge_threshold)
    choose_short = directions.can_short and short_edge > max(long_edge, cfg.edge_threshold)

    buy_amount = 0.0
    sell_amount = 0.0
    buy_price = ref
    sell_price = ref

    if choose_long:
        dip_gap = min(max(downside_mag * cfg.entry_alpha, 0.0), 0.03)
        tp_gap = min(max(upside_mag * cfg.exit_alpha, 0.003), 0.08)
        buy_price = _clip_price(ref * (1.0 - dip_gap), ref)
        sell_price = _clip_price(max(ref * (1.0 + tp_gap), buy_price * 1.003), ref)
        intensity = min(1.0, max(0.0, long_edge / max(cfg.edge_to_full_size, 1e-6)))
        buy_amount = 100.0 * intensity
    elif choose_short:
        rally_gap = min(max(upside_mag * cfg.entry_alpha, 0.0), 0.03)
        cover_gap = min(max(downside_mag * cfg.exit_alpha, 0.003), 0.08)
        sell_price = _clip_price(ref * (1.0 + rally_gap), ref)
        buy_price = _clip_price(min(ref * (1.0 - cover_gap), sell_price * 0.997), ref)
        intensity = min(1.0, max(0.0, short_edge / max(cfg.edge_to_full_size, 1e-6)))
        sell_amount = 100.0 * intensity

    trade_amount = max(buy_amount, sell_amount)
    return {
        "timestamp": pd.Timestamp(row["timestamp"]),
        "symbol": symbol,
        "buy_price": float(buy_price),
        "sell_price": float(sell_price),
        "buy_amount": float(buy_amount),
        "sell_amount": float(sell_amount),
        "trade_amount": float(trade_amount),
        "xgb_long_edge": float(long_edge),
        "xgb_short_edge": float(short_edge),
        "predicted_high_p50_h1": float(ref * (1.0 + pred_high)),
        "predicted_low_p50_h1": float(ref * (1.0 + pred_low)),
        "predicted_close_p50_h1": float(ref * (1.0 + pred_close)),
    }


def build_actions_and_bars(
    scored_df: pd.DataFrame,
    cfg: SearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_rows = [build_action_row(row, cfg) for _, row in scored_df.iterrows()]
    actions = pd.DataFrame(action_rows)

    bars = scored_df[
        [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    pred = actions[
        [
            "timestamp",
            "symbol",
            "predicted_high_p50_h1",
            "predicted_low_p50_h1",
            "predicted_close_p50_h1",
        ]
    ].copy()
    pred["predicted_high_p50_h24"] = pred["predicted_high_p50_h1"]
    pred["predicted_low_p50_h24"] = pred["predicted_low_p50_h1"]
    pred["predicted_close_p50_h24"] = pred["predicted_close_p50_h1"]
    bars = bars.merge(pred, on=["timestamp", "symbol"], how="left")
    return bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True), actions.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)


def compute_selection_score(metrics: list[WindowMetrics]) -> float:
    if not metrics:
        return float("-inf")
    min_sortino = min(item.sortino for item in metrics)
    mean_return = float(np.mean([item.total_return_pct for item in metrics]))
    mean_smooth = float(np.mean([item.pnl_smoothness_score for item in metrics]))
    mean_goodness = float(np.mean([item.goodness_score for item in metrics]))
    max_drawdown = max(item.max_drawdown_pct for item in metrics)
    return min_sortino + 0.05 * mean_return + 2.0 * mean_smooth + 0.35 * mean_goodness - 0.15 * max_drawdown


def evaluate_windows(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    symbols: list[str],
    eval_windows: list[int],
    cfg: SearchConfig,
    initial_cash: float,
) -> list[WindowMetrics]:
    results: list[WindowMetrics] = []
    latest_ts = pd.Timestamp(bars["timestamp"].max())
    for window_days in eval_windows:
        cutoff = latest_ts - pd.Timedelta(days=int(window_days))
        bars_window = bars[bars["timestamp"] >= cutoff].copy()
        actions_window = actions[actions["timestamp"] >= cutoff].copy()
        sim_cfg = PortfolioConfig(
            initial_cash=float(initial_cash),
            max_positions=int(cfg.max_positions),
            min_edge=float(cfg.edge_threshold),
            max_hold_hours=int(cfg.max_hold_hours),
            enforce_market_hours=True,
            close_at_eod=bool(cfg.close_at_eod),
            symbols=symbols,
            trade_amount_scale=100.0,
            decision_lag_bars=1,
            market_order_entry=bool(cfg.market_order_entry),
            entry_selection_mode=str(cfg.entry_selection_mode),
            entry_allocator_mode=str(cfg.entry_allocator_mode),
            entry_allocator_edge_power=float(cfg.entry_allocator_edge_power),
            sim_backend="python",
        )
        sim = run_portfolio_simulation(bars_window, actions_window, sim_cfg, horizon=1)
        results.append(
            WindowMetrics(
                window_days=int(window_days),
                total_return_pct=float(sim.metrics.get("total_return", 0.0) * 100.0),
                sortino=float(sim.metrics.get("sortino", 0.0)),
                max_drawdown_pct=float(sim.metrics.get("max_drawdown", 0.0) * 100.0),
                pnl_smoothness=float(sim.metrics.get("pnl_smoothness", 0.0)),
                pnl_smoothness_score=float(sim.metrics.get("pnl_smoothness_score", 0.0)),
                goodness_score=float(sim.metrics.get("goodness_score", 0.0)),
                num_buys=int(sim.metrics.get("num_buys", 0)),
                num_sells=int(sim.metrics.get("num_sells", 0)),
            )
        )
    return results


def iter_search_configs(args: argparse.Namespace) -> list[SearchConfig]:
    configs: list[SearchConfig] = []
    for values in itertools.product(
        parse_int_list(args.label_horizon_grid),
        parse_token_list(args.label_basis_grid),
        parse_float_list(args.residual_scale_grid),
        parse_float_list(args.risk_penalty_grid),
        parse_float_list(args.entry_alpha_grid),
        parse_float_list(args.exit_alpha_grid),
        parse_float_list(args.edge_threshold_grid),
        parse_float_list(args.edge_to_full_size_grid),
        parse_int_list(args.max_positions_grid),
        parse_int_list(args.max_hold_hours_grid),
        parse_bool_list(args.close_at_eod_grid),
        parse_bool_list(args.market_order_entry_grid),
        parse_token_list(args.entry_selection_mode_grid),
        parse_token_list(args.entry_allocator_mode_grid),
        parse_float_list(args.entry_allocator_edge_power_grid),
    ):
        configs.append(SearchConfig(*values))
    return configs


def save_models(
    models: dict[str, xgb.XGBRegressor],
    output_dir: Path,
) -> None:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.save_model(str(model_dir / f"{name}.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronos2 + XGBoost hourly Alpaca stock baseline")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--eval-windows", default="7,14,30")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--label-horizon-grid", default="2,4")
    parser.add_argument("--label-basis-grid", default="reference_close")
    parser.add_argument("--residual-scale-grid", default="0.0,0.5,1.0")
    parser.add_argument("--risk-penalty-grid", default="0.75,1.0")
    parser.add_argument("--entry-alpha-grid", default="0.25,0.5")
    parser.add_argument("--exit-alpha-grid", default="0.75")
    parser.add_argument("--edge-threshold-grid", default="0.002,0.004")
    parser.add_argument("--edge-to-full-size-grid", default="0.02")
    parser.add_argument("--max-positions-grid", default="5")
    parser.add_argument("--max-hold-hours-grid", default="2,4")
    parser.add_argument("--close-at-eod-grid", default="0,1")
    parser.add_argument("--market-order-entry-grid", default="0,1")
    parser.add_argument("--entry-selection-mode-grid", default="edge_rank,first_trigger")
    parser.add_argument("--entry-allocator-mode-grid", default="legacy,concentrated")
    parser.add_argument("--entry-allocator-edge-power-grid", default="2.0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    symbols = parse_csv_list(args.symbols)
    forecast_horizons = tuple(parse_int_list(args.forecast_horizons))
    eval_windows = parse_int_list(args.eval_windows)
    search_configs = iter_search_configs(args)
    if not search_configs:
        raise ValueError("No search configs generated")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path("experiments") / f"xgb_chronos_baseline_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    dataset_keys = sorted({(cfg.label_horizon_hours, cfg.label_basis) for cfg in search_configs})
    dataset_cache: dict[
        tuple[int, str],
        tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, xgb.XGBRegressor], list[str]],
    ] = {}

    for label_horizon, label_basis in dataset_keys:
        train_df, holdout_df = prepare_dataset(
            symbols=symbols,
            data_root=args.data_root,
            cache_root=args.cache_root,
            sequence_length=int(args.sequence_length),
            forecast_horizons=forecast_horizons,
            label_horizon_hours=int(label_horizon),
            label_basis=str(label_basis),
            validation_days=int(args.validation_days),
        )
        feature_columns = resolve_feature_columns(forecast_horizons, train_df)
        models, _, encoded_columns = train_models(train_df, feature_columns=feature_columns, seed=int(args.seed) + int(label_horizon) * 10)
        dataset_cache[(int(label_horizon), str(label_basis))] = (
            train_df,
            holdout_df,
            feature_columns,
            models,
            encoded_columns,
        )

    for cfg in search_configs:
        train_df, holdout_df, feature_columns, models, encoded_columns = dataset_cache[
            (int(cfg.label_horizon_hours), str(cfg.label_basis))
        ]
        scored_holdout = score_rows(
            holdout_df,
            train_df,
            feature_columns=feature_columns,
            models=models,
            residual_scale=float(cfg.residual_scale),
        )
        bars, actions = build_actions_and_bars(scored_holdout, cfg)
        window_metrics = evaluate_windows(
            bars,
            actions,
            symbols=symbols,
            eval_windows=eval_windows,
            cfg=cfg,
            initial_cash=float(args.initial_cash),
        )
        selection_score = compute_selection_score(window_metrics)
        result = {
            "config": asdict(cfg),
            "selection_score": float(selection_score),
            "window_metrics": [asdict(metric) for metric in window_metrics],
            "feature_columns": feature_columns,
            "encoded_feature_count": int(len(encoded_columns)),
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(scored_holdout)),
        }
        all_results.append(result)
        if best_result is None or float(result["selection_score"]) > float(best_result["selection_score"]):
            best_result = result

    if best_result is None:
        raise RuntimeError("No baseline result was produced")

    best_cfg = SearchConfig(**best_result["config"])
    train_df, holdout_df, feature_columns, models, _ = dataset_cache[
        (int(best_cfg.label_horizon_hours), str(best_cfg.label_basis))
    ]
    scored_holdout = score_rows(
        holdout_df,
        train_df,
        feature_columns=feature_columns,
        models=models,
        residual_scale=float(best_cfg.residual_scale),
    )
    best_bars, best_actions = build_actions_and_bars(scored_holdout, best_cfg)

    save_models(models, output_dir)
    best_actions.to_parquet(output_dir / "best_actions.parquet", index=False)
    best_bars.to_parquet(output_dir / "best_bars.parquet", index=False)
    (output_dir / "summary.json").write_text(json.dumps(best_result, indent=2, sort_keys=True))
    (output_dir / "results.json").write_text(json.dumps(all_results, indent=2, sort_keys=True))

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "forecast_horizons": list(forecast_horizons),
        "validation_days": int(args.validation_days),
        "eval_windows": eval_windows,
        "best": best_result,
        "output_dir": str(output_dir),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
