#!/usr/bin/env python3
"""Experimental allocation refiners for the work-steal daily Binance strategy."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from binance_worksteal.robust_eval import EvaluationWindow, build_recent_windows, build_start_state_config
from binance_worksteal.strategy import (
    EntrySizingContext,
    WorkStealConfig,
    compute_atr,
    compute_sma,
    compute_volume_ratio,
    get_fee,
    load_daily_bars,
    load_hourly_bars,
    run_worksteal_backtest,
)
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, summarize_scenario_results

LIVE_SYMBOLS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AVAXUSD", "LINKUSD",
    "AAVEUSD", "LTCUSD", "XRPUSD", "DOTUSD", "UNIUSD", "NEARUSD",
    "APTUSD", "ICPUSD", "SHIBUSD", "ADAUSD", "FILUSD", "ARBUSD",
    "OPUSD", "INJUSD", "SUIUSD", "TIAUSD", "SEIUSD", "ATOMUSD",
    "ALGOUSD", "BCHUSD", "BNBUSD", "TRXUSD", "PEPEUSD", "POLUSD",
]
SCREEN_START_STATES = ["flat", "BTCUSD", "ETHUSD"]
NUMERIC_COLUMNS = [
    "score",
    "candidate_rank_frac",
    "candidate_count",
    "slots_remaining_frac",
    "cash_frac",
    "equity_utilization",
    "market_breadth",
    "signal_close_delta",
    "execution_close_delta",
    "signal_range_pct",
    "sma20_gap",
    "mom3",
    "mom5",
    "mom20",
    "atr14_pct",
    "volume_ratio",
    "is_fdusd",
    "hold_base_asset",
    "has_forecast",
    "forecast_close_delta",
    "forecast_low_delta",
    "forecast_high_delta",
    "forecast_band_pct",
    "forecast_age_hours",
]


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), float(low)), float(high)))


def _safe_pct_delta(current: float, anchor: float) -> float:
    current_value = float(current)
    anchor_value = float(anchor)
    if current_value <= 0.0 or anchor_value <= 0.0:
        return 0.0
    return float((current_value - anchor_value) / anchor_value)


def _symbol_cache_candidates(symbol: str) -> list[str]:
    value = str(symbol or "").strip().upper()
    base = value[:-3] if value.endswith("USD") else value
    candidates = [value]
    if value.endswith("USD"):
        candidates.append(f"{base}USDT")
        candidates.append(f"{base}FDUSD")
    if base == "RNDR":
        candidates.extend(["RENDERUSD", "RENDERUSDT"])
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


class ForecastLookup:
    def __init__(self, cache_dir: Path, *, max_age_hours: float = 72.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_age = pd.Timedelta(hours=float(max_age_hours))
        self._frames: dict[str, pd.DataFrame] = {}

    def _load_frame(self, symbol: str) -> pd.DataFrame | None:
        for candidate in _symbol_cache_candidates(symbol):
            path = self.cache_dir / f"{candidate}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if frame.empty or "timestamp" not in frame.columns:
                continue
            frame = frame.copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            if frame.empty:
                continue
            self._frames[symbol] = frame
            return frame
        self._frames[symbol] = pd.DataFrame()
        return None

    def lookup(self, symbol: str, timestamp: pd.Timestamp) -> dict[str, float] | None:
        symbol_key = str(symbol or "").strip().upper()
        frame = self._frames.get(symbol_key)
        if frame is None:
            frame = self._load_frame(symbol_key)
        if frame is None or frame.empty:
            return None

        ts = _as_utc_timestamp(timestamp)
        idx = int(frame["timestamp"].searchsorted(ts, side="right") - 1)
        if idx < 0:
            return None
        row = frame.iloc[idx]
        row_ts = _as_utc_timestamp(row["timestamp"])
        if ts - row_ts > self.max_age:
            return None
        result: dict[str, float] = {}
        for key in (
            "predicted_close_p50",
            "predicted_close_p10",
            "predicted_close_p90",
            "predicted_high_p50",
            "predicted_low_p50",
        ):
            value = row.get(key)
            if value is None or not np.isfinite(float(value)):
                return None
            result[key] = float(value)
        result["forecast_age_hours"] = float((ts - row_ts).total_seconds() / 3600.0)
        return result


def build_feature_row(context: EntrySizingContext, forecast_lookup: ForecastLookup | None) -> dict[str, float | str]:
    signal_close = float(context.signal_bar.get("close", context.fill_price) or context.fill_price)
    signal_high = float(context.signal_bar.get("high", signal_close) or signal_close)
    signal_low = float(context.signal_bar.get("low", signal_close) or signal_close)
    execution_close = signal_close
    if context.execution_bar is not None:
        execution_close = float(context.execution_bar.get("close", signal_close) or signal_close)

    history = context.history
    sma20 = compute_sma(history, min(20, max(len(history), 1))) if not history.empty else signal_close
    atr_period = min(14, max(len(history) - 1, 1))
    atr14 = compute_atr(history, atr_period) if not history.empty else 0.0
    volume_ratio = compute_volume_ratio(history, min(20, max(len(history) - 1, 1))) if not history.empty else 1.0

    def momentum(period: int) -> float:
        if history.empty or len(history) <= period:
            return 0.0
        anchor = float(history.iloc[-(period + 1)]["close"])
        return _safe_pct_delta(signal_close, anchor)

    feature_row: dict[str, float | str] = {
        "timestamp": _as_utc_timestamp(context.timestamp).isoformat(),
        "signal_timestamp": _as_utc_timestamp(context.signal_timestamp).isoformat(),
        "symbol": context.symbol,
        "score": float(context.score),
        "candidate_rank_frac": 0.0
        if context.candidate_count <= 1
        else float(context.candidate_rank - 1) / float(context.candidate_count - 1),
        "candidate_count": float(context.candidate_count),
        "slots_remaining_frac": float(context.slots_remaining) / max(float(context.candidate_count), 1.0),
        "cash_frac": float(context.cash) / max(float(context.current_equity), 1.0),
        "equity_utilization": 1.0 - float(context.cash) / max(float(context.current_equity), 1.0),
        "market_breadth": float(context.market_breadth),
        "signal_close_delta": _safe_pct_delta(signal_close, context.fill_price),
        "execution_close_delta": _safe_pct_delta(execution_close, context.fill_price),
        "signal_range_pct": 0.0 if signal_close <= 0.0 else float((signal_high - signal_low) / signal_close),
        "sma20_gap": _safe_pct_delta(signal_close, sma20),
        "mom3": momentum(3),
        "mom5": momentum(5),
        "mom20": momentum(20),
        "atr14_pct": 0.0 if signal_close <= 0.0 else float(atr14 / signal_close),
        "volume_ratio": float(volume_ratio),
        "is_fdusd": 1.0 if get_fee(context.symbol, WorkStealConfig()) == 0.0 else 0.0,
        "hold_base_asset": 1.0 if context.hold_base_asset else 0.0,
        "has_forecast": 0.0,
        "forecast_close_delta": 0.0,
        "forecast_low_delta": 0.0,
        "forecast_high_delta": 0.0,
        "forecast_band_pct": 0.0,
        "forecast_age_hours": 999.0,
    }

    if forecast_lookup is not None:
        row = forecast_lookup.lookup(context.symbol, context.timestamp)
        if row is not None:
            feature_row["has_forecast"] = 1.0
            feature_row["forecast_close_delta"] = _safe_pct_delta(float(row["predicted_close_p50"]), context.fill_price)
            feature_row["forecast_low_delta"] = _safe_pct_delta(float(row["predicted_low_p50"]), context.fill_price)
            feature_row["forecast_high_delta"] = _safe_pct_delta(float(row["predicted_high_p50"]), context.fill_price)
            feature_row["forecast_band_pct"] = 0.0 if context.fill_price <= 0.0 else (
                float(row["predicted_close_p90"] - row["predicted_close_p10"]) / float(context.fill_price)
            )
            feature_row["forecast_age_hours"] = float(row["forecast_age_hours"])

    return feature_row


def heuristic_score_scale(features: dict[str, float | str], *, min_scale: float = 0.0, max_scale: float = 1.35) -> float:
    edge = float(features["signal_close_delta"])
    execution_edge = float(features["execution_close_delta"])
    rank_strength = 1.0 - float(features["candidate_rank_frac"])
    momentum = max(float(features["mom3"]), float(features["mom5"]), 0.0)
    atr_penalty = float(features["atr14_pct"])
    breadth_penalty = max(float(features["market_breadth"]) - 0.45, 0.0)
    raw = (
        0.60
        + 2.25 * max(edge, 0.0)
        + 1.20 * max(execution_edge, 0.0)
        + 0.15 * rank_strength
        + 0.35 * momentum
        - 2.50 * atr_penalty
        - 0.40 * breadth_penalty
    )
    if float(features["cash_frac"]) < 0.35:
        raw *= 0.92
    return _clip(raw, min_scale, max_scale)


def heuristic_chronos_scale(
    features: dict[str, float | str],
    *,
    min_scale: float = 0.0,
    max_scale: float = 1.35,
) -> float:
    raw = heuristic_score_scale(features, min_scale=min_scale, max_scale=max_scale)
    if float(features["has_forecast"]) <= 0.0:
        return _clip(raw * 0.95, min_scale, max_scale)
    raw += 2.4 * float(features["forecast_close_delta"])
    raw += 1.2 * min(float(features["forecast_low_delta"]), 0.0)
    raw += 0.6 * max(float(features["forecast_high_delta"]), 0.0)
    raw -= 1.1 * max(float(features["forecast_band_pct"]), 0.0)
    raw -= 0.02 * max(float(features["forecast_age_hours"]) - 24.0, 0.0)
    return _clip(raw, min_scale, max_scale)


@dataclass(frozen=True)
class RefinerConfig:
    min_scale: float = 0.0
    max_scale: float = 1.35
    hidden_size: int = 64
    epochs: int = 250
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 42


@dataclass(frozen=True)
class FeatureEncoder:
    numeric_columns: tuple[str, ...]
    symbol_columns: tuple[str, ...]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame.loc[:, list(self.numeric_columns)].astype(np.float32).to_numpy(copy=True)
        symbol_matrix = np.zeros((len(frame), len(self.symbol_columns)), dtype=np.float32)
        symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.symbol_columns)}
        for row_idx, symbol in enumerate(frame["symbol"].astype(str).tolist()):
            col_idx = symbol_to_idx.get(symbol.upper())
            if col_idx is not None:
                symbol_matrix[row_idx, col_idx] = 1.0
        return np.concatenate([numeric, symbol_matrix], axis=1)


class AllocationResidualHead(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, *, min_scale: float, max_scale: float) -> None:
        super().__init__()
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.net(x)).squeeze(-1)
        return self.min_scale + (self.max_scale - self.min_scale) * scale


@dataclass
class LearnedRefiner:
    encoder: FeatureEncoder
    model: AllocationResidualHead
    forecast_lookup: ForecastLookup | None
    config: RefinerConfig
    train_rows: int
    val_rows: int
    train_loss: float
    val_loss: float

    def scale(self, context: EntrySizingContext) -> float:
        feature_row = build_feature_row(context, self.forecast_lookup)
        frame = pd.DataFrame([feature_row])
        x = torch.from_numpy(self.encoder.transform(frame))
        with torch.no_grad():
            value = float(self.model(x).item())
        return _clip(value, self.config.min_scale, self.config.max_scale)


def _build_encoder(frame: pd.DataFrame) -> FeatureEncoder:
    symbols = tuple(sorted(str(symbol).upper() for symbol in frame["symbol"].dropna().unique().tolist()))
    return FeatureEncoder(numeric_columns=tuple(NUMERIC_COLUMNS), symbol_columns=symbols)


def _target_scale_from_row(row: pd.Series, config: RefinerConfig) -> float:
    utility = 12.0 * float(row["realized_return"])
    if row["exit_reason"] == "profit_target":
        utility += 0.45
    if row["exit_reason"] in {"stop_loss", "max_dd_exit", "margin_call"}:
        utility -= 1.15
    if row["exit_reason"] == "trailing_stop":
        utility += 0.10
    if row["exit_reason"] == "max_hold":
        utility -= 0.25
    utility -= min(float(row["hold_hours"]) / 240.0, 0.25)
    target = 1.0 / (1.0 + np.exp(-utility))
    return float(config.min_scale + (config.max_scale - config.min_scale) * target)


def fit_learned_refiner(
    train_examples: pd.DataFrame,
    *,
    forecast_lookup: ForecastLookup | None,
    config: RefinerConfig,
) -> LearnedRefiner:
    if train_examples.empty:
        raise ValueError("train_examples must not be empty.")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    frame = train_examples.copy().sort_values("timestamp").reset_index(drop=True)
    frame["target_scale"] = frame.apply(_target_scale_from_row, axis=1, config=config)
    split_idx = max(1, int(len(frame) * 0.8))
    if split_idx >= len(frame):
        split_idx = len(frame) - 1
    train_df = frame.iloc[:split_idx].reset_index(drop=True)
    val_df = frame.iloc[split_idx:].reset_index(drop=True)
    if val_df.empty:
        val_df = train_df.iloc[-1:].copy()
        train_df = train_df.iloc[:-1].reset_index(drop=True)
    if train_df.empty:
        train_df = val_df.copy()

    encoder = _build_encoder(frame)
    x_train = torch.from_numpy(encoder.transform(train_df))
    y_train = torch.from_numpy(train_df["target_scale"].to_numpy(dtype=np.float32))
    x_val = torch.from_numpy(encoder.transform(val_df))
    y_val = torch.from_numpy(val_df["target_scale"].to_numpy(dtype=np.float32))

    model = AllocationResidualHead(
        input_dim=x_train.shape[1],
        hidden_size=config.hidden_size,
        min_scale=config.min_scale,
        max_scale=config.max_scale,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    for _epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        train_loss = loss_fn(pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = float(loss_fn(val_pred, y_val).item())
            train_value = float(train_loss.item())
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_value
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return LearnedRefiner(
        encoder=encoder,
        model=model,
        forecast_lookup=forecast_lookup,
        config=config,
        train_rows=len(train_df),
        val_rows=len(val_df),
        train_loss=best_train_loss,
        val_loss=best_val_loss,
    )


def collect_training_examples(
    *,
    all_bars: dict[str, pd.DataFrame],
    intraday_bars: dict[str, pd.DataFrame],
    config: WorkStealConfig,
    windows: Sequence[EvaluationWindow],
    start_states: Sequence[str],
    forecast_lookup: ForecastLookup | None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for window in windows:
        for start_state in start_states:
            recorded_contexts: list[EntrySizingContext] = []

            def _record_context(context: EntrySizingContext) -> float:
                recorded_contexts.append(context)
                return 1.0

            start_label, scenario_config = build_start_state_config(
                base_config=config,
                all_bars=all_bars,
                start_state=start_state,
                start_date=window.start_date,
                end_date=window.end_date,
                starting_equity=config.initial_cash,
            )
            _equity_df, trades, _metrics = run_worksteal_backtest(
                {sym: frame.copy() for sym, frame in all_bars.items()},
                scenario_config,
                start_date=window.start_date,
                end_date=window.end_date,
                intraday_bars=intraday_bars,
                allocation_scale_fn=_record_context,
            )

            pending_contexts: deque[EntrySizingContext] = deque(recorded_contexts)
            open_positions: dict[str, deque[tuple[EntrySizingContext, object]]] = defaultdict(deque)
            for trade in trades:
                if trade.side in {"buy", "short"}:
                    if not pending_contexts:
                        raise ValueError("Recorded entry contexts do not align with trade log entries.")
                    context = pending_contexts.popleft()
                    if context.symbol != trade.symbol:
                        raise ValueError(
                            f"Entry context mismatch: expected {context.symbol}, observed trade {trade.symbol}."
                        )
                    open_positions[trade.symbol].append((context, trade))
                    continue
                if trade.side not in {"sell", "cover"}:
                    continue
                if not open_positions[trade.symbol]:
                    continue
                context, entry_trade = open_positions[trade.symbol].popleft()
                feature_row = build_feature_row(context, forecast_lookup)
                cost_basis = float(entry_trade.notional + max(entry_trade.fee, 0.0))
                realized_return = 0.0 if cost_basis <= 0.0 else float(trade.pnl / cost_basis)
                hold_hours = float((trade.timestamp - entry_trade.timestamp).total_seconds() / 3600.0)
                feature_row.update(
                    {
                        "window_label": window.label,
                        "start_state": start_label,
                        "entry_timestamp": _as_utc_timestamp(entry_trade.timestamp).isoformat(),
                        "exit_timestamp": _as_utc_timestamp(trade.timestamp).isoformat(),
                        "entry_price": float(entry_trade.price),
                        "exit_price": float(trade.price),
                        "entry_fee": float(entry_trade.fee),
                        "exit_fee": float(trade.fee),
                        "pnl": float(trade.pnl),
                        "realized_return": realized_return,
                        "hold_hours": hold_hours,
                        "exit_reason": str(trade.reason),
                    }
                )
                rows.append(feature_row)
    return pd.DataFrame(rows)


def _make_scale_callback(
    variant: str,
    *,
    forecast_lookup: ForecastLookup | None,
    learned_refiner: LearnedRefiner | None,
    min_scale: float,
    max_scale: float,
) -> Callable[[EntrySizingContext], float]:
    if variant == "fixed":
        return lambda _context: 1.0
    if variant == "heuristic_score":
        return lambda context: heuristic_score_scale(
            build_feature_row(context, forecast_lookup),
            min_scale=min_scale,
            max_scale=max_scale,
        )
    if variant == "heuristic_chronos":
        return lambda context: heuristic_chronos_scale(
            build_feature_row(context, forecast_lookup),
            min_scale=min_scale,
            max_scale=max_scale,
        )
    if variant == "learned_refiner":
        if learned_refiner is None:
            raise ValueError("learned_refiner callback requested before fitting the model.")
        return learned_refiner.scale
    raise ValueError(f"Unknown variant: {variant}.")


def evaluate_variants(
    *,
    all_bars: dict[str, pd.DataFrame],
    intraday_bars: dict[str, pd.DataFrame],
    config: WorkStealConfig,
    windows: Sequence[EvaluationWindow],
    start_states: Sequence[str],
    variants: Sequence[str],
    forecast_lookup: ForecastLookup | None,
    learned_refiner: LearnedRefiner | None,
    min_scale: float,
    max_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    for variant in variants:
        scale_callback = _make_scale_callback(
            variant,
            forecast_lookup=forecast_lookup,
            learned_refiner=learned_refiner,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        variant_scenarios: list[dict[str, float | str]] = []
        for window in windows:
            for start_state in start_states:
                start_label, scenario_config = build_start_state_config(
                    base_config=config,
                    all_bars=all_bars,
                    start_state=start_state,
                    start_date=window.start_date,
                    end_date=window.end_date,
                    starting_equity=config.initial_cash,
                )
                equity_df, trades, metrics = run_worksteal_backtest(
                    {sym: frame.copy() for sym, frame in all_bars.items()},
                    scenario_config,
                    start_date=window.start_date,
                    end_date=window.end_date,
                    intraday_bars=intraday_bars,
                    allocation_scale_fn=scale_callback,
                )
                equity_values = (
                    equity_df["equity"].astype(float).to_numpy(copy=False)
                    if not equity_df.empty
                    else np.asarray([], dtype=np.float64)
                )
                row = {
                    "variant": variant,
                    "window_label": window.label,
                    "start_state": start_label,
                    "return_pct": float(metrics.get("total_return_pct", 0.0)),
                    "annualized_return_pct": float(metrics.get("total_return_pct", 0.0))
                    if float(metrics.get("n_days", 0.0) or 0.0) <= 0
                    else float(((1.0 + float(metrics.get("total_return", 0.0))) ** (365.0 / max(float(metrics["n_days"]), 1.0)) - 1.0) * 100.0)
                    if float(metrics.get("total_return", 0.0)) > -1.0
                    else -100.0,
                    "sortino": float(metrics.get("sortino", 0.0)),
                    "max_drawdown_pct": abs(float(metrics.get("max_drawdown_pct", 0.0))),
                    "pnl_smoothness": float(compute_pnl_smoothness_from_equity(equity_values)),
                    "trade_count": float(len(trades)),
                    "n_days": float(metrics.get("n_days", 0.0)),
                }
                scenario_rows.append(row)
                variant_scenarios.append(row)
        summary = summarize_scenario_results(variant_scenarios)
        summary_rows.append({"variant": variant, **summary})
    return pd.DataFrame(summary_rows), pd.DataFrame(scenario_rows)


def build_runtime_config() -> WorkStealConfig:
    return WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.005,
        entry_proximity_bps=25.0,
        profit_target_pct=0.03,
        stop_loss_pct=0.08,
        max_positions=5,
        max_position_pct=0.10,
        max_hold_days=30,
        lookback_days=20,
        ref_price_method="sma",
        sma_filter_period=0,
        market_breadth_filter=0.0,
        trailing_stop_pct=0.03,
        max_drawdown_exit=0.25,
        enable_shorts=False,
        max_leverage=1.0,
        maker_fee=0.001,
        fdusd_fee=0.0,
        initial_cash=10000.0,
        rebalance_seeded_positions=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="trainingdatadailybinance")
    parser.add_argument("--hourly-data-dir", default="trainingdatahourly/crypto")
    parser.add_argument("--forecast-cache-dir", default="binanceneural/forecast_cache/h24")
    parser.add_argument("--symbols", nargs="+", default=LIVE_SYMBOLS)
    parser.add_argument("--end-date", default="2026-03-14")
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--window-count", type=int, default=3)
    parser.add_argument("--screen-start-states", nargs="+", default=SCREEN_START_STATES)
    parser.add_argument("--use-all-live-start-states", action="store_true")
    parser.add_argument("--output-dir", default="analysis/worksteal_hourly_20260318/allocation_refiner_v1")
    parser.add_argument("--min-scale", type=float, default=0.0)
    parser.add_argument("--max-scale", type=float, default=1.35)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_bars = load_daily_bars(args.data_dir, list(args.symbols))
    intraday_bars = load_hourly_bars(args.hourly_data_dir, list(args.symbols))
    if not all_bars:
        raise ValueError("No daily bars loaded for the requested symbols.")
    if not intraday_bars:
        raise ValueError("No hourly bars loaded for the requested symbols.")

    forecast_lookup = ForecastLookup(Path(args.forecast_cache_dir))
    base_config = build_runtime_config()
    windows = build_recent_windows(end_date=args.end_date, window_days=args.window_days, window_count=args.window_count)
    latest_window = windows[:1]
    train_windows = windows[1:] or windows[:1]
    loaded_symbols = list(all_bars.keys())
    screen_start_states = [
        str(item).upper() if str(item).lower() != "flat" else "flat"
        for item in args.screen_start_states
        if str(item).lower() == "flat" or str(item).upper() in loaded_symbols
    ]
    full_start_states = ["flat"] + loaded_symbols if args.use_all_live_start_states else screen_start_states

    train_examples = collect_training_examples(
        all_bars=all_bars,
        intraday_bars=intraday_bars,
        config=base_config,
        windows=train_windows,
        start_states=full_start_states,
        forecast_lookup=forecast_lookup,
    )
    train_examples.to_csv(output_dir / "train_examples.csv", index=False)

    refiner_config = RefinerConfig(
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
        hidden_size=int(args.hidden_size),
        epochs=int(args.epochs),
        seed=int(args.seed),
    )
    learned_refiner = fit_learned_refiner(
        train_examples,
        forecast_lookup=forecast_lookup,
        config=refiner_config,
    )

    with (output_dir / "fit_summary.json").open("w") as handle:
        json.dump(
            {
                "config": asdict(refiner_config),
                "train_rows": learned_refiner.train_rows,
                "val_rows": learned_refiner.val_rows,
                "train_loss": learned_refiner.train_loss,
                "val_loss": learned_refiner.val_loss,
            },
            handle,
            indent=2,
        )

    variants = ["fixed", "heuristic_score", "heuristic_chronos", "learned_refiner"]
    screen_summary, screen_scenarios = evaluate_variants(
        all_bars=all_bars,
        intraday_bars=intraday_bars,
        config=base_config,
        windows=latest_window,
        start_states=screen_start_states,
        variants=variants,
        forecast_lookup=forecast_lookup,
        learned_refiner=learned_refiner,
        min_scale=refiner_config.min_scale,
        max_scale=refiner_config.max_scale,
    )
    screen_summary.to_csv(output_dir / "screen_summary.csv", index=False)
    screen_scenarios.to_csv(output_dir / "screen_scenarios.csv", index=False)

    candidate_order = screen_summary.sort_values("robust_score", ascending=False)["variant"].tolist()
    full_variants = candidate_order[:2]
    if "fixed" not in full_variants:
        full_variants = ["fixed", *full_variants[:1]]
    full_variants = list(dict.fromkeys(full_variants))
    full_summary, full_scenarios = evaluate_variants(
        all_bars=all_bars,
        intraday_bars=intraday_bars,
        config=base_config,
        windows=windows,
        start_states=full_start_states,
        variants=full_variants,
        forecast_lookup=forecast_lookup,
        learned_refiner=learned_refiner,
        min_scale=refiner_config.min_scale,
        max_scale=refiner_config.max_scale,
    )
    full_summary.to_csv(output_dir / "full_summary.csv", index=False)
    full_scenarios.to_csv(output_dir / "full_scenarios.csv", index=False)
    print(full_summary.sort_values("robust_score", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
