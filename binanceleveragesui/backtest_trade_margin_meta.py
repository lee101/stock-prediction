#!/usr/bin/env python3
"""Backtest the live two-model meta margin bot over arbitrary time windows."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Backtests reuse live selection code paths but must never pollute live runtime logs.
os.environ.setdefault("MARGIN_META_DISABLE_LOG", "1")

from src.forecast_horizon_utils import resolve_required_forecast_horizons
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.execution import resolve_symbol_rules
from binanceleveragesui import trade_margin_meta as live_meta
from binanceleveragesui.validate_sim_vs_live import (
    generate_hourly_signals,
    load_5m_bars,
    set_seeds,
    simulate_5m_with_trace,
)

DATA_ROOT = REPO / "trainingdatahourlybinance"
FORECAST_CACHE = REPO / "binanceneural/forecast_cache"
DEFAULT_DOGE_CKPT = REPO / "binanceleveragesui/checkpoints/r5_DOGE_rw05_drop15/binanceneural_20260303_001154/epoch_008.pt"
DEFAULT_AAVE_CKPT = REPO / "binanceleveragesui/checkpoints/r5_AAVE_rw05_strides_long/binanceneural_20260303_023521/epoch_002.pt"
DEFAULT_MARGIN_HOURLY_RATE = 0.0000025457
FIVE_MIN_PERIODS_PER_YEAR = 12 * 24 * 365


def _resolve_initial_model_name(args) -> str:
    return str(getattr(args, "initial_model", "") or "").strip().lower()


def _to_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _latest_5m_timestamp(symbol: str) -> pd.Timestamp:
    path = REPO / "trainingdata5min" / f"{symbol}.csv"
    frame = pd.read_csv(path, usecols=["timestamp"])
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError(f"No valid 5m timestamps found for {symbol}")
    return ts.max()


def resolve_window(args, model_configs: dict[str, dict]) -> tuple[pd.Timestamp, pd.Timestamp]:
    if args.end:
        end_ts = _to_utc_timestamp(args.end)
    else:
        end_ts = min(_latest_5m_timestamp(cfg["symbol"]) for cfg in model_configs.values())

    if args.start:
        start_ts = _to_utc_timestamp(args.start)
    else:
        start_ts = end_ts - pd.Timedelta(days=float(args.days))

    if start_ts >= end_ts:
        raise ValueError(f"start must be before end: start={start_ts} end={end_ts}")
    return start_ts, end_ts


def _history_warmup_start(args, start_ts: pd.Timestamp) -> pd.Timestamp:
    warmup_hours = max(
        int(args.lookback),
        int(getattr(args, "profit_gate_lookback_hours", 0)),
        int(args.max_hold_hours),
        24,
    ) + 6
    return start_ts - pd.Timedelta(hours=warmup_hours)


def _resolve_checkpoint_forecast_horizons(feature_columns, horizon: int) -> tuple[int, ...]:
    return resolve_required_forecast_horizons(
        (int(horizon),),
        feature_columns=feature_columns,
        fallback_horizons=(int(horizon),),
    )


def _load_frame(
    data_symbol: str,
    checkpoint_path: Path,
    horizon: int,
    sequence_length: int,
    *,
    data_root: Path,
    forecast_cache_root: Path,
):
    model, normalizer, feature_columns, meta = load_policy_checkpoint(
        str(checkpoint_path),
        device="cuda",
        data_root=Path(data_root),
        forecast_cache_root=Path(forecast_cache_root),
    )
    seq_len = int(meta.get("sequence_length", sequence_length))
    forecast_horizons = _resolve_checkpoint_forecast_horizons(feature_columns, horizon)
    dm = ChronosSolDataModule(
        symbol=data_symbol,
        data_root=Path(data_root),
        forecast_cache_root=Path(forecast_cache_root),
        forecast_horizons=forecast_horizons,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=seq_len,
        split_config=SplitConfig(val_days=1, test_days=1),
        cache_only=True,
        max_history_days=365,
    )
    return model, normalizer, feature_columns, meta, dm.full_frame.copy(), forecast_horizons


def build_signal_histories_through(
    signal_maps: dict[str, dict[pd.Timestamp, dict]],
    cutoff_ts: pd.Timestamp,
) -> dict[str, live_meta.SignalHistory]:
    histories = {name: live_meta.SignalHistory() for name in signal_maps}
    for name, signals in signal_maps.items():
        for signal_hour in sorted(signals):
            if signal_hour > cutoff_ts:
                break
            live_meta._upsert_signal_history(histories[name], signals[signal_hour])
    return histories


def _make_sim_args(args, start_ts: pd.Timestamp, *, rules) -> argparse.Namespace:
    min_notional = float(rules.min_notional) if getattr(rules, "min_notional", None) else float(args.min_notional)
    tick_size = float(rules.tick_size) if getattr(rules, "tick_size", None) else float(args.tick_size)
    step_size = float(rules.step_size) if getattr(rules, "step_size", None) else float(args.step_size)
    raw_long_max_leverage = getattr(args, "long_max_leverage", None)
    long_max_leverage = float(args.max_leverage if raw_long_max_leverage is None else raw_long_max_leverage)
    raw_short_max_leverage = getattr(args, "short_max_leverage", None)
    short_max_leverage = float(long_max_leverage if raw_short_max_leverage is None else raw_short_max_leverage)
    return argparse.Namespace(
        fee=float(args.fee),
        fill_buffer_pct=float(args.fill_buffer_pct),
        initial_cash=float(args.initial_cash),
        start=start_ts.isoformat(),
        realistic=True,
        expiry_minutes=int(args.expiry_minutes),
        max_fill_fraction=float(args.max_fill_fraction),
        min_notional=min_notional,
        tick_size=tick_size,
        step_size=step_size,
        max_hold_hours=float(args.max_hold_hours),
        max_leverage=float(args.max_leverage),
        long_max_leverage=long_max_leverage,
        short_max_leverage=short_max_leverage,
        margin_hourly_rate=float(args.margin_hourly_rate),
        verbose=bool(args.verbose),
        live_like=True,
        use_order_expiry=bool(args.use_order_expiry),
        reprice_threshold=float(args.reprice_threshold),
        max_position_notional=args.max_position_notional,
        allow_short=bool(getattr(args, "allow_short", False)),
    )


def _entry_ts_or_none(raw_value) -> pd.Timestamp | None:
    if not raw_value:
        return None
    ts = pd.Timestamp(raw_value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _starting_mark_price(bars_5m: pd.DataFrame, start_ts: pd.Timestamp) -> float:
    if bars_5m.empty:
        raise ValueError("Cannot resolve starting mark price from empty 5m bars.")
    window = bars_5m[bars_5m["timestamp"] >= start_ts]
    if window.empty:
        window = bars_5m
    return max(0.0, float(window.iloc[0]["close"]))


def _mark_price_at_or_before(bars_5m: pd.DataFrame, ts: pd.Timestamp) -> float:
    if bars_5m.empty:
        return 0.0
    window = bars_5m[bars_5m["timestamp"] <= ts]
    if window.empty:
        window = bars_5m
    return max(0.0, float(window.iloc[-1]["close"]))


def _inventory_blocks_meta_rotation(args, *, inventory: float, bars_5m: pd.DataFrame, ts: pd.Timestamp, rules) -> bool:
    mark_price = _mark_price_at_or_before(bars_5m, ts)
    return bool(
        live_meta._has_effective_position(
            inventory,
            abs(float(inventory)) * mark_price,
            step_size=float(getattr(rules, "step_size", 0.0) or 0.0),
            max_position_notional=getattr(args, "max_position_notional", None),
        )
    )


def resolve_model_initial_state(
    args,
    *,
    name: str,
    start_ts: pd.Timestamp,
    bars_5m: pd.DataFrame,
    initial_equity: float,
) -> tuple[float, pd.Timestamp | None, float]:
    initial_model = _resolve_initial_model_name(args)
    initial_inv = float(getattr(args, "initial_inv", 0.0) or 0.0)
    if name != initial_model or abs(initial_inv) <= 1e-12:
        return 0.0, None, float(initial_equity)
    mark_price = _starting_mark_price(bars_5m, start_ts)
    adjusted_cash = float(initial_equity) - initial_inv * mark_price
    entry_ts = _entry_ts_or_none(getattr(args, "initial_entry_ts", None))
    if entry_ts is None:
        entry_ts = start_ts
    return initial_inv, entry_ts, adjusted_cash


def compute_equity_stats(trace: pd.DataFrame, initial_cash: float) -> dict:
    if trace.empty:
        return {
            "final_equity": float(initial_cash),
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "bars": 0,
            "sharpe_ratio": None,
            "sortino_ratio": None,
        }
    eq = trace["equity"].to_numpy(dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    returns = np.diff(eq) / np.clip(eq[:-1], 1e-12, None)
    sharpe_ratio = None
    sortino_ratio = None
    if returns.size > 0:
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        if np.isfinite(std_return) and std_return > 1e-12:
            sharpe_ratio = mean_return / std_return * float(np.sqrt(FIVE_MIN_PERIODS_PER_YEAR))
        downside = returns[returns < 0.0]
        if downside.size > 0:
            downside_std = float(np.std(downside))
            if np.isfinite(downside_std) and downside_std > 1e-12:
                sortino_ratio = mean_return / downside_std * float(np.sqrt(FIVE_MIN_PERIODS_PER_YEAR))
    return {
        "final_equity": float(eq[-1]),
        "return_pct": float(eq[-1] / initial_cash - 1.0) * 100.0,
        "max_drawdown_pct": float(dd.min()) * 100.0,
        "bars": int(len(trace)),
        "sharpe_ratio": None if sharpe_ratio is None else float(sharpe_ratio),
        "sortino_ratio": None if sortino_ratio is None else float(sortino_ratio),
    }


def _normalize_trade_side(side: str) -> str:
    normalized = str(side or "").lower()
    if normalized == "force_buy":
        return "buy"
    if normalized == "force_sell":
        return "sell"
    return normalized


def summarize_trades(trades: list[dict], *, initial_inventory: float = 0.0) -> dict[str, int]:
    summary = {
        "trade_count": int(len(trades)),
        "buy_count": 0,
        "sell_count": 0,
        "long_entry_count": 0,
        "long_exit_count": 0,
        "short_entry_count": 0,
        "short_exit_count": 0,
    }
    inventory = float(initial_inventory)
    for trade in trades:
        side = _normalize_trade_side(trade.get("side", ""))
        qty = max(0.0, float(trade.get("qty", 0.0)))
        if qty <= 0.0 or side not in {"buy", "sell"}:
            continue
        previous_inventory = inventory
        if side == "buy":
            summary["buy_count"] += 1
            inventory += qty
        else:
            summary["sell_count"] += 1
            inventory -= qty
        previous_side = "long" if previous_inventory > 1e-12 else ("short" if previous_inventory < -1e-12 else "")
        current_side = "long" if inventory > 1e-12 else ("short" if inventory < -1e-12 else "")
        if previous_side == "" and current_side == "long":
            summary["long_entry_count"] += 1
        elif previous_side == "" and current_side == "short":
            summary["short_entry_count"] += 1
        elif previous_side == "long" and current_side == "":
            summary["long_exit_count"] += 1
        elif previous_side == "short" and current_side == "":
            summary["short_exit_count"] += 1
    return summary


def run_single_symbol_backtest(args, start_ts: pd.Timestamp, end_ts: pd.Timestamp, *, name: str, signals, bars_5m, rules):
    sim_args = _make_sim_args(args, start_ts, rules=rules)
    initial_inv, initial_entry_ts, starting_cash = resolve_model_initial_state(
        args,
        name=name,
        start_ts=start_ts,
        bars_5m=bars_5m,
        initial_equity=float(args.initial_cash),
    )
    sim_args.initial_cash = float(starting_cash)
    trades, final_eq, cash, inv, trace = simulate_5m_with_trace(
        sim_args,
        signals,
        bars_5m,
        initial_inv=initial_inv,
        initial_entry_ts=initial_entry_ts,
        stop_after_cycle=False,
    )
    summary = compute_equity_stats(trace, float(args.initial_cash))
    trade_summary = summarize_trades(trades, initial_inventory=initial_inv)
    summary.update(
        {
            "model": name,
            "cash": float(cash),
            "inventory": float(inv),
            "final_equity": float(final_eq),
            "initial_inventory": float(initial_inv),
            "initial_entry_ts": initial_entry_ts.isoformat() if initial_entry_ts is not None else None,
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            **trade_summary,
        }
    )
    return summary, trace


def run_meta_backtest(
    args,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    signal_maps: dict[str, dict[pd.Timestamp, dict]],
    bars_by_model: dict[str, pd.DataFrame],
    rules_by_model: dict[str, object],
) -> dict:
    common_ts = None
    for frame in bars_by_model.values():
        ts = set(frame["timestamp"])
        common_ts = ts if common_ts is None else (common_ts & ts)
    common_index = sorted(ts for ts in (common_ts or set()) if start_ts <= ts <= end_ts)

    current_ts = start_ts
    current_equity = float(args.initial_cash)
    bootstrap_model = _resolve_initial_model_name(args)
    bootstrap_inv = float(getattr(args, "initial_inv", 0.0) or 0.0)
    bootstrap_entry_ts = _entry_ts_or_none(getattr(args, "initial_entry_ts", None))
    if abs(bootstrap_inv) > 1e-12 and bootstrap_entry_ts is None:
        bootstrap_entry_ts = start_ts
    active_model = bootstrap_model if bootstrap_model in signal_maps else ""
    switches: list[dict] = []
    segment_summaries: list[dict] = []
    equity_records: list[dict] = []

    while current_ts <= end_ts:
        selection_cutoff = current_ts.floor("h") - pd.Timedelta(hours=1)
        histories = build_signal_histories_through(signal_maps, selection_cutoff)

        if not active_model:
            score_snapshot = {
                name: float(
                    live_meta._run_hypothetical_score(
                        histories[name],
                        int(args.lookback),
                        float(live_meta.MODELS[name]["maker_fee"]),
                        float(args.max_leverage),
                        str(args.selection_metric),
                        allow_short=bool(args.allow_short),
                        max_long_leverage=getattr(args, "long_max_leverage", None),
                        max_short_leverage=getattr(args, "short_max_leverage", None),
                    )
                )
                for name in signal_maps
            }
            profit_gate_returns = None
            if (
                int(getattr(args, "profit_gate_lookback_hours", 0)) > 0
                and str(getattr(args, "profit_gate_mode", "hypothetical")) == "live_like"
            ):
                profit_gate_returns = live_meta.compute_live_like_profit_gate_returns(
                    histories,
                    asof_ts=current_ts,
                    lookback_hours=int(getattr(args, "profit_gate_lookback_hours", 0)),
                    args=args,
                    rules_by_model=rules_by_model,
                    initial_cash=float(current_equity),
                    bars_by_model=bars_by_model,
                    simulate_with_trace=simulate_5m_with_trace,
                )
            chosen = live_meta.select_model(
                histories,
                int(args.lookback),
                float(args.max_leverage),
                metric=str(args.selection_metric),
                selection_mode=str(args.selection_mode),
                cash_threshold=float(args.cash_threshold),
                current_model="",
                switch_margin=float(args.switch_margin),
                min_score_gap=float(args.min_score_gap),
                profit_gate_mode=str(getattr(args, "profit_gate_mode", "hypothetical")),
                profit_gate_lookback_hours=int(getattr(args, "profit_gate_lookback_hours", 0)),
                profit_gate_min_return=float(getattr(args, "profit_gate_min_return", 0.0)),
                profit_gate_returns=profit_gate_returns,
                allow_short=bool(args.allow_short),
                max_long_leverage=getattr(args, "long_max_leverage", None),
                max_short_leverage=getattr(args, "short_max_leverage", None),
            )
            if chosen != active_model:
                switches.append(
                    {
                        "ts": current_ts.isoformat(),
                        "selected": chosen or "cash",
                        "selection_cutoff": selection_cutoff.isoformat(),
                        "scores": score_snapshot,
                        "profit_returns": profit_gate_returns or {},
                    }
                )
            active_model = chosen

        if not active_model:
            next_eval_ts = min(end_ts + pd.Timedelta(minutes=5), current_ts.floor("h") + pd.Timedelta(hours=1))
            for ts in common_index:
                if current_ts <= ts < next_eval_ts:
                    equity_records.append({"timestamp": ts, "equity": current_equity, "model": "cash"})
            current_ts = next_eval_ts
            continue

        segment_bars = bars_by_model[active_model]
        segment_bars = segment_bars[(segment_bars["timestamp"] >= current_ts) & (segment_bars["timestamp"] <= end_ts)].reset_index(drop=True)
        if segment_bars.empty:
            break

        sim_args = _make_sim_args(args, current_ts, rules=rules_by_model[active_model])
        segment_initial_inv = bootstrap_inv if active_model == bootstrap_model and abs(bootstrap_inv) > 1e-12 else 0.0
        segment_initial_entry_ts = bootstrap_entry_ts if abs(segment_initial_inv) > 1e-12 else None
        mark_price = _starting_mark_price(segment_bars, current_ts)
        sim_args.initial_cash = float(current_equity) - segment_initial_inv * mark_price
        trades, final_eq, cash, inv, trace = simulate_5m_with_trace(
            sim_args,
            signal_maps[active_model],
            segment_bars,
            initial_inv=segment_initial_inv,
            initial_entry_ts=segment_initial_entry_ts,
            stop_after_cycle=True,
        )
        if trace.empty:
            break

        for row in trace.itertuples(index=False):
            equity_records.append(
                {
                    "timestamp": row.timestamp,
                    "equity": float(row.equity),
                    "model": active_model,
                }
            )

        trade_summary = summarize_trades(trades, initial_inventory=segment_initial_inv)
        segment_summary = {
            "model": active_model,
            "start": current_ts.isoformat(),
            "end": pd.Timestamp(trace["timestamp"].iloc[-1]).isoformat(),
            "final_equity": float(final_eq),
            "initial_inventory": float(segment_initial_inv),
            "initial_entry_ts": segment_initial_entry_ts.isoformat() if segment_initial_entry_ts is not None else None,
            **trade_summary,
        }
        segment_summaries.append(segment_summary)
        current_equity = float(final_eq)
        bootstrap_inv = 0.0
        bootstrap_entry_ts = None

        final_trace_ts = pd.Timestamp(trace["timestamp"].iloc[-1])
        if _inventory_blocks_meta_rotation(
            args,
            inventory=float(inv),
            bars_5m=segment_bars,
            ts=final_trace_ts,
            rules=rules_by_model[active_model],
        ):
            current_ts = pd.Timestamp(trace["timestamp"].iloc[-1]) + pd.Timedelta(minutes=5)
            break

        if trades:
            active_model = ""
            current_ts = final_trace_ts + pd.Timedelta(minutes=5)
            continue

        current_ts = end_ts + pd.Timedelta(minutes=5)

    trace_df = (
        pd.DataFrame(equity_records, columns=["timestamp", "equity", "model"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    stats = compute_equity_stats(trace_df, float(args.initial_cash))
    stats.update(
        {
            "trade_count": int(sum(segment["trade_count"] for segment in segment_summaries)),
            "switch_count": int(len(switches)),
            "segments": segment_summaries,
            "switches": switches,
        }
    )
    return {"summary": stats, "trace": trace_df}


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest the live two-model meta margin bot over arbitrary windows")
    parser.add_argument("--start", default=None, help="UTC ISO start timestamp. Overrides --days.")
    parser.add_argument("--end", default=None, help="UTC ISO end timestamp. Defaults to latest common 5m bar.")
    parser.add_argument("--days", type=float, default=7.0, help="Trailing window size in days when --start is omitted.")
    parser.add_argument("--model-a-name", default=live_meta.MODEL_SLOT_DEFAULTS[0][1])
    parser.add_argument("--model-a-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["symbol"])
    parser.add_argument("--model-a-data-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["data_symbol"])
    parser.add_argument("--model-a-base-asset", default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["base_asset"])
    parser.add_argument("--model-a-maker-fee", type=float, default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["maker_fee"])
    parser.add_argument("--model-a-checkpoint", "--doge-checkpoint", dest="model_a_checkpoint", default=str(DEFAULT_DOGE_CKPT))
    parser.add_argument("--model-b-name", default=live_meta.MODEL_SLOT_DEFAULTS[1][1])
    parser.add_argument("--model-b-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["symbol"])
    parser.add_argument("--model-b-data-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["data_symbol"])
    parser.add_argument("--model-b-base-asset", default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["base_asset"])
    parser.add_argument("--model-b-maker-fee", type=float, default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["maker_fee"])
    parser.add_argument("--model-b-checkpoint", "--aave-checkpoint", dest="model_b_checkpoint", default=str(DEFAULT_AAVE_CKPT))
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument(
        "--initial-model",
        default="",
        help="Optional model active at window start. Combine with --initial-inv to seed a carried position.",
    )
    parser.add_argument(
        "--initial-inv",
        type=float,
        default=0.0,
        help="Optional signed starting inventory for --initial-model. Positive=long, negative=short. Zero seeds a flat active model.",
    )
    parser.add_argument("--initial-entry-ts", default=None, help="Optional ISO timestamp describing when the starting position was opened.")
    parser.add_argument("--max-leverage", type=float, default=2.30)
    parser.add_argument("--long-max-leverage", type=float, default=None)
    parser.add_argument("--short-max-leverage", type=float, default=None)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--selection-mode", default="winner_cash", choices=live_meta.SUPPORTED_SELECTION_MODES)
    parser.add_argument("--selection-metric", default="calmar", choices=live_meta.SUPPORTED_SELECTION_METRICS)
    parser.add_argument("--cash-threshold", type=float, default=0.0)
    parser.add_argument("--switch-margin", type=float, default=0.005)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--profit-gate-mode", default="hypothetical", choices=live_meta.SUPPORTED_PROFIT_GATE_MODES)
    parser.add_argument("--profit-gate-lookback-hours", type=int, default=0)
    parser.add_argument("--profit-gate-min-return", type=float, default=0.0)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--intensity-scale", type=float, default=5.0)
    parser.add_argument("--max-hold-hours", type=float, default=6.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    parser.add_argument("--expiry-minutes", type=int, default=90)
    parser.add_argument("--max-fill-fraction", type=float, default=0.01)
    parser.add_argument("--min-notional", type=float, default=5.0)
    parser.add_argument("--tick-size", type=float, default=0.00001)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--reprice-threshold", type=float, default=0.003)
    parser.add_argument("--use-order-expiry", action="store_true")
    parser.add_argument("--margin-hourly-rate", type=float, default=DEFAULT_MARGIN_HOURLY_RATE)
    parser.add_argument("--max-position-notional", type=float, default=None)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--forecast-cache", default=str(FORECAST_CACHE))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_specs = live_meta.build_model_specs_from_args(args)
    model_configs = {
        spec["name"]: {
            "symbol": spec["symbol"],
            "data_symbol": spec["data_symbol"],
            "base_asset": spec["base_asset"],
            "maker_fee": float(spec["maker_fee"]),
            "checkpoint": Path(spec["checkpoint"]) if spec.get("checkpoint") is not None else None,
        }
        for spec in model_specs
    }
    live_meta.apply_model_specs(model_specs)
    start_ts, end_ts = resolve_window(args, model_configs)
    torch.use_deterministic_algorithms(True)
    set_seeds(42)

    signal_start_ts = _history_warmup_start(args, start_ts)
    signal_args = argparse.Namespace(
        start=signal_start_ts.isoformat(),
        horizon=int(args.horizon),
        intensity_scale=float(args.intensity_scale),
        sequence_length=int(args.sequence_length),
    )

    signal_maps: dict[str, dict[pd.Timestamp, dict]] = {}
    bars_by_model: dict[str, pd.DataFrame] = {}
    rules_by_model: dict[str, object] = {}
    individual_results: dict[str, dict] = {}
    forecast_horizons_by_model: dict[str, list[int]] = {}
    data_root = Path(args.data_root)
    forecast_cache_root = Path(args.forecast_cache)

    for name, cfg in model_configs.items():
        checkpoint_path = cfg["checkpoint"]
        if checkpoint_path is None:
            raise ValueError(f"Missing checkpoint for model '{name}'.")
        rules_by_model[name] = resolve_symbol_rules(cfg["symbol"])
        model, normalizer, feature_columns, meta, frame, forecast_horizons = _load_frame(
            cfg["data_symbol"],
            checkpoint_path,
            int(args.horizon),
            int(args.sequence_length),
            data_root=data_root,
            forecast_cache_root=forecast_cache_root,
        )
        forecast_horizons_by_model[name] = [int(h) for h in forecast_horizons]
        signal_args.symbol = cfg["symbol"]
        signal_maps[name] = generate_hourly_signals(signal_args, frame, model, normalizer, feature_columns, meta)
        bars_by_model[name] = load_5m_bars(cfg["symbol"], signal_start_ts - pd.Timedelta(hours=1), end_ts)
        summary, _trace = run_single_symbol_backtest(
            args,
            start_ts,
            end_ts,
            name=name,
            signals=signal_maps[name],
            bars_5m=bars_by_model[name],
            rules=rules_by_model[name],
        )
        individual_results[name] = summary

    meta_result = run_meta_backtest(
        args,
        start_ts,
        end_ts,
        signal_maps=signal_maps,
        bars_by_model=bars_by_model,
        rules_by_model=rules_by_model,
    )

    report = {
        "window": {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
        "config": {
            "models": [
                {
                    "name": str(spec["name"]),
                    "symbol": str(spec["symbol"]),
                    "data_symbol": str(spec["data_symbol"]),
                    "base_asset": str(spec["base_asset"]),
                    "maker_fee": float(spec["maker_fee"]),
                    "checkpoint": (
                        None
                        if spec.get("checkpoint") is None
                        else str(Path(spec["checkpoint"]).resolve())
                    ),
                }
                for spec in model_specs
            ],
            "initial_cash": float(args.initial_cash),
            "initial_model": str(args.initial_model or ""),
            "initial_inv": float(args.initial_inv),
            "initial_entry_ts": args.initial_entry_ts,
            "max_leverage": float(args.max_leverage),
            "long_max_leverage": float(args.long_max_leverage if args.long_max_leverage is not None else args.max_leverage),
            "short_max_leverage": float(args.short_max_leverage if args.short_max_leverage is not None else (args.long_max_leverage if args.long_max_leverage is not None else args.max_leverage)),
            "allow_short": bool(args.allow_short),
            "lookback": int(args.lookback),
            "selection_mode": str(args.selection_mode),
            "selection_metric": str(args.selection_metric),
            "cash_threshold": float(args.cash_threshold),
            "switch_margin": float(args.switch_margin),
            "min_score_gap": float(args.min_score_gap),
            "profit_gate_mode": str(getattr(args, "profit_gate_mode", "hypothetical")),
            "profit_gate_lookback_hours": int(getattr(args, "profit_gate_lookback_hours", 0)),
            "profit_gate_min_return": float(getattr(args, "profit_gate_min_return", 0.0)),
            "intensity_scale": float(args.intensity_scale),
            "max_hold_hours": float(args.max_hold_hours),
            "data_root": str(data_root.resolve()),
            "forecast_cache": str(forecast_cache_root.resolve()),
            "resolved_forecast_horizons": forecast_horizons_by_model,
        },
        "individual": individual_results,
        "meta": meta_result["summary"],
    }

    print(json.dumps(report, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
