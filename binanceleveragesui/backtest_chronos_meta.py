#!/usr/bin/env python3
"""Chronos-2 PnL forecast meta-selector experiment.

Uses Chronos-2 to forecast each model's equity curve 6h ahead and picks
the model with the best forecasted growth.  Compare against the existing
omega/calmar-based meta-switcher.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("MARGIN_META_DISABLE_LOG", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from binanceleveragesui.backtest_trade_margin_meta import (
    _history_warmup_start,
    _load_frame,
    _make_sim_args,
    _starting_mark_price,
    _mark_price_at_or_before,
    _inventory_blocks_meta_rotation,
    compute_equity_stats,
    resolve_model_initial_state,
    resolve_window,
    run_single_symbol_backtest,
    summarize_trades,
    DEFAULT_DOGE_CKPT,
    DEFAULT_AAVE_CKPT,
    DEFAULT_MARGIN_HOURLY_RATE,
)
from binanceleveragesui.validate_sim_vs_live import (
    generate_hourly_signals,
    load_5m_bars,
    set_seeds,
    simulate_5m_with_trace,
)
from binanceleveragesui import trade_margin_meta as live_meta
from binanceneural.execution import resolve_symbol_rules


def resample_trace_hourly(trace_5m: pd.DataFrame) -> pd.Series:
    """Last equity value per hour from a 5m trace."""
    df = trace_5m[["timestamp", "equity"]].copy()
    df = df.set_index("timestamp").sort_index()
    hourly = df["equity"].resample("1h").last().dropna()
    return hourly


def forecast_equity_growth(
    pipeline, equity_array: np.ndarray, prediction_length: int = 6
) -> float:
    """Forecast equity growth over prediction_length hours using Chronos-2."""
    if len(equity_array) < 3:
        return 0.0
    current = float(equity_array[-1])
    if current <= 0:
        return 0.0
    tensor = torch.tensor(equity_array, dtype=torch.float32)
    with torch.no_grad():
        forecasts = pipeline.predict([tensor], prediction_length=prediction_length)
    t = forecasts[0]  # shape (1, num_quantiles, prediction_length)
    median_idx = t.shape[1] // 2
    forecast_final = float(t[0, median_idx, -1].item())
    return (forecast_final / current) - 1.0


def chronos_select_model(
    pipeline,
    hourly_equities: dict[str, pd.Series],
    current_ts: pd.Timestamp,
    context_hours: int = 48,
    prediction_hours: int = 6,
    growth_threshold: float = 0.0,
) -> tuple[str, dict]:
    """Pick the model with highest forecasted equity growth."""
    forecasts = {}
    for name, eq_series in hourly_equities.items():
        available = eq_series[eq_series.index <= current_ts]
        if len(available) < 3:
            forecasts[name] = 0.0
            continue
        context = available.values[-context_hours:]
        forecasts[name] = forecast_equity_growth(
            pipeline, context, prediction_length=prediction_hours
        )

    best_name = max(forecasts, key=forecasts.get)
    best_growth = forecasts[best_name]
    if best_growth <= growth_threshold:
        return "", forecasts
    return best_name, forecasts


def run_chronos_meta_backtest(
    args,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    signal_maps: dict[str, dict[pd.Timestamp, dict]],
    bars_by_model: dict[str, pd.DataFrame],
    rules_by_model: dict,
    hourly_equities: dict[str, pd.Series],
    pipeline,
) -> dict:
    """Meta-backtest using Chronos-2 forecasted equity growth for model selection."""
    context_hours = int(args.chronos_context_hours)
    prediction_hours = int(args.chronos_prediction_hours)
    growth_threshold = float(args.chronos_growth_threshold)

    current_ts = start_ts
    current_equity = float(args.initial_cash)
    active_model = ""
    switches: list[dict] = []
    segment_summaries: list[dict] = []
    equity_records: list[dict] = []

    common_ts = None
    for frame in bars_by_model.values():
        ts_set = set(frame["timestamp"])
        common_ts = ts_set if common_ts is None else (common_ts & ts_set)
    common_index = sorted(ts for ts in (common_ts or set()) if start_ts <= ts <= end_ts)

    while current_ts <= end_ts:
        if not active_model:
            chosen, forecasts = chronos_select_model(
                pipeline,
                hourly_equities,
                current_ts,
                context_hours=context_hours,
                prediction_hours=prediction_hours,
                growth_threshold=growth_threshold,
            )
            if chosen != active_model:
                switches.append({
                    "ts": current_ts.isoformat(),
                    "selected": chosen or "cash",
                    "forecasts": {k: round(v, 6) for k, v in forecasts.items()},
                })
                if args.verbose:
                    fcast_str = " ".join(f"{k}={v:+.4f}" for k, v in forecasts.items())
                    print(f"[chronos-meta] {current_ts} select={chosen or 'cash'} {fcast_str}")
            active_model = chosen

        if not active_model:
            next_eval_ts = min(
                end_ts + pd.Timedelta(minutes=5),
                current_ts.floor("h") + pd.Timedelta(hours=1),
            )
            for ts in common_index:
                if current_ts <= ts < next_eval_ts:
                    equity_records.append({"timestamp": ts, "equity": current_equity, "model": "cash"})
            current_ts = next_eval_ts
            continue

        segment_bars = bars_by_model[active_model]
        segment_bars = segment_bars[
            (segment_bars["timestamp"] >= current_ts) & (segment_bars["timestamp"] <= end_ts)
        ].reset_index(drop=True)
        if segment_bars.empty:
            break

        sim_args = _make_sim_args(args, current_ts, rules=rules_by_model[active_model])
        mark_price = _starting_mark_price(segment_bars, current_ts)
        sim_args.initial_cash = float(current_equity)

        trades, final_eq, cash, inv, trace = simulate_5m_with_trace(
            sim_args,
            signal_maps[active_model],
            segment_bars,
            initial_inv=0.0,
            initial_entry_ts=None,
            stop_after_cycle=True,
        )
        if trace.empty:
            break

        for row in trace.itertuples(index=False):
            equity_records.append({
                "timestamp": row.timestamp,
                "equity": float(row.equity),
                "model": active_model,
            })

        trade_summary = summarize_trades(trades, initial_inventory=0.0)
        segment_summaries.append({
            "model": active_model,
            "start": current_ts.isoformat(),
            "end": pd.Timestamp(trace["timestamp"].iloc[-1]).isoformat(),
            "final_equity": float(final_eq),
            **trade_summary,
        })
        current_equity = float(final_eq)

        final_trace_ts = pd.Timestamp(trace["timestamp"].iloc[-1])
        if _inventory_blocks_meta_rotation(
            args,
            inventory=float(inv),
            bars_5m=segment_bars,
            ts=final_trace_ts,
            rules=rules_by_model[active_model],
        ):
            current_ts = final_trace_ts + pd.Timedelta(minutes=5)
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
    stats.update({
        "trade_count": int(sum(s["trade_count"] for s in segment_summaries)),
        "switch_count": int(len(switches)),
        "segments": segment_summaries,
        "switches": switches,
    })
    return {"summary": stats, "trace": trace_df}


def parse_args():
    p = argparse.ArgumentParser(description="Chronos-2 PnL forecast meta-selector backtest")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--days", type=float, default=7.0)
    p.add_argument("--model-a-name", default=live_meta.MODEL_SLOT_DEFAULTS[0][1])
    p.add_argument("--model-a-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["symbol"])
    p.add_argument("--model-a-data-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["data_symbol"])
    p.add_argument("--model-a-base-asset", default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["base_asset"])
    p.add_argument("--model-a-maker-fee", type=float, default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["maker_fee"])
    p.add_argument("--model-a-checkpoint", "--doge-checkpoint", dest="model_a_checkpoint", default=str(DEFAULT_DOGE_CKPT))
    p.add_argument("--model-b-name", default=live_meta.MODEL_SLOT_DEFAULTS[1][1])
    p.add_argument("--model-b-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["symbol"])
    p.add_argument("--model-b-data-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["data_symbol"])
    p.add_argument("--model-b-base-asset", default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["base_asset"])
    p.add_argument("--model-b-maker-fee", type=float, default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["maker_fee"])
    p.add_argument("--model-b-checkpoint", "--aave-checkpoint", dest="model_b_checkpoint", default=str(DEFAULT_AAVE_CKPT))
    p.add_argument("--initial-cash", type=float, default=10000.0)
    p.add_argument("--max-leverage", type=float, default=2.0)
    p.add_argument("--long-max-leverage", type=float, default=None)
    p.add_argument("--short-max-leverage", type=float, default=None)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--sequence-length", type=int, default=72)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--intensity-scale", type=float, default=5.0)
    p.add_argument("--max-hold-hours", type=float, default=6.0)
    p.add_argument("--lookback", type=int, default=1)
    p.add_argument("--profit-gate-lookback-hours", type=int, default=24)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--fill-buffer-pct", type=float, default=0.0)
    p.add_argument("--expiry-minutes", type=int, default=90)
    p.add_argument("--max-fill-fraction", type=float, default=0.01)
    p.add_argument("--min-notional", type=float, default=5.0)
    p.add_argument("--tick-size", type=float, default=0.00001)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--reprice-threshold", type=float, default=0.003)
    p.add_argument("--use-order-expiry", action="store_true")
    p.add_argument("--margin-hourly-rate", type=float, default=DEFAULT_MARGIN_HOURLY_RATE)
    p.add_argument("--max-position-notional", type=float, default=None)
    p.add_argument("--data-root", default=str(REPO / "trainingdatahourlybinance"))
    p.add_argument("--forecast-cache", default=str(REPO / "binanceneural/forecast_cache"))
    p.add_argument("--chronos-model-id", default="amazon/chronos-2")
    p.add_argument("--chronos-context-hours", type=int, default=48)
    p.add_argument("--chronos-prediction-hours", type=int, default=6)
    p.add_argument("--chronos-growth-threshold", type=float, default=0.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    set_seeds(42)

    model_configs = {
        args.model_a_name: {
            "symbol": args.model_a_symbol,
            "data_symbol": args.model_a_data_symbol,
            "base_asset": args.model_a_base_asset,
            "maker_fee": args.model_a_maker_fee,
            "checkpoint": args.model_a_checkpoint,
        },
        args.model_b_name: {
            "symbol": args.model_b_symbol,
            "data_symbol": args.model_b_data_symbol,
            "base_asset": args.model_b_base_asset,
            "maker_fee": args.model_b_maker_fee,
            "checkpoint": args.model_b_checkpoint,
        },
    }

    live_meta.MODELS = {
        name: {"maker_fee": cfg["maker_fee"], "symbol": cfg["symbol"]}
        for name, cfg in model_configs.items()
    }

    start_ts, end_ts = resolve_window(args, model_configs)
    warmup_start = _history_warmup_start(args, start_ts)
    print(f"Window: {start_ts} -> {end_ts} (warmup from {warmup_start})")

    signal_maps = {}
    bars_by_model = {}
    rules_by_model = {}

    for name, cfg in model_configs.items():
        print(f"\nLoading {name} ({cfg['data_symbol']})...")
        model, normalizer, feature_columns, meta, frame, forecast_horizons = _load_frame(
            cfg["data_symbol"],
            Path(cfg["checkpoint"]),
            int(args.horizon),
            int(args.sequence_length),
            data_root=Path(args.data_root),
            forecast_cache_root=Path(args.forecast_cache),
        )
        sig_args = argparse.Namespace(
            start=(warmup_start - pd.Timedelta(hours=2)).isoformat(),
            sequence_length=int(meta.get("sequence_length", args.sequence_length)),
            horizon=int(args.horizon),
            intensity_scale=float(args.intensity_scale),
            symbol=cfg["symbol"],
        )
        signals = generate_hourly_signals(sig_args, frame, model, normalizer, feature_columns, meta)
        signal_maps[name] = signals
        print(f"  {name}: {len(signals)} hourly signals")

        bars_5m = load_5m_bars(
            cfg["symbol"],
            warmup_start - pd.Timedelta(hours=1),
            end_ts,
            data_root=args.data_root,
        )
        bars_by_model[name] = bars_5m
        print(f"  {name}: {len(bars_5m)} 5m bars")

        rules = resolve_symbol_rules(cfg["symbol"])
        rules_by_model[name] = rules

    # Phase 1: pre-compute full equity curves per model
    print("\n=== Phase 1: Individual equity curves ===")
    individual_results = {}
    hourly_equities = {}
    for name in model_configs:
        summary, trace = run_single_symbol_backtest(
            args, warmup_start, end_ts,
            name=name, signals=signal_maps[name],
            bars_5m=bars_by_model[name], rules=rules_by_model[name],
        )
        individual_results[name] = summary
        hourly_equities[name] = resample_trace_hourly(trace)
        print(f"  {name}: {summary['return_pct']:+.2f}% ret, sort={summary.get('sortino_ratio') or 'n/a'}, {summary['trade_count']} trades")

    # Phase 2: Chronos-2 meta selection
    print("\n=== Phase 2: Chronos-2 meta-selector ===")
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        args.chronos_model_id, device_map="cuda"
    )
    print(f"  Loaded {args.chronos_model_id}")

    result = run_chronos_meta_backtest(
        args, start_ts, end_ts,
        signal_maps=signal_maps,
        bars_by_model=bars_by_model,
        rules_by_model=rules_by_model,
        hourly_equities=hourly_equities,
        pipeline=pipeline,
    )

    meta_stats = result["summary"]
    print(f"\n=== Results ===")
    for name, r in individual_results.items():
        print(f"  {name} standalone: {r['return_pct']:+.2f}% sort={r.get('sortino_ratio') or 'n/a'}")
    print(f"  chronos-meta:     {meta_stats['return_pct']:+.2f}% sort={meta_stats.get('sortino_ratio') or 'n/a'} "
          f"trades={meta_stats['trade_count']} switches={meta_stats['switch_count']}")

    output = {
        "window": {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
        "config": {
            "models": [
                {"name": n, **{k: str(v) if isinstance(v, Path) else v for k, v in c.items()}}
                for n, c in model_configs.items()
            ],
            "initial_cash": float(args.initial_cash),
            "chronos_model_id": args.chronos_model_id,
            "chronos_context_hours": int(args.chronos_context_hours),
            "chronos_prediction_hours": int(args.chronos_prediction_hours),
            "chronos_growth_threshold": float(args.chronos_growth_threshold),
        },
        "individual": individual_results,
        "chronos_meta": meta_stats,
    }

    print(json.dumps(output, indent=2, default=str))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(output, indent=2, default=str))
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
