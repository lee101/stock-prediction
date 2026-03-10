#!/usr/bin/env python3
"""Evaluate a parity-first 5m current-price execution policy on Binance checkpoints."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]

from src.current_price_locked_execution import CurrentPriceLockedConfig, simulate_current_price_locked
from src.forecast_horizon_utils import resolve_required_forecast_horizons
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.execution import resolve_symbol_rules
from binanceleveragesui.backtest_trade_margin_meta import compute_equity_stats, summarize_trades
from binanceleveragesui.trade_margin_meta import MODELS, apply_model_specs, build_model_specs_from_args
from binanceleveragesui.validate_sim_vs_live import generate_hourly_signals, load_5m_bars, set_seeds

DEFAULT_DEPLOY_MANIFEST = REPO / "deployments/binance-meta-margin/20260308_h1h6_e3_omega_s004_gate24_livelike_full_v1/deploy.json"


def _parse_days(raw: str) -> list[int]:
    return [int(chunk) for chunk in str(raw).split(",") if str(chunk).strip()]


def _latest_5m_timestamp(symbol: str) -> pd.Timestamp:
    path = REPO / "trainingdata5min" / f"{symbol}.csv"
    frame = pd.read_csv(path, usecols=["timestamp"])
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError(f"No valid 5m timestamps found for {symbol}")
    return ts.max()


def _resolve_output_root(value: str | None) -> Path:
    if value:
        return Path(value)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO / "experiments" / f"current_price_locked_5m_{stamp}"


def _load_manifest(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _build_config_grid(runtime: dict) -> list[CurrentPriceLockedConfig]:
    long_lev = float(runtime.get("max_long_leverage", runtime.get("max_leverage", 1.0)))
    short_lev = float(runtime.get("max_short_leverage", long_lev))
    allow_short = bool(runtime.get("allow_short", False))
    base = {
        "fee": 0.001,
        "spread_bps": 4.0,
        "slippage_bps": 2.0,
        "lock_minutes": 60,
        "cooldown_minutes_after_exit": 60,
        "max_hold_hours": float(runtime.get("max_hold_hours", 6.0)),
        "allow_short": allow_short,
        "long_max_leverage": long_lev,
        "short_max_leverage": short_lev if allow_short else 0.0,
        "min_notional": 5.0,
        "step_size": 0.0,
        "max_position_notional": runtime.get("max_position_notional"),
    }
    return [
        CurrentPriceLockedConfig(name="edge12_lock4", min_expected_edge_bps=12.0, min_profit_exit_bps=4.0, **base),
        CurrentPriceLockedConfig(name="edge12_lock8", min_expected_edge_bps=12.0, min_profit_exit_bps=8.0, **base),
        CurrentPriceLockedConfig(name="edge20_lock4", min_expected_edge_bps=20.0, min_profit_exit_bps=4.0, **base),
        CurrentPriceLockedConfig(name="edge20_lock8", min_expected_edge_bps=20.0, min_profit_exit_bps=8.0, **base),
    ]


def _load_model_surface(
    *,
    symbol: str,
    data_symbol: str,
    checkpoint_path: Path,
    data_root: Path,
    forecast_cache: Path,
    horizon: int,
    sequence_length: int,
    signal_start: pd.Timestamp,
    end_ts: pd.Timestamp,
):
    model, normalizer, feature_columns, meta = load_policy_checkpoint(
        str(checkpoint_path),
        device="cuda",
        data_root=data_root,
        forecast_cache_root=forecast_cache,
    )
    seq_len = int(meta.get("sequence_length", sequence_length))
    forecast_horizons = resolve_required_forecast_horizons(
        (int(horizon),),
        feature_columns=feature_columns,
        fallback_horizons=(int(horizon),),
    )
    dm = ChronosSolDataModule(
        symbol=data_symbol,
        data_root=data_root,
        forecast_cache_root=forecast_cache,
        forecast_horizons=forecast_horizons,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-2",
        sequence_length=seq_len,
        split_config=SplitConfig(val_days=1, test_days=1),
        cache_only=True,
        max_history_days=365,
    )
    frame = dm.full_frame.copy()
    frame = frame[frame["timestamp"] <= end_ts].reset_index(drop=True)
    signal_args = SimpleNamespace(
        start=signal_start.isoformat(),
        horizon=int(horizon),
        intensity_scale=5.0,
        sequence_length=int(sequence_length),
        symbol=symbol,
    )
    signals = generate_hourly_signals(signal_args, frame, model, normalizer, feature_columns, meta)
    return signals, load_5m_bars(symbol, signal_start - pd.Timedelta(hours=1), end_ts)


def _seed_state(mode: str, *, initial_equity: float, seed_notional: float, first_close: float, step_size: float):
    normalized = str(mode or "cash").strip().lower()
    qty = 0.0
    cash = float(initial_equity)
    entry_price = None
    if normalized in {"long", "short"}:
        raw_qty = max(0.0, float(seed_notional)) / max(1e-12, float(first_close))
        if step_size > 0.0:
            raw_qty = int(raw_qty / step_size) * step_size
        qty = raw_qty if normalized == "long" else -raw_qty
        cash = float(initial_equity) - qty * float(first_close)
        entry_price = float(first_close)
    return cash, qty, entry_price


def _scenario_rows_for_symbol(
    *,
    symbol_name: str,
    symbol: str,
    data_symbol: str,
    checkpoint_path: Path,
    signals: dict[pd.Timestamp, dict],
    bars_5m: pd.DataFrame,
    rules,
    windows: list[int],
    start_modes: list[str],
    configs: list[CurrentPriceLockedConfig],
    initial_equity: float,
    seed_notional: float,
):
    rows: list[dict] = []
    full_end_ts = bars_5m["timestamp"].max()
    for window_days in windows:
        start_ts = full_end_ts - pd.Timedelta(days=int(window_days))
        window_bars = bars_5m[(bars_5m["timestamp"] >= start_ts) & (bars_5m["timestamp"] <= full_end_ts)].reset_index(drop=True)
        if window_bars.empty:
            continue
        first_close = float(window_bars.iloc[0]["close"])
        for start_mode in start_modes:
            starting_cash, starting_qty, starting_entry_price = _seed_state(
                start_mode,
                initial_equity=initial_equity,
                seed_notional=seed_notional,
                first_close=first_close,
                step_size=float(getattr(rules, "step_size", 0.0) or 0.0),
            )
            for config in configs:
                config_with_rules = CurrentPriceLockedConfig(
                    **{
                        **asdict(config),
                        "min_notional": float(getattr(rules, "min_notional", config.min_notional) or config.min_notional),
                        "step_size": float(getattr(rules, "step_size", config.step_size) or config.step_size),
                    }
                )
                trades, final_eq, cash, qty, trace, counters = simulate_current_price_locked(
                    config_with_rules,
                    signals,
                    window_bars,
                    start_ts=start_ts,
                    initial_cash=starting_cash,
                    initial_qty=starting_qty,
                    initial_entry_price=starting_entry_price,
                    initial_entry_ts=start_ts if abs(starting_qty) > 1e-12 else None,
                )
                stats = compute_equity_stats(trace, initial_equity)
                trade_summary = summarize_trades(trades, initial_inventory=starting_qty)
                rows.append(
                    {
                        "config_name": config.name,
                        "symbol_name": symbol_name,
                        "symbol": symbol,
                        "data_symbol": data_symbol,
                        "checkpoint": str(checkpoint_path.resolve()),
                        "window_days": int(window_days),
                        "start_mode": start_mode,
                        "start": start_ts.isoformat(),
                        "end": full_end_ts.isoformat(),
                        "return_pct": float(stats["return_pct"]),
                        "max_drawdown_pct": float(stats["max_drawdown_pct"]),
                        "final_equity": float(final_eq),
                        "cash": float(cash),
                        "inventory": float(qty),
                        "bars": int(stats["bars"]),
                        "sharpe_ratio": stats["sharpe_ratio"],
                        "sortino_ratio": stats["sortino_ratio"],
                        "blocked_loss_exit_count": int(counters["blocked_loss_exit_count"]),
                        "blocked_reentry_count": int(counters["blocked_reentry_count"]),
                        **trade_summary,
                    }
                )
    return rows


def _aggregate_config_rows(configs: list[CurrentPriceLockedConfig], scenario_rows: list[dict]) -> list[dict]:
    ranked: list[dict] = []
    for config in configs:
        rows = [row for row in scenario_rows if row["config_name"] == config.name]
        if not rows:
            continue
        returns = np.array([row["return_pct"] for row in rows], dtype=np.float64)
        sortinos = np.array(
            [row["sortino_ratio"] if row["sortino_ratio"] is not None else -1e9 for row in rows],
            dtype=np.float64,
        )
        drawdowns = np.array([row["max_drawdown_pct"] for row in rows], dtype=np.float64)
        ranked.append(
            {
                "config": asdict(config),
                "scenario_count": int(len(rows)),
                "positive_scenarios": int(np.sum(returns > 0.0)),
                "min_return_pct": float(np.min(returns)),
                "mean_return_pct": float(np.mean(returns)),
                "median_return_pct": float(np.median(returns)),
                "min_sortino": float(np.min(sortinos)),
                "mean_sortino": float(np.mean(sortinos)),
                "mean_max_drawdown_pct": float(np.mean(drawdowns)),
                "blocked_loss_exit_total": int(sum(row["blocked_loss_exit_count"] for row in rows)),
                "blocked_reentry_total": int(sum(row["blocked_reentry_count"] for row in rows)),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["min_return_pct"],
            row["mean_return_pct"],
            row["mean_sortino"],
            -row["mean_max_drawdown_pct"],
        ),
        reverse=True,
    )
    return ranked


def _write_summary_md(
    path: Path,
    *,
    output_root: Path,
    deploy_manifest: Path,
    windows: list[int],
    initial_equity: float,
    seed_notional: float,
    config_rows: list[dict],
    scenario_rows: list[dict],
) -> None:
    best = config_rows[0] if config_rows else None
    lines = [
        "# 5m Current-Price Locked Execution Summary",
        "",
        "Parity-first evaluation settings:",
        "",
        f"- deploy manifest: `{deploy_manifest}`",
        f"- windows: `{','.join(str(day) + 'd' for day in windows)}`",
        f"- starting equity: `${initial_equity:,.2f}`",
        f"- seeded starting notional: `${seed_notional:,.2f}` long/short probes",
        f"- artifacts root: `{output_root}`",
        "",
    ]
    if best is not None:
        cfg = best["config"]
        lines.extend(
            [
                "## Best config",
                "",
                f"- name: `{cfg['name']}`",
                f"- edge guard: `{cfg['min_expected_edge_bps']:.1f} bps`",
                f"- profit lock: `{cfg['min_profit_exit_bps']:.1f} bps`",
                f"- spread/slippage: `{cfg['spread_bps']:.1f}+{cfg['slippage_bps']:.1f} bps`",
                f"- robustness: min return `{best['min_return_pct']:+.4f}%`, mean return `{best['mean_return_pct']:+.4f}%`, mean sortino `{best['mean_sortino']:.4f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Ranked configs",
            "",
            "| Rank | Name | Min Ret % | Mean Ret % | Mean Sortino | Mean DD % | Blocked Loss Exits | Blocked Reentries |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(config_rows, start=1):
        cfg = row["config"]
        lines.append(
            f"| {idx} | {cfg['name']} | {row['min_return_pct']:+.4f} | {row['mean_return_pct']:+.4f} | "
            f"{row['mean_sortino']:.4f} | {row['mean_max_drawdown_pct']:.4f} | "
            f"{row['blocked_loss_exit_total']} | {row['blocked_reentry_total']} |"
        )

    if best is not None:
        lines.extend(
            [
                "",
                "## Best config scenarios",
                "",
                "| Symbol | Window | Start | Ret % | Max DD % | Trades | Blocked Loss Exits | Blocked Reentries |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        best_rows = [row for row in scenario_rows if row["config_name"] == best["config"]["name"]]
        for row in sorted(best_rows, key=lambda item: (item["symbol"], item["window_days"], item["start_mode"])):
            lines.append(
                f"| {row['symbol']} | {row['window_days']}d | {row['start_mode']} | "
                f"{row['return_pct']:+.4f} | {row['max_drawdown_pct']:.4f} | {row['trade_count']} | "
                f"{row['blocked_loss_exit_count']} | {row['blocked_reentry_count']} |"
            )

    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `summary.json`",
            "- `summary.md`",
            "- `scenario_rows.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a 5m current-price locked execution policy")
    parser.add_argument("--deploy-manifest", default=str(DEFAULT_DEPLOY_MANIFEST))
    parser.add_argument("--model-a-name", default="doge")
    parser.add_argument("--model-a-symbol", default="DOGEUSDT")
    parser.add_argument("--model-a-data-symbol", default="DOGEUSD")
    parser.add_argument("--model-a-base-asset", default="DOGE")
    parser.add_argument("--model-a-maker-fee", type=float, default=0.001)
    parser.add_argument("--model-a-checkpoint", default=None)
    parser.add_argument("--model-b-name", default="aave")
    parser.add_argument("--model-b-symbol", default="AAVEUSDT")
    parser.add_argument("--model-b-data-symbol", default="AAVEUSD")
    parser.add_argument("--model-b-base-asset", default="AAVE")
    parser.add_argument("--model-b-maker-fee", type=float, default=0.001)
    parser.add_argument("--model-b-checkpoint", default=None)
    parser.add_argument("--windows", default="1,7,30")
    parser.add_argument("--start-modes", default="cash,long,short")
    parser.add_argument("--initial-equity", type=float, default=2_500.0)
    parser.add_argument("--seed-notional", type=float, default=5.0)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.deploy_manifest)
    deploy = _load_manifest(manifest_path)
    runtime = deploy["winner"]["runtime"]
    if args.model_a_checkpoint is None:
        args.model_a_checkpoint = runtime["doge_checkpoint"]
    if args.model_b_checkpoint is None:
        args.model_b_checkpoint = runtime["aave_checkpoint"]

    output_root = _resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    windows = _parse_days(args.windows)
    start_modes = [chunk.strip().lower() for chunk in str(args.start_modes).split(",") if chunk.strip()]
    model_specs = build_model_specs_from_args(args)
    apply_model_specs(model_specs)

    torch.use_deterministic_algorithms(True)
    set_seeds(42)

    data_root = Path(runtime["data_root"])
    forecast_cache = Path(runtime["forecast_cache"])
    latest_end_ts = min(_latest_5m_timestamp(spec["symbol"]) for spec in model_specs)
    max_window_days = max(windows)
    signal_start = latest_end_ts - pd.Timedelta(days=max_window_days) - pd.Timedelta(hours=6)

    config_grid = _build_config_grid(runtime)
    scenario_rows: list[dict] = []
    for spec in model_specs:
        rules = resolve_symbol_rules(spec["symbol"])
        signals, bars_5m = _load_model_surface(
            symbol=spec["symbol"],
            data_symbol=spec["data_symbol"],
            checkpoint_path=Path(spec["checkpoint"]),
            data_root=data_root,
            forecast_cache=forecast_cache,
            horizon=int(args.horizon),
            sequence_length=int(args.sequence_length),
            signal_start=signal_start,
            end_ts=latest_end_ts,
        )
        symbol_rows = _scenario_rows_for_symbol(
            symbol_name=spec["name"],
            symbol=spec["symbol"],
            data_symbol=spec["data_symbol"],
            checkpoint_path=Path(spec["checkpoint"]),
            signals=signals,
            bars_5m=bars_5m,
            rules=rules,
            windows=windows,
            start_modes=start_modes,
            configs=config_grid,
            initial_equity=float(args.initial_equity),
            seed_notional=float(args.seed_notional),
        )
        scenario_rows.extend(symbol_rows)

    config_rows = _aggregate_config_rows(config_grid, scenario_rows)
    summary = {
        "manifest": {
            "deploy_manifest": str(manifest_path.resolve()),
            "latest_end_ts": latest_end_ts.isoformat(),
            "windows": windows,
            "start_modes": start_modes,
            "initial_equity": float(args.initial_equity),
            "seed_notional": float(args.seed_notional),
            "output_root": str(output_root.resolve()),
        },
        "ranked_configs": config_rows,
        "scenario_rows": scenario_rows,
        "selected": config_rows[0] if config_rows else None,
    }

    (output_root / "scenario_rows.json").write_text(json.dumps(scenario_rows, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(
        output_root / "summary.md",
        output_root=output_root,
        deploy_manifest=manifest_path.resolve(),
        windows=windows,
        initial_equity=float(args.initial_equity),
        seed_notional=float(args.seed_notional),
        config_rows=config_rows,
        scenario_rows=scenario_rows,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
