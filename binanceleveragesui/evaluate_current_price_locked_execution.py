#!/usr/bin/env python3
"""Evaluate a parity-first 5m current-price locked execution policy."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.binan.binance_margin import get_margin_trades
from src.current_price_locked_execution import CurrentPriceLockedConfig, simulate_current_price_locked
from src.forecast_horizon_utils import resolve_required_forecast_horizons
from src.margin_position_utils import position_side_from_qty
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.execution import resolve_symbol_rules
from binanceleveragesui.backtest_trade_margin_meta import compute_equity_stats, summarize_trades
from binanceleveragesui.validate_sim_vs_live import (
    generate_hourly_signals,
    load_5m_bars,
    match_trades,
    pull_prod_fills,
    set_seeds,
    simulate_5m_with_trace,
)

DEPLOY_ROOT = REPO / "deployments/binance-meta-margin/20260308_h1h6_e3_omega_s004_gate24_livelike_full_v1"
LOG_ROOT = DEPLOY_ROOT / "strategy_state" / "margin_logs"
DEFAULT_OUTPUT_ROOT = REPO / "experiments" / f"current_price_locked_execution_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
DEFAULT_INITIAL_EQUITY = 2485.0
DEFAULT_PROBE_NOTIONAL = 5.10
DEFAULT_MODELS = {
    "doge": {
        "symbol": "DOGEUSDT",
        "data_symbol": "DOGEUSD",
        "checkpoint": DEPLOY_ROOT / "checkpoints" / "doge_epoch_003.pt",
        "data_root": REPO / "trainingdatahourlybinance",
        "forecast_cache": DEPLOY_ROOT / "forecast_cache",
    },
    "aave": {
        "symbol": "AAVEUSDT",
        "data_symbol": "AAVEUSD",
        "checkpoint": DEPLOY_ROOT / "checkpoints" / "aave_epoch_003.pt",
        "data_root": REPO / "trainingdatahourlybinance",
        "forecast_cache": DEPLOY_ROOT / "forecast_cache",
    },
}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    symbol_name: str
    start: pd.Timestamp
    end: pd.Timestamp
    initial_equity: float
    initial_qty: float = 0.0
    initial_entry_price: float = 0.0
    initial_entry_ts: pd.Timestamp | None = None


def _to_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _serialize_records(records: list[dict]) -> list[dict]:
    return [{key: _json_value(value) for key, value in row.items()} for row in records]


def _serialize_frame(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    records = frame.to_dict(orient="records")
    return _serialize_records(records)


def _latest_5m_timestamp(symbol: str) -> pd.Timestamp:
    path = REPO / "trainingdata5min" / f"{symbol}.csv"
    frame = pd.read_csv(path, usecols=["timestamp"])
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError(f"No valid 5m timestamps found for {symbol}")
    return ts.max()


def _starting_mark_price(bars_5m: pd.DataFrame, start_ts: pd.Timestamp) -> float:
    window = bars_5m[bars_5m["timestamp"] >= start_ts]
    if window.empty:
        window = bars_5m
    if window.empty:
        raise ValueError("Cannot resolve starting mark price from empty 5m bars.")
    return max(0.0, float(window.iloc[0]["close"]))


def _seed_qty_from_notional(price: float, step_size: float, notional: float) -> float:
    if price <= 0.0:
        return 0.0
    raw_qty = float(notional) / float(price)
    if step_size <= 0.0:
        return raw_qty
    return int(raw_qty / step_size) * float(step_size)


def _build_baseline_args(
    *,
    initial_cash: float,
    max_position_notional: float,
    long_max_leverage: float,
    short_max_leverage: float,
    rules,
) -> SimpleNamespace:
    return SimpleNamespace(
        fee=0.001,
        fill_buffer_pct=0.0005,
        initial_cash=float(initial_cash),
        realistic=True,
        expiry_minutes=90,
        max_fill_fraction=0.01,
        min_notional=float(getattr(rules, "min_notional", 5.0) or 5.0),
        tick_size=float(getattr(rules, "tick_size", 0.00001) or 0.00001),
        step_size=float(getattr(rules, "step_size", 1.0) or 1.0),
        max_hold_hours=6.0,
        max_leverage=2.3,
        long_max_leverage=float(long_max_leverage),
        short_max_leverage=float(short_max_leverage),
        margin_hourly_rate=0.0,
        verbose=False,
        live_like=True,
        use_order_expiry=False,
        reprice_threshold=0.003,
        max_position_notional=float(max_position_notional),
        allow_short=True,
    )


def _load_model_inputs(
    *,
    symbol_name: str,
    symbol: str,
    data_symbol: str,
    checkpoint: Path,
    data_root: Path,
    forecast_cache: Path,
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, normalizer, feature_columns, meta = load_policy_checkpoint(
        str(checkpoint),
        device=device,
        data_root=data_root,
        forecast_cache_root=forecast_cache,
    )
    seq_len = int(meta.get("sequence_length", 72))
    forecast_horizons = resolve_required_forecast_horizons(
        (1,),
        feature_columns=feature_columns,
        fallback_horizons=(1,),
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
    frame = frame[frame["timestamp"] <= signal_end].reset_index(drop=True)
    signal_args = SimpleNamespace(
        start=signal_start.isoformat(),
        horizon=1,
        intensity_scale=5.0,
        sequence_length=seq_len,
        symbol=symbol,
    )
    signals = generate_hourly_signals(signal_args, frame, model, normalizer, feature_columns, meta)
    bars = load_5m_bars(symbol, signal_start - pd.Timedelta(hours=1), signal_end)
    rules = resolve_symbol_rules(symbol)
    return {
        "name": symbol_name,
        "symbol": symbol,
        "signals": signals,
        "bars": bars,
        "rules": rules,
        "forecast_horizons": [int(h) for h in forecast_horizons],
        "checkpoint": str(checkpoint.resolve()),
    }


def _reconstruct_initial_position_with_price(
    symbol: str,
    start_ms: int,
    *,
    lookback_hours: int = 72,
) -> tuple[float, float, pd.Timestamp | None]:
    raw: list[dict] = []
    cursor = start_ms - lookback_hours * 3600 * 1000
    while cursor < start_ms:
        chunk_end = min(cursor + 24 * 3600 * 1000, start_ms)
        raw.extend(
            get_margin_trades(
                symbol,
                start_time=cursor,
                end_time=chunk_end,
                limit=1000,
            )
        )
        cursor = chunk_end
    inventory = 0.0
    avg_entry_price = 0.0
    entry_ts: pd.Timestamp | None = None
    for trade in sorted(raw, key=lambda row: int(row["time"])):
        qty = float(trade["qty"])
        price = float(trade["price"])
        ts = pd.Timestamp(int(trade["time"]), unit="ms", tz="UTC")
        is_buy = bool(trade.get("isBuyer"))
        if is_buy:
            if inventory >= 0.0:
                new_inventory = inventory + qty
                avg_entry_price = (
                    (inventory * avg_entry_price + qty * price) / new_inventory
                    if new_inventory > 0.0
                    else 0.0
                )
                inventory = new_inventory
                entry_ts = ts
            else:
                cover_qty = min(abs(inventory), qty)
                inventory += cover_qty
                if abs(inventory) <= 1e-12:
                    avg_entry_price = 0.0
                    entry_ts = None
                remaining = qty - cover_qty
                if remaining > 1e-12:
                    inventory = remaining
                    avg_entry_price = price
                    entry_ts = ts
        else:
            if inventory <= 0.0:
                new_abs_inventory = abs(inventory) + qty
                avg_entry_price = (
                    (abs(inventory) * avg_entry_price + qty * price) / new_abs_inventory
                    if new_abs_inventory > 0.0
                    else 0.0
                )
                inventory -= qty
                entry_ts = ts
            else:
                sell_qty = min(inventory, qty)
                inventory -= sell_qty
                if inventory <= 1e-12:
                    avg_entry_price = 0.0
                    entry_ts = None
                remaining = qty - sell_qty
                if remaining > 1e-12:
                    inventory = -remaining
                    avg_entry_price = price
                    entry_ts = ts
    return float(inventory), float(avg_entry_price), entry_ts


def _build_scenarios(model_inputs: dict[str, dict], *, initial_equity: float, probe_notional: float) -> tuple[list[Scenario], list[Scenario]]:
    latest_end = min(_latest_5m_timestamp(model["symbol"]) for model in model_inputs.values()).floor("5min")
    robust: list[Scenario] = []
    parity: list[Scenario] = []
    for name, bundle in model_inputs.items():
        step_size = float(getattr(bundle["rules"], "step_size", 1.0) or 1.0)
        for days in (1, 7, 30):
            end = latest_end
            start = end - pd.Timedelta(days=days)
            robust.append(
                Scenario(
                    name=f"{name}_flat_{days}d",
                    kind="robust",
                    symbol_name=name,
                    start=start,
                    end=end,
                    initial_equity=float(initial_equity),
                )
            )
        seed_start = latest_end - pd.Timedelta(days=7)
        seed_price = _starting_mark_price(bundle["bars"], seed_start)
        seed_qty = _seed_qty_from_notional(seed_price, step_size, probe_notional)
        robust.extend(
            [
                Scenario(
                    name=f"{name}_long_seed_7d",
                    kind="robust",
                    symbol_name=name,
                    start=seed_start,
                    end=latest_end,
                    initial_equity=float(initial_equity),
                    initial_qty=float(seed_qty),
                    initial_entry_price=float(seed_price),
                    initial_entry_ts=seed_start,
                ),
                Scenario(
                    name=f"{name}_short_seed_7d",
                    kind="robust",
                    symbol_name=name,
                    start=seed_start,
                    end=latest_end,
                    initial_equity=float(initial_equity),
                    initial_qty=-float(seed_qty),
                    initial_entry_price=float(seed_price),
                    initial_entry_ts=seed_start,
                ),
            ]
        )

    parity_windows = {
        "doge": ("2026-03-08T06:30:00+00:00", "2026-03-08T09:30:00+00:00"),
        "aave": ("2026-03-08T11:00:00+00:00", "2026-03-08T18:00:00+00:00"),
    }
    for name, (start_raw, end_raw) in parity_windows.items():
        start = _to_utc_timestamp(start_raw)
        end = _to_utc_timestamp(end_raw)
        symbol = model_inputs[name]["symbol"]
        initial_qty, initial_entry_price, initial_entry_ts = _reconstruct_initial_position_with_price(
            symbol,
            int(start.timestamp() * 1000),
            lookback_hours=96,
        )
        parity.append(
            Scenario(
                name=f"{name}_prod_window",
                kind="parity",
                symbol_name=name,
                start=start,
                end=end,
                initial_equity=float(initial_equity),
                initial_qty=float(initial_qty),
                initial_entry_price=float(initial_entry_price),
                initial_entry_ts=initial_entry_ts,
            )
        )
    return robust, parity


def _scenario_start_cash(bundle: dict, scenario: Scenario) -> float:
    start_price = _starting_mark_price(bundle["bars"], scenario.start)
    return float(scenario.initial_equity) - float(scenario.initial_qty) * float(start_price)


def _subset_bars(bundle: dict, scenario: Scenario) -> pd.DataFrame:
    mask = (bundle["bars"]["timestamp"] >= scenario.start) & (bundle["bars"]["timestamp"] <= scenario.end)
    return bundle["bars"].loc[mask].reset_index(drop=True)


def _normalize_trade_row(trade: dict) -> dict:
    return {
        "ts": pd.Timestamp(trade["ts"]),
        "side": str(trade["side"]),
        "price": float(trade["price"]),
        "qty": float(trade["qty"]),
        **({"reason": str(trade["reason"])} if "reason" in trade else {}),
    }


def _run_baseline(bundle: dict, scenario: Scenario, *, max_position_notional: float, short_max_leverage: float) -> dict:
    bars = _subset_bars(bundle, scenario)
    args = _build_baseline_args(
        initial_cash=_scenario_start_cash(bundle, scenario),
        max_position_notional=max_position_notional,
        long_max_leverage=2.3,
        short_max_leverage=short_max_leverage,
        rules=bundle["rules"],
    )
    args.start = scenario.start.isoformat()
    trades, final_eq, cash, inv, trace = simulate_5m_with_trace(
        args,
        bundle["signals"],
        bars,
        initial_inv=float(scenario.initial_qty),
        initial_entry_ts=scenario.initial_entry_ts,
        stop_after_cycle=False,
    )
    normalized_trades = [_normalize_trade_row(trade) for trade in trades]
    summary = compute_equity_stats(trace, float(scenario.initial_equity))
    summary.update(summarize_trades(normalized_trades, initial_inventory=float(scenario.initial_qty)))
    summary.update(
        {
            "final_equity": float(final_eq),
            "final_cash": float(cash),
            "final_inventory": float(inv),
            "position_side": position_side_from_qty(float(inv), step_size=float(getattr(bundle["rules"], "step_size", 0.0) or 0.0)),
            "trades": normalized_trades,
        }
    )
    return summary


def _run_locked(
    bundle: dict,
    scenario: Scenario,
    config: CurrentPriceLockedConfig,
    *,
    signal_schedule: pd.DataFrame | None = None,
) -> dict:
    bars = _subset_bars(bundle, scenario)
    config_with_rules = CurrentPriceLockedConfig(
        **{
            **asdict(config),
            "min_notional": float(getattr(bundle["rules"], "min_notional", config.min_notional) or config.min_notional),
            "step_size": float(getattr(bundle["rules"], "step_size", config.step_size) or config.step_size),
        }
    )
    trades, final_eq, cash, qty, trace, counters = simulate_current_price_locked(
        config_with_rules,
        bundle["signals"],
        bars,
        start_ts=scenario.start,
        initial_cash=_scenario_start_cash(bundle, scenario),
        initial_qty=float(scenario.initial_qty),
        initial_entry_price=float(scenario.initial_entry_price),
        initial_entry_ts=scenario.initial_entry_ts,
        signal_schedule=signal_schedule,
    )
    normalized_trades = [_normalize_trade_row(trade) for trade in trades]
    summary = compute_equity_stats(trace, float(scenario.initial_equity))
    summary.update(summarize_trades(normalized_trades, initial_inventory=float(scenario.initial_qty)))
    summary.update(counters)
    summary.update(
        {
            "final_equity": float(final_eq),
            "final_cash": float(cash),
            "final_inventory": float(qty),
            "position_side": position_side_from_qty(float(qty), step_size=float(config_with_rules.step_size)),
            "trades": normalized_trades,
        }
    )
    return summary


def _summarize_match_rate(prod_fills: pd.DataFrame, matches: list[dict], unmatched_sim_count: int) -> dict:
    matched_rows = [row for row in matches if row.get("matched")]
    total_prod = int(len(prod_fills))
    price_diffs = [abs(float(row["diff_bps"])) for row in matched_rows if "diff_bps" in row]
    time_diffs = [
        abs((pd.Timestamp(row["sim_ts"]) - pd.Timestamp(row["prod_ts"])).total_seconds()) / 60.0
        for row in matched_rows
        if row.get("sim_ts") is not None
    ]
    return {
        "prod_fill_count": total_prod,
        "matched_fill_count": int(len(matched_rows)),
        "unmatched_prod_count": int(total_prod - len(matched_rows)),
        "unmatched_sim_count": int(unmatched_sim_count),
        "match_rate_pct": float((len(matched_rows) / total_prod) * 100.0) if total_prod else 100.0,
        "mean_abs_price_diff_bps": float(sum(price_diffs) / len(price_diffs)) if price_diffs else 0.0,
        "mean_abs_time_diff_minutes": float(sum(time_diffs) / len(time_diffs)) if time_diffs else 0.0,
    }


def _run_parity_match(prod_fills: pd.DataFrame, sim_trades: list[dict]) -> dict:
    matches, unmatched = match_trades(prod_fills, sim_trades)
    parity = _summarize_match_rate(prod_fills, matches, unmatched_sim_count=int(len(unmatched)))
    parity["matches"] = _serialize_records(matches)
    parity["unmatched_sim_trades"] = _serialize_records(unmatched.to_dict(orient="records")) if not unmatched.empty else []
    return parity


def _candidate_grid(max_position_notional: float) -> list[dict]:
    candidates: list[dict] = []
    for spread_bps in (2.0, 4.0):
        for slippage_bps in (1.0, 2.0):
            for min_expected_edge_bps in (12.0, 20.0):
                for min_profit_exit_bps in (0.0, 4.0):
                    config = CurrentPriceLockedConfig(
                        name=(
                            f"locked_sp{int(spread_bps):02d}_sl{int(slippage_bps):02d}_"
                            f"edge{int(min_expected_edge_bps):02d}_lock{int(min_profit_exit_bps):02d}"
                        ),
                        fee=0.001,
                        spread_bps=spread_bps,
                        slippage_bps=slippage_bps,
                        min_expected_edge_bps=min_expected_edge_bps,
                        min_profit_exit_bps=min_profit_exit_bps,
                        lock_minutes=60,
                        cooldown_minutes_after_exit=60,
                        max_hold_hours=6.0,
                        allow_short=True,
                        long_max_leverage=2.3,
                        short_max_leverage=0.04,
                        min_notional=5.0,
                        step_size=0.0,
                        max_position_notional=max_position_notional,
                    )
                    candidates.append({"name": config.name, "config": config})
    return candidates


def _load_logged_signal_schedules(*, start: pd.Timestamp, end: pd.Timestamp, model_names: list[str]) -> dict[str, pd.DataFrame]:
    rows_by_model: dict[str, list[dict]] = {name: [] for name in model_names}
    file_days = pd.date_range((start - pd.Timedelta(days=1)).floor("D"), end.floor("D"), freq="D", tz="UTC")
    for day in file_days:
        path = LOG_ROOT / f"margin-meta_{day.strftime('%Y%m%d')}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("event") not in {"signal_refresh", "signal_bootstrap"}:
                    continue
                model = str(row.get("model", "")).strip().lower()
                if model not in rows_by_model:
                    continue
                ts = _to_utc_timestamp(row["ts"])
                if ts > end + pd.Timedelta(hours=1):
                    continue
                rows_by_model[model].append(
                    {
                        "effective_ts": ts,
                        "signal_hour": row.get("signal_hour") or row.get("last_signal_hour"),
                        "buy_price": float(row.get("buy_price", 0.0)),
                        "sell_price": float(row.get("sell_price", 0.0)),
                        "buy_amount": float(row.get("buy_amount", 0.0)),
                        "sell_amount": float(row.get("sell_amount", 0.0)),
                        "source_event": str(row.get("event", "")),
                    }
                )
    return {
        name: pd.DataFrame(rows).sort_values("effective_ts").reset_index(drop=True) if rows else pd.DataFrame()
        for name, rows in rows_by_model.items()
    }


def _aggregate_logged_parity(results: dict[str, dict]) -> dict:
    rows = list(results.values())
    matches = [float(row["parity"]["match_rate_pct"]) for row in rows]
    price_diffs = [float(row["parity"]["mean_abs_price_diff_bps"]) for row in rows]
    time_diffs = [float(row["parity"]["mean_abs_time_diff_minutes"]) for row in rows]
    return {
        "scenario_count": int(len(rows)),
        "mean_match_rate_pct": float(sum(matches) / len(matches)) if matches else 0.0,
        "mean_abs_price_diff_bps": float(sum(price_diffs) / len(price_diffs)) if price_diffs else 0.0,
        "mean_abs_time_diff_minutes": float(sum(time_diffs) / len(time_diffs)) if time_diffs else 0.0,
    }


def _aggregate_candidate(name: str, scenario_results: dict[str, dict]) -> dict:
    robust_rows = [row for row in scenario_results.values() if row["scenario_kind"] == "robust"]
    parity_rows = [row for row in scenario_results.values() if row["scenario_kind"] == "parity"]
    robust_returns = [float(row["summary"]["return_pct"]) for row in robust_rows]
    robust_dd = [float(row["summary"]["max_drawdown_pct"]) for row in robust_rows]
    robust_trades = [int(row["summary"]["trade_count"]) for row in robust_rows]
    parity_matches = [float(row["parity"]["match_rate_pct"]) for row in parity_rows]
    parity_price_diffs = [float(row["parity"]["mean_abs_price_diff_bps"]) for row in parity_rows]
    parity_time_diffs = [float(row["parity"]["mean_abs_time_diff_minutes"]) for row in parity_rows]
    blocked_loss_exits = sum(int(row["summary"].get("blocked_loss_exit_count", 0)) for row in robust_rows + parity_rows)
    blocked_reentries = sum(int(row["summary"].get("blocked_reentry_count", 0)) for row in robust_rows + parity_rows)
    return {
        "name": name,
        "robust_mean_return_pct": float(sum(robust_returns) / len(robust_returns)) if robust_returns else 0.0,
        "robust_min_return_pct": float(min(robust_returns)) if robust_returns else 0.0,
        "robust_mean_max_drawdown_pct": float(sum(robust_dd) / len(robust_dd)) if robust_dd else 0.0,
        "robust_mean_trade_count": float(sum(robust_trades) / len(robust_trades)) if robust_trades else 0.0,
        "parity_mean_match_rate_pct": float(sum(parity_matches) / len(parity_matches)) if parity_matches else 0.0,
        "parity_mean_abs_price_diff_bps": float(sum(parity_price_diffs) / len(parity_price_diffs)) if parity_price_diffs else 0.0,
        "parity_mean_abs_time_diff_minutes": float(sum(parity_time_diffs) / len(parity_time_diffs)) if parity_time_diffs else 0.0,
        "blocked_loss_exit_count": int(blocked_loss_exits),
        "blocked_reentry_count": int(blocked_reentries),
    }


def _winner_key(row: dict) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["parity_mean_match_rate_pct"]),
        -float(row["parity_mean_abs_price_diff_bps"]),
        float(row["robust_min_return_pct"]),
        float(row["robust_mean_return_pct"]),
        -float(row["blocked_loss_exit_count"]),
        -float(row["robust_mean_trade_count"]),
    )


def _render_summary_markdown(
    *,
    output_root: Path,
    baseline_aggregate: dict,
    ranked_rows: list[dict],
    winner_name: str,
    winner_config: dict,
    baseline_results: dict[str, dict],
    winner_results: dict[str, dict],
    winner_logged_signal_results: dict[str, dict],
    winner_logged_signal_aggregate: dict,
) -> str:
    lines = [
        "# Current-Price Locked 5m Execution Evaluation",
        "",
        f"Output root: `{output_root}`",
        "",
        "## Winner",
        f"- Candidate: `{winner_name}`",
        f"- Config: `{json.dumps(winner_config, sort_keys=True)}`",
        "",
        "## Aggregate Comparison",
        "",
        "| Strategy | Mean parity match % | Mean abs price diff (bps) | Mean abs time diff (min) | Robust mean return % | Robust min return % | Mean max DD % | Mean trade count | Blocked loss exits | Blocked reentries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| baseline_live_like | {baseline_aggregate['parity_mean_match_rate_pct']:.2f} | "
            f"{baseline_aggregate['parity_mean_abs_price_diff_bps']:.2f} | "
            f"{baseline_aggregate['parity_mean_abs_time_diff_minutes']:.2f} | "
            f"{baseline_aggregate['robust_mean_return_pct']:.4f} | "
            f"{baseline_aggregate['robust_min_return_pct']:.4f} | "
            f"{baseline_aggregate['robust_mean_max_drawdown_pct']:.4f} | "
            f"{baseline_aggregate['robust_mean_trade_count']:.2f} | "
            f"{baseline_aggregate['blocked_loss_exit_count']} | "
            f"{baseline_aggregate['blocked_reentry_count']} |"
        ),
    ]
    winner_row = next(row for row in ranked_rows if row["name"] == winner_name)
    lines.append(
        (
            f"| {winner_row['name']} | {winner_row['parity_mean_match_rate_pct']:.2f} | "
            f"{winner_row['parity_mean_abs_price_diff_bps']:.2f} | "
            f"{winner_row['parity_mean_abs_time_diff_minutes']:.2f} | "
            f"{winner_row['robust_mean_return_pct']:.4f} | "
            f"{winner_row['robust_min_return_pct']:.4f} | "
            f"{winner_row['robust_mean_max_drawdown_pct']:.4f} | "
            f"{winner_row['robust_mean_trade_count']:.2f} | "
            f"{winner_row['blocked_loss_exit_count']} | "
            f"{winner_row['blocked_reentry_count']} |"
        )
    )
    lines.extend(
        [
            "",
            "## Candidate Ranking",
            "",
            "| Candidate | Mean parity match % | Mean abs price diff (bps) | Robust min return % | Robust mean return % | Mean max DD % | Mean trade count | Blocked loss exits | Blocked reentries |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ranked_rows:
        lines.append(
            (
                f"| {row['name']} | {row['parity_mean_match_rate_pct']:.2f} | "
                f"{row['parity_mean_abs_price_diff_bps']:.2f} | "
                f"{row['robust_min_return_pct']:.4f} | {row['robust_mean_return_pct']:.4f} | "
                f"{row['robust_mean_max_drawdown_pct']:.4f} | {row['robust_mean_trade_count']:.2f} | "
                f"{row['blocked_loss_exit_count']} | {row['blocked_reentry_count']} |"
            )
        )
    lines.extend(
        [
            "",
            "## Scenario Deltas",
            "",
            "| Scenario | Baseline return % | Winner return % | Baseline trades | Winner trades | Baseline parity match % | Winner parity match % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario_name in sorted(baseline_results):
        baseline_row = baseline_results[scenario_name]
        winner_row = winner_results[scenario_name]
        lines.append(
            (
                f"| {scenario_name} | {baseline_row['summary']['return_pct']:.4f} | "
                f"{winner_row['summary']['return_pct']:.4f} | "
                f"{baseline_row['summary']['trade_count']} | "
                f"{winner_row['summary']['trade_count']} | "
                f"{baseline_row.get('parity', {}).get('match_rate_pct', 0.0):.2f} | "
                f"{winner_row.get('parity', {}).get('match_rate_pct', 0.0):.2f} |"
            )
        )
    lines.extend(
        [
            "",
            "## Winner With Logged Live Signals",
            "",
            "| Scenario | Match % | Mean abs price diff (bps) | Mean abs time diff (min) | Trades |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario_name in sorted(winner_logged_signal_results):
        row = winner_logged_signal_results[scenario_name]
        lines.append(
            (
                f"| {scenario_name} | {row['parity']['match_rate_pct']:.2f} | "
                f"{row['parity']['mean_abs_price_diff_bps']:.2f} | "
                f"{row['parity']['mean_abs_time_diff_minutes']:.2f} | "
                f"{row['summary']['trade_count']} |"
            )
        )
    lines.extend(
        [
            "",
            (
                "Logged-signal aggregate: "
                f"match `{winner_logged_signal_aggregate['mean_match_rate_pct']:.2f}%`, "
                f"price diff `{winner_logged_signal_aggregate['mean_abs_price_diff_bps']:.2f} bps`, "
                f"time diff `{winner_logged_signal_aggregate['mean_abs_time_diff_minutes']:.2f} min`."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a parity-first 5m current-price execution policy.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--initial-equity", type=float, default=DEFAULT_INITIAL_EQUITY)
    parser.add_argument("--probe-notional", type=float, default=DEFAULT_PROBE_NOTIONAL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    set_seeds(42)

    latest_end = min(_latest_5m_timestamp(cfg["symbol"]) for cfg in DEFAULT_MODELS.values()).floor("5min")
    earliest_start = min(
        _to_utc_timestamp("2026-03-08T06:30:00+00:00"),
        latest_end - pd.Timedelta(days=30),
    )
    signal_start = earliest_start - pd.Timedelta(hours=48)

    model_inputs = {
        name: _load_model_inputs(
            symbol_name=name,
            symbol=cfg["symbol"],
            data_symbol=cfg["data_symbol"],
            checkpoint=Path(cfg["checkpoint"]),
            data_root=Path(cfg["data_root"]),
            forecast_cache=Path(cfg["forecast_cache"]),
            signal_start=signal_start,
            signal_end=latest_end,
        )
        for name, cfg in DEFAULT_MODELS.items()
    }

    robust_scenarios, parity_scenarios = _build_scenarios(
        model_inputs,
        initial_equity=float(args.initial_equity),
        probe_notional=float(args.probe_notional),
    )
    scenarios = robust_scenarios + parity_scenarios

    parity_truth: dict[str, dict] = {}
    for scenario in parity_scenarios:
        bundle = model_inputs[scenario.symbol_name]
        start_ms = int(scenario.start.timestamp() * 1000)
        end_ms = int(scenario.end.timestamp() * 1000)
        prod_fills, raw_trades, orders = pull_prod_fills(bundle["symbol"], start_ms, end_ms)
        parity_truth[scenario.name] = {
            "prod_fills": prod_fills,
            "raw_trades": raw_trades,
            "orders": orders,
        }
    logged_signal_schedules = _load_logged_signal_schedules(
        start=min(scenario.start for scenario in parity_scenarios),
        end=max(scenario.end for scenario in parity_scenarios),
        model_names=list(DEFAULT_MODELS.keys()),
    )

    baseline_results: dict[str, dict] = {}
    for scenario in scenarios:
        bundle = model_inputs[scenario.symbol_name]
        summary = _run_baseline(
            bundle,
            scenario,
            max_position_notional=float(args.probe_notional),
            short_max_leverage=0.04,
        )
        result = {
            "scenario_name": scenario.name,
            "scenario_kind": scenario.kind,
            "symbol": bundle["symbol"],
            "summary": summary,
        }
        if scenario.kind == "parity":
            result["parity"] = _run_parity_match(parity_truth[scenario.name]["prod_fills"], summary["trades"])
        baseline_results[scenario.name] = result

    candidate_results: dict[str, dict[str, dict]] = {}
    ranked_rows: list[dict] = []
    candidates = _candidate_grid(float(args.probe_notional))
    for candidate in candidates:
        scenario_results: dict[str, dict] = {}
        for scenario in scenarios:
            bundle = model_inputs[scenario.symbol_name]
            summary = _run_locked(bundle, scenario, candidate["config"])
            result = {
                "scenario_name": scenario.name,
                "scenario_kind": scenario.kind,
                "symbol": bundle["symbol"],
                "summary": summary,
            }
            if scenario.kind == "parity":
                result["parity"] = _run_parity_match(parity_truth[scenario.name]["prod_fills"], summary["trades"])
            scenario_results[scenario.name] = result
        candidate_results[candidate["name"]] = scenario_results
        ranked_rows.append(_aggregate_candidate(candidate["name"], scenario_results))

    baseline_aggregate = _aggregate_candidate("baseline_live_like", baseline_results)
    ranked_rows = sorted(ranked_rows, key=_winner_key, reverse=True)
    winner_name = ranked_rows[0]["name"]
    winner_config = next(asdict(candidate["config"]) for candidate in candidates if candidate["name"] == winner_name)
    winner_config_obj = next(candidate["config"] for candidate in candidates if candidate["name"] == winner_name)

    winner_logged_signal_results: dict[str, dict] = {}
    for scenario in parity_scenarios:
        bundle = model_inputs[scenario.symbol_name]
        summary = _run_locked(
            bundle,
            scenario,
            winner_config_obj,
            signal_schedule=logged_signal_schedules.get(scenario.symbol_name),
        )
        winner_logged_signal_results[scenario.name] = {
            "scenario_name": scenario.name,
            "scenario_kind": scenario.kind,
            "symbol": bundle["symbol"],
            "summary": summary,
            "parity": _run_parity_match(parity_truth[scenario.name]["prod_fills"], summary["trades"]),
        }
    winner_logged_signal_aggregate = _aggregate_logged_parity(winner_logged_signal_results)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "initial_equity": float(args.initial_equity),
            "probe_notional": float(args.probe_notional),
            "deploy_root": str(DEPLOY_ROOT),
            "models": {
                name: {
                    "symbol": bundle["symbol"],
                    "checkpoint": bundle["checkpoint"],
                    "forecast_horizons": bundle["forecast_horizons"],
                }
                for name, bundle in model_inputs.items()
            },
        },
        "baseline_aggregate": baseline_aggregate,
        "candidate_ranking": ranked_rows,
        "winner": {"name": winner_name, "config": winner_config},
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "logged_signal_schedule": {name: _serialize_frame(frame) for name, frame in logged_signal_schedules.items()},
        "live_truth": {
            scenario.name: {
                "symbol": model_inputs[scenario.symbol_name]["symbol"],
                "prod_fills": _serialize_frame(parity_truth[scenario.name]["prod_fills"]),
                "raw_trades": _serialize_frame(parity_truth[scenario.name]["raw_trades"]),
                "orders": _serialize_frame(parity_truth[scenario.name]["orders"]),
            }
            for scenario in parity_scenarios
        },
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "winner_logged_signal_results": winner_logged_signal_results,
        "winner_logged_signal_aggregate": winner_logged_signal_aggregate,
    }
    (args.output_root / "live_truth.json").write_text(
        json.dumps(summary["live_truth"], indent=2, default=str),
        encoding="utf-8",
    )
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary_md = _render_summary_markdown(
        output_root=args.output_root,
        baseline_aggregate=baseline_aggregate,
        ranked_rows=ranked_rows,
        winner_name=winner_name,
        winner_config=winner_config,
        baseline_results=baseline_results,
        winner_results=candidate_results[winner_name],
        winner_logged_signal_results=winner_logged_signal_results,
        winner_logged_signal_aggregate=winner_logged_signal_aggregate,
    )
    (args.output_root / "summary.md").write_text(summary_md, encoding="utf-8")
    print(summary_md)


if __name__ == "__main__":
    main()
