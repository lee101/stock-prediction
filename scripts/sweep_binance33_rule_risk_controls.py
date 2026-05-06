#!/usr/bin/env python3
"""Sweep risk controls around high-upside Binance33 deterministic rules."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from scripts.sweep_binance33_rules import RuleConfig, _candidate_starts, _default_configs, _make_policy, _slice_window
from scripts.sweep_binance33_xgb import _monthly_equivalent_return, _passes_production_target
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


def _parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(f"expected at least one float in {value!r}")
    return out


def _parse_int_list(value: str) -> list[int]:
    out = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(f"expected at least one int in {value!r}")
    return out


def _monthly_stats(returns: list[float], eval_days: int) -> np.ndarray:
    return np.asarray([_monthly_equivalent_return(ret, int(eval_days)) for ret in returns], dtype=np.float64)


def _masked_data(data: MktdData, *, exclude_symbols: set[str]) -> MktdData:
    if not exclude_symbols:
        return data
    tradable = (
        np.asarray(data.tradable, dtype=np.uint8).copy()
        if data.tradable is not None
        else np.ones((data.num_timesteps, data.num_symbols), dtype=np.uint8)
    )
    for idx, symbol in enumerate(data.symbols):
        if str(symbol).upper() in exclude_symbols:
            tradable[:, idx] = 0
    return MktdData(
        version=data.version,
        symbols=list(data.symbols),
        features=data.features,
        prices=data.prices,
        tradable=tradable,
    )


def _eval_config(
    data: MktdData,
    config: RuleConfig,
    *,
    eval_days: int,
    stride: int,
    slippage_bps: float,
    fee_rate: float,
    fill_buffer_bps: float,
    decision_lag: int,
    max_leverage: float,
    max_hold_bars: int,
    circuit_breaker_dd_pct: float,
    periods_per_year: float,
    short_borrow_apr: float,
    target_monthly_pct: float,
    target_max_dd_pct: float,
) -> dict[str, float | int | str]:
    returns: list[float] = []
    sortinos: list[float] = []
    maxdds: list[float] = []
    smooths: list[float] = []
    ulcers: list[float] = []
    trades: list[int] = []
    early: list[int] = []
    for start in _candidate_starts(data, eval_days, stride=int(stride)):
        window = _slice_window(data, start, eval_days)
        raw_policy = _make_policy(window, config, decision_lag=int(decision_lag))
        circuit_dd = max(0.0, float(circuit_breaker_dd_pct)) / 100.0
        if circuit_dd > 0.0:
            obs_equity_base = window.num_symbols * int(window.features.shape[2])
            circuit_state = {"peak": 1.0, "halted": False}

            def policy(obs: np.ndarray) -> int:
                equity_scaled = float(obs[obs_equity_base + 0] + obs[obs_equity_base + 1])
                if np.isfinite(equity_scaled) and equity_scaled > circuit_state["peak"]:
                    circuit_state["peak"] = equity_scaled
                peak = max(1e-9, float(circuit_state["peak"]))
                dd = (peak - equity_scaled) / peak if np.isfinite(equity_scaled) else 0.0
                if circuit_state["halted"] or dd >= circuit_dd:
                    circuit_state["halted"] = True
                    return 0
                return int(raw_policy(obs))

        else:
            policy = raw_policy
        result = simulate_daily_policy(
            window,
            policy,
            max_steps=int(eval_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            fill_buffer_bps=float(fill_buffer_bps),
            max_leverage=float(max_leverage),
            max_hold_bars=int(max_hold_bars),
            periods_per_year=float(periods_per_year),
            short_borrow_apr=float(short_borrow_apr),
            enable_drawdown_profit_early_exit=False,
            enable_metric_threshold_early_exit=False,
        )
        curve = np.asarray(result.equity_curve if result.equity_curve is not None else [], dtype=np.float64)
        returns.append(float(result.total_return))
        sortinos.append(float(result.sortino))
        maxdds.append(float(result.max_drawdown))
        smooths.append(float(compute_pnl_smoothness_from_equity(curve)))
        ulcers.append(float(compute_ulcer_index(curve)))
        trades.append(int(result.num_trades))
        early.append(int(result.stopped_early))

    arr = np.asarray(returns, dtype=np.float64)
    monthly = _monthly_stats(returns, int(eval_days))
    row: dict[str, float | int | str] = {
        "config": config.name,
        "max_leverage": float(max_leverage),
        "max_hold_bars": int(max_hold_bars),
        "circuit_breaker_dd_pct": float(circuit_breaker_dd_pct),
        "eval_days": int(eval_days),
        "slip_bps": float(slippage_bps),
        "fill_buffer_bps": float(fill_buffer_bps),
        "decision_lag": int(decision_lag),
        "short_borrow_apr": float(short_borrow_apr),
        "median_pct": float(100.0 * np.percentile(arr, 50)),
        "p10_pct": float(100.0 * np.percentile(arr, 10)),
        "p90_pct": float(100.0 * np.percentile(arr, 90)),
        "median_monthly_pct": float(100.0 * np.percentile(monthly, 50)),
        "p10_monthly_pct": float(100.0 * np.percentile(monthly, 10)),
        "worst_monthly_pct": float(100.0 * np.min(monthly)),
        "best_monthly_pct": float(100.0 * np.max(monthly)),
        "neg_windows": int(np.sum(arr < 0.0)),
        "windows": int(arr.size),
        "p90_dd_pct": float(100.0 * np.percentile(np.asarray(maxdds, dtype=np.float64), 90)),
        "median_smooth": float(np.percentile(np.asarray(smooths, dtype=np.float64), 50)),
        "median_ulcer": float(np.percentile(np.asarray(ulcers, dtype=np.float64), 50)),
        "median_sortino": float(np.percentile(np.asarray(sortinos, dtype=np.float64), 50)),
        "median_trades": float(np.percentile(np.asarray(trades, dtype=np.float64), 50)),
        "stopped_early_windows": int(np.sum(np.asarray(early, dtype=np.int64) > 0)),
        "target_monthly_pct": float(target_monthly_pct),
        "target_max_dd_pct": float(target_max_dd_pct),
    }
    row["passes_target"] = int(
        _passes_production_target(
            row,
            target_monthly_pct=float(target_monthly_pct),
            max_dd_pct=float(target_max_dd_pct),
        )
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep risk controls for Binance33 deterministic rules.")
    parser.add_argument("--data-path", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_rule_risk_controls.csv"))
    parser.add_argument("--config-contains", default="reversal5_short_bottom_rb1")
    parser.add_argument("--eval-days", default="120")
    parser.add_argument("--slippage-bps", default="20")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", default="5,10,20")
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--max-leverage", default="1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--max-hold-bars", default="0,1,2,3,5,8")
    parser.add_argument("--circuit-breaker-dd-pct", default="0")
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.08)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--target-monthly-pct", type=float, default=27.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    parser.add_argument("--exclude-symbols", default="")
    parser.add_argument("--max-configs", type=int, default=0)
    args = parser.parse_args()

    data = read_mktd(args.data_path)
    exclude = {part.strip().upper() for part in str(args.exclude_symbols).split(",") if part.strip()}
    data = _masked_data(data, exclude_symbols=exclude)
    filters = [part.strip() for part in str(args.config_contains).split(",") if part.strip()]
    configs = _default_configs()
    if filters:
        configs = [config for config in configs if any(token in config.name for token in filters)]
    if args.max_configs > 0:
        configs = configs[: int(args.max_configs)]
    if not configs:
        raise ValueError("No configs selected")

    eval_days_grid = _parse_int_list(args.eval_days)
    slippage_grid = _parse_float_list(args.slippage_bps)
    fill_buffer_grid = _parse_float_list(args.fill_buffer_bps)
    leverage_grid = _parse_float_list(args.max_leverage)
    max_hold_grid = _parse_int_list(args.max_hold_bars)
    circuit_breaker_grid = _parse_float_list(args.circuit_breaker_dd_pct)

    fieldnames = [
        "config",
        "max_leverage",
        "max_hold_bars",
        "circuit_breaker_dd_pct",
        "eval_days",
        "slip_bps",
        "fill_buffer_bps",
        "decision_lag",
        "short_borrow_apr",
        "median_pct",
        "p10_pct",
        "p90_pct",
        "median_monthly_pct",
        "p10_monthly_pct",
        "worst_monthly_pct",
        "best_monthly_pct",
        "neg_windows",
        "windows",
        "p90_dd_pct",
        "median_smooth",
        "median_ulcer",
        "median_sortino",
        "median_trades",
        "stopped_early_windows",
        "target_monthly_pct",
        "target_max_dd_pct",
        "passes_target",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            for eval_days in eval_days_grid:
                for slip in slippage_grid:
                    for fill_buffer in fill_buffer_grid:
                        for max_leverage in leverage_grid:
                            for max_hold in max_hold_grid:
                                for circuit_dd in circuit_breaker_grid:
                                    row = _eval_config(
                                        data,
                                        config,
                                        eval_days=int(eval_days),
                                        stride=int(args.stride),
                                        slippage_bps=float(slip),
                                        fee_rate=float(args.fee_rate),
                                        fill_buffer_bps=float(fill_buffer),
                                        decision_lag=int(args.decision_lag),
                                        max_leverage=float(max_leverage),
                                        max_hold_bars=int(max_hold),
                                        circuit_breaker_dd_pct=float(circuit_dd),
                                        periods_per_year=float(args.periods_per_year),
                                        short_borrow_apr=float(args.short_borrow_apr),
                                        target_monthly_pct=float(args.target_monthly_pct),
                                        target_max_dd_pct=float(args.target_max_dd_pct),
                                    )
                                    rows.append(row)
                                    writer.writerow(row)

    best = sorted(
        rows,
        key=lambda row: (
            int(row["passes_target"]),
            float(row["median_monthly_pct"]) - max(0.0, float(row["p90_dd_pct"]) - float(args.target_max_dd_pct)),
        ),
        reverse=True,
    )[:20]
    print("\n=== Best risk-control rows ===")
    for row in best:
        print(
            f"{row['median_monthly_pct']:+7.2f}%/mo p10={row['p10_monthly_pct']:+7.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} dd90={row['p90_dd_pct']:.2f}% "
            f"trades={row['median_trades']:.0f} pass={row['passes_target']} "
            f"lev={row['max_leverage']} hold={row['max_hold_bars']} cb={row['circuit_breaker_dd_pct']} "
            f"fill={row['fill_buffer_bps']} {row['config']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
