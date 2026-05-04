#!/usr/bin/env python3
"""Random-search linear cross-sectional Binance33 rules."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from scripts.sweep_binance33_rules import FEATURES, _candidate_starts, _slice_window
from scripts.sweep_binance33_xgb import _monthly_equivalent_return, _passes_production_target
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


FEATURE_NAMES = (
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "ma_delta_20d",
    "rsi_14",
    "trend_60d",
    "drawdown_20d",
    "drawdown_60d",
    "log_volume_z20d",
)


@dataclass(frozen=True)
class LinearRule:
    name: str
    coeffs: np.ndarray
    rebalance_days: int
    min_abs_score: float
    btc_min_return_20d: float


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


def _zscore_columns(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    for col in range(values.shape[1]):
        series = np.asarray(values[:, col], dtype=np.float64)
        mask = valid & np.isfinite(series)
        if int(mask.sum()) <= 1:
            continue
        mean = float(series[mask].mean())
        std = float(series[mask].std())
        out[mask, col] = (series[mask] - mean) / max(std, 1e-6)
    return out


def _score_vector(data: MktdData, rule: LinearRule, t: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(data.features[t][:, [FEATURES[name] for name in FEATURE_NAMES]], dtype=np.float64)
    valid = np.isfinite(raw).all(axis=1)
    if data.tradable is not None:
        valid &= np.asarray(data.tradable[t], dtype=bool)
    z = _zscore_columns(raw, valid)
    scores = z @ np.asarray(rule.coeffs, dtype=np.float64)
    scores = np.where(valid & np.isfinite(scores), scores, np.nan)
    return scores, valid


def _make_policy(data: MktdData, rule: LinearRule, *, decision_lag: int):
    symbols = [symbol.upper() for symbol in data.symbols]
    btc_idx = next((idx for idx, symbol in enumerate(symbols) if symbol.startswith("BTC")), 0)
    pending: list[int] = []
    state = {"step": 0, "last_action": 0}

    def raw_policy(_obs: np.ndarray) -> int:
        step = int(state["step"])
        state["step"] = step + 1
        if step > 0 and int(rule.rebalance_days) > 1 and step % int(rule.rebalance_days) != 0:
            return int(state["last_action"])

        feat_idx = max(0, min(step - 1, data.num_timesteps - 1))
        btc_mom = float(data.features[feat_idx, btc_idx, FEATURES["return_20d"]])
        if rule.btc_min_return_20d > -9.0 and (not np.isfinite(btc_mom) or btc_mom < rule.btc_min_return_20d):
            state["last_action"] = 0
            return 0

        tradable_idx = max(0, min(step, data.num_timesteps - 1))
        scores, valid = _score_vector(data, rule, tradable_idx)
        if not valid.any() or not np.isfinite(scores[valid]).any():
            state["last_action"] = 0
            return 0
        valid_idx = np.flatnonzero(valid & np.isfinite(scores))
        chosen = int(valid_idx[int(np.nanargmin(scores[valid_idx]))])
        edge = -float(scores[chosen])
        action = 1 + data.num_symbols + chosen if edge >= float(rule.min_abs_score) else 0
        state["last_action"] = int(action)
        return int(action)

    if decision_lag <= 0:
        return raw_policy

    def lagged_policy(obs: np.ndarray) -> int:
        action_now = int(raw_policy(obs))
        pending.append(action_now)
        if len(pending) <= int(decision_lag):
            return 0
        return int(pending.pop(0))

    return lagged_policy


def _default_rules(rng: np.random.Generator, n_random: int) -> list[LinearRule]:
    rules: list[LinearRule] = []
    seed_coeffs = [
        ("neg_ret5", {"return_5d": -1.0}),
        ("neg_ret5_ret20", {"return_5d": -0.75, "return_20d": -0.25}),
        ("winner_rsi", {"return_5d": -0.65, "rsi_14": -0.25, "log_volume_z20d": -0.10}),
        ("winner_lowdd", {"return_5d": -0.70, "drawdown_20d": 0.20, "volatility_20d": 0.10}),
    ]
    for base_name, mapping in seed_coeffs:
        coeff = np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
        for key, value in mapping.items():
            coeff[FEATURE_NAMES.index(key)] = float(value)
        for rebalance in (1, 3, 7):
            for threshold in (0.0, 0.25, 0.5):
                for btc_gate in (-0.05, 0.0):
                    rules.append(
                        LinearRule(
                            name=f"{base_name}_rb{rebalance}_thr{threshold:g}_btcmom20gt{btc_gate:g}",
                            coeffs=coeff,
                            rebalance_days=rebalance,
                            min_abs_score=threshold,
                            btc_min_return_20d=btc_gate,
                        )
                    )
    for idx in range(int(n_random)):
        coeff = rng.normal(0.0, 1.0, size=len(FEATURE_NAMES))
        # Bias toward shorting recent winners while still exploring composites.
        coeff[FEATURE_NAMES.index("return_5d")] -= 1.25
        coeff[FEATURE_NAMES.index("return_20d")] -= 0.35
        denom = float(np.sum(np.abs(coeff)))
        coeff = coeff / max(denom, 1e-9)
        rebalance = int(rng.choice([1, 3, 7, 14], p=[0.55, 0.25, 0.15, 0.05]))
        threshold = float(rng.choice([0.0, 0.2, 0.35, 0.5, 0.75], p=[0.35, 0.25, 0.2, 0.15, 0.05]))
        btc_gate = float(rng.choice([-0.08, -0.05, -0.02, 0.0], p=[0.15, 0.45, 0.25, 0.15]))
        rules.append(
            LinearRule(
                name=f"rand{idx:04d}_rb{rebalance}_thr{threshold:g}_btcmom20gt{btc_gate:g}",
                coeffs=coeff,
                rebalance_days=rebalance,
                min_abs_score=threshold,
                btc_min_return_20d=btc_gate,
            )
        )
    return rules


def _eval_rule(
    data: MktdData,
    rule: LinearRule,
    *,
    eval_days: int,
    stride: int,
    slippage_bps: float,
    fee_rate: float,
    fill_buffer_bps: float,
    decision_lag: int,
    max_leverage: float,
    periods_per_year: float,
    short_borrow_apr: float,
    target_monthly_pct: float,
    target_max_dd_pct: float,
) -> dict[str, float | int | str]:
    returns: list[float] = []
    maxdds: list[float] = []
    sortinos: list[float] = []
    smooths: list[float] = []
    ulcers: list[float] = []
    trades: list[int] = []
    for start in _candidate_starts(data, eval_days, stride=int(stride)):
        window = _slice_window(data, start, eval_days)
        result = simulate_daily_policy(
            window,
            _make_policy(window, rule, decision_lag=int(decision_lag)),
            max_steps=int(eval_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            fill_buffer_bps=float(fill_buffer_bps),
            max_leverage=float(max_leverage),
            periods_per_year=float(periods_per_year),
            short_borrow_apr=float(short_borrow_apr),
            enable_drawdown_profit_early_exit=False,
            enable_metric_threshold_early_exit=False,
        )
        curve = np.asarray(result.equity_curve if result.equity_curve is not None else [], dtype=np.float64)
        returns.append(float(result.total_return))
        maxdds.append(float(result.max_drawdown))
        sortinos.append(float(result.sortino))
        smooths.append(float(compute_pnl_smoothness_from_equity(curve)))
        ulcers.append(float(compute_ulcer_index(curve)))
        trades.append(int(result.num_trades))
    arr = np.asarray(returns, dtype=np.float64)
    monthly = np.asarray([_monthly_equivalent_return(float(ret), int(eval_days)) for ret in arr], dtype=np.float64)
    row: dict[str, float | int | str] = {
        "config": rule.name,
        "coeffs_json": json.dumps({name: round(float(value), 6) for name, value in zip(FEATURE_NAMES, rule.coeffs)}),
        "max_leverage": float(max_leverage),
        "rebalance_days": int(rule.rebalance_days),
        "min_abs_score": float(rule.min_abs_score),
        "btc_min_return_20d": float(rule.btc_min_return_20d),
        "eval_days": int(eval_days),
        "slip_bps": float(slippage_bps),
        "fill_buffer_bps": float(fill_buffer_bps),
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
    parser = argparse.ArgumentParser(description="Search linear Binance33 short rules.")
    parser.add_argument("--data-path", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_linear_rule_search.csv"))
    parser.add_argument("--n-random", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--max-leverage", default="1.0,1.25,1.5")
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.08)
    parser.add_argument("--target-monthly-pct", type=float, default=27.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    args = parser.parse_args()

    data = read_mktd(args.data_path)
    rng = np.random.default_rng(int(args.seed))
    rules = _default_rules(rng, int(args.n_random))
    leverage_grid = _parse_float_list(args.max_leverage)
    rows: list[dict[str, float | int | str]] = []
    fieldnames = [
        "config",
        "coeffs_json",
        "max_leverage",
        "rebalance_days",
        "min_abs_score",
        "btc_min_return_20d",
        "eval_days",
        "slip_bps",
        "fill_buffer_bps",
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
        "target_monthly_pct",
        "target_max_dd_pct",
        "passes_target",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rule_idx, rule in enumerate(rules, start=1):
            for max_leverage in leverage_grid:
                row = _eval_rule(
                    data,
                    rule,
                    eval_days=int(args.eval_days),
                    stride=int(args.stride),
                    slippage_bps=float(args.slippage_bps),
                    fee_rate=float(args.fee_rate),
                    fill_buffer_bps=float(args.fill_buffer_bps),
                    decision_lag=int(args.decision_lag),
                    max_leverage=float(max_leverage),
                    periods_per_year=float(args.periods_per_year),
                    short_borrow_apr=float(args.short_borrow_apr),
                    target_monthly_pct=float(args.target_monthly_pct),
                    target_max_dd_pct=float(args.target_max_dd_pct),
                )
                rows.append(row)
                writer.writerow(row)
            if rule_idx % 50 == 0:
                print(f"evaluated {rule_idx}/{len(rules)} rules")

    best = sorted(
        rows,
        key=lambda row: (
            int(row["passes_target"]),
            float(row["median_monthly_pct"]) - max(0.0, float(row["p90_dd_pct"]) - float(args.target_max_dd_pct)),
        ),
        reverse=True,
    )[:20]
    print("\n=== Best linear rules ===")
    for row in best:
        print(
            f"{row['median_monthly_pct']:+7.2f}%/mo p10={row['p10_monthly_pct']:+7.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} dd90={row['p90_dd_pct']:.2f}% "
            f"trades={row['median_trades']:.0f} pass={row['passes_target']} "
            f"lev={row['max_leverage']} {row['config']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
