#!/usr/bin/env python3
"""Portfolio-pack deterministic Binance33 daily rules.

This is the diversified companion to ``sweep_binance33_rules.py``. The original
rule sweep picks one symbol at a time through the single-position simulator.
Here we keep the same lagged feature rules but allocate across the top-N
eligible symbols in a simple portfolio simulation, so we can test whether the
recent short-reversion edge survives diversification and lower per-symbol risk.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, read_mktd
from scripts.sweep_binance33_rules import (
    FEATURES,
    RuleConfig,
    _candidate_starts,
    _default_configs,
    _score_vector,
    _slice_window,
)
from scripts.sweep_binance33_xgb import _monthly_equivalent_return, _passes_production_target
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


@dataclass(frozen=True)
class PackConfig:
    top_n: int
    allocation_mode: str
    max_gross: float
    max_weight: float


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


def _parse_str_list(value: str) -> list[str]:
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(f"expected at least one value in {value!r}")
    return out


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0.0, equity / peak - 1.0, 0.0)
    return float(max(0.0, -np.nanmin(dd)))


def _sortino_from_equity(equity: np.ndarray, *, periods_per_year: float = 365.0) -> float:
    if equity.size <= 1:
        return 0.0
    prev = equity[:-1]
    cur = equity[1:]
    valid = np.isfinite(prev) & np.isfinite(cur) & (prev > 0.0)
    if not valid.any():
        return 0.0
    returns = cur[valid] / prev[valid] - 1.0
    downside = returns[returns < 0.0]
    if downside.size == 0:
        return float("inf") if float(returns.mean()) > 0.0 else 0.0
    denom = float(downside.std())
    if denom <= 1e-12:
        return 0.0
    return float(returns.mean() / denom * np.sqrt(float(periods_per_year)))


def _selected_indices(data: MktdData, config: RuleConfig, *, t: int, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    symbols = [sym.upper() for sym in data.symbols]
    btc_idx = next((idx for idx, sym in enumerate(symbols) if sym.startswith("BTC")), 0)
    feat = data.features[t]
    btc_mom = float(feat[btc_idx, FEATURES["return_20d"]])
    if config.mode != "regime_long_top_short_bottom" and config.btc_min_return_20d > -9.0:
        if not np.isfinite(btc_mom) or btc_mom < float(config.btc_min_return_20d):
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    scores = _score_vector(feat, config.score)
    valid = np.isfinite(scores)
    if data.tradable is not None:
        valid &= np.asarray(data.tradable[t], dtype=bool)
    threshold = float(config.min_abs_score)

    if config.mode == "long_top":
        candidates = np.flatnonzero(valid & (scores >= threshold))
        ordered = candidates[np.argsort(scores[candidates])[::-1]]
        signs = np.ones(min(len(ordered), int(top_n)), dtype=np.float64)
    elif config.mode == "short_bottom":
        candidates = np.flatnonzero(valid & (-scores >= threshold))
        ordered = candidates[np.argsort(scores[candidates])]
        signs = -np.ones(min(len(ordered), int(top_n)), dtype=np.float64)
    elif config.mode == "long_or_short":
        long_idx = np.flatnonzero(valid & (scores >= threshold))
        short_idx = np.flatnonzero(valid & (-scores >= threshold))
        signed = [(int(idx), 1.0, abs(float(scores[idx]))) for idx in long_idx]
        signed.extend((int(idx), -1.0, abs(float(scores[idx]))) for idx in short_idx)
        signed.sort(key=lambda item: item[2], reverse=True)
        chosen = signed[: int(top_n)]
        return (
            np.asarray([item[0] for item in chosen], dtype=np.int64),
            np.asarray([item[1] for item in chosen], dtype=np.float64),
        )
    elif config.mode == "regime_long_top_short_bottom":
        if np.isfinite(btc_mom) and btc_mom >= max(config.btc_min_return_20d, 0.0):
            candidates = np.flatnonzero(valid & (scores >= threshold))
            ordered = candidates[np.argsort(scores[candidates])[::-1]]
            signs = np.ones(min(len(ordered), int(top_n)), dtype=np.float64)
        else:
            candidates = np.flatnonzero(valid & (-scores >= threshold))
            ordered = candidates[np.argsort(scores[candidates])]
            signs = -np.ones(min(len(ordered), int(top_n)), dtype=np.float64)
    else:
        raise ValueError(f"unknown mode: {config.mode}")
    return ordered[: int(top_n)].astype(np.int64, copy=False), signs


def _target_weights(data: MktdData, rule: RuleConfig, cfg: PackConfig, *, t: int) -> np.ndarray:
    t = max(0, min(int(t), data.num_timesteps - 1))
    idx, signs = _selected_indices(data, rule, t=t, top_n=int(cfg.top_n))
    weights = np.zeros(data.num_symbols, dtype=np.float64)
    if idx.size == 0:
        return weights
    scores = np.abs(_score_vector(data.features[t], rule.score)[idx]).astype(np.float64, copy=False)
    scores = np.maximum(scores, 1e-12)
    vol = np.asarray(data.features[t, idx, FEATURES["volatility_20d"]], dtype=np.float64)
    vol = np.where(np.isfinite(vol) & (vol > 1e-6), vol, np.nan)
    if cfg.allocation_mode == "equal":
        raw = np.ones_like(scores)
    elif cfg.allocation_mode == "score":
        raw = scores
    elif cfg.allocation_mode == "inv_vol":
        raw = 1.0 / np.where(np.isfinite(vol), vol, np.inf)
    elif cfg.allocation_mode == "inv_vol_score":
        raw = scores / np.where(np.isfinite(vol), vol, np.inf)
    else:
        raise ValueError(f"unknown allocation_mode: {cfg.allocation_mode}")
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    if float(raw.sum()) <= 0.0:
        return weights
    gross = max(0.0, float(cfg.max_gross))
    alloc = raw / float(raw.sum()) * gross
    max_weight = max(0.0, float(cfg.max_weight))
    if max_weight > 0.0:
        alloc = np.minimum(alloc, max_weight)
    weights[idx] = signs * alloc
    gross_now = float(np.abs(weights).sum())
    if gross_now > gross and gross_now > 0.0:
        weights *= gross / gross_now
    return weights


def _simulate_window(
    data: MktdData,
    rule: RuleConfig,
    cfg: PackConfig,
    *,
    eval_days: int,
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    decision_lag: int,
) -> dict[str, float | int]:
    close = np.asarray(data.prices[:, :, P_CLOSE], dtype=np.float64)
    weights = np.zeros(data.num_symbols, dtype=np.float64)
    equity = 1.0
    curve = [equity]
    trades = 0
    cost_per_turnover = float(fee_rate) + (float(slippage_bps) + float(fill_buffer_bps)) / 10_000.0
    for step in range(int(eval_days)):
        target = weights
        signal_t = int(step) - int(decision_lag)
        if signal_t >= 0 and (step == 0 or step % max(1, int(rule.rebalance_days)) == 0):
            target = _target_weights(data, rule, cfg, t=signal_t)
        p0 = close[step]
        p1 = close[step + 1]
        tradable = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
        if data.tradable is not None:
            tradable &= np.asarray(data.tradable[step], dtype=bool)
            tradable &= np.asarray(data.tradable[step + 1], dtype=bool)
        target = np.where(tradable, target, 0.0)
        turnover = float(np.abs(target - weights).sum())
        if turnover > 1e-9:
            trades += int(np.count_nonzero(np.abs(target - weights) > 1e-9))
        daily_ret = np.zeros(data.num_symbols, dtype=np.float64)
        daily_ret[tradable] = p1[tradable] / p0[tradable] - 1.0
        equity = max(1e-9, equity * (1.0 + float(np.dot(target, daily_ret)) - turnover * cost_per_turnover))
        weights = target
        curve.append(equity)
    final_turnover = float(np.abs(weights).sum())
    if final_turnover > 1e-9:
        trades += int(np.count_nonzero(np.abs(weights) > 1e-9))
        equity = max(1e-9, equity * (1.0 - final_turnover * cost_per_turnover))
        curve[-1] = equity
    equity_curve = np.asarray(curve, dtype=np.float64)
    return {
        "total_return": float(equity_curve[-1] - 1.0),
        "max_drawdown": _max_drawdown(equity_curve),
        "sortino": _sortino_from_equity(equity_curve),
        "smooth": float(compute_pnl_smoothness_from_equity(equity_curve)),
        "ulcer": float(compute_ulcer_index(equity_curve)),
        "trades": int(trades),
    }


def _eval_config(
    data: MktdData,
    rule: RuleConfig,
    cfg: PackConfig,
    *,
    eval_days: int,
    stride: int,
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    decision_lag: int,
) -> dict[str, float | int | str]:
    returns: list[float] = []
    maxdds: list[float] = []
    sortinos: list[float] = []
    smooths: list[float] = []
    ulcers: list[float] = []
    trades: list[int] = []
    for start in _candidate_starts(data, eval_days, stride=int(stride)):
        window = _slice_window(data, start, eval_days)
        result = _simulate_window(
            window,
            rule,
            cfg,
            eval_days=int(eval_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            fill_buffer_bps=float(fill_buffer_bps),
            decision_lag=int(decision_lag),
        )
        returns.append(float(result["total_return"]))
        maxdds.append(float(result["max_drawdown"]))
        sortinos.append(float(result["sortino"]))
        smooths.append(float(result["smooth"]))
        ulcers.append(float(result["ulcer"]))
        trades.append(int(result["trades"]))
    arr = np.asarray(returns, dtype=np.float64)
    monthly = np.asarray([_monthly_equivalent_return(float(ret), int(eval_days)) for ret in arr], dtype=np.float64)
    return {
        "config": rule.name,
        "top_n": int(cfg.top_n),
        "allocation_mode": cfg.allocation_mode,
        "max_gross": float(cfg.max_gross),
        "max_weight": float(cfg.max_weight),
        "eval_days": int(eval_days),
        "slip_bps": float(slippage_bps),
        "median_pct": float(100.0 * np.percentile(arr, 50)),
        "p10_pct": float(100.0 * np.percentile(arr, 10)),
        "p90_pct": float(100.0 * np.percentile(arr, 90)),
        "median_monthly_pct": float(100.0 * np.percentile(monthly, 50)),
        "p10_monthly_pct": float(100.0 * np.percentile(monthly, 10)),
        "worst_monthly_pct": float(100.0 * np.min(monthly)),
        "best_monthly_pct": float(100.0 * np.max(monthly)),
        "neg_windows": int(np.sum(arr < 0.0)),
        "windows": int(arr.size),
        "p90_dd_pct": float(100.0 * np.percentile(np.asarray(maxdds), 90)),
        "median_sortino": float(np.percentile(np.asarray(sortinos), 50)),
        "median_smooth": float(np.percentile(np.asarray(smooths), 50)),
        "median_ulcer": float(np.percentile(np.asarray(ulcers), 50)),
        "median_trades": float(np.percentile(np.asarray(trades), 50)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep deterministic Binance33 portfolio-pack rules.")
    parser.add_argument("--data-path", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_rule_portfolio_pack.csv"))
    parser.add_argument("--eval-days", default="30,120")
    parser.add_argument("--slippage-bps", default="20")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--config-contains", default="reversal5_short_bottom_rb1")
    parser.add_argument("--top-n-grid", default="1,2,3,5")
    parser.add_argument("--allocation-modes", default="equal,inv_vol_score")
    parser.add_argument("--max-gross-grid", default="1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--max-weight-grid", default="1.0,0.5,0.33")
    parser.add_argument("--target-monthly-pct", type=float, default=27.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    parser.add_argument("--require-production-target", action="store_true")
    args = parser.parse_args()

    data = read_mktd(args.data_path)
    filters = [part.strip() for part in str(args.config_contains).split(",") if part.strip()]
    rules = _default_configs()
    if filters:
        rules = [rule for rule in rules if any(token in rule.name for token in filters)]
    pack_configs = [
        PackConfig(top_n=top_n, allocation_mode=mode, max_gross=max_gross, max_weight=max_weight)
        for top_n, mode, max_gross, max_weight in product(
            _parse_int_list(args.top_n_grid),
            _parse_str_list(args.allocation_modes),
            _parse_float_list(args.max_gross_grid),
            _parse_float_list(args.max_weight_grid),
        )
    ]
    eval_days = _parse_int_list(args.eval_days)
    slippages = _parse_float_list(args.slippage_bps)
    fieldnames = [
        "config",
        "top_n",
        "allocation_mode",
        "max_gross",
        "max_weight",
        "eval_days",
        "slip_bps",
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
        "median_sortino",
        "median_smooth",
        "median_ulcer",
        "median_trades",
        "target_monthly_pct",
        "target_max_dd_pct",
        "passes_target",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rule in rules:
            for cfg in pack_configs:
                for days in eval_days:
                    for slip in slippages:
                        row = _eval_config(
                            data,
                            rule,
                            cfg,
                            eval_days=int(days),
                            stride=int(args.stride),
                            fee_rate=float(args.fee_rate),
                            slippage_bps=float(slip),
                            fill_buffer_bps=float(args.fill_buffer_bps),
                            decision_lag=int(args.decision_lag),
                        )
                        row["target_monthly_pct"] = float(args.target_monthly_pct)
                        row["target_max_dd_pct"] = float(args.target_max_dd_pct)
                        row["passes_target"] = int(
                            _passes_production_target(
                                row,
                                target_monthly_pct=float(args.target_monthly_pct),
                                max_dd_pct=float(args.target_max_dd_pct),
                            )
                        )
                        rows.append(row)
                        writer.writerow(row)
                        fh.flush()

    promotion_days = max(eval_days)
    promotion_slip = max(slippages)
    subset = [
        row
        for row in rows
        if int(row["eval_days"]) == int(promotion_days) and float(row["slip_bps"]) == float(promotion_slip)
    ]
    subset.sort(key=lambda item: float(item["median_monthly_pct"]), reverse=True)
    print(f"\n=== Best rule portfolio packs {promotion_days}d slip{promotion_slip:g} ===")
    for row in subset[:20]:
        print(
            f"{row['median_monthly_pct']:+7.2f}%/mo p10/mo={row['p10_monthly_pct']:+7.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} dd90={row['p90_dd_pct']:.2f}% "
            f"pass={row['passes_target']} {row['config']} top={row['top_n']} "
            f"{row['allocation_mode']} gross={row['max_gross']} maxw={row['max_weight']}"
        )
    promoted = [row for row in subset if int(row["passes_target"]) == 1]
    print(f"\n=== Production target candidates {promotion_days}d slip{promotion_slip:g} ===")
    if promoted:
        for row in promoted[:10]:
            print(
                f"{row['median_monthly_pct']:+7.2f}%/mo p10/mo={row['p10_monthly_pct']:+7.2f}% "
                f"dd90={row['p90_dd_pct']:.2f}% {row['config']} top={row['top_n']}"
            )
    else:
        print("none")
    print(f"\nwrote {args.out}")
    if args.require_production_target and not promoted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
