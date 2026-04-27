#!/usr/bin/env python3
"""Research-only multi-position Binance33 portfolio packing for daily XGB scores.

This is not the pufferlib single-position ground-truth simulator. It is a fast
portfolio-packing experiment to test whether XGB daily cross-sectional scores
become smoother when spread across multiple independent pairs at 1x gross.
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
from scripts.sweep_binance33_xgb import (
    FEATURES,
    Experiment,
    _build_dataset,
    _candidate_starts,
    _experiments,
    _monthly_equivalent_return,
    _passes_production_target,
    _precompute_scores,
    _slice_window,
    _train_xgb,
)
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


@dataclass(frozen=True)
class PackConfig:
    top_n: int
    allocation_mode: str
    min_abs_score: float
    score_temp: float
    target_vol: float
    max_weight: float
    max_gross: float


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError(f"expected at least one float in {value!r}")
    return parsed


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError(f"expected at least one int in {value!r}")
    return parsed


def _parse_str_list(value: str) -> list[str]:
    parsed = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError(f"expected at least one value in {value!r}")
    return parsed


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
        return float("inf") if returns.mean() > 0.0 else 0.0
    denom = float(downside.std())
    if denom <= 1e-12:
        return 0.0
    return float(returns.mean() / denom * np.sqrt(float(periods_per_year)))


def _select_candidates(
    data: MktdData,
    scores: np.ndarray,
    exp: Experiment,
    cfg: PackConfig,
    *,
    t: int,
) -> list[tuple[int, float]]:
    valid = np.isfinite(scores)
    if data.tradable is not None:
        valid &= np.asarray(data.tradable[t], dtype=bool)
    if not valid.any():
        return []

    btc_idx = next((idx for idx, symbol in enumerate(data.symbols) if symbol.upper().startswith("BTC")), 0)
    btc_mom = float(data.features[t, btc_idx, FEATURES["return_20d"]])
    threshold = max(float(exp.min_abs_score), float(cfg.min_abs_score))
    selected: list[tuple[int, float]] = []

    def add_longs() -> None:
        long_idx = np.flatnonzero(valid & (scores >= threshold))
        ordered = long_idx[np.argsort(scores[long_idx])[::-1]]
        selected.extend((int(idx), 1.0) for idx in ordered[: max(0, int(cfg.top_n))])

    def add_shorts() -> None:
        short_idx = np.flatnonzero(valid & (-scores >= threshold))
        ordered = short_idx[np.argsort(scores[short_idx])]
        selected.extend((int(idx), -1.0) for idx in ordered[: max(0, int(cfg.top_n))])

    if exp.btc_gate > -9.0 and (not np.isfinite(btc_mom) or btc_mom < exp.btc_gate):
        if exp.mode == "long_top":
            return []
    if exp.mode == "long_top":
        add_longs()
    elif exp.mode == "short_bottom":
        add_shorts()
    elif exp.mode == "long_or_short":
        add_longs()
        add_shorts()
        selected.sort(key=lambda item: abs(float(scores[item[0]])), reverse=True)
        selected = selected[: max(0, int(cfg.top_n))]
    elif exp.mode == "regime":
        if np.isfinite(btc_mom) and btc_mom >= max(exp.btc_gate, 0.0):
            add_longs()
        else:
            add_shorts()
    else:
        raise ValueError(f"unknown experiment mode: {exp.mode}")
    return selected


def _target_weights(
    data: MktdData,
    scores_by_t: np.ndarray,
    exp: Experiment,
    cfg: PackConfig,
    *,
    t: int,
) -> np.ndarray:
    t = max(0, min(int(t), data.num_timesteps - 1))
    scores = np.asarray(scores_by_t[t], dtype=np.float64)
    selected = _select_candidates(data, scores, exp, cfg, t=t)
    weights = np.zeros(data.num_symbols, dtype=np.float64)
    if not selected:
        return weights

    idx = np.asarray([item[0] for item in selected], dtype=np.int64)
    signs = np.asarray([item[1] for item in selected], dtype=np.float64)
    edge = np.abs(scores[idx]).astype(np.float64, copy=False)
    edge = np.maximum(edge, 1e-12)
    vol = np.asarray(data.features[t, idx, FEATURES["volatility_20d"]], dtype=np.float64)
    vol = np.where(np.isfinite(vol) & (vol > 1e-6), vol, np.nan)

    mode = str(cfg.allocation_mode)
    if mode == "equal":
        raw = np.ones_like(edge)
    elif mode == "score":
        raw = edge
    elif mode == "softmax":
        temp = max(1e-6, float(cfg.score_temp))
        centered = edge / temp
        centered = centered - float(np.max(centered))
        raw = np.exp(centered)
    elif mode == "inv_vol":
        raw = 1.0 / np.where(np.isfinite(vol), vol, np.inf)
    elif mode == "inv_vol_score":
        raw = edge / np.where(np.isfinite(vol), vol, np.inf)
    else:
        raise ValueError(f"unknown allocation mode: {cfg.allocation_mode}")

    if float(cfg.target_vol) > 0.0:
        vol_scale = float(cfg.target_vol) / np.where(np.isfinite(vol), vol, np.inf)
        raw = raw * np.clip(vol_scale, 0.0, 3.0)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    if float(raw.sum()) <= 0.0:
        return weights

    gross = max(0.0, float(cfg.max_gross))
    max_weight = max(0.0, float(cfg.max_weight))
    alloc = raw / float(raw.sum()) * gross
    if max_weight > 0.0:
        alloc = np.minimum(alloc, max_weight)
        alloc_sum = float(alloc.sum())
        if alloc_sum > gross and alloc_sum > 0.0:
            alloc = alloc / alloc_sum * gross
    weights[idx] = signs * alloc
    gross_now = float(np.abs(weights).sum())
    if gross_now > gross and gross_now > 0.0:
        weights *= gross / gross_now
    return weights


def _simulate_pack_window(
    data: MktdData,
    scores_by_t: np.ndarray,
    exp: Experiment,
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
    cost_per_turnover = float(fee_rate) + (float(slippage_bps) + float(fill_buffer_bps)) / 10000.0

    for step in range(int(eval_days)):
        target = weights
        signal_t = int(step) - int(decision_lag)
        if signal_t >= 0 and (step == 0 or step % max(1, int(exp.rebalance_days)) == 0):
            target = _target_weights(data, scores_by_t, exp, cfg, t=signal_t)

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
        pnl = float(np.dot(target, daily_ret))
        cost = turnover * cost_per_turnover
        equity = max(1e-9, equity * (1.0 + pnl - cost))
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


def _eval_pack(
    data: MktdData,
    scores_by_t: np.ndarray,
    exp: Experiment,
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
    for start in _candidate_starts(data, eval_days, stride=stride):
        window = _slice_window(data, start, eval_days)
        window_scores = np.asarray(scores_by_t[start : start + int(eval_days) + 1], dtype=np.float64)
        result = _simulate_pack_window(
            window,
            window_scores,
            exp,
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
    monthly = np.asarray(
        [_monthly_equivalent_return(float(total_return), int(eval_days)) for total_return in arr], dtype=np.float64
    )
    return {
        "experiment": exp.name,
        "horizon": int(exp.horizon),
        "label": exp.label,
        "mode": exp.mode,
        "rebalance_days": int(exp.rebalance_days),
        "top_n": int(cfg.top_n),
        "allocation_mode": cfg.allocation_mode,
        "min_abs_score": float(cfg.min_abs_score),
        "score_temp": float(cfg.score_temp),
        "target_vol": float(cfg.target_vol),
        "max_weight": float(cfg.max_weight),
        "max_gross": float(cfg.max_gross),
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
        "median_smooth": float(np.percentile(np.asarray(smooths), 50)),
        "median_ulcer": float(np.percentile(np.asarray(ulcers), 50)),
        "median_sortino": float(np.percentile(np.asarray(sortinos), 50)),
        "median_trades": float(np.percentile(np.asarray(trades), 50)),
        "failed_fast": 0,
        "fail_reason": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Binance33 XGB daily multi-position portfolio packs.")
    parser.add_argument("--train-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_train.bin"))
    parser.add_argument("--eval-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_xgb_portfolio_pack.csv"))
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-names", default="xgb08,xgb11,xgb14")
    parser.add_argument("--eval-days", default="30,120")
    parser.add_argument("--slippage-bps", default="20")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--top-n-grid", default="1,2,3,5,8")
    parser.add_argument("--allocation-modes", default="equal,score,softmax,inv_vol,inv_vol_score")
    parser.add_argument("--min-abs-score-grid", default="0.0")
    parser.add_argument("--score-temp-grid", default="0.005,0.01,0.02")
    parser.add_argument("--target-vol-grid", default="0.0")
    parser.add_argument("--max-weight-grid", default="1.0,0.5,0.33,0.2")
    parser.add_argument("--max-gross-grid", default="1.0")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--target-monthly-pct", type=float, default=27.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    parser.add_argument("--require-production-target", action="store_true")
    args = parser.parse_args()

    train_data = read_mktd(args.train_data)
    eval_data = read_mktd(args.eval_data)
    experiments = _experiments()
    requested = [part.strip() for part in str(args.experiment_names).split(",") if part.strip()]
    by_name = {exp.name: exp for exp in experiments}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError(f"unknown experiment names: {', '.join(missing)}")
    experiments = [by_name[name] for name in requested]

    eval_days = _parse_int_list(args.eval_days)
    slippages = _parse_float_list(args.slippage_bps)
    pack_configs = [
        PackConfig(
            top_n=top_n,
            allocation_mode=allocation_mode,
            min_abs_score=min_abs_score,
            score_temp=score_temp,
            target_vol=target_vol,
            max_weight=max_weight,
            max_gross=max_gross,
        )
        for top_n, allocation_mode, min_abs_score, score_temp, target_vol, max_weight, max_gross in product(
            _parse_int_list(args.top_n_grid),
            _parse_str_list(args.allocation_modes),
            _parse_float_list(args.min_abs_score_grid),
            _parse_float_list(args.score_temp_grid),
            _parse_float_list(args.target_vol_grid),
            _parse_float_list(args.max_weight_grid),
            _parse_float_list(args.max_gross_grid),
        )
    ]

    fieldnames = [
        "experiment",
        "horizon",
        "label",
        "mode",
        "rebalance_days",
        "top_n",
        "allocation_mode",
        "min_abs_score",
        "score_temp",
        "target_vol",
        "max_weight",
        "max_gross",
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
        "median_smooth",
        "median_ulcer",
        "median_sortino",
        "median_trades",
        "failed_fast",
        "fail_reason",
        "target_monthly_pct",
        "target_max_dd_pct",
        "passes_target",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for exp_idx, exp in enumerate(experiments, start=1):
            x_train, y_train = _build_dataset(train_data, horizon=exp.horizon, label=exp.label)
            model = _train_xgb(x_train, y_train, exp, rounds=int(args.rounds), device=str(args.device))
            scores_by_t = _precompute_scores(eval_data, model)
            for cfg in pack_configs:
                for days in eval_days:
                    for slip in slippages:
                        row = _eval_pack(
                            eval_data,
                            scores_by_t,
                            exp,
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
            print(f"evaluated {exp_idx}/{len(experiments)} {exp.name}", flush=True)

    promotion_days = max(eval_days)
    promotion_slip = max(slippages)
    subset = [
        row
        for row in rows
        if int(row["eval_days"]) == int(promotion_days) and float(row["slip_bps"]) == float(promotion_slip)
    ]
    subset.sort(key=lambda row: float(row["median_monthly_pct"]), reverse=True)
    print(f"\n=== Best portfolio packs {promotion_days}d slip{promotion_slip:g} ===")
    for row in subset[:20]:
        print(
            f"{row['median_monthly_pct']:+7.2f}%/mo p10/mo={row['p10_monthly_pct']:+7.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} dd90={row['p90_dd_pct']:.2f}% "
            f"pass={row['passes_target']} {row['experiment']} top={row['top_n']} "
            f"{row['allocation_mode']} maxw={row['max_weight']} gross={row['max_gross']}"
        )
    promoted = [row for row in subset if int(row["passes_target"]) == 1]
    print(
        f"\n=== Production target candidates {promotion_days}d slip{promotion_slip:g} "
        f"target={float(args.target_monthly_pct):g}%/mo ==="
    )
    if promoted:
        for row in promoted[:10]:
            print(
                f"{row['median_monthly_pct']:+7.2f}%/mo p10/mo={row['p10_monthly_pct']:+7.2f}% "
                f"dd90={row['p90_dd_pct']:.2f}% {row['experiment']} top={row['top_n']} {row['allocation_mode']}"
            )
    else:
        print("none")
    print(f"\nwrote {args.out}")
    if args.require_production_target and not promoted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
