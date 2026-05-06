#!/usr/bin/env python3
"""Search forecastable Binance33 sub-universes and evaluate on untouched val.

Workflow:
1. Split the training MKTD panel into a prefix train section and a calibration
   tail.
2. Train candidate XGB rankers on the prefix and score the calibration tail.
3. Rank symbols by out-of-sample forecastability on that calibration tail.
4. Retrain on full raw or augmented training data restricted to top-K symbols.
5. Evaluate the restricted policy on the untouched validation panel.

The point is to test the "drop hard-to-forecast pairs" hypothesis without
looking at validation performance when choosing the universe.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, read_mktd
from scripts.sweep_binance33_xgb import (
    Experiment,
    _build_dataset,
    _eval_model,
    _experiments,
    _feature_matrix,
    _precompute_scores,
    _train_xgb,
    _passes_production_target,
)


def _parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError(f"expected at least one integer in {value!r}")
    return items


def _parse_float_list(value: str) -> list[float]:
    items = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError(f"expected at least one float in {value!r}")
    return items


def _slice_data(data: MktdData, start: int, end: int) -> MktdData:
    start = max(0, int(start))
    end = min(int(end), data.num_timesteps)
    if end <= start:
        raise ValueError(f"empty data slice start={start} end={end}")
    return MktdData(
        version=int(data.version),
        symbols=list(data.symbols),
        features=data.features[start:end].copy(),
        prices=data.prices[start:end].copy(),
        tradable=data.tradable[start:end].copy() if data.tradable is not None else None,
    )


def _masked_data(data: MktdData, symbols: Sequence[str]) -> MktdData:
    allowed = {str(sym).upper() for sym in symbols}
    mask = np.asarray([sym.upper() in allowed for sym in data.symbols], dtype=np.uint8)
    tradable = np.ones((data.num_timesteps, data.num_symbols), dtype=np.uint8)
    if data.tradable is not None:
        tradable = np.asarray(data.tradable, dtype=np.uint8).copy()
    tradable &= mask[None, :]
    return MktdData(
        version=int(data.version),
        symbols=list(data.symbols),
        features=data.features,
        prices=data.prices,
        tradable=tradable,
    )


def _target_matrix(data: MktdData, *, horizon: int, label: str) -> np.ndarray:
    close = np.asarray(data.prices[:, :, P_CLOSE], dtype=np.float64)
    out = np.full((data.num_timesteps, data.num_symbols), np.nan, dtype=np.float64)
    h = int(horizon)
    for t in range(0, data.num_timesteps - h):
        p0 = close[t]
        p1 = close[t + h]
        valid = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
        if data.tradable is not None:
            valid &= np.asarray(data.tradable[t], dtype=bool)
            valid &= np.asarray(data.tradable[t + h], dtype=bool)
        ret = np.full(data.num_symbols, np.nan, dtype=np.float64)
        ret[valid] = p1[valid] / p0[valid] - 1.0
        if label == "rank":
            ranked = np.full(data.num_symbols, np.nan, dtype=np.float64)
            if valid.sum() > 1:
                order = np.flatnonzero(valid)[np.argsort(ret[valid])]
                ranked[order] = np.linspace(-1.0, 1.0, int(valid.sum()), dtype=np.float64)
            ret = ranked
        elif label == "voladj":
            vol = np.maximum(np.asarray(data.features[t, :, 4], dtype=np.float64), 0.01)
            ret = ret / vol
        out[t] = ret
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 20:
        return 0.0
    xv = x[valid]
    yv = y[valid]
    if float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def _symbol_scores(data: MktdData, scores_by_t: np.ndarray, exp: Experiment) -> list[tuple[str, float, float, float]]:
    targets = _target_matrix(data, horizon=int(exp.horizon), label=str(exp.label))
    out: list[tuple[str, float, float, float]] = []
    for idx, symbol in enumerate(data.symbols):
        pred = np.asarray(scores_by_t[:, idx], dtype=np.float64)
        target = np.asarray(targets[:, idx], dtype=np.float64)
        ic = _safe_corr(pred, target)
        valid = np.isfinite(pred) & np.isfinite(target)
        if int(valid.sum()) <= 0:
            sign_acc = 0.0
            spread = 0.0
        else:
            sign_acc = float(np.mean(np.sign(pred[valid]) == np.sign(target[valid])))
            q_hi = np.nanpercentile(pred[valid], 75)
            q_lo = np.nanpercentile(pred[valid], 25)
            spread = float(np.nanmean(target[valid & (pred >= q_hi)]) - np.nanmean(target[valid & (pred <= q_lo)]))
        score = 100.0 * ic + 10.0 * spread + 5.0 * (sign_acc - 0.5)
        out.append((symbol, float(score), float(ic), float(sign_acc)))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def _experiment_by_name(names: str) -> list[Experiment]:
    requested = [part.strip() for part in str(names).split(",") if part.strip()]
    by_name = {exp.name: exp for exp in _experiments()}
    if not requested:
        return list(by_name.values())
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError(f"unknown experiments: {', '.join(missing)}")
    return [by_name[name] for name in requested]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_train.bin"))
    parser.add_argument("--aug-train-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_aug_train.bin"))
    parser.add_argument("--eval-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_easy_universe.csv"))
    parser.add_argument("--calibration-days", type=int, default=360)
    parser.add_argument("--experiment-names", default="xgb08,xgb11,xgb14,xgb17")
    parser.add_argument("--top-k-grid", default="3,5,8,12,16,24,33")
    parser.add_argument("--train-variants", default="raw,aug")
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-days", default="30,120")
    parser.add_argument("--slippage-bps", default="20")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-leverage", default="1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--target-monthly-pct", type=float, default=27.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    parser.add_argument("--require-production-target", action="store_true")
    args = parser.parse_args()

    raw_train = read_mktd(args.train_data)
    aug_train = read_mktd(args.aug_train_data) if str(args.aug_train_data) else raw_train
    eval_data = read_mktd(args.eval_data)
    cal_days = min(max(30, int(args.calibration_days)), raw_train.num_timesteps - 30)
    prefix = _slice_data(raw_train, 0, raw_train.num_timesteps - cal_days)
    cal = _slice_data(raw_train, raw_train.num_timesteps - cal_days, raw_train.num_timesteps)
    eval_days = _parse_int_list(args.eval_days)
    slippages = _parse_float_list(args.slippage_bps)
    leverages = _parse_float_list(args.max_leverage)
    top_ks = _parse_int_list(args.top_k_grid)
    train_variants = [part.strip().lower() for part in str(args.train_variants).split(",") if part.strip()]

    fieldnames = [
        "experiment",
        "horizon",
        "label",
        "mode",
        "size_mode",
        "level_mode",
        "rebalance_days",
        "action_allocation_bins",
        "action_level_bins",
        "action_max_offset_bps",
        "score_scale",
        "target_vol",
        "train_variant",
        "top_k",
        "selected_symbols",
        "calibration_score_mean",
        "calibration_ic_mean",
        "calibration_sign_acc_mean",
        "max_leverage",
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
    rows: list[dict[str, object]] = []
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for exp in _experiment_by_name(args.experiment_names):
            x_prefix, y_prefix = _build_dataset(prefix, horizon=exp.horizon, label=exp.label)
            cal_model = _train_xgb(x_prefix, y_prefix, exp, rounds=int(args.rounds), device=str(args.device))
            cal_scores = _precompute_scores(cal, cal_model)
            ranked_symbols = _symbol_scores(cal, cal_scores, exp)
            for top_k in top_ks:
                selected = [symbol for symbol, _score, _ic, _acc in ranked_symbols[: min(int(top_k), len(ranked_symbols))]]
                selected_rows = ranked_symbols[: min(int(top_k), len(ranked_symbols))]
                score_mean = float(np.mean([item[1] for item in selected_rows])) if selected_rows else 0.0
                ic_mean = float(np.mean([item[2] for item in selected_rows])) if selected_rows else 0.0
                sign_acc_mean = float(np.mean([item[3] for item in selected_rows])) if selected_rows else 0.0
                for train_variant in train_variants:
                    source_train = aug_train if train_variant == "aug" else raw_train
                    masked_train = _masked_data(source_train, selected)
                    x_train, y_train = _build_dataset(masked_train, horizon=exp.horizon, label=exp.label)
                    final_exp = replace(exp, name=f"{exp.name}_{train_variant}_top{top_k}")
                    model = _train_xgb(x_train, y_train, final_exp, rounds=int(args.rounds), device=str(args.device))
                    eval_scores = _precompute_scores(eval_data, model)
                    disallowed = np.asarray([symbol.upper() not in {s.upper() for s in selected} for symbol in eval_data.symbols])
                    eval_scores[:, disallowed] = np.nan
                    for leverage in leverages:
                        for days in eval_days:
                            for slip in slippages:
                                row = _eval_model(
                                    eval_data,
                                    model,
                                    final_exp,
                                    eval_days=int(days),
                                    slippage_bps=float(slip),
                                    stride=int(args.stride),
                                    max_leverage=float(leverage),
                                    fee_rate=float(args.fee_rate),
                                    fill_buffer_bps=float(args.fill_buffer_bps),
                                    decision_lag=int(args.decision_lag),
                                    scores_by_t=eval_scores,
                                )
                                row["experiment"] = exp.name
                                row["train_variant"] = train_variant
                                row["top_k"] = int(top_k)
                                row["selected_symbols"] = ",".join(selected)
                                row["calibration_score_mean"] = score_mean
                                row["calibration_ic_mean"] = ic_mean
                                row["calibration_sign_acc_mean"] = sign_acc_mean
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
            print(f"evaluated {exp.name}", flush=True)

    promotion_days = max(eval_days)
    promotion_slip = max(slippages)
    subset = [
        row
        for row in rows
        if int(row["eval_days"]) == int(promotion_days)
        and float(row["slip_bps"]) == float(promotion_slip)
    ]
    subset.sort(key=lambda row: float(row["median_monthly_pct"]), reverse=True)
    print(f"\n=== Best easy-universe XGB {promotion_days}d slip{promotion_slip:g} ===")
    for row in subset[:20]:
        print(
            f"{row['median_monthly_pct']:+7.2f}%/mo p10/mo={row['p10_monthly_pct']:+7.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} dd90={row['p90_dd_pct']:.2f}% "
            f"pass={row['passes_target']} {row['experiment']} {row['train_variant']} "
            f"top={row['top_k']} lev={row['max_leverage']} syms={row['selected_symbols']}"
        )
    promoted = [row for row in subset if int(row["passes_target"]) == 1]
    print(f"\n=== Production target candidates {promotion_days}d slip{promotion_slip:g} ===")
    if promoted:
        for row in promoted[:10]:
            print(
                f"{row['median_monthly_pct']:+7.2f}%/mo p10/mo={row['p10_monthly_pct']:+7.2f}% "
                f"dd90={row['p90_dd_pct']:.2f}% {row['experiment']} {row['train_variant']} top={row['top_k']}"
            )
    else:
        print("none")
    print(f"\nwrote {args.out}")
    if args.require_production_target and not promoted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
