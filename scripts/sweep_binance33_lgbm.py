#!/usr/bin/env python3
"""Train/evaluate Binance33 LightGBM cross-sectional policies."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd
from scripts.sweep_binance33_xgb import (
    Experiment,
    _build_dataset,
    _eval_model,
    _experiments,
    _feature_matrix,
    _parse_float_list,
    _parse_str_list,
    _passes_production_target,
)


def _train_lgbm(x_train: np.ndarray, y_train: np.ndarray, exp: Experiment, *, rounds: int, device: str):
    import lightgbm as lgb

    dtrain = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": float(exp.eta),
        "num_leaves": max(7, min(255, 2 ** int(exp.max_depth))),
        "max_depth": int(exp.max_depth),
        "feature_fraction": float(exp.colsample),
        "bagging_fraction": float(exp.subsample),
        "bagging_freq": 1,
        "min_data_in_leaf": max(5, int(round(float(exp.min_child_weight)))),
        "lambda_l2": float(exp.reg_lambda),
        "verbosity": -1,
        "seed": 1337,
        "force_col_wise": True,
    }
    if str(device).lower() in {"gpu", "cuda"}:
        params["device_type"] = "gpu"
    try:
        return lgb.train(params, dtrain, num_boost_round=int(rounds))
    except Exception:
        params.pop("device_type", None)
        return lgb.train(params, dtrain, num_boost_round=int(rounds))


def _precompute_lgbm_scores(data, model) -> np.ndarray:
    rows = []
    for t in range(data.num_timesteps):
        scores = model.predict(_feature_matrix(data, t))
        rows.append(np.asarray(scores, dtype=np.float64))
    return np.vstack(rows).astype(np.float64, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Binance33 LightGBM cross-sectional policies.")
    parser.add_argument("--train-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_train.bin"))
    parser.add_argument("--eval-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_lgbm_sweep.csv"))
    parser.add_argument("--rounds", type=int, default=180)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-names", default="xgb08,xgb11,xgb14,xgb17")
    parser.add_argument("--eval-days", default="120")
    parser.add_argument("--slippage-bps", default="20")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-leverage", default="1.0,1.25,1.5")
    parser.add_argument("--size-modes", default="full")
    parser.add_argument("--level-modes", default="mid")
    parser.add_argument("--allocation-bins", type=int, default=1)
    parser.add_argument("--level-bins", type=int, default=1)
    parser.add_argument("--max-offset-bps", type=float, default=0.0)
    parser.add_argument("--score-scale", type=float, default=0.05)
    parser.add_argument("--target-vol", type=float, default=0.05)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--target-monthly-pct", type=float, default=27.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    args = parser.parse_args()

    train_data = read_mktd(args.train_data)
    eval_data = read_mktd(args.eval_data)
    by_name = {exp.name: exp for exp in _experiments()}
    names = [part.strip() for part in str(args.experiment_names).split(",") if part.strip()]
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"unknown experiment names: {', '.join(missing)}")
    size_modes = _parse_str_list(args.size_modes, default=["full"])
    level_modes = _parse_str_list(args.level_modes, default=["mid"])
    experiments: list[Experiment] = []
    for base in [by_name[name] for name in names]:
        for size_mode in size_modes:
            for level_mode in level_modes:
                suffix = "_lgbm"
                if size_mode != "full":
                    suffix += f"_{size_mode}"
                if level_mode != "mid":
                    suffix += f"_{level_mode}"
                experiments.append(
                    replace(
                        base,
                        name=f"{base.name}{suffix}",
                        size_mode=size_mode,
                        level_mode=level_mode,
                        score_scale=float(args.score_scale),
                        target_vol=float(args.target_vol),
                    )
                )

    eval_days = [int(part.strip()) for part in str(args.eval_days).split(",") if part.strip()]
    slippages = [float(part.strip()) for part in str(args.slippage_bps).split(",") if part.strip()]
    leverages = _parse_float_list(args.max_leverage)
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
    rows: list[dict[str, float | int | str]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, exp in enumerate(experiments, start=1):
            x_train, y_train = _build_dataset(train_data, horizon=exp.horizon, label=exp.label)
            model = _train_lgbm(x_train, y_train, exp, rounds=int(args.rounds), device=str(args.device))
            scores_by_t = _precompute_lgbm_scores(eval_data, model)
            for days in eval_days:
                for slip in slippages:
                    for leverage in leverages:
                        row = _eval_model(
                            eval_data,
                            None,
                            exp,
                            eval_days=int(days),
                            slippage_bps=float(slip),
                            stride=int(args.stride),
                            max_leverage=float(leverage),
                            fee_rate=float(args.fee_rate),
                            fill_buffer_bps=float(args.fill_buffer_bps),
                            decision_lag=int(args.decision_lag),
                            scores_by_t=scores_by_t,
                            action_allocation_bins=int(args.allocation_bins),
                            action_level_bins=int(args.level_bins),
                            action_max_offset_bps=float(args.max_offset_bps),
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
                        writer.writerow(row)
                        rows.append(row)
            print(f"evaluated {idx}/{len(experiments)} {exp.name}")

    best = sorted(
        rows,
        key=lambda row: (
            int(row["passes_target"]),
            float(row["median_monthly_pct"]) - max(0.0, float(row["p90_dd_pct"]) - float(args.target_max_dd_pct)),
        ),
        reverse=True,
    )[:20]
    print("\n=== Best LightGBM rows ===")
    for row in best:
        print(
            f"{row['median_monthly_pct']:+7.2f}%/mo p10={row['p10_monthly_pct']:+7.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} dd90={row['p90_dd_pct']:.2f}% "
            f"trades={row['median_trades']:.0f} pass={row['passes_target']} "
            f"lev={row['max_leverage']} {row['experiment']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
