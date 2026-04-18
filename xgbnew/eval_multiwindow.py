#!/usr/bin/env python3
"""Realistic multi-window evaluation for the XGBoost daily strategy.

Trains XGBoost on a fixed train span, then evaluates rolling out-of-sample
windows with realistic daily execution assumptions:
  - explicit entry/exit fill buffer around the trade bar
  - shared stock fee defaults from the workspace
  - rolling 120-day windows by default

Also reports:
  - Chronos open-to-close MAE on the same OOS rows when cache data is available
  - XGBoost calibrated return MAE on the same OOS rows
  - an optional small hyperparameter sweep over XGBoost settings
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loss_utils import TRADING_FEE
from src.forecast_cache_metrics import compute_mae_percent
from xgbnew.backtest import BacktestConfig, simulate
from xgbnew.dataset import build_daily_dataset, load_chronos_cache
from xgbnew.features import DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_MONTH = 21.0


@dataclass(frozen=True)
class SweepConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    top_n: int
    xgb_weight: float
    leverage: float = 1.0
    random_state: int = 42


def _monthly_return(total_ret: float, n_days: int) -> float:
    try:
        return (1.0 + total_ret / 100.0) ** (TRADING_DAYS_PER_MONTH / n_days) - 1.0
    except Exception:
        return 0.0


def _load_symbols(path: Path) -> list[str]:
    syms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().split("#", 1)[0].strip().upper()
        if s:
            syms.append(s)
    return syms


def _parse_int_grid(raw: str | None, default: int) -> list[int]:
    text = (raw or "").strip()
    if not text:
        return [int(default)]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_grid(raw: str | None, default: float) -> list[float]:
    text = (raw or "").strip()
    if not text:
        return [float(default)]
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _fit_linear_return_calibration(prob_up: pd.Series, actual_return: pd.Series) -> tuple[float, float]:
    x = prob_up.to_numpy(dtype=np.float64, copy=False)
    y = actual_return.to_numpy(dtype=np.float64, copy=False)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        baseline = float(np.nanmean(y[mask])) if bool(mask.any()) else 0.0
        return baseline, 0.0
    x_fit = x[mask]
    y_fit = y[mask]
    if np.allclose(x_fit, x_fit[0]):
        return float(np.mean(y_fit)), 0.0
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    return float(intercept), float(slope)


def _predict_calibrated_returns(prob_up: pd.Series, calibration: tuple[float, float]) -> pd.Series:
    intercept, slope = calibration
    pred = intercept + slope * prob_up.to_numpy(dtype=np.float64, copy=False)
    return pd.Series(np.clip(pred, -0.5, 0.5), index=prob_up.index, name="pred_return_xgb")


def _compute_xgb_mae(
    model: XGBStockModel,
    calibration: tuple[float, float],
    df: pd.DataFrame,
) -> dict[str, float]:
    if df.empty:
        return {"count": 0, "return_mae_pct_points": 0.0, "return_mae_percent": 0.0, "direction_mae": 0.0}
    prob_up = model.predict_scores(df)
    pred_ret = _predict_calibrated_returns(prob_up, calibration)
    actual_ret = df["target_oc"].astype(float)
    mae_return_pct_points = float(np.mean(np.abs((pred_ret - actual_ret) * 100.0)))
    direction_mae = float(np.mean(np.abs(prob_up.to_numpy(dtype=np.float64) - df["target_oc_up"].to_numpy(dtype=np.float64))))
    return {
        "count": int(len(df)),
        "return_mae_pct_points": mae_return_pct_points,
        "return_mae_percent": float(compute_mae_percent(mae_return_pct_points, actual_ret.to_numpy(dtype=np.float64) * 100.0)),
        "direction_mae": direction_mae,
    }


def _compute_chronos_mae(df: pd.DataFrame) -> dict[str, float] | None:
    if df.empty or "chronos_available" not in df.columns or "chronos_oc_return" not in df.columns:
        return None
    mask = (
        df["chronos_available"].astype(float) > 0.0
    ) & df["chronos_oc_return"].notna() & df["target_oc"].notna()
    if not bool(mask.any()):
        return None
    pred = df.loc[mask, "chronos_oc_return"].astype(float).to_numpy(dtype=np.float64, copy=False)
    actual = (df.loc[mask, "target_oc"].astype(float).to_numpy(dtype=np.float64, copy=False) * 100.0)
    mae = float(np.mean(np.abs(pred - actual)))
    return {
        "count": int(mask.sum()),
        "return_mae_pct_points": mae,
        "return_mae_percent": float(compute_mae_percent(mae, actual)),
    }


def _build_windows(all_trading_days: list[date], *, window_days: int, stride_days: int) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    idx = 0
    while idx + window_days <= len(all_trading_days):
        span = all_trading_days[idx : idx + window_days]
        windows.append((span[0], span[-1]))
        idx += stride_days
    return windows


def _combined_scores_from_predictions(
    df: pd.DataFrame,
    xgb_scores: pd.Series,
    *,
    xgb_weight: float,
    chronos_col: str = "chronos_oc_return",
) -> pd.Series:
    if chronos_col in df.columns and (df[chronos_col] != 0).any():
        chron_vals = df[chronos_col].fillna(0.0)
        if "date" in df.columns:
            chron_norm = chron_vals.groupby(df["date"]).rank(pct=True)
        else:
            chron_norm = chron_vals.rank(pct=True)
        chron_norm = chron_norm.fillna(0.5)
    else:
        chron_norm = pd.Series(0.5, index=df.index)
    combined = xgb_weight * xgb_scores.reindex(df.index).astype(float) + (1.0 - xgb_weight) * chron_norm
    return combined.rename("combined_score")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path, default=REPO / "symbol_lists/stocks_wide_1000_v1.txt")
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--chronos-cache", type=Path, default=REPO / "analysis/top2_backtest/forecast_cache")
    p.add_argument("--train-start", default="2021-01-01")
    p.add_argument("--train-end", default="2023-12-31")
    p.add_argument("--oos-start", default="2024-01-02")
    p.add_argument("--oos-end", default="")
    p.add_argument("--window-days", type=int, default=120)
    p.add_argument("--stride-days", type=int, default=21)
    p.add_argument("--top-n", type=int, default=2)
    p.add_argument("--top-n-grid", default="")
    p.add_argument("--xgb-weight", type=float, default=1.0)
    p.add_argument("--xgb-weight-grid", default="")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--leverage-grid", default="")
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--fee-rate", type=float, default=float(TRADING_FEE))
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--min-dollar-vol", type=float, default=5e6)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-estimators-grid", default="")
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--max-depth-grid", default="")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--learning-rate-grid", default="")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--random-state-grid",
        default="",
        help="Comma-separated seeds to sweep XGB random_state across (Bonferroni / stability check).",
    )
    p.add_argument("--output-dir", type=Path, default=REPO / "analysis/xgbnew_multiwindow")
    p.add_argument("--model-save-path", type=Path, default=None)
    p.add_argument(
        "--device",
        default=None,
        help="XGBoost device (e.g. 'cuda'). Omit for CPU. Requires xgboost built with USE_CUDA.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _config_grid(args: argparse.Namespace) -> list[SweepConfig]:
    configs = []
    for n_estimators, max_depth, learning_rate, top_n, xgb_weight, leverage, random_state in itertools.product(
        _parse_int_grid(args.n_estimators_grid, args.n_estimators),
        _parse_int_grid(args.max_depth_grid, args.max_depth),
        _parse_float_grid(args.learning_rate_grid, args.learning_rate),
        _parse_int_grid(args.top_n_grid, args.top_n),
        _parse_float_grid(args.xgb_weight_grid, args.xgb_weight),
        _parse_float_grid(args.leverage_grid, args.leverage),
        _parse_int_grid(args.random_state_grid, args.random_state),
    ):
        configs.append(
            SweepConfig(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=float(learning_rate),
                top_n=int(top_n),
                xgb_weight=float(xgb_weight),
                leverage=float(leverage),
                random_state=int(random_state),
            )
        )
    return configs


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    symbols = _load_symbols(args.symbols_file)
    oos_start = date.fromisoformat(args.oos_start)
    oos_end_str = args.oos_end or date.today().isoformat()
    oos_end = date.fromisoformat(oos_end_str)

    chronos_cache = {}
    if args.chronos_cache.exists():
        chronos_cache = load_chronos_cache(args.chronos_cache)

    print(
        f"[xgb-eval] {len(symbols)} symbols | train {args.train_start}–{args.train_end} | "
        f"OOS {args.oos_start}–{oos_end_str} | windows={args.window_days}d stride={args.stride_days}d",
        flush=True,
    )
    print(
        f"[xgb-eval] realistic costs: fee_rate={float(args.fee_rate) * 10000:.3f}bps "
        f"fill_buffer={float(args.fill_buffer_bps):.2f}bps commission_bps={float(args.commission_bps):.2f}",
        flush=True,
    )

    t0 = time.perf_counter()
    train_df, _, oos_df = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        val_start=oos_start,
        val_end=oos_end,
        test_start=oos_start,
        test_end=oos_end,
        chronos_cache=chronos_cache if chronos_cache else None,
        min_dollar_vol=args.min_dollar_vol,
    )
    print(f"[xgb-eval] dataset built in {time.perf_counter()-t0:.1f}s | train={len(train_df):,} oos={len(oos_df):,}", flush=True)
    requested_symbol_count = len(symbols)
    train_symbol_count = int(train_df["symbol"].nunique()) if "symbol" in train_df.columns else 0
    oos_symbol_count = int(oos_df["symbol"].nunique()) if "symbol" in oos_df.columns else 0
    missing_train_symbols = max(requested_symbol_count - train_symbol_count, 0)
    missing_oos_symbols = max(requested_symbol_count - oos_symbol_count, 0)
    print(
        f"[xgb-eval] coverage: requested={requested_symbol_count} "
        f"train_symbols={train_symbol_count} oos_symbols={oos_symbol_count} "
        f"missing_train={missing_train_symbols} missing_oos={missing_oos_symbols}",
        flush=True,
    )

    if len(train_df) < 1000:
        print("ERROR: Too few training rows.", file=sys.stderr)
        return 1
    if len(oos_df) < 100:
        print("ERROR: No OOS data found.", file=sys.stderr)
        return 1

    all_trading_days = sorted(oos_df["date"].unique())
    windows = _build_windows(all_trading_days, window_days=int(args.window_days), stride_days=int(args.stride_days))
    if not windows:
        print("ERROR: No OOS windows could be built.", file=sys.stderr)
        return 1

    chronos_mae = _compute_chronos_mae(oos_df)

    model_cache: dict[
        tuple[int, int, float, int],
        tuple[XGBStockModel, tuple[float, float], dict[str, float], pd.Series]
    ] = {}
    sweep_results: list[dict[str, object]] = []
    configs = _config_grid(args)

    print(f"[xgb-eval] evaluating {len(configs)} config(s)", flush=True)
    for idx, cfg in enumerate(configs, start=1):
        model_key = (cfg.n_estimators, cfg.max_depth, cfg.learning_rate, cfg.random_state)
        if model_key not in model_cache:
            print(
                f"[xgb-eval] train model {idx}/{len(configs)} "
                f"(n_estimators={cfg.n_estimators}, max_depth={cfg.max_depth}, "
                f"lr={cfg.learning_rate}, seed={cfg.random_state})",
                flush=True,
            )
            model = XGBStockModel(
                device=args.device,
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                random_state=cfg.random_state,
            )
            model.fit(train_df, DAILY_FEATURE_COLS, verbose=args.verbose)
            train_prob = model.predict_scores(train_df)
            calibration = _fit_linear_return_calibration(train_prob, train_df["target_oc"])
            oos_prob = model.predict_scores(oos_df)
            xgb_mae = _compute_xgb_mae(model, calibration, oos_df)
            model_cache[model_key] = (model, calibration, xgb_mae, oos_prob)
            if args.model_save_path and len(model_cache) == 1:
                args.model_save_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(args.model_save_path)
        model, calibration, xgb_mae, oos_prob = model_cache[model_key]

        backtest_cfg = BacktestConfig(
            top_n=cfg.top_n,
            leverage=cfg.leverage,
            xgb_weight=cfg.xgb_weight,
            commission_bps=float(args.commission_bps),
            min_dollar_vol=args.min_dollar_vol,
            fee_rate=float(args.fee_rate),
            fill_buffer_bps=float(args.fill_buffer_bps),
        )
        combined_scores = _combined_scores_from_predictions(
            oos_df,
            oos_prob,
            xgb_weight=cfg.xgb_weight,
            chronos_col=backtest_cfg.chronos_col,
        )

        window_results = []
        for w_start, w_end in windows:
            w_df = oos_df[(oos_df["date"] >= w_start) & (oos_df["date"] <= w_end)]
            if len(w_df) < 5:
                continue
            w_scores = combined_scores.loc[w_df.index]
            result = simulate(w_df, model, backtest_cfg, precomputed_scores=w_scores)
            n_days = len(result.day_results)
            monthly = _monthly_return(result.total_return_pct, max(n_days, 1)) * 100.0
            window_results.append(
                {
                    "w_start": str(w_start),
                    "w_end": str(w_end),
                    "n_trading_days": n_days,
                    "total_return_pct": result.total_return_pct,
                    "monthly_return_pct": monthly,
                    "sharpe": result.sharpe_ratio,
                    "sortino": result.sortino_ratio,
                    "max_dd_pct": result.max_drawdown_pct,
                    "win_rate_pct": result.win_rate_pct,
                    "dir_acc_pct": result.directional_accuracy_pct,
                    "total_trades": result.total_trades,
                    "avg_fee_bps": result.avg_fee_bps,
                    "avg_spread_bps": result.avg_spread_bps,
                }
            )

        if not window_results:
            continue

        monthly_rets = np.array([r["monthly_return_pct"] for r in window_results], dtype=np.float64)
        total_rets = np.array([r["total_return_pct"] for r in window_results], dtype=np.float64)
        sortinos = np.array([r["sortino"] for r in window_results], dtype=np.float64)
        n_neg = int(np.sum(monthly_rets < 0.0))
        sweep_results.append(
            {
                "config": {
                    "n_estimators": cfg.n_estimators,
                    "max_depth": cfg.max_depth,
                    "learning_rate": cfg.learning_rate,
                    "top_n": cfg.top_n,
                    "xgb_weight": cfg.xgb_weight,
                    "leverage": cfg.leverage,
                    "random_state": cfg.random_state,
                },
                "median_monthly_pct": float(np.median(monthly_rets)),
                "p10_monthly_pct": float(np.percentile(monthly_rets, 10)),
                "p90_monthly_pct": float(np.percentile(monthly_rets, 90)),
                "mean_monthly_pct": float(np.mean(monthly_rets)),
                "median_total_ret_pct": float(np.median(total_rets)),
                "median_sortino": float(np.median(sortinos)),
                "n_windows": int(len(window_results)),
                "n_neg_monthly": n_neg,
                "xgb_mae": xgb_mae,
                "chronos_mae": chronos_mae,
                "windows": window_results,
            }
        )

    if not sweep_results:
        print("ERROR: No window results.", file=sys.stderr)
        return 1

    sweep_results.sort(
        key=lambda item: (
            float(item["median_monthly_pct"]),
            float(item["p10_monthly_pct"]),
            float(item["median_sortino"]),
        ),
        reverse=True,
    )
    best = sweep_results[0]

    print(f"\n{'='*84}")
    print("  XGBoost 120d Realistic Eval")
    print(f"  OOS: {args.oos_start} → {oos_end_str} | windows={best['n_windows']} | "
          f"fee={float(args.fee_rate) * 10000:.3f}bps | fill_buffer={float(args.fill_buffer_bps):.2f}bps")
    print(f"{'='*84}")
    if chronos_mae:
        print(
            f"  Chronos OOS MAE   : {chronos_mae['return_mae_pct_points']:.3f} pct-pts "
            f"({chronos_mae['return_mae_percent']:.2f}% scaled, n={chronos_mae['count']})"
        )
    xgb_mae = best["xgb_mae"]
    if isinstance(xgb_mae, dict):
        print(
            f"  XGB OOS MAE       : {float(xgb_mae['return_mae_pct_points']):.3f} pct-pts "
            f"({float(xgb_mae['return_mae_percent']):.2f}% scaled, dir_mae={float(xgb_mae['direction_mae']):.4f}, "
            f"n={int(xgb_mae['count'])})"
        )
    print(f"\n  Best config       : {best['config']}")
    print(f"  Median monthly%   : {float(best['median_monthly_pct']):+.2f}%")
    print(f"  P10 monthly%      : {float(best['p10_monthly_pct']):+.2f}%")
    print(f"  Median sortino    : {float(best['median_sortino']):.2f}")
    print(f"  Neg windows       : {int(best['n_neg_monthly'])}/{int(best['n_windows'])}")

    if len(sweep_results) > 1:
        print("\n  Sweep ranking:")
        print("  " + "-" * 84)
        for row in sweep_results[:10]:
            print(
                f"  cfg={row['config']} "
                f"median_monthly={float(row['median_monthly_pct']):+7.2f}% "
                f"p10={float(row['p10_monthly_pct']):+7.2f}% "
                f"sortino={float(row['median_sortino']):5.2f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "train_start": args.train_start,
        "train_end": args.train_end,
        "oos_start": args.oos_start,
        "oos_end": oos_end_str,
        "window_days": int(args.window_days),
        "stride_days": int(args.stride_days),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "fee_rate": float(args.fee_rate),
        "commission_bps": float(args.commission_bps),
        "coverage": {
            "requested_symbol_count": requested_symbol_count,
            "train_symbol_count": train_symbol_count,
            "oos_symbol_count": oos_symbol_count,
            "missing_train_symbols": missing_train_symbols,
            "missing_oos_symbols": missing_oos_symbols,
        },
        "chronos_mae": chronos_mae,
        "median_monthly_pct": float(best["median_monthly_pct"]),
        "p10_monthly_pct": float(best["p10_monthly_pct"]),
        "median_sortino": float(best["median_sortino"]),
        "windows": best["windows"],
        "best": best,
        "sweep_results": sweep_results,
    }
    out_path = args.output_dir / f"multiwindow_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
