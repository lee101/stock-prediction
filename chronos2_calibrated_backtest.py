#!/usr/bin/env python3
"""
Calibrated backtest for Chronos2 models.

Evaluates a trained + calibrated Chronos2 model on held-out test data using
its buy/sell thresholds (calibration.json). Reports:
  - Annualized Sharpe
  - Win rate
  - Trade count, hold rate
  - Per-symbol breakdown (if --per-symbol)
  - Comparison vs buy-and-hold

This is the correct way to assess model quality after calibration, as opposed
to just looking at MAE (which doesn't account for trading economics).

Usage:
    python chronos2_calibrated_backtest.py \\
        --model-id chronos2_finetuned/stocks_all_v2/finetuned-ckpt \\
        --cal-data-dir trainingdata \\
        --fee-bps 10 \\
        --per-symbol

    # Compare two models:
    python chronos2_calibrated_backtest.py \\
        --model-id chronos2_finetuned/stocks_all_v3/finetuned-ckpt \\
        --compare-model chronos2_finetuned/stocks_all_v2/finetuned-ckpt \\
        --cal-data-dir trainingdata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chronos2_linear_calibration import (
    CalibrationParams,
    collect_predictions,
    compute_sharpe,
)
from chronos2_stock_augmentation import load_all_series


# ---------------------------------------------------------------------------
# Backtest stats
# ---------------------------------------------------------------------------

def run_backtest(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    actual: np.ndarray,
    prev_close: np.ndarray,
    symbols: List[str],
    params: CalibrationParams,
    fee_bps: float = 10.0,
) -> Dict:
    """
    Run calibrated backtest on collected predictions.

    Returns dict with overall and per-symbol stats.
    """
    eps = 1e-8
    pred_ret = (q50 - prev_close) / (np.abs(prev_close) + eps)
    act_ret  = (actual - prev_close) / (np.abs(prev_close) + eps)
    unc      = (q90 - q10) / (np.abs(prev_close) + eps)
    skewness = ((q90 - q50) - (q50 - q10)) / (np.abs(prev_close) + eps)

    # Signal: weighted median return + optional skewness component
    signals = pred_ret * params.signal_weight + skewness * params.skew_weight + params.signal_bias
    uncertainties = unc

    fee = fee_bps / 10_000.0
    pnl = []
    positions = []
    position = 0
    n_trades = 0
    conf_thresh = params.confidence_threshold

    for i in range(len(signals)):
        sig = signals[i]
        ret = act_ret[i]
        unc_i = uncertainties[i]

        # Confidence filter
        if conf_thresh > 0.0 and unc_i > conf_thresh:
            desired = 0
        elif sig > params.buy_threshold:
            desired = 1
        elif params.allow_short and sig < -params.sell_threshold:
            desired = -1
        else:
            desired = 0

        transition_cost = fee if desired != position else 0.0
        if desired != position:
            n_trades += 1

        if position == 1:
            pnl.append(ret - transition_cost)
        elif position == -1:
            pnl.append(-ret - transition_cost)
        else:
            pnl.append(-transition_cost)

        positions.append(position)
        position = desired

    arr = np.array(pnl)
    pos_arr = np.array(positions)
    n = len(arr)

    # Buy-and-hold Sharpe
    bnh_sharpe = float(act_ret.mean() / (act_ret.std() + 1e-10) * np.sqrt(252))

    stats: Dict = {
        "n_windows": n,
        "n_trades": n_trades,
        "hold_rate": float((pos_arr == 0).mean()),
        "long_rate": float((pos_arr == 1).mean()),
        "short_rate": float((pos_arr == -1).mean()),
        "mean_daily_pnl_bps": float(arr.mean() * 10000),
        "sharpe_annualized": float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252)),
        "bnh_sharpe_annualized": bnh_sharpe,
        "win_rate": float((arr[arr != 0] > 0).mean()) if (arr != 0).any() else float("nan"),
        "pnl_30d_est_pct": float(arr.mean() * 30 * 100),  # rough monthly estimate
        "calibration": params.to_dict(),
    }

    # Per-symbol breakdown
    if any(s for s in symbols):
        syms_arr = np.array(symbols)
        per_sym = {}
        for sym in sorted(set(symbols)):
            if not sym:
                continue
            mask = syms_arr == sym
            if mask.sum() < 10:
                continue
            sub = arr[mask]
            sub_act = act_ret[mask]
            per_sym[sym] = {
                "n": int(mask.sum()),
                "sharpe": float(sub.mean() / (sub.std() + 1e-10) * np.sqrt(252)),
                "bnh_sharpe": float(sub_act.mean() / (sub_act.std() + 1e-10) * np.sqrt(252)),
                "mean_daily_pnl_bps": float(sub.mean() * 10000),
                "win_rate": float((sub[sub != 0] > 0).mean()) if (sub != 0).any() else float("nan"),
            }
        stats["per_symbol"] = per_sym

    return stats


def _print_stats(stats: Dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Sharpe={stats['sharpe_annualized']:.3f}  "
          f"(BnH={stats['bnh_sharpe_annualized']:.3f})  "
          f"trades={stats['n_trades']}  hold={stats['hold_rate']*100:.1f}%  "
          f"win_rate={stats.get('win_rate', float('nan'))*100:.1f}%  "
          f"monthly≈{stats['pnl_30d_est_pct']:.2f}%  "
          f"n={stats['n_windows']}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrated backtest for Chronos2")
    p.add_argument("--model-id",    required=True)
    p.add_argument("--cal-data-dir", default="trainingdata")
    p.add_argument("--calibration",  default=None,
                   help="Path to calibration JSON (default: <model-id>/calibration.json)")
    p.add_argument("--compare-model", default=None,
                   help="Second model to compare against (uses its own calibration.json)")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--device-map", default="cuda")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--max-windows", type=int, default=5000)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--fee-bps",     type=float, default=10.0)
    p.add_argument("--per-symbol",  action="store_true")
    p.add_argument("--output",      default=None,
                   help="Save results JSON to this path")
    return p.parse_args(argv)


def _load_calibration(model_id: str, cal_path: Optional[str]) -> CalibrationParams:
    path = Path(cal_path) if cal_path else Path(model_id) / "calibration.json"
    if not path.exists():
        print(f"WARNING: calibration not found at {path}, using defaults")
        return CalibrationParams(model_id=model_id)
    d = json.loads(path.read_text())
    return CalibrationParams.from_dict(d)


def main(argv=None) -> int:
    args = parse_args(argv)
    cal_dir = Path(args.cal_data_dir)

    # Load series
    print(f"Loading series from {cal_dir} ...")
    all_series = load_all_series(cal_dir, min_length=50)
    if not all_series:
        print("ERROR: No series found"); return 1
    print(f"Loaded {len(all_series)} series")

    # Use last 60 bars as test (same split as calibration)
    test_series = []
    for s in all_series:
        arr = s["target"]
        sym = s.get("symbol", "")
        T = arr.shape[-1]
        n_test = min(60, T - args.context_length)
        if n_test > 0:
            test_series.append({"target": arr[:, -(args.context_length + n_test):], "symbol": sym})
    print(f"Using {len(test_series)} series for backtest")

    # Collect predictions for primary model
    print(f"\nRunning inference for: {args.model_id}")
    q10, q50, q90, actual, prev_close, symbols = collect_predictions(
        model_id=args.model_id,
        series_list=test_series,
        context_length=args.context_length,
        device_map=args.device_map,
        torch_dtype_str=args.torch_dtype,
        max_windows=args.max_windows,
        batch_size=args.batch_size,
    )

    # Primary model backtest
    params = _load_calibration(args.model_id, args.calibration)
    stats = run_backtest(q10, q50, q90, actual, prev_close, symbols, params, args.fee_bps)

    print("\n=== Primary model ===")
    _print_stats(stats, args.model_id.split("/")[-2] if "/" in args.model_id else args.model_id)

    if args.per_symbol and "per_symbol" in stats:
        top = sorted(stats["per_symbol"].items(), key=lambda x: x[1]["sharpe"], reverse=True)[:10]
        print("\nTop 10 symbols by Sharpe:")
        for sym, s in top:
            print(f"  {sym:8s}: Sharpe={s['sharpe']:+.3f}  BnH={s['bnh_sharpe']:+.3f}  n={s['n']}")

    results = {"primary": stats}

    # Compare model
    if args.compare_model:
        print(f"\nRunning inference for compare: {args.compare_model}")
        q10c, q50c, q90c, actualc, prev_c, symsc = collect_predictions(
            model_id=args.compare_model,
            series_list=test_series,
            context_length=args.context_length,
            device_map=args.device_map,
            torch_dtype_str=args.torch_dtype,
            max_windows=args.max_windows,
            batch_size=args.batch_size,
        )
        params_c = _load_calibration(args.compare_model, None)
        stats_c = run_backtest(q10c, q50c, q90c, actualc, prev_c, symsc, params_c, args.fee_bps)

        print("\n=== Compare model ===")
        _print_stats(stats_c, args.compare_model.split("/")[-2] if "/" in args.compare_model else args.compare_model)

        delta = stats["sharpe_annualized"] - stats_c["sharpe_annualized"]
        print(f"\nDelta Sharpe (primary - compare): {delta:+.3f}")
        results["compare"] = stats_c
        results["delta_sharpe"] = delta

    if args.output:
        # Remove non-serializable parts
        out = json.dumps(results, indent=2, default=str)
        Path(args.output).write_text(out)
        print(f"\nSaved → {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
