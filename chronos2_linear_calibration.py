#!/usr/bin/env python3
"""
Post-training linear calibration for Chronos2 buy/sell thresholds.

After fine-tuning Chronos2, the raw quantile predictions can be calibrated
to improve trading performance. This script:

1. Runs the model on a held-out calibration set to collect:
   - Predicted quantiles (q10, q50, q90) for the close price
   - Actual future close prices

2. Fits a simple linear threshold model that maps raw predictions to
   buy/sell signals:
     signal_pct = w * predicted_return_q50 + b
     buy  if signal_pct >  buy_threshold
     sell if signal_pct < -sell_threshold

   where predicted_return = (q50_pred - close_now) / close_now

3. Constraints:
   - buy_threshold  ∈ [-max_shift_bps, +max_shift_bps]  (default: 8 bps)
   - sell_threshold ∈ [-max_shift_bps, +max_shift_bps]  (default: 8 bps)
   - sell_threshold > buy_threshold - min_gap  (ensure sell > buy for profitability)
   - If allow_short=False: sell signals are capped at "exit long" only

4. Saves calibration params to JSON, compatible with the Chronos2 wrapper.

Usage:
    python chronos2_linear_calibration.py \\
        --model-id chronos2_finetuned/stocks_all_v1/finetuned-ckpt \\
        --cal-data-dir trainingdata \\
        --output-path chronos2_finetuned/stocks_all_v1/calibration.json \\
        --max-shift-bps 8 \\
        --allow-short
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chronos2_stock_augmentation import load_all_series, OHLC_COLS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationParams:
    """Linear calibration parameters for Chronos2 buy/sell thresholds."""

    # Multiplier applied to the raw predicted return (default: 1.0 = no scaling)
    signal_weight: float = 1.0

    # Additive bias on the signal (default: 0.0)
    signal_bias: float = 0.0

    # Buy threshold in fractional return space
    # If predicted_return * signal_weight + signal_bias > buy_threshold: go long
    buy_threshold: float = 0.001   # 10 bps default

    # Sell/exit threshold
    # If predicted_return * signal_weight + signal_bias < -sell_threshold: sell/short
    sell_threshold: float = 0.001  # 10 bps default

    # Whether shorting is permitted
    allow_short: bool = False

    # Source model
    model_id: str = ""

    # Calibration set info
    n_cal_windows: int = 0
    cal_sharpe: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def apply(self, predicted_return: float) -> str:
        """Return 'buy', 'sell', or 'hold'."""
        signal = predicted_return * self.signal_weight + self.signal_bias
        if signal > self.buy_threshold:
            return "buy"
        if self.allow_short and signal < -self.sell_threshold:
            return "sell"
        if not self.allow_short and signal < -self.sell_threshold:
            return "exit"  # close long position
        return "hold"


# ---------------------------------------------------------------------------
# Collect predictions from model
# ---------------------------------------------------------------------------

def collect_predictions(
    model_id: str,
    series_list: List[dict],
    context_length: int = 512,
    prediction_length: int = 1,
    device_map: str = "cuda",
    torch_dtype_str: str = "bfloat16",
    max_windows: int = 5000,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference on sliding windows of each series.
    Batches windows together for efficient GPU utilization.

    Returns:
        q10: (N,) predicted 10th percentile of close
        q50: (N,) predicted 50th percentile (median)
        q90: (N,) predicted 90th percentile
        actual: (N,) actual close at the next timestep
        prev_close: (N,) close at the last context bar (to compute returns)
    """
    import torch

    try:
        from chronos import Chronos2Pipeline
    except ImportError as e:
        raise RuntimeError("chronos-forecasting required") from e

    dtype = None
    if torch_dtype_str:
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}.get(torch_dtype_str.lower())
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device_map, torch_dtype=dtype)

    # Find quantile indices for q10, q50, q90
    try:
        qs = list(pipeline.model.chronos_config.quantiles)
        i10 = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.10))
        i50 = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.50))
        i90 = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.90))
    except Exception:
        i10, i50, i90 = 0, 10, 20  # fallback indices

    all_q10, all_q50, all_q90, all_actual, all_prev = [], [], [], [], []
    window_count = 0

    # Collect all windows first, then process in batches
    pending_ctx: List[Any] = []
    pending_labels: List[Tuple[float, float]] = []  # (fut_close, ctx_close)

    for s in series_list:
        arr = s["target"]  # (4, T)
        T = arr.shape[-1]
        if T < context_length + prediction_length:
            continue

        for start in range(context_length, T - prediction_length + 1, prediction_length):
            if window_count >= max_windows:
                break

            ctx_arr = arr[:, start - context_length : start]
            fut_close = float(arr[3, start])
            ctx_close = float(arr[3, start - 1])

            pending_ctx.append(torch.from_numpy(ctx_arr).float())
            pending_labels.append((fut_close, ctx_close))
            window_count += 1

        if window_count >= max_windows:
            break

    print(f"Running inference on {len(pending_ctx)} windows (batch={batch_size})...")
    # Process in batches
    for batch_start in range(0, len(pending_ctx), batch_size):
        batch_ctx = pending_ctx[batch_start : batch_start + batch_size]
        batch_labels = pending_labels[batch_start : batch_start + batch_size]

        try:
            preds = pipeline.predict(batch_ctx, prediction_length=prediction_length,
                                     batch_size=len(batch_ctx))
        except Exception:
            continue

        for j, (pred_t, (fut_close, ctx_close)) in enumerate(zip(preds, batch_labels)):
            pred_np = pred_t.detach().cpu().numpy()  # (4, n_quantiles, pred_len)
            if pred_np.ndim != 3 or pred_np.shape[0] < 4:
                continue
            all_q10.append(float(pred_np[3, i10, 0]))
            all_q50.append(float(pred_np[3, i50, 0]))
            all_q90.append(float(pred_np[3, i90, 0]))
            all_actual.append(fut_close)
            all_prev.append(ctx_close)

    print(f"Collected {len(all_actual)} calibration windows")
    return (
        np.array(all_q10, dtype=np.float64),
        np.array(all_q50, dtype=np.float64),
        np.array(all_q90, dtype=np.float64),
        np.array(all_actual, dtype=np.float64),
        np.array(all_prev, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Grid-search calibration
# ---------------------------------------------------------------------------

def compute_sharpe(
    signals: np.ndarray,
    actual_returns: np.ndarray,
    buy_thresh: float,
    sell_thresh: float,
    allow_short: bool,
    fee_bps: float = 10.0,
) -> float:
    """
    Compute Sharpe-like score for a given threshold pair on in-sample data.

    Fee is charged only on position transitions (entry/exit), not every held day.
    A "hold" day (flat) earns 0. P&L is collected as a daily series over all
    windows (including flat days) so the denominator reflects real volatility.

    signals:        predicted returns (q50 - prev_close) / prev_close
    actual_returns: (actual_close - prev_close) / prev_close
    """
    fee = fee_bps / 10_000.0
    pnl: List[float] = []
    position = 0  # 0=flat, 1=long, -1=short
    n_trades = 0

    for sig, ret in zip(signals, actual_returns):
        # Determine desired position from signal
        if sig > buy_thresh:
            desired = 1
        elif allow_short and sig < -sell_thresh:
            desired = -1
        else:
            desired = 0

        # Pay transition fee when position changes
        if desired != position:
            transition_cost = fee
            n_trades += 1
        else:
            transition_cost = 0.0

        # Collect P&L: return from held position minus any transition cost
        if position == 1:
            pnl.append(ret - transition_cost)
        elif position == -1:
            pnl.append(-ret - transition_cost)
        else:
            # flat — only cost is entry fee if we're entering this bar
            pnl.append(-transition_cost)

        position = desired

    if n_trades == 0:
        return -999.0
    arr = np.array(pnl)
    mean = arr.mean()
    std  = arr.std() + 1e-10
    return float(mean / std * np.sqrt(252))  # annualise roughly


def fit_calibration(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    actual: np.ndarray,
    prev_close: np.ndarray,
    *,
    max_shift_bps: float = 8.0,
    min_gap_bps: float = 2.0,
    allow_short: bool = False,
    fee_bps: float = 10.0,
    grid_steps: int = 17,
    search_signal_weight: bool = True,
    model_id: str = "",
) -> CalibrationParams:
    """
    Grid-search over (signal_weight, buy_threshold, sell_threshold) to maximise
    Sharpe ratio on the calibration set.

    signal_weight scales the raw predicted return before thresholding:
      signal = (q50 - prev_close) / prev_close * signal_weight
      buy  when signal > buy_threshold
      sell when signal < -sell_threshold  (with allow_short=True)
      exit when signal < -sell_threshold  (without allow_short)

    Threshold range: ±max_shift_bps around 0.
    Constraint: sell_threshold > buy_threshold - min_gap (no accidental straddle).

    Returns CalibrationParams with the best-found thresholds.
    """
    eps = 1e-8
    predicted_return = (q50 - prev_close) / (np.abs(prev_close) + eps)
    actual_return    = (actual - prev_close) / (np.abs(prev_close) + eps)

    max_shift = max_shift_bps / 10_000.0
    min_gap   = min_gap_bps   / 10_000.0

    # Grid over thresholds in fractional return space
    thresh_vals = np.linspace(-max_shift, max_shift, grid_steps)

    # Signal weight candidates: sub-1 = less sensitive, >1 = more sensitive
    weight_vals = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] if search_signal_weight else [1.0]

    best_sharpe  = -999.0
    best_buy     = 0.0
    best_sell    = 0.0
    best_weight  = 1.0

    for w in weight_vals:
        scaled_return = predicted_return * w
        for buy_t in thresh_vals:
            for sell_t in thresh_vals:
                # Constraint: sell_t must be larger than buy_t - min_gap
                if allow_short and sell_t < buy_t - min_gap:
                    continue
                sharpe = compute_sharpe(
                    scaled_return, actual_return, float(buy_t), float(sell_t),
                    allow_short=allow_short, fee_bps=fee_bps,
                )
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_buy    = float(buy_t)
                    best_sell   = float(sell_t)
                    best_weight = float(w)

    params = CalibrationParams(
        signal_weight=best_weight,
        signal_bias=0.0,
        buy_threshold=best_buy,
        sell_threshold=best_sell,
        allow_short=allow_short,
        model_id=model_id,
        n_cal_windows=len(actual),
        cal_sharpe=best_sharpe,
    )
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate Chronos2 buy/sell thresholds")
    p.add_argument("--model-id",    required=True,
                   help="Model path or HF id (e.g. chronos2_finetuned/stocks_all_v1/finetuned-ckpt)")
    p.add_argument("--cal-data-dir", default="trainingdata",
                   help="Directory of CSV files to use as calibration data (default: trainingdata/)")
    p.add_argument("--output-path", default=None,
                   help="Where to save calibration JSON (default: <model-id>/calibration.json)")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--device-map", default="cuda")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--max-windows", type=int, default=5000)
    p.add_argument("--max-shift-bps", type=float, default=8.0)
    p.add_argument("--min-gap-bps",   type=float, default=2.0)
    p.add_argument("--fee-bps",       type=float, default=10.0)
    p.add_argument("--allow-short",   action="store_true")
    p.add_argument("--grid-steps",    type=int, default=17)
    p.add_argument("--batch-size",    type=int, default=32,
                   help="GPU batch size for inference (default: 32)")
    p.add_argument("--no-search-signal-weight", action="store_true",
                   help="Fix signal_weight=1.0, only search thresholds (faster)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    cal_dir = Path(args.cal_data_dir)
    if not cal_dir.exists():
        print(f"ERROR: calibration data dir not found: {cal_dir}")
        return 1

    output_path = Path(args.output_path) if args.output_path else Path(args.model_id) / "calibration.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load calibration series
    print(f"Loading calibration series from {cal_dir} ...")
    all_series = load_all_series(cal_dir, min_length=50)
    if not all_series:
        print("ERROR: No valid series found")
        return 1
    print(f"Loaded {len(all_series)} series")

    # Use last 60 bars of each series as calibration window (unseen by training)
    cal_series = []
    for s in all_series:
        arr = s["target"]
        T = arr.shape[-1]
        n_cal = min(60, T - args.context_length)
        if n_cal > 0:
            cal_series.append({"target": arr[:, -(args.context_length + n_cal):]})

    print(f"Using {len(cal_series)} series for calibration")

    # Collect predictions
    q10, q50, q90, actual, prev_close = collect_predictions(
        model_id=args.model_id,
        series_list=cal_series,
        context_length=args.context_length,
        device_map=args.device_map,
        torch_dtype_str=args.torch_dtype,
        max_windows=args.max_windows,
        batch_size=args.batch_size,
    )

    if len(actual) < 50:
        print(f"Too few calibration windows ({len(actual)}); cannot fit calibration.")
        return 1

    # Fit calibration
    search_w = not getattr(args, "no_search_signal_weight", False)
    print(f"Fitting calibration ({len(actual)} windows, max_shift={args.max_shift_bps}bps"
          f"{', weight-search' if search_w else ''}) ...")
    params = fit_calibration(
        q10=q10, q50=q50, q90=q90,
        actual=actual, prev_close=prev_close,
        max_shift_bps=args.max_shift_bps,
        min_gap_bps=args.min_gap_bps,
        allow_short=args.allow_short,
        fee_bps=args.fee_bps,
        grid_steps=args.grid_steps,
        search_signal_weight=search_w,
        model_id=args.model_id,
    )

    print(f"Best thresholds: buy={params.buy_threshold*1e4:.1f}bps  "
          f"sell={params.sell_threshold*1e4:.1f}bps  "
          f"weight={params.signal_weight:.2f}  "
          f"sharpe={params.cal_sharpe:.3f}  "
          f"allow_short={params.allow_short}")

    output_path.write_text(json.dumps(params.to_dict(), indent=2))
    print(f"Saved calibration → {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
