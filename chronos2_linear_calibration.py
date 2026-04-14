#!/usr/bin/env python3
"""
Post-training linear calibration for Chronos2 buy/sell thresholds.

After fine-tuning Chronos2, the raw quantile predictions can be calibrated
to improve trading performance. This script:

1. Runs the model on a held-out calibration set to collect:
   - Predicted quantiles (q10, q50, q90) for the close price
   - Actual future close prices

2. Fits a threshold model that maps raw predictions to buy/sell signals:
     signal = w * predicted_return_q50
     buy   if signal > buy_threshold
     sell  if signal < -sell_threshold  (allow_short=True)
     exit  if signal < -sell_threshold  (allow_short=False)

   where predicted_return = (q50_pred - close_now) / close_now

3. Optionally filters by uncertainty: skip trades when
     uncertainty = (q90 - q10) / prev_close > confidence_threshold
   This forces the model to only trade when the quantile interval is narrow.

4. Two-phase grid search: coarse pass over ±max_shift_bps, then fine pass
   (±2bps around best) for precise threshold placement.

5. Per-symbol mode: calibrate each symbol independently and save per-symbol
   JSONs under hyperparams/chronos2/{SYM}_calibration.json.

Constraints:
   - buy_threshold, sell_threshold ∈ [-max_shift_bps, +max_shift_bps]
   - sell_threshold > buy_threshold - min_gap
   - If allow_short=False: sell capped at "exit long"

Usage:
    python chronos2_linear_calibration.py \\
        --model-id chronos2_finetuned/stocks_all_v3/finetuned-ckpt \\
        --cal-data-dir trainingdata \\
        --output-path chronos2_finetuned/stocks_all_v3/finetuned-ckpt/calibration.json \\
        --max-shift-bps 20 \\
        --allow-short

    # Per-symbol calibration:
    python chronos2_linear_calibration.py \\
        --model-id chronos2_finetuned/stocks_all_v3/finetuned-ckpt \\
        --cal-data-dir trainingdata \\
        --per-symbol \\
        --hyperparams-dir hyperparams/chronos2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    # Weight on quantile skewness component: (q90-q50)-(q50-q10) / prev_close
    # Positive skew_weight means we add upside-skewed predictions' signal bonus.
    # 0.0 = disabled (default for backward compatibility)
    skew_weight: float = 0.0

    # Buy threshold in fractional return space
    # If signal > buy_threshold: go long
    buy_threshold: float = 0.001   # 10 bps default

    # Sell/exit threshold
    # If signal < -sell_threshold: sell/short (or exit long)
    sell_threshold: float = 0.001  # 10 bps default

    # Whether shorting is permitted
    allow_short: bool = False

    # Uncertainty (confidence) filter: skip trades when
    # (q90 - q10) / prev_close > confidence_threshold (0 = disabled)
    confidence_threshold: float = 0.0

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

    def apply(self, predicted_return: float, uncertainty: float = 0.0,
              skewness: float = 0.0) -> str:
        """
        Return 'buy', 'sell', 'exit', or 'hold'.

        Args:
            predicted_return: (q50 - prev_close) / prev_close
            uncertainty: (q90 - q10) / prev_close  — skip if above confidence_threshold
            skewness: ((q90-q50) - (q50-q10)) / prev_close  — optional tail skew signal
        """
        # Confidence filter: hold if model is too uncertain
        if self.confidence_threshold > 0.0 and uncertainty > self.confidence_threshold:
            return "hold"
        signal = predicted_return * self.signal_weight + skewness * self.skew_weight + self.signal_bias
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run model inference on sliding windows of each series.
    Batches windows together for efficient GPU utilization.

    Returns:
        q10:       (N,) predicted 10th percentile of close
        q50:       (N,) predicted 50th percentile (median)
        q90:       (N,) predicted 90th percentile
        actual:    (N,) actual close at the next timestep
        prev_close:(N,) close at the last context bar (to compute returns)
        symbols:   [N]  symbol name for each window (empty string if unknown)
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

    all_q10, all_q50, all_q90, all_actual, all_prev, all_syms = [], [], [], [], [], []
    window_count = 0

    # Collect all windows first, then process in batches
    pending_ctx: List[Any] = []
    pending_labels: List[Tuple[float, float, str]] = []  # (fut_close, ctx_close, symbol)

    for s in series_list:
        arr = s["target"]  # (4, T)
        sym = s.get("symbol", "")
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
            pending_labels.append((fut_close, ctx_close, sym))
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

        for j, (pred_t, (fut_close, ctx_close, sym)) in enumerate(zip(preds, batch_labels)):
            pred_np = pred_t.detach().cpu().numpy()  # (4, n_quantiles, pred_len)
            if pred_np.ndim != 3 or pred_np.shape[0] < 4:
                continue
            all_q10.append(float(pred_np[3, i10, 0]))
            all_q50.append(float(pred_np[3, i50, 0]))
            all_q90.append(float(pred_np[3, i90, 0]))
            all_actual.append(fut_close)
            all_prev.append(ctx_close)
            all_syms.append(sym)

    print(f"Collected {len(all_actual)} calibration windows")
    return (
        np.array(all_q10, dtype=np.float64),
        np.array(all_q50, dtype=np.float64),
        np.array(all_q90, dtype=np.float64),
        np.array(all_actual, dtype=np.float64),
        np.array(all_prev, dtype=np.float64),
        all_syms,
    )


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------

def collect_ensemble_predictions(
    model_ids: List[str],
    series_list: List[dict],
    context_length: int = 512,
    prediction_length: int = 1,
    device_map: str = "cuda",
    torch_dtype_str: str = "bfloat16",
    max_windows: int = 5000,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run inference with multiple models and average their quantile predictions.

    Loads each model sequentially (to avoid GPU OOM), collects predictions,
    then returns the per-quantile average across all models.

    Returns same signature as collect_predictions():
        q10, q50, q90, actual, prev_close, symbols
    """
    if len(model_ids) == 1:
        return collect_predictions(
            model_ids[0], series_list, context_length, prediction_length,
            device_map, torch_dtype_str, max_windows, batch_size,
        )

    all_q10_runs: List[np.ndarray] = []
    all_q50_runs: List[np.ndarray] = []
    all_q90_runs: List[np.ndarray] = []
    actual_ref = None
    prev_ref = None
    syms_ref = None

    for mid in model_ids:
        print(f"\n[Ensemble] Collecting predictions from: {mid}")
        q10, q50, q90, actual, prev, syms = collect_predictions(
            mid, series_list, context_length, prediction_length,
            device_map, torch_dtype_str, max_windows, batch_size,
        )
        all_q10_runs.append(q10)
        all_q50_runs.append(q50)
        all_q90_runs.append(q90)
        if actual_ref is None:
            actual_ref = actual
            prev_ref = prev
            syms_ref = syms
        else:
            # Sanity check: all models should produce same number of windows
            if len(actual) != len(actual_ref):  # type: ignore[arg-type]
                print(f"WARNING: {mid} produced {len(actual)} windows vs {len(actual_ref)} — truncating to min")
                n = min(len(actual), len(actual_ref))  # type: ignore[arg-type]
                all_q10_runs[-1] = q10[:n]
                all_q50_runs[-1] = q50[:n]
                all_q90_runs[-1] = q90[:n]
                actual_ref = actual_ref[:n]
                prev_ref = prev_ref[:n]  # type: ignore[index]
                syms_ref = syms_ref[:n]  # type: ignore[index]

    # Average quantiles across models (approximate ensemble median)
    q10_avg = np.mean(all_q10_runs, axis=0)
    q50_avg = np.mean(all_q50_runs, axis=0)
    q90_avg = np.mean(all_q90_runs, axis=0)

    return q10_avg, q50_avg, q90_avg, actual_ref, prev_ref, syms_ref  # type: ignore[return-value]


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
    uncertainties: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.0,
    skewness: Optional[np.ndarray] = None,
    skew_weight: float = 0.0,
) -> float:
    """
    Compute Sharpe-like score for a given threshold pair on in-sample data.

    Fee is charged only on position transitions (entry/exit), not every held day.
    A "hold" day (flat) earns 0. P&L is collected as a daily series over all
    windows (including flat days) so the denominator reflects real volatility.

    Args:
        signals:              predicted returns (q50 - prev_close) / prev_close
        actual_returns:       (actual_close - prev_close) / prev_close
        uncertainties:        (q90 - q10) / prev_close per window (optional)
        confidence_threshold: skip trade if uncertainty > this value (0 = disabled)
    """
    fee = fee_bps / 10_000.0
    pnl: List[float] = []
    position = 0  # 0=flat, 1=long, -1=short
    n_trades = 0
    use_conf = confidence_threshold > 0.0 and uncertainties is not None
    use_skew = skew_weight != 0.0 and skewness is not None

    for i, (sig, ret) in enumerate(zip(signals, actual_returns)):
        # Add skewness component to signal if enabled
        effective_sig = sig + (float(skewness[i]) * skew_weight if use_skew else 0.0)  # type: ignore[index]
        # Confidence filter: force flat if model is uncertain
        if use_conf and uncertainties[i] > confidence_threshold:  # type: ignore[index]
            desired = 0
        elif effective_sig > buy_thresh:
            desired = 1
        elif allow_short and effective_sig < -sell_thresh:
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


def _run_grid(
    predicted_return: np.ndarray,
    actual_return: np.ndarray,
    uncertainties: np.ndarray,
    thresh_vals: np.ndarray,
    weight_vals: List[float],
    conf_vals: List[float],
    allow_short: bool,
    min_gap: float,
    fee_bps: float,
    skewness: Optional[np.ndarray] = None,
    skew_weight_vals: Optional[List[float]] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Inner grid search loop.
    Returns (best_sharpe, best_buy, best_sell, best_weight, best_conf, best_skew_weight).
    """
    best_sharpe = -999.0
    best_buy = 0.0
    best_sell = 0.0
    best_weight = 1.0
    best_conf = 0.0
    best_skew_w = 0.0

    _skew_vals: List[float] = skew_weight_vals if skew_weight_vals else [0.0]

    for w in weight_vals:
        scaled_return = predicted_return * w
        for buy_t in thresh_vals:
            for sell_t in thresh_vals:
                if allow_short and sell_t < buy_t - min_gap:
                    continue
                for conf in conf_vals:
                    for sw in _skew_vals:
                        sharpe = compute_sharpe(
                            scaled_return, actual_return, float(buy_t), float(sell_t),
                            allow_short=allow_short, fee_bps=fee_bps,
                            uncertainties=uncertainties,
                            confidence_threshold=float(conf),
                            skewness=skewness,
                            skew_weight=float(sw),
                        )
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_buy = float(buy_t)
                            best_sell = float(sell_t)
                            best_weight = float(w)
                            best_conf = float(conf)
                            best_skew_w = float(sw)

    return best_sharpe, best_buy, best_sell, best_weight, best_conf, best_skew_w


def fit_calibration(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    actual: np.ndarray,
    prev_close: np.ndarray,
    *,
    max_shift_bps: float = 20.0,
    min_gap_bps: float = 2.0,
    allow_short: bool = False,
    fee_bps: float = 10.0,
    grid_steps: int = 17,
    search_signal_weight: bool = True,
    search_confidence: bool = True,
    model_id: str = "",
) -> CalibrationParams:
    """
    Two-phase grid-search over (signal_weight, buy_threshold, sell_threshold,
    confidence_threshold) to maximise Sharpe ratio on the calibration set.

    Phase 1 — coarse search over ±max_shift_bps with grid_steps points.
    Phase 2 — fine search ±2bps around the best found in phase 1.

    signal_weight scales the raw predicted return before thresholding.
    confidence_threshold filters high-uncertainty bars (wide q90-q10 interval).

    Constraint: sell_threshold > buy_threshold - min_gap.
    """
    eps = 1e-8
    predicted_return = (q50 - prev_close) / (np.abs(prev_close) + eps)
    actual_return    = (actual - prev_close) / (np.abs(prev_close) + eps)
    uncertainties    = (q90 - q10) / (np.abs(prev_close) + eps)
    # Skewness: positive = right-tail heavy (upside surprise potential)
    skewness         = ((q90 - q50) - (q50 - q10)) / (np.abs(prev_close) + eps)

    max_shift = max_shift_bps / 10_000.0
    min_gap   = min_gap_bps   / 10_000.0

    # --- Coarse phase ---
    thresh_vals_coarse = np.linspace(-max_shift, max_shift, grid_steps)
    weight_vals = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] if search_signal_weight else [1.0]

    # Confidence threshold: 0 = disabled, else % of prev_close
    # Grid over uncertainty percentiles: 25th, 50th, 75th (+ no filter)
    if search_confidence and len(uncertainties) > 10:
        pcts = np.percentile(uncertainties, [25, 50, 75])
        conf_vals: List[float] = [0.0] + [float(p) for p in pcts]
    else:
        conf_vals = [0.0]

    # Skew weight search: [0, 0.5, 1.0, 2.0] — 0 means disabled
    skew_weight_vals: List[float] = [0.0, 0.5, 1.0, 2.0]

    best_sharpe, best_buy, best_sell, best_weight, best_conf, best_skew_w = _run_grid(
        predicted_return, actual_return, uncertainties,
        thresh_vals_coarse, weight_vals, conf_vals,
        allow_short, min_gap, fee_bps,
        skewness=skewness, skew_weight_vals=skew_weight_vals,
    )

    # --- Fine phase: ±2bps around best, 17 points each ---
    fine_bps = 2.0 / 10_000.0
    thresh_fine = np.linspace(
        max(-max_shift, best_buy - fine_bps),
        min( max_shift, best_buy + fine_bps),
        17,
    )
    sell_fine = np.linspace(
        max(-max_shift, best_sell - fine_bps),
        min( max_shift, best_sell + fine_bps),
        17,
    )
    # Combine both ranges for both buy and sell
    thresh_both = np.unique(np.concatenate([thresh_fine, sell_fine]))

    sharpe2, buy2, sell2, weight2, conf2, skew2 = _run_grid(
        predicted_return, actual_return, uncertainties,
        thresh_both, [best_weight], [best_conf],
        allow_short, min_gap, fee_bps,
        skewness=skewness, skew_weight_vals=[best_skew_w],
    )
    if sharpe2 > best_sharpe:
        best_sharpe, best_buy, best_sell, best_weight, best_conf, best_skew_w = (
            sharpe2, buy2, sell2, weight2, conf2, skew2)

    # --- Phase 3: refine signal_weight around best (±40% in 9 steps) ---
    if search_signal_weight:
        w_fine = np.linspace(best_weight * 0.6, best_weight * 1.4, 9)
        sharpe3, buy3, sell3, weight3, conf3, skew3 = _run_grid(
            predicted_return, actual_return, uncertainties,
            np.array([best_buy, best_sell]),
            list(w_fine), [best_conf],
            allow_short, min_gap, fee_bps,
            skewness=skewness, skew_weight_vals=[best_skew_w],
        )
        if sharpe3 > best_sharpe:
            best_sharpe, best_buy, best_sell, best_weight, best_conf, best_skew_w = (
                sharpe3, buy3, sell3, weight3, conf3, skew3)

    params = CalibrationParams(
        signal_weight=best_weight,
        signal_bias=0.0,
        skew_weight=best_skew_w,
        buy_threshold=best_buy,
        sell_threshold=best_sell,
        allow_short=allow_short,
        confidence_threshold=best_conf,
        model_id=model_id,
        n_cal_windows=len(actual),
        cal_sharpe=best_sharpe,
    )
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_params(params: CalibrationParams) -> None:
    conf_str = f"  conf={params.confidence_threshold*1e4:.1f}bps" if params.confidence_threshold > 0 else ""
    print(f"  buy={params.buy_threshold*1e4:.1f}bps  sell={params.sell_threshold*1e4:.1f}bps"
          f"  weight={params.signal_weight:.2f}{conf_str}  sharpe={params.cal_sharpe:.3f}"
          f"  allow_short={params.allow_short}  n={params.n_cal_windows}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate Chronos2 buy/sell thresholds")
    p.add_argument("--model-id",    required=True,
                   help="Model path or HF id (e.g. chronos2_finetuned/stocks_all_v3/finetuned-ckpt)")
    p.add_argument("--cal-data-dir", default="trainingdata",
                   help="Directory of CSV files to use as calibration data (default: trainingdata/)")
    p.add_argument("--output-path", default=None,
                   help="Where to save calibration JSON (default: <model-id>/calibration.json)")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--device-map", default="cuda")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--max-windows", type=int, default=5000)
    p.add_argument("--max-shift-bps", type=float, default=20.0,
                   help="Search range ±N bps around 0 (default: 20)")
    p.add_argument("--min-gap-bps",   type=float, default=2.0)
    p.add_argument("--fee-bps",       type=float, default=10.0)
    p.add_argument("--allow-short",   action="store_true")
    p.add_argument("--grid-steps",    type=int, default=17)
    p.add_argument("--batch-size",    type=int, default=32,
                   help="GPU batch size for inference (default: 32)")
    p.add_argument("--no-search-signal-weight", action="store_true",
                   help="Fix signal_weight=1.0, only search thresholds (faster)")
    p.add_argument("--no-search-confidence", action="store_true",
                   help="Disable uncertainty/confidence filter search")
    p.add_argument("--per-symbol", action="store_true",
                   help="Calibrate each symbol independently and save per-symbol JSONs")
    p.add_argument("--hyperparams-dir", default="hyperparams/chronos2",
                   help="Directory to write per-symbol calibration JSONs (default: hyperparams/chronos2)")
    p.add_argument("--min-windows-per-symbol", type=int, default=30,
                   help="Minimum windows to fit per-symbol calibration (default: 30)")
    p.add_argument("--ensemble-models", nargs="*", default=None,
                   help="Additional model IDs to ensemble with --model-id (quantile averaging)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    cal_dir = Path(args.cal_data_dir)
    if not cal_dir.exists():
        print(f"ERROR: calibration data dir not found: {cal_dir}")
        return 1

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
        sym = s.get("symbol", "")
        T = arr.shape[-1]
        n_cal = min(60, T - args.context_length)
        if n_cal > 0:
            cal_series.append({"target": arr[:, -(args.context_length + n_cal):], "symbol": sym})

    print(f"Using {len(cal_series)} series for calibration")

    # Collect predictions (one pass — reuse for global + per-symbol)
    # If --ensemble-models given, average quantiles across all models
    ensemble_ids = [args.model_id] + (args.ensemble_models or [])
    collect_fn = collect_ensemble_predictions if len(ensemble_ids) > 1 else collect_predictions
    collect_kwargs: dict = dict(
        series_list=cal_series,
        context_length=args.context_length,
        device_map=args.device_map,
        torch_dtype_str=args.torch_dtype,
        max_windows=args.max_windows,
    )
    if len(ensemble_ids) > 1:
        print(f"Ensemble mode: averaging predictions from {len(ensemble_ids)} models")
        collect_kwargs["model_ids"] = ensemble_ids
    else:
        collect_kwargs["model_id"] = args.model_id

    q10, q50, q90, actual, prev_close, symbols = collect_fn(
        **collect_kwargs,
        batch_size=args.batch_size,
    )

    if len(actual) < 50:
        print(f"Too few calibration windows ({len(actual)}); cannot fit calibration.")
        return 1

    search_w = not getattr(args, "no_search_signal_weight", False)
    search_c = not getattr(args, "no_search_confidence", False)
    fit_kwargs: Dict[str, Any] = dict(
        max_shift_bps=args.max_shift_bps,
        min_gap_bps=args.min_gap_bps,
        fee_bps=args.fee_bps,
        grid_steps=args.grid_steps,
        search_signal_weight=search_w,
        search_confidence=search_c,
        model_id=args.model_id,
    )

    # ---- Global calibration ----
    output_path = (Path(args.output_path) if args.output_path
                   else Path(args.model_id) / "calibration.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFitting global calibration ({len(actual)} windows, ±{args.max_shift_bps}bps"
          f"{', weight-search' if search_w else ''}"
          f"{', conf-search' if search_c else ''}) ...")
    params = fit_calibration(q10=q10, q50=q50, q90=q90, actual=actual,
                             prev_close=prev_close, allow_short=args.allow_short, **fit_kwargs)
    print("Global result:"); _print_params(params)
    output_path.write_text(json.dumps(params.to_dict(), indent=2))
    print(f"Saved → {output_path}")

    # Short variant
    if not args.allow_short:
        out_short = output_path.parent / "calibration_short.json"
        params_short = fit_calibration(q10=q10, q50=q50, q90=q90, actual=actual,
                                       prev_close=prev_close, allow_short=True, **fit_kwargs)
        print("Short variant:"); _print_params(params_short)
        out_short.write_text(json.dumps(params_short.to_dict(), indent=2))
        print(f"Saved → {out_short}")

    # ---- Per-symbol calibration ----
    if args.per_symbol:
        syms_arr = np.array(symbols)
        unique_syms = [s for s in sorted(set(symbols)) if s]
        print(f"\nPer-symbol calibration for {len(unique_syms)} symbols ...")
        hp_dir = Path(args.hyperparams_dir)
        hp_dir.mkdir(parents=True, exist_ok=True)

        n_saved = 0
        for sym in unique_syms:
            mask = syms_arr == sym
            n = int(mask.sum())
            if n < args.min_windows_per_symbol:
                continue
            sym_params = fit_calibration(
                q10=q10[mask], q50=q50[mask], q90=q90[mask],
                actual=actual[mask], prev_close=prev_close[mask],
                allow_short=args.allow_short, **fit_kwargs,
            )
            sym_out = hp_dir / f"{sym}_calibration.json"
            sym_out.write_text(json.dumps(sym_params.to_dict(), indent=2))
            n_saved += 1
            if n_saved <= 5 or sym_params.cal_sharpe > params.cal_sharpe + 0.05:
                print(f"  {sym:8s} (n={n:4d}):"); _print_params(sym_params)

        print(f"Saved {n_saved} per-symbol calibrations → {hp_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
