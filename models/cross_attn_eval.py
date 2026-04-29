"""OOS eval for the cross-attention transformer.

Loads a trained checkpoint, scores every (date, symbol) row in the OOS
window, then runs the same windowed backtest used by
``xgbnew.sweep_ensemble_grid``: 30-day windows, 7-day stride, top_n=1,
hold_through, deploy fees.

Outputs a JSON with median/p10/n_neg/maxDD/goodness_score + the ensemble's
per-window monthly returns + a comparison line vs the XGB 15-seed baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.cross_attn_data import build_panel
from models.cross_attn_transformer_v1 import CrossAttnConfig, CrossAttnTransformerV1
from xgbnew.backtest import BacktestConfig, simulate
from xgbnew.dataset import build_daily_dataset
from xgbnew.sweep_ensemble_grid import (
    FEE_REGIMES,
    compute_goodness,
    compute_robust_goodness,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more cross_attn checkpoint .pt files")
    p.add_argument("--panel", type=Path,
                   default=Path("analysis/cross_attn_transformer/panel_v1.npz"))
    p.add_argument("--symbols-file",
                   default=Path("symbol_lists/stocks_wide_1000_v1.txt"))
    p.add_argument("--data-root", default=Path("trainingdata"))
    p.add_argument("--oos-start", type=str, default="2025-01-02")
    p.add_argument("--oos-end", type=str, default="2026-04-28")
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--leverage-grid", type=str, default="1.0,1.5,2.0")
    p.add_argument("--min-score-grid", type=str, default="0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--fee-regime", type=str, default="deploy",
                   choices=list(FEE_REGIMES.keys()))
    p.add_argument("--hold-through", action="store_true", default=True)
    p.add_argument("--no-hold-through", dest="hold_through", action="store_false")
    p.add_argument("--out-json", required=True, type=Path)
    return p.parse_args()


@torch.no_grad()
def score_panel_with_model(
    model: CrossAttnTransformerV1,
    X_norm_t: torch.Tensor,
    valid_t: torch.Tensor,
    seq_len: int,
    day_indices: list[int],
    *,
    chunk_size: int = 1,
) -> np.ndarray:
    """Return scores of shape (D, S) — sigmoid probability for each (day, symbol).

    Only rows in ``day_indices`` are scored; other days remain 0.0.
    """
    D, S = valid_t.shape
    out = np.zeros((D, S), dtype=np.float32)
    model.eval()
    for d in day_indices:
        mask = valid_t[d]
        if int(mask.sum().item()) < 1:
            continue
        start = d - seq_len + 1
        if start < 0:
            continue
        window = X_norm_t[start:d + 1].transpose(0, 1).contiguous()  # (S, T, F)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(window, valid_mask=mask)
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        out[d, :] = prob
        # zero-out positions where the symbol was not active that day
        out[d, ~mask.cpu().numpy()] = 0.0
    return out


def load_model(ckpt_path: Path, device: torch.device) -> CrossAttnTransformerV1:
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = CrossAttnConfig(**obj["config"])
    model = CrossAttnTransformerV1(cfg).to(device)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model


def _build_windows(days, window_days: int, stride_days: int):
    if len(days) < window_days:
        return []
    out = []
    i = 0
    while i + window_days <= len(days):
        span = days[i: i + window_days]
        out.append((span[0], span[-1]))
        i += stride_days
    return out


def _monthly_return(total_pct: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    return (1.0 + total_pct / 100.0) ** (21.0 / n_days) - 1.0


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda")

    panel = np.load(args.panel, allow_pickle=True)
    X = panel["X"]
    valid = panel["valid"]
    dates = panel["dates"]            # datetime64[D]
    symbols = panel["symbols"]
    train_end_idx = int((dates <= np.datetime64(str(panel["train_end"]))).sum())

    # Re-derive normalization from each checkpoint (they were trained on the
    # same panel so feature_mean/std should be identical, but we re-load them
    # from the first checkpoint to be safe).
    first_ckpt = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    mean = first_ckpt["feature_mean"]
    std = first_ckpt["feature_std"]
    cfg_first = CrossAttnConfig(**first_ckpt["config"])
    seq_len = cfg_first.seq_len

    X_norm = ((X - mean) / std).astype(np.float32)
    np.clip(X_norm, -5.0, 5.0, out=X_norm)
    X_norm[~valid] = 0.0
    X_norm_t = torch.from_numpy(X_norm).to(device)
    valid_t = torch.from_numpy(valid).to(device)

    # Determine OOS day indices
    oos_start = np.datetime64(args.oos_start)
    oos_end = np.datetime64(args.oos_end)
    oos_mask = (dates >= oos_start) & (dates <= oos_end)
    oos_idx = np.where(oos_mask)[0].tolist()
    oos_idx = [d for d in oos_idx if d - seq_len + 1 >= 0]
    logger.info("OOS days: %d (first=%s, last=%s)",
                len(oos_idx), dates[oos_idx[0]], dates[oos_idx[-1]])

    # Score every checkpoint, then average to form the ensemble probability.
    ensemble_scores = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    for ck in args.checkpoints:
        logger.info("scoring with %s", ck)
        model = load_model(Path(ck), device)
        t0 = time.time()
        s = score_panel_with_model(
            model, X_norm_t, valid_t, seq_len, oos_idx,
        )
        logger.info("  scored in %.1fs", time.time() - t0)
        ensemble_scores += s
        del model
        torch.cuda.empty_cache()
    ensemble_scores /= float(len(args.checkpoints))

    # Build the OOS DataFrame (same shape XGB sweep uses).
    syms_list = [s.upper() for s in symbols]
    syms_file = list({l.strip().upper() for l in open(args.symbols_file)
                      if l.strip() and not l.startswith("#")})
    use_syms = sorted(set(syms_list) & set(syms_file))
    logger.info("symbols intersection: %d", len(use_syms))

    # Re-build the OOS DataFrame using the same xgbnew dataset pipeline so
    # backtest sees actual_open/actual_close/spread_bps/dolvol_20d_log etc.
    _, _, oos_df = build_daily_dataset(
        Path(args.data_root), syms_list,
        train_start=date(2020, 1, 1),
        train_end=date(2020, 1, 2),                            # no-op
        val_start=date(2020, 1, 3), val_end=date(2020, 1, 4),  # no-op
        test_start=pd.Timestamp(args.oos_start).date(),
        test_end=pd.Timestamp(args.oos_end).date(),
        fast_features=True,
    )
    if oos_df is None or len(oos_df) == 0:
        raise RuntimeError("OOS DataFrame empty")
    oos_df = oos_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Build (date, symbol) → score lookup from ensemble_scores.
    sym_to_idx = {s: i for i, s in enumerate(syms_list)}
    date_to_idx = {pd.Timestamp(d).date(): i for i, d in enumerate(dates)}
    oos_df_dates = oos_df["date"].to_numpy()
    oos_df_syms = oos_df["symbol"].to_numpy()
    score_arr = np.zeros(len(oos_df), dtype=np.float64)
    for i in range(len(oos_df)):
        di = date_to_idx.get(oos_df_dates[i], -1)
        si = sym_to_idx.get(oos_df_syms[i], -1)
        if di >= 0 and si >= 0:
            score_arr[i] = ensemble_scores[di, si]
    oos_df["_score"] = score_arr
    scores = pd.Series(score_arr, index=oos_df.index, name="ensemble_score")

    # Build trading days in OOS window
    days = sorted(pd.unique(oos_df["date"]))
    windows = _build_windows(days, args.window_days, args.stride_days)
    logger.info("backtest windows: %d  (window_days=%d stride=%d)",
                len(windows), args.window_days, args.stride_days)

    fees = FEE_REGIMES[args.fee_regime]
    leverage_grid = [float(x) for x in args.leverage_grid.split(",")]
    min_score_grid = [float(x) for x in args.min_score_grid.split(",")]

    cells = []
    for lev in leverage_grid:
        for ms in min_score_grid:
            cfg = BacktestConfig(
                top_n=int(args.top_n),
                leverage=float(lev),
                xgb_weight=1.0,
                fee_rate=float(fees["fee_rate"]),
                fill_buffer_bps=float(fees["fill_buffer_bps"]),
                commission_bps=float(fees["commission_bps"]),
                hold_through=bool(args.hold_through),
                min_score=float(ms),
            )
            monthlies: list[float] = []
            sortinos: list[float] = []
            dds: list[float] = []
            window_starts: list[str] = []
            for w_start, w_end in windows:
                w_df = oos_df[(oos_df["date"] >= w_start) & (oos_df["date"] <= w_end)]
                if len(w_df) < 5:
                    continue
                w_scores = scores.loc[w_df.index]
                res = simulate(w_df, None, cfg, precomputed_scores=w_scores)
                # Elapsed days: prefer all unique dates in the slice
                elapsed = max(len(pd.unique(w_df["date"])), 1)
                monthly = _monthly_return(res.total_return_pct, elapsed) * 100.0
                monthlies.append(monthly)
                sortinos.append(res.sortino_ratio)
                dds.append(res.max_drawdown_pct)
                window_starts.append(pd.Timestamp(w_start).date().isoformat())

            n = len(monthlies)
            if n == 0:
                continue
            arr = np.array(monthlies)
            p10 = float(np.percentile(arr, 10))
            worst_dd = float(np.max(dds))
            n_neg = int(np.sum(arr < 0))
            goodness = compute_goodness(p10, worst_dd, n_neg, n)
            robust = compute_robust_goodness(arr, worst_dd)
            cell = dict(
                leverage=lev, min_score=ms, top_n=args.top_n,
                fee_regime=args.fee_regime, hold_through=bool(args.hold_through),
                n_windows=n,
                median_monthly_pct=float(np.median(arr)),
                p10_monthly_pct=p10,
                worst_dd_pct=worst_dd,
                n_neg=n_neg,
                median_sortino=float(np.median(sortinos)),
                goodness_score=float(goodness),
                robust_goodness_score=float(robust),
                monthly_return_pcts=[float(x) for x in arr],
                window_drawdown_pcts=[float(x) for x in dds],
                window_start_dates=window_starts,
            )
            cells.append(cell)
            logger.info(
                "lev=%.2f ms=%.2f  med=%+.2f%%/mo  p10=%+.2f  neg=%d/%d  ddW=%.2f  sortino=%.2f  good=%+.2f",
                lev, ms, cell["median_monthly_pct"], p10, n_neg, n, worst_dd,
                cell["median_sortino"], goodness,
            )

    out = dict(
        checkpoints=[str(c) for c in args.checkpoints],
        panel=str(args.panel),
        oos_start=args.oos_start, oos_end=args.oos_end,
        window_days=int(args.window_days), stride_days=int(args.stride_days),
        top_n=int(args.top_n), fee_regime=args.fee_regime,
        hold_through=bool(args.hold_through),
        cells=cells,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s (cells=%d)", args.out_json, len(cells))


if __name__ == "__main__":
    main()
