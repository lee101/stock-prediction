#!/usr/bin/env python3
"""Validate a checkpoint across many time windows (1d, 7d, 30d, 60d, 120d, 150d)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost

DATA_ROOT = REPO / "trainingdatahourlybinance"
FORECAST_CACHE = REPO / "binanceneural/forecast_cache"

CKPT = REPO / "binanceleveragesui/checkpoints/DOGEUSD_r4_R4_h384_cosine/binanceneural_20260301_065549"
EVAL_SYMBOLS = ["DOGEUSD", "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "AAVEUSD"]
WINDOWS = [1, 7, 30, 60, 120, 150]
SAMPLE_EPOCHS = [1, 2, 3, 5, 8, 10, 15, 20]


def eval_window(ckpt_path, symbol, test_days, val_days=30):
    model, normalizer, feature_columns, meta = load_policy_checkpoint(ckpt_path, device="cuda")
    seq_len = meta.get("sequence_length", 72)
    dm = ChronosSolDataModule(
        symbol=symbol, data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
        context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32, model_id="amazon/chronos-t5-small",
        sequence_length=seq_len,
        split_config=SplitConfig(val_days=val_days, test_days=test_days),
        cache_only=True, max_history_days=365,
    )
    actions = generate_actions_from_frame(
        model=model, frame=dm.test_frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=seq_len, horizon=1,
    )
    bars = dm.test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
    cfg = LeverageConfig(
        symbol=symbol, max_leverage=1.0, can_short=False,
        maker_fee=0.001, margin_hourly_rate=0.0, initial_cash=10000.0,
        fill_buffer_pct=0.0013, decision_lag_bars=1, min_edge=0.0,
        max_hold_bars=6, intensity_scale=5.0,
    )
    r = simulate_with_margin_cost(bars, actions, cfg)
    return r


def main():
    # first find best epoch across sample epochs on DOGE 30d
    logger.info("Finding best epoch on DOGE 30d...")
    best_ep, best_sort = 0, -999
    for ep in SAMPLE_EPOCHS:
        ckpt = CKPT / f"epoch_{ep:03d}.pt"
        if not ckpt.exists():
            continue
        try:
            r = eval_window(ckpt, "DOGEUSD", test_days=30)
            s = r["sortino"]
            logger.info(f"  ep{ep}: sort={s:.1f} ret={r['total_return']*100:.1f}% dd={r['max_drawdown']*100:.1f}%")
            if s > best_sort:
                best_sort = s
                best_ep = ep
        except Exception as e:
            logger.warning(f"  ep{ep} failed: {e}")

    logger.info(f"\nBest epoch: {best_ep} (sort={best_sort:.1f})")
    best_ckpt = CKPT / f"epoch_{best_ep:03d}.pt"

    # multi-window eval on all symbols
    logger.info(f"\n{'='*100}")
    logger.info(f"MULTI-WINDOW VALIDATION: R4_h384_cosine ep{best_ep}")
    logger.info(f"{'='*100}")

    header = f"{'Symbol':<10}"
    for w in WINDOWS:
        header += f" {'%dd'%w:>10}"
    logger.info(header)
    logger.info("-" * (10 + 11 * len(WINDOWS)))

    all_sortinos = {w: [] for w in WINDOWS}
    all_returns = {w: [] for w in WINDOWS}
    all_dds = {w: [] for w in WINDOWS}

    for sym in EVAL_SYMBOLS:
        row = f"{sym:<10}"
        for w in WINDOWS:
            try:
                r = eval_window(best_ckpt, sym, test_days=w)
                s = r["sortino"]
                ret = r["total_return"] * 100
                dd = r["max_drawdown"] * 100
                all_sortinos[w].append(s)
                all_returns[w].append(ret)
                all_dds[w].append(dd)
                row += f" {s:>6.1f}/{ret:>+.0f}%"
            except Exception as e:
                row += f" {'ERR':>10}"
                all_sortinos[w].append(-999)
        logger.info(row)

    # summary stats per window
    logger.info("-" * (10 + 11 * len(WINDOWS)))
    mean_row = f"{'Mean':.<10}"
    pos_row = f"{'Positive':.<10}"
    dd_row = f"{'Worst DD':.<10}"
    for w in WINDOWS:
        vals = [x for x in all_sortinos[w] if x > -999]
        mean_row += f" {np.mean(vals):>10.1f}" if vals else f" {'N/A':>10}"
        pos_row += f" {sum(1 for x in vals if x > 0)}/{len(vals):>8}"
        dds = all_dds.get(w, [])
        dd_row += f" {min(dds):>9.1f}%" if dds else f" {'N/A':>10}"
    logger.info(mean_row)
    logger.info(pos_row)
    logger.info(dd_row)

    # also test leverage at 2x on best window
    logger.info(f"\n=== Leverage test (DOGE, 30d) ===")
    for lev in [1.0, 1.5, 2.0, 3.0]:
        model, normalizer, feature_columns, meta = load_policy_checkpoint(best_ckpt, device="cuda")
        seq_len = meta.get("sequence_length", 72)
        dm = ChronosSolDataModule(
            symbol="DOGEUSD", data_root=DATA_ROOT,
            forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
            context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32, model_id="amazon/chronos-t5-small",
            sequence_length=seq_len,
            split_config=SplitConfig(val_days=30, test_days=30),
            cache_only=True, max_history_days=365,
        )
        actions = generate_actions_from_frame(
            model=model, frame=dm.test_frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=seq_len, horizon=1,
        )
        bars = dm.test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
        cfg = LeverageConfig(
            symbol="DOGEUSD", max_leverage=lev, can_short=False,
            maker_fee=0.001, margin_hourly_rate=0.0000025457 if lev > 1 else 0.0,
            initial_cash=10000.0, fill_buffer_pct=0.0013, decision_lag_bars=1,
            min_edge=0.0, max_hold_bars=6, intensity_scale=5.0,
        )
        r = simulate_with_margin_cost(bars, actions, cfg)
        logger.info(f"  {lev}x: Sort={r['sortino']:.1f} Ret={r['total_return']*100:.1f}% DD={r['max_drawdown']*100:.1f}% Trades={r['num_trades']}")


if __name__ == "__main__":
    main()
