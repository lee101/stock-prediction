#!/usr/bin/env python3
"""Simulate a meta-strategy that switches between DOGE and AAVE based on trailing performance."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost

DATA_ROOT = REPO / "trainingdatahourlybinance"
FORECAST_CACHE = REPO / "binanceneural/forecast_cache"
MARGIN_HOURLY_RATE = 0.0000025457

DOGE_CKPT = REPO / "binanceleveragesui/checkpoints/DOGEUSD_r4_R4_h384_cosine/binanceneural_20260301_065549/epoch_001.pt"
AAVE_CKPT = REPO / "binanceleveragesui/checkpoints/AAVEUSD_sweep_AAVE_h384_cosine_rw05/binanceneural_20260302_102218/epoch_003.pt"


def load_model_and_actions(ckpt_path, symbol, val_days=30, test_days=90):
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
    return bars, actions


def run_individual_sim(bars, actions, symbol, maker_fee, max_leverage):
    cfg = LeverageConfig(
        symbol=symbol, max_leverage=max_leverage, can_short=False,
        maker_fee=maker_fee, margin_hourly_rate=MARGIN_HOURLY_RATE if max_leverage > 1 else 0.0,
        initial_cash=10000.0, fill_buffer_pct=0.0005,
        decision_lag_bars=1, min_edge=0.0, max_hold_bars=6,
        intensity_scale=5.0,
    )
    return simulate_with_margin_cost(bars, actions, cfg)


def per_bar_sim(bars, actions, maker_fee, max_leverage, symbol):
    """Run sim and return per-bar equity array aligned with timestamps."""
    bars_s = bars.sort_values("timestamp").reset_index(drop=True)
    actions_s = actions.sort_values("timestamp").reset_index(drop=True)
    merged = bars_s.merge(actions_s, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))

    if True:  # decision_lag_bars=1
        for col in ['buy_price', 'sell_price', 'buy_amount', 'sell_amount']:
            if col in merged.columns:
                merged[col] = merged[col].shift(1)
        merged = merged.dropna(subset=['buy_price']).reset_index(drop=True)

    cash = 10000.0
    inventory = 0.0
    fill_buf = 0.0005
    iscale = 5.0
    margin_rate = MARGIN_HOURLY_RATE if max_leverage > 1 else 0.0
    bars_held = 0

    records = []
    for _, row in merged.iterrows():
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        buy_price = float(row.get("buy_price", 0) or 0)
        sell_price = float(row.get("sell_price", 0) or 0)
        buy_amount = min(float(row.get("buy_amount", 0) or 0) * iscale, 100.0) / 100.0
        sell_amount = min(float(row.get("sell_amount", 0) or 0) * iscale, 100.0) / 100.0

        equity = cash + inventory * close

        if cash < 0:
            interest = abs(cash) * margin_rate
            cash -= interest

        # Force close
        if inventory > 0 and bars_held >= 6:
            fp = close * 0.999
            cash += inventory * fp * (1 - maker_fee)
            inventory = 0.0
            bars_held = 0
            records.append({"timestamp": row["timestamp"], "equity": cash, "in_position": False})
            continue

        sold = False
        if sell_amount > 0 and sell_price > 0 and high >= sell_price * (1 + fill_buf):
            if inventory > 0:
                sq = min(sell_amount * inventory, inventory)
                if sq > 0:
                    cash += sq * sell_price * (1 - maker_fee)
                    inventory -= sq
                    sold = True
                    if inventory <= 0:
                        bars_held = 0

        if not sold and buy_amount > 0 and buy_price > 0 and low <= buy_price * (1 - fill_buf):
            max_bv = max_leverage * max(equity, 0) - inventory * buy_price
            if max_bv > 0:
                bq = buy_amount * max_bv / (buy_price * (1 + maker_fee))
                if bq > 0:
                    cash -= bq * buy_price * (1 + maker_fee)
                    inventory += bq

        if inventory > 0:
            bars_held += 1
        else:
            bars_held = 0

        records.append({
            "timestamp": row["timestamp"],
            "equity": cash + inventory * close,
            "in_position": inventory > 0,
        })

    return pd.DataFrame(records)


def meta_sim(doge_bars_df, aave_bars_df, max_leverage=2.0, lookback_hours=168, sit_out_if_negative=False):
    """Simulate switching between DOGE and AAVE based on trailing performance.
    If sit_out_if_negative=True, hold cash when both models have negative trailing sortino."""
    common_ts = set(doge_bars_df["timestamp"]) & set(aave_bars_df["timestamp"])
    doge_bars_df = doge_bars_df[doge_bars_df["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    aave_bars_df = aave_bars_df[aave_bars_df["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)

    n = len(doge_bars_df)
    doge_eq = doge_bars_df["equity"].values
    aave_eq = aave_bars_df["equity"].values

    def trailing_sortino(eq_arr, end_idx, window):
        start = max(0, end_idx - window)
        if end_idx - start < 24:
            return 0.0
        sub = eq_arr[start:end_idx+1]
        rets = np.diff(sub) / (np.abs(sub[:-1]) + 1e-10)
        if len(rets) < 2:
            return 0.0
        neg = rets[rets < 0]
        dd = np.std(neg) if len(neg) > 1 else 1e-10
        return float(np.mean(rets)) / (dd + 1e-10)

    doge_in = doge_bars_df["in_position"].values
    aave_in = aave_bars_df["in_position"].values

    active = None  # "doge", "aave", or "cash"
    meta_equity = [10000.0]
    switch_log = []
    bars_in_cash = 0

    doge_rel = np.ones(n)
    aave_rel = np.ones(n)
    for i in range(1, n):
        doge_rel[i] = doge_eq[i] / (doge_eq[i-1] + 1e-10) if doge_eq[i-1] > 0 else 1.0
        aave_rel[i] = aave_eq[i] / (aave_eq[i-1] + 1e-10) if aave_eq[i-1] > 0 else 1.0

    for i in range(n):
        ds = trailing_sortino(doge_eq, i, lookback_hours)
        as_ = trailing_sortino(aave_eq, i, lookback_hours)

        if active is None:
            if sit_out_if_negative and ds <= 0 and as_ <= 0:
                active = "cash"
            else:
                active = "aave" if as_ > ds else "doge"
            switch_log.append((i, active, ds, as_))

        if active == "cash":
            meta_equity.append(meta_equity[-1])
            bars_in_cash += 1
        else:
            rel = doge_rel[i] if active == "doge" else aave_rel[i]
            meta_equity.append(meta_equity[-1] * rel)

        in_pos = (doge_in[i] if active == "doge" else aave_in[i]) if active != "cash" else False
        was_in = False
        if i > 0 and active != "cash":
            was_in = doge_in[i-1] if active == "doge" else aave_in[i-1]

        should_reeval = (was_in and not in_pos) or (active == "cash" and i % 24 == 0)

        if should_reeval:
            if sit_out_if_negative and ds <= 0 and as_ <= 0:
                new_active = "cash"
            else:
                new_active = "aave" if as_ > ds else "doge"
            if new_active != active:
                switch_log.append((i, new_active, ds, as_))
                active = new_active

    meta_equity = np.array(meta_equity[1:])
    timestamps = doge_bars_df["timestamp"].values

    return meta_equity, timestamps, switch_log, bars_in_cash


def compute_stats(eq_arr, label):
    ret = np.diff(eq_arr) / (np.abs(eq_arr[:-1]) + 1e-10)
    total_ret = (eq_arr[-1] - eq_arr[0]) / eq_arr[0]
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / (peak + 1e-10)
    max_dd = dd.min()
    neg = ret[ret < 0]
    dd_std = np.std(neg) if len(neg) > 1 else 1e-10
    sortino = float(np.mean(ret)) / (dd_std + 1e-10)
    return {
        "label": label,
        "sortino": round(sortino, 2),
        "total_return": round(total_ret * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "final_eq": round(float(eq_arr[-1]), 2),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--lookback", type=int, default=168, help="trailing hours for model selection")
    args = parser.parse_args()

    logger.info("Loading DOGE model...")
    doge_bars, doge_actions = load_model_and_actions(DOGE_CKPT, "DOGEUSD", test_days=args.test_days)
    logger.info("Loading AAVE model...")
    aave_bars, aave_actions = load_model_and_actions(AAVE_CKPT, "AAVEUSD", test_days=args.test_days)

    logger.info("Running per-bar sims...")
    doge_pbar = per_bar_sim(doge_bars, doge_actions, maker_fee=0.001, max_leverage=args.max_leverage, symbol="DOGEUSD")
    aave_pbar = per_bar_sim(aave_bars, aave_actions, maker_fee=0.001, max_leverage=args.max_leverage, symbol="AAVEUSD")

    logger.info(f"DOGE: {len(doge_pbar)} bars, AAVE: {len(aave_pbar)} bars")

    doge_s = compute_stats(doge_pbar["equity"].values, "DOGE_only")
    aave_s = compute_stats(aave_pbar["equity"].values, "AAVE_only")
    logger.info(f"  DOGE only: Sort={doge_s['sortino']:>7.1f} Ret={doge_s['total_return']:>+8.1f}% DD={doge_s['max_dd']:>6.1f}%")
    logger.info(f"  AAVE only: Sort={aave_s['sortino']:>7.1f} Ret={aave_s['total_return']:>+8.1f}% DD={aave_s['max_dd']:>6.1f}%")

    for test_d in [30, 60, 90, 120]:
        logger.info(f"\n=== {test_d}d window ===")
        doge_b, doge_a = load_model_and_actions(DOGE_CKPT, "DOGEUSD", test_days=test_d)
        aave_b, aave_a = load_model_and_actions(AAVE_CKPT, "AAVEUSD", test_days=test_d)
        dp = per_bar_sim(doge_b, doge_a, 0.001, args.max_leverage, "DOGEUSD")
        ap = per_bar_sim(aave_b, aave_a, 0.001, args.max_leverage, "AAVEUSD")

        ds = compute_stats(dp["equity"].values, "DOGE")
        as_ = compute_stats(ap["equity"].values, "AAVE")
        logger.info(f"  DOGE:       Sort={ds['sortino']:>6.2f} Ret={ds['total_return']:>+8.1f}% DD={ds['max_dd']:>6.1f}%")
        logger.info(f"  AAVE:       Sort={as_['sortino']:>6.2f} Ret={as_['total_return']:>+8.1f}% DD={as_['max_dd']:>6.1f}%")

        for lb in [6, 12, 24, 48, 72, 168, 336, 720]:
            meta_eq, _, sw, _ = meta_sim(dp, ap, args.max_leverage, lb, sit_out_if_negative=False)
            ms = compute_stats(meta_eq, f"meta_{lb}h")
            meta_eq2, _, sw2, cash_bars = meta_sim(dp, ap, args.max_leverage, lb, sit_out_if_negative=True)
            ms2 = compute_stats(meta_eq2, f"meta_{lb}h_safe")
            logger.info(f"  META {lb:>3}h:   Sort={ms['sortino']:>6.2f} Ret={ms['total_return']:>+8.1f}% DD={ms['max_dd']:>6.1f}% sw={len(sw)}")
            logger.info(f"  SAFE {lb:>3}h:   Sort={ms2['sortino']:>6.2f} Ret={ms2['total_return']:>+8.1f}% DD={ms2['max_dd']:>6.1f}% sw={len(sw2)} cash={cash_bars}h")


if __name__ == "__main__":
    main()
