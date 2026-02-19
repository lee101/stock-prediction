#!/usr/bin/env python3
"""Compare SUI OHLC forecast MAE with and without BTC/ETH/SOL covariates.

Tests 3 approaches:
1. Univariate: SUI OHLC only (current production)
2. Multivariate: SUI OHLC jointly predicted (no cross-symbol)
3. Cross-learning: SUI + BTC + ETH + SOL jointly with cross_learning=True
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path("trainingdatahourlybinance")
SYMBOLS = ["SUIUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"]
TARGET = "SUIUSDT"
HORIZONS = [1, 4, 24]
CONTEXT_LENGTH = 512
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
HOLDOUT_HOURS = 24 * 30  # 30 days holdout for evaluation


def load_ohlc(symbol: str) -> pd.DataFrame:
    csv_path = DATA_ROOT / f"{symbol}.csv"
    if not csv_path.exists():
        alt = symbol.replace("USDT", "USD")
        csv_path = Path("trainingdatahourly/crypto") / f"{alt}.csv"
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def compute_mae(actual: pd.DataFrame, predicted: dict, targets=("close", "high", "low")) -> dict:
    results = {}
    for t in targets:
        key = f"predicted_{t}_p50"
        if key in predicted:
            mask = actual[t].notna() & pd.Series(predicted[key]).notna()
            if mask.sum() > 0:
                mae = np.mean(np.abs(actual[t].values[mask] - np.array(predicted[key])[mask]))
                mae_pct = mae / np.mean(np.abs(actual[t].values[mask])) * 100
                results[f"{t}_mae"] = float(mae)
                results[f"{t}_mae_pct"] = float(mae_pct)
    return results


def eval_univariate(wrapper, sui_df: pd.DataFrame, eval_df: pd.DataFrame, horizon: int) -> dict:
    """Current approach: predict SUI OHLC independently."""
    context = sui_df.tail(CONTEXT_LENGTH).copy()
    context["symbol"] = TARGET
    batch = wrapper.predict_ohlc(
        context, symbol=TARGET, prediction_length=horizon,
        context_length=CONTEXT_LENGTH, quantile_levels=QUANTILE_LEVELS,
    )
    preds = {}
    for col in ["close", "high", "low"]:
        key = f"predicted_{col}_p50"
        med = batch.median
        if isinstance(med, pd.DataFrame) and col in med.columns:
            preds[key] = med[col].values.tolist()
        elif hasattr(batch, "quantile"):
            q = batch.quantile(0.5)
            if isinstance(q, pd.DataFrame) and col in q.columns:
                preds[key] = q[col].values.tolist()
    return preds


def eval_multivariate(wrapper, sui_df: pd.DataFrame, eval_df: pd.DataFrame, horizon: int) -> dict:
    """Predict SUI OHLC jointly (multivariate, no cross-symbol)."""
    context = sui_df.tail(CONTEXT_LENGTH).copy()
    context["symbol"] = TARGET
    batch = wrapper.predict_ohlc_multivariate(
        context, symbol=TARGET, prediction_length=horizon,
        context_length=CONTEXT_LENGTH, quantile_levels=QUANTILE_LEVELS,
    )
    preds = {}
    for col in ["close", "high", "low"]:
        key = f"predicted_{col}_p50"
        med = batch.median
        if isinstance(med, pd.DataFrame) and col in med.columns:
            preds[key] = med[col].values.tolist()
    return preds


def eval_cross_learning(wrapper, all_dfs: dict, eval_df: pd.DataFrame, horizon: int) -> dict:
    """Predict SUI with BTC/ETH/SOL cross-learning."""
    contexts = []
    symbols = []
    for sym in SYMBOLS:
        df = all_dfs[sym].tail(CONTEXT_LENGTH).copy()
        df["symbol"] = sym
        contexts.append(df)
        symbols.append(sym)

    results = wrapper.predict_ohlc_joint(
        contexts, symbols=symbols, prediction_length=horizon,
        context_length=CONTEXT_LENGTH, quantile_levels=QUANTILE_LEVELS,
        predict_batches_jointly=True,
    )
    sui_batch = results[0]
    preds = {}
    for col in ["close", "high", "low"]:
        key = f"predicted_{col}_p50"
        med = sui_batch.median
        if isinstance(med, pd.DataFrame) and col in med.columns:
            preds[key] = med[col].values.tolist()
    return preds


def main():
    from src.models.chronos2_wrapper import Chronos2OHLCWrapper
    import os

    model_id = os.environ.get("CHRONOS2_MODEL_ID_OVERRIDE",
        "chronos2_finetuned/binance_lora_20260208_newpairs_SUIUSDT/finetuned-ckpt")
    logger.info("Loading Chronos2 from {}", model_id)
    wrapper = Chronos2OHLCWrapper.from_pretrained(model_id=model_id, device_map="cuda")

    all_dfs = {}
    for sym in SYMBOLS:
        df = load_ohlc(sym)
        logger.info("{}: {} rows, {} to {}", sym, len(df),
                     df["timestamp"].min(), df["timestamp"].max())
        all_dfs[sym] = df

    sui_full = all_dfs[TARGET]
    # Align all series to common timestamps
    common_ts = set(sui_full["timestamp"])
    for sym in SYMBOLS:
        common_ts &= set(all_dfs[sym]["timestamp"])
    common_ts = sorted(common_ts)
    logger.info("Common timestamps: {}", len(common_ts))

    for sym in SYMBOLS:
        all_dfs[sym] = all_dfs[sym][all_dfs[sym]["timestamp"].isin(common_ts)].reset_index(drop=True)

    sui_full = all_dfs[TARGET]

    # Rolling evaluation over holdout window
    n_total = len(sui_full)
    n_eval_start = n_total - HOLDOUT_HOURS
    eval_points = list(range(n_eval_start, n_total - max(HORIZONS), 24))  # every 24h
    logger.info("Eval points: {} (every 24h over {}d holdout)", len(eval_points), HOLDOUT_HOURS // 24)

    all_results = {h: {"univariate": [], "multivariate": [], "cross_learning": []} for h in HORIZONS}

    for i, idx in enumerate(eval_points):
        for horizon in HORIZONS:
            if idx + horizon > n_total:
                continue
            context_end = idx
            actual_slice = sui_full.iloc[context_end:context_end + horizon]
            if len(actual_slice) < horizon:
                continue

            sui_context = sui_full.iloc[:context_end]
            context_dfs = {sym: all_dfs[sym].iloc[:context_end] for sym in SYMBOLS}

            try:
                uni_preds = eval_univariate(wrapper, sui_context, actual_slice, horizon)
                uni_mae = compute_mae(actual_slice, uni_preds)
                all_results[horizon]["univariate"].append(uni_mae)
            except Exception as e:
                logger.warning("Univariate h{} point {} failed: {}", horizon, i, e)

            try:
                mv_preds = eval_multivariate(wrapper, sui_context, actual_slice, horizon)
                mv_mae = compute_mae(actual_slice, mv_preds)
                all_results[horizon]["multivariate"].append(mv_mae)
            except Exception as e:
                logger.warning("Multivariate h{} point {} failed: {}", horizon, i, e)

            try:
                cl_preds = eval_cross_learning(wrapper, context_dfs, actual_slice, horizon)
                cl_mae = compute_mae(actual_slice, cl_preds)
                all_results[horizon]["cross_learning"].append(cl_mae)
            except Exception as e:
                logger.warning("Cross-learning h{} point {} failed: {}", horizon, i, e)

        if (i + 1) % 5 == 0:
            logger.info("Progress: {}/{} eval points", i + 1, len(eval_points))

    # Aggregate results
    summary = {}
    for horizon in HORIZONS:
        summary[f"h{horizon}"] = {}
        for method in ["univariate", "multivariate", "cross_learning"]:
            entries = all_results[horizon][method]
            if not entries:
                continue
            agg = {}
            for key in entries[0]:
                vals = [e[key] for e in entries if key in e]
                if vals:
                    agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            summary[f"h{horizon}"][method] = agg

    # Print comparison
    print("\n=== SUI OHLC FORECAST MAE COMPARISON ===")
    for horizon in HORIZONS:
        print(f"\n--- h{horizon} ---")
        for method in ["univariate", "multivariate", "cross_learning"]:
            data = summary.get(f"h{horizon}", {}).get(method, {})
            if not data:
                print(f"  {method}: NO DATA")
                continue
            close_mae = data.get("close_mae_pct", {}).get("mean", -1)
            high_mae = data.get("high_mae_pct", {}).get("mean", -1)
            low_mae = data.get("low_mae_pct", {}).get("mean", -1)
            print(f"  {method}: close={close_mae:.3f}% high={high_mae:.3f}% low={low_mae:.3f}%")

    output_path = Path("binanceleveragesui/covariate_forecast_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
