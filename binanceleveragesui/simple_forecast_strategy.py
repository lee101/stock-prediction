#!/usr/bin/env python3
"""Simple trading strategies from Chronos2 h1 forecasts.

No neural policy - forecast features -> buy/sell decisions.
Strategies: threshold, linear, quantile, gbm, vol_scaled
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost

DATA_ROOT = REPO / "trainingdatahourlybinance"
CACHE_ROOT = REPO / "binanceneural/forecast_cache"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "LTCUSDT", "UNIUSDT", "AAVEUSDT", "ATOMUSDT", "APTUSDT",
    "TRXUSDT", "SHIBUSDT", "SUIUSDT", "NEARUSDT",
    "OPUSDT", "ARBUSDT",
]

FEATURE_COLS = [
    "delta_close_p50_h1", "delta_high_p50_h1", "delta_low_p50_h1",
    "forecast_spread_h1", "return_1h", "return_4h", "return_24h",
    "volatility_24h", "range_pct",
    "delta_close_ma3", "delta_close_ma6", "momentum_6h",
]


def pair_to_data_symbol(pair: str) -> str:
    if pair == "SUIUSDT":
        return "SUIUSDT"
    usd = pair.rstrip("T") if pair.endswith("USDT") else pair
    if (DATA_ROOT / f"{usd}.csv").exists():
        return usd
    if (DATA_ROOT / f"{pair}.csv").exists():
        return pair
    return usd


def load_data(dsym: str) -> Optional[pd.DataFrame]:
    csv = DATA_ROOT / f"{dsym}.csv"
    if not csv.exists():
        return None
    price = pd.read_csv(csv, parse_dates=["timestamp"])
    if price["timestamp"].dt.tz is None:
        price["timestamp"] = price["timestamp"].dt.tz_localize("UTC")
    price["symbol"] = price["symbol"].fillna(dsym.upper())

    h1 = CACHE_ROOT / "h1" / f"{dsym.upper()}.parquet"
    if not h1.exists():
        return None
    fc = pd.read_parquet(h1)
    fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
    rename = {c: c + "_h1" for c in fc.columns if c.startswith("predicted_") and not c.endswith("_h1")}
    if rename:
        fc = fc.rename(columns=rename)

    merged = price.merge(fc, on=["timestamp", "symbol"], how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"{dsym}: {len(merged)} merged rows")
    return merged


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ref = df["close"].astype(float)

    for col in ["predicted_close_p50_h1", "predicted_high_p50_h1", "predicted_low_p50_h1"]:
        if col in df.columns:
            df[col.replace("predicted_", "delta_")] = (df[col] - ref) / ref

    if "predicted_close_p10_h1" in df.columns and "predicted_close_p90_h1" in df.columns:
        df["forecast_spread_h1"] = (df["predicted_close_p90_h1"] - df["predicted_close_p10_h1"]) / ref

    df["return_1h"] = ref.pct_change(1)
    df["return_4h"] = ref.pct_change(4)
    df["return_24h"] = ref.pct_change(24)
    df["volatility_24h"] = df["return_1h"].rolling(24).std()
    df["range_pct"] = (df["high"] - df["low"]).abs() / ref.replace(0, np.nan)

    # smoothed forecast deltas - reduces noise
    if "delta_close_p50_h1" in df.columns:
        df["delta_close_ma3"] = df["delta_close_p50_h1"].rolling(3).mean()
        df["delta_close_ma6"] = df["delta_close_p50_h1"].rolling(6).mean()

    # price momentum
    df["momentum_6h"] = ref.pct_change(6)

    keep = [c for c in FEATURE_COLS if c in df.columns]
    keep += ["return_1h", "return_4h", "return_24h", "volatility_24h", "range_pct"]
    keep = list(set(keep))
    return df.dropna(subset=keep).reset_index(drop=True)


def split_data(df: pd.DataFrame, test_days: int = 30, val_days: int = 30):
    end = df["timestamp"].max()
    test_start = end - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    train = df[df["timestamp"] < val_start].copy()
    val = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)].copy()
    test = df[df["timestamp"] >= test_start].copy()
    return train, val, test


def _get_feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def _make_actions_from_arrays(df, buy_prices, sell_prices, buy_amounts, sell_amounts):
    result = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "symbol": df["symbol"].values,
        "buy_price": np.maximum(buy_prices, 0),
        "sell_price": np.maximum(sell_prices, 0),
        "buy_amount": buy_amounts,
        "sell_amount": sell_amounts,
    })
    if result["timestamp"].dt.tz is None:
        result["timestamp"] = result["timestamp"].dt.tz_localize("UTC")
    return result


# ── Strategy: Threshold ──────────────────────────────────────────────────────

def threshold_actions(df, buy_thresh, edge_mult=1.0):
    close = df["close"].values.astype(float)
    dc = df["delta_close_p50_h1"].values.astype(float) if "delta_close_p50_h1" in df.columns else np.zeros(len(df))
    dh = df["delta_high_p50_h1"].values.astype(float) if "delta_high_p50_h1" in df.columns else np.zeros(len(df))
    dl = df["delta_low_p50_h1"].values.astype(float) if "delta_low_p50_h1" in df.columns else np.zeros(len(df))

    buy_mask = dc > buy_thresh
    sell_mask = dc < -buy_thresh

    bp = np.where(buy_mask, close * (1 + dl * edge_mult), 0.0)
    sp = np.where(buy_mask, close * (1 + dh * edge_mult),
                  np.where(sell_mask, close * (1 + dc * 0.5), 0.0))
    ba = np.where(buy_mask, np.minimum(np.abs(dc) / 0.01, 100.0), 0.0)
    sa = np.where(buy_mask | sell_mask, 100.0, 0.0)

    return _make_actions_from_arrays(df, bp, sp, ba, sa)


# ── Strategy: Linear ─────────────────────────────────────────────────────────

def fit_linear(train, feat_cols, horizon=1):
    from sklearn.linear_model import Ridge
    X = train[feat_cols].values
    y = train["close"].pct_change(horizon).shift(-horizon).values
    mask = np.isfinite(y)
    model = Ridge(alpha=1.0)
    model.fit(X[mask], y[mask])
    return model


def linear_actions(df, model, feat_cols, buy_thresh=0.001, scale=1.0):
    close = df["close"].values.astype(float)
    preds = model.predict(df[feat_cols].values)

    buy_mask = preds > buy_thresh
    sell_mask = preds < -buy_thresh

    bp = np.where(buy_mask, close * (1 - 0.001), 0.0)
    sp = np.where(buy_mask, close * (1 + np.abs(preds) * scale),
                  np.where(sell_mask, close * (1 + preds * 0.5), 0.0))
    ba = np.where(buy_mask, np.minimum(np.abs(preds) / 0.002 * 20, 100.0), 0.0)
    sa = np.where(buy_mask | sell_mask, 100.0, 0.0)

    return _make_actions_from_arrays(df, bp, sp, ba, sa)


# ── Strategy: GBM ────────────────────────────────────────────────────────────

def fit_gbm(train, feat_cols, horizon=1):
    from sklearn.ensemble import GradientBoostingRegressor
    X = train[feat_cols].values
    y = train["close"].pct_change(horizon).shift(-horizon).values
    mask = np.isfinite(y)
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=1337,
    )
    model.fit(X[mask], y[mask])
    return model


# ── Strategy: Quantile ───────────────────────────────────────────────────────

def quantile_actions(df, min_spread=0.003, edge_frac=0.5):
    close = df["close"].values.astype(float)
    p10 = df.get("predicted_close_p10_h1", pd.Series(close, index=df.index)).values.astype(float)
    p50 = df.get("predicted_close_p50_h1", pd.Series(close, index=df.index)).values.astype(float)
    p90 = df.get("predicted_close_p90_h1", pd.Series(close, index=df.index)).values.astype(float)
    ph = df.get("predicted_high_p50_h1", pd.Series(close, index=df.index)).values.astype(float)
    pl = df.get("predicted_low_p50_h1", pd.Series(close, index=df.index)).values.astype(float)

    spread = (p90 - p10) / close
    direction = (p50 - close) / close

    buy_mask = (direction > 0.0005) & (spread > min_spread)
    sell_mask = direction < -0.0005

    bp = np.where(buy_mask, close + (pl - close) * edge_frac, 0.0)
    sp = np.where(buy_mask, close + (ph - close) * edge_frac,
                  np.where(sell_mask, close + (p50 - close) * edge_frac, 0.0))
    confidence = np.minimum(direction / 0.005, 1.0)
    ba = np.where(buy_mask, confidence * 100.0, 0.0)
    sa = np.where(buy_mask | sell_mask, 100.0, 0.0)

    return _make_actions_from_arrays(df, bp, sp, ba, sa)


# ── Strategy: Vol-Scaled ─────────────────────────────────────────────────────

def vol_scaled_actions(df, vol_mult=1.0, min_delta=0.001):
    """Set buy/sell prices based on volatility bands + forecast direction."""
    close = df["close"].values.astype(float)
    dc = df["delta_close_p50_h1"].values.astype(float) if "delta_close_p50_h1" in df.columns else np.zeros(len(df))
    vol = df["volatility_24h"].values.astype(float) if "volatility_24h" in df.columns else np.full(len(df), 0.01)
    vol = np.maximum(vol, 0.001)

    buy_mask = dc > min_delta

    bp = np.where(buy_mask, close * (1 - vol * vol_mult), 0.0)
    sp = np.where(buy_mask, close * (1 + vol * vol_mult),
                  np.where(dc < -min_delta, close * (1 - vol * vol_mult * 0.5), 0.0))
    # size inversely proportional to vol (less vol = more confident)
    conf = np.minimum(np.abs(dc) / (vol + 1e-6), 2.0) / 2.0
    ba = np.where(buy_mask, conf * 100.0, 0.0)
    sa = np.where(buy_mask | (dc < -min_delta), 100.0, 0.0)

    return _make_actions_from_arrays(df, bp, sp, ba, sa)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(bars, actions, dsym, label):
    cfg = LeverageConfig(
        symbol=dsym, max_leverage=1.0, can_short=False,
        maker_fee=0.001, margin_hourly_rate=0.0, initial_cash=10000.0,
        fill_buffer_pct=0.0005, decision_lag_bars=1, min_edge=0.0,
        max_hold_bars=6, intensity_scale=5.0,
    )
    r = simulate_with_margin_cost(bars, actions, cfg)
    dd = abs(r["max_drawdown"]) if r["max_drawdown"] != 0 else 1e-6
    r["calmar"] = r["total_return"] / dd
    r["composite"] = r["sortino"] * (1 - min(dd, 1.0))
    r["label"] = label
    r["symbol"] = dsym
    return r


def sweep_symbol(pair: str, strategies: List[str] = None) -> List[dict]:
    dsym = pair_to_data_symbol(pair)
    df = load_data(dsym)
    if df is None:
        return []

    df = add_features(df)
    train, val, test = split_data(df)
    logger.info(f"{dsym}: train={len(train)} val={len(val)} test={len(test)}")

    if len(test) < 48:
        logger.warning(f"{dsym}: test too small ({len(test)})")
        return []

    bars = test[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
    feat_cols = _get_feat_cols(train)
    results = []

    if strategies is None:
        strategies = ["threshold", "linear", "gbm", "quantile", "vol_scaled"]

    if "threshold" in strategies:
        for bt in [0.0005, 0.001, 0.002, 0.003]:
            for em in [0.5, 1.0, 1.5]:
                acts = threshold_actions(test, buy_thresh=bt, edge_mult=em)
                r = evaluate(bars, acts, dsym, f"thresh_bt{bt}_em{em}")
                r["strategy"] = "threshold"
                results.append(r)

    if "linear" in strategies and len(feat_cols) >= 3 and len(train) > 100:
        for horizon in [1, 3, 6]:
            model = fit_linear(train, feat_cols, horizon=horizon)
            for bt in [0.0005, 0.001, 0.002]:
                for sc in [0.5, 1.0, 2.0]:
                    acts = linear_actions(test, model, feat_cols, buy_thresh=bt, scale=sc)
                    r = evaluate(bars, acts, dsym, f"linear_h{horizon}_bt{bt}_sc{sc}")
                    r["strategy"] = "linear"
                    results.append(r)

    if "gbm" in strategies and len(feat_cols) >= 3 and len(train) > 200:
        for horizon in [1, 3, 6]:
            model = fit_gbm(train, feat_cols, horizon=horizon)
            for bt in [0.0005, 0.001, 0.002]:
                for sc in [0.5, 1.0, 2.0]:
                    acts = linear_actions(test, model, feat_cols, buy_thresh=bt, scale=sc)
                    r = evaluate(bars, acts, dsym, f"gbm_h{horizon}_bt{bt}_sc{sc}")
                    r["strategy"] = "gbm"
                    results.append(r)

    if "quantile" in strategies:
        for ms in [0.002, 0.003, 0.005]:
            for ef in [0.3, 0.5, 0.7, 1.0]:
                acts = quantile_actions(test, min_spread=ms, edge_frac=ef)
                r = evaluate(bars, acts, dsym, f"quant_ms{ms}_ef{ef}")
                r["strategy"] = "quantile"
                results.append(r)

    if "vol_scaled" in strategies:
        for vm in [0.5, 1.0, 1.5, 2.0]:
            for md in [0.0005, 0.001, 0.002]:
                acts = vol_scaled_actions(test, vol_mult=vm, min_delta=md)
                r = evaluate(bars, acts, dsym, f"vol_vm{vm}_md{md}")
                r["strategy"] = "vol_scaled"
                results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--strategies", type=str, default="threshold,linear,gbm,quantile,vol_scaled")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else SYMBOLS
    strategies = args.strategies.split(",")

    all_results = {}
    for pair in symbols:
        dsym = pair_to_data_symbol(pair)
        logger.info(f"\n{'='*60}\n{dsym}\n{'='*60}")
        results = sweep_symbol(pair, strategies)
        if results:
            all_results[dsym] = results
            best = max(results, key=lambda r: r.get("composite", r.get("sortino", -999)))
            logger.info(f"  BEST: {best['label']} Sort={best['sortino']:.2f} "
                        f"Ret={best['total_return']*100:.1f}% DD={best['max_drawdown']*100:.1f}% "
                        f"Trades={best['num_trades']}")

    # Summary
    print(f"\n{'Symbol':<12} {'Strategy':<35} {'Sort':>7} {'Ret%':>7} {'DD%':>7} {'Calmar':>7} {'Trades':>7}")
    print("-" * 90)
    rows = []
    for sym, results in sorted(all_results.items()):
        best = max(results, key=lambda r: r.get("composite", r.get("sortino", -999)))
        rows.append((sym, best))
    rows.sort(key=lambda x: x[1].get("composite", x[1].get("sortino", -999)), reverse=True)
    for sym, best in rows:
        print(f"{sym:<12} {best['label']:<35} {best['sortino']:>7.2f} "
              f"{best['total_return']*100:>6.1f}% {best['max_drawdown']*100:>6.1f}% "
              f"{best.get('calmar', 0):>7.2f} {best['num_trades']:>7}")

    out = REPO / "binanceleveragesui/sweep_results/simple_strategy_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info(f"Results saved: {out}")


if __name__ == "__main__":
    main()
