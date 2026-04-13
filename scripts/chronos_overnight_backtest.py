#!/usr/bin/env python3
"""
Chronos2 overnight close-to-close strategy backtest.

Strategy
--------
Each trading day D (at market close):
  1. Run Chronos2 (with per-symbol preaug) to predict close return for D+1.
  2. Rank all liquid stocks by predicted return.
  3. Buy top-K at close of D (Market On Close order).
  4. Sell at close of D+1.
  5. Optionally short bottom-K (predicted < -threshold).

This matches Chronos2's prediction target exactly (close-to-close).

Key differences from chronos_top2_backtest.py:
  - Execution: close[D] → close[D+1]  (not open[D+1] → close[D+1])
  - Uses per-symbol best preaug strategy from preaugstrategies/best/ or chronos2/
  - Adds minimum return threshold filter
  - Adds market regime filter (optional)
  - Supports batch inference for speed

Usage
-----
  python scripts/chronos_overnight_backtest.py \\
      --start-date 2024-01-01 --end-date 2026-04-01 \\
      --top-k 5 --min-predicted-return 0.5 \\
      --min-volume-m 5 --commission-bps 10 \\
      --output-dir analysis/overnight_backtest

  # Use fine-tuned v2 model:
  python scripts/chronos_overnight_backtest.py \\
      --model-id chronos2_finetuned/stocks_all_v2/finetuned-ckpt \\
      --start-date 2024-01-01 --end-date 2026-04-01 \\
      --top-k 5 --min-predicted-return 0.5 --min-volume-m 10

  # Aggressive top-2 with high threshold:
  python scripts/chronos_overnight_backtest.py \\
      --top-k 2 --min-predicted-return 2.0 --min-volume-m 20 \\
      --start-date 2025-01-01 --end-date 2026-04-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from preaug.strategies import AUGMENTATION_REGISTRY

log = logging.getLogger("overnight_backtest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return None
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "open", "volume"])
        df = df[df["close"] > 0]
        return df
    except Exception:
        return None


def load_universe(data_dir: Path, preaug_dir: Path, min_volume_m: float, min_rows: int = 300) -> dict:
    """Return {symbol: {'df': DataFrame, 'strategy': str, 'strategy_obj': aug}}"""
    universe = {}
    min_vol = min_volume_m * 1e6
    for csv_path in sorted(data_dir.glob("*.csv")):
        sym = csv_path.stem.upper()
        df = load_csv(csv_path)
        if df is None or len(df) < min_rows:
            continue
        avg_vol = df["volume"].tail(60).mean()
        if avg_vol < min_vol:
            continue
        avg_price = df["close"].tail(60).mean()
        if avg_price < 1.0:
            continue
        # Load preaug strategy
        strategy_name = "differencing"  # default
        preaug_file = preaug_dir / f"{sym}.json"
        if preaug_file.exists():
            try:
                pd_cfg = json.loads(preaug_file.read_text())
                strategy_name = pd_cfg.get("best_strategy", "differencing")
            except Exception:
                pass
        aug_cls = AUGMENTATION_REGISTRY.get(strategy_name)
        if aug_cls is None:
            aug_cls = AUGMENTATION_REGISTRY["differencing"]
            strategy_name = "differencing"
        universe[sym] = {
            "df": df,
            "strategy": strategy_name,
            "aug_cls": aug_cls,
        }
    return universe


# ---------------------------------------------------------------------------
# Forecast cache
# ---------------------------------------------------------------------------

def _cache_key(sym: str, day: date, context_length: int, model_id: str) -> str:
    return f"{sym}_{day.isoformat()}_{context_length}_{Path(model_id).name}"


# ---------------------------------------------------------------------------
# Single-symbol forecast
# ---------------------------------------------------------------------------

def predict_return(
    wrapper,
    sym_data: dict,
    target_day: date,
    context_length: int,
    cache: dict,
    hp_root: Path,
) -> Optional[float]:
    """Predict close-to-close return % for target_day using data strictly before it."""
    df = sym_data["df"]
    aug_cls = sym_data["aug_cls"]

    key = _cache_key(sym_data.get("sym", "?"), target_day, context_length, wrapper.model_id or "base")
    if key in cache:
        return cache[key]

    # Context: all rows with date < target_day
    target_dt = pd.Timestamp(target_day, tz="UTC")
    ctx = df[df["timestamp"] < target_dt].copy()
    if len(ctx) < 20:
        return None

    # Trim to context_length
    if len(ctx) > context_length:
        ctx = ctx.tail(context_length).copy()

    last_close = float(ctx["close"].iloc[-1])
    if last_close <= 0:
        return None

    # Apply preaug transform
    aug = aug_cls()
    try:
        ctx_aug = aug.transform_dataframe(ctx)
    except Exception:
        return None

    # Predict
    try:
        batch = wrapper.predict_ohlc_batch(
            [ctx_aug],
            symbols=None,
            prediction_length=1,
            context_length=context_length,
            batch_size=1,
        )
        if not batch:
            return None
        pred_batch = batch[0]
        # Get median predicted value (close column)
        close_col = "close"
        q50_col = "q_0.5" if "q_0.5" in pred_batch.prediction_df.columns else pred_batch.prediction_df.columns[0]
        # inverse transform
        pred_arr = pred_batch.prediction_df[[close_col]].values if close_col in pred_batch.prediction_df.columns else pred_batch.prediction_df.iloc[:, :1].values
        pred_inverse = aug.inverse_transform_predictions(pred_arr, ctx, columns=[close_col])
        pred_close = float(pred_inverse[0, 0]) if hasattr(pred_inverse, 'shape') else float(pred_inverse[0])
    except Exception as e:
        log.debug(f"predict failed: {e}")
        return None

    ret_pct = (pred_close - last_close) / last_close * 100.0
    cache[key] = ret_pct
    return ret_pct


# ---------------------------------------------------------------------------
# Backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    wrapper,
    universe: dict,
    start_date: date,
    end_date: date,
    top_k: int,
    min_predicted_return_pct: float,
    commission_bps: float,
    fill_bps: float,
    context_length: int,
    hp_root: Path,
    regime_symbol: Optional[str] = None,
    regime_lookback: int = 20,
    short_bottom_k: bool = False,
    cache_dir: Optional[Path] = None,
) -> dict:
    """Run overnight close-to-close backtest. Returns results dict."""

    # Restore cache
    cache: dict = {}
    if cache_dir and cache_dir.exists():
        for cf in cache_dir.glob("*.json"):
            try:
                cache.update(json.loads(cf.read_text()))
            except Exception:
                pass
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Build sorted trading day list
    all_timestamps: set = set()
    for info in universe.values():
        df = info["df"]
        mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
        all_timestamps.update(df[mask]["timestamp"].dt.date.tolist())
    trading_days = sorted(all_timestamps)

    # Add sym to each entry for cache key
    for sym, info in universe.items():
        info["sym"] = sym

    equity = 10_000.0
    daily_returns = []
    daily_log = []
    cost_pct = (commission_bps * 2 + fill_bps * 2) / 100.0  # round-trip

    for day_idx, day in enumerate(trading_days):
        next_day = None
        # Find next trading day
        for nd in trading_days[day_idx + 1:]:
            next_day = nd
            break
        if next_day is None:
            break

        # Regime filter
        if regime_symbol and regime_symbol in universe:
            reg_df = universe[regime_symbol]["df"]
            reg_ctx = reg_df[reg_df["timestamp"].dt.date < day]
            if len(reg_ctx) >= regime_lookback:
                ma_short = reg_ctx["close"].tail(regime_lookback // 2).mean()
                ma_long = reg_ctx["close"].tail(regime_lookback).mean()
                if ma_short < ma_long:
                    # Bearish regime — skip longs
                    daily_log.append({"day": str(day), "regime": "bear", "trades": []})
                    continue

        # Generate predictions for all symbols
        preds = {}
        for sym, info in universe.items():
            ret = predict_return(wrapper, info, day, context_length, cache, hp_root)
            if ret is not None and np.isfinite(ret):
                preds[sym] = ret

        if not preds:
            continue

        # Select top-K longs
        ranked = sorted(preds.items(), key=lambda x: -x[1])
        longs = [(s, r) for s, r in ranked if r >= min_predicted_return_pct][:top_k]

        # Select bottom-K shorts (if enabled)
        shorts = []
        if short_bottom_k:
            rev_ranked = sorted(preds.items(), key=lambda x: x[1])
            shorts = [(s, r) for s, r in rev_ranked if r <= -min_predicted_return_pct][:top_k]

        trades = []
        for sym, pred_ret in longs + [(s, -r) for s, r in shorts]:
            is_short = (sym, -pred_ret) in shorts

            df = universe[sym]["df"]
            # Close of day (entry price)
            entry_row = df[df["timestamp"].dt.date == day]
            # Close of next_day (exit price)
            exit_row = df[df["timestamp"].dt.date == next_day]
            if entry_row.empty or exit_row.empty:
                continue

            entry_price = float(entry_row["close"].iloc[0])
            exit_price = float(exit_row["close"].iloc[0])
            if entry_price <= 0:
                continue

            gross_pct = (exit_price - entry_price) / entry_price * 100.0
            if is_short:
                gross_pct = -gross_pct
            net_pct = gross_pct - cost_pct * 100.0

            trades.append({
                "symbol": sym,
                "side": "short" if is_short else "long",
                "predicted_return_pct": pred_ret if not is_short else -pred_ret,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return_pct": gross_pct,
                "net_return_pct": net_pct,
            })

        if not trades:
            daily_log.append({"day": str(day), "trades": [], "equity": equity})
            continue

        day_ret = float(np.mean([t["net_return_pct"] for t in trades]))
        equity *= (1 + day_ret / 100.0)
        daily_returns.append(day_ret)
        daily_log.append({
            "day": str(day),
            "equity": equity,
            "day_return_pct": day_ret,
            "n_trades": len(trades),
            "trades": [f"{t['symbol']}(pred={t['predicted_return_pct']:+.2f}%,net={t['net_return_pct']:+.2f}%)" for t in trades[:3]],
        })

        if day_idx % 20 == 0:
            log.info(f"[{day_idx+1}/{len(trading_days)}] {day}: eq=${equity:.0f} day={day_ret:+.2f}% trades={len(trades)}")

    # Save cache
    if cache_dir:
        (cache_dir / "forecast_cache.json").write_text(json.dumps(cache))

    # Summary stats
    n_days = len(daily_returns)
    if n_days == 0:
        return {"error": "no trading days", "daily_log": daily_log}

    arr = np.array(daily_returns)
    total_ret = (equity - 10_000) / 10_000 * 100.0
    months = n_days / 21.0
    monthly_ret = ((equity / 10_000) ** (1 / max(months, 0.1)) - 1) * 100.0
    ann_ret = ((equity / 10_000) ** (252 / max(n_days, 1)) - 1) * 100.0
    sharpe = (arr.mean() / (arr.std() + 1e-9)) * (252 ** 0.5)
    downside = arr[arr < 0]
    sortino = (arr.mean() / (downside.std() + 1e-9)) * (252 ** 0.5) if len(downside) > 1 else float("inf")
    win_rate = float((arr > 0).mean() * 100)
    neg_days = int((arr < -cost_pct * 100 * 0.5).sum())
    max_dd = 0.0
    peak = 10_000.0
    eq = 10_000.0
    for r in arr:
        eq *= (1 + r / 100)
        peak = max(peak, eq)
        max_dd = min(max_dd, (eq - peak) / peak * 100)

    summary = {
        "initial_cash": 10_000,
        "final_equity": round(equity, 2),
        "total_return_pct": round(total_ret, 2),
        "monthly_return_pct": round(monthly_ret, 2),
        "ann_return_pct": round(ann_ret, 2),
        "sharpe_ann": round(sharpe, 3),
        "sortino_ann": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 2),
        "n_trading_days": n_days,
        "commission_bps": commission_bps,
        "fill_bps": fill_bps,
        "top_k": top_k,
        "min_predicted_return_pct": min_predicted_return_pct,
    }
    return {"summary": summary, "daily_log": daily_log}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="trainingdata")
    p.add_argument("--preaug-dir", default="preaugstrategies/best",
                   help="Directory with per-symbol preaug JSON (tries chronos2/ as fallback)")
    p.add_argument("--model-id", default="amazon/chronos-2",
                   help="Chronos2 model ID or path to fine-tuned checkpoint")
    p.add_argument("--hp-root", default="hyperparams", help="Hyperparam root dir")
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-predicted-return", type=float, default=0.5,
                   help="Min predicted return %% to enter long (default: 0.5)")
    p.add_argument("--short-bottom-k", action="store_true",
                   help="Also short bottom-K stocks")
    p.add_argument("--min-volume-m", type=float, default=10.0,
                   help="Min avg daily volume in $M (default: 10)")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--commission-bps", type=float, default=10.0)
    p.add_argument("--fill-bps", type=float, default=5.0)
    p.add_argument("--regime-filter", action="store_true",
                   help="Skip longs in bearish SPY regime")
    p.add_argument("--output-dir", default="analysis/overnight_backtest")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--backend", default="cute_compiled_fp32",
                   choices=["cute_compiled_fp32", "auto", "original"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-symbols", type=int, default=0,
                   help="Limit universe size for testing (0=all)")
    args = p.parse_args()

    data_dir = REPO / args.data_dir
    # Fallback preaug dir logic: try best/, then chronos2/
    preaug_dir = REPO / args.preaug_dir
    if not preaug_dir.exists():
        preaug_dir = REPO / "preaugstrategies/chronos2"
    output_dir = REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "forecast_cache"

    end_date_obj = date.fromisoformat(args.end_date) if args.end_date else date.today()
    start_date_obj = date.fromisoformat(args.start_date)

    log.info(f"Loading universe from {data_dir}...")
    universe = load_universe(data_dir, preaug_dir, args.min_volume_m)
    if args.max_symbols > 0:
        keys = list(universe.keys())[:args.max_symbols]
        universe = {k: universe[k] for k in keys}
    log.info(f"Universe: {len(universe)} liquid symbols (vol≥${args.min_volume_m}M)")

    if not universe:
        log.error("No symbols in universe — check data-dir and min-volume-m")
        return 1

    # Load model
    log.info(f"Loading Chronos2: {args.model_id}")
    from src.models.chronos2_wrapper import Chronos2Wrapper
    hp_root = REPO / args.hp_root
    # Build a minimal hyperparam config for the wrapper
    sample_hp = {
        "model_id": args.model_id,
        "device_map": "cuda",
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "quantile_levels": [0.1, 0.5, 0.9],
        "aggregation": "median",
        "sample_count": 0,
        "scaler": "none",
        "use_multivariate": False,
        "predict_kwargs": {},
    }
    wrapper = Chronos2Wrapper(
        config=type("cfg", (), sample_hp)(),
        backend=args.backend,
        hp_root=str(hp_root),
    )

    regime_symbol = "SPY" if args.regime_filter else None

    log.info(f"Running backtest: {start_date_obj} → {end_date_obj}, top_k={args.top_k}, min_ret={args.min_predicted_return}%")
    t0 = time.time()
    results = run_backtest(
        wrapper=wrapper,
        universe=universe,
        start_date=start_date_obj,
        end_date=end_date_obj,
        top_k=args.top_k,
        min_predicted_return_pct=args.min_predicted_return,
        commission_bps=args.commission_bps,
        fill_bps=args.fill_bps,
        context_length=args.context_length,
        hp_root=hp_root,
        regime_symbol=regime_symbol,
        short_bottom_k=args.short_bottom_k,
        cache_dir=cache_dir,
    )
    elapsed = time.time() - t0

    # Print summary
    if "summary" in results:
        s = results["summary"]
        print("\n" + "=" * 60)
        print("  OVERNIGHT CLOSE-TO-CLOSE BACKTEST SUMMARY")
        print("=" * 60)
        print(f"  Model      : {args.model_id}")
        print(f"  Period     : {start_date_obj} → {end_date_obj}")
        print(f"  Universe   : {len(universe)} stocks (vol≥${args.min_volume_m}M)")
        print(f"  Strategy   : top-{args.top_k}, min_pred={args.min_predicted_return}%")
        print(f"  Regime filter: {'SPY-based' if args.regime_filter else 'none'}")
        print(f"  Initial    : ${s['initial_cash']:,.0f}")
        print(f"  Final      : ${s['final_equity']:,.0f}")
        print(f"  Total ret  : {s['total_return_pct']:+.2f}%")
        print(f"  Monthly ret: {s['monthly_return_pct']:+.2f}%")
        print(f"  Ann. ret   : {s['ann_return_pct']:+.2f}%")
        print(f"  Sharpe     : {s['sharpe_ann']:.2f}")
        print(f"  Sortino    : {s['sortino_ann']:.2f}")
        print(f"  Max DD     : {s['max_drawdown_pct']:.2f}%")
        print(f"  Win rate   : {s['win_rate_pct']:.1f}%")
        print(f"  Trading days: {s['n_trading_days']}")
        print(f"  Elapsed    : {elapsed:.0f}s")
        print("=" * 60)
    else:
        print(f"No results: {results.get('error', 'unknown')}")

    # Save output
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"overnight_summary_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
