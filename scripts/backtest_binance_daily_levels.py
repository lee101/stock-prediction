#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Allow running as `python scripts/...` without needing PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.chronos2_params import resolve_chronos2_params
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.tradinglib.daily_level_simulator import (
    DailyLevelSimulationConfig,
    simulate_daily_levels_on_intraday_bars,
)


def _parse_floats(raw: str) -> List[float]:
    items: List[float] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        items.append(float(token))
    return items


def _q_tag(q: float) -> str:
    pct = int(round(float(q) * 100))
    return f"p{pct:d}"


def _load_ohlc_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} missing timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise KeyError(f"{path} missing required column {col!r}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


@dataclass(frozen=True)
class ForecastSpec:
    symbol: str
    target_ts: pd.Timestamp
    issued_at: pd.Timestamp
    context: pd.DataFrame
    batch_symbol: str


def _build_forecast_specs(
    daily: pd.DataFrame,
    *,
    symbol: str,
    start_idx: int,
    end_idx: int,
    context_length: int,
    min_context: int = 16,
) -> List[ForecastSpec]:
    if start_idx < 0 or end_idx > len(daily) or start_idx >= end_idx:
        return []
    specs: List[ForecastSpec] = []
    for ordinal, idx in enumerate(range(start_idx, end_idx)):
        ctx_end = idx
        ctx_start = max(0, ctx_end - int(context_length))
        context = daily.iloc[ctx_start:ctx_end].copy()
        if len(context) < min_context:
            continue
        target_ts = pd.Timestamp(daily.iloc[idx]["timestamp"])
        issued_at = pd.Timestamp(context["timestamp"].max())
        suffix = target_ts.strftime("%Y%m%dT%H%M%SZ")
        batch_symbol = f"{symbol}__{suffix}__{ordinal}"
        specs.append(
            ForecastSpec(
                symbol=symbol,
                target_ts=target_ts,
                issued_at=issued_at,
                context=context[["timestamp", "open", "high", "low", "close"]],
                batch_symbol=batch_symbol,
            )
        )
    return specs


def _build_forecast_frame(
    wrapper: Chronos2OHLCWrapper,
    specs: Sequence[ForecastSpec],
    *,
    prediction_length: int,
    context_length: int,
    quantile_levels: Sequence[float],
    batch_size: int,
    predict_kwargs: Optional[Dict[str, object]],
) -> pd.DataFrame:
    if not specs:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    q_levels = tuple(float(q) for q in quantile_levels)
    for start in range(0, len(specs), max(1, int(batch_size))):
        chunk = specs[start : start + max(1, int(batch_size))]
        contexts = [spec.context for spec in chunk]
        symbols = [spec.batch_symbol for spec in chunk]
        batches = wrapper.predict_ohlc_batch(
            contexts,
            symbols=symbols,
            prediction_length=int(prediction_length),
            context_length=int(context_length),
            quantile_levels=q_levels,
            batch_size=int(batch_size),
            predict_kwargs=predict_kwargs,
        )
        for spec, batch in zip(chunk, batches):
            quantiles = batch.quantile_frames
            if not quantiles:
                continue
            row: Dict[str, object] = {
                "timestamp": spec.target_ts,
                "symbol": spec.symbol,
                "issued_at": spec.issued_at,
            }
            for q, qdf in quantiles.items():
                if qdf is None or qdf.empty:
                    continue
                # predict_ohlc_batch uses future timestamps derived from delta; with prediction_length=1,
                # we expect the forecast index to equal spec.target_ts.
                if spec.target_ts not in qdf.index:
                    # Fall back to the first row if the timestamp doesn't align exactly (defensive).
                    q_row = qdf.iloc[0]
                else:
                    q_row = qdf.loc[spec.target_ts]
                tag = _q_tag(float(q))
                for col in ("open", "high", "low", "close"):
                    if col in q_row.index:
                        row[f"predicted_{col}_{tag}"] = float(q_row[col])
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_records(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["issued_at"] = pd.to_datetime(out["issued_at"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out = out.drop_duplicates(subset=["timestamp", "symbol"], keep="last").reset_index(drop=True)
    return out


def _slice_hourly_window(hourly: pd.DataFrame, *, start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    start_day = pd.Timestamp(start_day).tz_convert("UTC") if pd.Timestamp(start_day).tzinfo else pd.Timestamp(start_day, tz="UTC")
    end_day = pd.Timestamp(end_day).tz_convert("UTC") if pd.Timestamp(end_day).tzinfo else pd.Timestamp(end_day, tz="UTC")
    # Include all intraday bars in [start_day, end_day + 1 day).
    start_ts = start_day.floor("D")
    end_ts = end_day.floor("D") + pd.Timedelta(days=1)
    ts = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce")
    mask = (ts >= start_ts) & (ts < end_ts)
    return hourly.loc[mask].reset_index(drop=True)


def _levels_from_forecasts(forecasts: pd.DataFrame, *, buy_q: float, sell_q: float) -> pd.DataFrame:
    buy_tag = _q_tag(buy_q)
    sell_tag = _q_tag(sell_q)
    buy_col = f"predicted_low_{buy_tag}"
    sell_col = f"predicted_high_{sell_tag}"
    if buy_col not in forecasts.columns:
        raise KeyError(f"Forecast frame missing {buy_col}")
    if sell_col not in forecasts.columns:
        raise KeyError(f"Forecast frame missing {sell_col}")
    levels = forecasts[["timestamp", buy_col, sell_col]].copy()
    levels = levels.rename(columns={buy_col: "buy_price", sell_col: "sell_price"})
    return levels


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest daily Chronos2-derived buy/sell levels on Binance intraday bars (multiple cycles per day).",
    )
    parser.add_argument("--symbol", default="SOLUSD", help="Internal symbol for model + levels (default: SOLUSD).")
    parser.add_argument(
        "--hourly-symbol",
        default="SOLFDUSD",
        help="Hourly CSV symbol to use for intraday simulation (default: SOLFDUSD).",
    )
    parser.add_argument("--hourly-root", type=Path, default=Path("trainingdatahourlybinance"))
    parser.add_argument("--daily-root", type=Path, default=Path("trainingdatadailybinance"))
    parser.add_argument("--val-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--context-length", type=int, default=None, help="Override Chronos context length.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override Chronos batch size.")
    parser.add_argument("--model-id", default=None, help="Override Chronos model_id (default: resolve from hyperparams).")
    parser.add_argument("--device-map", default=None, help="Override device_map (default: resolve from hyperparams).")
    parser.add_argument("--buy-quantiles", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--sell-quantiles", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--maker-fee", type=float, default=0.0)
    parser.add_argument("--min-spread-pct", type=float, default=0.0)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbol = str(args.symbol).strip().upper()
    hourly_symbol = str(args.hourly_symbol).strip().upper()
    val_days = max(1, int(args.val_days))
    test_days = max(1, int(args.test_days))

    daily_path = Path(args.daily_root) / f"{symbol}.csv"
    hourly_path = Path(args.hourly_root) / f"{hourly_symbol}.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing daily CSV: {daily_path}")
    if not hourly_path.exists():
        raise FileNotFoundError(f"Missing hourly CSV: {hourly_path}")

    daily = _load_ohlc_csv(daily_path)
    hourly = _load_ohlc_csv(hourly_path)
    if len(daily) < (val_days + test_days + 32):
        raise ValueError(f"Not enough daily history ({len(daily)} rows) for val={val_days} test={test_days}.")

    # Split by row count to keep the window definitions stable and leakage-free.
    total = len(daily)
    test_start_idx = total - test_days
    val_start_idx = test_start_idx - val_days
    # Forecast + trade for val+test days, but forecast for day[i] uses context up to i-1.
    forecast_start_idx = val_start_idx
    forecast_end_idx = total

    params = resolve_chronos2_params(symbol, frequency="daily", default_prediction_length=1)
    model_id = str(args.model_id or params.get("model_id") or "amazon/chronos-2")
    device_map = args.device_map or params.get("device_map") or "cuda"
    context_length = int(args.context_length or params.get("context_length") or 512)
    batch_size = int(args.batch_size or params.get("batch_size") or 32)
    predict_kwargs = params.get("predict_kwargs")
    predict_kwargs = dict(predict_kwargs) if isinstance(predict_kwargs, dict) else None

    buy_qs = sorted({float(q) for q in _parse_floats(args.buy_quantiles)})
    sell_qs = sorted({float(q) for q in _parse_floats(args.sell_quantiles)})
    quantiles = sorted({*buy_qs, *sell_qs, 0.5})

    logger.info(
        "Backtest setup: symbol={} hourly_symbol={} daily_rows={} hourly_rows={} val_days={} test_days={}",
        symbol,
        hourly_symbol,
        len(daily),
        len(hourly),
        val_days,
        test_days,
    )
    logger.info("Chronos params: model_id={} device_map={} context_length={} batch_size={}", model_id, device_map, context_length, batch_size)
    logger.info("Quantile sweep: buy={} sell={}", buy_qs, sell_qs)

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id=model_id,
        device_map=device_map,
        default_context_length=context_length,
        default_batch_size=batch_size,
        quantile_levels=tuple(quantiles),
        preaugmentation_dirs=[Path("preaugstrategies") / "chronos2"],
    )

    specs = _build_forecast_specs(
        daily,
        symbol=symbol,
        start_idx=forecast_start_idx,
        end_idx=forecast_end_idx,
        context_length=context_length,
        min_context=max(16, min(64, context_length // 8)),
    )
    logger.info("Generating forecasts for {} day(s)...", len(specs))
    forecasts = _build_forecast_frame(
        wrapper,
        specs,
        prediction_length=1,
        context_length=context_length,
        quantile_levels=quantiles,
        batch_size=batch_size,
        predict_kwargs=predict_kwargs,
    )
    if forecasts.empty:
        raise RuntimeError("No forecasts generated (empty forecast frame).")

    val_start_day = pd.Timestamp(daily.iloc[val_start_idx]["timestamp"]).floor("D")
    test_start_day = pd.Timestamp(daily.iloc[test_start_idx]["timestamp"]).floor("D")
    end_day = pd.Timestamp(daily.iloc[-1]["timestamp"]).floor("D")

    # Helper to evaluate a (buy_q, sell_q) pair on a window.
    sim_cfg = DailyLevelSimulationConfig(
        initial_cash=float(args.initial_cash),
        maker_fee=float(args.maker_fee),
        allocation_fraction=1.0,
        min_spread_pct=float(args.min_spread_pct),
        close_at_eod=True,
        allow_reentry_same_bar=False,
        periods_per_year=24.0 * 365.0,
    )

    def _eval_window(start_day: pd.Timestamp, end_day_inclusive: pd.Timestamp, buy_q: float, sell_q: float):
        levels = _levels_from_forecasts(forecasts, buy_q=buy_q, sell_q=sell_q)
        levels = levels[(levels["timestamp"] >= start_day) & (levels["timestamp"] <= end_day_inclusive)].reset_index(drop=True)
        bars_win = _slice_hourly_window(hourly, start_day=start_day, end_day=end_day_inclusive)
        result = simulate_daily_levels_on_intraday_bars(bars_win, levels, config=sim_cfg)
        return result

    val_end_day = (test_start_day - pd.Timedelta(days=1)).floor("D")
    test_end_day = end_day

    best = None
    best_key = None
    scores: List[Dict[str, object]] = []
    for buy_q in buy_qs:
        for sell_q in sell_qs:
            if sell_q <= buy_q:
                continue
            res = _eval_window(val_start_day, val_end_day, buy_q, sell_q)
            score = float(res.metrics.total_return)
            sortino = float(res.metrics.sortino)
            scores.append(
                {
                    "buy_q": buy_q,
                    "sell_q": sell_q,
                    "val_total_return": score,
                    "val_sortino": sortino,
                    "val_final_equity": float(res.equity_curve.iloc[-1]),
                    "val_trades": len(res.trades),
                }
            )
            key = (score, sortino)
            if best is None or key > best_key:
                best = (buy_q, sell_q)
                best_key = key

    assert best is not None
    best_buy_q, best_sell_q = best
    logger.info("Best on val: buy_q={} sell_q={} total_return={:.6f} sortino={:.3f}", best_buy_q, best_sell_q, best_key[0], best_key[1])

    val_best_res = _eval_window(val_start_day, val_end_day, best_buy_q, best_sell_q)
    test_best_res = _eval_window(test_start_day, test_end_day, best_buy_q, best_sell_q)

    payload = {
        "run_id": time.strftime("%Y%m%d_%H%M%SZ", time.gmtime()),
        "symbol": symbol,
        "hourly_symbol": hourly_symbol,
        "data": {
            "daily_path": str(daily_path),
            "hourly_path": str(hourly_path),
            "daily_rows": int(len(daily)),
            "hourly_rows": int(len(hourly)),
            "val_days": int(val_days),
            "test_days": int(test_days),
            "val_start_day": str(val_start_day),
            "val_end_day": str(val_end_day),
            "test_start_day": str(test_start_day),
            "test_end_day": str(test_end_day),
        },
        "chronos": {
            "model_id": model_id,
            "device_map": device_map,
            "context_length": context_length,
            "batch_size": batch_size,
            "quantiles": quantiles,
            "predict_kwargs": predict_kwargs,
        },
        "strategy": {
            "best_buy_q": best_buy_q,
            "best_sell_q": best_sell_q,
            "maker_fee": float(args.maker_fee),
            "min_spread_pct": float(args.min_spread_pct),
            "initial_cash": float(args.initial_cash),
        },
        "val_best": {
            **asdict(val_best_res.metrics),
            "trades": len(val_best_res.trades),
            "final_equity": float(val_best_res.equity_curve.iloc[-1]),
        },
        "test_best": {
            **asdict(test_best_res.metrics),
            "trades": len(test_best_res.trades),
            "final_equity": float(test_best_res.equity_curve.iloc[-1]),
        },
        "sweep": scores,
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("reports/binance_daily_levels") / f"{symbol}_daily_levels_{payload['run_id']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote report: {}", out_path)

    print("VAL  best:", json.dumps(payload["val_best"], indent=2, sort_keys=True))
    print("TEST best:", json.dumps(payload["test_best"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

