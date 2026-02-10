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
from src.tradinglib.direction_filter import filter_forecasts_by_predicted_close_return


def _parse_floats(raw: str) -> List[float]:
    items: List[float] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        items.append(float(token))
    return items


def _infer_periods_per_year(bars: pd.DataFrame, *, fallback: float = 24.0 * 365.0) -> float:
    """Infer bar frequency from timestamp deltas and convert to periods/year."""
    if bars is None or bars.empty or "timestamp" not in bars.columns:
        return float(fallback)
    ts = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna().sort_values()
    if len(ts) < 3:
        return float(fallback)
    deltas = ts.diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return float(fallback)
    step = float(deltas.median())
    if not np.isfinite(step) or step <= 0:
        return float(fallback)
    seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
    return float(seconds_per_year / step)


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


def _apply_price_offset(levels: pd.DataFrame, *, price_offset_pct: float) -> pd.DataFrame:
    pct = float(price_offset_pct)
    if pct == 0.0:
        return levels
    frame = levels.copy()
    frame["buy_price"] = pd.to_numeric(frame["buy_price"], errors="coerce") * (1.0 - pct)
    frame["sell_price"] = pd.to_numeric(frame["sell_price"], errors="coerce") * (1.0 + pct)
    return frame


def _trade_stats(trades) -> Dict[str, float]:
    if not trades:
        return {
            "trades_total": 0.0,
            "buys_total": 0.0,
            "sells_total": 0.0,
            "cycles_total": 0.0,
            "trades_per_day_mean": 0.0,
            "trades_per_day_max": 0.0,
            "cycles_per_day_mean": 0.0,
            "cycles_per_day_max": 0.0,
        }

    df = pd.DataFrame({"timestamp": [t.timestamp for t in trades], "side": [t.side for t in trades]})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df["day"] = df["timestamp"].dt.floor("D")

    total = float(len(df))
    buys = float((df["side"].astype(str).str.lower().str.startswith("b")).sum())
    sells = float((df["side"].astype(str).str.lower().str.startswith("s")).sum())

    trades_per_day = df.groupby("day", sort=True).size()
    buys_per_day = df[df["side"].astype(str).str.lower().str.startswith("b")].groupby("day", sort=True).size()
    sells_per_day = df[df["side"].astype(str).str.lower().str.startswith("s")].groupby("day", sort=True).size()
    days = trades_per_day.index
    cycles_per_day = np.minimum(buys_per_day.reindex(days, fill_value=0), sells_per_day.reindex(days, fill_value=0))

    return {
        "trades_total": total,
        "buys_total": buys,
        "sells_total": sells,
        "cycles_total": float(cycles_per_day.sum()),
        "trades_per_day_mean": float(trades_per_day.mean()) if not trades_per_day.empty else 0.0,
        "trades_per_day_max": float(trades_per_day.max()) if not trades_per_day.empty else 0.0,
        "cycles_per_day_mean": float(cycles_per_day.mean()) if len(cycles_per_day) else 0.0,
        "cycles_per_day_max": float(cycles_per_day.max()) if len(cycles_per_day) else 0.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest daily Chronos2-derived buy/sell levels on Binance intraday bars (multiple cycles per day).",
    )
    parser.add_argument("--symbol", default="SOLUSD", help="Internal symbol for model + levels (default: SOLUSD).")
    parser.add_argument(
        "--hourly-symbol",
        default="SOLFDUSD",
        help="CSV symbol to use for intraday simulation (default: SOLFDUSD).",
    )
    parser.add_argument(
        "--intraday-symbol",
        default=None,
        help="Alias for --hourly-symbol (preferred name for non-hourly bars).",
    )
    parser.add_argument("--hourly-root", type=Path, default=Path("trainingdatahourlybinance"))
    parser.add_argument(
        "--intraday-root",
        type=Path,
        default=None,
        help="Alias for --hourly-root (preferred name for non-hourly bars).",
    )
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
    parser.add_argument(
        "--min-spread-mode",
        type=str,
        default="skip",
        choices=["skip", "widen", "none"],
        help="How to enforce min spread between buy/sell (default: skip).",
    )
    parser.add_argument("--min-spread-pcts", default=None, help="Comma-separated sweep values. Overrides --min-spread-pct.")
    parser.add_argument("--price-offset-pcts", default="0.0", help="Comma-separated offsets (buy down, sell up).")
    parser.add_argument("--allocation-fraction", type=float, default=1.0)
    parser.add_argument("--close-at-eod", action="store_true", default=True)
    parser.add_argument("--no-close-at-eod", action="store_false", dest="close_at_eod")
    parser.add_argument("--allow-reentry-same-bar", action="store_true", default=False)
    parser.add_argument("--allow-roundtrip-same-bar", action="store_true", default=False)
    parser.add_argument(
        "--intrabar-assumption",
        type=str,
        default="none",
        choices=["none", "ohlc"],
        help="Within-bar path assumption when only OHLC is known (default: none).",
    )
    parser.add_argument("--stop-loss-pcts", default="0.0", help="Comma-separated stop-loss percentages (0 disables).")
    parser.add_argument(
        "--stop-loss-lockout-until-next-day",
        action="store_true",
        default=False,
        help="After a stop-loss, do not re-enter until the next UTC day (default: off).",
    )
    parser.add_argument("--record-per-bar", action="store_true", default=False, help="Include per-bar frame in results (slower).")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--min-predicted-close-return-pct",
        type=float,
        default=None,
        help="If set, only trade days where predicted_close_p50 >= prev_close*(1+threshold).",
    )
    parser.add_argument(
        "--min-predicted-close-return-pcts",
        default=None,
        help="Comma-separated sweep values. Overrides --min-predicted-close-return-pct.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="sortino_then_annualized",
        choices=[
            "total_return",
            "annualized_return",
            "sortino",
            "sortino_then_annualized",
            "sortino_then_mdd_then_annualized",
            "calmar_then_sortino",
        ],
        help="Validation objective for selecting best parameters.",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Print top-K parameter sets by validation objective.")
    parser.add_argument(
        "--evaluate-top-k-on-test",
        type=int,
        default=0,
        help="If >0, evaluate test metrics for top-K val configs (useful for analysis, but leaks test into selection).",
    )
    parser.add_argument("--forecasts-path", type=Path, default=None, help="Optional cache path for forecast frame (csv/parquet).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbol = str(args.symbol).strip().upper()
    hourly_symbol = str(args.intraday_symbol or args.hourly_symbol).strip().upper()
    hourly_root = Path(args.intraday_root) if args.intraday_root is not None else Path(args.hourly_root)
    val_days = max(1, int(args.val_days))
    test_days = max(1, int(args.test_days))

    daily_path = Path(args.daily_root) / f"{symbol}.csv"
    hourly_path = hourly_root / f"{hourly_symbol}.csv"
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

    offsets = sorted({float(x) for x in _parse_floats(args.price_offset_pcts)})
    if not offsets:
        offsets = [0.0]
    stop_losses = sorted({float(x) for x in _parse_floats(args.stop_loss_pcts)})
    if not stop_losses:
        stop_losses = [0.0]
    if args.min_spread_pcts:
        min_spreads = sorted({float(x) for x in _parse_floats(args.min_spread_pcts)})
    else:
        min_spreads = [float(args.min_spread_pct)]
    if not min_spreads:
        min_spreads = [0.0]

    if args.min_predicted_close_return_pcts:
        min_pred_close_returns: List[Optional[float]] = sorted(
            {float(x) for x in _parse_floats(args.min_predicted_close_return_pcts)}
        )
    elif args.min_predicted_close_return_pct is not None:
        min_pred_close_returns = [float(args.min_predicted_close_return_pct)]
    else:
        min_pred_close_returns = [None]
    if not min_pred_close_returns:
        min_pred_close_returns = [None]
    for thr in min_pred_close_returns:
        if thr is None:
            continue
        if not np.isfinite(float(thr)):
            raise ValueError("--min-predicted-close-return-pct(s) must be finite when provided.")

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

    need_close_p50 = any(thr is not None for thr in min_pred_close_returns)
    required_cols = set()
    for buy_q in buy_qs:
        required_cols.add(f"predicted_low_{_q_tag(buy_q)}")
    for sell_q in sell_qs:
        required_cols.add(f"predicted_high_{_q_tag(sell_q)}")
    if need_close_p50:
        required_cols.add("predicted_close_p50")

    min_context = max(16, min(64, context_length // 8))
    required_days: set[pd.Timestamp] = set()
    for idx in range(forecast_start_idx, forecast_end_idx):
        ctx_end = idx
        ctx_start = max(0, ctx_end - int(context_length))
        if (ctx_end - ctx_start) < int(min_context):
            continue
        ts = pd.Timestamp(daily.iloc[idx]["timestamp"])
        if pd.isna(ts):
            continue
        required_days.add(ts)

    forecasts: pd.DataFrame = pd.DataFrame()
    cache_path = Path(args.forecasts_path) if args.forecasts_path else None
    if cache_path is not None and cache_path.exists():
        if cache_path.suffix.lower() == ".parquet":
            forecasts = pd.read_parquet(cache_path)
        else:
            forecasts = pd.read_csv(cache_path)
        if "timestamp" in forecasts.columns:
            forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"], utc=True, errors="coerce")
        if "issued_at" in forecasts.columns:
            forecasts["issued_at"] = pd.to_datetime(forecasts["issued_at"], utc=True, errors="coerce")
        forecasts = forecasts.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        missing_cols = sorted(required_cols - set(forecasts.columns))
        have_days = set(pd.to_datetime(forecasts["timestamp"], utc=True, errors="coerce").dropna())
        missing_days = sorted(required_days - have_days)
        if missing_cols or missing_days:
            missing_cols_preview = missing_cols[:10]
            if len(missing_cols) > 10:
                missing_cols_preview = [*missing_cols_preview, "..."]
            logger.warning(
                "Cached forecasts {} incomplete; regenerating (missing_cols={} missing_days={}).",
                cache_path,
                missing_cols_preview,
                len(missing_days),
            )
            forecasts = pd.DataFrame()
        else:
            logger.info("Loaded cached forecasts: {} (rows={})", cache_path, len(forecasts))

    if forecasts.empty:
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
            min_context=min_context,
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
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.suffix.lower() == ".parquet":
                forecasts.to_parquet(cache_path, index=False)
            else:
                forecasts.to_csv(cache_path, index=False)
            logger.info("Wrote cached forecasts: {}", cache_path)

    forecasts_by_thr: Dict[Optional[float], pd.DataFrame] = {}
    for thr in min_pred_close_returns:
        norm_thr: Optional[float] = float(thr) if thr is not None else None
        if norm_thr is None:
            forecasts_by_thr[None] = forecasts
            continue
        filtered = filter_forecasts_by_predicted_close_return(
            forecasts,
            daily,
            min_return_pct=float(norm_thr),
        )
        forecasts_by_thr[norm_thr] = filtered
        logger.info(
            "Predicted-close filter: kept {}/{} day(s) (min_return_pct={:.4f}).",
            len(filtered),
            len(forecasts),
            float(norm_thr),
        )

    val_start_day = pd.Timestamp(daily.iloc[val_start_idx]["timestamp"]).floor("D")
    test_start_day = pd.Timestamp(daily.iloc[test_start_idx]["timestamp"]).floor("D")
    end_day = pd.Timestamp(daily.iloc[-1]["timestamp"]).floor("D")

    # Helper to evaluate a (buy_q, sell_q) pair on a window.
    inferred_periods = _infer_periods_per_year(hourly, fallback=24.0 * 365.0)
    sim_base_cfg = {
        "initial_cash": float(args.initial_cash),
        "maker_fee": float(args.maker_fee),
        "allocation_fraction": float(args.allocation_fraction),
        "close_at_eod": bool(args.close_at_eod),
        "allow_reentry_same_bar": bool(args.allow_reentry_same_bar),
        "allow_roundtrip_same_bar": bool(args.allow_roundtrip_same_bar),
        "intrabar_assumption": str(args.intrabar_assumption),
        "periods_per_year": float(inferred_periods),
        "record_per_bar": bool(args.record_per_bar),
        "stop_loss_lockout_until_next_day": bool(args.stop_loss_lockout_until_next_day),
    }

    val_end_day = (test_start_day - pd.Timedelta(days=1)).floor("D")
    test_end_day = end_day
    bars_val = _slice_hourly_window(hourly, start_day=val_start_day, end_day=val_end_day)
    bars_test = _slice_hourly_window(hourly, start_day=test_start_day, end_day=test_end_day)

    def _eval_window(
        forecasts_win: pd.DataFrame,
        bars_win: pd.DataFrame,
        start_day: pd.Timestamp,
        end_day_inclusive: pd.Timestamp,
        buy_q: float,
        sell_q: float,
        price_offset: float,
        min_spread: float,
        stop_loss_pct: float,
    ):
        levels = _levels_from_forecasts(forecasts_win, buy_q=buy_q, sell_q=sell_q)
        levels = levels[(levels["timestamp"] >= start_day) & (levels["timestamp"] <= end_day_inclusive)].reset_index(drop=True)
        levels = _apply_price_offset(levels, price_offset_pct=price_offset)

        sim_cfg = DailyLevelSimulationConfig(
            **sim_base_cfg,
            min_spread_pct=float(min_spread),
            min_spread_mode=str(args.min_spread_mode),
            stop_loss_pct=float(stop_loss_pct),
        )
        result = simulate_daily_levels_on_intraday_bars(bars_win, levels, config=sim_cfg)
        return result

    best = None
    best_key = None
    scores: List[Dict[str, object]] = []
    for min_pred_close_return in min_pred_close_returns:
        norm_thr: Optional[float] = float(min_pred_close_return) if min_pred_close_return is not None else None
        forecasts_for_thr = forecasts_by_thr.get(norm_thr)
        if forecasts_for_thr is None or forecasts_for_thr.empty:
            continue
        for buy_q in buy_qs:
            for sell_q in sell_qs:
                if sell_q <= buy_q:
                    continue
                for price_offset in offsets:
                    for min_spread in min_spreads:
                        for stop_loss in stop_losses:
                            res = _eval_window(
                                forecasts_for_thr,
                                bars_val,
                                val_start_day,
                                val_end_day,
                                buy_q,
                                sell_q,
                                price_offset,
                                min_spread,
                                stop_loss,
                            )
                            metrics = res.metrics
                            val_total_return = float(metrics.total_return)
                            val_sortino = float(metrics.sortino)
                            val_ann = float(metrics.annualized_return)
                            val_mdd = float(metrics.max_drawdown)
                            val_calmar = float(metrics.calmar)
                            val_pf = float(metrics.profit_factor)
                            rec = {
                                "min_predicted_close_return_pct": norm_thr,
                                "buy_q": buy_q,
                                "sell_q": sell_q,
                                "price_offset_pct": float(price_offset),
                                "min_spread_pct": float(min_spread),
                                "stop_loss_pct": float(stop_loss),
                                "val_total_return": val_total_return,
                                "val_annualized_return": val_ann,
                                "val_sortino": val_sortino,
                                "val_max_drawdown": val_mdd,
                                "val_calmar": val_calmar,
                                "val_profit_factor": val_pf,
                                "val_final_equity": float(res.equity_curve.iloc[-1]),
                                "val_trades": len(res.trades),
                            }
                            scores.append(rec)

                            if args.objective == "total_return":
                                key = (val_total_return, val_sortino, val_ann)
                            elif args.objective == "annualized_return":
                                key = (val_ann, val_sortino, val_total_return)
                            elif args.objective == "sortino":
                                key = (val_sortino, val_ann, val_total_return)
                            elif args.objective == "sortino_then_mdd_then_annualized":
                                key = (val_sortino, val_mdd, val_ann, val_total_return)
                            elif args.objective == "calmar_then_sortino":
                                key = (val_calmar, val_sortino, val_ann, val_total_return)
                            else:
                                key = (val_sortino, val_ann, val_total_return)

                            if best is None or key > best_key:
                                best = (norm_thr, buy_q, sell_q, float(price_offset), float(min_spread), float(stop_loss))
                                best_key = key

    assert best is not None
    best_min_pred_close_return, best_buy_q, best_sell_q, best_price_offset, best_min_spread, best_stop_loss = best
    logger.info(
        "Best on val: min_predicted_close_return_pct={} buy_q={} sell_q={} offset={} min_spread={} stop_loss={} objective={}",
        best_min_pred_close_return,
        best_buy_q,
        best_sell_q,
        best_price_offset,
        best_min_spread,
        best_stop_loss,
        args.objective,
    )

    best_forecasts = forecasts_by_thr.get(best_min_pred_close_return)
    if best_forecasts is None:
        raise RuntimeError(f"Internal error: missing forecasts for threshold={best_min_pred_close_return!r}")

    val_best_res = _eval_window(
        best_forecasts,
        bars_val,
        val_start_day,
        val_end_day,
        best_buy_q,
        best_sell_q,
        best_price_offset,
        best_min_spread,
        best_stop_loss,
    )
    test_best_res = _eval_window(
        best_forecasts,
        bars_test,
        test_start_day,
        test_end_day,
        best_buy_q,
        best_sell_q,
        best_price_offset,
        best_min_spread,
        best_stop_loss,
    )

    payload = {
        "run_id": time.strftime("%Y%m%d_%H%M%SZ", time.gmtime()),
        "symbol": symbol,
        "hourly_symbol": hourly_symbol,
        "data": {
            "daily_path": str(daily_path),
            "hourly_path": str(hourly_path),
            "daily_rows": int(len(daily)),
            "hourly_rows": int(len(hourly)),
            "intraday_periods_per_year": float(inferred_periods),
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
            "min_spread_mode": str(args.min_spread_mode),
            "min_spread_pct": float(best_min_spread),
            "price_offset_pct": float(best_price_offset),
            "stop_loss_pct": float(best_stop_loss),
            "stop_loss_lockout_until_next_day": bool(args.stop_loss_lockout_until_next_day),
            "allocation_fraction": float(args.allocation_fraction),
            "close_at_eod": bool(args.close_at_eod),
            "allow_reentry_same_bar": bool(args.allow_reentry_same_bar),
            "allow_roundtrip_same_bar": bool(args.allow_roundtrip_same_bar),
            "intrabar_assumption": str(args.intrabar_assumption),
            "min_predicted_close_return_pct": best_min_pred_close_return,
            "initial_cash": float(args.initial_cash),
        },
        "val_best": {
            **asdict(val_best_res.metrics),
            "trades": len(val_best_res.trades),
            "final_equity": float(val_best_res.equity_curve.iloc[-1]),
            **_trade_stats(val_best_res.trades),
        },
        "test_best": {
            **asdict(test_best_res.metrics),
            "trades": len(test_best_res.trades),
            "final_equity": float(test_best_res.equity_curve.iloc[-1]),
            **_trade_stats(test_best_res.trades),
        },
        "sweep": scores,
    }

    top_val: List[Dict[str, object]] = []
    want_top = max(int(args.top_k or 0), int(args.evaluate_top_k_on_test or 0))
    if want_top > 0 and scores:
        if args.objective == "total_return":
            sort_key = lambda r: (r["val_total_return"], r["val_sortino"], r["val_annualized_return"])
        elif args.objective == "annualized_return":
            sort_key = lambda r: (r["val_annualized_return"], r["val_sortino"], r["val_total_return"])
        elif args.objective == "sortino":
            sort_key = lambda r: (r["val_sortino"], r["val_annualized_return"], r["val_total_return"])
        elif args.objective == "sortino_then_mdd_then_annualized":
            sort_key = lambda r: (r["val_sortino"], r["val_max_drawdown"], r["val_annualized_return"], r["val_total_return"])
        elif args.objective == "calmar_then_sortino":
            sort_key = lambda r: (r["val_calmar"], r["val_sortino"], r["val_annualized_return"], r["val_total_return"])
        else:
            sort_key = lambda r: (r["val_sortino"], r["val_annualized_return"], r["val_total_return"])
        top_val = sorted(scores, key=sort_key, reverse=True)[:want_top]

    if int(args.top_k or 0) > 0 and top_val:
        print("TOP:", json.dumps(top_val[: int(args.top_k)], indent=2, sort_keys=True))

    if int(args.evaluate_top_k_on_test or 0) > 0 and top_val:
        top_test: List[Dict[str, object]] = []
        for rec in top_val[: int(args.evaluate_top_k_on_test)]:
            thr = rec.get("min_predicted_close_return_pct")
            forecasts_for_thr = forecasts_by_thr.get(float(thr) if thr is not None else None)
            if forecasts_for_thr is None or forecasts_for_thr.empty:
                continue
            res_test = _eval_window(
                forecasts_for_thr,
                bars_test,
                test_start_day,
                test_end_day,
                float(rec["buy_q"]),
                float(rec["sell_q"]),
                float(rec["price_offset_pct"]),
                float(rec["min_spread_pct"]),
                float(rec["stop_loss_pct"]),
            )
            m = res_test.metrics
            enriched = dict(rec)
            enriched.update(
                {
                    "test_total_return": float(m.total_return),
                    "test_annualized_return": float(m.annualized_return),
                    "test_sortino": float(m.sortino),
                    "test_max_drawdown": float(m.max_drawdown),
                    "test_profit_factor": float(m.profit_factor),
                    "test_trades": len(res_test.trades),
                    "test_final_equity": float(res_test.equity_curve.iloc[-1]),
                }
            )
            top_test.append(enriched)
        payload["top_k_test"] = top_test
        print("TOP_TEST:", json.dumps(top_test, indent=2, sort_keys=True))

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
