#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.chronos_daily_comparison import (
    DailyOHLC,
    aggregate_hourly_prediction_frame_to_daily,
    available_daily_days,
    complete_utc_days_from_hourly,
    load_ohlc_csv,
    normalize_symbol_token,
    ohlc_error_percent_by_column,
    ohlc_mape_percent,
)
from src.models.chronos2_wrapper import Chronos2OHLCWrapper


DEFAULT_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD")


@dataclass(frozen=True)
class ComparisonRow:
    symbol: str
    day: str
    daily_direct_mape_pct: float
    hourly_24h_agg_mape_pct: float
    daily_direct_close_error_pct: float
    hourly_24h_close_error_pct: float
    daily_direct: dict[str, float]
    hourly_24h_agg: dict[str, float]
    actual: dict[str, float]


def _discover_symbols(daily_root: Path, hourly_root: Path) -> list[str]:
    daily_symbols = {path.stem.upper() for path in Path(daily_root).glob("*.csv")}
    hourly_symbols = {path.stem.upper() for path in Path(hourly_root).glob("*.csv")}
    return sorted(daily_symbols & hourly_symbols)


def _parse_symbols(raw: Optional[str], *, daily_root: Path, hourly_root: Path, all_available: bool) -> list[str]:
    if all_available:
        symbols = _discover_symbols(daily_root, hourly_root)
    elif raw:
        symbols = [normalize_symbol_token(token) for token in str(raw).split(",") if token.strip()]
    else:
        symbols = list(DEFAULT_SYMBOLS)
    if not symbols:
        raise ValueError("No symbols requested.")
    return symbols


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    raise TypeError(f"Unsupported JSON value {type(value)!r}")


def _daily_row_to_ohlc(row: pd.Series) -> DailyOHLC:
    day = pd.to_datetime(row["timestamp"], utc=True, errors="coerce")
    if pd.isna(day):
        raise ValueError(f"Invalid daily timestamp: {row.get('timestamp')!r}")
    return DailyOHLC(
        timestamp=pd.Timestamp(day).floor("D"),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
    )


def _batch_to_daily_ohlc(batch) -> DailyOHLC:
    quantile_frames = getattr(batch, "quantile_frames", None) or {}
    median = quantile_frames.get(0.5)
    if median is None or median.empty:
        raise ValueError("Chronos batch missing non-empty 0.5 quantile frame.")
    return aggregate_hourly_prediction_frame_to_daily(median)


def _single_step_batch_to_daily_ohlc(batch) -> DailyOHLC:
    quantile_frames = getattr(batch, "quantile_frames", None) or {}
    median = quantile_frames.get(0.5)
    if median is None or median.empty:
        raise ValueError("Chronos batch missing non-empty 0.5 quantile frame.")
    row = median.sort_index().iloc[0]
    ts = pd.Timestamp(median.sort_index().index[0]).floor("D")
    return DailyOHLC(
        timestamp=ts,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
    )


def _context_before_day(frame: pd.DataFrame, *, day: pd.Timestamp, limit: int) -> pd.DataFrame:
    context = frame.loc[pd.to_datetime(frame["timestamp"], utc=True) < day].copy()
    if limit > 0 and len(context) > limit:
        context = context.tail(limit).reset_index(drop=True)
    return context.reset_index(drop=True)


def _actual_hourly_window(frame: pd.DataFrame, *, day: pd.Timestamp, hours: int) -> pd.DataFrame:
    end = day + pd.Timedelta(hours=int(hours))
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[(ts >= day) & (ts < end)].copy().sort_values("timestamp").reset_index(drop=True)


def evaluate_daily_vs_hourly(
    *,
    symbols: Sequence[str],
    daily_root: Path,
    hourly_root: Path,
    model_id: str,
    device_map: str,
    context_days: int,
    context_hours: int,
    batch_size: int,
    val_days: int,
    hourly_prediction_length: int,
    cross_learning: bool,
) -> tuple[list[ComparisonRow], dict[str, Any]]:
    normalized_symbols = [normalize_symbol_token(symbol) for symbol in symbols]
    daily_frames: dict[str, pd.DataFrame] = {}
    hourly_frames: dict[str, pd.DataFrame] = {}
    for symbol in normalized_symbols:
        daily_path = Path(daily_root) / f"{symbol}.csv"
        hourly_path = Path(hourly_root) / f"{symbol}.csv"
        if not daily_path.exists() or not hourly_path.exists():
            logger.warning("Skipping {} due to missing daily/hourly CSV.", symbol)
            continue
        daily_frames[symbol] = load_ohlc_csv(daily_path)
        hourly_frames[symbol] = load_ohlc_csv(hourly_path)

    available_symbols = [symbol for symbol in normalized_symbols if symbol in daily_frames and symbol in hourly_frames]
    if not available_symbols:
        raise RuntimeError("No requested symbols had both daily and hourly data.")

    eligible_days_by_symbol: dict[str, list[pd.Timestamp]] = {}
    all_eval_days: set[pd.Timestamp] = set()
    for symbol in available_symbols:
        daily_days = set(available_daily_days(daily_frames[symbol]))
        hourly_days = set(complete_utc_days_from_hourly(hourly_frames[symbol], expected_bars=hourly_prediction_length))
        eligible = sorted(daily_days & hourly_days)
        if not eligible:
            continue
        eligible = eligible[-max(1, int(val_days)) :]
        eligible_days_by_symbol[symbol] = eligible
        all_eval_days.update(eligible)

    if not eligible_days_by_symbol:
        raise RuntimeError("No requested symbols had complete daily/hourly validation days.")

    eval_days = sorted(all_eval_days)
    logger.info(
        "Evaluating {} symbols across {} union day(s): {} -> {}",
        len(eligible_days_by_symbol),
        len(eval_days),
        eval_days[0],
        eval_days[-1],
    )

    daily_wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id=model_id,
        device_map=device_map,
        default_context_length=max(32, int(context_days)),
        default_batch_size=max(1, int(batch_size)),
        quantile_levels=(0.5,),
    )
    hourly_wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id=model_id,
        device_map=device_map,
        default_context_length=max(64, int(context_hours)),
        default_batch_size=max(1, int(batch_size)),
        quantile_levels=(0.5,),
    )

    predict_kwargs = {"predict_batches_jointly": True} if cross_learning else {}
    rows: list[ComparisonRow] = []

    for day in eval_days:
        symbol_payloads: list[tuple[str, DailyOHLC, pd.DataFrame, pd.DataFrame]] = []
        for symbol in available_symbols:
            eligible_days = eligible_days_by_symbol.get(symbol)
            if eligible_days is None or day not in eligible_days:
                continue
            daily_frame = daily_frames[symbol]
            hourly_frame = hourly_frames[symbol]

            day_mask = pd.to_datetime(daily_frame["timestamp"], utc=True).dt.floor("D") == day
            if not bool(day_mask.any()):
                continue
            actual_daily = _daily_row_to_ohlc(daily_frame.loc[day_mask].iloc[0])
            daily_context = _context_before_day(daily_frame, day=day, limit=context_days)
            hourly_context = _context_before_day(hourly_frame, day=day, limit=context_hours)
            actual_hourly = _actual_hourly_window(hourly_frame, day=day, hours=hourly_prediction_length)

            if len(daily_context) < max(8, min(32, context_days // 4)):
                continue
            if len(hourly_context) < max(48, min(96, context_hours // 4)):
                continue
            if len(actual_hourly) < hourly_prediction_length:
                continue

            symbol_payloads.append((symbol, actual_daily, daily_context, hourly_context))

        if not symbol_payloads:
            logger.warning("No eligible symbol contexts for {}", day)
            continue

        day_symbols = [item[0] for item in symbol_payloads]
        daily_batches = daily_wrapper.predict_ohlc_batch(
            [item[2] for item in symbol_payloads],
            symbols=day_symbols,
            prediction_length=1,
            context_length=context_days,
            batch_size=batch_size,
            predict_kwargs=predict_kwargs,
        )
        hourly_batches = hourly_wrapper.predict_ohlc_batch(
            [item[3] for item in symbol_payloads],
            symbols=day_symbols,
            prediction_length=hourly_prediction_length,
            context_length=context_hours,
            batch_size=batch_size,
            predict_kwargs=predict_kwargs,
        )

        if len(daily_batches) != len(symbol_payloads):
            raise RuntimeError(f"Daily batch count mismatch for {day}: {len(daily_batches)} != {len(symbol_payloads)}")
        if len(hourly_batches) != len(symbol_payloads):
            raise RuntimeError(f"Hourly batch count mismatch for {day}: {len(hourly_batches)} != {len(symbol_payloads)}")

        for payload, daily_batch, hourly_batch in zip(symbol_payloads, daily_batches, hourly_batches):
            symbol, actual_daily, _, _ = payload
            daily_pred = _single_step_batch_to_daily_ohlc(daily_batch)
            hourly_pred = _batch_to_daily_ohlc(hourly_batch)
            daily_errors = ohlc_error_percent_by_column(daily_pred, actual_daily)
            hourly_errors = ohlc_error_percent_by_column(hourly_pred, actual_daily)
            rows.append(
                ComparisonRow(
                    symbol=symbol,
                    day=pd.Timestamp(day).date().isoformat(),
                    daily_direct_mape_pct=ohlc_mape_percent(daily_pred, actual_daily),
                    hourly_24h_agg_mape_pct=ohlc_mape_percent(hourly_pred, actual_daily),
                    daily_direct_close_error_pct=float(daily_errors["close"]),
                    hourly_24h_close_error_pct=float(hourly_errors["close"]),
                    daily_direct={k: float(v) for k, v in daily_pred.as_dict().items() if k != "timestamp"},
                    hourly_24h_agg={k: float(v) for k, v in hourly_pred.as_dict().items() if k != "timestamp"},
                    actual={k: float(v) for k, v in actual_daily.as_dict().items() if k != "timestamp"},
                )
            )

    if not rows:
        raise RuntimeError("Evaluation produced no comparison rows.")

    summary = _summarize_rows(
        rows,
        model_id=model_id,
        symbols=available_symbols,
        eval_days=eval_days,
        eligible_days_by_symbol=eligible_days_by_symbol,
        cross_learning=cross_learning,
        context_days=context_days,
        context_hours=context_hours,
        hourly_prediction_length=hourly_prediction_length,
        batch_size=batch_size,
    )
    return rows, summary


def _summarize_rows(
    rows: Sequence[ComparisonRow],
    *,
    model_id: str,
    symbols: Sequence[str],
    eval_days: Sequence[pd.Timestamp],
    eligible_days_by_symbol: dict[str, list[pd.Timestamp]],
    cross_learning: bool,
    context_days: int,
    context_hours: int,
    hourly_prediction_length: int,
    batch_size: int,
) -> dict[str, Any]:
    frame = pd.DataFrame([asdict(row) for row in rows])
    per_symbol: dict[str, Any] = {}
    for symbol, group in frame.groupby("symbol", sort=True):
        per_symbol[str(symbol)] = {
            "rows": int(len(group)),
            "daily_direct_mape_pct_mean": float(group["daily_direct_mape_pct"].mean()),
            "hourly_24h_agg_mape_pct_mean": float(group["hourly_24h_agg_mape_pct"].mean()),
            "daily_direct_close_error_pct_mean": float(group["daily_direct_close_error_pct"].mean()),
            "hourly_24h_close_error_pct_mean": float(group["hourly_24h_close_error_pct"].mean()),
            "hourly_minus_daily_mape_pct": float(
                group["hourly_24h_agg_mape_pct"].mean() - group["daily_direct_mape_pct"].mean()
            ),
        }

    overall = {
        "rows": int(len(frame)),
        "daily_direct_mape_pct_mean": float(frame["daily_direct_mape_pct"].mean()),
        "hourly_24h_agg_mape_pct_mean": float(frame["hourly_24h_agg_mape_pct"].mean()),
        "daily_direct_close_error_pct_mean": float(frame["daily_direct_close_error_pct"].mean()),
        "hourly_24h_close_error_pct_mean": float(frame["hourly_24h_close_error_pct"].mean()),
        "hourly_minus_daily_mape_pct": float(
            frame["hourly_24h_agg_mape_pct"].mean() - frame["daily_direct_mape_pct"].mean()
        ),
        "winner": (
            "hourly_24h_aggregated"
            if float(frame["hourly_24h_agg_mape_pct"].mean()) < float(frame["daily_direct_mape_pct"].mean())
            else "daily_direct"
        ),
    }
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_id": model_id,
        "symbols": list(symbols),
        "cross_learning": bool(cross_learning),
        "context_days": int(context_days),
        "context_hours": int(context_hours),
        "hourly_prediction_length": int(hourly_prediction_length),
        "batch_size": int(batch_size),
        "eval_days": [pd.Timestamp(day).date().isoformat() for day in eval_days],
        "eligible_days_by_symbol": {
            symbol: [pd.Timestamp(day).date().isoformat() for day in days]
            for symbol, days in sorted(eligible_days_by_symbol.items())
        },
        "overall": overall,
        "per_symbol": per_symbol,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct daily Chronos2 forecasts against 24-hour hourly forecasts aggregated back into daily bars.",
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated symbols.")
    parser.add_argument("--all-available-symbols", action="store_true", help="Use the daily/hourly symbol intersection.")
    parser.add_argument("--daily-root", type=Path, default=REPO / "trainingdatadailybinance")
    parser.add_argument("--hourly-root", type=Path, default=REPO / "trainingdatahourly/crypto")
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--context-days", type=int, default=180)
    parser.add_argument("--context-hours", type=int, default=24 * 90)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--hourly-prediction-length", type=int, default=24)
    parser.add_argument("--cross-learning", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    symbols = _parse_symbols(
        args.symbols,
        daily_root=Path(args.daily_root),
        hourly_root=Path(args.hourly_root),
        all_available=bool(args.all_available_symbols),
    )

    rows, summary = evaluate_daily_vs_hourly(
        symbols=symbols,
        daily_root=Path(args.daily_root),
        hourly_root=Path(args.hourly_root),
        model_id=str(args.model_id),
        device_map=str(args.device_map),
        context_days=int(args.context_days),
        context_hours=int(args.context_hours),
        batch_size=int(args.batch_size),
        val_days=int(args.val_days),
        hourly_prediction_length=int(args.hourly_prediction_length),
        cross_learning=bool(args.cross_learning),
    )

    payload = {
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
    if args.output_json:
        output_path = Path(args.output_json)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_path = REPO / "analysis" / f"chronos2_daily_granularity_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")

    overall = summary["overall"]
    logger.info(
        "Wrote comparison report to {} | winner={} | daily_direct_mape={:.4f}% | hourly_24h_agg_mape={:.4f}%",
        output_path,
        overall["winner"],
        overall["daily_direct_mape_pct_mean"],
        overall["hourly_24h_agg_mape_pct_mean"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
