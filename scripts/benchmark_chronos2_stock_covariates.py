#!/usr/bin/env python3
"""Benchmark Chronos2 stock forecasts with correlated and calendar covariates.

This is the Alpaca-stock analogue of the BitBank BTC/ETH experiment: compare
plain OHLC Chronos2 forecasts against the same target with SPY/QQQ or
top-correlated peer return/range/body covariates on recent hourly holdouts.
It can also test calendar/session covariates, which are known in the future
and therefore live-feasible. The script does not trade or update configs; it
is a promotion gate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chronos2_wrapper import Chronos2OHLCWrapper


TARGET_COLS = ("open", "high", "low", "close")
DEFAULT_SYMBOLS = ("AAPL", "MSFT", "NVDA", "TSLA")
DEFAULT_ANCHORS = ("SPY", "QQQ")


@dataclass
class TrialResult:
    symbol: str
    cutoff: int
    peers: list[str]
    baseline_mae_pct: float
    peer_mae_pct: float
    time_mae_pct: float
    combined_mae_pct: float
    baseline_direction_pct: float
    peer_direction_pct: float
    time_direction_pct: float
    combined_direction_pct: float
    baseline_return_mae_bps: float
    peer_return_mae_bps: float
    time_return_mae_bps: float
    combined_return_mae_bps: float
    baseline_latency_s: float
    peer_latency_s: float
    time_latency_s: float
    combined_latency_s: float

    @property
    def peer_improvement_pct(self) -> float:
        if self.baseline_mae_pct <= 0 or not math.isfinite(self.baseline_mae_pct):
            return 0.0
        return (self.baseline_mae_pct - self.peer_mae_pct) / self.baseline_mae_pct * 100.0

    @property
    def time_improvement_pct(self) -> float:
        if self.baseline_mae_pct <= 0 or not math.isfinite(self.baseline_mae_pct):
            return 0.0
        return (self.baseline_mae_pct - self.time_mae_pct) / self.baseline_mae_pct * 100.0

    @property
    def combined_improvement_pct(self) -> float:
        if self.baseline_mae_pct <= 0 or not math.isfinite(self.baseline_mae_pct):
            return 0.0
        return (self.baseline_mae_pct - self.combined_mae_pct) / self.baseline_mae_pct * 100.0


def _parse_csv_list(raw: str | Sequence[str]) -> list[str]:
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = []
        for item in raw:
            values.extend(str(item).split(","))
    return [value.strip().upper() for value in values if value.strip()]


def _load_frame(data_root: Path, symbol: str) -> pd.DataFrame:
    path = data_root / f"{symbol.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing hourly data for {symbol}: {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} missing timestamp column")
    for col in TARGET_COLS:
        if col not in df.columns:
            raise KeyError(f"{path} missing {col} column")
    df = df[["timestamp", *TARGET_COLS]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    for col in TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(TARGET_COLS))
    df = df[(df[list(TARGET_COLS)] > 0).all(axis=1)]
    return df.reset_index(drop=True)


def _covariate_payload(peer_df: pd.DataFrame, peer: str) -> pd.DataFrame:
    name = "".join(ch.lower() if ch.isalnum() else "_" for ch in peer).strip("_")
    close = peer_df["close"].astype("float64")
    open_ = peer_df["open"].astype("float64")
    high = peer_df["high"].astype("float64")
    low = peer_df["low"].astype("float64")
    log_close = np.log(np.maximum(close.to_numpy(dtype=np.float64), 1e-12))
    returns = np.diff(log_close, prepend=log_close[0])
    range_pct = ((high - low) / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    body_pct = ((close - open_) / open_.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame(
        {
            "timestamp": peer_df["timestamp"],
            f"cov_{name}_ret": np.clip(returns, -0.20, 0.20),
            f"cov_{name}_range": np.clip(range_pct.to_numpy(dtype=np.float64), 0.0, 0.30),
            f"cov_{name}_body": np.clip(body_pct.to_numpy(dtype=np.float64), -0.30, 0.30),
        }
    )


def _add_covariates(context: pd.DataFrame, peer_contexts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    result = context.copy()
    columns: list[str] = []
    for peer, peer_df in peer_contexts.items():
        payload = _covariate_payload(peer_df, peer)
        before = set(result.columns)
        result = result.merge(payload, on="timestamp", how="left")
        new_cols = sorted(set(result.columns) - before)
        for col in new_cols:
            result[col] = result[col].ffill().bfill().fillna(0.0).astype("float32")
        columns.extend(new_cols)
    return result, columns


def _time_covariate_payload(timestamps: pd.Series | pd.DatetimeIndex | Sequence[pd.Timestamp]) -> pd.DataFrame:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    hour = ts.dt.hour.astype("float64")
    dow = ts.dt.dayofweek.astype("float64")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "cov_hour_sin": np.sin(2.0 * np.pi * hour / 24.0).astype("float32"),
            "cov_hour_cos": np.cos(2.0 * np.pi * hour / 24.0).astype("float32"),
            "cov_dow_sin": np.sin(2.0 * np.pi * dow / 7.0).astype("float32"),
            "cov_dow_cos": np.cos(2.0 * np.pi * dow / 7.0).astype("float32"),
        }
    )


def _add_time_covariates(
    context: pd.DataFrame,
    future_timestamps: pd.Series | pd.DatetimeIndex | Sequence[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    context_payload = _time_covariate_payload(context["timestamp"])
    future_payload = _time_covariate_payload(future_timestamps)
    cov_cols = [col for col in context_payload.columns if col != "timestamp"]
    enriched = context.merge(context_payload, on="timestamp", how="left")
    for col in cov_cols:
        enriched[col] = enriched[col].ffill().bfill().fillna(0.0).astype("float32")
    return enriched, future_payload, cov_cols


def _infer_forecast_timestamps(context: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(context["timestamp"], utc=True, errors="coerce").dropna()
    if len(timestamps) < 2:
        step = pd.Timedelta(hours=1)
        last_ts = timestamps.iloc[-1] if len(timestamps) else pd.Timestamp.now(tz="UTC")
    else:
        freq = pd.infer_freq(timestamps)
        if freq is not None:
            step = pd.tseries.frequencies.to_offset(freq)
            last_ts = timestamps.iloc[-1]
            return pd.DatetimeIndex([last_ts + step * (i + 1) for i in range(horizon)])
        deltas = timestamps.diff().dropna()
        try:
            step = deltas.value_counts().idxmax()
        except Exception:
            step = deltas.median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta(hours=1)
        # Mirror Chronos2OHLCWrapper.build_panel(): irregular contexts are
        # reindexed to the modal step before Chronos sees them, so future_df
        # timestamps must follow that reindexed grid rather than raw holdout
        # rows.
        full_index = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq=step, tz="UTC")
        last_ts = full_index[-1] if len(full_index) else timestamps.iloc[-1]
    return pd.DatetimeIndex([last_ts + step * (i + 1) for i in range(horizon)])


def _select_correlated_peers(
    symbol: str,
    frames: dict[str, pd.DataFrame],
    *,
    anchors: Sequence[str],
    max_peers: int,
    lookback: int,
    cutoff: int,
) -> list[str]:
    candidates = [peer for peer in anchors if peer in frames and peer != symbol]
    if max_peers <= 0:
        return []
    target = frames[symbol].iloc[max(0, cutoff - lookback) : cutoff][["timestamp", "close"]].copy()
    if target.empty:
        return []
    start_ts = target["timestamp"].iloc[0]
    end_ts = target["timestamp"].iloc[-1]
    target["target_ret"] = np.log(target["close"]).diff()
    scored: list[tuple[float, str]] = []
    for peer in candidates:
        peer_frame = frames[peer]
        peer_frame = peer_frame[(peer_frame["timestamp"] >= start_ts) & (peer_frame["timestamp"] <= end_ts)]
        peer_frame = peer_frame[["timestamp", "close"]].copy()
        peer_frame["peer_ret"] = np.log(peer_frame["close"]).diff()
        merged = target[["timestamp", "target_ret"]].merge(peer_frame[["timestamp", "peer_ret"]], on="timestamp")
        merged = merged.dropna()
        if len(merged) < 24:
            continue
        corr = float(merged["target_ret"].corr(merged["peer_ret"]))
        if math.isfinite(corr):
            scored.append((abs(corr), peer))
    scored.sort(reverse=True)
    return [peer for _, peer in scored[:max_peers]]


def _close_mae_pct(predicted: Sequence[float], actual: Sequence[float]) -> float:
    pred = np.asarray(predicted, dtype=np.float64)
    obs = np.asarray(actual, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(obs) & (np.abs(obs) > 1e-12)
    if not mask.any():
        return math.inf
    return float(np.mean(np.abs(pred[mask] - obs[mask]) / np.abs(obs[mask])) * 100.0)


def _return_mae_bps(predicted: Sequence[float], actual: Sequence[float], base: float) -> float:
    if base <= 0:
        return math.inf
    pred_ret = (np.asarray(predicted, dtype=np.float64) - base) / base
    obs_ret = (np.asarray(actual, dtype=np.float64) - base) / base
    return float(np.mean(np.abs(pred_ret - obs_ret)) * 10_000.0)


def _direction_pct(predicted: Sequence[float], actual: Sequence[float], start_close: float) -> float:
    matches = 0
    count = 0
    prev_pred = start_close
    prev_actual = start_close
    for pred, obs in zip(predicted, actual):
        pred_move = float(pred) - prev_pred
        actual_move = float(obs) - prev_actual
        if pred_move != 0.0 and actual_move != 0.0:
            matches += int((pred_move > 0.0) == (actual_move > 0.0))
            count += 1
        prev_pred = float(pred)
        prev_actual = float(obs)
    return matches / count * 100.0 if count else 0.0


def _predict_close_path(
    wrapper: Chronos2OHLCWrapper,
    context: pd.DataFrame,
    *,
    symbol: str,
    horizon: int,
    context_length: int,
    quantiles: Sequence[float],
    batch_size: int,
    covariate_cols: Sequence[str] = (),
    future_covariates: pd.DataFrame | None = None,
) -> tuple[list[float], float]:
    kwargs: dict[str, Any] = {}
    if covariate_cols:
        kwargs["known_future_covariates"] = list(covariate_cols)
    if future_covariates is not None:
        kwargs["future_covariates"] = future_covariates
    start = time.perf_counter()
    batch = wrapper.predict_ohlc(
        context,
        symbol=symbol,
        prediction_length=horizon,
        context_length=context_length,
        quantile_levels=quantiles,
        batch_size=batch_size,
        **kwargs,
    )
    latency = time.perf_counter() - start
    q50 = batch.quantile(0.5).reset_index(drop=True)
    return [float(value) for value in q50["close"].iloc[:horizon]], latency


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = _parse_csv_list(args.symbols)
    anchors = _parse_csv_list(args.anchors)
    needed = sorted(set(symbols) | set(anchors))
    frames = {symbol: _load_frame(args.data_root, symbol) for symbol in needed}

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id=args.model_id,
        device_map=args.device_map,
        target_columns=TARGET_COLS,
        default_context_length=args.context_length,
        quantile_levels=tuple(args.quantiles),
        default_batch_size=args.batch_size,
        pipeline_backend="chronos",
    )

    trials: list[TrialResult] = []
    try:
        for symbol in symbols:
            frame = frames[symbol]
            if len(frame) < args.context_length + args.horizon:
                print(f"{symbol}: skipped, only {len(frame)} rows")
                continue
            max_cutoff = len(frame) - args.horizon
            cutoffs = [max_cutoff - args.horizon * i for i in range(args.cutoffs)]
            for cutoff in sorted(c for c in cutoffs if c >= args.context_length):
                context = frame.iloc[cutoff - args.context_length : cutoff].copy()
                actual = frame.iloc[cutoff : cutoff + args.horizon].copy()
                peers = _select_correlated_peers(
                    symbol,
                    frames,
                    anchors=anchors,
                    max_peers=args.max_peers,
                    lookback=args.peer_lookback,
                    cutoff=cutoff,
                )
                start_ts = context["timestamp"].iloc[0]
                end_ts = context["timestamp"].iloc[-1]
                peer_contexts = {
                    peer: frames[peer][
                        (frames[peer]["timestamp"] >= start_ts) & (frames[peer]["timestamp"] <= end_ts)
                    ].copy()
                    for peer in peers
                    if not frames[peer][
                        (frames[peer]["timestamp"] >= start_ts) & (frames[peer]["timestamp"] <= end_ts)
                    ].empty
                }
                cov_context, cov_cols = _add_covariates(context, peer_contexts)
                forecast_timestamps = _infer_forecast_timestamps(context, args.horizon)
                time_context, time_future, time_cols = _add_time_covariates(context, forecast_timestamps)
                combined_context = cov_context.copy()
                combined_time_payload = _time_covariate_payload(combined_context["timestamp"])
                combined_context = combined_context.merge(combined_time_payload, on="timestamp", how="left")
                combined_cols = [*cov_cols, *time_cols]
                for col in time_cols:
                    combined_context[col] = combined_context[col].ffill().bfill().fillna(0.0).astype("float32")
                combined_future = time_future.copy()
                for col in cov_cols:
                    combined_future[col] = np.float32(0.0)

                baseline_close, baseline_latency = _predict_close_path(
                    wrapper,
                    context,
                    symbol=symbol,
                    horizon=args.horizon,
                    context_length=args.context_length,
                    quantiles=args.quantiles,
                    batch_size=args.batch_size,
                )
                peer_close, peer_latency = _predict_close_path(
                    wrapper,
                    cov_context,
                    symbol=symbol,
                    horizon=args.horizon,
                    context_length=args.context_length,
                    quantiles=args.quantiles,
                    batch_size=args.batch_size,
                    covariate_cols=cov_cols,
                )
                time_close, time_latency = _predict_close_path(
                    wrapper,
                    time_context,
                    symbol=symbol,
                    horizon=args.horizon,
                    context_length=args.context_length,
                    quantiles=args.quantiles,
                    batch_size=args.batch_size,
                    covariate_cols=time_cols,
                    future_covariates=time_future,
                )
                combined_close, combined_latency = _predict_close_path(
                    wrapper,
                    combined_context,
                    symbol=symbol,
                    horizon=args.horizon,
                    context_length=args.context_length,
                    quantiles=args.quantiles,
                    batch_size=args.batch_size,
                    covariate_cols=combined_cols,
                    future_covariates=combined_future,
                )
                actual_close = [float(value) for value in actual["close"]]
                start_close = float(context["close"].iloc[-1])
                trial = TrialResult(
                    symbol=symbol,
                    cutoff=cutoff,
                    peers=peers,
                    baseline_mae_pct=_close_mae_pct(baseline_close, actual_close),
                    peer_mae_pct=_close_mae_pct(peer_close, actual_close),
                    time_mae_pct=_close_mae_pct(time_close, actual_close),
                    combined_mae_pct=_close_mae_pct(combined_close, actual_close),
                    baseline_direction_pct=_direction_pct(baseline_close, actual_close, start_close),
                    peer_direction_pct=_direction_pct(peer_close, actual_close, start_close),
                    time_direction_pct=_direction_pct(time_close, actual_close, start_close),
                    combined_direction_pct=_direction_pct(combined_close, actual_close, start_close),
                    baseline_return_mae_bps=_return_mae_bps(baseline_close, actual_close, start_close),
                    peer_return_mae_bps=_return_mae_bps(peer_close, actual_close, start_close),
                    time_return_mae_bps=_return_mae_bps(time_close, actual_close, start_close),
                    combined_return_mae_bps=_return_mae_bps(combined_close, actual_close, start_close),
                    baseline_latency_s=baseline_latency,
                    peer_latency_s=peer_latency,
                    time_latency_s=time_latency,
                    combined_latency_s=combined_latency,
                )
                trials.append(trial)
                print(
                    f"{symbol} cutoff={cutoff} peers={','.join(peers) or '-'} "
                    f"base={trial.baseline_mae_pct:.3f}% "
                    f"peer={trial.peer_mae_pct:.3f}%({trial.peer_improvement_pct:+.2f}%) "
                    f"time={trial.time_mae_pct:.3f}%({trial.time_improvement_pct:+.2f}%) "
                    f"combo={trial.combined_mae_pct:.3f}%({trial.combined_improvement_pct:+.2f}%) "
                    f"dir={trial.baseline_direction_pct:.1f}%/"
                    f"{trial.peer_direction_pct:.1f}%/{trial.time_direction_pct:.1f}%/{trial.combined_direction_pct:.1f}%"
                )
                time.sleep(args.pause)
    finally:
        wrapper.unload()

    if not trials:
        raise RuntimeError("no benchmark trials completed")
    baseline_mae = statistics.mean(t.baseline_mae_pct for t in trials)
    peer_mae = statistics.mean(t.peer_mae_pct for t in trials)
    time_mae = statistics.mean(t.time_mae_pct for t in trials)
    combined_mae = statistics.mean(t.combined_mae_pct for t in trials)
    baseline_ret = statistics.mean(t.baseline_return_mae_bps for t in trials)
    peer_ret = statistics.mean(t.peer_return_mae_bps for t in trials)
    time_ret = statistics.mean(t.time_return_mae_bps for t in trials)
    combined_ret = statistics.mean(t.combined_return_mae_bps for t in trials)
    baseline_dir = statistics.mean(t.baseline_direction_pct for t in trials)
    peer_dir = statistics.mean(t.peer_direction_pct for t in trials)
    time_dir = statistics.mean(t.time_direction_pct for t in trials)
    combined_dir = statistics.mean(t.combined_direction_pct for t in trials)
    peer_improvement = (baseline_mae - peer_mae) / baseline_mae * 100.0 if baseline_mae > 0 else 0.0
    time_improvement = (baseline_mae - time_mae) / baseline_mae * 100.0 if baseline_mae > 0 else 0.0
    combined_improvement = (baseline_mae - combined_mae) / baseline_mae * 100.0 if baseline_mae > 0 else 0.0
    peer_win_rate = sum(t.peer_mae_pct < t.baseline_mae_pct for t in trials) / len(trials) * 100.0
    time_win_rate = sum(t.time_mae_pct < t.baseline_mae_pct for t in trials) / len(trials) * 100.0
    combined_win_rate = sum(t.combined_mae_pct < t.baseline_mae_pct for t in trials) / len(trials) * 100.0
    summary = {
        "trials": len(trials),
        "symbols": symbols,
        "anchors": anchors,
        "baseline_mae_pct": baseline_mae,
        "peer_mae_pct": peer_mae,
        "time_mae_pct": time_mae,
        "combined_mae_pct": combined_mae,
        "peer_improvement_pct": peer_improvement,
        "time_improvement_pct": time_improvement,
        "combined_improvement_pct": combined_improvement,
        "peer_win_rate_pct": peer_win_rate,
        "time_win_rate_pct": time_win_rate,
        "combined_win_rate_pct": combined_win_rate,
        "baseline_return_mae_bps": baseline_ret,
        "peer_return_mae_bps": peer_ret,
        "time_return_mae_bps": time_ret,
        "combined_return_mae_bps": combined_ret,
        "baseline_direction_pct": baseline_dir,
        "peer_direction_pct": peer_dir,
        "time_direction_pct": time_dir,
        "combined_direction_pct": combined_dir,
        "trial_results": [
            asdict(t)
            | {
                "peer_improvement_pct": t.peer_improvement_pct,
                "time_improvement_pct": t.time_improvement_pct,
                "combined_improvement_pct": t.combined_improvement_pct,
            }
            for t in trials
        ],
    }
    print(
        f"SUMMARY trials={len(trials)} base_mae={baseline_mae:.3f}% "
        f"peer={peer_mae:.3f}%({peer_improvement:+.2f}%, win={peer_win_rate:.1f}%) "
        f"time={time_mae:.3f}%({time_improvement:+.2f}%, win={time_win_rate:.1f}%) "
        f"combo={combined_mae:.3f}%({combined_improvement:+.2f}%, win={combined_win_rate:.1f}%) "
        f"dir={baseline_dir:.1f}%/{peer_dir:.1f}%/{time_dir:.1f}%/{combined_dir:.1f}% "
        f"ret_mae_bps={baseline_ret:.1f}/{peer_ret:.1f}/{time_ret:.1f}/{combined_ret:.1f}"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS))
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--context-length", type=int, default=360)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--cutoffs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--max-peers", type=int, default=2)
    parser.add_argument("--peer-lookback", type=int, default=240)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
