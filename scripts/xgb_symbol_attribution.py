#!/usr/bin/env python3
"""Per-symbol attribution for an XGB daily candidate.

This is a research tool, not a deploy path. It replays a fixed pre-trained
ensemble with a single BacktestConfig over rolling OOS windows and attributes
daily portfolio return back to the symbols that generated it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xgbnew.backtest import (
    PRODUCTION_STOCK_FEE_RATE,
    BacktestConfig,
    DayResult,
    _trade_allocation_weights,
    simulate,
)
from xgbnew.dataset import build_daily_dataset, load_chronos_cache
from xgbnew.features import (
    DAILY_DISPERSION_FEATURE_COLS,
    DAILY_RANK_FEATURE_COLS,
    LIVE_SUPPORTED_FEATURE_COLS,
)
from xgbnew.model_registry import load_any_model
from xgbnew.sweep_ensemble_grid import (
    FEE_REGIMES,
    _build_windows,
    _ensemble_score_mean_std,
    _load_symbols,
)


logger = logging.getLogger(__name__)


@dataclass
class SymbolAttribution:
    symbol: str
    pnl_dollars: float = 0.0
    contribution_pct_sum: float = 0.0
    trade_count: int = 0
    long_count: int = 0
    short_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    avg_score: float = 0.0
    avg_trade_net_pct: float = 0.0
    worst_trade_net_pct: float = 0.0
    best_trade_net_pct: float = 0.0
    worst_intraday_dd_pct: float = 0.0
    first_day: str = ""
    last_day: str = ""


def _model_feature_mode(models: list[Any], paths: list[Path]) -> tuple[bool, bool]:
    first_features: tuple[str, ...] | None = None
    first_path: Path | None = None
    for model, path in zip(models, paths, strict=True):
        raw_features = getattr(model, "feature_cols", None)
        if (
            not isinstance(raw_features, (list, tuple))
            or not raw_features
            or not all(isinstance(col, str) and col for col in raw_features)
        ):
            raise ValueError(f"{path}: model feature_cols must be a non-empty list")
        features = tuple(raw_features)
        unsupported = sorted(set(features) - LIVE_SUPPORTED_FEATURE_COLS)
        if unsupported:
            raise ValueError(
                f"{path}: model feature_cols contains unsupported live features: {unsupported}"
            )
        if first_features is None:
            first_features = features
            first_path = path
        elif features != first_features:
            raise ValueError(
                "Ensemble feature_cols mismatch: "
                f"{path} differs from {first_path}"
            )
    assert first_features is not None
    needs_ranks = any(c in first_features for c in DAILY_RANK_FEATURE_COLS)
    needs_disp = any(c in first_features for c in DAILY_DISPERSION_FEATURE_COLS)
    return needs_ranks, needs_disp


def aggregate_symbol_attribution(day_results: list[DayResult], config: BacktestConfig) -> list[dict[str, Any]]:
    """Attribute realized daily return back to symbols.

    The simulator stores day-level returns after allocation weighting. Rebuild
    the same weights for each day and apportion the day return to each trade.
    If a future config applies an extra day-level scalar, preserve exact
    attribution by scaling raw trade contributions back to ``daily_return_pct``.
    """
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "pnl_dollars": 0.0,
        "contribution_pct_sum": 0.0,
        "trade_count": 0,
        "long_count": 0,
        "short_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "score_sum": 0.0,
        "net_pct_sum": 0.0,
        "worst_trade_net_pct": float("inf"),
        "best_trade_net_pct": float("-inf"),
        "worst_intraday_dd_pct": 0.0,
        "first_day": "",
        "last_day": "",
    })
    for day in day_results:
        if not day.trades:
            continue
        weights = _trade_allocation_weights(
            day.trades,
            mode=config.allocation_mode,
            temperature=config.allocation_temp,
            short_allocation_scale=config.short_allocation_scale,
        )
        raw_contribs = np.asarray(
            [float(w) * float(t.net_return_pct) for w, t in zip(weights, day.trades)],
            dtype=np.float64,
        )
        raw_total = float(raw_contribs.sum())
        scale = float(day.daily_return_pct) / raw_total if abs(raw_total) > 1e-12 else 1.0
        for raw_contrib, weight, trade in zip(raw_contribs, weights, day.trades):
            sym = str(trade.symbol)
            contrib_pct = float(raw_contrib) * scale
            bucket = stats[sym]
            bucket["pnl_dollars"] += float(day.equity_start) * contrib_pct / 100.0
            bucket["contribution_pct_sum"] += contrib_pct
            bucket["trade_count"] += 1
            if int(trade.side) < 0:
                bucket["short_count"] += 1
            else:
                bucket["long_count"] += 1
            if contrib_pct > 0:
                bucket["win_count"] += 1
            elif contrib_pct < 0:
                bucket["loss_count"] += 1
            bucket["score_sum"] += float(trade.score)
            bucket["net_pct_sum"] += float(trade.net_return_pct)
            bucket["worst_trade_net_pct"] = min(
                float(bucket["worst_trade_net_pct"]),
                float(trade.net_return_pct),
            )
            bucket["best_trade_net_pct"] = max(
                float(bucket["best_trade_net_pct"]),
                float(trade.net_return_pct),
            )
            bucket["worst_intraday_dd_pct"] = max(
                float(bucket["worst_intraday_dd_pct"]),
                float(weight) * float(trade.intraday_worst_dd_pct) * scale,
            )
            day_str = str(day.day)
            if not bucket["first_day"] or day_str < bucket["first_day"]:
                bucket["first_day"] = day_str
            if not bucket["last_day"] or day_str > bucket["last_day"]:
                bucket["last_day"] = day_str

    rows: list[dict[str, Any]] = []
    for sym, bucket in stats.items():
        n = int(bucket["trade_count"])
        if n <= 0:
            continue
        rows.append(asdict(SymbolAttribution(
            symbol=sym,
            pnl_dollars=float(bucket["pnl_dollars"]),
            contribution_pct_sum=float(bucket["contribution_pct_sum"]),
            trade_count=n,
            long_count=int(bucket["long_count"]),
            short_count=int(bucket["short_count"]),
            win_count=int(bucket["win_count"]),
            loss_count=int(bucket["loss_count"]),
            avg_score=float(bucket["score_sum"]) / n,
            avg_trade_net_pct=float(bucket["net_pct_sum"]) / n,
            worst_trade_net_pct=float(bucket["worst_trade_net_pct"]),
            best_trade_net_pct=float(bucket["best_trade_net_pct"]),
            worst_intraday_dd_pct=float(bucket["worst_intraday_dd_pct"]),
            first_day=str(bucket["first_day"]),
            last_day=str(bucket["last_day"]),
        )))
    rows.sort(key=lambda r: float(r["pnl_dollars"]), reverse=True)
    return rows


def _window_metrics(result) -> dict[str, Any]:
    return {
        "monthly_return_pct": float(result.monthly_return_pct),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "sortino_ratio": float(result.sortino_ratio),
        "total_trades": int(result.total_trades),
        "active_days": int(len(result.day_results)),
        "worst_intraday_dd_pct": float(result.worst_intraday_dd_pct),
        "stopped_early": bool(result.stopped_early),
        "stop_reason": str(result.stop_reason),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XGB candidate per-symbol attribution")
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    p.add_argument("--chronos-cache", type=Path, default=Path("analysis/xgbnew_daily/chronos_cache.parquet"))
    p.add_argument("--model-paths", type=str, required=True)
    p.add_argument("--blend-mode", choices=["mean", "median"], default="mean")
    p.add_argument("--train-start", type=str, default="2020-01-01")
    p.add_argument("--train-end", type=str, default="2024-12-31")
    p.add_argument("--oos-start", type=str, required=True)
    p.add_argument("--oos-end", type=str, required=True)
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--stride-days", type=int, default=30)
    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--short-n", type=int, default=0)
    p.add_argument("--max-short-score", type=float, default=0.45)
    p.add_argument("--short-allocation-scale", type=float, default=0.5)
    p.add_argument("--leverage", type=float, default=2.0)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--hold-through", action="store_true")
    p.add_argument("--fee-regime", choices=sorted(FEE_REGIMES), default="prod10bps")
    p.add_argument("--fill-buffer-bps", type=float, default=-1.0)
    p.add_argument("--min-dollar-vol", type=float, default=50_000_000.0)
    p.add_argument("--inference-min-dolvol", type=float, default=50_000_000.0)
    p.add_argument("--inference-min-vol", type=float, default=0.12)
    p.add_argument("--inference-max-vol", type=float, default=0.0)
    p.add_argument("--inference-max-spread-bps", type=float, default=30.0)
    p.add_argument("--allocation-mode", choices=["equal", "softmax", "score_norm"], default="equal")
    p.add_argument("--allocation-temp", type=float, default=1.0)
    p.add_argument("--regime-cs-iqr-max", type=float, default=0.0)
    p.add_argument("--regime-cs-skew-min", type=float, default=-1e9)
    p.add_argument("--max-ret-20d-rank-pct", type=float, default=1.0)
    p.add_argument("--min-ret-5d-rank-pct", type=float, default=0.0)
    p.add_argument("--inv-vol-target-ann", type=float, default=0.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    model_paths = [Path(p.strip()) for p in args.model_paths.split(",") if p.strip()]
    if not model_paths:
        raise ValueError("--model-paths cannot be empty")

    t0 = time.perf_counter()
    models = [load_any_model(path) for path in model_paths]
    needs_ranks, needs_disp = _model_feature_mode(models, model_paths)
    chronos_cache = (
        load_chronos_cache(args.chronos_cache)
        if args.chronos_cache.exists()
        else None
    )
    symbols = _load_symbols(args.symbols_file)
    _, _, oos_df = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        val_start=date.fromisoformat(args.oos_start),
        val_end=date.fromisoformat(args.oos_end),
        test_start=date.fromisoformat(args.oos_start),
        test_end=date.fromisoformat(args.oos_end),
        chronos_cache=chronos_cache,
        min_dollar_vol=float(args.min_dollar_vol),
        fast_features=False,
        include_cross_sectional_ranks=needs_ranks,
        include_cross_sectional_dispersion=needs_disp,
    )
    scores, _ = _ensemble_score_mean_std(oos_df, models, args.blend_mode)
    all_days = sorted(oos_df["date"].unique())
    windows = _build_windows(all_days, int(args.window_days), int(args.stride_days))
    fee = FEE_REGIMES[str(args.fee_regime)]
    fill_buffer_bps = (
        float(fee["fill_buffer_bps"])
        if float(args.fill_buffer_bps) < 0.0
        else float(args.fill_buffer_bps)
    )
    cfg = BacktestConfig(
        top_n=int(args.top_n),
        short_n=int(args.short_n),
        max_short_score=float(args.max_short_score),
        short_allocation_scale=float(args.short_allocation_scale),
        leverage=float(args.leverage),
        min_score=float(args.min_score),
        hold_through=bool(args.hold_through),
        fee_rate=float(fee.get("fee_rate", PRODUCTION_STOCK_FEE_RATE)),
        commission_bps=float(fee.get("commission_bps", 0.0)),
        fill_buffer_bps=fill_buffer_bps,
        min_dollar_vol=float(args.inference_min_dolvol),
        min_vol_20d=float(args.inference_min_vol),
        max_vol_20d=float(args.inference_max_vol),
        max_spread_bps=float(args.inference_max_spread_bps),
        allocation_mode=str(args.allocation_mode),
        allocation_temp=float(args.allocation_temp),
        regime_cs_iqr_max=float(args.regime_cs_iqr_max),
        regime_cs_skew_min=float(args.regime_cs_skew_min),
        max_ret_20d_rank_pct=float(args.max_ret_20d_rank_pct),
        min_ret_5d_rank_pct=float(args.min_ret_5d_rank_pct),
        inv_vol_target_ann=float(args.inv_vol_target_ann),
    )

    window_payloads = []
    all_day_results: list[DayResult] = []
    for start, end in windows:
        mask = (oos_df["date"] >= start) & (oos_df["date"] <= end)
        result = simulate(
            oos_df.loc[mask],
            model=models[0],
            config=cfg,
            precomputed_scores=scores.loc[mask],
        )
        window_payloads.append({
            "start": str(start),
            "end": str(end),
            **_window_metrics(result),
        })
        all_day_results.extend(result.day_results)

    monthly = np.asarray([w["monthly_return_pct"] for w in window_payloads], dtype=np.float64)
    drawdowns = np.asarray([w["max_drawdown_pct"] for w in window_payloads], dtype=np.float64)
    attribution = aggregate_symbol_attribution(all_day_results, cfg)
    positive_symbols = [
        row["symbol"] for row in attribution
        if float(row["pnl_dollars"]) > 0.0 and int(row["trade_count"]) >= 2
    ]
    payload = {
        "config": {
            **vars(args),
            "symbols_file": str(args.symbols_file),
            "data_root": str(args.data_root),
            "chronos_cache": str(args.chronos_cache),
            "model_paths": [str(p) for p in model_paths],
            "output": str(args.output),
        },
        "summary": {
            "n_windows": len(window_payloads),
            "median_monthly_pct": float(np.median(monthly)) if monthly.size else 0.0,
            "p10_monthly_pct": float(np.percentile(monthly, 10)) if monthly.size else 0.0,
            "worst_drawdown_pct": float(np.max(drawdowns)) if drawdowns.size else 0.0,
            "n_negative_windows": int(np.sum(monthly < 0.0)) if monthly.size else 0,
            "n_attributed_symbols": len(attribution),
            "elapsed_seconds": time.perf_counter() - t0,
        },
        "candidate_positive_symbols": positive_symbols,
        "symbol_attribution": attribution,
        "windows": window_payloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
