"""Per-symbol attribution for XGB daily ensemble backtests.

This is a diagnostic companion to ``sweep_ensemble_grid``. It reuses the same
model loading, feature construction, score blending, and simulator knobs, then
aggregates simulated trades by symbol so we can tell whether a candidate edge
is broad, concentrated, or just one name carrying the result.
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xgbnew.artifacts import (
    write_dataframe_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from xgbnew.backtest import (
    BacktestConfig,
    DayResult,
    DayTrade,
    _trade_allocation_weights,
    simulate,
)
from xgbnew.dataset import build_daily_dataset, load_chronos_cache, load_fm_latents
from xgbnew.features import (
    DAILY_DISPERSION_FEATURE_COLS,
    DAILY_RANK_FEATURE_COLS,
    LIVE_SUPPORTED_FEATURE_COLS,
)
from xgbnew.model_registry import load_any_model
from xgbnew.sweep_ensemble_grid import (
    FEE_REGIMES,
    _build_windows,
    _elapsed_window_days,
    _ensemble_score_mean_std,
    _infer_required_fm_latents,
    _monthly_return,
    _uncertainty_adjusted_scores,
)


logger = logging.getLogger(__name__)


@dataclass
class WindowAttribution:
    window_start: str
    window_end: str
    elapsed_days: int
    active_days: int
    total_return_pct: float
    monthly_return_pct: float
    sortino: float
    max_dd_pct: float
    top_symbol: str
    top_symbol_contribution_pct: float
    worst_symbol: str
    worst_symbol_contribution_pct: float


@dataclass
class SymbolAccumulator:
    symbol: str
    trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    unique_days: set[str] = field(default_factory=set)
    net_return_pcts: list[float] = field(default_factory=list)
    allocated_contribution_pcts: list[float] = field(default_factory=list)
    weighted_intraday_dd_pcts: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    windows_positive: int = 0
    windows_negative: int = 0
    best_window_contribution_pct: float = 0.0
    worst_window_contribution_pct: float = 0.0

    def add_trade(self, day: date, trade: DayTrade, weight: float) -> None:
        contribution = float(weight) * float(trade.net_return_pct)
        intraday_dd = float(weight) * float(trade.intraday_worst_dd_pct)
        self.trades += 1
        if int(trade.side) >= 0:
            self.long_trades += 1
        else:
            self.short_trades += 1
        self.unique_days.add(day.isoformat())
        self.net_return_pcts.append(float(trade.net_return_pct))
        self.allocated_contribution_pcts.append(contribution)
        self.weighted_intraday_dd_pcts.append(intraday_dd)
        self.scores.append(float(trade.score))

    def finish_window(self, contribution_pct: float) -> None:
        contribution_pct = float(contribution_pct)
        if contribution_pct > 0.0:
            self.windows_positive += 1
        elif contribution_pct < 0.0:
            self.windows_negative += 1
        self.best_window_contribution_pct = max(
            self.best_window_contribution_pct,
            contribution_pct,
        )
        self.worst_window_contribution_pct = min(
            self.worst_window_contribution_pct,
            contribution_pct,
        )

    def row(self) -> dict[str, Any]:
        net = np.asarray(self.net_return_pcts, dtype=np.float64)
        contrib = np.asarray(self.allocated_contribution_pcts, dtype=np.float64)
        scores = np.asarray(self.scores, dtype=np.float64)
        intraday = np.asarray(self.weighted_intraday_dd_pcts, dtype=np.float64)
        return {
            "symbol": self.symbol,
            "trades": int(self.trades),
            "unique_traded_days": int(len(self.unique_days)),
            "long_trades": int(self.long_trades),
            "short_trades": int(self.short_trades),
            "portfolio_contribution_pct": float(contrib.sum()) if contrib.size else 0.0,
            "avg_allocated_contribution_pct": float(contrib.mean()) if contrib.size else 0.0,
            "median_allocated_contribution_pct": float(np.median(contrib)) if contrib.size else 0.0,
            "win_rate_pct": float(np.mean(contrib > 0.0) * 100.0) if contrib.size else 0.0,
            "avg_trade_net_return_pct": float(net.mean()) if net.size else 0.0,
            "median_trade_net_return_pct": float(np.median(net)) if net.size else 0.0,
            "worst_trade_net_return_pct": float(net.min()) if net.size else 0.0,
            "best_trade_net_return_pct": float(net.max()) if net.size else 0.0,
            "max_weighted_intraday_dd_pct": float(intraday.max()) if intraday.size else 0.0,
            "avg_score": float(scores.mean()) if scores.size else 0.0,
            "median_score": float(np.median(scores)) if scores.size else 0.0,
            "windows_positive": int(self.windows_positive),
            "windows_negative": int(self.windows_negative),
            "best_window_contribution_pct": float(self.best_window_contribution_pct),
            "worst_window_contribution_pct": float(self.worst_window_contribution_pct),
        }


@dataclass
class ReplayArtifacts:
    oos_df: pd.DataFrame
    scores: pd.Series
    score_std: pd.Series
    models: list[Any]
    ensemble_needs_ranks: bool
    ensemble_needs_dispersion: bool


def _parse_symbols_file(path: Path) -> list[str]:
    out: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            symbol = line.strip()
            if symbol and not symbol.startswith("#"):
                out.append(symbol)
    return out


def _resolve_model_paths(spec: str) -> list[Path]:
    out: list[Path] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if any(ch in token for ch in "*?["):
            out.extend(Path(match) for match in sorted(glob.glob(token)))  # noqa: PTH207
        else:
            out.append(Path(token))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in out:
        key = str(path.expanduser().resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _parse_date_arg(name: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"--{name} must be an ISO date, got {value!r}") from exc


def _finite_float_arg(name: str, value: float, *, min_value: float | None = None) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"--{name} must be finite")
    if min_value is not None and out < min_value:
        raise ValueError(f"--{name} must be >= {min_value:g}")
    return out


def _load_models_and_feature_mode(model_paths: list[Path]) -> tuple[list[Any], tuple[str, ...], bool, bool, int | None]:
    if not model_paths:
        raise ValueError("no model paths resolved")
    models: list[Any] = []
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(f"model path not found: {path}")
        models.append(load_any_model(path))

    feature_cols: tuple[str, ...] | None = None
    first_path: Path | None = None
    fm_n_latents: int | None = None
    for model, path in zip(models, model_paths, strict=True):
        raw = getattr(model, "feature_cols", None)
        if (
            not isinstance(raw, (list, tuple))
            or not raw
            or not all(isinstance(col, str) and col for col in raw)
        ):
            raise ValueError(f"{path}: model feature_cols must be a non-empty list")
        cols = tuple(raw)
        unsupported = sorted(set(cols) - LIVE_SUPPORTED_FEATURE_COLS)
        if unsupported:
            raise ValueError(
                f"{path}: model feature_cols contains unsupported live features: {unsupported}",
            )
        if feature_cols is None:
            feature_cols = cols
            first_path = path
        elif cols != feature_cols:
            raise ValueError(
                f"ensemble feature_cols mismatch: {path} does not match {first_path}",
            )
        inferred = _infer_required_fm_latents(cols, path)
        if inferred is not None:
            fm_n_latents = max(fm_n_latents or 0, int(inferred))

    assert feature_cols is not None
    needs_ranks = any(col in feature_cols for col in DAILY_RANK_FEATURE_COLS)
    needs_disp = any(col in feature_cols for col in DAILY_DISPERSION_FEATURE_COLS)
    return models, feature_cols, needs_ranks, needs_disp, fm_n_latents


def _load_spy_close_by_date(path: Path | None) -> pd.Series:
    if path is None or not path.exists():
        raise FileNotFoundError(f"SPY CSV not found: {path}")
    spy_df = pd.read_csv(path, usecols=["timestamp", "close"])
    spy_df["timestamp"] = pd.to_datetime(spy_df["timestamp"], utc=True, errors="coerce")
    spy_df = spy_df.dropna(subset=["timestamp", "close"]).drop_duplicates(subset=["timestamp"])
    spy_df["date"] = spy_df["timestamp"].dt.date
    spy_df = spy_df.sort_values("timestamp")
    return spy_df.groupby("date")["close"].last().astype(float).sort_index()


def build_replay_artifacts(
    *,
    symbols: list[str],
    data_root: Path,
    model_paths: list[Path],
    train_start: date,
    train_end: date,
    oos_start: date,
    oos_end: date,
    blend_mode: str,
    chronos_cache_path: Path | None,
    fm_latents_path: Path | None,
    fm_n_latents: int | None,
    min_dollar_vol: float,
    fast_features: bool,
    score_uncertainty_penalty: float,
) -> ReplayArtifacts:
    models, _, needs_ranks, needs_disp, inferred_fm_n = _load_models_and_feature_mode(model_paths)

    chronos_cache = None
    if chronos_cache_path is not None and chronos_cache_path.exists():
        chronos_cache = load_chronos_cache(chronos_cache_path)

    fm_latents = None
    selected_fm_n = fm_n_latents
    if inferred_fm_n is not None:
        if fm_latents_path is None:
            raise ValueError("model feature_cols require FM latents; pass --fm-latents-path")
        selected_fm_n = max(int(selected_fm_n or 0), int(inferred_fm_n))
    if fm_latents_path is not None:
        if selected_fm_n is None or int(selected_fm_n) <= 0:
            raise ValueError("--fm-n-latents must be positive when --fm-latents-path is used")
        fm_latents = load_fm_latents(fm_latents_path)
        if fm_latents is None:
            raise ValueError(f"--fm-latents-path not found: {fm_latents_path}")

    _, _, oos_df = build_daily_dataset(
        data_root=data_root,
        symbols=symbols,
        train_start=train_start,
        train_end=train_end,
        val_start=oos_start,
        val_end=oos_end,
        test_start=oos_start,
        test_end=oos_end,
        chronos_cache=chronos_cache,
        min_dollar_vol=float(min_dollar_vol),
        fast_features=bool(fast_features),
        include_cross_sectional_ranks=needs_ranks,
        include_cross_sectional_dispersion=needs_disp,
        fm_latents=fm_latents,
        fm_n_latents=int(selected_fm_n) if fm_latents is not None else None,
    )
    scores, score_std = _ensemble_score_mean_std(oos_df, models, blend_mode)
    scores = _uncertainty_adjusted_scores(scores, score_std, score_uncertainty_penalty)
    return ReplayArtifacts(
        oos_df=oos_df,
        scores=scores,
        score_std=score_std,
        models=models,
        ensemble_needs_ranks=needs_ranks,
        ensemble_needs_dispersion=needs_disp,
    )


def _window_symbol_contributions(
    day_results: list[DayResult],
    config: BacktestConfig,
    accumulators: dict[str, SymbolAccumulator],
) -> dict[str, float]:
    contributions: dict[str, float] = {}
    for day_result in day_results:
        trades = list(day_result.trades)
        weights = _trade_allocation_weights(
            trades,
            mode=config.allocation_mode,
            temperature=config.allocation_temp,
            short_allocation_scale=config.short_allocation_scale,
            min_secondary_allocation=config.min_secondary_allocation,
        )
        for trade, weight in zip(trades, weights, strict=True):
            symbol = str(trade.symbol)
            accumulator = accumulators.setdefault(symbol, SymbolAccumulator(symbol=symbol))
            accumulator.add_trade(day_result.day, trade, float(weight))
            contributions[symbol] = contributions.get(symbol, 0.0) + float(weight) * float(trade.net_return_pct)
    return contributions


def aggregate_symbol_attribution(
    *,
    window_results: list[tuple[date, date, int, Any]],
    config: BacktestConfig,
) -> tuple[list[dict[str, Any]], list[WindowAttribution]]:
    accumulators: dict[str, SymbolAccumulator] = {}
    window_rows: list[WindowAttribution] = []

    for w_start, w_end, elapsed_days, result in window_results:
        by_symbol = _window_symbol_contributions(
            list(result.day_results),
            config,
            accumulators,
        )
        for symbol, contribution in by_symbol.items():
            accumulators[symbol].finish_window(contribution)

        sorted_contrib = sorted(by_symbol.items(), key=lambda item: item[1])
        worst_symbol, worst_contribution = sorted_contrib[0] if sorted_contrib else ("", 0.0)
        top_symbol, top_contribution = sorted_contrib[-1] if sorted_contrib else ("", 0.0)
        window_rows.append(
            WindowAttribution(
                window_start=w_start.isoformat(),
                window_end=w_end.isoformat(),
                elapsed_days=int(elapsed_days),
                active_days=int(len(result.day_results)),
                total_return_pct=float(result.total_return_pct),
                monthly_return_pct=float(_monthly_return(result.total_return_pct, max(int(elapsed_days), 1)) * 100.0),
                sortino=float(result.sortino_ratio),
                max_dd_pct=float(result.max_drawdown_pct),
                top_symbol=top_symbol,
                top_symbol_contribution_pct=float(top_contribution),
                worst_symbol=worst_symbol,
                worst_symbol_contribution_pct=float(worst_contribution),
            ),
        )

    rows = [acc.row() for acc in accumulators.values()]
    rows.sort(
        key=lambda row: (
            float(row["portfolio_contribution_pct"]),
            float(row["avg_allocated_contribution_pct"]),
        ),
        reverse=True,
    )
    return rows, window_rows


def replay_attribution(
    *,
    artifacts: ReplayArtifacts,
    window_days: int,
    stride_days: int,
    config: BacktestConfig,
    spy_close_by_date: pd.Series | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[WindowAttribution]]:
    all_days = sorted(artifacts.oos_df["date"].unique())
    windows = _build_windows(all_days, int(window_days), int(stride_days))
    if not windows:
        raise RuntimeError("no eval windows; check OOS date range")

    window_results: list[tuple[date, date, int, Any]] = []
    for w_start, w_end in windows:
        w_df = artifacts.oos_df[
            (artifacts.oos_df["date"] >= w_start)
            & (artifacts.oos_df["date"] <= w_end)
        ]
        if len(w_df) < 5:
            continue
        w_scores = artifacts.scores.loc[w_df.index]
        result = simulate(
            w_df,
            None,  # type: ignore[arg-type]
            config,
            precomputed_scores=w_scores,
            spy_close_by_date=spy_close_by_date,
        )
        elapsed_days = _elapsed_window_days(w_df, result)
        window_results.append((w_start, w_end, elapsed_days, result))

    symbol_rows, window_rows = aggregate_symbol_attribution(
        window_results=window_results,
        config=config,
    )
    monthly = np.asarray([row.monthly_return_pct for row in window_rows], dtype=np.float64)
    dds = np.asarray([row.max_dd_pct for row in window_rows], dtype=np.float64)
    sortinos = np.asarray([row.sortino for row in window_rows], dtype=np.float64)
    active = np.asarray(
        [
            float(row.active_days) / max(float(row.elapsed_days), 1.0) * 100.0
            for row in window_rows
        ],
        dtype=np.float64,
    )
    summary = {
        "n_windows": int(len(window_rows)),
        "median_monthly_pct": float(np.median(monthly)) if monthly.size else 0.0,
        "p10_monthly_pct": float(np.percentile(monthly, 10)) if monthly.size else 0.0,
        "median_sortino": float(np.median(sortinos)) if sortinos.size else 0.0,
        "worst_dd_pct": float(dds.max()) if dds.size else 0.0,
        "n_neg": int(np.sum(monthly < 0.0)) if monthly.size else 0,
        "median_active_day_pct": float(np.median(active)) if active.size else 0.0,
        "window_monthly_return_pcts": [float(x) for x in monthly],
        "window_drawdown_pcts": [float(x) for x in dds],
    }
    return summary, symbol_rows, window_rows


def _build_config(args: argparse.Namespace) -> BacktestConfig:
    if args.fee_regime not in FEE_REGIMES:
        raise ValueError(f"unknown fee regime: {args.fee_regime}")
    fees = dict(FEE_REGIMES[str(args.fee_regime)])
    fill_buffer = (
        float(args.fill_buffer_bps)
        if float(args.fill_buffer_bps) >= 0.0
        else float(fees["fill_buffer_bps"])
    )
    return BacktestConfig(
        top_n=int(args.top_n),
        short_n=int(args.short_n),
        max_short_score=float(args.max_short_score),
        short_allocation_scale=float(args.short_allocation_scale),
        min_picks=int(args.min_picks),
        leverage=float(args.leverage),
        xgb_weight=1.0,
        min_score=float(args.min_score),
        min_dollar_vol=float(args.inference_min_dolvol),
        max_spread_bps=float(args.inference_max_spread_bps),
        min_vol_20d=float(args.inference_min_vol_20d),
        max_vol_20d=float(args.inference_max_vol_20d),
        fee_rate=float(fees["fee_rate"]),
        commission_bps=float(fees["commission_bps"]),
        fill_buffer_bps=fill_buffer,
        opportunistic_watch_n=int(args.opportunistic_watch_n),
        opportunistic_entry_discount_bps=float(args.opportunistic_entry_discount_bps),
        hold_through=bool(args.hold_through),
        skip_prob=float(args.skip_prob),
        skip_seed=int(args.skip_seed),
        regime_gate_window=int(args.regime_gate_window),
        vol_target_ann=float(args.vol_target_ann),
        inv_vol_target_ann=float(args.inv_vol_target_ann),
        inv_vol_floor=float(args.inv_vol_floor),
        inv_vol_cap=float(args.inv_vol_cap),
        max_ret_20d_rank_pct=float(args.max_ret_20d_rank_pct),
        min_ret_5d_rank_pct=float(args.min_ret_5d_rank_pct),
        regime_cs_iqr_max=float(args.regime_cs_iqr_max),
        regime_cs_skew_min=float(args.regime_cs_skew_min),
        no_picks_fallback_symbol=str(args.no_picks_fallback_symbol or ""),
        no_picks_fallback_alloc_scale=float(args.no_picks_fallback_alloc_scale),
        conviction_scaled_alloc=bool(args.conviction_scaled_alloc),
        conviction_alloc_low=float(args.conviction_alloc_low),
        conviction_alloc_high=float(args.conviction_alloc_high),
        allocation_mode=str(args.allocation_mode),
        allocation_temp=float(args.allocation_temp),
        min_secondary_allocation=float(args.min_secondary_allocation),
        overnight_max_gross_leverage=(
            None
            if args.overnight_max_gross_leverage is None
            else float(args.overnight_max_gross_leverage)
        ),
    )


def _markdown_report(
    *,
    summary: dict[str, Any],
    config: BacktestConfig,
    symbol_rows: list[dict[str, Any]],
    window_rows: list[WindowAttribution],
    output_json: Path,
) -> str:
    def _fmt(value: Any) -> str:
        return f"{float(value):+.2f}"

    lines = [
        "# XGB Symbol Attribution",
        "",
        f"- JSON: `{output_json}`",
        f"- Windows: {summary['n_windows']}",
        f"- Median monthly: {_fmt(summary['median_monthly_pct'])}%",
        f"- P10 monthly: {_fmt(summary['p10_monthly_pct'])}%",
        f"- Worst drawdown: {float(summary['worst_dd_pct']):.2f}%",
        f"- Negative windows: {summary['n_neg']}",
        f"- Active day median: {float(summary['median_active_day_pct']):.2f}%",
        f"- Config: top_n={config.top_n}, leverage={config.leverage:.2f}, "
        f"hold_through={config.hold_through}, min_score={config.min_score:.3f}, "
        f"min_vol_20d={config.min_vol_20d:.3f}, min_dolvol={config.min_dollar_vol:.0f}",
        "",
        "## Top Contributors",
        "",
        "| symbol | trades | unique days | contribution % | win % | worst trade % | max intraday dd % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in symbol_rows[:12]:
        lines.append(
            f"| {row['symbol']} | {row['trades']} | {row['unique_traded_days']} | "
            f"{float(row['portfolio_contribution_pct']):+.2f} | "
            f"{float(row['win_rate_pct']):.1f} | "
            f"{float(row['worst_trade_net_return_pct']):+.2f} | "
            f"{float(row['max_weighted_intraday_dd_pct']):.2f} |",
        )

    lines.extend([
        "",
        "## Worst Contributors",
        "",
        "| symbol | trades | unique days | contribution % | win % | worst trade % | max intraday dd % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in list(reversed(symbol_rows[-12:])):
        lines.append(
            f"| {row['symbol']} | {row['trades']} | {row['unique_traded_days']} | "
            f"{float(row['portfolio_contribution_pct']):+.2f} | "
            f"{float(row['win_rate_pct']):.1f} | "
            f"{float(row['worst_trade_net_return_pct']):+.2f} | "
            f"{float(row['max_weighted_intraday_dd_pct']):.2f} |",
        )

    lines.extend([
        "",
        "## Windows",
        "",
        "| start | end | monthly % | dd % | top symbol | top % | worst symbol | worst % |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | ---: |",
    ])
    for row in window_rows:
        lines.append(
            f"| {row.window_start} | {row.window_end} | "
            f"{row.monthly_return_pct:+.2f} | {row.max_dd_pct:.2f} | "
            f"{row.top_symbol} | {row.top_symbol_contribution_pct:+.2f} | "
            f"{row.worst_symbol} | {row.worst_symbol_contribution_pct:+.2f} |",
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    parser.add_argument("--model-paths", required=True, help="Comma-separated paths; globs are accepted.")
    parser.add_argument("--blend-mode", choices=["mean", "median"], default="mean")
    parser.add_argument("--chronos-cache", type=Path, default=Path("analysis/xgbnew_daily/chronos_cache.parquet"))
    parser.add_argument("--fm-latents-path", type=Path, default=None)
    parser.add_argument("--fm-n-latents", type=int, default=0)
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--oos-start", default="2025-01-02")
    parser.add_argument("--oos-end", default="")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--stride-days", type=int, default=15)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--short-n", type=int, default=0)
    parser.add_argument("--max-short-score", type=float, default=0.45)
    parser.add_argument("--short-allocation-scale", type=float, default=0.5)
    parser.add_argument("--min-picks", type=int, default=0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--fee-regime", choices=list(FEE_REGIMES), default="prod10bps")
    parser.add_argument("--fill-buffer-bps", type=float, default=-1.0, help="-1 uses the fee regime default.")
    parser.add_argument("--inference-min-dolvol", type=float, default=5_000_000.0)
    parser.add_argument("--inference-max-spread-bps", type=float, default=30.0)
    parser.add_argument("--inference-min-vol-20d", type=float, default=0.0)
    parser.add_argument("--inference-max-vol-20d", type=float, default=0.0)
    parser.add_argument("--opportunistic-watch-n", type=int, default=0)
    parser.add_argument("--opportunistic-entry-discount-bps", type=float, default=0.0)
    parser.add_argument("--hold-through", action="store_true")
    parser.add_argument("--skip-prob", type=float, default=0.0)
    parser.add_argument("--skip-seed", type=int, default=0)
    parser.add_argument("--regime-gate-window", type=int, default=0)
    parser.add_argument("--vol-target-ann", type=float, default=0.0)
    parser.add_argument("--spy-csv", type=Path, default=Path("trainingdata/SPY.csv"))
    parser.add_argument("--inv-vol-target-ann", type=float, default=0.0)
    parser.add_argument("--inv-vol-floor", type=float, default=0.05)
    parser.add_argument("--inv-vol-cap", type=float, default=3.0)
    parser.add_argument("--max-ret-20d-rank-pct", type=float, default=1.0)
    parser.add_argument("--min-ret-5d-rank-pct", type=float, default=0.0)
    parser.add_argument("--regime-cs-iqr-max", type=float, default=0.0)
    parser.add_argument("--regime-cs-skew-min", type=float, default=-1e9)
    parser.add_argument("--no-picks-fallback-symbol", default="")
    parser.add_argument("--no-picks-fallback-alloc-scale", type=float, default=0.0)
    parser.add_argument("--conviction-scaled-alloc", action="store_true")
    parser.add_argument("--conviction-alloc-low", type=float, default=0.55)
    parser.add_argument("--conviction-alloc-high", type=float, default=0.85)
    parser.add_argument("--allocation-mode", choices=["equal", "score_norm", "softmax", "worksteal"], default="equal")
    parser.add_argument("--allocation-temp", type=float, default=1.0)
    parser.add_argument("--min-secondary-allocation", type=float, default=0.0)
    parser.add_argument("--score-uncertainty-penalty", type=float, default=0.0)
    parser.add_argument("--overnight-max-gross-leverage", type=float, default=None)
    parser.add_argument("--fast-features", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/xgbnew_symbol_attribution"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[date, date, date, date]:
    train_start = _parse_date_arg("train-start", str(args.train_start))
    train_end = _parse_date_arg("train-end", str(args.train_end))
    oos_start = _parse_date_arg("oos-start", str(args.oos_start))
    oos_end = (
        _parse_date_arg("oos-end", str(args.oos_end))
        if args.oos_end
        else datetime.now(UTC).date()
    )
    if train_start > train_end:
        raise ValueError("--train-start must be <= --train-end")
    if oos_start > oos_end:
        raise ValueError("--oos-start must be <= --oos-end")
    if int(args.window_days) <= 0:
        raise ValueError("--window-days must be positive")
    if int(args.stride_days) <= 0:
        raise ValueError("--stride-days must be positive")
    if int(args.top_n) <= 0:
        raise ValueError("--top-n must be positive")
    if int(args.short_n) < 0:
        raise ValueError("--short-n must be nonnegative")
    if int(args.min_picks) < 0:
        raise ValueError("--min-picks must be nonnegative")
    leverage = _finite_float_arg("leverage", args.leverage, min_value=0.0)
    if leverage <= 0.0:
        raise ValueError("--leverage must be positive")
    _finite_float_arg("min-score", args.min_score)
    _finite_float_arg("fill-buffer-bps", args.fill_buffer_bps, min_value=-1.0)
    _finite_float_arg("inference-min-dolvol", args.inference_min_dolvol, min_value=0.0)
    _finite_float_arg("inference-max-spread-bps", args.inference_max_spread_bps, min_value=0.0)
    _finite_float_arg("inference-min-vol-20d", args.inference_min_vol_20d, min_value=0.0)
    _finite_float_arg("inference-max-vol-20d", args.inference_max_vol_20d, min_value=0.0)
    _finite_float_arg("skip-prob", args.skip_prob, min_value=0.0)
    if float(args.skip_prob) > 1.0:
        raise ValueError("--skip-prob must be <= 1")
    _finite_float_arg("allocation-temp", args.allocation_temp, min_value=1e-12)
    min_secondary = _finite_float_arg(
        "min-secondary-allocation",
        args.min_secondary_allocation,
        min_value=0.0,
    )
    if min_secondary > 1.0:
        raise ValueError("--min-secondary-allocation must be <= 1")
    _finite_float_arg("score-uncertainty-penalty", args.score_uncertainty_penalty, min_value=0.0)
    _finite_float_arg("inv-vol-floor", args.inv_vol_floor, min_value=1e-12)
    _finite_float_arg("inv-vol-cap", args.inv_vol_cap, min_value=1e-12)
    if int(args.fm_n_latents) < 0:
        raise ValueError("--fm-n-latents must be nonnegative")
    if args.fm_latents_path is not None and not Path(args.fm_latents_path).exists():
        raise ValueError(f"--fm-latents-path not found: {args.fm_latents_path}")
    return train_start, train_end, oos_start, oos_end


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        train_start, train_end, oos_start, oos_end = _validate_args(args)
        symbols = _parse_symbols_file(args.symbols_file)
        model_paths = _resolve_model_paths(args.model_paths)
        if not model_paths:
            raise ValueError("no model paths resolved")
        config = _build_config(args)
        spy_close = None
        if int(args.regime_gate_window) > 0 or float(args.vol_target_ann) > 0.0:
            spy_close = _load_spy_close_by_date(args.spy_csv)
        artifacts = build_replay_artifacts(
            symbols=symbols,
            data_root=args.data_root,
            model_paths=model_paths,
            train_start=train_start,
            train_end=train_end,
            oos_start=oos_start,
            oos_end=oos_end,
            blend_mode=str(args.blend_mode),
            chronos_cache_path=args.chronos_cache,
            fm_latents_path=args.fm_latents_path,
            fm_n_latents=(int(args.fm_n_latents) if int(args.fm_n_latents) > 0 else None),
            min_dollar_vol=float(args.inference_min_dolvol),
            fast_features=bool(args.fast_features),
            score_uncertainty_penalty=float(args.score_uncertainty_penalty),
        )
        summary, symbol_rows, window_rows = replay_attribution(
            artifacts=artifacts,
            window_days=int(args.window_days),
            stride_days=int(args.stride_days),
            config=config,
            spy_close_by_date=spy_close,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"symbol_attribution: {exc}", flush=True)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"symbol_attribution_{timestamp}"
    json_path = args.output_dir / f"{stem}.json"
    symbols_csv = args.output_dir / f"{stem}_symbols.csv"
    windows_csv = args.output_dir / f"{stem}_windows.csv"
    md_path = args.output_dir / f"{stem}.md"

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "symbols_file": str(args.symbols_file),
        "data_root": str(args.data_root),
        "model_paths": [str(path) for path in model_paths],
        "oos_start": oos_start.isoformat(),
        "oos_end": oos_end.isoformat(),
        "window_days": int(args.window_days),
        "stride_days": int(args.stride_days),
        "blend_mode": str(args.blend_mode),
        "ensemble_needs_ranks": bool(artifacts.ensemble_needs_ranks),
        "ensemble_needs_dispersion": bool(artifacts.ensemble_needs_dispersion),
        "config": asdict(config),
        "summary": summary,
        "symbols": symbol_rows,
        "windows": [asdict(row) for row in window_rows],
    }
    write_json_atomic(json_path, payload)
    write_dataframe_csv_atomic(symbols_csv, pd.DataFrame(symbol_rows))
    write_dataframe_csv_atomic(windows_csv, pd.DataFrame([asdict(row) for row in window_rows]))
    write_text_atomic(
        md_path,
        _markdown_report(
            summary=summary,
            config=config,
            symbol_rows=symbol_rows,
            window_rows=window_rows,
            output_json=json_path,
        ),
    )
    print(f"[symbol-attribution] wrote {json_path}", flush=True)
    print(f"[symbol-attribution] wrote {symbols_csv}", flush=True)
    print(f"[symbol-attribution] wrote {windows_csv}", flush=True)
    print(f"[symbol-attribution] wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
