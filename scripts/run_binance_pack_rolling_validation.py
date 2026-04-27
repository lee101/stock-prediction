#!/usr/bin/env python3
"""Validate Binance portfolio-pack configs across rolling 120d windows.

This is the guardrail after a sweep finds attractive single-window PnL: train
the same forecaster shape on each window, replay candidate pack configs under
multiple binary-fill slippage cells, then rank by worst-case robustness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.sweep_binance_hourly_portfolio_pack import (
    PackConfig,
    _discover_symbols,
    _filter_liquid_frames,
    _load_hourly_frames,
    _parse_float_list,
    _to_utc,
    build_model_frame,
    evaluate_pack,
    fit_forecasters,
    score_eval_rows,
)


PACK_FIELDS = tuple(field.name for field in fields(PackConfig))
PACK_INT_FIELDS = {"max_positions", "max_pending_entries", "entry_ttl_hours", "max_hold_hours"}
PACK_STR_FIELDS = {"entry_selection_mode", "entry_allocator_mode"}
PACK_DEFAULTS: dict[str, Any] = {
    "min_recent_ret_24h": -1.0,
    "min_recent_ret_72h": -1.0,
    "max_recent_vol_72h": 0.0,
}

SUMMARY_FIELDS = [
    "candidate_id",
    "base_candidate_id",
    "min_take_profit_bps",
    "source",
    "source_row",
    "cells",
    "negative_cells",
    "median_monthly_return_pct",
    "worst_monthly_return_pct",
    "mean_monthly_return_pct",
    "best_monthly_return_pct",
    "median_total_return_pct",
    "worst_total_return_pct",
    "median_sortino",
    "worst_sortino",
    "median_max_drawdown_pct",
    "worst_max_drawdown_pct",
    "min_num_sells",
    "median_num_sells",
    "rolling_score",
    "config_json",
]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source: str
    source_row: int
    source_rank_value: float
    cfg: PackConfig


def _split_paths(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        for part in str(value).split(","):
            token = part.strip()
            if token:
                paths.append(Path(token))
    if not paths:
        raise ValueError("at least one candidate CSV is required")
    return paths


def _float_or_default(value: Any, default: Any = None) -> float:
    if value is None or value == "":
        if default is None:
            raise ValueError("missing required float value")
        return float(default)
    return float(value)


def _candidate_from_row(row: dict[str, Any]) -> PackConfig:
    values: dict[str, Any] = {}
    for name in PACK_FIELDS:
        raw = row.get(name, PACK_DEFAULTS.get(name))
        if name in PACK_STR_FIELDS:
            if raw is None or str(raw).strip() == "":
                raise ValueError(f"candidate row missing {name}")
            values[name] = str(raw)
        elif name in PACK_INT_FIELDS:
            values[name] = int(float(_float_or_default(raw, PACK_DEFAULTS.get(name))))
        else:
            values[name] = float(_float_or_default(raw, PACK_DEFAULTS.get(name)))
    return PackConfig(**values)


def _config_hash(cfg: PackConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def load_candidate_configs(
    paths: Sequence[Path],
    *,
    top_k_per_file: int,
    rank_by: str,
    min_source_sells: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for path in paths:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        rank_key = rank_by if rank_by in rows[0] else "monthly_return_pct"

        def rank_value(item: tuple[int, dict[str, str]]) -> float:
            try:
                return float(item[1].get(rank_key, "-inf"))
            except ValueError:
                return float("-inf")

        ranked = sorted(enumerate(rows), key=rank_value, reverse=True)
        if int(top_k_per_file) > 0:
            ranked = ranked[: int(top_k_per_file)]
        for row_idx, row in ranked:
            if int(float(row.get("num_sells", 0) or 0)) < int(min_source_sells):
                continue
            cfg = _candidate_from_row(row)
            candidate_id = f"cfg_{_config_hash(cfg)}"
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    source=str(path),
                    source_row=int(row_idx),
                    source_rank_value=rank_value((row_idx, row)),
                    cfg=cfg,
                )
            )
    if not candidates:
        raise ValueError("no candidate configs survived filtering")
    return candidates


def rolling_score(summary: dict[str, Any], *, min_trades: int) -> float:
    """Rank configs by worst-window return, stability, and non-idle behavior."""
    trade_shortfall = max(0.0, float(min_trades) - float(summary["min_num_sells"]))
    return float(
        summary["worst_monthly_return_pct"]
        + 0.50 * summary["median_monthly_return_pct"]
        + 2.0 * summary["median_sortino"]
        - 1.25 * summary["worst_max_drawdown_pct"]
        - 18.0 * summary["negative_cells"]
        - 2.0 * trade_shortfall
    )


def summarize_results(rows: list[dict[str, Any]], *, min_trades: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for candidate_id, group in frame.groupby("candidate_id", sort=False):
        first = group.iloc[0].to_dict()
        monthly = group["monthly_return_pct"].astype(float)
        total = group["total_return_pct"].astype(float)
        sortino = group["sortino"].astype(float)
        drawdown = group["max_drawdown_pct"].astype(float)
        sells = group["num_sells"].astype(float)
        summary = {
            "candidate_id": candidate_id,
            "base_candidate_id": first.get("base_candidate_id", candidate_id),
            "min_take_profit_bps": float(first.get("min_take_profit_bps", 0.0)),
            "source": first["source"],
            "source_row": int(first["source_row"]),
            "cells": int(len(group)),
            "negative_cells": int((monthly < 0.0).sum()),
            "median_monthly_return_pct": float(monthly.median()),
            "worst_monthly_return_pct": float(monthly.min()),
            "mean_monthly_return_pct": float(monthly.mean()),
            "best_monthly_return_pct": float(monthly.max()),
            "median_total_return_pct": float(total.median()),
            "worst_total_return_pct": float(total.min()),
            "median_sortino": float(sortino.median()),
            "worst_sortino": float(sortino.min()),
            "median_max_drawdown_pct": float(drawdown.median()),
            "worst_max_drawdown_pct": float(drawdown.max()),
            "min_num_sells": int(sells.min()),
            "median_num_sells": float(sells.median()),
            "config_json": first["config_json"],
        }
        summary["rolling_score"] = rolling_score(summary, min_trades=min_trades)
        summaries.append(summary)
    summaries.sort(key=lambda item: float(item["rolling_score"]), reverse=True)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _eval_args(
    args: argparse.Namespace,
    *,
    fill_buffer_bps: float,
    min_take_profit_bps: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        min_take_profit_bps=float(min_take_profit_bps),
        max_entry_gap_bps=float(args.max_entry_gap_bps),
        max_exit_gap_bps=float(args.max_exit_gap_bps),
        fee_rate=float(args.fee_rate),
        top_candidates_per_hour=int(args.top_candidates_per_hour),
        initial_cash=float(args.initial_cash),
        entry_intensity_power=float(args.entry_intensity_power),
        entry_min_intensity_fraction=float(args.entry_min_intensity_fraction),
        force_close_slippage_bps=float(args.force_close_slippage_bps),
        margin_annual_rate=float(args.margin_annual_rate),
        decision_lag=int(args.decision_lag),
        entry_allocator_max_single_position_fraction=float(args.entry_allocator_max_single_position_fraction),
        entry_allocator_reserve_fraction=float(args.entry_allocator_reserve_fraction),
        fill_buffer_bps=float(fill_buffer_bps),
        min_result_trades=int(args.min_result_trades),
        disable_drawdown_profit_early_exit=bool(args.disable_drawdown_profit_early_exit),
    )


def _load_scored_window(
    raw_frames: dict[str, pd.DataFrame],
    *,
    args: argparse.Namespace,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frames, liquidity_metrics = _filter_liquid_frames(
        raw_frames,
        end=end,
        lookback_days=int(args.liquidity_lookback_days),
        min_median_dollar_volume=float(args.min_median_dollar_volume),
        max_symbols=int(args.max_symbols_by_dollar_volume),
    )
    if not liquidity_metrics.empty:
        top = liquidity_metrics[liquidity_metrics["symbol"].isin(frames)].head(8)
        print(
            f"liquidity selected {len(frames)}/{len(liquidity_metrics)} symbols: "
            + ",".join(f"{row.symbol}:{row.median_dollar_volume:.0f}" for row in top.itertuples(index=False)),
            flush=True,
        )
    eval_start = end - pd.Timedelta(days=int(args.eval_days))
    train_start = eval_start - pd.Timedelta(days=int(args.train_days))
    feature_start = train_start - pd.Timedelta(days=10)
    model_frame, feature_cols = build_model_frame(
        frames,
        start=feature_start,
        end=end,
        horizon=int(args.label_horizon),
    )
    print(
        f"window end={end.isoformat()} rows={len(model_frame):,} symbols={model_frame['symbol'].nunique()} "
        f"train=[{train_start}, {eval_start}) eval=[{eval_start}, {end}]",
        flush=True,
    )
    models = fit_forecasters(
        model_frame,
        feature_cols,
        train_end=eval_start,
        rounds=int(args.rounds),
        device=str(args.device),
    )
    scored = score_eval_rows(model_frame, feature_cols, models, eval_start=eval_start, eval_end=end)
    print(f"scored rows={len(scored):,} symbols={scored['symbol'].nunique()}", flush=True)
    return scored


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling validation for Binance portfolio-pack configs.")
    parser.add_argument("--candidate-csv", nargs="+", required=True)
    parser.add_argument("--top-k-per-file", type=int, default=12)
    parser.add_argument("--rank-by", default="selection_score")
    parser.add_argument("--min-source-sells", type=int, default=0)

    parser.add_argument("--hourly-root", type=Path, default=Path("binance_spot_hourly"))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--min-bars", type=int, default=5000)
    parser.add_argument("--min-symbols-per-hour", type=int, default=20)
    parser.add_argument("--liquidity-lookback-days", type=int, default=90)
    parser.add_argument("--min-median-dollar-volume", type=float, default=0.0)
    parser.add_argument("--max-symbols-by-dollar-volume", type=int, default=0)
    parser.add_argument("--train-days", type=int, default=720)
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--end-dates", default="2025-07-22T21:00:00+00:00,2025-11-19T21:00:00+00:00,2026-03-19T21:00:00+00:00")
    parser.add_argument("--label-horizon", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fill-buffer-bps-grid", default="5,10,20")

    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--force-close-slippage-bps", type=float, default=10.0)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0625)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--min-take-profit-bps", type=float, default=35.0)
    parser.add_argument(
        "--min-take-profit-bps-grid",
        default="",
        help="Optional comma grid for exit-level validation; defaults to --min-take-profit-bps.",
    )
    parser.add_argument("--max-entry-gap-bps", type=float, default=120.0)
    parser.add_argument("--max-exit-gap-bps", type=float, default=250.0)
    parser.add_argument("--top-candidates-per-hour", type=int, default=15)
    parser.add_argument("--entry-intensity-power", type=float, default=1.0)
    parser.add_argument("--entry-min-intensity-fraction", type=float, default=0.0)
    parser.add_argument("--entry-allocator-max-single-position-fraction", type=float, default=0.35)
    parser.add_argument("--entry-allocator-reserve-fraction", type=float, default=0.05)
    parser.add_argument("--min-result-trades", type=int, default=20)
    parser.set_defaults(disable_drawdown_profit_early_exit=True)
    parser.add_argument(
        "--allow-drawdown-profit-early-exit",
        dest="disable_drawdown_profit_early_exit",
        action="store_false",
        help="Use sweep-style fail-fast simulation instead of full-window validation.",
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path, default=Path(f"analysis/binance_pack_rolling_validation_{stamp}.csv"))
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_paths = _split_paths(args.candidate_csv)
    candidates = load_candidate_configs(
        candidate_paths,
        top_k_per_file=int(args.top_k_per_file),
        rank_by=str(args.rank_by),
        min_source_sells=int(args.min_source_sells),
    )
    end_dates = [_to_utc(value) for value in str(args.end_dates).split(",") if value.strip()]
    fill_buffers = _parse_float_list(str(args.fill_buffer_bps_grid))
    take_profit_values = (
        _parse_float_list(str(args.min_take_profit_bps_grid))
        if str(args.min_take_profit_bps_grid).strip()
        else [float(args.min_take_profit_bps)]
    )
    symbols = _discover_symbols(args.hourly_root, args.symbols)
    raw_frames = _load_hourly_frames(args.hourly_root, symbols=symbols, min_bars=int(args.min_bars))
    summary_path = args.summary_out or args.out.with_name(args.out.stem + "_summary.csv")
    manifest_path = args.out.with_name(args.out.stem + "_manifest.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "args": {key: str(value) for key, value in vars(args).items()},
                "candidate_count": len(candidates),
                "candidate_sources": [str(path) for path in candidate_paths],
            },
            indent=2,
        )
        + "\n"
    )

    detail_rows: list[dict[str, Any]] = []
    print(
        f"validating {len(candidates)} configs over {len(end_dates)} windows x "
        f"{len(fill_buffers)} fill cells x {len(take_profit_values)} take-profit cells"
    )
    for end_idx, end in enumerate(end_dates, start=1):
        scored = _load_scored_window(raw_frames, args=args, end=end)
        for fill_buffer_bps in fill_buffers:
            for min_take_profit_bps in take_profit_values:
                eval_args = _eval_args(
                    args,
                    fill_buffer_bps=float(fill_buffer_bps),
                    min_take_profit_bps=float(min_take_profit_bps),
                )
                for cand_idx, candidate in enumerate(candidates, start=1):
                    variant_id = (
                        candidate.candidate_id
                        if len(take_profit_values) == 1
                        else f"{candidate.candidate_id}_tp{float(min_take_profit_bps):g}"
                    )
                    row, _bars, _actions, _result = evaluate_pack(
                        scored,
                        cfg=candidate.cfg,
                        label_horizon=int(args.label_horizon),
                        args=eval_args,
                    )
                    row.update(
                        {
                            "candidate_id": variant_id,
                            "base_candidate_id": candidate.candidate_id,
                            "candidate_index": cand_idx,
                            "source": candidate.source,
                            "source_row": candidate.source_row,
                            "source_rank_value": candidate.source_rank_value,
                            "window_index": end_idx,
                            "validation_end": end.isoformat(),
                            "train_days": int(args.train_days),
                            "rounds": int(args.rounds),
                            "label_horizon": int(args.label_horizon),
                            "fill_buffer_bps": float(fill_buffer_bps),
                            "min_take_profit_bps": float(min_take_profit_bps),
                            "config_json": json.dumps(asdict(candidate.cfg), sort_keys=True),
                        }
                    )
                    detail_rows.append(row)
                    _write_csv(args.out, detail_rows)
                    summary_rows = summarize_results(detail_rows, min_trades=int(args.min_result_trades))
                    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
                    print(
                        f"window={end_idx}/{len(end_dates)} fill={fill_buffer_bps:g} "
                        f"tp={min_take_profit_bps:g} cfg={cand_idx}/{len(candidates)} "
                        f"monthly={row['monthly_return_pct']:+.2f}% ret={row['total_return_pct']:+.2f}% "
                        f"dd={row['max_drawdown_pct']:.2f}% trades={row['num_sells']}",
                        flush=True,
                    )

    summary_rows = summarize_results(detail_rows, min_trades=int(args.min_result_trades))
    print("\n=== Best rolling configs ===")
    for row in summary_rows[:10]:
        print(
            f"score={row['rolling_score']:+.2f} worst_monthly={row['worst_monthly_return_pct']:+.2f}% "
            f"median_monthly={row['median_monthly_return_pct']:+.2f}% "
            f"worst_dd={row['worst_max_drawdown_pct']:.2f}% neg={row['negative_cells']} "
            f"min_sells={row['min_num_sells']} {row['candidate_id']}",
            flush=True,
        )
    print(f"\nwrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
