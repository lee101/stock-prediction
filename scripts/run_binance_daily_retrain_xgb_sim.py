#!/usr/bin/env python3
"""Replay Binance hourly XGB packs with daily walk-forward retraining.

This is slower than the normal pack sweep because it deliberately mirrors the
"train today, trade tomorrow, retrain" idea: for each evaluation day, fit fresh
XGB high/low/close forecasters using only targets that would have been known at
that day's start, score that day's hourly rows, then concatenate those scored
rows and run the normal binary-fill portfolio simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_binance_pack_rolling_validation import load_candidate_configs  # noqa: E402
from scripts.sweep_binance_hourly_portfolio_pack import (  # noqa: E402
    PackConfig,
    _choose_end_timestamp,
    _discover_symbols,
    _filter_liquid_frames,
    _load_hourly_frames,
    _monthly_return,
    _to_utc,
    build_model_frame,
    evaluate_pack,
    fit_forecasters,
    score_eval_rows,
)


STABLE_BASE_ASSETS = {
    "BUSD",
    "DAI",
    "FDUSD",
    "TUSD",
    "USDC",
    "USDP",
    "USDS",
    "USDT",
}


def controlled_clamp_config() -> PackConfig:
    """Best controlled clamp from the 2026-05-06 refreshed short sweep."""
    return PackConfig(
        risk_penalty=0.35,
        cvar_weight=0.5,
        entry_gap_bps=30.0,
        entry_alpha=0.2,
        exit_alpha=0.85,
        edge_threshold=0.0025,
        edge_to_full_size=0.03,
        min_close_ret=0.0005,
        close_edge_weight=0.35,
        min_upside_downside_ratio=0.0,
        min_recent_ret_24h=-1.0,
        min_recent_ret_72h=-1.0,
        max_recent_vol_72h=0.03,
        regime_cs_skew_min=-0.35,
        vol_target_ann=0.6,
        inv_vol_target_ann=2.0,
        inv_vol_floor=0.15,
        inv_vol_cap=1.5,
        max_positions=2,
        max_pending_entries=4,
        entry_ttl_hours=1,
        max_hold_hours=16,
        max_leverage=2.0,
        entry_selection_mode="first_trigger",
        entry_allocator_mode="concentrated",
        entry_allocator_edge_power=1.2,
    )


def daily_eval_windows(eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(eval_start).tz_convert("UTC")
    end = pd.Timestamp(eval_end).tz_convert("UTC")
    if end < start:
        raise ValueError(f"eval_end {end} is before eval_start {start}")
    current = start.floor("D")
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while current <= end:
        window_start = max(start, current)
        # Hourly Binance bars are timestamped on the hour; use an inclusive end.
        window_end = min(end, current + pd.Timedelta(days=1) - pd.Timedelta(hours=1))
        if window_start <= window_end:
            windows.append((window_start, window_end))
        current += pd.Timedelta(days=1)
    return windows


def _filter_live_tradable_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        raw = str(symbol).upper().strip()
        base = raw[:-4] if raw.endswith("USDT") else raw
        if base and base not in STABLE_BASE_ASSETS:
            out[raw] = frame
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _eval_args(args: argparse.Namespace, *, fill_buffer_bps: float) -> argparse.Namespace:
    return argparse.Namespace(
        min_take_profit_bps=float(args.min_take_profit_bps),
        max_entry_gap_bps=float(args.max_entry_gap_bps),
        max_exit_gap_bps=float(args.max_exit_gap_bps),
        fee_rate=float(args.fee_rate),
        top_candidates_per_hour=int(args.top_candidates_per_hour),
        top_candidates_include_inactive=bool(args.top_candidates_include_inactive),
        initial_cash=float(args.initial_cash),
        entry_intensity_power=float(args.entry_intensity_power),
        entry_min_intensity_fraction=float(args.entry_min_intensity_fraction),
        force_close_slippage_bps=float(args.force_close_slippage_bps),
        margin_annual_rate=float(args.margin_annual_rate),
        decision_lag=int(args.decision_lag),
        entry_allocator_max_single_position_fraction=float(args.entry_allocator_max_single_position_fraction),
        entry_allocator_reserve_fraction=float(args.entry_allocator_reserve_fraction),
        entry_allocator_min_active_positions=int(args.entry_allocator_min_active_positions),
        entry_allocator_min_second_position_fraction=float(args.entry_allocator_min_second_position_fraction),
        entry_drawdown_pause_threshold=float(args.entry_drawdown_pause_threshold),
        entry_drawdown_resume_threshold=float(args.entry_drawdown_resume_threshold),
        entry_drawdown_cooldown_hours=int(args.entry_drawdown_cooldown_hours),
        entry_drawdown_scale_start=float(args.entry_drawdown_scale_start),
        entry_drawdown_scale_full=float(args.entry_drawdown_scale_full),
        entry_drawdown_scale_floor=float(args.entry_drawdown_scale_floor),
        entry_diversify_buckets=int(args.entry_diversify_buckets),
        entry_corr_window_bars=int(args.entry_corr_window_bars),
        entry_corr_min_periods=int(args.entry_corr_min_periods),
        entry_corr_max_signed=float(args.entry_corr_max_signed),
        fill_buffer_bps=float(fill_buffer_bps),
        side_mode=str(args.side_mode),
        entry_block_hours_utc=str(args.entry_block_hours_utc),
        force_exit_hours_utc=str(args.force_exit_hours_utc),
        min_result_trades=int(args.min_result_trades),
        disable_drawdown_profit_early_exit=bool(args.disable_drawdown_profit_early_exit),
    )


def _load_candidates(args: argparse.Namespace) -> list[tuple[str, PackConfig]]:
    if args.candidate_csv:
        loaded = load_candidate_configs(
            [Path(path) for path in args.candidate_csv],
            top_k_per_file=int(args.top_k_per_file),
            rank_by=str(args.rank_by),
            min_source_sells=int(args.min_source_sells),
        )
        return [(candidate.candidate_id, candidate.cfg) for candidate in loaded]
    return [("controlled_clamp_20260506", controlled_clamp_config())]


def build_daily_retrain_scores(
    frames: dict[str, pd.DataFrame],
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    train_days: int,
    label_horizon: int,
    rounds: int,
    device: str,
) -> pd.DataFrame:
    feature_start = eval_start - pd.Timedelta(days=int(train_days) + 10)
    model_frame, feature_cols = build_model_frame(
        frames,
        start=feature_start,
        end=eval_end,
        horizon=int(label_horizon),
        require_targets=False,
    )
    scored_parts: list[pd.DataFrame] = []
    for idx, (day_start, day_end) in enumerate(daily_eval_windows(eval_start, eval_end), start=1):
        train_end = day_start - pd.Timedelta(hours=int(label_horizon))
        train_start = train_end - pd.Timedelta(days=int(train_days))
        frame = model_frame[(model_frame["timestamp"] >= train_start) & (model_frame["timestamp"] <= day_end)].copy()
        if frame.empty:
            raise ValueError(f"empty model frame for daily retrain window ending {day_end}")
        models = fit_forecasters(
            frame,
            feature_cols,
            train_end=train_end,
            rounds=int(rounds),
            device=str(device),
        )
        scored = score_eval_rows(frame, feature_cols, models, eval_start=day_start, eval_end=day_end)
        scored_parts.append(scored)
        print(
            f"daily retrain {idx}/{len(daily_eval_windows(eval_start, eval_end))} "
            f"train=[{train_start}, {train_end}) eval=[{day_start}, {day_end}] rows={len(scored):,}",
            flush=True,
        )
    return pd.concat(scored_parts, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily walk-forward retrain simulation for Binance hourly XGB packs.")
    parser.add_argument("--candidate-csv", nargs="*", default=[])
    parser.add_argument("--top-k-per-file", type=int, default=4)
    parser.add_argument("--rank-by", default="selection_score")
    parser.add_argument("--min-source-sells", type=int, default=0)

    parser.add_argument("--hourly-root", type=Path, default=Path("binance_spot_hourly"))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--min-bars", type=int, default=5000)
    parser.add_argument("--min-symbols-per-hour", type=int, default=20)
    parser.add_argument("--liquidity-lookback-days", type=int, default=90)
    parser.add_argument("--min-median-dollar-volume", type=float, default=0.0)
    parser.add_argument("--max-symbols-by-dollar-volume", type=int, default=36)
    parser.add_argument("--train-days", type=int, default=720)
    parser.add_argument("--eval-days", type=int, default=14)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--label-horizon", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fill-buffer-bps-grid", default="5")

    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--force-close-slippage-bps", type=float, default=10.0)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0625)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--min-take-profit-bps", type=float, default=35.0)
    parser.add_argument("--max-entry-gap-bps", type=float, default=120.0)
    parser.add_argument("--max-exit-gap-bps", type=float, default=250.0)
    parser.add_argument("--top-candidates-per-hour", type=int, default=10)
    parser.add_argument("--top-candidates-include-inactive", action="store_true")
    parser.add_argument("--entry-intensity-power", type=float, default=1.0)
    parser.add_argument("--entry-min-intensity-fraction", type=float, default=0.0)
    parser.add_argument("--entry-allocator-max-single-position-fraction", type=float, default=0.8)
    parser.add_argument("--entry-allocator-reserve-fraction", type=float, default=0.0)
    parser.add_argument("--entry-allocator-min-active-positions", type=int, default=2)
    parser.add_argument("--entry-allocator-min-second-position-fraction", type=float, default=0.2)
    parser.add_argument("--entry-drawdown-pause-threshold", type=float, default=0.0)
    parser.add_argument("--entry-drawdown-resume-threshold", type=float, default=0.0)
    parser.add_argument("--entry-drawdown-cooldown-hours", type=int, default=0)
    parser.add_argument("--entry-drawdown-scale-start", type=float, default=0.045)
    parser.add_argument("--entry-drawdown-scale-full", type=float, default=0.15)
    parser.add_argument("--entry-drawdown-scale-floor", type=float, default=0.32)
    parser.add_argument("--entry-diversify-buckets", type=int, default=0)
    parser.add_argument("--entry-corr-window-bars", type=int, default=168)
    parser.add_argument("--entry-corr-min-periods", type=int, default=48)
    parser.add_argument("--entry-corr-max-signed", type=float, default=0.8)
    parser.add_argument("--side-mode", choices=("long", "short", "both"), default="short")
    parser.add_argument("--entry-block-hours-utc", default="")
    parser.add_argument("--force-exit-hours-utc", default="")
    parser.add_argument("--min-result-trades", type=int, default=10)
    parser.set_defaults(disable_drawdown_profit_early_exit=True)
    parser.add_argument(
        "--allow-drawdown-profit-early-exit", dest="disable_drawdown_profit_early_exit", action="store_false"
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path, default=Path(f"analysis/binance_daily_retrain_xgb_sim_{stamp}.csv"))
    parser.add_argument("--scored-out", type=Path, default=None)
    parser.add_argument("--manifest-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = _discover_symbols(args.hourly_root, str(args.symbols))
    raw_frames = _load_hourly_frames(args.hourly_root, symbols=symbols, min_bars=int(args.min_bars))
    raw_frames = _filter_live_tradable_frames(raw_frames)
    end = (
        _to_utc(args.end_date)
        if str(args.end_date).strip()
        else _choose_end_timestamp(raw_frames, int(args.min_symbols_per_hour))
    )
    eval_start = end - pd.Timedelta(days=int(args.eval_days))
    frames, liquidity = _filter_liquid_frames(
        raw_frames,
        end=eval_start,
        lookback_days=int(args.liquidity_lookback_days),
        min_median_dollar_volume=float(args.min_median_dollar_volume),
        max_symbols=int(args.max_symbols_by_dollar_volume),
    )
    frames = _filter_live_tradable_frames(frames)
    if not frames:
        raise RuntimeError("live-tradable filter removed all symbols")
    print(
        f"daily-retrain universe={len(frames)} eval=[{eval_start}, {end}] "
        f"train_days={args.train_days} rounds={args.rounds} side={args.side_mode}",
        flush=True,
    )
    if not liquidity.empty:
        top = liquidity[liquidity["symbol"].isin(frames)].head(8)
        print(
            "liquidity top="
            + ",".join(f"{row.symbol}:{row.median_dollar_volume:.0f}" for row in top.itertuples(index=False)),
            flush=True,
        )

    scored = build_daily_retrain_scores(
        frames,
        eval_start=eval_start,
        eval_end=end,
        train_days=int(args.train_days),
        label_horizon=int(args.label_horizon),
        rounds=int(args.rounds),
        device=str(args.device),
    )
    if args.scored_out:
        args.scored_out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.scored_out, index=False)
        print(f"wrote {args.scored_out}")

    candidates = _load_candidates(args)
    fill_buffers = [float(part.strip()) for part in str(args.fill_buffer_bps_grid).split(",") if part.strip()]
    rows: list[dict[str, Any]] = []
    for candidate_id, cfg in candidates:
        for fill_buffer_bps in fill_buffers:
            sim_args = _eval_args(args, fill_buffer_bps=fill_buffer_bps)
            row, _bars, _actions, _result = evaluate_pack(
                scored,
                cfg=cfg,
                label_horizon=int(args.label_horizon),
                args=sim_args,
            )
            row.update(
                {
                    "candidate_id": candidate_id,
                    "fill_buffer_bps": fill_buffer_bps,
                    "daily_retrain": True,
                    "train_days": int(args.train_days),
                    "rounds": int(args.rounds),
                    "label_horizon": int(args.label_horizon),
                    "config_json": json.dumps(asdict(cfg), sort_keys=True),
                }
            )
            rows.append(row)
            _write_csv(args.out, rows)
            print(
                f"{candidate_id} fill={fill_buffer_bps:g} monthly={row['monthly_return_pct']:+.2f}% "
                f"ret={row['total_return_pct']:+.2f}% dd={row['max_drawdown_pct']:.2f}% "
                f"sortino={row['sortino']:.2f} trades={row['num_sells']}",
                flush=True,
            )
    best = max(rows, key=lambda item: float(item["selection_score"])) if rows else None
    manifest_path = args.manifest_out or args.out.with_name(args.out.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "args": vars(args),
                    "eval_start": eval_start,
                    "eval_end": end,
                    "symbols": sorted(frames),
                    "rows": len(rows),
                    "best": best,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {args.out}")
    print(f"wrote {manifest_path}")
    if best:
        total_return = float(best["total_return_pct"]) / 100.0
        monthly = 100.0 * _monthly_return(total_return, pd.Timestamp(best["start"]), pd.Timestamp(best["end"]))
        print(f"best={best['candidate_id']} monthly={monthly:+.2f}% dd={best['max_drawdown_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
