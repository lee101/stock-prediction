#!/usr/bin/env python3
"""Epoch-level pair search for DOGE/AAVE meta selector configs."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from binanceleveragesui.sim_meta_switcher import compute_stats
from binanceleveragesui.sweep_meta_daily_winners import (
    Candidate,
    align_equity_curves,
    build_equity_series,
    compose_meta_equity_mode,
    parse_csv_list,
    parse_int_list,
    slice_window,
)


@dataclass(frozen=True)
class EpochSource:
    prefix: str
    symbol: str
    directory: Path


def parse_source_spec(raw: str) -> EpochSource:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid source spec '{raw}'. Expected PREFIX:SYMBOL:DIR")
    prefix, symbol, dir_raw = parts[0].strip(), parts[1].strip().upper(), parts[2].strip()
    if not prefix or not symbol or not dir_raw:
        raise ValueError(f"Invalid source spec '{raw}'. Empty prefix/symbol/dir.")
    directory = Path(dir_raw).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Source dir does not exist: {directory}")
    return EpochSource(prefix=prefix, symbol=symbol, directory=directory)


def build_candidates(sources: list[EpochSource], epochs: list[int]) -> list[Candidate]:
    out: list[Candidate] = []
    for src in sources:
        for epoch in epochs:
            ckpt = src.directory / f"epoch_{int(epoch):03d}.pt"
            if not ckpt.exists():
                continue
            out.append(
                Candidate(
                    name=f"{src.prefix}_ep{int(epoch):03d}",
                    symbol=src.symbol,
                    checkpoint=ckpt,
                )
            )
    return out


def rank_key(row: dict) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["min_sortino"]),
        float(row["beats"]),
        float(row["mean_sortino"]),
        float(row["min_return_pct"]),
        float(row["mean_return_pct"]),
        -float(row["mean_dd_pct"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Epoch-level DOGE/AAVE pair search for meta selector.")
    parser.add_argument(
        "--doge-source",
        action="append",
        required=True,
        help="PREFIX:SYMBOL:DIR (repeatable). Example: doge_drop15:DOGEUSD:/path/to/run",
    )
    parser.add_argument(
        "--aave-source",
        action="append",
        required=True,
        help="PREFIX:SYMBOL:DIR (repeatable). Example: aave_strides:AAVEUSD:/path/to/run",
    )
    parser.add_argument("--epochs", default="1,2,3,4,5,6,7,8,10,12,15,20")
    parser.add_argument("--metrics", default="calmar,sortino")
    parser.add_argument("--lookbacks", default="1,2")
    parser.add_argument("--windows", default="30,60,90,120")
    parser.add_argument("--mode", default="winner_cash")
    parser.add_argument("--cash-threshold", type=float, default=0.0)
    parser.add_argument("--switch-margin", type=float, default=0.005)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = [m.lower() for m in parse_csv_list(args.metrics)]
    lookbacks = parse_int_list(args.lookbacks)
    windows = parse_int_list(args.windows)
    epochs = parse_int_list(args.epochs)
    if not windows or not lookbacks or not epochs:
        raise ValueError("epochs/lookbacks/windows cannot be empty")

    doge_sources = [parse_source_spec(x) for x in args.doge_source]
    aave_sources = [parse_source_spec(x) for x in args.aave_source]
    doge_candidates = build_candidates(doge_sources, epochs)
    aave_candidates = build_candidates(aave_sources, epochs)
    if not doge_candidates or not aave_candidates:
        raise ValueError(
            f"Need non-empty candidate sets. doge={len(doge_candidates)} aave={len(aave_candidates)}"
        )

    all_candidates = doge_candidates + aave_candidates
    max_window = max(windows)
    curves = {}
    logger.info(
        "Loading candidate curves: doge={} aave={} total={} ...",
        len(doge_candidates),
        len(aave_candidates),
        len(all_candidates),
    )
    for c in all_candidates:
        logger.info("  loading {}", c.name)
        curves[c.name] = build_equity_series(
            c,
            val_days=args.val_days,
            test_days=max_window,
            maker_fee=args.maker_fee,
            max_leverage=args.max_leverage,
        )
    curves = align_equity_curves(curves)

    rows = []
    for d, a in itertools.product(doge_candidates, aave_candidates):
        for metric, lb in itertools.product(metrics, lookbacks):
            window_rows = []
            for days in windows:
                sub = slice_window({d.name: curves[d.name], a.name: curves[a.name]}, days)
                eq, winners, switches = compose_meta_equity_mode(
                    sub,
                    lookback_days=lb,
                    metric=metric,
                    fallback_strategy=d.name,
                    tie_break_order=[d.name, a.name],
                    mode=args.mode,
                    cash_threshold=float(args.cash_threshold),
                    switch_margin=float(args.switch_margin),
                    softmax_temperature=1.0,
                    min_score_gap=float(args.min_score_gap),
                )
                meta = compute_stats(eq.values, "meta")
                b1 = compute_stats(sub[d.name].values, d.name)
                b2 = compute_stats(sub[a.name].values, a.name)
                best = max((b1, b2), key=lambda x: (x["sortino"], x["total_return"], -x["max_dd"]))
                window_rows.append(
                    {
                        "days": int(days),
                        "meta": meta,
                        "best": best,
                        "switches": int(switches),
                        "winner_days": int(len(winners)),
                    }
                )

            s = [w["meta"]["sortino"] for w in window_rows]
            r = [w["meta"]["total_return"] for w in window_rows]
            dd = [w["meta"]["max_dd"] for w in window_rows]
            beats = sum(1 for w in window_rows if w["meta"]["sortino"] > w["best"]["sortino"])
            rows.append(
                {
                    "doge": d.name,
                    "aave": a.name,
                    "mode": args.mode,
                    "metric": metric,
                    "lookback_days": int(lb),
                    "cash_threshold": float(args.cash_threshold),
                    "switch_margin": float(args.switch_margin),
                    "min_score_gap": float(args.min_score_gap),
                    "min_sortino": float(np.min(s)),
                    "mean_sortino": float(np.mean(s)),
                    "min_return_pct": float(np.min(r)),
                    "mean_return_pct": float(np.mean(r)),
                    "mean_dd_pct": float(np.mean(dd)),
                    "beats": int(beats),
                    "windows": window_rows,
                    "doge_checkpoint": str(d.checkpoint),
                    "aave_checkpoint": str(a.checkpoint),
                }
            )

    rows.sort(key=rank_key, reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    logger.info("Saved {}", args.output)
    logger.info("Top {}:", min(args.top_k, len(rows)))
    for i, row in enumerate(rows[: args.top_k], 1):
        logger.info(
            "{:>2}. {} + {} | {} lb={} | minS={:.2f} meanS={:.2f} minR={:+.2f}% meanR={:+.2f}% meanDD={:.2f}% beats={}",
            i,
            row["doge"],
            row["aave"],
            row["metric"],
            row["lookback_days"],
            row["min_sortino"],
            row["mean_sortino"],
            row["min_return_pct"],
            row["mean_return_pct"],
            row["mean_dd_pct"],
            row["beats"],
        )


if __name__ == "__main__":
    main()
