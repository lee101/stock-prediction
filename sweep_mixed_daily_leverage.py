#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from trade_mixed_daily import DEFAULT_CHECKPOINT, run_backtest


def _parse_float_csv(raw: str) -> list[float]:
    values: list[float] = []
    for token in str(raw).split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        values.append(float(cleaned))
    if not values:
        raise ValueError("at least one leverage value is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep mixed daily leverage settings and rank by Sortino.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--leverage-values", default="1,2,3,4")
    parser.add_argument("--output-json", default="strategy_state/mixed_daily_leverage_sweep.json")
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    for leverage in _parse_float_csv(args.leverage_values):
        run_args = SimpleNamespace(
            checkpoint=args.checkpoint,
            symbols=args.symbols,
            data_root=args.data_root,
            start=args.start,
            end=args.end,
            max_steps=args.max_steps,
            min_days=args.min_days,
            fee_rate=args.fee_rate,
            max_leverage=float(leverage),
            periods_per_year=args.periods_per_year,
            short_borrow_apr=args.short_borrow_apr,
            cash=args.cash,
            current_symbol=None,
            current_direction=None,
            position_qty=0.0,
            entry_price=0.0,
            hold_days=0,
            output_json=None,
            cpu=bool(args.cpu),
            sample=bool(args.sample),
            backtest=True,
            once=False,
            audit=False,
        )
        report = run_backtest(run_args)
        enriched = dict(report)
        enriched["max_leverage"] = float(leverage)
        results.append(enriched)

    ranked = sorted(
        results,
        key=lambda row: (
            float(row.get("sortino", 0.0)),
            float(row.get("annualized_return", 0.0)),
            float(row.get("total_return", 0.0)),
        ),
        reverse=True,
    )
    payload = {
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "date_range": {"start": args.start, "end": args.end},
        "leverage_values": _parse_float_csv(args.leverage_values),
        "best_by_sortino": ranked[0] if ranked else None,
        "results": ranked,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
