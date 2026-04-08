#!/usr/bin/env python3
"""Run a production-equivalent daily stock variant sweep."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import trade_daily_stock_prod as daily_stock


VariantSpec = daily_stock.BacktestVariantSpec


@dataclass(frozen=True)
class VariantPreset:
    name: str
    description: str
    variants: tuple[VariantSpec, ...]


CURRENT_LIVE_VARIANT = VariantSpec(name="current_live_12p5", allocation_pct=12.5)
SINGLE_STATIC_25_VARIANT = VariantSpec(name="single_static_25", allocation_pct=25.0)
PORTFOLIO2_STATIC_50_VARIANT = VariantSpec(name="portfolio2_static_50", allocation_pct=50.0, multi_position=2)
PORTFOLIO3_STATIC_50_VARIANT = VariantSpec(name="portfolio3_static_50", allocation_pct=50.0, multi_position=3)


PRESET_VARIANTS: dict[str, VariantPreset] = {
    "current_vs_candidates": VariantPreset(
        name="current_vs_candidates",
        description="Current live-equivalent config plus the strongest server-aware static candidates.",
        variants=(
            CURRENT_LIVE_VARIANT,
            SINGLE_STATIC_25_VARIANT,
            PORTFOLIO2_STATIC_50_VARIANT,
            PORTFOLIO3_STATIC_50_VARIANT,
        ),
    ),
    "current_only": VariantPreset(
        name="current_only",
        description="Only the current live-equivalent single-position configuration.",
        variants=(CURRENT_LIVE_VARIANT,),
    ),
    "promising_only": VariantPreset(
        name="promising_only",
        description="Only the short-window variants that beat the current live-equivalent baseline.",
        variants=(
            SINGLE_STATIC_25_VARIANT,
            PORTFOLIO2_STATIC_50_VARIANT,
            PORTFOLIO3_STATIC_50_VARIANT,
        ),
    ),
}


def _preset_choices() -> list[str]:
    return sorted(PRESET_VARIANTS)


def _resolve_preset(name: str) -> VariantPreset:
    return PRESET_VARIANTS[str(name).strip()]


def _normalize_symbols(values: list[str] | None) -> list[str]:
    if not values:
        return list(daily_stock.DEFAULT_SYMBOLS)
    symbols: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for item in str(raw).split(","):
            symbol = item.strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def _variant_payload(spec: VariantSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "allocation_pct": float(spec.allocation_pct),
        "allocation_sizing_mode": spec.allocation_sizing_mode,
        "multi_position": int(spec.multi_position),
        "multi_position_min_prob_ratio": float(spec.multi_position_min_prob_ratio),
        "buying_power_multiplier": float(spec.buying_power_multiplier),
    }


def _monthly_return(total_return: float, *, days: int) -> float:
    return float((1.0 + float(total_return)) ** (21.0 / float(days)) - 1.0)


def _table_for_results(rows: list[dict[str, object]]) -> str:
    headers = [
        "name",
        "alloc",
        "multi",
        "total_return",
        "monthly_return",
        "sortino",
        "max_drawdown",
        "trades",
    ]
    printable: list[dict[str, str]] = []
    for row in rows:
        printable.append(
            {
                "name": str(row["name"]),
                "alloc": f"{float(row['allocation_pct']):g}",
                "multi": str(int(row["multi_position"])),
                "total_return": f"{float(row['total_return']):+.4%}",
                "monthly_return": f"{float(row['monthly_return']):+.4%}",
                "sortino": f"{float(row['sortino']):+.3f}",
                "max_drawdown": f"{float(row['max_drawdown']):+.4%}",
                "trades": f"{float(row['trades']):g}",
            }
        )
    widths = {
        header: max(len(header), *(len(item[header]) for item in printable)) if printable else len(header)
        for header in headers
    }
    lines = [
        " ".join(header.ljust(widths[header]) for header in headers),
        " ".join("-" * widths[header] for header in headers),
    ]
    for item in printable:
        lines.append(" ".join(item[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a server-aware daily stock variant sweep.")
    parser.add_argument(
        "--preset",
        choices=_preset_choices(),
        default="current_vs_candidates",
        help="Named variant set to evaluate.",
    )
    parser.add_argument("--days", type=int, default=120, help="Backtest trading days.")
    parser.add_argument("--checkpoint", default=daily_stock.DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=daily_stock.DEFAULT_DATA_DIR)
    parser.add_argument("--symbols", action="append", default=None, help="Optional comma-separated symbol override.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without running the sweep.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preset = _resolve_preset(args.preset)
    variants = list(preset.variants)
    symbols = _normalize_symbols(args.symbols)
    payload = {
        "preset": preset.name,
        "preset_description": preset.description,
        "days": int(args.days),
        "checkpoint": str(Path(args.checkpoint)),
        "data_dir": str(args.data_dir),
        "symbols": symbols,
        "variants": [_variant_payload(item) for item in variants],
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    results = daily_stock.run_backtest_variant_matrix_via_trading_server(
        checkpoint=args.checkpoint,
        symbols=symbols,
        data_dir=args.data_dir,
        days=args.days,
        variants=variants,
        extra_checkpoints=list(daily_stock.DEFAULT_EXTRA_CHECKPOINTS),
    )
    ranked: list[dict[str, object]] = []
    for row in results:
        enriched = dict(row)
        enriched["monthly_return"] = _monthly_return(float(row["total_return"]), days=int(args.days))
        ranked.append(enriched)
    ranked.sort(key=lambda item: float(item["monthly_return"]), reverse=True)

    if args.json:
        print(json.dumps({"config": payload, "results": ranked}, indent=2, sort_keys=True))
    else:
        print(f"Daily stock variant sweep: preset={preset.name} days={args.days} symbols={len(symbols)}")
        print(preset.description)
        print(_table_for_results(ranked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
