#!/usr/bin/env python3
"""Run a production-equivalent daily stock variant sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import trade_daily_stock_prod as daily_stock
from src.daily_stock_variant_presets import preset_choices, resolve_variant_preset


VariantSpec = daily_stock.BacktestVariantSpec


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


def _emit_json_payload(payload: dict[str, object], *, output_json: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    _write_json_payload(rendered, output_json=output_json)
    print(rendered)


def _write_json_payload(rendered: str, *, output_json: str | None) -> None:
    if output_json:
        output_path = Path(output_json)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"{rendered}\n", encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Failed to write JSON report to {output_path}: {exc}") from exc


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


def _table_for_multi_window_summary(rows: list[dict[str, object]]) -> str:
    headers = [
        "name",
        "alloc",
        "multi",
        "avg_monthly",
        "min_monthly",
        "avg_sortino",
        "worst_drawdown",
        "windows",
    ]
    printable: list[dict[str, str]] = []
    for row in rows:
        printable.append(
            {
                "name": str(row["name"]),
                "alloc": f"{float(row['allocation_pct']):g}",
                "multi": str(int(row["multi_position"])),
                "avg_monthly": f"{float(row['avg_monthly_return']):+.4%}",
                "min_monthly": f"{float(row['min_monthly_return']):+.4%}",
                "avg_sortino": f"{float(row['avg_sortino']):+.3f}",
                "worst_drawdown": f"{float(row['worst_max_drawdown']):+.4%}",
                "windows": str(int(row["window_count"])),
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


def _resolve_days(days: int, windows: list[int] | None) -> list[int]:
    if windows:
        resolved = [int(item) for item in windows]
    else:
        resolved = [int(days)]
    return list(dict.fromkeys(item for item in resolved if int(item) > 0))


def _summarize_multi_window_results(
    windows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not windows:
        return []
    grouped: dict[str, list[dict[str, object]]] = {}
    for window in windows:
        for row in window["results"]:
            grouped.setdefault(str(row["name"]), []).append(row)
    summary: list[dict[str, object]] = []
    for name, rows in grouped.items():
        exemplar = rows[0]
        monthly_returns = [float(row["monthly_return"]) for row in rows]
        sortinos = [float(row["sortino"]) for row in rows]
        max_drawdowns = [float(row["max_drawdown"]) for row in rows]
        summary.append(
            {
                "name": name,
                "allocation_pct": float(exemplar["allocation_pct"]),
                "allocation_sizing_mode": exemplar["allocation_sizing_mode"],
                "multi_position": int(exemplar["multi_position"]),
                "multi_position_min_prob_ratio": float(exemplar["multi_position_min_prob_ratio"]),
                "buying_power_multiplier": float(exemplar["buying_power_multiplier"]),
                "avg_monthly_return": sum(monthly_returns) / len(monthly_returns),
                "min_monthly_return": min(monthly_returns),
                "avg_sortino": sum(sortinos) / len(sortinos),
                "worst_max_drawdown": min(max_drawdowns),
                "window_count": len(rows),
            }
        )
    summary.sort(
        key=lambda item: (float(item["min_monthly_return"]), float(item["avg_monthly_return"])),
        reverse=True,
    )
    return summary


def _single_window_baseline_comparison(
    rows: list[dict[str, object]],
    *,
    baseline_name: str = "current_live_12p5",
) -> dict[str, object] | None:
    baseline = next((row for row in rows if str(row["name"]) == baseline_name), None)
    if baseline is None:
        return None
    baseline_metric = float(baseline["monthly_return"])
    beaters = [
        {
            "name": str(row["name"]),
            "candidate_monthly_return": float(row["monthly_return"]),
            "delta_monthly_return": float(row["monthly_return"]) - baseline_metric,
        }
        for row in rows
        if str(row["name"]) != baseline_name and float(row["monthly_return"]) > baseline_metric
    ]
    beaters.sort(key=lambda item: float(item["delta_monthly_return"]), reverse=True)
    return {
        "baseline_name": baseline_name,
        "baseline_monthly_return": baseline_metric,
        "beaters": beaters,
    }


def _multi_window_baseline_comparison(
    rows: list[dict[str, object]],
    *,
    baseline_name: str = "current_live_12p5",
) -> dict[str, object] | None:
    baseline = next((row for row in rows if str(row["name"]) == baseline_name), None)
    if baseline is None:
        return None
    baseline_avg = float(baseline["avg_monthly_return"])
    baseline_min = float(baseline["min_monthly_return"])
    beaters = [
        {
            "name": str(row["name"]),
            "candidate_avg_monthly_return": float(row["avg_monthly_return"]),
            "candidate_min_monthly_return": float(row["min_monthly_return"]),
            "delta_avg_monthly_return": float(row["avg_monthly_return"]) - baseline_avg,
            "delta_min_monthly_return": float(row["min_monthly_return"]) - baseline_min,
        }
        for row in rows
        if str(row["name"]) != baseline_name
        and float(row["avg_monthly_return"]) > baseline_avg
    ]
    beaters.sort(key=lambda item: float(item["delta_avg_monthly_return"]), reverse=True)
    return {
        "baseline_name": baseline_name,
        "baseline_avg_monthly_return": baseline_avg,
        "baseline_min_monthly_return": baseline_min,
        "beaters": beaters,
    }


def _format_single_window_baseline_comparison(comparison: dict[str, object] | None) -> str | None:
    if not comparison:
        return None
    baseline_name = str(comparison["baseline_name"])
    baseline_monthly = float(comparison["baseline_monthly_return"])
    beaters = list(comparison["beaters"])
    if not beaters:
        return f"Baseline {baseline_name}: {baseline_monthly:+.4%} monthly. No tested variant beat it."
    lead = ", ".join(
        f"{item['name']} ({float(item['delta_monthly_return']):+.4%})"
        for item in beaters
    )
    return f"Baseline {baseline_name}: {baseline_monthly:+.4%} monthly. Beat baseline: {lead}"


def _format_multi_window_baseline_comparison(comparison: dict[str, object] | None) -> str | None:
    if not comparison:
        return None
    baseline_name = str(comparison["baseline_name"])
    baseline_avg = float(comparison["baseline_avg_monthly_return"])
    baseline_min = float(comparison["baseline_min_monthly_return"])
    beaters = list(comparison["beaters"])
    if not beaters:
        return (
            f"Baseline {baseline_name}: avg {baseline_avg:+.4%}, worst {baseline_min:+.4%}. "
            "No tested variant beat its average monthly return."
        )
    lead = ", ".join(
        f"{item['name']} (avg {float(item['delta_avg_monthly_return']):+.4%}, "
        f"worst {float(item['delta_min_monthly_return']):+.4%})"
        for item in beaters
    )
    return (
        f"Baseline {baseline_name}: avg {baseline_avg:+.4%}, worst {baseline_min:+.4%}. "
        f"Beat baseline: {lead}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a server-aware daily stock variant sweep.")
    parser.add_argument(
        "--preset",
        choices=preset_choices(),
        default="current_vs_candidates",
        help="Named variant set to evaluate.",
    )
    parser.add_argument("--days", type=int, default=120, help="Backtest trading days.")
    parser.add_argument(
        "--window",
        action="append",
        type=int,
        default=None,
        help="Optional repeatable backtest window in trading days. Overrides single-window ranking with a multi-window summary.",
    )
    parser.add_argument("--checkpoint", default=daily_stock.DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=daily_stock.DEFAULT_DATA_DIR)
    parser.add_argument("--symbols", action="append", default=None, help="Optional comma-separated symbol override.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.add_argument("--output-json", default=None, help="Optional path to write the JSON report.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without running the sweep.")
    args = parser.parse_args(argv)
    if not _resolve_days(args.days, args.window):
        parser.error("at least one positive backtest window is required")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        preset = resolve_variant_preset(args.preset)
        variants = list(preset.variants)
        symbols = _normalize_symbols(args.symbols)
        days_list = _resolve_days(args.days, args.window)
        payload = {
            "preset": preset.name,
            "preset_description": preset.description,
            "days": int(args.days),
            "days_list": days_list,
            "checkpoint": str(Path(args.checkpoint)),
            "data_dir": str(args.data_dir),
            "symbols": symbols,
            "variants": [_variant_payload(item) for item in variants],
        }
        if args.dry_run:
            _emit_json_payload(payload, output_json=args.output_json)
            return 0

        window_results: list[dict[str, object]] = []
        for days in days_list:
            try:
                results = daily_stock.run_backtest_variant_matrix_via_trading_server(
                    checkpoint=args.checkpoint,
                    symbols=symbols,
                    data_dir=args.data_dir,
                    days=days,
                    variants=variants,
                    extra_checkpoints=list(daily_stock.DEFAULT_EXTRA_CHECKPOINTS),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Backtest sweep failed for preset {preset.name} at {int(days)} trading days: {exc}"
                ) from exc
            ranked: list[dict[str, object]] = []
            for row in results:
                enriched = dict(row)
                enriched["monthly_return"] = _monthly_return(float(row["total_return"]), days=int(days))
                ranked.append(enriched)
            ranked.sort(key=lambda item: float(item["monthly_return"]), reverse=True)
            window_results.append({"days": int(days), "results": ranked})

        if len(window_results) == 1:
            ranked = list(window_results[0]["results"])
            comparison = _single_window_baseline_comparison(ranked)
            report_payload = {"config": payload, "results": ranked, "baseline_comparison": comparison}
            if args.json:
                _emit_json_payload(report_payload, output_json=args.output_json)
            else:
                if args.output_json:
                    _write_json_payload(
                        json.dumps(report_payload, indent=2, sort_keys=True),
                        output_json=args.output_json,
                    )
                print(f"Daily stock variant sweep: preset={preset.name} days={days_list[0]} symbols={len(symbols)}")
                print(preset.description)
                print(_table_for_results(ranked))
                message = _format_single_window_baseline_comparison(comparison)
                if message:
                    print(message)
            return 0

        summary = _summarize_multi_window_results(window_results)
        comparison = _multi_window_baseline_comparison(summary)
        report_payload = {
            "config": payload,
            "windows": window_results,
            "summary": summary,
            "baseline_comparison": comparison,
        }

        if args.json:
            _emit_json_payload(report_payload, output_json=args.output_json)
        else:
            if args.output_json:
                _write_json_payload(
                    json.dumps(report_payload, indent=2, sort_keys=True),
                    output_json=args.output_json,
                )
            print(
                "Daily stock variant sweep: "
                f"preset={preset.name} windows={','.join(str(item) for item in days_list)} symbols={len(symbols)}"
            )
            print(preset.description)
            print(_table_for_multi_window_summary(summary))
            message = _format_multi_window_baseline_comparison(comparison)
            if message:
                print(message)
        return 0
    except RuntimeError as exc:
        print(f"daily stock variant sweep failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
