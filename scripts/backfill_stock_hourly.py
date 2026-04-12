#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hourly_data_utils import HourlyDataIssue, HourlyDataStatus, HourlyDataValidator  # noqa: E402
from src.stock_symbol_inputs import load_symbols_file, normalize_symbols  # noqa: E402

try:  # pragma: no cover - optional during check-only mode outside full env
    from src.hourly_data_refresh import HourlyDataRefresher  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - lazy fallback handled in main
    HourlyDataRefresher = None  # type: ignore[assignment]


def _resolve_symbols(values: Sequence[str] | None, *, symbols_file: str | None) -> list[str]:
    raw: list[str] = []
    if symbols_file:
        raw.extend(load_symbols_file(symbols_file))
    for value in values or ():
        raw.extend(part.strip() for part in str(value).split(",") if part.strip())
    normalized, _duplicates, _ignored = normalize_symbols(raw)
    return normalized


def _status_payload(statuses: Sequence[HourlyDataStatus]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for status in statuses:
        rows.append(
            {
                "symbol": status.symbol,
                "path": str(status.path),
                "last_timestamp": status.latest_timestamp.isoformat(),
                "latest_close": float(status.latest_close),
                "staleness_hours": float(status.staleness_hours),
            }
        )
    return rows


def _issue_payload(issues: Sequence[HourlyDataIssue]) -> list[dict[str, str]]:
    return [
        {
            "symbol": issue.symbol,
            "reason": issue.reason,
            "detail": issue.detail,
        }
        for issue in issues
    ]


def build_summary(
    *,
    symbols: Sequence[str],
    before_statuses: Sequence[HourlyDataStatus],
    before_issues: Sequence[HourlyDataIssue],
    after_statuses: Sequence[HourlyDataStatus],
    after_issues: Sequence[HourlyDataIssue],
) -> dict[str, object]:
    before_ready = {status.symbol for status in before_statuses}
    after_ready = {status.symbol for status in after_statuses}
    return {
        "symbols": list(symbols),
        "symbol_count": len(symbols),
        "before": {
            "ready_count": len(before_statuses),
            "issue_count": len(before_issues),
            "ready_symbols": sorted(before_ready),
            "issue_symbols": sorted({issue.symbol for issue in before_issues}),
            "statuses": _status_payload(before_statuses),
            "issues": _issue_payload(before_issues),
        },
        "after": {
            "ready_count": len(after_statuses),
            "issue_count": len(after_issues),
            "ready_symbols": sorted(after_ready),
            "issue_symbols": sorted({issue.symbol for issue in after_issues}),
            "statuses": _status_payload(after_statuses),
            "issues": _issue_payload(after_issues),
        },
        "newly_ready_symbols": sorted(after_ready - before_ready),
        "still_missing_symbols": sorted({issue.symbol for issue in after_issues if issue.reason == "missing"}),
        "still_stale_symbols": sorted({issue.symbol for issue in after_issues if issue.reason == "stale"}),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill hourly stock data for a symbol set.")
    parser.add_argument("--symbols", action="append", default=[], help="Comma-separated symbols.")
    parser.add_argument("--symbols-file", default=None, help="Optional newline-delimited symbol file.")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly"))
    parser.add_argument("--backfill-hours", type=int, default=24 * 365 * 2)
    parser.add_argument("--overlap-hours", type=int, default=2)
    parser.add_argument("--max-staleness-hours", type=float, default=24.0 * 365.0 * 5.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = _resolve_symbols(args.symbols, symbols_file=args.symbols_file)
    validator = HourlyDataValidator(Path(args.data_root), max_staleness_hours=float(args.max_staleness_hours))
    before_statuses, before_issues = validator.filter_ready(symbols)

    if args.check_only:
        after_statuses, after_issues = before_statuses, before_issues
    else:
        refresher_cls = HourlyDataRefresher
        if refresher_cls is None:
            from src.hourly_data_refresh import HourlyDataRefresher as refresher_cls  # noqa: PLC0415

        refresher = refresher_cls(
            Path(args.data_root),
            validator,
            backfill_hours=int(args.backfill_hours),
            overlap_hours=int(args.overlap_hours),
            sleep_seconds=float(args.sleep_seconds),
        )
        after_statuses, after_issues = refresher.refresh(symbols)

    summary = build_summary(
        symbols=symbols,
        before_statuses=before_statuses,
        before_issues=before_issues,
        after_statuses=after_statuses,
        after_issues=after_issues,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        "Hourly coverage "
        f"{summary['before']['ready_count']}/{summary['symbol_count']} -> "
        f"{summary['after']['ready_count']}/{summary['symbol_count']}"
    )
    if summary["newly_ready_symbols"]:
        print("Newly ready: " + ", ".join(summary["newly_ready_symbols"]))
    if summary["still_missing_symbols"]:
        print("Still missing: " + ", ".join(summary["still_missing_symbols"]))
    if summary["still_stale_symbols"]:
        print("Still stale: " + ", ".join(summary["still_stale_symbols"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
