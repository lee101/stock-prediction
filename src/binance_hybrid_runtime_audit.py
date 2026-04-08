from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.binan.hybrid_cycle_trace import DEFAULT_TRACE_DIR, load_cycle_snapshots
from src.binance_hybrid_launch import DEFAULT_LAUNCH_SCRIPT, parse_launch_script
from src.binance_hybrid_prod_verify import evaluate_live_snapshot_health
from src.binance_hybrid_snapshot_activity import (
    detail_has_activity,
    iter_snapshot_order_symbols,
    normalize_symbols,
    safe_float,
    snapshot_allowed_market_symbols,
)


_DEGRADED_STATUSES = frozenset(
    {
        "failed",
        "context_error",
        "gemini_error_rl_flat",
        "invalid_allocation_plan",
        "invalid_allocation_plan_rl_flat",
        "no_active_tradable_symbols",
        "no_contexts",
        "rl_signal_error",
    }
)
_DEGRADED_ALLOCATION_SOURCES = frozenset(
    {
        "rl_flat_cash_flatten",
        "rl_flat_no_action",
        "rl_only_fallback",
    }
)
_OVERRIDE_ALLOCATION_SOURCES = frozenset({"rl_override_gemini_cash"})


def _normalize_checkpoint(path: str | Path | None) -> str | None:
    if path is None or str(path).strip() == "":
        return None
    return str(Path(path).expanduser().resolve(strict=False))


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).strip().replace("Z", "+00:00")
    ts = datetime.fromisoformat(normalized)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _extract_snapshot_symbols(snapshot: dict[str, Any]) -> list[str]:
    requested = normalize_symbols(snapshot.get("requested_tradable_symbols"))
    active = normalize_symbols(snapshot.get("active_tradable_symbols"))
    return requested or active


@dataclass(frozen=True)
class BinanceHybridRuntimeAuditResult:
    launch_script: str
    trace_dir: str
    window_start: str | None
    window_end: str | None
    launch_checkpoint: str | None
    launch_symbols: list[str]
    launch_leverage: float
    snapshot_count: int
    healthy_completed_count: int
    gemini_call_skipped_count: int
    degraded_status_count: int
    degraded_allocation_source_count: int
    override_allocation_source_count: int
    checkpoint_missing_count: int
    checkpoint_mismatch_count: int
    symbols_mismatch_count: int
    leverage_mismatch_count: int
    status_counts: dict[str, int]
    allocation_source_counts: dict[str, int]
    unexpected_symbol_activity_counts: dict[str, int]
    unexpected_order_symbol_counts: dict[str, int]
    degraded_examples: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_runtime_audit_launch_mismatch_issues(
    result: BinanceHybridRuntimeAuditResult,
    *,
    require_recent_snapshots: bool = False,
    require_checkpoint_match: bool = True,
) -> list[str]:
    issues: list[str] = []
    if require_recent_snapshots and result.snapshot_count == 0:
        issues.append("no recent live allocation snapshots found in audit window")
    if result.checkpoint_mismatch_count:
        issues.append(
            f"{result.checkpoint_mismatch_count} recent live snapshot(s) disagree with the launch checkpoint"
        )
    if require_checkpoint_match and result.checkpoint_missing_count:
        issues.append(
            f"{result.checkpoint_missing_count} recent live snapshot(s) are missing rl_checkpoint"
        )
    if result.symbols_mismatch_count:
        issues.append(
            f"{result.symbols_mismatch_count} recent live snapshot(s) disagree with the launch symbols"
        )
    if result.leverage_mismatch_count:
        issues.append(
            f"{result.leverage_mismatch_count} recent live snapshot(s) disagree with the launch leverage"
        )
    if result.unexpected_symbol_activity_counts:
        issues.append(
            "unexpected symbol activity outside launch universe: "
            + ", ".join(sorted(result.unexpected_symbol_activity_counts))
        )
    if result.unexpected_order_symbol_counts:
        issues.append(
            "unexpected order symbols outside launch universe: "
            + ", ".join(sorted(result.unexpected_order_symbol_counts))
        )
    return issues


def build_runtime_audit_health_issues(
    result: BinanceHybridRuntimeAuditResult,
    *,
    require_recent_snapshots: bool = False,
    min_healthy_completed_count: int = 1,
    max_degraded_status_count: int = 0,
    max_degraded_fallback_count: int = 0,
    max_gemini_call_skipped_count: int = 0,
) -> list[str]:
    issues: list[str] = []
    if require_recent_snapshots and result.snapshot_count == 0:
        issues.append("no recent live allocation snapshots found in audit window")
    if result.healthy_completed_count < int(min_healthy_completed_count):
        issues.append(
            "recent live runtime has too few healthy completed allocation cycles "
            f"({result.healthy_completed_count} < {int(min_healthy_completed_count)})"
        )
    if result.degraded_status_count > int(max_degraded_status_count):
        issues.append(
            "recent live runtime has too many degraded status cycles "
            f"({result.degraded_status_count} > {int(max_degraded_status_count)})"
        )
    if result.degraded_allocation_source_count > int(max_degraded_fallback_count):
        issues.append(
            "recent live runtime has too many degraded fallback allocation cycles "
            f"({result.degraded_allocation_source_count} > {int(max_degraded_fallback_count)})"
        )
    if result.gemini_call_skipped_count > int(max_gemini_call_skipped_count):
        issues.append(
            "recent live runtime skipped Gemini too often "
            f"({result.gemini_call_skipped_count} > {int(max_gemini_call_skipped_count)})"
        )
    return issues


def audit_binance_hybrid_runtime(
    *,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    trace_dir: str | Path = DEFAULT_TRACE_DIR,
    start: str | None = None,
    end: str | None = None,
    hours: float | None = 24.0,
    max_examples: int = 5,
) -> BinanceHybridRuntimeAuditResult:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    end_ts = _parse_utc(end) if end else datetime.now(UTC)
    start_ts = _parse_utc(start) if start else None
    if start_ts is None and hours is not None:
        start_ts = end_ts - timedelta(hours=float(hours))

    snapshots = [
        snapshot
        for snapshot in load_cycle_snapshots(
            log_dir=Path(trace_dir),
            start=start_ts.isoformat() if start_ts is not None else None,
            end=end_ts.isoformat() if end_ts is not None else None,
            live_only=True,
        )
        if str(snapshot.get("cycle_kind") or "") == "allocation"
    ]

    launch_checkpoint = _normalize_checkpoint(launch_cfg.rl_checkpoint)
    launch_symbols = list(launch_cfg.symbols)
    launch_leverage = float(launch_cfg.leverage)

    status_counts: Counter[str] = Counter()
    allocation_source_counts: Counter[str] = Counter()
    unexpected_symbol_activity_counts: Counter[str] = Counter()
    unexpected_order_symbol_counts: Counter[str] = Counter()

    healthy_completed_count = 0
    gemini_call_skipped_count = 0
    degraded_status_count = 0
    degraded_allocation_source_count = 0
    override_allocation_source_count = 0
    checkpoint_missing_count = 0
    checkpoint_mismatch_count = 0
    symbols_mismatch_count = 0
    leverage_mismatch_count = 0
    degraded_examples: list[dict[str, str]] = []

    for snapshot in snapshots:
        status = str(snapshot.get("status") or "").strip() or "<missing>"
        allocation_source = str(snapshot.get("allocation_source") or "").strip() or "<none>"
        status_counts[status] += 1
        allocation_source_counts[allocation_source] += 1

        degraded_status = status in _DEGRADED_STATUSES
        degraded_source = allocation_source in _DEGRADED_ALLOCATION_SOURCES
        if degraded_status:
            degraded_status_count += 1
        if degraded_source:
            degraded_allocation_source_count += 1
        if allocation_source in _OVERRIDE_ALLOCATION_SOURCES:
            override_allocation_source_count += 1
        if bool(snapshot.get("gemini_call_skipped")):
            gemini_call_skipped_count += 1
        snapshot_checkpoint = _normalize_checkpoint(snapshot.get("rl_checkpoint"))
        if launch_checkpoint is None:
            if snapshot_checkpoint is not None:
                checkpoint_mismatch_count += 1
        elif snapshot_checkpoint is None:
            checkpoint_missing_count += 1
        elif snapshot_checkpoint != launch_checkpoint:
            checkpoint_mismatch_count += 1

        snapshot_symbols = _extract_snapshot_symbols(snapshot)
        if snapshot_symbols and snapshot_symbols != launch_symbols:
            symbols_mismatch_count += 1

        snapshot_leverage = safe_float(snapshot.get("requested_leverage"))
        if (
            snapshot_leverage is not None
            and not math.isclose(snapshot_leverage, launch_leverage, rel_tol=0.0, abs_tol=1e-9)
        ):
            leverage_mismatch_count += 1

        allowed_market_symbols = snapshot_allowed_market_symbols(snapshot, launch_symbols)
        for detail in snapshot.get("symbols_detail") or []:
            if not isinstance(detail, dict):
                continue
            symbol = str(detail.get("symbol") or "").strip().upper()
            if symbol and symbol not in launch_symbols and detail_has_activity(detail):
                unexpected_symbol_activity_counts[symbol] += 1

        if allowed_market_symbols:
            for order_symbol in iter_snapshot_order_symbols(snapshot):
                if order_symbol not in allowed_market_symbols:
                    unexpected_order_symbol_counts[order_symbol] += 1

        snapshot_confirmed, _snapshot_reason, _snapshot_meta = evaluate_live_snapshot_health(
            snapshot,
            expected_checkpoint_norm=launch_checkpoint,
            expect_no_rl_checkpoint=launch_checkpoint is None,
            expected_symbols_list=launch_symbols,
            launch_symbols=launch_symbols,
            expected_leverage=launch_leverage,
        )
        if status == "completed" and snapshot_confirmed is True:
            healthy_completed_count += 1

        if (degraded_status or degraded_source) and len(degraded_examples) < max(0, int(max_examples)):
            reasoning = ""
            allocation_plan = snapshot.get("allocation_plan")
            if isinstance(allocation_plan, dict):
                reasoning = str(allocation_plan.get("reasoning") or "").strip()
            error = str(snapshot.get("error") or "").strip()
            degraded_examples.append(
                {
                    "cycle_started_at": str(snapshot.get("cycle_started_at") or ""),
                    "status": status,
                    "allocation_source": allocation_source,
                    "reason": error or reasoning,
                }
            )

    return BinanceHybridRuntimeAuditResult(
        launch_script=str(Path(launch_script).resolve()),
        trace_dir=str(Path(trace_dir).resolve()),
        window_start=start_ts.isoformat() if start_ts is not None else None,
        window_end=end_ts.isoformat(),
        launch_checkpoint=launch_checkpoint,
        launch_symbols=launch_symbols,
        launch_leverage=launch_leverage,
        snapshot_count=len(snapshots),
        healthy_completed_count=healthy_completed_count,
        gemini_call_skipped_count=gemini_call_skipped_count,
        degraded_status_count=degraded_status_count,
        degraded_allocation_source_count=degraded_allocation_source_count,
        override_allocation_source_count=override_allocation_source_count,
        checkpoint_missing_count=checkpoint_missing_count,
        checkpoint_mismatch_count=checkpoint_mismatch_count,
        symbols_mismatch_count=symbols_mismatch_count,
        leverage_mismatch_count=leverage_mismatch_count,
        status_counts=dict(status_counts),
        allocation_source_counts=dict(allocation_source_counts),
        unexpected_symbol_activity_counts=dict(unexpected_symbol_activity_counts),
        unexpected_order_symbol_counts=dict(unexpected_order_symbol_counts),
        degraded_examples=degraded_examples,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit recent Binance hybrid production runtime snapshots for degraded states and launch drift."
    )
    parser.add_argument("--launch-script", default=str(DEFAULT_LAUNCH_SCRIPT))
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--output-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = audit_binance_hybrid_runtime(
        launch_script=args.launch_script,
        trace_dir=args.trace_dir,
        start=args.start or None,
        end=args.end or None,
        hours=args.hours,
        max_examples=args.max_examples,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    print("Binance hybrid runtime audit")
    print(f"  launch:      {result.launch_script}")
    print(f"  trace dir:   {result.trace_dir}")
    print(f"  window:      {result.window_start or '<all>'} -> {result.window_end}")
    print(f"  checkpoint:  {result.launch_checkpoint or 'RL disabled'}")
    print(f"  symbols:     {' '.join(result.launch_symbols)}")
    print(f"  leverage:    {result.launch_leverage:g}")
    print("")
    print(f"Snapshots:            {result.snapshot_count}")
    print(f"Healthy completed:    {result.healthy_completed_count}")
    print(f"Gemini skipped:       {result.gemini_call_skipped_count}")
    print(f"Degraded statuses:    {result.degraded_status_count}")
    print(f"Degraded fallbacks:   {result.degraded_allocation_source_count}")
    print(f"Gemini overrides:     {result.override_allocation_source_count}")
    print(f"Checkpoint missing:   {result.checkpoint_missing_count}")
    print(f"Checkpoint mismatch:  {result.checkpoint_mismatch_count}")
    print(f"Symbols mismatch:     {result.symbols_mismatch_count}")
    print(f"Leverage mismatch:    {result.leverage_mismatch_count}")
    print("")
    print("Status counts:")
    for status, count in sorted(result.status_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {status}: {count}")
    print("Allocation sources:")
    for source, count in sorted(result.allocation_source_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {source}: {count}")
    if result.unexpected_symbol_activity_counts:
        print("Unexpected symbol activity:")
        for symbol, count in sorted(
            result.unexpected_symbol_activity_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {symbol}: {count}")
    if result.unexpected_order_symbol_counts:
        print("Unexpected order symbols:")
        for symbol, count in sorted(
            result.unexpected_order_symbol_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {symbol}: {count}")
    if result.degraded_examples:
        print("Degraded examples:")
        for example in result.degraded_examples:
            print(
                "  "
                f"{example['cycle_started_at']} "
                f"status={example['status']} "
                f"source={example['allocation_source']} "
                f"reason={example['reason'] or '<none>'}"
            )
    if args.output_json:
        print(f"\nJSON: {Path(args.output_json).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
