#!/usr/bin/env python3
"""Audit exit coverage for open positions and optionally fix missing watchers.

Ensures each open position has an active exit watcher (take-profit) and
reports whether an exit order exists. Designed for PAPER mode audits.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import alpaca_wrapper  # noqa: E402
from stock.state import get_state_dir, get_paper_suffix, resolve_state_suffix  # noqa: E402
from src.exit_plan_tracker import get_exit_tracker, ExitPlan  # noqa: E402
from src.process_utils import spawn_close_position_at_maxdiff_takeprofit  # noqa: E402


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class PositionInfo:
    symbol: str
    norm_symbol: str
    qty: float
    side: str  # entry side: buy for long, sell for short
    current_price: Optional[float]


@dataclass
class ExitAudit:
    position: PositionInfo
    exit_watchers: List[dict]
    active_exit_watchers: List[dict]
    exit_orders: List[object]
    exit_plan: Optional[ExitPlan]


def _load_exit_watchers(watcher_dir: Path) -> Dict[str, List[dict]]:
    watchers: Dict[str, List[dict]] = {}
    if not watcher_dir.exists():
        return watchers

    for path in watcher_dir.glob("*_exit_*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload.get("mode") != "exit":
            continue
        symbol = payload.get("symbol")
        if not symbol:
            continue
        payload["_path"] = str(path)
        norm = _normalize_symbol(symbol)
        watchers.setdefault(norm, []).append(payload)
    return watchers


def _group_orders(orders: Iterable[object]) -> Dict[str, List[object]]:
    grouped: Dict[str, List[object]] = {}
    for order in orders:
        symbol = getattr(order, "symbol", None)
        if not symbol:
            continue
        norm = _normalize_symbol(str(symbol))
        grouped.setdefault(norm, []).append(order)
    return grouped


def _collect_positions() -> List[PositionInfo]:
    positions = alpaca_wrapper.get_all_positions()
    results: List[PositionInfo] = []
    for pos in positions:
        try:
            qty = float(pos.qty)
        except (TypeError, ValueError):
            continue
        if qty == 0:
            continue
        symbol = str(pos.symbol)
        side = "buy" if qty > 0 else "sell"
        current_price = _safe_float(getattr(pos, "current_price", None))
        results.append(
            PositionInfo(
                symbol=symbol,
                norm_symbol=_normalize_symbol(symbol),
                qty=qty,
                side=side,
                current_price=current_price,
            )
        )
    return results


def _exit_side_for(position: PositionInfo) -> str:
    return "sell" if position.qty > 0 else "buy"


def _resolve_exit_price(
    audit: ExitAudit,
    fallback_bps: float,
) -> Optional[float]:
    if audit.exit_plan:
        return float(audit.exit_plan.exit_price)

    # Use existing exit order if present
    for order in audit.exit_orders:
        limit_price = _safe_float(getattr(order, "limit_price", None))
        if limit_price and limit_price > 0:
            return limit_price

    # Fallback to current price +/- bps
    if audit.position.current_price:
        current = audit.position.current_price
        bump = fallback_bps / 10000.0
        if audit.position.qty > 0:
            return current * (1.0 + bump)
        return current * (1.0 - bump)

    return None


def _resolve_expiry_minutes(plan: Optional[ExitPlan], default_minutes: int) -> int:
    if plan is None:
        return default_minutes
    try:
        deadline = plan.deadline_dt
    except Exception:
        return default_minutes
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds() / 60)
    if remaining <= 0:
        return default_minutes
    return max(1, remaining)


def audit_exit_coverage(
    *,
    fix: bool,
    fallback_bps: float,
    default_expiry_minutes: int,
    dry_run: bool,
) -> Tuple[List[ExitAudit], List[str]]:
    positions = _collect_positions()
    orders = alpaca_wrapper.get_orders(use_cache=False)
    orders_by_symbol = _group_orders(orders)

    suffix = resolve_state_suffix()
    watcher_dir = get_state_dir() / f"maxdiff_watchers{get_paper_suffix()}{suffix or ''}"
    watchers_by_symbol = _load_exit_watchers(watcher_dir)

    exit_tracker = get_exit_tracker()
    plans_by_symbol = {
        _normalize_symbol(plan.symbol): plan for plan in exit_tracker.get_all_plans()
    }

    audits: List[ExitAudit] = []
    fixes: List[str] = []

    for position in positions:
        watchers = watchers_by_symbol.get(position.norm_symbol, [])
        entry_side = position.side
        active_watchers = [
            w for w in watchers if w.get("active") and w.get("side") == entry_side
        ]

        exit_side = _exit_side_for(position)
        exit_orders = [
            o
            for o in orders_by_symbol.get(position.norm_symbol, [])
            if getattr(o, "side", "").lower() == exit_side
        ]

        plan = plans_by_symbol.get(position.norm_symbol)

        audit = ExitAudit(
            position=position,
            exit_watchers=watchers,
            active_exit_watchers=active_watchers,
            exit_orders=exit_orders,
            exit_plan=plan,
        )
        audits.append(audit)

        if fix and not active_watchers:
            takeprofit_price = _resolve_exit_price(audit, fallback_bps)
            if takeprofit_price is None:
                fixes.append(f"{position.symbol}: missing exit watcher (no price source)")
                continue
            expiry_minutes = _resolve_expiry_minutes(plan, default_expiry_minutes)
            if dry_run:
                fixes.append(
                    f"[DRY RUN] spawn close watcher for {position.symbol} @ {takeprofit_price:.6f}"
                )
                continue
            spawn_close_position_at_maxdiff_takeprofit(
                symbol=position.symbol,
                side=entry_side,
                takeprofit_price=takeprofit_price,
                target_qty=abs(position.qty),
                expiry_minutes=expiry_minutes,
                entry_strategy=None,
            )
            fixes.append(
                f"spawned close watcher for {position.symbol} @ {takeprofit_price:.6f}"
            )

    return audits, fixes


def _print_report(audits: List[ExitAudit], fixes: List[str]) -> None:
    print("=" * 72)
    print("Exit Coverage Report")
    print("=" * 72)
    print(f"Positions: {len(audits)}")
    print()

    for audit in audits:
        pos = audit.position
        exit_side = _exit_side_for(pos)
        has_watchers = len(audit.active_exit_watchers) > 0
        has_orders = len(audit.exit_orders) > 0
        plan = audit.exit_plan
        plan_info = (
            f"plan_tp={plan.exit_price:.6f} deadline={plan.exit_deadline}"
            if plan
            else "plan=none"
        )
        print(
            f"{pos.symbol:<8} qty={pos.qty:>12.6f} side={pos.side:<4} "
            f"exit_side={exit_side:<4} watcher={'Y' if has_watchers else 'N'} "
            f"order={'Y' if has_orders else 'N'} {plan_info}"
        )

    if fixes:
        print("\nFixes:")
        for entry in fixes:
            print(f"- {entry}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exit watcher/order coverage")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Spawn missing exit watchers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without spawning watchers.",
    )
    parser.add_argument(
        "--fallback-bps",
        type=float,
        default=10.0,
        help="Fallback take-profit distance in bps when no plan/order exists.",
    )
    parser.add_argument(
        "--default-expiry-minutes",
        type=int,
        default=24 * 60,
        help="Default expiry minutes for watchers when no plan is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audits, fixes = audit_exit_coverage(
        fix=args.fix,
        fallback_bps=args.fallback_bps,
        default_expiry_minutes=args.default_expiry_minutes,
        dry_run=args.dry_run,
    )

    _print_report(audits, fixes)


if __name__ == "__main__":
    main()
