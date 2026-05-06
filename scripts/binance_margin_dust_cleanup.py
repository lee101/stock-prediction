#!/usr/bin/env python3
"""Plan or execute Binance cross-margin dust cleanup."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.binan.binance_margin import (  # noqa: E402
    exchange_small_liability_assets,
    get_margin_dust_assets,
    get_small_liability_exchange_assets,
    transfer_margin_dust,
)


@dataclass(frozen=True)
class DustCleanupPlan:
    liability_assets: list[str]
    margin_dust_assets: list[str]
    liability_rows: list[dict[str, Any]]
    margin_dust_rows: list[dict[str, Any]]
    total_margin_dust_bnb: float
    execute: bool
    liability_result: dict[str, Any] | None = None
    margin_dust_result: dict[str, Any] | None = None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_cleanup_plan(
    *,
    include_liability_assets: set[str] | None = None,
    include_margin_dust_assets: set[str] | None = None,
) -> DustCleanupPlan:
    liability_rows = get_small_liability_exchange_assets()
    margin_dust_payload = get_margin_dust_assets()
    margin_dust_rows = [row for row in margin_dust_payload.get("details", []) if isinstance(row, dict)]
    liability_assets = sorted(
        {str(row.get("asset") or "").upper() for row in liability_rows if str(row.get("asset") or "").strip()}
    )
    margin_dust_assets = sorted(
        {str(row.get("asset") or "").upper() for row in margin_dust_rows if str(row.get("asset") or "").strip()}
    )
    if include_liability_assets is not None:
        liability_assets = [asset for asset in liability_assets if asset in include_liability_assets]
    if include_margin_dust_assets is not None:
        margin_dust_assets = [asset for asset in margin_dust_assets if asset in include_margin_dust_assets]
    return DustCleanupPlan(
        liability_assets=liability_assets,
        margin_dust_assets=margin_dust_assets,
        liability_rows=liability_rows,
        margin_dust_rows=margin_dust_rows,
        total_margin_dust_bnb=_safe_float(margin_dust_payload.get("totalTransferBNB")),
        execute=False,
    )


def execute_cleanup(plan: DustCleanupPlan) -> DustCleanupPlan:
    if os.getenv("ALLOW_BINANCE_DUST_CLEANUP") != "1":
        raise RuntimeError("refusing dust cleanup: set ALLOW_BINANCE_DUST_CLEANUP=1")
    liability_result = exchange_small_liability_assets(plan.liability_assets) if plan.liability_assets else None
    margin_dust_result = transfer_margin_dust(plan.margin_dust_assets) if plan.margin_dust_assets else None
    return DustCleanupPlan(
        liability_assets=plan.liability_assets,
        margin_dust_assets=plan.margin_dust_assets,
        liability_rows=plan.liability_rows,
        margin_dust_rows=plan.margin_dust_rows,
        total_margin_dust_bnb=plan.total_margin_dust_bnb,
        execute=True,
        liability_result=liability_result,
        margin_dust_result=margin_dust_result,
    )


def _parse_asset_filter(value: str) -> set[str] | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liability-assets", default="", help="Optional comma filter for liability cleanup.")
    parser.add_argument("--margin-dust-assets", default="", help="Optional comma filter for margin dust conversion.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    plan = build_cleanup_plan(
        include_liability_assets=_parse_asset_filter(args.liability_assets),
        include_margin_dust_assets=_parse_asset_filter(args.margin_dust_assets),
    )
    if args.execute:
        plan = execute_cleanup(plan)

    payload = asdict(plan)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    mode = "EXECUTED" if plan.execute else "DRY RUN"
    print(f"Binance margin dust cleanup ({mode})")
    print(f"small liability assets: {', '.join(plan.liability_assets) or '(none)'}")
    print(f"margin dust assets: {', '.join(plan.margin_dust_assets) or '(none)'}")
    print(f"estimated margin dust BNB: {plan.total_margin_dust_bnb:.8f}")
    if plan.liability_result is not None:
        print("liability result:")
        print(json.dumps(plan.liability_result, indent=2, sort_keys=True))
    if plan.margin_dust_result is not None:
        print("margin dust result:")
        print(json.dumps(plan.margin_dust_result, indent=2, sort_keys=True))
    if not plan.execute and (plan.liability_assets or plan.margin_dust_assets):
        print("set ALLOW_BINANCE_DUST_CLEANUP=1 and pass --execute to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
