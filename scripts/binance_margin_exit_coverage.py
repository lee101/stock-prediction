#!/usr/bin/env python3
"""Audit cross-margin positions that lack matching reduce/exit orders.

Read-only by default. This is intended to make the "N positions but M sell
orders" situation explicit without eyeballing ``scripts/binance_status.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.binan import binance_wrapper as bw
from src.binan.binance_margin import get_margin_account, get_open_margin_orders


STABLE_ASSETS = {"USDT", "FDUSD", "BUSD", "USDC"}


@dataclass(frozen=True)
class CoverageRow:
    asset: str
    pair: str
    pair_candidates: tuple[str, ...]
    net_qty: float
    est_value_usdt: float
    open_sell_qty: float
    open_buy_qty: float
    sell_coverage_ratio: float
    status: str


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _asset_price_usdt(asset: str) -> float:
    asset = str(asset or "").upper().strip()
    if not asset:
        return 0.0
    if asset in {"USDT", "FDUSD", "BUSD", "USDC"}:
        return 1.0
    for pair in (f"{asset}USDT", f"{asset}FDUSD", f"{asset}BUSD"):
        try:
            price = _safe_float(bw.get_symbol_price(pair))
        except Exception:
            continue
        if price > 0.0:
            return price
    return 0.0


def _remaining_qty(order: dict) -> float:
    return max(0.0, _safe_float(order.get("origQty")) - _safe_float(order.get("executedQty")))


def _pair_for_asset(asset: str) -> str:
    return f"{asset.upper()}USDT"


def _pair_candidates_for_asset(asset: str) -> tuple[str, ...]:
    base = asset.upper()
    return tuple(dict.fromkeys([f"{base}USDT", f"{base}FDUSD", f"{base}BUSD", f"{base}USDC"]))


def load_coverage(*, min_value_usdt: float) -> list[CoverageRow]:
    account = get_margin_account()
    open_orders = list(get_open_margin_orders())
    sell_qty_by_pair: dict[str, float] = {}
    buy_qty_by_pair: dict[str, float] = {}
    for order in open_orders:
        pair = str(order.get("symbol") or "").upper().strip()
        side = str(order.get("side") or "").upper().strip()
        if not pair:
            continue
        if side == "SELL":
            sell_qty_by_pair[pair] = sell_qty_by_pair.get(pair, 0.0) + _remaining_qty(order)
        elif side == "BUY":
            buy_qty_by_pair[pair] = buy_qty_by_pair.get(pair, 0.0) + _remaining_qty(order)

    rows: list[CoverageRow] = []
    for entry in account.get("userAssets", []):
        asset = str(entry.get("asset") or "").upper().strip()
        if not asset or asset in STABLE_ASSETS:
            continue
        net_qty = _safe_float(entry.get("netAsset"))
        if net_qty <= 0.0:
            continue
        price = _asset_price_usdt(asset)
        est_value = net_qty * price
        if est_value < float(min_value_usdt):
            continue
        pair = _pair_for_asset(asset)
        pair_candidates = _pair_candidates_for_asset(asset)
        open_sell_qty = sum(sell_qty_by_pair.get(candidate, 0.0) for candidate in pair_candidates)
        open_buy_qty = sum(buy_qty_by_pair.get(candidate, 0.0) for candidate in pair_candidates)
        ratio = open_sell_qty / max(net_qty, 1e-12)
        if ratio >= 0.98:
            status = "covered"
        elif open_sell_qty > 0.0:
            status = "partial"
        else:
            status = "missing"
        rows.append(
            CoverageRow(
                asset=asset,
                pair=pair,
                pair_candidates=pair_candidates,
                net_qty=float(net_qty),
                est_value_usdt=float(est_value),
                open_sell_qty=float(open_sell_qty),
                open_buy_qty=float(open_buy_qty),
                sell_coverage_ratio=float(ratio),
                status=status,
            )
        )
    rows.sort(key=lambda row: abs(row.est_value_usdt), reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Binance cross-margin exit order coverage.")
    parser.add_argument("--min-value-usdt", type=float, default=12.0)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args()

    rows = load_coverage(min_value_usdt=float(args.min_value_usdt))
    summary = {
        "positions": len(rows),
        "covered": sum(1 for row in rows if row.status == "covered"),
        "partial": sum(1 for row in rows if row.status == "partial"),
        "missing": sum(1 for row in rows if row.status == "missing"),
        "rows": [asdict(row) for row in rows],
    }
    exit_code = 1 if summary["missing"] or summary["partial"] else 0
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return exit_code

    print(
        f"margin exit coverage: positions={summary['positions']} "
        f"covered={summary['covered']} partial={summary['partial']} missing={summary['missing']}"
    )
    print("asset pair       net_qty       value   sell_qty  buy_qty  cover  status")
    for row in rows:
        print(
            f"{row.asset:<5} {row.pair:<9} {row.net_qty:>11.6g} "
            f"${row.est_value_usdt:>8.2f} {row.open_sell_qty:>9.6g} "
            f"{row.open_buy_qty:>8.6g} {row.sell_coverage_ratio:>6.2f} {row.status}"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
