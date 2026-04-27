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
from binanceneural.execution import quantize_price, quantize_qty, resolve_symbol_rules
from src.binan.binance_margin import (
    create_margin_limit_sell,
    get_margin_account,
    get_open_margin_orders,
)


STABLE_ASSETS = {"USDT", "FDUSD", "BUSD", "USDC"}


@dataclass(frozen=True)
class CoverageRow:
    asset: str
    pair: str
    pair_candidates: tuple[str, ...]
    net_qty: float
    free_qty: float
    market_price: float
    est_value_usdt: float
    open_sell_qty: float
    open_buy_qty: float
    coverage_gap_qty: float
    sell_coverage_ratio: float
    status: str


@dataclass(frozen=True)
class RepairPlan:
    asset: str
    pair: str
    quantity: float
    price: float
    notional_usdt: float
    status: str
    reason: str
    order: dict | None = None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _asset_price_and_pair(asset: str) -> tuple[float, str]:
    asset = str(asset or "").upper().strip()
    if not asset:
        return 0.0, ""
    if asset in {"USDT", "FDUSD", "BUSD", "USDC"}:
        return 1.0, asset
    fallback_pair = _pair_for_asset(asset)
    for pair in _pair_candidates_for_asset(asset):
        try:
            price = _safe_float(bw.get_symbol_price(pair))
        except Exception:
            continue
        if price > 0.0:
            return price, pair
    return 0.0, fallback_pair


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
        free_qty = _safe_float(entry.get("free"), default=net_qty)
        price, price_pair = _asset_price_and_pair(asset)
        est_value = net_qty * price
        if est_value < float(min_value_usdt):
            continue
        pair = price_pair or _pair_for_asset(asset)
        pair_candidates = _pair_candidates_for_asset(asset)
        open_sell_qty = sum(sell_qty_by_pair.get(candidate, 0.0) for candidate in pair_candidates)
        open_buy_qty = sum(buy_qty_by_pair.get(candidate, 0.0) for candidate in pair_candidates)
        ratio = open_sell_qty / max(net_qty, 1e-12)
        gap_qty = max(0.0, net_qty - open_sell_qty)
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
                free_qty=float(free_qty),
                market_price=float(price),
                est_value_usdt=float(est_value),
                open_sell_qty=float(open_sell_qty),
                open_buy_qty=float(open_buy_qty),
                coverage_gap_qty=float(gap_qty),
                sell_coverage_ratio=float(ratio),
                status=status,
            )
        )
    rows.sort(key=lambda row: abs(row.est_value_usdt), reverse=True)
    return rows


def build_repair_plan(
    rows: list[CoverageRow],
    *,
    target_markup_pct: float,
    min_order_value_usdt: float,
) -> list[RepairPlan]:
    plans: list[RepairPlan] = []
    markup = max(0.0, float(target_markup_pct))
    min_notional = max(0.0, float(min_order_value_usdt))
    for row in rows:
        if row.status == "covered":
            continue
        gap_qty = max(0.0, float(row.coverage_gap_qty))
        place_qty = min(gap_qty, max(0.0, float(row.free_qty)))
        target_price = float(row.market_price) * (1.0 + markup)
        if gap_qty <= 0.0:
            plans.append(RepairPlan(row.asset, row.pair, 0.0, target_price, 0.0, "skipped", "no_coverage_gap"))
            continue
        if place_qty <= 0.0:
            plans.append(RepairPlan(row.asset, row.pair, 0.0, target_price, 0.0, "skipped", "no_free_qty"))
            continue
        if target_price <= 0.0:
            plans.append(RepairPlan(row.asset, row.pair, place_qty, 0.0, 0.0, "skipped", "no_market_price"))
            continue
        notional = place_qty * target_price
        if notional < min_notional:
            plans.append(
                RepairPlan(row.asset, row.pair, place_qty, target_price, notional, "skipped", "below_min_order_value")
            )
            continue
        reason = "partial_gap" if row.status == "partial" else "missing_exit"
        plans.append(RepairPlan(row.asset, row.pair, place_qty, target_price, notional, "planned", reason))
    return plans


def place_repair_orders(plans: list[RepairPlan], *, side_effect_type: str) -> list[RepairPlan]:
    placed: list[RepairPlan] = []
    for plan in plans:
        if plan.status != "planned":
            placed.append(plan)
            continue
        try:
            rules = resolve_symbol_rules(plan.pair)
            price = quantize_price(plan.price, tick_size=rules.tick_size, side="sell")
            qty = quantize_qty(plan.quantity, step_size=rules.step_size)
            if rules.min_qty is not None and qty < rules.min_qty:
                placed.append(RepairPlan(plan.asset, plan.pair, qty, price, qty * price, "skipped", "below_min_qty"))
                continue
            if rules.min_notional is not None and qty * price < rules.min_notional:
                placed.append(
                    RepairPlan(plan.asset, plan.pair, qty, price, qty * price, "skipped", "below_min_notional")
                )
                continue
            order = create_margin_limit_sell(
                plan.pair,
                qty,
                price,
                side_effect_type=str(side_effect_type or "NO_SIDE_EFFECT"),
            )
            placed.append(RepairPlan(plan.asset, plan.pair, qty, price, qty * price, "placed", plan.reason, order))
        except Exception as exc:
            placed.append(
                RepairPlan(
                    plan.asset,
                    plan.pair,
                    plan.quantity,
                    plan.price,
                    plan.notional_usdt,
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                )
            )
    return placed


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Binance cross-margin exit order coverage.")
    parser.add_argument("--min-value-usdt", type=float, default=12.0)
    parser.add_argument("--target-markup-pct", type=float, default=0.20)
    parser.add_argument("--min-order-value-usdt", type=float, default=12.0)
    parser.add_argument("--side-effect-type", default="NO_SIDE_EFFECT")
    parser.add_argument("--execute", action="store_true", help="Actually place missing target SELL orders")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args()

    rows = load_coverage(min_value_usdt=float(args.min_value_usdt))
    repair_plan = build_repair_plan(
        rows,
        target_markup_pct=float(args.target_markup_pct),
        min_order_value_usdt=float(args.min_order_value_usdt),
    )
    if args.execute:
        repair_plan = place_repair_orders(repair_plan, side_effect_type=str(args.side_effect_type))
    summary = {
        "positions": len(rows),
        "covered": sum(1 for row in rows if row.status == "covered"),
        "partial": sum(1 for row in rows if row.status == "partial"),
        "missing": sum(1 for row in rows if row.status == "missing"),
        "repair_planned": sum(1 for row in repair_plan if row.status == "planned"),
        "repair_placed": sum(1 for row in repair_plan if row.status == "placed"),
        "repair_failed": sum(1 for row in repair_plan if row.status == "failed"),
        "execute": bool(args.execute),
        "rows": [asdict(row) for row in rows],
        "repair_plan": [asdict(row) for row in repair_plan],
    }
    exit_code = 1 if summary["missing"] or summary["partial"] else 0
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return exit_code

    print(
        f"margin exit coverage: positions={summary['positions']} "
        f"covered={summary['covered']} partial={summary['partial']} missing={summary['missing']} "
        f"repair_planned={summary['repair_planned']} repair_placed={summary['repair_placed']}"
    )
    print("asset pair       net_qty       free       value   sell_qty  buy_qty  gap_qty  cover  status")
    for row in rows:
        print(
            f"{row.asset:<5} {row.pair:<9} {row.net_qty:>11.6g} "
            f"{row.free_qty:>9.6g} ${row.est_value_usdt:>8.2f} "
            f"{row.open_sell_qty:>9.6g} {row.open_buy_qty:>8.6g} "
            f"{row.coverage_gap_qty:>8.6g} {row.sell_coverage_ratio:>6.2f} {row.status}"
        )
    if repair_plan:
        action = "placed" if args.execute else "dry-run plan"
        print(f"\nexit repair {action}:")
        print("asset pair       qty          price       notional  status   reason")
        for plan in repair_plan:
            print(
                f"{plan.asset:<5} {plan.pair:<9} {plan.quantity:>11.6g} "
                f"{plan.price:>12.6g} ${plan.notional_usdt:>8.2f} {plan.status:<8} {plan.reason}"
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
