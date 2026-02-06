from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

from loguru import logger

# Allow running as `python scripts/...` without needing PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _coerce_amount(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def compute_spendable_quote(*, free_quote: float, leave_quote: float, max_spend: float | None) -> float:
    free_quote = _coerce_amount(free_quote)
    leave_quote = max(0.0, _coerce_amount(leave_quote))
    spendable = max(0.0, free_quote - leave_quote)
    if max_spend is not None:
        cap = max(0.0, _coerce_amount(max_spend))
        spendable = min(spendable, cap)
    return spendable


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Binance USDT -> FDUSD (spot) via FDUSDUSDT.")
    parser.add_argument("--pair", default="FDUSDUSDT", help="Conversion symbol (default: FDUSDUSDT).")
    parser.add_argument("--from-asset", default="USDT", help="Source asset (default: USDT).")
    parser.add_argument("--to-asset", default="FDUSD", help="Target asset (default: FDUSD).")
    parser.add_argument("--leave-usdt", type=float, default=10.0, help="Leave this much USDT unconverted (default: 10).")
    parser.add_argument("--max-spend-usdt", type=float, default=None, help="Cap conversion spend in USDT (default: no cap).")
    parser.add_argument("--min-spend-usdt", type=float, default=None, help="Skip if spendable < this (default: minNotional).")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Place the live market order (default is dry-run/test order).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    from src.binan import binance_wrapper

    args = _parse_args(argv)
    pair = str(args.pair).strip().upper()
    asset_from = str(args.from_asset).strip().upper()
    asset_to = str(args.to_asset).strip().upper()

    client = binance_wrapper.get_client()
    if client is None:
        logger.error("Binance client unavailable; check credentials / connectivity.")
        return 2

    info = client.get_symbol_info(pair)
    if not isinstance(info, dict):
        logger.error(f"Binance symbol {pair} not available on this endpoint.")
        return 2
    base = str(info.get("baseAsset") or "").upper()
    quote = str(info.get("quoteAsset") or "").upper()
    if base != asset_to or quote != asset_from:
        logger.warning(f"{pair} is base={base} quote={quote} (expected {asset_to}/{asset_from}).")

    free_from = binance_wrapper.get_asset_free_balance(asset_from, client=client)
    free_to = binance_wrapper.get_asset_free_balance(asset_to, client=client)
    logger.info(f"Balances: {asset_from} free={free_from:.8f} | {asset_to} free={free_to:.8f}")

    min_notional = binance_wrapper.get_min_notional(pair, client=client) or 0.0
    min_spend = _coerce_amount(args.min_spend_usdt) if args.min_spend_usdt is not None else float(min_notional)

    spendable = compute_spendable_quote(
        free_quote=free_from,
        leave_quote=float(args.leave_usdt),
        max_spend=args.max_spend_usdt,
    )
    if spendable <= 0:
        logger.info("No spendable USDT (after leave buffer). Nothing to do.")
        return 0
    if spendable < min_spend:
        logger.info(f"Spendable {spendable:.8f} below min_spend {min_spend:.8f}; skipping.")
        return 0

    last_price = binance_wrapper.get_symbol_price(pair, client=client)
    if last_price is not None and last_price > 0:
        est_to = spendable / last_price
        logger.info(f"{pair} price={last_price:.6f} => estimated {asset_to} bought ~= {est_to:.8f}")
    else:
        logger.info(f"Unable to fetch latest {pair} price; proceeding without estimate.")

    dry_run = not bool(args.execute)
    if dry_run:
        logger.warning("Dry-run mode: using Binance test order endpoint; no funds will be moved.")
    else:
        logger.warning("EXECUTE mode: placing live Binance market order.")

    order = binance_wrapper.create_market_buy_quote(
        pair,
        quote_amount=spendable,
        client=client,
        dry_run=dry_run,
    )
    logger.info(f"Order response: {order}")

    if not dry_run:
        # Fetch updated balances for visibility.
        free_from_after = binance_wrapper.get_asset_free_balance(asset_from, client=client)
        free_to_after = binance_wrapper.get_asset_free_balance(asset_to, client=client)
        logger.info(
            f"Balances after: {asset_from} free={free_from_after:.8f} | {asset_to} free={free_to_after:.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
