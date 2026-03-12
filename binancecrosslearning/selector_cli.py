from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence

from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig


def _parse_symbol_list(raw: Optional[str]) -> tuple[str, ...]:
    if raw is None:
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for token in str(raw).split(","):
        symbol = token.strip().upper()
        if not symbol or symbol in seen:
            continue
        cleaned.append(symbol)
        seen.add(symbol)
    return tuple(cleaned)


def add_selector_realism_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument(
        "--long-only-symbols",
        default=None,
        help="Optional comma-separated symbols that may only be traded long.",
    )
    parser.add_argument(
        "--short-only-symbols",
        default=None,
        help="Optional comma-separated symbols that may only be traded short.",
    )
    parser.add_argument(
        "--select-fillable-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true, only rank entry candidates whose limit prices would have filled on the bar. "
            "Disable for more live-like select-then-fill behaviour."
        ),
    )
    parser.add_argument("--decision-lag-bars", type=int, default=0)
    parser.add_argument("--bar-margin", type=float, default=0.0)
    parser.add_argument("--limit-fill-model", default="binary", choices=["binary", "penetration"])
    parser.add_argument("--touch-fill-fraction", type=float, default=0.0)
    parser.add_argument("--margin-interest-annual", type=float, default=0.0)
    parser.add_argument("--short-borrow-cost-annual", type=float, default=0.0)
    parser.add_argument("--max-leverage-stock", type=float, default=1.0)
    parser.add_argument("--long-max-leverage-stock", type=float, default=None)
    parser.add_argument("--short-max-leverage-stock", type=float, default=None)
    parser.add_argument("--max-leverage-crypto", type=float, default=1.0)
    parser.add_argument("--long-max-leverage-crypto", type=float, default=None)
    parser.add_argument("--short-max-leverage-crypto", type=float, default=None)


def build_selection_config_from_args(
    args: argparse.Namespace,
    *,
    symbols: Sequence[str],
    fee_by_symbol: Dict[str, float],
    periods_by_symbol: Dict[str, float],
) -> SelectionConfig:
    return SelectionConfig(
        initial_cash=float(args.initial_cash),
        min_edge=float(args.min_edge),
        risk_weight=float(args.risk_weight),
        edge_mode=str(args.edge_mode),
        max_volume_fraction=args.max_volume_fraction,
        max_hold_hours=args.max_hold_hours,
        allow_reentry_same_bar=bool(args.allow_reentry_same_bar),
        enforce_market_hours=not bool(args.no_enforce_market_hours),
        close_at_eod=not bool(args.no_close_at_eod),
        fee_by_symbol=fee_by_symbol,
        periods_per_year_by_symbol=periods_by_symbol,
        symbols=list(symbols),
        allow_short=bool(getattr(args, "allow_short", False)),
        long_only_symbols=_parse_symbol_list(getattr(args, "long_only_symbols", None)) or None,
        short_only_symbols=_parse_symbol_list(getattr(args, "short_only_symbols", None)) or None,
        max_leverage_stock=float(getattr(args, "max_leverage_stock", 1.0)),
        long_max_leverage_stock=(
            None
            if getattr(args, "long_max_leverage_stock", None) is None
            else float(args.long_max_leverage_stock)
        ),
        short_max_leverage_stock=(
            None
            if getattr(args, "short_max_leverage_stock", None) is None
            else float(args.short_max_leverage_stock)
        ),
        max_leverage_crypto=float(getattr(args, "max_leverage_crypto", 1.0)),
        long_max_leverage_crypto=(
            None
            if getattr(args, "long_max_leverage_crypto", None) is None
            else float(args.long_max_leverage_crypto)
        ),
        short_max_leverage_crypto=(
            None
            if getattr(args, "short_max_leverage_crypto", None) is None
            else float(args.short_max_leverage_crypto)
        ),
        margin_interest_annual=float(getattr(args, "margin_interest_annual", 0.0)),
        short_borrow_cost_annual=float(getattr(args, "short_borrow_cost_annual", 0.0)),
        decision_lag_bars=int(getattr(args, "decision_lag_bars", 0)),
        bar_margin=float(getattr(args, "bar_margin", 0.0)),
        limit_fill_model=str(getattr(args, "limit_fill_model", "binary")),
        touch_fill_fraction=float(getattr(args, "touch_fill_fraction", 0.0)),
        select_fillable_only=bool(getattr(args, "select_fillable_only", True)),
    )


__all__ = [
    "add_selector_realism_args",
    "build_selection_config_from_args",
]
