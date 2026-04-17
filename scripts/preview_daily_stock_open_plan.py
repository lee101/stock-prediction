#!/usr/bin/env python3
"""Preview what the daily stock production bot would do at market open.

This is intentionally planning-only. It reuses the real `run_once` logic in
dry-run mode, optionally forces the market-open branch, and prints a compact
"paper trading style" summary without claiming any writer lease or submitting
orders.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import trade_daily_stock_prod as daily_stock  # noqa: E402


DEFAULT_LIVE_SERVER_ACCOUNT = "live_prod"
DEFAULT_LIVE_SERVER_BOT_ID = "daily_stock_sortino_v1"


def _upper_or_none(value: object) -> str | None:
    text = str(value or "").strip().upper()
    return text or None


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


@contextmanager
def _suppress_logs(enabled: bool):
    if not enabled:
        yield
        return
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable)


def _build_runtime_config(args: argparse.Namespace) -> daily_stock.CliRuntimeConfig:
    return daily_stock.CliRuntimeConfig(
        paper=bool(args.paper),
        symbols=[str(symbol).upper() for symbol in args.symbols],
        checkpoint=str(args.checkpoint),
        extra_checkpoints=None if args.no_ensemble else list(args.extra_checkpoints),
        data_dir=str(args.data_dir),
        data_source="alpaca",
        allocation_pct=float(args.allocation_pct),
        execution_backend="trading_server",
        server_account=str(args.server_account),
        server_bot_id=str(args.server_bot_id),
        server_url=str(args.server_url) if args.server_url else None,
        dry_run=True,
        backtest=False,
        backtest_days=60,
        backtest_starting_cash=float(daily_stock.DEFAULT_BACKTEST_STARTING_CASH),
        daemon=False,
        compare_server_parity=False,
        allocation_sizing_mode=str(args.allocation_sizing_mode),
        multi_position=int(args.multi_position),
        multi_position_min_prob_ratio=float(args.multi_position_min_prob_ratio),
        min_open_confidence=float(args.min_open_confidence),
        min_open_value_estimate=float(args.min_open_value_estimate),
        print_payload=False,
        allow_unsafe_checkpoint_loading=bool(args.allow_unsafe_checkpoint_loading),
        meta_selector=bool(args.meta_selector),
        meta_top_k=int(args.meta_top_k),
        meta_lookback=int(args.meta_lookback),
    )


def _load_live_state() -> daily_stock.StrategyState:
    return daily_stock.load_state(daily_stock.STATE_PATH)


def build_preview_result(
    payload: Mapping[str, object],
    *,
    state_active_symbol: str | None,
    state_pending_close_symbol: str | None,
    allocation_pct: float,
    allocation_sizing_mode: str,
    min_open_confidence: float,
    forced_market_open: bool,
    server_account: str,
    server_bot_id: str,
) -> dict[str, object]:
    snapshot = payload.get("server_snapshot")
    snapshot_map = snapshot if isinstance(snapshot, Mapping) else {}
    positions_raw = snapshot_map.get("positions")
    positions = positions_raw if isinstance(positions_raw, Mapping) else {}
    quotes_raw = payload.get("quotes")
    quotes = quotes_raw if isinstance(quotes_raw, Mapping) else {}

    desired_symbol = (
        _upper_or_none(payload.get("symbol"))
        if str(payload.get("direction") or "").strip().lower() == "long"
        else None
    )
    managed_symbol = _upper_or_none(state_active_symbol)
    managed_position_raw = positions.get(managed_symbol) if managed_symbol else None
    managed_position = managed_position_raw if isinstance(managed_position_raw, Mapping) else None

    would_submit = bool(payload.get("execution_would_submit"))
    execution_status = str(payload.get("execution_status") or "")
    skip_reason = str(payload.get("execution_skip_reason") or "") or None
    allow_open_reason = str(payload.get("allow_open_reason") or "") or None

    open_leg: dict[str, object] | None = None
    close_leg: dict[str, object] | None = None
    summary: str

    if managed_position is not None and (desired_symbol is None or desired_symbol != managed_symbol):
        close_qty = abs(_coerce_float(managed_position.get("qty")))
        close_price = _coerce_float(
            quotes.get(managed_symbol),
            _coerce_float(managed_position.get("current_price"), _coerce_float(managed_position.get("avg_entry_price"))),
        )
        close_leg = {
            "symbol": managed_symbol,
            "qty": close_qty,
            "ref_price": close_price,
        }

    if desired_symbol:
        signal_like = SimpleNamespace(
            confidence=_coerce_float(payload.get("confidence"), 0.0),
            allocation_pct=_coerce_float(payload.get("allocation_fraction"), 0.0),
        )
        price = _coerce_float(quotes.get(desired_symbol), 0.0)
        effective_allocation_pct = daily_stock.resolved_signal_allocation_pct(
            signal_like,
            base_allocation_pct=float(allocation_pct),
            sizing_mode=str(allocation_sizing_mode),
            min_open_confidence=float(min_open_confidence),
        )
        qty = (
            daily_stock.compute_target_qty_from_server_snapshot(
                snapshot=snapshot_map,
                quotes={str(key).upper(): _coerce_float(value) for key, value in quotes.items()},
                price=price,
                allocation_pct=effective_allocation_pct,
            )
            if snapshot_map and price > 0.0
            else 0.0
        )
        open_leg = {
            "symbol": desired_symbol,
            "qty": qty,
            "ref_price": price,
            "limit_price": daily_stock._marketable_limit_price(price, "buy") if price > 0.0 else 0.0,
            "effective_allocation_pct": effective_allocation_pct,
        }

    if would_submit:
        if close_leg and open_leg:
            summary = f"Would rotate {close_leg['symbol']} -> {open_leg['symbol']}"
        elif close_leg:
            summary = f"Would close {close_leg['symbol']}"
        elif open_leg:
            summary = f"Would open {open_leg['symbol']}"
        else:
            summary = "Would submit an order"
    elif execution_status == "no_action_flat_signal":
        summary = "No order; signal is flat"
    elif execution_status == "blocked_open_gate":
        symbol_text = desired_symbol or "signal"
        summary = f"No order; blocked from opening {symbol_text}"
    elif execution_status == "skipped_market_closed":
        summary = "No order; market is closed"
    elif execution_status == "skipped_stale_bars":
        summary = "No order; inference bars are stale"
    elif execution_status == "no_action_executor_declined" and managed_symbol and desired_symbol == managed_symbol:
        summary = f"No order; keep holding {managed_symbol}"
    else:
        summary = "No order"

    positions_view = [
        {
            "symbol": str(symbol).upper(),
            "qty": _coerce_float(position.get("qty")),
            "avg_entry_price": _coerce_float(position.get("avg_entry_price")),
            "current_price": _coerce_float(position.get("current_price")),
        }
        for symbol, position in sorted(positions.items())
        if isinstance(position, Mapping)
    ]

    return {
        "preview_mode": "forced_market_open" if forced_market_open else "respect_clock",
        "server_account": server_account,
        "server_bot_id": server_bot_id,
        "signal": {
            "action": payload.get("action"),
            "symbol": payload.get("symbol"),
            "direction": payload.get("direction"),
            "confidence": _coerce_float(payload.get("confidence")),
            "value_estimate": _coerce_float(payload.get("value_estimate")),
        },
        "bars": {
            "fresh": bool(payload.get("bars_fresh")),
            "latest_bar_timestamp": payload.get("latest_bar_timestamp"),
            "market_open": bool(payload.get("market_open")),
            "bar_data_source": payload.get("bar_data_source"),
            "quote_data_source": payload.get("quote_data_source"),
        },
        "account": {
            "cash": _coerce_float(snapshot_map.get("cash")),
            "equity": _coerce_float(snapshot_map.get("equity")),
            "buying_power": _coerce_float(snapshot_map.get("buying_power")),
            "position_count": len(positions_view),
        },
        "state": {
            "active_symbol": managed_symbol,
            "pending_close_symbol": _upper_or_none(state_pending_close_symbol),
        },
        "positions": positions_view,
        "plan": {
            "summary": summary,
            "would_submit": would_submit,
            "execution_status": execution_status,
            "execution_skip_reason": skip_reason,
            "allow_open_reason": allow_open_reason,
            "close_leg": close_leg,
            "open_leg": open_leg,
        },
        "payload": dict(payload),
    }


def format_preview_text(result: Mapping[str, object]) -> str:
    signal = result["signal"] if isinstance(result.get("signal"), Mapping) else {}
    bars = result["bars"] if isinstance(result.get("bars"), Mapping) else {}
    account = result["account"] if isinstance(result.get("account"), Mapping) else {}
    state = result["state"] if isinstance(result.get("state"), Mapping) else {}
    plan = result["plan"] if isinstance(result.get("plan"), Mapping) else {}
    positions = result.get("positions")
    positions_list = positions if isinstance(positions, list) else []

    lines = [
        "Daily Stock Open Preview",
        f"mode: dry-run via trading_server on {result.get('server_account')} / {result.get('server_bot_id')}",
        f"clock: {'forced open preview' if result.get('preview_mode') == 'forced_market_open' else 'actual Alpaca clock'}",
        f"plan: {plan.get('summary')}",
        (
            "signal: "
            f"{signal.get('action')} "
            f"symbol={signal.get('symbol') or 'N/A'} "
            f"confidence={_coerce_float(signal.get('confidence')) * 100.0:.1f}% "
            f"value_est={_coerce_float(signal.get('value_estimate')):.4f}"
        ),
        (
            "bars: "
            f"fresh={bool(bars.get('fresh'))} "
            f"latest={bars.get('latest_bar_timestamp') or 'n/a'} "
            f"market_open={bool(bars.get('market_open'))}"
        ),
        (
            "account: "
            f"equity=${_coerce_float(account.get('equity')):,.2f} "
            f"cash=${_coerce_float(account.get('cash')):,.2f} "
            f"buying_power=${_coerce_float(account.get('buying_power')):,.2f}"
        ),
        (
            "state: "
            f"active_symbol={state.get('active_symbol') or 'none'} "
            f"pending_close={state.get('pending_close_symbol') or 'none'}"
        ),
    ]

    if positions_list:
        positions_text = ", ".join(
            (
                f"{position.get('symbol')} qty={_coerce_float(position.get('qty')):.4f} "
                f"avg={_coerce_float(position.get('avg_entry_price')):.4f} "
                f"last={_coerce_float(position.get('current_price')):.4f}"
            )
            for position in positions_list
            if isinstance(position, Mapping)
        )
        lines.append(f"positions: {positions_text}")
    else:
        lines.append("positions: none")

    close_leg = plan.get("close_leg")
    if isinstance(close_leg, Mapping):
        lines.append(
            "close leg: "
            f"{close_leg.get('symbol')} qty={_coerce_float(close_leg.get('qty')):.4f} "
            f"ref_price={_coerce_float(close_leg.get('ref_price')):.4f}"
        )

    open_leg = plan.get("open_leg")
    if isinstance(open_leg, Mapping):
        lines.append(
            "open leg: "
            f"{open_leg.get('symbol')} qty={_coerce_float(open_leg.get('qty')):.4f} "
            f"ref_price={_coerce_float(open_leg.get('ref_price')):.4f} "
            f"limit={_coerce_float(open_leg.get('limit_price')):.4f} "
            f"alloc={_coerce_float(open_leg.get('effective_allocation_pct')):.2f}%"
        )

    if str(plan.get("allow_open_reason") or "").strip():
        lines.append(f"gate: {plan.get('allow_open_reason')}")
    elif str(plan.get("execution_skip_reason") or "").strip():
        lines.append(f"reason: {plan.get('execution_skip_reason')}")

    lines.append("safety: dry-run only, no writer lease claimed, no order submitted")
    return "\n".join(lines)


def _run_preview(config: daily_stock.CliRuntimeConfig, *, force_market_open: bool, quiet: bool) -> dict[str, object]:
    payload = daily_stock._preflight_config_payload(
        config,
        include_checkpoint_load_diagnostics=False,
    )
    if not payload["ready"]:
        raise RuntimeError(daily_stock._format_runtime_preflight_failure(payload))

    state = _load_live_state()

    run_kwargs = dict(
        checkpoint=config.checkpoint,
        symbols=config.symbols,
        paper=config.paper,
        allocation_pct=config.allocation_pct,
        allocation_sizing_mode=config.allocation_sizing_mode,
        dry_run=True,
        data_source=config.data_source,
        data_dir=config.data_dir,
        extra_checkpoints=config.extra_checkpoints,
        execution_backend=config.execution_backend,
        server_account=config.server_account,
        server_bot_id=config.server_bot_id,
        server_url=config.server_url,
        multi_position=config.multi_position,
        multi_position_min_prob_ratio=config.multi_position_min_prob_ratio,
        min_open_confidence=config.min_open_confidence,
        min_open_value_estimate=config.min_open_value_estimate,
        allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
        meta_selector=config.meta_selector,
        meta_top_k=config.meta_top_k,
        meta_lookback=config.meta_lookback,
    )

    with _suppress_logs(quiet):
        if force_market_open:
            class _OpenClockClient:
                def get_clock(self):
                    return SimpleNamespace(is_open=True)

            with patch.object(daily_stock, "build_trading_client", return_value=_OpenClockClient()):
                payload = daily_stock.run_once(**run_kwargs)
        else:
            payload = daily_stock.run_once(**run_kwargs)

    return build_preview_result(
        payload,
        state_active_symbol=state.active_symbol,
        state_pending_close_symbol=state.pending_close_symbol,
        allocation_pct=config.allocation_pct,
        allocation_sizing_mode=config.allocation_sizing_mode,
        min_open_confidence=config.min_open_confidence,
        forced_market_open=force_market_open,
        server_account=config.server_account,
        server_bot_id=config.server_bot_id,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the daily stock live bot's market-open plan.")
    parser.add_argument("--paper", action="store_true", help="Use paper account mode for the preview.")
    parser.add_argument("--checkpoint", default=daily_stock.DEFAULT_CHECKPOINT)
    parser.add_argument("--extra-checkpoints", nargs="*", default=list(daily_stock.DEFAULT_EXTRA_CHECKPOINTS))
    parser.add_argument("--no-ensemble", action="store_true", help="Disable the extra ensemble checkpoints.")
    parser.add_argument("--data-dir", default=daily_stock.DEFAULT_DATA_DIR)
    parser.add_argument("--allocation-pct", type=float, default=12.5)
    parser.add_argument(
        "--allocation-sizing-mode",
        choices=["static", "confidence_scaled"],
        default=daily_stock.DEFAULT_ALLOCATION_SIZING_MODE,
    )
    parser.add_argument("--symbols", nargs="*", default=list(daily_stock.DEFAULT_SYMBOLS))
    parser.add_argument("--server-account", default=DEFAULT_LIVE_SERVER_ACCOUNT)
    parser.add_argument("--server-bot-id", default=DEFAULT_LIVE_SERVER_BOT_ID)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--multi-position", type=int, default=daily_stock.DEFAULT_MULTI_POSITION)
    parser.add_argument(
        "--multi-position-min-prob-ratio",
        type=float,
        default=daily_stock.DEFAULT_MULTI_POSITION_MIN_PROB_RATIO,
    )
    parser.add_argument(
        "--min-open-confidence",
        type=float,
        default=daily_stock.DEFAULT_MIN_OPEN_CONFIDENCE,
    )
    parser.add_argument(
        "--min-open-value-estimate",
        type=float,
        default=daily_stock.DEFAULT_MIN_OPEN_VALUE_ESTIMATE,
    )
    parser.add_argument("--allow-unsafe-checkpoint-loading", action="store_true")
    parser.add_argument("--meta-selector", action="store_true")
    parser.add_argument("--meta-top-k", type=int, default=1)
    parser.add_argument("--meta-lookback", type=int, default=3)
    parser.add_argument(
        "--respect-clock",
        action="store_true",
        help="Use the real Alpaca market clock instead of forcing the market-open branch.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the text summary.")
    parser.add_argument("--verbose", action="store_true", help="Keep the underlying strategy logs visible.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _build_runtime_config(args)
    try:
        result = _run_preview(
            config,
            force_market_open=not bool(args.respect_clock),
            quiet=not bool(args.verbose),
        )
    except Exception as exc:
        print(f"daily stock open preview failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_preview_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
