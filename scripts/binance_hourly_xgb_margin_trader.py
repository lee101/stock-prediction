#!/usr/bin/env python3
"""One-cycle Binance hourly XGB margin-pack trader.

This is the deploy surface for the current hourly portfolio-pack candidate.
It is deliberately dry-run by default. Live order placement requires:

* ``--execute``
* ``ALLOW_BINANCE_XGB_LIVE_TRADING=1``
* a clean Binance live-writer process audit
* fresh enough hourly data
* existing margin positions already covered by matching exit orders

For short entries, the entry order is a cross-margin ``SELL`` with
``AUTO_BORROW_REPAY``. A protective ``BUY``/``AUTO_REPAY`` order is placed only
after a short position exists; placing it before the short entry fills can create
an accidental long.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from binanceneural.execution import quantize_price, quantize_qty, resolve_symbol_rules
from scripts.binance_margin_exit_coverage import build_repair_plan, load_coverage, place_repair_orders
from scripts.sweep_binance_hourly_portfolio_pack import (
    PackConfig,
    _choose_end_timestamp,
    _discover_symbols,
    _filter_liquid_frames,
    _load_hourly_frames,
    build_actions_and_bars,
    build_model_frame,
    fit_forecasters,
    score_eval_rows,
)
from src.binan.binance_margin import cancel_margin_order, create_margin_order, get_margin_order
from src.binance_live_process_audit import audit_running_binance_live_processes
from src.hourly_trader_utils import EntryAllocationCandidate, allocate_concentrated_entry_budget


BEST_LABEL = "h12_short_corr168_08_aggr2777_dd1998_20260504"
MIN_TRADE_USDT = 12.0
STABLE_BASE_ASSETS = frozenset({"USDT", "FDUSD", "BUSD", "USDC", "USDP", "TUSD", "DAI", "AEUR"})
TERMINAL_ORDER_STATUSES = frozenset({"FILLED", "CANCELED", "REJECTED", "EXPIRED"})
LIVE_REFRESH_SYMBOLS = (
    "AAVE",
    "ADA",
    "APT",
    "ARB",
    "AVAX",
    "BCH",
    "BNB",
    "BONK",
    "BTC",
    "DOGE",
    "DOT",
    "ENA",
    "ETH",
    "FET",
    "FIL",
    "HBAR",
    "ICP",
    "LINK",
    "LTC",
    "NEAR",
    "ONDO",
    "PAXG",
    "PEPE",
    "POL",
    "RENDER",
    "SEI",
    "SHIB",
    "SOL",
    "SUI",
    "TAO",
    "TON",
    "TRUMP",
    "TRX",
    "UNI",
    "WLD",
    "XRP",
)


@dataclass(frozen=True)
class XGBLiveCandidate:
    symbol: str
    entry_price: float
    exit_price: float
    edge: float
    trade_amount: float
    current_price: float
    required_move_frac: float
    target_notional_usdt: float
    raw_qty: float
    quantized_qty: float
    quantized_entry_price: float
    quantized_exit_price: float
    notional_usdt: float
    skipped_reason: str = ""


@dataclass(frozen=True)
class XGBOrderPayload:
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float
    side_effect_type: str
    time_in_force: str
    kind: str


def best_pack_config() -> PackConfig:
    return PackConfig(
        risk_penalty=0.35,
        cvar_weight=0.5,
        entry_gap_bps=30.0,
        entry_alpha=0.15,
        exit_alpha=0.85,
        edge_threshold=0.002,
        edge_to_full_size=0.02,
        min_close_ret=0.001,
        close_edge_weight=0.25,
        min_upside_downside_ratio=0.0,
        min_recent_ret_24h=-1.0,
        min_recent_ret_72h=-1.0,
        max_recent_vol_72h=0.025,
        regime_cs_skew_min=-0.45,
        vol_target_ann=0.75,
        inv_vol_target_ann=2.5,
        inv_vol_floor=0.15,
        inv_vol_cap=1.8,
        max_positions=2,
        max_pending_entries=4,
        entry_ttl_hours=1,
        max_hold_hours=12,
        max_leverage=2.55,
        entry_selection_mode="first_trigger",
        entry_allocator_mode="concentrated",
        entry_allocator_edge_power=1.5,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _enforce_min_second_share(values: list[float], min_fraction: float) -> list[float]:
    if len(values) < 2:
        return values
    total = sum(max(0.0, float(v)) for v in values)
    if total <= 0.0:
        return values
    floor = min(max(float(min_fraction), 0.0), 0.5) * total
    ordered = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    second = ordered[1]
    if values[second] >= floor:
        return values
    needed = floor - values[second]
    first = ordered[0]
    take = min(needed, max(0.0, values[first] - floor))
    adjusted = list(values)
    adjusted[first] -= take
    adjusted[second] += take
    return adjusted


def _latest_common_timestamp(frames: dict[str, pd.DataFrame], min_symbols_per_hour: int) -> pd.Timestamp:
    return _choose_end_timestamp(frames, min(max(1, int(min_symbols_per_hour)), len(frames)))


def _filter_live_tradable_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    filtered: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        raw = str(symbol).upper().strip()
        base = raw[:-4] if raw.endswith("USDT") else raw
        if not base or base in STABLE_BASE_ASSETS:
            continue
        filtered[raw] = frame
    return filtered


def _load_and_score_latest(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    symbols = _discover_symbols(Path(args.hourly_root), str(args.symbols))
    frames = _load_hourly_frames(Path(args.hourly_root), symbols=symbols, min_bars=int(args.min_bars))
    frames = _filter_live_tradable_frames(frames)
    if not frames:
        raise RuntimeError("live tradable filter removed all symbols")
    end = _latest_common_timestamp(frames, int(args.min_symbols_per_hour))
    frames, liquidity = _filter_liquid_frames(
        frames,
        end=end,
        lookback_days=int(args.liquidity_lookback_days),
        min_median_dollar_volume=float(args.min_median_dollar_volume),
        max_symbols=int(args.max_symbols_by_dollar_volume),
    )
    frames = _filter_live_tradable_frames(frames)
    if not frames:
        raise RuntimeError("live tradable filter removed all symbols")
    liquidity = liquidity[liquidity["symbol"].astype(str).str.upper().isin(frames)].reset_index(drop=True)
    selected_symbols = sorted(frames)
    available_timestamps = pd.DatetimeIndex(
        sorted(pd.concat([df["timestamp"] for df in frames.values()], ignore_index=True).drop_duplicates())
    )
    available_timestamps = available_timestamps[available_timestamps <= end]
    decision_lag = max(0, int(args.decision_lag))
    if len(available_timestamps) <= decision_lag:
        raise RuntimeError(f"not enough hourly bars to apply decision_lag={decision_lag}")
    decision_ts = pd.Timestamp(available_timestamps[-1 - decision_lag]).tz_convert("UTC")
    train_end = end - pd.Timedelta(hours=int(args.label_horizon))
    train_start = train_end - pd.Timedelta(days=int(args.train_days))
    feature_start = train_start - pd.Timedelta(days=10)

    model_frame, feature_cols = build_model_frame(
        frames,
        start=feature_start,
        end=end,
        horizon=int(args.label_horizon),
        require_targets=False,
    )
    models = fit_forecasters(
        model_frame,
        feature_cols,
        train_end=train_end,
        rounds=int(args.rounds),
        device=str(args.device),
    )
    scored = score_eval_rows(
        model_frame,
        feature_cols,
        models,
        eval_start=decision_ts,
        eval_end=decision_ts,
    )
    cfg = best_pack_config()
    bars, actions = build_actions_and_bars(
        scored,
        cfg=cfg,
        label_horizon=int(args.label_horizon),
        min_take_profit_bps=float(args.min_take_profit_bps),
        max_entry_gap_bps=float(args.max_entry_gap_bps),
        max_exit_gap_bps=float(args.max_exit_gap_bps),
        fee_rate=float(args.fee_rate),
        top_candidates_per_hour=int(args.top_candidates_per_hour),
        top_candidates_include_inactive=False,
        entry_block_hours_utc=str(args.entry_block_hours_utc),
        side_mode="short",
    )
    actions = actions.merge(
        bars[["timestamp", "symbol", "close"]],
        on=["timestamp", "symbol"],
        how="left",
    )
    if not liquidity.empty:
        actions = actions.merge(
            liquidity[["symbol", "median_dollar_volume"]],
            on="symbol",
            how="left",
        )
    return end, decision_ts, bars, actions, frames, selected_symbols


def _pair_corr_ok(
    frames: dict[str, pd.DataFrame],
    *,
    decision_ts: pd.Timestamp,
    selected_symbols: list[str],
    candidate_symbol: str,
    window_bars: int,
    min_periods: int,
    max_signed_corr: float,
) -> bool:
    if not selected_symbols or int(window_bars) <= 0 or float(max_signed_corr) >= 1.0:
        return True
    series: dict[str, pd.Series] = {}
    for symbol in [*selected_symbols, candidate_symbol]:
        df = frames.get(symbol)
        if df is None or df.empty:
            return True
        close = (
            df[df["timestamp"] <= decision_ts]
            .set_index("timestamp")["close"]
            .astype(float)
            .pct_change(fill_method=None)
            .dropna()
            .tail(int(window_bars))
        )
        series[symbol] = close
    cand = series[candidate_symbol]
    for symbol in selected_symbols:
        aligned = pd.concat([series[symbol], cand], axis=1, join="inner").dropna()
        if len(aligned) < int(min_periods):
            continue
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if math.isfinite(corr) and corr > float(max_signed_corr):
            return False
    return True


def _current_price(symbol: str, fallback: float, *, use_live_price: bool) -> float:
    if use_live_price:
        try:
            from src.binan import binance_wrapper

            price = binance_wrapper.get_symbol_price(symbol)
            if price is not None and float(price) > 0.0:
                return float(price)
        except Exception:
            pass
    return float(fallback)


def _active_margin_symbols() -> set[str]:
    try:
        from src.binan.binance_margin import get_margin_account, get_open_margin_orders
    except Exception:
        return set()

    active: set[str] = set()
    try:
        account = get_margin_account()
        for entry in account.get("userAssets", []):
            asset = str(entry.get("asset") or "").upper().strip()
            if not asset or asset in {"USDT", "FDUSD", "BUSD", "USDC"}:
                continue
            try:
                net_qty = float(entry.get("netAsset", 0.0) or 0.0)
            except (TypeError, ValueError):
                net_qty = 0.0
            if abs(net_qty) > 1e-12:
                active.add(f"{asset}USDT")
    except Exception:
        pass

    try:
        for order in get_open_margin_orders():
            symbol = str(order.get("symbol") or "").upper().strip()
            if symbol:
                active.add(symbol)
    except Exception:
        pass
    return active


def build_order_payloads(candidate: XGBLiveCandidate) -> tuple[XGBOrderPayload, XGBOrderPayload]:
    """Return entry plus post-fill exit payloads for a short candidate."""

    return (
        XGBOrderPayload(
            symbol=candidate.symbol,
            side="SELL",
            order_type="LIMIT",
            quantity=float(candidate.quantized_qty),
            price=float(candidate.quantized_entry_price),
            side_effect_type="AUTO_BORROW_REPAY",
            time_in_force="GTC",
            kind="short_entry",
        ),
        XGBOrderPayload(
            symbol=candidate.symbol,
            side="BUY",
            order_type="LIMIT",
            quantity=float(candidate.quantized_qty),
            price=float(candidate.quantized_exit_price),
            side_effect_type="AUTO_REPAY",
            time_in_force="GTC",
            kind="short_exit_after_fill",
        ),
    )


def select_live_candidates(
    actions: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    *,
    decision_ts: pd.Timestamp,
    equity_usdt: float,
    risk_scale: float,
    use_live_prices: bool,
    active_symbols: set[str],
    max_total_entry_notional_usdt: float,
) -> list[XGBLiveCandidate]:
    cfg = best_pack_config()
    active = actions[
        (actions["sell_amount"].astype(float) > 0.0)
        & (actions["sell_price"].astype(float) > 0.0)
        & (actions["buy_price"].astype(float) > 0.0)
    ].copy()
    if active.empty:
        return []

    rows: list[dict[str, Any]] = []
    for row in active.itertuples(index=False):
        symbol = str(row.symbol).upper()
        if symbol in active_symbols:
            continue
        current = _current_price(symbol, float(getattr(row, "close", row.sell_price)), use_live_price=use_live_prices)
        required_move = max(0.0, (float(row.sell_price) * (1.0 + 0.0005) - current) / max(current, 1e-12))
        rows.append(
            {
                "symbol": symbol,
                "entry_price": float(row.sell_price),
                "exit_price": float(row.buy_price),
                "edge": float(row.xgb_edge),
                "trade_amount": float(row.sell_amount),
                "current_price": float(current),
                "required_move_frac": float(required_move),
            }
        )
    rows.sort(key=lambda item: (item["required_move_frac"], -item["edge"], item["symbol"]))

    selected: list[dict[str, Any]] = []
    for row in rows:
        if len(selected) >= int(cfg.max_positions):
            break
        if not _pair_corr_ok(
            frames,
            decision_ts=decision_ts,
            selected_symbols=[item["symbol"] for item in selected],
            candidate_symbol=str(row["symbol"]),
            window_bars=168,
            min_periods=48,
            max_signed_corr=0.8,
        ):
            continue
        selected.append(row)

    if len(selected) < 2:
        return []

    deployable_budget = max(0.0, float(equity_usdt)) * float(cfg.max_leverage) * max(0.0, float(risk_scale))
    cap = max(0.0, float(max_total_entry_notional_usdt))
    if cap > 0.0:
        deployable_budget = min(deployable_budget, cap)
    slot_budget = deployable_budget / max(int(cfg.max_positions), 1)
    allocations = allocate_concentrated_entry_budget(
        [
            EntryAllocationCandidate(
                symbol=str(row["symbol"]),
                edge=float(row["edge"]),
                intensity_fraction=min(max(float(row["trade_amount"]) / 100.0, 0.0), 1.0),
                slot_budget=float(slot_budget),
            )
            for row in selected
        ],
        max_positions=int(cfg.max_positions),
        deployable_budget=deployable_budget,
        edge_power=float(cfg.entry_allocator_edge_power),
        max_single_position_fraction=0.8,
    )
    allocations = _enforce_min_second_share(allocations, 0.2)

    candidates: list[XGBLiveCandidate] = []
    for row, target_notional in zip(selected, allocations):
        symbol = str(row["symbol"])
        entry_price = float(row["entry_price"])
        exit_price = float(row["exit_price"])
        raw_qty = max(0.0, float(target_notional) / max(entry_price, 1e-12))
        try:
            rules = resolve_symbol_rules(symbol)
            q_entry = quantize_price(entry_price, tick_size=rules.tick_size, side="sell")
            q_exit = quantize_price(exit_price, tick_size=rules.tick_size, side="buy")
            q_qty = quantize_qty(raw_qty, step_size=rules.step_size)
            min_qty = float(rules.min_qty or 0.0)
            min_notional = float(rules.min_notional or MIN_TRADE_USDT)
        except Exception:
            q_entry = entry_price
            q_exit = exit_price
            q_qty = raw_qty
            min_qty = 0.0
            min_notional = MIN_TRADE_USDT
        notional = q_qty * q_entry
        reason = ""
        if q_qty <= 0.0 or q_qty < min_qty:
            reason = "below_min_qty"
        elif notional < max(MIN_TRADE_USDT, min_notional):
            reason = "below_min_notional"
        candidates.append(
            XGBLiveCandidate(
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                edge=float(row["edge"]),
                trade_amount=float(row["trade_amount"]),
                current_price=float(row["current_price"]),
                required_move_frac=float(row["required_move_frac"]),
                target_notional_usdt=float(target_notional),
                raw_qty=float(raw_qty),
                quantized_qty=float(q_qty),
                quantized_entry_price=float(q_entry),
                quantized_exit_price=float(q_exit),
                notional_usdt=float(notional),
                skipped_reason=reason,
            )
        )
    return [candidate for candidate in candidates if not candidate.skipped_reason]


def _margin_equity_usdt(fallback: float) -> float:
    if fallback > 0.0:
        return float(fallback)
    try:
        from src.binan import binance_wrapper
        from src.binan.binance_margin import get_margin_account

        account = get_margin_account()
        total_net_btc = float(account.get("totalNetAssetOfBtc", 0.0) or 0.0)
        btc_price = float(binance_wrapper.get_symbol_price("BTCUSDT") or 0.0)
        if total_net_btc > 0.0 and btc_price > 0.0:
            return total_net_btc * btc_price
    except Exception:
        pass
    return 0.0


def _load_drawdown_state(path: Path, equity: float) -> tuple[dict[str, Any], float]:
    state = {}
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except Exception:
            state = {}
    high_water = max(float(state.get("equity_high_water_usdt", 0.0) or 0.0), float(equity))
    drawdown = max(0.0, (high_water - float(equity)) / max(high_water, 1e-12))
    start = 0.045
    full = 0.15
    floor = 0.32
    if drawdown <= start:
        scale = 1.0
    elif drawdown >= full:
        scale = floor
    else:
        frac = (drawdown - start) / max(full - start, 1e-12)
        scale = 1.0 - frac * (1.0 - floor)
    state.update(
        {
            "updated_at": datetime.now(UTC).isoformat(),
            "equity_high_water_usdt": high_water,
            "last_equity_usdt": float(equity),
            "drawdown": float(drawdown),
            "risk_scale": float(scale),
        }
    )
    return state, float(scale)


def _preflight_live(args: argparse.Namespace, *, end: pd.Timestamp) -> None:
    if os.getenv("ALLOW_BINANCE_XGB_LIVE_TRADING") != "1":
        raise RuntimeError("refusing live orders: set ALLOW_BINANCE_XGB_LIVE_TRADING=1")
    audit = audit_running_binance_live_processes(
        allowed_counts_by_kind={"hybrid": 0, "xgb_hourly_pack": 1}
    )
    if not audit.ok:
        raise RuntimeError(f"refusing live orders: {audit.reason}")
    age_hours = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(end)).total_seconds() / 3600.0
    if age_hours > float(args.max_data_staleness_hours):
        raise RuntimeError(
            f"refusing live orders: hourly data is stale ({age_hours:.2f}h > {args.max_data_staleness_hours}h)"
        )
    rows = load_coverage(min_value_usdt=MIN_TRADE_USDT)
    missing = [row for row in rows if row.status != "covered"]
    if missing and not bool(args.repair_existing_coverage):
        details = ", ".join(f"{row.asset}/{row.direction}/{row.status}" for row in missing)
        raise RuntimeError(f"refusing live entries: existing margin positions lack exit coverage: {details}")
    if missing:
        repair_plan = build_repair_plan(
            rows,
            target_markup_pct=float(args.coverage_repair_markup_pct),
            min_order_value_usdt=MIN_TRADE_USDT,
        )
        placed = place_repair_orders(
            repair_plan,
            side_effect_type="NO_SIDE_EFFECT",
            short_side_effect_type="AUTO_REPAY",
        )
        failed = [plan for plan in placed if plan.status == "failed"]
        if failed:
            details = ", ".join(f"{plan.asset}:{plan.reason}" for plan in failed)
            raise RuntimeError(f"exit-coverage repair failed: {details}")


def _safe_order_id(order: dict[str, Any]) -> int | None:
    try:
        return int(order.get("orderId") or order.get("order_id"))
    except (TypeError, ValueError):
        return None


def _safe_order_status(order: dict[str, Any]) -> str:
    return str(order.get("status") or "").upper().strip()


def _safe_executed_qty(order: dict[str, Any]) -> float:
    try:
        return max(0.0, float(order.get("executedQty", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _refresh_margin_order(symbol: str, entry_order: dict[str, Any]) -> dict[str, Any]:
    order_id = _safe_order_id(entry_order)
    if order_id is None:
        return dict(entry_order)
    try:
        refreshed = get_margin_order(symbol, order_id=order_id)
    except Exception:
        return dict(entry_order)
    return refreshed if isinstance(refreshed, dict) and refreshed else dict(entry_order)


def _place_short_exit_order(candidate: XGBLiveCandidate, quantity: float) -> dict[str, Any]:
    _, exit_after_fill = build_order_payloads(candidate)
    qty = min(max(0.0, float(quantity)), max(0.0, float(candidate.quantized_qty)))
    try:
        rules = resolve_symbol_rules(candidate.symbol)
        qty = quantize_qty(qty, step_size=rules.step_size)
        min_qty = float(rules.min_qty or 0.0)
        min_notional = float(rules.min_notional or MIN_TRADE_USDT)
    except Exception:
        min_qty = 0.0
        min_notional = MIN_TRADE_USDT
    notional = qty * float(exit_after_fill.price)
    if qty <= 0.0 or qty < min_qty:
        return {"status": "skipped", "reason": "below_min_qty", "quantity": qty, "price": exit_after_fill.price}
    if notional < max(MIN_TRADE_USDT, min_notional):
        return {
            "status": "skipped",
            "reason": "below_min_notional",
            "quantity": qty,
            "price": exit_after_fill.price,
            "notional_usdt": notional,
        }
    try:
        order = create_margin_order(
            exit_after_fill.symbol,
            exit_after_fill.side,
            exit_after_fill.order_type,
            qty,
            price=exit_after_fill.price,
            side_effect_type=exit_after_fill.side_effect_type,
            time_in_force=exit_after_fill.time_in_force,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "quantity": qty,
            "price": exit_after_fill.price,
            "notional_usdt": notional,
        }
    return {"status": "placed", "quantity": qty, "price": exit_after_fill.price, "notional_usdt": notional, "order": order}


def _sync_exit_for_filled_delta(record: dict[str, Any], candidate: XGBLiveCandidate) -> None:
    entry_status = record.get("entry_status") if isinstance(record.get("entry_status"), dict) else record.get("entry_order", {})
    executed_qty = min(float(candidate.quantized_qty), _safe_executed_qty(entry_status))
    exit_qty = sum(
        float(exit_order.get("quantity", 0.0) or 0.0)
        for exit_order in record.get("exit_orders", [])
        if exit_order.get("status") == "placed"
    )
    delta = executed_qty - exit_qty
    if delta <= 0.0:
        return
    record.setdefault("exit_orders", []).append(_place_short_exit_order(candidate, delta))


def _cancel_unfilled_entry(record: dict[str, Any], candidate: XGBLiveCandidate) -> None:
    entry_status = record.get("entry_status") if isinstance(record.get("entry_status"), dict) else record.get("entry_order", {})
    if _safe_order_status(entry_status) in TERMINAL_ORDER_STATUSES:
        return
    order_id = _safe_order_id(entry_status) or _safe_order_id(record.get("entry_order", {}))
    if order_id is None:
        record["cancel_order"] = {"status": "skipped", "reason": "missing_order_id"}
        return
    try:
        record["cancel_order"] = cancel_margin_order(candidate.symbol, order_id=order_id)
    except Exception as exc:
        record["cancel_order"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "orderId": order_id}
        return
    record["entry_status"] = _refresh_margin_order(candidate.symbol, record.get("entry_order", {}))
    _sync_exit_for_filled_delta(record, candidate)


def _place_entry_orders(
    candidates: list[XGBLiveCandidate],
    *,
    wait_seconds: float,
    poll_seconds: float,
    cancel_unfilled_entries: bool,
) -> list[dict[str, Any]]:
    """Place short entries, then cover any filled quantity with BUY exits."""

    placed: list[dict[str, Any]] = []
    for candidate in candidates:
        entry, _exit_after_fill = build_order_payloads(candidate)
        order = create_margin_order(
            entry.symbol,
            entry.side,
            entry.order_type,
            entry.quantity,
            price=entry.price,
            side_effect_type=entry.side_effect_type,
            time_in_force=entry.time_in_force,
        )
        record = {
            "candidate": asdict(candidate),
            "entry_order": order,
            "entry_status": order,
            "exit_orders": [],
            "cancel_order": None,
        }
        _sync_exit_for_filled_delta(record, candidate)
        placed.append(record)

    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    pending = {idx for idx, record in enumerate(placed) if _safe_order_status(record["entry_status"]) not in TERMINAL_ORDER_STATUSES}
    while pending and time.monotonic() <= deadline:
        for idx in list(pending):
            candidate = candidates[idx]
            record = placed[idx]
            record["entry_status"] = _refresh_margin_order(candidate.symbol, record["entry_order"])
            _sync_exit_for_filled_delta(record, candidate)
            if _safe_order_status(record["entry_status"]) in TERMINAL_ORDER_STATUSES:
                pending.discard(idx)
        if pending and time.monotonic() < deadline:
            time.sleep(max(0.1, float(poll_seconds)))

    if cancel_unfilled_entries:
        for idx in list(pending):
            _cancel_unfilled_entry(placed[idx], candidates[idx])
    else:
        for idx in list(pending):
            placed[idx]["left_open"] = True
    return placed


def _coverage_summary(rows: list[Any]) -> dict[str, Any]:
    return {
        "positions": len(rows),
        "covered": sum(1 for row in rows if row.status == "covered"),
        "partial": sum(1 for row in rows if row.status == "partial"),
        "missing": sum(1 for row in rows if row.status == "missing"),
        "rows": [asdict(row) for row in rows],
    }


def _load_post_execute_coverage(args: argparse.Namespace) -> list[Any]:
    settle_seconds = max(0.0, float(getattr(args, "post_cycle_settle_seconds", 0.0) or 0.0))
    if settle_seconds > 0.0:
        time.sleep(settle_seconds)
    return load_coverage(min_value_usdt=MIN_TRADE_USDT)


def _existing_margin_gross_usdt(rows: list[Any]) -> float:
    total = 0.0
    for row in rows:
        try:
            total += max(0.0, float(row.est_value_usdt))
        except (TypeError, ValueError):
            continue
    return float(total)


def _refresh_hourly_data(args: argparse.Namespace) -> int:
    symbols_text = str(getattr(args, "refresh_symbols", "") or getattr(args, "symbols", "") or "").strip()
    symbols = symbols_text.split() if symbols_text else list(LIVE_REFRESH_SYMBOLS)
    start_days = max(1, int(getattr(args, "refresh_start_days", 4)))
    start = (datetime.now(UTC) - timedelta(days=start_days)).date().isoformat()
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "download_binance_data.py"),
        "--hourly-only",
        "--hourly-dir",
        str(getattr(args, "hourly_root")),
        "--start",
        start,
        "--symbols",
        *symbols,
    ]
    completed = subprocess.run(cmd, cwd=REPO, check=False)
    if completed.returncode != 0:
        print(
            f"[{datetime.now(UTC).isoformat()}] hourly data refresh returned {completed.returncode}; "
            "continuing so the freshness gate can decide",
            file=sys.stderr,
        )
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Binance hourly XGB margin-pack cycle.")
    parser.add_argument("--hourly-root", type=Path, default=Path("binance_spot_hourly"))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--min-bars", type=int, default=5000)
    parser.add_argument("--min-symbols-per-hour", type=int, default=20)
    parser.add_argument("--liquidity-lookback-days", type=int, default=90)
    parser.add_argument("--min-median-dollar-volume", type=float, default=0.0)
    parser.add_argument("--max-symbols-by-dollar-volume", type=int, default=36)
    parser.add_argument("--train-days", type=int, default=720)
    parser.add_argument("--label-horizon", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=220)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--min-take-profit-bps", type=float, default=35.0)
    parser.add_argument("--max-entry-gap-bps", type=float, default=120.0)
    parser.add_argument("--max-exit-gap-bps", type=float, default=250.0)
    parser.add_argument("--top-candidates-per-hour", type=int, default=10)
    parser.add_argument("--entry-block-hours-utc", default="")
    parser.add_argument("--equity-usdt", type=float, default=0.0)
    parser.add_argument("--max-total-entry-notional-usdt", type=float, default=0.0)
    parser.add_argument("--state-path", type=Path, default=Path("strategy_state/binance_xgb_short_pack_state.json"))
    parser.add_argument("--json-out", type=Path, default=Path("analysis/binance_hourly_xgb_margin_plan_latest.json"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repair-existing-coverage", action="store_true")
    parser.add_argument("--coverage-repair-markup-pct", type=float, default=0.20)
    parser.add_argument("--max-data-staleness-hours", type=float, default=4.0)
    parser.add_argument("--entry-fill-wait-seconds", type=float, default=60.0)
    parser.add_argument("--entry-fill-poll-seconds", type=float, default=3.0)
    parser.add_argument("--post-cycle-settle-seconds", type=float, default=10.0)
    parser.add_argument("--leave-unfilled-entries-open", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--run-on-start", action="store_true")
    parser.add_argument("--cycle-minutes", type=float, default=60.0)
    parser.add_argument("--error-sleep-seconds", type=float, default=300.0)
    parser.add_argument("--refresh-data-before-cycle", action="store_true")
    parser.add_argument("--refresh-start-days", type=int, default=4)
    parser.add_argument("--refresh-symbols", default="")
    parser.add_argument("--use-live-prices", action="store_true", default=True)
    parser.add_argument("--no-live-prices", action="store_false", dest="use_live_prices")
    return parser


def run_once(args: argparse.Namespace) -> int:
    if bool(getattr(args, "refresh_data_before_cycle", False)):
        _refresh_hourly_data(args)
    end, decision_ts, _bars, actions, frames, selected_symbols = _load_and_score_latest(args)
    equity = _margin_equity_usdt(float(args.equity_usdt))
    if equity <= 0.0:
        raise RuntimeError("could not determine positive margin equity; pass --equity-usdt for dry-run")

    state, risk_scale = _load_drawdown_state(Path(args.state_path), equity)
    active_symbols = _active_margin_symbols() if bool(args.use_live_prices) else set()
    coverage_rows = load_coverage(min_value_usdt=MIN_TRADE_USDT) if bool(args.use_live_prices) else []
    existing_gross = _existing_margin_gross_usdt(coverage_rows)
    configured_entry_cap = max(0.0, float(args.max_total_entry_notional_usdt))
    live_gross_cap = max(0.0, float(equity)) * float(best_pack_config().max_leverage) * max(0.0, float(risk_scale))
    remaining_entry_cap = max(0.0, live_gross_cap - existing_gross) if bool(args.use_live_prices) else configured_entry_cap
    if configured_entry_cap > 0.0:
        remaining_entry_cap = min(remaining_entry_cap, configured_entry_cap)
    if bool(args.use_live_prices) and remaining_entry_cap < MIN_TRADE_USDT:
        candidates = []
    else:
        candidates = select_live_candidates(
            actions,
            frames,
            decision_ts=decision_ts,
            equity_usdt=equity,
            risk_scale=risk_scale,
            use_live_prices=bool(args.use_live_prices),
            active_symbols=active_symbols,
            max_total_entry_notional_usdt=remaining_entry_cap,
        )
    payloads = [build_order_payloads(candidate) for candidate in candidates]
    payload: dict[str, Any] = {
        "label": BEST_LABEL,
        "mode": "live" if args.execute else "dry_run",
        "generated_at": datetime.now(UTC),
        "data_end": end,
        "decision_timestamp": decision_ts,
        "data_age_hours": (pd.Timestamp.now(tz="UTC") - end).total_seconds() / 3600.0,
        "selected_symbols": selected_symbols,
        "config": asdict(best_pack_config()),
        "equity_usdt": float(equity),
        "drawdown_state": state,
        "existing_margin_gross_usdt": float(existing_gross),
        "max_total_entry_notional_usdt": float(remaining_entry_cap),
        "active_margin_symbols": sorted(active_symbols),
        "candidates": [asdict(candidate) for candidate in candidates],
        "orders": [
            {"entry": asdict(entry), "exit_after_fill": asdict(exit_after_fill)}
            for entry, exit_after_fill in payloads
        ],
        "exit_coverage": [asdict(row) for row in coverage_rows],
        "post_execute_exit_coverage": None,
        "placed": [],
    }

    if args.execute:
        _preflight_live(args, end=end)
        payload["placed"] = _place_entry_orders(
            candidates,
            wait_seconds=float(args.entry_fill_wait_seconds),
            poll_seconds=float(args.entry_fill_poll_seconds),
            cancel_unfilled_entries=not bool(args.leave_unfilled_entries_open),
        )
        post_rows = _load_post_execute_coverage(args)
        payload["post_execute_exit_coverage"] = _coverage_summary(post_rows)
        missing = [row for row in post_rows if row.status != "covered"]
        if missing:
            details = ", ".join(f"{row.asset}/{row.direction}/{row.status}" for row in missing)
            payload["post_execute_error"] = f"post-entry exit coverage failed: {details}"
            _write_json(Path(args.json_out), payload)
            raise RuntimeError(str(payload["post_execute_error"]))
        Path(args.state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.state_path).write_text(json.dumps(_json_safe(state), indent=2, sort_keys=True) + "\n")

    _write_json(Path(args.json_out), payload)
    print(
        f"xgb-hourly-pack {payload['mode']} label={BEST_LABEL} "
        f"data_end={end.isoformat()} decision_ts={decision_ts.isoformat()} "
        f"equity={equity:.2f} risk_scale={risk_scale:.3f} candidates={len(candidates)} "
        f"json={args.json_out}"
    )
    for candidate in candidates:
        entry, exit_after_fill = build_order_payloads(candidate)
        print(
            f"  {candidate.symbol}: SELL {entry.quantity:.8g} @ {entry.price:.8g} "
            f"notional=${candidate.notional_usdt:.2f}; post-fill BUY exit @ {exit_after_fill.price:.8g}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not bool(args.daemon):
        return run_once(args)

    if not bool(args.run_on_start):
        time.sleep(max(1.0, float(args.cycle_minutes) * 60.0))
    while True:
        try:
            run_once(args)
            sleep_seconds = max(1.0, float(args.cycle_minutes) * 60.0)
        except Exception as exc:
            print(
                f"[{datetime.now(UTC).isoformat()}] xgb-hourly-pack cycle failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            sleep_seconds = max(1.0, float(args.error_sleep_seconds))
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
