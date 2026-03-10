#!/usr/bin/env python3
"""Meta-switcher margin bot: switches between two configured symbols using trailing meta scores."""
from __future__ import annotations

import argparse, json, os, random, sys, time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binan import binance_wrapper
from src.binan.binance_margin import (
    cancel_all_margin_orders,
    cancel_margin_order,
    create_margin_order,
    get_margin_account,
    get_margin_asset_balance,
    get_max_borrowable,
    get_margin_borrowed_balance,
    get_margin_free_balance,
    get_open_margin_orders,
    margin_repay_all,
)
from src.price_guard import enforce_gap
from src.process_utils import enforce_min_spread
from src.margin_position_utils import (
    choose_flat_entry_side,
    directional_signal,
    position_notional,
    position_side_from_qty,
    remaining_entry_notional,
)
from src.forecast_horizon_utils import resolve_required_forecast_horizons

from binanceneural.execution import resolve_symbol_rules, quantize_qty, quantize_price
from binanceneural.inference import generate_latest_action
from binanceneural.trade_binance_hourly import _ensure_valid_levels

from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from unified_hourly_experiment.meta_selector import score_trailing_returns

TAG = "margin-meta"
STATE_FILE = Path("strategy_state/margin_meta_state.json")
RUNTIME_STATE_FILE = Path("strategy_state/margin_meta_runtime.json")
RUNTIME_STATE_VERSION = 1
MIN_POSITION_NOTIONAL = 5.0
_LOG_DIR = Path("strategy_state/margin_logs")
FORECAST_CONTEXT_HOURS = 512
FORECAST_QUANTILES = (0.1, 0.5, 0.9)
FORECAST_BATCH_SIZE = 32
FORECAST_STALE_TOLERANCE_HOURS = 1
FORECAST_REBUILD_LOOKBACK_HOURS = 96
_FORECAST_REBUILD_GUARD: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}
ORDER_REPRICE_THRESHOLD = 0.003
LIVE_PROBE_MAX_POSITION_NOTIONAL = 5.10
PROBE_POSITION_NOTIONAL_RATIO = 0.80
DETERMINISTIC_SEED = 42
SIGNAL_BOOTSTRAP_EXTRA_HOURS = 2

# Model configs
DEFAULT_MODELS = {
    "doge": {
        "symbol": "DOGEUSDT",
        "data_symbol": "DOGEUSD",
        "base_asset": "DOGE",
        "maker_fee": 0.001,
    },
    "aave": {
        "symbol": "AAVEUSDT",
        "data_symbol": "AAVEUSD",
        "base_asset": "AAVE",
        "maker_fee": 0.001,
    },
}
MODEL_SLOT_DEFAULTS = (
    ("model_a", "doge", DEFAULT_MODELS["doge"]),
    ("model_b", "aave", DEFAULT_MODELS["aave"]),
)
MODELS = {name: dict(cfg) for name, cfg in DEFAULT_MODELS.items()}

SUPPORTED_SELECTION_METRICS = (
    "return",
    "sortino",
    "sharpe",
    "calmar",
    "omega",
    "gain_pain",
    "p10",
    "median",
)
SUPPORTED_SELECTION_MODES = ("winner", "winner_cash")
SUPPORTED_PROFIT_GATE_MODES = ("hypothetical", "live_like")
PROFIT_GATE_BAR_INTERVAL = pd.Timedelta(minutes=5)


def _normalize_model_name(value: str, *, fallback: str) -> str:
    name = str(value or fallback).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")
    if not name:
        raise ValueError("Model name must not be empty.")
    return name


def build_model_specs_from_args(args) -> list[dict]:
    specs: list[dict] = []
    seen_names: set[str] = set()
    seen_symbols: set[str] = set()
    for slot, fallback_name, defaults in MODEL_SLOT_DEFAULTS:
        name = _normalize_model_name(getattr(args, f"{slot}_name", fallback_name), fallback=fallback_name)
        symbol = str(getattr(args, f"{slot}_symbol", defaults["symbol"]) or "").strip().upper()
        data_symbol = str(getattr(args, f"{slot}_data_symbol", defaults["data_symbol"]) or "").strip().upper()
        base_asset = str(getattr(args, f"{slot}_base_asset", defaults["base_asset"]) or "").strip().upper()
        maker_fee = float(getattr(args, f"{slot}_maker_fee", defaults["maker_fee"]))
        checkpoint_raw = getattr(args, f"{slot}_checkpoint", None)
        checkpoint = None if checkpoint_raw in (None, "") else Path(checkpoint_raw)
        if not symbol:
            raise ValueError(f"{slot} symbol must not be empty.")
        if not data_symbol:
            raise ValueError(f"{slot} data symbol must not be empty.")
        if not base_asset:
            raise ValueError(f"{slot} base asset must not be empty.")
        if name in seen_names:
            raise ValueError(f"Model names must be unique, got duplicate '{name}'.")
        if symbol in seen_symbols:
            raise ValueError(f"Model symbols must be unique, got duplicate '{symbol}'.")
        seen_names.add(name)
        seen_symbols.add(symbol)
        spec = {
            "slot": slot,
            "name": name,
            "symbol": symbol,
            "data_symbol": data_symbol,
            "base_asset": base_asset,
            "maker_fee": maker_fee,
            "checkpoint": checkpoint,
        }
        specs.append(spec)
    return specs


def build_model_configs_from_args(args) -> dict[str, dict]:
    return {
        spec["name"]: {
            "symbol": spec["symbol"],
            "data_symbol": spec["data_symbol"],
            "base_asset": spec["base_asset"],
            "maker_fee": float(spec["maker_fee"]),
        }
        for spec in build_model_specs_from_args(args)
    }


def apply_model_specs(model_specs: list[dict]) -> dict[str, dict]:
    global MODELS
    MODELS = {
        spec["name"]: {
            "symbol": spec["symbol"],
            "data_symbol": spec["data_symbol"],
            "base_asset": spec["base_asset"],
            "maker_fee": float(spec["maker_fee"]),
        }
        for spec in model_specs
    }
    return MODELS


def _log_event(event_type: str, **data):
    argv0 = Path(sys.argv[0]).name.lower()
    if (
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("PYTEST_VERSION")
        or os.environ.get("MARGIN_META_DISABLE_LOG") == "1"
        or "pytest" in argv0
    ):
        return
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = _LOG_DIR / f"{TAG}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type, **data}
        with open(log_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


@dataclass
class MetaState:
    active_model: str = ""
    in_position: bool = False
    position_side: str = ""
    open_ts: Optional[str] = None
    open_price: float = 0.0

    def hours_held(self) -> float:
        if not self.open_ts:
            return 0.0
        try:
            opened = datetime.fromisoformat(self.open_ts)
        except Exception:
            return 0.0
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        held = (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0
        return max(0.0, held)

    def save(self, path: Path = STATE_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "active_model": self.active_model,
            "in_position": self.in_position,
            "position_side": self.position_side,
            "open_ts": self.open_ts,
            "open_price": self.open_price,
        }))

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> MetaState:
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text())
            return cls(
                active_model=d.get("active_model", ""),
                in_position=d.get("in_position", False),
                position_side=str(d.get("position_side", "") or ""),
                open_ts=d.get("open_ts"),
                open_price=d.get("open_price", 0.0),
            )
        except Exception:
            return cls()


@dataclass
class SignalHistory:
    """Track hypothetical signals for calmar computation."""
    timestamps: list = field(default_factory=list)
    buy_prices: list = field(default_factory=list)
    sell_prices: list = field(default_factory=list)
    buy_amounts: list = field(default_factory=list)
    sell_amounts: list = field(default_factory=list)
    closes: list = field(default_factory=list)
    equities: list = field(default_factory=list)
    max_entries: int = 500

    def add(self, ts, buy_p, sell_p, buy_a, sell_a, close, equity):
        self.timestamps.append(ts)
        self.buy_prices.append(buy_p)
        self.sell_prices.append(sell_p)
        self.buy_amounts.append(buy_a)
        self.sell_amounts.append(sell_a)
        self.closes.append(close)
        self.equities.append(equity)
        if len(self.timestamps) > self.max_entries:
            for lst in [self.timestamps, self.buy_prices, self.sell_prices,
                        self.buy_amounts, self.sell_amounts, self.closes, self.equities]:
                del lst[0]

    def trailing_calmar(self, lookback: int = 12) -> float:
        if len(self.equities) < 3:
            return 0.0
        eq = np.array(self.equities[-lookback:])
        if len(eq) < 3:
            return 0.0
        ret = eq[-1] / eq[0] - 1
        peak = np.maximum.accumulate(eq)
        dd = ((eq - peak) / (peak + 1e-10)).min()
        return ret / (abs(dd) + 1e-10) if abs(dd) > 1e-10 else ret * 100

    def trailing_sortino(self, lookback: int = 12) -> float:
        if len(self.equities) < 3:
            return 0.0
        eq = np.array(self.equities[-lookback:])
        if len(eq) < 3:
            return 0.0
        rets = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
        neg = rets[rets < 0]
        dd = np.std(neg) if len(neg) > 1 else 1e-10
        return float(np.mean(rets)) / (dd + 1e-10)


def _normalize_profit_gate_mode(value: str) -> str:
    mode = str(value or "hypothetical").strip().lower()
    if mode not in SUPPORTED_PROFIT_GATE_MODES:
        raise ValueError(
            f"Unsupported profit gate mode '{value}'. Expected one of {SUPPORTED_PROFIT_GATE_MODES}."
        )
    return mode


def _normalize_utc_timestamp(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _signal_history_to_hourly_signals(history: SignalHistory, *, symbol: str = "") -> dict[pd.Timestamp, dict]:
    signals: dict[pd.Timestamp, dict] = {}
    size = min(
        len(history.timestamps),
        len(history.buy_prices),
        len(history.sell_prices),
        len(history.buy_amounts),
        len(history.sell_amounts),
        len(history.closes),
    )
    for idx in range(size):
        signal_hour = _normalize_hour_like(history.timestamps[idx])
        if signal_hour is None:
            continue
        signals[signal_hour] = {
            "symbol": symbol,
            "buy_price": float(history.buy_prices[idx]),
            "sell_price": float(history.sell_prices[idx]),
            "buy_amount": float(history.buy_amounts[idx]),
            "sell_amount": float(history.sell_amounts[idx]),
            "close": float(history.closes[idx]),
            "signal_hour": signal_hour,
        }
    return signals


def _resolve_profit_gate_5m_root(args=None) -> Path:
    candidates: list[Path] = []
    preferred_roots: list[Path] = []
    explicit_root = getattr(args, "profit_gate_5m_root", None) if args is not None else None
    if explicit_root:
        root = Path(explicit_root).expanduser()
        preferred_roots.append(root)
        candidates.append(root)
    env_root = os.environ.get("BINANCE_5M_DATA_ROOT") or os.environ.get("TRAININGDATA5MIN_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        preferred_roots.append(root)
        candidates.append(root)
    data_root = getattr(args, "data_root", None) if args is not None else None
    if data_root:
        try:
            resolved_data_root = Path(data_root).expanduser().resolve()
        except Exception:
            resolved_data_root = Path(data_root).expanduser()
        root = resolved_data_root.parent / "trainingdata5min"
        preferred_roots.append(root)
        candidates.append(root)
    candidates.append(REPO_ROOT / "trainingdata5min")
    candidates.append(Path.cwd() / "trainingdata5min")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)

    for candidate in deduped:
        if candidate in preferred_roots:
            return candidate
    for candidate in deduped:
        if candidate.exists():
            return candidate
    return deduped[0] if deduped else (REPO_ROOT / "trainingdata5min")


def _normalize_profit_gate_5m_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame()
    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"])
    if normalized.empty:
        return pd.DataFrame()
    normalized = normalized.sort_values("timestamp")
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if "symbol" not in normalized.columns:
        normalized["symbol"] = str(symbol).upper()
    return normalized


def _expected_profit_gate_5m_bars(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> int:
    if start_ts > end_ts:
        return 0
    return int((end_ts - start_ts) / PROFIT_GATE_BAR_INTERVAL) + 1


def _has_complete_profit_gate_5m_window(frame: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    if frame is None or frame.empty:
        return False
    window = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)]
    if window.empty:
        return False
    expected_bars = _expected_profit_gate_5m_bars(start_ts, end_ts)
    if len(window) < expected_bars:
        return False
    first_ts = window["timestamp"].iloc[0]
    last_ts = window["timestamp"].iloc[-1]
    return first_ts <= start_ts and last_ts >= end_ts


def _fetch_profit_gate_5m_bars(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    resolved_start = _normalize_utc_timestamp(start_ts)
    resolved_end = _normalize_utc_timestamp(end_ts)
    if resolved_start is None or resolved_end is None or resolved_start > resolved_end:
        return pd.DataFrame()

    request_start = resolved_start.floor("5min")
    request_end = resolved_end.floor("5min")
    interval_ms = int(PROFIT_GATE_BAR_INTERVAL.total_seconds() * 1000)
    request_end_open_ms = int(request_end.timestamp() * 1000)
    request_end_ms = int((request_end + PROFIT_GATE_BAR_INTERVAL).timestamp() * 1000) - 1
    cursor_ms = int(request_start.timestamp() * 1000)
    frames: list[pd.DataFrame] = []
    client = binance_wrapper._resolve_client()

    while cursor_ms <= request_end_ms:
        raw = client.get_klines(
            symbol=str(symbol).upper(),
            interval="5m",
            startTime=cursor_ms,
            endTime=request_end_ms,
            limit=1000,
        )
        if not raw:
            break

        rows = []
        last_open_ms = None
        for kline in raw:
            open_ms = int(kline[0])
            if open_ms > request_end_open_ms:
                continue
            last_open_ms = open_ms
            volume = float(kline[5])
            quote_volume = float(kline[7])
            close_price = float(kline[4])
            rows.append(
                {
                    "timestamp": pd.Timestamp(open_ms, unit="ms", tz="UTC"),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": close_price,
                    "volume": volume,
                    "trade_count": int(kline[8]),
                    "vwap": (quote_volume / volume) if volume > 0.0 else close_price,
                    "symbol": str(symbol).upper(),
                }
            )
        if rows:
            frames.append(pd.DataFrame(rows))
        if last_open_ms is None:
            break
        next_cursor_ms = last_open_ms + interval_ms
        if next_cursor_ms <= cursor_ms:
            break
        cursor_ms = next_cursor_ms
        if len(raw) < 1000 and last_open_ms >= request_end_open_ms:
            break

    if not frames:
        return pd.DataFrame()
    return _normalize_profit_gate_5m_frame(pd.concat(frames, ignore_index=True), symbol=symbol)


def _append_profit_gate_5m_bars(path: Path, frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    new_rows = _normalize_profit_gate_5m_frame(frame, symbol=symbol)
    if path.exists():
        try:
            existing = _normalize_profit_gate_5m_frame(pd.read_csv(path), symbol=symbol)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined = _normalize_profit_gate_5m_frame(combined, symbol=symbol)
    if combined.empty:
        return combined
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def _load_profit_gate_5m_bars(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, *, args=None) -> pd.DataFrame:
    root = _resolve_profit_gate_5m_root(args)
    path = root / f"{str(symbol).upper()}.csv"
    try:
        frame = _normalize_profit_gate_5m_frame(pd.read_csv(path), symbol=symbol) if path.exists() else pd.DataFrame()
    except Exception:
        frame = pd.DataFrame()

    if not _has_complete_profit_gate_5m_window(frame, start_ts, end_ts):
        print(f"[{TAG}] profit gate 5m refresh needed for {symbol} ({start_ts} -> {end_ts}) at {path}")
        try:
            refreshed = _fetch_profit_gate_5m_bars(symbol, start_ts, end_ts)
        except Exception as exc:
            print(f"[{TAG}] profit gate 5m refresh failed for {symbol}: {exc}")
            refreshed = pd.DataFrame()
        if not refreshed.empty:
            frame = _append_profit_gate_5m_bars(path, refreshed, symbol=symbol)
            print(f"[{TAG}] profit gate 5m refresh wrote {len(refreshed)} bars for {symbol} to {path}")

    if frame.empty:
        return pd.DataFrame()
    mask = (frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)
    return frame.loc[mask].sort_values("timestamp").reset_index(drop=True)


def _make_profit_gate_sim_args(args, *, start_ts: pd.Timestamp, initial_cash: float, rules, fee: float):
    raw_long_max_leverage = getattr(args, "max_long_leverage", None)
    long_max_leverage = float(args.max_leverage if raw_long_max_leverage is None else raw_long_max_leverage)
    raw_short_max_leverage = getattr(args, "max_short_leverage", None)
    short_max_leverage = float(long_max_leverage if raw_short_max_leverage is None else raw_short_max_leverage)
    return argparse.Namespace(
        fee=float(fee),
        fill_buffer_pct=float(getattr(args, "fill_buffer_pct", 0.0005)),
        initial_cash=float(initial_cash),
        start=start_ts.isoformat(),
        realistic=True,
        expiry_minutes=int(getattr(args, "expiry_minutes", 90)),
        max_fill_fraction=float(getattr(args, "max_fill_fraction", 0.01)),
        min_notional=float(
            getattr(rules, "min_notional", getattr(args, "min_notional", 5.0))
            or getattr(args, "min_notional", 5.0)
        ),
        tick_size=float(
            getattr(rules, "tick_size", getattr(args, "tick_size", 0.00001))
            or getattr(args, "tick_size", 0.00001)
        ),
        step_size=float(
            getattr(rules, "step_size", getattr(args, "step_size", 1.0))
            or getattr(args, "step_size", 1.0)
        ),
        max_hold_hours=float(getattr(args, "max_hold_hours", 0.0) or 0.0),
        max_leverage=float(getattr(args, "max_leverage", 1.0)),
        long_max_leverage=long_max_leverage,
        short_max_leverage=short_max_leverage,
        margin_hourly_rate=float(getattr(args, "margin_hourly_rate", 0.0)),
        verbose=False,
        live_like=True,
        use_order_expiry=bool(getattr(args, "use_order_expiry", False)),
        reprice_threshold=float(getattr(args, "reprice_threshold", ORDER_REPRICE_THRESHOLD)),
        max_position_notional=getattr(args, "max_position_notional", None),
        allow_short=bool(getattr(args, "allow_short", False)),
    )


def compute_live_like_profit_gate_returns(
    histories: dict,
    *,
    asof_ts,
    lookback_hours: int,
    args,
    rules_by_model: dict[str, object],
    initial_cash: float,
    bars_by_model: Optional[dict[str, pd.DataFrame]] = None,
    simulate_with_trace=None,
) -> dict[str, float]:
    if lookback_hours <= 0:
        return {}

    resolved_asof = _normalize_utc_timestamp(asof_ts)
    if resolved_asof is None:
        return {str(name): 0.0 for name in histories}

    end_ts = resolved_asof.floor("5min")
    if end_ts == resolved_asof:
        end_ts -= pd.Timedelta(minutes=5)
    start_ts = end_ts - pd.Timedelta(hours=float(lookback_hours))
    bars_start_ts = start_ts - pd.Timedelta(hours=1)
    starting_cash = max(0.0, float(initial_cash or 0.0))
    if start_ts >= end_ts or starting_cash <= 0.0:
        return {str(name): 0.0 for name in histories}

    if simulate_with_trace is None:
        from binanceleveragesui.validate_sim_vs_live import simulate_5m_with_trace as _simulate_with_trace

        simulate_with_trace = _simulate_with_trace

    returns: dict[str, float] = {}
    for name, history in histories.items():
        model_cfg = MODELS.get(name, {})
        symbol = str(model_cfg.get("symbol", "") or "")
        if not symbol:
            returns[str(name)] = 0.0
            continue

        signals = _signal_history_to_hourly_signals(history, symbol=symbol)
        if not signals:
            returns[str(name)] = 0.0
            continue

        rules = rules_by_model.get(name)
        if rules is None:
            rules = resolve_symbol_rules(symbol)

        if bars_by_model and name in bars_by_model:
            bars = bars_by_model[name]
            if not bars.empty:
                bars = bars.copy()
                bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
                bars = bars.dropna(subset=["timestamp"])
                bars = bars[(bars["timestamp"] >= bars_start_ts) & (bars["timestamp"] <= end_ts)].reset_index(drop=True)
        else:
            bars = _load_profit_gate_5m_bars(symbol, bars_start_ts, end_ts, args=args)

        if bars is None or bars.empty:
            returns[str(name)] = 0.0
            continue

        sim_args = _make_profit_gate_sim_args(
            args,
            start_ts=start_ts,
            initial_cash=starting_cash,
            rules=rules,
            fee=float(model_cfg.get("maker_fee", getattr(args, "fee", 0.001))),
        )
        try:
            _trades, final_eq, _cash, _inv, _trace = simulate_with_trace(
                sim_args,
                signals,
                bars,
                initial_inv=0.0,
                initial_entry_ts=None,
            )
            returns[str(name)] = float(final_eq / starting_cash - 1.0) if np.isfinite(final_eq) else 0.0
        except Exception as exc:
            print(f"[{TAG}] live-like profit gate simulation failed for {name}: {exc}")
            returns[str(name)] = 0.0

    return returns


def _refresh_price_csv(symbol, data_symbol, data_root):
    try:
        from binance_data_wrapper import fetch_binance_hourly_bars
    except ImportError:
        return
    csv_path = data_root / f"{data_symbol.upper()}.csv"
    if not csv_path.exists():
        return
    existing = pd.read_csv(csv_path)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    last_ts = existing["timestamp"].max()
    try:
        new = fetch_binance_hourly_bars(symbol, start=last_ts, end=datetime.now(timezone.utc))
    except Exception:
        return
    if new is None or len(new) == 0:
        return
    new = new.reset_index()
    new["symbol"] = data_symbol.upper()
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
    new = new[new["timestamp"] > last_ts]
    if len(new) == 0:
        return
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"[{TAG}] data {data_symbol}: +{len(new)} rows")


def _latest_price_timestamp(data_symbol: str, data_root: Path) -> Optional[pd.Timestamp]:
    csv_path = Path(data_root) / f"{data_symbol.upper()}.csv"
    if not csv_path.exists():
        return None
    try:
        frame = pd.read_csv(csv_path, usecols=["timestamp"])
    except Exception:
        return None
    if frame.empty:
        return None
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return None
    return ts.max()


def _latest_cache_timestamp(data_symbol: str, forecast_cache: Path, horizon: int) -> Optional[pd.Timestamp]:
    cache_path = Path(forecast_cache) / f"h{int(horizon)}" / f"{data_symbol.upper()}.parquet"
    if not cache_path.exists():
        return None
    try:
        frame = pd.read_parquet(cache_path, columns=["timestamp"])
    except Exception:
        return None
    if frame.empty:
        return None
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return None
    return ts.max()


def _rebuild_forecast_cache(
    *,
    data_symbol: str,
    data_root: Path,
    forecast_cache: Path,
    forecast_horizons: tuple[int, ...],
    forecast_model_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    reason: str,
) -> None:
    now_hour = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
    guard_key = (data_symbol.upper(), now_hour)
    if guard_key in _FORECAST_REBUILD_GUARD:
        return
    _FORECAST_REBUILD_GUARD[guard_key] = now_hour

    from binancechronossolexperiment.forecasts import build_forecast_bundle

    horizons_str = ",".join(str(int(h)) for h in forecast_horizons)
    print(
        f"[{TAG}] forecast rebuild {data_symbol} reason={reason} "
        f"h={horizons_str} start={start} end={end}"
    )
    try:
        build_forecast_bundle(
            symbol=data_symbol.upper(),
            data_root=Path(data_root),
            cache_root=Path(forecast_cache),
            horizons=forecast_horizons,
            context_hours=FORECAST_CONTEXT_HOURS,
            quantile_levels=FORECAST_QUANTILES,
            batch_size=FORECAST_BATCH_SIZE,
            model_id=str(forecast_model_id),
            cache_only=False,
            start=start,
            end=end,
        )
    except Exception as exc:
        print(f"[{TAG}] forecast rebuild failed for {data_symbol}: {exc}")


def _ensure_forecast_freshness(
    *,
    data_symbol: str,
    data_root: Path,
    forecast_cache: Path,
    forecast_horizons: tuple[int, ...],
    forecast_model_id: str,
    sequence_length: int,
) -> None:
    latest_price_ts = _latest_price_timestamp(data_symbol, data_root)
    if latest_price_ts is None:
        return

    cutoff = latest_price_ts - pd.Timedelta(hours=float(FORECAST_STALE_TOLERANCE_HOURS))
    stale_horizons: list[int] = []
    stale_parts: list[str] = []
    for horizon in forecast_horizons:
        cache_ts = _latest_cache_timestamp(data_symbol, forecast_cache, horizon)
        if cache_ts is None or cache_ts < cutoff:
            stale_horizons.append(int(horizon))
            cache_label = "missing" if cache_ts is None else str(cache_ts)
            stale_parts.append(f"h{int(horizon)}={cache_label}")
    if not stale_horizons:
        return

    lookback_hours = max(int(sequence_length) + 48, FORECAST_REBUILD_LOOKBACK_HOURS)
    start = latest_price_ts - pd.Timedelta(hours=float(lookback_hours))
    _rebuild_forecast_cache(
        data_symbol=data_symbol,
        data_root=Path(data_root),
        forecast_cache=Path(forecast_cache),
        forecast_horizons=tuple(stale_horizons),
        forecast_model_id=str(forecast_model_id),
        start=start,
        end=latest_price_ts,
        reason=f"stale_cache({';'.join(stale_parts)})",
    )


def _load_live_frame(
    symbol,
    data_root,
    forecast_cache,
    forecast_horizons,
    sequence_length,
    *,
    forecast_model_id: str,
    cache_only=True,
):
    dm = ChronosSolDataModule(
        symbol=symbol, data_root=data_root,
        forecast_cache_root=forecast_cache,
        forecast_horizons=forecast_horizons,
        context_hours=FORECAST_CONTEXT_HOURS, quantile_levels=FORECAST_QUANTILES,
        batch_size=FORECAST_BATCH_SIZE, model_id=str(forecast_model_id),
        sequence_length=sequence_length,
        split_config=SplitConfig(val_days=1, test_days=1),
        cache_only=cache_only,
    )
    return dm.full_frame


def _get_total_margin_equity() -> float:
    try:
        margin_acct = get_margin_account()
        total_net_btc = float(margin_acct.get("totalNetAssetOfBtc", 0))
        btc_price = float(binance_wrapper.get_symbol_price("BTCUSDT"))
        equity = total_net_btc * btc_price
        if np.isfinite(equity) and equity > 0.0:
            return float(equity)
    except Exception:
        pass
    return 0.0


def _get_margin_equity_for(symbol, base_asset):
    usdt_entry = get_margin_asset_balance("USDT")
    asset_entry = get_margin_asset_balance(base_asset)
    usdt_free = float(usdt_entry.get("free", 0)) if usdt_entry else 0.0
    usdt_locked = float(usdt_entry.get("locked", 0)) if usdt_entry else 0.0
    usdt_borrowed = float(usdt_entry.get("borrowed", 0)) if usdt_entry else 0.0
    usdt_net = float(usdt_entry.get("netAsset", 0)) if usdt_entry else 0.0
    asset_free = float(asset_entry.get("free", 0)) if asset_entry else 0.0
    asset_locked = float(asset_entry.get("locked", 0)) if asset_entry else 0.0
    asset_borrowed = float(asset_entry.get("borrowed", 0)) if asset_entry else 0.0
    asset_net = float(asset_entry.get("netAsset", 0)) if asset_entry else 0.0
    asset_total = asset_free + asset_locked
    try:
        market_price = float(binance_wrapper.get_symbol_price(symbol))
    except Exception:
        market_price = 0.0
    asset_value = asset_total * market_price
    position_value = position_notional(asset_net, market_price)
    position_side = position_side_from_qty(asset_net)
    # Use Binance total net (all assets) instead of 2-asset manual calc
    try:
        margin_acct = get_margin_account()
        total_net_btc = float(margin_acct.get("totalNetAssetOfBtc", 0))
        btc_price = float(binance_wrapper.get_symbol_price("BTCUSDT"))
        equity = total_net_btc * btc_price
    except Exception:
        equity = usdt_net + asset_net * market_price
    return {
        "usdt_free": usdt_free, "usdt_locked": usdt_locked,
        "usdt_borrowed": usdt_borrowed, "usdt_net": usdt_net,
        "asset_free": asset_free, "asset_locked": asset_locked,
        "asset_borrowed": asset_borrowed, "asset_net": asset_net,
        "asset_total": asset_total, "asset_value": asset_value,
        "position_value": position_value, "position_side": position_side,
        "market_price": market_price,
        "equity": equity,
    }


def _repay_outstanding(symbol, asset="USDT"):
    borrowed = get_margin_borrowed_balance(asset)
    if borrowed <= 0.01:
        return
    cancel_all_margin_orders(symbol)
    try:
        margin_repay_all(asset)
        print(f"[{TAG}] repaid {borrowed:.4f} {asset}")
    except Exception as exc:
        free = get_margin_free_balance(asset)
        if free > 0.01:
            from src.binan.binance_margin import margin_repay
            try:
                margin_repay(asset, free * 0.999)
                print(f"[{TAG}] partial repay {free:.4f} {asset}")
            except Exception as exc2:
                print(f"[{TAG}] repay failed: {exc2}")
        else:
            print(f"[{TAG}] repay failed: {exc}")


def _list_open_side_orders(symbol: str, side: str):
    side_u = str(side).upper()
    try:
        orders = get_open_margin_orders(symbol)
    except Exception as exc:
        print(f"[{TAG}] open-order check failed for {symbol}: {exc}")
        return []

    live = []
    for order in orders:
        if str(order.get("side", "")).upper() != side_u:
            continue
        status = str(order.get("status", "")).upper()
        if status not in {"NEW", "PARTIALLY_FILLED"}:
            continue
        live.append(order)
    return live


def _remaining_order_qty(order: dict) -> float:
    try:
        orig = float(order.get("origQty", 0))
    except Exception:
        orig = 0.0
    try:
        executed = float(order.get("executedQty", 0))
    except Exception:
        executed = 0.0
    return max(0.0, orig - executed)


def _sum_remaining_qty(orders) -> float:
    return float(sum(_remaining_order_qty(order) for order in orders))


def _parse_order_timestamp(order: dict) -> Optional[datetime]:
    for key in ("time", "updateTime", "transactTime"):
        raw = order.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except Exception:
            continue
        if val <= 0:
            continue
        unit = "ms" if val > 1e12 else "s"
        try:
            ts = pd.Timestamp(int(val), unit=unit, tz="UTC")
        except Exception:
            continue
        return ts.to_pydatetime()
    return None


def _order_age_minutes(order: dict, now: Optional[datetime] = None) -> Optional[float]:
    ts = _parse_order_timestamp(order)
    if ts is None:
        return None
    ref = now or datetime.now(timezone.utc)
    age = (ref - ts).total_seconds() / 60.0
    return max(0.0, float(age))


def _empty_exit():
    return {"id": None, "price": 0.0, "qty": 0.0, "symbol": "", "side": "", "kind": ""}


def _normalize_hour_like(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("h")


def _serialize_hour_like(value) -> Optional[str]:
    ts = _normalize_hour_like(value)
    return ts.isoformat() if ts is not None else None


def _signal_history_to_payload(history: SignalHistory) -> dict:
    return {
        "timestamps": list(history.timestamps),
        "buy_prices": [float(v) for v in history.buy_prices],
        "sell_prices": [float(v) for v in history.sell_prices],
        "buy_amounts": [float(v) for v in history.buy_amounts],
        "sell_amounts": [float(v) for v in history.sell_amounts],
        "closes": [float(v) for v in history.closes],
        "equities": [float(v) for v in history.equities],
        "max_entries": int(history.max_entries),
    }


def _signal_history_from_payload(payload: dict) -> SignalHistory:
    history = SignalHistory(max_entries=max(1, int(payload.get("max_entries", 500))))
    series = {
        "timestamps": list(payload.get("timestamps", [])),
        "buy_prices": list(payload.get("buy_prices", [])),
        "sell_prices": list(payload.get("sell_prices", [])),
        "buy_amounts": list(payload.get("buy_amounts", [])),
        "sell_amounts": list(payload.get("sell_amounts", [])),
        "closes": list(payload.get("closes", [])),
        "equities": list(payload.get("equities", [])),
    }
    size = min((len(values) for values in series.values()), default=0)
    if size <= 0:
        return history
    trim = slice(-min(size, history.max_entries), None)
    history.timestamps = [str(v) for v in series["timestamps"][trim]]
    history.buy_prices = [float(v) for v in series["buy_prices"][trim]]
    history.sell_prices = [float(v) for v in series["sell_prices"][trim]]
    history.buy_amounts = [float(v) for v in series["buy_amounts"][trim]]
    history.sell_amounts = [float(v) for v in series["sell_amounts"][trim]]
    history.closes = [float(v) for v in series["closes"][trim]]
    history.equities = [float(v) for v in series["equities"][trim]]
    return history


def _serialize_signal(signal: Optional[dict]) -> dict:
    if not signal:
        return {}
    return {
        "symbol": str(signal.get("symbol", "")),
        "buy_price": float(signal.get("buy_price", 0.0)),
        "sell_price": float(signal.get("sell_price", 0.0)),
        "buy_amount": float(signal.get("buy_amount", 0.0)),
        "sell_amount": float(signal.get("sell_amount", 0.0)),
        "close": float(signal.get("close", 0.0)),
        "signal_hour": _serialize_hour_like(signal.get("signal_hour")),
    }


def _deserialize_signal(payload: dict) -> dict:
    if not payload:
        return {}
    return {
        "symbol": str(payload.get("symbol", "")),
        "buy_price": float(payload.get("buy_price", 0.0)),
        "sell_price": float(payload.get("sell_price", 0.0)),
        "buy_amount": float(payload.get("buy_amount", 0.0)),
        "sell_amount": float(payload.get("sell_amount", 0.0)),
        "close": float(payload.get("close", 0.0)),
        "signal_hour": _normalize_hour_like(payload.get("signal_hour")),
        "ts": time.monotonic(),
    }


def _serialize_order(order: Optional[dict]) -> dict:
    if not order or not order.get("id"):
        return _empty_exit()
    return {
        "id": int(order["id"]),
        "price": float(order.get("price", 0.0)),
        "qty": float(order.get("qty", 0.0)),
        "symbol": str(order.get("symbol", "")),
        "side": str(order.get("side", "")),
        "kind": str(order.get("kind", "")),
        "signal_hour": _serialize_hour_like(order.get("signal_hour")),
    }


def _deserialize_order(payload: dict) -> dict:
    if not payload or not payload.get("id"):
        return _empty_exit()
    return {
        "id": int(payload["id"]),
        "price": float(payload.get("price", 0.0)),
        "qty": float(payload.get("qty", 0.0)),
        "symbol": str(payload.get("symbol", "")),
        "side": str(payload.get("side", "")),
        "kind": str(payload.get("kind", "")),
        "signal_hour": _normalize_hour_like(payload.get("signal_hour")),
    }


def _restore_histories_from_snapshot(payload: dict) -> dict[str, SignalHistory]:
    restored = {name: SignalHistory() for name in MODELS}
    for name in MODELS:
        history_payload = payload.get(name)
        if isinstance(history_payload, dict):
            restored[name] = _signal_history_from_payload(history_payload)
    return restored


def _restore_signals_from_snapshot(payload: dict) -> dict[str, dict]:
    restored: dict[str, dict] = {}
    for name in MODELS:
        signal_payload = payload.get(name)
        if isinstance(signal_payload, dict) and signal_payload:
            restored[name] = _deserialize_signal(signal_payload)
    return restored


def _runtime_signature(args, *, forecast_horizons: tuple[int, ...], model_specs: list[dict]) -> dict:
    return {
        "models": [
            {
                "name": str(spec["name"]),
                "symbol": str(spec["symbol"]),
                "data_symbol": str(spec["data_symbol"]),
                "base_asset": str(spec["base_asset"]),
                "maker_fee": float(spec["maker_fee"]),
                "checkpoint": (
                    None
                    if spec.get("checkpoint") is None
                    else str(Path(spec["checkpoint"]).resolve())
                ),
            }
            for spec in model_specs
        ],
        "data_root": str(Path(args.data_root).resolve()),
        "forecast_cache": str(Path(args.forecast_cache).resolve()),
        "forecast_model_id": str(getattr(args, "forecast_model_id", "amazon/chronos-t5-small")),
        "forecast_horizons": [int(h) for h in forecast_horizons],
        "lookback": int(args.lookback),
        "selection_metric": str(args.selection_metric),
        "selection_mode": str(args.selection_mode),
        "cash_threshold": float(args.cash_threshold),
        "switch_margin": float(args.switch_margin),
        "min_score_gap": float(args.min_score_gap),
        "profit_gate_mode": str(getattr(args, "profit_gate_mode", "hypothetical")),
        "profit_gate_lookback_hours": int(getattr(args, "profit_gate_lookback_hours", 0)),
        "profit_gate_min_return": float(getattr(args, "profit_gate_min_return", 0.0)),
        "horizon": int(args.horizon),
        "sequence_length": int(args.sequence_length),
        "intensity_scale": float(args.intensity_scale),
        "min_gap_pct": float(args.min_gap_pct),
        "max_hold_hours": int(args.max_hold_hours) if args.max_hold_hours is not None else None,
        "max_leverage": float(args.max_leverage),
        "allow_short": bool(getattr(args, "allow_short", False)),
        "max_long_leverage": (
            None if getattr(args, "max_long_leverage", None) is None else float(args.max_long_leverage)
        ),
        "max_short_leverage": (
            None if getattr(args, "max_short_leverage", None) is None else float(args.max_short_leverage)
        ),
        "max_position_notional": (
            None if args.max_position_notional is None else float(args.max_position_notional)
        ),
    }


def _load_runtime_snapshot(path: Path = RUNTIME_STATE_FILE) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if int(payload.get("version", 0)) != RUNTIME_STATE_VERSION:
        return None
    return payload


def _runtime_snapshot_is_compatible(payload: Optional[dict], signature: dict) -> bool:
    if not payload:
        return False
    return payload.get("signature") == signature


def _merge_state_with_snapshot(state: MetaState, snapshot_payload: Optional[dict]) -> MetaState:
    if not isinstance(snapshot_payload, dict):
        return state
    merged = MetaState(
        active_model=state.active_model,
        in_position=state.in_position,
        position_side=state.position_side,
        open_ts=state.open_ts,
        open_price=state.open_price,
    )
    snapshot_active = str(snapshot_payload.get("active_model", "") or "")
    snapshot_in_position = bool(snapshot_payload.get("in_position", False))
    snapshot_position_side = str(snapshot_payload.get("position_side", "") or "")
    snapshot_open_ts = snapshot_payload.get("open_ts")
    try:
        snapshot_open_price = float(snapshot_payload.get("open_price", 0.0))
    except Exception:
        snapshot_open_price = 0.0

    if not merged.active_model and snapshot_active:
        merged.active_model = snapshot_active
    if not merged.in_position and snapshot_in_position:
        merged.in_position = True
    if not merged.position_side and snapshot_position_side:
        merged.position_side = snapshot_position_side
    if not merged.open_ts and snapshot_open_ts:
        merged.open_ts = str(snapshot_open_ts)
    if merged.open_price <= 0.0 and snapshot_open_price > 0.0:
        merged.open_price = snapshot_open_price
    return merged


def _snapshot_signals_are_current(payload: dict, latest_signal_hours: dict[str, Optional[pd.Timestamp]]) -> bool:
    for name in MODELS:
        latest_hour = _normalize_hour_like(latest_signal_hours.get(name))
        signal_payload = payload.get(name) if isinstance(payload, dict) else None
        signal_hour = _normalize_hour_like(signal_payload.get("signal_hour")) if isinstance(signal_payload, dict) else None
        if latest_hour is None or signal_hour is None or signal_hour != latest_hour:
            return False
    return True


def _save_runtime_snapshot(
    *,
    state: MetaState,
    histories: dict[str, SignalHistory],
    signals: dict[str, dict],
    entry_order: dict,
    exit_order: dict,
    signature: dict,
    path: Path = RUNTIME_STATE_FILE,
) -> None:
    payload = {
        "version": RUNTIME_STATE_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "state": {
            "active_model": state.active_model,
            "in_position": state.in_position,
            "position_side": state.position_side,
            "open_ts": state.open_ts,
            "open_price": state.open_price,
        },
        "histories": {name: _signal_history_to_payload(histories[name]) for name in MODELS},
        "signals": {name: _serialize_signal(signals.get(name)) for name in MODELS},
        "entry_order": _serialize_order(entry_order),
        "exit_order": _serialize_order(exit_order),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except Exception:
        pass


def _merge_recovered_order_with_snapshot(recovered: dict, snapshot_order: Optional[dict]) -> dict:
    if not recovered.get("id"):
        return recovered
    if not snapshot_order or snapshot_order.get("id") != recovered.get("id"):
        return recovered
    if snapshot_order.get("symbol") != recovered.get("symbol"):
        return recovered
    merged = dict(recovered)
    snapshot_signal_hour = _normalize_hour_like(snapshot_order.get("signal_hour"))
    if snapshot_signal_hour is not None:
        merged["signal_hour"] = snapshot_signal_hour
    return merged


def _is_same_signal_hour(order: dict, signal: dict) -> bool:
    order_hour = _normalize_hour_like(order.get("signal_hour"))
    signal_hour = _normalize_hour_like(signal.get("signal_hour"))
    return order_hour is not None and signal_hour is not None and order_hour == signal_hour


def _check_margin_order(symbol: str, order_id: int):
    try:
        client = binance_wrapper._resolve_client()
        return client.get_margin_order(symbol=symbol, orderId=int(order_id), isIsolated="FALSE")
    except Exception as exc:
        print(f"[{TAG}] order check failed {symbol}/{order_id}: {exc}")
        return None


def _place_direct_order(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    rules,
    *,
    side_effect: str,
    kind: str,
):
    side_l = str(side).lower()
    px = quantize_price(price, tick_size=rules.tick_size, side=side_l)
    qty = quantize_qty(qty, step_size=rules.step_size)
    if qty <= 0 or px <= 0:
        return None
    if rules.min_notional and qty * px < rules.min_notional:
        return None
    order = create_margin_order(
        symbol,
        side_l.upper(),
        "LIMIT",
        qty,
        price=px,
        side_effect_type=side_effect,
        time_in_force="GTC",
    )
    oid = order.get("orderId")
    print(f"[{TAG}] placed {kind}: {symbol} {side_l.upper()} {qty:.4f} @ {px:.4f} orderId={oid}")
    return {"id": oid, "price": px, "qty": qty, "symbol": symbol, "side": side_l, "kind": kind}


def _place_direct_exit(
    symbol: str,
    qty: float,
    price: float,
    rules,
    *,
    position_side: str,
    kind: str = "exit",
):
    side = "buy" if str(position_side).lower() == "short" else "sell"
    side_effect = "AUTO_REPAY"
    return _place_direct_order(
        symbol,
        side,
        qty,
        price,
        rules,
        side_effect=side_effect,
        kind=kind,
    )


def _place_direct_entry(symbol: str, qty: float, price: float, rules, *, entry_side: str):
    side = str(entry_side).lower()
    side_effect = "AUTO_BORROW_REPAY" if side == "sell" else "MARGIN_BUY"
    kind = "short_entry" if side == "sell" else "entry"
    return _place_direct_order(
        symbol,
        side,
        qty,
        price,
        rules,
        side_effect=side_effect,
        kind=kind,
    )


def _cancel_tracked_order(order: dict, label: str = "order"):
    if not order.get("id"):
        return
    try:
        cancel_margin_order(order["symbol"], order_id=int(order["id"]))
        print(f"[{TAG}] cancelled {label} orderId={order['id']}")
    except Exception as exc:
        print(f"[{TAG}] cancel {label} failed: {exc}")


def _recover_open_orders(symbol: str, side: Optional[str] = None):
    if side:
        orders = _list_open_side_orders(symbol, side)
    else:
        orders = []
        for current_side in ("BUY", "SELL"):
            orders.extend(_list_open_side_orders(symbol, current_side))
    if not orders:
        return _empty_exit()
    if len(orders) > 1:
        print(f"[{TAG}] found {len(orders)} open orders, keeping newest, cancelling rest")
        orders.sort(key=lambda o: int(o.get("orderId", 0)))
        for old in orders[:-1]:
            try:
                cancel_margin_order(symbol, order_id=int(old["orderId"]))
            except Exception:
                pass
        orders = orders[-1:]
    o = orders[0]
    side_l = str(o.get("side", "")).lower()
    kind = "entry" if side_l in {"buy", "sell"} else ""
    remaining = _remaining_order_qty(o)
    qty = remaining if remaining > 0 else float(o.get("origQty", 0))
    signal_hour = _normalize_hour_like(_parse_order_timestamp(o))
    recovered = {
        "id": int(o["orderId"]),
        "price": float(o["price"]),
        "qty": qty,
        "symbol": symbol,
        "signal_hour": signal_hour,
        "side": side_l,
        "kind": kind,
    }
    print(
        f"[{TAG}] recovered open order: {symbol} {side_l.upper()} "
        f"{recovered['qty']:.4f} @ {recovered['price']:.4f} id={recovered['id']}"
    )
    return recovered


def _recover_exit_orders(symbol: str, *, position_side: str):
    expected = "BUY" if str(position_side).lower() == "short" else "SELL"
    recovered = _recover_open_orders(symbol, expected)
    if recovered.get("id"):
        recovered["kind"] = "exit"
    return recovered


def _recover_entry_orders(symbol: str, *, entry_side: Optional[str] = None):
    expected = None if entry_side is None else str(entry_side).upper()
    recovered = _recover_open_orders(symbol, expected)
    if recovered.get("id"):
        recovered["kind"] = "entry"
    return recovered


def _inventory_mostly_locked(asset_free: float, asset_locked: float, *, lock_ratio: float = 0.95) -> bool:
    free = max(0.0, float(asset_free))
    locked = max(0.0, float(asset_locked))
    total = free + locked
    if total <= 0.0 or locked <= 0.0:
        return False
    return (locked / total) >= max(0.0, min(1.0, float(lock_ratio)))


def _normalize_state_open_ts_for_position(
    state: MetaState,
    active_symbol: str,
    *,
    max_hold_hours: Optional[int],
    future_tolerance_minutes: float = 5.0,
) -> bool:
    if not state.in_position:
        return False

    now = datetime.now(timezone.utc)
    opened: Optional[datetime] = None
    if state.open_ts:
        try:
            opened = datetime.fromisoformat(state.open_ts)
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
        except Exception:
            opened = None

    if opened is not None and opened <= (now + timedelta(minutes=future_tolerance_minutes)):
        return False

    reason = "missing_or_invalid" if opened is None else "future_ts"
    replacement: Optional[datetime] = None
    if active_symbol:
        try:
            exit_side = "BUY" if str(state.position_side).lower() == "short" else "SELL"
            open_exits = _list_open_side_orders(active_symbol, exit_side)
            order_times = [ts for order in open_exits if (ts := _parse_order_timestamp(order)) is not None]
            if order_times:
                replacement = min(order_times)
                reason = f"{reason}_from_open_order"
        except Exception:
            pass

    if replacement is None:
        back_hours = float(max_hold_hours) if max_hold_hours and max_hold_hours > 0 else 1.0
        replacement = now - timedelta(hours=back_hours)
        reason = f"{reason}_fallback"

    state.open_ts = replacement.astimezone(timezone.utc).isoformat()
    state.save()
    print(f"[{TAG}] corrected open_ts ({reason}) -> {state.open_ts}")
    _log_event("open_ts_corrected", model=state.active_model, reason=reason, open_ts=state.open_ts)
    return True


def _cap_buy_notional(target_notional: float, usdt_free: float, max_borrowable_usdt: float) -> float:
    target = max(0.0, float(target_notional))
    available_quote = max(0.0, float(usdt_free)) + max(0.0, float(max_borrowable_usdt))
    return min(target, available_quote)


def _cap_short_notional(
    target_notional: float,
    market_price: float,
    asset_free: float,
    max_borrowable_asset: float,
) -> float:
    target = max(0.0, float(target_notional))
    price = max(0.0, float(market_price))
    if price <= 0.0:
        return 0.0
    available_qty = max(0.0, float(asset_free)) + max(0.0, float(max_borrowable_asset))
    return min(target, available_qty * price)


def _cap_position_notional(
    target_notional: float,
    current_asset_notional: float,
    max_position_notional: Optional[float],
) -> float:
    target = max(0.0, float(target_notional))
    if max_position_notional is None:
        return target
    cap = max(0.0, float(max_position_notional))
    remaining = max(0.0, cap - max(0.0, float(current_asset_notional)))
    return min(target, remaining)


def _aligned_position_notional(
    asset_net: float,
    market_price: float,
    *,
    side: str,
) -> float:
    if position_side_from_qty(asset_net) != str(side or "").strip().lower():
        return 0.0
    return position_notional(asset_net, market_price)


def _cap_directional_entry_notional(
    target_notional: float,
    asset_net: float,
    market_price: float,
    *,
    side: str,
    max_position_notional: Optional[float],
) -> float:
    target = max(0.0, float(target_notional))
    if max_position_notional is None:
        return target
    cap = max(0.0, float(max_position_notional))
    aligned = _aligned_position_notional(asset_net, market_price, side=side)
    opposing = max(0.0, position_notional(asset_net, market_price) - aligned)
    remaining = max(0.0, cap - aligned + opposing)
    return min(target, remaining)


def _remaining_target_entry_notional(
    *,
    side: str,
    equity: float,
    asset_net: float,
    market_price: float,
    usdt_free: float,
    asset_free: float,
    max_borrowable_usdt: float,
    max_borrowable_asset: float,
    long_max_leverage: float,
    short_max_leverage: float,
    max_position_notional: Optional[float],
) -> float:
    target_notional = remaining_entry_notional(
        side=side,
        equity=equity,
        current_qty=asset_net,
        market_price=market_price,
        long_max_leverage=long_max_leverage,
        short_max_leverage=short_max_leverage,
    )
    if str(side or "").strip().lower() == "short":
        capped = _cap_short_notional(
            target_notional,
            market_price,
            asset_free,
            max_borrowable_asset,
        )
    else:
        capped = _cap_buy_notional(
            target_notional,
            usdt_free,
            max_borrowable_usdt,
        )
    return _cap_directional_entry_notional(
        capped,
        asset_net,
        market_price,
        side=side,
        max_position_notional=max_position_notional,
    )


def _order_needs_resize_to_target(
    order: dict,
    *,
    target_qty: float,
    reference_price: float,
    step_size: float,
    min_notional: float,
) -> bool:
    if not order.get("id"):
        return False
    working_qty = max(0.0, float(order.get("qty", 0.0)))
    desired_qty = max(0.0, float(target_qty))
    qty_gap = abs(desired_qty - working_qty)
    step = max(0.0, float(step_size or 0.0))
    if qty_gap <= 0.0:
        return False
    if step > 0.0 and qty_gap + 1e-12 < step:
        return False
    price = max(0.0, float(reference_price))
    gap_notional = qty_gap * price
    desired_notional = desired_qty * price
    material_threshold = max(max(0.0, float(min_notional or 0.0)) * 0.25, desired_notional * 0.01)
    return gap_notional + 1e-12 >= material_threshold


def _resolve_max_position_notional(
    max_position_notional: Optional[float],
    *,
    dry_run: bool,
    disable_probe_cap: bool = False,
    paper_env: Optional[str] = None,
) -> Optional[float]:
    if max_position_notional is not None:
        return max(0.0, float(max_position_notional))
    if disable_probe_cap:
        return None
    if dry_run:
        return None
    effective_paper = str(os.environ.get("PAPER", "0") if paper_env is None else paper_env).strip()
    if effective_paper == "0":
        return LIVE_PROBE_MAX_POSITION_NOTIONAL
    return None


def _set_inference_seeds(seed: int = DETERMINISTIC_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is not None and hasattr(cuda_backends, "matmul"):
        try:
            cuda_backends.matmul.allow_tf32 = False
        except Exception:
            pass


def _resolve_directional_leverages(
    max_leverage: float,
    *,
    max_long_leverage: Optional[float] = None,
    max_short_leverage: Optional[float] = None,
) -> tuple[float, float]:
    base = max(0.0, float(max_leverage))
    long_lev = base if max_long_leverage is None else max(0.0, float(max_long_leverage))
    short_lev = base if max_short_leverage is None else max(0.0, float(max_short_leverage))
    return long_lev, short_lev


def _maker_limit_ok(side: str, limit_price: float, market_price: float) -> bool:
    side_l = str(side).lower()
    if side_l == "buy":
        return float(limit_price) < float(market_price)
    if side_l == "sell":
        return float(limit_price) > float(market_price)
    return False


def _effective_position_notional_threshold(max_position_notional: Optional[float]) -> float:
    threshold = float(MIN_POSITION_NOTIONAL)
    if max_position_notional is None:
        return threshold
    cap = max(0.0, float(max_position_notional))
    if cap <= 0.0:
        return threshold
    return min(threshold, cap * PROBE_POSITION_NOTIONAL_RATIO)


def _has_effective_position(
    asset_total: float,
    asset_value: float,
    *,
    step_size: float,
    max_position_notional: Optional[float],
) -> bool:
    qty = abs(float(asset_total))
    value = max(0.0, float(asset_value))
    step = max(0.0, float(step_size or 0.0))
    if qty <= 0.0:
        return False
    if step > 0.0 and qty + 1e-12 < step:
        return False
    return value + 1e-12 >= _effective_position_notional_threshold(max_position_notional)


def _promote_detected_position(
    state: MetaState,
    *,
    model: str,
    position_side: str,
    market_price: float,
    asset_total: float,
    asset_value: float,
    entry_order: Optional[dict] = None,
) -> dict:
    tracked_entry = entry_order if entry_order and entry_order.get("id") else None
    if not state.open_ts:
        state.open_ts = datetime.now(timezone.utc).isoformat()

    tracked_price = float(tracked_entry.get("price", 0.0)) if tracked_entry else 0.0
    if tracked_price > 0.0:
        state.open_price = tracked_price
    elif state.open_price <= 0.0 and market_price > 0.0:
        state.open_price = market_price

    state.in_position = True
    state.position_side = str(position_side or "")
    state.save()

    if tracked_entry:
        fill_qty = max(0.0, float(tracked_entry.get("qty", 0.0))) or abs(float(asset_total))
        fill_price = tracked_price if tracked_price > 0.0 else max(0.0, float(market_price))
        print(
            f"[{TAG}] detected filled entry for {model}/{position_side}: "
            f"qty={fill_qty:.4f} price={fill_price:.4f} source=balance_detection"
        )
        _log_event(
            "entry_filled",
            model=model,
            position_side=position_side,
            price=fill_price,
            qty=fill_qty,
            source="balance_detection",
        )
        return _empty_exit()

    print(f"[{TAG}] detected {model}/{position_side} position: {asset_total:.4f} (${asset_value:.2f})")
    _log_event(
        "position_detected",
        model=model,
        position_side=position_side,
        market_price=market_price,
        qty=asset_total,
        asset_value=asset_value,
        source="balance_detection",
    )
    return _empty_exit()


def _collect_managed_positions(
    rules: dict,
    *,
    max_position_notional: Optional[float],
) -> list[dict]:
    detected_positions: list[dict] = []
    for name, cfg in MODELS.items():
        bal = _get_margin_equity_for(cfg["symbol"], cfg["base_asset"])
        rule = rules[name]
        position_value = float(bal.get("position_value", bal.get("asset_value", 0.0)))
        if not _has_effective_position(
            bal["asset_net"],
            position_value,
            step_size=rule.step_size,
            max_position_notional=max_position_notional,
        ):
            continue
        detected_positions.append(
            {
                "model": name,
                "symbol": cfg["symbol"],
                "base_asset": cfg["base_asset"],
                "position_value": position_value,
                "position_side": str(bal.get("position_side", "") or ""),
                "balance": bal,
            }
        )
    detected_positions.sort(key=lambda row: float(row["position_value"]), reverse=True)
    return detected_positions


def _managed_position_details(detected_positions: list[dict]) -> dict[str, dict]:
    details: dict[str, dict] = {}
    for position in detected_positions:
        bal = position["balance"]
        details[str(position["model"])] = {
            "position_value": round(float(position["position_value"]), 4),
            "position_side": str(position.get("position_side", "") or ""),
            "asset_net": round(float(bal.get("asset_net", 0.0)), 8),
        }
    return details


def _reconcile_managed_positions(
    state: MetaState,
    rules: dict,
    *,
    max_position_notional: Optional[float],
) -> dict:
    detected_positions = _collect_managed_positions(
        rules,
        max_position_notional=max_position_notional,
    )
    details = _managed_position_details(detected_positions)
    chosen_position = None
    chosen_model = ""
    if detected_positions:
        available = {str(position["model"]) for position in detected_positions}
        chosen_model = state.active_model if state.active_model in available else str(detected_positions[0]["model"])
        chosen_position = next(position for position in detected_positions if position["model"] == chosen_model)
        chosen_bal = chosen_position["balance"]
        chosen_side = str(chosen_position.get("position_side", "") or "")
        needs_reconcile = (
            state.active_model != chosen_model
            or not state.in_position
            or str(state.position_side or "") != chosen_side
        )
        if needs_reconcile:
            chosen_cfg = MODELS[chosen_model]
            print(
                f"[{TAG}] WARNING: live {chosen_cfg['base_asset']} {chosen_side or 'unknown'} position "
                f"(${chosen_position['position_value']:.2f}), state says "
                f"model={state.active_model} in_position={state.in_position} side={state.position_side}"
            )
            print(
                f"[{TAG}] AUTO-RECONCILE: setting active_model={chosen_model}, "
                f"in_position=True, position_side={chosen_side or 'unknown'}"
            )
            if state.active_model != chosen_model or str(state.position_side or "") != chosen_side:
                state.open_ts = None
                state.open_price = 0.0
            state.active_model = chosen_model
            state.in_position = True
            state.position_side = chosen_side
            if not state.open_ts:
                state.open_ts = datetime.now(timezone.utc).isoformat()
            if state.open_price <= 0 and chosen_bal["market_price"] > 0:
                state.open_price = float(chosen_bal["market_price"])
            state.save()
    return {
        "positions": detected_positions,
        "details": details,
        "chosen_model": chosen_model,
        "chosen_position": chosen_position,
        "conflict": len(detected_positions) > 1,
    }


def _signal_from_action(action: dict, last_row: pd.Series, *, intensity_scale: float, symbol: str) -> dict:
    buy_price = float(action.get("buy_price", 0))
    sell_price = float(action.get("sell_price", 0))
    buy_amount = max(0.0, min(100.0, float(action.get("buy_amount", 0)) * intensity_scale))
    sell_amount = max(0.0, min(100.0, float(action.get("sell_amount", 0)) * intensity_scale))

    min_spread = 0.002
    if buy_price > 0 and sell_price > 0 and sell_price <= buy_price * (1 + min_spread):
        mid = (buy_price + sell_price) / 2
        buy_price = mid * (1 - min_spread / 2)
        sell_price = mid * (1 + min_spread / 2)

    close = float(last_row.get("close", 0))
    signal_hour = _normalize_hour_like(last_row.get("timestamp"))
    return {
        "symbol": symbol,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "buy_amount": buy_amount,
        "sell_amount": sell_amount,
        "close": close,
        "ts": time.monotonic(),
        "signal_hour": signal_hour,
    }


def _upsert_signal_history(history: SignalHistory, signal: dict) -> None:
    signal_hour = _normalize_hour_like(signal.get("signal_hour"))
    history_ts = signal_hour.isoformat() if signal_hour is not None else datetime.now(timezone.utc).isoformat()

    if history.timestamps and signal_hour is not None:
        last_hour = _normalize_hour_like(history.timestamps[-1])
        if last_hour == signal_hour:
            history.timestamps[-1] = history_ts
            history.buy_prices[-1] = float(signal.get("buy_price", 0.0))
            history.sell_prices[-1] = float(signal.get("sell_price", 0.0))
            history.buy_amounts[-1] = float(signal.get("buy_amount", 0.0))
            history.sell_amounts[-1] = float(signal.get("sell_amount", 0.0))
            history.closes[-1] = float(signal.get("close", 0.0))
            history.equities[-1] = float(signal.get("close", 0.0))
            return

    history.add(
        history_ts,
        float(signal.get("buy_price", 0.0)),
        float(signal.get("sell_price", 0.0)),
        float(signal.get("buy_amount", 0.0)),
        float(signal.get("sell_amount", 0.0)),
        float(signal.get("close", 0.0)),
        float(signal.get("close", 0.0)),
    )


def _load_fresh_live_frame(
    *,
    symbol: str,
    data_symbol: str,
    data_root: Path,
    forecast_cache: Path,
    forecast_horizons: tuple[int, ...],
    forecast_model_id: str,
    sequence_length: int,
) -> pd.DataFrame:
    _refresh_price_csv(symbol, data_symbol, data_root)
    _ensure_forecast_freshness(
        data_symbol=data_symbol,
        data_root=Path(data_root),
        forecast_cache=Path(forecast_cache),
        forecast_horizons=tuple(int(h) for h in forecast_horizons),
        forecast_model_id=str(forecast_model_id),
        sequence_length=int(sequence_length),
    )

    latest_price_ts = _latest_price_timestamp(data_symbol, Path(data_root))
    try:
        frame = _load_live_frame(
            data_symbol,
            data_root,
            forecast_cache,
            forecast_horizons,
            sequence_length,
            forecast_model_id=str(forecast_model_id),
            cache_only=True,
        )
    except Exception as exc:
        if latest_price_ts is None:
            raise
        lookback_hours = max(int(sequence_length) + 48, FORECAST_REBUILD_LOOKBACK_HOURS)
        start = latest_price_ts - pd.Timedelta(hours=float(lookback_hours))
        _rebuild_forecast_cache(
            data_symbol=data_symbol,
            data_root=Path(data_root),
            forecast_cache=Path(forecast_cache),
            forecast_horizons=tuple(int(h) for h in forecast_horizons),
            forecast_model_id=str(forecast_model_id),
            start=start,
            end=latest_price_ts,
            reason=f"load_failure({exc})",
        )
        frame = _load_live_frame(
            data_symbol,
            data_root,
            forecast_cache,
            forecast_horizons,
            sequence_length,
            forecast_model_id=str(forecast_model_id),
            cache_only=True,
        )

    if latest_price_ts is not None and not frame.empty:
        frame_ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").max()
        cutoff = latest_price_ts - pd.Timedelta(hours=float(FORECAST_STALE_TOLERANCE_HOURS))
        if pd.isna(frame_ts) or frame_ts < cutoff:
            lookback_hours = max(int(sequence_length) + 48, FORECAST_REBUILD_LOOKBACK_HOURS)
            start = latest_price_ts - pd.Timedelta(hours=float(lookback_hours))
            _rebuild_forecast_cache(
                data_symbol=data_symbol,
                data_root=Path(data_root),
                forecast_cache=Path(forecast_cache),
                forecast_horizons=tuple(int(h) for h in forecast_horizons),
                forecast_model_id=str(forecast_model_id),
                start=start,
                end=latest_price_ts,
                reason=f"frame_stale(frame_ts={frame_ts},price_ts={latest_price_ts})",
            )
            frame = _load_live_frame(
                data_symbol,
                data_root,
                forecast_cache,
                forecast_horizons,
                sequence_length,
                forecast_model_id=str(forecast_model_id),
                cache_only=True,
            )
    return frame


def _bootstrap_signal_history_from_frame(
    history: SignalHistory,
    frame: pd.DataFrame,
    model,
    normalizer,
    feature_columns,
    *,
    horizon: int,
    sequence_length: int,
    intensity_scale: float,
    history_hours: int,
    symbol: str,
) -> list[dict]:
    if frame.empty:
        return []

    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return []

    usable_hours = max(int(history_hours), 1)
    start_ts = ts.max() - pd.Timedelta(hours=usable_hours)
    start_idx_arr = frame.index[ts >= start_ts]
    start_idx = int(start_idx_arr[0]) if len(start_idx_arr) > 0 else 0

    seeded: list[dict] = []
    for bar_idx in range(start_idx, len(frame)):
        sub_frame = frame.iloc[: bar_idx + 1].copy()
        _set_inference_seeds()
        action = generate_latest_action(
            model=model,
            frame=sub_frame,
            feature_columns=feature_columns,
            normalizer=normalizer,
            sequence_length=sequence_length,
            horizon=horizon,
            require_gpu=True,
        )
        last_row = frame.iloc[bar_idx]
        signal = _signal_from_action(
            action,
            last_row,
            intensity_scale=intensity_scale,
            symbol=symbol,
        )
        _upsert_signal_history(history, signal)
        seeded.append(signal)
    return seeded


def _refresh_signal(model, normalizer, feature_columns, *, horizon, sequence_length,
                    intensity_scale, data_root, forecast_cache, forecast_horizons,
                    forecast_model_id,
                    data_symbol, symbol):
    frame = _load_fresh_live_frame(
        symbol=symbol,
        data_symbol=data_symbol,
        data_root=Path(data_root),
        forecast_cache=Path(forecast_cache),
        forecast_horizons=tuple(int(h) for h in forecast_horizons),
        forecast_model_id=str(forecast_model_id),
        sequence_length=int(sequence_length),
    )
    _set_inference_seeds()
    action = generate_latest_action(
        model=model, frame=frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=sequence_length,
        horizon=horizon, require_gpu=True,
    )
    last_row = frame.iloc[-1]
    return _signal_from_action(
        action,
        last_row,
        intensity_scale=float(intensity_scale),
        symbol=symbol,
    )


def _run_hypothetical_score(
    history: SignalHistory,
    lookback: int,
    maker_fee: float,
    max_leverage: float,
    metric: str,
    *,
    allow_short: bool = False,
    max_long_leverage: Optional[float] = None,
    max_short_leverage: Optional[float] = None,
) -> float:
    """Run a mini-sim on recent signal history and return the requested score."""
    n = len(history.timestamps)
    if n < 3:
        return 0.0

    # Prefer true time-based window (hours) when timestamps are parseable.
    start = 0
    if lookback > 0:
        try:
            ts = pd.to_datetime(history.timestamps, utc=True, errors="coerce")
            if len(ts) == n and ts.notna().all():
                end_ts = ts[-1]
                cutoff = end_ts - pd.Timedelta(hours=float(lookback))
                idx = np.where(ts >= cutoff)[0]
                if len(idx) > 0:
                    start = int(idx[0])
                else:
                    start = n - 1
            else:
                start = max(0, n - lookback)
        except Exception:
            start = max(0, n - lookback)

    # Ensure we always have at least 2 transitions to score.
    start = min(start, max(0, n - 3))

    cash = 10000.0
    inventory = 0.0
    long_lev, short_lev = _resolve_directional_leverages(
        max_leverage,
        max_long_leverage=max_long_leverage,
        max_short_leverage=max_short_leverage,
    )
    bars_held = 0
    eq_curve = [cash]

    for i in range(start, n):
        close = history.closes[i]
        buy_p = history.buy_prices[i]
        sell_p = history.sell_prices[i]
        buy_a = min(history.buy_amounts[i], 100.0) / 100.0
        sell_a = min(history.sell_amounts[i], 100.0) / 100.0

        equity = cash + inventory * close

        if inventory > 0 and bars_held >= 6:
            cash += inventory * close * 0.999 * (1 - maker_fee)
            inventory = 0.0
            bars_held = 0
            eq_curve.append(cash)
            continue
        if inventory < 0 and bars_held >= 6:
            cover_qty = abs(inventory)
            cash -= cover_qty * close * 1.001 * (1 + maker_fee)
            inventory = 0.0
            bars_held = 0
            eq_curve.append(cash)
            continue

        sold = False
        if sell_a > 0 and sell_p > 0:
            if inventory > 0:
                sq = min(sell_a * inventory, inventory)
                if sq > 0:
                    cash += sq * sell_p * (1 - maker_fee)
                    inventory -= sq
                    sold = True
                    if inventory <= 0:
                        bars_held = 0
            elif allow_short and short_lev > 0:
                max_sv = short_lev * max(equity, 0.0) - abs(min(inventory, 0.0)) * sell_p
                if max_sv > 0:
                    sq = sell_a * max_sv / (sell_p * (1 + maker_fee))
                    if sq > 0:
                        cash += sq * sell_p * (1 - maker_fee)
                        inventory -= sq
                        sold = True

        if not sold and buy_a > 0 and buy_p > 0:
            if inventory < 0:
                bq = min(buy_a * abs(inventory), abs(inventory))
                if bq > 0:
                    cash -= bq * buy_p * (1 + maker_fee)
                    inventory += bq
                    if inventory >= 0:
                        bars_held = 0
            else:
                max_bv = long_lev * max(equity, 0.0) - max(inventory, 0.0) * buy_p
                if max_bv > 0:
                    bq = buy_a * max_bv / (buy_p * (1 + maker_fee))
                    if bq > 0:
                        cash -= bq * buy_p * (1 + maker_fee)
                        inventory += bq

        if abs(inventory) > 0:
            bars_held += 1
        else:
            bars_held = 0
        eq_curve.append(cash + inventory * close)

    eq = np.asarray(eq_curve, dtype=np.float64)
    if len(eq) < 3 or not np.isfinite(eq).all():
        return 0.0

    returns = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0.0
    return float(score_trailing_returns(returns, metric))


def _recent_profit_gate_returns(
    histories: dict,
    *,
    lookback_hours: int,
    max_leverage: float,
    allow_short: bool = False,
    max_long_leverage: Optional[float] = None,
    max_short_leverage: Optional[float] = None,
) -> dict[str, float]:
    if lookback_hours <= 0:
        return {}
    returns: dict[str, float] = {}
    for name, hist in histories.items():
        fee = MODELS[name]["maker_fee"]
        score_kwargs = {}
        if allow_short or max_long_leverage is not None or max_short_leverage is not None:
            score_kwargs = {
                "allow_short": allow_short,
                "max_long_leverage": max_long_leverage,
                "max_short_leverage": max_short_leverage,
            }
        returns[name] = float(
            _run_hypothetical_score(
                hist,
                int(lookback_hours),
                fee,
                max_leverage,
                "return",
                **score_kwargs,
            )
        )
    return returns


def select_model(
    histories: dict,
    lookback: int,
    max_leverage: float,
    metric: str,
    selection_mode: str,
    cash_threshold: float,
    current_model: str = "",
    switch_margin: float = 0.0,
    min_score_gap: float = 0.0,
    *,
    allow_short: bool = False,
    max_long_leverage: Optional[float] = None,
    max_short_leverage: Optional[float] = None,
    profit_gate_lookback_hours: int = 0,
    profit_gate_min_return: float = 0.0,
    profit_gate_mode: str = "hypothetical",
    profit_gate_returns: Optional[dict[str, float]] = None,
) -> str:
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unsupported selection metric '{metric}'. Expected one of {SUPPORTED_SELECTION_METRICS}.")
    if selection_mode not in SUPPORTED_SELECTION_MODES:
        raise ValueError(f"Unsupported selection mode '{selection_mode}'. Expected one of {SUPPORTED_SELECTION_MODES}.")
    if min_score_gap < 0:
        raise ValueError(f"min_score_gap must be >= 0, got {min_score_gap}")
    if profit_gate_lookback_hours < 0:
        raise ValueError(f"profit_gate_lookback_hours must be >= 0, got {profit_gate_lookback_hours}")
    profit_gate_mode = _normalize_profit_gate_mode(profit_gate_mode)

    scores = {}
    for name, hist in histories.items():
        fee = MODELS[name]["maker_fee"]
        score_kwargs = {}
        if allow_short or max_long_leverage is not None or max_short_leverage is not None:
            score_kwargs = {
                "allow_short": allow_short,
                "max_long_leverage": max_long_leverage,
                "max_short_leverage": max_short_leverage,
            }
        s = _run_hypothetical_score(
            hist,
            lookback,
            fee,
            max_leverage,
            metric,
            **score_kwargs,
        )
        scores[name] = s

    if profit_gate_lookback_hours > 0 and profit_gate_returns is not None:
        profit_returns = {str(name): float(value) for name, value in profit_gate_returns.items()}
    elif profit_gate_lookback_hours > 0 and profit_gate_mode == "live_like":
        raise ValueError("profit_gate_returns must be provided when profit_gate_mode='live_like'")
    else:
        profit_returns = _recent_profit_gate_returns(
            histories,
            lookback_hours=int(profit_gate_lookback_hours),
            max_leverage=max_leverage,
            allow_short=allow_short,
            max_long_leverage=max_long_leverage,
            max_short_leverage=max_short_leverage,
        )
    blocked_by_profit_gate = {
        name: recent_return
        for name, recent_return in profit_returns.items()
        if recent_return <= float(profit_gate_min_return)
    }

    candidate_scores = {
        name: score for name, score in scores.items() if name not in blocked_by_profit_gate
    }
    if not candidate_scores:
        print(
            f"[{TAG}] model selection ({selection_mode}/{metric}): "
            + " ".join(f"{k}={v:.4f}" for k, v in scores.items())
            + " -> cash (profit_gate, "
            + f"lookback={profit_gate_lookback_hours}h, min_return={profit_gate_min_return:.4f}, "
            + "blocked="
            + ",".join(
                f"{k}:{v:.4f}" for k, v in sorted(blocked_by_profit_gate.items())
            )
            + ")"
        )
        _log_event(
            "model_select",
            mode=selection_mode,
            metric=metric,
            threshold=cash_threshold,
            switch_margin=switch_margin,
            min_score_gap=min_score_gap,
            profit_gate_mode=profit_gate_mode,
            profit_gate_lookback_hours=int(profit_gate_lookback_hours),
            profit_gate_min_return=float(profit_gate_min_return),
            profit_returns=profit_returns,
            blocked_models=blocked_by_profit_gate,
            score_gap=None,
            scores=scores,
            selected="cash",
        )
        return ""

    ranked = sorted(((float(score), name) for name, score in candidate_scores.items()), reverse=True)
    best_score, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else float("-inf")
    score_gap = float(best_score - second_score) if np.isfinite(second_score) else float("inf")

    current_model = str(current_model or "")
    current_score = float(candidate_scores[current_model]) if current_model in candidate_scores else None

    # Hysteresis: only switch model when challenger is better by switch_margin.
    if current_score is not None and current_model in candidate_scores and current_model != best:
        if (best_score - current_score) <= switch_margin:
            best = current_model
            best_score = current_score

    should_cash = selection_mode == "winner_cash" and (best_score <= cash_threshold or score_gap < min_score_gap)
    if should_cash:
        # Hysteresis around cash boundary.
        if (
            current_score is not None
            and current_model in scores
            and current_model == best
            and current_score > (cash_threshold - switch_margin)
            and score_gap >= max(0.0, min_score_gap - switch_margin)
        ):
            print(
                f"[{TAG}] model selection ({selection_mode}/{metric}): "
                + " ".join(f"{k}={v:.4f}" for k, v in scores.items())
                + (
                    " recent_return="
                    + " ".join(f"{k}={v:.4f}" for k, v in profit_returns.items())
                    if profit_returns
                    else ""
                )
                + f" gap={score_gap:.4f} -> {current_model} (cash blocked by hysteresis)"
            )
            _log_event(
                "model_select",
                mode=selection_mode,
                metric=metric,
                threshold=cash_threshold,
                switch_margin=switch_margin,
                min_score_gap=min_score_gap,
                profit_gate_mode=profit_gate_mode,
                profit_gate_lookback_hours=int(profit_gate_lookback_hours),
                profit_gate_min_return=float(profit_gate_min_return),
                profit_returns=profit_returns,
                blocked_models=blocked_by_profit_gate,
                score_gap=score_gap,
                scores=scores,
                selected=current_model,
            )
            return current_model
        reason = "threshold" if best_score <= cash_threshold else "low_gap"
        print(
            f"[{TAG}] model selection ({selection_mode}/{metric}): "
            + " ".join(f"{k}={v:.4f}" for k, v in scores.items())
            + (
                " recent_return="
                + " ".join(f"{k}={v:.4f}" for k, v in profit_returns.items())
                if profit_returns
                else ""
            )
            + f" gap={score_gap:.4f} -> cash ({reason}, threshold={cash_threshold:.4f}, min_gap={min_score_gap:.4f})"
        )
        _log_event(
            "model_select",
            mode=selection_mode,
            metric=metric,
            threshold=cash_threshold,
            switch_margin=switch_margin,
            min_score_gap=min_score_gap,
            profit_gate_mode=profit_gate_mode,
            profit_gate_lookback_hours=int(profit_gate_lookback_hours),
            profit_gate_min_return=float(profit_gate_min_return),
            profit_returns=profit_returns,
            blocked_models=blocked_by_profit_gate,
            score_gap=score_gap,
            scores=scores,
            selected="cash",
        )
        return ""
    print(
        f"[{TAG}] model selection ({selection_mode}/{metric}): "
        + " ".join(f"{k}={v:.4f}" for k, v in scores.items())
        + (
            " recent_return="
            + " ".join(f"{k}={v:.4f}" for k, v in profit_returns.items())
            if profit_returns
            else ""
        )
        + f" gap={score_gap:.4f} -> {best}"
    )
    _log_event(
        "model_select",
        mode=selection_mode,
        metric=metric,
        threshold=cash_threshold,
        switch_margin=switch_margin,
        min_score_gap=min_score_gap,
        profit_gate_mode=profit_gate_mode,
        profit_gate_lookback_hours=int(profit_gate_lookback_hours),
        profit_gate_min_return=float(profit_gate_min_return),
        profit_returns=profit_returns,
        blocked_models=blocked_by_profit_gate,
        score_gap=score_gap,
        scores=scores,
        selected=best,
    )
    return best


def main():
    parser = argparse.ArgumentParser(description="Meta-switcher margin bot")
    parser.add_argument("--model-a-name", default=MODEL_SLOT_DEFAULTS[0][1])
    parser.add_argument("--model-a-symbol", default=MODEL_SLOT_DEFAULTS[0][2]["symbol"])
    parser.add_argument("--model-a-data-symbol", default=MODEL_SLOT_DEFAULTS[0][2]["data_symbol"])
    parser.add_argument("--model-a-base-asset", default=MODEL_SLOT_DEFAULTS[0][2]["base_asset"])
    parser.add_argument("--model-a-maker-fee", type=float, default=MODEL_SLOT_DEFAULTS[0][2]["maker_fee"])
    parser.add_argument("--model-a-checkpoint", "--doge-checkpoint", dest="model_a_checkpoint", required=True)
    parser.add_argument("--model-b-name", default=MODEL_SLOT_DEFAULTS[1][1])
    parser.add_argument("--model-b-symbol", default=MODEL_SLOT_DEFAULTS[1][2]["symbol"])
    parser.add_argument("--model-b-data-symbol", default=MODEL_SLOT_DEFAULTS[1][2]["data_symbol"])
    parser.add_argument("--model-b-base-asset", default=MODEL_SLOT_DEFAULTS[1][2]["base_asset"])
    parser.add_argument("--model-b-maker-fee", type=float, default=MODEL_SLOT_DEFAULTS[1][2]["maker_fee"])
    parser.add_argument("--model-b-checkpoint", "--aave-checkpoint", dest="model_b_checkpoint", required=True)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--max-long-leverage", type=float, default=None)
    parser.add_argument("--max-short-leverage", type=float, default=None)
    parser.add_argument("--lookback", type=int, default=12, help="trailing hours for model selection")
    parser.add_argument(
        "--selection-metric",
        default="calmar",
        choices=SUPPORTED_SELECTION_METRICS,
        help="score used for meta model selection",
    )
    parser.add_argument(
        "--selection-mode",
        default="winner",
        choices=SUPPORTED_SELECTION_MODES,
        help="winner: always choose top model; winner_cash: stay flat when best score <= cash-threshold",
    )
    parser.add_argument(
        "--cash-threshold",
        type=float,
        default=0.0,
        help="only used in winner_cash mode; if best score <= threshold then sit out in cash",
    )
    parser.add_argument(
        "--switch-margin",
        type=float,
        default=0.0,
        help="hysteresis margin: challenger score must beat current score by this amount to switch",
    )
    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=0.0,
        help="confidence gate: in winner_cash mode, stay in cash if (best - second_best) < min-score-gap",
    )
    parser.add_argument(
        "--profit-gate-mode",
        default="hypothetical",
        choices=SUPPORTED_PROFIT_GATE_MODES,
        help="profit gate source: hypothetical score replay or live-like 5m execution replay",
    )
    parser.add_argument(
        "--profit-gate-lookback-hours",
        type=int,
        default=0,
        help="when > 0, block models whose trailing simulated return over this many hours is <= --profit-gate-min-return",
    )
    parser.add_argument(
        "--profit-gate-min-return",
        type=float,
        default=0.0,
        help="minimum trailing simulated return required by --profit-gate-lookback-hours; 0.0 means strictly profitable",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--intensity-scale", type=float, default=5.0)
    parser.add_argument("--min-gap-pct", type=float, default=0.0003)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--expiry-minutes", type=int, default=90)
    parser.add_argument("--price-tolerance", type=float, default=0.0008)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--forecast-cache", default="binanceneural/forecast_cache")
    parser.add_argument("--forecast-model-id", default="amazon/chronos-t5-small")
    parser.add_argument("--forecast-horizons", default="1")
    parser.add_argument("--cycle-minutes", type=int, default=5)
    parser.add_argument("--fast-check-seconds", type=int, default=30)
    parser.add_argument(
        "--max-position-notional",
        type=float,
        default=None,
        help="hard cap on gross symbol notional for probe trading; applies to both initial entries and add-ons",
    )
    parser.add_argument(
        "--disable-probe-cap",
        action="store_true",
        help="disable the implicit live probe cap when --max-position-notional is omitted",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.max_position_notional = _resolve_max_position_notional(
        args.max_position_notional,
        dry_run=args.dry_run,
        disable_probe_cap=bool(args.disable_probe_cap),
    )
    if args.max_position_notional == LIVE_PROBE_MAX_POSITION_NOTIONAL and not args.dry_run:
        print(f"[{TAG}] live probe cap enabled by default: max_position_notional={args.max_position_notional:.2f}")
    elif args.max_position_notional is None and not args.dry_run:
        print(f"[{TAG}] probe cap disabled: max_position_notional=None")

    model_specs = build_model_specs_from_args(args)
    apply_model_specs(model_specs)

    data_root = Path(args.data_root)
    forecast_cache = Path(args.forecast_cache)
    requested_forecast_horizons = tuple(int(h) for h in args.forecast_horizons.split(",") if str(h).strip())
    signal_interval = args.cycle_minutes * 60
    fast_interval = args.fast_check_seconds
    long_max_leverage, short_max_leverage = _resolve_directional_leverages(
        args.max_leverage,
        max_long_leverage=args.max_long_leverage,
        max_short_leverage=args.max_short_leverage,
    )
    if not args.allow_short:
        short_max_leverage = 0.0

    models_loaded = {}
    combined_feature_columns: list[str] = []
    for spec in model_specs:
        checkpoint = spec.get("checkpoint")
        if checkpoint is None:
            raise ValueError(f"Missing checkpoint for {spec['name']}")
        print(f"[{TAG}] loading {str(spec['name']).upper()} model from {checkpoint}")
        model, norm, feat, _cfg = load_policy_checkpoint(
            str(checkpoint),
            data_root=Path(args.data_root),
            forecast_cache_root=Path(args.forecast_cache),
        )
        models_loaded[str(spec["name"])] = (model, norm, feat)
        combined_feature_columns.extend(list(feat))

    forecast_horizons = resolve_required_forecast_horizons(
        requested_forecast_horizons,
        feature_columns=combined_feature_columns,
        fallback_horizons=(int(args.horizon),),
    )
    if forecast_horizons != requested_forecast_horizons:
        print(
            f"[{TAG}] expanded forecast horizons from {requested_forecast_horizons} "
            f"to {forecast_horizons} based on checkpoint features"
        )
    runtime_signature = _runtime_signature(args, forecast_horizons=forecast_horizons, model_specs=model_specs)
    runtime_snapshot = _load_runtime_snapshot()

    # Resolve rules for both symbols
    rules = {str(spec["name"]): resolve_symbol_rules(str(spec["symbol"])) for spec in model_specs}

    # Signal histories for both models
    histories = {name: SignalHistory() for name in MODELS}

    state = MetaState.load()
    signals = {}
    last_signal_time = 0.0
    snapshot_entry_order = _empty_exit()
    snapshot_exit_order = _empty_exit()
    snapshot_compatible = _runtime_snapshot_is_compatible(runtime_snapshot, runtime_signature)
    if snapshot_compatible:
        state = _merge_state_with_snapshot(state, runtime_snapshot.get("state"))
        snapshot_entry_order = _deserialize_order(runtime_snapshot.get("entry_order"))
        snapshot_exit_order = _deserialize_order(runtime_snapshot.get("exit_order"))
        latest_signal_hours = {
            name: _normalize_hour_like(_latest_price_timestamp(MODELS[name]["data_symbol"], data_root))
            for name in MODELS
        }
        if _snapshot_signals_are_current(runtime_snapshot.get("signals", {}), latest_signal_hours):
            histories = _restore_histories_from_snapshot(runtime_snapshot.get("histories", {}))
            signals = _restore_signals_from_snapshot(runtime_snapshot.get("signals", {}))
            if len(signals) == len(MODELS):
                last_signal_time = time.monotonic()
                restored_hours = {
                    name: _serialize_hour_like(signal.get("signal_hour"))
                    for name, signal in signals.items()
                }
                print(f"[{TAG}] restored runtime snapshot for {', '.join(sorted(signals))}: {restored_hours}")
                _log_event("runtime_snapshot_restored", signal_hours=restored_hours)
            else:
                signals = {}
                last_signal_time = 0.0
        else:
            print(f"[{TAG}] runtime snapshot is stale for current market hour, bootstrapping fresh history")

    bootstrap_hours = max(
        int(args.lookback),
        int(args.profit_gate_lookback_hours),
        int(args.max_hold_hours or 0),
        SIGNAL_BOOTSTRAP_EXTRA_HOURS,
    ) + SIGNAL_BOOTSTRAP_EXTRA_HOURS
    for name, (model, norm, feat) in models_loaded.items():
        if name in signals:
            continue
        mcfg = MODELS[name]
        try:
            frame = _load_fresh_live_frame(
                symbol=mcfg["symbol"],
                data_symbol=mcfg["data_symbol"],
                data_root=data_root,
                forecast_cache=forecast_cache,
                forecast_horizons=forecast_horizons,
                forecast_model_id=args.forecast_model_id,
                sequence_length=args.sequence_length,
            )
            seeded = _bootstrap_signal_history_from_frame(
                histories[name],
                frame,
                model,
                norm,
                feat,
                horizon=args.horizon,
                sequence_length=args.sequence_length,
                intensity_scale=args.intensity_scale,
                history_hours=bootstrap_hours,
                symbol=mcfg["symbol"],
            )
            if seeded:
                signals[name] = seeded[-1]
                last_signal = signals[name]
                print(
                    f"[{TAG}] bootstrapped {name} history with {len(seeded)} points "
                    f"through {last_signal.get('signal_hour')}"
                )
                _log_event(
                    "signal_bootstrap",
                    model=name,
                    points=len(seeded),
                    last_signal_hour=last_signal.get("signal_hour"),
                    buy_price=last_signal["buy_price"],
                    sell_price=last_signal["sell_price"],
                    buy_amount=last_signal["buy_amount"],
                    sell_amount=last_signal["sell_amount"],
                )
        except Exception as exc:
            print(f"[{TAG}] bootstrap history failed for {name}: {exc}")
    if signals:
        last_signal_time = time.monotonic()

    last_managed_conflict_models: tuple[str, ...] | None = None

    # Startup orphan detection
    try:
        managed_positions = _reconcile_managed_positions(
            state,
            rules,
            max_position_notional=args.max_position_notional,
        )
        if managed_positions["conflict"]:
            detail = managed_positions["details"]
            print(f"[{TAG}] WARNING: multiple managed positions detected: {detail}")
            _log_event("managed_position_conflict", positions=detail)
            last_managed_conflict_models = tuple(sorted(detail))
    except Exception as exc:
        print(f"[{TAG}] orphan check failed: {exc}")

    if state.in_position and state.active_model in MODELS:
        try:
            _normalize_state_open_ts_for_position(
                state,
                MODELS[state.active_model]["symbol"],
                max_hold_hours=args.max_hold_hours,
            )
        except Exception as exc:
            print(f"[{TAG}] open_ts normalization failed: {exc}")

    # Recover orders from exchange on startup
    exit_order = _empty_exit()
    entry_order = _empty_exit()
    if state.active_model in MODELS:
        sym = MODELS[state.active_model]["symbol"]
        if state.in_position:
            exit_order = _merge_recovered_order_with_snapshot(
                _recover_exit_orders(sym, position_side=state.position_side),
                snapshot_exit_order,
            )
            add_side = "sell" if str(state.position_side).lower() == "short" else "buy"
            entry_order = _merge_recovered_order_with_snapshot(
                _recover_entry_orders(sym, entry_side=add_side),
                snapshot_entry_order,
            )
        else:
            snapshot_side = snapshot_entry_order.get("side") if snapshot_entry_order.get("id") else None
            entry_order = _merge_recovered_order_with_snapshot(
                _recover_entry_orders(sym, entry_side=snapshot_side),
                snapshot_entry_order,
            )

    def persist_runtime() -> None:
        _save_runtime_snapshot(
            state=state,
            histories=histories,
            signals=signals,
            entry_order=entry_order,
            exit_order=exit_order,
            signature=runtime_signature,
        )

    def persist_and_sleep(seconds: int = fast_interval) -> None:
        persist_runtime()
        time.sleep(seconds)

    persist_runtime()

    print(
        f"[{TAG}] meta-switcher ready, models={','.join(MODELS)}, lookback={args.lookback}, leverage={args.max_leverage}x, "
        f"mode={args.selection_mode}, metric={args.selection_metric}, "
        f"cash_threshold={args.cash_threshold}, switch_margin={args.switch_margin}, "
        f"min_score_gap={args.min_score_gap}, "
        f"profit_gate={args.profit_gate_mode}:{args.profit_gate_lookback_hours}h>{args.profit_gate_min_return:.4f}, "
        f"forecast_model_id={args.forecast_model_id}, "
        f"max_position_notional={args.max_position_notional}, "
        f"allow_short={bool(args.allow_short)}, max_long_leverage={args.max_long_leverage}, "
        f"max_short_leverage={args.max_short_leverage}"
    )
    if state.active_model:
        print(f"[{TAG}] resuming with active_model={state.active_model}, in_position={state.in_position}")

    while True:
        now = time.monotonic()

        # Refresh signals from both models
        if now - last_signal_time >= signal_interval or not signals:
            for name, (model, norm, feat) in models_loaded.items():
                mcfg = MODELS[name]
                try:
                    sig = _refresh_signal(
                        model, norm, feat,
                        horizon=args.horizon, sequence_length=args.sequence_length,
                        intensity_scale=args.intensity_scale,
                        data_root=data_root, forecast_cache=forecast_cache,
                        forecast_horizons=forecast_horizons,
                        forecast_model_id=args.forecast_model_id,
                        data_symbol=mcfg["data_symbol"], symbol=mcfg["symbol"],
                    )
                    signals[name] = sig
                    _upsert_signal_history(histories[name], sig)
                    print(f"[{TAG}] {name}: buy={sig['buy_price']:.4f}({sig['buy_amount']:.0f}%) sell={sig['sell_price']:.4f}({sig['sell_amount']:.0f}%)")
                    _log_event(
                        "signal_refresh",
                        model=name,
                        buy_price=sig["buy_price"],
                        sell_price=sig["sell_price"],
                        buy_amount=sig["buy_amount"],
                        sell_amount=sig["sell_amount"],
                        close=sig["close"],
                        signal_hour=sig.get("signal_hour"),
                    )
                except Exception as exc:
                    print(f"[{TAG}] {name} signal error: {exc}")
            last_signal_time = now
            persist_runtime()

        if not signals:
            print(f"[{TAG}] no signals yet, sleeping...")
            persist_and_sleep()
            continue

        managed_positions = {
            "positions": [],
            "details": {},
            "chosen_model": "",
            "chosen_position": None,
            "conflict": False,
        }
        try:
            managed_positions = _reconcile_managed_positions(
                state,
                rules,
                max_position_notional=args.max_position_notional,
            )
            if managed_positions["conflict"]:
                detail = managed_positions["details"]
                conflict_models = tuple(sorted(detail))
                if conflict_models != last_managed_conflict_models:
                    print(f"[{TAG}] WARNING: multiple managed positions detected: {detail}")
                    _log_event("managed_position_conflict", positions=detail)
                last_managed_conflict_models = conflict_models
            else:
                last_managed_conflict_models = None
        except Exception as exc:
            print(f"[{TAG}] managed position reconcile failed: {exc}")

        managed_conflict = bool(managed_positions["conflict"])
        managed_position_detail = managed_positions["details"]
        if managed_conflict and entry_order["id"]:
            order_label = "add" if state.in_position else "entry"
            print(f"[{TAG}] pausing new risk while managed positions conflict; cancelling working {order_label}")
            _cancel_tracked_order(entry_order, order_label)
            entry_order = _empty_exit()

        # Determine active model
        active = state.active_model
        active_cfg = MODELS.get(active, {})
        active_symbol = active_cfg.get("symbol", "")
        active_base = active_cfg.get("base_asset", "")

        if state.in_position and active_symbol:
            try:
                _normalize_state_open_ts_for_position(
                    state,
                    active_symbol,
                    max_hold_hours=args.max_hold_hours,
                )
            except Exception as exc:
                print(f"[{TAG}] open_ts normalization failed: {exc}")

        if active and active_symbol:
            r = rules[active]
            bal = _get_margin_equity_for(active_symbol, active_base)
            equity = bal["equity"]
            market_price = bal["market_price"]
            asset_total = bal["asset_total"]
            asset_free = bal["asset_free"]
            asset_locked = bal["asset_locked"]
            asset_value = bal["asset_value"]
            asset_net = bal["asset_net"]
            position_side = str(bal.get("position_side", "") or "")
            position_value = float(bal.get("position_value", asset_value))
            usdt_free = bal["usdt_free"]

            if market_price <= 0:
                print(f"[{TAG}] price fetch failed for {active_symbol}, skipping cycle")
                persist_and_sleep()
                continue

            # Detect position
            if _has_effective_position(
                asset_net,
                position_value,
                step_size=r.step_size,
                max_position_notional=args.max_position_notional,
            ):
                if not state.in_position:
                    entry_order = _promote_detected_position(
                        state,
                        model=active,
                        position_side=position_side,
                        market_price=market_price,
                        asset_total=asset_net,
                        asset_value=position_value,
                        entry_order=entry_order,
                    )
            else:
                if state.in_position:
                    print(f"[{TAG}] {active} position closed")
                    _log_event("position_closed", model=active)
                    _cancel_tracked_order(exit_order, "exit")
                    exit_order = _empty_exit()
                    _cancel_tracked_order(entry_order, "add")
                    entry_order = _empty_exit()
                    state.in_position = False
                    state.position_side = ""
                    state.open_ts = None
                    state.open_price = 0.0
                    profit_gate_returns = None
                    if int(args.profit_gate_lookback_hours) > 0 and str(args.profit_gate_mode) == "live_like":
                        profit_gate_returns = compute_live_like_profit_gate_returns(
                            histories,
                            asof_ts=datetime.now(timezone.utc),
                            lookback_hours=int(args.profit_gate_lookback_hours),
                            args=args,
                            rules_by_model=rules,
                            initial_cash=max(float(equity), 0.0),
                        )
                    # Select new model for next trade
                    new_model = select_model(
                        histories,
                        args.lookback,
                        args.max_leverage,
                        metric=args.selection_metric,
                        selection_mode=args.selection_mode,
                        cash_threshold=args.cash_threshold,
                        current_model=active,
                        switch_margin=args.switch_margin,
                        min_score_gap=args.min_score_gap,
                        profit_gate_mode=args.profit_gate_mode,
                        profit_gate_lookback_hours=args.profit_gate_lookback_hours,
                        profit_gate_min_return=args.profit_gate_min_return,
                        profit_gate_returns=profit_gate_returns,
                        allow_short=bool(args.allow_short),
                        max_long_leverage=args.max_long_leverage,
                        max_short_leverage=args.max_short_leverage,
                    )
                    if new_model and new_model != active:
                        print(f"[{TAG}] SWITCHING {active} -> {new_model}")
                        _log_event("switch", from_model=active, to_model=new_model)
                    elif not new_model:
                        print(f"[{TAG}] SWITCHING {active} -> cash")
                        _log_event("switch", from_model=active, to_model="cash")
                    state.active_model = new_model
                    active = new_model
                    if active:
                        active_cfg = MODELS[active]
                        active_symbol = active_cfg["symbol"]
                        active_base = active_cfg["base_asset"]
                    else:
                        active_cfg = {}
                        active_symbol = ""
                        active_base = ""
                    state.save()
                if active_symbol and not entry_order["id"]:
                    _repay_outstanding(active_symbol)
                    if args.allow_short:
                        _repay_outstanding(active_symbol, asset=active_base)

            current_leverage = position_value / equity if equity > 0 else 0.0
            print(
                f"[{TAG}] [{active}] eq=${equity:.2f} {active_base}_net={asset_net:.4f} "
                f"side={position_side or 'flat'} pos=${position_value:.2f} lev={current_leverage:.1f}x price={market_price:.4f}"
            )
            _log_event("state", model=active, equity=equity, market_price=market_price,
                       asset_value=asset_value, position_value=position_value, asset_net=asset_net,
                       position_side=position_side, leverage=current_leverage,
                       in_position=state.in_position, hours_held=state.hours_held(),
                       managed_positions=managed_position_detail, managed_conflict=managed_conflict)

            sig = signals.get(active)
            if not sig:
                print(f"[{TAG}] no signal for {active}")
                persist_and_sleep()
                continue
            long_sig = directional_signal(sig, side="long")
            short_sig = directional_signal(sig, side="short")
            validated_levels = None
            if sig.get("buy_price", 0) > 0 and sig.get("sell_price", 0) > 0:
                validated_levels = _ensure_valid_levels(
                    active_symbol,
                    sig["buy_price"],
                    sig["sell_price"],
                    min_gap_pct=args.min_gap_pct,
                    rules=r,
                )
            if validated_levels:
                validated_buy, validated_sell = validated_levels
                long_sig = directional_signal(
                    {
                        "buy_price": validated_buy,
                        "sell_price": validated_sell,
                        "buy_amount": sig["buy_amount"],
                        "sell_amount": sig["sell_amount"],
                    },
                    side="long",
                )
                short_sig = directional_signal(
                    {
                        "buy_price": validated_buy,
                        "sell_price": validated_sell,
                        "buy_amount": sig["buy_amount"],
                        "sell_amount": sig["sell_amount"],
                    },
                    side="short",
                )
            active_position_side = str(state.position_side or position_side or "")
            flat_entry_side = choose_flat_entry_side(sig, allow_short=bool(args.allow_short and short_max_leverage > 0))
            sig_hour = sig.get("signal_hour")
            if exit_order.get("id") and exit_order.get("signal_hour") is None:
                exit_order["signal_hour"] = sig_hour
            if entry_order.get("id") and entry_order.get("signal_hour") is None:
                entry_order["signal_hour"] = sig_hour

            if state.in_position:
                hours = state.hours_held()
                force_close = args.max_hold_hours is not None and hours >= args.max_hold_hours
                position_side_name = active_position_side or "long"
                exit_side = "buy" if position_side_name == "short" else "sell"
                add_side = "sell" if position_side_name == "short" else "buy"
                exit_sig = short_sig if position_side_name == "short" else long_sig
                add_sig = short_sig if position_side_name == "short" else long_sig
                desired_add_notional = 0.0
                desired_add_qty = 0.0
                if add_sig.entry_amount > 0 and add_sig.entry_price > 0 and equity > 0:
                    max_borrowable_asset = get_max_borrowable(active_base) if position_side_name == "short" and short_max_leverage > 0 else 0.0
                    max_borrowable_usdt = get_max_borrowable("USDT") if position_side_name == "long" and long_max_leverage > 1.0 else 0.0
                    desired_add_notional = _remaining_target_entry_notional(
                        side=position_side_name,
                        equity=equity,
                        asset_net=asset_net,
                        market_price=market_price,
                        usdt_free=usdt_free,
                        asset_free=asset_free,
                        max_borrowable_usdt=max_borrowable_usdt,
                        max_borrowable_asset=max_borrowable_asset,
                        long_max_leverage=long_max_leverage,
                        short_max_leverage=short_max_leverage,
                        max_position_notional=args.max_position_notional,
                    )
                    desired_add_qty = quantize_qty(
                        desired_add_notional / add_sig.entry_price,
                        step_size=r.step_size,
                    )

                if exit_order["id"]:
                    if exit_order["symbol"] != active_symbol or str(exit_order.get("side", "")) != exit_side:
                        print(f"[{TAG}] exit order mismatch, cancelling")
                        _cancel_tracked_order(exit_order, "exit")
                        exit_order = _empty_exit()
                    else:
                        oinfo = _check_margin_order(active_symbol, exit_order["id"])
                        if oinfo:
                            ostatus = oinfo.get("status", "")
                            if ostatus == "FILLED":
                                print(f"[{TAG}] exit FILLED id={exit_order['id']} price={exit_order['price']:.4f}")
                                _log_event(
                                    "exit_filled",
                                    model=active,
                                    position_side=position_side_name,
                                    side=exit_side,
                                    price=exit_order["price"],
                                    qty=exit_order["qty"],
                                )
                                exit_order = _empty_exit()
                            elif ostatus in ("CANCELED", "REJECTED", "EXPIRED"):
                                print(f"[{TAG}] exit order {ostatus} id={exit_order['id']}")
                                exit_order = _empty_exit()
                            elif force_close:
                                print(f"[{TAG}] FORCE CLOSE: cancelling exit at {exit_order['price']:.4f}")
                                _cancel_tracked_order(exit_order, "exit")
                                exit_order = _empty_exit()
                                time.sleep(0.3)
                            elif exit_sig.exit_price > 0:
                                price_diff = abs(exit_order["price"] - exit_sig.exit_price) / max(exit_order["price"], 1e-9)
                                if _is_same_signal_hour(exit_order, sig):
                                    print(
                                        f"[{TAG}] exit working (sticky hour): id={exit_order['id']} "
                                        f"{exit_order['qty']:.4f}@{exit_order['price']:.4f} ({hours:.1f}h)"
                                    )
                                    persist_and_sleep()
                                    continue
                                if _maker_limit_ok(exit_side, exit_sig.exit_price, market_price) and price_diff > ORDER_REPRICE_THRESHOLD:
                                    print(
                                        f"[{TAG}] updating exit: {exit_order['price']:.4f} -> {exit_sig.exit_price:.4f} "
                                        f"(diff={price_diff:.4f})"
                                    )
                                    _cancel_tracked_order(exit_order, "exit")
                                    exit_order = _empty_exit()
                                    time.sleep(0.3)
                                else:
                                    print(
                                        f"[{TAG}] exit working: id={exit_order['id']} "
                                        f"{exit_order['qty']:.4f}@{exit_order['price']:.4f} ({hours:.1f}h)"
                                    )
                                    persist_and_sleep()
                                    continue
                            else:
                                print(
                                    f"[{TAG}] exit working: id={exit_order['id']} "
                                    f"{exit_order['qty']:.4f}@{exit_order['price']:.4f} ({hours:.1f}h)"
                                )
                                persist_and_sleep()
                                continue
                        else:
                            recovered = _recover_exit_orders(active_symbol, position_side=position_side_name)
                            if recovered["id"]:
                                recovered["signal_hour"] = (
                                    recovered.get("signal_hour") or exit_order.get("signal_hour") or sig.get("signal_hour")
                                )
                                exit_order = recovered
                                print(f"[{TAG}] recovered working exit after order-check failure")
                                persist_and_sleep()
                                continue
                            print(f"[{TAG}] exit order check unavailable, keeping tracker to avoid duplicate replace")
                            persist_and_sleep()
                            continue

                if not exit_order["id"]:
                    recovered = _recover_exit_orders(active_symbol, position_side=position_side_name)
                    if recovered["id"]:
                        recovered["signal_hour"] = recovered.get("signal_hour") or sig.get("signal_hour")
                        exit_order = recovered
                        print(f"[{TAG}] using recovered working exit order")
                        persist_and_sleep()
                        continue
                    if force_close:
                        print(f"[{TAG}] FORCE CLOSE {active}/{position_side_name} after {hours:.1f}h")
                        target_exit_price = market_price * (1.001 if position_side_name == "short" else 0.999)
                        exit_qty = quantize_qty(abs(asset_net), step_size=r.step_size)
                        if exit_qty > 0:
                            try:
                                exit_order = _place_direct_exit(
                                    active_symbol,
                                    exit_qty,
                                    target_exit_price,
                                    r,
                                    position_side=position_side_name,
                                    kind="force_exit",
                                )
                                if exit_order:
                                    exit_order["signal_hour"] = sig.get("signal_hour")
                                    _log_event(
                                        "force_close",
                                        model=active,
                                        position_side=position_side_name,
                                        price=target_exit_price,
                                        qty=exit_qty,
                                    )
                                else:
                                    exit_order = _empty_exit()
                            except Exception as exc:
                                print(f"[{TAG}] force-close order failed: {exc}")
                                exit_order = _empty_exit()
                    elif exit_sig.exit_amount > 0 and exit_sig.exit_price > 0:
                        if not _maker_limit_ok(exit_side, exit_sig.exit_price, market_price):
                            print(
                                f"[{TAG}] exit {exit_side}={exit_sig.exit_price:.4f} not maker vs market={market_price:.4f}, skipping"
                            )
                        else:
                            if position_side_name == "short":
                                exit_qty = quantize_qty(
                                    abs(asset_net) * max(0.0, min(1.0, exit_sig.exit_amount / 100.0)),
                                    step_size=r.step_size,
                                )
                            else:
                                exit_qty = quantize_qty(
                                    max(0.0, asset_free) * max(0.0, min(1.0, exit_sig.exit_amount / 100.0)),
                                    step_size=r.step_size,
                                )
                            if exit_qty > 0:
                                try:
                                    exit_order = _place_direct_exit(
                                        active_symbol,
                                        exit_qty,
                                        exit_sig.exit_price,
                                        r,
                                        position_side=position_side_name,
                                    )
                                    if exit_order:
                                        exit_order["signal_hour"] = sig.get("signal_hour")
                                        _log_event(
                                            "exit_attempt",
                                            model=active,
                                            position_side=position_side_name,
                                            side=exit_side,
                                            price=exit_sig.exit_price,
                                            qty=exit_qty,
                                        )
                                    else:
                                        exit_order = _empty_exit()
                                except Exception as exc:
                                    print(f"[{TAG}] exit order failed: {exc}")
                                    exit_order = _empty_exit()
                            else:
                                print(f"[{TAG}] holding {active}/{position_side_name} ({hours:.1f}h), exit qty too small")
                    else:
                        print(f"[{TAG}] holding {active}/{position_side_name} ({hours:.1f}h), no exit signal")

                if entry_order["id"]:
                    expected_side = add_side
                    if entry_order["symbol"] != active_symbol or str(entry_order.get("side", "")) != expected_side:
                        _cancel_tracked_order(entry_order, "add")
                        entry_order = _empty_exit()
                    else:
                        oinfo = _check_margin_order(active_symbol, entry_order["id"])
                        if oinfo:
                            ostatus = oinfo.get("status", "")
                            if ostatus == "FILLED":
                                print(f"[{TAG}] add FILLED id={entry_order['id']}")
                                _log_event(
                                    "add_filled",
                                    model=active,
                                    position_side=position_side_name,
                                    side=expected_side,
                                    price=entry_order["price"],
                                    qty=entry_order["qty"],
                                )
                                entry_order = _empty_exit()
                            elif ostatus in ("CANCELED", "REJECTED", "EXPIRED"):
                                entry_order = _empty_exit()
                            elif add_sig.entry_price > 0:
                                price_diff = abs(entry_order["price"] - add_sig.entry_price) / max(entry_order["price"], 1e-9)
                                needs_resize = _order_needs_resize_to_target(
                                    entry_order,
                                    target_qty=desired_add_qty,
                                    reference_price=add_sig.entry_price,
                                    step_size=r.step_size,
                                    min_notional=r.min_notional,
                                )
                                if _is_same_signal_hour(entry_order, sig):
                                    if needs_resize:
                                        print(
                                            f"[{TAG}] resizing add during sticky hour: "
                                            f"{entry_order['qty']:.4f} -> {desired_add_qty:.4f}"
                                        )
                                        _cancel_tracked_order(entry_order, "add")
                                        entry_order = _empty_exit()
                                        time.sleep(0.3)
                                    else:
                                        print(
                                            f"[{TAG}] add working (sticky hour): id={entry_order['id']} "
                                            f"{entry_order['qty']:.4f}@{entry_order['price']:.4f}"
                                        )
                                        persist_and_sleep()
                                        continue
                                elif needs_resize or (
                                    _maker_limit_ok(add_side, add_sig.entry_price, market_price) and price_diff > ORDER_REPRICE_THRESHOLD
                                ):
                                    print(
                                        f"[{TAG}] updating add: {entry_order['price']:.4f} -> {add_sig.entry_price:.4f} "
                                        f"(diff={price_diff:.4f}, qty={entry_order['qty']:.4f}->{desired_add_qty:.4f})"
                                    )
                                    _cancel_tracked_order(entry_order, "add")
                                    entry_order = _empty_exit()
                                    time.sleep(0.3)
                                else:
                                    print(
                                        f"[{TAG}] add working: id={entry_order['id']} "
                                        f"{entry_order['qty']:.4f}@{entry_order['price']:.4f}"
                                    )
                                    persist_and_sleep()
                                    continue
                        else:
                            recovered = _recover_entry_orders(active_symbol, entry_side=expected_side)
                            if recovered["id"]:
                                recovered["signal_hour"] = (
                                    recovered.get("signal_hour") or entry_order.get("signal_hour") or sig.get("signal_hour")
                                )
                                entry_order = recovered
                            else:
                                print(f"[{TAG}] add-order check unavailable, keeping tracker to avoid duplicate replace")

                if (
                    not managed_conflict
                    and not force_close
                    and desired_add_qty > 0
                    and add_sig.entry_amount > 0
                    and add_sig.entry_price > 0
                ):
                    if desired_add_notional > r.min_notional:
                        if not _maker_limit_ok(add_side, add_sig.entry_price, market_price):
                            print(
                                f"[{TAG}] add {add_side}={add_sig.entry_price:.4f} not maker vs market={market_price:.4f}, skipping"
                            )
                        elif not entry_order["id"]:
                            add_qty = desired_add_qty
                            if add_qty > 0 and (not r.min_notional or add_qty * add_sig.entry_price >= r.min_notional):
                                try:
                                    entry_order = _place_direct_entry(
                                        active_symbol,
                                        add_qty,
                                        add_sig.entry_price,
                                        r,
                                        entry_side=add_side,
                                    )
                                    if entry_order:
                                        entry_order["signal_hour"] = sig.get("signal_hour")
                                        print(
                                            f"[{TAG}] ADD {active}/{position_side_name} "
                                            f"{add_side}={add_sig.entry_price:.4f} qty={add_qty:.4f}"
                                        )
                                    else:
                                        entry_order = _empty_exit()
                                except Exception as exc:
                                    print(f"[{TAG}] add order failed: {exc}")
                                    entry_order = _empty_exit()
            else:
                desired_entry_side = ""
                desired_sig = None
                desired_position_side = ""
                if flat_entry_side == "short" and args.allow_short and short_max_leverage > 0:
                    desired_entry_side = "sell"
                    desired_sig = short_sig
                    desired_position_side = "short"
                elif flat_entry_side == "long":
                    desired_entry_side = "buy"
                    desired_sig = long_sig
                    desired_position_side = "long"
                desired_entry_notional = 0.0
                desired_entry_qty = 0.0
                if desired_sig and desired_sig.entry_amount > 0 and desired_sig.entry_price > 0 and equity > 0:
                    max_borrowable_asset = get_max_borrowable(active_base) if desired_position_side == "short" and short_max_leverage > 0 else 0.0
                    max_borrowable_usdt = get_max_borrowable("USDT") if desired_position_side == "long" and long_max_leverage > 1.0 else 0.0
                    desired_entry_notional = _remaining_target_entry_notional(
                        side=desired_position_side,
                        equity=equity,
                        asset_net=asset_net,
                        market_price=market_price,
                        usdt_free=usdt_free,
                        asset_free=asset_free,
                        max_borrowable_usdt=max_borrowable_usdt,
                        max_borrowable_asset=max_borrowable_asset,
                        long_max_leverage=long_max_leverage,
                        short_max_leverage=short_max_leverage,
                        max_position_notional=args.max_position_notional,
                    )
                    desired_entry_qty = quantize_qty(
                        desired_entry_notional / desired_sig.entry_price,
                        step_size=r.step_size,
                    )

                if entry_order["id"]:
                    if entry_order["symbol"] != active_symbol:
                        _cancel_tracked_order(entry_order, "entry")
                        entry_order = _empty_exit()
                    else:
                        oinfo = _check_margin_order(active_symbol, entry_order["id"])
                        if oinfo:
                            ostatus = oinfo.get("status", "")
                            if ostatus == "FILLED":
                                filled_side = "short" if str(entry_order.get("side", "")) == "sell" else "long"
                                print(f"[{TAG}] entry FILLED id={entry_order['id']} price={entry_order['price']:.4f}")
                                state.in_position = True
                                state.position_side = filled_side
                                if not state.open_ts:
                                    state.open_ts = datetime.now(timezone.utc).isoformat()
                                if float(entry_order["price"]) > 0:
                                    state.open_price = float(entry_order["price"])
                                state.save()
                                _log_event(
                                    "entry_filled",
                                    model=active,
                                    position_side=filled_side,
                                    side=entry_order.get("side"),
                                    price=entry_order["price"],
                                    qty=entry_order["qty"],
                                )
                                entry_order = _empty_exit()
                            elif ostatus in ("CANCELED", "REJECTED", "EXPIRED"):
                                print(f"[{TAG}] entry order {ostatus}")
                                entry_order = _empty_exit()
                            elif desired_sig and desired_sig.entry_price > 0:
                                price_diff = abs(entry_order["price"] - desired_sig.entry_price) / max(entry_order["price"], 1e-9)
                                needs_resize = (
                                    str(entry_order.get("side", "")) == desired_entry_side
                                    and _order_needs_resize_to_target(
                                        entry_order,
                                        target_qty=desired_entry_qty,
                                        reference_price=desired_sig.entry_price,
                                        step_size=r.step_size,
                                        min_notional=r.min_notional,
                                    )
                                )
                                if _is_same_signal_hour(entry_order, sig):
                                    if needs_resize:
                                        print(
                                            f"[{TAG}] resizing entry during sticky hour: "
                                            f"{entry_order['qty']:.4f} -> {desired_entry_qty:.4f}"
                                        )
                                        _cancel_tracked_order(entry_order, "entry")
                                        entry_order = _empty_exit()
                                        time.sleep(0.3)
                                    else:
                                        print(
                                            f"[{TAG}] entry working (sticky hour): id={entry_order['id']} "
                                            f"{entry_order['qty']:.4f}@{entry_order['price']:.4f}"
                                        )
                                        persist_and_sleep()
                                        continue
                                if (
                                    str(entry_order.get("side", "")) != desired_entry_side
                                    or needs_resize
                                    or (_maker_limit_ok(desired_entry_side, desired_sig.entry_price, market_price) and price_diff > ORDER_REPRICE_THRESHOLD)
                                ):
                                    print(
                                        f"[{TAG}] updating entry: {entry_order['price']:.4f} -> {desired_sig.entry_price:.4f} "
                                        f"(diff={price_diff:.4f}, qty={entry_order['qty']:.4f}->{desired_entry_qty:.4f})"
                                    )
                                    _cancel_tracked_order(entry_order, "entry")
                                    entry_order = _empty_exit()
                                    time.sleep(0.3)
                                else:
                                    print(
                                        f"[{TAG}] entry working: id={entry_order['id']} "
                                        f"{entry_order['qty']:.4f}@{entry_order['price']:.4f}"
                                    )
                                    persist_and_sleep()
                                    continue
                            else:
                                print(
                                    f"[{TAG}] entry working: id={entry_order['id']} "
                                    f"{entry_order['qty']:.4f}@{entry_order['price']:.4f}"
                                )
                                persist_and_sleep()
                                continue
                        else:
                            recovered = _recover_entry_orders(active_symbol, entry_side=desired_entry_side or None)
                            if recovered["id"]:
                                recovered["signal_hour"] = (
                                    recovered.get("signal_hour") or entry_order.get("signal_hour") or sig.get("signal_hour")
                                )
                                entry_order = recovered
                                print(f"[{TAG}] recovered working entry after order-check failure")
                                persist_and_sleep()
                                continue
                            print(f"[{TAG}] entry order check unavailable, keeping tracker to avoid duplicate replace")
                            persist_and_sleep()
                            continue

                if (
                    not managed_conflict
                    and not entry_order["id"]
                    and desired_sig
                    and desired_sig.entry_amount > 0
                    and desired_sig.entry_price > 0
                    and equity > 0
                ):
                    recovered = _recover_entry_orders(active_symbol, entry_side=desired_entry_side)
                    if recovered["id"]:
                        recovered["signal_hour"] = recovered.get("signal_hour") or sig.get("signal_hour")
                        entry_order = recovered
                        print(f"[{TAG}] using recovered working entry order")
                        persist_and_sleep()
                        continue
                    if not _maker_limit_ok(desired_entry_side, desired_sig.entry_price, market_price):
                        print(
                            f"[{TAG}] {desired_position_side} entry {desired_entry_side}={desired_sig.entry_price:.4f} "
                            f"not maker vs market={market_price:.4f}, skipping"
                        )
                    else:
                        entry_qty = desired_entry_qty
                        if entry_qty > 0:
                            try:
                                entry_order = _place_direct_entry(
                                    active_symbol,
                                    entry_qty,
                                    desired_sig.entry_price,
                                    r,
                                    entry_side=desired_entry_side,
                                )
                                if entry_order:
                                    entry_order["signal_hour"] = sig.get("signal_hour")
                                    _log_event(
                                        "entry_attempt",
                                        model=active,
                                        position_side=desired_position_side,
                                        side=desired_entry_side,
                                        price=desired_sig.entry_price,
                                        qty=entry_qty,
                                    )
                                else:
                                    entry_order = _empty_exit()
                            except Exception as exc:
                                print(f"[{TAG}] entry order failed: {exc}")
                                entry_order = _empty_exit()
                        else:
                            print(f"[{TAG}] {active} {desired_position_side} target already satisfied or qty too small")
                elif managed_conflict:
                    print(f"[{TAG}] flat entry paused while managed positions conflict: {managed_position_detail}")
                elif not entry_order["id"] and equity <= 0:
                    print(f"[{TAG}] no equity in margin account")
                elif not entry_order["id"]:
                    print(f"[{TAG}] {active} flat, no entry signal")
        else:
            # No active model yet, select one
            profit_gate_returns = None
            if int(args.profit_gate_lookback_hours) > 0 and str(args.profit_gate_mode) == "live_like":
                profit_gate_returns = compute_live_like_profit_gate_returns(
                    histories,
                    asof_ts=datetime.now(timezone.utc),
                    lookback_hours=int(args.profit_gate_lookback_hours),
                    args=args,
                    rules_by_model=rules,
                    initial_cash=max(_get_total_margin_equity(), 0.0),
                )
            active = select_model(
                histories,
                args.lookback,
                args.max_leverage,
                metric=args.selection_metric,
                selection_mode=args.selection_mode,
                cash_threshold=args.cash_threshold,
                current_model="",
                switch_margin=args.switch_margin,
                min_score_gap=args.min_score_gap,
                profit_gate_mode=args.profit_gate_mode,
                profit_gate_lookback_hours=args.profit_gate_lookback_hours,
                profit_gate_min_return=args.profit_gate_min_return,
                profit_gate_returns=profit_gate_returns,
                allow_short=bool(args.allow_short),
                max_long_leverage=args.max_long_leverage,
                max_short_leverage=args.max_short_leverage,
            )
            if active:
                state.active_model = active
                state.save()
                print(f"[{TAG}] initial model: {active}")
            else:
                state.active_model = ""
                state.save()
                print(f"[{TAG}] staying in cash (no model selected)")

        print(f"[{TAG}] sleeping {fast_interval}s...")
        persist_and_sleep()


if __name__ == "__main__":
    main()
