"""Work-stealing dip-buying strategy for large crypto universe.

Supports:
- Long dip-buying (buy on X% dip from recent high)
- Short pump-selling (short on X% pump from recent low)
- Leverage (margin mode, configurable max)
- FDUSD 0% fee for BTC/ETH, USDT 10bps for rest
- Margin interest costs
- Work-stealing: limited positions, best candidates fill first
"""
from __future__ import annotations

import heapq
import numpy as np
import pandas as pd
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from src.market_sim_early_exit import evaluate_drawdown_vs_profit_early_exit, print_early_exit

FDUSD_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"}
MARGIN_ANNUAL_RATE = 0.0625  # 6.25% per year
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class WorkStealConfig:
    dip_pct: float = 0.10
    dip_pct_fallback: Tuple[float, ...] = field(default_factory=tuple)
    proximity_pct: float = 0.03
    profit_target_pct: float = 0.05
    stop_loss_pct: float = 0.08
    max_positions: int = 5
    max_hold_days: int = 14
    lookback_days: int = 20
    ref_price_method: str = "high"
    maker_fee: float = 0.001  # default for USDT pairs
    fdusd_fee: float = 0.0    # 0% for FDUSD pairs
    initial_cash: float = 10_000.0
    equal_weight: bool = True
    trailing_stop_pct: float = 0.0
    reentry_cooldown_days: int = 1
    max_leverage: float = 1.0  # 1.0 = no leverage
    enable_shorts: bool = False
    short_pump_pct: float = 0.10
    margin_annual_rate: float = MARGIN_ANNUAL_RATE
    max_position_pct: float = 0.25
    # Trend filters
    sma_filter_period: int = 0
    market_breadth_filter: float = 0.0
    rsi_filter: int = 0
    volume_spike_filter: float = 0.0
    # Risk management
    max_drawdown_exit: float = 0.25
    deleverage_threshold: float = 0.0
    target_leverage: float = 0.0
    entry_proximity_bps: float = 3000.0
    risk_off_ref_price_method: str = "high"
    risk_off_market_breadth_filter: float = 0.70
    risk_off_trigger_sma_period: int = 30
    risk_off_trigger_momentum_period: int = 7
    rebalance_seeded_positions: bool = True
    # SMA check method: "current" (legacy: close>=SMA), "pre_dip" (any of last 5 closes>=SMA), "none"
    sma_check_method: str = "pre_dip"
    adaptive_dip: bool = False
    risk_off_momentum_threshold: float = -0.05
    momentum_period: int = 0
    momentum_min: float = -0.10
    initial_holdings: Dict[str, float] = field(default_factory=dict)
    base_asset_symbol: str = ""
    base_asset_sma_filter_period: int = 0
    base_asset_momentum_period: int = 0
    base_asset_min_momentum: float = 0.0
    base_asset_rebalance_min_cash: float = 1.0
    forecast_bias_weight: float = 0.0
    # When enabled, only count entries if the signal bar actually touched the limit price.
    realistic_fill: bool = False
    # Kept for compatibility with older audit scripts; daily strategy only has one checkpoint per bar.
    daily_checkpoint_only: bool = False


@dataclass
class Position:
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_date: pd.Timestamp
    quantity: float  # positive for both long/short (abs size)
    cost_basis: float
    peak_price: float  # for trailing stop (long: highest, short: lowest)
    target_exit_price: float
    stop_price: float
    margin_borrowed: float = 0.0  # USDT borrowed for leverage


@dataclass
class TradeLog:
    timestamp: pd.Timestamp
    symbol: str
    side: str  # "buy" or "sell" or "short" or "cover"
    price: float
    quantity: float
    notional: float
    fee: float
    pnl: float = 0.0
    reason: str = ""
    direction: str = "long"


ENTRY_SIDES = frozenset(("buy", "short"))
EXIT_SIDES = frozenset(("sell", "cover"))


def get_entry_trades(trades) -> list:
    return [trade for trade in (trades or []) if getattr(trade, "side", None) in ENTRY_SIDES]


def get_exit_trades(trades) -> list:
    return [trade for trade in (trades or []) if getattr(trade, "side", None) in EXIT_SIDES]


def count_completed_trades(trades) -> int:
    return len(get_exit_trades(trades))


def compute_avg_hold_days_from_trades(trades) -> float:
    entries_by_symbol = {}
    hold_days = []
    for trade in trades or []:
        side = getattr(trade, "side", None)
        if side in ENTRY_SIDES:
            entries_by_symbol[getattr(trade, "symbol", None)] = getattr(trade, "timestamp", None)
            continue
        if side not in EXIT_SIDES:
            continue
        symbol = getattr(trade, "symbol", None)
        if symbol not in entries_by_symbol:
            continue
        entry_timestamp = entries_by_symbol.pop(symbol)
        exit_timestamp = getattr(trade, "timestamp", None)
        if entry_timestamp is None or exit_timestamp is None:
            continue
        hold_days.append(max(1, (exit_timestamp - entry_timestamp).days))
    return float(np.mean(hold_days)) if hold_days else 0.0


@dataclass
class EntrySizingContext:
    timestamp: pd.Timestamp
    signal_timestamp: pd.Timestamp
    symbol: str
    direction: str
    score: float
    fill_price: float
    candidate_rank: int
    candidate_count: int
    slots_remaining: int
    current_position_count: int
    cash: float
    base_equity: float
    current_equity: float
    market_breadth: float
    hold_base_asset: bool
    signal_bar: pd.Series
    history: pd.DataFrame
    execution_bar: Optional[pd.Series] = None


@dataclass
class PendingEntry:
    symbol: str
    direction: str
    score: float
    fill_price: float
    signal_date: pd.Timestamp
    signal_bar: pd.Series


@dataclass
class _PreparedBacktestSymbol:
    bars: pd.DataFrame
    rows: tuple[pd.Series, ...]
    timestamp_ns: np.ndarray
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    volumes: Optional[np.ndarray]


class _LazyHistoryContext(Mapping[str, pd.DataFrame]):
    def __init__(
        self,
        prepared: Dict[str, _PreparedBacktestSymbol],
        current_indices: Dict[str, int],
    ) -> None:
        self.prepared = prepared
        self.current_indices = current_indices
        self._cache: Dict[str, pd.DataFrame] = {}

    def __getitem__(self, symbol: str) -> pd.DataFrame:
        if symbol not in self.current_indices:
            raise KeyError(symbol)
        cached = self._cache.get(symbol)
        if cached is None:
            idx = self.current_indices[symbol]
            cached = self.prepared[symbol].bars.iloc[: idx + 1]
            self._cache[symbol] = cached
        return cached

    def __iter__(self) -> Iterator[str]:
        return iter(self.current_indices)

    def __len__(self) -> int:
        return len(self.current_indices)


@dataclass
class _SymbolMetricCache:
    history: Optional[pd.DataFrame]
    bar: pd.Series
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    volumes: Optional[np.ndarray]
    close: float
    low: float
    high: float
    _sma_cache: dict[int, float] = field(default_factory=dict)
    _momentum_cache: dict[int, Optional[float]] = field(default_factory=dict)
    _rsi_cache: dict[int, float] = field(default_factory=dict)
    _volume_ratio_cache: dict[int, float] = field(default_factory=dict)
    _atr_cache: dict[int, float] = field(default_factory=dict)
    _ref_high_cache: dict[tuple[str, int], float] = field(default_factory=dict)
    _ref_low_cache: dict[int, float] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.closes)

    def prev_close(self) -> Optional[float]:
        if self.length < 2:
            return None
        return float(self.closes[-2])

    def sma(self, period: int) -> float:
        cached = self._sma_cache.get(period)
        if cached is not None:
            return cached
        if self.length == 0:
            value = 0.0
        elif self.length < period:
            value = float(np.mean(self.closes))
        else:
            value = float(np.mean(self.closes[-period:]))
        self._sma_cache[period] = value
        return value

    def momentum(self, period: int) -> Optional[float]:
        if period in self._momentum_cache:
            return self._momentum_cache[period]
        if self.length <= period:
            value = None
        else:
            past_close = float(self.closes[-(period + 1)])
            value = None if past_close <= 0.0 else (self.close - past_close) / past_close
        self._momentum_cache[period] = value
        return value

    def passes_sma_filter(self, config: "WorkStealConfig") -> bool:
        if config.sma_filter_period <= 0 or self.length == 0:
            return True
        sma = self.sma(config.sma_filter_period)
        if config.sma_check_method == "current":
            return self.close >= sma
        if config.sma_check_method == "pre_dip":
            n_check = min(5, self.length - 1)
            if n_check > 0:
                recent_closes = self.closes[-(n_check + 1):-1]
                return bool(np.any(recent_closes >= sma))
            return self.close >= sma
        return True

    def rsi(self, period: int = 14) -> float:
        cached = self._rsi_cache.get(period)
        if cached is not None:
            return cached
        if self.length < period + 1:
            value = 50.0
        else:
            closes = self.closes[-(period + 1):]
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            if avg_loss == 0:
                value = 100.0
            else:
                rs = avg_gain / avg_loss
                value = float(100.0 - 100.0 / (1.0 + rs))
        self._rsi_cache[period] = value
        return value

    def volume_ratio(self, period: int = 20) -> float:
        cached = self._volume_ratio_cache.get(period)
        if cached is not None:
            return cached
        if self.volumes is None or self.length < period + 1:
            value = 1.0
        else:
            avg_vol = float(np.mean(self.volumes[-(period + 1):-1]))
            current_vol = float(self.volumes[-1])
            value = 1.0 if avg_vol <= 0 else float(current_vol / avg_vol)
        self._volume_ratio_cache[period] = value
        return value

    def atr(self, period: int = 14) -> float:
        cached = self._atr_cache.get(period)
        if cached is not None:
            return cached
        if self.length == 0:
            value = 0.0
        elif self.length < period + 1:
            value = float(self.highs[-1] - self.lows[-1])
        else:
            high = self.highs[-period:]
            low = self.lows[-period:]
            close = self.closes[-(period + 1):-1]
            tr = np.maximum(high - low, np.maximum(np.abs(high - close), np.abs(low - close)))
            value = float(np.mean(tr))
        self._atr_cache[period] = value
        return value

    def ref_high(self, method: str, lookback: int) -> float:
        key = (method, lookback)
        cached = self._ref_high_cache.get(key)
        if cached is not None:
            return cached
        if self.length == 0:
            value = 0.0
        elif self.length < 2:
            value = float(self.close)
        else:
            start = max(0, self.length - lookback)
            if method == "sma":
                value = float(np.mean(self.closes[start:]))
            elif method == "close":
                value = float(np.max(self.closes[start:]))
            else:
                value = float(np.max(self.highs[start:]))
        self._ref_high_cache[key] = value
        return value

    def ref_low(self, lookback: int) -> float:
        cached = self._ref_low_cache.get(lookback)
        if cached is not None:
            return cached
        if self.length == 0:
            value = 0.0
        elif self.length < 2:
            value = float(self.close)
        else:
            start = max(0, self.length - lookback)
            value = float(np.min(self.lows[start:]))
        self._ref_low_cache[lookback] = value
        return value

    def buy_target(self, ref_high: float, config: "WorkStealConfig") -> float:
        if ref_high <= 0:
            return 0.0
        if config.adaptive_dip:
            atr = self.atr(14)
            dip = min(config.dip_pct, max(0.05, 2.5 * atr / ref_high))
            return ref_high * (1.0 - dip)
        return ref_high * (1.0 - config.dip_pct)


def _prepare_backtest_symbol_frame(df: pd.DataFrame) -> _PreparedBacktestSymbol:
    timestamps = df["timestamp"]
    tz = getattr(timestamps.dtype, "tz", None)
    prepared_df = df
    prepared_ts = timestamps
    if not (tz is not None and str(tz) == "UTC" and not timestamps.hasnans and timestamps.is_monotonic_increasing and timestamps.is_unique):
        prepared_df = df.copy()
        prepared_df["timestamp"] = pd.to_datetime(
            prepared_df["timestamp"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        prepared_df = prepared_df.dropna(subset=["timestamp"])
        prepared_ts = prepared_df["timestamp"]
        if not prepared_ts.is_monotonic_increasing or not prepared_ts.is_unique:
            prepared_df = prepared_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            prepared_ts = prepared_df["timestamp"]
    return _PreparedBacktestSymbol(
        bars=prepared_df,
        rows=tuple(prepared_df.iloc[idx] for idx in range(len(prepared_df))),
        timestamp_ns=prepared_ts.array.asi8,
        closes=prepared_df["close"].to_numpy(dtype=float, copy=False),
        highs=prepared_df["high"].to_numpy(dtype=float, copy=False),
        lows=prepared_df["low"].to_numpy(dtype=float, copy=False),
        volumes=prepared_df["volume"].to_numpy(dtype=float, copy=False) if "volume" in prepared_df.columns else None,
    )


def prepare_backtest_bars(all_bars: Dict[str, pd.DataFrame]) -> Dict[str, _PreparedBacktestSymbol]:
    return {
        sym: _prepare_backtest_symbol_frame(df)
        for sym, df in all_bars.items()
    }


def _collect_backtest_dates(
    prepared: Dict[str, _PreparedBacktestSymbol],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[list[pd.Timestamp], Optional[int]]:
    all_date_ns: set[int] = set()
    start_ts = pd.Timestamp(start_date, tz="UTC") if start_date else None
    end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else None
    start_ns = start_ts.value if start_ts is not None else None
    end_ns = end_ts.value if end_ts is not None else None

    for data in prepared.values():
        ts_ns = data.timestamp_ns
        if start_ns is None and end_ns is None:
            all_date_ns.update(ts_ns.tolist())
            continue
        mask = np.ones(len(ts_ns), dtype=bool)
        if start_ns is not None:
            mask &= ts_ns >= start_ns
        if end_ns is not None:
            mask &= ts_ns <= end_ns
        all_date_ns.update(ts_ns[mask].tolist())

    all_dates = [pd.Timestamp(ns, tz="UTC") for ns in sorted(all_date_ns)]
    first_date_ns = all_dates[0].value if all_dates else None
    return all_dates, first_date_ns


def get_fee(symbol: str, config: WorkStealConfig) -> float:
    if symbol in FDUSD_SYMBOLS:
        return config.fdusd_fee
    return config.maker_fee


def _prepare_backtest_symbol_data(
    all_bars: Dict[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[Dict[str, _PreparedBacktestSymbol], list[pd.Timestamp], Optional[int]]:
    prepared = prepare_backtest_bars(all_bars)
    all_dates, first_date_ns = _collect_backtest_dates(
        prepared,
        start_date=start_date,
        end_date=end_date,
    )
    return prepared, all_dates, first_date_ns


def _initialize_backtest_cursors(
    prepared: Dict[str, _PreparedBacktestSymbol],
    first_date_ns: Optional[int],
) -> Dict[str, int]:
    if first_date_ns is None:
        return {sym: 0 for sym in prepared}
    return {
        sym: int(np.searchsorted(data.timestamp_ns, first_date_ns, side="left"))
        for sym, data in prepared.items()
    }


def _build_daily_market_context(
    prepared: Dict[str, _PreparedBacktestSymbol],
    next_indices: Dict[str, int],
    date: pd.Timestamp,
) -> tuple[Dict[str, pd.Series], Mapping[str, pd.DataFrame]]:
    date_ns = date.value
    current_bars: Dict[str, pd.Series] = {}
    current_indices: Dict[str, int] = {}
    for sym, data in prepared.items():
        idx = next_indices[sym]
        if idx >= len(data.timestamp_ns) or data.timestamp_ns[idx] != date_ns:
            continue
        next_indices[sym] = idx + 1
        current_bars[sym] = data.rows[idx]
        current_indices[sym] = idx
    return current_bars, _LazyHistoryContext(prepared, current_indices)


def _build_symbol_metric_cache(
    current_bars: Dict[str, pd.Series],
    history: Mapping[str, pd.DataFrame],
) -> Dict[str, _SymbolMetricCache]:
    cache: Dict[str, _SymbolMetricCache] = {}
    lazy_history = history if isinstance(history, _LazyHistoryContext) else None
    for sym, bar in current_bars.items():
        hist = None
        if lazy_history is not None:
            idx = lazy_history.current_indices.get(sym)
            data = lazy_history.prepared.get(sym)
            if idx is None or data is None:
                continue
            closes = data.closes[: idx + 1]
            highs = data.highs[: idx + 1]
            lows = data.lows[: idx + 1]
            volumes = data.volumes[: idx + 1] if data.volumes is not None else None
        else:
            hist = history.get(sym)
            if hist is None or hist.empty:
                continue
            closes = hist["close"].to_numpy(dtype=float, copy=False)
            highs = hist["high"].to_numpy(dtype=float, copy=False)
            lows = hist["low"].to_numpy(dtype=float, copy=False)
            volumes = hist["volume"].to_numpy(dtype=float, copy=False) if "volume" in hist.columns else None
        cache[sym] = _SymbolMetricCache(
            history=hist,
            bar=bar,
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            close=float(bar["close"]),
            low=float(bar["low"]),
            high=float(bar["high"]),
        )
    return cache


def compute_ref_price(bars: pd.DataFrame, method: str, lookback: int) -> float:
    if bars.empty:
        return 0.0
    if len(bars) < 2:
        return float(bars["close"].iloc[-1])
    window = bars.tail(lookback)
    if method == "high":
        return float(window["high"].max())
    elif method == "sma":
        return float(window["close"].mean())
    elif method == "close":
        return float(window["close"].max())
    return float(window["high"].max())


def compute_ref_low(bars: pd.DataFrame, lookback: int) -> float:
    if bars.empty:
        return 0.0
    if len(bars) < 2:
        return float(bars["close"].iloc[-1])
    window = bars.tail(lookback)
    return float(window["low"].min())


def compute_sma(bars: pd.DataFrame, period: int) -> float:
    if bars.empty:
        return 0.0
    closes = bars["close"].to_numpy(dtype=float, copy=False)
    if len(closes) < period:
        return float(np.mean(closes))
    return float(np.mean(closes[-period:]))


def compute_rsi(bars: pd.DataFrame, period: int = 14) -> float:
    if bars.empty or len(bars) < period + 1:
        return 50.0
    closes = bars["close"].values[-(period+1):]
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def compute_volume_ratio(bars: pd.DataFrame, period: int = 20) -> float:
    if bars.empty or len(bars) < period + 1 or "volume" not in bars.columns:
        return 1.0
    avg_vol = bars["volume"].iloc[-(period+1):-1].mean()
    current_vol = bars["volume"].iloc[-1]
    if avg_vol <= 0:
        return 1.0
    return float(current_vol / avg_vol)


def compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
    if bars.empty:
        return 0.0
    if len(bars) < period + 1:
        return float(bars["high"].iloc[-1] - bars["low"].iloc[-1])
    high = bars["high"].values[-period:]
    low = bars["low"].values[-period:]
    close = bars["close"].values[-(period+1):-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - close), np.abs(low - close)))
    return float(np.mean(tr))


def _compute_margin_interest(pos: Position, current_date: pd.Timestamp, rate: float) -> float:
    if pos.margin_borrowed <= 0:
        return 0.0
    days_held = max(1, (current_date - pos.entry_date).days)
    daily_rate = rate / 365.0
    return pos.margin_borrowed * daily_rate * days_held


def _entry_fee_paid(pos: Position, config: WorkStealConfig) -> float:
    return pos.quantity * pos.entry_price * get_fee(pos.symbol, config)


def _mark_to_market_position_value(
    pos: Position,
    price: float,
    current_date: pd.Timestamp,
    config: WorkStealConfig,
) -> float:
    interest = _compute_margin_interest(pos, current_date, config.margin_annual_rate)
    if pos.direction == "long":
        return pos.quantity * price - pos.margin_borrowed - interest
    unrealized_pnl = pos.quantity * (pos.entry_price - price)
    return pos.cost_basis + unrealized_pnl - _entry_fee_paid(pos, config) - interest


def _close_position_accounting(
    pos: Position,
    exit_price: float,
    current_date: pd.Timestamp,
    config: WorkStealConfig,
) -> tuple[float, float, float]:
    fee_rate = get_fee(pos.symbol, config)
    interest = _compute_margin_interest(pos, current_date, config.margin_annual_rate)
    exit_fee = pos.quantity * exit_price * fee_rate

    if pos.direction == "long":
        proceeds = pos.quantity * exit_price * (1 - fee_rate)
        cash_delta = proceeds - pos.margin_borrowed - interest
        pnl = cash_delta - pos.cost_basis
        return float(cash_delta), float(pnl), float(exit_fee + interest)

    cash_delta = (
        pos.cost_basis
        + pos.quantity * (pos.entry_price - exit_price)
        - _entry_fee_paid(pos, config)
        - exit_fee
        - interest
    )
    pnl = cash_delta - pos.cost_basis
    return float(cash_delta), float(pnl), float(exit_fee + interest)


def _risk_off_triggered_cached(
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    symbol_metrics: Dict[str, _SymbolMetricCache],
) -> bool:
    if config.risk_off_trigger_sma_period > 0:
        n_above = 0
        n_total = 0
        for sym, bar in current_bars.items():
            metrics = symbol_metrics.get(sym)
            if metrics is not None:
                if metrics.length < config.risk_off_trigger_sma_period:
                    continue
                n_total += 1
                if metrics.close >= metrics.sma(config.risk_off_trigger_sma_period):
                    n_above += 1
                continue
            hist = history.get(sym)
            if hist is None or len(hist) < config.risk_off_trigger_sma_period:
                continue
            n_total += 1
            if float(bar["close"]) >= compute_sma(hist, config.risk_off_trigger_sma_period):
                n_above += 1
        if n_total > 0 and (n_above / n_total) < 0.5:
            return True

    if config.risk_off_trigger_momentum_period > 0:
        momentums: list[float] = []
        for sym, bar in current_bars.items():
            metrics = symbol_metrics.get(sym)
            if metrics is not None:
                momentum = metrics.momentum(config.risk_off_trigger_momentum_period)
                if momentum is not None:
                    momentums.append(momentum)
                continue
            hist = history.get(sym)
            if hist is None or len(hist) <= config.risk_off_trigger_momentum_period:
                continue
            past_close = float(hist.iloc[-(config.risk_off_trigger_momentum_period + 1)]["close"])
            if past_close <= 0.0:
                continue
            momentums.append((float(bar["close"]) - past_close) / past_close)
        if momentums and float(np.mean(momentums)) < config.risk_off_momentum_threshold:
            return True
    return False


def _risk_off_triggered(
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
) -> bool:
    return _risk_off_triggered_cached(current_bars, history, config, {})


_ORIGINAL_RISK_OFF_TRIGGERED = _risk_off_triggered


def passes_sma_filter(
    bars: pd.DataFrame,
    config: WorkStealConfig,
    close: float,
    metrics: Optional[_SymbolMetricCache] = None,
) -> bool:
    if metrics is not None:
        return metrics.passes_sma_filter(config)
    if config.sma_filter_period <= 0:
        return True
    if bars.empty:
        return True
    sma = compute_sma(bars, config.sma_filter_period)
    if config.sma_check_method == "current":
        return close >= sma
    elif config.sma_check_method == "pre_dip":
        n_check = min(5, len(bars) - 1)
        if n_check > 0:
            recent_closes = bars["close"].values[-(n_check + 1):-1]
            return any(float(c) >= sma for c in recent_closes)
        return close >= sma
    return True  # "none"


def compute_buy_target(bars: pd.DataFrame, ref_high: float, config: WorkStealConfig) -> float:
    if ref_high <= 0:
        return 0.0
    if config.adaptive_dip:
        atr = compute_atr(bars, 14)
        dip = min(config.dip_pct, max(0.05, 2.5 * atr / ref_high))
        return ref_high * (1.0 - dip)
    return ref_high * (1.0 - config.dip_pct)


def compute_breadth_ratio(
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    symbol_metrics: Optional[Dict[str, _SymbolMetricCache]] = None,
) -> Tuple[float, int, int]:
    n_total = 0
    n_dipping = 0
    for sym in current_bars:
        metrics = symbol_metrics.get(sym) if symbol_metrics is not None else None
        if metrics is not None:
            if metrics.length < 5:
                continue
            prev_close = metrics.prev_close()
            if prev_close is None:
                continue
            n_total += 1
            if metrics.close < prev_close:
                n_dipping += 1
            continue
        if sym not in history or len(history[sym]) < 5:
            continue
        n_total += 1
        prev_close = float(history[sym].iloc[-2]["close"])
        if float(current_bars[sym]["close"]) < prev_close:
            n_dipping += 1
    ratio = n_dipping / max(n_total, 1)
    return ratio, n_dipping, n_total


def compute_market_breadth_skip(
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    symbol_metrics: Optional[Dict[str, _SymbolMetricCache]] = None,
) -> bool:
    if config.market_breadth_filter > 0:
        ratio, _, _ = compute_breadth_ratio(current_bars, history, symbol_metrics=symbol_metrics)
        return ratio > config.market_breadth_filter
    return False


def resolve_entry_config(
    *,
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
) -> WorkStealConfig:
    return resolve_entry_regime(
        current_bars=current_bars,
        history=history,
        config=config,
    ).config


@dataclass
class SymbolDiagnostic:
    symbol: str
    close: float = 0.0
    ref_high: float = 0.0
    buy_target: float = 0.0
    dist_pct: float = 0.0
    sma_value: float = 0.0
    sma_pass: bool = True
    momentum_ret: float = 0.0
    filter_reason: str = ""
    is_candidate: bool = False


@dataclass(frozen=True)
class EntryRegimeState:
    config: WorkStealConfig
    risk_off: bool
    market_breadth_skip: bool
    market_breadth_ratio: float = 0.0
    market_breadth_dipping_count: int = 0
    market_breadth_total_count: int = 0

    @property
    def skip_entries(self) -> bool:
        return self.risk_off or self.market_breadth_skip


def resolve_entry_regime(
    *,
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    symbol_metrics: Optional[Dict[str, _SymbolMetricCache]] = None,
) -> EntryRegimeState:
    use_cached_risk_off = (
        symbol_metrics is not None and _risk_off_triggered is _ORIGINAL_RISK_OFF_TRIGGERED
    )
    risk_off = (
        _risk_off_triggered_cached(current_bars, history, config, symbol_metrics)
        if use_cached_risk_off
        else _risk_off_triggered(current_bars, history, config)
    )
    if risk_off:
        entry_config = replace(
            config,
            ref_price_method=str(config.risk_off_ref_price_method or config.ref_price_method),
            market_breadth_filter=float(max(config.market_breadth_filter, config.risk_off_market_breadth_filter)),
        )
    else:
        entry_config = config
    market_breadth_ratio = 0.0
    market_breadth_dipping_count = 0
    market_breadth_total_count = 0
    market_breadth_skip = False
    if entry_config.market_breadth_filter > 0:
        market_breadth_ratio, market_breadth_dipping_count, market_breadth_total_count = compute_breadth_ratio(
            current_bars,
            history,
            symbol_metrics=symbol_metrics,
        )
        market_breadth_skip = market_breadth_ratio > entry_config.market_breadth_filter
    return EntryRegimeState(
        config=entry_config,
        risk_off=bool(risk_off),
        market_breadth_skip=bool(market_breadth_skip),
        market_breadth_ratio=float(market_breadth_ratio),
        market_breadth_dipping_count=int(market_breadth_dipping_count),
        market_breadth_total_count=int(market_breadth_total_count),
    )


def _rank_candidates_desc(
    candidates: list,
    *,
    max_candidates: int | None = None,
) -> list:
    indexed_candidates = list(enumerate(candidates))
    if max_candidates is None:
        ranked = sorted(indexed_candidates, key=lambda item: (-item[1][2], item[0]))
    else:
        if max_candidates <= 0:
            return []
        ranked = heapq.nlargest(
            max_candidates * 2,
            indexed_candidates,
            key=lambda item: (item[1][2], -item[0]),
        )
        ranked.sort(key=lambda item: (-item[1][2], item[0]))
    return [candidate for _, candidate in ranked]



def _dedupe_ranked_candidates(
    candidates: list,
    *,
    max_candidates: int | None = None,
) -> list:
    deduped: list = []
    seen_symbols: set[str] = set()
    for candidate in candidates:
        symbol = candidate[0]
        if symbol in seen_symbols:
            continue
        deduped.append(candidate)
        seen_symbols.add(symbol)
        if max_candidates is not None and len(deduped) >= max_candidates:
            break
    return deduped



def build_entry_candidates(
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    positions: Dict[str, object],
    last_exit: Dict[str, pd.Timestamp],
    date: pd.Timestamp,
    config: WorkStealConfig,
    base_symbol: str | None = None,
    diagnostics: Optional[List] = None,
    symbol_metrics: Optional[Dict[str, _SymbolMetricCache]] = None,
    max_candidates: int | None = None,
) -> list:
    candidates = []
    for sym, bar in current_bars.items():
        metrics = symbol_metrics.get(sym) if symbol_metrics is not None else None
        hist: Optional[pd.DataFrame] = None
        history_loaded = False

        def load_history() -> Optional[pd.DataFrame]:
            nonlocal hist, history_loaded
            if not history_loaded:
                hist = history.get(sym)
                history_loaded = True
            return hist

        diag = SymbolDiagnostic(symbol=sym) if diagnostics is not None else None
        if base_symbol is not None and sym == base_symbol:
            if diag:
                diag.filter_reason = "base_asset"
                diagnostics.append(diag)
            continue
        if sym in positions:
            if diag:
                diag.filter_reason = "already_held"
                diagnostics.append(diag)
            continue
        if sym in last_exit and (date - last_exit[sym]).days < config.reentry_cooldown_days:
            if diag:
                diag.filter_reason = "cooldown"
                diagnostics.append(diag)
            continue
        if metrics is not None:
            hist_len = metrics.length
        else:
            hist = load_history()
            hist_len = 0 if hist is None else len(hist)
        if hist_len < config.lookback_days:
            if diag:
                diag.filter_reason = "insufficient_history"
                diagnostics.append(diag)
            continue

        close = metrics.close if metrics is not None else float(bar["close"])
        low_bar = metrics.low if metrics is not None else float(bar["low"])
        high_bar = metrics.high if metrics is not None else float(bar["high"])
        if diag:
            diag.close = close

        if config.momentum_period > 0:
            mom_ret = metrics.momentum(config.momentum_period) if metrics is not None else None
            if mom_ret is None:
                hist = load_history()
                if hist is not None and len(hist) > config.momentum_period:
                    past_close = float(hist.iloc[-(config.momentum_period + 1)]["close"])
                    if past_close > 0.0:
                        mom_ret = (close - past_close) / past_close
            if mom_ret is not None:
                if diag:
                    diag.momentum_ret = mom_ret
                if mom_ret < config.momentum_min:
                    if diag:
                        diag.filter_reason = f"momentum({mom_ret:.3f}<{config.momentum_min:.3f})"
                        diagnostics.append(diag)
                    continue

        hist_for_filters = load_history() if metrics is None else None
        sma_ok = passes_sma_filter(
            hist_for_filters if hist_for_filters is not None else pd.DataFrame(),
            config,
            close,
            metrics=metrics,
        )
        if diag:
            diag.sma_pass = sma_ok
            if config.sma_filter_period > 0:
                if metrics is not None:
                    diag.sma_value = metrics.sma(config.sma_filter_period)
                elif hist_for_filters is not None:
                    diag.sma_value = compute_sma(hist_for_filters, config.sma_filter_period)
        if not sma_ok:
            if diag:
                diag.filter_reason = f"sma_filter(close={close:.2f},sma={diag.sma_value:.2f})"
                diagnostics.append(diag)
            continue

        if config.rsi_filter > 0:
            if metrics is not None:
                rsi = metrics.rsi(14)
            else:
                hist = load_history()
                rsi = compute_rsi(hist if hist is not None else pd.DataFrame(), 14)
            if rsi > config.rsi_filter:
                if diag:
                    diag.filter_reason = f"rsi({rsi:.1f}>{config.rsi_filter})"
                    diagnostics.append(diag)
                continue

        if config.volume_spike_filter > 0:
            if metrics is not None:
                vol_ratio = metrics.volume_ratio(20)
            else:
                hist = load_history()
                vol_ratio = compute_volume_ratio(hist if hist is not None else pd.DataFrame(), 20)
            if vol_ratio < config.volume_spike_filter:
                if diag:
                    diag.filter_reason = f"volume({vol_ratio:.2f}<{config.volume_spike_filter:.2f})"
                    diagnostics.append(diag)
                continue

        if metrics is not None:
            ref_high = metrics.ref_high(config.ref_price_method, config.lookback_days)
            buy_target = metrics.buy_target(ref_high, config)
        else:
            hist = load_history()
            ref_high = compute_ref_price(hist if hist is not None else pd.DataFrame(), config.ref_price_method, config.lookback_days)
            buy_target = compute_buy_target(hist if hist is not None else pd.DataFrame(), ref_high, config)
        if not np.isfinite(ref_high) or ref_high <= 0 or not np.isfinite(buy_target):
            if diag:
                diag.ref_high = ref_high
                diag.buy_target = buy_target
                diag.filter_reason = "invalid_ref_high"
                diagnostics.append(diag)
            continue
        dist_long = (close - buy_target) / ref_high
        if diag:
            diag.ref_high = ref_high
            diag.buy_target = buy_target
            diag.dist_pct = dist_long

        if dist_long <= config.proximity_pct:
            strict_long_fill = low_bar <= buy_target
            if config.realistic_fill and not strict_long_fill:
                if diag:
                    diag.filter_reason = "strict_fill_not_touched"
                    diagnostics.append(diag)
                continue
            dip_score = -dist_long
            fill_price = buy_target if config.realistic_fill else max(buy_target, low_bar)
            candidates.append((sym, "long", dip_score, fill_price, bar))
            if diag:
                diag.is_candidate = True
                diag.filter_reason = ""
                diagnostics.append(diag)
        else:
            if diag:
                diag.filter_reason = f"proximity(dist={dist_long:.4f}>{config.proximity_pct:.4f})"
                diagnostics.append(diag)

        if config.enable_shorts:
            if metrics is not None:
                ref_low = metrics.ref_low(config.lookback_days)
            else:
                hist = load_history()
                ref_low = compute_ref_low(hist if hist is not None else pd.DataFrame(), config.lookback_days)
            if not np.isfinite(ref_low) or ref_low <= 0:
                continue
            short_target = ref_low * (1 + config.short_pump_pct)
            dist_short = (short_target - close) / ref_low
            if dist_short <= config.proximity_pct:
                strict_short_fill = high_bar >= short_target
                if config.realistic_fill and not strict_short_fill:
                    continue
                pump_score = -dist_short
                fill_price = short_target if config.realistic_fill else min(short_target, high_bar)
                candidates.append((sym, "short", pump_score, fill_price, bar))

    unique_symbol_count = len({candidate[0] for candidate in candidates})
    ranked_candidates = _rank_candidates_desc(candidates, max_candidates=max_candidates)
    deduped_candidates = _dedupe_ranked_candidates(
        ranked_candidates,
        max_candidates=max_candidates,
    )
    if (
        max_candidates is not None
        and len(deduped_candidates) < min(max_candidates, unique_symbol_count)
    ):
        ranked_candidates = _rank_candidates_desc(candidates)
        deduped_candidates = _dedupe_ranked_candidates(
            ranked_candidates,
            max_candidates=max_candidates,
        )
    return deduped_candidates

def build_tiered_entry_candidates(
    *,
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    positions: Dict[str, object],
    last_exit: Dict[str, pd.Timestamp],
    date: pd.Timestamp,
    config: WorkStealConfig,
    base_symbol: str | None = None,
    max_candidates: int | None = None,
    diagnostics: Optional[List[SymbolDiagnostic]] = None,
    symbol_metrics: Optional[Dict[str, _SymbolMetricCache]] = None,
) -> tuple[list, Dict[str, int]]:
    raw_tiers = list(config.dip_pct_fallback) if config.dip_pct_fallback else [config.dip_pct]
    tiers: list[float] = []
    seen_tiers: set[float] = set()
    for raw_tier in raw_tiers:
        try:
            tier = float(raw_tier)
        except (TypeError, ValueError):
            continue
        if tier <= 0.0 or tier in seen_tiers:
            continue
        tiers.append(tier)
        seen_tiers.add(tier)
    if not tiers:
        tiers = [float(config.dip_pct)]

    occupied_symbols = set(positions)
    all_candidates = []
    tier_map: Dict[str, int] = {}
    remaining = max_candidates if max_candidates is not None else None

    for tier_idx, dip_pct in enumerate(tiers):
        if remaining is not None and remaining <= 0:
            break
        tier_config = replace(config, dip_pct=dip_pct)
        tier_diagnostics: Optional[List[SymbolDiagnostic]] = [] if diagnostics is not None and tier_idx == 0 else None
        candidates = build_entry_candidates(
            date=date,
            current_bars=current_bars,
            history=history,
            positions={sym: True for sym in occupied_symbols},
            last_exit=last_exit,
            config=tier_config,
            base_symbol=base_symbol,
            diagnostics=tier_diagnostics,
            symbol_metrics=symbol_metrics,
            max_candidates=remaining,
        )
        if diagnostics is not None and tier_diagnostics:
            diagnostics.extend(tier_diagnostics)

        for candidate in candidates:
            sym = candidate[0]
            if sym in occupied_symbols:
                continue
            all_candidates.append(candidate)
            tier_map[sym] = tier_idx
            occupied_symbols.add(sym)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break

    return all_candidates, tier_map


def _normalize_base_asset_symbol(config: WorkStealConfig) -> str | None:
    value = str(config.base_asset_symbol or "").strip().upper()
    return value or None


def _base_asset_should_hold(
    *,
    base_symbol: str | None,
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    symbol_metrics: Optional[Dict[str, _SymbolMetricCache]] = None,
) -> bool:
    if base_symbol is None:
        return False
    bar = current_bars.get(base_symbol)
    metrics = symbol_metrics.get(base_symbol) if symbol_metrics is not None else None
    hist = None
    if bar is None:
        return False
    if metrics is None:
        hist = history.get(base_symbol)
        if hist is None or hist.empty:
            return False
    elif metrics.length <= 0:
        return False

    close = metrics.close if metrics is not None else float(bar["close"])
    if close <= 0:
        return False

    if config.base_asset_sma_filter_period > 0:
        hist_len = metrics.length if metrics is not None else len(hist)
        if hist_len < config.base_asset_sma_filter_period:
            return False
        sma = metrics.sma(config.base_asset_sma_filter_period) if metrics is not None else compute_sma(hist, config.base_asset_sma_filter_period)
        if close < sma:
            return False

    if config.base_asset_momentum_period > 0:
        momentum = metrics.momentum(config.base_asset_momentum_period) if metrics is not None else None
        if momentum is None:
            if metrics is not None:
                return False
            if len(hist) <= config.base_asset_momentum_period:
                return False
            past_close = float(hist.iloc[-(config.base_asset_momentum_period + 1)]["close"])
            if past_close > 0:
                momentum = (close - past_close) / past_close
        if momentum is not None and momentum < config.base_asset_min_momentum:
            return False

    return True


def _buy_base_asset(
    *,
    cash: float,
    base_qty: float,
    price: float,
    fee_rate: float,
    min_cash: float,
) -> tuple[float, float]:
    if cash <= min_cash or price <= 0:
        return float(cash), float(base_qty)
    deployable_cash = float(cash - min_cash)
    if deployable_cash <= 0:
        return float(cash), float(base_qty)
    denom = price * (1.0 + fee_rate)
    if denom <= 0:
        return float(cash), float(base_qty)
    qty = deployable_cash / denom
    if qty <= 0:
        return float(cash), float(base_qty)
    cost = qty * denom
    return float(cash - cost), float(base_qty + qty)


def _sell_base_asset_for_cash(
    *,
    cash: float,
    base_qty: float,
    price: float,
    fee_rate: float,
    required_cash: float,
) -> tuple[float, float]:
    if required_cash <= cash or base_qty <= 0 or price <= 0:
        return float(cash), float(base_qty)
    net_price = price * (1.0 - fee_rate)
    if net_price <= 0:
        return float(cash), float(base_qty)
    needed = required_cash - cash
    qty = min(base_qty, needed / net_price)
    if qty <= 0:
        return float(cash), float(base_qty)
    proceeds = qty * net_price
    return float(cash + proceeds), float(base_qty - qty)


def _sell_all_base_asset(
    *,
    cash: float,
    base_qty: float,
    price: float,
    fee_rate: float,
) -> tuple[float, float]:
    if base_qty <= 0 or price <= 0:
        return float(cash), float(base_qty)
    proceeds = base_qty * price * (1.0 - fee_rate)
    return float(cash + proceeds), 0.0


def _compute_starting_equity(
    config: WorkStealConfig,
    current_bars: Dict[str, pd.Series],
) -> float:
    equity = float(config.initial_cash)
    for symbol, quantity in (config.initial_holdings or {}).items():
        if float(quantity) <= 0.0 or symbol not in current_bars:
            continue
        equity += float(quantity) * float(current_bars[symbol]["close"])
    return equity


def _seed_initial_holdings(
    *,
    date: pd.Timestamp,
    current_bars: Dict[str, pd.Series],
    config: WorkStealConfig,
    positions: Dict[str, Position],
    base_symbol: str | None,
) -> float:
    base_qty = 0.0
    for symbol, quantity in (config.initial_holdings or {}).items():
        qty = float(quantity)
        if qty <= 0.0 or symbol not in current_bars:
            continue
        bar = current_bars[symbol]
        price = float(bar["close"])
        if price <= 0.0:
            continue
        if base_symbol is not None and symbol == base_symbol:
            base_qty += qty
            continue
        positions[symbol] = Position(
            symbol=symbol,
            direction="long",
            entry_price=price,
            entry_date=date,
            quantity=qty,
            cost_basis=qty * price,
            peak_price=float(bar["high"]),
            target_exit_price=price * (1.0 + config.profit_target_pct),
            stop_price=price * (1.0 - config.stop_loss_pct),
            margin_borrowed=0.0,
        )
    return float(base_qty)


def _compute_rebalance_keep_symbols(
    *,
    date: pd.Timestamp,
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    last_exit: Dict[str, pd.Timestamp],
    config: WorkStealConfig,
    base_symbol: str | None,
) -> set[str]:
    del date, history, last_exit
    keep_symbols = {
        str(symbol).upper()
        for symbol, quantity in (config.initial_holdings or {}).items()
        if float(quantity) > 0.0 and symbol in current_bars and symbol != base_symbol
    }
    return keep_symbols


def _apply_seeded_rebalance(
    *,
    timestamp: pd.Timestamp,
    current_prices: Dict[str, float],
    positions: Dict[str, Position],
    trades: List[TradeLog],
    last_exit: Dict[str, pd.Timestamp],
    cash: float,
    config: WorkStealConfig,
    keep_symbols: set[str],
) -> float:
    if not config.rebalance_seeded_positions:
        return float(cash)
    updated_cash = float(cash)
    for symbol in list(positions):
        if symbol in keep_symbols:
            continue
        position = positions.pop(symbol)
        price = float(current_prices.get(symbol, position.entry_price))
        fee_rate = get_fee(symbol, config)
        proceeds = position.quantity * price * (1.0 - fee_rate)
        pnl = proceeds - position.cost_basis
        updated_cash += proceeds
        trades.append(
            TradeLog(
                timestamp=timestamp,
                symbol=symbol,
                side="sell" if position.direction == "long" else "cover",
                price=price,
                quantity=position.quantity,
                notional=position.quantity * price,
                fee=position.quantity * price * fee_rate,
                pnl=pnl,
                reason="seeded_rebalance",
                direction=position.direction,
            )
        )
        last_exit[symbol] = timestamp
    return float(updated_cash)


def _execution_bar_for_signal(
    intraday_bars: Optional[Dict[str, pd.DataFrame]],
    symbol: str,
    signal_timestamp: pd.Timestamp,
) -> Optional[pd.Series]:
    if not intraday_bars:
        return None
    frame = intraday_bars.get(symbol)
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return None
    ts = pd.Timestamp(signal_timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    next_ts = ts + pd.Timedelta(days=1)
    hourly = frame.copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce")
    hourly = hourly.dropna(subset=["timestamp"])
    hourly = hourly[(hourly["timestamp"] >= ts) & (hourly["timestamp"] < next_ts)]
    if hourly.empty:
        return None
    return hourly.sort_values("timestamp").iloc[0]


def run_worksteal_backtest(
    all_bars: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    forecast_data: Optional[Dict] = None,
    intraday_bars: Optional[Dict[str, pd.DataFrame]] = None,
    allocation_scale_fn=None,
    prepared_bars: Optional[Dict[str, _PreparedBacktestSymbol]] = None,
) -> Tuple[pd.DataFrame, List[TradeLog], Dict[str, float]]:
    if prepared_bars is None:
        prepared, all_dates, first_date_ns = _prepare_backtest_symbol_data(
            all_bars,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        prepared = prepared_bars
        all_dates, first_date_ns = _collect_backtest_dates(
            prepared,
            start_date=start_date,
            end_date=end_date,
        )
    next_indices = _initialize_backtest_cursors(prepared, first_date_ns)

    cash = config.initial_cash
    base_symbol = _normalize_base_asset_symbol(config)
    base_qty = 0.0
    positions: Dict[str, Position] = {}
    trades: List[TradeLog] = []
    equity_rows: List[Dict] = []
    equity_values: List[float] = []
    last_exit: Dict[str, pd.Timestamp] = {}
    pending_entries: list[PendingEntry] = []
    candidates_generated = 0
    candidates_visible = 0
    entries_executed = 0
    peak_equity = config.initial_cash
    initial_holdings_seeded = False
    seeded_positions_rebalanced = False
    starting_equity = float(config.initial_cash)

    for date in all_dates:
        current_bars, history = _build_daily_market_context(prepared, next_indices, date)

        if not current_bars:
            continue

        if not initial_holdings_seeded:
            base_qty += _seed_initial_holdings(
                date=date,
                current_bars=current_bars,
                config=config,
                positions=positions,
                base_symbol=base_symbol,
            )
            starting_equity = _compute_starting_equity(config, current_bars)
            initial_holdings_seeded = True

        if initial_holdings_seeded and not seeded_positions_rebalanced:
            current_prices = {sym: float(bar["close"]) for sym, bar in current_bars.items()}
            keep_symbols = _compute_rebalance_keep_symbols(
                date=date,
                current_bars=current_bars,
                history=history,
                last_exit=last_exit,
                config=config,
                base_symbol=base_symbol,
            )
            cash = _apply_seeded_rebalance(
                timestamp=date,
                current_prices=current_prices,
                positions=positions,
                trades=trades,
                last_exit=last_exit,
                cash=cash,
                config=config,
                keep_symbols=keep_symbols,
            )
            seeded_positions_rebalanced = True

        symbol_metrics = _build_symbol_metric_cache(current_bars, history)

        # Compute current equity for position sizing
        inv_value = 0.0
        for sym, pos in positions.items():
            price = float(current_bars[sym]["close"]) if sym in current_bars else pos.entry_price
            inv_value += _mark_to_market_position_value(pos, price, date, config)
        base_value = 0.0
        if base_symbol is not None and base_symbol in current_bars:
            base_value = base_qty * float(current_bars[base_symbol]["close"])
        current_equity = cash + inv_value + base_value

        # 1. Check exits
        symbols_to_exit = []
        for sym, pos in list(positions.items()):
            if sym not in current_bars:
                continue
            bar = current_bars[sym]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])

            exit_price = None
            exit_reason = ""

            if pos.direction == "long":
                pos.peak_price = max(pos.peak_price, high)

                if high >= pos.target_exit_price:
                    exit_price = pos.target_exit_price
                    exit_reason = "profit_target"
                elif low <= pos.stop_price:
                    exit_price = pos.stop_price
                    exit_reason = "stop_loss"
                elif config.trailing_stop_pct > 0:
                    trail = pos.peak_price * (1 - config.trailing_stop_pct)
                    if low <= trail:
                        exit_price = trail
                        exit_reason = "trailing_stop"
            else:  # short
                pos.peak_price = min(pos.peak_price, low)

                if low <= pos.target_exit_price:
                    exit_price = pos.target_exit_price
                    exit_reason = "profit_target"
                elif high >= pos.stop_price:
                    exit_price = pos.stop_price
                    exit_reason = "stop_loss"
                elif config.trailing_stop_pct > 0:
                    trail = pos.peak_price * (1 + config.trailing_stop_pct)
                    if high >= trail:
                        exit_price = trail
                        exit_reason = "trailing_stop"

            # Margin call: force close short if loss exceeds collateral
            if exit_price is None and pos.direction == "short":
                unrealized_loss = pos.quantity * (close - pos.entry_price)
                if unrealized_loss > pos.cost_basis * 0.8:  # 80% of collateral
                    exit_price = close
                    exit_reason = "margin_call"

            if exit_price is None and config.max_hold_days > 0:
                held = (date - pos.entry_date).days
                if held >= config.max_hold_days:
                    exit_price = close
                    exit_reason = "max_hold"

            if exit_price is not None:
                symbols_to_exit.append((sym, exit_price, exit_reason))

        for sym, exit_price, reason in symbols_to_exit:
            pos = positions[sym]
            cash_delta, pnl, fee = _close_position_accounting(pos, exit_price, date, config)
            cash += cash_delta
            side = "sell" if pos.direction == "long" else "cover"

            trades.append(TradeLog(
                timestamp=date, symbol=sym, side=side,
                price=exit_price, quantity=pos.quantity,
                notional=pos.quantity * exit_price,
                fee=fee,
                pnl=pnl, reason=reason, direction=pos.direction,
            ))
            last_exit[sym] = date
            del positions[sym]

        # 1b. Deleveraging: close worst position if over-leveraged
        if config.deleverage_threshold > 0 and positions:
            delev_notional = 0.0
            delev_inv = 0.0
            worst_sym = None
            worst_pnl = float("inf")
            for sym, pos in positions.items():
                price = float(current_bars[sym]["close"]) if sym in current_bars else pos.entry_price
                delev_notional += pos.quantity * price
                delev_inv += _mark_to_market_position_value(pos, price, date, config)
                if pos.direction == "long":
                    pnl_pct = (price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - price) / pos.entry_price
                if pnl_pct < worst_pnl:
                    worst_pnl = pnl_pct
                    worst_sym = sym
            delev_equity = cash + delev_inv
            if delev_equity > 0 and worst_sym is not None:
                current_lev = delev_notional / delev_equity
                if current_lev > config.max_leverage + config.deleverage_threshold:
                    pos = positions[worst_sym]
                    close_p = float(current_bars[worst_sym]["close"])
                    cash_delta, pnl, fee = _close_position_accounting(pos, close_p, date, config)
                    cash += cash_delta
                    side = "sell" if pos.direction == "long" else "cover"
                    trades.append(TradeLog(
                        timestamp=date, symbol=worst_sym, side=side,
                        price=close_p, quantity=pos.quantity,
                        notional=pos.quantity * close_p,
                        fee=fee,
                        pnl=pnl, reason="deleverage", direction=pos.direction,
                    ))
                    last_exit[worst_sym] = date
                    del positions[worst_sym]
        hold_base_asset = _base_asset_should_hold(
            base_symbol=base_symbol,
            current_bars=current_bars,
            history=history,
            config=config,
            symbol_metrics=symbol_metrics,
        )
        if base_symbol is not None and base_symbol in current_bars and not hold_base_asset:
            base_fee = get_fee(base_symbol, config)
            cash, base_qty = _sell_all_base_asset(
                cash=cash,
                base_qty=base_qty,
                price=float(current_bars[base_symbol]["close"]),
                fee_rate=base_fee,
            )

        if config.daily_checkpoint_only and pending_entries and len(positions) < config.max_positions:
            next_pending_entries: list[PendingEntry] = []
            for order in pending_entries:
                if len(positions) >= config.max_positions:
                    continue
                if order.symbol in positions or order.symbol not in current_bars:
                    continue
                fee_rate = get_fee(order.symbol, config)
                bar = current_bars[order.symbol]
                low = float(bar["low"])
                high = float(bar["high"])
                touch_ok = low <= order.fill_price if order.direction == "long" else high >= order.fill_price
                if not touch_ok:
                    continue

                max_alloc = config.initial_cash * config.max_position_pct * config.max_leverage
                if order.direction == "long":
                    if base_symbol is not None and hold_base_asset and base_symbol in current_bars:
                        base_fee = get_fee(base_symbol, config)
                        cash, base_qty = _sell_base_asset_for_cash(
                            cash=cash,
                            base_qty=base_qty,
                            price=float(current_bars[base_symbol]["close"]),
                            fee_rate=base_fee,
                            required_cash=max_alloc,
                        )
                    quantity = max_alloc / (order.fill_price * (1 + fee_rate))
                    if quantity <= 0.0:
                        continue
                    available_cash = float(cash)
                    actual_cost = quantity * order.fill_price * (1 + fee_rate)
                    borrowed = max(0.0, actual_cost - available_cash)
                    equity_used = actual_cost - borrowed
                    cash -= equity_used
                    positions[order.symbol] = Position(
                        symbol=order.symbol,
                        direction="long",
                        entry_price=order.fill_price,
                        entry_date=date,
                        quantity=quantity,
                        cost_basis=equity_used,
                        peak_price=float(bar["high"]),
                        target_exit_price=order.fill_price * (1 + config.profit_target_pct),
                        stop_price=order.fill_price * (1 - config.stop_loss_pct),
                        margin_borrowed=borrowed,
                    )
                    trades.append(TradeLog(
                        timestamp=date,
                        symbol=order.symbol,
                        side="buy",
                        price=order.fill_price,
                        quantity=quantity,
                        notional=quantity * order.fill_price,
                        fee=quantity * order.fill_price * fee_rate,
                        reason=f"dip_buy(score={order.score:.4f})",
                        direction="long",
                    ))
                    entries_executed += 1
                else:
                    alloc = min(max_alloc, config.initial_cash * config.max_position_pct)
                    quantity = alloc / order.fill_price
                    if quantity <= 0.0:
                        continue
                    margin_required = alloc * 0.5
                    if cash < margin_required:
                        continue
                    cash -= margin_required
                    borrowed = quantity * order.fill_price
                    positions[order.symbol] = Position(
                        symbol=order.symbol,
                        direction="short",
                        entry_price=order.fill_price,
                        entry_date=date,
                        quantity=quantity,
                        cost_basis=margin_required,
                        peak_price=float(bar["low"]),
                        target_exit_price=order.fill_price * (1 - config.profit_target_pct),
                        stop_price=order.fill_price * (1 + config.stop_loss_pct),
                        margin_borrowed=borrowed,
                    )
                    trades.append(TradeLog(
                        timestamp=date,
                        symbol=order.symbol,
                        side="short",
                        price=order.fill_price,
                        quantity=quantity,
                        notional=quantity * order.fill_price,
                        fee=quantity * order.fill_price * fee_rate,
                        reason=f"pump_short(score={order.score:.4f})",
                        direction="short",
                    ))
                    entries_executed += 1
            pending_entries = next_pending_entries
        elif config.daily_checkpoint_only and pending_entries:
            pending_entries = []

        entry_regime = resolve_entry_regime(
            current_bars=current_bars,
            history=history,
            config=config,
            symbol_metrics=symbol_metrics,
        )
        entry_config = entry_regime.config
        skip_entries = entry_regime.skip_entries

        # 3. Check entries (work-stealing)
        if len(positions) < config.max_positions and not skip_entries:
            slots = config.max_positions - len(positions)
            candidate_limit = None if (forecast_data and config.forecast_bias_weight > 0) else slots
            if entry_config.dip_pct_fallback:
                candidates, _tier_map = build_tiered_entry_candidates(
                    date=date,
                    current_bars=current_bars,
                    history=history,
                    positions=positions,
                    last_exit=last_exit,
                    config=entry_config,
                    base_symbol=base_symbol,
                    max_candidates=candidate_limit,
                    symbol_metrics=symbol_metrics,
                )
            else:
                candidates = build_entry_candidates(
                    date=date,
                    current_bars=current_bars,
                    history=history,
                    positions=positions,
                    last_exit=last_exit,
                    config=entry_config,
                    base_symbol=base_symbol,
                    symbol_metrics=symbol_metrics,
                    max_candidates=candidate_limit,
                )

            candidates_need_sort = bool(entry_config.dip_pct_fallback)
            if forecast_data and config.forecast_bias_weight > 0:
                from binance_worksteal.forecast_integration import get_forecast_multiplier
                adjusted = []
                for sym, direction, score, fill_price, bar in candidates:
                    fm = get_forecast_multiplier(sym, date, forecast_data, float(bar["close"]))
                    adjusted_score = score * (1 + config.forecast_bias_weight * fm)
                    adjusted.append((sym, direction, adjusted_score, fill_price, bar))
                candidates = adjusted
                candidates_need_sort = True

            if candidates_need_sort:
                candidates.sort(key=lambda x: x[2], reverse=True)
            candidates_generated += len(candidates)

            # Size new entries from starting equity to avoid compounding while
            # still respecting seeded start-state holdings.
            base_equity = starting_equity
            visible_candidates = candidates[:slots]
            candidates_visible += len(visible_candidates)
            market_breadth = entry_regime.market_breadth_ratio
            if (
                allocation_scale_fn is not None
                and visible_candidates
                and entry_config.market_breadth_filter <= 0
            ):
                market_breadth, _, _ = compute_breadth_ratio(
                    current_bars,
                    history,
                    symbol_metrics=symbol_metrics,
                )
            if config.daily_checkpoint_only:
                pending_entries = []
            for rank, (sym, direction, score, fill_price, bar) in enumerate(visible_candidates, start=1):
                if sym in positions:
                    continue
                if cash <= 0 and direction == "long" and config.max_leverage <= 1.0 and base_qty <= 0:
                    continue

                fee_rate = get_fee(sym, config)
                # Size based on initial equity, not current (prevents leverage spiral)
                max_alloc = base_equity * config.max_position_pct * config.max_leverage
                allocation_scale = 1.0
                if allocation_scale_fn is not None:
                    context = EntrySizingContext(
                        timestamp=date,
                        signal_timestamp=date,
                        symbol=sym,
                        direction=direction,
                        score=float(score),
                        fill_price=float(fill_price),
                        candidate_rank=rank,
                        candidate_count=len(visible_candidates),
                        slots_remaining=max(0, slots - rank + 1),
                        current_position_count=len(positions),
                        cash=float(cash),
                        base_equity=float(base_equity),
                        current_equity=float(current_equity),
                        market_breadth=float(market_breadth),
                        hold_base_asset=bool(hold_base_asset),
                        signal_bar=bar.copy(),
                        history=history.get(sym, pd.DataFrame()).copy(),
                        execution_bar=_execution_bar_for_signal(intraday_bars, sym, date),
                    )
                    allocation_scale = float(allocation_scale_fn(context))
                    if not np.isfinite(allocation_scale):
                        allocation_scale = 0.0
                    allocation_scale = max(0.0, allocation_scale)
                scaled_alloc = max_alloc * allocation_scale
                if scaled_alloc <= 0.0:
                    continue

                if config.daily_checkpoint_only:
                    pending_entries.append(
                        PendingEntry(
                            symbol=sym,
                            direction=direction,
                            score=float(score),
                            fill_price=float(fill_price),
                            signal_date=date,
                            signal_bar=bar.copy(),
                        )
                    )
                    continue

                if direction == "long":
                    if base_symbol is not None and hold_base_asset and base_symbol in current_bars:
                        base_fee = get_fee(base_symbol, config)
                        cash, base_qty = _sell_base_asset_for_cash(
                            cash=cash,
                            base_qty=base_qty,
                            price=float(current_bars[base_symbol]["close"]),
                            fee_rate=base_fee,
                            required_cash=scaled_alloc,
                        )
                    alloc = scaled_alloc
                    quantity = alloc / (fill_price * (1 + fee_rate))
                    if quantity <= 0:
                        continue

                    available_cash = float(cash)
                    actual_cost = quantity * fill_price * (1 + fee_rate)
                    borrowed = max(0.0, actual_cost - available_cash)
                    equity_used = actual_cost - borrowed
                    cash -= equity_used

                    positions[sym] = Position(
                        symbol=sym, direction="long",
                        entry_price=fill_price, entry_date=date,
                        quantity=quantity, cost_basis=equity_used,
                        peak_price=float(bar["high"]),
                        target_exit_price=fill_price * (1 + config.profit_target_pct),
                        stop_price=fill_price * (1 - config.stop_loss_pct),
                        margin_borrowed=borrowed,
                    )
                    trades.append(TradeLog(
                        timestamp=date, symbol=sym, side="buy",
                        price=fill_price, quantity=quantity,
                        notional=quantity * fill_price,
                        fee=quantity * fill_price * fee_rate,
                        reason=f"dip_buy(score={score:.4f})",
                        direction="long",
                    ))
                    entries_executed += 1
                else:  # short
                    alloc = min(scaled_alloc, base_equity * config.max_position_pct * allocation_scale)
                    quantity = alloc / fill_price
                    if quantity <= 0:
                        continue
                    # Borrow asset, sell it - but margin collateral comes from cash
                    margin_required = alloc * 0.5  # 50% margin requirement
                    if cash < margin_required:
                        continue
                    cash -= margin_required  # lock up margin
                    borrowed = quantity * fill_price

                    positions[sym] = Position(
                        symbol=sym, direction="short",
                        entry_price=fill_price, entry_date=date,
                        quantity=quantity, cost_basis=margin_required,
                        peak_price=float(bar["low"]),
                        target_exit_price=fill_price * (1 - config.profit_target_pct),
                        stop_price=fill_price * (1 + config.stop_loss_pct),
                        margin_borrowed=borrowed,
                    )
                    trades.append(TradeLog(
                        timestamp=date, symbol=sym, side="short",
                        price=fill_price, quantity=quantity,
                        notional=quantity * fill_price,
                        fee=quantity * fill_price * fee_rate,
                        reason=f"pump_short(score={score:.4f})",
                        direction="short",
                    ))
                    entries_executed += 1

        if base_symbol is not None and hold_base_asset and base_symbol in current_bars:
            base_fee = get_fee(base_symbol, config)
            cash, base_qty = _buy_base_asset(
                cash=cash,
                base_qty=base_qty,
                price=float(current_bars[base_symbol]["close"]),
                fee_rate=base_fee,
                min_cash=max(0.0, float(config.base_asset_rebalance_min_cash)),
            )

        # 3. Compute equity
        inventory_value = 0.0
        for sym, pos in positions.items():
            price = float(current_bars[sym]["close"]) if sym in current_bars else pos.entry_price
            inventory_value += _mark_to_market_position_value(pos, price, date, config)

        base_asset_value = 0.0
        if base_symbol is not None and base_symbol in current_bars:
            base_asset_value = base_qty * float(current_bars[base_symbol]["close"])

        equity = cash + inventory_value + base_asset_value

        total_notional = 0.0
        for sym, pos in positions.items():
            if sym in current_bars:
                total_notional += pos.quantity * float(current_bars[sym]["close"])
            else:
                total_notional += pos.quantity * pos.entry_price

        equity_rows.append({
            "timestamp": date,
            "equity": equity,
            "cash": cash,
            "inventory_value": inventory_value,
            "base_asset_symbol": base_symbol or "",
            "base_asset_qty": base_qty,
            "base_asset_value": base_asset_value,
            "n_positions": len(positions),
            "n_long": sum(1 for p in positions.values() if p.direction == "long"),
            "n_short": sum(1 for p in positions.values() if p.direction == "short"),
            "positions": ",".join(f"{p.direction[0]}:{s}" for s, p in positions.items()),
            "leverage": total_notional / max(equity, 1) if equity > 0 else 0,
        })
        equity_values.append(float(equity))
        if equity > peak_equity:
            peak_equity = float(equity)

        profit_vs_drawdown_exit = evaluate_drawdown_vs_profit_early_exit(
            equity_values,
            total_steps=len(all_dates),
            label="binance_worksteal.run_worksteal_backtest",
        )

        # Early exit on max drawdown or when drawdown already exceeds profit halfway through.
        if profit_vs_drawdown_exit.should_stop or (config.max_drawdown_exit > 0 and len(equity_rows) > 1):
            dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
            trigger_max_dd = config.max_drawdown_exit > 0 and dd < -config.max_drawdown_exit
            if profit_vs_drawdown_exit.should_stop or trigger_max_dd:
                # Force close all positions
                for sym, pos in list(positions.items()):
                    if sym in current_bars:
                        close_p = float(current_bars[sym]["close"])
                    else:
                        close_p = pos.entry_price
                    cash_delta, pnl, fee = _close_position_accounting(pos, close_p, date, config)
                    cash += cash_delta
                    trades.append(TradeLog(
                        timestamp=date, symbol=sym,
                        side="sell" if pos.direction == "long" else "cover",
                        price=close_p, quantity=pos.quantity,
                        notional=pos.quantity * close_p,
                        fee=fee,
                        pnl=pnl, reason="max_dd_exit",
                        direction=pos.direction,
                    ))
                positions.clear()
                if base_symbol is not None and base_symbol in current_bars:
                    base_fee = get_fee(base_symbol, config)
                    cash, base_qty = _sell_all_base_asset(
                        cash=cash,
                        base_qty=base_qty,
                        price=float(current_bars[base_symbol]["close"]),
                        fee_rate=base_fee,
                    )
                n_days_active = len(equity_rows)
                if profit_vs_drawdown_exit.should_stop:
                    print_early_exit(profit_vs_drawdown_exit)
                    print(
                        f"  EARLY EXIT: drawdown exceeded profit after {n_days_active}d, "
                        f"equity=${equity:.0f} -> ${cash:.0f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  EARLY EXIT: DD={dd:.1%} after {n_days_active}d, "
                        f"equity=${equity:.0f} -> ${cash:.0f}",
                        flush=True,
                    )
                equity_rows[-1]["equity"] = cash
                equity_rows[-1]["cash"] = cash
                equity_rows[-1]["base_asset_qty"] = base_qty
                equity_rows[-1]["base_asset_value"] = 0.0
                equity_rows[-1]["n_positions"] = 0
                break

    equity_df = pd.DataFrame(equity_rows)
    metrics = compute_metrics(equity_df, config, trades)
    if metrics:
        metrics["candidates_generated"] = int(candidates_generated)
        metrics["candidates_visible"] = int(candidates_visible)
        metrics["entries_executed"] = int(entries_executed)
        metrics["fill_rate"] = float(entries_executed) / max(float(candidates_generated), 1.0)
        metrics["visible_fill_rate"] = float(entries_executed) / max(float(candidates_visible), 1.0)
    return equity_df, trades, metrics


def compute_metrics(equity_df: pd.DataFrame, config: WorkStealConfig,
                    trades: Optional[List[TradeLog]] = None) -> Dict[str, float]:
    if equity_df.empty or len(equity_df) < 2:
        return {}
    values = equity_df["equity"].values.astype(float)
    if not np.isfinite(values).all() or values[0] <= 0:
        return {}
    returns = np.diff(values) / np.clip(values[:-1], 1e-8, None)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    total_return = (values[-1] - values[0]) / values[0]
    mean_ret = float(returns.mean())
    downside = returns[returns < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 1e-8

    sortino = mean_ret / max(downside_std, 1e-8) * np.sqrt(365)
    ret_std = float(returns.std()) if len(returns) > 1 else 1e-8
    sharpe = mean_ret / max(ret_std, 1e-8) * np.sqrt(365)

    peak = np.maximum.accumulate(values)
    drawdown = np.where(peak > 0, (values - peak) / peak, 0.0)
    max_dd = float(drawdown.min())

    n_days = len(equity_df)
    win_rate = 0.0
    exits: List[TradeLog] = []
    if trades:
        exits = get_exit_trades(trades)
        wins = [t for t in exits if t.pnl > 0]
        win_rate = len(wins) / len(exits) * 100 if exits else 0

    return {
        "total_return": float(total_return),
        "total_return_pct": float(total_return * 100),
        "sortino": float(sortino),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd * 100),
        "n_days": n_days,
        "n_orders": int(len(trades or [])),
        "n_trades": int(len(exits)),
        "final_equity": float(values[-1]),
        "mean_daily_return": float(mean_ret),
        "win_rate": win_rate,
    }


def load_daily_bars(data_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    from pathlib import Path

    def _read_ohlcv_csv(path: Path, *, symbol: str, min_rows: int) -> Optional[pd.DataFrame]:
        try:
            frame = pd.read_csv(path)
        except (OSError, UnicodeDecodeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            warnings.warn(
                f"Skipping daily bars for {symbol} from {path}: failed to read CSV ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        missing_columns = [column for column in OHLCV_COLUMNS if column not in frame.columns]
        if missing_columns:
            warnings.warn(
                f"Skipping daily bars for {symbol} from {path}: missing columns {missing_columns}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
        for column in OHLCV_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=OHLCV_COLUMNS).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if len(frame) <= min_rows:
            if not frame.empty:
                warnings.warn(
                    f"Skipping daily bars for {symbol} from {path}: only {len(frame)} valid rows after cleaning.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None
        return frame.reset_index(drop=True)

    data_path = Path(data_dir)
    result = {}
    for sym in symbols:
        base = sym.replace("USD", "").replace("USDT", "")
        # Prefer USDT (Binance Vision, more up-to-date) over USD (Alpaca)
        candidates = [f"{base}USDT.csv", f"{sym}.csv", f"{base}USD.csv"]
        for fname in candidates:
            fpath = data_path / fname
            if fpath.exists():
                df = _read_ohlcv_csv(fpath, symbol=f"{base}USD", min_rows=30)
                if df is None:
                    continue
                df["symbol"] = f"{base}USD"
                result[f"{base}USD"] = df
                break
    return result


def load_hourly_bars(data_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    from pathlib import Path

    def _read_ohlcv_csv(path: Path, *, symbol: str, min_rows: int) -> Optional[pd.DataFrame]:
        try:
            frame = pd.read_csv(path)
        except (OSError, UnicodeDecodeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            warnings.warn(
                f"Skipping hourly bars for {symbol} from {path}: failed to read CSV ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        missing_columns = [column for column in OHLCV_COLUMNS if column not in frame.columns]
        if missing_columns:
            warnings.warn(
                f"Skipping hourly bars for {symbol} from {path}: missing columns {missing_columns}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
        for column in OHLCV_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=OHLCV_COLUMNS).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if len(frame) <= min_rows:
            if not frame.empty:
                warnings.warn(
                    f"Skipping hourly bars for {symbol} from {path}: only {len(frame)} valid rows after cleaning.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None
        return frame[OHLCV_COLUMNS].reset_index(drop=True)

    result = {}
    data_path = Path(data_dir)
    for sym in symbols:
        candidates = [
            data_path / "crypto" / f"{sym}.csv",
            data_path / "stocks" / f"{sym}.csv",
            data_path / f"{sym}.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            frame = _read_ohlcv_csv(path, symbol=sym, min_rows=1)
            if frame is None:
                continue
            result[sym] = frame
            break
    return result


def print_results(equity_df: pd.DataFrame, trades: List[TradeLog], metrics: Dict[str, float]):
    print(f"\n{'='*60}")
    print(f"WORK-STEALING BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Sortino:      {metrics.get('sortino', 0):.2f}")
    print(f"Sharpe:       {metrics.get('sharpe', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Win Rate:     {metrics.get('win_rate', 0):.1f}%")
    print(f"Final Equity: ${metrics.get('final_equity', 0):.2f}")
    print(f"Days:         {metrics.get('n_days', 0)}")

    buys = get_entry_trades(trades)
    exits = get_exit_trades(trades)
    winning = [t for t in exits if t.pnl > 0]
    losing = [t for t in exits if t.pnl <= 0]

    print(f"\nEntries: {len(buys)} ({sum(1 for t in buys if t.direction=='long')} long, "
          f"{sum(1 for t in buys if t.direction=='short')} short)")
    print(f"Exits:   {len(exits)}")
    if exits:
        if winning:
            print(f"Avg Win:  ${np.mean([t.pnl for t in winning]):.2f}")
        if losing:
            print(f"Avg Loss: ${np.mean([t.pnl for t in losing]):.2f}")
        total_pnl = sum(t.pnl for t in exits)
        total_fees = sum(t.fee for t in trades)
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Total Fees: ${total_fees:.2f}")

    sym_pnl: Dict[str, float] = {}
    sym_trades: Dict[str, int] = {}
    for t in exits:
        sym_pnl[t.symbol] = sym_pnl.get(t.symbol, 0) + t.pnl
        sym_trades[t.symbol] = sym_trades.get(t.symbol, 0) + 1

    if sym_pnl:
        print(f"\nPer-Symbol PnL:")
        for sym in sorted(sym_pnl, key=lambda s: sym_pnl[s], reverse=True):
            print(f"  {sym:12s} ${sym_pnl[sym]:>8.2f} ({sym_trades[sym]} trades)")

    reasons: Dict[str, int] = {}
    for t in exits:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    if reasons:
        print(f"\nExit Reasons:")
        for r, c in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {r:20s} {c}")
