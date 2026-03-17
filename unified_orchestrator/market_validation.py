"""Deterministic portfolio-level RL checkpoint validation on the hourly market simulator.

This module replays the currently deployed RL bridge flow on local bar data, then
hands generated actions to ``HourlyTraderMarketSimulator`` so fills, cancels,
reserved cash, and working-order replacement are modeled the same way as the
live hourly trader.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm_hourly_trader.gemini_wrapper import TradePlan
from newnanoalpacahourlyexp.marketsimulator.hourly_trader import (
    HourlyTraderMarketSimulator,
    HourlyTraderSimulationConfig,
)
from src.robust_trading_metrics import (
    compute_max_drawdown,
    compute_market_sim_goodness_score,
    compute_pnl_smoothness_from_equity,
)
from unified_orchestrator.orchestrator import (
    CRYPTO_CHECKPOINT_CANDIDATES,
    CRYPTO_FORECAST_CACHE_CANDIDATES,
    CRYPTO_SYMBOLS,
    MAX_HOLD_HOURS,
    MIN_CONFIDENCE_CRYPTO,
    STOCK_CHECKPOINT_CANDIDATES,
    STOCK_FORECAST_CACHE_CANDIDATES,
    STOCK_SYMBOLS,
    TRAILING_STOP_PCT,
    _build_crypto_rl_signal_map,
    _build_stock_rl_signal_map,
    _choose_forecast_cache_root,
    _forecast_at,
    _load_forecast_frames,
    _num_symbols_from_obs_size,
    _read_trained_symbols_for_checkpoint,
)
from unified_orchestrator.rl_gemini_bridge import RLGeminiBridge, RLSignal

AssetClass = Literal["crypto", "stock"]
DecisionCadence = Literal["hourly", "daily"]

MAX_POSITION_PCT = 0.50
CRYPTO_FALLBACK_CAP = 0.40
STOCK_BASE_ALLOCATION = 0.20
CRYPTO_FEE_BUFFER_PCT = 0.0016
STOCK_FEE_BUFFER_PCT = 0.0010


@dataclass(frozen=True)
class ValidationConfig:
    asset_class: AssetClass
    days: int = 30
    initial_cash: float = 10_000.0
    decision_cadence: DecisionCadence = "hourly"
    decision_lag_bars: int = 1
    cancel_ack_delay_bars: int = 1
    fill_buffer_bps: float = 5.0
    partial_fill_on_touch: bool = True
    keep_similar_orders: bool = True
    price_tol_pct: float = 0.0003
    qty_tol_pct: float = 0.05
    qty_tol_notional_usd: float = 100.0
    max_leverage: float = 1.0
    allow_position_adds: bool = False
    max_hold_hours: int | None = None
    trailing_stop_pct: float | None = None
    force_exit_offset_pct: float = 0.1


@dataclass(frozen=True)
class ValidationResult:
    checkpoint: str
    asset_class: str
    eval_symbols: list[str]
    signal_universe: list[str]
    start_timestamp: str
    end_timestamp: str
    bars: int
    fills: int
    trade_count: int
    win_rate: float
    return_pct: float
    final_equity: float
    sortino: float
    max_drawdown_pct: float
    pnl_smoothness: float
    goodness_score: float


def _default_eval_symbols(asset_class: AssetClass) -> list[str]:
    if asset_class == "crypto":
        return list(CRYPTO_SYMBOLS)
    return list(STOCK_SYMBOLS)


def _default_candidates(asset_class: AssetClass) -> list[Path]:
    if asset_class == "crypto":
        return [Path(path) for path in CRYPTO_CHECKPOINT_CANDIDATES if Path(path).exists()]
    return [Path(path) for path in STOCK_CHECKPOINT_CANDIDATES if Path(path).exists()]


def _load_local_bars(symbol: str, asset_class: AssetClass) -> pd.DataFrame:
    symbol = str(symbol).upper()
    if asset_class == "crypto":
        candidates = [
            REPO / "trainingdatahourly" / "crypto" / f"{symbol}.csv",
            REPO / "binance_spot_hourly" / f"{symbol.replace('USD', 'USDT')}.csv",
        ]
    else:
        candidates = [
            REPO / "trainingdatahourly" / "stocks" / f"{symbol}_hist.pkl",
            REPO / "trainingdatahourly" / "stocks" / f"{symbol}.csv",
        ]

    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(f"No local bars found for {symbol} ({asset_class})")

    if source.suffix == ".pkl":
        raw = pd.read_pickle(source)
        frame = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
    else:
        frame = pd.read_csv(source)

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Unsupported bar payload for {source}")

    frame = frame.copy()
    if "timestamp" not in frame.columns:
        if "date" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["date"], utc=True)
        elif isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index()
            if frame.columns[0] != "timestamp":
                frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        else:
            raise ValueError(f"{source} is missing timestamp/date columns")
    else:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    for column in ("open", "high", "low", "close"):
        if column not in frame.columns:
            raise ValueError(f"{source} is missing required column {column!r}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame["symbol"] = symbol
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return frame.loc[:, ["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


def _lookup_history(frame: pd.DataFrame, timestamp: pd.Timestamp, *, lookback: int = 72) -> pd.DataFrame | None:
    if frame.empty:
        return None
    idx = frame["timestamp"].searchsorted(timestamp, side="right")
    if idx <= 0:
        return None
    start = max(0, int(idx) - int(lookback))
    return frame.iloc[start:int(idx)].reset_index(drop=True)


def _atr_pct(frame: pd.DataFrame | None, *, window: int = 12) -> float:
    if frame is None or frame.empty:
        return 0.003
    sample = frame.tail(int(window))
    close = float(sample["close"].iloc[-1])
    if close <= 0.0:
        return 0.003
    atr = float((sample["high"] - sample["low"]).mean())
    return max(0.0005, atr / close)


def _forecast_edges(
    current_price: float,
    forecast_1h: dict | None,
    forecast_24h: dict | None,
) -> tuple[float, float]:
    if current_price <= 0.0:
        return 0.0, 0.0

    upside_targets: list[float] = []
    downside_targets: list[float] = []
    for forecast in (forecast_1h, forecast_24h):
        if not forecast:
            continue
        for key in ("predicted_close_p50", "predicted_high_p50", "predicted_close_p90"):
            value = float(forecast.get(key, 0.0) or 0.0)
            if value > 0.0:
                upside_targets.append(value / current_price - 1.0)
        for key in ("predicted_low_p50", "predicted_close_p10"):
            value = float(forecast.get(key, 0.0) or 0.0)
            if value > 0.0:
                downside_targets.append(1.0 - value / current_price)

    upside = max(0.0, max(upside_targets) if upside_targets else 0.0)
    downside = max(0.0, max(downside_targets) if downside_targets else 0.0)
    return upside, downside


def build_market_sim_trade_plan(
    *,
    signal: RLSignal | None,
    current_price: float,
    history_frame: pd.DataFrame | None,
    forecast_1h: dict | None,
    forecast_24h: dict | None,
    asset_class: AssetClass,
    confidence_threshold: float,
) -> TradePlan:
    """Convert the RL hint into a deterministic limit-order plan for offline replay."""

    atr_pct = _atr_pct(history_frame)
    upside_pct, downside_pct = _forecast_edges(current_price, forecast_1h, forecast_24h)
    fee_buffer = CRYPTO_FEE_BUFFER_PCT if asset_class == "crypto" else STOCK_FEE_BUFFER_PCT
    min_entry_pullback = 0.0005 if asset_class == "crypto" else 0.00025
    max_entry_pullback = 0.0040 if asset_class == "crypto" else 0.0025
    max_target_pct = 0.03 if asset_class == "crypto" else 0.02

    signal = signal or RLSignal(
        symbol_idx=-1,
        symbol_name="",
        direction="flat",
        confidence=0.0,
        logit_gap=0.0,
        allocation_pct=0.0,
    )
    is_long = signal.direction == "long" and float(signal.confidence) >= float(confidence_threshold)

    base_pullback = 0.35 * atr_pct
    level_offset_pct = float(signal.level_offset_bps) / 10_000.0
    # Strong upside and positive level offsets move the entry closer to market.
    entry_pullback = base_pullback - min(upside_pct, 0.02) * 0.30 - level_offset_pct
    entry_pullback = min(max(entry_pullback, min_entry_pullback), max_entry_pullback)
    buy_price = max(current_price * (1.0 - entry_pullback), current_price * 0.95)

    long_target_pct = max(
        fee_buffer + 0.0010,
        0.60 * atr_pct,
        0.60 * min(upside_pct, max_target_pct),
    )
    hold_target_pct = max(
        fee_buffer * 0.75,
        0.35 * atr_pct,
        0.20 * min(upside_pct, max_target_pct),
    )
    if downside_pct > 0.0 and not is_long:
        hold_target_pct = min(hold_target_pct, max(fee_buffer * 0.5, 0.0010))

    target_pct = long_target_pct if is_long else hold_target_pct
    target_pct = min(max_target_pct, max(target_pct, fee_buffer + 0.0005))
    sell_price = current_price * (1.0 + target_pct)
    sell_price = max(sell_price, buy_price * (1.0 + fee_buffer + 0.0005))

    return TradePlan(
        direction="long" if is_long else "hold",
        buy_price=float(buy_price),
        sell_price=float(sell_price),
        confidence=float(signal.confidence),
        reasoning="deterministic_market_validation",
        allocation_pct=float(max(0.0, signal.allocation_pct) * 100.0),
    )


def _scaled_buy_amount_pct(
    *,
    signal: RLSignal | None,
    spec_alloc_bins: int,
    long_signal_count: int,
    asset_class: AssetClass,
) -> float:
    signal = signal or RLSignal(
        symbol_idx=-1,
        symbol_name="",
        direction="flat",
        confidence=0.0,
        logit_gap=0.0,
        allocation_pct=0.0,
    )
    if signal.direction != "long":
        return 0.0

    if asset_class == "crypto":
        base_alloc = min(CRYPTO_FALLBACK_CAP, 1.0 / max(int(long_signal_count), 1))
    else:
        base_alloc = STOCK_BASE_ALLOCATION

    if int(spec_alloc_bins) > 1:
        desired_alloc = base_alloc + (MAX_POSITION_PCT - base_alloc) * max(0.0, min(1.0, float(signal.allocation_pct)))
    else:
        desired_alloc = base_alloc

    desired_alloc = min(MAX_POSITION_PCT, max(base_alloc, desired_alloc))
    return max(0.0, min(100.0, desired_alloc / MAX_POSITION_PCT * 100.0))


def _candidate_timestamps(bar_frames: dict[str, pd.DataFrame], eval_symbols: list[str], days: int) -> tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]:
    usable = {sym: frame for sym, frame in bar_frames.items() if sym in eval_symbols and not frame.empty}
    if not usable:
        raise ValueError("No usable evaluation symbols loaded")

    end_ts = min(frame["timestamp"].max() for frame in usable.values())
    start_ts = end_ts - pd.Timedelta(days=int(days))
    timestamps = sorted(
        {
            pd.Timestamp(ts)
            for frame in usable.values()
            for ts in frame.loc[(frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts), "timestamp"]
        }
    )
    if not timestamps:
        raise ValueError("No timestamps available in the requested validation window")
    return pd.Timestamp(start_ts), pd.Timestamp(end_ts), timestamps


def _decision_timestamps(
    timestamps: list[pd.Timestamp],
    *,
    cadence: DecisionCadence,
) -> list[pd.Timestamp]:
    if cadence == "hourly":
        return list(timestamps)
    if cadence != "daily":
        raise ValueError(f"Unsupported decision cadence: {cadence}")

    first_by_day: dict[pd.Timestamp, pd.Timestamp] = {}
    for ts in timestamps:
        day = pd.Timestamp(ts).floor("D")
        first_by_day.setdefault(day, pd.Timestamp(ts))
    return list(first_by_day.values())


def build_validation_frames(
    *,
    checkpoint_path: str | Path,
    config: ValidationConfig,
    eval_symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Timestamp, pd.Timestamp]:
    checkpoint = Path(checkpoint_path)
    bridge = RLGeminiBridge(checkpoint_path=str(checkpoint))
    spec = bridge.get_checkpoint_spec()
    signal_universe = _read_trained_symbols_for_checkpoint(
        checkpoint,
        _num_symbols_from_obs_size(spec.obs_size),
    )
    eval_symbols = [str(sym).upper() for sym in (eval_symbols or _default_eval_symbols(config.asset_class))]

    load_symbols = list(dict.fromkeys([*signal_universe, *eval_symbols]))
    bar_frames: dict[str, pd.DataFrame] = {}
    missing_eval: list[str] = []
    for sym in load_symbols:
        try:
            bar_frames[sym] = _load_local_bars(sym, config.asset_class)
        except FileNotFoundError:
            if sym in eval_symbols:
                missing_eval.append(sym)
    if missing_eval:
        raise FileNotFoundError(
            f"Missing local bars for evaluation symbols: {', '.join(sorted(missing_eval))}"
        )
    start_ts, end_ts, timestamps = _candidate_timestamps(bar_frames, eval_symbols, config.days)

    if config.asset_class == "crypto":
        forecast_root = _choose_forecast_cache_root(load_symbols, CRYPTO_FORECAST_CACHE_CANDIDATES)
        forecasts_1h, forecasts_24h = _load_forecast_frames(load_symbols, forecast_root)
        confidence_threshold = MIN_CONFIDENCE_CRYPTO
    else:
        forecast_root = _choose_forecast_cache_root(load_symbols, STOCK_FORECAST_CACHE_CANDIDATES)
        forecasts_1h, forecasts_24h = _load_forecast_frames(load_symbols, forecast_root)
        confidence_threshold = 0.50

    bars_rows: list[dict[str, float | str | pd.Timestamp]] = []
    action_rows: list[dict[str, float | str | pd.Timestamp]] = []
    history_frames_by_ts: dict[pd.Timestamp, dict[str, pd.DataFrame]] = {}
    current_rows_by_ts: dict[pd.Timestamp, dict[str, pd.Series]] = {}

    for ts in timestamps:
        history_frames: dict[str, pd.DataFrame] = {}
        current_rows: dict[str, pd.Series] = {}
        for sym, frame in bar_frames.items():
            history = _lookup_history(frame, ts, lookback=72)
            if history is not None and not history.empty:
                history_frames[sym] = history
                current_rows[sym] = history.iloc[-1]
        history_frames_by_ts[pd.Timestamp(ts)] = history_frames
        current_rows_by_ts[pd.Timestamp(ts)] = current_rows

        for sym in eval_symbols:
            row = current_rows.get(sym)
            if row is None:
                continue
            bars_rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )

    decision_timestamps = _decision_timestamps(timestamps, cadence=config.decision_cadence)
    decision_rows: list[dict[str, float | str | pd.Timestamp]] = []
    for ts in decision_timestamps:
        history_frames = history_frames_by_ts[pd.Timestamp(ts)]
        current_rows = current_rows_by_ts[pd.Timestamp(ts)]

        if config.asset_class == "crypto":
            signal_map = _build_crypto_rl_signal_map(history_frames, bridge)
        else:
            signal_map = _build_stock_rl_signal_map(history_frames, bridge, forecasts_1h, forecasts_24h)

        long_signal_count = sum(
            1
            for sym in eval_symbols
            for signal in [signal_map.get(sym)]
            if signal is not None and signal.direction == "long" and signal.confidence >= confidence_threshold
        )

        for sym in eval_symbols:
            row = current_rows.get(sym)
            if row is None:
                continue

            history = history_frames.get(sym)
            current_price = float(row["close"])
            forecast_1h = _forecast_at(forecasts_1h.get(sym), ts)
            forecast_24h = _forecast_at(forecasts_24h.get(sym), ts)
            signal = signal_map.get(sym)
            plan = build_market_sim_trade_plan(
                signal=signal,
                current_price=current_price,
                history_frame=history,
                forecast_1h=forecast_1h,
                forecast_24h=forecast_24h,
                asset_class=config.asset_class,
                confidence_threshold=confidence_threshold,
            )
            buy_amount = _scaled_buy_amount_pct(
                signal=signal,
                spec_alloc_bins=spec.alloc_bins,
                long_signal_count=long_signal_count,
                asset_class=config.asset_class,
            )
            decision_rows.append(
                {
                    "timestamp": ts,
                    "decision_timestamp": ts,
                    "symbol": sym,
                    "buy_price": float(plan.buy_price),
                    "sell_price": float(plan.sell_price),
                    "buy_amount": float(buy_amount),
                    "sell_amount": 100.0 if plan.sell_price > 0 else 0.0,
                }
            )

    if config.decision_cadence == "daily":
        plan_by_day_symbol = {
            (pd.Timestamp(row["decision_timestamp"]).floor("D"), str(row["symbol"]).upper()): row
            for row in decision_rows
        }
        for ts in timestamps:
            current_rows = current_rows_by_ts[pd.Timestamp(ts)]
            day = pd.Timestamp(ts).floor("D")
            for sym in eval_symbols:
                if sym not in current_rows:
                    continue
                row = plan_by_day_symbol.get((day, sym))
                if row is None:
                    continue
                action_rows.append(
                    {
                        "timestamp": ts,
                        "decision_timestamp": row["decision_timestamp"],
                        "symbol": sym,
                        "buy_price": float(row["buy_price"]),
                        "sell_price": float(row["sell_price"]),
                        "buy_amount": float(row["buy_amount"]),
                        "sell_amount": float(row["sell_amount"]),
                    }
                )
    else:
        action_rows = decision_rows

    bars_df = pd.DataFrame(bars_rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    actions_df = pd.DataFrame(action_rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return bars_df, actions_df, signal_universe, start_ts, end_ts


def _sim_config_for(asset_class: AssetClass, initial_cash: float) -> ValidationConfig:
    if asset_class == "crypto":
        return ValidationConfig(
            asset_class=asset_class,
            initial_cash=float(initial_cash),
            max_leverage=1.0,
            allow_position_adds=True,
            max_hold_hours=MAX_HOLD_HOURS,
            trailing_stop_pct=TRAILING_STOP_PCT,
        )
    return ValidationConfig(
        asset_class=asset_class,
        initial_cash=float(initial_cash),
        max_leverage=2.0,
        allow_position_adds=False,
    )


def run_validation(
    *,
    checkpoint_path: str | Path,
    asset_class: AssetClass,
    days: int = 30,
    initial_cash: float = 10_000.0,
    decision_cadence: DecisionCadence = "hourly",
    eval_symbols: list[str] | None = None,
) -> ValidationResult:
    sim_profile = _sim_config_for(asset_class, initial_cash)
    config = ValidationConfig(
        asset_class=asset_class,
        days=int(days),
        initial_cash=float(initial_cash),
        decision_cadence=decision_cadence,
        decision_lag_bars=sim_profile.decision_lag_bars,
        cancel_ack_delay_bars=sim_profile.cancel_ack_delay_bars,
        fill_buffer_bps=sim_profile.fill_buffer_bps,
        partial_fill_on_touch=sim_profile.partial_fill_on_touch,
        keep_similar_orders=sim_profile.keep_similar_orders,
        price_tol_pct=sim_profile.price_tol_pct,
        qty_tol_pct=sim_profile.qty_tol_pct,
        qty_tol_notional_usd=sim_profile.qty_tol_notional_usd,
        max_leverage=sim_profile.max_leverage,
        allow_position_adds=sim_profile.allow_position_adds,
        max_hold_hours=sim_profile.max_hold_hours,
        trailing_stop_pct=sim_profile.trailing_stop_pct,
        force_exit_offset_pct=sim_profile.force_exit_offset_pct,
    )

    bars_df, actions_df, signal_universe, start_ts, end_ts = build_validation_frames(
        checkpoint_path=checkpoint_path,
        config=config,
        eval_symbols=eval_symbols,
    )

    sim = HourlyTraderMarketSimulator(
        HourlyTraderSimulationConfig(
            initial_cash=config.initial_cash,
            allocation_pct=MAX_POSITION_PCT,
            allocation_mode="per_symbol",
            max_leverage=config.max_leverage,
            allow_short=False,
            allow_position_adds=config.allow_position_adds,
            always_full_exit=True,
            enforce_market_hours=(asset_class == "stock"),
            decision_lag_bars=config.decision_lag_bars,
            cancel_ack_delay_bars=config.cancel_ack_delay_bars,
            fill_buffer_bps=config.fill_buffer_bps,
            partial_fill_on_touch=config.partial_fill_on_touch,
            keep_similar_orders=config.keep_similar_orders,
            price_tol_pct=config.price_tol_pct,
            qty_tol_pct=config.qty_tol_pct,
            qty_tol_notional_usd=config.qty_tol_notional_usd,
            symbols=eval_symbols or _default_eval_symbols(asset_class),
            max_hold_hours=config.max_hold_hours,
            trailing_stop_pct=config.trailing_stop_pct,
            force_exit_offset_pct=config.force_exit_offset_pct,
        )
    )
    result = sim.run(bars_df, actions_df)

    exit_fills = [fill for fill in result.fills if fill.kind == "exit"]
    entry_prices: dict[str, float] = {}
    wins = 0
    closed_trades = 0
    for fill in result.fills:
        if fill.kind == "entry":
            entry_prices[fill.symbol] = float(fill.price)
        elif fill.kind == "exit" and fill.symbol in entry_prices:
            closed_trades += 1
            if float(fill.price) > float(entry_prices[fill.symbol]):
                wins += 1
            entry_prices.pop(fill.symbol, None)

    final_equity = float(result.equity_curve.iloc[-1])
    total_return = final_equity / float(config.initial_cash) - 1.0
    smoothness = compute_pnl_smoothness_from_equity(result.equity_curve.to_numpy(dtype=float))
    max_drawdown_frac = float(compute_max_drawdown(result.equity_curve.to_numpy(dtype=float)))
    goodness = compute_market_sim_goodness_score(
        total_return=float(total_return),
        sortino=float(result.metrics.get("sortino", 0.0)),
        max_drawdown=float(max_drawdown_frac),
        pnl_smoothness=float(smoothness),
        trade_count=closed_trades,
        period_count=len(result.per_hour),
    )

    return ValidationResult(
        checkpoint=str(Path(checkpoint_path)),
        asset_class=asset_class,
        eval_symbols=[str(sym).upper() for sym in (eval_symbols or _default_eval_symbols(asset_class))],
        signal_universe=list(signal_universe),
        start_timestamp=start_ts.isoformat(),
        end_timestamp=end_ts.isoformat(),
        bars=int(len(result.per_hour)),
        fills=int(len(result.fills)),
        trade_count=int(closed_trades),
        win_rate=float(wins / closed_trades) if closed_trades else 0.0,
        return_pct=float(total_return * 100.0),
        final_equity=float(final_equity),
        sortino=float(result.metrics.get("sortino", 0.0)),
        max_drawdown_pct=float(max_drawdown_frac * 100.0),
        pnl_smoothness=float(smoothness),
        goodness_score=float(goodness),
    )


def _print_summary(results: list[ValidationResult]) -> None:
    print(
        "checkpoint | return_pct | sortino | max_dd_pct | trades | win_rate | goodness | bars"
    )
    for item in sorted(results, key=lambda row: row.goodness_score, reverse=True):
        checkpoint_name = Path(item.checkpoint).parent.name
        print(
            f"{checkpoint_name:24s} | "
            f"{item.return_pct:8.2f} | "
            f"{item.sortino:7.2f} | "
            f"{item.max_drawdown_pct:10.2f} | "
            f"{item.trade_count:6d} | "
            f"{item.win_rate:8.1%} | "
            f"{item.goodness_score:8.2f} | "
            f"{item.bars:4d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic 30-day RL checkpoint market validation")
    parser.add_argument("--asset-class", choices=["crypto", "stock"], default="crypto")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--decision-cadence", choices=["hourly", "daily"], default="hourly")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--checkpoint", action="append", default=None, help="Checkpoint path(s) to validate")
    parser.add_argument("--write-json", default=None, help="Optional path to persist the validation summary")
    args = parser.parse_args()

    asset_class: AssetClass = args.asset_class
    checkpoints = [Path(path) for path in (args.checkpoint or [])]
    if not checkpoints:
        checkpoints = _default_candidates(asset_class)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found for asset_class={asset_class}")

    results = [
        run_validation(
            checkpoint_path=checkpoint,
            asset_class=asset_class,
            days=int(args.days),
            initial_cash=float(args.cash),
            decision_cadence=args.decision_cadence,
            eval_symbols=args.symbols,
        )
        for checkpoint in checkpoints
    ]
    _print_summary(results)

    if args.write_json:
        payload = [asdict(item) for item in results]
        output_path = Path(args.write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
