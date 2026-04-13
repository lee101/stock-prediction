"""Validate RL model performance through the Python marketsimulator.

Runs checkpoint inference on historical bars, converts the resulting signals
into simulator actions, and evaluates them through ``marketsimulator.py``.

Notes:
- The shared-cash simulator is long-only. Short signals are flattened and
  counted in the action-generation summary unless ``--long-only`` masks them
  out during inference.
- Orders are emitted at the current bar close. With the default simulator
  fill rules this behaves like a close-priced limit that typically fills on the
  same bar, which is suitable for pipeline validation but still optimistic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from pufferlib_market.export_data_daily import compute_daily_features as compute_daily_feature_history
from pufferlib_market.inference import PPOTrader, TradingSignal, compute_hourly_features as compute_hourly_feature_snapshot
from pufferlib_market.inference_daily import DailyPPOTrader


def load_hourly_bars(symbol: str, data_root: str = "trainingdatahourly") -> pd.DataFrame:
    """Load hourly OHLCV for a symbol."""
    root = Path(data_root)
    for subdir in ["crypto", "stocks", ""]:
        path = root / subdir / f"{symbol}.csv" if subdir else root / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["symbol"] = symbol
            return df.sort_values("timestamp")
    raise FileNotFoundError(f"No data for {symbol}")


def load_daily_bars(symbol: str, data_root: str = "trainingdata/train") -> pd.DataFrame:
    """Load daily OHLCV for a symbol."""
    root = Path(data_root)
    for subdir in ["crypto", "stocks", ""]:
        path = root / subdir / f"{symbol}.csv" if subdir else root / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            ts_col = "timestamp" if "timestamp" in df.columns else "date"
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
            df["symbol"] = symbol
            return df.sort_values("timestamp")
    raise FileNotFoundError(f"No data for {symbol}")


FEE_TIERS = {
    "fdusd": 0.0,       # Binance FDUSD promotional
    "usdt": 0.001,      # Binance USDT standard
    "conservative": 0.0015,  # Conservative estimate
}

SLIPPAGE_BPS = {
    "BTCUSD": 2,
    "ETHUSD": 3,
    "SOLUSD": 5,
    "BNBUSD": 5,
    "LTCUSD": 8,
    "AVAXUSD": 10,
    "DOGEUSD": 8,
    "LINKUSD": 8,
    "AAVEUSD": 12,
}


def _load_marketsimulator_api():
    spec = importlib.util.spec_from_file_location(
        "_validate_marketsim_legacy_module",
        REPO_ROOT / "marketsimulator.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load legacy marketsimulator.py module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimulationConfig, module.run_shared_cash_simulation


def _minimum_history_bars(timeframe: str) -> int:
    return 72 if str(timeframe).strip().lower() == "hourly" else 60


def _load_trader_for_timeframe(
    *,
    checkpoint: str,
    symbols: list[str],
    timeframe: str,
    device: str,
    long_only: bool,
):
    normalized = str(timeframe).strip().lower()
    effective_long_only = bool(long_only or normalized == "daily")
    if normalized == "daily":
        return DailyPPOTrader(checkpoint, device=device, long_only=effective_long_only, symbols=symbols)
    return PPOTrader(checkpoint, device=device, long_only=effective_long_only, symbols=symbols)


def _align_symbol_frames(
    bars: pd.DataFrame,
    *,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    if bars.empty:
        raise ValueError("bars must not be empty")
    common_timestamps: pd.Index | None = None
    indexed_frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = bars[bars["symbol"].astype(str).str.upper() == symbol].copy()
        if frame.empty:
            raise ValueError(f"No bars available for {symbol}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = (
            frame.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .set_index("timestamp")
        )
        indexed_frames[symbol] = frame
        common_timestamps = frame.index if common_timestamps is None else common_timestamps.intersection(frame.index)
    if common_timestamps is None or common_timestamps.empty:
        raise ValueError("No common timestamps across all requested symbols")
    common_timestamps = common_timestamps.sort_values()
    aligned: dict[str, pd.DataFrame] = {}
    for symbol, frame in indexed_frames.items():
        aligned_frame = frame.loc[common_timestamps].reset_index()
        aligned_frame["symbol"] = symbol
        aligned[symbol] = aligned_frame
    return aligned


def _set_trader_shadow_state(
    trader,
    *,
    symbol_to_index: dict[str, int],
    held_symbol: str | None,
    hold_bars: int,
    entry_price: float,
    step_index: int,
) -> None:
    # The C training env starts with INITIAL_CASH=10_000 and buys
    # qty = INITIAL_CASH / entry_price units, so obs[base+1] = pos_val /
    # INITIAL_CASH ≈ 1.0 when fully invested.  Using position_qty=1.0 here
    # would give obs[base+1] = cur_price / 10_000 — up to 8.5× off for BTC at
    # $85k.  Use the same normalization as the C env.
    _ep = float(entry_price) if held_symbol and entry_price and entry_price > 0 else 0.0
    trader.cash = 10_000.0
    trader.position_qty = (10_000.0 / _ep) if (held_symbol and _ep > 0) else 0.0
    trader.entry_price = _ep
    trader.current_position = symbol_to_index[held_symbol] if held_symbol else None
    trader.step = int(max(0, step_index))
    trader.hold_hours = int(max(0, hold_bars))
    if hasattr(trader, "hold_days"):
        trader.hold_days = int(max(0, hold_bars))


def _signal_allocation(signal: TradingSignal) -> float:
    return float(np.clip(float(getattr(signal, "allocation_pct", 1.0) or 0.0), 0.0, 1.0))


def _compute_hourly_feature_history(price_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized hourly feature history matching pufferlib_market.inference.compute_hourly_features()."""
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    volume = price_df["volume"].astype(float)

    ret_1h = close.diff().fillna(0.0) / close.clip(lower=1e-8)
    ret_4h = (close - close.shift(4)) / close.shift(4).clip(lower=1e-8)
    ret_24h = (close - close.shift(24)) / close.shift(24).clip(lower=1e-8)

    ma_24 = close.rolling(24, min_periods=1).mean()
    ma_72 = close.rolling(72, min_periods=1).mean()
    ema_24 = close.ewm(span=24, min_periods=1).mean()

    atr_24 = (high - low).rolling(24, min_periods=1).mean()
    vol_ratio = volume / (volume.rolling(24, min_periods=1).mean() + 1e-8)
    range_pos = (close - low) / (high - low + 1e-8)
    ma_ratio = close / (ma_24 + 1e-8)
    macd_proxy = ema_24 - ma_72

    def _zscore(series: pd.Series, window: int = 72) -> pd.Series:
        roll = series.rolling(window, min_periods=1)
        return (series - roll.mean()) / (roll.std() + 1e-8)

    features = pd.DataFrame(
        {
            "f0": _zscore(ret_1h),
            "f1": _zscore(ret_4h),
            "f2": _zscore(ret_24h),
            "f3": _zscore(ma_ratio - 1.0),
            "f4": _zscore(range_pos - 0.5),
            "f5": _zscore(vol_ratio - 1.0),
            "f6": _zscore(atr_24 / close.clip(lower=1e-8)),
            "f7": _zscore(macd_proxy / close.clip(lower=1e-8)),
            "f8": 0.0,
            "f9": 0.0,
            "f10": 0.0,
            "f11": 0.0,
            "f12": 0.0,
            "f13": 0.0,
            "f14": 0.0,
            "f15": 0.0,
        },
        index=price_df.index,
    )
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)

    # The legacy inference helper uses prefix-local np.roll() calls for 4h/24h
    # returns. That only affects early rows before the longest lag plus z-score
    # window have fully rolled through, so patch those rows back to the exact
    # legacy snapshots while keeping the rest vectorized.
    warmup_rows = min(len(price_df), 96)
    if warmup_rows > 0:
        legacy_rows = [
            compute_hourly_feature_snapshot(price_df.iloc[: idx + 1])
            for idx in range(warmup_rows)
        ]
        features.iloc[:warmup_rows] = np.asarray(legacy_rows, dtype=np.float32)
    return features


def _build_feature_history(
    aligned_frames: dict[str, pd.DataFrame],
    *,
    timeframe: str,
) -> dict[str, pd.DataFrame]:
    normalized = str(timeframe).strip().lower()
    feature_history: dict[str, pd.DataFrame] = {}
    for symbol, frame in aligned_frames.items():
        ohlcv = frame[["open", "high", "low", "close", "volume"]].copy()
        if normalized == "hourly":
            feature_history[symbol] = _compute_hourly_feature_history(ohlcv)
        else:
            feature_history[symbol] = compute_daily_feature_history(ohlcv)
    return feature_history


def _prepare_action_inputs(
    aligned_frames: dict[str, pd.DataFrame],
    feature_history: dict[str, pd.DataFrame],
    *,
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, tuple[pd.Timestamp, ...]]:
    feature_blocks = [
        feature_history[symbol].to_numpy(dtype=np.float32, copy=False)
        for symbol in symbols
    ]
    close_columns = [
        aligned_frames[symbol]["close"].to_numpy(dtype=np.float64, copy=False)
        for symbol in symbols
    ]
    timestamps = tuple(
        pd.to_datetime(aligned_frames[symbols[0]]["timestamp"], utc=True).tolist()
    )
    feature_cube = np.stack(feature_blocks, axis=1)
    close_matrix = np.stack(close_columns, axis=1)
    return feature_cube, close_matrix, timestamps


def _generate_policy_actions(
    *,
    bars: pd.DataFrame,
    checkpoint: str,
    symbols: list[str],
    timeframe: str,
    device: str,
    long_only: bool,
    min_history_bars: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    normalized_timeframe = str(timeframe).strip().lower()
    if normalized_timeframe not in {"hourly", "daily"}:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")

    aligned_frames = _align_symbol_frames(bars, symbols=symbols)
    trader = _load_trader_for_timeframe(
        checkpoint=checkpoint,
        symbols=symbols,
        timeframe=normalized_timeframe,
        device=device,
        long_only=long_only,
    )
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols)}
    feature_history = _build_feature_history(aligned_frames, timeframe=normalized_timeframe)
    feature_cube, close_matrix, timestamps = _prepare_action_inputs(
        aligned_frames,
        feature_history,
        symbols=symbols,
    )
    total_steps = len(next(iter(aligned_frames.values())))
    history_bars = int(_minimum_history_bars(normalized_timeframe) if min_history_bars is None else min_history_bars)
    if total_steps <= history_bars:
        raise ValueError(
            f"Need more aligned history for {normalized_timeframe} inference: "
            f"have {total_steps} bars, need > {history_bars}"
        )

    held_symbol: str | None = None
    hold_bars = 0
    entry_price = 0.0
    generated_rows: list[dict[str, object]] = []
    buy_rows = 0
    sell_rows = 0
    short_signals_flattened = 0

    for idx in range(history_bars, total_steps):
        _set_trader_shadow_state(
            trader,
            symbol_to_index=symbol_to_index,
            held_symbol=held_symbol,
            hold_bars=hold_bars,
            entry_price=entry_price,
            step_index=idx,
        )

        prices = {
            symbol: float(close_matrix[idx, symbol_idx])
            for symbol_idx, symbol in enumerate(symbols)
        }
        features = feature_cube[idx]
        signal = trader.get_signal(features, prices)

        timestamp = pd.Timestamp(timestamps[idx])
        per_symbol_rows = {
            symbol: {
                "timestamp": timestamp,
                "symbol": symbol,
                "buy_price": 0.0,
                "sell_price": 0.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
            }
            for symbol in symbols
        }

        desired_symbol = str(signal.symbol).upper() if signal.symbol and signal.direction == "long" else None
        should_close = held_symbol is not None and held_symbol != desired_symbol
        if should_close and held_symbol is not None:
            per_symbol_rows[held_symbol]["sell_price"] = prices[held_symbol]
            per_symbol_rows[held_symbol]["sell_amount"] = 1.0
            sell_rows += 1

        if signal.direction == "short":
            short_signals_flattened += 1

        if desired_symbol is not None and desired_symbol in per_symbol_rows and held_symbol != desired_symbol:
            allocation = _signal_allocation(signal)
            if allocation > 0.0:
                per_symbol_rows[desired_symbol]["buy_price"] = prices[desired_symbol]
                per_symbol_rows[desired_symbol]["buy_amount"] = allocation
                buy_rows += 1

        generated_rows.extend(per_symbol_rows.values())

        if desired_symbol is None:
            held_symbol = None
            entry_price = 0.0
            hold_bars = 0
        elif held_symbol == desired_symbol:
            hold_bars += 1
        else:
            held_symbol = desired_symbol
            entry_price = prices[desired_symbol]
            hold_bars = 1

    actions = pd.DataFrame(generated_rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    stats = {
        "history_bars": int(history_bars),
        "aligned_timestamps": int(total_steps),
        "action_timestamps": int(actions["timestamp"].nunique()),
        "buy_rows": int(buy_rows),
        "sell_rows": int(sell_rows),
        "short_signals_flattened": int(short_signals_flattened),
    }
    return actions, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--fee-tier", choices=list(FEE_TIERS.keys()), default="fdusd")
    parser.add_argument("--timeframe", choices=["hourly", "daily"], default="hourly")
    parser.add_argument("--start-date", default="2025-06-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--max-hold-hours", type=int, default=None)
    parser.add_argument("--hidden-size", type=float, default=1024, help="Deprecated compatibility flag; ignored.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Mask short actions during inference before simulation.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the final JSON summary without progress lines.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the final JSON summary.",
    )
    return parser.parse_args(argv)


def _emit_validation_summary(
    summary: dict[str, object],
    *,
    output_json: str | None,
) -> None:
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if output_json:
        Path(output_json).write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    fee = FEE_TIERS[args.fee_tier]

    if not args.json_only:
        print(f"Validating {args.checkpoint}")
        print(f"  Symbols: {symbols}")
        print(f"  Fee tier: {args.fee_tier} ({fee*100:.2f}%)")
        print(f"  Timeframe: {args.timeframe}")
        print(f"  Period: {args.start_date} to {args.end_date or 'latest'}")

    all_bars = []
    for sym in symbols:
        if args.timeframe == "hourly":
            df = load_hourly_bars(sym)
        else:
            df = load_daily_bars(sym)
        all_bars.append(df)
    bars = pd.concat(all_bars, ignore_index=True)
    bars = bars[bars["timestamp"] >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        bars = bars[bars["timestamp"] <= pd.Timestamp(args.end_date, tz="UTC")]
    if bars.empty:
        raise ValueError("No bars remain after applying the requested date range")

    if not args.json_only:
        print(f"  Loaded {len(bars)} bars ({bars['timestamp'].min()} to {bars['timestamp'].max()})")

    actions, action_stats = _generate_policy_actions(
        bars=bars,
        checkpoint=args.checkpoint,
        symbols=symbols,
        timeframe=args.timeframe,
        device=str(args.device),
        long_only=bool(args.long_only),
    )
    if not args.json_only:
        print(
            "  Generated actions: "
            f"{action_stats['action_timestamps']} timestamps, "
            f"{action_stats['buy_rows']} buys, "
            f"{action_stats['sell_rows']} sells, "
            f"{action_stats['short_signals_flattened']} short signals flattened"
        )

    simulation_config_cls, run_shared_cash_simulation = _load_marketsimulator_api()
    config = simulation_config_cls(
        maker_fee=fee,
        initial_cash=args.initial_cash,
        max_hold_hours=args.max_hold_hours,
    )
    result = run_shared_cash_simulation(bars, actions, config)

    summary = {
        "checkpoint": str(args.checkpoint),
        "symbols": symbols,
        "timeframe": str(args.timeframe),
        "fee_tier": str(args.fee_tier),
        "fee_rate": float(fee),
        "long_only": bool(args.long_only),
        "period": {
            "start": str(args.start_date),
            "end": str(args.end_date) if args.end_date is not None else None,
        },
        "action_generation": action_stats,
        "metrics": {
            "total_return": float(result.metrics.get("total_return", 0.0)),
            "sortino": float(result.metrics.get("sortino", 0.0)),
            "trades": int(sum(len(sr.trades) for sr in result.per_symbol.values())),
        },
    }

    n_periods = len(result.combined_equity)
    if args.timeframe == "hourly":
        years = n_periods / 8760
    else:
        years = n_periods / 365
    total_ret = summary["metrics"]["total_return"]
    if years > 0 and total_ret > -1:
        annualized = (1 + total_ret) ** (1 / years) - 1
        summary["metrics"]["annualized_return"] = float(annualized)
        summary["period_years"] = float(years)

    _emit_validation_summary(summary, output_json=args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
