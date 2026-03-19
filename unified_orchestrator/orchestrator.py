"""Unified Stock+Crypto Trading Orchestrator.

24/7 hourly loop that coordinates Alpaca (stocks and crypto).
Runs the right system at the right time with cross-asset awareness.

Usage:
  python -m unified_orchestrator.orchestrator --dry-run --once
  python -m unified_orchestrator.orchestrator --live
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-trading-agent-binance"))

from loguru import logger

from unified_orchestrator.state import (
    build_snapshot,
    save_snapshot,
    PendingOrder,
    Position,
    UnifiedPortfolioSnapshot,
)
from unified_orchestrator.prompt_builder import build_unified_prompt
from unified_orchestrator.alpaca_watcher import AlpacaCryptoWatcher, OrderPair
from unified_orchestrator.position_tracker import (
    load_entry_times,
    save_entry_times,
    update_entry_times,
    get_force_exit_symbols,
    load_peak_prices,
    save_peak_prices,
    update_peak_prices,
    get_trailing_stop_symbols,
)
from unified_orchestrator.backout import select_backout_candidates, execute_backout
from unified_orchestrator.conditional_orders import (
    execute_plan,
    read_pending_fills,
    TradingPlan,
    TradingStep,
)

from llm_hourly_trader.providers import call_llm
from llm_hourly_trader.gemini_wrapper import TradePlan

from unified_orchestrator.rl_gemini_bridge import (
    RLGeminiBridge,
    RLSignal,
    build_portfolio_observation,
)


# ---------------------------------------------------------------------------
# Crypto signal generation
# ---------------------------------------------------------------------------

# RL+Gemini hybrid bridges (lazy-loaded, separate for stocks vs crypto)
_rl_bridge: RLGeminiBridge | None = None
_rl_bridge_stock: RLGeminiBridge | None = None

# Validated checkpoints (confirmed on held-out data):
#   stocks: +50% median / 30-day window, Sortino=25 (featlag1, fee=5bps, long-only)
#   crypto: mass_daily/tp0.15_s314 = Sortino 4.18, +73.2% on 120d market sim (daily RL)
#           Confirmed best crypto5 model (217 ckpts evaluated 2026-03-19)
#           Runners-up: tp0.10_s42 (S=4.01), tp0.20_s123 (S=3.86, best 180d)
#           Old 300M models overfit badly (-3% to -16% OOS) — do NOT use
STOCK_CHECKPOINT_CANDIDATES = [
    REPO / "pufferlib_market/checkpoints/stocks13_featlag1_fee5bps_longonly_run4/best.pt",
    REPO / "pufferlib_market/checkpoints/stocks13_issuedat_featlag1_fee5bps_longonly_run5/best.pt",
]
CRYPTO_CHECKPOINT_CANDIDATES = [
    # Comprehensive eval 2026-03-19: 217 checkpoints, 5 periods (30-180d)
    # Best crypto5-compatible (obs_size=90) by 120d Sortino:
    #   #1 tp0.15_s314:  Sortino 4.18, +73.2% 120d, MaxDD 25.6%, WR 67%
    #   #2 tp0.10_s42:   Sortino 4.01, +62.3% 120d, MaxDD 21.3%, WR 57%
    #   #3 tp0.20_s123:  Sortino 3.86, +63.2% 120d, MaxDD 22.2%, WR 64%, best 180d (+81.1%, S=3.04)
    #   #4 tp0.05_s123:  Sortino 3.78, +65.0% 120d, MaxDD 20.8%, WR 59%
    REPO / "pufferlib_market/checkpoints/mass_daily/tp0.15_s314/best.pt",
    REPO / "pufferlib_market/checkpoints/mass_daily/tp0.10_s42/best.pt",
    REPO / "pufferlib_market/checkpoints/mass_daily/tp0.20_s123/best.pt",
    REPO / "pufferlib_market/checkpoints/mass_daily/tp0.05_s123/best.pt",
    REPO / "pufferlib_market/checkpoints/autoresearch/longonly_forecast/best.pt",
    REPO / "pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt",
    REPO / "pufferlib_market/checkpoints/autoresearch/ent_01/best.pt",
    REPO / "pufferlib_market/checkpoints/autoresearch/reg_combo_2/best.pt",
]
CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
MIN_ACTIONABLE_CRYPTO_NOTIONAL = 1.0

# Max hours to hold a crypto position before forcing exit at near-market price.
# Confirmed optimal by hold-sweep: Sortino=20.69 at 6h vs 11.66 at 4h, 6.16 at 12h.
MAX_HOLD_HOURS = 6

# Minimum LLM confidence to act on a buy signal.
# Exp02: 0.7 optimal on Mar 2-9 window, but HURT on Mar 7-14 (-40.7% vs -2.2%).
# Exp07: Not robust across windows. Reverted to 0.4. Trailing stop is the real winner.
MIN_CONFIDENCE_CRYPTO = 0.4

# Trailing stop: exit if price drops this % from peak since entry.
# Exp06 result: 0.3% trail → Sortino=76.5, MaxDD=4.7% (vs 30.9/28.3% baseline).
TRAILING_STOP_PCT = 0.3

# Leverage settings for stocks
MAX_INTRADAY_LEVERAGE = 4.0   # Alpaca PDT 4x intraday
MAX_OVERNIGHT_LEVERAGE = 2.0  # Must deleverage to 2x before market close
MARGIN_INTEREST_ANNUAL = 0.0625  # 6.25% annual on leveraged portion
DELEVERAGE_MINUTES_BEFORE_CLOSE = 60  # Start deleveraging 1h before close
_CHECKPOINT_DATA_HINTS = {
    "stocks13_featlag1_fee5bps_longonly_run4": REPO / "pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_start20250915_featlag1.bin",
    "stocks13_issuedat_featlag1_fee5bps_longonly_run5": REPO / "pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_start20250915_issuedat_featlag1.bin",
    "tp0.15_s314": REPO / "pufferlib_market/data/crypto5_daily_train.bin",
    "tp0.10_s42": REPO / "pufferlib_market/data/crypto5_daily_train.bin",
    "tp0.20_s123": REPO / "pufferlib_market/data/crypto5_daily_train.bin",
    "tp0.05_s123": REPO / "pufferlib_market/data/crypto5_daily_train.bin",
    "longonly_forecast": REPO / "pufferlib_market/data/crypto6_train.bin",
    "slip_5bps": REPO / "pufferlib_market/data/crypto6_train.bin",
    "ent_01": REPO / "pufferlib_market/data/crypto6_train.bin",
    "reg_combo_2": REPO / "pufferlib_market/data/crypto6_train.bin",
}
STOCK_FORECAST_CACHE_CANDIDATES = [
    REPO / "binanceneural/forecast_cache",
    REPO / "unified_hourly_experiment/forecast_cache",
    REPO / "alpacanewccrosslearning/forecast_cache/mega24_plus_yelp_novol_baseline_20260207_lb2400",
]
CRYPTO_FORECAST_CACHE_CANDIDATES = [
    REPO / "alpacanewccrosslearning/forecast_cache/crypto13_novol_20260208_lb4000",
    REPO / "binanceneural/forecast_cache",
    REPO / "alpacanewccrosslearning/forecast_cache",
]


def get_rl_bridge(checkpoint_path: str = "", hidden_size: int = 1024) -> RLGeminiBridge | None:
    """Get or create the crypto RL+Gemini bridge (lazy singleton)."""
    global _rl_bridge
    if _rl_bridge is not None:
        return _rl_bridge
    if not checkpoint_path:
        for c in CRYPTO_CHECKPOINT_CANDIDATES:
            if c.exists():
                checkpoint_path = str(c)
                break
    if not checkpoint_path:
        return None
    _rl_bridge = RLGeminiBridge(checkpoint_path=checkpoint_path, hidden_size=hidden_size)
    logger.info(f"  Crypto RL bridge loaded: {checkpoint_path}")
    return _rl_bridge


def get_rl_bridge_stock(checkpoint_path: str = "", hidden_size: int = 1024) -> RLGeminiBridge | None:
    """Get or create the stock RL+Gemini bridge (lazy singleton)."""
    global _rl_bridge_stock
    if _rl_bridge_stock is not None:
        return _rl_bridge_stock
    if not checkpoint_path:
        for c in STOCK_CHECKPOINT_CANDIDATES:
            if c.exists():
                checkpoint_path = str(c)
                break
    if not checkpoint_path:
        logger.warning("  No stock RL checkpoint found — falling back to Gemini-only signals")
        return None
    _rl_bridge_stock = RLGeminiBridge(checkpoint_path=checkpoint_path, hidden_size=hidden_size)
    logger.info(f"  Stock RL bridge loaded: {checkpoint_path}")
    return _rl_bridge_stock

CRYPTO_PAIRS = {"BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT",
                "LTCUSD": "LTCUSDT", "AVAXUSD": "AVAXUSDT",
                "DOGEUSD": "DOGEUSDT", "SUIUSD": "SUIUSDT", "AAVEUSD": "AAVEUSDT"}

STOCK_SYMBOLS = ["NVDA", "PLTR", "META", "MSFT", "NET"]


def _has_actionable_crypto_position(pos: Position | None) -> bool:
    """Return True when a crypto position is large enough to manage live."""

    if pos is None:
        return False
    qty = float(getattr(pos, "qty", 0.0) or 0.0)
    if qty <= 0.0:
        return False
    try:
        market_value = float(pos.market_value)
    except Exception:
        current = float(getattr(pos, "current_price", 0.0) or 0.0)
        avg = float(getattr(pos, "avg_price", 0.0) or 0.0)
        market_value = qty * (current if current > 0.0 else avg)
    return market_value >= MIN_ACTIONABLE_CRYPTO_NOTIONAL


def _with_fallback_crypto_exit_target(
    sym: str,
    plan: TradePlan,
    held_position: Position | None,
    current_price: float,
    fee_bps: float = 16.0,
) -> TradePlan:
    """Ensure live held crypto positions always retain a take-profit target."""

    if not _has_actionable_crypto_position(held_position):
        return plan

    sell_price = float(getattr(plan, "sell_price", 0.0) or 0.0)
    if sell_price > float(current_price):
        return plan

    fee_floor = 1.0 + max(float(fee_bps), 0.0) / 10_000.0 + 0.0005
    avg_price = float(getattr(held_position, "avg_price", 0.0) or 0.0)
    current_floor = float(current_price) * 1.008
    target_floor = max(current_floor, avg_price * fee_floor if avg_price > 0.0 else 0.0)
    fallback_sell = max(float(current_price) * 1.001, target_floor)
    logger.warning(
        f"  {sym}: held crypto position missing actionable exit target "
        f"(sell=${sell_price:.2f}) — using fallback sell=${fallback_sell:.2f}"
    )
    return TradePlan(
        direction=plan.direction,
        buy_price=float(getattr(plan, "buy_price", 0.0) or 0.0),
        sell_price=float(fallback_sell),
        confidence=max(float(getattr(plan, "confidence", 0.0) or 0.0), 0.05),
        reasoning=(
            f"{getattr(plan, 'reasoning', '')} | fallback_exit_target"
            if getattr(plan, "reasoning", "")
            else "fallback_exit_target"
        ),
        allocation_pct=float(getattr(plan, "allocation_pct", 0.0) or 0.0),
    )


def _get_crypto_rl_trader():
    """Lazy-load PPOTrader for crypto RL signals."""
    global _crypto_rl_trader
    if "_crypto_rl_trader" not in globals():
        _crypto_rl_trader = None
    if _crypto_rl_trader is not None:
        return _crypto_rl_trader
    for c in CRYPTO_CHECKPOINT_CANDIDATES:
        if c.exists():
            try:
                from pufferlib_market.inference import PPOTrader
                bridge = RLGeminiBridge(checkpoint_path=str(c))
                trained_symbols = _read_trained_symbols_for_checkpoint(
                    c,
                    _num_symbols_from_obs_size(bridge.get_checkpoint_spec().obs_size),
                )
                _crypto_rl_trader = PPOTrader(
                    str(c), device="cpu", symbols=trained_symbols,
                )
                logger.info(f"  Crypto RL trader loaded: {c}")
                return _crypto_rl_trader
            except Exception as e:
                logger.warning(f"  Failed to load RL trader from {c}: {e}")
    return None


_crypto_rl_trader = None


def _checkpoint_data_path(checkpoint_path: Path) -> Path | None:
    path = _CHECKPOINT_DATA_HINTS.get(checkpoint_path.parent.name)
    if path is not None and path.exists():
        return path
    return None


def _read_mktd_symbols(data_path: Path) -> list[str]:
    with data_path.open("rb") as handle:
        header = handle.read(64)
        if len(header) < 24:
            raise ValueError(f"Incomplete MKTD header in {data_path}")
        magic, _version, num_symbols, _num_timesteps, _feat_count, _price_count = struct.unpack(
            "<4sIIIII",
            header[:24],
        )
        if magic != b"MKTD":
            raise ValueError(f"Unexpected MKTD magic {magic!r} in {data_path}")
        symbols = []
        for _ in range(int(num_symbols)):
            raw = handle.read(16)
            symbols.append(raw.split(b"\x00", 1)[0].decode("ascii"))
    return symbols


def _num_symbols_from_obs_size(obs_size: int) -> int:
    remainder = int(obs_size) - 5
    if remainder <= 0 or remainder % 17 != 0:
        raise ValueError(f"Cannot infer symbol count from obs_size={obs_size}")
    return remainder // 17


def _read_trained_symbols_for_checkpoint(checkpoint_path: Path, expected_num_symbols: int) -> list[str]:
    data_path = _checkpoint_data_path(checkpoint_path)
    if data_path is None:
        raise FileNotFoundError(f"No market-data hint configured for checkpoint {checkpoint_path}")
    symbols = _read_mktd_symbols(data_path)
    if len(symbols) != int(expected_num_symbols):
        raise ValueError(
            f"Checkpoint {checkpoint_path} expects {expected_num_symbols} symbols but {data_path} lists {len(symbols)}"
        )
    return symbols


def _choose_forecast_cache_root(symbols: list[str], candidates: list[Path]) -> Path | None:
    best_root: Path | None = None
    best_coverage = -1
    for root in candidates:
        if not root.exists():
            continue
        coverage = sum(
            1
            for sym in symbols
            if (root / "h1" / f"{sym}.parquet").exists() and (root / "h24" / f"{sym}.parquet").exists()
        )
        if coverage > best_coverage:
            best_root = root
            best_coverage = coverage
            if coverage == len(symbols):
                break
    return best_root


def _select_symbol_frame(raw_df, symbol_key: str):
    import pandas as pd

    if raw_df is None or len(raw_df) == 0:
        return None

    frame = raw_df
    if isinstance(frame.index, pd.MultiIndex):
        level_names = [name or "" for name in frame.index.names]
        if "symbol" in level_names:
            available = frame.index.get_level_values("symbol")
            if symbol_key not in available:
                return None
            frame = frame.xs(symbol_key, level="symbol")
    elif "symbol" in frame.columns:
        frame = frame[frame["symbol"] == symbol_key]
        if frame.empty:
            return None

    frame = frame.reset_index()
    if "timestamp" not in frame.columns:
        if "index" in frame.columns:
            frame = frame.rename(columns={"index": "timestamp"})
        else:
            for column in frame.columns:
                if "timestamp" in str(column).lower():
                    frame = frame.rename(columns={column: "timestamp"})
                    break
    if "timestamp" not in frame.columns:
        return None

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame if not frame.empty else None


def _history_from_frame(frame) -> list[dict]:
    history = []
    for _, row in frame.iterrows():
        history.append(
            {
                "timestamp": str(row.get("timestamp", ""))[:16],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            }
        )
    return history


def _forecast_at(forecast_df, timestamp):
    if forecast_df is None or getattr(forecast_df, "empty", True):
        return None
    cutoff = forecast_df[forecast_df.index <= timestamp]
    if cutoff.empty:
        return None
    row = cutoff.iloc[-1]
    return {
        "predicted_close_p50": float(row.get("predicted_close_p50", 0.0)),
        "predicted_close_p10": float(row.get("predicted_close_p10", 0.0)),
        "predicted_close_p90": float(row.get("predicted_close_p90", 0.0)),
        "predicted_high_p50": float(row.get("predicted_high_p50", 0.0)),
        "predicted_low_p50": float(row.get("predicted_low_p50", 0.0)),
    }


def _load_forecast_frames(symbols: list[str], cache_root: Path | None):
    if cache_root is None:
        return {}, {}

    from pufferlib_market.export_data_hourly_forecast import _read_forecast

    forecasts_1h = {}
    forecasts_24h = {}
    for sym in symbols:
        try:
            forecasts_1h[sym] = _read_forecast(sym, cache_root, 1)
        except FileNotFoundError:
            continue
        try:
            forecasts_24h[sym] = _read_forecast(sym, cache_root, 24)
        except FileNotFoundError:
            continue
    return forecasts_1h, forecasts_24h


def _decode_bridge_signal_map(
    bridge: RLGeminiBridge,
    features,
    symbol_names: list[str],
) -> dict[str, RLSignal]:
    obs = build_portfolio_observation(features)
    signals = bridge.get_rl_signals(
        obs,
        num_symbols=len(symbol_names),
        symbol_names=symbol_names,
        top_k=len(symbol_names),
    )
    return {signal.symbol_name: signal for signal in signals}


def _build_crypto_rl_signal_map(history_frames: dict[str, object], bridge: RLGeminiBridge) -> dict[str, RLSignal]:
    import numpy as np
    from pufferlib_market.inference import compute_hourly_features

    spec = bridge.get_checkpoint_spec()
    trained_symbols = _read_trained_symbols_for_checkpoint(
        bridge.checkpoint_path,
        _num_symbols_from_obs_size(spec.obs_size),
    )

    features = np.zeros((len(trained_symbols), 16), dtype=np.float32)
    valid_symbols = []
    for idx, sym in enumerate(trained_symbols):
        frame = history_frames.get(sym)
        if frame is None or len(frame) < 72:
            logger.warning(f"  Crypto RL skipped: {sym} only has {0 if frame is None else len(frame)} bars (need 72); using zeros")
            continue
        features[idx] = compute_hourly_features(frame)
        valid_symbols.append(sym)
    if not valid_symbols:
        logger.warning("  Crypto RL: no symbols had enough data, returning empty signals")
        return {}
    return _decode_bridge_signal_map(bridge, features, trained_symbols)


def _build_stock_rl_signal_map(
    history_frames: dict[str, object],
    bridge: RLGeminiBridge,
    forecasts_1h: dict[str, object],
    forecasts_24h: dict[str, object],
) -> dict[str, RLSignal]:
    import numpy as np
    import pandas as pd
    from pufferlib_market.export_data_hourly_forecast import compute_features

    spec = bridge.get_checkpoint_spec()
    trained_symbols = _read_trained_symbols_for_checkpoint(
        bridge.checkpoint_path,
        _num_symbols_from_obs_size(spec.obs_size),
    )

    features = np.zeros((len(trained_symbols), 16), dtype=np.float32)
    valid_symbols = []
    for idx, sym in enumerate(trained_symbols):
        frame = history_frames.get(sym)
        if frame is None or len(frame) < 72:
            logger.warning(f"  Stock RL skipped: {sym} only has {0 if frame is None else len(frame)} bars (need 72); using zeros")
            continue
        price_df = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True, errors="coerce").dt.floor("h")
        price_df = price_df.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last").set_index("timestamp")
        feature_df = compute_features(price_df, forecasts_1h.get(sym, pd.DataFrame()), forecasts_24h.get(sym, pd.DataFrame()))
        features[idx] = feature_df.iloc[-1].to_numpy(dtype=np.float32, copy=False)
        valid_symbols.append(sym)
    if not valid_symbols:
        logger.warning("  Stock RL: no symbols had enough data, returning empty signals")
        return {}
    return _decode_bridge_signal_map(bridge, features, trained_symbols)


def _format_rl_hint(signal: RLSignal | None, source: str) -> str:
    if signal is None or signal.direction == "flat":
        return ""
    return (
        f"\nRL MODEL SIGNAL: direction={signal.direction} confidence={signal.confidence:.2f} "
        f"allocation={signal.allocation_pct:.0%} ({source})"
    )


def _fetch_crypto_history_frames(data_client, symbols: list[str], now, *, lookback_hours: int = 78):
    """Fetch hourly bars for each crypto symbol individually.

    Alpaca's batch CryptoBarsRequest is unreliable (only returns one symbol's
    data when multiple are requested).  Individual requests are slower but
    consistent.
    """
    from datetime import timedelta

    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    frames = {}
    for sym in symbols:
        request_sym = sym[:-3] + "/USD" if "/" not in sym and sym.endswith("USD") else sym
        try:
            req = CryptoBarsRequest(
                symbol_or_symbols=request_sym,
                timeframe=TimeFrame.Hour,
                start=now - timedelta(hours=lookback_hours),
                end=now,
                limit=72,
            )
            bars = data_client.get_crypto_bars(req)
            raw_df = getattr(bars, "df", None)
            frame = _select_symbol_frame(raw_df, request_sym)
            if frame is not None:
                frames[sym] = frame
        except Exception as e:
            logger.debug(f"  {sym}: bars fetch error: {e}")
    return frames


def _fetch_stock_history_frames(data_client, symbols: list[str], now, *, lookback_hours: int = 78):
    from datetime import timedelta

    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    # Fetch per-symbol individually — batch requests return unreliable multi-index
    # structures (only one symbol's data often appears).
    frames = {}
    for sym in symbols:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Hour,
                start=now - timedelta(hours=lookback_hours),
                end=now,
                limit=72,
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)
            raw_df = getattr(bars, "df", None)
            frame = _select_symbol_frame(raw_df, sym)
            if frame is not None:
                frames[sym] = frame
        except Exception as e:
            logger.debug(f"  {sym}: stock bars fetch error: {e}")
    return frames


def get_crypto_signals(
    symbols: list[str],
    snapshot: UnifiedPortfolioSnapshot,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    review_thinking_level: str | None = None,
    reprompt_passes: int = 1,
    reprompt_policy: str = "always",
    review_max_confidence: float | None = None,
    review_model: str | None = None,
    dry_run: bool = True,
) -> dict[str, TradePlan]:
    """Generate LLM trading signals for crypto symbols using Alpaca historical data.

    If a validated RL checkpoint is available, computes RL model signals and
    injects them as hints into the LLM prompt.
    """
    from alpaca.data.historical import CryptoHistoricalDataClient
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

    signals = {}
    data_client = CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    now = datetime.now(timezone.utc)

    rl_bridge = get_rl_bridge()
    rl_symbol_universe: list[str] = []
    if rl_bridge is not None:
        try:
            rl_symbol_universe = _read_trained_symbols_for_checkpoint(
                rl_bridge.checkpoint_path,
                _num_symbols_from_obs_size(rl_bridge.get_checkpoint_spec().obs_size),
            )
        except Exception as e:
            logger.warning(f"  Crypto RL disabled: {e}")
            rl_bridge = None

    fetch_symbols = list(dict.fromkeys([*symbols, *rl_symbol_universe]))
    history_frames = _fetch_crypto_history_frames(data_client, fetch_symbols, now)

    rl_signal_map: dict[str, RLSignal] = {}
    if rl_bridge is not None:
        try:
            rl_signal_map = _build_crypto_rl_signal_map(history_frames, rl_bridge)
        except Exception as e:
            logger.warning(f"  Crypto RL signal generation failed: {e}")

    # SMA-24 trend filter: suppress LONG hints when price is below 24h SMA.
    # The longonly_forecast model cannot short, so its LONG hints in downtrends
    # push the LLM into losing positions.  Zeroing them out lets the LLM use
    # its own bearish judgment instead.
    for sym, sig in list(rl_signal_map.items()):
        if sig.direction != "long":
            continue
        frame = history_frames.get(sym)
        if frame is None or len(frame) < 24:
            continue
        sma24 = float(frame["close"].iloc[-24:].mean())
        current = float(frame["close"].iloc[-1])
        if current < sma24:
            logger.info(f"  {sym}: SMA-24 filter suppressed RL LONG hint (price={current:.4f} < sma24={sma24:.4f})")
            rl_signal_map[sym] = RLSignal(
                symbol_idx=sig.symbol_idx,
                symbol_name=sig.symbol_name,
                direction="flat",
                confidence=0.0,
                logit_gap=0.0,
                allocation_pct=0.0,
            )

    forecast_root = _choose_forecast_cache_root(fetch_symbols, CRYPTO_FORECAST_CACHE_CANDIDATES)
    forecast_frames_1h, forecast_frames_24h = _load_forecast_frames(fetch_symbols, forecast_root)

    for sym in symbols:
        frame = history_frames.get(sym)
        if frame is None or len(frame) < 12:
            logger.warning(f"  {sym}: insufficient bars ({0 if frame is None else len(frame)})")
            continue

        history = _history_from_frame(frame)
        if len(history) < 12:
            logger.warning(f"  {sym}: insufficient normalized history ({len(history)})")
            continue

        current_price = history[-1]["close"]
        timestamp = frame["timestamp"].iloc[-1]
        forecast_1h = _forecast_at(forecast_frames_1h.get(sym), timestamp)
        forecast_24h = _forecast_at(forecast_frames_24h.get(sym), timestamp)
        rl_hint = _format_rl_hint(
            rl_signal_map.get(sym),
            "autoresearch/slip_5bps, +5.2%/30d OOS validated",
        )

        # Inject bearish context into prompt when price is below 24h SMA
        trend_warning = ""
        if frame is not None and len(frame) >= 24:
            _sma24 = float(frame["close"].iloc[-24:].mean())
            if current_price < _sma24:
                trend_warning = (
                    f"\nTREND CAUTION: {sym} price (${current_price:.4f}) is BELOW its 24h SMA "
                    f"(${_sma24:.4f}). Market is in a SHORT-TERM DOWNTREND. "
                    f"Strongly prefer HOLD/FLAT unless there is very clear reversal evidence."
                )

        held_pos = snapshot.alpaca_positions.get(sym)
        actionable_held_pos = held_pos if _has_actionable_crypto_position(held_pos) else None
        prompt = build_unified_prompt(
            symbol=sym,
            history_rows=history,
            current_price=current_price,
            snapshot=snapshot,
            asset_class="crypto",
            forecast_1h=forecast_1h,
            forecast_24h=forecast_24h,
            held_position=actionable_held_pos,
        ) + rl_hint + trend_warning

        try:
            plan = call_llm(
                prompt,
                model=model,
                thinking_level=thinking_level,
                review_thinking_level=review_thinking_level,
                reprompt_passes=reprompt_passes,
                reprompt_policy=reprompt_policy,
                review_max_confidence=review_max_confidence,
                review_model=review_model,
            )
            plan = _with_fallback_crypto_exit_target(sym, plan, actionable_held_pos, current_price)
            signals[sym] = plan
            alloc_str = f", alloc={plan.allocation_pct:.0f}%" if plan.allocation_pct > 0 else ""
            logger.info(
                f"  {sym}: {plan.direction} (conf={plan.confidence:.2f}, "
                f"buy=${plan.buy_price:.2f}, sell=${plan.sell_price:.2f}{alloc_str})"
            )
        except Exception as e:
            logger.error(f"  {sym}: LLM error: {e}")

    # Hard SMA-24 guard: override LONG → HOLD for any symbol where price < SMA-24.
    # This fires even when the LLM ignores the TREND CAUTION prompt message.
    from llm_hourly_trader.gemini_wrapper import TradePlan as _TradePlan
    for sym, plan in list(signals.items()):
        if plan.direction != "long":
            continue
        frame = history_frames.get(sym)
        if frame is None or len(frame) < 24:
            continue
        _sma24g = float(frame["close"].iloc[-24:].mean())
        _price_g = float(frame["close"].iloc[-1])
        if _price_g < _sma24g:
            logger.info(
                f"  {sym}: SMA-24 HARD BLOCK overrides LONG→HOLD "
                f"(price={_price_g:.4f} < sma24={_sma24g:.4f})"
            )
            signals[sym] = _TradePlan(
                direction="hold",
                buy_price=0.0,
                sell_price=plan.sell_price,
                confidence=plan.confidence,
                reasoning=f"SMA-24 trend block; {plan.reasoning[:60]}",
                allocation_pct=0,
            )

    return signals


# ---------------------------------------------------------------------------
# Crypto execution
# ---------------------------------------------------------------------------

def validate_plan_safety(plan: TradePlan, current_price: float,
                         fee_bps: float = 20.0) -> tuple[bool, str]:
    """Validate a trade plan is safe to execute in production.

    Checks:
    1. sell_price > buy_price + fees (must be profitable if both fill)
    2. Prices within reasonable range of current (not >5% away)
    3. Confidence > 0 for non-hold plans
    """
    if plan.direction == "hold":
        return True, "hold"

    if plan.buy_price <= 0 and plan.sell_price <= 0:
        return False, "no prices set"

    # For longs: sell must be above buy + fees
    if plan.direction == "long" and plan.buy_price > 0 and plan.sell_price > 0:
        fee_cost = plan.buy_price * fee_bps / 10000 * 2  # round-trip
        if plan.sell_price <= plan.buy_price + fee_cost:
            return False, (f"sell ${plan.sell_price:.2f} <= buy ${plan.buy_price:.2f} "
                          f"+ fees ${fee_cost:.2f}")

    # Prices should be within 5% of current
    if plan.buy_price > 0:
        pct_diff = abs(plan.buy_price - current_price) / current_price
        if pct_diff > 0.05:
            return False, f"buy_price ${plan.buy_price:.2f} is {pct_diff:.1%} from current ${current_price:.2f}"

    if plan.sell_price > 0:
        pct_diff = abs(plan.sell_price - current_price) / current_price
        if pct_diff > 0.05:
            return False, f"sell_price ${plan.sell_price:.2f} is {pct_diff:.1%} from current ${current_price:.2f}"

    return True, "ok"


def _place_crypto_tp_sell(alpaca, sym: str, pos, sell_price: float,
                          dry_run: bool, orders: list[dict]) -> None:
    """Place a take-profit sell order for a crypto position.

    Uses a slightly reduced qty (floor to 8 decimals minus epsilon) to avoid
    Alpaca's 'insufficient balance' error from floating-point qty mismatch.
    """
    import math
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    # Floor qty to 8 decimals minus a tiny buffer to avoid exceeding available
    sell_qty = math.floor(pos.qty * 1e8 - 1) / 1e8
    if sell_qty <= 0:
        return

    # Cancel any existing sell orders for this symbol first
    import time as _time
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    canceled_any = False
    try:
        open_orders = alpaca.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        for order in open_orders:
            sym_norm = str(order.symbol).replace("/", "")
            if sym_norm == sym and order.side.value.lower() == "sell":
                if not dry_run:
                    alpaca.cancel_order_by_id(str(order.id))
                    canceled_any = True
                logger.info(f"  {sym}: canceled old TP sell {order.id} @ {order.limit_price}")
    except Exception as e:
        logger.debug(f"  {sym}: error canceling old sells: {e}")

    # Brief pause after cancel to let Alpaca release held balance
    if canceled_any:
        _time.sleep(0.5)

    logger.info(f"  {sym}: placing TP sell {sell_qty:.8f} @ ${sell_price:.2f}")
    if not dry_run:
        try:
            req = LimitOrderRequest(
                symbol=sym,
                qty=round(sell_qty, 8),
                side=OrderSide.SELL,
                type="limit",
                time_in_force=TimeInForce.GTC,
                limit_price=round(sell_price, 2),
            )
            result = alpaca.submit_order(req)
            orders.append({"symbol": sym, "action": "sell_tp", "price": sell_price,
                           "qty": sell_qty, "order_id": str(result.id)})
        except Exception as e:
            logger.error(f"  {sym}: TP sell error: {e}")
    else:
        orders.append({"symbol": sym, "action": "sell_tp", "price": sell_price,
                       "qty": sell_qty, "dry_run": True})


def _place_crypto_force_exit_sell(alpaca, sym: str, pos, dry_run: bool,
                                   orders: list[dict]) -> None:
    """Force-exit a position that has exceeded MAX_HOLD_HOURS.

    Places an aggressive limit sell 0.1% below current price so it fills
    quickly — mirrors the backtest simulator's 'exit at close' after max_hold.
    Cancels any open orders for this symbol first.
    """
    import math
    from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

    sell_qty = math.floor(pos.qty * 1e8 - 1) / 1e8
    if sell_qty <= 0:
        return

    # Cancel all open orders for this symbol (buy AND sell)
    try:
        open_orders = alpaca.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        for order in open_orders:
            sym_norm = str(order.symbol).replace("/", "")
            if sym_norm == sym:
                if not dry_run:
                    alpaca.cancel_order_by_id(str(order.id))
                logger.info(f"  {sym}: [FORCE EXIT] canceled order {order.id}")
    except Exception as e:
        logger.debug(f"  {sym}: error canceling orders for force exit: {e}")

    # Sell at 0.1% below current — fills quickly on liquid crypto
    exit_price = round(pos.current_price * 0.999, 2)
    logger.info(f"  {sym}: FORCE EXIT {sell_qty:.8f} @ ${exit_price:.2f} "
                f"(held > {MAX_HOLD_HOURS}h, current ${pos.current_price:.2f})")
    if not dry_run:
        try:
            req = LimitOrderRequest(
                symbol=sym,
                qty=round(sell_qty, 8),
                side=OrderSide.SELL,
                type="limit",
                time_in_force=TimeInForce.GTC,
                limit_price=exit_price,
            )
            result = alpaca.submit_order(req)
            orders.append({"symbol": sym, "action": "force_exit", "price": exit_price,
                           "qty": sell_qty, "order_id": str(result.id)})
        except Exception as e:
            logger.error(f"  {sym}: force exit order error: {e}")
    else:
        orders.append({"symbol": sym, "action": "force_exit", "price": exit_price,
                       "qty": sell_qty, "dry_run": True})


def execute_crypto_signals(
    signals: dict[str, TradePlan],
    snapshot: UnifiedPortfolioSnapshot,
    dry_run: bool = True,
    alpaca_client=None,
) -> list[dict]:
    """Execute crypto trading signals on Alpaca."""
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    orders = []
    alpaca = alpaca_client or TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)

    # Normalize Alpaca's "ETH/USD" symbol format back to "ETHUSD" for lookup
    def _norm(s: str) -> str:
        return s.replace("/", "")

    def _refresh_alpaca_state() -> None:
        """Refresh cash, positions, and open orders after any broker-side mutations."""
        account = alpaca.get_account()
        snapshot.alpaca_cash = float(account.cash)
        snapshot.alpaca_buying_power = float(account.buying_power)

        refreshed_positions: dict[str, Position] = {}
        for pos in alpaca.get_all_positions():
            symbol = _norm(str(pos.symbol))
            refreshed_positions[symbol] = Position(
                symbol=symbol,
                qty=float(pos.qty),
                avg_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                unrealized_pnl=float(pos.unrealized_pl),
                broker="alpaca",
            )
        snapshot.alpaca_positions = refreshed_positions

        refreshed_orders: list[PendingOrder] = []
        refreshed_open_orders = alpaca.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        for order in refreshed_open_orders:
            refreshed_orders.append(PendingOrder(
                symbol=_norm(str(order.symbol)),
                side=order.side.value.lower(),
                qty=float(order.qty),
                limit_price=float(order.limit_price) if order.limit_price else 0.0,
                broker="alpaca",
                order_id=str(order.id),
            ))
        snapshot.alpaca_pending_orders = refreshed_orders

    # Cancel stale pending buy orders for crypto symbols (GTC orders from last cycle
    # have stale prices — replace with fresh signals each hour)
    crypto_sym_set = {sym for sym in signals}
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    open_orders = alpaca.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    canceled_for: set[str] = set()
    for order in open_orders:
        sym_norm = _norm(str(order.symbol))
        if sym_norm in crypto_sym_set and order.side.value.lower() == "buy":
            if dry_run:
                logger.info(
                    f"  {sym_norm}: [DRY RUN] would cancel stale buy order {order.id} @ {order.limit_price}"
                )
                continue
            try:
                alpaca.cancel_order_by_id(str(order.id))
                canceled_for.add(sym_norm)
                logger.info(f"  {sym_norm}: canceled stale buy order {order.id} @ {order.limit_price}")
            except Exception as e:
                logger.warning(f"  {sym_norm}: failed to cancel stale order: {e}")

    # LLM calls and order cancellation can both leave the cycle snapshot stale.
    # Refresh from Alpaca immediately before sizing so we do not size off cash
    # reserved by canceled orders or positions that already filled out.
    _refresh_alpaca_state()

    # ── Position entry-time tracking ────────────────────────────────────────
    # Detect new fills since the last cycle, enforce MAX_HOLD_HOURS force exit.
    # This mirrors the backtest simulator's max_hold_hours close-at-market.
    # Use CRYPTO_SYMBOLS (not just signals keys) so tracking works even when
    # signals is empty due to a transient API failure.
    all_crypto_syms = set(CRYPTO_SYMBOLS)
    crypto_pos = {
        sym: pos
        for sym, pos in snapshot.alpaca_positions.items()
        if sym in all_crypto_syms and _has_actionable_crypto_position(pos)
    }
    entry_times = load_entry_times()
    entry_times = update_entry_times(entry_times, crypto_pos)
    force_exit_syms = set(get_force_exit_symbols(entry_times, MAX_HOLD_HOURS))
    save_entry_times(entry_times)

    # Force-exit positions that exceeded the max hold window
    for sym in force_exit_syms:
        pos = snapshot.alpaca_positions.get(sym)
        if pos and pos.qty > 0:
            _place_crypto_force_exit_sell(alpaca, sym, pos, dry_run, orders)

    # ── Trailing stop tracking ────────────────────────────────────────────
    # Track peak price since entry for each position. If current price drops
    # TRAILING_STOP_PCT% from peak, force exit. Exp06: 0.3% trail gives
    # Sortino=76.5, MaxDD=4.7% (vs 30.9/28.3% with LLM TP alone).
    peak_prices = load_peak_prices()
    peak_prices = update_peak_prices(peak_prices, crypto_pos)
    trailing_stop_syms = set(get_trailing_stop_symbols(
        peak_prices, crypto_pos, trail_pct=TRAILING_STOP_PCT,
    ))
    # Don't double-exit symbols already force-exiting
    trailing_stop_syms -= force_exit_syms
    save_peak_prices(peak_prices)

    for sym in trailing_stop_syms:
        pos = snapshot.alpaca_positions.get(sym)
        if pos and pos.qty > 0:
            _place_crypto_force_exit_sell(alpaca, sym, pos, dry_run, orders)

    # Place take-profit sells on all other existing crypto positions
    exit_syms = force_exit_syms | trailing_stop_syms
    for sym, plan in signals.items():
        if sym in exit_syms:
            continue  # already force-exiting or trailing-stop exiting
        pos = snapshot.alpaca_positions.get(sym)
        if pos and pos.qty > 0 and plan.sell_price > 0 and plan.sell_price > pos.current_price:
            _place_crypto_tp_sell(alpaca, sym, pos, plan.sell_price, dry_run, orders)

    # Sizing: use LLM's allocation_pct when available, else fall back to equal split
    num_long_signals = sum(
        1 for plan in signals.values()
        if plan.direction == "long" and plan.confidence >= MIN_CONFIDENCE_CRYPTO
    )
    fallback_pct = min(0.40, 1.0 / max(num_long_signals, 1))
    logger.info(f"  Sizing: {num_long_signals} long signals, fallback {fallback_pct*100:.0f}% per symbol")

    # Track cash reserved by orders placed in this cycle to avoid oversizing
    cash_reserved_this_cycle = 0.0

    for sym, plan in signals.items():
        try:
            if plan.direction != "long" or plan.confidence < MIN_CONFIDENCE_CRYPTO:
                continue

            # Don't buy into a position we're actively exiting
            if sym in exit_syms:
                logger.info(f"  {sym}: skipping buy — exit in progress")
                continue

            # Safety validation
            pos = snapshot.alpaca_positions.get(sym)
            current_price = pos.current_price if pos else plan.buy_price
            safe, reason = validate_plan_safety(plan, current_price, fee_bps=16.0)
            if not safe:
                logger.warning(f"  {sym}: SAFETY BLOCKED - {reason}")
                continue

            # Check existing position
            pos_value = pos.market_value if pos else 0.0
            # Use LLM allocation_pct if provided (1-100%), else fallback to equal split
            alloc = plan.allocation_pct / 100.0 if plan.allocation_pct > 0 else fallback_pct
            alloc = min(alloc, 0.50)  # hard cap at 50% per symbol
            max_position = snapshot.alpaca_cash * alloc
            logger.info(f"  {sym}: alloc={alloc*100:.0f}% → max ${max_position:.0f}")

            if pos_value > 0 and pos_value >= max_position:
                logger.info(f"  {sym}: already at max position (${pos_value:.0f})")
                continue

            # Calculate order size — subtract cash already reserved this cycle
            remaining_cash = snapshot.alpaca_cash - cash_reserved_this_cycle
            trade_size = min(max_position - pos_value, remaining_cash * 0.99)  # 1% buffer

            if trade_size < 12:  # Min notional ~$10
                logger.info(f"  {sym}: trade too small (${trade_size:.2f})")
                continue

            buy_price = plan.buy_price if plan.buy_price > 0 else 0.0
            if buy_price <= 0:
                continue

            qty = trade_size / buy_price

            logger.info(f"  {sym}: BUY {qty:.6f} @ ${buy_price:.2f} (${trade_size:.0f})")
            if not dry_run:
                req = LimitOrderRequest(
                    symbol=sym,
                    qty=round(qty, 8),
                    side=OrderSide.BUY,
                    type="limit",
                    time_in_force=TimeInForce.GTC,
                    limit_price=round(buy_price, 2),
                )
                result = alpaca.submit_order(req)
                cash_reserved_this_cycle += trade_size
                orders.append({"symbol": sym, "action": "buy", "price": buy_price,
                                "qty": qty, "order_id": str(result.id)})
            else:
                logger.info(f"    [DRY RUN]")
                cash_reserved_this_cycle += trade_size
                orders.append({"symbol": sym, "action": "buy", "price": buy_price,
                                "qty": qty, "dry_run": True})

        except Exception as e:
            logger.error(f"  {sym}: execution error: {e}")

    return orders


# ---------------------------------------------------------------------------
# Stock signal generation (Alpaca + Gemini)
# ---------------------------------------------------------------------------

def get_stock_signals(
    symbols: list[str],
    snapshot: UnifiedPortfolioSnapshot,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    review_thinking_level: str | None = None,
    reprompt_passes: int = 1,
    reprompt_policy: str = "always",
    review_max_confidence: float | None = None,
    review_model: str | None = None,
    dry_run: bool = True,
) -> dict[str, TradePlan]:
    """Generate LLM trading signals for stock symbols using Alpaca OHLCV data."""
    from alpaca.data.historical import StockHistoricalDataClient
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

    data_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    signals = {}
    now = datetime.now(timezone.utc)

    rl_bridge = get_rl_bridge_stock()
    rl_symbol_universe: list[str] = []
    if rl_bridge is not None:
        try:
            rl_symbol_universe = _read_trained_symbols_for_checkpoint(
                rl_bridge.checkpoint_path,
                _num_symbols_from_obs_size(rl_bridge.get_checkpoint_spec().obs_size),
            )
        except Exception as e:
            logger.warning(f"  Stock RL disabled: {e}")
            rl_bridge = None

    fetch_symbols = list(dict.fromkeys([*symbols, *rl_symbol_universe]))
    history_frames = _fetch_stock_history_frames(data_client, fetch_symbols, now)

    forecast_root = _choose_forecast_cache_root(fetch_symbols, STOCK_FORECAST_CACHE_CANDIDATES)
    forecast_frames_1h, forecast_frames_24h = _load_forecast_frames(fetch_symbols, forecast_root)

    rl_signal_map: dict[str, RLSignal] = {}
    if rl_bridge is not None:
        try:
            rl_signal_map = _build_stock_rl_signal_map(
                history_frames,
                rl_bridge,
                forecast_frames_1h,
                forecast_frames_24h,
            )
        except Exception as e:
            logger.warning(f"  Stock RL signal generation failed: {e}")

    # SMA-24 trend filter: suppress LONG hints when price is below 24h SMA.
    for sym, sig in list(rl_signal_map.items()):
        if sig.direction != "long":
            continue
        frame = history_frames.get(sym)
        if frame is None or len(frame) < 24:
            continue
        sma24 = float(frame["close"].iloc[-24:].mean())
        current = float(frame["close"].iloc[-1])
        if current < sma24:
            logger.info(f"  {sym}: SMA-24 filter suppressed RL LONG hint (price={current:.2f} < sma24={sma24:.2f})")
            rl_signal_map[sym] = RLSignal(
                symbol_idx=sig.symbol_idx,
                symbol_name=sig.symbol_name,
                direction="flat",
                confidence=0.0,
                logit_gap=0.0,
                allocation_pct=0.0,
            )

    for sym in symbols:
        try:
            frame = history_frames.get(sym)
            if frame is None or len(frame) < 12:
                logger.warning(f"  {sym}: insufficient bars ({0 if frame is None else len(frame)})")
                continue

            history = _history_from_frame(frame)
            current_price = history[-1]["close"]
            timestamp = frame["timestamp"].iloc[-1]
            forecast_1h = _forecast_at(forecast_frames_1h.get(sym), timestamp)
            forecast_24h = _forecast_at(forecast_frames_24h.get(sym), timestamp)
            rl_hint = _format_rl_hint(
                rl_signal_map.get(sym),
                "stocks13_featlag1 validated PPO model",
            )

            # Inject bearish context when price is below 24h SMA
            stock_trend_warning = ""
            if frame is not None and len(frame) >= 24:
                _sma24s = float(frame["close"].iloc[-24:].mean())
                if current_price < _sma24s:
                    stock_trend_warning = (
                        f"\nTREND CAUTION: {sym} price (${current_price:.2f}) is BELOW its 24h SMA "
                        f"(${_sma24s:.2f}). SHORT-TERM DOWNTREND. "
                        f"Strongly prefer HOLD/FLAT unless there is very clear reversal evidence."
                    )

            prompt = build_unified_prompt(
                symbol=sym,
                history_rows=history,
                current_price=current_price,
                snapshot=snapshot,
                asset_class="stock",
                forecast_1h=forecast_1h,
                forecast_24h=forecast_24h,
            ) + rl_hint + stock_trend_warning

            plan = call_llm(
                prompt,
                model=model,
                thinking_level=thinking_level,
                review_thinking_level=review_thinking_level,
                reprompt_passes=reprompt_passes,
                reprompt_policy=reprompt_policy,
                review_max_confidence=review_max_confidence,
                review_model=review_model,
            )

            ok, reason = validate_plan_safety(plan, current_price, fee_bps=10.0)
            if not ok:
                logger.warning(f"  {sym}: plan REJECTED by safety check — {reason}")
                continue

            signals[sym] = plan
            alloc_str = f" alloc={plan.allocation_pct:.0f}%" if plan.allocation_pct > 0 else ""
            logger.info(
                f"  {sym}: {plan.direction} conf={plan.confidence:.2f} "
                f"buy=${plan.buy_price:.2f} sell=${plan.sell_price:.2f}{alloc_str} | {plan.reasoning[:60]}"
            )
        except Exception as e:
            logger.error(f"  {sym}: error: {e}")

    return signals


_STOCK_FEE_BPS = 5.0  # 5 bps round-trip; safety margin for limit price checks


def _validate_trade_plan(plan: TradePlan, symbol: str) -> tuple[bool, str]:
    """Safety gate: ensure the plan is safe to execute live.

    Rules (must ALL pass):
      1. direction must be long, short, or hold
      2. buy_price must be positive when direction == long
      3. sell_price > buy_price * (1 + 2*fee) — ensures the round-trip is profitable
      4. confidence >= 0.4
      5. sell_price > 0 when direction == long
    """
    fee_factor = 1 + 2 * _STOCK_FEE_BPS / 10_000
    if plan.direction not in ("long", "short", "hold"):
        return False, f"unknown direction: {plan.direction!r}"
    if plan.direction == "long":
        if plan.buy_price <= 0:
            return False, "buy_price must be > 0 for long"
        if plan.sell_price <= 0:
            return False, "sell_price must be > 0 for long"
        if plan.sell_price <= plan.buy_price * fee_factor:
            return False, (f"sell_price ${plan.sell_price:.2f} <= buy_price ${plan.buy_price:.2f} "
                           f"* fee_factor {fee_factor:.5f} — not profitable after fees")
    if plan.confidence < 0.4:
        return False, f"confidence {plan.confidence:.2f} < threshold 0.40"
    return True, "ok"


STOCK_PEAKS_FILE = Path("strategy_state/stock_peaks.json")


def execute_stock_signals(
    signals: dict[str, TradePlan],
    snapshot: UnifiedPortfolioSnapshot,
    dry_run: bool = True,
) -> list[dict]:
    """Execute stock trading signals on Alpaca with safety validation."""
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    orders = []
    alpaca = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)

    # ── Stock trailing stop tracking ──────────────────────────────────────
    # Same mechanism as crypto: track peak price, exit if price drops 0.3%.
    # Exp06 stocks: trail 0.3% → Sortino 89.4, +6.77% (vs -1.09% baseline).
    stock_syms = set(signals.keys())
    stock_pos = {
        sym: pos for sym, pos in snapshot.alpaca_positions.items()
        if sym in stock_syms and pos.qty > 0
    }
    stock_peaks = {}
    try:
        if STOCK_PEAKS_FILE.exists():
            stock_peaks = json.loads(STOCK_PEAKS_FILE.read_text())
    except Exception:
        stock_peaks = {}

    stock_peaks = update_peak_prices(stock_peaks, stock_pos)
    trail_exit_syms = set(get_trailing_stop_symbols(
        stock_peaks, stock_pos, trail_pct=TRAILING_STOP_PCT,
    ))

    # Remove exited positions from peaks
    for sym in list(stock_peaks):
        if sym not in stock_pos:
            del stock_peaks[sym]
    try:
        STOCK_PEAKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STOCK_PEAKS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(stock_peaks, indent=2))
        tmp.replace(STOCK_PEAKS_FILE)
    except Exception:
        pass

    # Force-exit trailing stop positions
    for sym in trail_exit_syms:
        pos = snapshot.alpaca_positions.get(sym)
        if pos and pos.qty > 0:
            logger.info(f"  {sym}: TRAILING STOP — selling {pos.qty} shares near market")
            if not dry_run:
                try:
                    # Cancel existing sells first
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    open_orders = alpaca.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
                    for order in open_orders:
                        if str(order.symbol) == sym and order.side.value.lower() == "sell":
                            alpaca.cancel_order_by_id(str(order.id))
                    import time as _time
                    _time.sleep(0.5)
                    sell_price = round(pos.current_price * 0.999, 2)
                    req = LimitOrderRequest(
                        symbol=sym, qty=pos.qty, side=OrderSide.SELL,
                        type="limit", time_in_force=TimeInForce.DAY,
                        limit_price=sell_price,
                    )
                    result = alpaca.submit_order(req)
                    orders.append({"symbol": sym, "action": "trailing_stop_sell",
                                   "price": sell_price, "qty": pos.qty,
                                   "order_id": str(result.id)})
                except Exception as e:
                    logger.error(f"  {sym}: trailing stop sell error: {e}")
            else:
                orders.append({"symbol": sym, "action": "trailing_stop_sell",
                               "qty": pos.qty, "dry_run": True})

    for sym, plan in signals.items():
        try:
            if sym in trail_exit_syms:
                continue  # already exiting via trailing stop

            # Safety validation — applied to ALL orders
            ok, reason = _validate_trade_plan(plan, sym)
            if not ok:
                logger.warning(f"  {sym}: REJECTED — {reason}")
                continue

            # Take-profit on existing position (set sell limit regardless of direction)
            pos = snapshot.alpaca_positions.get(sym)
            if pos and pos.qty > 0 and plan.sell_price > 0 and plan.sell_price > pos.current_price:
                logger.info(f"  {sym}: updating take-profit sell @ ${plan.sell_price:.2f} ({pos.qty:.2f} shares)")
                if not dry_run:
                    # Cancel any existing sell orders for this symbol before placing new TP
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    try:
                        open_orders = alpaca.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
                        for order in open_orders:
                            if str(order.symbol) == sym and order.side.value.lower() == "sell":
                                alpaca.cancel_order_by_id(str(order.id))
                                logger.info(f"  {sym}: canceled old sell order {order.id}")
                    except Exception as e:
                        logger.debug(f"  {sym}: error canceling old sell: {e}")
                    req = LimitOrderRequest(
                        symbol=sym,
                        qty=pos.qty,
                        side=OrderSide.SELL,
                        type="limit",
                        time_in_force=TimeInForce.DAY,
                        limit_price=round(plan.sell_price, 2),
                    )
                    result = alpaca.submit_order(req)
                    orders.append({"symbol": sym, "action": "sell_tp", "price": plan.sell_price,
                                   "qty": pos.qty, "order_id": str(result.id)})
                else:
                    orders.append({"symbol": sym, "action": "sell_tp", "price": plan.sell_price,
                                   "qty": pos.qty, "dry_run": True})
                continue

            # New long entry
            if plan.direction != "long" or plan.confidence < 0.5 or plan.buy_price <= 0:
                continue

            # Don't add to a position we already hold
            if pos and pos.qty > 0:
                continue

            # Leverage-aware sizing: use up to 4x intraday, 2x near close
            equity = max(snapshot.total_stock_value, 1.0)
            long_val = sum(p.market_value for p in snapshot.alpaca_positions.values())
            current_leverage = long_val / equity if equity > 0 else 0

            # Determine max leverage based on time to close
            if snapshot.minutes_to_close is not None and snapshot.minutes_to_close <= DELEVERAGE_MINUTES_BEFORE_CLOSE:
                max_lev = MAX_OVERNIGHT_LEVERAGE
            else:
                max_lev = MAX_INTRADAY_LEVERAGE

            max_total_exposure = equity * max_lev
            room_for_new = max(0, max_total_exposure - long_val)

            # Use LLM allocation_pct if provided, else 20% of equity
            alloc = plan.allocation_pct / 100.0 if plan.allocation_pct > 0 else 0.20
            alloc = min(alloc, 0.50)  # hard cap at 50% per stock
            max_position = equity * alloc
            available = min(room_for_new, snapshot.alpaca_buying_power * 0.95)
            trade_usd = min(max_position, available)
            logger.info(f"  {sym}: alloc={alloc*100:.0f}% lev={current_leverage:.2f}x/{max_lev:.0f}x → max ${trade_usd:.0f}")

            if trade_usd < 50:
                logger.info(f"  {sym}: insufficient buying power (${trade_usd:.0f})")
                continue

            qty = trade_usd / plan.buy_price
            qty = round(qty, 2)
            if qty < 0.01:
                continue

            logger.info(f"  {sym}: BUY {qty:.2f} @ ${plan.buy_price:.2f} (${trade_usd:.0f})")
            if not dry_run:
                req = LimitOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=OrderSide.BUY,
                    type="limit",
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(plan.buy_price, 2),
                )
                result = alpaca.submit_order(req)
                orders.append({"symbol": sym, "action": "buy", "price": plan.buy_price,
                                "qty": qty, "order_id": str(result.id)})
            else:
                logger.info(f"    [DRY RUN]")
                orders.append({"symbol": sym, "action": "buy", "price": plan.buy_price,
                                "qty": qty, "dry_run": True})

        except Exception as e:
            logger.error(f"  {sym}: execution error: {e}")

    return orders


# ---------------------------------------------------------------------------
# End-of-day deleveraging
# ---------------------------------------------------------------------------

def deleverage_to_target(
    snapshot: UnifiedPortfolioSnapshot,
    target_leverage: float = MAX_OVERNIGHT_LEVERAGE,
    dry_run: bool = True,
    use_limit: bool = True,
    limit_offset_pct: float = 0.05,
) -> list[dict]:
    """Sell stock positions to bring leverage down to target.

    Called in the final hour before market close. Sells the smallest/worst
    positions first to minimize disruption to high-conviction holdings.

    Args:
        use_limit: Use limit orders near current price instead of market orders.
                   Limit price = current_price * (1 - limit_offset_pct/100).
        limit_offset_pct: Percent below current price for limit (default 0.05%).
    """
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    equity = max(snapshot.total_stock_value, 1.0)
    stock_positions = {
        sym: pos for sym, pos in snapshot.alpaca_positions.items()
        if sym not in set(CRYPTO_SYMBOLS) and pos.qty > 0
    }
    long_val = sum(p.market_value for p in stock_positions.values())
    current_lev = long_val / equity

    if current_lev <= target_leverage:
        logger.info(f"  Deleverage: {current_lev:.2f}x <= {target_leverage:.1f}x target — no action needed")
        return []

    excess = long_val - equity * target_leverage
    logger.info(f"  Deleverage: {current_lev:.2f}x → {target_leverage:.1f}x | need to sell ${excess:,.0f}")

    # Sort positions: sell smallest first, then worst PnL
    ranked = sorted(
        stock_positions.items(),
        key=lambda kv: (kv[1].market_value, kv[1].unrealized_pnl),
    )

    orders = []
    alpaca = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)
    sold_value = 0.0

    for sym, pos in ranked:
        if sold_value >= excess:
            break

        # Sell full position if it fits, partial otherwise
        remaining_excess = excess - sold_value
        if pos.market_value <= remaining_excess * 1.2:
            sell_qty = pos.qty
        else:
            sell_qty = round(remaining_excess / pos.current_price, 2)
            sell_qty = min(sell_qty, pos.qty)

        if sell_qty < 0.01:
            continue

        # Limit order near market price — fills quickly without market order slippage
        limit_price = round(pos.current_price * (1 - limit_offset_pct / 100), 2)
        sell_value = sell_qty * pos.current_price
        order_type = "limit" if use_limit else "market"
        logger.info(f"  Deleverage SELL: {sym} {sell_qty:.2f} shares @ ${limit_price:.2f} "
                    f"({order_type}, ${sell_value:,.0f})")

        if not dry_run:
            try:
                req = LimitOrderRequest(
                    symbol=sym,
                    qty=sell_qty,
                    side=OrderSide.SELL,
                    type="limit",
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
                result = alpaca.submit_order(req)
                orders.append({
                    "symbol": sym, "action": "deleverage_sell",
                    "qty": sell_qty, "limit_price": limit_price,
                    "order_id": str(result.id),
                })
                sold_value += sell_value
            except Exception as e:
                logger.error(f"  {sym}: deleverage sell error: {e}")
        else:
            orders.append({
                "symbol": sym, "action": "deleverage_sell",
                "qty": sell_qty, "limit_price": limit_price,
                "dry_run": True,
            })
            sold_value += sell_value

    new_lev = (long_val - sold_value) / equity
    logger.info(f"  Deleverage complete: {current_lev:.2f}x → {new_lev:.2f}x ({len(orders)} sells)")
    return orders


# ---------------------------------------------------------------------------
# Main trading cycle
# ---------------------------------------------------------------------------

def run_cycle(
    crypto_symbols: list[str],
    stock_symbols: list[str] | None = None,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    review_thinking_level: str | None = None,
    reprompt_passes: int = 1,
    reprompt_policy: str = "always",
    review_max_confidence: float | None = None,
    review_model: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Run one unified trading cycle."""
    now = datetime.now(timezone.utc)
    logger.info(f"\n{'=' * 70}")
    logger.info(f"UNIFIED TRADING CYCLE: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

    # 1. Build snapshot
    snapshot = build_snapshot(now)
    logger.info(f"Regime: {snapshot.regime}")
    equity = max(snapshot.total_stock_value, 1.0)
    long_val = sum(p.market_value for p in snapshot.alpaca_positions.values())
    lev = long_val / equity if equity > 0 else 0
    margin_cost_day = max(0, long_val - equity) * MARGIN_INTEREST_ANNUAL / 365
    logger.info(f"Equity: ${equity:,.0f} | Positions: ${long_val:,.0f} | "
                f"Leverage: {lev:.2f}x | Margin cost: ${margin_cost_day:.2f}/day")
    if snapshot.alpaca_positions:
        crypto_syms = set(CRYPTO_SYMBOLS)
        for sym, pos in snapshot.alpaca_positions.items():
            if sym in crypto_syms:
                prefix = "Crypto" if _has_actionable_crypto_position(pos) else "Crypto dust"
                logger.info(f"  {prefix}: {sym} {pos.qty:.6f} @ ${pos.current_price:.2f} (${pos.market_value:.0f})")
            else:
                logger.info(f"  Stock: {sym} {pos.qty} @ ${pos.avg_price:.2f} (${pos.market_value:.0f})")
    logger.info(f"{'=' * 70}")

    results = {"regime": snapshot.regime, "orders": []}

    # 2. Handle regime-specific logic
    if snapshot.regime == "PRE_MARKET":
        logger.info("\n--- PRE-MARKET: Checking crypto backout opportunities ---")
        # TODO: Load best stock edges from meta-selector
        best_stock_edges = {}  # Will integrate with meta_live_runtime
        candidates = select_backout_candidates(snapshot, best_stock_edges)
        if candidates:
            backout_results = execute_backout(candidates, dry_run=dry_run)
            results["backout"] = backout_results
        else:
            logger.info("  No backout candidates")

    # 3. Generate and execute crypto signals (always, except pure stock hours)
    if snapshot.regime in ("CRYPTO_ONLY", "PRE_MARKET", "POST_MARKET", "STOCK_HOURS"):
        logger.info(f"\n--- CRYPTO SIGNALS ({len(crypto_symbols)} symbols) ---")
        crypto_signals = get_crypto_signals(
            crypto_symbols,
            snapshot,
            model,
            thinking_level,
            review_thinking_level,
            reprompt_passes,
            reprompt_policy,
            review_max_confidence,
            review_model,
            dry_run,
        )
        if crypto_signals:
            crypto_orders = execute_crypto_signals(crypto_signals, snapshot, dry_run)
            results["orders"].extend(crypto_orders)

            # Spawn intra-hour watcher for buy orders placed this cycle
            watcher_pairs = []
            for order in crypto_orders:
                if order.get("action") == "buy" and order.get("qty", 0) > 0:
                    sym = order["symbol"]
                    plan = crypto_signals.get(sym)
                    if plan and plan.sell_price > 0:
                        watcher_pairs.append(OrderPair(
                            symbol=sym,
                            buy_price=order["price"],
                            sell_price=plan.sell_price,
                            target_qty=order["qty"],
                        ))
            if watcher_pairs:
                # Stop previous watcher if still running
                prev_watcher = results.get("_watcher")
                if prev_watcher and prev_watcher.is_alive():
                    prev_watcher.stop()
                    prev_watcher.join(timeout=5)
                watcher = AlpacaCryptoWatcher(
                    alpaca_client=None,  # will create its own client
                    pairs=watcher_pairs,
                    expiry_minutes=55,
                    dry_run=dry_run,
                )
                watcher.start()
                results["_watcher"] = watcher
                logger.info(f"  Watcher spawned for {len(watcher_pairs)} pairs")

    # 4. Stock signals during market hours (Gemini-driven)
    if snapshot.regime == "STOCK_HOURS":
        # 4a. End-of-day deleveraging (final hour before close)
        if snapshot.minutes_to_close is not None and snapshot.minutes_to_close <= DELEVERAGE_MINUTES_BEFORE_CLOSE:
            logger.info(f"\n--- DELEVERAGE CHECK ({snapshot.minutes_to_close} min to close) ---")
            delev_orders = deleverage_to_target(snapshot, MAX_OVERNIGHT_LEVERAGE, dry_run)
            if delev_orders:
                results["orders"].extend(delev_orders)
                # Refresh snapshot after sells
                snapshot = build_snapshot(datetime.now(timezone.utc))

        # 4b. Generate and execute stock signals
        syms = stock_symbols or STOCK_SYMBOLS
        logger.info(f"\n--- STOCK SIGNALS ({len(syms)} symbols) ---")
        # Log leverage info
        equity = max(snapshot.total_stock_value, 1.0)
        long_val = sum(p.market_value for p in snapshot.alpaca_positions.values()
                       if p.symbol not in set(CRYPTO_SYMBOLS))
        logger.info(f"  Leverage: {long_val / equity:.2f}x | Equity: ${equity:,.0f} | "
                    f"Margin cost: ${max(0, long_val - equity) * MARGIN_INTEREST_ANNUAL / 365:.2f}/day")
        stock_signals = get_stock_signals(
            syms,
            snapshot,
            model,
            thinking_level,
            review_thinking_level,
            reprompt_passes,
            reprompt_policy,
            review_max_confidence,
            review_model,
            dry_run,
        )
        if stock_signals:
            stock_orders = execute_stock_signals(stock_signals, snapshot, dry_run)
            results["orders"].extend(stock_orders)
            results["stock_signals"] = {s: {"direction": p.direction, "confidence": p.confidence}
                                         for s, p in stock_signals.items()}

    # 5. Check conditional order triggers
    pending_fills = read_pending_fills(since_minutes=65)
    if pending_fills:
        logger.info(f"\n--- CONDITIONAL TRIGGERS: {len(pending_fills)} recent fills ---")
        for fill in pending_fills:
            logger.info(f"  Fill: {fill['symbol']} {fill['action']} @ ${fill.get('fill_price', '?')}")

    # 6. Persist state
    save_snapshot(snapshot)
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Cycle complete: {len(results.get('orders', []))} orders")
    logger.info(f"{'=' * 70}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Stock+Crypto Trading Orchestrator")
    parser.add_argument("--crypto-symbols", nargs="+", default=CRYPTO_SYMBOLS)
    parser.add_argument("--stock-symbols", nargs="+", default=STOCK_SYMBOLS)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--thinking-level", default="HIGH")
    parser.add_argument("--review-thinking-level", default=None,
                        help="Optional thinking level for pass 2+. Defaults to --thinking-level.")
    parser.add_argument("--reprompt-passes", type=int, default=1,
                        help="Total Gemini plan passes per symbol. 1 = current behavior, 2 = review once.")
    parser.add_argument("--reprompt-policy", choices=["always", "actionable", "entry_only"], default="always",
                        help="When to run pass 2+. 'actionable' reviews any order-managing plan; 'entry_only' only reviews plans with a buy target.")
    parser.add_argument("--review-max-confidence", type=float, default=None,
                        help="Optional cap on first-pass confidence for running pass 2+. Example: 0.60 only reviews plans at 60% confidence or below.")
    parser.add_argument("--review-model", default=None,
                        help="Optional alternate model for review pass 2+. Defaults to the primary model.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=3600)
    args = parser.parse_args()

    dry_run = not args.live
    if args.live:
        logger.warning("LIVE TRADING MODE")
        time.sleep(3)

    while True:
        try:
            run_cycle(
                crypto_symbols=args.crypto_symbols,
                stock_symbols=args.stock_symbols,
                model=args.model,
                thinking_level=args.thinking_level,
                review_thinking_level=args.review_thinking_level,
                reprompt_passes=args.reprompt_passes,
                reprompt_policy=args.reprompt_policy,
                review_max_confidence=args.review_max_confidence,
                review_model=args.review_model,
                dry_run=dry_run,
            )
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            import traceback
            traceback.print_exc()

        if args.once:
            break

        # Sleep until :01 of next hour
        now = datetime.now(timezone.utc)
        next_hour = now.replace(minute=1, second=0, microsecond=0)
        if next_hour <= now:
            from datetime import timedelta
            next_hour += timedelta(hours=1)
        sleep_secs = (next_hour - now).total_seconds()
        logger.info(f"Next cycle at {next_hour.strftime('%H:%M')} UTC ({sleep_secs:.0f}s)")
        time.sleep(sleep_secs)


if __name__ == "__main__":
    main()
