#!/usr/bin/env python3
"""
Production daily stock RL trading bot.

This trader runs a long-only daily PPO policy on U.S. equities. It uses the
previous completed trading day's bar set for inference and places a single
market order shortly after the regular session opens.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.inference_daily import DailyPPOTrader, compute_daily_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("daily_stock_rl")

EASTERN = ZoneInfo("America/New_York")
RUN_AFTER_OPEN_ET = dt_time(hour=9, minute=35)

DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "META",
    "TSLA",
    "SPY",
    "QQQ",
    "PLTR",
    "JPM",
    "V",
    "AMZN",
]
DEFAULT_CHECKPOINT = "pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_10/best.pt"
# 15-model ensemble: tp10+s15+s36+gamma_995+muon_wd_005+h1024_a40+s1731+gamma995_s2006+s1401+s1726+s1523+s2617+s2033+s2495+s1835 (2026-03-28)
# NOTE: Original s735 (screen_best) was deleted from disk (2026-03-28). Replaced with s1731 (screen_best).
# Exhaustive 111-window @fee=10bps,fill=5bps: 0/111 neg, med=58.8%, p10=48.6%
# (Original 10-model with s735 was: 0/111 neg, med=65.8%, p10=55.6% — s735 provided unique diversity)
# s1835 screen_best adds: +0.6% p10 (seed=1835, neg=1 screen, med=7.87%)
# s2495 screen_best adds: +2.0% p10 (seed=2495, QUALIFIED, neg=5, med=21.13%)
# s2033 screen_best adds: +2.6% p10 (seed=2033, QUALIFIED, neg=5, med=19.23%)
# s2617 screen_best adds: +2.0% p10 (seed=2617, QUALIFIED, neg=2, med=16.95%)
# s1523 screen_best adds: +4.6% p10 (seed=1523 retrain, 3M-step screen ckpt)
# s1731 screen_best adds: +4.1% p10 (seed=1731, 3M-step screen ckpt update=61)
# s1401 screen_best adds: +2.9% p10 (seed=1401, 3M-step screen ckpt)
# s1726 screen_best adds: +0.4% p10 (seed=1726, 3M-step screen ckpt update=65)
# gamma995_s2006 screen_best: (gamma=0.995, seed=2006, 3M-step screen ckpt)
# REVERT NOTE (2026-03-27): 7-model (+resmlp_a40) → med=57.2%, p10=42.1% (-3.3% p10!)
#   HURT: resmlp_a40, s28, tp03, s241, s541, s310, stock_ent_05
# 16-model bar: 16-model exhaustive p10 >= 48.6% @fill_bps=5
DEFAULT_EXTRA_CHECKPOINTS = [
    "pufferlib_market/checkpoints/stocks12_seed_sweep/tp05_s15/best.pt",
    "pufferlib_market/checkpoints/stocks12_seed_sweep/tp05_s36/best.pt",
    "pufferlib_market/checkpoints/stocks12_v2_sweep/stock_gamma_995/best.pt",
    "pufferlib_market/checkpoints/stocks12_v2_sweep/muon_wd_005/best.pt",
    "pufferlib_market/checkpoints/stocks12_v2_sweep/h1024_a40/best.pt",
    "pufferlib_market/checkpoints/stocks12_s1731_screen/screen_best.pt",
    "pufferlib_market/checkpoints/stocks12_gamma995_s2006/screen_best.pt",
    "pufferlib_market/checkpoints/stocks12_s1401_screen/screen_best.pt",
    "pufferlib_market/checkpoints/stocks12_s1726_screen/screen_best.pt",
    "pufferlib_market/checkpoints/stocks12_s1523_screen/screen_best.pt",
    "pufferlib_market/checkpoints/stocks12_s2617_screen/screen_best.pt",
    "pufferlib_market/checkpoints/stocks12_s2033_screen/screen_best.pt",
    "pufferlib_market/checkpoints/stocks12_s2495_screen/screen_best.pt",
    "pufferlib_market/checkpoints/stocks12_s1835_screen/screen_best.pt",
]
DEFAULT_DATA_DIR = "trainingdata"
DEFAULT_ALLOCATION_PCT = 25.0
STATE_PATH = REPO / "strategy_state/daily_stock_rl_state.json"
SIGNAL_LOG_PATH = REPO / "strategy_state/daily_stock_rl_signals.jsonl"


@dataclass
class StrategyState:
    active_symbol: Optional[str] = None
    active_qty: float = 0.0
    entry_price: float = 0.0
    entry_date: Optional[str] = None
    last_run_date: Optional[str] = None
    last_signal_action: Optional[str] = None
    last_signal_timestamp: Optional[str] = None
    last_order_id: Optional[str] = None


@dataclass(frozen=True)
class PortfolioContext:
    cash: float = 10_000.0
    current_symbol: Optional[str] = None
    position_qty: float = 0.0
    entry_price: float = 0.0
    hold_days: int = 0


def _normalize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    lower_map = {str(col).lower(): col for col in frame.columns}
    ts_col = lower_map.get("timestamp") or lower_map.get("date")
    required = ["open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in lower_map]
    if ts_col is None or missing:
        raise ValueError(f"Daily frame missing columns: timestamp/date + {required}")

    normalized = frame.rename(columns={src: src.lower() for src in frame.columns}).copy()
    normalized["timestamp"] = pd.to_datetime(normalized[ts_col.lower()], utc=True)
    normalized = normalized[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp")
    normalized = normalized.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
    for column in required:
        normalized[column] = normalized[column].astype(float)
    return normalized


def _align_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not frames:
        raise ValueError("No daily frames to align")

    common_index: Optional[pd.DatetimeIndex] = None
    aligned: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        indexed = _normalize_daily_frame(frame).set_index("timestamp").sort_index()
        common_index = indexed.index if common_index is None else common_index.intersection(indexed.index)
        aligned[symbol] = indexed

    if common_index is None or len(common_index) == 0:
        raise ValueError("No common daily timestamps across symbols")

    result: dict[str, pd.DataFrame] = {}
    for symbol, indexed in aligned.items():
        trimmed = indexed.loc[common_index].copy()
        trimmed.index.name = "timestamp"
        result[symbol] = trimmed.reset_index()
    return result


def load_local_daily_frames(
    symbols: Iterable[str],
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    min_days: int = 120,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    base = REPO / data_dir
    for symbol in symbols:
        path = base / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing local daily data for {symbol}: {path}")
        frame = _normalize_daily_frame(pd.read_csv(path))
        if len(frame) < min_days:
            raise ValueError(f"{symbol}: only {len(frame)} rows in {path}, need at least {min_days}")
        frames[symbol] = frame
    return _align_frames(frames)


def _drop_incomplete_session(frame: pd.DataFrame, *, now: datetime) -> pd.DataFrame:
    if frame.empty:
        return frame
    latest_ts = pd.Timestamp(frame["timestamp"].iloc[-1])
    latest_et = latest_ts.tz_convert(EASTERN)
    now_et = now.astimezone(EASTERN)
    if latest_et.date() >= now_et.date():
        return frame.iloc[:-1].reset_index(drop=True)
    return frame


def build_trading_client(*, paper: bool):
    from alpaca.trading.client import TradingClient
    from env_real import ALP_KEY_ID, ALP_KEY_ID_PROD, ALP_SECRET_KEY, ALP_SECRET_KEY_PROD

    key_id = ALP_KEY_ID if paper else (ALP_KEY_ID_PROD or ALP_KEY_ID)
    secret = ALP_SECRET_KEY if paper else (ALP_SECRET_KEY_PROD or ALP_SECRET_KEY)
    return TradingClient(key_id, secret, paper=paper)


def build_data_client(*, paper: bool):
    from alpaca.data import StockHistoricalDataClient
    from env_real import ALP_KEY_ID, ALP_KEY_ID_PROD, ALP_SECRET_KEY, ALP_SECRET_KEY_PROD

    key_id = ALP_KEY_ID if paper else (ALP_KEY_ID_PROD or ALP_KEY_ID)
    secret = ALP_SECRET_KEY if paper else (ALP_SECRET_KEY_PROD or ALP_SECRET_KEY)
    return StockHistoricalDataClient(key_id, secret)


def _frames_from_alpaca_bars(
    *,
    symbols: Sequence[str],
    bars_df: pd.DataFrame | None,
    now: datetime,
) -> dict[str, pd.DataFrame]:
    if bars_df is None or len(bars_df) == 0:
        return {}

    ordered_symbols = [str(symbol).upper() for symbol in symbols]
    frames: dict[str, pd.DataFrame] = {}
    if isinstance(bars_df.index, pd.MultiIndex):
        grouped = bars_df.reset_index().groupby("symbol", sort=False)
        for symbol, group in grouped:
            normalized = _normalize_daily_frame(group)
            frames[str(symbol).upper()] = _drop_incomplete_session(normalized, now=now)
        return frames

    flat = bars_df.reset_index()
    if "symbol" in flat.columns:
        grouped = flat.groupby("symbol", sort=False)
        for symbol, group in grouped:
            normalized = _normalize_daily_frame(group)
            frames[str(symbol).upper()] = _drop_incomplete_session(normalized, now=now)
        return frames

    if len(ordered_symbols) != 1:
        raise RuntimeError("Expected Alpaca bars dataframe to include symbol information for multi-symbol requests")
    normalized = _normalize_daily_frame(flat)
    frames[ordered_symbols[0]] = _drop_incomplete_session(normalized, now=now)
    return frames


def _request_alpaca_daily_frames(
    *,
    client,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    feed,
    now: datetime,
) -> dict[str, pd.DataFrame]:
    from alpaca.data import StockBarsRequest, TimeFrame

    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="raw",
        feed=feed,
    )
    bars = client.get_stock_bars(request)
    bars_df = getattr(bars, "df", None)
    return _frames_from_alpaca_bars(symbols=symbols, bars_df=bars_df, now=now)


def load_alpaca_daily_frames(
    symbols: Iterable[str],
    *,
    paper: bool,
    min_days: int = 120,
    history_days: int = 420,
    now: Optional[datetime] = None,
    data_client=None,
) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import DataFeed

    now = now or datetime.now(timezone.utc)
    client = data_client or build_data_client(paper=paper)
    ordered_symbols = [str(symbol).upper() for symbol in symbols]
    start = now - timedelta(days=history_days)

    frames: dict[str, pd.DataFrame] = {}
    insufficient_counts: dict[str, int] = {}
    fetch_errors: list[str] = []

    def _accept_resolved(candidates: dict[str, pd.DataFrame]) -> None:
        for symbol, frame in candidates.items():
            rows = len(frame)
            if rows < min_days:
                insufficient_counts[symbol] = max(rows, insufficient_counts.get(symbol, 0))
                continue
            frames[symbol] = frame
            insufficient_counts.pop(symbol, None)

    def _remaining_symbols() -> list[str]:
        return [symbol for symbol in ordered_symbols if symbol not in frames]

    for feed in (DataFeed.IEX, DataFeed.SIP):
        remaining = _remaining_symbols()
        if not remaining:
            break
        try:
            _accept_resolved(
                _request_alpaca_daily_frames(
                    client=client,
                    symbols=remaining,
                    start=start,
                    end=now,
                    feed=feed,
                    now=now,
                )
            )
        except Exception as exc:
            fetch_errors.append(f"{getattr(feed, 'value', feed)} batch: {exc}")

        for symbol in list(_remaining_symbols()):
            try:
                _accept_resolved(
                    _request_alpaca_daily_frames(
                        client=client,
                        symbols=[symbol],
                        start=start,
                        end=now,
                        feed=feed,
                        now=now,
                    )
                )
            except Exception as exc:
                fetch_errors.append(f"{getattr(feed, 'value', feed)} {symbol}: {exc}")

    if not frames:
        details = f" ({'; '.join(fetch_errors)})" if fetch_errors else ""
        raise RuntimeError(f"No Alpaca daily bars returned{details}")

    missing = _remaining_symbols()
    if missing:
        missing_with_counts = [
            f"{symbol} ({insufficient_counts[symbol]} bars)"
            if symbol in insufficient_counts
            else symbol
            for symbol in missing
        ]
        if any(symbol in insufficient_counts for symbol in missing):
            raise ValueError(
                f"Missing enough Alpaca daily bars for: {', '.join(missing_with_counts)}; need at least {min_days}"
            )
        raise RuntimeError(f"Missing Alpaca daily bars for: {', '.join(missing_with_counts)}")
    return _align_frames({symbol: frames[symbol] for symbol in ordered_symbols})


def load_state(path: Path = STATE_PATH) -> StrategyState:
    if not path.exists():
        return StrategyState()
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return StrategyState()
    return StrategyState(
        active_symbol=payload.get("active_symbol"),
        active_qty=float(payload.get("active_qty", 0.0) or 0.0),
        entry_price=float(payload.get("entry_price", 0.0) or 0.0),
        entry_date=payload.get("entry_date"),
        last_run_date=payload.get("last_run_date"),
        last_signal_action=payload.get("last_signal_action"),
        last_signal_timestamp=payload.get("last_signal_timestamp"),
        last_order_id=payload.get("last_order_id"),
    )


def save_state(state: StrategyState, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))


def append_signal_log(payload: dict, path: Path = SIGNAL_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def latest_close_prices(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    return {symbol: float(frame["close"].iloc[-1]) for symbol, frame in frames.items()}


def load_latest_quotes_with_source(
    symbols: Iterable[str],
    *,
    paper: bool,
    fallback_prices: dict[str, float],
    data_client=None,
) -> tuple[dict[str, float], str, dict[str, str]]:
    from alpaca.data import StockLatestQuoteRequest
    from alpaca.data.enums import DataFeed

    client = data_client or build_data_client(paper=paper)
    request = StockLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX)
    try:
        quotes = client.get_stock_latest_quote(request)
    except Exception as exc:
        logger.warning("Falling back to previous close prices for quotes: %s", exc)
        prices = {symbol: float(fallback_prices[symbol]) for symbol in symbols}
        sources = {symbol: "close_fallback" for symbol in symbols}
        return prices, "close_fallback", sources

    prices: dict[str, float] = {}
    quote_source_by_symbol: dict[str, str] = {}
    for symbol in symbols:
        quote = quotes.get(symbol)
        ask = float(getattr(quote, "ask_price", 0.0) or 0.0) if quote is not None else 0.0
        bid = float(getattr(quote, "bid_price", 0.0) or 0.0) if quote is not None else 0.0
        last_price = ask or bid or float(fallback_prices[symbol])
        prices[symbol] = last_price
        quote_source_by_symbol[symbol] = "alpaca" if (ask > 0.0 or bid > 0.0) else "close_fallback"

    overall_source = (
        "alpaca"
        if all(source == "alpaca" for source in quote_source_by_symbol.values())
        else "mixed_fallback"
    )
    return prices, overall_source, quote_source_by_symbol


def load_latest_quotes(
    symbols: Iterable[str],
    *,
    paper: bool,
    fallback_prices: dict[str, float],
    data_client=None,
) -> dict[str, float]:
    prices, _, _ = load_latest_quotes_with_source(
        symbols,
        paper=paper,
        fallback_prices=fallback_prices,
        data_client=data_client,
    )
    return prices


def load_inference_frames(
    symbols: Iterable[str],
    *,
    paper: bool,
    data_dir: str,
    now: datetime,
    data_client=None,
) -> tuple[dict[str, pd.DataFrame], str]:
    try:
        frames = load_alpaca_daily_frames(symbols, paper=paper, data_client=data_client, now=now)
        return frames, "alpaca"
    except Exception as exc:
        logger.warning("Falling back to local daily CSVs for inference bars: %s", exc)
        frames = load_local_daily_frames(symbols, data_dir=data_dir)
        return frames, "local_fallback"


def _load_bare_policy(checkpoint_path: str, obs_size: int, num_actions: int, device: str):
    """Load a policy nn.Module from a checkpoint without full PPOTrader overhead."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Support multiple checkpoint formats: direct state_dict, {"model": sd}, {"model_state_dict": sd}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    encoder_key = [k for k in state_dict if "encoder" in k and "weight" in k]
    if encoder_key:
        hidden = state_dict[encoder_key[0]].shape[0]
        has_encoder_norm = any("encoder_norm" in k for k in state_dict)
        from pufferlib_market.train import TradingPolicy
        policy = TradingPolicy(obs_size, num_actions, hidden, use_encoder_norm=has_encoder_norm)
    else:
        input_proj_key = [k for k in state_dict if "input_proj" in k and "weight" in k]
        hidden = state_dict[input_proj_key[0]].shape[0] if input_proj_key else 256
        from pufferlib_market.inference import Policy
        policy = Policy(obs_size, num_actions, hidden, 3)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(torch.device(device))
    policy.eval()
    return policy


def _ensemble_softmax_signal(
    primary: DailyPPOTrader,
    extra_policies: list,
    features: np.ndarray,
    prices: dict,
):
    """Softmax-average probabilities across primary + extra policies, return TradingSignal."""
    obs = primary.build_observation(features, prices)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(primary.device)
    all_probs = []
    value_est = 0.0
    with torch.inference_mode():
        logits, value = primary.policy(obs_t)
        all_probs.append(F.softmax(logits, dim=-1))
        value_est = float(value.item())
        for pol in extra_policies:
            logits_i, _ = pol(obs_t)
            all_probs.append(F.softmax(logits_i, dim=-1))
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    action = int(avg_probs.argmax(dim=-1).item())
    confidence = float(avg_probs[0, action].item())
    return primary._decode_action(action, confidence, value_est)


def build_signal(
    checkpoint: str,
    frames: dict[str, pd.DataFrame],
    *,
    device: str = "cpu",
    portfolio: PortfolioContext = PortfolioContext(),
    extra_checkpoints: Optional[list] = None,
):
    aligned = _align_frames(frames)
    indexed = {
        symbol: frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
        for symbol, frame in aligned.items()
    }
    prices = {symbol: float(frame["close"].iloc[-1]) for symbol, frame in aligned.items()}
    trader = DailyPPOTrader(checkpoint, device=device, long_only=True, symbols=list(indexed.keys()))
    trader.cash = float(portfolio.cash)
    trader.position_qty = float(portfolio.position_qty)
    trader.entry_price = float(portfolio.entry_price)
    trader.hold_days = int(max(0, portfolio.hold_days))
    trader.hold_hours = trader.hold_days
    trader.step = min(trader.hold_days, trader.max_steps)
    trader.current_position = None
    if portfolio.current_symbol:
        symbol_upper = portfolio.current_symbol.upper()
        if symbol_upper in trader.SYMBOLS and trader.position_qty > 0:
            trader.current_position = trader.SYMBOLS.index(symbol_upper)

    if extra_checkpoints:
        features = np.zeros((trader.num_symbols, 16), dtype=np.float32)
        for i, sym in enumerate(trader.SYMBOLS):
            if sym in indexed:
                features[i] = compute_daily_features(indexed[sym])
        extra_policies = [
            _load_bare_policy(
                str((REPO / p).resolve()) if not Path(p).is_absolute() else p,
                trader.obs_size,
                trader.num_actions,
                device,
            )
            for p in extra_checkpoints
        ]
        signal = _ensemble_softmax_signal(trader, extra_policies, features, prices)
        logger.info("Ensemble signal (%d policies, softmax_avg)", 1 + len(extra_policies))
    else:
        signal = trader.get_daily_signal(indexed, prices)

    if signal.direction == "short":
        logger.warning("Checkpoint produced short signal on long-only path; flattening")
        signal = signal.__class__(
            action="flat",
            symbol=None,
            direction=None,
            confidence=float(signal.confidence),
            value_estimate=float(signal.value_estimate),
            allocation_pct=0.0,
            level_offset_bps=0.0,
    )
    return signal, prices


def latest_bar_timestamp(frames: dict[str, pd.DataFrame]) -> pd.Timestamp:
    if not frames:
        raise ValueError("No frames available")
    latest = pd.Timestamp(next(iter(frames.values()))["timestamp"].iloc[-1])
    return latest.tz_localize("UTC") if latest.tzinfo is None else latest.tz_convert("UTC")


def bars_are_fresh(*, latest_bar: pd.Timestamp, now: datetime, max_age_days: int = 5) -> bool:
    latest_bar = latest_bar.tz_localize("UTC") if latest_bar.tzinfo is None else latest_bar.tz_convert("UTC")
    age_days = (now.date() - latest_bar.date()).days
    return age_days <= max(0, int(max_age_days))


def _signed_position_qty(position) -> float:
    qty = float(getattr(position, "qty", 0.0) or 0.0)
    side = str(getattr(position, "side", "long") or "long").lower()
    if side == "short" and qty > 0:
        return -qty
    return qty


def positions_by_symbol(client, symbols: Iterable[str]) -> dict[str, object]:
    target = {str(symbol).upper() for symbol in symbols}
    positions: dict[str, object] = {}
    for position in client.get_all_positions():
        symbol = str(getattr(position, "symbol", "")).upper()
        if symbol in target:
            positions[symbol] = position
    return positions


def _market_order_side_for_qty(qty: float) -> str:
    return "sell" if qty > 0 else "buy"


def submit_market_order(client, *, symbol: str, qty: float, side: str):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if qty <= 0:
        raise ValueError("qty must be positive")
    side_value = OrderSide.BUY if side == "buy" else OrderSide.SELL
    request = MarketOrderRequest(
        symbol=symbol,
        qty=round(float(qty), 4),
        side=side_value,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(request)


def compute_target_qty(*, account, price: float, allocation_pct: float) -> float:
    portfolio_value = float(getattr(account, "portfolio_value", 0.0) or 0.0)
    buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
    if price <= 0 or portfolio_value <= 0 or buying_power <= 0:
        return 0.0
    target_notional = portfolio_value * max(0.0, allocation_pct) / 100.0
    target_notional = min(target_notional, buying_power * 0.95)
    if target_notional <= 0:
        return 0.0
    return round(target_notional / price, 4)


def execute_signal(
    signal,
    *,
    client,
    quotes: dict[str, float],
    state: StrategyState,
    symbols: Iterable[str],
    allocation_pct: float,
    dry_run: bool,
    now: Optional[datetime] = None,
    allow_open: bool = True,
) -> bool:
    now = now or datetime.now(timezone.utc)
    symbol_set = [str(symbol).upper() for symbol in symbols]
    live_positions = positions_by_symbol(client, symbol_set)
    managed_symbol = state.active_symbol.upper() if state.active_symbol else None
    desired_symbol = signal.symbol.upper() if signal.symbol and signal.direction == "long" else None

    unmanaged = sorted(symbol for symbol in live_positions if symbol != managed_symbol)
    if unmanaged:
        logger.warning(
            "Found unmanaged position(s) in the strategy universe: %s. Refusing to place new orders.",
            ", ".join(unmanaged),
        )
        return False

    managed_position = live_positions.get(managed_symbol) if managed_symbol else None
    if desired_symbol and managed_symbol == desired_symbol and managed_position is not None:
        logger.info("Holding existing managed position in %s", desired_symbol)
        return False

    if managed_position is not None and managed_symbol is not None:
        qty = _signed_position_qty(managed_position)
        close_side = _market_order_side_for_qty(qty)
        logger.info("Closing managed position: %s qty=%.4f side=%s", managed_symbol, abs(qty), close_side)
        if not dry_run:
            order = submit_market_order(client, symbol=managed_symbol, qty=abs(qty), side=close_side)
            state.last_order_id = str(getattr(order, "id", ""))
            state.active_symbol = None
            state.active_qty = 0.0
            state.entry_price = 0.0
            state.entry_date = None

    if desired_symbol is None:
        logger.info("Signal is flat; no new position opened")
        return managed_position is not None

    if not allow_open:
        logger.warning("Skipping new position open for %s because execution safety gate is active", desired_symbol)
        return managed_position is not None

    account = client.get_account()
    price = float(quotes.get(desired_symbol, 0.0) or 0.0)
    qty = compute_target_qty(account=account, price=price, allocation_pct=allocation_pct)
    if qty <= 0:
        logger.warning("Computed zero-sized order for %s at %.4f", desired_symbol, price)
        return False

    logger.info(
        "Opening managed position: %s qty=%.4f @ %.4f (alloc=%.1f%%)",
        desired_symbol,
        qty,
        price,
        allocation_pct,
    )
    if not dry_run:
        order = submit_market_order(client, symbol=desired_symbol, qty=qty, side="buy")
        state.last_order_id = str(getattr(order, "id", ""))
        state.active_symbol = desired_symbol
        state.active_qty = qty
        state.entry_price = price
        state.entry_date = now.astimezone(EASTERN).date().isoformat()
    return True


def build_portfolio_context(
    *,
    state: StrategyState,
    live_positions: dict[str, object],
    account,
    now: Optional[datetime] = None,
) -> PortfolioContext:
    now = now or datetime.now(timezone.utc)
    cash = float(
        getattr(account, "cash", 0.0)
        or getattr(account, "buying_power", 0.0)
        or getattr(account, "portfolio_value", 10_000.0)
        or 10_000.0
    )
    if not state.active_symbol:
        return PortfolioContext(cash=cash)

    symbol = state.active_symbol.upper()
    position = live_positions.get(symbol)
    if position is None:
        return PortfolioContext(cash=cash)

    hold_days = 0
    if state.entry_date:
        try:
            entry_day = datetime.fromisoformat(state.entry_date).date()
            hold_days = max(0, (now.astimezone(EASTERN).date() - entry_day).days)
        except ValueError:
            hold_days = 0

    entry_price = float(state.entry_price or getattr(position, "avg_entry_price", 0.0) or 0.0)
    return PortfolioContext(
        cash=cash,
        current_symbol=symbol,
        position_qty=abs(_signed_position_qty(position)),
        entry_price=entry_price,
        hold_days=hold_days,
    )


def adopt_existing_position(
    *,
    state: StrategyState,
    live_positions: dict[str, object],
    now: Optional[datetime] = None,
) -> bool:
    now = now or datetime.now(timezone.utc)
    if state.active_symbol or len(live_positions) != 1:
        return False
    symbol, position = next(iter(live_positions.items()))
    qty = abs(_signed_position_qty(position))
    if qty <= 0:
        return False
    state.active_symbol = symbol
    state.active_qty = qty
    state.entry_price = float(getattr(position, "avg_entry_price", 0.0) or 0.0)
    state.entry_date = now.astimezone(EASTERN).date().isoformat()
    logger.warning("Adopting pre-existing %s position into daily stock RL state", symbol)
    return True


def should_run_today(*, now: datetime, is_market_open: bool, last_run_date: Optional[str]) -> bool:
    if not is_market_open:
        return False
    now_et = now.astimezone(EASTERN)
    if now_et.weekday() >= 5:
        return False
    if last_run_date == now_et.date().isoformat():
        return False
    return now_et.timetz().replace(tzinfo=None) >= RUN_AFTER_OPEN_ET


def seconds_until_next_check(*, now: datetime, is_market_open: bool, next_open: Optional[datetime]) -> float:
    now_et = now.astimezone(EASTERN)
    if is_market_open and now_et.timetz().replace(tzinfo=None) < RUN_AFTER_OPEN_ET:
        target_et = datetime.combine(now_et.date(), RUN_AFTER_OPEN_ET, tzinfo=EASTERN)
        return max(30.0, (target_et - now_et).total_seconds())
    if next_open is not None:
        target_ts = pd.Timestamp(next_open)
        if target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize("UTC")
        else:
            target_ts = target_ts.tz_convert("UTC")
        target = target_ts.to_pydatetime()
        target += timedelta(minutes=5)
        return max(60.0, (target - now).total_seconds())
    return 300.0 if is_market_open else 900.0


def _signal_payload(signal, *, checkpoint: str, quotes: dict[str, float]) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "action": signal.action,
        "symbol": signal.symbol,
        "direction": signal.direction,
        "confidence": float(signal.confidence),
        "value_estimate": float(signal.value_estimate),
        "quotes": {symbol: float(price) for symbol, price in quotes.items()},
    }


def run_backtest(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    data_dir: str,
    days: int,
) -> dict[str, float]:
    frames = load_local_daily_frames(symbols, data_dir=data_dir, min_days=days + 120)
    indexed = {
        symbol: frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
        for symbol, frame in frames.items()
    }
    min_len = min(len(frame) for frame in indexed.values())
    start = min_len - days
    if start < 1:
        raise ValueError(f"Need at least {days + 1} aligned days for backtest")

    trader = DailyPPOTrader(checkpoint, device="cpu", long_only=True, symbols=list(indexed.keys()))
    cash = 10_000.0
    position: Optional[tuple[str, float, float]] = None
    equity_curve: list[float] = []
    trades = 0

    for idx in range(start, min_len):
        prices = {symbol: float(frame["close"].iloc[idx]) for symbol, frame in indexed.items()}
        trader.cash = cash
        trader.current_position = None
        trader.position_qty = 0.0
        trader.entry_price = 0.0
        if position is not None:
            pos_symbol, qty, entry_price = position
            trader.current_position = trader.SYMBOLS.index(pos_symbol)
            trader.position_qty = qty
            trader.entry_price = entry_price
        signal = trader.get_daily_signal(
            {symbol: frame.iloc[: idx + 1] for symbol, frame in indexed.items()},
            prices,
        )

        equity = cash
        if position is not None:
            pos_symbol, qty, _ = position
            equity += qty * prices[pos_symbol]
        equity_curve.append(equity)

        if position is not None and (signal.symbol != position[0] or signal.direction != "long"):
            pos_symbol, qty, _ = position
            cash += qty * prices[pos_symbol]
            position = None
            trades += 1
            trader.update_state(0, 0.0, "")

        if position is None and signal.symbol and signal.direction == "long":
            qty = cash / prices[signal.symbol]
            cash -= qty * prices[signal.symbol]
            position = (signal.symbol, qty, prices[signal.symbol])
            trader.update_state(trader.SYMBOLS.index(signal.symbol) + 1, prices[signal.symbol], signal.symbol)

        trader.step_day()

    if position is not None:
        equity_curve.append(cash + position[1] * indexed[position[0]]["close"].iloc[min_len - 1])
    else:
        equity_curve.append(cash)

    curve = np.asarray(equity_curve, dtype=np.float64)
    total_return = float(curve[-1] / curve[0] - 1.0)
    daily_returns = np.diff(curve) / np.clip(curve[:-1], 1e-8, None)
    downside = daily_returns[daily_returns < 0.0]
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 1e-8
    sortino = float(np.mean(daily_returns) / downside_dev * np.sqrt(252.0)) if len(daily_returns) else 0.0
    max_dd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
    annualized = float((1.0 + total_return) ** (252.0 / max(1, days)) - 1.0)

    results = {
        "total_return": total_return,
        "annualized_return": annualized,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "trades": float(trades),
    }
    logger.info("Backtest results: %s", json.dumps(results, sort_keys=True))
    return results


def run_once(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    paper: bool,
    allocation_pct: float,
    dry_run: bool,
    data_source: str,
    data_dir: str,
    state_path: Path = STATE_PATH,
    extra_checkpoints: Optional[list] = None,
) -> dict:
    now = datetime.now(timezone.utc)
    state = load_state(state_path)
    quote_data_source = "local"
    quote_source_by_symbol: dict[str, str] = {}
    latest_bar = None
    market_open = None
    if data_source == "alpaca":
        client = build_trading_client(paper=paper)
        data_client = build_data_client(paper=paper)
        frames, bar_data_source = load_inference_frames(
            symbols,
            paper=paper,
            data_dir=data_dir,
            now=now,
            data_client=data_client,
        )
        close_prices = latest_close_prices(frames)
        latest_bar = latest_bar_timestamp(frames)
        quotes, quote_data_source, quote_source_by_symbol = load_latest_quotes_with_source(
            symbols,
            paper=paper,
            fallback_prices=close_prices,
            data_client=data_client,
        )
        live_positions = positions_by_symbol(client, symbols)
        if not dry_run:
            adopt_existing_position(state=state, live_positions=live_positions, now=now)
        portfolio = build_portfolio_context(
            state=state,
            live_positions=live_positions,
            account=client.get_account(),
            now=now,
        )
        try:
            market_open = bool(getattr(client.get_clock(), "is_open", False))
        except Exception as exc:
            logger.warning("Could not read Alpaca market clock: %s", exc)
            market_open = False
    else:
        frames = load_local_daily_frames(symbols, data_dir=data_dir)
        quotes = latest_close_prices(frames)
        portfolio = PortfolioContext()
        bar_data_source = "local"
        latest_bar = latest_bar_timestamp(frames)

    signal, close_prices = build_signal(checkpoint, frames, portfolio=portfolio, extra_checkpoints=extra_checkpoints)
    bars_fresh = bars_are_fresh(latest_bar=latest_bar, now=now) if latest_bar is not None else False

    payload = _signal_payload(signal, checkpoint=checkpoint, quotes=quotes)
    payload["close_prices"] = close_prices
    payload["bar_data_source"] = bar_data_source
    payload["quote_data_source"] = quote_data_source
    payload["quote_source_by_symbol"] = dict(quote_source_by_symbol)
    payload["latest_bar_timestamp"] = latest_bar.isoformat() if latest_bar is not None else None
    payload["bars_fresh"] = bars_fresh
    append_signal_log(payload)

    logger.info("%s", "=" * 60)
    logger.info("DAILY STOCK RL SIGNAL (%s)", now.strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("%s", "=" * 60)
    logger.info("Action:     %s", signal.action)
    logger.info("Symbol:     %s", signal.symbol or "N/A")
    logger.info("Direction:  %s", signal.direction or "N/A")
    logger.info("Confidence: %.1f%%", float(signal.confidence) * 100.0)
    logger.info("Value est:  %.4f", float(signal.value_estimate))
    logger.info("Bars:       %s latest=%s fresh=%s", bar_data_source, latest_bar.isoformat() if latest_bar is not None else "n/a", bars_fresh)
    logger.info("Quotes:     %s", quote_data_source)
    if quote_source_by_symbol:
        fallback_symbols = sorted(symbol for symbol, source in quote_source_by_symbol.items() if source != "alpaca")
        if fallback_symbols:
            logger.info("Quote fallbacks: %s", ", ".join(fallback_symbols))

    executed = False
    if data_source == "alpaca":
        allow_open = (
            not signal.symbol
            or quote_source_by_symbol.get(signal.symbol, quote_data_source) == "alpaca"
        )
        if not dry_run and not bool(market_open):
            logger.warning("Market is closed; skipping order placement")
        elif not dry_run and not bars_fresh:
            logger.warning("Latest inference bar is stale; skipping order placement")
        else:
            executed = execute_signal(
                signal,
                client=client,
                quotes=quotes,
                state=state,
                symbols=symbols,
                allocation_pct=allocation_pct,
                dry_run=dry_run,
                now=now,
                allow_open=allow_open,
            )
    else:
        logger.info("Local data mode selected; skipping execution")

    should_advance_state = (
        not dry_run
        and (data_source != "alpaca" or bool(market_open))
    )
    if should_advance_state:
        state.last_run_date = now.astimezone(EASTERN).date().isoformat()
        state.last_signal_action = signal.action
        state.last_signal_timestamp = now.isoformat()
        save_state(state, path=state_path)
    payload["executed"] = executed
    logger.info("%s", "=" * 60)
    return payload


def run_daemon(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    paper: bool,
    allocation_pct: float,
    dry_run: bool,
    data_dir: str,
    extra_checkpoints: Optional[list] = None,
) -> None:
    logger.info("Starting daily stock RL daemon")
    while True:
        state = load_state()
        client = build_trading_client(paper=paper)
        clock = client.get_clock()
        now = datetime.now(timezone.utc)
        if should_run_today(
            now=now,
            is_market_open=bool(getattr(clock, "is_open", False)),
            last_run_date=state.last_run_date,
        ):
            try:
                run_once(
                    checkpoint=checkpoint,
                    symbols=symbols,
                    paper=paper,
                    allocation_pct=allocation_pct,
                    dry_run=dry_run,
                    data_source="alpaca",
                    data_dir=data_dir,
                    extra_checkpoints=extra_checkpoints,
                )
            except Exception as exc:
                logger.exception("Daily stock RL cycle failed: %s", exc)
            time.sleep(60.0)
            continue

        sleep_seconds = seconds_until_next_check(
            now=now,
            is_market_open=bool(getattr(clock, "is_open", False)),
            next_open=getattr(clock, "next_open", None),
        )
        logger.info("Sleeping %.1f minutes", sleep_seconds / 60.0)
        time.sleep(sleep_seconds)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production daily stock RL trader")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--extra-checkpoints", nargs="*", default=None,
                        help="Additional checkpoints for ensemble (default: DEFAULT_EXTRA_CHECKPOINTS)")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Disable ensemble, use --checkpoint alone")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--data-source", choices=["alpaca", "local"], default="alpaca")
    parser.add_argument("--allocation-pct", type=float, default=DEFAULT_ALLOCATION_PCT)
    parser.add_argument("--once", action="store_true", help="Run one inference cycle")
    parser.add_argument("--daemon", action="store_true", help="Run as a daemon around market open")
    parser.add_argument("--dry-run", action="store_true", help="Print signals without placing orders")
    parser.add_argument("--paper", action="store_true", default=True, help="Use the Alpaca paper account")
    parser.add_argument("--live", action="store_true", help="Use the Alpaca live account")
    parser.add_argument("--backtest", action="store_true", help="Run a local historical backtest")
    parser.add_argument("--backtest-days", type=int, default=60)
    parser.add_argument("--symbols", nargs="+", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    paper = not args.live
    symbols = [str(symbol).upper() for symbol in (args.symbols or DEFAULT_SYMBOLS)]
    checkpoint = str((REPO / args.checkpoint).resolve()) if not Path(args.checkpoint).is_absolute() else args.checkpoint

    def _resolve(p: str) -> str:
        return str((REPO / p).resolve()) if not Path(p).is_absolute() else p

    if args.no_ensemble:
        extra_checkpoints: Optional[list] = None
    elif args.extra_checkpoints is not None:
        extra_checkpoints = [_resolve(p) for p in args.extra_checkpoints]
    else:
        extra_checkpoints = [_resolve(p) for p in DEFAULT_EXTRA_CHECKPOINTS]

    if args.backtest:
        run_backtest(
            checkpoint=checkpoint,
            symbols=symbols,
            data_dir=args.data_dir,
            days=args.backtest_days,
        )
        return

    if args.daemon:
        run_daemon(
            checkpoint=checkpoint,
            symbols=symbols,
            paper=paper,
            allocation_pct=args.allocation_pct,
            dry_run=args.dry_run,
            data_dir=args.data_dir,
            extra_checkpoints=extra_checkpoints,
        )
        return

    run_once(
        checkpoint=checkpoint,
        symbols=symbols,
        paper=paper,
        allocation_pct=args.allocation_pct,
        dry_run=args.dry_run,
        data_source=args.data_source,
        data_dir=args.data_dir,
        extra_checkpoints=extra_checkpoints,
    )


if __name__ == "__main__":
    main()
