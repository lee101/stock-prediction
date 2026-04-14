#!/usr/bin/env python3
"""
Daily crypto30 RL ensemble trading bot for Binance.

Loads 4-seed PPO ensemble, fetches daily OHLCV for 30 crypto symbols,
runs softmax-average ensemble inference, places spot orders once per day.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.inference_daily import DailyPPOTrader, compute_daily_features
from pufferlib_market.inference import TradingSignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("crypto30_daily")

SYMBOLS_30 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "AAVEUSDT", "LTCUSDT", "XRPUSDT", "DOTUSDT",
    "UNIUSDT", "NEARUSDT", "APTUSDT", "ICPUSDT", "SHIBUSDT",
    "ADAUSDT", "FILUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "SUIUSDT", "TIAUSDT", "SEIUSDT", "ATOMUSDT", "ALGOUSDT",
    "BCHUSDT", "BNBUSDT", "TRXUSDT", "PEPEUSDT", "POLUSDT",
]

# Internal symbol names (must match training data order)
# Note: MATIC was renamed to POL on Binance but training data used MATIC
INTERNAL_SYMBOLS = [s.replace("USDT", "USD") for s in SYMBOLS_30]
# Map POL back to MATIC for model compatibility
_RENAME = {"POLUSD": "MATICUSD"}
INTERNAL_SYMBOLS = [_RENAME.get(s, s) for s in INTERNAL_SYMBOLS]

DEFAULT_CHECKPOINTS = [
    "pufferlib_market/checkpoints/crypto30_ensemble/s2.pt",
    "pufferlib_market/checkpoints/crypto30_ensemble/s19.pt",
    "pufferlib_market/checkpoints/crypto30_ensemble/s21.pt",
    "pufferlib_market/checkpoints/crypto30_ensemble/s23.pt",
]

STATE_DIR = REPO / "strategy_state" / "crypto30_daily"
STATE_FILE = STATE_DIR / "state.json"
SIGNAL_LOG = STATE_DIR / "signals.jsonl"

KLINES_URL = "https://api.binance.com/api/v3/klines"
LOOKBACK_DAYS = 90

# Reverse rename: internal MATICUSD -> POLUSDT for Binance API
_INTERNAL_TO_BINANCE = {"MATICUSD": "POLUSDT"}


def to_binance_symbol(internal: str) -> str:
    if internal in _INTERNAL_TO_BINANCE:
        return _INTERNAL_TO_BINANCE[internal]
    return internal.replace("USD", "USDT")


@dataclass
class PortfolioState:
    cash_usd: float = 10000.0
    position_symbol: Optional[str] = None
    position_qty: float = 0.0
    entry_price: float = 0.0
    hold_days: int = 0
    episode_step: int = 0
    total_value: float = 10000.0


def fetch_daily_klines(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch daily OHLCV from Binance REST API."""
    import requests
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    all_klines = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": symbol, "interval": "1d",
            "startTime": current, "endTime": end_ms, "limit": 1000,
        }
        for attempt in range(5):
            try:
                resp = requests.get(KLINES_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(int(resp.headers.get("Retry-After", 10)))
                    continue
                resp.raise_for_status()
                break
            except Exception:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise
        data = resp.json()
        if not data:
            break
        all_klines.extend(data)
        current = data[-1][0] + 86400000
    if not all_klines:
        return pd.DataFrame()
    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


class Crypto30Ensemble:
    """4-model softmax-average ensemble for crypto30 daily trading."""

    def __init__(self, checkpoint_paths: list[str], device: str = "cpu", regime_ma_period: int = 15):
        self.device = torch.device(device)
        self.symbols = INTERNAL_SYMBOLS
        self.regime_ma_period = regime_ma_period
        self.btc_symbol = "BTCUSD"
        self.num_symbols = len(self.symbols)

        self.traders = []
        for path in checkpoint_paths:
            trader = DailyPPOTrader(
                path, device=device, long_only=True,
                symbols=self.symbols,
                allow_unsafe_checkpoint_loading=True,
            )
            self.traders.append(trader)
            log.info("loaded %s (arch=%s h=%d actions=%d)",
                     Path(path).name, trader.arch, trader.hidden_size, trader.num_actions)

        self.features_per_sym = self.traders[0].features_per_sym
        self.num_actions = self.traders[0].num_actions

    def is_bull_regime(self, daily_dfs: dict[str, pd.DataFrame]) -> bool:
        """Check if BTC is above its MA (bull regime). Returns True if no filter or bull."""
        if self.regime_ma_period <= 0:
            return True
        btc_df = daily_dfs.get(self.btc_symbol)
        if btc_df is None or len(btc_df) < self.regime_ma_period:
            return True  # not enough data, allow trading
        btc_close = btc_df["close"].values
        ma = float(np.mean(btc_close[-self.regime_ma_period:]))
        current = float(btc_close[-1])
        is_bull = current > ma
        log.info("BTC regime: close=%.2f MA%d=%.2f %s",
                 current, self.regime_ma_period, ma, "BULL" if is_bull else "BEAR")
        return is_bull

    def get_ensemble_signal(
        self,
        daily_dfs: dict[str, pd.DataFrame],
        prices: dict[str, float],
        portfolio: PortfolioState,
    ) -> TradingSignal:
        """Run ensemble inference with softmax-average. Respects BTC regime filter."""
        if not self.is_bull_regime(daily_dfs):
            return TradingSignal("flat", None, None, 0.0, 0.0)

        fps = self.features_per_sym
        features = np.zeros((self.num_symbols, fps), dtype=np.float32)
        for i, sym in enumerate(self.symbols):
            if sym in daily_dfs and len(daily_dfs[sym]) >= 60:
                features[i, :16] = compute_daily_features(daily_dfs[sym])

        # Build observation (must match C sim layout)
        obs = np.zeros(self.num_symbols * fps + 5 + self.num_symbols, dtype=np.float32)
        obs[:self.num_symbols * fps] = features.flatten()

        base = self.num_symbols * fps
        scale = portfolio.total_value if portfolio.total_value > 0 else 10000.0
        obs[base + 0] = portfolio.cash_usd / scale
        pos_val = portfolio.position_qty * prices.get(portfolio.position_symbol, 0.0) if portfolio.position_symbol else 0.0
        obs[base + 1] = pos_val / scale
        pnl = (pos_val - portfolio.position_qty * portfolio.entry_price) if portfolio.position_symbol else 0.0
        obs[base + 2] = pnl / scale
        obs[base + 3] = portfolio.hold_days / 365.0
        obs[base + 4] = portfolio.episode_step / 365.0

        if portfolio.position_symbol and portfolio.position_symbol in self.symbols:
            idx = self.symbols.index(portfolio.position_symbol)
            obs[base + 5 + idx] = 1.0

        obs_t = torch.from_numpy(obs).to(self.device).unsqueeze(0)

        # Ensemble: softmax-average
        probs_sum = None
        values = []
        with torch.no_grad():
            for trader in self.traders:
                logits, value = trader.policy(obs_t)
                probs = F.softmax(logits, dim=-1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
                values.append(value.item())

        avg_probs = probs_sum / len(self.traders)
        avg_logits = torch.log(avg_probs + 1e-8)

        # Mask shorts (actions > 1 + num_symbols)
        short_start = 1 + self.num_symbols
        if avg_logits.shape[-1] > short_start:
            avg_logits[0, short_start:] = float("-inf")

        action = int(avg_logits.argmax(dim=-1).item())
        confidence = float(avg_probs[0, action].item())
        avg_value = float(np.mean(values))

        if action == 0:
            return TradingSignal("flat", None, None, confidence, avg_value)
        elif action <= self.num_symbols:
            sym = self.symbols[action - 1]
            return TradingSignal(f"long_{sym}", sym, "long", confidence, avg_value)
        else:
            sym = self.symbols[action - 1 - self.num_symbols]
            return TradingSignal(f"short_{sym}", sym, "short", confidence, avg_value)


def load_state() -> PortfolioState:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            d = json.load(f)
        return PortfolioState(**{k: d[k] for k in PortfolioState.__dataclass_fields__ if k in d})
    return PortfolioState()


def save_state(state: PortfolioState):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(asdict(state), f, indent=2)


def log_signal(signal: TradingSignal, state: PortfolioState, prices: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": signal.action,
        "symbol": signal.symbol,
        "direction": signal.direction,
        "confidence": round(signal.confidence, 4),
        "value_estimate": round(signal.value_estimate, 4),
        "portfolio_value": round(state.total_value, 2),
        "cash": round(state.cash_usd, 2),
        "position": state.position_symbol,
    }
    with open(SIGNAL_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")


def execute_binance_order(signal: TradingSignal, state: PortfolioState, prices: dict, dry_run: bool = True):
    """Execute order on Binance. dry_run=True for paper trading."""
    if signal.action == "flat":
        if state.position_symbol:
            binance_sym = to_binance_symbol(state.position_symbol)
            price = prices.get(state.position_symbol, 0.0)
            value = state.position_qty * price
            log.info("SELL %s qty=%.6f price=%.2f value=%.2f",
                     binance_sym, state.position_qty, price, value)
            if not dry_run:
                _place_market_order(binance_sym, "SELL", state.position_qty)
            state.cash_usd += value
            state.position_symbol = None
            state.position_qty = 0.0
            state.entry_price = 0.0
            state.hold_days = 0
    elif signal.direction == "long" and signal.symbol:
        # Close existing position first if different symbol
        if state.position_symbol and state.position_symbol != signal.symbol:
            old_price = prices.get(state.position_symbol, 0.0)
            old_value = state.position_qty * old_price
            old_binance = to_binance_symbol(state.position_symbol)
            log.info("SELL %s (rotate) qty=%.6f price=%.2f", old_binance, state.position_qty, old_price)
            if not dry_run:
                _place_market_order(old_binance, "SELL", state.position_qty)
            state.cash_usd += old_value
            state.position_symbol = None
            state.position_qty = 0.0
            state.entry_price = 0.0
            state.hold_days = 0

        if state.position_symbol is None:
            price = prices.get(signal.symbol, 0.0)
            if price > 0:
                alloc = state.cash_usd * 0.95  # keep 5% cash buffer
                qty = alloc / price
                binance_sym = to_binance_symbol(signal.symbol)
                log.info("BUY %s qty=%.6f price=%.2f alloc=%.2f",
                         binance_sym, qty, price, alloc)
                if not dry_run:
                    _place_market_order(binance_sym, "BUY", qty)
                state.position_symbol = signal.symbol
                state.position_qty = qty
                state.entry_price = price
                state.cash_usd -= alloc
                state.hold_days = 0


def _place_market_order(symbol: str, side: str, quantity: float):
    """Place market order on Binance."""
    from binance.client import Client
    try:
        from env_real import BINANCE_API_KEY, BINANCE_SECRET
        api_key, api_secret = BINANCE_API_KEY, BINANCE_SECRET
    except ImportError:
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
    client = Client(api_key, api_secret)

    # Get symbol info for precision
    info = client.get_symbol_info(symbol)
    step_size = None
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
            break
    if step_size and step_size > 0:
        precision = max(0, int(round(-np.log10(step_size))))
        quantity = round(quantity, precision)

    order = client.create_order(
        symbol=symbol, side=side, type="MARKET",
        quantity=quantity,
    )
    log.info("order filled: %s", json.dumps(order, default=str))
    return order


def run_once(ensemble: Crypto30Ensemble, dry_run: bool = True):
    """Run one inference cycle."""
    state = load_state()
    log.info("portfolio: cash=%.2f pos=%s qty=%.6f hold=%dd step=%d",
             state.cash_usd, state.position_symbol, state.position_qty,
             state.hold_days, state.episode_step)

    # Fetch daily data for all 30 symbols
    log.info("fetching daily klines for %d symbols...", len(SYMBOLS_30))
    daily_dfs = {}
    prices = {}
    for binance_sym, internal_sym in zip(SYMBOLS_30, INTERNAL_SYMBOLS):
        try:
            df = fetch_daily_klines(binance_sym, days=LOOKBACK_DAYS)
            if len(df) >= 60:
                daily_dfs[internal_sym] = df
                prices[internal_sym] = float(df["close"].iloc[-1])
            else:
                log.warning("%s: only %d bars (need 60+)", binance_sym, len(df))
        except Exception as e:
            log.error("failed to fetch %s: %s", binance_sym, e)
    log.info("fetched %d/%d symbols", len(daily_dfs), len(SYMBOLS_30))

    # Update portfolio value
    if state.position_symbol and state.position_symbol in prices:
        pos_val = state.position_qty * prices[state.position_symbol]
        state.total_value = state.cash_usd + pos_val
    else:
        state.total_value = state.cash_usd

    # Get ensemble signal (includes BTC regime filter)
    signal = ensemble.get_ensemble_signal(daily_dfs, prices, state)
    log.info("signal: %s conf=%.4f value=%.4f", signal.action, signal.confidence, signal.value_estimate)

    log_signal(signal, state, prices)

    # Execute
    execute_binance_order(signal, state, prices, dry_run=dry_run)

    # Advance state
    state.episode_step += 1
    if state.position_symbol:
        state.hold_days += 1
    if state.position_symbol and state.position_symbol in prices:
        state.total_value = state.cash_usd + state.position_qty * prices[state.position_symbol]
    else:
        state.total_value = state.cash_usd

    save_state(state)
    log.info("portfolio after: value=%.2f cash=%.2f pos=%s",
             state.total_value, state.cash_usd, state.position_symbol)


def run_daemon(ensemble: Crypto30Ensemble, dry_run: bool = True, interval_hours: float = 24.0):
    """Run continuously, trading once per interval."""
    log.info("starting daemon (interval=%.1fh, dry_run=%s)", interval_hours, dry_run)
    while True:
        try:
            now = datetime.now(timezone.utc)
            log.info("=== %s ===", now.strftime("%Y-%m-%d %H:%M UTC"))
            run_once(ensemble, dry_run=dry_run)
        except Exception as e:
            log.error("run failed: %s", e, exc_info=True)
        time.sleep(interval_hours * 3600)


def run_backtest(ensemble: Crypto30Ensemble, days: int = 90, fee_rate: float = 0.001):
    """Backtest on historical daily data from Binance."""
    log.info("fetching historical data for backtest (%d days)...", days)
    all_dfs = {}
    for binance_sym, internal_sym in zip(SYMBOLS_30, INTERNAL_SYMBOLS):
        try:
            df = fetch_daily_klines(binance_sym, days=days + LOOKBACK_DAYS)
            if len(df) >= 60:
                all_dfs[internal_sym] = df
        except Exception as e:
            log.error("failed to fetch %s: %s", binance_sym, e)
    log.info("fetched %d/%d symbols", len(all_dfs), len(SYMBOLS_30))

    min_len = min(len(df) for df in all_dfs.values())
    test_start = max(60, min_len - days)
    log.info("backtest window: %d days (from idx %d to %d)", min_len - test_start, test_start, min_len)

    state = PortfolioState()
    equity_curve = []

    for t in range(test_start, min_len):
        daily_dfs = {}
        prices = {}
        for sym, df in all_dfs.items():
            window = df.iloc[:t + 1]
            if len(window) >= 60:
                daily_dfs[sym] = window
                prices[sym] = float(window["close"].iloc[-1])

        if state.position_symbol and state.position_symbol in prices:
            pos_val = state.position_qty * prices[state.position_symbol]
            state.total_value = state.cash_usd + pos_val

        signal = ensemble.get_ensemble_signal(daily_dfs, prices, state)

        old_pos = state.position_symbol
        execute_binance_order(signal, state, prices, dry_run=True)

        # Apply fee on trades
        if old_pos != state.position_symbol:
            if old_pos is not None:
                state.cash_usd *= (1 - fee_rate)
            if state.position_symbol is not None:
                fee_cost = state.position_qty * state.entry_price * fee_rate
                state.cash_usd -= fee_cost

        state.episode_step += 1
        if state.position_symbol:
            state.hold_days += 1
        if state.position_symbol and state.position_symbol in prices:
            state.total_value = state.cash_usd + state.position_qty * prices[state.position_symbol]
        else:
            state.total_value = state.cash_usd

        date_str = list(all_dfs.values())[0].index[t].strftime("%Y-%m-%d")
        equity_curve.append((date_str, state.total_value, state.position_symbol, signal.action))

    initial = 10000.0
    final = state.total_value
    ret_pct = (final / initial - 1) * 100
    log.info("=== BACKTEST RESULTS ===")
    log.info("period: %s to %s (%d days)", equity_curve[0][0], equity_curve[-1][0], len(equity_curve))
    log.info("initial: $%.2f  final: $%.2f  return: %.2f%%", initial, final, ret_pct)

    peak = initial
    max_dd = 0.0
    for _, val, _, _ in equity_curve:
        peak = max(peak, val)
        dd = (peak - val) / peak
        max_dd = max(max_dd, dd)
    log.info("max drawdown: %.2f%%", max_dd * 100)

    trades = sum(1 for i in range(1, len(equity_curve))
                 if equity_curve[i][2] != equity_curve[i - 1][2])
    log.info("trades: %d", trades)

    return equity_curve


def main():
    parser = argparse.ArgumentParser(description="Crypto30 Daily RL Ensemble Trader")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["once", "daemon", "backtest"], default="once")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="Disable dry-run (REAL MONEY)")
    parser.add_argument("--interval-hours", type=float, default=24.0)
    parser.add_argument("--backtest-days", type=int, default=90)
    parser.add_argument("--regime-ma", type=int, default=15, help="BTC MA period for regime filter (0=disabled)")
    args = parser.parse_args()

    dry_run = not args.live

    ensemble = Crypto30Ensemble(args.checkpoints, device=args.device, regime_ma_period=args.regime_ma)
    log.info("ensemble loaded: %d models, %d symbols, %d actions",
             len(ensemble.traders), ensemble.num_symbols, ensemble.num_actions)

    if args.mode == "once":
        run_once(ensemble, dry_run=dry_run)
    elif args.mode == "daemon":
        run_daemon(ensemble, dry_run=dry_run, interval_hours=args.interval_hours)
    elif args.mode == "backtest":
        run_backtest(ensemble, days=args.backtest_days)


if __name__ == "__main__":
    main()
