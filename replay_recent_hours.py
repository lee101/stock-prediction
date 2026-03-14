#!/usr/bin/env python3
"""Replay RL model on recent hours and compare to actual production decisions.

Pulls fresh bars from Alpaca, runs RL inference, and shows side-by-side
with what the orchestrator actually did.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from pufferlib_market.inference import PPOTrader, compute_hourly_features


CHECKPOINT = str(REPO / "pufferlib_market/checkpoints/autoresearch/longonly_forecast/best.pt")
SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
LOOKBACK_HOURS = 96  # need history for feature computation
DISPLAY_HOURS = 18   # show recent hours


def fetch_bars(symbols: list[str], hours: int) -> dict[str, pd.DataFrame]:
    """Fetch hourly bars from Alpaca."""
    client = CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)

    result = {}
    for sym in symbols:
        alpaca_sym = sym.replace("USD", "/USD")
        req = CryptoBarsRequest(symbol_or_symbols=alpaca_sym, timeframe=TimeFrame.Hour, start=start)
        bars = client.get_crypto_bars(req)
        rows = []
        for key in bars.data:
            for b in bars.data[key]:
                rows.append({
                    "timestamp": b.timestamp,
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                })
        if rows:
            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            result[sym] = df
    return result


def main():
    print("Fetching bars from Alpaca...")
    bars = fetch_bars(SYMBOLS, LOOKBACK_HOURS)

    print(f"\nLoading RL model: {CHECKPOINT}")
    trader = PPOTrader(CHECKPOINT, device="cpu", long_only=True, symbols=SYMBOLS)

    # Known prod actions from order log (filled only)
    prod_actions = {
        "2026-03-14 01:00": "SELL BTC 0.1395 @$70671, SELL SOL 112.09 @$87.68, SELL LTC @$55.26",
        "2026-03-14 02:00": "BUY AVAX 327.94 @$9.73, SELL ETH 1.448 @$2093.70",
        "2026-03-14 03:00": "tried buy SOL+AVAX — canceled",
        "2026-03-14 04:00": "tried buy ETH+LTC — all canceled",
        "2026-03-14 05:00": "tried buy SOL+ETH — all canceled",
        "2026-03-14 06:00": "BUY ETH 4.12 @$2088-89, BUY SOL 60.16 @$88.05",
        "2026-03-14 07:00": "SELL AVAX 327 @$9.58, then BUY AVAX 101.9 @$9.64",
    }

    # Run inference on each recent bar
    cutoff = datetime.now(timezone.utc) - timedelta(hours=DISPLAY_HOURS)

    print(f"\n{'='*100}")
    print(f"{'HOUR':>20s} | {'RL SIGNAL':>30s} | {'CONF':>5s} | {'PROD ACTION':<50s}")
    print(f"{'='*100}")

    for hour_offset in range(DISPLAY_HOURS, 0, -1):
        target_time = datetime.now(timezone.utc) - timedelta(hours=hour_offset)
        target_hour = target_time.replace(minute=0, second=0, microsecond=0)
        hour_str = target_hour.strftime("%Y-%m-%d %H:%M")

        # Build features for each symbol at this timestamp
        all_features = np.zeros((len(SYMBOLS), 16), dtype=np.float32)
        prices = {}
        valid = True

        for i, sym in enumerate(SYMBOLS):
            if sym not in bars:
                valid = False
                break
            df = bars[sym]
            # Find bars up to this hour
            mask = df["timestamp"] <= target_hour
            if mask.sum() < 24:
                valid = False
                break
            sub = df[mask].tail(96).reset_index(drop=True)
            features = compute_hourly_features(sub)
            all_features[i] = features
            prices[sym] = float(sub.iloc[-1]["close"])

        if not valid:
            continue

        signal = trader.get_signal(all_features, prices)

        # Look up prod action
        prod_key = hour_str
        prod = prod_actions.get(prod_key, "hold / TP adjustments")

        # Format signal
        if signal.symbol:
            sig_str = f"{signal.direction} {signal.symbol}"
        else:
            sig_str = "flat (no trade)"

        print(f"{hour_str:>20s} | {sig_str:>30s} | {signal.confidence:>5.1%} | {prod:<50s}")

        # Show prices
        btc_p = prices.get("BTCUSD", 0)
        eth_p = prices.get("ETHUSD", 0)
        sol_p = prices.get("SOLUSD", 0)
        ltc_p = prices.get("LTCUSD", 0)
        avax_p = prices.get("AVAXUSD", 0)
        # print price bar below
        # Show volume for context
        vol_line = []
        for sym in SYMBOLS:
            if sym in bars:
                df = bars[sym]
                mask = (df["timestamp"] == target_hour)
                if mask.any():
                    v = df.loc[mask, "volume"].iloc[0]
                    vol_line.append(f"{sym[:3]}:{v:.1f}")
        if vol_line:
            print(f"{'':>20s} | {'prices: BTC=' + f'${btc_p:.0f}':>30s} | {'':>5s} | vol: {', '.join(vol_line)}")
        print(f"{'':>20s} | {'ETH=$' + f'{eth_p:.0f} SOL=${sol_p:.1f} LTC=${ltc_p:.1f} AVAX=${avax_p:.2f}':>30s} |")
        print(f"{'-'*100}")

    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")

    # Show current positions value
    print("\nCurrent LIVE positions:")
    print("  AVAXUSD  101.73  @ $9.62   ($979)")
    print("  ETHUSD     4.12  @ $2080   ($8,554)")
    print("  SOLUSD    60.10  @ $87.47  ($5,262)")
    print("  BTC/LTC: flat")
    print(f"\n  Equity: $40,053 | Cash: $1,486")

    # Show volume concern
    print("\n⚠ VOLUME CONCERN:")
    for sym in SYMBOLS:
        if sym in bars:
            df = bars[sym]
            recent = df[df["timestamp"] >= cutoff]
            if len(recent) > 0:
                avg_vol = recent["volume"].mean()
                zero_hours = (recent["volume"] == 0).sum()
                print(f"  {sym}: avg hourly vol = {avg_vol:.2f}, zero-volume hours: {zero_hours}/{len(recent)}")


if __name__ == "__main__":
    main()
