#!/usr/bin/env python3
"""Validate which Binance symbols are cross-margin eligible."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance import Client
from src.binan.binance_wrapper import _resolve_client
from src.binan.binance_margin import get_max_borrowable, get_margin_interest_rate

TARGET_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "UNIUSDT", "AAVEUSDT", "ATOMUSDT", "APTUSDT",
    "TRXUSDT", "SHIBUSDT", "BCHUSDT", "SUIUSDT", "NEARUSDT",
]


def main():
    client = _resolve_client()
    # get all cross-margin assets
    account = client.get_margin_account()
    margin_assets = {a["asset"] for a in account.get("userAssets", [])}

    base_assets = [p.replace("USDT", "") for p in TARGET_PAIRS]
    rates = get_margin_interest_rate(base_assets, client=client)

    print(f"{'Pair':<12} {'Base':<6} {'Margin?':<8} {'Borrowable':<12} {'HourlyRate':<12} {'AnnualRate':<10}")
    print("-" * 65)

    eligible = []
    for pair in TARGET_PAIRS:
        base = pair.replace("USDT", "")
        in_margin = base in margin_assets
        borrowable = get_max_borrowable(base, client=client) if in_margin else 0.0
        rate = rates.get(base, 0.0)
        annual = rate * 24 * 365 * 100

        status = "YES" if in_margin and borrowable > 0 else "NO"
        print(f"{pair:<12} {base:<6} {status:<8} {borrowable:<12.4f} {rate:<12.10f} {annual:<10.2f}%")

        if in_margin and borrowable > 0:
            eligible.append({
                "pair": pair,
                "base": base,
                "max_borrowable": borrowable,
                "hourly_rate": rate,
            })

    print(f"\n{len(eligible)}/{len(TARGET_PAIRS)} pairs are margin-eligible")
    for e in eligible:
        print(f"  {e['pair']}")

    return eligible


if __name__ == "__main__":
    main()
