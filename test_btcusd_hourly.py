#!/usr/bin/env python3
"""Quick test to fetch latest BTCUSD hourly data and current price."""

from datetime import datetime, timezone, timedelta
from alpaca.data import TimeFrame, TimeFrameUnit
from alpaca_wrapper import download_symbol_history, latest_data

# Get last 4 hourly bars for BTCUSD
symbol = "BTCUSD"
end = datetime.now(timezone.utc)
start = end - timedelta(hours=10)  # Get a bit extra to ensure we have 4 bars

print(f"Fetching hourly BTCUSD data (include_latest=True)...")
print(f"Start: {start}")
print(f"End: {end}")
print()

df = download_symbol_history(
    symbol=symbol,
    start=start,
    end=end,
    include_latest=True,  # This should update the last bar with current pricing
    timeframe=TimeFrame(1, TimeFrameUnit.Hour)
)

print(f"Total bars retrieved: {len(df)}")
print()
print("Latest 4 bars:")
print("=" * 100)
print(df.tail(4).to_string())
print("=" * 100)
print()

# Get the latest quote directly
print("Latest BTCUSD quote:")
print("-" * 100)
quote = latest_data(symbol)
print(f"Ask Price: ${quote.ask_price}")
print(f"Bid Price: ${quote.bid_price}")
print(f"Mid Price: ${(quote.ask_price + quote.bid_price) / 2.0}")
print(f"Timestamp: {quote.timestamp}")
print("-" * 100)
