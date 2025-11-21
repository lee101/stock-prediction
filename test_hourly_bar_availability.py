#!/usr/bin/env python3
"""Test when Alpaca makes the latest hourly crypto bar available."""

import sys
from datetime import datetime, timezone, timedelta
sys.path.insert(0, ".")

from alpaca_wrapper import download_symbol_history
from alpaca.data import TimeFrame, TimeFrameUnit

def check_latest_bar(symbol="BTCUSD"):
    """Check what the latest hourly bar timestamp is."""

    # Current time
    now = datetime.now(timezone.utc)

    # Expected latest completed hour
    # If it's 08:05:00, the latest completed hour should be 08:00:00
    # If it's 08:00:05, the latest completed hour should be 07:00:00
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    if now.minute == 0 and now.second < 60:
        # We're in the first minute of the hour, so the "completed" hour is the previous one
        expected_latest = current_hour - timedelta(hours=1)
    else:
        expected_latest = current_hour

    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Expected latest completed hour: {expected_latest.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()

    # Fetch the most recent bars
    df = download_symbol_history(
        symbol=symbol,
        start=now - timedelta(hours=3),
        end=now,
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        include_latest=True,
    )

    if df.empty:
        print("ERROR: No bars returned")
        return

    print(f"Retrieved {len(df)} bars:")

    # Handle both index and column timestamp
    if 'timestamp' in df.columns:
        for idx, row in df.tail(5).iterrows():
            print(f"  {row['timestamp']} | close={row['close']:.2f}")
        latest_timestamp = df.iloc[-1]['timestamp']
    else:
        # Timestamp is the index
        for idx in df.tail(5).index:
            row = df.loc[idx]
            print(f"  {idx} | close={row['close']:.2f}")
        latest_timestamp = df.index[-1]

    print()
    print(f"Latest bar timestamp: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Expected latest:      {expected_latest.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Calculate lag
    if latest_timestamp == expected_latest:
        print("✅ Latest bar is available immediately!")
    else:
        lag = expected_latest - latest_timestamp
        lag_hours = lag.total_seconds() / 3600
        print(f"❌ Latest bar is {lag_hours:.2f} hours behind (lag: {lag})")
        print(f"   This means the bar for the current hour is not yet available.")

    # Seconds since the hour
    seconds_since_hour = (now - current_hour).total_seconds()
    print()
    print(f"Current offset from hour boundary: {seconds_since_hour:.1f} seconds")

    if latest_timestamp == expected_latest and seconds_since_hour < 60:
        print(f"✅ Data was available within {seconds_since_hour:.1f} seconds of the hour")
    elif latest_timestamp < expected_latest:
        print("⚠️  Hourly bar not yet available - recommend waiting longer after the hour")
        print("   Suggestion: Run bot at :01:00 or :02:00 instead of :00:30")

if __name__ == "__main__":
    check_latest_bar()
