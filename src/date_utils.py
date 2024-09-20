from datetime import datetime

import pytz


def is_nyse_trading_day_ending():
    # Get current time in UTC
    now_utc = datetime.now(pytz.timezone('UTC'))

    # Convert to NYSE time
    now_nyse = now_utc.astimezone(pytz.timezone('America/New_York'))

    # Check if it's the end of the trading day
    return now_nyse.hour in [14, 15, 16, 17]  # NYSE closes at 16:00 EST

def is_nyse_trading_day_now():
    # Get current time in UTC
    now_utc = datetime.now(pytz.timezone('UTC'))

    # Convert to NYSE time
    now_nyse = now_utc.astimezone(pytz.timezone('America/New_York'))

    # Check if it's a weekday (Monday = 0, Sunday = 6)
    if now_nyse.weekday() >= 5:
        return False

    # Check if it's during trading hours (9:30 AM to 4:00 PM)
    market_open = now_nyse.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_nyse.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_nyse <= market_close
