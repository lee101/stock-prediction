from datetime import datetime

import pytz


def is_nyse_trading_day_ending():
    # Get current time in UTC
    now_utc = datetime.now(pytz.timezone('UTC'))

    # Convert to NYSE time
    now_nyse = now_utc.astimezone(pytz.timezone('America/New_York'))

    # Check if it's the end of the trading day
    return now_nyse.hour in [14, 15, 16, 17]  # NYSE closes at 16:00 EST
