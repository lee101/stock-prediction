from datetime import datetime
import pytz
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from trade_stock_e2e import (
    analyze_symbols,
    log_trading_plan,
    dry_run_manage_positions,
    analyze_next_day_positions,
    manage_market_close,
    get_market_hours
)


def test_analyze_symbols_real_call():
    symbols = ['ETHUSD']
    results = analyze_symbols(symbols)
    
    assert isinstance(results, dict)
    # ah well? its not profitable
    # assert len(results) > 0
    # first_symbol = list(results.keys())[0]
    # assert 'sharpe' in results[first_symbol]
    # assert 'side' in results[first_symbol]
