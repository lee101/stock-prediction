import pandas as pd
from datetime import timedelta
from freezegun import freeze_time
from unittest.mock import patch

import trade_stock_e2e
from fake_alpaca import fake_alpaca

DATA_FILE = "WIKI-AAPL.csv"


def fake_backtest_forecasts(symbol: str, num_simulations: int = 7):
    data = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values("Date")
    rows = []
    for i in range(num_simulations):
        if i + 8 >= len(data):
            break
        window = data.iloc[i:i+8]
        close = window.iloc[-2]["Close"]
        predicted_close = window.iloc[-1]["Close"]
        predicted_high = window.iloc[-1]["High"]
        predicted_low = window.iloc[-1]["Low"]
        ret = (window.iloc[-1]["Close"] - close) / close
        rows.append({
            "date": window.iloc[-2]["Date"],
            "close": close,
            "predicted_close": predicted_close,
            "predicted_high": predicted_high,
            "predicted_low": predicted_low,
            "simple_strategy_return": ret,
            "all_signals_strategy_return": ret / 2,
            "entry_takeprofit_return": ret / 2,
            "highlow_return": ret / 2,
        })
    return pd.DataFrame(rows)


def run_backtest(days: int = 14):
    symbols = ["AAPL"]
    previous_picks = {}
    all_results = {}
    for day in range(days):
        current_time = pd.Timestamp("2021-01-01") + timedelta(days=day)
        with freeze_time(current_time):
            with patch("trade_stock_e2e.backtest_forecasts", fake_backtest_forecasts), \
                 patch("trade_stock_e2e.alpaca_wrapper.get_all_positions", fake_alpaca.get_all_positions), \
                 patch("trade_stock_e2e.alpaca_wrapper.open_order_at_price_or_all", fake_alpaca.open_order_at_price_or_all), \
                 patch("trade_stock_e2e.alpaca_wrapper.total_buying_power", fake_alpaca.total_buying_power, create=True):
                all_results = trade_stock_e2e.analyze_symbols(symbols)
                current_picks = {s: d for s, d in all_results.items() if d["avg_return"] > 0}
                trade_stock_e2e.manage_positions(current_picks, previous_picks, all_results)
                previous_picks = trade_stock_e2e.manage_market_close(symbols, previous_picks, all_results)
    return fake_alpaca.get_all_positions()


if __name__ == "__main__":
    positions = run_backtest()
    for p in positions:
        print(f"{p.side} {p.qty} {p.symbol}")
