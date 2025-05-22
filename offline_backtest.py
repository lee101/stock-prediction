import pandas as pd
from datetime import timedelta
from freezegun import freeze_time
from unittest.mock import patch

import trade_stock_e2e
from fake_alpaca import fake_alpaca

DATA_FILE = "WIKI-AAPL.csv"

# Will hold the day's price for patch functions
CURRENT_ROW = {}


def fake_backtest_forecasts(symbol: str, num_simulations: int = 7):
    """Return a deterministic sliding window forecast using historical data."""
    data = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values("Date")
    rows = []
    for i in range(num_simulations):
        if i + 8 >= len(data):
            break
        window = data.iloc[i : i + 8]
        close = window.iloc[-2]["Close"]
        predicted_close = window.iloc[-1]["Close"]
        predicted_high = window.iloc[-1]["High"]
        predicted_low = window.iloc[-1]["Low"]
        ret = (window.iloc[-1]["Close"] - close) / close
        rows.append(
            {
                "date": window.iloc[-2]["Date"],
                "close": close,
                "predicted_close": predicted_close,
                "predicted_high": predicted_high,
                "predicted_low": predicted_low,
                "simple_strategy_return": ret,
                "all_signals_strategy_return": ret / 2,
                "entry_takeprofit_return": ret / 2,
                "highlow_return": ret / 2,
            }
        )
    return pd.DataFrame(rows)


def fake_ramp_into_position(symbol: str, side: str = "buy"):
    price = CURRENT_ROW.get("Close", 0)
    qty = trade_stock_e2e.get_qty(symbol, price)
    fake_alpaca.open_order_at_price_or_all(symbol, qty, side, price)


def fake_backout_near_market(symbol: str):
    price = CURRENT_ROW.get("Close", 0)
    fake_alpaca.close_position(symbol, price)


def fake_spawn_close_position_at_takeprofit(symbol: str, tp_price: float):
    fake_alpaca.close_position(symbol, tp_price)


def run_backtest(days: int = 14):
    data = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values("Date")
    symbols = ["AAPL"]
    previous_picks = {}
    all_results = {}

    for day in range(days):
        row = data.iloc[day]
        CURRENT_ROW.clear()
        CURRENT_ROW.update(row)
        trade_stock_e2e.alpaca_wrapper.total_buying_power = fake_alpaca.total_buying_power
        current_time = row["Date"]

        with freeze_time(current_time):
            with patch("trade_stock_e2e.backtest_forecasts", fake_backtest_forecasts), \
                patch("trade_stock_e2e.alpaca_wrapper.get_all_positions", fake_alpaca.get_all_positions), \
                patch("trade_stock_e2e.alpaca_wrapper.open_order_at_price_or_all", fake_alpaca.open_order_at_price_or_all), \
                patch("trade_stock_e2e.ramp_into_position", fake_ramp_into_position), \
                patch("trade_stock_e2e.backout_near_market", fake_backout_near_market), \
                patch(
                    "trade_stock_e2e.spawn_close_position_at_takeprofit",
                    fake_spawn_close_position_at_takeprofit,
                ):
                all_results = trade_stock_e2e.analyze_symbols(symbols)
                current_picks = {s: d for s, d in all_results.items() if d["avg_return"] > 0}
                trade_stock_e2e.manage_positions(current_picks, previous_picks, all_results)
                previous_picks = trade_stock_e2e.manage_market_close(symbols, previous_picks, all_results)

        trade_stock_e2e.alpaca_wrapper.total_buying_power = fake_alpaca.total_buying_power

    last_close = data.iloc[days - 1]["Close"]
    fake_alpaca.close_all_positions(last_close)
    return fake_alpaca.cash - fake_alpaca.starting_cash


if __name__ == "__main__":
    profit = run_backtest()
    print(f"Final cash: {fake_alpaca.cash:.2f}")
    print(f"Profit: {profit:.2f}")
