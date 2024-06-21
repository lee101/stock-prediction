from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

# keys required for stock historical data client
client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

# multi symbol request - single symbol is similar
multisymbol_request_params = StockLatestQuoteRequest(symbol_or_symbols=["SPY", "GLD", "TLT"])

latest_multisymbol_quotes = client.get_stock_latest_quote(multisymbol_request_params)

gld_latest_ask_price = latest_multisymbol_quotes["GLD"]
print(gld_latest_ask_price)
##
# symbol='GLD' timestamp=datetime.datetime(2022, 10, 21, 20, 13, 54, 490814, tzinfo=datetime.timezone.utc) ask_exchange='V' ask_price=154.14 ask_size=3.0 bid_exchange='V' bid_price=154.05 bid_size=5.0 conditions=['R'] tape='B'
print(gld_latest_ask_price.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
