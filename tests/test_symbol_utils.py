import pytest

from src.symbol_utils import is_crypto_symbol
from src import fees


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("BTCUSD", True),
        ("ETHUSD", True),
        ("AVAXUSD", True),
        ("BCHUSD", True),
        ("APTUSD", True),
        ("BTC/USDT", True),
        ("BTC-USDT", True),
        ("ETHUSDT", True),
        ("BTCFDUSD", True),
        ("BTC/FDUSD", True),
        ("AVAXFDUSD", True),
        ("BCHFDUSD", True),
        ("APTFDUSD", True),
        ("POLFDUSD", True),
        ("BTCU", True),
        ("UUSDT", True),
        ("AEURUSDT", True),
        ("SOLUSDT", True),
        ("AAPL", False),
        ("MSFT", False),
        ("U", False),
        ("MU", False),
        ("BTC/EUR", False),
    ],
)
def test_is_crypto_symbol_stable_quotes(symbol, expected):
    assert is_crypto_symbol(symbol) is expected


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("BTCUSD", True),
        ("BTCUSDT", True),
        ("BTC/USDT", True),
        ("BTCFDUSD", True),
        ("AVAXFDUSD", True),
        ("POLFDUSD", True),
        ("AEURUSDT", True),
        ("BTCU", True),
        ("USDT", True),
        ("AAPL", False),
        ("U", False),
        ("MU", False),
    ],
)
def test_fee_crypto_detection(symbol, expected):
    assert fees._is_crypto_symbol(symbol) is expected
