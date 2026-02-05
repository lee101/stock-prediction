from enum import Enum

from src.comparisons import is_buy_side, is_sell_side, normalize_side


class DummySide(Enum):
    LONG = "long"
    SHORT = "short"


def test_normalize_side_enum_value():
    assert normalize_side(DummySide.LONG) == "long"
    assert normalize_side(DummySide.SHORT) == "short"


def test_normalize_side_dotted_strings():
    assert normalize_side("OrderSide.SELL") == "sell"
    assert normalize_side("PositionSide.LONG") == "long"
    assert is_sell_side("orderside.sell")
    assert is_buy_side("positionside.long")


def test_buy_sell_side_enum():
    assert is_buy_side(DummySide.LONG)
    assert is_sell_side(DummySide.SHORT)
