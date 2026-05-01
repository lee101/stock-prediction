from types import SimpleNamespace

import numpy as np

from pufferlib_market.hourly_replay import Position
from scripts.plotly_backtest_space import _action_payload, _decode_action, _default_symbol, _positions_by_bar_payload


def test_positions_by_bar_uses_position_history_for_held_spans():
    result = SimpleNamespace(
        position_history=[
            Position(sym=1, is_short=False, qty=2.5, entry_price=125.0),
            Position(sym=1, is_short=False, qty=2.5, entry_price=125.0),
            None,
        ]
    )

    rows = _positions_by_bar_payload(result, ["BTCUSD", "ETHUSD"], total_bars=4)

    assert rows[0] == []
    assert rows[1] == [
        {
            "bar_index": 1,
            "symbol": "ETHUSD",
            "side": "long",
            "qty": 2.5,
            "entry_price": 125.0,
            "weight": None,
        }
    ]
    assert rows[2][0]["symbol"] == "ETHUSD"
    assert rows[2][0]["side"] == "long"
    assert rows[3] == []


def test_decode_action_maps_flat_long_short_and_unknown():
    symbols = ["BTCUSD", "ETHUSD"]

    assert _decode_action(0, symbols) == {"symbol": "", "side": "flat", "symbol_index": -1}
    assert _decode_action(1, symbols) == {"symbol": "BTCUSD", "side": "long", "symbol_index": 0}
    assert _decode_action(2, symbols) == {"symbol": "ETHUSD", "side": "long", "symbol_index": 1}
    assert _decode_action(3, symbols) == {"symbol": "BTCUSD", "side": "short", "symbol_index": 0}
    assert _decode_action(4, symbols) == {"symbol": "ETHUSD", "side": "short", "symbol_index": 1}
    assert _decode_action(5, symbols) == {"symbol": "", "side": "unknown", "symbol_index": -1}


def test_action_payload_includes_decoded_symbol_and_side():
    result = SimpleNamespace(actions=np.asarray([0, 2, 4], dtype=np.int64))

    rows = _action_payload(result, ["d0", "d1", "d2"], ["BTCUSD", "ETHUSD"])

    assert rows == [
        {"bar_index": 0, "x": "d0", "action": 0, "symbol": "", "side": "flat", "symbol_index": -1},
        {"bar_index": 1, "x": "d1", "action": 2, "symbol": "ETHUSD", "side": "long", "symbol_index": 1},
        {"bar_index": 2, "x": "d2", "action": 4, "symbol": "ETHUSD", "side": "short", "symbol_index": 1},
    ]


def test_default_symbol_prefers_latest_held_position_over_first_trade():
    trade = SimpleNamespace(symbol="BCHUSD")
    result = SimpleNamespace(
        position_history=[
            Position(sym=2, is_short=False, qty=1.0, entry_price=10.0),
            Position(sym=1, is_short=False, qty=1.0, entry_price=20.0),
        ],
        trade_events=[trade],
    )

    assert _default_symbol(result, ["BCHUSD", "ETHUSD", "SOLUSD"]) == "ETHUSD"
