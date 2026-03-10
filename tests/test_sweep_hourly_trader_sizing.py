from __future__ import annotations

import pandas as pd

from newnanoalpacahourlyexp.sweep_hourly_trader_sizing import _load_replay_inputs, _parse_bool_list


def test_parse_bool_list_supports_common_tokens() -> None:
    assert _parse_bool_list("true,false,1,0,yes,no") == [True, False, True, False, True, False]


def test_load_replay_inputs_normalizes_symbols_and_applies_subset(tmp_path) -> None:
    bars = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "symbol": "ETH/USD", "high": 1.0, "low": 1.0, "close": 1.0},
            {"timestamp": "2026-01-01T00:00:00Z", "symbol": "SOLUSD", "high": 1.0, "low": 1.0, "close": 1.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "symbol": "ETH/USD", "buy_price": 1.0, "sell_price": 2.0, "buy_amount": 10.0, "sell_amount": 0.0},
            {"timestamp": "2026-01-01T00:00:00Z", "symbol": "SOLUSD", "buy_price": 1.0, "sell_price": 2.0, "buy_amount": 10.0, "sell_amount": 0.0},
        ]
    )
    bars_path = tmp_path / "bars.csv"
    actions_path = tmp_path / "actions.csv"
    bars.to_csv(bars_path, index=False)
    actions.to_csv(actions_path, index=False)

    loaded_bars, loaded_actions, symbols = _load_replay_inputs(bars_path, actions_path, symbols=["ETHUSD"])

    assert symbols == ["ETHUSD"]
    assert loaded_bars["symbol"].tolist() == ["ETHUSD"]
    assert loaded_actions["symbol"].tolist() == ["ETHUSD"]
