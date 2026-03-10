from __future__ import annotations

import json

from newnanoalpacahourlyexp.run_hourly_trader_sim import _load_initial_state


def test_load_initial_state_supports_alias_fields(tmp_path) -> None:
    path = tmp_path / "initial_state.json"
    path.write_text(
        json.dumps(
            {
                "cash": 47000.5,
                "positions": [
                    {"symbol": "ETH/USD", "quantity": "0.25"},
                ],
                "open_orders": [
                    {
                        "symbol": "ETH/USD",
                        "side": "buy",
                        "quantity": "6.1",
                        "price": "1928.73",
                        "kind": "entry",
                        "created_at": "2026-03-05T17:00:07Z",
                    }
                ],
            }
        )
    )

    initial_cash, positions, open_orders = _load_initial_state(path)

    assert initial_cash == 47000.5
    assert positions == {"ETHUSD": 0.25}
    assert len(open_orders) == 1
    assert open_orders[0].symbol == "ETHUSD"
    assert open_orders[0].qty == 6.1
    assert open_orders[0].limit_price == 1928.73
    assert open_orders[0].placed_at.isoformat() == "2026-03-05T17:00:07+00:00"
