import json
from datetime import datetime, timedelta, timezone

from stockagent_pctline.data_formatter import PctLineData

from stockagent3 import agent as agent_module
from stockagent3.expiry_watchers import compute_expiry_at


def _pct(symbol: str) -> PctLineData:
    return PctLineData(symbol=symbol, last_close=100.0, lines="+0.01000,+0.02000,-0.01000", num_days=1)


def test_parse_allocations_clamps_and_skips_crypto_short():
    raw = json.dumps(
        {
            "overall_confidence": 0.8,
            "reasoning": "test",
            "allocations": {
                "BTCUSD": {
                    "alloc": 0.6,
                    "direction": "short",
                    "confidence": 0.9,
                    "rationale": "bad",
                    "leverage": 2,
                },
                "AAPL": {
                    "alloc": 0.7,
                    "direction": "long",
                    "confidence": 0.7,
                    "rationale": "good",
                    "leverage": 3.5,
                },
            },
        }
    )

    allocation = agent_module._parse_allocations_response(raw, symbols=["BTCUSD", "AAPL"])
    assert "BTCUSD" not in allocation.allocations
    assert "AAPL" in allocation.allocations
    assert allocation.allocations["AAPL"].leverage == 2.0
    assert allocation.allocations["AAPL"].alloc <= 1.0


def test_parse_trade_plan_enforces_signs_and_expiry():
    pct_data = {"AAPL": _pct("AAPL"), "BTCUSD": _pct("BTCUSD")}
    raw = json.dumps(
        {
            "overall_confidence": 0.7,
            "positions": [
                {
                    "symbol": "AAPL",
                    "target_alloc": 0.5,
                    "direction": "long",
                    "leverage": 1.5,
                    "entry_bps": 50,
                    "exit_bps": -20,
                    "stop_bps": 10,
                    "entry_mode": "limit",
                    "entry_expiry_days": 6,
                    "hold_expiry_days": 3,
                    "confidence": 0.6,
                },
                {
                    "symbol": "BTCUSD",
                    "target_alloc": 0.4,
                    "direction": "short",
                    "leverage": 2.0,
                    "entry_bps": 20,
                    "exit_bps": -50,
                    "entry_mode": "market",
                    "entry_expiry_days": 2,
                    "hold_expiry_days": 2,
                },
            ],
        }
    )

    plan = agent_module._parse_trade_plan_response(raw, symbols=["AAPL", "BTCUSD"], pct_data=pct_data)
    assert len(plan.positions) == 1
    pos = plan.positions[0]
    assert pos.symbol == "AAPL"
    assert pos.entry_bps <= 0
    assert pos.exit_bps > 0
    assert pos.stop_bps is not None and pos.stop_bps < 0
    assert pos.entry_mode == "watch"
    assert pos.entry_expiry_days <= pos.hold_expiry_days


def test_compute_expiry_at_aligns_stock_close():
    now = datetime(2025, 12, 19, 15, 0, tzinfo=timezone.utc)  # Friday
    expiry = compute_expiry_at("AAPL", 1, now=now)
    # Next trading day close (Monday) at 16:00 ET
    assert expiry.weekday() in {0, 1, 2, 3, 4}
    assert expiry.hour in {20, 21}  # 16:00 ET in UTC depending on DST


def test_compute_expiry_at_crypto_calendar():
    now = datetime(2025, 12, 19, 0, 0, tzinfo=timezone.utc)
    expiry = compute_expiry_at("BTCUSD", 2, now=now)
    assert expiry.date() == (now + timedelta(days=2)).date()
