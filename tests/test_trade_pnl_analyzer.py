from datetime import datetime, timezone

from src.trade_pnl_analyzer import FillEvent, compute_round_trips, normalize_fill_activities


def _dt(idx: int) -> datetime:
    return datetime(2025, 1, 1, 10, 0, idx, tzinfo=timezone.utc)


def test_round_trip_long_with_fees():
    fills = [
        FillEvent("AAPL", "buy", 10, 100.0, 0.1, _dt(0)),
        FillEvent("AAPL", "sell", 10, 102.0, 0.1, _dt(1)),
    ]

    trades = compute_round_trips(fills)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.direction == "long"
    assert trade.entry_price == 100.0
    assert trade.exit_price == 102.0
    assert round(trade.fees, 4) == 0.2
    assert round(trade.net, 4) == 19.8
    assert round(trade.gross, 4) == 20.0


def test_flip_generates_two_trades():
    fills = [
        FillEvent("TSLA", "buy", 5, 10.0, 0.0, _dt(0)),
        FillEvent("TSLA", "sell", 10, 11.0, 0.0, _dt(1)),
        FillEvent("TSLA", "buy", 5, 9.0, 0.0, _dt(2)),
    ]

    trades = compute_round_trips(fills)

    assert len(trades) == 2
    long_trade, short_trade = trades
    assert round(long_trade.net, 4) == 5.0
    assert long_trade.direction == "long"
    assert short_trade.direction == "short"
    assert round(short_trade.net, 4) == 10.0
    assert short_trade.entry_price == 11.0
    assert short_trade.exit_price == 9.0


def test_residual_qty_within_epsilon_treated_closed():
    fills = [
        FillEvent("BTCUSD", "buy", 1.0, 100.0, 0.0, _dt(0)),
        FillEvent("BTCUSD", "sell", 0.9999995, 110.0, 0.0, _dt(1)),
    ]

    trades = compute_round_trips(fills)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.quantity == 0.9999995
    # residual is tiny; should still be realized
    assert trade.net > 9.9


def test_entry_and_exit_fees_counted_once():
    fills = [
        FillEvent("ETHUSD", "buy", 2, 100.0, 0.2, _dt(0)),   # entry fee 0.2
        FillEvent("ETHUSD", "sell", 2, 105.0, 0.3, _dt(1)),  # exit fee 0.3
    ]
    trades = compute_round_trips(fills)
    trade = trades[0]
    assert round(trade.gross, 4) == 10.0  # price move * qty
    assert round(trade.fees, 4) == 0.5    # 0.2 + 0.3
    assert round(trade.net, 4) == 9.5

def test_normalize_fill_activities_filters_and_parses():
    activities = [
        {
            "activity_type": "FILL",
            "symbol": "aapl",
            "side": "buy",
            "qty": "2",
            "price": "150.5",
            "fee": "0.15",
            "transaction_time": "2025-01-01T12:00:00Z",
        },
        {"activity_type": "DIV"},  # ignored
    ]

    fills = normalize_fill_activities(activities)
    assert len(fills) == 1
    fill = fills[0]
    assert fill.symbol == "AAPL"
    assert fill.side == "buy"
    assert fill.qty == 2.0
    assert fill.price == 150.5
    assert fill.fee == 0.15
    assert fill.transacted_at.tzinfo == timezone.utc
