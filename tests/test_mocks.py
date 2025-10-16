import uuid

try:
    from alpaca.trading import Position
except ImportError:  # pragma: no cover - fallback for environments without Alpaca SDK
    class Position:  # type: ignore[override]
        """Lightweight stand-in for alpaca.trading.Position used in CI."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


def test_mocks():
    btc_position = Position(symbol='BTCUSD', qty=1000, side='long', avg_entry_price=18000, unrealized_plpc=0.1,
                            unrealized_pl=0.1, market_value=5000,
                            asset_id=uuid.uuid4(),
                            exchange='FTXU',
                            asset_class='crypto',
                            cost_basis=1,
                            unrealized_intraday_pl=1,
                            unrealized_intraday_plpc=1,
                            current_price=100000,
                            lastday_price=1,
                            change_today=1,
                            )
