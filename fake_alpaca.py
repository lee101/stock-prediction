class FakePosition:
    def __init__(self, symbol, qty, side):
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.market_value = qty

class FakeAlpaca:
    def __init__(self, starting_cash=100000):
        self.positions = []
        self.cash = starting_cash

    @property
    def total_buying_power(self):
        return self.cash

    def get_all_positions(self):
        return self.positions

    def open_order_at_price_or_all(self, symbol, qty, side, price):
        pos = FakePosition(symbol, qty, side)
        self.positions.append(pos)
        self.cash -= qty * price
        return pos

    def close_position(self, symbol):
        self.positions = [p for p in self.positions if p.symbol != symbol]

fake_alpaca = FakeAlpaca()
