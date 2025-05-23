class FakePosition:
    def __init__(self, symbol, qty, side, entry_price, entry_date=None):
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.entry_price = entry_price
        self.entry_date = entry_date

    @property
    def market_value(self):
        return self.qty * self.entry_price


class FakeAlpaca:
    """Minimal in-memory trading simulator."""

    def __init__(self, starting_cash=100000):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions = []
        self.closed = []

    @property
    def total_buying_power(self):
        return self.cash

    def get_all_positions(self):
        return list(self.positions)

    def open_order_at_price_or_all(self, symbol, qty, side, price):
        pos = FakePosition(symbol, qty, side, price)
        self.positions.append(pos)
        if side == "buy":
            self.cash -= qty * price
        else:  # short adds cash
            self.cash += qty * price
        return pos

    def close_position(self, symbol, price):
        remaining = []
        for p in self.positions:
            if p.symbol == symbol:
                if p.side == "buy":
                    self.cash += p.qty * price
                    profit = (price - p.entry_price) * p.qty
                else:
                    self.cash -= p.qty * price
                    profit = (p.entry_price - price) * p.qty
                self.closed.append((p, price, profit))
            else:
                remaining.append(p)
        self.positions = remaining

    def close_all_positions(self, price):
        for p in list(self.positions):
            self.close_position(p.symbol, price)


fake_alpaca = FakeAlpaca()
