import sys
import types


def test_entry_requires_cash_refreshes_account(monkeypatch):
    """Verify entry cash checks refresh the Alpaca account cache before using globals."""

    # Install minimal Alpaca stubs before importing the CLI.
    trade_module = types.ModuleType("alpaca_trade_api")

    class _DummyREST:
        def __init__(self, *args, **kwargs):
            self._orders = []

        def get_all_positions(self):
            return []

        def get_account(self):
            return types.SimpleNamespace(equity=1.0, cash=1.0, multiplier=1.0, buying_power=1.0)

        def get_clock(self):
            return types.SimpleNamespace(is_open=True)

        def get_orders(self):
            return []

    trade_module.REST = _DummyREST
    sys.modules["alpaca_trade_api"] = trade_module
    sys.modules.setdefault("alpaca_trade_api.rest", types.ModuleType("alpaca_trade_api.rest")).APIError = Exception

    from scripts import maxdiff_cli

    calls = {"count": 0}

    def fake_refresh(force: bool = False):
        calls["count"] += 1
        maxdiff_cli.alpaca_wrapper.cash = 1000.0
        maxdiff_cli.alpaca_wrapper.total_buying_power = 2000.0
        maxdiff_cli.alpaca_wrapper.equity = 1000.0

    monkeypatch.setattr(maxdiff_cli.alpaca_wrapper, "refresh_account_cache", fake_refresh)
    maxdiff_cli.alpaca_wrapper.cash = 10.0
    maxdiff_cli.alpaca_wrapper.total_buying_power = 10.0
    maxdiff_cli.alpaca_wrapper.equity = 10.0

    assert maxdiff_cli._entry_requires_cash("buy", price=25.0, qty=10.0) is True
    assert calls["count"] == 1
