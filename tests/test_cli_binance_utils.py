from typer.testing import CliRunner

import binance_cli


runner = CliRunner()


def test_open_orders_empty(monkeypatch):
    monkeypatch.setattr(binance_cli.binance_wrapper, "get_open_orders", lambda *args, **kwargs: [])
    result = runner.invoke(binance_cli.app, ["open-orders"])
    assert result.exit_code == 0
    assert "No open orders." in result.stdout


def test_open_orders_with_symbols(monkeypatch):
    calls = []

    def fake_get_open_orders(symbol=None):
        calls.append(symbol)
        return [
            {
                "symbol": symbol or "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "100.0",
                "origQty": "0.5",
                "executedQty": "0.0",
            }
        ]

    monkeypatch.setattr(binance_cli.binance_wrapper, "get_open_orders", fake_get_open_orders)
    result = runner.invoke(binance_cli.app, ["open-orders", "--symbol", "BTCUSDT,ETHUSDT"])
    assert result.exit_code == 0
    assert calls == ["BTCUSDT", "ETHUSDT"]
    assert "Open Orders:" in result.stdout
    assert "BTCUSDT" in result.stdout
    assert "ETHUSDT" in result.stdout


def test_daily_pnl(monkeypatch):
    monkeypatch.setattr(
        binance_cli.binance_wrapper,
        "get_prev_day_pnl_usdt",
        lambda: {
            "prev_total_btc": 1.0,
            "latest_total_btc": 1.1,
            "delta_btc": 0.1,
            "btc_price_usdt": 50000.0,
            "delta_usdt": 5000.0,
            "prev_update_time": "2026-02-01T00:00:00+00:00",
            "latest_update_time": "2026-02-02T00:00:00+00:00",
        },
    )
    result = runner.invoke(binance_cli.app, ["daily-pnl"])
    assert result.exit_code == 0
    assert "Previous-day PnL" in result.stdout
    assert "delta_usdt" in result.stdout


def test_trade_pnl_basic(monkeypatch):
    trades = [
        {"price": "100.0", "qty": "1.0", "isBuyer": True, "time": 1, "commission": "0.10", "commissionAsset": "USDT"},
        {"price": "110.0", "qty": "1.0", "isBuyer": True, "time": 2, "commission": "0.10", "commissionAsset": "USDT"},
        {"price": "120.0", "qty": "1.5", "isBuyer": False, "time": 3, "commission": "0.15", "commissionAsset": "USDT"},
    ]

    monkeypatch.setattr(binance_cli, "_fetch_trades_window", lambda *args, **kwargs: trades)

    result = runner.invoke(binance_cli.app, ["trade-pnl", "--symbol", "SOLUSD", "--days", "1", "--limit", "10"])
    assert result.exit_code == 0
    assert "SOLUSD" in result.stdout
    assert "realized=$25.00" in result.stdout
    assert "net=$24.65" in result.stdout


def test_holdings_summary_snapshots(monkeypatch, tmp_path):
    snapshots = [
        {
            "total_usdt": 100.0,
            "assets": [
                {"asset": "USDT", "amount": 50.0, "price_usdt": 1.0, "value_usdt": 50.0, "free": 50.0, "locked": 0.0},
                {"asset": "BTC", "amount": 0.001, "price_usdt": 50000.0, "value_usdt": 50.0, "free": 0.001, "locked": 0.0},
            ],
        },
        {
            "total_usdt": 120.0,
            "assets": [
                {"asset": "USDT", "amount": 60.0, "price_usdt": 1.0, "value_usdt": 60.0, "free": 60.0, "locked": 0.0},
                {"asset": "BTC", "amount": 0.001, "price_usdt": 60000.0, "value_usdt": 60.0, "free": 0.001, "locked": 0.0},
            ],
        },
    ]
    iterator = iter(snapshots)

    def fake_account_value(*args, **kwargs):
        return next(iterator)

    monkeypatch.setattr(binance_cli.binance_wrapper, "get_account_value_usdt", fake_account_value)
    db_path = tmp_path / "holdings.db"

    result = runner.invoke(binance_cli.app, ["holdings-summary", "--db-path", str(db_path)])
    assert result.exit_code == 0
    assert "Holdings Snapshot" in result.stdout

    result = runner.invoke(binance_cli.app, ["holdings-summary", "--db-path", str(db_path)])
    assert result.exit_code == 0
    assert "Comparison" in result.stdout
