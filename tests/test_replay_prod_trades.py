"""Tests for replay_prod_trades.py log parsing and simulation."""

import tempfile
from datetime import datetime
import uuid

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from replay_prod_trades import parse_log, simulate, build_comparison, Trade, PortfolioSnapshot


SAMPLE_LOG = """\
2026-03-22 04:00:14.585 | INFO     | __main__:run_trading_cycle:1358 - Portfolio: FDUSD=0.00, USDT=3.81, borrowable_usdt=11266.62, total=3313.41
2026-03-22 04:00:17.316 | INFO     | __main__:run_trading_cycle:1396 -   FORCED EXIT DOGEUSD: held 6.6h > 6.0h, market selling 17658.31906227
2026-03-22 04:00:19.043 | INFO     | __main__:place_margin_limit_sell:1046 - MARGIN SELL DOGEUSDT: qty=17658.0, price=0.09321, notional=1645.90
2026-03-22 04:00:19.268 | INFO     | __main__:run_trading_cycle:1414 -   Forced exit sell @ $0.09
2026-03-22 04:00:23.337 | INFO     | __main__:place_margin_limit_buy:1004 - MARGIN BUY ETHUSDT: qty=1.5412, price=2149.87, notional=3313.38
2026-03-22 04:00:26.767 | INFO     | __main__:place_margin_limit_buy:1004 - MARGIN BUY SOLUSDT: qty=27.762, price=89.51, notional=2484.98
2026-03-22 05:00:37.035 | INFO     | __main__:run_trading_cycle:1358 - Portfolio: FDUSD=0.00, USDT=0.00, borrowable_usdt=6180.53, total=3327.14
2026-03-22 05:00:44.498 | INFO     | __main__:place_margin_limit_sell:1046 - MARGIN SELL ETHUSDT: qty=1.5396, price=2175.63, notional=3349.60
2026-03-22 05:00:47.033 | INFO     | __main__:place_margin_limit_sell:1046 - MARGIN SELL SOLUSDT: qty=27.735, price=90.7, notional=2515.56
"""

SAMPLE_LOG_NEW_PORTFOLIO = """\
2026-03-22 19:04:24.262 | INFO     | __main__:run_hybrid_trading_cycle:1902 - Portfolio: $3068.33 | Cash: $2109.05 | borrowable_usdt=11504.53
2026-03-22 20:23:13.249 | INFO     | __main__:run_hybrid_trading_cycle:1902 - Portfolio: $3068.33 | Cash: $2109.05 | borrowable_usdt=11506.72
"""


def _write_tmp(content):
    temp_dir = tempfile.mkdtemp(prefix="replay_prod_trades_", dir=os.getcwd())
    path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.log")
    with open(path, "w") as handle:
        handle.write(content)
    os.chmod(path, 0o600)
    return path


def _cleanup_tmp(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    try:
        os.rmdir(os.path.dirname(path))
    except OSError:
        pass


def test_parse_trades():
    path = _write_tmp(SAMPLE_LOG)
    cutoff = datetime(2026, 3, 22)
    trades, snaps = parse_log(path, cutoff)
    _cleanup_tmp(path)

    assert len(trades) == 5
    assert trades[0].side == "SELL"
    assert trades[0].symbol == "DOGEUSDT"
    assert trades[0].is_forced_exit is True
    assert trades[1].side == "BUY"
    assert trades[1].symbol == "ETHUSDT"
    assert abs(trades[1].price - 2149.87) < 0.01
    assert trades[2].side == "BUY"
    assert trades[2].symbol == "SOLUSDT"
    assert trades[3].side == "SELL"
    assert trades[3].symbol == "ETHUSDT"
    assert trades[3].is_forced_exit is False


def test_parse_portfolio_old_format():
    path = _write_tmp(SAMPLE_LOG)
    cutoff = datetime(2026, 3, 22)
    _, snaps = parse_log(path, cutoff)
    _cleanup_tmp(path)

    assert len(snaps) == 2
    assert abs(snaps[0].total - 3313.41) < 0.01
    assert abs(snaps[1].total - 3327.14) < 0.01


def test_parse_portfolio_new_format():
    path = _write_tmp(SAMPLE_LOG_NEW_PORTFOLIO)
    cutoff = datetime(2026, 3, 22)
    _, snaps = parse_log(path, cutoff)
    _cleanup_tmp(path)

    assert len(snaps) == 2
    assert abs(snaps[0].total - 3068.33) < 0.01
    assert abs(snaps[0].cash - 2109.05) < 0.01


def test_simulate_buy_sell():
    trades = [
        Trade(ts=datetime(2026, 3, 22, 4), symbol="ETHUSDT", side="BUY",
              qty=1.0, price=2000.0, notional=2000.0),
        Trade(ts=datetime(2026, 3, 22, 8), symbol="ETHUSDT", side="SELL",
              qty=1.0, price=2100.0, notional=2100.0),
    ]
    state = simulate(trades, 3000.0)

    assert len(state.trade_pnls) == 1
    tp = state.trade_pnls[0]
    assert tp["symbol"] == "ETHUSDT"
    assert abs(tp["gross_pnl"] - 100.0) < 0.01
    assert tp["fee"] > 0
    assert tp["margin_interest"] > 0
    assert tp["hours_held"] == 4.0

    margin = 2000.0 * 0.0625 * 4.0 / 8760.0
    expected_net = 100.0 - 2100.0 * 0.001 - margin
    assert abs(tp["net_pnl"] - expected_net) < 0.01


def test_simulate_fees_correct():
    trades = [
        Trade(ts=datetime(2026, 3, 22, 4), symbol="BTCUSDT", side="BUY",
              qty=0.01, price=80000.0, notional=800.0),
        Trade(ts=datetime(2026, 3, 22, 5), symbol="BTCUSDT", side="SELL",
              qty=0.01, price=80000.0, notional=800.0),
    ]
    state = simulate(trades, 5000.0)
    assert abs(state.total_fees - 1.60) < 0.01  # 2 * 800 * 0.001


def test_simulate_forced_exit_flagged():
    trades = [
        Trade(ts=datetime(2026, 3, 22, 4), symbol="DOGEUSDT", side="BUY",
              qty=10000.0, price=0.09, notional=900.0),
        Trade(ts=datetime(2026, 3, 22, 11), symbol="DOGEUSDT", side="SELL",
              qty=10000.0, price=0.089, notional=890.0, is_forced_exit=True),
    ]
    state = simulate(trades, 3000.0)
    assert state.trade_pnls[0]["forced_exit"] is True
    assert state.trade_pnls[0]["hours_held"] == 7.0


def test_simulate_multiple_positions():
    trades = [
        Trade(ts=datetime(2026, 3, 22, 4), symbol="ETHUSDT", side="BUY",
              qty=1.0, price=2000.0, notional=2000.0),
        Trade(ts=datetime(2026, 3, 22, 4, 30), symbol="SOLUSDT", side="BUY",
              qty=10.0, price=90.0, notional=900.0),
        Trade(ts=datetime(2026, 3, 22, 8), symbol="ETHUSDT", side="SELL",
              qty=1.0, price=2100.0, notional=2100.0),
        Trade(ts=datetime(2026, 3, 22, 8), symbol="SOLUSDT", side="SELL",
              qty=10.0, price=85.0, notional=850.0),
    ]
    state = simulate(trades, 5000.0)
    assert "ETHUSDT" in state.per_symbol_pnl
    assert "SOLUSDT" in state.per_symbol_pnl
    assert state.per_symbol_pnl["ETHUSDT"] > 0
    assert state.per_symbol_pnl["SOLUSDT"] < 0


def test_cutoff_filtering():
    log = """\
2026-03-19 10:00:00.000 | INFO     | __main__:place_margin_limit_buy:1004 - MARGIN BUY ETHUSDT: qty=1.0, price=2000.0, notional=2000.0
2026-03-22 10:00:00.000 | INFO     | __main__:place_margin_limit_sell:1046 - MARGIN SELL ETHUSDT: qty=1.0, price=2100.0, notional=2100.0
"""
    path = _write_tmp(log)
    cutoff = datetime(2026, 3, 21)
    trades, _ = parse_log(path, cutoff)
    _cleanup_tmp(path)
    assert len(trades) == 1
    assert trades[0].side == "SELL"


def test_equity_curve():
    snaps = [
        PortfolioSnapshot(ts=datetime(2026, 3, 22, 3), total=3000.0, cash=3000.0),
        PortfolioSnapshot(ts=datetime(2026, 3, 22, 9), total=3100.0, cash=3100.0),
    ]
    trades = [
        Trade(ts=datetime(2026, 3, 22, 4), symbol="ETHUSDT", side="BUY",
              qty=1.0, price=2000.0, notional=2000.0),
        Trade(ts=datetime(2026, 3, 22, 8), symbol="ETHUSDT", side="SELL",
              qty=1.0, price=2100.0, notional=2100.0),
    ]
    curve = build_comparison(snaps, trades, 3000.0)
    assert len(curve) == 2
    assert curve[0]["actual"] == 3000.0
    assert curve[0]["simulated"] == 3000.0


def test_position_averaging():
    trades = [
        Trade(ts=datetime(2026, 3, 22, 4), symbol="ETHUSDT", side="BUY",
              qty=1.0, price=2000.0, notional=2000.0),
        Trade(ts=datetime(2026, 3, 22, 5), symbol="ETHUSDT", side="BUY",
              qty=1.0, price=2200.0, notional=2200.0),
        Trade(ts=datetime(2026, 3, 22, 8), symbol="ETHUSDT", side="SELL",
              qty=2.0, price=2150.0, notional=4300.0),
    ]
    state = simulate(trades, 10000.0)
    tp = state.trade_pnls[0]
    assert abs(tp["entry_price"] - 2100.0) < 0.01
    assert abs(tp["gross_pnl"] - 100.0) < 0.01


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
