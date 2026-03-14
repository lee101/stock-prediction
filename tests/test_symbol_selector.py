"""Tests for the meta-strategy symbol selector."""
import tempfile
from datetime import datetime, timezone, timedelta

from unified_orchestrator.symbol_selector import SymbolSelector


def test_allow_all_with_no_data():
    """Symbols with no trades should be allowed."""
    selector = SymbolSelector(["BTCUSD", "ETHUSD", "SOLUSD"])
    assert selector.get_allowed_symbols() == ["BTCUSD", "ETHUSD", "SOLUSD"]


def test_allow_with_insufficient_trades():
    """Symbols with fewer trades than min_trades should be allowed."""
    selector = SymbolSelector(["BTCUSD", "ETHUSD"], min_trades=3)
    selector.record_trade("BTCUSD", pnl_pct=-1.0)
    selector.record_trade("BTCUSD", pnl_pct=-2.0)
    # Only 2 trades, min_trades=3, so still allowed
    assert selector.should_trade("BTCUSD")


def test_block_low_win_rate():
    """Symbols with consistently losing trades should be blocked."""
    selector = SymbolSelector(["BTCUSD", "ETHUSD"], min_trades=3, min_win_rate=0.3)
    now = datetime.now(timezone.utc)
    for i in range(5):
        selector.record_trade("BTCUSD", pnl_pct=-1.0, timestamp=now - timedelta(hours=i))
    # 0% win rate, should be blocked
    assert not selector.should_trade("BTCUSD")
    # ETHUSD has no trades, should be allowed
    assert selector.should_trade("ETHUSD")
    assert selector.get_allowed_symbols() == ["ETHUSD"]


def test_allow_good_performer():
    """Symbols with good win rate should pass."""
    selector = SymbolSelector(["BTCUSD"], min_trades=3, min_win_rate=0.3)
    now = datetime.now(timezone.utc)
    selector.record_trade("BTCUSD", pnl_pct=1.5, timestamp=now - timedelta(hours=3))
    selector.record_trade("BTCUSD", pnl_pct=0.8, timestamp=now - timedelta(hours=2))
    selector.record_trade("BTCUSD", pnl_pct=-0.3, timestamp=now - timedelta(hours=1))
    # 2/3 = 66% win rate, should pass
    assert selector.should_trade("BTCUSD")


def test_prune_old_trades():
    """Trades older than lookback window should be pruned."""
    selector = SymbolSelector(["BTCUSD"], lookback_hours=24, min_trades=3, min_win_rate=0.3)
    now = datetime.now(timezone.utc)
    # Add old losing trades (> 24h ago)
    for i in range(5):
        selector.record_trade("BTCUSD", pnl_pct=-2.0, timestamp=now - timedelta(hours=48 + i))
    # Add recent winning trades
    for i in range(3):
        selector.record_trade("BTCUSD", pnl_pct=1.0, timestamp=now - timedelta(hours=i))
    # Old trades should be pruned, only recent ones remain
    assert selector.should_trade("BTCUSD")


def test_persistence():
    """Test save/load cycle."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    now = datetime.now(timezone.utc)
    selector1 = SymbolSelector(["BTCUSD", "ETHUSD"], persist_path=path)
    selector1.record_trade("BTCUSD", pnl_pct=1.5, timestamp=now)
    selector1.record_trade("ETHUSD", pnl_pct=-0.5, timestamp=now)

    # Load from disk
    selector2 = SymbolSelector(["BTCUSD", "ETHUSD"], persist_path=path)
    stats = selector2._stats
    assert stats["BTCUSD"].num_trades == 1
    assert stats["ETHUSD"].num_trades == 1
    assert stats["BTCUSD"].trades[0].pnl_pct == 1.5


def test_symbol_weights():
    """Test weight computation."""
    selector = SymbolSelector(["BTCUSD", "ETHUSD"], min_trades=2)
    now = datetime.now(timezone.utc)
    # BTCUSD: 2 wins
    selector.record_trade("BTCUSD", pnl_pct=2.0, timestamp=now - timedelta(hours=2))
    selector.record_trade("BTCUSD", pnl_pct=1.0, timestamp=now - timedelta(hours=1))
    # ETHUSD: 2 losses
    selector.record_trade("ETHUSD", pnl_pct=-1.0, timestamp=now - timedelta(hours=2))
    selector.record_trade("ETHUSD", pnl_pct=-2.0, timestamp=now - timedelta(hours=1))

    weights = selector.get_symbol_weights()
    assert weights["BTCUSD"] > weights["ETHUSD"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_summary():
    """Test summary string generation."""
    selector = SymbolSelector(["BTCUSD"])
    now = datetime.now(timezone.utc)
    selector.record_trade("BTCUSD", pnl_pct=1.0, timestamp=now)
    summary = selector.summary()
    assert "BTCUSD" in summary
    assert "1 trades" in summary
