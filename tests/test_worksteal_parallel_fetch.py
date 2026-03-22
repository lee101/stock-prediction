"""Tests for parallel bar fetching and universe file loading."""
import time
from unittest.mock import patch

import pandas as pd
import pytest

from binance_worksteal.trade_live import (
    _fetch_all_bars,
    load_universe_file,
    build_arg_parser,
)


def _make_bars(symbol: str, n: int = 35) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="D"),
        "open": [100.0] * n,
        "high": [105.0] * n,
        "low": [95.0] * n,
        "close": [102.0] * n,
        "volume": [1000.0] * n,
        "symbol": [symbol] * n,
    })


class TestParallelFetch:
    def test_parallel_fetch_basic(self):
        symbols = ["BTCUSD", "ETHUSD", "SOLUSD"]
        bars_map = {s: _make_bars(s) for s in symbols}

        def mock_fetch(client, sym, lookback):
            return bars_map.get(sym, pd.DataFrame())

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            result = _fetch_all_bars(None, symbols, 20, max_workers=3)

        assert set(result.keys()) == set(symbols)
        for sym in symbols:
            assert len(result[sym]) == 35

    def test_parallel_fetch_one_failure(self):
        symbols = ["BTCUSD", "ETHUSD", "FAILUSD"]
        bars_map = {
            "BTCUSD": _make_bars("BTCUSD"),
            "ETHUSD": _make_bars("ETHUSD"),
        }

        def mock_fetch(client, sym, lookback):
            if sym == "FAILUSD":
                raise RuntimeError("API error")
            return bars_map.get(sym, pd.DataFrame())

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            result = _fetch_all_bars(None, symbols, 20, max_workers=3)

        assert "BTCUSD" in result
        assert "ETHUSD" in result
        assert "FAILUSD" not in result

    def test_parallel_fetch_empty_bars_skipped(self):
        symbols = ["BTCUSD", "EMPTYUSD"]

        def mock_fetch(client, sym, lookback):
            if sym == "EMPTYUSD":
                return pd.DataFrame()
            return _make_bars(sym)

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            result = _fetch_all_bars(None, symbols, 20, max_workers=2)

        assert "BTCUSD" in result
        assert "EMPTYUSD" not in result

    def test_parallel_fetch_insufficient_bars_skipped(self):
        symbols = ["BTCUSD", "SHORTUSD"]

        def mock_fetch(client, sym, lookback):
            if sym == "SHORTUSD":
                return _make_bars(sym, n=5)  # too few
            return _make_bars(sym)

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            result = _fetch_all_bars(None, symbols, 20, max_workers=2)

        assert "BTCUSD" in result
        assert "SHORTUSD" not in result

    def test_parallel_is_faster_than_sequential(self):
        symbols = [f"SYM{i}USD" for i in range(20)]

        def mock_fetch(client, sym, lookback):
            time.sleep(0.05)
            return _make_bars(sym)

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            t0 = time.monotonic()
            result = _fetch_all_bars(None, symbols, 20, max_workers=10)
            elapsed = time.monotonic() - t0

        assert len(result) == 20
        # 20 symbols * 50ms = 1s sequential; parallel should be well under
        assert elapsed < 0.5


class TestUniverseFile:
    def test_load_dict_format(self, tmp_path):
        f = tmp_path / "universe.yaml"
        f.write_text(
            "symbols:\n"
            "  - symbol: BTCUSD\n"
            "    fee_tier: fdusd\n"
            "  - symbol: ETHUSD\n"
            "    fee_tier: usdt\n"
        )
        result = load_universe_file(str(f))
        assert result == ["BTCUSD", "ETHUSD"]

    def test_load_string_format(self, tmp_path):
        f = tmp_path / "universe.yaml"
        f.write_text(
            "symbols:\n"
            "  - BTCUSD\n"
            "  - SOLUSD\n"
            "  - DOGEUSD\n"
        )
        result = load_universe_file(str(f))
        assert result == ["BTCUSD", "SOLUSD", "DOGEUSD"]

    def test_normalizes_usdt_suffix(self, tmp_path):
        f = tmp_path / "universe.yaml"
        f.write_text(
            "symbols:\n"
            "  - symbol: BTCUSDT\n"
            "  - ETHUSDT\n"
        )
        result = load_universe_file(str(f))
        assert result == ["BTCUSD", "ETHUSD"]

    def test_missing_symbols_key(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("pairs:\n  - BTCUSD\n")
        with pytest.raises(ValueError, match="symbols"):
            load_universe_file(str(f))

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        with pytest.raises(ValueError):
            load_universe_file(str(f))


class TestMaxSymbolsCap:
    def test_max_symbols_parser_default(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.max_symbols == 100

    def test_max_symbols_custom(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--max-symbols", "50"])
        assert args.max_symbols == 50

    def test_universe_file_parser(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--universe-file", "foo.yaml"])
        assert args.universe_file == "foo.yaml"

    def test_symbols_override_with_cap(self):
        parser = build_arg_parser()
        syms = [f"SYM{i}USD" for i in range(10)]
        args = parser.parse_args(["--symbols"] + syms + ["--max-symbols", "3"])
        assert args.max_symbols == 3
        assert len(args.symbols) == 10  # parser stores all, main() slices
