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

        dummy_client = object()

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            result = _fetch_all_bars(dummy_client, symbols, 20, max_workers=3)

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

        dummy_client = object()

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            result = _fetch_all_bars(dummy_client, symbols, 20, max_workers=3)

        assert "BTCUSD" in result
        assert "ETHUSD" in result
        assert "FAILUSD" not in result

    def test_parallel_fetch_empty_bars_skipped(self):
        symbols = ["BTCUSD", "EMPTYUSD"]

        def mock_fetch(client, sym, lookback):
            if sym == "EMPTYUSD":
                return pd.DataFrame()
            return _make_bars(sym)

        dummy_client = object()

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            result = _fetch_all_bars(dummy_client, symbols, 20, max_workers=2)

        assert "BTCUSD" in result
        assert "EMPTYUSD" not in result

    def test_parallel_fetch_insufficient_bars_skipped(self):
        symbols = ["BTCUSD", "SHORTUSD"]

        def mock_fetch(client, sym, lookback):
            if sym == "SHORTUSD":
                return _make_bars(sym, n=5)  # too few
            return _make_bars(sym)

        dummy_client = object()

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            result = _fetch_all_bars(dummy_client, symbols, 20, max_workers=2)

        assert "BTCUSD" in result
        assert "SHORTUSD" not in result

    def test_parallel_is_faster_than_sequential(self):
        symbols = [f"SYM{i}USD" for i in range(20)]

        def mock_fetch(client, sym, lookback):
            time.sleep(0.05)
            return _make_bars(sym)

        dummy_client = object()

        with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
            t0 = time.monotonic()
            result = _fetch_all_bars(dummy_client, symbols, 20, max_workers=10)
            elapsed = time.monotonic() - t0

        assert len(result) == 20
        # 20 symbols * 50ms = 1s sequential; parallel should be well under
        assert elapsed < 0.5

    def test_local_fetch_batches_load_daily_bars_by_directory(self):
        symbols = ["BTCUSD", "ETHUSD", "SOLUSD"]
        calls = []

        def mock_load_daily_bars(data_dir, requested_symbols):
            calls.append((data_dir, list(requested_symbols)))
            if data_dir == "trainingdatadailybinance":
                return {"BTCUSD": _make_bars("BTCUSD", n=60)}
            if data_dir == "trainingdata/train":
                return {"ETHUSD": _make_bars("ETHUSD", n=55), "SOLUSD": _make_bars("SOLUSD", n=5)}
            raise AssertionError(f"unexpected data_dir: {data_dir}")

        with patch("binance_worksteal.trade_live.load_daily_bars", side_effect=mock_load_daily_bars):
            result = _fetch_all_bars(None, symbols, 20, max_workers=3)

        assert set(result) == {"BTCUSD", "ETHUSD"}
        assert calls == [
            ("trainingdatadailybinance", ["BTCUSD", "ETHUSD", "SOLUSD"]),
            ("trainingdata/train", ["ETHUSD", "SOLUSD"]),
        ]
        assert len(result["BTCUSD"]) == 35
        assert len(result["ETHUSD"]) == 35

    def test_local_fetch_falls_back_to_fetch_daily_bars_when_monkeypatched(self):
        symbols = ["BTCUSD", "ETHUSD"]
        calls = []

        def mock_fetch(client, sym, lookback):
            calls.append((client, sym, lookback))
            return _make_bars(sym, n=50)

        with patch("binance_worksteal.trade_live.load_daily_bars", side_effect=AssertionError("should not batch-load")):
            with patch("binance_worksteal.trade_live.fetch_daily_bars", side_effect=mock_fetch):
                result = _fetch_all_bars(None, symbols, 20, max_workers=2)

        assert set(result) == set(symbols)
        assert calls == [
            (None, "BTCUSD", 30),
            (None, "ETHUSD", 30),
        ]

    def test_local_fetch_continues_after_first_directory_load_failure(self):
        symbols = ["BTCUSD", "ETHUSD"]
        calls = []

        def mock_load_daily_bars(data_dir, requested_symbols):
            calls.append((data_dir, list(requested_symbols)))
            if data_dir == "trainingdatadailybinance":
                raise RuntimeError("disk error")
            if data_dir == "trainingdata/train":
                return {
                    "BTCUSD": _make_bars("BTCUSD", n=55),
                    "ETHUSD": _make_bars("ETHUSD", n=52),
                }
            raise AssertionError(f"unexpected data_dir: {data_dir}")

        with patch("binance_worksteal.trade_live.load_daily_bars", side_effect=mock_load_daily_bars):
            result = _fetch_all_bars(None, symbols, 20, max_workers=2)

        assert set(result) == set(symbols)
        assert calls == [
            ("trainingdatadailybinance", ["BTCUSD", "ETHUSD"]),
            ("trainingdata/train", ["BTCUSD", "ETHUSD"]),
        ]
        assert len(result["BTCUSD"]) == 35
        assert len(result["ETHUSD"]) == 35

    def test_local_fetch_ignores_invalid_directory_response_shape(self):
        symbols = ["BTCUSD", "ETHUSD"]
        calls = []

        def mock_load_daily_bars(data_dir, requested_symbols):
            calls.append((data_dir, list(requested_symbols)))
            if data_dir == "trainingdatadailybinance":
                return ["not", "a", "mapping"]
            if data_dir == "trainingdata/train":
                return {
                    "BTCUSD": _make_bars("BTCUSD", n=60),
                    "ETHUSD": _make_bars("ETHUSD", n=58),
                }
            raise AssertionError(f"unexpected data_dir: {data_dir}")

        with patch("binance_worksteal.trade_live.load_daily_bars", side_effect=mock_load_daily_bars):
            result = _fetch_all_bars(None, symbols, 20, max_workers=2)

        assert set(result) == set(symbols)
        assert calls == [
            ("trainingdatadailybinance", ["BTCUSD", "ETHUSD"]),
            ("trainingdata/train", ["BTCUSD", "ETHUSD"]),
        ]
        assert len(result["BTCUSD"]) == 35
        assert len(result["ETHUSD"]) == 35

    def test_local_fetch_ignores_invalid_symbol_payload_and_tries_next_directory(self):
        symbols = ["BTCUSD", "ETHUSD"]
        calls = []

        def mock_load_daily_bars(data_dir, requested_symbols):
            calls.append((data_dir, list(requested_symbols)))
            if data_dir == "trainingdatadailybinance":
                return {
                    "BTCUSD": {"close": 100.0},
                    "ETHUSD": _make_bars("ETHUSD", n=60),
                }
            if data_dir == "trainingdata/train":
                return {
                    "BTCUSD": _make_bars("BTCUSD", n=59),
                }
            raise AssertionError(f"unexpected data_dir: {data_dir}")

        with patch("binance_worksteal.trade_live.load_daily_bars", side_effect=mock_load_daily_bars):
            result = _fetch_all_bars(None, symbols, 20, max_workers=2)

        assert set(result) == set(symbols)
        assert calls == [
            ("trainingdatadailybinance", ["BTCUSD", "ETHUSD"]),
            ("trainingdata/train", ["BTCUSD"]),
        ]
        assert len(result["BTCUSD"]) == 35
        assert len(result["ETHUSD"]) == 35


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

    def test_invalid_yaml_reports_value_error(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("symbols: [BTCUSD\n")
        with pytest.raises(ValueError, match="Invalid universe YAML"):
            load_universe_file(str(f))


    def test_symbols_value_must_be_a_list(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("symbols: BTCUSD\n")
        with pytest.raises(ValueError, match="symbols.*list"):
            load_universe_file(str(f))

    def test_symbol_entry_missing_symbol_field_raises_value_error(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("symbols:\n  - fee_tier: usdt\n")
        with pytest.raises(ValueError, match="missing 'symbol' field"):
            load_universe_file(str(f))

    def test_invalid_min_notional_raises_value_error(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("symbols:\n  - symbol: BTCUSD\n    min_notional: nope\n")
        with pytest.raises(ValueError, match="Invalid min_notional"):
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
