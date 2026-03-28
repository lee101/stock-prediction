"""Tests for expanded symbol universe, data validation, and volume filtering."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from binance_worksteal.data import (
    DEFAULT_SYMBOLS,
    EXPANDED_SYMBOLS,
    ORIGINAL_30_SYMBOLS,
    validate_symbol_data,
    filter_symbols_by_volume,
    discover_symbols,
    generate_symbols_arg,
    load_symbol_data,
    compute_features,
    FEATURE_NAMES,
)
from binance_worksteal.backtest import (
    FULL_UNIVERSE,
    ORIGINAL_30_UNIVERSE,
    EXPANDED_UNIVERSE,
)


def _write_symbol_csv(path: Path, n_days: int = 400) -> None:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")
    close = np.linspace(100.0, 200.0, n_days)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close - 1.0,
        "high": close + 1.0,
        "low": close - 2.0,
        "close": close,
        "volume": np.full(n_days, 10_000.0),
    })
    df.to_csv(path, index=False)


class TestSymbolLists:
    def test_original_30_preserved(self):
        assert len(ORIGINAL_30_SYMBOLS) == 30
        assert "BTCUSDT" in ORIGINAL_30_SYMBOLS
        assert "MATICUSDT" in ORIGINAL_30_SYMBOLS

    def test_expanded_symbols_count(self):
        assert len(EXPANDED_SYMBOLS) == 40

    def test_default_symbols_is_union(self):
        assert DEFAULT_SYMBOLS == ORIGINAL_30_SYMBOLS + EXPANDED_SYMBOLS
        assert len(DEFAULT_SYMBOLS) == 70

    def test_no_duplicates_in_default(self):
        assert len(DEFAULT_SYMBOLS) == len(set(DEFAULT_SYMBOLS))

    def test_all_usdt_suffix(self):
        for sym in DEFAULT_SYMBOLS:
            assert sym.endswith("USDT"), f"{sym} missing USDT suffix"

    def test_backtest_universe_matches(self):
        assert len(ORIGINAL_30_UNIVERSE) == 30
        assert len(EXPANDED_UNIVERSE) == 40
        assert len(FULL_UNIVERSE) == 70

    def test_backtest_universe_usd_suffix(self):
        for sym in FULL_UNIVERSE:
            assert sym.endswith("USD"), f"{sym} missing USD suffix"
            assert not sym.endswith("USDT"), f"{sym} has USDT suffix (should be USD)"

    def test_backtest_and_data_symbols_correspond(self):
        data_bases = sorted(s.replace("USDT", "") for s in DEFAULT_SYMBOLS)
        bt_bases = sorted(s.replace("USD", "") for s in FULL_UNIVERSE)
        assert data_bases == bt_bases

    def test_expanded_candidates_present(self):
        expanded_bases = {s.replace("USDT", "") for s in EXPANDED_SYMBOLS}
        assert "HBAR" in expanded_bases
        assert "FET" in expanded_bases
        assert "RENDER" in expanded_bases
        assert "THETA" in expanded_bases
        assert "FTM" in expanded_bases
        assert "TON" in expanded_bases


class TestValidateSymbolData:
    def _make_df(self, n_days=400, start="2023-01-01"):
        dates = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
        rng = np.random.RandomState(42)
        close = 100.0 + rng.randn(n_days).cumsum()
        close = np.maximum(close, 1.0)
        return pd.DataFrame({
            "timestamp": dates,
            "open": close + rng.randn(n_days) * 0.5,
            "high": close + abs(rng.randn(n_days)) * 2,
            "low": close - abs(rng.randn(n_days)) * 2,
            "close": close,
            "volume": rng.rand(n_days) * 1e6,
        })

    def test_valid_data_passes(self):
        df = self._make_df(400)
        ok, reason = validate_symbol_data(df, min_bars=365)
        assert ok
        assert reason == "ok"

    def test_none_data_fails(self):
        ok, reason = validate_symbol_data(None)
        assert not ok
        assert "no data" in reason

    def test_empty_data_fails(self):
        ok, reason = validate_symbol_data(pd.DataFrame())
        assert not ok
        assert "no data" in reason

    def test_too_few_bars_fails(self):
        df = self._make_df(100)
        ok, reason = validate_symbol_data(df, min_bars=365)
        assert not ok
        assert "365" in reason

    def test_missing_columns_fails(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=400, tz="UTC"), "close": range(400)})
        ok, reason = validate_symbol_data(df, min_bars=30)
        assert not ok
        assert "missing columns" in reason

    def test_missing_bars_pct_fails(self):
        df = self._make_df(400)
        keep = list(range(0, 400, 3))
        df = df.iloc[keep].copy()
        ok, reason = validate_symbol_data(df, min_bars=100, max_missing_pct=0.05)
        assert not ok
        assert "missing bars" in reason

    def test_negative_close_fails(self):
        df = self._make_df(400)
        df.loc[50, "close"] = -1.0
        ok, reason = validate_symbol_data(df, min_bars=30)
        assert not ok
        assert "non-positive" in reason

    def test_lenient_missing_pct_passes(self):
        df = self._make_df(400)
        ok, reason = validate_symbol_data(df, min_bars=30, max_missing_pct=0.50)
        assert ok


class TestFilterSymbolsByVolume:
    def test_filter_supports_legacy_usd_filename(self, tmp_path):
        _write_symbol_csv(tmp_path / "INJUSD.csv")
        passed = filter_symbols_by_volume(
            str(tmp_path), symbols=["INJUSDT"], min_avg_daily_volume_usd=0.0,
        )
        assert passed == ["INJUSDT"]

    def test_filter_with_real_data(self):
        data_dir = "trainingdata/train"
        p = Path(data_dir)
        if not p.exists() or not list(p.glob("*USDT.csv")):
            pytest.skip("No training data available")
        passed = filter_symbols_by_volume(
            data_dir, symbols=["BTCUSDT", "ETHUSDT"], min_avg_daily_volume_usd=1_000_000.0,
        )
        assert "BTCUSDT" in passed
        assert "ETHUSDT" in passed

    def test_filter_rejects_missing_symbols(self):
        passed = filter_symbols_by_volume(
            "trainingdata/train", symbols=["NONEXISTENT999USDT"], min_avg_daily_volume_usd=0.0,
        )
        assert len(passed) == 0

    def test_filter_with_zero_threshold(self):
        data_dir = "trainingdata/train"
        p = Path(data_dir)
        if not p.exists() or not list(p.glob("*USDT.csv")):
            pytest.skip("No training data available")
        passed = filter_symbols_by_volume(
            data_dir, symbols=ORIGINAL_30_SYMBOLS, min_avg_daily_volume_usd=0.0,
        )
        assert len(passed) == 30

    def test_high_threshold_filters_some(self):
        data_dir = "trainingdata/train"
        p = Path(data_dir)
        if not p.exists() or not list(p.glob("*USDT.csv")):
            pytest.skip("No training data available")
        all_passed = filter_symbols_by_volume(
            data_dir, symbols=ORIGINAL_30_SYMBOLS, min_avg_daily_volume_usd=0.0,
        )
        high_passed = filter_symbols_by_volume(
            data_dir, symbols=ORIGINAL_30_SYMBOLS, min_avg_daily_volume_usd=1e12,
        )
        assert len(high_passed) < len(all_passed)


class TestDiscoverSymbols:
    def test_discover_normalizes_legacy_usd_filename(self, tmp_path):
        _write_symbol_csv(tmp_path / "INJUSD.csv")
        found = discover_symbols(str(tmp_path), min_bars=100, min_avg_daily_volume_usd=0.0)
        assert found == ["INJUSDT"]

    def test_discover_returns_list(self):
        data_dir = "trainingdata/train"
        p = Path(data_dir)
        if not p.exists() or not list(p.glob("*USDT.csv")):
            pytest.skip("No training data available")
        found = discover_symbols(data_dir, min_bars=100, min_avg_daily_volume_usd=0.0)
        assert isinstance(found, list)
        assert len(found) > 0
        for sym in found:
            assert sym.endswith("USDT")

    def test_discover_nonexistent_dir(self):
        found = discover_symbols("/tmp/nonexistent_dir_xyz123")
        assert found == []

    def test_discover_respects_min_bars(self):
        data_dir = "trainingdata/train"
        p = Path(data_dir)
        if not p.exists() or not list(p.glob("*USDT.csv")):
            pytest.skip("No training data available")
        loose = discover_symbols(data_dir, min_bars=100, min_avg_daily_volume_usd=0.0)
        strict = discover_symbols(data_dir, min_bars=9999, min_avg_daily_volume_usd=0.0)
        assert len(strict) <= len(loose)


class TestGenerateSymbolsArg:
    def test_generate_supports_legacy_usd_filename(self, tmp_path):
        _write_symbol_csv(tmp_path / "INJUSD.csv")
        assert generate_symbols_arg(str(tmp_path), min_bars=100, min_avg_daily_volume_usd=0.0) == "INJUSD"
        assert (
            generate_symbols_arg(
                str(tmp_path),
                min_bars=100,
                min_avg_daily_volume_usd=0.0,
                use_usd_suffix=False,
            )
            == "INJUSDT"
        )

    def test_generate_returns_string(self):
        data_dir = "trainingdata/train"
        p = Path(data_dir)
        if not p.exists() or not list(p.glob("*USDT.csv")):
            pytest.skip("No training data available")
        arg = generate_symbols_arg(data_dir, min_bars=100, min_avg_daily_volume_usd=0.0)
        assert isinstance(arg, str)
        parts = arg.split()
        assert len(parts) > 0
        for part in parts:
            assert part.endswith("USD")
            assert not part.endswith("USDT")

    def test_generate_usdt_suffix(self):
        data_dir = "trainingdata/train"
        p = Path(data_dir)
        if not p.exists() or not list(p.glob("*USDT.csv")):
            pytest.skip("No training data available")
        arg = generate_symbols_arg(data_dir, min_bars=100, min_avg_daily_volume_usd=0.0, use_usd_suffix=False)
        parts = arg.split()
        for part in parts:
            assert part.endswith("USDT")


class TestDownloadScript:
    def test_symbol_lists_in_download_script(self):
        from scripts.download_binance_data import SYMBOLS, ORIGINAL_30, EXPANDED_40
        assert len(ORIGINAL_30) == 30
        assert len(EXPANDED_40) == 40
        assert len(SYMBOLS) == 70
        assert "BTC" in SYMBOLS
        assert "HBAR" in SYMBOLS
        assert "TON" in SYMBOLS

    def test_no_duplicates_in_download_symbols(self):
        from scripts.download_binance_data import SYMBOLS
        assert len(SYMBOLS) == len(set(SYMBOLS))


class TestBackwardsCompatibility:
    def test_original_30_unchanged(self):
        expected = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
            "AAVEUSDT", "LTCUSDT", "XRPUSDT", "DOTUSDT", "UNIUSDT", "NEARUSDT",
            "APTUSDT", "ICPUSDT", "SHIBUSDT", "ADAUSDT", "FILUSDT", "ARBUSDT",
            "OPUSDT", "INJUSDT", "SUIUSDT", "TIAUSDT", "SEIUSDT", "ATOMUSDT",
            "ALGOUSDT", "BCHUSDT", "BNBUSDT", "TRXUSDT", "PEPEUSDT", "MATICUSDT",
        ]
        assert ORIGINAL_30_SYMBOLS == expected

    def test_original_30_are_first_in_default(self):
        assert DEFAULT_SYMBOLS[:30] == ORIGINAL_30_SYMBOLS

    def test_load_symbol_data_still_works(self):
        data_dir = "trainingdata/train"
        p = Path(data_dir)
        if not p.exists():
            pytest.skip("No training data available")
        df = load_symbol_data(data_dir, "BTCUSDT")
        if df is None:
            pytest.skip("BTCUSDT data not available")
        assert "timestamp" in df.columns
        assert "close" in df.columns
        assert len(df) > 30

    def test_load_symbol_data_supports_legacy_usd_filename(self, tmp_path):
        _write_symbol_csv(tmp_path / "INJUSD.csv")
        df = load_symbol_data(str(tmp_path), "INJUSDT")
        assert df is not None
        assert "timestamp" in df.columns
        assert len(df) == 400

    def test_compute_features_unchanged(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.rand(50) * 100 + 50,
            "high": np.random.rand(50) * 100 + 55,
            "low": np.random.rand(50) * 100 + 45,
            "close": np.random.rand(50) * 100 + 50,
            "volume": np.random.rand(50) * 1e6,
        })
        feats = compute_features(df)
        assert list(feats.columns) == FEATURE_NAMES
        assert len(feats) == 50
