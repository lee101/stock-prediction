"""Tests for audit_stock_splits.py

Tests are self-contained and do not require access to trainingdata/ or
pufferlib_market/data/ (which are gitignored).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import helpers from the module under test
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_stock_splits import (
    AuditRecord,
    REAL_EVENT_LOOKUP,
    MIN_SPLIT_FACTOR,
    _apply_split_to_csv,
    _backup_csv,
    _is_crypto_symbol,
    _is_valid_ticker,
    _load_csv_normalized,
    _price_ratio_at_split,
    _scan_big_drops,
    audit_symbol,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_csv(rows: list[dict], path: Path) -> Path:
    """Write a minimal OHLCV CSV to *path* and return *path*."""
    fieldnames = ["timestamp", "open", "high", "low", "close", "volume"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            # fill missing price/volume columns with defaults
            w.writerow({k: row.get(k, 1.0) for k in fieldnames})
    return path


def _row(date_str: str, close: float, volume: float = 1_000_000.0) -> dict:
    return {
        "timestamp": f"{date_str} 05:00:00+00:00",
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
    }


# ---------------------------------------------------------------------------
# _is_valid_ticker
# ---------------------------------------------------------------------------


class TestIsValidTicker:
    def test_standard_tickers(self):
        for sym in ["AAPL", "MSFT", "GOOG", "A", "V", "SPY", "QQQ"]:
            assert _is_valid_ticker(sym), sym

    def test_dot_tickers(self):
        assert _is_valid_ticker("BRK.B")
        assert _is_valid_ticker("BF.B")

    def test_hyphen_ticker(self):
        assert _is_valid_ticker("BRK-B")

    def test_digit_ticker(self):
        assert _is_valid_ticker("MP2")

    def test_comma_rejected(self):
        assert not _is_valid_ticker("AAPL,MSFT")

    def test_lowercase_rejected(self):
        assert not _is_valid_ticker("aapl")
        assert not _is_valid_ticker("download_summary")

    def test_empty_rejected(self):
        assert not _is_valid_ticker("")


# ---------------------------------------------------------------------------
# _is_crypto_symbol
# ---------------------------------------------------------------------------


class TestIsCryptoSymbol:
    def test_crypto_pair(self):
        assert _is_crypto_symbol("BTCUSD")
        assert _is_crypto_symbol("ETHUSD")
        assert _is_crypto_symbol("SOLUSD")
        assert _is_crypto_symbol("AAVEUSD")

    def test_stock_not_crypto(self):
        assert not _is_crypto_symbol("MSFT")
        assert not _is_crypto_symbol("NVDA")
        assert not _is_crypto_symbol("AAPL")

    def test_comma_is_crypto_like(self):
        assert _is_crypto_symbol("AAPL,MSFT")


# ---------------------------------------------------------------------------
# _load_csv_normalized
# ---------------------------------------------------------------------------


class TestLoadCsvNormalized:
    def test_parses_timestamps(self, tmp_path):
        p = _make_csv([_row("2022-01-03", 100.0), _row("2022-01-04", 102.0)], tmp_path / "X.csv")
        df = _load_csv_normalized(p)
        assert len(df) == 2
        assert str(df["timestamp"].dt.tz) == "UTC"
        # All timestamps normalized to midnight
        assert (df["timestamp"].dt.time == pd.Timestamp("00:00:00").time()).all()

    def test_missing_timestamp_column_returns_empty(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("symbol,price\nAAPL,100\n")
        df = _load_csv_normalized(bad_csv)
        assert df.empty

    def test_sorted_by_date(self, tmp_path):
        p = _make_csv(
            [_row("2022-01-05", 105.0), _row("2022-01-03", 100.0), _row("2022-01-04", 102.0)],
            tmp_path / "X.csv",
        )
        df = _load_csv_normalized(p)
        dates = df["timestamp"].tolist()
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# _price_ratio_at_split
# ---------------------------------------------------------------------------


class TestPriceRatioAtSplit:
    def _df(self, rows: list[tuple[str, float]]) -> pd.DataFrame:
        data = {
            "timestamp": pd.to_datetime([r[0] for r in rows], utc=True),
            "close": [r[1] for r in rows],
        }
        df = pd.DataFrame(data)
        df["timestamp"] = df["timestamp"].dt.normalize()
        return df.sort_values("timestamp").reset_index(drop=True)

    def test_10_to_1_split(self):
        df = self._df([("2024-06-09", 120.0), ("2024-06-10", 12.0)])
        split_ts = pd.Timestamp("2024-06-10", tz="UTC")
        ratio = _price_ratio_at_split(df, split_ts)
        assert ratio == pytest.approx(0.1, rel=0.01)

    def test_already_adjusted(self):
        df = self._df([("2024-06-09", 12.05), ("2024-06-10", 12.10)])
        split_ts = pd.Timestamp("2024-06-10", tz="UTC")
        ratio = _price_ratio_at_split(df, split_ts)
        assert ratio == pytest.approx(1.004, rel=0.01)

    def test_no_data_after_split_returns_none(self):
        df = self._df([("2024-06-09", 120.0)])
        split_ts = pd.Timestamp("2024-06-10", tz="UTC")
        assert _price_ratio_at_split(df, split_ts) is None

    def test_split_before_data_start_returns_none(self):
        df = self._df([("2024-07-01", 120.0), ("2024-07-02", 12.0)])
        split_ts = pd.Timestamp("2024-06-10", tz="UTC")  # before data
        assert _price_ratio_at_split(df, split_ts) is None


# ---------------------------------------------------------------------------
# _scan_big_drops
# ---------------------------------------------------------------------------


class TestScanBigDrops:
    def _df(self, rows: list[tuple[str, float]]) -> pd.DataFrame:
        data = {
            "timestamp": pd.to_datetime([r[0] for r in rows], utc=True),
            "close": [r[1] for r in rows],
        }
        df = pd.DataFrame(data)
        df["timestamp"] = df["timestamp"].dt.normalize()
        return df.sort_values("timestamp").reset_index(drop=True)

    def test_detects_35_pct_drop(self):
        df = self._df([("2022-04-19", 200.0), ("2022-04-20", 130.0)])
        drops = _scan_big_drops(df)
        assert len(drops) == 1
        date_str, pct = drops[0]
        assert date_str == "2022-04-20"
        assert pct == pytest.approx(-0.35, rel=0.01)

    def test_small_drop_not_flagged(self):
        df = self._df([("2022-01-03", 100.0), ("2022-01-04", 95.0)])
        assert _scan_big_drops(df) == []

    def test_duplicate_rows_per_day_deduped(self):
        # Same date, two prices — should only compare across day boundaries
        data = {
            "timestamp": pd.to_datetime(
                ["2022-01-03", "2022-01-03", "2022-01-04"], utc=True
            ),
            "close": [400.0, 40.0, 38.0],
        }
        df = pd.DataFrame(data)
        # After dedup by last, 2022-01-03 = 40.0, 2022-01-04 = 38.0 → only -5% drop
        drops = _scan_big_drops(df)
        assert drops == []

    def test_no_drops_in_flat_series(self):
        rows = [(f"2022-01-{i:02d}", 100.0) for i in range(3, 20)]
        df = self._df(rows)
        assert _scan_big_drops(df) == []


# ---------------------------------------------------------------------------
# _backup_csv
# ---------------------------------------------------------------------------


class TestBackupCsv:
    def test_creates_backup(self, tmp_path):
        csv_path = tmp_path / "TEST.csv"
        csv_path.write_text("timestamp,close\n2022-01-03,100\n")
        _backup_csv(csv_path)
        backup = csv_path.with_suffix(".csv.pre_split_backup")
        assert backup.exists()
        assert backup.read_text() == csv_path.read_text()

    def test_does_not_overwrite_existing_backup(self, tmp_path):
        csv_path = tmp_path / "TEST.csv"
        csv_path.write_text("timestamp,close\n2022-01-03,100\n")
        backup = csv_path.with_suffix(".csv.pre_split_backup")
        backup.write_text("ORIGINAL BACKUP")
        _backup_csv(csv_path)  # should not overwrite
        assert backup.read_text() == "ORIGINAL BACKUP"


# ---------------------------------------------------------------------------
# _apply_split_to_csv
# ---------------------------------------------------------------------------


class TestApplySplitToCsv:
    def test_10_to_1_split_divides_prices_multiplies_volume(self, tmp_path):
        rows = [
            _row("2024-06-08", 1000.0, volume=100_000.0),
            _row("2024-06-09", 1020.0, volume=110_000.0),
            _row("2024-06-10", 102.0, volume=1_100_000.0),   # post-split day
            _row("2024-06-11", 103.0, volume=1_200_000.0),
        ]
        p = _make_csv(rows, tmp_path / "NVDA.csv")
        _apply_split_to_csv(p, "2024-06-10", 10.0)

        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        pre = df[df["timestamp"] < pd.Timestamp("2024-06-10", tz="UTC")]
        post = df[df["timestamp"] >= pd.Timestamp("2024-06-10", tz="UTC")]

        # Pre-split prices should be divided by 10
        assert pre.iloc[0]["close"] == pytest.approx(100.0, rel=1e-4)
        assert pre.iloc[1]["close"] == pytest.approx(102.0, rel=1e-4)
        # Pre-split volume should be multiplied by 10
        assert pre.iloc[0]["volume"] == pytest.approx(1_000_000.0, rel=1e-4)
        # Post-split prices unchanged
        assert post.iloc[0]["close"] == pytest.approx(102.0, rel=1e-4)
        # Post-split volume unchanged
        assert post.iloc[0]["volume"] == pytest.approx(1_100_000.0, rel=1e-4)

    def test_creates_backup_before_modifying(self, tmp_path):
        rows = [_row("2024-06-09", 1000.0), _row("2024-06-10", 100.0)]
        p = _make_csv(rows, tmp_path / "NVDA.csv")
        original_content = p.read_text()
        _apply_split_to_csv(p, "2024-06-10", 10.0)
        backup = p.with_suffix(".csv.pre_split_backup")
        assert backup.exists()
        assert backup.read_text() == original_content


# ---------------------------------------------------------------------------
# audit_symbol
# ---------------------------------------------------------------------------


class TestAuditSymbol:
    def _make_splits(self, items: list[tuple[str, float]]) -> pd.Series:
        idx = pd.DatetimeIndex(
            [pd.Timestamp(d, tz="America/New_York") for d, _ in items]
        )
        return pd.Series([f for _, f in items], index=idx, name="Stock Splits")

    def test_unadjusted_split_detected_and_fixed(self, tmp_path):
        """A 10:1 split that hasn't been applied should be FIXED."""
        rows = [
            _row("2024-06-08", 1000.0),
            _row("2024-06-09", 1020.0),
            _row("2024-06-10", 102.0),   # price drops by ~1/10 on split day
            _row("2024-06-11", 103.0),
        ]
        p = _make_csv(rows, tmp_path / "NVDA.csv")
        splits = self._make_splits([("2024-06-10", 10.0)])

        records = audit_symbol("NVDA", p, None, splits, dry_run=False)
        fixed = [r for r in records if r.status == "FIXED"]
        assert len(fixed) == 1
        assert fixed[0].factor == 10.0
        assert fixed[0].split_date == "2024-06-10"

    def test_already_adjusted_split_is_ok(self, tmp_path):
        """A split already applied should be reported OK (no big price drop)."""
        rows = [
            _row("2024-06-08", 102.0),
            _row("2024-06-09", 103.0),
            _row("2024-06-10", 104.0),   # no drop — already adjusted
            _row("2024-06-11", 103.5),
        ]
        p = _make_csv(rows, tmp_path / "NVDA.csv")
        splits = self._make_splits([("2024-06-10", 10.0)])

        records = audit_symbol("NVDA", p, None, splits, dry_run=False)
        ok_recs = [r for r in records if r.status == "OK"]
        fixed_recs = [r for r in records if "FIXED" in r.status]
        assert len(ok_recs) == 1
        assert len(fixed_recs) == 0

    def test_dry_run_does_not_modify_file(self, tmp_path):
        rows = [
            _row("2024-06-09", 1020.0),
            _row("2024-06-10", 102.0),
        ]
        p = _make_csv(rows, tmp_path / "NVDA.csv")
        original_content = p.read_text()
        splits = self._make_splits([("2024-06-10", 10.0)])

        records = audit_symbol("NVDA", p, None, splits, dry_run=True)
        assert p.read_text() == original_content  # file unchanged
        assert any("dry" in r.status.lower() for r in records)

    def test_real_event_classified_correctly(self, tmp_path):
        """NFLX 2022-04-20 earnings crash should be REAL_EVENT."""
        rows = [
            _row("2022-04-19", 348.0),
            _row("2022-04-20", 226.0),  # -35% earnings crash
            _row("2022-04-21", 230.0),
        ]
        p = _make_csv(rows, tmp_path / "NFLX.csv")
        splits = self._make_splits([])  # no splits from yfinance in this period

        records = audit_symbol("NFLX", p, None, splits, dry_run=True)
        real_events = [r for r in records if r.status == "REAL_EVENT"]
        assert len(real_events) == 1
        assert real_events[0].split_date == "2022-04-20"

    def test_unrecognized_drop_flagged(self, tmp_path):
        rows = [
            _row("2022-05-10", 100.0),
            _row("2022-05-11", 65.0),   # -35%, unknown cause
        ]
        p = _make_csv(rows, tmp_path / "UNKWN.csv")
        splits = self._make_splits([])

        records = audit_symbol("UNKWN", p, None, splits, dry_run=True)
        unrecog = [r for r in records if r.status == "UNRECOGNIZED_DROP"]
        assert len(unrecog) == 1

    def test_split_before_csv_start_skipped(self, tmp_path):
        """A split that occurred before our data starts should be silently skipped."""
        rows = [
            _row("2024-01-03", 50.0),
            _row("2024-01-04", 51.0),
        ]
        p = _make_csv(rows, tmp_path / "AAPL.csv")
        # Split in 2020 — before our CSV data
        splits = self._make_splits([("2020-08-31", 4.0)])

        records = audit_symbol("AAPL", p, None, splits, dry_run=True)
        assert records == []

    def test_spin_off_adjustment_not_treated_as_split(self, tmp_path):
        """Fractional yfinance 'splits' (e.g. 1.128 spin-off) should be ignored."""
        rows = [
            _row("2023-10-01", 250.0),
            _row("2023-10-02", 222.0),  # ~11% drop, could match 1/1.128
            _row("2023-10-03", 225.0),
        ]
        p = _make_csv(rows, tmp_path / "DHR.csv")
        # 1.128 is a spin-off adjustment (below MIN_SPLIT_FACTOR threshold)
        splits = self._make_splits([("2023-10-02", 1.128)])

        records = audit_symbol("DHR", p, None, splits, dry_run=True)
        fixed = [r for r in records if "FIXED" in r.status]
        assert len(fixed) == 0, f"Spin-off should not be FIXED: {records}"

    def test_empty_csv_handled_gracefully(self, tmp_path):
        """A CSV with no data rows should not crash the audit."""
        p = tmp_path / "EMPTY.csv"
        p.write_text("timestamp,open,high,low,close,volume\n")
        splits = self._make_splits([("2024-06-10", 10.0)])
        records = audit_symbol("EMPTY", p, None, splits, dry_run=True)
        assert isinstance(records, list)

    def test_csv_without_timestamp_col_handled_gracefully(self, tmp_path):
        """A CSV without a timestamp column should not crash."""
        p = tmp_path / "BAD.csv"
        p.write_text("symbol,price\nBAD,100\n")
        splits = self._make_splits([("2024-06-10", 10.0)])
        records = audit_symbol("BAD", p, None, splits, dry_run=True)
        assert isinstance(records, list)


# ---------------------------------------------------------------------------
# MIN_SPLIT_FACTOR
# ---------------------------------------------------------------------------


def test_min_split_factor_filters_fractional_splits():
    """Ensure MIN_SPLIT_FACTOR is between 1.0 and 2.0 to catch real splits only."""
    assert 1.0 < MIN_SPLIT_FACTOR < 2.0


# ---------------------------------------------------------------------------
# REAL_EVENT_LOOKUP
# ---------------------------------------------------------------------------


def test_known_real_events_populated():
    assert ("NFLX", "2022-04-20") in REAL_EVENT_LOOKUP
    assert ("META", "2022-02-03") in REAL_EVENT_LOOKUP
    assert ("INTC", "2024-08-01") in REAL_EVENT_LOOKUP
