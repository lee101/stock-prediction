from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.autoresearch_stock.prepare as stock_prepare
from src.autoresearch_stock.prepare import ISO_FORMAT, evaluate_model, load_live_spread_profile, prepare_task, resolve_task_config
from src.trade_stock_utils import expected_cost_bps


def _hourly_market_timestamps(count: int) -> list[pd.Timestamp]:
    timestamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2024-01-02 00:00:00+00:00")
    market_hours = [14, 15, 16, 17, 18, 19, 20]
    while len(timestamps) < count:
        if day.weekday() < 5:
            for hour in market_hours:
                timestamps.append(day + pd.Timedelta(hours=hour, minutes=30))
                if len(timestamps) >= count:
                    break
        day += pd.Timedelta(days=1)
    return timestamps


def _daily_timestamps(count: int) -> list[pd.Timestamp]:
    timestamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2024-01-02 05:00:00+00:00")
    while len(timestamps) < count:
        if day.weekday() < 5:
            timestamps.append(day)
        day += pd.Timedelta(days=1)
    return timestamps


def _write_symbol_csv(root: Path, symbol: str, timestamps: list[pd.Timestamp], *, sign: float) -> None:
    rows: list[dict[str, float | str]] = []
    price = 100.0 + (2.5 if sign > 0 else 7.5)
    for index, timestamp in enumerate(timestamps):
        drift = sign * 0.002 + 0.0008 * np.sin(index / 4.0)
        open_price = price
        close_price = open_price * (1.0 + drift)
        high_price = max(open_price, close_price) * 1.0025
        low_price = min(open_price, close_price) * 0.9975
        volume = 1_000_000.0 + index * 1_000.0
        rows.append(
            {
                "timestamp": str(timestamp),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "vwap": (open_price + close_price) / 2.0,
                "symbol": symbol,
            }
        )
        price = close_price
    pd.DataFrame(rows).to_csv(root / f"{symbol}.csv", index=False)


class _ZeroModel(torch.nn.Module):
    def forward(self, features: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        del symbol_ids
        return torch.zeros((batch_size, 3), dtype=torch.float32)


def test_load_live_spread_profile_uses_db_and_fallback(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE spread_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            bid REAL,
            ask REAL,
            spread_ratio REAL NOT NULL,
            spread_absolute REAL,
            spread_bps REAL
        )
        """
    )
    now = datetime.now(tz=timezone.utc)
    for index in range(10):
        recorded_at = (now - timedelta(hours=index)).strftime(ISO_FORMAT)
        conn.execute(
            """
            INSERT INTO spread_observations (
                recorded_at, symbol, bid, ask, spread_ratio, spread_absolute, spread_bps
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (recorded_at, "AAPL", 100.0, 100.1, 1.001, 0.1, 12.5),
        )
    conn.commit()
    conn.close()

    profile = load_live_spread_profile(
        ["AAPL", "DBX"],
        db_path=db_path,
        lookback_days=7,
        now=now,
    )

    assert profile["AAPL"] == 12.5
    assert profile["DBX"] == float(expected_cost_bps("DBX"))


def test_prepare_task_and_evaluate_model_hourly(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    timestamps = _hourly_market_timestamps(96)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(data_root, "DBX", timestamps, sign=-1.0)

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL", "DBX"),
        data_root=data_root,
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8, 16),
        max_positions=2,
        dashboard_db_path=tmp_path / "missing.db",
    )
    task = prepare_task(config)

    assert task.train_features.shape[0] > 0
    assert task.train_features.shape[1] == 8
    assert task.train_features.shape[2] == len(task.feature_names)
    assert len(task.scenarios) == 2

    result = evaluate_model(_ZeroModel(), task, device=torch.device("cpu"), batch_size=32)
    summary = result["summary"]
    assert summary["scenario_count"] == 2.0
    assert np.isfinite(summary["robust_score"])


def test_prepare_task_hourly_does_not_depend_on_path_exists_precheck(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    timestamps = _hourly_market_timestamps(96)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(data_root, "DBX", timestamps, sign=-1.0)

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL", "DBX"),
        data_root=data_root,
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8, 16),
        max_positions=2,
        dashboard_db_path=tmp_path / "missing.db",
    )
    original_exists = Path.exists

    def _fake_exists(self: Path) -> bool:
        if self.name in {"AAPL.csv", "DBX.csv"} and self.parent == data_root:
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", _fake_exists)

    task = prepare_task(config)

    assert task.train_features.shape[0] > 0
    assert len(task.scenarios) == 2


def test_try_read_symbol_bars_from_path_returns_none_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    assert stock_prepare._try_read_symbol_bars_from_path(missing, "AAPL") is None


def test_prepare_task_daily(tmp_path: Path) -> None:
    data_root = tmp_path / "daily"
    data_root.mkdir()
    timestamps = _daily_timestamps(220)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(data_root, "DBX", timestamps, sign=-1.0)

    config = resolve_task_config(
        frequency="daily",
        symbols=("AAPL", "DBX"),
        data_root=data_root,
        sequence_length=12,
        hold_bars=5,
        eval_windows=(20, 40),
        max_positions=2,
        dashboard_db_path=tmp_path / "missing.db",
    )
    task = prepare_task(config)

    assert task.train_features.shape[0] > 0
    assert len(task.scenarios) == 2
