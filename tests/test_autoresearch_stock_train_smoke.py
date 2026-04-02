from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="autoresearch trainer requires CUDA")


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


def test_train_smoke_hourly(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    timestamps = _hourly_market_timestamps(96)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(data_root, "DBX", timestamps, sign=-1.0)

    env = os.environ.copy()
    env["AUTORESEARCH_STOCK_TIME_BUDGET_SECONDS"] = "1"
    env["AUTORESEARCH_STOCK_CHECKPOINT_ROOT"] = str(tmp_path / "checkpoints")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_stock.train",
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL,DBX",
            "--sequence-length",
            "8",
            "--hold-bars",
            "3",
            "--eval-windows",
            "8,16",
            "--max-positions",
            "2",
            "--batch-size",
            "16",
            "--eval-batch-size",
            "16",
            "--hidden-size",
            "16",
            "--layers",
            "1",
            "--disable-auto-lr-find",
            "--dashboard-db",
            str(tmp_path / "missing.db"),
        ],
        cwd="/nvme0n1-disk/code/stock-prediction",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "robust_score:" in proc.stdout
    assert "training_seconds:" in proc.stdout


def test_train_smoke_hourly_timestamp_budget_head(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    timestamps = _hourly_market_timestamps(96)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(data_root, "DBX", timestamps, sign=-1.0)

    env = os.environ.copy()
    env["AUTORESEARCH_STOCK_TIME_BUDGET_SECONDS"] = "1"
    env["AUTORESEARCH_STOCK_CHECKPOINT_ROOT"] = str(tmp_path / "checkpoints")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_stock.train",
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL,DBX",
            "--sequence-length",
            "8",
            "--hold-bars",
            "3",
            "--eval-windows",
            "8,16",
            "--max-positions",
            "2",
            "--batch-size",
            "16",
            "--eval-batch-size",
            "16",
            "--hidden-size",
            "16",
            "--layers",
            "1",
            "--disable-auto-lr-find",
            "--dashboard-db",
            str(tmp_path / "missing.db"),
            "--timestamp-budget-head",
        ],
        cwd="/nvme0n1-disk/code/stock-prediction",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "robust_score:" in proc.stdout
    assert "training_seconds:" in proc.stdout


def test_train_smoke_hourly_budget_guided_keep_count(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    timestamps = _hourly_market_timestamps(96)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(data_root, "DBX", timestamps, sign=-1.0)

    env = os.environ.copy()
    env["AUTORESEARCH_STOCK_TIME_BUDGET_SECONDS"] = "1"
    env["AUTORESEARCH_STOCK_CHECKPOINT_ROOT"] = str(tmp_path / "checkpoints")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_stock.train",
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL,DBX",
            "--sequence-length",
            "8",
            "--hold-bars",
            "3",
            "--eval-windows",
            "8,16",
            "--max-positions",
            "2",
            "--batch-size",
            "16",
            "--eval-batch-size",
            "16",
            "--hidden-size",
            "16",
            "--layers",
            "1",
            "--disable-auto-lr-find",
            "--dashboard-db",
            str(tmp_path / "missing.db"),
            "--budget-guided-keep-count",
        ],
        cwd="/nvme0n1-disk/code/stock-prediction",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "robust_score:" in proc.stdout
    assert "training_seconds:" in proc.stdout


def test_train_smoke_hourly_continuous_budget_thresholds(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    timestamps = _hourly_market_timestamps(96)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(data_root, "DBX", timestamps, sign=-1.0)

    env = os.environ.copy()
    env["AUTORESEARCH_STOCK_TIME_BUDGET_SECONDS"] = "1"
    env["AUTORESEARCH_STOCK_CHECKPOINT_ROOT"] = str(tmp_path / "checkpoints")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_stock.train",
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL,DBX",
            "--sequence-length",
            "8",
            "--hold-bars",
            "3",
            "--eval-windows",
            "8,16",
            "--max-positions",
            "2",
            "--batch-size",
            "16",
            "--eval-batch-size",
            "16",
            "--hidden-size",
            "16",
            "--layers",
            "1",
            "--disable-auto-lr-find",
            "--dashboard-db",
            str(tmp_path / "missing.db"),
            "--continuous-budget-thresholds",
        ],
        cwd="/nvme0n1-disk/code/stock-prediction",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "robust_score:" in proc.stdout
    assert "training_seconds:" in proc.stdout


def test_train_smoke_hourly_budget_entropy_confidence(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    timestamps = _hourly_market_timestamps(96)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(data_root, "DBX", timestamps, sign=-1.0)

    env = os.environ.copy()
    env["AUTORESEARCH_STOCK_TIME_BUDGET_SECONDS"] = "1"
    env["AUTORESEARCH_STOCK_CHECKPOINT_ROOT"] = str(tmp_path / "checkpoints")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_stock.train",
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL,DBX",
            "--sequence-length",
            "8",
            "--hold-bars",
            "3",
            "--eval-windows",
            "8,16",
            "--max-positions",
            "2",
            "--batch-size",
            "16",
            "--eval-batch-size",
            "16",
            "--hidden-size",
            "16",
            "--layers",
            "1",
            "--disable-auto-lr-find",
            "--dashboard-db",
            str(tmp_path / "missing.db"),
            "--budget-entropy-confidence",
        ],
        cwd="/nvme0n1-disk/code/stock-prediction",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "robust_score:" in proc.stdout
    assert "training_seconds:" in proc.stdout


def test_train_smoke_hourly_budget_consensus_dispersion(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    timestamps = _hourly_market_timestamps(96)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(data_root, "DBX", timestamps, sign=-1.0)

    env = os.environ.copy()
    env["AUTORESEARCH_STOCK_TIME_BUDGET_SECONDS"] = "1"
    env["AUTORESEARCH_STOCK_CHECKPOINT_ROOT"] = str(tmp_path / "checkpoints")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoresearch_stock.train",
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL,DBX",
            "--sequence-length",
            "8",
            "--hold-bars",
            "3",
            "--eval-windows",
            "8,16",
            "--max-positions",
            "2",
            "--batch-size",
            "16",
            "--eval-batch-size",
            "16",
            "--hidden-size",
            "16",
            "--layers",
            "1",
            "--disable-auto-lr-find",
            "--dashboard-db",
            str(tmp_path / "missing.db"),
            "--budget-consensus-dispersion",
        ],
        cwd="/nvme0n1-disk/code/stock-prediction",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "robust_score:" in proc.stdout
    assert "training_seconds:" in proc.stdout
