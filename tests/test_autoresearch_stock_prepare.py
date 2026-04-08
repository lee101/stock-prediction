from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_prepare_main_help_documents_default_data_roots(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        stock_prepare.main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "trainingdatahourly/stocks" in output
    assert "trainingdata for daily" in output
    assert "built-in" in output and "hourly 8-symbol set" in output


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


def test_build_sequence_block_uses_ordered_windows_and_filters_nan_targets() -> None:
    frame = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "future_high_return": [0.1, 0.2, 0.3, np.nan, 0.5],
            "future_low_return": [-0.1, -0.2, -0.3, np.nan, -0.5],
            "future_close_return": [0.01, 0.02, 0.03, np.nan, 0.05],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T14:30:00Z",
                    "2024-01-02T15:30:00Z",
                    "2024-01-02T16:30:00Z",
                    "2024-01-02T17:30:00Z",
                    "2024-01-02T18:30:00Z",
                ],
                utc=True,
            ),
        }
    )

    sequences, targets, rows = stock_prepare._build_sequence_block(
        frame,
        feature_names=("f1", "f2"),
        sequence_length=3,
        row_mask=np.array([True, True, True, True, True], dtype=bool),
        include_targets=True,
    )

    assert sequences.shape == (2, 3, 2)
    np.testing.assert_allclose(
        sequences,
        np.array(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        targets,
        np.array(
            [
                [0.3, -0.3, 0.03],
                [0.5, -0.5, 0.05],
            ],
            dtype=np.float32,
        ),
    )
    assert rows["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02T16:30:00Z"),
        pd.Timestamp("2024-01-02T18:30:00Z"),
    ]


def test_lag_action_frame_shifts_actions_per_symbol_to_future_bars() -> None:
    actions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T14:30:00Z",
                    "2024-01-02T15:30:00Z",
                    "2024-01-02T14:30:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "side": ["buy", "sell", "buy"],
            "strength": [0.9, 0.4, 0.7],
        }
    )
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T14:30:00Z",
                    "2024-01-02T15:30:00Z",
                    "2024-01-02T16:30:00Z",
                    "2024-01-02T14:30:00Z",
                    "2024-01-02T15:30:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
            "close": [100.0, 101.0, 102.0, 200.0, 201.0],
        }
    )

    shifted = stock_prepare.lag_action_frame(actions, bars, lag=1)

    assert shifted["symbol"].tolist() == ["AAPL", "MSFT", "AAPL"]
    assert shifted["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02T15:30:00Z"),
        pd.Timestamp("2024-01-02T15:30:00Z"),
        pd.Timestamp("2024-01-02T16:30:00Z"),
    ]
    assert shifted["side"].tolist() == ["buy", "buy", "sell"]


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


def test_describe_task_inputs_uses_lightweight_metadata_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    _write_symbol_csv(data_root, "AAPL", _hourly_market_timestamps(16), sign=1.0)

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL",),
        data_root=data_root,
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )

    monkeypatch.setattr(
        stock_prepare,
        "_read_symbol_bars_from_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full loader should not run for input inspection")),
    )

    payload = stock_prepare.describe_task_inputs(config)

    assert payload["all_symbols_ready"] is True
    assert payload["symbols"][0]["status"] == "ready"
    assert payload["symbols"][0]["rows"] > 0


def test_resolve_task_config_rejects_path_like_symbol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Unsupported symbol"):
        resolve_task_config(
            frequency="hourly",
            symbols=("../secret",),
            data_root=tmp_path / "hourly",
            sequence_length=8,
            hold_bars=3,
            eval_windows=(8,),
            dashboard_db_path=tmp_path / "missing.db",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"recent_overlay_bars": -1}, "recent_overlay_bars must be non-negative"),
        ({"initial_cash": 0.0}, "initial_cash must be positive"),
        ({"max_positions": 0}, "max_positions must be at least 1"),
        ({"max_volume_fraction": 0.0}, "max_volume_fraction must be > 0 and <= 1"),
        ({"max_volume_fraction": 1.5}, "max_volume_fraction must be > 0 and <= 1"),
        ({"min_edge_bps": -0.5}, "min_edge_bps must be non-negative"),
        ({"entry_slippage_bps": -1.0}, "entry_slippage_bps must be non-negative"),
        ({"exit_slippage_bps": -1.0}, "exit_slippage_bps must be non-negative"),
        ({"spread_lookback_days": 0}, "spread_lookback_days must be at least 1"),
        ({"annual_leverage_rate": -0.01}, "annual_leverage_rate must be non-negative"),
        ({"max_gross_leverage": 0.5}, "max_gross_leverage must be at least 1.0"),
    ],
)
def test_resolve_task_config_rejects_invalid_numeric_overrides(
    tmp_path: Path,
    overrides: dict[str, float | int],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "frequency": "hourly",
        "symbols": ("AAPL",),
        "data_root": tmp_path / "hourly",
        "sequence_length": 8,
        "hold_bars": 3,
        "eval_windows": (8,),
        "dashboard_db_path": tmp_path / "missing.db",
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        resolve_task_config(**kwargs)


def test_resolve_task_input_check_workers_honors_env_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_STOCK_INPUT_CHECK_WORKERS", "99")

    assert stock_prepare._resolve_task_input_check_worker_config(3) == (3, "env")


def test_resolve_task_input_check_workers_rejects_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_STOCK_INPUT_CHECK_WORKERS", "not-an-int")

    with pytest.raises(ValueError, match="AUTORESEARCH_STOCK_INPUT_CHECK_WORKERS"):
        stock_prepare._resolve_task_input_check_worker_config(3)


def test_load_symbol_bars_error_lists_available_csvs_in_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    _write_symbol_csv(data_root, "DBX", _hourly_market_timestamps(16), sign=1.0)

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL",),
        data_root=data_root,
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )

    with pytest.raises(FileNotFoundError, match=r"data_root=.*DBX\.csv") as excinfo:
        stock_prepare._load_symbol_bars("AAPL", config)

    assert "Point --data-root at a directory containing per-symbol CSVs" in str(excinfo.value)


def test_load_symbol_bars_error_reports_missing_recent_overlay_root(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    recent_root = tmp_path / "recent"

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL",),
        data_root=data_root,
        recent_data_root=recent_root,
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )

    with pytest.raises(FileNotFoundError, match=r"recent_data_root=.*\(missing\)") as excinfo:
        stock_prepare._load_symbol_bars("AAPL", config)

    assert "checked:" in str(excinfo.value)


def test_describe_task_inputs_marks_invalid_csv_without_raising(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    (data_root / "AAPL.csv").write_text("timestamp,open,close\n2024-01-02T14:30:00Z,100,100.5\n", encoding="utf-8")

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL",),
        data_root=data_root,
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )

    payload = stock_prepare.describe_task_inputs(config)

    assert payload["all_symbols_ready"] is False
    assert payload["status_counts"] == {
        "ready": 0,
        "missing": 0,
        "invalid": 1,
        "partial": 0,
        "other": 0,
    }
    assert payload["symbols"][0]["status"] == "invalid"
    assert "missing required columns" in str(payload["symbols"][0]["primary_error"])


def test_describe_task_inputs_marks_invalid_recent_overlay_as_partial(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    recent_root = tmp_path / "recent"
    data_root.mkdir()
    recent_root.mkdir()
    _write_symbol_csv(data_root, "AAPL", _hourly_market_timestamps(16), sign=1.0)
    (recent_root / "AAPL.csv").write_text(
        "timestamp,open,close\n2024-01-02T14:30:00Z,100,100.5\n",
        encoding="utf-8",
    )

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL",),
        data_root=data_root,
        recent_data_root=recent_root,
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )

    payload = stock_prepare.describe_task_inputs(config)

    assert payload["all_symbols_ready"] is False
    assert payload["status_counts"] == {
        "ready": 0,
        "missing": 0,
        "invalid": 0,
        "partial": 1,
        "other": 0,
    }
    assert payload["symbols"][0]["status"] == "partial"
    assert payload["symbols"][0]["primary_rows"] > 0
    assert "missing required columns" in str(payload["symbols"][0]["recent_error"])


def test_run_task_input_check_renders_output_and_exit_code(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    _write_symbol_csv(data_root, "AAPL", _hourly_market_timestamps(16), sign=1.0)

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL",),
        data_root=data_root,
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )

    text_result = stock_prepare.run_task_input_check(config, text_output=True)
    json_result = stock_prepare.run_task_input_check(config, text_output=False)

    assert text_result.exit_code == 0
    assert "Autoresearch Stock Input Check" in text_result.rendered_output
    assert text_result.payload["all_symbols_ready"] is True
    assert json.loads(json_result.rendered_output)["all_symbols_ready"] is True
    assert json_result.exit_code == 0


def test_task_input_symbol_paths_normalizes_symbol_and_recent_root(tmp_path: Path) -> None:
    config = resolve_task_config(
        frequency="hourly",
        symbols=("aapl",),
        data_root=tmp_path / "hourly",
        recent_data_root=tmp_path / "recent",
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )

    symbol_paths = stock_prepare._task_input_symbol_paths(config, "aapl")

    assert symbol_paths.symbol == "AAPL"
    assert symbol_paths.primary == config.data_root / "AAPL.csv"
    assert symbol_paths.recent == config.recent_data_root / "AAPL.csv"


def test_describe_task_inputs_parallel_path_preserves_symbol_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = resolve_task_config(
        frequency="hourly",
        symbols=("MSFT", "AAPL"),
        data_root=tmp_path / "hourly",
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )
    monkeypatch.setattr(
        stock_prepare,
        "_resolve_task_input_check_worker_config",
        lambda _count: (2, stock_prepare.TaskInputWorkerSource.AUTO),
    )

    seen: dict[str, object] = {}

    class _FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            seen["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def map(self, fn, values):
            return [fn(value) for value in values]

    def _fake_describe_symbol(_config, symbol_name: str) -> dict[str, object]:
        return {
            "symbol": symbol_name,
            "status": stock_prepare.TaskInputStatus.READY,
            "rows": 1,
            "start_timestamp": None,
            "end_timestamp": None,
            "primary_path": f"{symbol_name}.csv",
            "primary_rows": 1,
            "primary_error": None,
            "recent_path": None,
            "recent_rows": 0,
            "recent_overlay_rows": 0,
            "recent_error": None,
        }

    monkeypatch.setattr(stock_prepare, "ThreadPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(stock_prepare, "_describe_task_input_symbol", _fake_describe_symbol)

    payload = stock_prepare.describe_task_inputs(config)

    assert seen["max_workers"] == 2
    assert payload["worker_count"] == 2
    assert payload["worker_source"] is stock_prepare.TaskInputWorkerSource.AUTO
    assert [row["symbol"] for row in payload["symbols"]] == ["MSFT", "AAPL"]
    assert payload["symbols"][0]["status"] is stock_prepare.TaskInputStatus.READY
    assert payload["symbols"][1]["status"] is stock_prepare.TaskInputStatus.READY
    assert payload["status_counts"]["ready"] == 2


def test_describe_task_inputs_parallel_path_isolates_unexpected_symbol_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = resolve_task_config(
        frequency="hourly",
        symbols=("MSFT", "AAPL"),
        data_root=tmp_path / "hourly",
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )
    monkeypatch.setattr(
        stock_prepare,
        "_resolve_task_input_check_worker_config",
        lambda _count: (2, stock_prepare.TaskInputWorkerSource.AUTO),
    )

    class _FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def map(self, fn, values):
            return [fn(value) for value in values]

    def _fake_describe_symbol(_config, symbol_name: str) -> dict[str, object]:
        if symbol_name == "AAPL":
            raise RuntimeError("boom")
        return {
            "symbol": symbol_name,
            "status": stock_prepare.TaskInputStatus.READY,
            "rows": 1,
            "start_timestamp": None,
            "end_timestamp": None,
            "primary_path": f"{symbol_name}.csv",
            "primary_rows": 1,
            "primary_error": None,
            "recent_path": None,
            "recent_rows": 0,
            "recent_overlay_rows": 0,
            "recent_error": None,
        }

    monkeypatch.setattr(stock_prepare, "ThreadPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(stock_prepare, "_describe_task_input_symbol", _fake_describe_symbol)

    payload = stock_prepare.describe_task_inputs(config)

    assert [row["symbol"] for row in payload["symbols"]] == ["MSFT", "AAPL"]
    assert payload["symbols"][0]["status"] is stock_prepare.TaskInputStatus.READY
    assert payload["symbols"][1]["status"] is stock_prepare.TaskInputStatus.INVALID
    assert "Unexpected input inspection failure: boom" in str(payload["symbols"][1]["primary_error"])
    assert payload["status_counts"] == {
        "ready": 1,
        "missing": 0,
        "invalid": 1,
        "partial": 0,
        "other": 0,
    }


def test_load_symbol_bars_prefers_recent_overlay_on_duplicate_timestamp(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    recent_root = tmp_path / "recent"
    data_root.mkdir()
    recent_root.mkdir()
    timestamps = _hourly_market_timestamps(4)
    _write_symbol_csv(data_root, "AAPL", timestamps, sign=1.0)
    _write_symbol_csv(recent_root, "AAPL", timestamps[-2:], sign=-1.0)

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL",),
        data_root=data_root,
        recent_data_root=recent_root,
        recent_overlay_bars=8,
        sequence_length=4,
        hold_bars=1,
        eval_windows=(4,),
        dashboard_db_path=tmp_path / "missing.db",
    )

    combined = stock_prepare._load_symbol_bars("AAPL", config)

    assert len(combined) == 4
    assert combined.iloc[-1]["timestamp"] == timestamps[-1]
    assert combined.iloc[-1]["close"] < combined.iloc[-2]["close"]


def test_load_symbol_bars_surfaces_recent_overlay_context(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    recent_root = tmp_path / "recent"
    data_root.mkdir()
    recent_root.mkdir()
    _write_symbol_csv(data_root, "AAPL", _hourly_market_timestamps(16), sign=1.0)
    (recent_root / "AAPL.csv").write_text(
        "timestamp,open,close\n2024-01-02T14:30:00Z,100,100.5\n",
        encoding="utf-8",
    )

    config = resolve_task_config(
        frequency="hourly",
        symbols=("AAPL",),
        data_root=data_root,
        recent_data_root=recent_root,
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8,),
        dashboard_db_path=tmp_path / "missing.db",
    )

    with pytest.raises(RuntimeError, match=r"Failed to load recent overlay dataset for AAPL") as excinfo:
        stock_prepare._load_symbol_bars("AAPL", config)

    assert "AAPL.csv" in str(excinfo.value)


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
