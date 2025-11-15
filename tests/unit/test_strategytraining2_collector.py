from datetime import datetime
import json
import threading

from strategytraining2 import collector


def test_log_strategy_snapshot_writes(monkeypatch, tmp_path):
    target = tmp_path / "dataset.jsonl"
    monkeypatch.setattr(collector, "DATASET_PATH", target)
    monkeypatch.setattr(collector, "_LOCK", threading.Lock())

    data = {
        "strategy": "simple",
        "side": "buy",
        "mode": "normal",
        "trade_mode": "normal",
        "gate_config": "-",
        "avg_return": 0.02,
        "rolling_sharpe": 1.5,
        "rolling_sortino": 2.5,
        "predicted_close": 101.0,
    }

    collector.log_strategy_snapshot("AAPL", data, datetime(2025, 1, 2))

    lines = target.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["symbol"] == "AAPL"
    assert payload["strategy"] == "simple"
    assert payload["avg_return"] == 0.02
