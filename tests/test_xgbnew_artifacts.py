from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
from xgbnew import artifacts


def test_write_json_atomic_overwrites_and_cleans_temp(tmp_path) -> None:
    out = tmp_path / "summary.json"
    out.write_text('{"old": true}\n', encoding="utf-8")

    artifacts.write_json_atomic(out, {"ok": True})

    assert json.loads(out.read_text(encoding="utf-8")) == {"ok": True}
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_write_json_atomic_supports_default_serializer(tmp_path) -> None:
    out = tmp_path / "summary.json"

    artifacts.write_json_atomic(out, {"path": Path("model.pt")}, default=str)

    assert json.loads(out.read_text(encoding="utf-8")) == {"path": "model.pt"}
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_write_json_atomic_supports_sorted_keys(tmp_path) -> None:
    out = tmp_path / "summary.json"

    artifacts.write_json_atomic(out, {"z": 1, "a": 2}, sort_keys=True)

    assert out.read_text(encoding="utf-8").splitlines()[1].strip() == '"a": 2,'
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_write_pickle_atomic_overwrites_and_cleans_temp(tmp_path) -> None:
    out = tmp_path / "model.pkl"
    out.write_bytes(b"old")

    artifacts.write_pickle_atomic(out, {"model": "new"})

    with out.open("rb") as handle:
        assert pickle.load(handle) == {"model": "new"}
    assert list(tmp_path.glob(".model.pkl.*.tmp")) == []


def test_write_dataframe_csv_atomic_overwrites_and_cleans_temp(tmp_path) -> None:
    out = tmp_path / "trades.csv"
    out.write_text("old,content\n", encoding="utf-8")

    artifacts.write_dataframe_csv_atomic(out, pd.DataFrame([{"symbol": "AAA", "fee_rate": 0.001}]))

    frame = pd.read_csv(out)
    assert frame.to_dict("records") == [{"symbol": "AAA", "fee_rate": 0.001}]
    assert list(tmp_path.glob(".trades.csv.*.tmp")) == []


def test_write_dict_rows_csv_atomic_overwrites_and_cleans_temp(tmp_path) -> None:
    out = tmp_path / "leaderboard.csv"
    out.write_text("old,content\n", encoding="utf-8")

    artifacts.write_dict_rows_csv_atomic(
        out,
        [{"checkpoint": "run_a/best.pt", "sortino": 1.5}],
        fieldnames=["checkpoint", "sortino"],
    )

    frame = pd.read_csv(out)
    assert frame.to_dict("records") == [{"checkpoint": "run_a/best.pt", "sortino": 1.5}]
    assert list(tmp_path.glob(".leaderboard.csv.*.tmp")) == []


def test_write_text_atomic_cleans_temp_on_replace_failure(tmp_path, monkeypatch) -> None:
    out = tmp_path / "summary.json"

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(artifacts, "_replace_path", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        artifacts.write_json_atomic(out, {"ok": True})

    assert not out.exists()
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_write_dict_rows_csv_atomic_cleans_temp_on_replace_failure(tmp_path, monkeypatch) -> None:
    out = tmp_path / "leaderboard.csv"

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(artifacts, "_replace_path", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        artifacts.write_dict_rows_csv_atomic(
            out,
            [{"checkpoint": "run_a/best.pt", "sortino": 1.5}],
            fieldnames=["checkpoint", "sortino"],
        )

    assert not out.exists()
    assert list(tmp_path.glob(".leaderboard.csv.*.tmp")) == []


def test_write_pickle_atomic_cleans_temp_on_replace_failure(tmp_path, monkeypatch) -> None:
    out = tmp_path / "model.pkl"
    out.write_bytes(b"old-model")

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(artifacts, "_replace_path", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        artifacts.write_pickle_atomic(out, {"model": "partial"})

    assert out.read_bytes() == b"old-model"
    assert list(tmp_path.glob(".model.pkl.*.tmp")) == []


def test_save_model_atomic_overwrites_and_cleans_temp(tmp_path) -> None:
    out = tmp_path / "model.pkl"
    out.write_bytes(b"old")

    class FakeModel:
        def save(self, path: Path) -> None:
            path.write_bytes(b"new-model")

    artifacts.save_model_atomic(FakeModel(), out)

    assert out.read_bytes() == b"new-model"
    assert list(tmp_path.glob(".model.pkl.*.tmp")) == []


def test_save_model_atomic_cleans_temp_on_replace_failure(tmp_path, monkeypatch) -> None:
    out = tmp_path / "model.pkl"
    out.write_bytes(b"old-model")

    class FakeModel:
        def save(self, path: Path) -> None:
            path.write_bytes(b"partial-model")

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(artifacts, "_replace_path", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        artifacts.save_model_atomic(FakeModel(), out)

    assert out.read_bytes() == b"old-model"
    assert list(tmp_path.glob(".model.pkl.*.tmp")) == []
