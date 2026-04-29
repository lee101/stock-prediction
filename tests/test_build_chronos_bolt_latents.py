from __future__ import annotations

import json
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scripts import build_chronos_bolt_latents as latents


def test_load_symbols_uppercases_dedupes_and_skips_comments(tmp_path: Path) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("\n# comment\naapl\nMSFT\naapl\n\nnvda\n", encoding="utf-8")

    assert latents._load_symbols(symbols_file) == ["AAPL", "MSFT", "NVDA"]


def test_load_csv_normalizes_daily_stock_data(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = []
    for i, day in enumerate(pd.date_range("2024-01-01", periods=60, freq="D")):
        rows.append({"Date": day.strftime("%Y-%m-%d"), "Close": 100 + i})
    rows.append({"Date": "bad-date", "Close": 999})
    rows.append({"Date": "2024-01-10", "Close": -1})
    pd.DataFrame(rows).to_csv(data_root / "AAPL.csv", index=False)

    df = latents._load_csv("AAPL", data_root)

    assert df is not None
    assert list(df.columns) == ["date", "close"]
    assert df["close"].min() > 0
    assert df["date"].is_monotonic_increasing


def test_load_csv_collapses_duplicate_trading_dates_to_last_close(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = []
    for i, day in enumerate(pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")):
        rows.append({"timestamp": day.isoformat(), "close": 100 + i})
        if i == 9:
            rows.append({"timestamp": day.replace(hour=20).isoformat(), "close": 999})
    pd.DataFrame(rows).to_csv(data_root / "AAPL.csv", index=False)

    df = latents._load_csv("AAPL", data_root)

    assert df is not None
    assert df["date"].is_unique
    assert len(df) == 60
    assert df.loc[df["date"] == date(2024, 1, 10), "close"].item() == 999


def test_build_index_table_requires_prior_context_without_lookahead(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    stock_dir = data_root / "stocks"
    stock_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"),
            "close": np.arange(100, 160, dtype=float),
        }
    ).to_csv(stock_dir / "AAPL.csv", index=False)

    pairs, closes_by_symbol, date_indices_by_symbol = latents._build_index_table(
        ["AAPL"],
        data_root,
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 6),
        context_length=3,
    )

    assert pairs == [
        ("AAPL", date(2024, 1, 4)),
        ("AAPL", date(2024, 1, 5)),
        ("AAPL", date(2024, 1, 6)),
    ]
    np.testing.assert_array_equal(closes_by_symbol["AAPL"][:4], np.array([100.0, 101.0, 102.0, 103.0]))
    assert date_indices_by_symbol["AAPL"][date(2024, 1, 4)] == 3


def test_make_context_uses_only_prior_closes_and_can_emit_log_returns() -> None:
    closes = np.array([10.0, 11.0, 12.0, 15.0, 16.0])

    np.testing.assert_array_equal(
        latents._make_context(closes, idx=4, context_length=3, return_log=False),
        np.array([11.0, 12.0, 15.0]),
    )

    got = latents._make_context(closes, idx=4, context_length=3, return_log=True)
    expected = np.array([0.0, np.log(12.0 / 11.0), np.log(15.0 / 12.0)])
    np.testing.assert_allclose(got, expected)


def test_embed_batch_pools_fake_chronos_embeddings() -> None:
    class FakePipe:
        def embed(self, ctx_t: torch.Tensor) -> tuple[torch.Tensor, None]:
            batch = ctx_t.shape[0]
            values = torch.arange(batch * 4 * 3, dtype=torch.float32).reshape(batch, 4, 3)
            return values, None

    contexts = np.zeros((2, 5), dtype=np.float32)

    last = latents._embed_batch(FakePipe(), contexts, "last")
    mean = latents._embed_batch(FakePipe(), contexts, "mean")
    first = latents._embed_batch(FakePipe(), contexts, "first")

    np.testing.assert_array_equal(last, np.array([[9, 10, 11], [21, 22, 23]], dtype=np.float32))
    np.testing.assert_array_equal(mean, np.array([[4.5, 5.5, 6.5], [16.5, 17.5, 18.5]], dtype=np.float32))
    np.testing.assert_array_equal(first, np.array([[0, 1, 2], [12, 13, 14]], dtype=np.float32))


def test_load_chronos_pipeline_forwards_revision(monkeypatch) -> None:
    calls = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls.append((model_id, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "chronos",
        types.SimpleNamespace(BaseChronosPipeline=FakePipeline),
    )

    latents._load_chronos_pipeline(
        model_id="fake/model",
        device="cpu",
        torch_dtype="float32",
        model_revision="abc123",
    )

    assert calls == [
        (
            "fake/model",
            {
                "device_map": "cpu",
                "dtype": torch.float32,
                "revision": "abc123",
            },
        )
    ]


def test_extract_model_commit_hash_prefers_pipeline_and_model_config() -> None:
    pipe = types.SimpleNamespace(
        config=types.SimpleNamespace(_commit_hash=None),
        model=types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash="model-sha")),
    )

    assert latents._extract_model_commit_hash(pipe) == "model-sha"

    pipe._commit_hash = "pipe-sha"
    assert latents._extract_model_commit_hash(pipe) == "pipe-sha"


def test_write_json_atomic_overwrites_and_cleans_temp_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"
    latents._write_json_atomic(path, {"version": 1})
    latents._write_json_atomic(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}

    def fail_write(_tmp_path: Path) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        latents._atomic_path_write(path, fail_write)

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert list(path.parent.glob(f".{path.name}.*")) == []


def test_write_npz_atomic_round_trips_arrays(tmp_path: Path) -> None:
    path = tmp_path / "components.pca.npz"

    latents._write_npz_atomic(path, values=np.array([1.0, 2.0], dtype=np.float32))

    with np.load(path) as loaded:
        np.testing.assert_array_equal(loaded["values"], np.array([1.0, 2.0], dtype=np.float32))


def test_write_parquet_atomic_uses_same_directory_temp_and_replaces(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "latents.parquet"
    df = pd.DataFrame({"x": [1, 2]})
    seen_paths = []

    def fake_to_parquet(self, tmp_path, *, index: bool, compression: str) -> None:
        assert self is df
        assert index is False
        assert compression == "zstd"
        tmp_path = Path(tmp_path)
        seen_paths.append(tmp_path)
        assert tmp_path.parent == path.parent
        assert tmp_path != path
        tmp_path.write_text("new parquet", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    path.write_text("old parquet", encoding="utf-8")

    latents._write_parquet_atomic(df, path)

    assert path.read_text(encoding="utf-8") == "new parquet"
    assert seen_paths
    assert not seen_paths[0].exists()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--start-date", "not-a-date"),
        ("--end-date", "not-a-date"),
        ("--context-length", "0"),
        ("--batch-size", "0"),
        ("--n-components", "-1"),
        ("--limit-symbols", "-1"),
    ],
)
def test_main_rejects_invalid_config_before_loading_symbols(
    tmp_path: Path,
    monkeypatch,
    flag: str,
    value: str,
) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\n", encoding="utf-8")

    def fail_load_symbols(_path: Path):
        raise AssertionError("symbols should not load for invalid config")

    monkeypatch.setattr(latents, "_load_symbols", fail_load_symbols)

    rc = latents.main(["--symbols-file", str(symbols_file), flag, value])

    assert rc == 2


def test_main_rejects_reversed_date_range_before_loading_symbols(tmp_path: Path, monkeypatch) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\n", encoding="utf-8")

    def fail_load_symbols(_path: Path):
        raise AssertionError("symbols should not load for invalid config")

    monkeypatch.setattr(latents, "_load_symbols", fail_load_symbols)

    rc = latents.main(
        [
            "--symbols-file",
            str(symbols_file),
            "--start-date",
            "2024-02-01",
            "--end-date",
            "2024-01-01",
        ]
    )

    assert rc == 2


@pytest.mark.parametrize(
    ("n_components", "embed_dim", "expected_message"),
    [
        ("5", 4, "raw embedding dim"),
        ("4", 8, "number of latent rows"),
    ],
)
def test_main_rejects_invalid_pca_components_before_writing_outputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
    n_components: str,
    embed_dim: int,
    expected_message: str,
) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\n", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"),
            "close": np.arange(100, 160, dtype=float),
        }
    ).to_csv(data_root / "AAPL.csv", index=False)

    class FakePipe:
        def embed(self, ctx_t: torch.Tensor) -> tuple[torch.Tensor, None]:
            batch = ctx_t.shape[0]
            return torch.ones((batch, 2, embed_dim), dtype=torch.float32), None

    class FakePipeline:
        @staticmethod
        def from_pretrained(_model_id: str, **_kwargs):
            return FakePipe()

    def fail_to_parquet(self, path, *, index: bool, compression: str) -> None:
        raise AssertionError("outputs should not be written for invalid PCA config")

    monkeypatch.setitem(
        sys.modules,
        "chronos",
        types.SimpleNamespace(BaseChronosPipeline=FakePipeline),
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)
    output = tmp_path / "latents.parquet"

    rc = latents.main(
        [
            "--symbols-file",
            str(symbols_file),
            "--data-root",
            str(data_root),
            "--start-date",
            "2024-01-04",
            "--end-date",
            "2024-01-06",
            "--context-length",
            "3",
            "--batch-size",
            "2",
            "--n-components",
            n_components,
            "--device",
            "cpu",
            "--torch-dtype",
            "float32",
            "--output",
            str(output),
        ]
    )

    assert rc == 2
    assert expected_message in capsys.readouterr().err
    assert not output.exists()


def test_main_writes_model_revision_and_commit_to_manifest(tmp_path: Path, monkeypatch) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\n", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"),
            "close": np.arange(100, 160, dtype=float),
        }
    ).to_csv(data_root / "AAPL.csv", index=False)

    calls = []

    class FakePipe:
        _commit_hash = "resolved-sha"

        def embed(self, ctx_t: torch.Tensor) -> tuple[torch.Tensor, None]:
            batch = ctx_t.shape[0]
            return torch.ones((batch, 2, 4), dtype=torch.float32), None

    class FakePipeline:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls.append((model_id, kwargs))
            return FakePipe()

    def fake_to_parquet(self, path, *, index: bool, compression: str) -> None:
        assert index is False
        assert compression == "zstd"
        Path(path).write_text("fake parquet", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "chronos",
        types.SimpleNamespace(BaseChronosPipeline=FakePipeline),
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    original_load_csv = latents._load_csv
    load_csv_calls = []

    def counted_load_csv(symbol: str, root: Path):
        load_csv_calls.append(symbol)
        return original_load_csv(symbol, root)

    monkeypatch.setattr(latents, "_load_csv", counted_load_csv)
    output = tmp_path / "latents.parquet"

    rc = latents.main(
        [
            "--symbols-file",
            str(symbols_file),
            "--data-root",
            str(data_root),
            "--start-date",
            "2024-01-04",
            "--end-date",
            "2024-01-06",
            "--context-length",
            "3",
            "--batch-size",
            "2",
            "--n-components",
            "0",
            "--model-id",
            "fake/model",
            "--model-revision",
            "requested-sha",
            "--device",
            "cpu",
            "--torch-dtype",
            "float32",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert calls[0][1]["revision"] == "requested-sha"
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_revision"] == "requested-sha"
    assert manifest["model_commit_hash"] == "resolved-sha"
    assert manifest["n_rows"] == 3
    assert manifest["n_components_after_pca"] == 4
    assert load_csv_calls == ["AAPL"]
