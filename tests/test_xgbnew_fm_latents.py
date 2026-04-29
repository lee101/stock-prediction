from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgbnew.dataset import attach_fm_latents, load_fm_latents


def _mock_parquet(monkeypatch, tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "fm_latents_test.parquet"
    path.touch()

    def fake_read_parquet(got_path: Path) -> pd.DataFrame:
        assert Path(got_path) == path
        return frame

    monkeypatch.setattr("xgbnew.dataset.pd.read_parquet", fake_read_parquet)
    return path


def test_load_fm_latents_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_fm_latents(tmp_path / "missing.parquet") is None


def test_load_fm_latents_normalizes_symbol_date_and_latent_dtypes(monkeypatch, tmp_path: Path) -> None:
    path = _mock_parquet(
        monkeypatch,
        tmp_path,
        pd.DataFrame(
            {
                "symbol": [" aapl ", "msft"],
                "date": ["2024-01-02", pd.Timestamp("2024-01-03", tz="UTC")],
                "latent_1": [2, 4],
                "latent_0": ["1.5", "3.5"],
            }
        ),
    )

    df = load_fm_latents(path)

    assert df is not None
    assert df["symbol"].to_list() == ["AAPL", "MSFT"]
    assert [str(value) for value in df["date"].to_list()] == ["2024-01-02", "2024-01-03"]
    assert df["latent_0"].dtype == np.float32
    assert df["latent_1"].dtype == np.float32
    np.testing.assert_array_equal(df["latent_0"].to_numpy(), np.array([1.5, 3.5], dtype=np.float32))


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"date": ["2024-01-02"], "latent_0": [1.0]}), "missing required"),
        (pd.DataFrame({"symbol": ["AAPL"], "date": ["2024-01-02"]}), "at least one"),
        (
            pd.DataFrame({"symbol": ["AAPL"], "date": ["2024-01-02"], "latent_x": [1.0]}),
            "invalid latent",
        ),
        (
            pd.DataFrame({"symbol": ["AAPL"], "date": ["2024-01-02"], "latent_1": [1.0]}),
            "contiguous",
        ),
        (
            pd.DataFrame({"symbol": [""], "date": ["2024-01-02"], "latent_0": [1.0]}),
            "empty symbols",
        ),
        (
            pd.DataFrame({"symbol": ["AAPL"], "date": ["not-a-date"], "latent_0": [1.0]}),
            "unparseable dates",
        ),
        (
            pd.DataFrame(
                {
                    "symbol": ["AAPL", "aapl"],
                    "date": ["2024-01-02", "2024-01-02"],
                    "latent_0": [1.0, 2.0],
                }
            ),
            "duplicate",
        ),
        (
            pd.DataFrame({"symbol": ["AAPL"], "date": ["2024-01-02"], "latent_0": [np.inf]}),
            "non-finite",
        ),
        (
            pd.DataFrame({"symbol": ["AAPL"], "date": ["2024-01-02"], "latent_0": ["bad"]}),
            "non-finite",
        ),
    ],
)
def test_load_fm_latents_rejects_malformed_artifacts(
    monkeypatch,
    tmp_path: Path,
    frame: pd.DataFrame,
    message: str,
) -> None:
    path = _mock_parquet(monkeypatch, tmp_path, frame)

    with pytest.raises(ValueError, match=message):
        load_fm_latents(path)


def test_attach_fm_latents_normalizes_direct_frames_and_marks_available() -> None:
    feat_df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOG"],
            "date": [pd.Timestamp("2024-01-02").date()] * 3,
        }
    )
    fm_df = pd.DataFrame(
        {
            "symbol": [" aapl ", "msft"],
            "date": ["2024-01-02", "2024-01-02"],
            "latent_0": ["1.25", "2.25"],
            "latent_1": [3, 4],
        }
    )

    merged = attach_fm_latents(feat_df, fm_df, n_latents=2, fillna=-1.0)

    assert merged["fm_available"].to_list() == [1.0, 1.0, 0.0]
    np.testing.assert_array_equal(
        merged["latent_0"].to_numpy(),
        np.array([1.25, 2.25, -1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        merged["latent_1"].to_numpy(),
        np.array([3.0, 4.0, -1.0], dtype=np.float32),
    )


def test_attach_fm_latents_rejects_duplicate_direct_rows() -> None:
    feat_df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2024-01-02").date()],
        }
    )
    fm_df = pd.DataFrame(
        {
            "symbol": ["AAPL", "aapl"],
            "date": ["2024-01-02", "2024-01-02"],
            "latent_0": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        attach_fm_latents(feat_df, fm_df, n_latents=1)


def test_attach_fm_latents_rejects_more_requested_latents_than_available() -> None:
    feat_df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2024-01-02").date()],
        }
    )
    fm_df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": ["2024-01-02"],
            "latent_0": [1.0],
        }
    )

    with pytest.raises(ValueError, match="requested 2"):
        attach_fm_latents(feat_df, fm_df, n_latents=2)
