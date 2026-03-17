from __future__ import annotations

import json

import pandas as pd

from prepare_hf_puffer_benchmark_data import prepare_benchmark_dataset


def test_prepare_benchmark_dataset_filters_and_writes_symbol_csvs(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    pd.DataFrame(
        {
            "timestamp": [
                "2024-12-30T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-03T00:00:00Z",
            ],
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "volume": [100, 110, 120],
        }
    ).to_csv(source_root / "AAPL.csv", index=False)

    pd.DataFrame(
        {
            "date": ["2024-12-29", "2025-01-02", "2025-01-04"],
            "open": [20.0, 21.0, 22.0],
            "high": [20.5, 21.5, 22.5],
            "low": [19.5, 20.5, 21.5],
            "close": [20.1, 21.1, 22.1],
            "volume": [200, 210, 220],
        }
    ).to_csv(source_root / "MSFT.csv", index=False)

    output_dir = tmp_path / "prepared"
    manifest = prepare_benchmark_dataset(
        source_root=source_root,
        output_dir=output_dir,
        symbols=["AAPL", "MSFT"],
        start_date="2025-01-01",
        end_date="2025-01-03",
    )

    assert [entry["symbol"] for entry in manifest["files"]] == ["AAPL", "MSFT"]

    aapl = pd.read_csv(output_dir / "AAPL.csv")
    msft = pd.read_csv(output_dir / "MSFT.csv")
    assert aapl["timestamp"].tolist() == ["2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"]
    assert msft["date"].tolist() == ["2025-01-02"]

    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_json["start_date"] == "2025-01-01"
    assert manifest_json["end_date"] == "2025-01-03"
    assert manifest_json["files"][0]["rows"] == 2
    assert manifest_json["files"][1]["rows"] == 1
