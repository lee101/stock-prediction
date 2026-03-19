from __future__ import annotations

import json
from pathlib import Path

from scripts.promote_chronos2_lora_reports import main


def _write_report(path: Path, *, symbol: str, output_dir: Path, val_mae_percent: float) -> None:
    payload = {
        "symbol": symbol,
        "output_dir": str(output_dir),
        "config": {
            "device_map": "cuda",
            "context_length": 1024,
            "batch_size": 64,
            "val_hours": 168,
            "test_hours": 168,
            "prediction_length": 1,
        },
        "val_metrics": {"mae": 1.0, "mae_percent": val_mae_percent, "pct_return_mae": 0.01, "rmse": 2.0, "count": 168},
        "test_metrics": {"mae": 2.0, "mae_percent": val_mae_percent + 0.1, "pct_return_mae": 0.02, "rmse": 3.0, "count": 168},
        "preaug_strategy": "differencing",
        "preaug_source": "sweep",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_promote_chronos2_lora_reports_selects_best_and_writes_config(tmp_path: Path) -> None:
    report_dir = tmp_path / "hourly_lora"
    report_dir.mkdir()
    out_dir = tmp_path / "hourly"
    out_dir.mkdir()

    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "symbol": "BTCUSD",
                "model": "chronos2",
                "config": {
                    "name": "hourly_ctx512_skip1_single",
                    "device_map": "cuda",
                    "context_length": 512,
                    "batch_size": 32,
                    "quantile_levels": [0.1, 0.5, 0.9],
                    "aggregation": "median",
                    "sample_count": 0,
                    "scaler": "none",
                    "predict_kwargs": {},
                    "skip_rates": [1],
                    "aggregation_method": "single",
                    "use_multivariate": False,
                },
                "validation": {},
                "test": {},
                "windows": {},
            }
        )
        + "\n"
    )

    # Two candidate runs; run2 is better.
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    (run1 / "finetuned-ckpt").mkdir(parents=True)
    (run2 / "finetuned-ckpt").mkdir(parents=True)

    _write_report(report_dir / "AAVEUSDT_lora_run1.json", symbol="AAVEUSDT", output_dir=run1, val_mae_percent=0.9)
    _write_report(report_dir / "AAVEUSDT_lora_run2.json", symbol="AAVEUSDT", output_dir=run2, val_mae_percent=0.4)

    rc = main(
        [
            "--report-dir",
            str(report_dir),
            "--output-dir",
            str(out_dir),
            "--template",
            str(template),
            "--symbols",
            "AAVEUSDT",
        ]
    )
    assert rc == 0

    written = json.loads((out_dir / "AAVEUSDT.json").read_text())
    assert written["symbol"] == "AAVEUSDT"
    assert written["config"]["model_id"] == str(run2 / "finetuned-ckpt")
    assert written["config"]["context_length"] == 1024
    assert written["config"]["batch_size"] == 64


def test_promote_chronos2_lora_reports_accepts_crypto_sweep_schema(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    out_dir = tmp_path / "hourly"
    out_dir.mkdir()

    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "symbol": "BTCUSD",
                "model": "chronos2",
                "config": {
                    "name": "hourly_ctx512_skip1_single",
                    "device_map": "cuda",
                    "context_length": 512,
                    "batch_size": 32,
                    "quantile_levels": [0.1, 0.5, 0.9],
                    "aggregation": "median",
                    "sample_count": 0,
                    "scaler": "none",
                    "predict_kwargs": {},
                    "skip_rates": [1],
                    "aggregation_method": "single",
                    "use_multivariate": False,
                },
                "validation": {},
                "test": {},
                "windows": {},
            }
        )
        + "\n"
    )

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    (run1 / "finetuned-ckpt").mkdir(parents=True)
    (run2 / "finetuned-ckpt").mkdir(parents=True)

    payload_1 = {
        "config": {
            "symbol": "DOGEUSDT",
            "context_length": 512,
            "batch_size": 32,
            "prediction_length": 24,
            "val_hours": 168,
            "test_hours": 168,
            "preaug": "baseline",
        },
        "output_dir": str(run1),
        "run_name": "DOGEUSDT_lora_baseline_ctx512_lr5e-05_r16_1",
        "val": {"mae_percent_mean": 3.0},
        "test": {"mae_percent_mean": 2.7},
    }
    payload_2 = {
        "config": {
            "symbol": "DOGEUSDT",
            "context_length": 512,
            "batch_size": 32,
            "prediction_length": 24,
            "val_hours": 168,
            "test_hours": 168,
            "preaug": "percent_change",
        },
        "output_dir": str(run2),
        "run_name": "DOGEUSDT_lora_percent_change_ctx512_lr5e-05_r16_2",
        "val": {"mae_percent_mean": 2.8},
        "test": {"mae_percent_mean": 2.9},
    }
    (report_dir / "DOGEUSDT_lora_run1.json").write_text(json.dumps(payload_1) + "\n")
    (report_dir / "DOGEUSDT_lora_run2.json").write_text(json.dumps(payload_2) + "\n")

    rc = main(
        [
            "--report-dir",
            str(report_dir),
            "--output-dir",
            str(out_dir),
            "--template",
            str(template),
            "--symbols",
            "DOGEUSDT",
        ]
    )
    assert rc == 0

    written = json.loads((out_dir / "DOGEUSDT.json").read_text())
    assert written["symbol"] == "DOGEUSDT"
    assert written["config"]["model_id"] == str(run2 / "finetuned-ckpt")
    assert written["config"]["context_length"] == 512
    assert written["config"]["batch_size"] == 32
    assert written["metadata"]["preaug_strategy"] == "percent_change"
    assert written["validation"]["mae_percent"] == 2.8
