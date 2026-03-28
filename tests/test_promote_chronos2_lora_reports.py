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


def test_promote_chronos2_lora_reports_can_prefer_stable_family_across_seeds(tmp_path: Path) -> None:
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

    stable_run_a = tmp_path / "stable_a"
    stable_run_b = tmp_path / "stable_b"
    noisy_run_a = tmp_path / "noisy_a"
    noisy_run_b = tmp_path / "noisy_b"
    for run_dir in (stable_run_a, stable_run_b, noisy_run_a, noisy_run_b):
        (run_dir / "finetuned-ckpt").mkdir(parents=True)

    payloads = [
        (
            report_dir / "ETHUSD_lora_stable_seed1.json",
            {
                "symbol": "ETHUSD",
                "output_dir": str(stable_run_a),
                "config": {
                    "device_map": "cuda",
                    "context_length": 512,
                    "batch_size": 32,
                    "learning_rate": 5e-5,
                    "prediction_length": 24,
                    "lora_r": 16,
                    "seed": 1337,
                    "preaug": "baseline",
                },
                "val_metrics": {"mae_percent": 0.55},
                "test_metrics": {"mae_percent": 0.60},
                "preaug_strategy": "baseline",
            },
        ),
        (
            report_dir / "ETHUSD_lora_stable_seed2.json",
            {
                "symbol": "ETHUSD",
                "output_dir": str(stable_run_b),
                "config": {
                    "device_map": "cuda",
                    "context_length": 512,
                    "batch_size": 32,
                    "learning_rate": 5e-5,
                    "prediction_length": 24,
                    "lora_r": 16,
                    "seed": 2027,
                    "preaug": "baseline",
                },
                "val_metrics": {"mae_percent": 0.57},
                "test_metrics": {"mae_percent": 0.62},
                "preaug_strategy": "baseline",
            },
        ),
        (
            report_dir / "ETHUSD_lora_noisy_seed1.json",
            {
                "symbol": "ETHUSD",
                "output_dir": str(noisy_run_a),
                "config": {
                    "device_map": "cuda",
                    "context_length": 512,
                    "batch_size": 32,
                    "learning_rate": 1e-4,
                    "prediction_length": 24,
                    "lora_r": 16,
                    "seed": 1337,
                    "preaug": "percent_change",
                },
                "val_metrics": {"mae_percent": 0.35},
                "test_metrics": {"mae_percent": 0.45},
                "preaug_strategy": "percent_change",
            },
        ),
        (
            report_dir / "ETHUSD_lora_noisy_seed2.json",
            {
                "symbol": "ETHUSD",
                "output_dir": str(noisy_run_b),
                "config": {
                    "device_map": "cuda",
                    "context_length": 512,
                    "batch_size": 32,
                    "learning_rate": 1e-4,
                    "prediction_length": 24,
                    "lora_r": 16,
                    "seed": 2027,
                    "preaug": "percent_change",
                },
                "val_metrics": {"mae_percent": 1.05},
                "test_metrics": {"mae_percent": 1.10},
                "preaug_strategy": "percent_change",
            },
        ),
    ]
    for path, payload in payloads:
        path.write_text(json.dumps(payload, indent=2) + "\n")

    rc = main(
        [
            "--report-dir",
            str(report_dir),
            "--output-dir",
            str(out_dir),
            "--template",
            str(template),
            "--symbols",
            "ETHUSD",
            "--selection-strategy",
            "stable_family",
            "--stability-penalty",
            "0.25",
            "--min-family-size",
            "2",
        ]
    )
    assert rc == 0

    written = json.loads((out_dir / "ETHUSD.json").read_text())
    assert written["config"]["model_id"] == str(stable_run_a / "finetuned-ckpt")
    assert written["metadata"]["selection_strategy"] == "stable_family"
    assert written["metadata"]["selection_family_size"] == 2
    assert written["metadata"]["selection_family_key"] is not None
    assert written["validation"]["mae_percent"] == 0.55
