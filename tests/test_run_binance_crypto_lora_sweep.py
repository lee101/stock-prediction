from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scripts.run_binance_crypto_lora_sweep import (
    best_by_symbol,
    parse_preaugs,
    parse_symbols,
    rank_results,
    render_summary_md,
    run_sweep,
)


def test_parse_symbols_and_preaugs_deduplicate_preserve_order() -> None:
    assert parse_symbols([" dogeusdt , aaveusdt ", "DOGEUSDT", "btcusdt"]) == [
        "DOGEUSDT",
        "AAVEUSDT",
        "BTCUSDT",
    ]
    assert parse_preaugs([" baseline,percent_change ", "baseline", "ROBUST_SCALING"]) == [
        "baseline",
        "percent_change",
        "robust_scaling",
    ]


def test_rank_results_and_render_summary_pick_lowest_validation_consistency() -> None:
    records = [
        {
            "symbol": "DOGEUSDT",
            "preaug": "baseline",
            "run_name": "doge_baseline",
            "val_consistency_score": 0.55,
            "val_mae_percent_mean": 0.80,
            "test_mae_percent_mean": 0.90,
        },
        {
            "symbol": "DOGEUSDT",
            "preaug": "percent_change",
            "run_name": "doge_pct",
            "val_consistency_score": 0.42,
            "val_mae_percent_mean": 0.70,
            "test_mae_percent_mean": 0.88,
        },
        {
            "symbol": "AAVEUSDT",
            "preaug": "robust_scaling",
            "run_name": "aave_robust",
            "val_consistency_score": 0.61,
            "val_mae_percent_mean": 0.95,
            "test_mae_percent_mean": 1.02,
        },
    ]

    ranked = rank_results(records)
    assert [record["run_name"] for record in ranked] == [
        "doge_pct",
        "doge_baseline",
        "aave_robust",
    ]
    best = best_by_symbol(records)
    assert best["DOGEUSDT"]["preaug"] == "percent_change"
    assert best["AAVEUSDT"]["preaug"] == "robust_scaling"

    rendered = render_summary_md(records)
    assert "`DOGEUSDT`: `percent_change`" in rendered
    assert "| 1 | DOGEUSDT | percent_change | 0.4200 | 0.7000 | 0.8800 | doge_pct |" in rendered


def test_run_sweep_writes_per_run_and_summary_files(tmp_path: Path, monkeypatch) -> None:
    @dataclass
    class FakeTrainConfig:
        symbol: str
        context_length: int
        prediction_length: int
        learning_rate: float
        num_steps: int
        lora_r: int
        preaug: str

    def fake_train_and_evaluate(cfg: FakeTrainConfig, data_path: Path, output_root: Path):
        output_dir = output_root / f"{cfg.symbol}_{cfg.preaug}"
        output_dir.mkdir(parents=True, exist_ok=True)
        base_score = {
            ("DOGEUSDT", "baseline"): 0.5,
            ("DOGEUSDT", "percent_change"): 0.4,
        }[(cfg.symbol, cfg.preaug)]
        return {
            "config": {
                "symbol": cfg.symbol,
                "preaug": cfg.preaug,
            },
            "run_name": f"{cfg.symbol}_{cfg.preaug}_run",
            "output_dir": str(output_dir),
            "val": {"mae_percent_mean": base_score + 0.1},
            "test": {"mae_percent_mean": base_score + 0.2},
            "val_consistency_score": base_score,
            "test_consistency_score": base_score + 0.05,
        }

    monkeypatch.setattr(
        "scripts.run_binance_crypto_lora_sweep._load_train_helpers",
        lambda: (FakeTrainConfig, fake_train_and_evaluate),
    )

    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "DOGEUSDT.csv").write_text("timestamp,open,high,low,close\n")

    results_dir = tmp_path / "results"
    summary = run_sweep(
        symbols=["DOGEUSDT"],
        preaugs=["baseline", "percent_change"],
        data_root=data_root,
        output_root=tmp_path / "outputs",
        results_dir=results_dir,
        context_length=512,
        prediction_length=24,
        learning_rate=5e-5,
        num_steps=600,
        lora_r=16,
    )

    assert summary["best_by_symbol"]["DOGEUSDT"]["preaug"] == "percent_change"
    assert (results_dir / "DOGEUSDT_baseline_run.json").exists()
    assert (results_dir / "DOGEUSDT_percent_change_run.json").exists()
    summary_json = json.loads((results_dir / "summary.json").read_text())
    assert summary_json["results"][0]["preaug"] == "percent_change"
    assert "percent_change" in (results_dir / "summary.md").read_text()
