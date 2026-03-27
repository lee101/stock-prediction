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


def test_rank_results_prefers_pnl_gate_then_validation_consistency() -> None:
    records = [
        {
            "symbol": "DOGEUSDT",
            "preaug": "baseline",
            "run_name": "doge_baseline",
            "val_consistency_score": 0.35,
            "val_mae_percent_mean": 0.75,
            "test_mae_percent_mean": 0.90,
            "pnl_eval_present": True,
            "pnl_window_count": 3,
            "pnl_accepted_window_count": 0,
            "pnl_all_windows_accept": False,
            "pnl_min_new_symbol_pnl": -10.0,
            "pnl_min_sortino_delta": -0.7,
            "pnl_mean_sortino_delta": -0.4,
            "pnl_mean_return_delta": -2.0,
            "pnl_mean_new_symbol_pnl": -8.0,
        },
        {
            "symbol": "DOGEUSDT",
            "preaug": "percent_change",
            "run_name": "doge_pct",
            "val_consistency_score": 0.42,
            "val_mae_percent_mean": 0.80,
            "test_mae_percent_mean": 0.88,
            "pnl_eval_present": True,
            "pnl_window_count": 3,
            "pnl_accepted_window_count": 2,
            "pnl_all_windows_accept": False,
            "pnl_min_new_symbol_pnl": 5.0,
            "pnl_min_sortino_delta": -0.2,
            "pnl_mean_sortino_delta": 0.3,
            "pnl_mean_return_delta": 1.1,
            "pnl_mean_new_symbol_pnl": 11.5,
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
        "aave_robust",
        "doge_baseline",
    ]
    best = best_by_symbol(records)
    assert best["DOGEUSDT"]["preaug"] == "percent_change"
    assert best["AAVEUSDT"]["preaug"] == "robust_scaling"

    rendered = render_summary_md(records)
    assert "`DOGEUSDT`: `percent_change`" in rendered
    assert "| 1 | DOGEUSDT | percent_change | pass | 2/3 | +0.3000 | +1.1000 | +11.50 | 0.4200 | 0.8000 | 0.8800 | doge_pct |" in rendered


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

    def fake_evaluate_pnl_for_result(*, result_path: Path, evaluation_config: dict[str, object]) -> dict[str, object]:
        assert evaluation_config["baseline_symbols"] == ["BTCUSD", "ETHUSD", "SOLUSD"]
        report_name = result_path.stem
        candidate_dir = Path(evaluation_config["local_output_root"]) / report_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        accepted = 0 if "baseline" in report_name else 2
        payload = {
            "eval_result_path": str(candidate_dir / "eval.json"),
            "eval_summary": {
                "window_count": 3,
                "accepted_window_count": accepted,
                "rejected_window_count": 3 - accepted,
                "all_windows_accept": False,
                "mean_return_delta": -0.5 if accepted == 0 else 1.25,
                "min_return_delta": -1.0 if accepted == 0 else 0.2,
                "mean_sortino_delta": -0.4 if accepted == 0 else 0.35,
                "min_sortino_delta": -0.6 if accepted == 0 else -0.1,
                "mean_max_dd_delta": 2.0 if accepted == 0 else -0.3,
                "max_max_dd_delta": 3.0 if accepted == 0 else 0.2,
                "mean_new_symbol_pnl": -12.0 if accepted == 0 else 21.0,
                "min_new_symbol_pnl": -15.0 if accepted == 0 else 4.0,
            },
        }
        (candidate_dir / "candidate_summary.json").write_text(json.dumps(payload) + "\n")
        return payload

    monkeypatch.setattr(
        "scripts.run_binance_crypto_lora_sweep._load_train_helpers",
        lambda: (FakeTrainConfig, fake_train_and_evaluate),
    )
    monkeypatch.setattr(
        "scripts.run_binance_crypto_lora_sweep.evaluate_pnl_for_result",
        fake_evaluate_pnl_for_result,
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
        evaluation_config={
            "baseline_symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],
            "windows": "30,7,1",
            "end": "2026-03-18",
            "signal_mode": "forecast_rule",
            "remote_host": "example-host",
            "remote_root": tmp_path / "remote",
            "remote_venv": ".venv313",
            "remote_data_root": "trainingdatahourlybinance",
            "remote_output_root": "analysis/lora_candidate_eval",
            "local_output_root": tmp_path / "candidate_eval",
            "data_root": "trainingdatahourly/crypto",
            "horizons": "1,24",
            "lookback_hours": 5000.0,
            "model": "gemini-3.1-flash-lite-preview",
            "thinking": "HIGH",
            "rate_limit": 0.2,
            "forecast_rule_total_cost_bps": 20.0,
            "forecast_rule_min_reward_risk": 1.1,
            "add_symbol_forecast_rule_total_cost_bps": None,
            "add_symbol_forecast_rule_min_reward_risk": None,
            "add_symbol_max_pos": None,
        },
    )

    assert summary["best_by_symbol"]["DOGEUSDT"]["preaug"] == "percent_change"
    assert summary["best_by_symbol"]["DOGEUSDT"]["pnl_accepted_window_count"] == 2
    assert (results_dir / "DOGEUSDT_baseline_run.json").exists()
    assert (results_dir / "DOGEUSDT_percent_change_run.json").exists()
    summary_json = json.loads((results_dir / "summary.json").read_text())
    assert summary_json["results"][0]["preaug"] == "percent_change"
    assert summary_json["results"][0]["pnl_mean_new_symbol_pnl"] == 21.0
    assert "PnL Gate" in (results_dir / "summary.md").read_text()
