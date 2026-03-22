from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.chronos2_lora_improvement_sweep import (
    NARROW_LORA_TARGETS,
    WIDE_LORA_TARGETS,
    ImprovementResult,
    ImprovementSweepConfig,
    build_train_cmd,
    compute_improvement,
    find_high_mae_symbols,
    generate_sweep_configs,
    load_dashboard_baselines,
    run_sweep,
    write_summary,
    write_summary_csv,
)


@pytest.mark.unit
def test_generate_sweep_configs_count():
    configs = generate_sweep_configs(
        symbols=["AAVEUSD", "GOOG"],
        lora_rs=(8, 16),
        learning_rates=(1e-5, 5e-5, 1e-4),
        preaugs=("baseline", "percent_change", "differencing"),
        context_lengths=(128, 256, 512),
        lora_target_sets=(NARROW_LORA_TARGETS, WIDE_LORA_TARGETS),
    )
    # 2 symbols * 3 preaugs * 3 ctx * 3 lr * 2 r * 2 target_sets = 216
    assert len(configs) == 216


@pytest.mark.unit
def test_generate_sweep_configs_narrow_only():
    configs = generate_sweep_configs(
        symbols=["BTCUSD"],
        lora_rs=(8,),
        learning_rates=(1e-4,),
        preaugs=("baseline",),
        context_lengths=(128,),
        lora_target_sets=(NARROW_LORA_TARGETS,),
    )
    assert len(configs) == 1
    assert configs[0].lora_targets == NARROW_LORA_TARGETS
    assert configs[0].lr_scheduler == "cosine"
    assert configs[0].warmup_ratio == 0.1


@pytest.mark.unit
def test_generate_sweep_configs_alpha_scales_with_r():
    configs = generate_sweep_configs(
        symbols=["ETHUSD"],
        lora_rs=(8, 16),
        learning_rates=(1e-4,),
        preaugs=("baseline",),
        context_lengths=(128,),
        lora_target_sets=(NARROW_LORA_TARGETS,),
    )
    for cfg in configs:
        assert cfg.lora_alpha == cfg.lora_r * 2


@pytest.mark.unit
def test_wide_targets_include_mlp():
    assert "gate_proj" in WIDE_LORA_TARGETS
    assert "up_proj" in WIDE_LORA_TARGETS
    assert "down_proj" in WIDE_LORA_TARGETS
    assert "q" in WIDE_LORA_TARGETS
    assert "k" in WIDE_LORA_TARGETS


@pytest.mark.unit
def test_find_high_mae_symbols():
    baselines = {"AAVEUSD": 4.49, "GOOG": 1.87, "BTCUSD": 0.36, "NET": 1.94}
    high = find_high_mae_symbols(baselines, threshold=2.0)
    assert "AAVEUSD" in high
    assert "GOOG" not in high
    assert "BTCUSD" not in high


@pytest.mark.unit
def test_find_high_mae_with_extras():
    baselines = {"AAVEUSD": 4.49, "GOOG": 1.87}
    high = find_high_mae_symbols(baselines, threshold=2.0, extra_symbols=["GOOG", "AVAXUSD"])
    assert "AAVEUSD" in high
    assert "GOOG" in high
    assert "AVAXUSD" in high


@pytest.mark.unit
def test_compute_improvement():
    assert compute_improvement(2.0, 1.8) == pytest.approx(10.0)
    assert compute_improvement(2.0, 2.0) == pytest.approx(0.0)
    assert compute_improvement(2.0, 2.1) == pytest.approx(-5.0)
    assert compute_improvement(None, 1.0) is None
    assert compute_improvement(0.0, 1.0) is None
    assert compute_improvement(2.0, None) is None


@pytest.mark.unit
def test_build_train_cmd(tmp_path: Path):
    cfg = ImprovementSweepConfig(
        symbol="AAVEUSD", preaug="differencing", context_length=256,
        batch_size=32, learning_rate=5e-5, num_steps=1000,
        prediction_length=24, lora_r=16, lora_alpha=32,
        lora_targets=WIDE_LORA_TARGETS, lr_scheduler="cosine",
        warmup_ratio=0.1,
    )
    cmd = build_train_cmd(
        run_id="test_run", cfg=cfg,
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
        results_dir=tmp_path / "results",
    )
    assert cmd[:2] == [sys.executable, "scripts/train_crypto_lora_sweep.py"]
    assert "--run-prefix" in cmd
    assert cmd[cmd.index("--run-prefix") + 1] == "test_run"
    assert cmd[cmd.index("--lora-r") + 1] == "16"
    assert cmd[cmd.index("--context-length") + 1] == "256"


@pytest.mark.unit
def test_load_dashboard_baselines(tmp_path: Path):
    csv_path = tmp_path / "dashboard.csv"
    csv_path.write_text("symbol,group,mae_pct\nAAVEUSD,crypto,4.49\nGOOG,stock,1.87\n")
    baselines = load_dashboard_baselines(csv_path)
    assert baselines["AAVEUSD"] == pytest.approx(4.49)
    assert baselines["GOOG"] == pytest.approx(1.87)


@pytest.mark.unit
def test_load_dashboard_baselines_missing_file(tmp_path: Path):
    baselines = load_dashboard_baselines(tmp_path / "nonexistent.csv")
    assert baselines == {}


@pytest.mark.unit
def test_run_sweep_with_mock(monkeypatch, tmp_path: Path):
    cfg = ImprovementSweepConfig(
        symbol="AAVEUSD", preaug="baseline", context_length=128,
        batch_size=32, learning_rate=1e-4, num_steps=100,
        prediction_length=24, lora_r=8, lora_alpha=16,
        lora_targets=NARROW_LORA_TARGETS, lr_scheduler="cosine",
        warmup_ratio=0.1,
    )
    results_dir = tmp_path / "results"
    output_root = tmp_path / "out"

    def _fake_run(cmd, cwd, capture_output, text):
        rp = results_dir / "sweep_AAVEUSD_lora_baseline_ctx128_lr1e-04_r8_20260319.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({
            "run_name": "sweep_AAVEUSD_lora_baseline_ctx128_lr1e-04_r8_20260319",
            "output_dir": str(output_root / "sweep_AAVEUSD"),
            "val": {"mae_percent_mean": 3.50},
            "test": {"mae_percent_mean": 3.80},
        }))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.chronos2_lora_improvement_sweep.subprocess.run", _fake_run)

    baselines = {"AAVEUSD": 4.49}
    results = run_sweep(
        run_id="sweep", configs=[cfg], baselines=baselines,
        data_root=tmp_path / "data", output_root=output_root,
        results_dir=results_dir, improvement_threshold=5.0,
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].val_mae_percent == pytest.approx(3.50)
    assert results[0].baseline_mae_percent == pytest.approx(4.49)
    # (4.49 - 3.50) / 4.49 * 100 = 22.05%
    assert results[0].improvement_pct == pytest.approx(22.05, abs=0.1)
    assert results[0].promoted is True


@pytest.mark.unit
def test_run_sweep_not_promoted(monkeypatch, tmp_path: Path):
    cfg = ImprovementSweepConfig(
        symbol="GOOG", preaug="baseline", context_length=128,
        batch_size=32, learning_rate=1e-5, num_steps=100,
        prediction_length=24, lora_r=8, lora_alpha=16,
        lora_targets=NARROW_LORA_TARGETS, lr_scheduler="cosine",
        warmup_ratio=0.1,
    )
    results_dir = tmp_path / "results"
    output_root = tmp_path / "out"

    def _fake_run(cmd, cwd, capture_output, text):
        rp = results_dir / "sweep_GOOG_lora_baseline_ctx128_lr1e-05_r8_20260319.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({
            "run_name": "sweep_GOOG_lora_baseline_ctx128_lr1e-05_r8_20260319",
            "output_dir": str(output_root / "sweep_GOOG"),
            "val": {"mae_percent_mean": 1.80},
            "test": {"mae_percent_mean": 1.90},
        }))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.chronos2_lora_improvement_sweep.subprocess.run", _fake_run)

    baselines = {"GOOG": 1.87}
    results = run_sweep(
        run_id="sweep", configs=[cfg], baselines=baselines,
        data_root=tmp_path / "data", output_root=output_root,
        results_dir=results_dir, improvement_threshold=5.0,
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    # (1.87 - 1.80) / 1.87 * 100 = 3.74% < 5%
    assert results[0].promoted is False


@pytest.mark.unit
def test_write_summary_json(tmp_path: Path):
    cfg = ImprovementSweepConfig(
        symbol="TEST", preaug="baseline", context_length=128,
        batch_size=32, learning_rate=1e-4, num_steps=100,
        prediction_length=24, lora_r=8, lora_alpha=16,
        lora_targets=NARROW_LORA_TARGETS, lr_scheduler="cosine",
        warmup_ratio=0.1,
    )
    results = [ImprovementResult(config=cfg, status="ok", val_mae_percent=1.5)]
    out = tmp_path / "summary.json"
    write_summary(out, run_id="test", results=results)
    data = json.loads(out.read_text())
    assert data["run_id"] == "test"
    assert data["total"] == 1


@pytest.mark.unit
def test_write_summary_csv(tmp_path: Path):
    cfg = ImprovementSweepConfig(
        symbol="TEST", preaug="baseline", context_length=128,
        batch_size=32, learning_rate=1e-4, num_steps=100,
        prediction_length=24, lora_r=8, lora_alpha=16,
        lora_targets=NARROW_LORA_TARGETS, lr_scheduler="cosine",
        warmup_ratio=0.1,
    )
    results = [ImprovementResult(config=cfg, status="ok", val_mae_percent=1.5, promoted=True)]
    out = tmp_path / "summary.csv"
    write_summary_csv(out, results)
    content = out.read_text()
    assert "TEST" in content
    assert "True" in content


@pytest.mark.unit
def test_symbol_dedup():
    configs = generate_sweep_configs(
        symbols=["AAVEUSD", "aaveusd", "AAVEUSD"],
        lora_rs=(8,),
        learning_rates=(1e-4,),
        preaugs=("baseline",),
        context_lengths=(128,),
        lora_target_sets=(NARROW_LORA_TARGETS,),
    )
    assert len(configs) == 1
