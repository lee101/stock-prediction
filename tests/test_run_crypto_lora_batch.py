from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_crypto_lora_batch import SweepConfig, build_train_cmd, run_batch


def test_build_train_cmd_forwards_run_prefix(tmp_path: Path) -> None:
    cfg = SweepConfig(
        symbol="BTCUSD",
        preaug="baseline",
        context_length=128,
        batch_size=16,
        learning_rate=5e-5,
        num_steps=100,
        prediction_length=24,
        lora_r=16,
    )

    cmd = build_train_cmd(
        run_id="probe_batch",
        cfg=cfg,
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
        results_dir=tmp_path / "results",
    )

    assert cmd[:2] == [sys.executable, "scripts/train_crypto_lora_sweep.py"]
    assert "--run-prefix" in cmd
    assert cmd[cmd.index("--run-prefix") + 1] == "probe_batch"


def test_run_batch_collects_result_json(monkeypatch, tmp_path: Path) -> None:
    cfg = SweepConfig(
        symbol="ETHUSD",
        preaug="percent_change",
        context_length=256,
        batch_size=16,
        learning_rate=1e-4,
        num_steps=50,
        prediction_length=24,
        lora_r=8,
    )
    results_dir = tmp_path / "results"
    output_root = tmp_path / "out"
    data_root = tmp_path / "data"
    data_root.mkdir()

    def _fake_run(cmd: list[str], cwd: Path, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        result_path = results_dir / "probe_run_ETHUSD_lora_percent_change_ctx256_lr1e-04_r8_20260316_120000.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "run_name": "probe_run_ETHUSD_lora_percent_change_ctx256_lr1e-04_r8_20260316_120000",
                    "output_dir": str(output_root / "probe_run_ETHUSD"),
                    "result_path": str(result_path),
                    "val": {"mae_percent_mean": 1.25},
                    "test": {"mae_percent_mean": 1.75},
                    "val_consistency_score": 1.5,
                    "test_consistency_score": 2.5,
                }
            )
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_crypto_lora_batch.subprocess.run", _fake_run)

    results = run_batch(
        run_id="probe_run",
        configs=[cfg],
        data_root=data_root,
        output_root=output_root,
        results_dir=results_dir,
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].run_name == "probe_run_ETHUSD_lora_percent_change_ctx256_lr1e-04_r8_20260316_120000"
    assert results[0].val_mae_percent == 1.25
    assert results[0].test_consistency_score == 2.5
