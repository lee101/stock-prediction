from __future__ import annotations

from pathlib import Path

from scripts.run_autoresearch_stock_group_sweep import (
    REPO,
    build_symbol_groups,
    build_train_command,
    parse_train_stdout,
    resolve_python_executable,
)


def test_build_symbol_groups_respects_group_size_and_limit():
    groups = build_symbol_groups(["AAPL", "MSFT", "NVDA"], group_size=2, max_groups=2)

    assert groups == [("AAPL", "MSFT"), ("AAPL", "NVDA")]


def test_parse_train_stdout_extracts_metrics():
    stdout = """
---
robust_score:      1.234500
val_loss:          0.456700
training_seconds:  12.3
checkpoint_dir:    /tmp/ckpts
saved_checkpoint:  /tmp/ckpts/latest.pt
best_checkpoint:   /tmp/ckpts/best.pt
"""

    parsed = parse_train_stdout(stdout)

    assert parsed["robust_score"] == 1.2345
    assert parsed["val_loss"] == 0.4567
    assert parsed["training_seconds"] == 12.3
    assert parsed["checkpoint_dir"] == "/tmp/ckpts"
    assert parsed["best_checkpoint"] == "/tmp/ckpts/best.pt"


def test_build_train_command_includes_experiment_flags(tmp_path: Path):
    command = build_train_command(
        python_executable="python",
        data_root=tmp_path / "hourly",
        symbols=("AAPL", "MSFT"),
        frequency="hourly",
        experiment="budget_entropy_confidence",
        hold_bars=3,
        eval_windows="8,16",
        max_positions=2,
        sequence_length=8,
        batch_size=16,
        eval_batch_size=32,
        hidden_size=64,
        layers=1,
        checkpoint_dir=tmp_path / "ckpts",
        dashboard_db="dashboards/metrics.db",
        disable_auto_lr_find=True,
        device="cpu",
        extra_args=("--check-inputs-text",),
    )

    joined = " ".join(command)
    assert "--budget-entropy-confidence" in joined
    assert "--disable-auto-lr-find" in joined
    assert "--symbols AAPL,MSFT" in joined
    assert "--check-inputs-text" in joined


def test_resolve_python_executable_prefers_cuda_capable_project_venv(monkeypatch):
    def fake_has_cuda(python_executable: str) -> bool:
        return python_executable == str(REPO / ".venv312" / "bin" / "python")

    monkeypatch.setattr("scripts.run_autoresearch_stock_group_sweep._python_has_cuda_torch", fake_has_cuda)
    monkeypatch.setattr("scripts.run_autoresearch_stock_group_sweep.sys.executable", "/usr/bin/python")

    resolved = resolve_python_executable(None)

    assert resolved == str(REPO / ".venv312" / "bin" / "python")
