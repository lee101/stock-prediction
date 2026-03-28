"""Tests for multi-seed dispatch support in scripts/dispatch_rl_training.py."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from unittest.mock import patch

import pytest

# The script is not a package module so we load it via importlib.
import importlib.util
import sys

_DISPATCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dispatch_rl_training.py"
_spec = importlib.util.spec_from_file_location("dispatch_rl_training", _DISPATCH_PATH)
dispatch = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(dispatch)  # type: ignore[union-attr]

# Real GPU data for tests that need actual rates/aliases (isolated from cross-test mocking).
_REAL_HOURLY_RATES = {
    "NVIDIA A100 80GB PCIe": 1.64,
    "NVIDIA A100-SXM4-80GB": 1.94,
    "NVIDIA H100 80GB HBM3": 3.89,
    "NVIDIA H100 SXM": 4.49,
    "NVIDIA GeForce RTX 4090": 0.69,
    "NVIDIA GeForce RTX 5090": 1.25,
}
_REAL_GPU_ALIASES = {
    "a100": "NVIDIA A100 80GB PCIe",
    "a100-sxm": "NVIDIA A100-SXM4-80GB",
    "h100": "NVIDIA H100 80GB HBM3",
    "h100-sxm": "NVIDIA H100 SXM",
    "4090": "NVIDIA GeForce RTX 4090",
    "5090": "NVIDIA GeForce RTX 5090",
}


# ---------------------------------------------------------------------------
# _resolve_seeds
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {"num_seeds": 1, "seeds": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_resolve_seeds_default_single() -> None:
    args = _make_args(num_seeds=1)
    assert dispatch._resolve_seeds(args) == [42]


def test_resolve_seeds_three() -> None:
    args = _make_args(num_seeds=3)
    assert dispatch._resolve_seeds(args) == [42, 123, 7]


def test_resolve_seeds_explicit_override() -> None:
    args = _make_args(num_seeds=1, seeds="10,20,30")
    assert dispatch._resolve_seeds(args) == [10, 20, 30]


def test_resolve_seeds_seeds_overrides_num_seeds() -> None:
    args = _make_args(num_seeds=5, seeds="99,17")
    assert dispatch._resolve_seeds(args) == [99, 17]


def test_resolve_seeds_clamps_to_pool() -> None:
    args = _make_args(num_seeds=10)
    # Pool only has 5; should return all 5.
    result = dispatch._resolve_seeds(args)
    assert result == dispatch.DEFAULT_SEEDS


# ---------------------------------------------------------------------------
# _read_leaderboard_metrics
# ---------------------------------------------------------------------------


def test_read_leaderboard_metrics_returns_none_when_missing(tmp_path: Path) -> None:
    metrics = dispatch._read_leaderboard_metrics(tmp_path / "nonexistent.csv")
    assert metrics == {
        "rank_metric": None,
        "rank_score": None,
        "val_return": None,
        "val_sortino": None,
        "generalization_score": None,
        "smooth_score": None,
        "holdout_robust_score": None,
        "replay_combo_score": None,
        "overfit_gap_score": None,
    }


def test_read_leaderboard_metrics_picks_best_row(tmp_path: Path) -> None:
    path = tmp_path / "lb.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "description",
                "rank_metric",
                "rank_score",
                "val_return",
                "val_sortino",
                "generalization_score",
                "smooth_score",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "description": "a",
            "rank_metric": "generalization_score",
            "rank_score": "1.0",
            "val_return": "0.05",
            "val_sortino": "1.2",
            "generalization_score": "0.8",
            "smooth_score": "0.4",
        })
        writer.writerow({
            "description": "b",
            "rank_metric": "generalization_score",
            "rank_score": "3.0",
            "val_return": "0.10",
            "val_sortino": "2.1",
            "generalization_score": "2.5",
            "smooth_score": "0.7",
        })
        writer.writerow({
            "description": "c",
            "rank_metric": "generalization_score",
            "rank_score": "0.5",
            "val_return": "0.01",
            "val_sortino": "0.8",
            "generalization_score": "0.1",
            "smooth_score": "0.2",
        })
    metrics = dispatch._read_leaderboard_metrics(path)
    assert metrics["rank_metric"] == "generalization_score"
    assert metrics["rank_score"] == pytest.approx(3.0)
    assert metrics["val_return"] == pytest.approx(0.10)
    assert metrics["val_sortino"] == pytest.approx(2.1)
    assert metrics["generalization_score"] == pytest.approx(2.5)
    assert metrics["smooth_score"] == pytest.approx(0.7)


def test_read_leaderboard_metrics_falls_back_to_val_return(tmp_path: Path) -> None:
    path = tmp_path / "lb.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["description", "val_return", "val_sortino"])
        writer.writeheader()
        writer.writerow({"description": "a", "val_return": "0.03", "val_sortino": "0.9"})
        writer.writerow({"description": "b", "val_return": "0.07", "val_sortino": "1.5"})
    metrics = dispatch._read_leaderboard_metrics(path)
    assert metrics["val_return"] == pytest.approx(0.07)
    assert metrics["val_sortino"] == pytest.approx(1.5)


def test_read_leaderboard_metrics_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("description,val_return,val_sortino\n")
    metrics = dispatch._read_leaderboard_metrics(path)
    assert metrics["val_return"] is None
    assert metrics["val_sortino"] is None
    assert metrics["rank_score"] is None


# ---------------------------------------------------------------------------
# _print_variance_report
# ---------------------------------------------------------------------------


def test_print_variance_report_creates_csv(tmp_path: Path, capsys) -> None:
    seed_results = [
        {
            "seed": 42,
            "rank_metric": "generalization_score",
            "rank_score": 1.8,
            "generalization_score": 1.8,
            "smooth_score": 0.62,
            "holdout_robust_score": 3.1,
            "replay_combo_score": 2.4,
            "overfit_gap_score": 0.4,
            "val_return": 0.083,
            "val_sortino": 1.42,
            "leaderboard_path": "a.csv",
            "exit_code": 0,
        },
        {
            "seed": 123,
            "rank_metric": "generalization_score",
            "rank_score": 1.4,
            "generalization_score": 1.4,
            "smooth_score": 0.58,
            "holdout_robust_score": 2.9,
            "replay_combo_score": 2.0,
            "overfit_gap_score": 0.7,
            "val_return": 0.061,
            "val_sortino": 1.18,
            "leaderboard_path": "b.csv",
            "exit_code": 0,
        },
        {
            "seed": 7,
            "rank_metric": "generalization_score",
            "rank_score": 2.1,
            "generalization_score": 2.1,
            "smooth_score": 0.71,
            "holdout_robust_score": 3.7,
            "replay_combo_score": 2.8,
            "overfit_gap_score": 0.3,
            "val_return": 0.092,
            "val_sortino": 1.61,
            "leaderboard_path": "c.csv",
            "exit_code": 0,
        },
    ]
    out_path = tmp_path / "pufferlib_market" / "test_multiseed.csv"
    dispatch._print_variance_report(seed_results, out_path)

    captured = capsys.readouterr()
    assert "Seed Results Summary" in captured.out
    assert "rank_score:" in captured.out
    assert "generalization:" in captured.out
    assert "Failed seeds: 0/3" in captured.out
    assert "non-deterministic" in captured.out

    assert out_path.exists()
    with open(out_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {int(r["seed"]) for r in rows} == {42, 123, 7}
    summary_path = out_path.with_suffix(".summary.json")
    assert summary_path.exists()
    summary = summary_path.read_text()
    assert '"failed_seed_count": 0' in summary
    assert '"rank_metrics": [' in summary


def test_print_variance_report_computes_correct_stats(tmp_path: Path, capsys) -> None:
    returns = [0.083, 0.061, 0.092]
    sortinos = [1.42, 1.18, 1.61]
    seed_results = [
        {
            "seed": idx,
            "rank_metric": "rank_score",
            "rank_score": r * 10.0,
            "generalization_score": r * 10.0,
            "smooth_score": 0.5,
            "holdout_robust_score": r * 20.0,
            "replay_combo_score": r * 15.0,
            "overfit_gap_score": 0.2,
            "val_return": r,
            "val_sortino": s,
            "leaderboard_path": "",
            "exit_code": 0,
        }
        for idx, (r, s) in enumerate(zip(returns, sortinos), start=1)
    ]
    dispatch._print_variance_report(seed_results, tmp_path / "ms.csv")
    captured = capsys.readouterr()

    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    mean_s = statistics.mean(sortinos)
    std_s = statistics.stdev(sortinos)

    assert f"{mean_r:+.1%}" in captured.out
    assert f"{std_r:.1%}" in captured.out
    assert f"{mean_s:+.2f}" in captured.out
    assert f"{std_s:+.2f}" not in captured.out
    assert f"{std_s:.2f}" in captured.out


def test_print_variance_report_handles_none_metrics(tmp_path: Path, capsys) -> None:
    seed_results = [
        {
            "seed": 42,
            "rank_metric": None,
            "rank_score": None,
            "generalization_score": None,
            "smooth_score": None,
            "holdout_robust_score": None,
            "replay_combo_score": None,
            "overfit_gap_score": None,
            "val_return": None,
            "val_sortino": None,
            "leaderboard_path": "",
            "exit_code": 1,
        },
    ]
    dispatch._print_variance_report(seed_results, tmp_path / "ms.csv")
    captured = capsys.readouterr()
    assert "n/a" in captured.out
    assert "Failed seeds: 1/1" in captured.out


# ---------------------------------------------------------------------------
# run_local passes --seed
# ---------------------------------------------------------------------------


def test_run_local_passes_seed_to_subprocess(monkeypatch, tmp_path: Path) -> None:
    captured_cmds: list[list[str]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(cmd, *, cwd=None, **kwargs):
        captured_cmds.append(list(cmd))
        return _FakeResult()

    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    args = argparse.Namespace(
        run_id="test_run",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=60,
        max_trials=2,
        leaderboard="",
        checkpoint_dir="",
        wandb_project="test",
        descriptions="",
        stocks=False,
    )
    dispatch.run_local(args, seed=123)
    assert captured_cmds, "subprocess.run should have been called"
    cmd = captured_cmds[0]
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "123"


def test_run_local_no_seed_when_none(monkeypatch, tmp_path: Path) -> None:
    captured_cmds: list[list[str]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(cmd, *, cwd=None, **kwargs):
        captured_cmds.append(list(cmd))
        return _FakeResult()

    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    args = argparse.Namespace(
        run_id="test_run",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=60,
        max_trials=2,
        leaderboard="",
        checkpoint_dir="",
        wandb_project="test",
        descriptions="",
        stocks=False,
    )
    dispatch.run_local(args, seed=None)
    cmd = captured_cmds[0]
    assert "--seed" not in cmd


# ---------------------------------------------------------------------------
# _build_remote_autoresearch_cmd passes --seed
# ---------------------------------------------------------------------------


def test_build_remote_autoresearch_cmd_includes_seed() -> None:
    args = argparse.Namespace(
        data_train="train.bin",
        data_val="val.bin",
        time_budget=60,
        max_trials=2,
        wandb_project="test",
        descriptions="",
        stocks=False,
    )
    cmd = dispatch._build_remote_autoresearch_cmd(
        args, "/workspace", "lb.csv", "checkpoints", seed=7
    )
    assert "--seed" in cmd
    assert "'7'" in cmd or "7" in cmd


def test_build_remote_autoresearch_cmd_no_seed_when_none() -> None:
    args = argparse.Namespace(
        data_train="train.bin",
        data_val="val.bin",
        time_budget=60,
        max_trials=2,
        wandb_project="test",
        descriptions="",
        stocks=False,
    )
    cmd = dispatch._build_remote_autoresearch_cmd(
        args, "/workspace", "lb.csv", "checkpoints", seed=None
    )
    assert "--seed" not in cmd


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


def test_estimate_cost_setup_overhead_constant_exists() -> None:
    """_SETUP_OVERHEAD_SECS must be defined and equal 1800 (30min)."""
    assert dispatch._SETUP_OVERHEAD_SECS == 1800


def test_estimate_cost_includes_overhead() -> None:
    """estimate_cost includes the 30min setup overhead in the total."""
    with (
        patch.object(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES),
        patch.object(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES),
    ):
        cost_zero_budget = dispatch.estimate_cost("NVIDIA A100 80GB PCIe", num_seeds=1, time_budget_secs=0)
        cost_with_budget = dispatch.estimate_cost("NVIDIA A100 80GB PCIe", num_seeds=1, time_budget_secs=300)
    # cost with budget must be strictly greater
    assert cost_with_budget > cost_zero_budget
    assert cost_zero_budget > 0  # overhead alone still costs something


def test_estimate_cost_alias_5090() -> None:
    """estimate_cost resolves '5090' alias to the 5090 display name."""
    with (
        patch.object(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES),
        patch.object(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES),
    ):
        cost_alias = dispatch.estimate_cost("5090", num_seeds=1, time_budget_secs=300)
        cost_full = dispatch.estimate_cost("NVIDIA GeForce RTX 5090", num_seeds=1, time_budget_secs=300)
    assert abs(cost_alias - cost_full) < 1e-9
    assert cost_alias > 0


def test_estimate_cost_zero_for_unknown() -> None:
    """Unknown GPU type should return 0 (rate = 0)."""
    with (
        patch.object(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES),
        patch.object(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES),
    ):
        cost = dispatch.estimate_cost("NVIDIA RTX 9999 Unknown")
    assert cost == 0.0


# ---------------------------------------------------------------------------
# --budget-limit argument parsing
# ---------------------------------------------------------------------------


def test_parse_args_budget_limit_default() -> None:
    args = dispatch.parse_args([
        "--data-train", "train.bin",
        "--data-val", "val.bin",
    ])
    assert hasattr(args, "budget_limit")
    assert args.budget_limit == 5.0


def test_parse_args_budget_limit_custom() -> None:
    args = dispatch.parse_args([
        "--data-train", "train.bin",
        "--data-val", "val.bin",
        "--budget-limit", "2.5",
    ])
    assert args.budget_limit == pytest.approx(2.5)


def test_parse_args_budget_limit_zero_disables() -> None:
    args = dispatch.parse_args([
        "--data-train", "train.bin",
        "--data-val", "val.bin",
        "--budget-limit", "0",
    ])
    assert args.budget_limit == 0.0


# ---------------------------------------------------------------------------
# run_remote budget gate
# ---------------------------------------------------------------------------


def test_run_remote_raises_on_budget_exceeded(monkeypatch) -> None:
    """run_remote should SystemExit before provisioning when cost exceeds limit."""
    args = argparse.Namespace(
        run_id="test",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=300,
        max_trials=50,
        gpu_type="a100",
        wandb_project="stock",
        checkpoint_dir="",
        leaderboard="",
        descriptions="",
        budget_limit=0.01,  # tiny limit — a100 will exceed this
        stocks=False,
    )

    # Ensure RunPodClient is never created
    create_calls = []

    def _mock_runpod_init(*a, **kw):
        create_calls.append(True)
        raise RuntimeError("should not be called")

    monkeypatch.setattr(dispatch, "RunPodClient", _mock_runpod_init)
    monkeypatch.setattr(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES)
    monkeypatch.setattr(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES)
    monkeypatch.setattr(dispatch, "resolve_gpu_type", lambda alias: _REAL_GPU_ALIASES.get(alias, alias))

    with pytest.raises(SystemExit):
        dispatch.run_remote(args)

    assert not create_calls, "RunPodClient should not have been instantiated before budget check"


def test_run_remote_proceeds_when_budget_zero(monkeypatch, capsys) -> None:
    """run_remote with budget_limit=0 skips the check and proceeds."""
    args = argparse.Namespace(
        run_id="test",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=300,
        max_trials=50,
        gpu_type="a100",
        wandb_project="stock",
        checkpoint_dir="",
        leaderboard="",
        descriptions="",
        budget_limit=0,  # disabled
        stocks=False,
    )

    monkeypatch.setattr(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES)
    monkeypatch.setattr(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES)
    monkeypatch.setattr(dispatch, "resolve_gpu_type", lambda alias: _REAL_GPU_ALIASES.get(alias, alias))
    # Make RunPodClient raise ValueError to abort before actual pod creation
    monkeypatch.setattr(dispatch, "RunPodClient", lambda: (_ for _ in ()).throw(ValueError("no key")))

    result = dispatch.run_remote(args)
    # Should return 1 (RunPodClient error) not SystemExit
    assert result == 1


# ---------------------------------------------------------------------------
# print_dry_run_plan shows cost estimate
# ---------------------------------------------------------------------------


def test_print_dry_run_plan_shows_cost_estimate(monkeypatch, capsys) -> None:
    """print_dry_run_plan always prints cost estimate for remote runs."""
    monkeypatch.setattr(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES)
    monkeypatch.setattr(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES)
    monkeypatch.setattr(dispatch, "resolve_gpu_type", lambda alias: _REAL_GPU_ALIASES.get(alias, alias))

    args = argparse.Namespace(
        run_id="dry_test",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=300,
        max_trials=50,
        gpu_type="a100",
        wandb_project="stock",
        checkpoint_dir="",
        leaderboard="",
        descriptions="",
        force_remote=False,
        vram_threshold_gb=16.0,
        budget_limit=5.0,
        seeds=[42],
        stocks=False,
    )
    dispatch.print_dry_run_plan(args, remote=True)
    captured = capsys.readouterr()
    assert "Cost estimate:" in captured.out
    assert "GPU:" in captured.out
    assert "Setup overhead:" in captured.out
    assert "Total:" in captured.out


def test_print_dry_run_plan_warns_over_budget(monkeypatch, capsys) -> None:
    """print_dry_run_plan shows warning if estimated cost exceeds budget_limit."""
    monkeypatch.setattr(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES)
    monkeypatch.setattr(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES)
    monkeypatch.setattr(dispatch, "resolve_gpu_type", lambda alias: _REAL_GPU_ALIASES.get(alias, alias))

    args = argparse.Namespace(
        run_id="dry_test",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=300,
        max_trials=50,
        gpu_type="a100",
        wandb_project="stock",
        checkpoint_dir="",
        leaderboard="",
        descriptions="",
        force_remote=False,
        vram_threshold_gb=16.0,
        budget_limit=0.01,  # very tight
        seeds=[42],
        stocks=False,
    )
    dispatch.print_dry_run_plan(args, remote=True)
    captured = capsys.readouterr()
    assert "exceeds budget limit" in captured.out
    assert "--budget-limit" in captured.out
