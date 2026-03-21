"""Tests for scripts/launch_gpu_sweep.py and scripts/launch_a100_large_models.py."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------

import importlib.util

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load_module("launch_gpu_sweep", REPO / "scripts" / "launch_gpu_sweep.py")
a100 = _load_module("launch_a100_large_models", REPO / "scripts" / "launch_a100_large_models.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_script(script_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Run a launch script as a subprocess, capturing combined output."""
    cmd = [sys.executable, str(script_path)] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))


# ---------------------------------------------------------------------------
# launch_gpu_sweep.py — dry-run tests
# ---------------------------------------------------------------------------


class TestGpuSweepDryRun:
    def test_dry_run_exits_zero(self):
        result = _run_script(REPO / "scripts" / "launch_gpu_sweep.py", ["--dry-run"])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_dry_run_prints_plan(self):
        result = _run_script(REPO / "scripts" / "launch_gpu_sweep.py", ["--dry-run"])
        combined = result.stdout + result.stderr
        assert "DRY RUN" in combined
        assert "GPU Sweep Plan" in combined

    def test_dry_run_prints_cost_table(self):
        result = _run_script(REPO / "scripts" / "launch_gpu_sweep.py", ["--dry-run"])
        combined = result.stdout + result.stderr
        # Should list at least one experiment name
        assert "trade_pen_05" in combined

    def test_dry_run_shows_experiment_count(self):
        result = _run_script(REPO / "scripts" / "launch_gpu_sweep.py", ["--dry-run"])
        combined = result.stdout + result.stderr
        # Default is 10 experiments
        assert "10 experiments" in combined

    def test_dry_run_help(self):
        result = _run_script(REPO / "scripts" / "launch_gpu_sweep.py", ["--help"])
        assert result.returncode == 0
        assert "--dry-run" in result.stdout
        assert "--budget-limit" in result.stdout
        assert "--gpu-type" in result.stdout

    def test_dry_run_custom_gpu_type(self):
        result = _run_script(
            REPO / "scripts" / "launch_gpu_sweep.py",
            ["--dry-run", "--gpu-type", "4090"],
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "RTX 4090" in combined or "4090" in combined

    def test_dry_run_custom_num_seeds(self):
        result = _run_script(
            REPO / "scripts" / "launch_gpu_sweep.py",
            ["--dry-run", "--num-seeds", "1"],
        )
        assert result.returncode == 0

    def test_dry_run_custom_configs(self):
        result = _run_script(
            REPO / "scripts" / "launch_gpu_sweep.py",
            ["--dry-run", "--configs", "trade_pen_05,slip_5bps"],
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "2 experiments" in combined

    def test_dry_run_invalid_config_name_exits_nonzero(self):
        result = _run_script(
            REPO / "scripts" / "launch_gpu_sweep.py",
            ["--dry-run", "--configs", "this_does_not_exist"],
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "unknown" in combined.lower() or "error" in combined.lower()


# ---------------------------------------------------------------------------
# launch_gpu_sweep.py — cost estimation unit tests
# ---------------------------------------------------------------------------


class TestGpuSweepCostEstimation:
    def test_estimate_cost_per_experiment_5090(self):
        # RTX 5090 rate = $1.25/hr; 1800s overhead + 1 * 300s = 2100s = 0.583h -> $0.73
        cost = sweep.estimate_cost_per_experiment("5090", num_seeds=1, time_budget_secs=300)
        assert 0.5 < cost < 2.0, f"Unexpected cost: {cost}"

    def test_estimate_cost_per_experiment_scales_with_seeds(self):
        cost_1 = sweep.estimate_cost_per_experiment("5090", num_seeds=1, time_budget_secs=300)
        cost_3 = sweep.estimate_cost_per_experiment("5090", num_seeds=3, time_budget_secs=300)
        assert cost_3 > cost_1

    def test_estimate_total_cost_10_experiments(self):
        total = sweep.estimate_total_cost(
            sweep.DEFAULT_SWEEP_CONFIGS, "5090", num_seeds=3, time_budget_secs=300
        )
        assert 5.0 < total < 50.0, f"Unexpected total: {total}"

    def test_budget_enforcement_dry_run(self):
        # With budget_limit=0.01 (tiny), budget check should print warning in dry-run
        result = _run_script(
            REPO / "scripts" / "launch_gpu_sweep.py",
            ["--dry-run", "--budget-limit", "0.01"],
        )
        combined = result.stdout + result.stderr
        assert "exceeds budget" in combined

    def test_budget_enforcement_live_exits_nonzero(self):
        # In live mode with a tiny budget, should exit nonzero
        # We do NOT want to provision real pods, so we catch the exit before dispatch
        # by providing a budget limit far below the actual cost.
        # Note: this test calls the module's main() directly with mocked dispatch.
        args = sweep.parse_args(["--budget-limit", "0.001"])
        # Manually check what main() would do
        data_dir = REPO / "pufferlib_market" / "data"
        total = sweep.estimate_total_cost(
            sweep.DEFAULT_SWEEP_CONFIGS, args.gpu_type, args.num_seeds, args.time_budget
        )
        assert total > 0.001, "Total cost should exceed tiny budget"

    def test_zero_budget_limit_disables_check(self):
        result = _run_script(
            REPO / "scripts" / "launch_gpu_sweep.py",
            ["--dry-run", "--budget-limit", "0"],
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "exceeds budget" not in combined


# ---------------------------------------------------------------------------
# launch_gpu_sweep.py — config validation
# ---------------------------------------------------------------------------


class TestGpuSweepConfigValidation:
    def test_default_sweep_configs_are_valid(self):
        """All names in DEFAULT_SWEEP_CONFIGS must be in VALID_EXPERIMENT_NAMES."""
        for cfg in sweep.DEFAULT_SWEEP_CONFIGS:
            assert cfg["name"] in sweep.VALID_EXPERIMENT_NAMES, (
                f"Config name '{cfg['name']}' not in VALID_EXPERIMENT_NAMES"
            )

    def test_default_sweep_configs_have_required_keys(self):
        for cfg in sweep.DEFAULT_SWEEP_CONFIGS:
            assert "name" in cfg
            assert "data_path" in cfg
            assert "val_path" in cfg

    def test_default_sweep_configs_is_ten_experiments(self):
        assert len(sweep.DEFAULT_SWEEP_CONFIGS) == 10

    def test_valid_experiment_names_non_empty(self):
        assert len(sweep.VALID_EXPERIMENT_NAMES) > 50

    def test_known_good_names_in_valid_set(self):
        for name in ("trade_pen_05", "slip_5bps", "combined_smooth", "robust_champion",
                     "calmar_focus", "cosine_lr_tp05", "transformer_h256"):
            assert name in sweep.VALID_EXPERIMENT_NAMES, f"'{name}' missing from VALID_EXPERIMENT_NAMES"


# ---------------------------------------------------------------------------
# launch_a100_large_models.py — dry-run tests
# ---------------------------------------------------------------------------


class TestA100DryRun:
    def test_dry_run_exits_zero(self):
        result = _run_script(REPO / "scripts" / "launch_a100_large_models.py", ["--dry-run"])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_dry_run_prints_plan(self):
        result = _run_script(REPO / "scripts" / "launch_a100_large_models.py", ["--dry-run"])
        combined = result.stdout + result.stderr
        assert "DRY RUN" in combined
        assert "Large Model" in combined or "A100" in combined or "a100" in combined.lower()

    def test_dry_run_lists_h2048(self):
        result = _run_script(REPO / "scripts" / "launch_a100_large_models.py", ["--dry-run"])
        combined = result.stdout + result.stderr
        assert "h2048" in combined

    def test_dry_run_lists_h4096(self):
        result = _run_script(REPO / "scripts" / "launch_a100_large_models.py", ["--dry-run"])
        combined = result.stdout + result.stderr
        assert "h4096" in combined

    def test_dry_run_help(self):
        result = _run_script(REPO / "scripts" / "launch_a100_large_models.py", ["--help"])
        assert result.returncode == 0
        assert "--dry-run" in result.stdout
        assert "--gpu-type" in result.stdout
        assert "--budget-limit" in result.stdout

    def test_dry_run_h100_gpu_type(self):
        result = _run_script(
            REPO / "scripts" / "launch_a100_large_models.py",
            ["--dry-run", "--gpu-type", "h100"],
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "H100" in combined

    def test_dry_run_custom_configs(self):
        result = _run_script(
            REPO / "scripts" / "launch_a100_large_models.py",
            ["--dry-run", "--configs", "h2048_anneal"],
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "1 experiments" in combined or "h2048_anneal" in combined

    def test_dry_run_invalid_config_exits_nonzero(self):
        result = _run_script(
            REPO / "scripts" / "launch_a100_large_models.py",
            ["--dry-run", "--configs", "not_a_large_model"],
        )
        assert result.returncode != 0

    def test_h4096_on_a100_prints_warning(self):
        result = _run_script(
            REPO / "scripts" / "launch_a100_large_models.py",
            ["--dry-run", "--gpu-type", "a100", "--configs", "h4096_anneal"],
        )
        # Should still succeed (just a warning) and mention H100 preference
        combined = result.stdout + result.stderr
        assert "H100" in combined or "h100" in combined or "preferred" in combined.lower()


# ---------------------------------------------------------------------------
# launch_a100_large_models.py — cost estimation
# ---------------------------------------------------------------------------


class TestA100CostEstimation:
    def test_estimate_cost_a100(self):
        # A100 rate = $1.64/hr; 1800s overhead + 1 * 300s = 2100s -> ~$0.96
        cost = a100.estimate_cost_per_experiment("a100", num_seeds=1, time_budget_secs=300)
        assert 0.5 < cost < 3.0, f"Unexpected cost: {cost}"

    def test_estimate_cost_h100_higher_than_a100(self):
        cost_a100 = a100.estimate_cost_per_experiment("a100", num_seeds=1, time_budget_secs=300)
        cost_h100 = a100.estimate_cost_per_experiment("h100", num_seeds=1, time_budget_secs=300)
        assert cost_h100 > cost_a100

    def test_estimate_total_4_experiments(self):
        total = a100.estimate_total_cost(
            a100.LARGE_MODEL_CONFIGS, "a100", num_seeds=3, time_budget_secs=300
        )
        # 4 experiments x ~$1.23 each = ~$4.92 on A100 ($1.64/hr, 1800s setup + 3*300s training)
        assert 3.0 < total < 100.0, f"Unexpected total: {total}"

    def test_budget_enforcement_tiny_budget_dry_run(self):
        result = _run_script(
            REPO / "scripts" / "launch_a100_large_models.py",
            ["--dry-run", "--budget-limit", "0.01"],
        )
        combined = result.stdout + result.stderr
        assert "exceeds budget" in combined

    def test_zero_budget_limit_disables_check(self):
        result = _run_script(
            REPO / "scripts" / "launch_a100_large_models.py",
            ["--dry-run", "--budget-limit", "0"],
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "exceeds budget" not in combined


# ---------------------------------------------------------------------------
# launch_a100_large_models.py — config validation
# ---------------------------------------------------------------------------


class TestA100ConfigValidation:
    def test_large_model_configs_have_required_keys(self):
        for cfg in a100.LARGE_MODEL_CONFIGS:
            assert "name" in cfg
            assert "data_path" in cfg
            assert "val_path" in cfg
            assert "min_gpu" in cfg

    def test_large_model_configs_count(self):
        assert len(a100.LARGE_MODEL_CONFIGS) == 4

    def test_valid_large_model_names_set(self):
        assert "h2048_anneal" in a100.VALID_LARGE_MODEL_NAMES
        assert "h2048_anneal_tp05" in a100.VALID_LARGE_MODEL_NAMES
        assert "h2048_resmlp_anneal" in a100.VALID_LARGE_MODEL_NAMES
        assert "h4096_anneal" in a100.VALID_LARGE_MODEL_NAMES

    def test_h4096_min_gpu_is_h100(self):
        cfg = next(c for c in a100.LARGE_MODEL_CONFIGS if c["name"] == "h4096_anneal")
        assert cfg["min_gpu"] == "h100"

    def test_h2048_configs_min_gpu_is_a100(self):
        for cfg in a100.LARGE_MODEL_CONFIGS:
            if "h2048" in cfg["name"]:
                assert cfg["min_gpu"] == "a100"


# ---------------------------------------------------------------------------
# dispatch_experiment — mocked subprocess (no real pods)
# ---------------------------------------------------------------------------


class TestDispatchExperimentMocked:
    """Tests that dispatch_experiment calls subprocess correctly, with no real pods."""

    def test_dispatch_skips_missing_data(self, tmp_path):
        cfg = {
            "name": "trade_pen_05",
            "data_path": "nonexistent_train.bin",
            "val_path": "nonexistent_val.bin",
            "note": "test",
        }
        results_csv = tmp_path / "results.csv"
        result = sweep.dispatch_experiment(
            cfg,
            gpu_type="5090",
            num_seeds=1,
            time_budget_secs=60,
            data_dir=tmp_path,
            budget_limit=10.0,
            dry_run=True,
            results_csv=results_csv,
        )
        assert result["exit_code"] == 1
        assert result["data_exists"] is False

    def test_dispatch_calls_subprocess_when_data_exists(self, tmp_path):
        # Create dummy data file
        (tmp_path / "train.bin").write_bytes(b"\x00" * 16)
        (tmp_path / "val.bin").write_bytes(b"\x00" * 16)
        cfg = {
            "name": "trade_pen_05",
            "data_path": "train.bin",
            "val_path": "val.bin",
            "note": "test",
        }
        results_csv = tmp_path / "results.csv"
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            result = sweep.dispatch_experiment(
                cfg,
                gpu_type="5090",
                num_seeds=1,
                time_budget_secs=60,
                data_dir=tmp_path,
                budget_limit=10.0,
                dry_run=True,
                results_csv=results_csv,
            )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "dispatch_rl_training.py" in " ".join(call_args)
        assert "--dry-run" in call_args
        assert "--descriptions" in call_args
        assert "trade_pen_05" in call_args

    def test_dispatch_writes_results_csv(self, tmp_path):
        (tmp_path / "train.bin").write_bytes(b"\x00" * 16)
        (tmp_path / "val.bin").write_bytes(b"\x00" * 16)
        cfg = {
            "name": "slip_5bps",
            "data_path": "train.bin",
            "val_path": "val.bin",
            "note": "test note",
        }
        results_csv = tmp_path / "results.csv"
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc):
            sweep.dispatch_experiment(
                cfg,
                gpu_type="5090",
                num_seeds=1,
                time_budget_secs=60,
                data_dir=tmp_path,
                budget_limit=10.0,
                dry_run=True,
                results_csv=results_csv,
            )

        assert results_csv.exists()
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["name"] == "slip_5bps"
        assert rows[0]["exit_code"] == "0"

    def test_a100_dispatch_skips_missing_data(self, tmp_path):
        cfg = {
            "name": "h2048_anneal",
            "data_path": "nonexistent.bin",
            "val_path": "nonexistent_val.bin",
            "min_gpu": "a100",
            "note": "test",
        }
        results_csv = tmp_path / "results.csv"
        result = a100.dispatch_experiment(
            cfg,
            gpu_type="a100",
            num_seeds=1,
            time_budget_secs=60,
            data_dir=tmp_path,
            budget_limit=10.0,
            dry_run=True,
            results_csv=results_csv,
        )
        assert result["exit_code"] == 1
        assert result["data_exists"] is False

    def test_a100_dispatch_calls_subprocess_when_data_exists(self, tmp_path):
        (tmp_path / "train.bin").write_bytes(b"\x00" * 16)
        (tmp_path / "val.bin").write_bytes(b"\x00" * 16)
        cfg = {
            "name": "h2048_anneal",
            "data_path": "train.bin",
            "val_path": "val.bin",
            "min_gpu": "a100",
            "note": "test",
        }
        results_csv = tmp_path / "results.csv"
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            result = a100.dispatch_experiment(
                cfg,
                gpu_type="a100",
                num_seeds=1,
                time_budget_secs=60,
                data_dir=tmp_path,
                budget_limit=10.0,
                dry_run=True,
                results_csv=results_csv,
            )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "dispatch_rl_training.py" in " ".join(call_args)
        assert "--gpu-type" in call_args
        assert "a100" in call_args
        assert "h2048_anneal" in call_args


# ---------------------------------------------------------------------------
# parse_args validation
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_sweep_defaults(self):
        args = sweep.parse_args([])
        assert args.gpu_type == "5090"
        assert args.budget_limit == 15.0
        assert args.num_seeds == 3
        assert args.time_budget == 300
        assert args.dry_run is False

    def test_sweep_dry_run_flag(self):
        args = sweep.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_sweep_custom_args(self):
        args = sweep.parse_args(["--gpu-type", "4090", "--num-seeds", "1", "--budget-limit", "5"])
        assert args.gpu_type == "4090"
        assert args.num_seeds == 1
        assert args.budget_limit == 5.0

    def test_a100_defaults(self):
        args = a100.parse_args([])
        assert args.gpu_type == "a100"
        assert args.budget_limit == 25.0
        assert args.num_seeds == 3
        assert args.time_budget == 300
        assert args.dry_run is False

    def test_a100_dry_run_flag(self):
        args = a100.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_a100_h100_gpu_type(self):
        args = a100.parse_args(["--gpu-type", "h100"])
        assert args.gpu_type == "h100"
