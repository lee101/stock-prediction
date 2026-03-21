"""Unit tests for scripts/wandb_dashboard.py — all wandb calls are mocked."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_scripts_dir = str(REPO / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_dashboard():
    """Import (or reload) wandb_dashboard fresh each call."""
    mod_name = "wandb_dashboard"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import wandb_dashboard as mod  # noqa: PLC0415
    return mod


# ---------------------------------------------------------------------------
# Test fixtures — build fake run dicts
# ---------------------------------------------------------------------------


def _make_run_dict(
    run_id: str = "abc123",
    name: str = "baseline_anneal_lr",
    val_return: float | None = 0.083,
    final_loss: float | None = 0.041,
    steps: int | None = 5_200_000,
    duration: float | None = 298.0,
    config: dict | None = None,
    sortino: float | None = 1.42,
    win_rate: float | None = 0.67,
    max_drawdown: float | None = -0.05,
    num_trades: float | None = 12.0,
    arch: str = "mlp",
    optimizer: str = "adamw",
    dataset: str = "crypto6",
    loss_curve: list[float] | None = None,
    val_return_curve: list[float] | None = None,
    group: str | None = None,
    state: str = "finished",
) -> dict:
    return {
        "id": run_id,
        "name": name,
        "state": state,
        "group": group,
        "config": config if config is not None else {
            "hidden_size": 1024, "lr": 3e-4, "anneal_lr": True,
            "ent_coef": 0.05, "seed": 42, "arch": arch,
            "fill_slippage_bps": 5.0, "trade_penalty": 0.0, "fee_rate": 0.001,
        },
        "config_raw": {},
        "val_return": val_return,
        "final_loss": final_loss,
        "steps": steps,
        "duration": duration,
        "loss_curve": loss_curve or [0.18, 0.12, 0.08, 0.05, 0.041],
        "val_return_curve": val_return_curve or [0.01, 0.03, 0.05, 0.08],
        "sortino": sortino,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades,
        "arch": arch,
        "optimizer": optimizer,
        "dataset": dataset,
    }


def _make_wandb_run(
    run_id: str = "abc123",
    name: str = "baseline_anneal_lr",
    val_return: float | None = 0.083,
    policy_loss: float | None = 0.041,
    steps: int | None = 5_200_000,
    runtime: float | None = 298.0,
    config: dict | None = None,
    sortino: float | None = 1.42,
    win_rate: float | None = 0.67,
    group: str | None = None,
) -> MagicMock:
    run = MagicMock()
    run.id = run_id
    run.name = name
    run.state = "finished"
    run.group = group
    summary_data: dict = {"_runtime": runtime, "_step": steps}
    if val_return is not None:
        summary_data["val/return"] = val_return
    if policy_loss is not None:
        summary_data["train/policy_loss"] = policy_loss
    if sortino is not None:
        summary_data["val/sortino"] = sortino
    if win_rate is not None:
        summary_data["val/win_rate"] = win_rate
    run.summary = summary_data
    run.config = config or {
        "hidden_size": 1024, "lr": 3e-4, "anneal_lr": True,
        "ent_coef": 0.05, "seed": 42,
    }
    run.history.return_value = [
        {"train/policy_loss": 0.18, "train/return": 0.01},
        {"train/policy_loss": 0.10, "train/return": 0.04},
        {"train/policy_loss": 0.05, "train/return": 0.08},
    ]
    return run


def _make_api_mock(wandb_runs: list[MagicMock] | None = None) -> MagicMock:
    wandb_mock = MagicMock()
    api_instance = MagicMock()
    wandb_mock.Api.return_value = api_instance
    run_list = wandb_runs if wandb_runs is not None else [_make_wandb_run()]
    api_instance.runs.return_value = run_list
    if run_list:
        api_instance.run.return_value = run_list[0]
    return wandb_mock


# ---------------------------------------------------------------------------
# Tests: markdown output
# ---------------------------------------------------------------------------


class TestDashboardMarkdown:
    def test_header_present(self):
        mod = _load_dashboard()
        runs = [_make_run_dict()]
        out = mod.format_dashboard(runs, project="stock", entity=None)
        assert "## WandB Trading Dashboard" in out

    def test_entity_in_header(self):
        mod = _load_dashboard()
        runs = [_make_run_dict()]
        out = mod.format_dashboard(runs, project="stock", entity="myorg")
        assert "myorg/stock" in out

    def test_no_runs_returns_graceful_message(self):
        mod = _load_dashboard()
        out = mod.format_dashboard([], project="stock", entity=None)
        assert "No runs found" in out
        assert "## WandB Trading Dashboard" in out

    def test_run_name_in_output(self):
        mod = _load_dashboard()
        runs = [_make_run_dict(name="slip_5bps_run")]
        out = mod.format_dashboard(runs, project="stock", entity=None)
        assert "slip_5bps_run" in out

    def test_val_return_formatted_as_pct(self):
        mod = _load_dashboard()
        runs = [_make_run_dict(val_return=0.083)]
        out = mod.format_dashboard(runs, project="stock", entity=None)
        assert "+8.3%" in out

    def test_best_run_section_present(self):
        mod = _load_dashboard()
        runs = [_make_run_dict(name="my_best_run", val_return=0.15)]
        out = mod.format_dashboard(runs, project="stock", entity=None)
        assert "### Best Run: my_best_run" in out

    def test_multiple_runs_sorted(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="low", val_return=0.02),
            _make_run_dict(run_id="r2", name="high", val_return=0.15),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None)
        # "high" should appear before "low" in the table output
        assert out.index("high") < out.index("low")

    def test_none_val_return_handled(self):
        mod = _load_dashboard()
        runs = [_make_run_dict(val_return=None)]
        out = mod.format_dashboard(runs, project="stock", entity=None)
        assert "## WandB Trading Dashboard" in out

    def test_single_run_label(self):
        mod = _load_dashboard()
        runs = [_make_run_dict()]
        out = mod.format_dashboard(runs, project="stock", entity=None)
        assert "1 run)" in out

    def test_multiple_runs_label(self):
        mod = _load_dashboard()
        runs = [_make_run_dict(), _make_run_dict(run_id="r2", name="run2")]
        out = mod.format_dashboard(runs, project="stock", entity=None)
        assert "2 runs)" in out


# ---------------------------------------------------------------------------
# Tests: compare-archs grouping
# ---------------------------------------------------------------------------


class TestCompareArchs:
    def test_arch_section_present(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="mlp_run", arch="mlp", val_return=0.08),
            _make_run_dict(run_id="r2", name="gru_run", arch="gru", val_return=0.12),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_archs=True)
        assert "Architecture" in out

    def test_arch_groups_separate_archs(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="mlp_run", arch="mlp", val_return=0.08),
            _make_run_dict(run_id="r2", name="transformer_run", arch="transformer", val_return=0.12),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_archs=True)
        assert "mlp" in out
        assert "transformer" in out

    def test_arch_best_run_per_group(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="mlp_low", arch="mlp", val_return=0.05),
            _make_run_dict(run_id="r2", name="mlp_high", arch="mlp", val_return=0.15),
            _make_run_dict(run_id="r3", name="gru_run", arch="gru", val_return=0.08),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_archs=True)
        # Best MLP run should be mlp_high (not mlp_low)
        assert "mlp_high" in out

    def test_arch_not_present_without_flag(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="mlp_run", arch="mlp"),
            _make_run_dict(run_id="r2", name="gru_run", arch="gru"),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_archs=False)
        assert "Comparison by Architecture" not in out

    def test_all_runs_same_arch(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="run1", arch="mlp"),
            _make_run_dict(run_id="r2", name="run2", arch="mlp"),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_archs=True)
        # Only one group: mlp
        assert "mlp" in out


# ---------------------------------------------------------------------------
# Tests: compare-optimizers grouping
# ---------------------------------------------------------------------------


class TestCompareOptimizers:
    def test_optimizer_section_present(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="adamw_run", optimizer="adamw", val_return=0.08),
            _make_run_dict(run_id="r2", name="muon_run", optimizer="muon", val_return=0.12),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_optimizers=True)
        assert "Optimizer" in out

    def test_optimizer_groups_shown(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="adamw_run", optimizer="adamw", val_return=0.08),
            _make_run_dict(run_id="r2", name="muon_run", optimizer="muon", val_return=0.12),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_optimizers=True)
        assert "adamw" in out
        assert "muon" in out

    def test_optimizer_not_present_without_flag(self):
        mod = _load_dashboard()
        runs = [_make_run_dict(name="run1", optimizer="adamw")]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_optimizers=False)
        assert "Comparison by Optimizer" not in out


# ---------------------------------------------------------------------------
# Tests: compare-datasets grouping
# ---------------------------------------------------------------------------


class TestCompareDatasets:
    def test_dataset_section_present(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="crypto6_run", dataset="crypto6", val_return=0.05),
            _make_run_dict(run_id="r2", name="fdusd_run", dataset="fdusd", val_return=0.10),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_datasets=True)
        assert "Dataset" in out

    def test_dataset_groups_shown(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(name="crypto6_run", dataset="crypto6", val_return=0.05),
            _make_run_dict(run_id="r2", name="mixed23_run", dataset="mixed23", val_return=0.10),
        ]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_datasets=True)
        assert "crypto6" in out
        assert "mixed23" in out

    def test_dataset_not_present_without_flag(self):
        mod = _load_dashboard()
        runs = [_make_run_dict(name="run1", dataset="crypto6")]
        out = mod.format_dashboard(runs, project="stock", entity=None, compare_datasets=False)
        assert "Comparison by Dataset" not in out


# ---------------------------------------------------------------------------
# Tests: arch/optimizer/dataset detection helpers
# ---------------------------------------------------------------------------


class TestDetectionHelpers:
    def test_detect_arch_from_config(self):
        mod = _load_dashboard()
        run = _make_run_dict(config={"arch": "transformer"})
        run["config_raw"] = {"arch": "transformer"}
        assert mod._detect_arch(run) == "transformer"

    def test_detect_arch_from_name_gru(self):
        mod = _load_dashboard()
        run = _make_run_dict(name="gru_h512_exp", config={})
        run["config_raw"] = {}
        assert mod._detect_arch(run) == "gru"

    def test_detect_arch_fallback_mlp(self):
        mod = _load_dashboard()
        run = _make_run_dict(name="plain_run", config={})
        run["config_raw"] = {}
        assert mod._detect_arch(run) == "mlp"

    def test_detect_optimizer_from_name_muon(self):
        mod = _load_dashboard()
        run = _make_run_dict(name="muon_lr3e4_run", config={})
        run["config_raw"] = {}
        assert mod._detect_optimizer(run) == "muon"

    def test_detect_optimizer_default_adamw(self):
        mod = _load_dashboard()
        run = _make_run_dict(name="baseline_run", config={})
        run["config_raw"] = {}
        assert mod._detect_optimizer(run) == "adamw"

    def test_detect_dataset_fdusd(self):
        mod = _load_dashboard()
        run = _make_run_dict(name="fdusd_trial", config={"data_path": "data/fdusd3_train.bin"})
        run["config_raw"] = {"data_path": "data/fdusd3_train.bin"}
        assert mod._detect_dataset(run) == "fdusd"

    def test_detect_dataset_mixed23_from_path(self):
        mod = _load_dashboard()
        run = _make_run_dict(name="some_run", config={"data_path": "data/mixed23_train.bin"})
        run["config_raw"] = {"data_path": "data/mixed23_train.bin"}
        assert mod._detect_dataset(run) == "mixed23"

    def test_detect_dataset_from_run_name(self):
        mod = _load_dashboard()
        run = _make_run_dict(name="crypto12_baseline", config={})
        run["config_raw"] = {}
        assert mod._detect_dataset(run) == "crypto12"


# ---------------------------------------------------------------------------
# Tests: fetch_runs with mocked WandB API
# ---------------------------------------------------------------------------


class TestFetchRuns:
    def test_fetch_returns_list(self):
        mod = _load_dashboard()
        wandb_mock = _make_api_mock([_make_wandb_run()])
        result = mod.fetch_runs(
            wandb=wandb_mock, project="stock", entity=None,
            run_id=None, group=None, last_n=5,
        )
        assert isinstance(result, list)
        assert len(result) == 1

    def test_fetch_sorted_descending(self):
        mod = _load_dashboard()
        runs = [
            _make_wandb_run(run_id="r1", name="low", val_return=0.02),
            _make_wandb_run(run_id="r2", name="high", val_return=0.15),
        ]
        wandb_mock = _make_api_mock(runs)
        result = mod.fetch_runs(
            wandb=wandb_mock, project="stock", entity=None,
            run_id=None, group=None, last_n=10,
        )
        assert result[0]["name"] == "high"
        assert result[1]["name"] == "low"

    def test_fetch_last_n_limits(self):
        mod = _load_dashboard()
        runs = [_make_wandb_run(run_id=f"r{i}", name=f"run{i}") for i in range(10)]
        wandb_mock = _make_api_mock(runs)
        result = mod.fetch_runs(
            wandb=wandb_mock, project="stock", entity=None,
            run_id=None, group=None, last_n=3,
        )
        assert len(result) == 3

    def test_fetch_run_id_specific(self):
        mod = _load_dashboard()
        run = _make_wandb_run(run_id="xyz789", name="specific_run")
        wandb_mock = _make_api_mock([run])
        result = mod.fetch_runs(
            wandb=wandb_mock, project="stock", entity=None,
            run_id="xyz789", group=None, last_n=5,
        )
        assert len(result) == 1
        assert result[0]["id"] == "xyz789"
        wandb_mock.Api.return_value.run.assert_called_once()
        wandb_mock.Api.return_value.runs.assert_not_called()

    def test_fetch_api_error_returns_empty(self):
        mod = _load_dashboard()
        wandb_mock = MagicMock()
        api_instance = MagicMock()
        wandb_mock.Api.return_value = api_instance
        api_instance.runs.side_effect = Exception("Project not found")
        result = mod.fetch_runs(
            wandb=wandb_mock, project="nonexistent", entity=None,
            run_id=None, group=None, last_n=5,
        )
        assert result == []

    def test_fetch_run_id_not_found_returns_empty(self):
        mod = _load_dashboard()
        wandb_mock = MagicMock()
        api_instance = MagicMock()
        wandb_mock.Api.return_value = api_instance
        api_instance.run.side_effect = Exception("Not found")
        result = mod.fetch_runs(
            wandb=wandb_mock, project="stock", entity=None,
            run_id="doesnotexist", group=None, last_n=5,
        )
        assert result == []

    def test_fetch_attaches_arch_tag(self):
        mod = _load_dashboard()
        run = _make_wandb_run(name="gru_h512_run", config={"arch": "gru"})
        wandb_mock = _make_api_mock([run])
        result = mod.fetch_runs(
            wandb=wandb_mock, project="stock", entity=None,
            run_id=None, group=None, last_n=5,
        )
        assert result[0]["arch"] == "gru"

    def test_fetch_attaches_dataset_tag(self):
        mod = _load_dashboard()
        run = _make_wandb_run(
            name="fdusd_trial",
            config={"data_path": "pufferlib_market/data/fdusd3_train.bin"},
        )
        wandb_mock = _make_api_mock([run])
        result = mod.fetch_runs(
            wandb=wandb_mock, project="stock", entity=None,
            run_id=None, group=None, last_n=5,
        )
        assert result[0]["dataset"] == "fdusd"

    def test_fetch_group_filter_passed_to_api(self):
        mod = _load_dashboard()
        wandb_mock = _make_api_mock()
        mod.fetch_runs(
            wandb=wandb_mock, project="stock", entity=None,
            run_id=None, group="autoresearch_20260321", last_n=5,
        )
        call_args = wandb_mock.Api.return_value.runs.call_args
        filters = call_args[1].get("filters")
        assert filters is not None
        assert filters.get("group") == "autoresearch_20260321"

    def test_fetch_no_group_passes_none_filters(self):
        mod = _load_dashboard()
        wandb_mock = _make_api_mock()
        mod.fetch_runs(
            wandb=wandb_mock, project="stock", entity=None,
            run_id=None, group=None, last_n=5,
        )
        call_args = wandb_mock.Api.return_value.runs.call_args
        filters = call_args[1].get("filters")
        assert filters is None


# ---------------------------------------------------------------------------
# Tests: formatting helpers
# ---------------------------------------------------------------------------


class TestFormattingHelpers:
    def test_pct_positive(self):
        mod = _load_dashboard()
        assert mod._pct(0.083) == "+8.3%"

    def test_pct_negative(self):
        mod = _load_dashboard()
        assert mod._pct(-0.05) == "-5.0%"

    def test_pct_none(self):
        mod = _load_dashboard()
        assert mod._pct(None) == "—"

    def test_fmt_duration_seconds(self):
        mod = _load_dashboard()
        assert mod._fmt_duration(45) == "45s"

    def test_fmt_duration_minutes(self):
        mod = _load_dashboard()
        assert mod._fmt_duration(298) == "4m 58s"

    def test_fmt_duration_hours(self):
        mod = _load_dashboard()
        assert mod._fmt_duration(3661) == "1h 1m 1s"

    def test_fmt_duration_none(self):
        mod = _load_dashboard()
        assert mod._fmt_duration(None) == "—"

    def test_fmt_curve_empty(self):
        mod = _load_dashboard()
        assert mod._fmt_curve([]) == "—"

    def test_fmt_curve_values(self):
        mod = _load_dashboard()
        out = mod._fmt_curve([0.18, 0.10, 0.05])
        assert "0.180 → 0.100 → 0.050" == out

    def test_fmt_config_keys_subset(self):
        mod = _load_dashboard()
        cfg = {"hidden_size": 1024, "lr": 3e-4, "seed": 42}
        out = mod._fmt_config(cfg, keys=["hidden_size", "lr"])
        assert "1024" in out
        assert "seed" not in out

    def test_fmt_config_empty(self):
        mod = _load_dashboard()
        assert mod._fmt_config({}) == "—"


# ---------------------------------------------------------------------------
# Tests: _group_runs and _best_in_group
# ---------------------------------------------------------------------------


class TestGroupHelpers:
    def test_group_runs_by_arch(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(arch="mlp"),
            _make_run_dict(run_id="r2", arch="gru"),
            _make_run_dict(run_id="r3", arch="mlp"),
        ]
        groups = mod._group_runs(runs, "arch")
        assert len(groups["mlp"]) == 2
        assert len(groups["gru"]) == 1

    def test_group_runs_unknown_key(self):
        mod = _load_dashboard()
        run = _make_run_dict()
        del run["arch"]  # remove key
        groups = mod._group_runs([run], "arch")
        assert "unknown" in groups

    def test_best_in_group_returns_highest(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(val_return=0.05),
            _make_run_dict(run_id="r2", val_return=0.15),
            _make_run_dict(run_id="r3", val_return=0.08),
        ]
        best = mod._best_in_group(runs)
        assert best is not None
        assert best["val_return"] == 0.15

    def test_best_in_group_all_none_returns_none(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(val_return=None),
            _make_run_dict(run_id="r2", val_return=None),
        ]
        best = mod._best_in_group(runs)
        assert best is None

    def test_best_in_group_empty_returns_none(self):
        mod = _load_dashboard()
        best = mod._best_in_group([])
        assert best is None


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgs:
    def test_parse_defaults(self):
        mod = _load_dashboard()
        args = mod.parse_args(["--project", "stock"])
        assert args.project == "stock"
        assert args.last_n_runs == 20
        assert args.entity is None
        assert args.run_id is None
        assert args.group is None
        assert not args.compare_archs
        assert not args.compare_optimizers
        assert not args.compare_datasets

    def test_parse_all_flags(self):
        mod = _load_dashboard()
        args = mod.parse_args([
            "--project", "myproj",
            "--entity", "myorg",
            "--group", "sweep1",
            "--last-n-runs", "10",
            "--compare-archs",
            "--compare-optimizers",
            "--compare-datasets",
        ])
        assert args.project == "myproj"
        assert args.entity == "myorg"
        assert args.group == "sweep1"
        assert args.last_n_runs == 10
        assert args.compare_archs
        assert args.compare_optimizers
        assert args.compare_datasets

    def test_parse_no_project_exits(self):
        mod = _load_dashboard()
        with pytest.raises(SystemExit):
            mod.parse_args([])


# ---------------------------------------------------------------------------
# Tests: graceful failure without API key
# ---------------------------------------------------------------------------


class TestApiKeyCheck:
    def test_check_api_key_with_env_var(self, monkeypatch):
        mod = _load_dashboard()
        monkeypatch.setenv("WANDB_API_KEY", "fakekey123")
        # Should not raise
        mod._check_api_key()

    def test_check_api_key_missing_exits(self, monkeypatch, tmp_path):
        mod = _load_dashboard()
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        # Point home to a temp dir with no netrc/wandb config
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            mod._check_api_key()
        assert exc_info.value.code == 1

    def test_import_wandb_missing_exits(self):
        mod = _load_dashboard()
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError("No module named 'wandb'")
            return original_import(name, *args, **kwargs)

        import unittest.mock as um
        with um.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(SystemExit) as exc_info:
                mod._import_wandb()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Tests: format_group_comparison edge cases
# ---------------------------------------------------------------------------


class TestFormatGroupComparison:
    def test_no_runs_returns_graceful(self):
        mod = _load_dashboard()
        out = mod.format_group_comparison([], "arch", "Architecture")
        assert "No runs" in out

    def test_all_no_val_return(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(arch="mlp", val_return=None),
            _make_run_dict(run_id="r2", arch="gru", val_return=None),
        ]
        out = mod.format_group_comparison(runs, "arch", "Architecture")
        # Should still render the table without crashing
        assert "Architecture" in out

    def test_convergence_curve_present(self):
        mod = _load_dashboard()
        runs = [
            _make_run_dict(arch="mlp", val_return_curve=[0.01, 0.05, 0.08]),
        ]
        out = mod.format_group_comparison(runs, "arch", "Architecture")
        assert "→" in out
