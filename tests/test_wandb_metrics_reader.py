"""Unit tests for scripts/wandb_metrics_reader.py — all wandb calls are mocked."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is importable
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Helpers — build fake wandb run objects (for fetch_runs mocking)
# ---------------------------------------------------------------------------


def _make_wandb_run(
    run_id: str = "abc123",
    name: str = "slip_5bps_s42",
    val_return: float | None = 0.083,
    policy_loss: float | None = 0.041,
    steps: int | None = 5_200_000,
    runtime: float | None = 298.0,
    config: dict | None = None,
    sortino: float | None = 1.42,
    win_rate: float | None = 0.67,
    train_return: float | None = None,
    max_drawdown: float | None = None,
    smooth_score: float | None = None,
    group: str | None = None,
    history_rows: list[dict] | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics a wandb.Run object."""
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
    if train_return is not None:
        summary_data["train/final_return"] = train_return
    if max_drawdown is not None:
        summary_data["val/max_drawdown"] = max_drawdown
    if smooth_score is not None:
        summary_data["smooth_score"] = smooth_score

    run.summary = summary_data
    run.config = config or {
        "hidden_size": 1024, "lr": 3e-4, "anneal_lr": True, "ent_coef": 0.05, "seed": 42,
    }

    rows = history_rows or [
        {"train/policy_loss": 0.18, "train/entropy": 2.1, "val/return": 0.01, "val/sortino": 0.20},
        {"train/policy_loss": 0.12, "train/entropy": 1.8, "val/return": 0.03, "val/sortino": 0.55},
        {"train/policy_loss": 0.08, "train/entropy": 1.4, "val/return": 0.05, "val/sortino": 0.90},
        {"train/policy_loss": 0.05, "train/entropy": 1.1, "val/return": 0.07, "val/sortino": 1.15},
        {"train/policy_loss": 0.041, "train/entropy": 0.9, "val/return": 0.083, "val/sortino": 1.42},
    ]
    run.history.return_value = rows
    return run


def _make_api_mock(wandb_runs: list[MagicMock] | None = None) -> MagicMock:
    """Return a mock wandb module whose Api().runs() returns wandb_runs."""
    wandb_mock = MagicMock()
    api_instance = MagicMock()
    wandb_mock.Api.return_value = api_instance

    run_list = wandb_runs if wandb_runs is not None else [_make_wandb_run()]
    api_instance.runs.return_value = run_list
    if run_list:
        api_instance.run.return_value = run_list[0]

    return wandb_mock


# ---------------------------------------------------------------------------
# Helper — plain run dict (used for format_markdown / format_json tests)
# ---------------------------------------------------------------------------


def _make_run_dict(
    run_id: str = "abc123",
    name: str = "slip_5bps_s42",
    val_return: float | None = 0.083,
    final_loss: float | None = 0.041,
    steps: int | None = 5_200_000,
    duration: float | None = 298.0,
    config: dict | None = None,
    sortino: float | None = 1.42,
    win_rate: float | None = 0.67,
    train_return: float | None = None,
    max_drawdown: float | None = None,
    smooth_score: float | None = None,
    loss_curve: list[float] | None = None,
    val_return_curve: list[float] | None = None,
    val_sortino_curve: list[float] | None = None,
    entropy_curve: list[float] | None = None,
    state: str = "finished",
) -> dict:
    return {
        "id": run_id,
        "name": name,
        "state": state,
        "config": config or {"hidden_size": 1024, "lr": 3e-4, "anneal_lr": True, "ent_coef": 0.05, "seed": 42},
        "val_return": val_return,
        "train_return": train_return,
        "max_drawdown": max_drawdown,
        "smooth_score": smooth_score,
        "final_loss": final_loss,
        "steps": steps,
        "duration": duration,
        "loss_curve": loss_curve or [0.18, 0.12, 0.08, 0.05, 0.041],
        "val_return_curve": val_return_curve or [0.01, 0.03, 0.05, 0.08],
        "val_sortino_curve": val_sortino_curve or [0.20, 0.55, 0.90, 1.42],
        "entropy_curve": entropy_curve or [2.1, 1.8, 1.4, 1.1, 0.9],
        "sortino": sortino,
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------


def _load_reader():
    """Import (or reload) the reader module fresh."""
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    mod_name = "wandb_metrics_reader"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import wandb_metrics_reader as mod  # noqa: PLC0415

    return mod


# ---------------------------------------------------------------------------
# Tests: markdown output
# ---------------------------------------------------------------------------


class TestMarkdownOutput:
    def test_output_contains_markdown_header(self):
        mod = _load_reader()
        runs = [_make_run_dict()]
        out = mod.format_markdown(runs, project="stock", entity=None, primary_metric="val/return")
        assert "## WandB Metrics" in out

    def test_output_contains_table_rows(self):
        mod = _load_reader()
        runs = [
            _make_run_dict(name="run_a"),
            _make_run_dict(run_id="xyz", name="run_b", val_return=0.05),
        ]
        out = mod.format_markdown(runs, project="stock", entity=None, primary_metric="val/return")
        assert "|-----|" in out
        assert "run_a" in out
        assert "run_b" in out

    def test_best_value_is_bolded(self):
        mod = _load_reader()
        runs = [
            _make_run_dict(name="best", val_return=0.15),
            _make_run_dict(run_id="r2", name="worse", val_return=0.05),
        ]
        out = mod.format_markdown(runs, project="stock", entity=None, primary_metric="val/return")
        assert "**+15.0%**" in out

    def test_best_run_section_present(self):
        mod = _load_reader()
        runs = [_make_run_dict(name="slip_5bps_s42", val_return=0.083)]
        out = mod.format_markdown(runs, project="stock", entity=None, primary_metric="val/return")
        assert "### Best Run: slip_5bps_s42" in out

    def test_no_runs_returns_graceful_message(self):
        mod = _load_reader()
        out = mod.format_markdown([], project="stock", entity=None, primary_metric="val/return")
        assert "No runs found" in out

    def test_entity_included_in_header(self):
        mod = _load_reader()
        runs = [_make_run_dict()]
        out = mod.format_markdown(runs, project="stock", entity="myorg", primary_metric="val/return")
        assert "myorg/stock" in out

    def test_config_fields_shown(self):
        mod = _load_reader()
        runs = [_make_run_dict(config={"hidden_size": 1024, "lr": 3e-4, "anneal_lr": True})]
        out = mod.format_markdown(runs, project="stock", entity=None, primary_metric="val/return")
        assert "1024" in out

    def test_loss_curve_shown(self):
        mod = _load_reader()
        runs = [_make_run_dict(loss_curve=[0.18, 0.12, 0.08, 0.05, 0.041])]
        out = mod.format_markdown(runs, project="stock", entity=None, primary_metric="val/return")
        assert "Train loss curve" in out
        assert "→" in out

    def test_missing_val_return_handled(self):
        mod = _load_reader()
        runs = [_make_run_dict(val_return=None)]
        # Should not crash; best_idx will be None, no best-run section
        out = mod.format_markdown(runs, project="stock", entity=None, primary_metric="val/return")
        assert "## WandB Metrics" in out

    def test_single_run_label_singular(self):
        mod = _load_reader()
        runs = [_make_run_dict()]
        out = mod.format_markdown(runs, project="stock", entity=None, primary_metric="val/return")
        assert "last 1 run)" in out

    def test_multiple_runs_label_plural(self):
        mod = _load_reader()
        runs = [_make_run_dict(), _make_run_dict(run_id="r2", name="run2")]
        out = mod.format_markdown(runs, project="stock", entity=None, primary_metric="val/return")
        assert "last 2 runs)" in out


# ---------------------------------------------------------------------------
# Tests: JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_is_valid(self):
        mod = _load_reader()
        runs = [_make_run_dict()]
        out = mod.format_json(runs, project="stock", entity=None)
        parsed = json.loads(out)
        assert "runs" in parsed
        assert parsed["project"] == "stock"

    def test_json_contains_run_data(self):
        mod = _load_reader()
        runs = [_make_run_dict(run_id="xyz", name="my_run", val_return=0.12)]
        out = mod.format_json(runs, project="stock", entity="org")
        parsed = json.loads(out)
        assert parsed["runs"][0]["name"] == "my_run"
        assert parsed["project"] == "org/stock"

    def test_json_entity_none_no_slash(self):
        mod = _load_reader()
        runs = [_make_run_dict()]
        out = mod.format_json(runs, project="stock", entity=None)
        parsed = json.loads(out)
        assert parsed["project"] == "stock"

    def test_json_fetched_at_present(self):
        mod = _load_reader()
        runs = [_make_run_dict()]
        out = mod.format_json(runs, project="stock", entity=None)
        parsed = json.loads(out)
        assert "fetched_at" in parsed


# ---------------------------------------------------------------------------
# Tests: last-n-runs limiting (via fetch_runs)
# ---------------------------------------------------------------------------


class TestLastNRuns:
    def test_last_n_limits_output(self):
        mod = _load_reader()
        all_runs = [_make_wandb_run(run_id=f"r{i}", name=f"run_{i}", val_return=0.01 * i) for i in range(10)]
        wandb_mock = _make_api_mock(all_runs)
        result = mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id=None,
            group=None,
            last_n=3,
        )
        assert len(result) == 3

    def test_last_n_1_returns_one(self):
        mod = _load_reader()
        all_runs = [_make_wandb_run(run_id=f"r{i}", name=f"run_{i}", val_return=0.01 * i) for i in range(5)]
        wandb_mock = _make_api_mock(all_runs)
        result = mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id=None,
            group=None,
            last_n=1,
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: specific run ID
# ---------------------------------------------------------------------------


class TestRunId:
    def test_run_id_fetches_single_run(self):
        mod = _load_reader()
        wandb_run = _make_wandb_run(run_id="abc123xyz", name="specific_run", val_return=0.07)
        wandb_mock = _make_api_mock([wandb_run])

        result = mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id="abc123xyz",
            group=None,
            last_n=5,
        )
        assert len(result) == 1
        assert result[0]["id"] == "abc123xyz"
        assert result[0]["name"] == "specific_run"
        # api.run (not api.runs) should have been called
        wandb_mock.Api.return_value.run.assert_called_once()
        wandb_mock.Api.return_value.runs.assert_not_called()

    def test_run_id_not_found_returns_empty(self):
        mod = _load_reader()
        wandb_mock = MagicMock()
        api_instance = MagicMock()
        wandb_mock.Api.return_value = api_instance
        api_instance.run.side_effect = Exception("Run not found")

        result = mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id="doesnotexist",
            group=None,
            last_n=5,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Tests: wandb not installed
# ---------------------------------------------------------------------------


class TestWandbNotInstalled:
    def test_exit_1_when_wandb_missing(self):
        mod = _load_reader()
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError("No module named 'wandb'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(SystemExit) as exc_info:
                mod._import_wandb()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Tests: no runs found
# ---------------------------------------------------------------------------


class TestNoRunsFound:
    def test_empty_runs_list_from_api(self):
        mod = _load_reader()
        wandb_mock = _make_api_mock(wandb_runs=[])
        result = mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id=None,
            group=None,
            last_n=5,
        )
        assert result == []

    def test_markdown_empty_runs(self):
        mod = _load_reader()
        out = mod.format_markdown([], project="stock", entity=None, primary_metric="val/return")
        assert "No runs found" in out
        assert "## WandB Metrics" in out

    def test_api_error_returns_empty(self):
        mod = _load_reader()
        wandb_mock = MagicMock()
        api_instance = MagicMock()
        wandb_mock.Api.return_value = api_instance
        api_instance.runs.side_effect = Exception("Project not found")

        result = mod.fetch_runs(
            wandb=wandb_mock,
            project="nonexistent",
            entity=None,
            run_id=None,
            group=None,
            last_n=5,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Tests: sorting
# ---------------------------------------------------------------------------


class TestSorting:
    def test_runs_sorted_descending(self):
        mod = _load_reader()
        wandb_runs = [
            _make_wandb_run(run_id="r1", name="low", val_return=0.02),
            _make_wandb_run(run_id="r2", name="high", val_return=0.15),
            _make_wandb_run(run_id="r3", name="mid", val_return=0.08),
        ]
        wandb_mock = _make_api_mock(wandb_runs)
        result = mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id=None,
            group=None,
            last_n=10,
        )
        assert result[0]["name"] == "high"
        assert result[1]["name"] == "mid"
        assert result[2]["name"] == "low"

    def test_none_val_return_goes_last(self):
        mod = _load_reader()
        wandb_runs = [
            _make_wandb_run(run_id="r1", name="has_metric", val_return=0.05),
            _make_wandb_run(run_id="r2", name="no_metric", val_return=None),
        ]
        wandb_mock = _make_api_mock(wandb_runs)
        result = mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id=None,
            group=None,
            last_n=10,
        )
        assert result[0]["name"] == "has_metric"
        assert result[1]["name"] == "no_metric"

    def test_blank_api_summary_uses_local_wandb_summary(self, tmp_path, monkeypatch):
        mod = _load_reader()
        run = _make_wandb_run(
            run_id="abc123",
            name="local_fallback",
            val_return=None,
            policy_loss=None,
            sortino=None,
            win_rate=None,
        )
        run.summary = {"_runtime": 12.0, "_step": 321}

        summary_dir = tmp_path / "wandb" / "run-20260328_000000-abc123" / "files"
        summary_dir.mkdir(parents=True)
        (summary_dir / "wandb-summary.json").write_text(
            json.dumps(
                {
                    "val/return": 0.12,
                    "val/sortino": 1.7,
                    "train/final_return": 0.18,
                }
            )
        )
        monkeypatch.setattr(mod, "REPO", tmp_path)

        result = mod.fetch_runs(
            wandb=_make_api_mock([run]),
            project="stock",
            entity=None,
            run_id=None,
            group=None,
            last_n=1,
        )

        assert result[0]["val_return"] == pytest.approx(0.12)
        assert result[0]["sortino"] == pytest.approx(1.7)
        assert result[0]["train_return"] == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# Tests: group filtering
# ---------------------------------------------------------------------------


class TestGroupFiltering:
    def test_group_filter_passed_to_api(self):
        mod = _load_reader()
        wandb_mock = _make_api_mock()

        mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id=None,
            group="my_sweep",
            last_n=5,
        )
        call_args = wandb_mock.Api.return_value.runs.call_args
        assert call_args is not None
        # filters is passed as keyword argument
        filters = call_args[1].get("filters")
        assert filters is not None
        assert filters.get("group") == "my_sweep"

    def test_no_group_passes_none_filters(self):
        mod = _load_reader()
        wandb_mock = _make_api_mock()

        mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id=None,
            group=None,
            last_n=5,
        )
        call_args = wandb_mock.Api.return_value.runs.call_args
        filters = call_args[1].get("filters")
        assert filters is None


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_fmt_duration_seconds(self):
        mod = _load_reader()
        assert mod._fmt_duration(45) == "45s"

    def test_fmt_duration_minutes(self):
        mod = _load_reader()
        assert mod._fmt_duration(298) == "4m 58s"

    def test_fmt_duration_hours(self):
        mod = _load_reader()
        assert mod._fmt_duration(3661) == "1h 1m 1s"

    def test_fmt_duration_none(self):
        mod = _load_reader()
        assert mod._fmt_duration(None) == "—"

    def test_sample_history_downsamples(self):
        mod = _load_reader()
        rows = [{"loss": float(i)} for i in range(100)]
        sampled = mod._sample_history(rows, "loss", n_points=10)
        assert len(sampled) <= 11  # at most n_points + 1 (last value appended)
        assert sampled[-1] == 99.0  # last value always included

    def test_sample_history_short_list_unchanged(self):
        mod = _load_reader()
        rows = [{"loss": float(i)} for i in range(5)]
        sampled = mod._sample_history(rows, "loss", n_points=10)
        assert sampled == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_sample_history_empty(self):
        mod = _load_reader()
        assert mod._sample_history([], "loss") == []

    def test_sample_history_no_key(self):
        mod = _load_reader()
        rows = [{"other": 1.0}]
        assert mod._sample_history(rows, "loss") == []

    def test_pct_positive(self):
        mod = _load_reader()
        assert mod._pct(0.083) == "+8.3%"

    def test_pct_negative(self):
        mod = _load_reader()
        assert mod._pct(-0.05) == "-5.0%"

    def test_pct_none(self):
        mod = _load_reader()
        assert mod._pct(None) == "—"

    def test_extract_config_nested_value(self):
        mod = _load_reader()
        cfg = {"hidden_size": {"value": 512, "desc": "hidden"}, "lr": 3e-4}
        result = mod._extract_config(cfg)
        assert result["hidden_size"] == 512
        assert result["lr"] == 3e-4

    def test_extract_config_plain_value(self):
        mod = _load_reader()
        cfg = {"hidden_size": 1024, "seed": 42}
        result = mod._extract_config(cfg)
        assert result["hidden_size"] == 1024
        assert result["seed"] == 42

    def test_fmt_config_empty(self):
        mod = _load_reader()
        assert mod._fmt_config({}) == "—"

    def test_fmt_config_with_values(self):
        mod = _load_reader()
        cfg = {"hidden_size": 1024, "anneal_lr": True}
        out = mod._fmt_config(cfg)
        assert "1024" in out
        assert "ann=True" in out

    def test_fmt_curve_empty(self):
        mod = _load_reader()
        assert mod._fmt_curve([]) == "—"

    def test_fmt_curve_values(self):
        mod = _load_reader()
        out = mod._fmt_curve([0.18, 0.10, 0.05])
        assert "0.180 → 0.100 → 0.050" == out

    def test_compute_stability_metrics(self):
        mod = _load_reader()
        run = _make_run_dict(
            train_return=0.24,
            val_return=0.12,
            max_drawdown=-0.05,
            smooth_score=0.8,
        )
        metrics = mod.compute_stability_metrics(run)
        assert metrics["loss_downhill_pct"] == pytest.approx(1.0)
        assert metrics["return_uphill_pct"] == pytest.approx(1.0)
        assert metrics["generalization_gap"] == pytest.approx(0.12)
        assert metrics["stability_score"] is not None
        assert 0.0 < metrics["stability_score"] <= 1.0

    def test_resolve_metric_value_uses_stability_metrics(self):
        mod = _load_reader()
        run = _make_run_dict()
        run.update(mod.compute_stability_metrics(run))
        assert mod._resolve_metric_value(run, "stability_score") is not None


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgs:
    def test_parse_args_defaults(self):
        mod = _load_reader()
        args = mod.parse_args(["--project", "stock"])
        assert args.project == "stock"
        assert args.last_n_runs == 5
        assert args.output_format == "markdown"
        assert args.metric == "val/return"
        assert args.entity is None
        assert args.run_id is None
        assert args.group is None

    def test_parse_args_all_flags(self):
        mod = _load_reader()
        args = mod.parse_args([
            "--project", "myproj",
            "--entity", "myorg",
            "--run-id", "abc",
            "--group", "sweep1",
            "--last-n-runs", "3",
            "--metric", "val/sortino",
            "--format", "json",
        ])
        assert args.project == "myproj"
        assert args.entity == "myorg"
        assert args.run_id == "abc"
        assert args.group == "sweep1"
        assert args.last_n_runs == 3
        assert args.metric == "val/sortino"
        assert args.output_format == "json"

    def test_format_invalid_choice_exits(self):
        mod = _load_reader()
        with pytest.raises(SystemExit):
            mod.parse_args(["--project", "stock", "--format", "yaml"])


class TestStabilityOutput:
    def test_markdown_shows_stability_and_gap(self):
        mod = _load_reader()
        run = _make_run_dict(
            train_return=0.18,
            val_return=0.09,
            smooth_score=0.5,
        )
        run.update(mod.compute_stability_metrics(run))
        out = mod.format_markdown([run], project="stock", entity=None, primary_metric="val/return")
        assert "Stability score" in out
        assert "Train/val gap" in out

    def test_fetch_runs_sorts_by_sortino_when_requested(self):
        mod = _load_reader()
        wandb_runs = [
            _make_wandb_run(run_id="r1", name="high_return_low_sortino", val_return=0.12, sortino=0.4),
            _make_wandb_run(run_id="r2", name="lower_return_high_sortino", val_return=0.08, sortino=1.8),
        ]
        wandb_mock = _make_api_mock(wandb_runs)
        result = mod.fetch_runs(
            wandb=wandb_mock,
            project="stock",
            entity=None,
            run_id=None,
            group=None,
            last_n=10,
            primary_metric="val/sortino",
        )
        assert result[0]["name"] == "lower_return_high_sortino"
