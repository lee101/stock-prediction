"""Tests for scripts/cutellm_dispatch.py — all RunPod API calls are mocked."""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the repo root importable regardless of cwd.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Module import (we import after sys.path is set)
# ---------------------------------------------------------------------------

import scripts.cutellm_dispatch as dispatch  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(pods=None):
    """Return a mock RunPodClient that lists the given pods."""
    mock_client = MagicMock()
    mock_client.list_pods.return_value = pods or []
    return mock_client


def _pod(name: str, pod_id: str = "pod-abc", status: str = "RUNNING", gpu_type: str = "") -> MagicMock:
    p = MagicMock()
    p.name = name
    p.id = pod_id
    p.status = status
    p.gpu_type = gpu_type
    return p


# ---------------------------------------------------------------------------
# parse_args — smoke tests for all subcommands
# ---------------------------------------------------------------------------

def test_parse_args_rl_defaults():
    args = dispatch.parse_args(["rl"])
    assert args.command == "rl"
    assert args.config == "crypto10_daily"
    assert args.dry_run is False
    assert args.budget_limit == dispatch.DEFAULT_BUDGET_LIMIT
    assert args.time_budget == dispatch.DEFAULT_RL_TIME_BUDGET
    assert args.max_trials == dispatch.DEFAULT_RL_MAX_TRIALS


def test_parse_args_rl_trade_pen_05():
    args = dispatch.parse_args(["rl", "--config", "trade_pen_05"])
    assert args.config == "trade_pen_05"


def test_parse_args_rl_dry_run():
    args = dispatch.parse_args(["rl", "--dry-run"])
    assert args.dry_run is True


def test_parse_args_rl_gpu_type():
    args = dispatch.parse_args(["rl", "--gpu-type", "a100"])
    assert args.gpu_type == "a100"


def test_parse_args_llm_defaults():
    args = dispatch.parse_args(["llm"])
    assert args.command == "llm"
    assert args.dry_run is False
    assert args.gpu_count == dispatch.DEFAULT_LLM_GPU_COUNT
    assert args.max_wallclock == dispatch.DEFAULT_LLM_MAX_WALLCLOCK


def test_parse_args_llm_experiment():
    args = dispatch.parse_args(["llm", "--experiment", "best_record_seed1337"])
    assert args.experiment == "best_record_seed1337"


def test_parse_args_llm_multi_gpu():
    args = dispatch.parse_args(["llm", "--gpu-type", "h100", "--gpu-count", "8"])
    assert args.gpu_type == "h100"
    assert args.gpu_count == 8


def test_parse_args_status():
    args = dispatch.parse_args(["status"])
    assert args.command == "status"


def test_parse_args_cost():
    args = dispatch.parse_args(["cost"])
    assert args.command == "cost"


def test_parse_args_requires_subcommand():
    with pytest.raises(SystemExit):
        dispatch.parse_args([])


# ---------------------------------------------------------------------------
# help output — all subcommands produce output
# ---------------------------------------------------------------------------

def test_help_top_level(capsys):
    with pytest.raises(SystemExit):
        dispatch.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "rl" in out
    assert "llm" in out
    assert "status" in out
    assert "cost" in out


def test_help_rl_subcommand(capsys):
    with pytest.raises(SystemExit):
        dispatch.parse_args(["rl", "--help"])
    out = capsys.readouterr().out
    assert "--config" in out
    assert "--dry-run" in out


def test_help_llm_subcommand(capsys):
    with pytest.raises(SystemExit):
        dispatch.parse_args(["llm", "--help"])
    out = capsys.readouterr().out
    assert "--experiment" in out
    assert "--gpu-count" in out


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_a100_single_gpu():
    # a100 = $1.64/hr; 300s training + 1800s setup = 2100s = 0.583 hr
    cost = dispatch.estimate_cost("a100", 300, gpu_count=1)
    expected = 1.64 * 1 * (2100 / 3600)
    assert abs(cost - expected) < 0.001


def test_estimate_cost_h100_8gpu():
    # h100 = $3.89/hr; 600s + 1800s = 2400s = 0.667 hr, 8 GPUs
    cost = dispatch.estimate_cost("h100", 600, gpu_count=8)
    expected = 3.89 * 8 * (2400 / 3600)
    assert abs(cost - expected) < 0.01


def test_estimate_cost_5090():
    cost = dispatch.estimate_cost("5090", 300, gpu_count=1)
    expected = 1.25 * 1 * (2100 / 3600)
    assert abs(cost - expected) < 0.001


def test_estimate_cost_zero_for_unknown_gpu():
    cost = dispatch.estimate_cost("unknown-gpu-xyz", 300)
    assert cost == 0.0


def test_estimate_cost_scales_with_gpu_count():
    c1 = dispatch.estimate_cost("a100", 300, gpu_count=1)
    c4 = dispatch.estimate_cost("a100", 300, gpu_count=4)
    assert abs(c4 - c1 * 4) < 0.001


# ---------------------------------------------------------------------------
# _enforce_budget
# ---------------------------------------------------------------------------

def test_enforce_budget_allows_under_limit():
    # Should not raise
    dispatch._enforce_budget(2.0, 5.0)


def test_enforce_budget_blocks_over_limit():
    with pytest.raises(SystemExit):
        dispatch._enforce_budget(6.0, 5.0)


def test_enforce_budget_zero_limit_never_blocks():
    # budget_limit=0 means no limit
    dispatch._enforce_budget(1000.0, 0.0)


def test_enforce_budget_exactly_at_limit_allows():
    dispatch._enforce_budget(5.0, 5.0)


def test_enforce_budget_dry_run_warns_not_raises(capsys):
    dispatch._enforce_budget(10.0, 5.0, raise_on_exceed=False)
    out = capsys.readouterr().out
    assert "budget warning" in out
    assert "10.00" in out


def test_cutellm_dispatch_uses_shared_safe_ssh_options() -> None:
    assert dispatch._SSH_OPTS == ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]


def test_cutellm_ssh_run_includes_stderr_context_on_failure(monkeypatch) -> None:
    def _fake_run(cmd, check=False, text=True, capture_output=True):
        return subprocess.CompletedProcess(cmd, 255, stdout="", stderr="permission denied")

    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="SSH command failed"):
        dispatch._ssh_run("1.2.3.4", 22, "echo hi")


def test_cutellm_scp_from_pod_prints_failure_context(monkeypatch, capsys, tmp_path: Path) -> None:
    def _fake_run(cmd, check=False, text=True, capture_output=True):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such file")

    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    ok = dispatch._scp_from_pod("1.2.3.4", 22, "/remote/missing.txt", tmp_path / "out.txt")

    output = capsys.readouterr().out
    assert ok is False
    assert "scp from pod failed for /remote/missing.txt" in output
    assert "no such file" in output


# ---------------------------------------------------------------------------
# load_llm_experiments
# ---------------------------------------------------------------------------

def test_load_llm_experiments_with_valid_file(tmp_path):
    exp_file = tmp_path / "default_experiments.json"
    exp_file.write_text(json.dumps({
        "experiments": [
            {"name": "exp_a", "description": "First", "train_script": "train_gpt.py"},
            {"name": "exp_b", "description": "Second", "train_script": "train_gpt_v2.py"},
        ]
    }))
    with patch.object(dispatch, "CUTELLM_DEFAULT_EXPERIMENTS_JSON", exp_file):
        exps = dispatch.load_llm_experiments()
    assert len(exps) == 2
    assert exps[0]["name"] == "exp_a"


def test_load_llm_experiments_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with patch.object(dispatch, "CUTELLM_DEFAULT_EXPERIMENTS_JSON", missing):
        exps = dispatch.load_llm_experiments()
    assert exps == []


def test_load_llm_experiments_bad_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json }")
    with patch.object(dispatch, "CUTELLM_DEFAULT_EXPERIMENTS_JSON", bad):
        exps = dispatch.load_llm_experiments()
    assert exps == []


# ---------------------------------------------------------------------------
# get_best_bpb
# ---------------------------------------------------------------------------

def test_get_best_bpb_parses_progress_md(tmp_path):
    md = tmp_path / "PROGRESS.md"
    md.write_text(
        "## Status\n"
        "| Our best score | **1.1748 BPB** -- some experiment |\n"
    )
    with patch.object(dispatch, "CUTELLM_PROGRESS_MD", md):
        bpb = dispatch.get_best_bpb()
    assert bpb is not None
    assert abs(bpb - 1.1748) < 1e-6


def test_get_best_bpb_missing_file(tmp_path):
    with patch.object(dispatch, "CUTELLM_PROGRESS_MD", tmp_path / "missing.md"):
        bpb = dispatch.get_best_bpb()
    assert bpb is None


def test_get_best_bpb_no_matching_line(tmp_path):
    md = tmp_path / "PROGRESS.md"
    md.write_text("No BPB info here.\n")
    with patch.object(dispatch, "CUTELLM_PROGRESS_MD", md):
        bpb = dispatch.get_best_bpb()
    assert bpb is None


# ---------------------------------------------------------------------------
# _select_llm_experiment
# ---------------------------------------------------------------------------

def test_select_llm_experiment_by_name():
    exps = [
        {"name": "alpha", "train_script": "a.py"},
        {"name": "beta", "train_script": "b.py"},
    ]
    args = MagicMock()
    args.experiment = "beta"
    selected = dispatch._select_llm_experiment(args, exps)
    assert selected is not None
    assert selected["name"] == "beta"


def test_select_llm_experiment_returns_first_when_no_name():
    exps = [{"name": "first", "train_script": "a.py"}]
    args = MagicMock()
    args.experiment = ""
    selected = dispatch._select_llm_experiment(args, exps)
    assert selected is not None
    assert selected["name"] == "first"


def test_select_llm_experiment_returns_none_when_not_found():
    exps = [{"name": "alpha", "train_script": "a.py"}]
    args = MagicMock()
    args.experiment = "nonexistent"
    selected = dispatch._select_llm_experiment(args, exps)
    assert selected is None


def test_select_llm_experiment_empty_list():
    args = MagicMock()
    args.experiment = "anything"
    selected = dispatch._select_llm_experiment(args, [])
    assert selected is None


# ---------------------------------------------------------------------------
# cmd_status — mock RunPodClient
# ---------------------------------------------------------------------------

def test_cmd_status_no_pods(capsys, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    mock_client = _make_mock_client(pods=[])
    with patch("scripts.cutellm_dispatch.RunPodClient", return_value=mock_client):
        rc = dispatch.cmd_status(MagicMock())
    out = capsys.readouterr().out
    assert "No active pods" in out
    assert rc == 0


def test_cmd_status_shows_pods(capsys, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    pods = [
        _pod("rl-dispatch-crypto10", "p1", "RUNNING"),
        _pod("llm-dispatch-exp", "p2", "RUNNING"),
    ]
    mock_client = _make_mock_client(pods=pods)
    with patch("scripts.cutellm_dispatch.RunPodClient", return_value=mock_client):
        rc = dispatch.cmd_status(MagicMock())
    out = capsys.readouterr().out
    assert "rl-dispatch-crypto10" in out
    assert "llm-dispatch-exp" in out
    assert rc == 0


def test_cmd_status_handles_client_error(capsys, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.list_pods.side_effect = RuntimeError("API down")
    with patch("scripts.cutellm_dispatch.RunPodClient", return_value=mock_client):
        rc = dispatch.cmd_status(MagicMock())
    assert rc == 1


def test_cmd_status_no_api_key(capsys, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with patch("scripts.cutellm_dispatch.RunPodClient", side_effect=ValueError("RUNPOD_API_KEY not set")):
        rc = dispatch.cmd_status(MagicMock())
    assert rc == 1


# ---------------------------------------------------------------------------
# cmd_cost — mock RunPodClient
# ---------------------------------------------------------------------------

def test_cmd_cost_no_pods(capsys, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    mock_client = _make_mock_client(pods=[])
    with patch("scripts.cutellm_dispatch.RunPodClient", return_value=mock_client):
        rc = dispatch.cmd_cost(MagicMock())
    out = capsys.readouterr().out
    assert "$0.00" in out
    assert rc == 0


def test_cmd_cost_categorises_rl_pods(capsys, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    pods = [
        _pod("rl-dispatch-crypto10", "p1", "RUNNING", "NVIDIA GeForce RTX 5090"),
    ]
    mock_client = _make_mock_client(pods=pods)
    with patch("scripts.cutellm_dispatch.RunPodClient", return_value=mock_client):
        rc = dispatch.cmd_cost(MagicMock())
    out = capsys.readouterr().out
    assert "RL" in out
    assert "stock-prediction" in out
    assert rc == 0


def test_cmd_cost_categorises_llm_pods(capsys, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    pods = [
        _pod("llm-dispatch-exp1", "p2", "RUNNING", "NVIDIA H100 80GB HBM3"),
    ]
    mock_client = _make_mock_client(pods=pods)
    with patch("scripts.cutellm_dispatch.RunPodClient", return_value=mock_client):
        rc = dispatch.cmd_cost(MagicMock())
    out = capsys.readouterr().out
    assert "LLM" in out or "cutellm" in out
    assert rc == 0


def test_cmd_cost_computes_correct_rates(capsys, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    pods = [
        _pod("rl-dispatch-a", "p1", "RUNNING", "NVIDIA GeForce RTX 5090"),
        _pod("llm-dispatch-b", "p2", "RUNNING", "NVIDIA H100 80GB HBM3"),
    ]
    mock_client = _make_mock_client(pods=pods)
    with patch("scripts.cutellm_dispatch.RunPodClient", return_value=mock_client):
        rc = dispatch.cmd_cost(MagicMock())
    out = capsys.readouterr().out
    # 5090 = $1.25/hr, H100 = $3.89/hr, total = $5.14/hr
    assert "1.25" in out
    assert "3.89" in out
    assert rc == 0


# ---------------------------------------------------------------------------
# dry-run rl — end-to-end output check
# ---------------------------------------------------------------------------

def test_rl_dry_run_trade_pen_05(capsys):
    args = dispatch.parse_args(["rl", "--config", "trade_pen_05", "--dry-run"])
    rc = dispatch.main(["rl", "--config", "trade_pen_05", "--dry-run"])
    out = capsys.readouterr().out
    assert "trade_pen_05" in out
    assert "dry-run" in out.lower()
    assert rc == 0


def test_rl_dry_run_default_config(capsys):
    rc = dispatch.main(["rl", "--dry-run"])
    out = capsys.readouterr().out
    assert "crypto10_daily" in out
    assert rc == 0


def test_rl_dry_run_shows_cost_estimate(capsys):
    rc = dispatch.main(["rl", "--dry-run", "--gpu-type", "a100"])
    out = capsys.readouterr().out
    assert "$" in out
    assert rc == 0


def test_rl_dry_run_shows_steps(capsys):
    rc = dispatch.main(["rl", "--dry-run"])
    out = capsys.readouterr().out
    assert "Steps:" in out
    assert "autoresearch_rl" in out


# ---------------------------------------------------------------------------
# dry-run llm — end-to-end output check
# ---------------------------------------------------------------------------

def test_llm_dry_run_default(capsys):
    with patch.object(dispatch, "CUTELLM_DEFAULT_EXPERIMENTS_JSON", Path("/nonexistent")):
        rc = dispatch.main(["llm", "--dry-run"])
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "LLM" in out
    assert rc == 0


def test_llm_dry_run_shows_cutellm_root(capsys):
    with patch.object(dispatch, "CUTELLM_DEFAULT_EXPERIMENTS_JSON", Path("/nonexistent")):
        rc = dispatch.main(["llm", "--dry-run"])
    out = capsys.readouterr().out
    assert "cutellm" in out.lower() or "CuteLLM" in out
    assert rc == 0


def test_llm_dry_run_shows_cost(capsys):
    with patch.object(dispatch, "CUTELLM_DEFAULT_EXPERIMENTS_JSON", Path("/nonexistent")):
        rc = dispatch.main(["llm", "--dry-run", "--gpu-type", "h100", "--gpu-count", "8"])
    out = capsys.readouterr().out
    assert "$" in out
    assert rc == 0


def test_llm_dry_run_with_experiments_file(capsys, tmp_path):
    exp_file = tmp_path / "exps.json"
    exp_file.write_text(json.dumps({
        "experiments": [
            {
                "name": "my_experiment",
                "description": "Test description for dry run",
                "train_script": "parameter-golf/train_gpt.py",
                "env_overrides": {"SEED": "1337"},
            }
        ]
    }))
    with patch.object(dispatch, "CUTELLM_DEFAULT_EXPERIMENTS_JSON", exp_file):
        rc = dispatch.main(["llm", "--dry-run", "--experiment", "my_experiment"])
    out = capsys.readouterr().out
    assert "my_experiment" in out
    assert rc == 0


def test_llm_dry_run_shows_best_bpb(capsys, tmp_path):
    md = tmp_path / "PROGRESS.md"
    md.write_text("| Our best score | **1.1748 BPB** -- notapplica |\n")
    with patch.object(dispatch, "CUTELLM_PROGRESS_MD", md), \
         patch.object(dispatch, "CUTELLM_DEFAULT_EXPERIMENTS_JSON", tmp_path / "no.json"):
        rc = dispatch.main(["llm", "--dry-run"])
    out = capsys.readouterr().out
    assert "1.1748" in out
    assert rc == 0


# ---------------------------------------------------------------------------
# dispatch_llm — runpod mocked; cutellm root mocked
# ---------------------------------------------------------------------------

def test_dispatch_llm_missing_cutellm_root(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    args = dispatch.parse_args(["llm"])
    with patch.object(dispatch, "CUTELLM_ROOT", tmp_path / "nonexistent"):
        rc = dispatch.dispatch_llm(args)
    out = capsys.readouterr().out
    assert "not found" in out
    assert rc == 1


def test_dispatch_llm_budget_exceeded(capsys, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    args = dispatch.parse_args(["llm", "--gpu-count", "8", "--gpu-type", "h100",
                                 "--budget-limit", "0.01"])
    with pytest.raises(SystemExit):
        dispatch.dispatch_llm(args)


def test_dispatch_llm_no_api_key(capsys, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    args = dispatch.parse_args(["llm"])
    with patch.object(dispatch, "CUTELLM_ROOT", Path("/tmp")), \
         patch("scripts.cutellm_dispatch.RunPodClient", side_effect=ValueError("RUNPOD_API_KEY not set")):
        rc = dispatch.dispatch_llm(args)
    assert rc == 1


# ---------------------------------------------------------------------------
# dispatch_rl — delegates to dispatch_rl_training; we mock the module
# ---------------------------------------------------------------------------

def test_dispatch_rl_delegates_to_dispatch_rl_training(monkeypatch):
    """dispatch_rl() should load and call dispatch_rl_training.main() with correct argv."""
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    args = dispatch.parse_args(["rl", "--config", "trade_pen_05"])

    mock_mod = MagicMock()
    mock_mod.main.return_value = 0

    with patch("importlib.util.spec_from_file_location") as mock_spec_fn, \
         patch("importlib.util.module_from_spec", return_value=mock_mod):
        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_spec_fn.return_value = mock_spec

        rc = dispatch.dispatch_rl(args)

    assert rc == 0
    assert mock_mod.main.called
    call_argv = mock_mod.main.call_args[0][0]
    assert "--data-train" in call_argv
    assert "--data-val" in call_argv
    assert "--force-remote" in call_argv
    assert "trade_pen_05" in call_argv  # description forwarded


def test_dispatch_rl_missing_dispatch_script(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    args = dispatch.parse_args(["rl"])
    with patch.object(dispatch, "REPO", tmp_path):
        rc = dispatch.dispatch_rl(args)
    out = capsys.readouterr().out
    assert "not found" in out
    assert rc == 1


# ---------------------------------------------------------------------------
# RL preset configs coverage
# ---------------------------------------------------------------------------

def test_rl_preset_configs_all_have_required_keys():
    for name, cfg in dispatch.RL_PRESET_CONFIGS.items():
        assert "data_train" in cfg, f"{name} missing data_train"
        assert "data_val" in cfg, f"{name} missing data_val"
        assert "note" in cfg, f"{name} missing note"


def test_rl_preset_configs_includes_trade_pen_05():
    assert "trade_pen_05" in dispatch.RL_PRESET_CONFIGS


def test_rl_preset_configs_includes_fdusd():
    assert "fdusd3_daily" in dispatch.RL_PRESET_CONFIGS
    assert "fdusd3_hourly" in dispatch.RL_PRESET_CONFIGS


# ---------------------------------------------------------------------------
# main() routes correctly
# ---------------------------------------------------------------------------

def test_main_status_called(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    mock_client = _make_mock_client(pods=[])
    with patch("scripts.cutellm_dispatch.RunPodClient", return_value=mock_client):
        rc = dispatch.main(["status"])
    assert rc == 0


def test_main_cost_called(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    mock_client = _make_mock_client(pods=[])
    with patch("scripts.cutellm_dispatch.RunPodClient", return_value=mock_client):
        rc = dispatch.main(["cost"])
    assert rc == 0


def test_main_rl_dry_run_returns_zero():
    rc = dispatch.main(["rl", "--dry-run"])
    assert rc == 0


def test_main_llm_dry_run_returns_zero():
    with patch.object(dispatch, "CUTELLM_DEFAULT_EXPERIMENTS_JSON", Path("/nonexistent")):
        rc = dispatch.main(["llm", "--dry-run"])
    assert rc == 0
