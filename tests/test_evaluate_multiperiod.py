"""Tests for multi-period evaluation."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pufferlib_market import evaluate_multiperiod as eval_mod


REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_CANDIDATES = [
    REPO / "pufferlib_market/checkpoints/autoresearch/longonly_forecast/best.pt",
    REPO / "pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt",
    REPO / "pufferlib_market/checkpoints/autoresearch_daily/slip_5bps/best.pt",
]
DATA_PATH = REPO / "pufferlib_market/data/crypto6_val.bin"


def _checkpoint_for_test() -> str:
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    pytest.skip("no compatible multiperiod smoke-test checkpoint is present in this workspace")


def test_multiperiod_runs_on_deployed_model():
    """Smoke test: run multi-period eval on the deployed longonly_forecast checkpoint."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.evaluate_multiperiod",
            "--checkpoint", _checkpoint_for_test(),
            "--data-path", str(DATA_PATH),
            "--deterministic", "--disable-shorts",
            "--periods", "1d,7d",
            "--json",
        ],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert len(data) == 1  # one checkpoint
    results = data[0]
    assert len(results) == 2  # 1d, 7d
    for r in results:
        assert "total_return" in r
        assert "sortino" in r
        assert "period" in r
        assert "arch" in r
        assert "hidden_size" in r
        assert "action_allocation_bins" in r
        assert "action_level_bins" in r
        assert "action_max_offset_bps" in r


def test_multiperiod_table_output():
    """Verify table output format."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.evaluate_multiperiod",
            "--checkpoint", _checkpoint_for_test(),
            "--data-path", str(DATA_PATH),
            "--deterministic", "--disable-shorts",
            "--periods", "1d",
        ],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0
    assert "Period" in result.stdout
    assert "Return%" in result.stdout
    assert "1d" in result.stdout


def test_multiperiod_custom_period():
    """Test custom period specification (e.g. 48h)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.evaluate_multiperiod",
            "--checkpoint", _checkpoint_for_test(),
            "--data-path", str(DATA_PATH),
            "--deterministic", "--disable-shorts",
            "--periods", "48h",
            "--json",
        ],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data[0][0]["eval_hours"] == 48


def test_multiperiod_rejects_nonpositive_period() -> None:
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.evaluate_multiperiod",
            "--checkpoint", "dummy.pt",
            "--data-path", str(DATA_PATH),
            "--periods", "0h",
        ],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode != 0
    assert "Period must be at least 1 hour: 0h" in result.stderr



def test_multiperiod_rejects_empty_checkpoint_list() -> None:
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.evaluate_multiperiod",
            "--checkpoints", " , , ",
            "--data-path", str(DATA_PATH),
        ],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode != 0
    assert "--checkpoints must include at least one checkpoint path" in result.stderr


def test_load_policy_rejects_metadata_mapping_without_model(tmp_path: Path):
    with (
        patch.object(eval_mod, "load_checkpoint_payload", return_value={"arch": "mlp", "hidden_size": 16}),
        pytest.raises(ValueError, match=r"Unsupported checkpoint format \(expected state_dict or dict with 'model'\)"),
    ):
        eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            1,
            arch="mlp",
            hidden_size=16,
            device=torch.device("cpu"),
        )


def test_load_policy_rejects_wrapped_checkpoint_with_invalid_model_payload(tmp_path: Path):
    with (
        patch.object(eval_mod, "load_checkpoint_payload", return_value={"model": {"arch": "mlp"}}),
        pytest.raises(KeyError, match="Checkpoint is missing a valid 'model' state_dict"),
    ):
        eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            1,
            arch="mlp",
            hidden_size=16,
            device=torch.device("cpu"),
        )


def test_load_policy_supports_bare_state_dict_payload(tmp_path: Path):
    num_symbols = 1
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols
    source_policy = eval_mod.TradingPolicy(obs_size, num_actions, hidden=16)
    source_state_dict = source_policy.state_dict()

    with patch.object(eval_mod, "load_checkpoint_payload", return_value=source_state_dict):
        loaded = eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols,
            arch="mlp",
            hidden_size=16,
            device=torch.device("cpu"),
        )

    assert isinstance(loaded.policy, eval_mod.TradingPolicy)
    assert loaded.num_actions == num_actions
    assert loaded.arch == "mlp"
    assert loaded.hidden_size == 16
    assert loaded.action_allocation_bins == 1
    assert loaded.action_level_bins == 1
    assert loaded.action_max_offset_bps == pytest.approx(0.0)


def test_load_policy_ignores_invalid_action_grid_metadata(tmp_path: Path):
    num_symbols = 1
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols
    source_policy = eval_mod.TradingPolicy(obs_size, num_actions, hidden=16)
    payload = {
        "model": source_policy.state_dict(),
        "action_allocation_bins": "oops",
        "action_level_bins": float("nan"),
    }

    with patch.object(eval_mod, "load_checkpoint_payload", return_value=payload):
        loaded = eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols,
            arch="mlp",
            hidden_size=16,
            device=torch.device("cpu"),
        )

    assert isinstance(loaded.policy, eval_mod.TradingPolicy)
    assert loaded.num_actions == num_actions
    assert loaded.action_allocation_bins == 1
    assert loaded.action_level_bins == 1


def test_load_policy_preserves_valid_action_max_offset_bps_metadata(tmp_path: Path):
    num_symbols = 1
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols
    source_policy = eval_mod.TradingPolicy(obs_size, num_actions, hidden=16)
    payload = {
        "model": source_policy.state_dict(),
        "action_max_offset_bps": 12.5,
    }

    with patch.object(eval_mod, "load_checkpoint_payload", return_value=payload):
        loaded = eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols,
            arch="mlp",
            hidden_size=16,
            device=torch.device("cpu"),
        )

    assert isinstance(loaded.policy, eval_mod.TradingPolicy)
    assert loaded.num_actions == num_actions
    assert loaded.action_allocation_bins == 1
    assert loaded.action_level_bins == 1
    assert loaded.action_max_offset_bps == pytest.approx(12.5)


def test_load_policy_rejects_incompatible_action_grid_metadata(tmp_path: Path):
    num_symbols = 1
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols
    source_policy = eval_mod.TradingPolicy(obs_size, num_actions, hidden=16)
    payload = {
        "model": source_policy.state_dict(),
        "action_allocation_bins": 2,
        "action_level_bins": 3,
    }

    with (
        patch.object(eval_mod, "load_checkpoint_payload", return_value=payload),
        pytest.raises(ValueError, match=r"Only supports alloc_bins=1 level_bins=1 \(got 2, 3\)"),
    ):
        eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols,
            arch="mlp",
            hidden_size=16,
            device=torch.device("cpu"),
        )


def test_evaluate_checkpoint_includes_effective_policy_metadata(tmp_path: Path):
    fake_data = SimpleNamespace(
        num_symbols=1,
        features=np.zeros((32, 1, 16), dtype=np.float32),
        symbols=["BTCUSDT"],
    )
    loaded = eval_mod.LoadedPolicy(
        policy=object(),
        arch="mlp",
        hidden_size=16,
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
        num_actions=3,
    )

    with (
        patch.object(eval_mod, "read_mktd", return_value=fake_data),
        patch.object(eval_mod, "load_policy", return_value=loaded),
        patch.object(
            eval_mod,
            "evaluate_period",
            return_value={
                "eval_hours": 24,
                "total_return": 0.1,
                "annualized_return": 0.15,
                "sortino": 1.2,
                "max_drawdown": 0.05,
                "num_trades": 4,
                "win_rate": 0.75,
                "avg_hold_steps": 6.0,
            },
        ),
    ):
        results = eval_mod.evaluate_checkpoint(
            str(tmp_path / "checkpoint.pt"),
            str(tmp_path / "data.bin"),
            {"1d": 24},
            arch="auto",
            hidden_size=None,
        )

    assert results == [
        {
            "eval_hours": 24,
            "total_return": 0.1,
            "annualized_return": 0.15,
            "sortino": 1.2,
            "max_drawdown": 0.05,
            "num_trades": 4,
            "win_rate": 0.75,
            "avg_hold_steps": 6.0,
            "period": "1d",
            "checkpoint": str(tmp_path / "checkpoint.pt"),
            "data_path": str(tmp_path / "data.bin"),
            "arch": "mlp",
            "hidden_size": 16,
            "action_allocation_bins": 1,
            "action_level_bins": 1,
            "action_max_offset_bps": 0.0,
        }
    ]


def test_format_table_includes_checkpoint_identity_effective_config_and_period_summary() -> None:
    table = eval_mod.format_table(
        [[
            {
                "period": "1d",
                "total_return": 0.1,
                "sortino": 1.2,
                "max_drawdown": 0.05,
                "num_trades": 4,
                "win_rate": 0.75,
                "avg_hold_steps": 6.0,
                "arch": "mlp",
                "hidden_size": 16,
                "action_allocation_bins": 1,
                "action_level_bins": 1,
                "action_max_offset_bps": 0.04,
            },
            {
                "period": "7d",
                "total_return": -0.2,
                "sortino": -0.3,
                "max_drawdown": 0.12,
                "num_trades": 2,
                "win_rate": 0.25,
                "avg_hold_steps": 10.0,
                "arch": "mlp",
                "hidden_size": 16,
                "action_allocation_bins": 1,
                "action_level_bins": 1,
                "action_max_offset_bps": 0.04,
            },
        ]],
        ["runs/best.pt"],
    )

    assert "Checkpoint: runs/best.pt" in table
    assert "Effective config: arch=mlp, hidden_size=16" in table
    assert "Action grid: alloc_bins=1 level_bins=1 max_offset_bps=0.04" in table
    assert "Period" in table
    assert "1d" in table
    assert "Best period: 1d (+10.00%)" in table
    assert "Worst period: 7d (-20.00%)" in table
    assert "Positive periods: 1/2" in table
