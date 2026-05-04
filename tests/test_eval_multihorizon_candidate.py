from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval_multihorizon_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_multihorizon_candidate", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_start_indices_is_deterministic():
    module = _load_module()
    starts_a = module.build_start_indices(
        num_timesteps=263,
        eval_days=100,
        n_windows=12,
        seed=1337,
    )
    starts_b = module.build_start_indices(
        num_timesteps=263,
        eval_days=100,
        n_windows=12,
        seed=1337,
    )
    assert starts_a == starts_b
    assert len(starts_a) == 12
    assert starts_a == sorted(starts_a)
    assert len(set(starts_a)) == len(starts_a)


def test_build_start_indices_recent_tail_bounds():
    module = _load_module()
    starts = module.build_start_indices(
        num_timesteps=263,
        eval_days=60,
        n_windows=10,
        seed=7,
        recent_within_days=90,
    )
    assert len(starts) == 10
    assert min(starts) >= 112
    assert max(starts) <= 202


def test_choose_recommendation_flags_promising_additive():
    module = _load_module()
    report = {
        "scenarios": {
            "baseline": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.10},
                }
            },
            "candidate": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.09},
                }
            },
            "baseline_plus_candidate": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.13},
                }
            },
        },
        "comparisons": {
            "candidate_vs_baseline": {
                "mean_delta_median_monthly_return": -0.01,
                "mean_delta_negative_windows": 0.5,
            },
            "baseline_plus_candidate_vs_baseline": {
                "mean_delta_median_monthly_return": 0.03,
                "mean_delta_negative_windows": -1.0,
            },
        },
    }
    rec = module.choose_recommendation(report)
    assert rec["status"] == "promising_additive"


def test_choose_recommendation_rejects_unproven_candidate():
    module = _load_module()
    report = {
        "scenarios": {
            "baseline": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.11},
                }
            },
            "candidate": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.08},
                }
            },
            "baseline_plus_candidate": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.10},
                }
            },
        },
        "comparisons": {
            "candidate_vs_baseline": {
                "mean_delta_median_monthly_return": -0.02,
                "mean_delta_negative_windows": 1.0,
            },
            "baseline_plus_candidate_vs_baseline": {
                "mean_delta_median_monthly_return": -0.01,
                "mean_delta_negative_windows": 0.25,
            },
        },
    }
    rec = module.choose_recommendation(report)
    assert rec["status"] == "not_proven"


def test_run_holdout_forwards_short_borrow_apr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    data_path = tmp_path / "data.bin"
    data_path.write_bytes(b"stub")
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(part) for part in cmd])
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "median_total_return": 0.1,
                        "p10_total_return": 0.05,
                        "negative_windows": 0,
                        "median_sortino": 2.0,
                    }
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    summary = module._run_holdout(
        scenario=module.Scenario(name="candidate", checkpoint="candidate.pt", extra_checkpoints=("extra.pt",)),
        data_path=data_path,
        eval_days=100,
        start_indices=[0, 1],
        fee_rate=0.001,
        slippage_bps=20,
        fill_buffer_bps=5.0,
        short_borrow_apr=0.0625,
        decision_lag=2,
        disable_shorts=True,
    )

    assert summary["slippage_bps"] == 20
    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[cmd.index("--short-borrow-apr") + 1] == "0.0625"
    assert "--disable-shorts" in cmd
    assert cmd[cmd.index("--extra-checkpoints") + 1] == "extra.pt"


def test_run_holdout_forwards_low_lag_diagnostic_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    data_path = tmp_path / "data.bin"
    data_path.write_bytes(b"stub")
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(part) for part in cmd])
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_text(
            json.dumps({"summary": {"median_total_return": 0.1, "p10_total_return": 0.05}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_holdout(
        scenario=module.Scenario(name="candidate", checkpoint="candidate.pt", extra_checkpoints=()),
        data_path=data_path,
        eval_days=30,
        start_indices=[0],
        fee_rate=0.001,
        slippage_bps=5,
        fill_buffer_bps=5.0,
        short_borrow_apr=0.0625,
        decision_lag=1,
        disable_shorts=True,
        allow_low_lag_diagnostics=True,
    )

    assert "--allow-low-lag-diagnostics" in commands[0]


@pytest.mark.parametrize(
    ("flag", "bad_value"),
    [
        ("--short-borrow-apr", "nan"),
        ("--short-borrow-apr", "-0.1"),
        ("--fee-rate", "-0.001"),
        ("--fill-buffer-bps", "-1"),
        ("--decision-lag", "1"),
        ("--horizons-days", "30,0"),
        ("--slippage-bps", "5,nan"),
        ("--n-windows", "0"),
        ("--recent-within-days", "-1"),
    ],
)
def test_main_rejects_invalid_config_before_data_load(
    flag: str,
    bad_value: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    read_calls = 0

    def fail_if_called(path):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("read_mktd should not be called")

    monkeypatch.setattr(module, "read_mktd", fail_if_called)

    rc = module.main(
        [
            "--data-path",
            str(tmp_path / "missing.bin"),
            flag,
            bad_value,
            "--out",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert read_calls == 0


def test_main_low_lag_diagnostic_opt_in_reaches_data_validation(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(FileNotFoundError, match="data path not found"):
        module.main(
            [
                "--data-path",
                str(tmp_path / "missing.bin"),
                "--decision-lag",
                "1",
                "--allow-low-lag-diagnostics",
                "--out",
                str(tmp_path / "out.json"),
            ]
        )


def test_script_uses_shared_atomic_json_writer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "from xgbnew.artifacts import write_json_atomic" in source
    assert ".write_text(json.dumps(report" not in source
