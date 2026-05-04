from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
CONFIDENCE_SCRIPT = REPO / "scripts" / "screened32_confidence_gate.py"
AGREEMENT_SCRIPT = REPO / "scripts" / "screened32_agreement_gate.py"


def _load_module(script: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script", "name", "gate_flag", "gate_bad_value"),
    [
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--thresholds", "0.1,nan"),
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--thresholds", "0.1,0.1"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--min-agree-counts", "1,nope"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--min-agree-counts", "1,1"),
    ],
)
def test_uncertainty_gates_reject_invalid_gate_grid_before_data_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: Path,
    name: str,
    gate_flag: str,
    gate_bad_value: str,
) -> None:
    module = _load_module(script, name)
    read_calls = 0

    def fail_if_called(path):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("read_mktd should not be called")

    monkeypatch.setattr(module, "read_mktd", fail_if_called)

    rc = module.main(
        [
            "--val-data",
            str(tmp_path / "missing.bin"),
            gate_flag,
            gate_bad_value,
            "--out",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert read_calls == 0
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize(
    ("script", "name", "flag", "bad_value"),
    [
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--decision-lag", "1"),
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--window-days", "0"),
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--fill-buffer-bps", "-1"),
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--fee-rate", "nan"),
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--slippage-bps", "-1"),
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--short-borrow-apr", "inf"),
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--max-leverage", "0"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--decision-lag", "1"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--window-days", "0"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--fill-buffer-bps", "-1"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--fee-rate", "nan"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--slippage-bps", "-1"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--short-borrow-apr", "inf"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--max-leverage", "0"),
    ],
)
def test_uncertainty_gates_reject_invalid_realism_config_before_data_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: Path,
    name: str,
    flag: str,
    bad_value: str,
) -> None:
    module = _load_module(script, name)
    read_calls = 0

    def fail_if_called(path):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("read_mktd should not be called")

    monkeypatch.setattr(module, "read_mktd", fail_if_called)

    rc = module.main(
        [
            "--val-data",
            str(tmp_path / "missing.bin"),
            flag,
            bad_value,
            "--out",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert read_calls == 0
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize(
    ("script", "name", "gate_flag", "gate_value"),
    [
        (CONFIDENCE_SCRIPT, "screened32_confidence_gate", "--thresholds", "0.2"),
        (AGREEMENT_SCRIPT, "screened32_agreement_gate", "--min-agree-counts", "2"),
    ],
)
def test_uncertainty_gates_low_lag_opt_in_reaches_data_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: Path,
    name: str,
    gate_flag: str,
    gate_value: str,
) -> None:
    module = _load_module(script, name)
    val_path = tmp_path / "val.bin"
    val_path.write_bytes(b"stub")
    read_calls = 0

    def fake_read_mktd(path):
        nonlocal read_calls
        read_calls += 1
        return SimpleNamespace(num_timesteps=1, num_symbols=1)

    monkeypatch.setattr(module, "read_mktd", fake_read_mktd)

    rc = module.main(
        [
            "--val-data",
            str(val_path),
            "--decision-lag",
            "1",
            "--allow-low-lag-diagnostics",
            "--window-days",
            "2",
            gate_flag,
            gate_value,
            "--out",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert read_calls == 1
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize(
    ("script", "name", "gate_flag", "gate_value", "result_key", "expected_gate_value"),
    [
        (
            CONFIDENCE_SCRIPT,
            "screened32_confidence_gate",
            "--thresholds",
            "0.25",
            "threshold",
            0.25,
        ),
        (
            AGREEMENT_SCRIPT,
            "screened32_agreement_gate",
            "--min-agree-counts",
            "3",
            "min_agree_count",
            3,
        ),
    ],
)
def test_uncertainty_gates_write_cost_provenance_through_atomic_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: Path,
    name: str,
    gate_flag: str,
    gate_value: str,
    result_key: str,
    expected_gate_value: float | int,
) -> None:
    module = _load_module(script, name)
    val_path = tmp_path / "val.bin"
    out_path = tmp_path / "gate.json"
    val_path.write_bytes(b"stub")

    monkeypatch.setattr(module, "DEFAULT_CHECKPOINT", "base_a.pt")
    monkeypatch.setattr(module, "DEFAULT_EXTRA_CHECKPOINTS", ())
    monkeypatch.setattr(
        module,
        "read_mktd",
        lambda path: SimpleNamespace(num_timesteps=3, num_symbols=1),
    )
    monkeypatch.setattr(
        module,
        "load_policy",
        lambda *args, **kwargs: SimpleNamespace(
            policy=object(),
            action_allocation_bins=1,
            action_level_bins=1,
            action_max_offset_bps=0.0,
        ),
    )
    monkeypatch.setattr(
        module,
        "_slice_window",
        lambda data, *, start, steps: SimpleNamespace(start=start, steps=steps),
    )
    observed_sim: list[dict] = []

    def fake_simulate_daily_policy(window, policy_fn, **kwargs):
        observed_sim.append(dict(kwargs))
        return SimpleNamespace(total_return=0.01, sortino=2.0, max_drawdown=0.03)

    monkeypatch.setattr(module, "simulate_daily_policy", fake_simulate_daily_policy)

    rc = module.main(
        [
            "--val-data",
            str(val_path),
            "--device",
            "cpu",
            "--window-days",
            "1",
            gate_flag,
            gate_value,
            "--fee-rate",
            "0.001",
            "--slippage-bps",
            "20",
            "--fill-buffer-bps",
            "5",
            "--short-borrow-apr",
            "0.0625",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    assert {call["fee_rate"] for call in observed_sim} == {0.001}
    assert {call["slippage_bps"] for call in observed_sim} == {20.0}
    assert {call["fill_buffer_bps"] for call in observed_sim} == {5.0}
    assert {call["short_borrow_apr"] for call in observed_sim} == {0.0625}

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["decision_lag"] == 2
    assert payload["fee_rate"] == 0.001
    assert payload["slippage_bps"] == 20.0
    assert payload["fill_buffer_bps"] == 5.0
    assert payload["short_borrow_apr"] == 0.0625
    assert payload["max_leverage"] == 1.0
    assert payload["results"][0][result_key] == expected_gate_value


@pytest.mark.parametrize("script", [CONFIDENCE_SCRIPT, AGREEMENT_SCRIPT])
def test_uncertainty_gate_scripts_use_shared_atomic_json_writer(script: Path) -> None:
    source = script.read_text(encoding="utf-8")

    assert "from xgbnew.artifacts import write_json_atomic" in source
    assert ".write_text(json.dumps(" not in source
