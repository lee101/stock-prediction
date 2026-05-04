from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "screened32_realism_gate_topk.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("screened32_realism_gate_topk", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("flag", "bad_value"),
    [
        ("--decision-lag", "1"),
        ("--decision-lag", "-1"),
        ("--window-days", "0"),
        ("--fill-buffer-bps", "-1"),
        ("--fee-rate", "nan"),
        ("--slippage-bps", "-1"),
        ("--short-borrow-apr", "inf"),
        ("--max-leverage", "0"),
        ("--min-prob-ratios", "0.3,nan"),
        ("--min-prob-ratios", "0.3,0.3"),
    ],
)
def test_main_rejects_invalid_config_before_data_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flag: str,
    bad_value: str,
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
            "--val-data",
            str(tmp_path / "missing.bin"),
            flag,
            bad_value,
            "--out-json",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert read_calls == 0
    assert not (tmp_path / "out.json").exists()


def test_low_lag_diagnostic_opt_in_reaches_data_validation(tmp_path: Path) -> None:
    module = _load_module()

    rc = module.main(
        [
            "--val-data",
            str(tmp_path / "missing.bin"),
            "--decision-lag",
            "1",
            "--allow-low-lag-diagnostics",
            "--out-json",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert not (tmp_path / "out.json").exists()


def test_main_writes_cost_provenance_through_atomic_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    val_path = tmp_path / "val.bin"
    out_path = tmp_path / "topk.json"
    val_path.write_bytes(b"stub")

    monkeypatch.setattr(module, "DEFAULT_CHECKPOINT", "base_a.pt")
    monkeypatch.setattr(module, "DEFAULT_EXTRA_CHECKPOINTS", ())
    monkeypatch.setattr(
        module,
        "read_mktd",
        lambda path: SimpleNamespace(num_timesteps=3, num_symbols=1, features=SimpleNamespace(shape=(3, 1, 1))),
    )

    observed_sim: list[dict] = []

    def fake_build_topk_policy_fn(**kwargs):
        def policy_fn(obs):
            return 0

        def reset_buffer():
            return None

        head = SimpleNamespace(
            action_allocation_bins=1,
            action_level_bins=1,
            action_max_offset_bps=0.0,
        )
        return policy_fn, reset_buffer, head

    def fake_slice_window(data, *, start, steps):
        return SimpleNamespace(start=start, steps=steps)

    def fake_simulate_daily_policy(window, policy_fn, **kwargs):
        observed_sim.append(dict(kwargs))
        return SimpleNamespace(total_return=0.01, sortino=1.5, max_drawdown=0.02)

    monkeypatch.setattr(module, "build_topk_policy_fn", fake_build_topk_policy_fn)
    monkeypatch.setattr(module, "_slice_window", fake_slice_window)
    monkeypatch.setattr(module, "simulate_daily_policy", fake_simulate_daily_policy)

    rc = module.main(
        [
            "--val-data",
            str(val_path),
            "--device",
            "cpu",
            "--window-days",
            "1",
            "--min-prob-ratios",
            "0.5",
            "--fee-rate",
            "0.001",
            "--slippage-bps",
            "20",
            "--fill-buffer-bps",
            "5",
            "--short-borrow-apr",
            "0.0625",
            "--out-json",
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
    assert len(payload["rows"]) == 1


def test_script_uses_shared_atomic_json_writer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "from xgbnew.artifacts import write_json_atomic" in source
    assert ".write_text(json.dumps(" not in source
