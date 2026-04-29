from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval_when_done.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_when_done", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exhaustive_eval_applies_realism_and_resets_decision_lag(monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    calls: list[dict[str, object]] = []

    class DummyPolicy:
        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=False):
            return None

        def state_dict(self):
            return {"weight": torch.zeros((4, 5))}

        def __call__(self, obs_t):
            return torch.tensor([[0.0, 0.0, 0.0, 1.0]]), None

    def fake_simulate_daily_policy(window, policy_fn, **kwargs):
        actions = [
            policy_fn(np.zeros(5, dtype=np.float32)),
            policy_fn(np.zeros(5, dtype=np.float32)),
        ]
        calls.append({"window": window, "actions": actions, **kwargs})
        return SimpleNamespace(total_return=0.10)

    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: {"model": {"weight": torch.zeros((4, 5))}})
    monkeypatch.setattr(module, "_infer_arch", lambda state_dict: "mlp")
    monkeypatch.setattr(module, "_infer_hidden_size", lambda state_dict, arch: 4)
    monkeypatch.setattr(module, "_infer_num_actions", lambda state_dict, fallback: 4)
    monkeypatch.setattr(module, "TradingPolicy", lambda **kwargs: DummyPolicy())
    monkeypatch.setattr(module, "_slice_window", lambda data, start, steps: f"window-{start}-{steps}")
    monkeypatch.setattr(module, "simulate_daily_policy", fake_simulate_daily_policy)

    result = module.exhaustive_eval(
        Path("checkpoint.pt"),
        SimpleNamespace(num_timesteps=3),
        eval_steps=1,
        fee=0.001,
        fill_bps=5.0,
        slippage_bps=20.0,
        short_borrow_apr=0.0625,
        decision_lag=1,
        max_hold_bars=6,
        max_leverage=1.5,
    )

    assert result["n"] == 2
    assert [call["actions"] for call in calls] == [[0, 3], [0, 3]]
    assert {call["fee_rate"] for call in calls} == {0.001}
    assert {call["fill_buffer_bps"] for call in calls} == {5.0}
    assert {call["slippage_bps"] for call in calls} == {20.0}
    assert {call["short_borrow_apr"] for call in calls} == {0.0625}
    assert {call["max_hold_bars"] for call in calls} == {6}
    assert {call["max_leverage"] for call in calls} == {1.5}
    assert {call["enable_drawdown_profit_early_exit"] for call in calls} == {False}


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--fee", "nan"),
        ("--fill-bps", "-1"),
        ("--slippage-bps", "inf"),
        ("--short-borrow-apr", "-0.01"),
        ("--decision-lag", "-1"),
        ("--max-hold-bars", "-1"),
        ("--max-leverage", "0"),
    ],
)
def test_main_rejects_invalid_realism_args_before_loading_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
):
    module = _load_module()

    monkeypatch.setattr(
        module,
        "read_mktd",
        lambda path: (_ for _ in ()).throw(AssertionError("read_mktd should not be called")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_when_done.py",
            "--ckpt-root",
            str(tmp_path / "ckpts"),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--out",
            str(tmp_path / "out.csv"),
            flag,
            value,
        ],
    )

    assert module.main() == 2
