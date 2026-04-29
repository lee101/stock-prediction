from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval_stocks12_seeds.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_stocks12_seeds", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_eval_checkpoint_applies_realism_and_resets_decision_lag(monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    calls: list[dict[str, object]] = []

    class DummyPolicy:
        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=False):
            return None

        def __call__(self, obs_t):
            return torch.tensor([[0.0, 0.0, 0.0, 1.0]]), None

    def fake_simulate_daily_policy(window, policy_fn, **kwargs):
        actions = [
            policy_fn(np.zeros(5, dtype=np.float32)),
            policy_fn(np.zeros(5, dtype=np.float32)),
        ]
        calls.append({"window": window, "actions": actions, **kwargs})
        return SimpleNamespace(total_return=0.10, sortino=2.0)

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: {"model": {"weight": torch.zeros((4, 5))}})
    monkeypatch.setattr(module, "_infer_arch", lambda state_dict: "mlp")
    monkeypatch.setattr(module, "_infer_hidden_size", lambda state_dict, arch: 4)
    monkeypatch.setattr(module, "_infer_num_actions", lambda state_dict, fallback: 4)
    monkeypatch.setattr(module, "TradingPolicy", lambda **kwargs: DummyPolicy())
    monkeypatch.setattr(module, "_slice_window", lambda data, start, steps: f"window-{start}-{steps}")
    monkeypatch.setattr(module, "simulate_daily_policy", fake_simulate_daily_policy)

    result = module.eval_checkpoint(
        Path("checkpoint.pt"),
        SimpleNamespace(num_timesteps=3),
        n_windows=2,
        eval_steps=1,
        seed=0,
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


def test_main_forwards_production_realism_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    checkpoint_root = tmp_path / "ckpts"
    trial = checkpoint_root / "seed_a"
    trial.mkdir(parents=True)
    (trial / "best.pt").write_bytes(b"checkpoint")
    data = SimpleNamespace(num_timesteps=100, num_symbols=12)
    captured: dict[str, object] = {}

    monkeypatch.setattr(module, "read_mktd", lambda path: data)

    def fake_eval_checkpoint(ckpt_path, passed_data, **kwargs):
        captured.update(kwargs)
        captured["ckpt_path"] = ckpt_path
        captured["data"] = passed_data
        return {
            "checkpoint": str(ckpt_path),
            "med": 30.0,
            "p10": 20.0,
            "p90": 40.0,
            "worst": 10.0,
            "best": 50.0,
            "neg": 0,
            "n": 50,
            "med_sortino": 3.0,
        }

    monkeypatch.setattr(module, "eval_checkpoint", fake_eval_checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_stocks12_seeds.py",
            "--checkpoint-root",
            str(checkpoint_root),
            "--data-path",
            str(tmp_path / "val.bin"),
        ],
    )

    assert module.main() == 0
    assert captured["ckpt_path"] == trial / "best.pt"
    assert captured["data"] is data
    assert captured["fee"] == 0.001
    assert captured["fill_bps"] == 5.0
    assert captured["slippage_bps"] == 20.0
    assert captured["short_borrow_apr"] == 0.0625
    assert captured["decision_lag"] == 2
    assert captured["max_hold_bars"] == 6
    assert captured["max_leverage"] == 1.0


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
            "eval_stocks12_seeds.py",
            "--checkpoint-root",
            str(tmp_path / "ckpts"),
            "--data-path",
            str(tmp_path / "val.bin"),
            flag,
            value,
        ],
    )

    assert module.main() == 2
