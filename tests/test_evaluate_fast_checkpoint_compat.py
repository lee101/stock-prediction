from __future__ import annotations

import json
import struct
import sys
from types import SimpleNamespace

import pytest
import torch
from pufferlib_market.evaluate_fast import (
    LoadedPolicy,
    TradingPolicy,
    WindowResult,
    _build_cli_summary_payload,
    _load_policy_for_eval,
    _summarize_window_results,
    fast_holdout_eval,
    main,
)
from torch import nn


@pytest.mark.unit
def test_load_policy_for_eval_ignores_encoder_norm_keys():
    obs_size = 21
    num_symbols = 1
    hidden = 8
    policy = TradingPolicy(obs_size, 3, hidden=hidden)
    state_dict = policy.state_dict()
    state_dict["encoder_norm.weight"] = torch.ones(hidden)
    state_dict["encoder_norm.bias"] = torch.zeros(hidden)

    loaded, loaded_state = _load_policy_for_eval(
        payload={"model": state_dict},
        obs_size=obs_size,
        num_symbols=num_symbols,
        arch="mlp",
        hidden_size=hidden,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded.policy, TradingPolicy)
    assert loaded.arch == "mlp"
    assert loaded.hidden_size == hidden
    assert loaded.action_allocation_bins == 1
    assert loaded.action_level_bins == 1
    assert loaded.action_max_offset_bps == 0.0
    assert "encoder_norm.weight" in loaded_state


@pytest.mark.unit
def test_load_policy_for_eval_rejects_wrapped_checkpoint_with_invalid_model_payload():
    with pytest.raises(KeyError, match="missing a valid 'model' state_dict"):
        _load_policy_for_eval(
            payload={"model": "bad-payload"},
            obs_size=21,
            num_symbols=1,
            arch="mlp",
            hidden_size=8,
            device=torch.device("cpu"),
        )


@pytest.mark.unit
def test_load_policy_for_eval_ignores_invalid_action_grid_metadata():
    obs_size = 21
    num_symbols = 1
    hidden = 8
    policy = TradingPolicy(obs_size, 3, hidden=hidden)
    state_dict = policy.state_dict()

    loaded, _ = _load_policy_for_eval(
        payload={
            "model": state_dict,
            "action_allocation_bins": "2.5",
            "action_level_bins": "nan",
        },
        obs_size=obs_size,
        num_symbols=num_symbols,
        arch="mlp",
        hidden_size=hidden,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded.policy, TradingPolicy)
    assert loaded.action_allocation_bins == 1
    assert loaded.action_level_bins == 1
    assert loaded.action_max_offset_bps == 0.0


@pytest.mark.unit
def test_summarize_window_results_reports_best_worst_and_positive_rate():
    summary = _summarize_window_results(
        [
            WindowResult(
                start_idx=12,
                total_return=0.15,
                sortino=1.4,
                max_drawdown=0.05,
                num_trades=4,
                win_rate=0.75,
                avg_hold_hours=10.0,
            ),
            WindowResult(
                start_idx=24,
                total_return=-0.10,
                sortino=-0.5,
                max_drawdown=0.20,
                num_trades=2,
                win_rate=0.0,
                avg_hold_hours=6.0,
            ),
            WindowResult(
                start_idx=36,
                total_return=0.0,
                sortino=0.1,
                max_drawdown=0.08,
                num_trades=1,
                win_rate=0.0,
                avg_hold_hours=3.0,
            ),
        ]
    )

    assert summary["median_total_return"] == pytest.approx(0.0)
    assert summary["positive_window_count"] == 1
    assert summary["positive_window_ratio"] == pytest.approx(1 / 3)
    assert summary["best_window"]["start_idx"] == 12
    assert summary["worst_window"]["start_idx"] == 24


@pytest.mark.unit
def test_fast_holdout_eval_exposes_resolved_checkpoint_config(tmp_path, monkeypatch):
    data_path = tmp_path / "market.bin"
    header = struct.pack("<4sIIIII", b"PMKT", 1, 1, 32, 16, 0)
    data_path.write_bytes(header + bytes(64 - len(header)))

    class FakePolicy(nn.Module):
        def __init__(self, num_actions: int):
            super().__init__()
            self.num_actions = num_actions

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = x.shape[0]
            logits = torch.zeros((batch, self.num_actions), dtype=torch.float32, device=x.device)
            values = torch.zeros((batch,), dtype=torch.float32, device=x.device)
            return logits, values

    env_rows = [
        {
            "total_return": 0.25,
            "sortino": 1.8,
            "max_drawdown": 0.04,
            "num_trades": 5,
            "win_rate": 0.8,
            "avg_hold_hours": 12.0,
        },
        {
            "total_return": -0.05,
            "sortino": -0.3,
            "max_drawdown": 0.12,
            "num_trades": 2,
            "win_rate": 0.0,
            "avg_hold_hours": 4.0,
        },
    ]

    class FakeBinding:
        def shared(self, *, data_path: str) -> None:
            self.data_path = data_path

        def vec_init(
            self,
            obs_bufs,
            act_bufs,
            rew_bufs,
            term_bufs,
            trunc_bufs,
            n_windows,
            seed,
            **kwargs,
        ):
            self.term_bufs = term_bufs
            self.trunc_bufs = trunc_bufs
            self.closed = False
            return "vec"

        def vec_set_offsets(self, handle, starts) -> None:
            self.starts = starts.copy()

        def vec_reset(self, handle, seed) -> None:
            self.term_bufs.fill(0)
            self.trunc_bufs.fill(0)

        def vec_step(self, handle) -> None:
            self.term_bufs[:] = 1
            self.trunc_bufs[:] = 0

        def vec_env_at(self, handle, idx):
            return idx

        def env_get(self, idx):
            return env_rows[idx]

        def vec_close(self, handle) -> None:
            self.closed = True

    fake_binding = FakeBinding()
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    import pufferlib_market  # noqa: PLC0415

    monkeypatch.setattr(pufferlib_market, "binding", fake_binding, raising=False)
    monkeypatch.setattr(
        "pufferlib_market.evaluate_fast.load_checkpoint_payload",
        lambda checkpoint_path, map_location=None: {"dummy": True},
    )
    monkeypatch.setattr(
        "pufferlib_market.evaluate_fast._load_policy_for_eval",
        lambda **kwargs: (
            LoadedPolicy(
                policy=FakePolicy(num_actions=13),
                arch="resmlp",
                hidden_size=32,
                action_allocation_bins=2,
                action_level_bins=3,
                action_max_offset_bps=12.5,
            ),
            {},
        ),
    )
    monkeypatch.setattr(
        "pufferlib_market.evaluate_fast.np.arange",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sampled fast path should not materialize candidate starts")),
    )

    result = fast_holdout_eval(
        "fake.pt",
        str(data_path),
        n_windows=2,
        eval_hours=4,
        device_str="cpu",
        use_compile=False,
    )

    assert result["checkpoint"] == "fake.pt"
    assert result["arch"] == "resmlp"
    assert result["hidden_size"] == 32
    assert result["action_allocation_bins"] == 2
    assert result["action_level_bins"] == 3
    assert result["action_max_offset_bps"] == pytest.approx(12.5)
    assert result["summary"]["best_window"]["total_return"] == pytest.approx(0.25)
    assert result["summary"]["worst_window"]["total_return"] == pytest.approx(-0.05)
    assert result["summary"]["positive_window_count"] == 1
    assert result["summary"]["positive_window_ratio"] == pytest.approx(0.5)
    assert len(result["windows"]) == 2
    assert fake_binding.closed is True


@pytest.mark.unit
def test_fast_holdout_eval_closes_vec_handle_on_runtime_failure(tmp_path, monkeypatch):
    data_path = tmp_path / "market.bin"
    header = struct.pack("<4sIIIII", b"PMKT", 1, 1, 32, 16, 0)
    data_path.write_bytes(header + bytes(64 - len(header)))

    class FakePolicy(nn.Module):
        def __init__(self, num_actions: int):
            super().__init__()
            self.num_actions = num_actions

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = x.shape[0]
            logits = torch.zeros((batch, self.num_actions), dtype=torch.float32, device=x.device)
            values = torch.zeros((batch,), dtype=torch.float32, device=x.device)
            return logits, values

    class FakeBinding:
        def shared(self, *, data_path: str) -> None:
            self.data_path = data_path

        def vec_init(
            self,
            obs_bufs,
            act_bufs,
            rew_bufs,
            term_bufs,
            trunc_bufs,
            n_windows,
            seed,
            **kwargs,
        ):
            self.closed = False
            return "vec"

        def vec_set_offsets(self, handle, starts) -> None:
            self.starts = starts.copy()

        def vec_reset(self, handle, seed) -> None:
            return None

        def vec_step(self, handle) -> None:
            raise RuntimeError("vec boom")

        def vec_close(self, handle) -> None:
            self.closed = True

    fake_binding = FakeBinding()
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    import pufferlib_market  # noqa: PLC0415

    monkeypatch.setattr(pufferlib_market, "binding", fake_binding, raising=False)
    monkeypatch.setattr(
        "pufferlib_market.evaluate_fast.load_checkpoint_payload",
        lambda checkpoint_path, map_location=None: {"dummy": True},
    )
    monkeypatch.setattr(
        "pufferlib_market.evaluate_fast._load_policy_for_eval",
        lambda **kwargs: (
            LoadedPolicy(
                policy=FakePolicy(num_actions=3),
                arch="mlp",
                hidden_size=16,
                action_allocation_bins=1,
                action_level_bins=1,
                action_max_offset_bps=0.0,
            ),
            {},
        ),
    )

    with pytest.raises(RuntimeError, match="vec boom"):
        fast_holdout_eval(
            "fake.pt",
            str(data_path),
            n_windows=2,
            eval_hours=4,
            device_str="cpu",
            use_compile=False,
        )

    assert fake_binding.closed is True


@pytest.mark.unit
def test_build_cli_summary_payload_merges_context_with_summary():
    payload = _build_cli_summary_payload(
        {
            "checkpoint": "run.pt",
            "data_path": "val.bin",
            "arch": "resmlp",
            "hidden_size": 32,
            "action_allocation_bins": 2,
            "action_level_bins": 3,
            "action_max_offset_bps": 12.5,
            "eval_hours": 24,
            "n_windows": 5,
            "n_completed": 4,
            "early_exit": True,
            "summary": {
                "median_total_return": 0.1,
                "best_window": {"start_idx": 12},
            },
        }
    )

    assert payload == {
        "checkpoint": "run.pt",
        "data_path": "val.bin",
        "arch": "resmlp",
        "hidden_size": 32,
        "action_allocation_bins": 2,
        "action_level_bins": 3,
        "action_max_offset_bps": 12.5,
        "eval_hours": 24,
        "n_windows": 5,
        "n_completed": 4,
        "early_exit": True,
        "median_total_return": 0.1,
        "best_window": {"start_idx": 12},
    }


@pytest.mark.unit
def test_main_prints_self_describing_single_run_summary(tmp_path, monkeypatch, capsys):
    out_path = tmp_path / "fast-summary.json"
    fake_args = SimpleNamespace(
        checkpoint="fake.pt",
        data_path="val.bin",
        eval_hours=24,
        n_windows=5,
        seed=1337,
        fee_rate=0.001,
        fill_slippage_bps=8.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
        short_borrow_apr=0.0,
        deterministic=True,
        disable_shorts=False,
        arch="auto",
        hidden_size=None,
        device="cpu",
        no_compile=True,
        early_exit_after=5,
        early_exit_threshold=-0.15,
        verbose=False,
        multi_windows=None,
        n_windows_per_size=8,
        out=str(out_path),
    )
    fake_result = {
        "checkpoint": "fake.pt",
        "data_path": "val.bin",
        "arch": "resmlp",
        "hidden_size": 32,
        "action_allocation_bins": 2,
        "action_level_bins": 3,
        "action_max_offset_bps": 12.5,
        "eval_hours": 24,
        "n_windows": 5,
        "n_completed": 4,
        "early_exit": True,
        "elapsed_s": 1.25,
        "summary": {
            "median_total_return": 0.1,
            "positive_window_count": 3,
            "best_window": {"start_idx": 12},
        },
    }

    monkeypatch.setattr("pufferlib_market.evaluate_fast.argparse.ArgumentParser.parse_args", lambda self: fake_args)
    monkeypatch.setattr("pufferlib_market.evaluate_fast.fast_holdout_eval", lambda *args, **kwargs: fake_result)

    main()

    captured = capsys.readouterr()
    summary_text, elapsed_line = captured.out.strip().split("\n\n")
    summary = json.loads(summary_text)
    assert summary["checkpoint"] == "fake.pt"
    assert summary["data_path"] == "val.bin"
    assert summary["arch"] == "resmlp"
    assert summary["hidden_size"] == 32
    assert summary["action_allocation_bins"] == 2
    assert summary["action_level_bins"] == 3
    assert summary["action_max_offset_bps"] == pytest.approx(12.5)
    assert summary["eval_hours"] == 24
    assert summary["n_windows"] == 5
    assert summary["n_completed"] == 4
    assert summary["early_exit"] is True
    assert summary["median_total_return"] == pytest.approx(0.1)
    assert summary["positive_window_count"] == 3
    assert elapsed_line == "Elapsed: 1.25s (4/5 windows)"
    assert json.loads(out_path.read_text()) == fake_result
