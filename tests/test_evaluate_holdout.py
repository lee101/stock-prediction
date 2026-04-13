from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pufferlib_market import evaluate_holdout as eval_mod
from pufferlib_market.hourly_replay import MktdData


def _fake_main_args(tmp_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "checkpoint": str(tmp_path / "checkpoint.pt"),
        "data_path": str(tmp_path / "data.mktd"),
        "eval_hours": 24,
        "n_windows": 3,
        "exhaustive": False,
        "seed": 1337,
        "end_within_hours": None,
        "fee_rate": 0.001,
        "slippage_bps": 0.0,
        "fill_buffer_bps": 5.0,
        "max_leverage": 1.0,
        "periods_per_year": 8760.0,
        "short_borrow_apr": 0.0,
        "disable_shorts": False,
        "shortable_symbols": None,
        "tradable_symbols": None,
        "decision_lag": 0,
        "deterministic": True,
        "no_early_stop": False,
        "device": "cpu",
        "out": None,
        "extra_checkpoints": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_data(num_timesteps: int, num_symbols: int = 1) -> MktdData:
    features = np.zeros((num_timesteps, num_symbols, 16), dtype=np.float32)
    prices = np.ones((num_timesteps, num_symbols, 5), dtype=np.float32)
    tradable = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
    return MktdData(
        version=2,
        symbols=[f"SYM{i}" for i in range(num_symbols)],
        features=features,
        prices=prices,
        tradable=tradable,
    )


def test_infer_action_grid_ignores_invalid_metadata() -> None:
    num_symbols = 1
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols
    source_policy = eval_mod.TradingPolicy(obs_size, num_actions, hidden=16)

    num_actions_out, alloc_bins, level_bins, max_offset_bps = eval_mod._infer_action_grid(
        payload={
            "action_allocation_bins": "oops",
            "action_level_bins": float("nan"),
            "action_max_offset_bps": "not-a-number",
        },
        state_dict=source_policy.state_dict(),
        num_symbols=num_symbols,
    )

    assert (num_actions_out, alloc_bins, level_bins, max_offset_bps) == (num_actions, 1, 1, 0.0)


def test_infer_action_grid_rejects_non_integral_level_bins_and_recovers_from_shape() -> None:
    num_symbols = 1
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 5
    source_policy = eval_mod.TradingPolicy(obs_size, num_actions, hidden=16)

    num_actions_out, alloc_bins, level_bins, max_offset_bps = eval_mod._infer_action_grid(
        payload={
            "action_allocation_bins": 1,
            "action_level_bins": 2.5,
            "action_max_offset_bps": 12.5,
        },
        state_dict=source_policy.state_dict(),
        num_symbols=num_symbols,
    )

    assert (num_actions_out, alloc_bins, level_bins, max_offset_bps) == (num_actions, 2, 1, 12.5)


def test_load_policy_rejects_metadata_mapping_without_model(tmp_path: Path) -> None:
    with (
        patch.object(eval_mod, "load_checkpoint_payload", return_value={"arch": "mlp", "hidden_size": 16}),
        pytest.raises(ValueError, match=r"Unsupported checkpoint format \(expected state_dict or dict with 'model'\)"),
    ):
        eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols=1,
            features_per_sym=16,
            device=torch.device("cpu"),
        )


def test_load_policy_rejects_wrapped_checkpoint_with_invalid_model_payload(tmp_path: Path) -> None:
    with (
        patch.object(eval_mod, "load_checkpoint_payload", return_value={"model": {"arch": "mlp"}}),
        pytest.raises(KeyError, match="Checkpoint is missing a valid 'model' state_dict"),
    ):
        eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols=1,
            features_per_sym=16,
            device=torch.device("cpu"),
        )


def test_load_policy_unknown_arch_falls_back_to_mlp(tmp_path: Path) -> None:
    # evaluate_holdout.py no longer rejects unknown archs — they fall through to
    # the TradingPolicy (mlp) branch for forward compatibility with new arch names.
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)

    with patch.object(
        eval_mod,
        "load_checkpoint_payload",
        return_value={"model": source_policy.state_dict(), "arch": "bogus"},
    ):
        loaded = eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols=1,
            features_per_sym=16,
            device=torch.device("cpu"),
        )
    assert loaded.policy is not None


def test_load_policy_supports_bare_state_dict_payload(tmp_path: Path) -> None:
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)

    with patch.object(eval_mod, "load_checkpoint_payload", return_value=source_policy.state_dict()):
        loaded = eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols=1,
            features_per_sym=16,
            device=torch.device("cpu"),
        )

    assert isinstance(loaded.policy, eval_mod.TradingPolicy)
    assert loaded.arch == "mlp"
    assert loaded.hidden_size == 16
    assert loaded.action_allocation_bins == 1
    assert loaded.action_level_bins == 1
    assert loaded.action_max_offset_bps == pytest.approx(0.0)


def test_load_policy_prefers_checkpoint_encoder_norm_flag(tmp_path: Path) -> None:
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)
    payload = {"model": source_policy.state_dict(), "use_encoder_norm": False}

    with (
        patch.object(eval_mod, "load_checkpoint_payload", return_value=payload),
        patch.object(eval_mod.TradingPolicy, "load_state_dict", return_value=([], [])),
    ):
        loaded = eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols=1,
            features_per_sym=16,
            device=torch.device("cpu"),
        )

    assert loaded.policy._use_encoder_norm is False


def test_mask_disallowed_symbols_masks_long_and_short_blocks() -> None:
    logits = torch.zeros((1, 9), dtype=torch.float32)

    masked = eval_mod._mask_disallowed_symbols(
        logits,
        num_symbols=2,
        per_symbol_actions=2,
        tradable_mask=torch.tensor([True, False], dtype=torch.bool),
    )

    min_val = torch.finfo(masked.dtype).min
    assert torch.isfinite(masked[0, 0])
    assert torch.isfinite(masked[0, 1])
    assert torch.isfinite(masked[0, 2])
    assert masked[0, 3] == min_val
    assert masked[0, 4] == min_val
    assert torch.isfinite(masked[0, 5])
    assert torch.isfinite(masked[0, 6])
    assert masked[0, 7] == min_val
    assert masked[0, 8] == min_val


def test_build_tradable_mask_rejects_unknown_symbols() -> None:
    with pytest.raises(ValueError, match="unknown symbols: SYM9"):
        eval_mod._build_tradable_mask(["SYM0", "SYM1"], "SYM0,SYM9")



def test_load_policy_raises_on_non_ignored_state_mismatch(tmp_path: Path) -> None:
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)

    with (
        patch.object(eval_mod, "load_checkpoint_payload", return_value=source_policy.state_dict()),
        patch.object(eval_mod.TradingPolicy, "load_state_dict", return_value=(["actor.weight"], [])),
        pytest.raises(RuntimeError, match="Checkpoint architecture mismatch"),
    ):
        eval_mod.load_policy(
            str(tmp_path / "checkpoint.pt"),
            num_symbols=1,
            features_per_sym=16,
            device=torch.device("cpu"),
        )


def test_main_supports_bare_state_dict_payload(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_args = _fake_main_args(tmp_path, out=str(tmp_path / "summary.json"))
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)
    fake_results = [
        SimpleNamespace(total_return=0.25, sortino=1.5, max_drawdown=0.1, num_trades=4, win_rate=0.75),
        SimpleNamespace(total_return=-0.10, sortino=-0.5, max_drawdown=0.2, num_trades=3, win_rate=0.25),
        SimpleNamespace(total_return=0.05, sortino=0.4, max_drawdown=0.08, num_trades=2, win_rate=0.50),
    ]

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(eval_mod, "load_checkpoint_payload", return_value=source_policy.state_dict()),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(40)),
        patch.object(eval_mod, "simulate_daily_policy", side_effect=fake_results) as simulate_mock,
    ):
        eval_mod.main()

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    out = json.loads(Path(fake_args.out).read_text())

    assert summary["checkpoint"] == str(Path(fake_args.checkpoint))
    assert summary["data_path"] == str(Path(fake_args.data_path))
    assert summary["arch"] == "mlp"
    assert summary["hidden_size"] == 16
    assert summary["action_allocation_bins"] == 1
    assert summary["action_level_bins"] == 1
    assert summary["action_max_offset_bps"] == pytest.approx(0.0)
    assert summary["eval_hours"] == fake_args.eval_hours
    assert summary["window_mode"] == "sampled"
    assert summary["candidate_window_count"] == 16
    assert summary["sampled_window_count"] == fake_args.n_windows
    assert summary["sampled_with_replacement"] is False
    assert summary["median_total_return"] == pytest.approx(0.05)
    assert summary["best_window"]["total_return"] == pytest.approx(0.25)
    assert summary["worst_window"]["total_return"] == pytest.approx(-0.10)
    assert out["window_selection"]["mode"] == "sampled"
    assert out["window_selection"]["candidate_window_count"] == 16
    assert out["window_selection"]["sampled_window_count"] == fake_args.n_windows
    assert out["window_selection"]["sampled_with_replacement"] is False
    assert out["window_selection"]["candidate_start_range"] == [0, 15]
    assert out["window_selection"]["sampled_start_indices"] == [window["start_idx"] for window in out["windows"]]
    assert out["summary"]["best_window"]["total_return"] == pytest.approx(0.25)
    assert out["summary"]["worst_window"]["total_return"] == pytest.approx(-0.10)
    assert out["summary"]["arch"] == "mlp"
    assert out["summary"]["action_max_offset_bps"] == pytest.approx(0.0)
    assert out["arch"] == "mlp"
    assert out["hidden_size"] == 16
    assert out["action_allocation_bins"] == 1
    assert out["action_level_bins"] == 1
    assert out["action_max_offset_bps"] == pytest.approx(0.0)
    assert len(out["windows"]) == fake_args.n_windows
    assert simulate_mock.call_count == fake_args.n_windows
    assert simulate_mock.call_args.kwargs["slippage_bps"] == pytest.approx(fake_args.slippage_bps)


def test_main_ignores_invalid_checkpoint_action_grid_metadata(tmp_path: Path) -> None:
    fake_args = _fake_main_args(tmp_path, out=str(tmp_path / "summary.json"))
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)
    fake_result = SimpleNamespace(
        total_return=0.1,
        sortino=1.0,
        max_drawdown=0.05,
        num_trades=2,
        win_rate=0.5,
    )

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(
            eval_mod,
            "load_checkpoint_payload",
            return_value={
                "model": source_policy.state_dict(),
                "action_allocation_bins": "oops",
                "action_level_bins": 2.5,
                "action_max_offset_bps": "bad",
            },
        ),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(40)),
        patch.object(eval_mod, "simulate_daily_policy", return_value=fake_result) as simulate_mock,
    ):
        eval_mod.main()

    out = json.loads(Path(fake_args.out).read_text())
    assert out["action_allocation_bins"] == 1
    assert out["action_level_bins"] == 1
    assert out["action_max_offset_bps"] == pytest.approx(0.0)
    for call in simulate_mock.call_args_list:
        assert call.kwargs["action_allocation_bins"] == 1
        assert call.kwargs["action_level_bins"] == 1
        assert call.kwargs["action_max_offset_bps"] == pytest.approx(0.0)


def test_main_reports_exhaustive_window_selection_metadata(tmp_path: Path) -> None:
    fake_args = _fake_main_args(tmp_path, exhaustive=True, out=str(tmp_path / "summary.json"))
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)
    fake_result = SimpleNamespace(
        total_return=0.1,
        sortino=0.8,
        max_drawdown=0.05,
        num_trades=2,
        win_rate=0.5,
    )

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(eval_mod, "load_checkpoint_payload", return_value=source_policy.state_dict()),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(28)),
        patch.object(eval_mod, "simulate_daily_policy", return_value=fake_result) as simulate_mock,
    ):
        eval_mod.main()

    out = json.loads(Path(fake_args.out).read_text())
    assert out["window_selection"]["mode"] == "exhaustive"
    assert out["window_selection"]["candidate_window_count"] == 4
    assert out["window_selection"]["sampled_window_count"] == 4
    assert out["window_selection"]["sampled_with_replacement"] is False
    assert out["window_selection"]["candidate_start_range"] == [0, 3]
    assert out["window_selection"]["sampled_start_indices"] == [0, 1, 2, 3]
    assert simulate_mock.call_count == 4


def test_main_reports_replacement_sampling_coverage_metadata(tmp_path: Path) -> None:
    fake_args = _fake_main_args(tmp_path, n_windows=6, out=str(tmp_path / "summary.json"))
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)
    fake_result = SimpleNamespace(
        total_return=0.1,
        sortino=0.8,
        max_drawdown=0.05,
        num_trades=2,
        win_rate=0.5,
    )

    class FakeRng:
        def choice(self, values, size, replace):
            assert values == 4
            assert size == 6
            assert replace is True
            return np.asarray([0, 1, 1, 2, 3, 3], dtype=np.int64)

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(eval_mod, "load_checkpoint_payload", return_value=source_policy.state_dict()),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(28)),
        patch.object(eval_mod.np.random, "default_rng", return_value=FakeRng()),
        patch.object(eval_mod, "simulate_daily_policy", return_value=fake_result) as simulate_mock,
    ):
        eval_mod.main()

    out = json.loads(Path(fake_args.out).read_text())
    summary = out["summary"]
    selection = out["window_selection"]

    assert summary["window_mode"] == "sampled"
    assert summary["candidate_window_count"] == 4
    assert summary["sampled_window_count"] == 6
    assert summary["unique_sampled_window_count"] == 4
    assert summary["duplicate_sample_count"] == 2
    assert summary["sampled_with_replacement"] is True
    assert selection["sampled_start_indices"] == [0, 1, 1, 2, 3, 3]
    assert selection["unique_sampled_window_count"] == 4
    assert selection["duplicate_sample_count"] == 2
    assert selection["sampled_with_replacement"] is True
    assert simulate_mock.call_count == 6


def test_main_sampled_mode_does_not_materialize_candidate_array(tmp_path: Path) -> None:
    fake_args = _fake_main_args(tmp_path, out=str(tmp_path / "summary.json"))
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)
    fake_result = SimpleNamespace(
        total_return=0.1,
        sortino=0.8,
        max_drawdown=0.05,
        num_trades=2,
        win_rate=0.5,
    )

    class FakeRng:
        def choice(self, values, size, replace):
            assert values == 16
            assert size == fake_args.n_windows
            assert replace is False
            return np.asarray([0, 2, 4], dtype=np.int64)

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(eval_mod, "load_checkpoint_payload", return_value=source_policy.state_dict()),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(40)),
        patch.object(eval_mod.np.random, "default_rng", return_value=FakeRng()),
        patch.object(eval_mod.np, "arange", side_effect=AssertionError("sampled mode should not build candidate arrays")),
        patch.object(eval_mod, "simulate_daily_policy", return_value=fake_result) as simulate_mock,
    ):
        eval_mod.main()

    out = json.loads(Path(fake_args.out).read_text())
    assert out["window_selection"]["sampled_start_indices"] == [0, 2, 4]
    assert simulate_mock.call_count == fake_args.n_windows


def test_main_moves_shortable_mask_once(tmp_path: Path) -> None:
    fake_args = _fake_main_args(
        tmp_path,
        out=str(tmp_path / "summary.json"),
        shortable_symbols="SYM0",
    )
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=16)
    fake_result = SimpleNamespace(
        total_return=0.1,
        sortino=0.8,
        max_drawdown=0.05,
        num_trades=2,
        win_rate=0.5,
    )

    class FakeMask:
        def __init__(self) -> None:
            self.to_calls: list[object] = []

        def to(self, *, device: object) -> FakeMask:
            self.to_calls.append(device)
            return self

    fake_mask = FakeMask()

    def _simulate_once(window, policy_fn, **kwargs):
        obs = np.zeros((22,), dtype=np.float32)
        policy_fn(obs)
        return fake_result

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(eval_mod, "load_checkpoint_payload", return_value=source_policy.state_dict()),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(40)),
        patch.object(eval_mod, "_build_shortable_mask", return_value=fake_mask),
        patch.object(eval_mod, "_mask_disallowed_shorts", side_effect=lambda logits, **kwargs: logits),
        patch.object(eval_mod, "simulate_daily_policy", side_effect=_simulate_once),
    ):
        eval_mod.main()

    assert fake_mask.to_calls == [torch.device("cpu")]


def test_main_applies_tradable_symbol_mask_and_reports_subset(tmp_path: Path) -> None:
    fake_args = _fake_main_args(
        tmp_path,
        out=str(tmp_path / "summary.json"),
        tradable_symbols="SYM0",
    )
    fake_result = SimpleNamespace(
        total_return=0.1,
        sortino=0.8,
        max_drawdown=0.05,
        num_trades=2,
        win_rate=0.5,
    )
    seen_actions: list[int] = []

    class FakePolicy:
        def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.tensor([[0.0, 3.0, 10.0, -1.0, 5.0]], dtype=torch.float32)
            value = torch.tensor([0.0], dtype=torch.float32)
            return logits, value

        def eval(self) -> FakePolicy:
            return self

    loaded = eval_mod.LoadedPolicy(
        policy=FakePolicy(),
        arch="mlp",
        hidden_size=16,
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
    )

    def _simulate_once(window, policy_fn, **kwargs):
        obs = np.zeros((39,), dtype=np.float32)
        seen_actions.append(policy_fn(obs))
        return fake_result

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(eval_mod, "load_policy", return_value=loaded),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(40, num_symbols=2)),
        patch.object(eval_mod, "simulate_daily_policy", side_effect=_simulate_once),
    ):
        eval_mod.main()

    out = json.loads(Path(fake_args.out).read_text())
    assert seen_actions == [1, 1, 1]
    assert out["tradable_symbols"] == ["SYM0"]
    assert out["summary"]["tradable_symbols"] == ["SYM0"]


def test_main_uses_shared_checkpoint_loader_boundary(tmp_path: Path) -> None:
    fake_args = _fake_main_args(tmp_path)
    expected_message = f"Failed to load checkpoint {Path(fake_args.checkpoint).resolve()}: no such file"

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(40)),
        patch.object(eval_mod, "load_checkpoint_payload", side_effect=RuntimeError(expected_message)),
        pytest.raises(RuntimeError, match=expected_message),
    ):
        eval_mod.main()
