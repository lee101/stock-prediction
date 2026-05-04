from __future__ import annotations

from pathlib import Path

import scripts.train_portfolio_bracket_real as module
import torch
from pufferlib_market.realism import PRODUCTION_FEE_BPS, PRODUCTION_FILL_BUFFER_BPS


def _data_payload(*, symbols: int = 2, timesteps: int = 8) -> dict:
    return {
        "T": timesteps,
        "S": symbols,
        "F": 4,
        "symbols": [f"S{i}" for i in range(symbols)],
        "prices": torch.ones(timesteps, symbols, 5),
        "tradable": torch.ones(timesteps, symbols, dtype=torch.bool),
        "features": torch.ones(timesteps, symbols, 4),
    }


def test_script_adds_gpu_trading_env_python_path() -> None:
    assert str(module.GPU_ENV_PYTHON) in module.sys.path


def test_invalid_args_fail_before_cuda_or_data_load(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "train.bin"
    data_path.write_bytes(b"stub")
    load_calls = 0

    def fail_load(_path):
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("data should not load")

    monkeypatch.setattr(module.gpu_trading_env, "load_bin_full", fail_load)
    monkeypatch.setattr(
        module.torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA should not be checked")),
    )

    rc = module.main([
        "--bin",
        str(data_path),
        "--lr",
        "nan",
    ])

    assert rc == 2
    assert load_calls == 0


def test_main_writes_checkpoints_and_reports_atomically(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    data_path.write_bytes(b"stub")
    val_path.write_bytes(b"stub")
    out_dir = tmp_path / "out"
    torch_writes: list[Path] = []
    json_writes: list[tuple[Path, dict]] = []
    env_params: list[dict] = []

    class FakeEnv:
        def __init__(self, *, batch_size: int, symbols: int = 2, timesteps: int = 8) -> None:
            self.B = batch_size
            self.S = symbols
            self.T = timesteps
            self.prices = torch.ones(timesteps, symbols, 5)

        def obs(self) -> torch.Tensor:
            return torch.ones(self.B, 3)

    class FakePolicy(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def to(self, _device: str):
            return self

    class FakeBuffer:
        rewards = torch.tensor([0.25])

    def fake_make_portfolio_bracket(*, B, prices, params, **_kwargs):
        env_params.append(dict(params))
        return FakeEnv(batch_size=B, timesteps=int(prices.size(0)))

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module.gpu_trading_env, "load_bin_full", lambda _path: _data_payload())
    monkeypatch.setattr(module.gpu_trading_env, "make_portfolio_bracket", fake_make_portfolio_bracket)
    monkeypatch.setattr(module, "PortfolioBracketActor", FakePolicy)
    monkeypatch.setattr(module, "collect_rollout", lambda *_args: (FakeBuffer(), torch.tensor([0.0])))
    monkeypatch.setattr(module, "compute_gae", lambda *_args: (torch.tensor([0.0]), torch.tensor([0.0])))
    monkeypatch.setattr(
        module,
        "ppo_update",
        lambda *_args: {"pg_loss": 0.1, "vf_loss": 0.2, "entropy": 0.3, "approx_kl": 0.01},
    )
    monkeypatch.setattr(
        module,
        "deterministic_eval",
        lambda *_args, **_kwargs: {
            "mean_return": 0.2,
            "median_return": 0.15,
            "p10_return": -0.01,
            "p25_return": 0.05,
            "p75_return": 0.25,
            "p90_return": 0.30,
            "frac_negative": 0.1,
            "final_eq_mean": 1.2,
            "num_steps": 2,
            "num_envs": 2,
        },
    )
    monkeypatch.setattr(module, "save_torch_atomic", lambda payload, path: torch_writes.append(Path(path)))
    monkeypatch.setattr(
        module,
        "write_json_atomic",
        lambda path, payload: json_writes.append((Path(path), payload)),
    )

    rc = module.main([
        "--bin",
        str(data_path),
        "--val-bin",
        str(val_path),
        "--B",
        "2",
        "--val-B",
        "2",
        "--rollout-steps",
        "1",
        "--iters",
        "1",
        "--epochs",
        "1",
        "--val-every",
        "1",
        "--val-steps",
        "2",
        "--out-dir",
        str(out_dir),
    ])

    assert rc == 0
    assert torch_writes == [out_dir / "policy_best.pt", out_dir / "policy.pt"]
    assert [path for path, _payload in json_writes] == [
        out_dir / "history.json",
        out_dir / "eval.json",
        out_dir / "eval_best.json",
    ]
    assert json_writes[0][1]["iter"] == [0]
    assert [params["fee_bps"] for params in env_params] == [PRODUCTION_FEE_BPS, PRODUCTION_FEE_BPS]
    assert [params["fill_buffer_bps"] for params in env_params] == [
        PRODUCTION_FILL_BUFFER_BPS,
        PRODUCTION_FILL_BUFFER_BPS,
    ]


def test_validation_allows_zero_cost_diagnostics_and_ignores_unused_val_frac(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    data_path.write_bytes(b"stub")
    val_path.write_bytes(b"stub")
    load_calls = 0

    def fail_load(_path):
        nonlocal load_calls
        load_calls += 1
        raise RuntimeError("stop after validation")

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.gpu_trading_env, "load_bin_full", fail_load)

    try:
        module.main([
            "--bin",
            str(data_path),
            "--val-bin",
            str(val_path),
            "--val-frac",
            "99.0",
            "--fee-bps",
            "0",
            "--fb-bps",
            "0",
            "--ent-coef",
            "0",
        ])
    except RuntimeError as exc:
        assert str(exc) == "stop after validation"
    else:
        raise AssertionError("expected patched data loader to stop after validation")

    assert load_calls == 1


def test_script_uses_shared_atomic_artifact_writers() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "save_torch_atomic(" in source
    assert "write_json_atomic(" in source
    assert "torch.save(" not in source
    assert ".write_text(json.dumps" not in source
