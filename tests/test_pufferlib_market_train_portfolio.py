from __future__ import annotations

from pathlib import Path

import pufferlib_market.train_portfolio as module


def test_train_config_writes_best_checkpoint_atomically(tmp_path, monkeypatch) -> None:
    class FakeObservationSpace:
        shape = (4,)

    class FakeEnv:
        observation_space = FakeObservationSpace()
        num_symbols = 2

        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class FakeTrainer:
        def __init__(self, _env, _policy, *, lr: float) -> None:
            self.lr = lr

        def collect_rollout(self, _num_steps: int) -> dict:
            return {}

        def update(self, _rollout: dict) -> dict[str, float]:
            return {"pg_loss": 0.1, "entropy": 0.2}

        def evaluate(self, *, num_episodes: int) -> dict[str, float]:
            return {"mean_return": 0.03, "mean_sortino": 1.25, "std_return": 0.0}

    writes: list[tuple[dict, Path]] = []

    monkeypatch.setattr(module, "PortfolioEnv", FakeEnv)
    monkeypatch.setattr(module, "PPOTrainer", FakeTrainer)
    monkeypatch.setattr(module, "save_torch_atomic", lambda payload, path: writes.append((payload, Path(path))))

    data_path = tmp_path / "portfolio.bin"
    data_path.write_bytes(b"stub")

    best_sortino = module.train_config(
        data_path=str(data_path),
        hidden_dim=8,
        num_layers=1,
        discrete_bins=3,
        lr=1e-3,
        total_steps=20_480,
        checkpoint_dir=str(tmp_path / "portfolio_ckpt"),
    )

    assert best_sortino == 1.25
    assert len(writes) == 1
    payload, path = writes[0]
    assert path == tmp_path / "portfolio_ckpt" / "best.pt"
    assert payload["sortino"] == 1.25
    assert payload["step"] == 20_480
    assert payload["config"] == {
        "hidden_dim": 8,
        "num_layers": 1,
        "discrete_bins": 3,
        "num_symbols": 2,
        "obs_dim": 4,
    }
    assert "policy" in payload


def test_train_config_short_run_still_evaluates_and_writes_checkpoint(tmp_path, monkeypatch) -> None:
    class FakeObservationSpace:
        shape = (4,)

    class FakeEnv:
        observation_space = FakeObservationSpace()
        num_symbols = 2

        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class FakeTrainer:
        def __init__(self, _env, _policy, *, lr: float) -> None:
            self.lr = lr

        def collect_rollout(self, num_steps: int) -> dict:
            assert num_steps == module.ROLLOUT_STEPS
            return {}

        def update(self, _rollout: dict) -> dict[str, float]:
            return {"pg_loss": 0.1, "entropy": 0.2}

        def evaluate(self, *, num_episodes: int) -> dict[str, float]:
            return {"mean_return": 0.01, "mean_sortino": 0.75, "std_return": 0.0}

    data_path = tmp_path / "portfolio.bin"
    data_path.write_bytes(b"stub")
    writes: list[tuple[dict, Path]] = []

    monkeypatch.setattr(module, "PortfolioEnv", FakeEnv)
    monkeypatch.setattr(module, "PPOTrainer", FakeTrainer)
    monkeypatch.setattr(module, "save_torch_atomic", lambda payload, path: writes.append((payload, Path(path))))

    best_sortino = module.train_config(
        data_path=str(data_path),
        hidden_dim=8,
        num_layers=1,
        discrete_bins=3,
        lr=1e-3,
        total_steps=1,
        checkpoint_dir=str(tmp_path / "portfolio_ckpt"),
    )

    assert best_sortino == 0.75
    assert len(writes) == 1
    assert writes[0][0]["step"] == module.ROLLOUT_STEPS
    assert writes[0][1] == tmp_path / "portfolio_ckpt" / "best.pt"


def test_main_rejects_invalid_config_before_env_work(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "portfolio.bin"
    data_path.write_bytes(b"stub")
    env_calls = 0

    def fail_env(*_args, **_kwargs):
        nonlocal env_calls
        env_calls += 1
        raise AssertionError("PortfolioEnv should not be constructed")

    monkeypatch.setattr(module, "PortfolioEnv", fail_env)

    rc = module.main([
        "--data-path",
        str(data_path),
        "--lr",
        "nan",
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
    ])

    assert rc == 2
    assert env_calls == 0


def test_search_ignores_single_run_only_model_knobs(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "portfolio.bin"
    data_path.write_bytes(b"stub")
    calls: list[dict] = []

    def fake_train_config(
        data_path: str,
        hidden_dim: int,
        num_layers: int,
        discrete_bins: int,
        lr: float,
        total_steps: int,
        checkpoint_dir: str,
    ) -> float:
        calls.append({
            "data_path": data_path,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "discrete_bins": discrete_bins,
            "lr": lr,
            "total_steps": total_steps,
            "checkpoint_dir": checkpoint_dir,
        })
        return float(len(calls))

    monkeypatch.setattr(module, "train_config", fake_train_config)

    rc = module.main([
        "--data-path",
        str(data_path),
        "--search",
        "--hidden-dim",
        "0",
        "--total-steps",
        "1",
        "--checkpoint-dir",
        str(tmp_path / "search"),
    ])

    assert rc == 0
    assert len(calls) == 4
    assert {call["hidden_dim"] for call in calls} == {128, 256, 512}


def test_train_portfolio_uses_atomic_checkpoint_helper() -> None:
    source = Path("pufferlib_market/train_portfolio.py").read_text(encoding="utf-8")

    assert "save_torch_atomic(" in source
    assert "torch.save(" not in source
