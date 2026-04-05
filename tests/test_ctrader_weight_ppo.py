from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import subprocess

from ctrader.market_sim_ffi import NativeWeightEnvConfig, NativeWeightEnvHandle, load_library
from ctrader.train_weight_ppo import (
    ContinuousWeightEnv,
    PPOTrainConfig,
    load_close_matrix,
    run_training,
)


REPO = Path("/home/lee/code/stock")
CTRADER_DIR = REPO / "ctrader"


def _write_close_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "close"])
        writer.writeheader()
        for ts, close in rows:
            writer.writerow({"timestamp": ts, "close": close})


def test_load_close_matrix_inner_joins_timestamps(tmp_path: Path) -> None:
    _write_close_csv(
        tmp_path / "AAAUSDT.csv",
        [
            ("2026-01-01T00:00:00Z", 100.0),
            ("2026-01-01T01:00:00Z", 101.0),
            ("2026-01-01T02:00:00Z", 102.0),
        ],
    )
    _write_close_csv(
        tmp_path / "BBBUSDT.csv",
        [
            ("2026-01-01T01:00:00Z", 50.0),
            ("2026-01-01T02:00:00Z", 51.0),
            ("2026-01-01T03:00:00Z", 52.0),
        ],
    )

    close = load_close_matrix(["AAAUSDT", "BBBUSDT"], data_root=tmp_path, min_rows=2)

    assert list(close.columns) == ["AAAUSDT", "BBBUSDT"]
    assert len(close) == 2
    assert close.iloc[0].tolist() == [101.0, 50.0]
    assert close.iloc[1].tolist() == [102.0, 51.0]


def test_weight_env_long_only_clamps_and_produces_finite_terminal_metrics() -> None:
    close = np.array(
        [
            [100.0, 100.0],
            [110.0, 100.0],
            [121.0, 100.0],
            [133.1, 100.0],
            [146.41, 100.0],
            [161.051, 100.0],
        ],
        dtype=np.float64,
    )
    config = PPOTrainConfig(
        lookback=3,
        episode_steps=2,
        rollout_steps=8,
        total_updates=1,
        max_gross_leverage=1.0,
        seed=7,
        device="cpu",
    )
    env = ContinuousWeightEnv(close, config)
    obs = env.reset(start_index=3)

    assert obs.shape == (env.obs_dim,)

    next_obs, reward, done, info = env.step(np.array([6.0, -6.0], dtype=np.float32))
    assert next_obs.shape == (env.obs_dim,)
    assert reward > 0.0
    assert not done
    assert float(np.sum(env.weights)) <= 1.000001
    assert float(np.min(env.weights)) >= 0.0

    _, reward, done, info = env.step(np.array([6.0, -6.0], dtype=np.float32))
    assert reward > 0.0
    assert done
    assert info["total_return"] > 0.0
    assert np.isfinite(info["annualized_return"])
    assert np.isfinite(info["sortino"])
    assert np.isfinite(info["max_drawdown"])
    assert info["max_drawdown"] >= 0.0


def test_run_training_smoke_returns_finite_annualized_metrics() -> None:
    rng = np.random.default_rng(123)
    rows = 320
    base = np.linspace(100.0, 130.0, rows)
    alt = 100.0 + np.cumsum(rng.normal(0.02, 0.15, size=rows))
    hedge = np.linspace(100.0, 96.0, rows)
    close = np.column_stack([base, alt, hedge])
    frame = pd.DataFrame(close, columns=["AAA", "BBB", "CCC"])

    result = run_training(
        frame,
        PPOTrainConfig(
            lookback=16,
            episode_steps=48,
            rollout_steps=128,
            total_updates=2,
            hidden_dim=64,
            batch_size=64,
            ppo_epochs=2,
            eval_every_updates=1,
            seed=5,
            device="cpu",
        ),
        train_fraction=0.7,
    )

    best = result["best_eval"]
    history = result["history"]

    assert len(history) == 2
    assert np.isfinite(best["annualized_return"])
    assert np.isfinite(best["sortino"])
    assert np.isfinite(best["max_drawdown"])
    assert best["max_drawdown"] >= 0.0
    assert best["final_equity"] > 0.0
    assert all(np.isfinite(row["grad_norm_mean"]) for row in history)
    assert json.loads(json.dumps(result))["best_eval"]["final_equity"] == best["final_equity"]


def test_native_weight_env_matches_python_env_one_step() -> None:
    subprocess.run(
        ["make", "libmarket_sim.so"],
        cwd=CTRADER_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    lib = load_library(CTRADER_DIR / "libmarket_sim.so")

    close = np.array(
        [
            [100.0, 100.0],
            [110.0, 100.0],
            [121.0, 100.0],
            [133.1, 100.0],
            [146.41, 100.0],
            [161.051, 100.0],
        ],
        dtype=np.float64,
    )
    config = PPOTrainConfig(
        lookback=3,
        episode_steps=2,
        rollout_steps=8,
        total_updates=1,
        max_gross_leverage=1.0,
        seed=7,
        device="cpu",
    )
    py_env = ContinuousWeightEnv(close, config)
    py_env.reset(start_index=3)
    py_obs = py_env._get_obs().astype(np.float64)

    native_cfg = NativeWeightEnvConfig(
        lookback=config.lookback,
        episode_steps=config.episode_steps,
        reward_scale=config.reward_scale,
    )
    with NativeWeightEnvHandle(
        close,
        native_cfg,
        {
            "initial_cash": config.initial_cash,
            "max_gross_leverage": config.max_gross_leverage,
            "fee_rate": config.fee_rate,
            "borrow_rate_per_period": config.borrow_rate_per_period,
            "periods_per_year": config.periods_per_year,
            "can_short": int(config.can_short),
        },
        library=lib,
    ) as native_env:
        native_env.reset(3)
        native_obs = native_env.get_obs()
        np.testing.assert_allclose(native_obs, py_obs, atol=1e-9)

        scores = np.array([6.0, -6.0], dtype=np.float64)
        _, py_reward, py_done, py_info = py_env.step(scores)
        native_info = native_env.step(scores)

        assert native_info.done == int(py_done)
        assert native_info.reward == pytest.approx(py_reward, abs=1e-9)
        assert native_info.period_return == pytest.approx(py_info["period_return"], abs=1e-9)
        assert native_info.equity == pytest.approx(py_info["equity"], abs=1e-9)
