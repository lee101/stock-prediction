from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from differentiable_loss_utils import (
    combined_sortino_pnl_loss as torch_combined_sortino_pnl_loss,
    compute_hourly_objective as torch_compute_hourly_objective,
    simulate_hourly_trades as torch_simulate_hourly_trades,
    simulate_hourly_trades_binary as torch_simulate_hourly_trades_binary,
)


def _sample_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    closes = rng.uniform(90.0, 110.0, size=(2, 8)).astype(np.float32)
    highs = closes + rng.uniform(0.1, 1.0, size=(2, 8)).astype(np.float32)
    lows = closes - rng.uniform(0.1, 1.0, size=(2, 8)).astype(np.float32)
    opens = closes + rng.uniform(-0.5, 0.5, size=(2, 8)).astype(np.float32)
    buy_prices = closes - 0.25
    sell_prices = closes + 0.35
    trade_intensity = rng.uniform(0.0, 0.8, size=(2, 8)).astype(np.float32)
    return {
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "opens": opens,
        "buy_prices": buy_prices,
        "sell_prices": sell_prices,
        "trade_intensity": trade_intensity,
        "buy_trade_intensity": trade_intensity * 0.9,
        "sell_trade_intensity": trade_intensity * 0.8,
    }


def _run_jax_losses_probe(case: str, inputs: dict[str, np.ndarray]) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        inputs_path = Path(tmpdir) / "inputs.npz"
        output_path = Path(tmpdir) / "output.json"
        np.savez(inputs_path, **inputs)

        script = """
import json
import sys
from pathlib import Path

import numpy as np

from binanceneural.jax_losses import (
    combined_sortino_pnl_loss,
    compute_hourly_objective,
    simulate_hourly_trades,
    simulate_hourly_trades_binary,
)

case = sys.argv[1]
inputs_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
archive = np.load(inputs_path)
inputs = {key: archive[key] for key in archive.files}

if case == "simulate_hourly_trades":
    result = simulate_hourly_trades(
        **inputs,
        maker_fee=0.001,
        initial_cash=1.0,
        temperature=5e-4,
        max_leverage=2.0,
        can_short=True,
        can_long=True,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        margin_annual_rate=0.0625,
    )
    payload = {
        "returns": np.asarray(result.returns).tolist(),
        "portfolio_values": np.asarray(result.portfolio_values).tolist(),
    }
elif case == "binary_and_objective":
    sim = simulate_hourly_trades_binary(
        **inputs,
        maker_fee=0.001,
        initial_cash=1.0,
        max_leverage=2.0,
        can_short=False,
        can_long=True,
        decision_lag_bars=0,
        market_order_entry=False,
        fill_buffer_pct=0.0,
        margin_annual_rate=0.0,
    )
    score, sortino, annual = compute_hourly_objective(
        sim.returns,
        periods_per_year=8760.0,
        return_weight=0.15,
        smoothness_penalty=0.0,
    )
    loss = combined_sortino_pnl_loss(
        sim.returns,
        target_sign=1.0,
        periods_per_year=8760.0,
        return_weight=0.15,
        smoothness_penalty=0.0,
    )
    payload = {
        "returns": np.asarray(sim.returns).tolist(),
        "score": np.asarray(score).tolist(),
        "sortino": np.asarray(sortino).tolist(),
        "annual": np.asarray(annual).tolist(),
        "loss": np.asarray(loss).tolist(),
    }
else:
    raise ValueError(f"Unknown case: {case}")

output_path.write_text(json.dumps(payload))
"""

        env = os.environ.copy()
        env.setdefault("JAX_PLATFORMS", "cpu")
        proc = subprocess.run(
            [sys.executable, "-c", script, case, str(inputs_path), str(output_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            combined_output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if proc.returncode < 0 or "Fatal Python error" in combined_output or "Aborted" in combined_output:
                pytest.skip(
                    "JAX losses test skipped due to native JAX/XLA compiler abort under this environment: "
                    f"{combined_output[-400:]}"
                )
            raise AssertionError(
                "JAX losses subprocess failed unexpectedly:\n"
                f"{combined_output or f'process exited with code {proc.returncode}'}"
            )
        return json.loads(output_path.read_text())


def test_simulate_hourly_trades_matches_torch() -> None:
    inputs = _sample_inputs()
    torch_result = torch_simulate_hourly_trades(
        **{key: torch.from_numpy(value) for key, value in inputs.items()},
        maker_fee=0.001,
        initial_cash=1.0,
        temperature=5e-4,
        max_leverage=2.0,
        can_short=True,
        can_long=True,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        margin_annual_rate=0.0625,
    )
    jax_result = _run_jax_losses_probe("simulate_hourly_trades", inputs)
    np.testing.assert_allclose(torch_result.returns.numpy(), np.asarray(jax_result["returns"]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        torch_result.portfolio_values.numpy(),
        np.asarray(jax_result["portfolio_values"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_binary_sim_and_objective_match_torch() -> None:
    inputs = _sample_inputs()
    torch_sim = torch_simulate_hourly_trades_binary(
        **{key: torch.from_numpy(value) for key, value in inputs.items()},
        maker_fee=0.001,
        initial_cash=1.0,
        max_leverage=2.0,
        can_short=False,
        can_long=True,
        decision_lag_bars=0,
        market_order_entry=False,
        fill_buffer_pct=0.0,
        margin_annual_rate=0.0,
    )
    jax_probe = _run_jax_losses_probe("binary_and_objective", inputs)
    np.testing.assert_allclose(torch_sim.returns.numpy(), np.asarray(jax_probe["returns"]), rtol=1e-6, atol=1e-6)

    torch_score, torch_sortino, torch_annual = torch_compute_hourly_objective(
        torch_sim.returns,
        periods_per_year=8760.0,
        return_weight=0.15,
        smoothness_penalty=0.0,
    )
    np.testing.assert_allclose(torch_score.numpy(), np.asarray(jax_probe["score"]), rtol=2e-5, atol=1e-4)
    np.testing.assert_allclose(torch_sortino.numpy(), np.asarray(jax_probe["sortino"]), rtol=2e-5, atol=1e-4)
    np.testing.assert_allclose(torch_annual.numpy(), np.asarray(jax_probe["annual"]), rtol=2e-5, atol=1e-4)

    torch_loss = torch_combined_sortino_pnl_loss(
        torch_sim.returns,
        target_sign=1.0,
        periods_per_year=8760.0,
        return_weight=0.15,
        smoothness_penalty=0.0,
    )
    np.testing.assert_allclose(torch_loss.numpy(), np.asarray(jax_probe["loss"]), rtol=2e-5, atol=1e-4)
