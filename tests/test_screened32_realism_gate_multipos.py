from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts import screened32_realism_gate_multipos as mod


def _fake_data() -> SimpleNamespace:
    return SimpleNamespace(
        num_symbols=1,
        num_timesteps=4,
        features=np.zeros((4, 1, 1), dtype=np.float32),
        prices=np.full((4, 1, 5), 100.0, dtype=np.float32),
    )


def _fake_head() -> SimpleNamespace:
    return SimpleNamespace(action_allocation_bins=1, action_level_bins=1)


def test_fill_buffer_bps_reduces_multipos_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "compute_top_k_targets",
        lambda **_kwargs: {0: 1.0},
    )
    common = {
        "data": _fake_data(),
        "loaded": [],
        "head": _fake_head(),
        "k": 1,
        "min_prob_ratio": 0.5,
        "total_alloc": 1.0,
        "decision_lag": 0,
        "fee_rate": 0.0,
        "slippage_bps": 0.0,
        "window_days": 3,
        "start_idx": 0,
        "device": torch.device("cpu"),
    }

    no_buffer = mod.simulate_multipos(fill_buffer_bps=0.0, **common)
    high_buffer = mod.simulate_multipos(fill_buffer_bps=100.0, **common)

    assert no_buffer["total_return"] == pytest.approx(0.0)
    assert high_buffer["total_return"] < no_buffer["total_return"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--fee-rate", "nan"], "fee_rate must be finite and non-negative"),
        (["--fill-buffer-bps", "-1"], "fill_buffer_bps must be finite and non-negative"),
        (["--slippage-bps", "inf"], "slippage_bps must be finite and non-negative"),
        (["--total-alloc", "0"], "total_alloc must be finite and positive"),
        (["--max-leverage", "0"], "max_leverage must be finite and positive"),
        (["--total-alloc", "1.5", "--max-leverage", "1.0"], "total_alloc must be <= max_leverage"),
        (["--min-prob-ratio", "0"], "min_prob_ratio must be finite and positive"),
        (["--k", "0"], "k must be positive"),
        (["--window-days", "0"], "window_days must be positive"),
        (["--decision-lag", "-1"], "decision_lag must be non-negative"),
        (["--decision-lag", "1"], "decision_lag below 2 requires --allow-low-lag-diagnostics"),
    ],
)
def test_invalid_config_fails_before_data_load(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
    argv: list[str],
    expected: str,
) -> None:
    def fail_read_mktd(*_args, **_kwargs):
        raise AssertionError("data should not be loaded for invalid configs")

    monkeypatch.setattr(mod, "read_mktd", fail_read_mktd)

    rc = mod.main(
        [
            "--val-data",
            str(tmp_path / "missing.bin"),
            "--out-json",
            str(tmp_path / "out.json"),
            *argv,
        ]
    )

    assert rc == 2
    assert expected in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_payload_records_cost_config() -> None:
    args = mod.parse_args([
        "--k",
        "3",
        "--total-alloc",
        "1.5",
        "--max-leverage",
        "2.0",
        "--fill-buffer-bps",
        "5",
        "--fee-rate",
        "0.001",
        "--slippage-bps",
        "20",
    ])

    payload = mod.build_payload(args, ["a.pt", "b.pt"], {"n_windows": 4})

    assert payload["ensemble"] == ["a", "b"]
    assert payload["k"] == 3
    assert payload["total_alloc"] == 1.5
    assert payload["max_leverage"] == 2.0
    assert payload["fill_buffer_bps"] == 5.0
    assert payload["fee_rate"] == 0.001
    assert payload["slippage_bps"] == 20.0
    assert payload["decision_lag"] == 2
    assert payload["summary"] == {"n_windows": 4}
