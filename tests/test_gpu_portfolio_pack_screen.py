from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from scripts.gpu_portfolio_pack_screen import (
    PackConfig,
    _members_from_artifact,
    _target_weights_from_scores,
    build_config_grid,
)


def test_build_config_grid_cross_product() -> None:
    configs = build_config_grid(
        pack_sizes=[1, 2],
        score_powers=[0.0],
        vol_powers=[0.0, 1.0],
        flat_gates=[0.0],
        gross_scales=[0.5, 1.0],
        rebalance_everys=[1, 5],
        rebalance_thresholds=[0.0],
    )

    assert len(configs) == 16
    assert configs[0] == PackConfig(
        pack_size=1,
        score_power=0.0,
        vol_power=0.0,
        flat_gate=0.0,
        gross_scale=0.5,
        rebalance_every=1,
        rebalance_threshold=0.0,
    )


def test_target_weights_equal_top_k_respects_gross_scale_and_tradable() -> None:
    long_probs = torch.tensor([[0.40, 0.30, 0.20], [0.10, 0.35, 0.25]], dtype=torch.float32)
    flat_probs = torch.tensor([0.05, 0.05], dtype=torch.float32)
    tradable = torch.tensor([[True, False, True], [True, True, True]])
    vol = torch.ones_like(long_probs)
    configs = [
        PackConfig(
            pack_size=2,
            score_power=0.0,
            vol_power=0.0,
            flat_gate=0.0,
            gross_scale=0.75,
            rebalance_every=1,
            rebalance_threshold=0.0,
        )
    ]

    weights = _target_weights_from_scores(
        long_probs=long_probs,
        flat_probs=flat_probs,
        tradable=tradable,
        vol=vol,
        configs=configs,
    )

    assert weights.shape == (1, 2, 3)
    assert torch.allclose(weights[0, 0], torch.tensor([0.375, 0.0, 0.375]))
    assert torch.allclose(weights[0, 1].sum(), torch.tensor(0.75))


def test_target_weights_flat_gate_can_force_cash() -> None:
    long_probs = torch.tensor([[0.08, 0.07, 0.06]], dtype=torch.float32)
    flat_probs = torch.tensor([0.10], dtype=torch.float32)
    tradable = torch.ones_like(long_probs, dtype=torch.bool)
    vol = torch.ones_like(long_probs)
    configs = [
        PackConfig(
            pack_size=3,
            score_power=1.0,
            vol_power=0.0,
            flat_gate=1.0,
            gross_scale=1.0,
            rebalance_every=1,
            rebalance_threshold=0.0,
        )
    ]

    weights = _target_weights_from_scores(
        long_probs=long_probs,
        flat_probs=flat_probs,
        tradable=tradable,
        vol=vol,
        configs=configs,
    )

    assert torch.equal(weights, torch.zeros_like(weights))


def test_target_weights_inverse_vol_prefers_lower_risk_name() -> None:
    long_probs = torch.tensor([[0.40, 0.40]], dtype=torch.float32)
    flat_probs = torch.tensor([0.0], dtype=torch.float32)
    tradable = torch.ones_like(long_probs, dtype=torch.bool)
    vol = torch.tensor([[0.05, 0.20]], dtype=torch.float32)
    configs = [
        PackConfig(
            pack_size=2,
            score_power=1.0,
            vol_power=1.0,
            flat_gate=0.0,
            gross_scale=1.0,
            rebalance_every=1,
            rebalance_threshold=0.0,
        )
    ]

    weights = _target_weights_from_scores(
        long_probs=long_probs,
        flat_probs=flat_probs,
        tradable=tradable,
        vol=vol,
        configs=configs,
    )

    assert weights[0, 0, 0] > weights[0, 0, 1]
    assert torch.allclose(weights.sum(), torch.tensor(1.0))


def test_members_from_artifact_selects_candidate_index(tmp_path: Path) -> None:
    artifact = tmp_path / "screen.json"
    artifact.write_text(
        json.dumps(
            {
                "results": [
                    {"candidate_index": 1, "members": ["a.pt"]},
                    {"candidate_index": 7, "members": ["b.pt", "c.pt"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _members_from_artifact(artifact, 7) == ["b.pt", "c.pt"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--fill-buffer-bps", "nan"], "fill_buffer_bps must be finite and non-negative"),
        (["--slippage-bps", "-1"], "slippage_bps must be finite and non-negative"),
        (["--fee-rate", "inf"], "fee_rate must be finite and non-negative"),
        (["--margin-apr", "-0.1"], "margin_apr must be finite and non-negative"),
        (["--neg-penalty", "nan"], "neg_penalty must be finite and non-negative"),
        (["--dd-penalty", "-0.1"], "dd_penalty must be finite and non-negative"),
        (["--turnover-penalty", "-0.1"], "turnover_penalty must be finite and non-negative"),
        (["--leverage", "0"], "leverage must be finite and positive"),
        (["--window-days", "0"], "window_days must be positive"),
        (["--vol-lookback", "0"], "vol_lookback must be positive"),
        (["--top-k", "0"], "top_k must be positive"),
        (["--max-windows", "0"], "max_windows must be positive when provided"),
        (["--candidate-index", "-1"], "candidate_index must be non-negative"),
        (["--pack-sizes", "0"], "pack_sizes: integer list values must be positive"),
        (["--score-powers", "nan"], "score_powers: list values must be finite"),
        (["--flat-gates", "-0.1"], "flat_gates must contain only non-negative values"),
    ],
)
def test_invalid_config_fails_before_cuda_or_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv: list[str],
    expected: str,
) -> None:
    from scripts import gpu_portfolio_pack_screen as mod

    def fail_cuda_available() -> bool:
        raise AssertionError("CUDA availability should not be checked for invalid configs")

    def fail_read_mktd(*args, **kwargs):
        raise AssertionError("data should not be loaded for invalid configs")

    monkeypatch.setattr(mod.torch.cuda, "is_available", fail_cuda_available)
    monkeypatch.setattr(mod, "read_mktd", fail_read_mktd)

    rc = mod.main(["--val-data", str(tmp_path / "missing.bin"), "--out", str(tmp_path / "out.json"), *argv])

    assert rc == 2
    assert expected in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()
