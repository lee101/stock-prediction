from __future__ import annotations

from pathlib import Path

import pytest
import torch
from scripts.gpu_factor_portfolio_search import (
    BASE_RECIPES,
    FactorConfig,
    _factor_score_bank,
    _recipe_matrix,
    _target_weights_from_factor_scores,
    build_factor_configs,
)


def test_build_factor_configs_cross_product() -> None:
    configs = build_factor_configs(
        BASE_RECIPES[:2],
        pack_sizes=[1, 2],
        score_powers=[0.0],
        vol_powers=[0.0],
        score_gates=[0.0],
        gross_scales=[0.5],
        rebalance_everys=[1, 5],
        rebalance_thresholds=[0.0],
    )

    assert len(configs) == 8
    assert configs[0].factor == BASE_RECIPES[0].name
    assert configs[0].pack_size == 1
    assert configs[1].rebalance_every == 5


def test_factor_score_bank_cross_sectional_zscore() -> None:
    recipes = BASE_RECIPES[:1]  # ret1_mom
    recipe_matrix = _recipe_matrix(recipes, device=torch.device("cpu"))
    features = torch.zeros(1, 2, 3, 16)
    features[:, 0, :, 0] = torch.tensor([[1.0, 2.0, 3.0]])
    tradable = torch.ones(1, 2, 3, dtype=torch.bool)

    scores = _factor_score_bank(
        features=features,
        tradable=tradable,
        recipe_matrix=recipe_matrix,
        step=1,
    )

    assert scores.shape == (1, 1, 3)
    assert torch.allclose(scores.mean(dim=2), torch.zeros(1, 1), atol=1e-6)
    assert scores[0, 0, 2] > scores[0, 0, 1] > scores[0, 0, 0]


def test_target_weights_from_factor_scores_uses_gate_and_gross() -> None:
    factor_scores = torch.tensor([[[1.2, 0.8, -0.5], [0.1, 2.0, 1.0]]], dtype=torch.float32)
    factor_ids = torch.tensor([0], dtype=torch.long)
    tradable = torch.ones(2, 3, dtype=torch.bool)
    vol = torch.ones(2, 3)
    configs = [
        FactorConfig(
            factor="ret1_mom",
            pack_size=2,
            score_power=0.0,
            vol_power=0.0,
            score_gate=0.5,
            gross_scale=0.75,
            rebalance_every=1,
            rebalance_threshold=0.0,
        )
    ]

    weights = _target_weights_from_factor_scores(
        factor_scores=factor_scores,
        factor_ids=factor_ids,
        tradable=tradable,
        vol=vol,
        configs=configs,
    )

    assert weights.shape == (1, 2, 3)
    assert torch.allclose(weights[0, 0], torch.tensor([0.375, 0.375, 0.0]))
    assert torch.allclose(weights[0, 1], torch.tensor([0.0, 0.375, 0.375]))


def test_target_weights_from_factor_scores_inverse_vol() -> None:
    factor_scores = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
    factor_ids = torch.tensor([0], dtype=torch.long)
    tradable = torch.ones(1, 2, dtype=torch.bool)
    vol = torch.tensor([[0.05, 0.20]])
    configs = [
        FactorConfig(
            factor="ret1_mom",
            pack_size=2,
            score_power=1.0,
            vol_power=1.0,
            score_gate=0.0,
            gross_scale=1.0,
            rebalance_every=1,
            rebalance_threshold=0.0,
        )
    ]

    weights = _target_weights_from_factor_scores(
        factor_scores=factor_scores,
        factor_ids=factor_ids,
        tradable=tradable,
        vol=vol,
        configs=configs,
    )

    assert weights[0, 0, 0] > weights[0, 0, 1]
    assert torch.allclose(weights.sum(), torch.tensor(1.0))


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
        (["--pack-sizes", "0"], "pack_sizes: integer list values must be positive"),
        (["--score-powers", "nan"], "score_powers: list values must be finite"),
        (["--score-gates", "-0.1"], "score_gates must contain only non-negative values"),
    ],
)
def test_invalid_config_fails_before_cuda_or_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv: list[str],
    expected: str,
) -> None:
    from scripts import gpu_factor_portfolio_search as mod

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
