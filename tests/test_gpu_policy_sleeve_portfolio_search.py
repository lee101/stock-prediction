from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.gpu_policy_sleeve_portfolio_search import (
    _eval_weight_candidates,
    build_candidate_weights,
)


def test_build_candidate_weights_includes_identity_and_normalizes() -> None:
    pool = [Path("a.pt"), Path("b.pt"), Path("c.pt")]
    weights = build_candidate_weights(
        pool=pool,
        sleeve_returns=np.array([0.1, 0.2, -0.1]),
        sleeve_p10=np.array([0.0, 0.1, -0.2]),
        random_candidates=4,
        min_members=2,
        max_members=3,
        max_count=2,
        seed=1,
        artifact_counts=[np.array([2.0, 1.0, 0.0])],
    )

    assert weights.shape[1] == 3
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert any(np.allclose(row, np.array([1.0, 0.0, 0.0])) for row in weights)
    assert any(np.allclose(row, np.array([2 / 3, 1 / 3, 0.0])) for row in weights)


def test_eval_weight_candidates_aggregates_equity_curves() -> None:
    equity_curve = torch.tensor(
        [
            [[10000.0, 11000.0, 12000.0]],
            [[10000.0, 9000.0, 8000.0]],
        ],
        dtype=torch.float32,
    )
    turnover = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
    weights = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)

    results = _eval_weight_candidates(
        weights=weights,
        equity_curve=equity_curve,
        turnover=turnover,
        labels=["winner", "loser"],
        window_days=2,
        batch_size=2,
        neg_penalty=0.0,
        dd_penalty=0.0,
        turnover_penalty=0.0,
        top_k=2,
    )

    assert results[0]["members"] == [("winner", 1.0)]
    assert results[0]["median_total_return"] > results[1]["median_total_return"]


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
        (["--min-members", "0"], "min_members must be positive"),
        (["--max-members", "0"], "max_members must be positive"),
        (["--max-count", "0"], "max_count must be positive"),
        (["--batch-size", "0"], "batch_size must be positive"),
        (["--top-k", "0"], "top_k must be positive"),
        (["--random-candidates", "-1"], "random_candidates must be non-negative"),
        (["--max-windows", "0"], "max_windows must be positive when provided"),
        (["--candidate-index", "-1"], "candidate_index values must be non-negative"),
        (["--min-members", "5", "--max-members", "4"], "max_members must be >= min_members"),
    ],
)
def test_invalid_config_fails_before_cuda_or_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv: list[str],
    expected: str,
) -> None:
    from scripts import gpu_policy_sleeve_portfolio_search as mod

    def fail_cuda_available() -> bool:
        raise AssertionError("CUDA availability should not be checked for invalid configs")

    def fail_load_pool(*args, **kwargs):
        raise AssertionError("checkpoint loading should not run for invalid configs")

    def fail_read_mktd(*args, **kwargs):
        raise AssertionError("data should not be loaded for invalid configs")

    monkeypatch.setattr(mod.torch.cuda, "is_available", fail_cuda_available)
    monkeypatch.setattr(mod, "_load_pool", fail_load_pool)
    monkeypatch.setattr(mod, "read_mktd", fail_read_mktd)

    rc = mod.main(
        [
            "--val-data",
            str(tmp_path / "missing.bin"),
            "--checkpoint-root",
            str(tmp_path / "missing-checkpoints"),
            "--out",
            str(tmp_path / "out.json"),
            *argv,
        ]
    )

    assert rc == 2
    assert expected in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()
