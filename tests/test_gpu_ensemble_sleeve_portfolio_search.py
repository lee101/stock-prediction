from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.gpu_ensemble_sleeve_portfolio_search import load_ensemble_specs


def test_load_ensemble_specs_deduplicates_normalized_counts(tmp_path: Path) -> None:
    artifact = tmp_path / "search.json"
    pool = [str(tmp_path / "A.pt"), str(tmp_path / "B.pt")]
    artifact.write_text(
        json.dumps(
            {
                "pool": pool,
                "results": [
                    {"candidate_index": 1, "counts": {"A": 2, "B": 1}},
                    {"candidate_index": 2, "counts": {"A": 4, "B": 2}},
                    {"candidate_index": 3, "counts": {"B": 1}},
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded_pool, specs = load_ensemble_specs([artifact], top_per_artifact=3)

    assert [p.name for p in loaded_pool] == ["A.pt", "B.pt"]
    weights = [counts / counts.sum() for _, counts in specs]
    assert any(np.allclose(w, [2 / 3, 1 / 3]) for w in weights)
    assert any(np.allclose(w, [0.0, 1.0]) for w in weights)
    assert sum(np.allclose(w, [2 / 3, 1 / 3]) for w in weights) == 1


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
        (["--top-per-artifact", "0"], "top_per_artifact must be positive"),
        (["--min-members", "0"], "min_members must be positive"),
        (["--max-members", "0"], "max_members must be positive"),
        (["--max-count", "0"], "max_count must be positive"),
        (["--batch-size", "0"], "batch_size must be positive"),
        (["--top-k", "0"], "top_k must be positive"),
        (["--random-candidates", "-1"], "random_candidates must be non-negative"),
        (["--max-windows", "0"], "max_windows must be positive when provided"),
        (["--min-members", "5", "--max-members", "4"], "max_members must be >= min_members"),
    ],
)
def test_invalid_config_fails_before_cuda_artifact_or_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv: list[str],
    expected: str,
) -> None:
    from scripts import gpu_ensemble_sleeve_portfolio_search as mod

    def fail_cuda_available() -> bool:
        raise AssertionError("CUDA availability should not be checked for invalid configs")

    def fail_load_specs(*args, **kwargs):
        raise AssertionError("artifact loading should not run for invalid configs")

    def fail_read_mktd(*args, **kwargs):
        raise AssertionError("data should not be loaded for invalid configs")

    monkeypatch.setattr(mod.torch.cuda, "is_available", fail_cuda_available)
    monkeypatch.setattr(mod, "load_ensemble_specs", fail_load_specs)
    monkeypatch.setattr(mod, "read_mktd", fail_read_mktd)

    rc = mod.main(
        [
            "--artifact",
            str(tmp_path / "missing-search.json"),
            "--val-data",
            str(tmp_path / "missing.bin"),
            "--out",
            str(tmp_path / "out.json"),
            *argv,
        ]
    )

    assert rc == 2
    assert expected in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()
