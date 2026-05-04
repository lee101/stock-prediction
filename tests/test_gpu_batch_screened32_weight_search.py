from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gpu_batch_screened32_weight_search.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gpu_batch_screened32_weight_search", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_counts_for_members_preserves_duplicate_weights():
    mod = _load_module()
    root = Path("/tmp/ckpts")
    pool = (root / "A.pt", root / "B.pt", root / "C.pt")

    counts = mod._counts_for_members(pool, [root / "A.pt", root / "B.pt", root / "A.pt", "C.pt"])

    assert counts == (2, 1, 1)


def test_repair_counts_respects_member_bounds_and_max_weight():
    mod = _load_module()
    rng = np.random.default_rng(7)

    low = mod._repair_counts(
        np.array([0, 0, 0, 0], dtype=np.int16),
        rng,
        max_weight=2,
        min_members=3,
        max_members=5,
        preferred=np.array([0, 1], dtype=np.int64),
    )
    high = mod._repair_counts(
        np.array([3, 3, 3, 3], dtype=np.int16),
        rng,
        max_weight=2,
        min_members=3,
        max_members=5,
        preferred=np.array([0, 1], dtype=np.int64),
    )

    assert 3 <= int(low.sum()) <= 5
    assert int(low.max()) <= 2
    assert 3 <= int(high.sum()) <= 5
    assert int(high.max()) <= 2


def test_make_population_is_deduped_and_within_bounds():
    mod = _load_module()
    root = Path("/tmp/ckpts")
    pool = tuple(root / f"M{i}.pt" for i in range(6))
    seeds = [(1, 1, 1, 0, 0, 0), (0, 2, 0, 2, 0, 0)]

    pop = mod.make_population(
        pool=pool,
        seeds=seeds,
        n_candidates=50,
        rng=np.random.default_rng(11),
        min_members=3,
        max_members=6,
        max_weight=2,
        mutation_prob=0.35,
        random_frac=0.3,
    )

    assert pop.shape[1] == len(pool)
    assert len({tuple(row.tolist()) for row in pop}) == len(pop)
    assert np.all(pop.sum(axis=1) >= 3)
    assert np.all(pop.sum(axis=1) <= 6)
    assert np.all(pop <= 2)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--fill-buffer-bps", "nan"], "fill_buffer_bps must be finite and non-negative"),
        (["--slippage-bps", "-1"], "slippage_bps must be finite and non-negative"),
        (["--fee-rate", "inf"], "fee_rate must be finite and non-negative"),
        (["--neg-penalty", "nan"], "neg_penalty must be finite and non-negative"),
        (["--dd-penalty", "-0.1"], "dd_penalty must be finite and non-negative"),
        (["--leverage", "0"], "leverage must be finite and positive"),
        (["--seed-top-k", "0"], "seed_top_k must be positive"),
        (["--candidates", "0"], "candidates must be positive"),
        (["--batch-size", "0"], "batch_size must be positive"),
        (["--min-members", "0"], "min_members must be positive"),
        (["--max-members", "0"], "max_members must be positive"),
        (["--max-weight", "0"], "max_weight must be positive"),
        (["--window-days", "0"], "window_days must be positive"),
        (["--top-k", "0"], "top_k must be positive"),
        (["--max-windows", "0"], "max_windows must be positive when provided"),
        (["--mutation-prob", "-0.1"], "mutation_prob must be between 0 and 1"),
        (["--random-frac", "1.1"], "random_frac must be between 0 and 1"),
        (["--min-members", "5", "--max-members", "4"], "max_members must be >= min_members"),
    ],
)
def test_invalid_config_fails_before_cuda_or_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv: list[str],
    expected: str,
):
    mod = _load_module()

    def fail_cuda_available() -> bool:
        raise AssertionError("CUDA availability should not be checked for invalid configs")

    def fail_read_mktd(*args, **kwargs):
        raise AssertionError("data should not be loaded for invalid configs")

    monkeypatch.setattr(mod.torch.cuda, "is_available", fail_cuda_available)
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
