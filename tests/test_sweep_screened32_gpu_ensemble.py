from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sweep_screened32_gpu_ensemble.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sweep_screened32_gpu_ensemble", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_monthly_from_total_uses_21_day_scaling():
    sweep = _load_module()

    assert sweep._monthly_from_total(0.10, 100) == pytest.approx(
        math.expm1(math.log1p(0.10) * (21.0 / 100.0))
    )
    assert sweep._monthly_from_total(0.0, 100) == 0.0


def test_build_initial_candidates_preserves_duplicate_baseline_weight():
    sweep = _load_module()
    root = Path("/tmp/ckpts")
    a = root / "A.pt"
    b = root / "B.pt"
    c = root / "C.pt"
    baseline = (a, b, a)

    candidates = sweep.build_initial_candidates(
        baseline_paths=baseline,
        all_paths=(a, b, c),
        include_singles=True,
        include_drop_one=True,
        include_add_one=True,
        include_dup_one=True,
    )
    by_label = {candidate.label: candidate.paths for candidate in candidates}

    assert by_label["baseline_v7"] == baseline
    assert by_label["drop1_A"] == (b, a)
    assert by_label["drop3_A"] == (a, b)
    assert by_label["add_C"] == baseline + (c,)
    assert by_label["dup_A"] == baseline + (a,)


def test_build_pair_candidates_uses_top_ranked_single_results():
    sweep = _load_module()
    mk = lambda label, score: sweep.ScreenResult(
        label=label,
        members=(f"/tmp/{label.removeprefix('single_')}.pt",),
        ensemble_size=1,
        median_total_return=0.0,
        p10_total_return=0.0,
        p90_total_return=0.0,
        median_monthly_return=0.0,
        p10_monthly_return=0.0,
        max_drawdown=0.0,
        median_drawdown=0.0,
        n_neg=0,
        n_windows=10,
        score=score,
    )

    pairs = sweep.build_pair_candidates(
        [mk("single_A", 3.0), mk("single_B", 2.0), mk("single_C", 1.0)],
        top_n=2,
    )

    assert [(candidate.label, candidate.paths) for candidate in pairs] == [
        ("pair_A_B", (Path("/tmp/A.pt"), Path("/tmp/B.pt")))
    ]


def test_sweep_uses_shared_atomic_json_writer():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "from xgbnew.artifacts import write_json_atomic" in source
    assert "def _write_json_atomic" not in source
