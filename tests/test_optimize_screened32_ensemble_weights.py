from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "optimize_screened32_ensemble_weights.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("optimize_screened32_ensemble_weights", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_counts_round_trip_preserves_integer_weights():
    mod = _load_module()
    root = Path("/tmp/ckpts")
    pool = (root / "A.pt", root / "B.pt", root / "C.pt")
    paths = (pool[0], pool[1], pool[0], pool[2], pool[0])

    counts = mod._counts_from_paths(pool, paths)

    assert counts == {"A": 3, "B": 1, "C": 1}
    assert mod._paths_from_counts(pool, counts) == (
        pool[0],
        pool[0],
        pool[0],
        pool[1],
        pool[2],
    )


def test_candidate_pool_include_accepts_stems_and_names(tmp_path):
    mod = _load_module()
    for name in ("A.pt", "B.pt", "C.pt"):
        (tmp_path / name).write_bytes(b"x")

    selected = mod._candidate_pool(tmp_path, ["B", "C.pt"])

    assert [path.name for path in selected] == ["B.pt", "C.pt"]


def test_seed_weighted_v8_counts_drops_d42_and_adds_known_weights():
    mod = _load_module()
    root = Path("/tmp/ckpts")
    pool = tuple(root / f"{stem}.pt" for stem in ("D_s42", "D_s28", "D_s24", "D_s57", "D_s72"))

    counts = mod._seed_weighted_v8_counts(pool)

    assert counts.get("D_s42", 0) == 0
    assert counts["D_s28"] >= 1
    assert counts["D_s24"] >= 1
    assert counts["D_s57"] >= 1
    assert counts["D_s72"] >= 1


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--decision-lag", "1"], "decision_lag must be exactly 2"),
        (["--fill-buffer-bps", "nan"], "fill_buffer_bps must be finite and non-negative"),
        (["--slippage-bps", "-1"], "slippage_bps must be finite and non-negative"),
        (["--fee-rate", "inf"], "fee_rate must be finite and non-negative"),
        (["--neg-penalty", "nan"], "neg_penalty must be finite and non-negative"),
        (["--dd-penalty", "-0.1"], "dd_penalty must be finite and non-negative"),
        (["--leverage", "0"], "leverage must be finite and positive"),
        (["--window-days", "0"], "window_days must be positive"),
        (["--trials", "0"], "trials must be positive"),
        (["--min-members", "0"], "min_members must be positive"),
        (["--max-members", "0"], "max_members must be positive"),
        (["--max-weight", "0"], "max_weight must be positive"),
        (["--top-k", "0"], "top_k must be positive"),
        (["--max-windows", "0"], "max_windows must be positive when provided"),
        (["--min-members", "5", "--max-members", "4"], "max_members must be >= min_members"),
    ],
)
def test_invalid_config_fails_before_cuda_optuna_or_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv: list[str],
    expected: str,
):
    mod = _load_module()
    original_import = builtins.__import__

    def fail_cuda_available() -> bool:
        raise AssertionError("CUDA availability should not be checked for invalid configs")

    def fail_read_mktd(*args, **kwargs):
        raise AssertionError("data should not be loaded for invalid configs")

    def fail_optuna_import(name, *args, **kwargs):
        if name == "optuna":
            raise AssertionError("optuna should not be imported for invalid configs")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(mod.torch.cuda, "is_available", fail_cuda_available)
    monkeypatch.setattr(mod, "read_mktd", fail_read_mktd)
    monkeypatch.setattr(builtins, "__import__", fail_optuna_import)

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
