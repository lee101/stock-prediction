from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "screened32_swap_in.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("screened32_swap_in", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_float_grid_rejects_empty_duplicate_and_non_finite() -> None:
    module = _load_module()

    assert module._parse_float_grid("0,5,20", name="slippage_bps_grid", min_value=0.0) == [
        0.0,
        5.0,
        20.0,
    ]
    with pytest.raises(ValueError, match="at least one"):
        module._parse_float_grid(" , ", name="slippage_bps_grid", min_value=0.0)
    with pytest.raises(ValueError, match="duplicate"):
        module._parse_float_grid("5,5", name="slippage_bps_grid", min_value=0.0)
    with pytest.raises(ValueError, match="finite"):
        module._parse_float_grid("5,nan", name="slippage_bps_grid", min_value=0.0)
    with pytest.raises(ValueError, match=">= 0"):
        module._parse_float_grid("-1,5", name="slippage_bps_grid", min_value=0.0)


def test_classify_swap_uses_worst_slippage_delta() -> None:
    module = _load_module()

    verdict, summary = module._classify_swap(
        [
            {"delta_median": 0.05, "delta_p10": 0.04, "delta_neg": 0, "delta_sortino": 1.0},
            {"delta_median": -0.01, "delta_p10": 0.02, "delta_neg": 0, "delta_sortino": 0.5},
        ]
    )

    assert verdict == "worse"
    assert summary["worst_delta_median"] == pytest.approx(-0.01)


def test_main_evaluates_default_slippage_grid_and_borrow_apr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    val_path = tmp_path / "val.bin"
    candidate_path = tmp_path / "candidate.pt"
    out_path = tmp_path / "swap.json"
    val_path.write_bytes(b"stub")
    candidate_path.write_bytes(b"stub")

    monkeypatch.setattr(module, "DEFAULT_CHECKPOINT", "base_a.pt")
    monkeypatch.setattr(module, "DEFAULT_EXTRA_CHECKPOINTS", ("base_b.pt",))
    monkeypatch.setattr(
        module,
        "read_mktd",
        lambda path: SimpleNamespace(num_symbols=1, num_timesteps=6, features=SimpleNamespace(shape=(6, 1, 1))),
    )
    monkeypatch.setattr(
        module,
        "load_policy",
        lambda *args, **kwargs: SimpleNamespace(
            policy=object(),
            action_allocation_bins=1,
            action_level_bins=1,
            action_max_offset_bps=0.0,
        ),
    )

    calls: list[dict] = []

    def fake_evaluate_subset(**kwargs):
        slip = float(kwargs["slippage_bps"])
        keep = tuple(kwargs["keep_indices"])
        calls.append(
            {
                "slippage_bps": slip,
                "short_borrow_apr": float(kwargs["short_borrow_apr"]),
                "keep_indices": keep,
            }
        )
        baseline_monthly = {0.0: 0.10, 5.0: 0.095, 10.0: 0.09, 20.0: 0.08}[slip]
        if keep == (0, 1):
            monthly = baseline_monthly
        elif keep == (1, 2):
            monthly = baseline_monthly + 0.003
        else:
            monthly = baseline_monthly + (0.02 if slip < 20.0 else -0.01)
        return {
            "median_total": monthly,
            "p10_total": monthly - 0.02,
            "median_monthly": monthly,
            "p10_monthly": monthly - 0.02,
            "median_sortino": 3.0 + monthly,
            "median_max_dd": 0.1,
            "n_neg": 0,
            "n_windows": 4,
            "slippage_bps": slip,
            "keep_indices": list(keep),
        }

    monkeypatch.setattr(module, "evaluate_subset", fake_evaluate_subset)

    rc = module.main(
        [
            "--candidate",
            str(candidate_path),
            "--val-data",
            str(val_path),
            "--window-days",
            "2",
            "--out",
            str(out_path),
            "--device",
            "cpu",
        ]
    )

    assert rc == 0
    assert {call["slippage_bps"] for call in calls} == {0.0, 5.0, 10.0, 20.0}
    assert {call["short_borrow_apr"] for call in calls} == {0.0625}

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["cell"]["slippage_bps_grid"] == [0.0, 5.0, 10.0, 20.0]
    assert payload["cell"]["short_borrow_apr"] == 0.0625
    assert payload["baseline"]["worst_cell"]["slippage_bps"] == 20.0
    assert payload["swaps"][0]["verdict"] == "win"
    assert payload["swaps"][0]["worst_delta_median"] == pytest.approx(0.003)
    assert payload["swaps"][1]["verdict"] == "worse"
    assert payload["swaps"][1]["worst_delta_median"] == pytest.approx(-0.01)
    assert [win["drop_idx"] for win in payload["wins"]] == [0]


def test_main_single_slippage_override_is_explicit_smoke_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    val_path = tmp_path / "val.bin"
    candidate_path = tmp_path / "candidate.pt"
    out_path = tmp_path / "swap.json"
    val_path.write_bytes(b"stub")
    candidate_path.write_bytes(b"stub")

    monkeypatch.setattr(module, "DEFAULT_CHECKPOINT", "base_a.pt")
    monkeypatch.setattr(module, "DEFAULT_EXTRA_CHECKPOINTS", ())
    monkeypatch.setattr(
        module,
        "read_mktd",
        lambda path: SimpleNamespace(num_symbols=1, num_timesteps=4, features=SimpleNamespace(shape=(4, 1, 1))),
    )
    monkeypatch.setattr(
        module,
        "load_policy",
        lambda *args, **kwargs: SimpleNamespace(
            policy=object(),
            action_allocation_bins=1,
            action_level_bins=1,
            action_max_offset_bps=0.0,
        ),
    )
    observed_slips: list[float] = []

    def fake_evaluate_subset(**kwargs):
        observed_slips.append(float(kwargs["slippage_bps"]))
        return {
            "median_total": 0.1,
            "p10_total": 0.08,
            "median_monthly": 0.1,
            "p10_monthly": 0.08,
            "median_sortino": 3.0,
            "median_max_dd": 0.1,
            "n_neg": 0,
            "n_windows": 2,
            "slippage_bps": float(kwargs["slippage_bps"]),
            "keep_indices": list(kwargs["keep_indices"]),
        }

    monkeypatch.setattr(module, "evaluate_subset", fake_evaluate_subset)

    rc = module.main(
        [
            "--candidate",
            str(candidate_path),
            "--val-data",
            str(val_path),
            "--window-days",
            "2",
            "--slippage-bps",
            "5",
            "--out",
            str(out_path),
            "--device",
            "cpu",
        ]
    )

    assert rc == 0
    assert set(observed_slips) == {5.0}
    assert json.loads(out_path.read_text(encoding="utf-8"))["cell"]["slippage_bps_grid"] == [5.0]


def test_main_rejects_invalid_realism_inputs_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    candidate_path = tmp_path / "candidate.pt"
    candidate_path.write_bytes(b"stub")
    read_calls = 0

    def fail_if_called(path):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("read_mktd should not be called")

    monkeypatch.setattr(module, "read_mktd", fail_if_called)

    rc = module.main(
        [
            "--candidate",
            str(candidate_path),
            "--val-data",
            str(tmp_path / "missing.bin"),
            "--slippage-bps-grid",
            "5,nan",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert read_calls == 0
