from __future__ import annotations

from pathlib import Path

import pytest
from scripts import screened32_diversity_screen as mod


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--fee-rate", "nan"], "fee_rate must be finite and non-negative"),
        (["--fill-buffer-bps", "-1"], "fill_buffer_bps must be finite and non-negative"),
        (["--slippage-bps", "inf"], "slippage_bps must be finite and non-negative"),
        (["--max-leverage", "0"], "max_leverage must be finite and positive"),
        (["--window-days", "0"], "window_days must be positive"),
        (["--decision-lag", "-1"], "decision_lag must be non-negative"),
        (["--decision-lag", "1"], "decision_lag below 2 requires --allow-low-lag-diagnostics"),
        (["--corr-threshold", "nan"], "corr_threshold must be finite"),
        (["--jaccard-threshold", "inf"], "jaccard_threshold must be finite"),
    ],
)
def test_invalid_config_fails_before_path_or_data_work(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv: list[str],
    expected: str,
) -> None:
    def fail_read_mktd(*_args, **_kwargs):
        raise AssertionError("data should not be loaded for invalid configs")

    monkeypatch.setattr(mod, "read_mktd", fail_read_mktd)

    rc = mod.main(
        [
            "--candidate-checkpoint",
            str(tmp_path / "missing.pt"),
            "--val-data",
            str(tmp_path / "missing.bin"),
            "--out-dir",
            str(tmp_path),
            *argv,
        ]
    )

    assert rc == 2
    assert expected in capsys.readouterr().err


def test_baseline_cache_contract_includes_cost_and_shorting_config() -> None:
    args = mod.parse_args([
        "--candidate-checkpoint",
        "candidate.pt",
        "--fee-rate",
        "0.001",
        "--slippage-bps",
        "20",
        "--fill-buffer-bps",
        "5",
        "--max-leverage",
        "1.0",
    ])
    base_ckpts = [Path("a.pt"), Path("b.pt")]
    payload = mod.build_baseline_cache_payload(
        args=args,
        base_ckpts=base_ckpts,
        baseline_rets=[0.1, -0.2],
        start_indices=[0, 1],
    )

    assert mod.baseline_cache_matches_config(payload, args=args, base_ckpts=base_ckpts, start_indices=[0, 1])

    for key, value in (
        ("fee_rate", 0.0000278),
        ("slippage_bps", 5.0),
        ("fill_buffer_bps", 10.0),
        ("max_leverage", 2.0),
        ("disable_shorts", False),
    ):
        changed = dict(payload)
        changed[key] = value
        assert not mod.baseline_cache_matches_config(
            changed,
            args=args,
            base_ckpts=base_ckpts,
            start_indices=[0, 1],
        )

    changed_indices = dict(payload)
    changed_indices["window_start_indices"] = [0, 2]
    assert not mod.baseline_cache_matches_config(
        changed_indices,
        args=args,
        base_ckpts=base_ckpts,
        start_indices=[0, 1],
    )


def test_result_payload_records_cost_config() -> None:
    args = mod.parse_args([
        "--candidate-checkpoint",
        "candidate.pt",
        "--fee-rate",
        "0.001",
        "--slippage-bps",
        "20",
        "--fill-buffer-bps",
        "5",
        "--max-leverage",
        "1.5",
        "--no-disable-shorts",
    ])

    payload = mod.build_result_payload(
        args=args,
        cand_path=Path("candidate.pt"),
        val_path=Path("val.bin"),
        base_med=0.1,
        cand_med=0.2,
        base_neg=1,
        cand_neg=0,
        corr=0.3,
        jaccard=0.4,
        overlap=[1],
        only_baseline=[2],
        only_cand=[3],
        cand_rets=[0.2],
        verdict="ADVISORY-PASS",
    )

    assert payload["fee_rate"] == 0.001
    assert payload["slippage_bps"] == 20.0
    assert payload["fill_buffer_bps"] == 5.0
    assert payload["max_leverage"] == 1.5
    assert payload["disable_shorts"] is False
    assert payload["candidate_window_returns"] == [0.2]
