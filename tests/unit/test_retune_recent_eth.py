from __future__ import annotations

from pathlib import Path

import pandas as pd

from fastalgorithms.eth_risk_ppo.retune_recent_eth import (
    build_default_candidates,
    build_buy_hold_benchmark,
    candidate_checkpoint_paths,
    load_strategy_trajectory,
    run_meta_sweep,
    score_summary,
    slice_recent_hourly_data,
)


def _write_hourly_csv(path: Path, *, start: str = "2026-01-01", rows: int = 12) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=rows, freq="h", tz="UTC"),
            "open": [100 + idx for idx in range(rows)],
            "high": [101 + idx for idx in range(rows)],
            "low": [99 + idx for idx in range(rows)],
            "close": [100 + idx for idx in range(rows)],
            "volume": [10.0 + idx for idx in range(rows)],
        }
    )
    frame.to_csv(path, index=False)


def test_slice_recent_hourly_data_keeps_requested_tail(tmp_path: Path) -> None:
    source = tmp_path / "trainingdatahourly" / "crypto"
    source.mkdir(parents=True)
    _write_hourly_csv(source / "ETHUSD.csv", rows=10)

    manifest = slice_recent_hourly_data(
        data_root=tmp_path / "trainingdatahourly",
        symbols=["ETHUSD"],
        recent_hours=4,
        output_dir=tmp_path / "recent",
    )

    written = pd.read_csv(tmp_path / "recent" / "ETHUSD.csv")
    assert len(written) == 4
    assert manifest["ETHUSD"]["rows"] == 4


def test_score_summary_and_meta_sweep_accept_recent_candidate_outputs(tmp_path: Path) -> None:
    price_csv = tmp_path / "ETHUSD.csv"
    _write_hourly_csv(price_csv, rows=24)
    benchmark = build_buy_hold_benchmark(csv_path=price_csv, windows_hours=[24, 168], fill_buffers_bps=[5.0, 10.0])

    summary = pd.DataFrame(
        [
            {"window_hours": 24, "fill_buffer_bps": 5.0, "total_return": 0.08, "sortino": 2.0, "max_drawdown": 0.02, "fills_total": 6, "mean_turnover": 0.12},
            {"window_hours": 24, "fill_buffer_bps": 10.0, "total_return": 0.07, "sortino": 1.8, "max_drawdown": 0.02, "fills_total": 6, "mean_turnover": 0.12},
            {"window_hours": 168, "fill_buffer_bps": 5.0, "total_return": 0.11, "sortino": 2.5, "max_drawdown": 0.03, "fills_total": 7, "mean_turnover": 0.10},
            {"window_hours": 168, "fill_buffer_bps": 10.0, "total_return": 0.09, "sortino": 2.2, "max_drawdown": 0.03, "fills_total": 7, "mean_turnover": 0.10},
        ]
    )

    scores = score_summary(summary, benchmark)
    assert isinstance(scores["robust_score"], float)
    assert scores["mean_fills"] >= 6.0

    up_path = tmp_path / "up.csv"
    down_path = tmp_path / "down.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-01", periods=6, freq="h", tz="UTC"),
            "equity": [1.0, 1.02, 1.03, 1.05, 1.06, 1.08],
            "in_position": [True, True, True, True, True, True],
        }
    ).to_csv(up_path, index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-01", periods=6, freq="h", tz="UTC"),
            "equity": [1.0, 0.99, 0.98, 1.00, 1.01, 1.00],
            "in_position": [True, False, False, True, True, False],
        }
    ).to_csv(down_path, index=False)

    strategy_frames = {
        "uptrend": load_strategy_trajectory(up_path),
        "mixed": load_strategy_trajectory(down_path),
    }
    meta_summary, best_meta = run_meta_sweep(strategy_frames=strategy_frames, lookbacks=[2, 4], initial_cash=10_000.0)

    assert "uptrend" in set(meta_summary["strategy"])
    assert any(name.startswith("meta_winner_lb2h") for name in meta_summary["strategy"])
    assert any(name.startswith("meta_softmax_lb2h") for name in meta_summary["strategy"])
    assert "equal_weight" in set(meta_summary["strategy"])
    assert best_meta["strategy"].startswith("meta_")


def test_build_default_candidates_expands_seeds() -> None:
    candidates = build_default_candidates(seeds=[42, 7])

    names = {candidate.name for candidate in candidates}
    assert "chronos2_h6_ctx1024_s42" in names
    assert "chronos2_h6_ctx1024_s7" in names
    assert all(candidate.env_overrides["SEED"] in {"42", "7"} for candidate in candidates)
    assert all(candidate.env_overrides["DETERMINISTIC_TRAINING"] == "1" for candidate in candidates)
    assert all(candidate.env_overrides["DISABLE_TF32"] == "1" for candidate in candidates)
    assert all(candidate.env_overrides["TORCH_NUM_THREADS"] == "1" for candidate in candidates)


def test_candidate_checkpoint_paths_prioritize_best_and_final(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "candidate"
    (artifact_dir / "best").mkdir(parents=True)
    (artifact_dir / "topk").mkdir(parents=True)
    best = artifact_dir / "best" / "best_model.zip"
    final = artifact_dir / "ppo_allocator_final.zip"
    step = artifact_dir / "topk" / "step_4096_reward_0.1000.zip"
    best.write_bytes(b"zip")
    final.write_bytes(b"zip")
    step.write_bytes(b"zip")

    checkpoints = candidate_checkpoint_paths(artifact_dir)

    assert checkpoints[0] == best.resolve()
    assert final.resolve() in checkpoints
    assert step.resolve() in checkpoints
