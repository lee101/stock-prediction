from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from binanceexp1.train_multiasset_selector_robust import (
    build_candidate_spec,
    build_search_command,
    choose_symbol_candidates,
    collect_top_checkpoints,
    load_configs,
    parse_symbol_map,
)


def test_load_configs_requires_named_dicts(tmp_path: Path) -> None:
    path = tmp_path / "configs.json"
    path.write_text(json.dumps([{"name": "ok"}, {"epochs": 1}]))

    try:
        load_configs(path)
    except ValueError as exc:
        assert "non-empty 'name'" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for missing config name")


def test_collect_top_checkpoints_orders_by_metric(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir()
    for epoch, score in [(1, 0.5), (2, 1.5), (3, 1.0)]:
        torch.save({"epoch": epoch, "metrics": {"score": score}}, ckpt_dir / f"epoch_{epoch:03d}.pt")

    candidates = collect_top_checkpoints(
        ckpt_dir,
        metric_name="score",
        top_k=2,
        run_name="run",
        source="trained",
    )

    assert [candidate.epoch for candidate in candidates] == [2, 3]
    assert [round(candidate.score, 2) for candidate in candidates] == [1.5, 1.0]


def test_choose_symbol_candidates_includes_baseline_and_formats_spec(tmp_path: Path) -> None:
    trained_dir = tmp_path / "trained"
    trained_dir.mkdir()
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    torch.save({"epoch": 1, "metrics": {"score": 0.9}}, trained_dir / "epoch_001.pt")
    torch.save({"epoch": 2, "metrics": {"score": 0.4}}, baseline_dir / "epoch_002.pt")

    trained = collect_top_checkpoints(
        trained_dir,
        metric_name="score",
        top_k=1,
        run_name="trained",
        source="trained",
    )
    selected = choose_symbol_candidates(
        [trained],
        baseline_paths=[baseline_dir / "epoch_002.pt"],
        metric_name="score",
        max_candidates=2,
    )
    spec = build_candidate_spec(
        ["BTCUSD"],
        {"BTCUSD": selected},
    )

    assert len(selected) == 2
    assert "BTCUSD=" in spec
    assert "epoch_001.pt" in spec
    assert "epoch_002.pt" in spec


def test_parse_symbol_map_accepts_directories(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "btc"
    ckpt_dir.mkdir()
    (ckpt_dir / "epoch_001.pt").write_text("x")
    (ckpt_dir / "epoch_003.pt").write_text("x")

    parsed = parse_symbol_map(f"BTCUSD={ckpt_dir}", symbols=["BTCUSD"])

    assert parsed["BTCUSD"][0].name == "epoch_003.pt"


def test_build_search_command_passes_max_history_hours(tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        sequence_length=96,
        forecast_horizons=(1, 6, 24),
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("experiments/cache"),
        validation_days=30.0,
        max_history_hours=24 * 120,
        use_forecast_interactions=True,
        search_window_hours="336",
        initial_cash=10_000.0,
        seed_position_fraction=1.0,
        default_intensity=6.0,
        default_offset=0.0,
        min_edge=0.0015,
        risk_weight=0.25,
        edge_mode="high_low",
        max_hold_hours=6,
        search_decision_lag_bars=2,
        search_fill_buffer_bps=20.0,
        max_volume_fraction=0.1,
        search_limit_fill_model="penetration",
        search_touch_fill_fraction=0.05,
        max_concurrent_positions=1,
        sortino_clip=10.0,
        min_trade_count_mean=6.0,
        cache_only=False,
        realistic_selection=True,
        require_all_positive=False,
        offset_map="ETHUSD=0.0003,SOLUSD=0.0005",
        intensity_map=None,
        work_steal=False,
        work_steal_min_profit_pct=0.001,
        work_steal_min_edge=0.005,
        work_steal_edge_margin=0.0,
    )

    cmd = build_search_command(
        args=args,
        candidate_spec="BTCUSD=a.pt;ETHUSD=b.pt;SOLUSD=c.pt",
        output_dir=tmp_path / "search",
    )

    joined = " ".join(cmd)
    assert "--max-history-hours 2880" in joined
    assert "--use-forecast-interactions" in joined
    assert "--limit-fill-model penetration" in joined
    assert "--touch-fill-fraction 0.05" in joined
