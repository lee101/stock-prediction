from __future__ import annotations

import json
from pathlib import Path

import torch

from binanceexp1.train_multiasset_selector_robust import (
    build_candidate_spec,
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
