from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import unified_hourly_experiment.rebuild_all_caches as rebuild_mod


def test_resolve_model_path_prefers_configured_checkpoint(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "chronos2_finetuned"
    preferred = model_root / "NVDA_model_a" / "finetuned-ckpt"
    preferred.mkdir(parents=True)

    monkeypatch.setattr(rebuild_mod, "MODEL_ROOT", model_root)

    resolved = rebuild_mod._resolve_model_path("NVDA", "NVDA_model_a")

    assert resolved == preferred


def test_resolve_model_path_falls_back_to_latest_symbol_checkpoint(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "chronos2_finetuned"
    older = model_root / "NVDA_old" / "finetuned-ckpt"
    newer = model_root / "NVDA_new" / "finetuned-ckpt"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    older_ts = 1_700_000_000
    newer_ts = older_ts + 100
    older.touch()
    newer.touch()
    older.parent.touch()
    newer.parent.touch()
    older.touch()
    newer.touch()
    older_stat_target = older
    newer_stat_target = newer
    import os
    os.utime(older_stat_target, (older_ts, older_ts))
    os.utime(newer_stat_target, (newer_ts, newer_ts))

    monkeypatch.setattr(rebuild_mod, "MODEL_ROOT", model_root)

    resolved = rebuild_mod._resolve_model_path("NVDA", "NVDA_missing")

    assert resolved == newer


def test_resolve_model_path_falls_back_to_configured_chronos_model_id(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "chronos2_finetuned"
    monkeypatch.setattr(rebuild_mod, "MODEL_ROOT", model_root)
    monkeypatch.setattr(
        rebuild_mod,
        "resolve_chronos2_params",
        lambda symbol, frequency="hourly": {"model_id": "amazon/chronos-2"},
    )

    resolved = rebuild_mod._resolve_model_path("TSLA", "TSLA_missing")

    assert resolved == "amazon/chronos-2"


def test_build_cache_uses_custom_horizons(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "chronos2_finetuned"
    preferred = model_root / "NVDA_model_a" / "finetuned-ckpt"
    preferred.mkdir(parents=True)

    captured: dict[str, object] = {}

    def _fake_run(cmd, capture_output, text, check):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(rebuild_mod, "MODEL_ROOT", model_root)
    monkeypatch.setattr(rebuild_mod.subprocess, "run", _fake_run)

    ok = rebuild_mod.build_cache(
        "NVDA",
        "NVDA_model_a",
        horizons="1,6",
        lookback_hours=2400,
        cache_root=tmp_path / "cache",
        data_root=tmp_path / "stocks",
    )

    assert ok is True
    cmd = captured["cmd"]
    assert "--horizons" in cmd
    assert cmd[cmd.index("--horizons") + 1] == "1,6"
    assert cmd[cmd.index("--lookback-hours") + 1] == "2400"


def test_build_selected_caches_respects_symbol_subset(monkeypatch) -> None:
    calls: list[tuple[str, str, str, int]] = []

    def _fake_build_cache(symbol: str, model_name: str, *, horizons: str, lookback_hours: int, cache_root, data_root):
        calls.append((symbol, model_name, horizons, lookback_hours))
        return symbol != "TSLA"

    monkeypatch.setattr(rebuild_mod, "build_cache", _fake_build_cache)
    monkeypatch.setattr(rebuild_mod, "BEST_MODELS", {"NVDA": "nvda_model", "TSLA": "tsla_model"})

    success, failed = rebuild_mod.build_selected_caches(
        symbols=["NVDA", "TSLA"],
        horizons="1,6",
        lookback_hours=3600,
    )

    assert success == 1
    assert failed == ["TSLA"]
    assert calls == [
        ("NVDA", "nvda_model", "1,6", 3600),
        ("TSLA", "tsla_model", "1,6", 3600),
    ]
